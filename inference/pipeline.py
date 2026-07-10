"""
inference/pipeline.py
─────────────────────────────────────────────────────────────────────────────
END-TO-END CLINICAL INFERENCE PIPELINE

Combines:
  1. Preprocessing (Reinhard normalisation, crop, resize)
  2. TTA prediction (8 augmented views, temperature-calibrated)
  3. Grad-CAM ensemble (GradCAM / GradCAM++ / EigenCAM / LayerCAM)
  4. Clinical staging + report generation

Returns a self-contained InferenceResult dataclass consumed by the GUI.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import io
import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    CLASSES, IDX_TO_CLASS, CLASS_TO_IDX,
    MODELS_DIR, RESULTS_DIR, IMAGE_SIZE,
    MALIGNANT_CLASSES,
)
from models.backbones import get_model
from preprocessing.preprocess import preprocess_image
from utils.tta import TTAPredictor, TTA_NAMES
from utils.gradcam_engine import (
    run_all_cam_methods,
    get_default_target_layer,
    overlay_cam_on_image,
    cam_to_heatmap,
    make_panel,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
_normalise = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# ─── Result dataclass ─────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    # ── Core prediction ──────────────────────────────────────────────
    predicted_class:  str   = ""
    predicted_idx:    int   = -1
    is_malignant:     bool  = False
    verdict:          str   = ""           # "MALIGNANT" | "BENIGN"
    confidence:       float = 0.0          # 0–1
    uncertainty:      float = 0.0          # std over TTA views
    staging:          str   = ""

    # ── Probabilities ────────────────────────────────────────────────
    probabilities:    Dict[str, float] = field(default_factory=dict)
    per_view_probs:   List             = field(default_factory=list)

    # ── Images (base64 PNG strings for GUI) ──────────────────────────
    original_b64:     str  = ""
    heatmap_b64:      str  = ""
    overlay_b64:      str  = ""
    panel_b64:        str  = ""            # 3-panel: orig | heatmap | overlay

    # ── CAM arrays (float32 numpy H×W) ───────────────────────────────
    cam_maps:         Dict[str, np.ndarray] = field(default_factory=dict)

    # ── Metadata ─────────────────────────────────────────────────────
    model_name:       str   = ""
    n_tta_views:      int   = 8
    inference_time_s: float = 0.0
    image_shape:      Tuple = (224, 224, 3)


def _ndarray_to_b64(img_bgr: np.ndarray) -> str:
    """Encode a BGR numpy image as base64 PNG string."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _ndarray_to_b64_rgb(img_rgb: np.ndarray) -> str:
    return _ndarray_to_b64(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


# ─── Model loader (singleton cache) ──────────────────────────────────────

_MODEL_CACHE: Dict[str, torch.nn.Module] = {}


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """Load and cache a model from its best checkpoint."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    ckpt = MODELS_DIR / f"{model_name}_best.pth"
    model = get_model(model_name, pretrained=False)

    if ckpt.exists():
        model.load_state_dict(
            torch.load(str(ckpt), map_location=device, weights_only=True))
    else:
        # Fallback: pretrained weights (demo mode without fine-tuned checkpoint)
        model = get_model(model_name, pretrained=True)

    model.to(device).eval()
    _MODEL_CACHE[model_name] = model
    return model


# ─── Main pipeline function ───────────────────────────────────────────────

def run_inference(image_input,
                  model_name: str = "ResNet50",
                  n_tta: int = 8,
                  temperature: float = 1.5,
                  cam_method: str = "Ensemble",
                  device: Optional[torch.device] = None) -> InferenceResult:
    """
    Run the full clinical inference pipeline on one image.

    Parameters
    ----------
    image_input : str (file path) | np.ndarray (BGR)
    model_name  : str  backbone name
    n_tta       : int  number of TTA augmentation views (1–8)
    temperature : float  softmax temperature
    cam_method  : str  'GradCAM' | 'GradCAM++' | 'EigenCAM' |
                        'LayerCAM' | 'Ensemble'
    device      : torch.device (auto-detected if None)

    Returns
    -------
    InferenceResult  (fully populated)
    """
    t0 = time.time()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load image ─────────────────────────────────────────────────────
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {image_input}")
    else:
        img_bgr = image_input.copy()

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    img_proc  = preprocess_image(img_bgr)          # Reinhard + crop + resize
    img_rgb   = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB).astype(np.uint8)

    # ── 3. Load model ─────────────────────────────────────────────────────
    model = load_model(model_name, device)

    # ── 4. TTA prediction ─────────────────────────────────────────────────
    tta = TTAPredictor(model, device,
                       n_augments=n_tta, temperature=temperature)
    tta_result = tta.predict(img_proc)

    # ── 5. Grad-CAM ───────────────────────────────────────────────────────
    tensor = _normalise(img_rgb).unsqueeze(0).to(device)

    try:
        target_layer = get_default_target_layer(model, model_name)
        all_cams     = run_all_cam_methods(
            model, tensor, target_layer,
            class_idx=tta_result["predicted_idx"])
        cam_selected = all_cams.get(cam_method,
                                     all_cams.get("Ensemble"))
    except Exception as e:
        print(f"  [WARN] CAM generation failed ({e}). Using blank cam.")
        cam_selected = np.zeros(IMAGE_SIZE, dtype=np.float32)
        all_cams     = {"Ensemble": cam_selected}

    # ── 6. Build overlay images ───────────────────────────────────────────
    h, w = img_proc.shape[:2]
    heatmap_bgr = cam_to_heatmap(cam_selected, (h, w))
    overlay_bgr = overlay_cam_on_image(img_proc, cam_selected, alpha=0.45)
    panel_bgr   = make_panel(img_proc, cam_selected,
                              title=f"{model_name}  |  {tta_result['predicted_class']}  "
                                    f"({tta_result['confidence']*100:.1f}%)")

    # ── 7. Assemble result ────────────────────────────────────────────────
    result = InferenceResult(
        predicted_class  = tta_result["predicted_class"],
        predicted_idx    = tta_result["predicted_idx"],
        is_malignant     = tta_result["is_malignant"],
        verdict          = tta_result["verdict"],
        confidence       = tta_result["confidence"],
        uncertainty      = tta_result["uncertainty"],
        staging          = tta_result["staging"],
        probabilities    = tta_result["probabilities"],
        per_view_probs   = tta_result["per_view_probs"],
        original_b64     = _ndarray_to_b64(img_proc),
        heatmap_b64      = _ndarray_to_b64(heatmap_bgr),
        overlay_b64      = _ndarray_to_b64(overlay_bgr),
        panel_b64        = _ndarray_to_b64(panel_bgr),
        cam_maps         = all_cams,
        model_name       = model_name,
        n_tta_views      = n_tta,
        inference_time_s = time.time() - t0,
        image_shape      = img_proc.shape,
    )

    return result


# ─── Batch inference for test set ─────────────────────────────────────────

def batch_inference(image_paths: List[str],
                    model_name: str = "ResNet50",
                    n_tta: int = 8,
                    device: Optional[torch.device] = None) -> List[InferenceResult]:
    """Run inference on a list of image paths."""
    results = []
    n = len(image_paths)
    for i, p in enumerate(image_paths):
        pct = (i + 1) / n * 100
        print(f"\r  Batch inference [{pct:5.1f}%]  {Path(p).name}", end="", flush=True)
        try:
            r = run_inference(p, model_name=model_name,
                              n_tta=n_tta, device=device)
            results.append(r)
        except Exception as e:
            print(f"\n  [ERROR] {p}: {e}")
    print()
    return results


# ─── CLI entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Clinical inference pipeline")
    parser.add_argument("--image",  required=True, help="Path to image")
    parser.add_argument("--model",  default="ResNet50")
    parser.add_argument("--n_tta",  type=int, default=8)
    parser.add_argument("--cam",    default="Ensemble",
                        choices=["GradCAM", "GradCAM++", "EigenCAM",
                                 "LayerCAM", "Ensemble"])
    args = parser.parse_args()

    result = run_inference(args.image, args.model, args.n_tta,
                           cam_method=args.cam)

    # Save overlay
    out_path = RESULTS_DIR / f"inference_{Path(args.image).stem}.png"
    cv2.imwrite(str(out_path), cv2.imdecode(
        np.frombuffer(base64.b64decode(result.panel_b64), np.uint8),
        cv2.IMREAD_COLOR))

    print(f"\n  Prediction  : {result.predicted_class}")
    print(f"  Verdict     : {result.verdict}")
    print(f"  Confidence  : {result.confidence*100:.2f}%")
    print(f"  Uncertainty : ±{result.uncertainty*100:.2f}%")
    print(f"  Staging     : {result.staging}")
    print(f"  Time        : {result.inference_time_s:.2f}s")
    print(f"  Panel saved : {out_path}")
