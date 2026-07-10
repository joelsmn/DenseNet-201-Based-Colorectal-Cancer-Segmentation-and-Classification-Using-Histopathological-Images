"""
utils/tta.py
─────────────────────────────────────────────────────────────────────────────
TEST TIME AUGMENTATION (TTA)

Applies N deterministic augmentation transforms at inference time,
averages the softmax probability vectors across all views, and returns
a single calibrated prediction.

TTA transforms (8 canonical views):
  1.  Original (no transform)
  2.  Horizontal flip
  3.  Vertical flip
  4.  90° rotation
  5.  180° rotation
  6.  270° rotation
  7.  Horizontal flip + 90° rotation
  8.  Vertical flip + 90° rotation

For each view, standard ImageNet normalisation is applied after the
geometric transform so the backbone receives properly normalised tensors.

Usage
─────
from utils.tta import TTAPredictor

predictor = TTAPredictor(model, device)
result    = predictor.predict(image_bgr_numpy)
# result['probabilities']  →  averaged, calibrated class probs
# result['predicted_class'] →  argmax class label
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    CLASSES, IDX_TO_CLASS, IMAGE_SIZE,
    MALIGNANT_CLASSES, BENIGN_CLASSES,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_normalise = T.Compose([T.ToTensor(),
                         T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# ─── Individual TTA views ─────────────────────────────────────────────────

def _to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    """Convert uint8 HWC RGB → normalised CHW float tensor."""
    return _normalise(img_rgb)


def _flip_h(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)

def _flip_v(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 0)

def _rot90(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def _rot180(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_180)

def _rot270(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def _fliph_rot90(img: np.ndarray) -> np.ndarray:
    return _rot90(_flip_h(img))

def _flipv_rot90(img: np.ndarray) -> np.ndarray:
    return _rot90(_flip_v(img))


TTA_TRANSFORMS: List[callable] = [
    lambda x: x,          # 1. original
    _flip_h,               # 2. horizontal flip
    _flip_v,               # 3. vertical flip
    _rot90,                # 4. 90°
    _rot180,               # 5. 180°
    _rot270,               # 6. 270°
    _fliph_rot90,          # 7. hflip + 90°
    _flipv_rot90,          # 8. vflip + 90°
]

TTA_NAMES = [
    "Original", "H-Flip", "V-Flip",
    "Rot-90", "Rot-180", "Rot-270",
    "HFlip+Rot90", "VFlip+Rot90",
]


# ─── Temperature scaling (simple calibration) ─────────────────────────────

def temperature_scale(logits: torch.Tensor, temperature: float = 1.5) -> torch.Tensor:
    """
    Divide logits by temperature before softmax.
    T > 1 softens predictions; T < 1 sharpens them.
    Default T=1.5 is a good prior for medical imaging models.
    """
    return logits / temperature


# ─── TTA Predictor ────────────────────────────────────────────────────────

class TTAPredictor:
    """
    Wraps a trained backbone and runs Test Time Augmentation inference.

    Parameters
    ----------
    model       : nn.Module  (already loaded, eval mode)
    device      : torch.device
    n_augments  : int        number of TTA views (1–8, default 8)
    temperature : float      softmax temperature for calibration
    """

    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 n_augments: int = 8,
                 temperature: float = 1.5):
        self.model       = model.eval()
        self.device      = device
        self.n_augments  = min(n_augments, len(TTA_TRANSFORMS))
        self.temperature = temperature
        self.transforms  = TTA_TRANSFORMS[:self.n_augments]
        self.names       = TTA_NAMES[:self.n_augments]

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray) -> Dict:
        """
        Run TTA on a single BGR image array.

        Returns dict:
          predicted_class  : str
          predicted_idx    : int
          is_malignant     : bool
          verdict          : str  ('MALIGNANT' | 'BENIGN')
          confidence       : float  (0–1)
          probabilities    : dict {class_name: float}
          per_view_probs   : list of per-view probability arrays (N_CLASSES,)
          uncertainty      : float  (std of confidence across views)
          staging          : str    clinical staging note
        """
        # Convert to RGB once
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.uint8)

        all_probs = []
        all_logits = []

        for tfm in self.transforms:
            aug_img = tfm(img_rgb)
            tensor  = _to_tensor(aug_img).unsqueeze(0).to(self.device)

            # Forward through backbone (strip head if needed)
            logit = self._forward(tensor)          # (1, NUM_CLASSES)
            all_logits.append(logit)
            prob = F.softmax(
                temperature_scale(logit, self.temperature), dim=1
            ).cpu().numpy()[0]
            all_probs.append(prob)

        # Average across views
        avg_probs = np.stack(all_probs, axis=0).mean(axis=0)  # (NUM_CLASSES,)
        pred_idx  = int(avg_probs.argmax())
        pred_cls  = IDX_TO_CLASS[pred_idx]
        confidence = float(avg_probs[pred_idx])

        # Uncertainty = std of per-view confidence for predicted class
        per_view_conf = [p[pred_idx] for p in all_probs]
        uncertainty   = float(np.std(per_view_conf))

        is_malignant = pred_cls in MALIGNANT_CLASSES
        verdict      = "MALIGNANT" if is_malignant else "BENIGN"
        staging      = _get_staging(pred_cls, confidence)

        return {
            "predicted_class":  pred_cls,
            "predicted_idx":    pred_idx,
            "is_malignant":     is_malignant,
            "verdict":          verdict,
            "confidence":       confidence,
            "probabilities":    {CLASSES[i]: float(avg_probs[i])
                                 for i in range(len(CLASSES))},
            "per_view_probs":   all_probs,
            "uncertainty":      uncertainty,
            "n_augments":       self.n_augments,
            "staging":          staging,
        }

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward through any backbone returning (1, NUM_CLASSES) logits.
        Simply calls model(tensor) directly — the fine-tuned head is already
        attached with the correct number of output classes.
        Handles dict outputs from NeurosymbolicGraphTransformer.
        """
        try:
            out = self.model(tensor)
            if isinstance(out, dict):
                out = out.get("logits_multiclass", list(out.values())[0])
            if out.dim() == 1:
                out = out.unsqueeze(0)
            return out
        except Exception as exc:
            import traceback; traceback.print_exc()
            return torch.zeros(1, len(CLASSES), device=tensor.device)


# ─── Clinical staging logic ───────────────────────────────────────────────

def _get_staging(predicted_class: str, confidence: float) -> str:
    """
    Map predicted class to a brief clinical staging note for display.
    """
    staging_map = {
        "Benign": (
            "Non-neoplastic. No dysplasia detected. "
            "Routine surveillance recommended."
        ),
        "Low Grade IN": (
            "Low-grade intraepithelial neoplasia (LGIN). "
            "Mild dysplasia present. Endoscopic follow-up in 3 years."
        ),
        "High Grade IN": (
            "High-grade intraepithelial neoplasia (HGIN). "
            "Severe dysplasia. Urgent endoscopic resection advised."
        ),
        "Polyp": (
            "Colorectal polyp detected. Histological subtyping required. "
            "Polypectomy + follow-up colonoscopy recommended."
        ),
        "Serrated Adenoma": (
            "Sessile serrated adenoma / polyp. Malignant potential present. "
            "Complete resection and 1-year surveillance colonoscopy."
        ),
        "Adenocarcinoma": (
            "Adenocarcinoma detected. Staging workup required (CT / MRI). "
            "Multidisciplinary team referral immediately."
        ),
    }
    note = staging_map.get(predicted_class, "Classification inconclusive.")
    conf_note = (
        "HIGH" if confidence > 0.80 else
        "MODERATE" if confidence > 0.55 else
        "LOW"
    )
    return f"[{conf_note} CONFIDENCE]  {note}"
