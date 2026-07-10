"""
utils/inference.py
─────────────────────────────────────────────────────────────────────────────
Single-image inference utility.

Usage (from VS Code terminal):
    python utils/inference.py --image path/to/image.png --model ResNet50

Outputs:
  • Predicted class (6-class)
  • Benign / Malignant verdict
  • Per-class probability table
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    CLASSES, IDX_TO_CLASS, MODELS_DIR,
    MALIGNANT_CLASSES, IMAGE_SIZE,
)
from models.backbones import get_model
from utils.datasets import get_transforms
from preprocessing.preprocess import preprocess_image


def predict_single(image_path: str,
                   model_name: str = "ResNet50",
                   device: torch.device = None) -> dict:
    """
    Run inference on a single image.

    Parameters
    ----------
    image_path : str  path to the image file
    model_name : str  name of the trained backbone
    device     : torch.device  (auto-detected if None)

    Returns
    -------
    dict with keys: predicted_class, is_malignant, probabilities
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = MODELS_DIR / f"{model_name}_best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = get_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.to(device).eval()

    # Load + preprocess image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_proc = preprocess_image(img_bgr)
    img_rgb  = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)

    transform = get_transforms("test")
    tensor    = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx   = int(probs.argmax())
    pred_class = IDX_TO_CLASS[pred_idx]
    is_malignant = pred_class in MALIGNANT_CLASSES

    result = {
        "predicted_class": pred_class,
        "is_malignant":    is_malignant,
        "verdict":         "MALIGNANT" if is_malignant else "BENIGN",
        "confidence":      float(probs[pred_idx]),
        "probabilities":   {CLASSES[i]: float(probs[i])
                            for i in range(len(CLASSES))},
    }
    return result


def _print_result(result: dict, model_name: str) -> None:
    print("\n" + "=" * 50)
    print(f"  Model       : {model_name}")
    print(f"  Prediction  : {result['predicted_class']}")
    print(f"  Verdict     : {result['verdict']}")
    print(f"  Confidence  : {result['confidence']*100:.2f}%")
    print("\n  Per-class probabilities:")
    print("  " + "-" * 36)
    for cls, prob in sorted(result["probabilities"].items(),
                             key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:20s} {prob*100:5.2f}%  {bar}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-image colorectal cancer prediction")
    parser.add_argument("--image", required=True,
                        help="Path to input histopathology image")
    parser.add_argument("--model", default="ResNet50",
                        help="Backbone model name (default: ResNet50)")
    args = parser.parse_args()

    result = predict_single(args.image, args.model)
    _print_result(result, args.model)
