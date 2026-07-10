"""
utils/gradcam.py
─────────────────────────────────────────────────────────────────────────────
Grad-CAM visualisation — highlights which regions of the histopathology
image contributed most to the model's decision.

Usage (terminal):
    python utils/gradcam.py --image path/to/img.png --model ResNet50 --layer layer4

Saves overlay to RESULTS_DIR/gradcam_<model>_<imagename>.png
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
from config.config import (CLASSES, IDX_TO_CLASS, MODELS_DIR,
                            RESULTS_DIR, IMAGE_SIZE)
from models.backbones import get_model
from utils.datasets import get_transforms
from preprocessing.preprocess import preprocess_image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._gradients   = None
        self._activations = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, tensor: torch.Tensor,
                 class_idx: int = None) -> np.ndarray:
        """
        Returns Grad-CAM heatmap as (H, W) float in [0, 1].
        """
        self.model.eval()
        output = self.model(tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        grads = self._gradients[0]          # (C, H, W)
        acts  = self._activations[0]        # (C, H, W)

        weights = grads.mean(dim=(1, 2))    # global average pool gradients
        cam     = (weights[:, None, None] * acts).sum(0)
        cam     = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam(img_bgr: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Overlay Grad-CAM heatmap on image."""
    h, w = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)
    return overlay


def _get_target_layer(model, model_name: str, layer_name: str = None):
    """Auto-select last conv layer if not specified."""
    if layer_name:
        parts = layer_name.split(".")
        m = model
        for p in parts:
            m = getattr(m, p)
        return m

    # Auto-select
    for name in ["layer4", "features", "blocks", "stages"]:
        if hasattr(model, name):
            return getattr(model, name)[-1]
    # Fallback: last Conv2d
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise ValueError("Could not auto-select target layer.")


def run_gradcam(image_path: str, model_name: str,
                layer_name: str = None) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = MODELS_DIR / f"{model_name}_best.pth"
    model  = get_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.to(device)

    target_layer = _get_target_layer(model, model_name, layer_name)
    gcam = GradCAM(model, target_layer)

    img_bgr = cv2.imread(image_path)
    img_proc = preprocess_image(img_bgr)
    img_rgb  = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
    tensor   = get_transforms("test")(img_rgb).unsqueeze(0).to(device)

    cam      = gcam.generate(tensor)
    overlay  = overlay_cam(img_proc, cam)

    out_name = f"gradcam_{model_name}_{Path(image_path).stem}.png"
    out_path = RESULTS_DIR / out_name
    cv2.imwrite(str(out_path), overlay)
    print(f"  Grad-CAM saved → {out_path}")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM visualisation")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="ResNet50")
    parser.add_argument("--layer", default=None,
                        help="Target layer name (e.g. layer4, features.denseblock4)")
    args = parser.parse_args()
    run_gradcam(args.image, args.model, args.layer)
