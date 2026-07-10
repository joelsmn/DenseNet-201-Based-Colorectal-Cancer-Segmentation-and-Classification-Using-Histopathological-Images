"""
utils/gradcam_engine.py  (v2 — fixed)
"""

import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import IMAGE_SIZE

OUTPUT_SIZE = IMAGE_SIZE


def _norm(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    mn, mx = cam.min(), cam.max()
    if mx - mn < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - mn) / (mx - mn)


def _resize_cam(cam: np.ndarray) -> np.ndarray:
    h, w = OUTPUT_SIZE
    if cam.shape == (h, w):
        return cam.astype(np.float32)
    return cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


class _CAMBase:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._act_list    = []
        self._grad_list   = []
        self._fwd_hook    = target_layer.register_forward_hook(self._save_act)
        self._bwd_hook    = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        self._act_list.append(out.detach().cpu())

    def _save_grad(self, module, gin, gout):
        g = gout[0]
        if g is not None:
            self._grad_list.append(g.detach().cpu())

    def _get_act(self):
        return self._act_list[-1] if self._act_list else None

    def _get_grad(self):
        return self._grad_list[-1] if self._grad_list else None

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def _blank(self):
        return np.zeros(OUTPUT_SIZE, dtype=np.float32)

    def _make_spatial(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3:
            return t
        if t.dim() == 2:
            N, C = t.shape
            side = int(N ** 0.5)
            if side * side == N and C > N:
                return t.permute(1, 0).reshape(C, side, side)
            elif N > C:
                return t.mean(0).unsqueeze(-1).unsqueeze(-1)
            else:
                side2 = int(C ** 0.5)
                if side2 * side2 == C:
                    return t.permute(1, 0).reshape(N, side2, side2)
                return t.unsqueeze(-1).unsqueeze(-1)
        return t.unsqueeze(-1).unsqueeze(-1)


class GradCAM(_CAMBase):
    def generate(self, tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        tensor = tensor.requires_grad_(True)
        try:
            output = self.model(tensor)
            if isinstance(output, dict):
                output = output.get("logits_multiclass", list(output.values())[0])
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            self.model.zero_grad()
            output[0, class_idx].backward()
            act  = self._get_act()
            grad = self._get_grad()
            if act is None:
                return self._blank()
            act = self._make_spatial(act)
            if grad is None:
                weights = torch.ones(act.shape[0]) / act.shape[0]
            else:
                grad    = self._make_spatial(grad)
                weights = grad.mean(dim=(1, 2))
            cam = F.relu((weights[:, None, None] * act).sum(0))
            return _resize_cam(_norm(cam.numpy()))
        except Exception as e:
            print(f"    [GradCAM error] {e}")
            return self._blank()


class GradCAMPlusPlus(_CAMBase):
    def generate(self, tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        tensor = tensor.requires_grad_(True)
        try:
            output = self.model(tensor)
            if isinstance(output, dict):
                output = output.get("logits_multiclass", list(output.values())[0])
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            self.model.zero_grad()
            output[0, class_idx].backward()
            act  = self._get_act()
            grad = self._get_grad()
            if act is None:
                return self._blank()
            act = self._make_spatial(act)
            if grad is None:
                return self._blank()
            grad      = self._make_spatial(grad)
            grads_sq  = grad ** 2
            grads_cu  = grad ** 3
            alpha_den = (2 * grads_sq + (grads_cu * act).sum(dim=(1, 2), keepdim=True) + 1e-8)
            alpha     = grads_sq / alpha_den
            weights   = (alpha * F.relu(grad)).sum(dim=(1, 2))
            cam       = F.relu((weights[:, None, None] * act).sum(0))
            return _resize_cam(_norm(cam.numpy()))
        except Exception as e:
            print(f"    [GradCAM++ error] {e}")
            return self._blank()


class EigenCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model     = model
        self._act_list = []
        self._fwd_hook = target_layer.register_forward_hook(self._save_act)

    def _save_act(self, module, inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        self._act_list.append(out.detach().cpu())

    def remove(self):
        self._fwd_hook.remove()

    def _make_spatial(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4: t = t[0]
        if t.dim() == 3: return t
        if t.dim() == 2:
            N, C = t.shape
            side = int(N ** 0.5)
            if side * side == N and C > N:
                return t.permute(1, 0).reshape(C, side, side)
            return t.mean(0).unsqueeze(-1).unsqueeze(-1)
        return t.unsqueeze(-1).unsqueeze(-1)

    def generate(self, tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        try:
            with torch.no_grad():
                self.model(tensor)
            if not self._act_list:
                return np.zeros(OUTPUT_SIZE, dtype=np.float32)
            act = self._make_spatial(self._act_list[-1])
            C, H, W = act.shape
            flat = act.reshape(C, -1).numpy().astype(np.float32)
            flat -= flat.mean(axis=1, keepdims=True)
            # Transpose so SVD rows <= cols (required for stable Vt[0])
            if flat.shape[0] > flat.shape[1]:
                flat_t = flat.T
            else:
                flat_t = flat
            try:
                _, _, Vt = np.linalg.svd(flat_t, full_matrices=False)
                proj = (flat.T @ Vt[0]) if flat.shape[0] <= flat.shape[1] else (flat.T @ Vt[0])
                proj = proj.reshape(-1)[:H * W].reshape(H, W)
            except (np.linalg.LinAlgError, ValueError):
                proj = flat.mean(axis=0).reshape(H, W)
            return _resize_cam(_norm(np.maximum(proj, 0)))
        except Exception as e:
            print(f"    [EigenCAM error] {e}")
            return np.zeros(OUTPUT_SIZE, dtype=np.float32)


class LayerCAM(_CAMBase):
    def generate(self, tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        tensor = tensor.requires_grad_(True)
        try:
            output = self.model(tensor)
            if isinstance(output, dict):
                output = output.get("logits_multiclass", list(output.values())[0])
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            self.model.zero_grad()
            output[0, class_idx].backward()
            act  = self._get_act()
            grad = self._get_grad()
            if act is None:
                return self._blank()
            act     = self._make_spatial(act)
            weights = F.relu(self._make_spatial(grad)) if grad is not None else torch.ones_like(act)
            cam     = F.relu((weights * act).sum(0))
            return _resize_cam(_norm(cam.numpy()))
        except Exception as e:
            print(f"    [LayerCAM error] {e}")
            return self._blank()


def cam_to_heatmap(cam: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h, w    = size
    resized = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    resized = np.clip(resized, 0, 1)
    return cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)


def overlay_cam_on_image(img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w    = img_bgr.shape[:2]
    heatmap = cam_to_heatmap(cam, (h, w))
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)


def make_panel(img_bgr: np.ndarray, cam: np.ndarray, title: str = "") -> np.ndarray:
    h, w    = img_bgr.shape[:2]
    heatmap = cam_to_heatmap(cam, (h, w))
    overlay = overlay_cam_on_image(img_bgr, cam)

    def _label(img, text):
        out = img.copy()
        cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        return out

    panel = np.concatenate([_label(img_bgr, "Original"), _label(heatmap, "Heatmap"), _label(overlay, "Overlay")], axis=1)
    if title:
        banner = np.zeros((30, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,230,255), 1, cv2.LINE_AA)
        panel = np.vstack([banner, panel])
    return panel


def get_default_target_layer(model: nn.Module, model_name: str = "") -> nn.Module:
    def _last_of(attr):
        m = getattr(model, attr, None)
        if m is None: return None
        if hasattr(m, '__getitem__'):
            try: return m[-1]
            except Exception: pass
        return m

    name = model_name.lower()

    if any(x in name for x in ["resnet", "seresnet", "cbam"]):
        l = _last_of("layer4")
        if l is not None: return l

    if "densenet" in name:
        feat = getattr(model, "features", None)
        if feat is not None:
            for child_name in reversed([n for n, _ in feat.named_children()]):
                if "denseblock" in child_name:
                    return getattr(feat, child_name)
            return feat

    if "efficientnet" in name:
        feats = getattr(model, "features", None)
        if feats is not None: return feats[-1]

    if "convnext" in name:
        s = getattr(model, "stages", None)
        if s is not None: return s[-1]

    if "xception" in name:
        for attr in ["block12", "conv4"]:
            m = getattr(model, attr, None)
            if m is not None: return m

    if "inception" in name:
        for attr in ["mixed_7a", "block8", "conv2d_7b"]:
            m = getattr(model, attr, None)
            if m is not None: return m

    if any(x in name for x in ["vit", "beit", "deit", "vision"]):
        blocks = getattr(model, "blocks", None)
        if blocks is not None and len(blocks) > 0: return blocks[-1]
        norm = getattr(model, "norm", None)
        if norm is not None: return norm

    if "swin" in name:
        layers = getattr(model, "layers", None)
        if layers is not None and len(layers) > 0: return layers[-1]
        norm = getattr(model, "norm", None)
        if norm is not None: return norm

    last_conv = last_norm = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d): last_conv = m
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)): last_norm = m

    if last_conv is not None: return last_conv
    if last_norm is not None: return last_norm
    raise ValueError(f"Cannot find target layer for '{model_name}'.")


def run_all_cam_methods(model: nn.Module, tensor: torch.Tensor,
                        target_layer: nn.Module, class_idx: Optional[int] = None) -> dict:
    results = {}
    for method_name, MethodClass in [("GradCAM", GradCAM), ("GradCAM++", GradCAMPlusPlus), ("LayerCAM", LayerCAM)]:
        obj = MethodClass(model, target_layer)
        cam = obj.generate(tensor.detach().clone().requires_grad_(True), class_idx)
        obj.remove()
        results[method_name] = _resize_cam(cam)

    ecam_obj = EigenCAM(model, target_layer)
    ecam     = ecam_obj.generate(tensor.detach().clone())
    ecam_obj.remove()
    results["EigenCAM"] = _resize_cam(ecam)

    stack = np.stack(list(results.values()), axis=0)
    results["Ensemble"] = _norm(stack.mean(axis=0))
    return results
