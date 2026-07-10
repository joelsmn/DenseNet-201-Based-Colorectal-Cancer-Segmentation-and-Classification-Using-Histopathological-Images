"""
models/backbones.py
─────────────────────────────────────────────────────────────────────────────
Backbone factory for the 15-model comparative sweep referenced throughout
the pipeline (training/trainer.py, evaluation/evaluate.py,
inference/pipeline.py, utils/gradcam.py, utils/inference.py) via:

    from models.backbones import get_model
    model = get_model(model_name, pretrained=True)

Matches config.config.MODEL_NAMES exactly:

    SwinTransformer, VisionTransformer, ConvNeXt, DenseNet121,
    DenseNet201, EfficientNetB4, EfficientNetV2, InceptionResNetV2,
    Xception, ResNet50, ResNet101, SEResNet, CBAMResNet, BEiT, DeiT

Design notes
────────────
- Standard CNN/transformer architectures (ResNet, DenseNet,
  EfficientNet, EfficientNetV2, ConvNeXt, ViT, Swin) are built from
  torchvision so no extra dependency is required for 11 of the 15
  models.
- InceptionResNetV2, Xception, BEiT, and DeiT are not in torchvision
  and are built via `timm` (already in requirements.txt). If timm is
  not installed, these four raise a clear ImportError telling the user
  to `pip install timm` rather than failing silently.
- SEResNet and CBAMResNet are plain ResNet50 with, respectively, the
  SqueezeExcitation and CBAM modules from models/attention.py inserted
  immediately after the stem (conv1) — the same insertion point used
  for the proposed CCA module on DenseNet201 elsewhere in this
  pipeline, so the "attention at the stem" comparison in the ATR's
  Table 1 discussion is architecturally apples-to-apples.
- Every returned model keeps at least one of the attribute names
  utils/gradcam.py's `_get_target_layer()` auto-selects on
  ("layer4", "features", "blocks", "stages") wherever the underlying
  architecture naturally has one, so Grad-CAM keeps working without
  changes to gradcam.py.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from models.xception import get_xception


sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import CLASSES
from models.attention import SqueezeExcitation, CBAM

NUM_CLASSES = len(CLASSES)


# ─────────────────────────────────────────────────────────────────────────
#  Torchvision-backed builders
# ─────────────────────────────────────────────────────────────────────────

def _tv_weights(pretrained: bool, enum_cls):
    return enum_cls.DEFAULT if pretrained else None


def _build_resnet(depth: int, pretrained: bool, num_classes: int):
    import torchvision.models as tv
    if depth == 50:
        model = tv.resnet50(weights=_tv_weights(pretrained, tv.ResNet50_Weights))
    elif depth == 101:
        model = tv.resnet101(weights=_tv_weights(pretrained, tv.ResNet101_Weights))
    else:
        raise ValueError(depth)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_densenet(depth: int, pretrained: bool, num_classes: int):
    import torchvision.models as tv
    if depth == 121:
        model = tv.densenet121(weights=_tv_weights(pretrained, tv.DenseNet121_Weights))
    elif depth == 201:
        model = tv.densenet201(weights=_tv_weights(pretrained, tv.DenseNet201_Weights))
    else:
        raise ValueError(depth)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def _build_efficientnet_b4(pretrained: bool, num_classes: int):
    import torchvision.models as tv
    model = tv.efficientnet_b4(
        weights=_tv_weights(pretrained, tv.EfficientNet_B4_Weights))
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, num_classes)
    return model


def _build_efficientnet_v2(pretrained: bool, num_classes: int):
    import torchvision.models as tv
    model = tv.efficientnet_v2_s(
        weights=_tv_weights(pretrained, tv.EfficientNet_V2_S_Weights))
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, num_classes)
    return model


def _build_convnext(pretrained: bool, num_classes: int):
    import torchvision.models as tv
    model = tv.convnext_base(
        weights=_tv_weights(pretrained, tv.ConvNeXt_Base_Weights))
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, num_classes)
    return model


def _build_vit(pretrained: bool, num_classes: int):
    import torchvision.models as tv
    model = tv.vit_b_16(weights=_tv_weights(pretrained, tv.ViT_B_16_Weights))
    in_feat = model.heads.head.in_features
    model.heads.head = nn.Linear(in_feat, num_classes)
    return model


def _build_swin(pretrained: bool, num_classes: int):
    import torchvision.models as tv
    model = tv.swin_b(weights=_tv_weights(pretrained, tv.Swin_B_Weights))
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


# ─────────────────────────────────────────────────────────────────────────
#  timm-backed builders (architectures absent from torchvision)
# ─────────────────────────────────────────────────────────────────────────

_TIMM_HELP = (
    "'{name}' requires the `timm` package (not in torchvision). "
    "Install it with:  pip install timm"
)


def _build_timm(timm_name: str, display_name: str, pretrained: bool,
                num_classes: int):
    try:
        import timm
    except ImportError as e:
        raise ImportError(_TIMM_HELP.format(name=display_name)) from e
    return timm.create_model(timm_name, pretrained=pretrained,
                             num_classes=num_classes)


# ─────────────────────────────────────────────────────────────────────────
#  SE-ResNet50 / CBAM-ResNet50 (attention inserted at the stem, matching
#  the CCA insertion point used elsewhere in this pipeline)
# ─────────────────────────────────────────────────────────────────────────

def _build_attention_resnet(attn_cls, pretrained: bool, num_classes: int):
    import torchvision.models as tv
    model = tv.resnet50(weights=_tv_weights(pretrained, tv.ResNet50_Weights))
    stem = model.conv1
    model.conv1 = nn.Sequential(stem, attn_cls(stem.out_channels))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ─────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────

_BUILDERS = {
    "ResNet50":          lambda p, n: _build_resnet(50, p, n),
    "ResNet101":         lambda p, n: _build_resnet(101, p, n),
    "DenseNet121":       lambda p, n: _build_densenet(121, p, n),
    "DenseNet201":       lambda p, n: _build_densenet(201, p, n),
    "EfficientNetB4":    lambda p, n: _build_efficientnet_b4(p, n),
    "EfficientNetV2":    lambda p, n: _build_efficientnet_v2(p, n),
    "ConvNeXt":          lambda p, n: _build_convnext(p, n),
    "VisionTransformer": lambda p, n: _build_vit(p, n),
    "SwinTransformer":   lambda p, n: _build_swin(p, n),
    "SEResNet":          lambda p, n: _build_attention_resnet(
        SqueezeExcitation, p, n),
    "CBAMResNet":        lambda p, n: _build_attention_resnet(CBAM, p, n),
    "InceptionResNetV2": lambda p, n: _build_timm(
        "inception_resnet_v2", "InceptionResNetV2", p, n),
    "Xception":          lambda p, n: get_xception(num_classes=n, pretrained=p),
    "BEiT":              lambda p, n: _build_timm(
        "beit_base_patch16_224", "BEiT", p, n),
    "DeiT":              lambda p, n: _build_timm(
        "deit_base_patch16_224", "DeiT", p, n),
}


def get_model(model_name: str, pretrained: bool = True,
             num_classes: int = None) -> nn.Module:
    """
    Build one of the 15 comparative backbones by name (see
    config.config.MODEL_NAMES for the exact list).

    Args:
        model_name: one of the keys in `_BUILDERS` (== MODEL_NAMES).
        pretrained: load ImageNet weights if True.
        num_classes: defaults to len(config.config.CLASSES).
    """
    if model_name not in _BUILDERS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Available: "
            f"{list(_BUILDERS.keys())}")
    n = num_classes or NUM_CLASSES
    return _BUILDERS[model_name](pretrained, n)


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    print(f"{'Model':20s}  {'Params':>12s}  {'Output':>14s}  Status")
    print("-" * 65)
    for name in _BUILDERS:
        try:
            model = get_model(name, pretrained=False)
            model.eval()
            with torch.no_grad():
                out = model(x)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"{name:20s}  {n_params:12,d}  {str(tuple(out.shape)):>14s}  OK")
        except ImportError as e:
            print(f"{name:20s}  {'—':>12s}  {'—':>14s}  SKIPPED ({e})")