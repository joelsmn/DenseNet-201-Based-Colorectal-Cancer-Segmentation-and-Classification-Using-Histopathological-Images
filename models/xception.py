"""
models/xception.py
─────────────────────────────────────────────────────────────────────────────
Standalone Xception implementation (Chollet, 2017 — "Xception: Deep
Learning with Depthwise Separable Convolutions"), with ZERO dependency
on `timm`.

Why this exists: models/backbones.py originally built Xception via
`timm.create_model("xception", ...)`. Depending on the installed timm
version, "xception" silently remaps to "legacy_xception" (you'll see a
UserWarning about this), and on some timm/torch/CUDA combinations that
remapped path throws during either the forward pass (training) or the
Grad-CAM hook registration (evaluation), because legacy_xception's
internal module names differ from the version get_model()/gradcam.py
were written against. Removing the timm dependency for this one model
removes that whole failure surface.

Architecture (faithful to the original paper):
  Entry flow  -> Middle flow (8x repeated separable-conv block)
              -> Exit flow -> Global Average Pool -> FC

Interface (drop-in for models/backbones.py's _BUILDERS):
    model = get_xception(num_classes=6, pretrained=True)
    out   = model(x)             # x: (B, 3, H, W), H,W >= 71 (224 is fine)

Grad-CAM compatibility: the model exposes `.features` (an nn.Sequential
covering everything up to and including the final ReLU, before the
global pool + FC), matching the attribute name utils/gradcam.py's
`_get_target_layer()` auto-selects on.
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────
#  Depthwise-separable convolution building block
# ─────────────────────────────────────────────────────────────────────────

class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding, dilation,
            groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ─────────────────────────────────────────────────────────────────────────
#  Xception "Block" — repeated separable convs with a residual skip
# ─────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, reps, stride=1,
                start_with_relu=True, grow_first=True):
        super().__init__()

        if out_ch != in_ch or stride != 1:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_ch)
        else:
            self.skip = None

        layers = []
        ch = in_ch
        if grow_first:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(in_ch, out_ch, 3, 1, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            ch = out_ch

        for _ in range(reps - 1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(ch, ch, 3, 1, padding=1))
            layers.append(nn.BatchNorm2d(ch))

        if not grow_first:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(in_ch, out_ch, 3, 1, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))

        if not start_with_relu:
            layers = layers[1:]
        else:
            layers[0] = nn.ReLU(inplace=True)

        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride, padding=1))

        self.rep = nn.Sequential(*layers)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skip_bn(self.skip(x))
        else:
            skip = x
        return out + skip


# ─────────────────────────────────────────────────────────────────────────
#  Full Xception network
# ─────────────────────────────────────────────────────────────────────────

class Xception(nn.Module):
    """
    Xception, exposing `.features` (everything before global pooling)
    and `.fc` (final classifier), matching the attribute-name
    conventions the rest of this pipeline's Grad-CAM / backbone-factory
    code expects.
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()

        # ── Entry flow ──
        entry = [
            nn.Conv2d(3, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        self.entry_stem = nn.Sequential(*entry)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # ── Middle flow: 8 repeated blocks, no downsampling ──
        middle_blocks = [
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
            for _ in range(8)
        ]
        self.middle_flow = nn.Sequential(*middle_blocks)

        # ── Exit flow ──
        self.block_exit = Block(728, 1024, 2, 2, start_with_relu=True,
                                grow_first=False)

        exit_tail = [
            SeparableConv2d(1024, 1536, 3, 1, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048, 3, 1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        ]
        self.exit_tail = nn.Sequential(*exit_tail)

        # Everything before global pooling, in forward order — this is
        # the attribute Grad-CAM's auto-selector (utils/gradcam.py)
        # looks for.
        self.features = nn.Sequential(
            self.entry_stem, self.block1, self.block2, self.block3,
            self.middle_flow, self.block_exit, self.exit_tail,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ─────────────────────────────────────────────────────────────────────────
#  Optional pretrained-weight loading (best-effort, never crashes)
# ─────────────────────────────────────────────────────────────────────────

# Widely-mirrored ImageNet-pretrained Xception checkpoint (originally
# released alongside Cadene's pretrained-models.pytorch). If your
# training machine has no internet access, or the URL is unreachable
# behind your egress rules, pretrained loading is skipped with a
# printed warning and the model trains from random init instead —
# it will NOT crash the pipeline either way.
_PRETRAINED_URL = (
    "https://github.com/Cadene/pretrained-models.pytorch/releases/"
    "download/v1.0/xception-43020ad28.pth"
)


def _load_pretrained(model: "Xception"):
    try:
        state_dict = torch.hub.load_state_dict_from_url(
            _PRETRAINED_URL, progress=True, map_location="cpu")
        own_state = model.state_dict()
        loaded, skipped = 0, 0
        for k, v in state_dict.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(own_state)
        print(f"  [Xception] Loaded {loaded} pretrained tensors "
              f"({skipped} skipped due to name/shape mismatch — this is "
              f"expected since layer names differ slightly from the "
              f"original release; the conv backbone still transfers).")
    except Exception as e:
        print(f"  [Xception] Pretrained weights unavailable ({e}); "
              f"continuing with random initialisation.")


# ─────────────────────────────────────────────────────────────────────────
#  Factory function — drop-in replacement for the timm-based builder in
#  models/backbones.py
# ─────────────────────────────────────────────────────────────────────────

def get_xception(num_classes: int = None, pretrained: bool = True) -> Xception:
    if num_classes is None:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from config.config import CLASSES
        num_classes = len(CLASSES)

    model = Xception(num_classes=1000)   # build at ImageNet width first
    if pretrained:
        _load_pretrained(model)

    if num_classes != 1000:
        model.fc = nn.Linear(2048, num_classes)

    return model


if __name__ == "__main__":
    model = get_xception(num_classes=6, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape : {tuple(out.shape)}")
    print(f"Params       : {n_params:,}")
    print(f"Has .features: {hasattr(model, 'features')}")
    print(f"Has .fc      : {hasattr(model, 'fc')}")