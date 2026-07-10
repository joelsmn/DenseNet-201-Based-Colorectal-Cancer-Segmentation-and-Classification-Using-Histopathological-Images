"""
segmentation/unet_variants.py
─────────────────────────────────────────────────────────────────────────────
Segmentation backbone zoo for the U-Net ablation study (ATR Reviewer #1,
comment 7 — "effectiveness and necessity of the U-Net segmentation step
prior to classification requires further validation").

Implements, in addition to the baseline UNet (segmentation/unet.py):

  1. UNetPlusPlus     — Zhou et al. 2018. Nested, dense skip pathways.
  2. AttentionUNet     — Oktay et al. 2018. Additive attention gates on
                          the skip connections.
  3. DeepLabV3Plus     — Chen et al. 2018. Atrous Spatial Pyramid
                          Pooling + separable-conv decoder. Backed by
                          torchvision's deeplabv3_resnet50 when available,
                          with a lightweight from-scratch fallback.
  4. TransUNetLite     — Chen et al. 2021. CNN encoder + a Transformer
                          bottleneck (multi-head self-attention over the
                          flattened bottleneck feature map) + CNN decoder.
                          A parameter-matched "lite" reproduction suitable
                          for a 224x224, single-GPU ablation budget rather
                          than the full ViT-B/16 encoder of the paper.
  5. SwinUNetLite      — Cao et al. 2021. Encoder/decoder built from
                          windowed multi-head self-attention (Swin) blocks
                          instead of convolutions. A "lite" 2-stage
                          reproduction for ablation-budget purposes.
  6. NNUNetStyleUNet   — Isensee et al. 2021 (nnU-Net). Not the full
                          self-configuring framework (which selects its
                          own patch size / stage count / augmentation from
                          dataset fingerprinting) — that is a *pipeline*,
                          not a fixed architecture, and is out of scope
                          for a single-GPU ablation. This class reproduces
                          nnU-Net's *architectural* signature relevant to
                          the ablation: deep supervision at 3 decoder
                          resolutions + instance normalisation + leaky
                          ReLU, so its contribution can be measured
                          against the plain U-Net under identical data
                          and identical training budget.

All variants share the interface of segmentation/unet.py's UNet:
    model = Variant(in_channels=3, out_channels=1)
    pred  = model(x)                 # (B, 1, H, W) in [0, 1]
    (NNUNetStyleUNet additionally returns aux deep-supervision heads
     when model.training is True — see class docstring.)

This uniform interface lets evaluation/ablation_unet_variants.py swap
segmentation backbones into the downstream graph-construction step
without touching any other pipeline code.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import UNET_FILTERS
from segmentation.unet import DoubleConv, DiceBCELoss  # reuse building block + loss


# ─────────────────────────────────────────────────────────────────────────
#  1. UNet++ (nested dense skip pathways)
# ─────────────────────────────────────────────────────────────────────────

class UNetPlusPlus(nn.Module):
    """
    UNet++ with 4 down-sampling levels and dense nested skip pathways
    X^{i,j}. Deep supervision is disabled by default (single final head)
    to keep the loss/metric interface identical to plain U-Net; set
    `deep_supervision=True` to average all four decoder-level outputs.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 filters=None, deep_supervision: bool = False):
        super().__init__()
        f = filters or UNET_FILTERS  # [64,128,256,512,1024]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=False)

        # Encoder column j=0
        self.conv0_0 = DoubleConv(in_channels, f[0])
        self.conv1_0 = DoubleConv(f[0], f[1])
        self.conv2_0 = DoubleConv(f[1], f[2])
        self.conv3_0 = DoubleConv(f[2], f[3])
        self.conv4_0 = DoubleConv(f[3], f[4])

        # Nested columns
        self.conv0_1 = DoubleConv(f[0] + f[1], f[0])
        self.conv1_1 = DoubleConv(f[1] + f[2], f[1])
        self.conv2_1 = DoubleConv(f[2] + f[3], f[2])
        self.conv3_1 = DoubleConv(f[3] + f[4], f[3])

        self.conv0_2 = DoubleConv(f[0] * 2 + f[1], f[0])
        self.conv1_2 = DoubleConv(f[1] * 2 + f[2], f[1])
        self.conv2_2 = DoubleConv(f[2] * 2 + f[3], f[2])

        self.conv0_3 = DoubleConv(f[0] * 3 + f[1], f[0])
        self.conv1_3 = DoubleConv(f[1] * 3 + f[2], f[1])

        self.conv0_4 = DoubleConv(f[0] * 4 + f[1], f[0])

        if deep_supervision:
            self.final1 = nn.Conv2d(f[0], out_channels, 1)
            self.final2 = nn.Conv2d(f[0], out_channels, 1)
            self.final3 = nn.Conv2d(f[0], out_channels, 1)
            self.final4 = nn.Conv2d(f[0], out_channels, 1)
        else:
            self.final = nn.Conv2d(f[0], out_channels, 1)

    def _up_to(self, x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="bilinear",
                               align_corners=False)
        return x

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat(
            [x0_0, self._up_to(self.up(x1_0), x0_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat(
            [x1_0, self._up_to(self.up(x2_0), x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat(
            [x0_0, x0_1, self._up_to(self.up(x1_1), x0_0)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat(
            [x2_0, self._up_to(self.up(x3_0), x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat(
            [x1_0, x1_1, self._up_to(self.up(x2_1), x1_0)], 1))
        x0_3 = self.conv0_3(torch.cat(
            [x0_0, x0_1, x0_2, self._up_to(self.up(x1_2), x0_0)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat(
            [x3_0, self._up_to(self.up(x4_0), x3_0)], 1))
        x2_2 = self.conv2_2(torch.cat(
            [x2_0, x2_1, self._up_to(self.up(x3_1), x2_0)], 1))
        x1_3 = self.conv1_3(torch.cat(
            [x1_0, x1_1, x1_2, self._up_to(self.up(x2_2), x1_0)], 1))
        x0_4 = self.conv0_4(torch.cat(
            [x0_0, x0_1, x0_2, x0_3, self._up_to(self.up(x1_3), x0_0)], 1))

        if self.deep_supervision:
            out = (self.final1(x0_1) + self.final2(x0_2) +
                   self.final3(x0_3) + self.final4(x0_4)) / 4.0
        else:
            out = self.final(x0_4)
        return torch.sigmoid(out)


# ─────────────────────────────────────────────────────────────────────────
#  2. Attention U-Net (additive attention gates)
# ─────────────────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018)."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=True),
            nn.BatchNorm2d(inter_ch))
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=True),
            nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear",
                                align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class _AttnUpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.gate = AttentionGate(out_ch, skip_ch, out_ch // 2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                               align_corners=False)
        skip = self.gate(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class AttentionUNet(nn.Module):
    """Standard U-Net encoder + attention-gated skip connections."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 filters=None):
        super().__init__()
        f = filters or UNET_FILTERS
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_channels, f[0])
        self.enc2 = DoubleConv(f[0], f[1])
        self.enc3 = DoubleConv(f[1], f[2])
        self.enc4 = DoubleConv(f[2], f[3])
        self.bottleneck = DoubleConv(f[3], f[4])

        self.dec4 = _AttnUpBlock(f[4], f[3], f[3])
        self.dec3 = _AttnUpBlock(f[3], f[2], f[2])
        self.dec2 = _AttnUpBlock(f[2], f[1], f[1])
        self.dec1 = _AttnUpBlock(f[1], f[0], f[0])

        self.out_conv = nn.Conv2d(f[0], out_channels, 1)

    def forward(self, x):
        s1 = self.enc1(x);   x = self.pool(s1)
        s2 = self.enc2(x);   x = self.pool(s2)
        s3 = self.enc3(x);   x = self.pool(s3)
        s4 = self.enc4(x);   x = self.pool(s4)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return torch.sigmoid(self.out_conv(x))


# ─────────────────────────────────────────────────────────────────────────
#  3. DeepLabV3+ (ASPP)
# ─────────────────────────────────────────────────────────────────────────

class _ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation,
                       dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class _ASPPPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        y = self.conv(self.pool(x))
        return F.interpolate(y, size=size, mode="bilinear",
                              align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                           nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        ])
        for r in rates:
            self.branches.append(_ASPPConv(in_ch, out_ch, r))
        self.branches.append(_ASPPPooling(in_ch, out_ch))
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout(0.3))

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.project(torch.cat(feats, dim=1))


class DeepLabV3Plus(nn.Module):
    """
    Lightweight from-scratch DeepLabV3+: small CNN encoder (stride-16) +
    ASPP + a decoder that fuses a low-level skip feature, matching the
    interface of the other segmentation variants (binary mask output,
    same input resolution).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 base_ch: int = 64):
        super().__init__()
        c = base_ch
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True))              # /2
        self.low_level = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True))               # /2
        self.stage2 = nn.Sequential(
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True))            # /4
        self.stage3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True))            # /8
        self.stage4 = nn.Sequential(
            nn.Conv2d(c * 4, c * 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 8), nn.ReLU(inplace=True))            # /16

        self.aspp = ASPP(c * 8, out_ch=256)

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(c, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1))

    def forward(self, x):
        in_size = x.shape[-2:]
        x = self.stem(x)
        low = self.low_level(x)          # /2, c channels
        x = self.stage2(low)             # /4
        x = self.stage3(x)               # /8
        x = self.stage4(x)               # /16
        x = self.aspp(x)                 # /16, 256 ch
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear",
                           align_corners=False)
        low = self.low_level_proj(low)
        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=in_size, mode="bilinear",
                           align_corners=False)
        return torch.sigmoid(x)


def build_deeplabv3_torchvision(out_channels: int = 1,
                                 pretrained: bool = True) -> nn.Module:
    """
    Optional path using torchvision's deeplabv3_resnet50 (ImageNet /
    COCO pretrained) with the classifier head replaced for binary
    segmentation. Falls back silently to the from-scratch DeepLabV3Plus
    above if torchvision weights cannot be downloaded (e.g. offline
    environment) — call build_deeplab() below rather than this directly.
    """
    import torchvision.models.segmentation as seg
    weights = "DEFAULT" if pretrained else None
    model = seg.deeplabv3_resnet50(weights=weights)
    model.classifier[4] = nn.Conv2d(256, out_channels, 1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, out_channels, 1)

    class _Wrapped(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)["out"]
            return torch.sigmoid(out)

    return _Wrapped(model)


def build_deeplab(out_channels: int = 1, prefer_pretrained: bool = True):
    """Try torchvision DeepLabV3 (ResNet-50, pretrained); fall back to
    the lightweight from-scratch DeepLabV3Plus if unavailable/offline."""
    if prefer_pretrained:
        try:
            return build_deeplabv3_torchvision(out_channels, pretrained=True)
        except Exception as e:
            print(f"  [DeepLabV3] Falling back to from-scratch ASPP "
                  f"decoder ({e})")
    return DeepLabV3Plus(out_channels=out_channels)


# ─────────────────────────────────────────────────────────────────────────
#  4. TransUNet-lite (CNN encoder + Transformer bottleneck + CNN decoder)
# ─────────────────────────────────────────────────────────────────────────

class _TransformerBottleneck(nn.Module):
    def __init__(self, channels: int, n_layers: int = 4, n_heads: int = 8,
                 mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads,
            dim_feedforward=int(channels * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pos_embed = None  # built lazily once spatial size is known

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)          # (B, HW, C)
        if (self.pos_embed is None or
                self.pos_embed.shape[1] != tokens.shape[1]):
            self.pos_embed = nn.Parameter(
                torch.zeros(1, tokens.shape[1], C, device=x.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class TransUNetLite(nn.Module):
    """
    CNN encoder (identical DoubleConv/DownBlock stack to the plain U-Net,
    for a fair ablation) with its bottleneck replaced by a multi-head
    self-attention Transformer stack operating on the flattened
    16x downsampled feature map, followed by the standard CNN decoder.
    A parameter- and compute-budget-matched reproduction of Chen et al.
    2021's TransUNet, sized for a 224x224 single-GPU ablation.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 filters=None, n_transformer_layers: int = 4,
                 n_heads: int = 8):
        super().__init__()
        f = filters or UNET_FILTERS
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_channels, f[0])
        self.enc2 = DoubleConv(f[0], f[1])
        self.enc3 = DoubleConv(f[1], f[2])
        self.enc4 = DoubleConv(f[2], f[3])

        self.bottleneck_conv = DoubleConv(f[3], f[4])
        self.transformer = _TransformerBottleneck(
            f[4], n_layers=n_transformer_layers, n_heads=n_heads)

        from segmentation.unet import UpBlock
        self.dec4 = UpBlock(f[4], f[3])
        self.dec3 = UpBlock(f[3], f[2])
        self.dec2 = UpBlock(f[2], f[1])
        self.dec1 = UpBlock(f[1], f[0])
        self.out_conv = nn.Conv2d(f[0], out_channels, 1)

    def forward(self, x):
        s1 = self.enc1(x);   x = self.pool(s1)
        s2 = self.enc2(x);   x = self.pool(s2)
        s3 = self.enc3(x);   x = self.pool(s3)
        s4 = self.enc4(x);   x = self.pool(s4)

        x = self.bottleneck_conv(x)
        x = self.transformer(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return torch.sigmoid(self.out_conv(x))


# ─────────────────────────────────────────────────────────────────────────
#  5. Swin-UNet-lite (windowed self-attention encoder/decoder)
# ─────────────────────────────────────────────────────────────────────────

class _WindowAttentionBlock(nn.Module):
    """A single, non-shifted windowed multi-head self-attention block
    (Liu et al. 2021, simplified — no relative position bias table for
    ablation-budget purposes; captures the core local-window attention
    mechanism that differentiates Swin from a plain CNN block)."""

    def __init__(self, dim: int, window_size: int = 7, n_heads: int = 4):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x):
        # x : (B, C, H, W)
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[2], x.shape[3]

        # partition into windows
        x = x.permute(0, 2, 3, 1)                      # (B, Hp, Wp, C)
        x = x.reshape(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)  # (B*nW, ws*ws, C)

        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        # merge windows back
        nW_h, nW_w = Hp // ws, Wp // ws
        x = x.reshape(B, nW_h, nW_w, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        x = x.permute(0, 3, 1, 2)                       # (B, C, Hp, Wp)
        if pad_h or pad_w:
            x = x[:, :, :H, :W]
        return x


class _SwinDown(nn.Module):
    def __init__(self, in_ch, out_ch, window_size=7, n_heads=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.block = _WindowAttentionBlock(out_ch, window_size, n_heads)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.proj(x)
        x = self.block(x)
        skip = x
        return self.pool(x), skip


class SwinUNetLite(nn.Module):
    """
    2-stage windowed-self-attention encoder/decoder ("lite" reproduction
    of Cao et al. 2021 Swin-UNet), sized for 224x224 single-GPU ablation:
    the first two encoder/decoder levels use Swin-style window attention
    blocks; the deeper levels fall back to convolutional DoubleConv
    blocks (matching TransUNetLite's compute budget) so the comparison
    against TransUNetLite isolates "windowed local attention" (Swin) vs
    "global attention at the bottleneck only" (TransUNet) rather than
    confounding it with encoder depth/capacity.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 filters=None, window_size: int = 7):
        super().__init__()
        f = filters or UNET_FILTERS
        self.stem = nn.Conv2d(in_channels, f[0], 3, padding=1)

        self.down1 = _SwinDown(f[0], f[0], window_size)
        self.down2 = _SwinDown(f[0], f[1], window_size)
        self.enc3 = DoubleConv(f[1], f[2])
        self.enc4 = DoubleConv(f[2], f[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(f[3], f[4])

        from segmentation.unet import UpBlock
        self.dec4 = UpBlock(f[4], f[3])
        self.dec3 = UpBlock(f[3], f[2])
        self.dec2 = UpBlock(f[2], f[1])
        self.dec1 = UpBlock(f[1], f[0])
        self.out_conv = nn.Conv2d(f[0], out_channels, 1)

    def forward(self, x):
        x = self.stem(x)
        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        s3 = self.enc3(x);  x = self.pool(s3)
        s4 = self.enc4(x);  x = self.pool(s4)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return torch.sigmoid(self.out_conv(x))


# ─────────────────────────────────────────────────────────────────────────
#  6. nnU-Net-style U-Net (instance norm + leaky ReLU + deep supervision)
# ─────────────────────────────────────────────────────────────────────────

class _DoubleConvIN(nn.Module):
    """DoubleConv with InstanceNorm + LeakyReLU, matching nnU-Net's
    default architectural template (as opposed to the BatchNorm+ReLU
    used by the plain U-Net baseline)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class NNUNetStyleUNet(nn.Module):
    """
    Reproduces nnU-Net's *architectural* signature (instance norm +
    leaky ReLU + deep supervision at 3 decoder resolutions) on the same
    5-level U-Net topology as the baseline, for a controlled ablation of
    "does the nnU-Net architectural recipe help on EBHI-SEG" separate
    from nnU-Net's self-configuring preprocessing pipeline (which is not
    reproduced here — it operates on dataset fingerprinting rather than
    a fixed architecture and is out of scope for this ablation).

    In train mode, forward() returns a dict with the full-resolution
    output plus two auxiliary lower-resolution outputs (upsampled to
    full res) for deep-supervision loss; in eval mode it returns only
    the final full-resolution mask, keeping the same call signature as
    every other variant in this file.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 filters=None):
        super().__init__()
        f = filters or UNET_FILTERS
        self.pool = nn.MaxPool2d(2)

        self.enc1 = _DoubleConvIN(in_channels, f[0])
        self.enc2 = _DoubleConvIN(f[0], f[1])
        self.enc3 = _DoubleConvIN(f[1], f[2])
        self.enc4 = _DoubleConvIN(f[2], f[3])
        self.bottleneck = _DoubleConvIN(f[3], f[4])

        self.up4 = nn.ConvTranspose2d(f[4], f[3], 2, stride=2)
        self.dec4 = _DoubleConvIN(f[3] * 2, f[3])
        self.up3 = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3 = _DoubleConvIN(f[2] * 2, f[2])
        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = _DoubleConvIN(f[1] * 2, f[1])
        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = _DoubleConvIN(f[0] * 2, f[0])

        self.out_final = nn.Conv2d(f[0], out_channels, 1)
        self.out_ds2 = nn.Conv2d(f[1], out_channels, 1)   # aux head @ dec2
        self.out_ds3 = nn.Conv2d(f[2], out_channels, 1)   # aux head @ dec3

    def _match(self, x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="bilinear",
                               align_corners=False)
        return x

    def forward(self, x):
        in_size = x.shape[-2:]
        s1 = self.enc1(x);   x = self.pool(s1)
        s2 = self.enc2(x);   x = self.pool(s2)
        s3 = self.enc3(x);   x = self.pool(s3)
        s4 = self.enc4(x);   x = self.pool(s4)
        x = self.bottleneck(x)

        x = self._match(self.up4(x), s4)
        d4 = self.dec4(torch.cat([x, s4], 1))

        x = self._match(self.up3(d4), s3)
        d3 = self.dec3(torch.cat([x, s3], 1))

        x = self._match(self.up2(d3), s2)
        d2 = self.dec2(torch.cat([x, s2], 1))

        x = self._match(self.up1(d2), s1)
        d1 = self.dec1(torch.cat([x, s1], 1))

        final = torch.sigmoid(self.out_final(d1))

        if self.training:
            ds2 = torch.sigmoid(F.interpolate(
                self.out_ds2(d2), size=in_size, mode="bilinear",
                align_corners=False))
            ds3 = torch.sigmoid(F.interpolate(
                self.out_ds3(d3), size=in_size, mode="bilinear",
                align_corners=False))
            return {"final": final, "ds2": ds2, "ds3": ds3}
        return final


class NNUNetDeepSupervisionLoss(nn.Module):
    """Weighted sum of Dice+BCE at full-res + 2 auxiliary decoder
    resolutions, per nnU-Net's deep-supervision recipe (weights
    halve at each shallower resolution)."""

    def __init__(self):
        super().__init__()
        self.base = DiceBCELoss()
        self.weights = {"final": 1.0, "ds2": 0.5, "ds3": 0.25}

    def forward(self, outputs: dict, target: torch.Tensor) -> torch.Tensor:
        total, wsum = 0.0, 0.0
        for k, w in self.weights.items():
            total += w * self.base(outputs[k], target)
            wsum += w
        return total / wsum


# ─────────────────────────────────────────────────────────────────────────
#  Registry — used by evaluation/ablation_unet_variants.py
# ─────────────────────────────────────────────────────────────────────────

def _plain_unet(**kw):
    from segmentation.unet import UNet
    return UNet(**kw)


SEGMENTATION_REGISTRY = {
    "UNet":            _plain_unet,
    "UNetPlusPlus":    UNetPlusPlus,
    "AttentionUNet":   AttentionUNet,
    "DeepLabV3":       lambda **kw: build_deeplab(
        out_channels=kw.get("out_channels", 1)),
    "TransUNetLite":   TransUNetLite,
    "SwinUNetLite":    SwinUNetLite,
    "NNUNetStyle":     NNUNetStyleUNet,
}


def build_segmentation_model(name: str, in_channels: int = 3,
                              out_channels: int = 1) -> nn.Module:
    if name not in SEGMENTATION_REGISTRY:
        raise ValueError(f"Unknown segmentation backbone '{name}'. "
                          f"Choices: {list(SEGMENTATION_REGISTRY)}")
    builder = SEGMENTATION_REGISTRY[name]
    try:
        return builder(in_channels=in_channels, out_channels=out_channels)
    except TypeError:
        # DeepLabV3 lambda only accepts out_channels
        return builder(out_channels=out_channels)


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    print(f"{'Backbone':16s}  {'Params':>12s}  {'Output shape':>18s}")
    print("-" * 52)
    for name in SEGMENTATION_REGISTRY:
        model = build_segmentation_model(name)
        model.eval()
        with torch.no_grad():
            out = model(x)
        if isinstance(out, dict):
            out = out["final"]
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:16s}  {n_params:12,d}  {str(tuple(out.shape)):>18s}")
