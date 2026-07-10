"""
models/attention.py
─────────────────────────────────────────────────────────────────────────────
Channel / spatial attention modules used across the CRC-AI ablation suite.

Implements
  1. ColourChannelAttention (CCA)  — the proposed module. Jointly pools
     GAP + GMP channel statistics and is inserted immediately after the
     stem convolution (conv0), i.e. BEFORE dense-block feature extraction,
     so stain-specific channel calibration happens before deep feature
     extraction rather than after it.
  2. SqueezeExcitation (SE)        — Hu et al., 2018. GAP-only channel
     attention (used for the SEResNet backbone / attention ablation).
  3. CBAM                          — Woo et al., 2018. Channel attention
     (GAP+GMP, shared MLP) followed by a *separate* spatial-attention
     branch (used for the CBAMResNet backbone / attention ablation).
  4. ECA (Efficient Channel Attention) — Wang et al., 2020. Local
     cross-channel interaction via a 1-D convolution, no dimensionality
     reduction, negligible parameter cost.

All four modules share the interface:
    module = AttentionModule(num_channels)
    y = module(x)     # x, y : (B, C, H, W)  same shape in/out

This makes them drop-in interchangeable for the attention ablation study
(evaluation/ablation_attention_variants.py) and the ATR Reviewer #1
comment-4 response (CCA vs SE vs CBAM vs ECA).
─────────────────────────────────────────────────────────────────────────────
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────
#  1. Colour Channel Attention (CCA) — proposed module
# ─────────────────────────────────────────────────────────────────────────

class ColourChannelAttention(nn.Module):
    """
    Jointly uses global-average-pooled (GAP) and global-max-pooled (GMP)
    channel statistics, fused through a shared bottleneck MLP, to produce
    a per-channel gate. Unlike SE (GAP only) and CBAM (GAP+GMP but with an
    *additional* spatial-attention branch applied deeper in the network),
    CCA is inserted directly after the stem convolution so that
    stain-specific channel recalibration (H&E colour balance) happens
    before any dense-block processing.

    Params (approx): 2 * C * (C / r) — under 0.05% of DenseNet201's ~20M
    parameters at r=16 for a 64-channel stem.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.gap(x))
        max_out = self.mlp(self.gmp(x))
        attn = self.gate(avg_out + max_out)         # (B, C, 1, 1)
        return x * attn


# ─────────────────────────────────────────────────────────────────────────
#  2. Squeeze-and-Excitation (SE)
# ─────────────────────────────────────────────────────────────────────────

class SqueezeExcitation(nn.Module):
    """Hu et al. 2018. GAP-only channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.gap(x))


# ─────────────────────────────────────────────────────────────────────────
#  3. CBAM — Convolutional Block Attention Module
# ─────────────────────────────────────────────────────────────────────────

class _CBAMChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x):
        attn = self.gate(self.mlp(self.gap(x)) + self.mlp(self.gmp(x)))
        return x * attn


class _CBAMSpatialGate(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        attn = self.gate(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """
    Woo et al. 2018. Channel attention (GAP+GMP, shared MLP) followed
    by a *separate, sequential* spatial-attention branch. This extra
    spatial branch is the key architectural difference vs. CCA.
    """

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7):
        super().__init__()
        self.channel_gate = _CBAMChannelGate(channels, reduction)
        self.spatial_gate = _CBAMSpatialGate(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x


# ─────────────────────────────────────────────────────────────────────────
#  4. ECA — Efficient Channel Attention
# ─────────────────────────────────────────────────────────────────────────

class ECA(nn.Module):
    """
    Wang et al. 2020. Avoids dimensionality reduction entirely; captures
    local cross-channel interaction with a single 1-D convolution whose
    kernel size is adaptively derived from the channel count. Cheapest
    of the four modules in parameter count.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1
        k = max(k, 3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k,
                               padding=(k - 1) // 2, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gap(x)                       # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)    # (B, 1, C)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = self.gate(y)
        return x * y


# ─────────────────────────────────────────────────────────────────────────
#  Identity (no-attention baseline)
# ─────────────────────────────────────────────────────────────────────────

class NoAttention(nn.Module):
    """Passthrough — used as the 'None' arm of the attention ablation."""

    def __init__(self, channels: int, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ─────────────────────────────────────────────────────────────────────────
#  Registry — used by evaluation/ablation_attention_variants.py
# ─────────────────────────────────────────────────────────────────────────

ATTENTION_REGISTRY = {
    "None": NoAttention,
    "SE":   SqueezeExcitation,
    "CBAM": CBAM,
    "ECA":  ECA,
    "CCA":  ColourChannelAttention,
}


def build_attention(name: str, channels: int, **kwargs) -> nn.Module:
    if name not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention module '{name}'. "
                          f"Choices: {list(ATTENTION_REGISTRY)}")
    return ATTENTION_REGISTRY[name](channels, **kwargs)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    # Quick sanity check + parameter-overhead report (stem = 64 channels,
    # matching DenseNet201's conv0 output).
    C = 64
    x = torch.randn(2, C, 56, 56)
    print(f"{'Module':10s}  {'Params':>10s}  {'% of 20M backbone':>20s}")
    print("-" * 45)
    for name, cls in ATTENTION_REGISTRY.items():
        mod = cls(C)
        y = mod(x)
        assert y.shape == x.shape, f"{name} changed tensor shape!"
        p = count_params(mod)
        print(f"{name:10s}  {p:10,d}  {p / 20_000_000 * 100:19.4f}%")
