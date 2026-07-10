"""
classifier/neurosymbolic_graph_transformer.py
─────────────────────────────────────────────────────────────────────────────
Stages 7–10 — FULL NEUROSYMBOLIC GRAPH TRANSFORMER PIPELINE

Pipeline
  (image, mask)
      │
      ├─► Backbone (CNN / Transformer)  → feat_vec  (B, feat_dim)
      │         [CCA layer after first conv]
      │
      ├─► U-Net segmented mask  →  rKNN graph  →  Graph Transformer
      │         with attention biases            → graph_feat  (B, graph_dim)
      │
      ├─► Symbolic Layer (KG + PathNetGate)  → sym_feat  (B, sym_dim)
      │
      └─► Fusion FC → Softmax  →  6-class logits
                                  +  binary Benign/Malignant output

Architecture detail
  • Graph Transformer : multi-head attention over node features with
      distance-based attention bias  (Ying et al., Graphormer 2021)
  • Subcellular feature extractor appended to node features
  • Adam optimiser, Cross-Entropy loss (label-smoothed + class-weighted)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (CLASSES, CLASS_TO_IDX, KNN_K, POS_ENC_DIM,
                            MALIGNANT_CLASSES, BENIGN_CLASSES)
from symbolic.knowledge_graph import SymbolicLayer

NUM_CLASSES = len(CLASSES)


# ─── Subcellular Feature Extractor (Stage 8) ──────────────────────────────

class SubcellularFeatureExtractor(nn.Module):
    """
    Extracts subcellular texture features from a node's patch region.
    Uses multi-scale LBP-like convolutional filters.

    Input  : image patch tensor (B, 3, H, W)
    Output : subcellular feature vector (B, sub_dim)
    """

    def __init__(self, sub_dim: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, groups=16), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, sub_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = out.flatten(1)
        return self.proj(out)


# ─── Graph Transformer with attention bias (Stage 9–10) ───────────────────

class GraphTransformerLayer(nn.Module):
    """
    Single Graph Transformer layer with spatial attention bias.
    Attention logits modified: A_ij += f(dist_ij)
    """

    def __init__(self, node_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = node_dim // n_heads
        assert node_dim % n_heads == 0

        self.q = nn.Linear(node_dim, node_dim)
        self.k = nn.Linear(node_dim, node_dim)
        self.v = nn.Linear(node_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)

        # Spatial bias: scalar for each head, based on distance bucket
        self.dist_bias = nn.Embedding(32, n_heads)   # 32 distance buckets
        self.dropout   = nn.Dropout(dropout)
        self.norm1     = nn.LayerNorm(node_dim)
        self.norm2     = nn.LayerNorm(node_dim)
        self.ffn       = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 2, node_dim),
        )

    def _dist_bucket(self, pos: torch.Tensor) -> torch.Tensor:
        """Bucket pairwise distances into 32 bins."""
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)      # (N, N, 2)
        dists = diff.norm(dim=-1)                        # (N, N)
        buckets = (dists / 10).long().clamp(0, 31)
        return buckets

    def forward(self, x: torch.Tensor,
                pos: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        N, D = x.shape
        H, d = self.n_heads, self.head_dim

        # QKV
        Q = self.q(x).view(N, H, d)
        K = self.k(x).view(N, H, d)
        V = self.v(x).view(N, H, d)

        # Attention scores
        scale = math.sqrt(d)
        attn = torch.einsum("nhd,mhd->hnm", Q, K) / scale   # (H, N, N)

        # Spatial bias
        buckets  = self._dist_bucket(pos)                    # (N, N)
        spat_bias = self.dist_bias(buckets)                  # (N, N, H)
        spat_bias = spat_bias.permute(2, 0, 1)               # (H, N, N)
        attn = attn + spat_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        out = torch.einsum("hnm,mhd->nhd", attn, V).reshape(N, D)
        out = self.out_proj(out)
        x   = self.norm1(x + out)
        x   = self.norm2(x + self.ffn(x))
        return x


class GraphTransformer(nn.Module):
    """
    Multi-layer Graph Transformer with readout.
    Input  : node features (N, node_dim), positions (N, 2)
    Output : graph-level embedding (graph_dim,)
    """

    def __init__(self, node_in: int, node_dim: int = 128,
                 graph_dim: int = 256, n_layers: int = 3, n_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(node_in, node_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(node_dim, n_heads) for _ in range(n_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(node_dim, graph_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, pos)
        # Global mean + max pooling readout
        g = (x.mean(0) + x.max(0).values) / 2
        return self.readout(g)


# ─── Full Neurosymbolic Graph Transformer (NSGT) ──────────────────────────

class NeurosymbolicGraphTransformer(nn.Module):
    """
    End-to-end model combining:
      • Backbone CNN/Transformer feature extractor
      • Graph Transformer over segmentation graph (with attention bias)
      • Symbolic Layer (KG + PathNet gate)
      • Fusion classifier

    Outputs
    -------
    logits       : (B, NUM_CLASSES)   multiclass
    binary_logit : (B, 2)             Benign vs Malignant
    """

    def __init__(self,
                 backbone: nn.Module,
                 feat_dim: int,
                 node_in:  int = 6 + POS_ENC_DIM,   # from graph construction
                 node_dim: int = 128,
                 graph_dim: int = 256,
                 sym_dim:   int = 256):
        super().__init__()
        self.backbone = backbone

        # Stage 9: Graph transformer
        self.graph_transformer = GraphTransformer(
            node_in=node_in, node_dim=node_dim,
            graph_dim=graph_dim, n_layers=3, n_heads=4)

        # Stage 6: Symbolic layer
        self.symbolic = SymbolicLayer(feat_dim=feat_dim,
                                      node_dim=32, kg_dim=64, out_dim=sym_dim)

        # Fusion
        fused_dim = feat_dim + graph_dim + sym_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Outputs
        self.head_multiclass = nn.Linear(256, NUM_CLASSES)
        self.head_binary     = nn.Linear(256, 2)   # Benign / Malignant

    def _extract_backbone_feat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through backbone, strip its original classification head
        and return the penultimate feature vector.
        """
        # We replace the backbone's fc/head temporarily
        # Works for both torchvision (model.fc) and timm (model.head)
        backbone = self.backbone
        if hasattr(backbone, 'fc'):
            original_fc, backbone.fc = backbone.fc, nn.Identity()
            feat = backbone(x)
            backbone.fc = original_fc
        elif hasattr(backbone, 'head'):
            original_head, backbone.head = backbone.head, nn.Identity()
            feat = backbone(x)
            backbone.head = original_head
        elif hasattr(backbone, 'classifier'):
            original_cls = backbone.classifier
            backbone.classifier = nn.Identity()
            feat = backbone(x)
            backbone.classifier = original_cls
        else:
            feat = backbone(x)

        if feat.dim() > 2:
            feat = feat.flatten(1)
        return feat

    def forward(self, x: torch.Tensor,
                graph_x: torch.Tensor,
                graph_pos: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x         : (B, 3, H, W)  image batch
        graph_x   : (N_total, node_in)  concatenated node features for batch
        graph_pos : (N_total, 2)        node positions for batch
        """
        # Backbone features
        feat = self._extract_backbone_feat(x)           # (B, feat_dim)

        # Graph features — process per image in batch
        graph_feat = self.graph_transformer(graph_x,
                                             graph_pos)  # (graph_dim,) → unsqueeze
        graph_feat = graph_feat.unsqueeze(0).expand(
            feat.size(0), -1)                            # (B, graph_dim)

        # Symbolic features
        sym_feat = self.symbolic(feat)                   # (B, sym_dim)

        # Fusion
        fused = torch.cat([feat, graph_feat, sym_feat], dim=-1)
        fused = self.fusion(fused)                       # (B, 256)

        logits_multi  = self.head_multiclass(fused)      # (B, NUM_CLASSES)
        logits_binary = self.head_binary(fused)          # (B, 2)

        return {
            "logits_multiclass": logits_multi,
            "logits_binary":     logits_binary,
        }


# ─── Label-smoothed focal cross-entropy ───────────────────────────────────

class FocalCrossEntropyLoss(nn.Module):
    """
    Focal loss variant of cross-entropy with label smoothing.
    Helps address the Normal ≈ Polyp ≈ High-grade IN confusion.
    """

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.1,
                 weight: torch.Tensor = None):
        super().__init__()
        self.gamma     = gamma
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(-1)
        log_prob = F.log_softmax(logits, dim=-1)

        # Label smoothing
        with torch.no_grad():
            smooth_target = torch.full_like(log_prob,
                                            self.smoothing / (n_cls - 1))
            smooth_target.scatter_(1, targets.unsqueeze(1),
                                   1.0 - self.smoothing)

        loss = -(smooth_target * log_prob).sum(dim=-1)

        # Focal weighting
        prob = torch.exp(-loss)
        focal_weight = (1 - prob) ** self.gamma
        loss = focal_weight * loss

        if self.weight is not None:
            w = self.weight[targets]
            loss = loss * w

        return loss.mean()
