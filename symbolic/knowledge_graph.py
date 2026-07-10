"""
symbolic/knowledge_graph.py
─────────────────────────────────────────────────────────────────────────────
Stage 6 — SYMBOLIC LAYER

Implements a Knowledge Graph (KG) encoding domain pathology rules
combined with a PathNet-style gating mechanism.

Knowledge Graph Nodes:
  - Cancer class nodes  (6)
  - Morphological feature nodes  (gland shape, lumen, nuclear density, …)
  - Staining feature nodes  (H&E hue, saturation bands)

Knowledge Graph Edges:
  - "is_associated_with"  (class → morphological feature)
  - "often_confused_with" (class ↔ class, for confusable groups)
  - "distinguishes"       (feature → class pair)

PathNet Integration:
  The KG embedding is projected and used as a GATE on the CNN/Transformer
  feature vector before the final classifier, forcing the model to consult
  symbolic constraints when making predictions.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import CLASSES, CLASS_TO_IDX, CONFUSABLE_GROUPS

NUM_CLASSES = len(CLASSES)

# ─── Knowledge graph vocabulary ───────────────────────────────────────────

# Morphological feature nodes
MORPH_FEATURES = [
    "glandular_architecture",
    "irregular_lumen",
    "serrated_epithelium",
    "nuclear_stratification",
    "high_nuclear_density",
    "stromal_invasion",
    "tubular_formation",
    "villous_architecture",
]

# H&E staining features
STAIN_FEATURES = [
    "dark_nuclear_stain",
    "pink_stroma",
    "mucin_production",
    "goblet_cells",
    "mitotic_figures",
]

ALL_NODES = CLASSES + MORPH_FEATURES + STAIN_FEATURES
NODE_TO_IDX = {n: i for i, n in enumerate(ALL_NODES)}
N_NODES = len(ALL_NODES)

# ─── Handcrafted KG edges (domain knowledge) ──────────────────────────────

KG_EDGES_RAW = [
    # class → morphology associations
    ("Benign",           "glandular_architecture"),
    ("Polyp",            "glandular_architecture"),
    ("Polyp",            "tubular_formation"),
    ("Polyp",            "villous_architecture"),
    ("Serrated Adenoma", "serrated_epithelium"),
    ("Serrated Adenoma", "irregular_lumen"),
    ("Serrated Adenoma", "glandular_architecture"),
    ("High Grade IN",    "nuclear_stratification"),
    ("High Grade IN",    "high_nuclear_density"),
    ("High Grade IN",    "irregular_lumen"),
    ("High Grade IN",    "mitotic_figures"),
    ("Low Grade IN",     "nuclear_stratification"),
    ("Low Grade IN",     "glandular_architecture"),
    ("Adenocarcinoma",   "stromal_invasion"),
    ("Adenocarcinoma",   "tubular_formation"),
    ("Adenocarcinoma",   "mitotic_figures"),
    ("Adenocarcinoma",   "high_nuclear_density"),

    # staining associations
    ("Benign",           "dark_nuclear_stain"),
    ("Benign",           "pink_stroma"),
    ("Benign",           "goblet_cells"),
    ("Serrated Adenoma", "mucin_production"),
    ("Adenocarcinoma",   "mucin_production"),
    ("High Grade IN",    "dark_nuclear_stain"),

    # confusable pairs (bidirectional)
    ("Benign",           "Polyp"),
    ("Benign",           "High Grade IN"),
    ("Benign",           "Serrated Adenoma"),
    ("Polyp",            "High Grade IN"),
    ("Polyp",            "Serrated Adenoma"),
    ("High Grade IN",    "Serrated Adenoma"),
]


def _build_adjacency() -> torch.Tensor:
    """Build N_NODES × N_NODES symmetric adjacency matrix."""
    adj = torch.zeros(N_NODES, N_NODES)
    for (src, dst) in KG_EDGES_RAW:
        if src in NODE_TO_IDX and dst in NODE_TO_IDX:
            i, j = NODE_TO_IDX[src], NODE_TO_IDX[dst]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    # Add self-loops
    adj += torch.eye(N_NODES)
    # Normalise
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
    return adj / deg


# ─── GCN layer for KG embedding ───────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return F.relu(self.W(adj @ x))


class KnowledgeGraphEncoder(nn.Module):
    """
    Two-layer GCN over the handcrafted KG.
    Produces a knowledge embedding of shape (N_NODES, kg_dim).
    """

    def __init__(self, node_dim: int = 32, kg_dim: int = 64):
        super().__init__()
        self.register_buffer("adj", _build_adjacency())

        # Node initial embeddings (learnable)
        self.node_embed = nn.Embedding(N_NODES, node_dim)
        self.gcn1 = GCNLayer(node_dim, kg_dim)
        self.gcn2 = GCNLayer(kg_dim,   kg_dim)

    def forward(self) -> torch.Tensor:
        """Returns (N_NODES, kg_dim) knowledge embeddings."""
        x = self.node_embed.weight          # (N, node_dim)
        x = self.gcn1(x, self.adj)
        x = self.gcn2(x, self.adj)
        return x                             # (N, kg_dim)

    def class_embeddings(self) -> torch.Tensor:
        """Extract only the NUM_CLASSES embeddings."""
        kg = self.forward()
        return kg[:NUM_CLASSES]              # (NUM_CLASSES, kg_dim)


# ─── PathNet Gate ─────────────────────────────────────────────────────────

class PathNetGate(nn.Module):
    """
    Gates the backbone feature vector using the KG class embeddings.

    Given
      feat_vec : (B, feat_dim) — CNN/Transformer features
      kg_embed : (NUM_CLASSES, kg_dim) — class-specific knowledge

    The gate computes a soft modulation per-class and returns a
    context-enriched feature for the final classifier.
    """

    def __init__(self, feat_dim: int, kg_dim: int, out_dim: int = 256):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, out_dim)
        self.kg_proj   = nn.Linear(kg_dim,   out_dim)
        self.gate_fc   = nn.Linear(out_dim * 2, out_dim)
        self.out_proj  = nn.Linear(out_dim, out_dim)

    def forward(self, feat: torch.Tensor,
                kg_class_embed: torch.Tensor) -> torch.Tensor:
        """
        feat           : (B, feat_dim)
        kg_class_embed : (NUM_CLASSES, kg_dim)  → averaged to (kg_dim,)
        Returns        : (B, out_dim)
        """
        f = self.feat_proj(feat)                            # (B, out_dim)
        k = self.kg_proj(kg_class_embed.mean(0))            # (out_dim,)
        k = k.unsqueeze(0).expand_as(f)                     # (B, out_dim)
        gate = torch.sigmoid(self.gate_fc(
            torch.cat([f, k], dim=-1)))                     # (B, out_dim)
        out = self.out_proj(f * gate)
        return F.relu(out)


# ─── Combined symbolic module ─────────────────────────────────────────────

class SymbolicLayer(nn.Module):
    """
    Full Stage-6 symbolic module:
      KnowledgeGraphEncoder  →  PathNetGate  →  gated feature vector
    """

    def __init__(self, feat_dim: int, node_dim: int = 32,
                 kg_dim: int = 64, out_dim: int = 256):
        super().__init__()
        self.kg_encoder = KnowledgeGraphEncoder(node_dim, kg_dim)
        self.path_gate  = PathNetGate(feat_dim, kg_dim, out_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        kg_embed = self.kg_encoder.class_embeddings()   # (C, kg_dim)
        gated    = self.path_gate(feat, kg_embed)       # (B, out_dim)
        return gated
