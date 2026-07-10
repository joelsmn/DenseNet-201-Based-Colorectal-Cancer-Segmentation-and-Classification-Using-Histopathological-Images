"""
graph/cellular_interaction_graph.py
─────────────────────────────────────────────────────────────────────────────
Stage 9 — CONVERT IMAGES TO CELLULAR INTERACTION GRAPHS

Extends basic graph_construction.py with richer edge features encoding
cell–cell interaction cues:

Edge features
  • Euclidean distance (normalised)
  • Colour similarity between adjacent cells (cosine of mean-RGB vectors)
  • Shared boundary length (overlap of convex hulls projected to 1D)
  • Relative area ratio

These edge features are concatenated and passed as `edge_attr` to the
Graph Transformer alongside the node features.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import numpy as np
import cv2
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import KNN_K, KNN_RADIUS, POS_ENC_DIM
from graph.graph_construction import (
    extract_node_features,
    build_rknn_edges,
    sinusoidal_pe,
)

EDGE_FEAT_DIM = 4   # dist, colour_sim, boundary, area_ratio


# ─── Edge feature computation ─────────────────────────────────────────────

def compute_edge_features(centroids:   np.ndarray,
                           node_feat:  np.ndarray,
                           edge_index: np.ndarray) -> np.ndarray:
    """
    Compute edge-level features for each (i,j) pair in edge_index.

    Parameters
    ----------
    centroids  : (N, 2)
    node_feat  : (N, F)  raw features before PE — first 3 cols are mean RGB
    edge_index : (2, E)

    Returns
    -------
    edge_attr  : (E, EDGE_FEAT_DIM)
    """
    if edge_index.shape[1] == 0:
        return np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32)

    h, w = centroids[:, 1].max() + 1, centroids[:, 0].max() + 1
    max_dist = np.sqrt(h**2 + w**2) + 1e-6

    src = edge_index[0]
    dst = edge_index[1]

    # 1. Normalised Euclidean distance
    diff  = centroids[src] - centroids[dst]
    dists = np.sqrt((diff**2).sum(axis=1)) / max_dist   # (E,)

    # 2. Colour similarity (cosine of mean-RGB vectors)
    c_src = node_feat[src, :3] + 1e-8
    c_dst = node_feat[dst, :3] + 1e-8
    cos   = (c_src * c_dst).sum(1) / (
                np.linalg.norm(c_src, axis=1) *
                np.linalg.norm(c_dst, axis=1) + 1e-8)    # (E,)

    # 3. Shared boundary proxy: 1 / (dist + 1)  (closer → higher)
    boundary = 1.0 / (dists * max_dist + 1.0)            # (E,)

    # 4. Area ratio  min(area_i, area_j) / max(area_i, area_j)
    area_src = node_feat[src, 3]
    area_dst = node_feat[dst, 3]
    area_ratio = (np.minimum(area_src, area_dst) /
                  (np.maximum(area_src, area_dst) + 1e-8))  # (E,)

    edge_attr = np.stack([dists, cos, boundary, area_ratio], axis=1)
    return edge_attr.astype(np.float32)


# ─── Full cellular interaction graph builder ──────────────────────────────

def image_to_cellular_graph(img_bgr:  np.ndarray,
                              mask_bin: np.ndarray,
                              label:    int = None):
    """
    Build a cellular interaction graph with node + edge features.

    Returns torch_geometric.data.Data with:
      x          : (N, 6 + POS_ENC_DIM)  node features + PE
      edge_index : (2, E)
      edge_attr  : (E, 4)                edge features
      pos        : (N, 2)                centroids
      y          : (1,) optional
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("Install torch_geometric: pip install torch_geometric")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    centroids, features = extract_node_features(img_rgb, mask_bin)
    edge_index          = build_rknn_edges(centroids)

    # Base features (before PE) are first 6 columns
    base_feat  = features[:, :6]
    edge_attr  = compute_edge_features(centroids, base_feat, edge_index)

    data = Data(
        x          = torch.tensor(features,   dtype=torch.float),
        edge_index = torch.tensor(edge_index, dtype=torch.long),
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float),
        pos        = torch.tensor(centroids,  dtype=torch.float),
    )
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)

    return data
