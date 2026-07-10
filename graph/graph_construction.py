"""
graph/graph_construction.py
─────────────────────────────────────────────────────────────────────────────
Stage 5 — GRAPH CONSTRUCTION

Method : Radius-based K-Nearest-Neighbour (rKNN) graph
         built from superpixel / nuclei centroids detected in the
         segmentation mask.

Steps
  1. Detect connected components (cell/nuclei regions) in binary mask.
  2. Compute centroid coordinates for each region → node positions.
  3. Build rKNN graph: connect two nodes if
       • their Euclidean distance ≤ KNN_RADIUS, AND
       • the edge is among the K nearest neighbours of each node.
  4. Attach sinusoidal positional encoding (PE) of dimension POS_ENC_DIM
     to each node feature vector.
  5. Extract node features: mean colour (RGB), area, eccentricity, solidity.

Output : torch_geometric.data.Data  object:
  • x       — node feature matrix  (N, F)
  • edge_index — COO edge list      (2, E)
  • pos     — centroid coordinates  (N, 2)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import math
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Tuple, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import KNN_K, KNN_RADIUS, POS_ENC_DIM


# ─── Positional encoding ──────────────────────────────────────────────────

def sinusoidal_pe(positions: np.ndarray, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding for 2D coordinates.
    positions : (N, 2)  normalised to [0, 1]
    Returns   : (N, d_model)
    """
    N   = positions.shape[0]
    enc = np.zeros((N, d_model), dtype=np.float32)
    half = d_model // 2

    for i in range(half):
        div = 10000 ** (2 * i / d_model)
        enc[:, 2 * i]     = np.sin(positions[:, 0] / div)
        enc[:, 2 * i + 1] = np.cos(positions[:, 1] / div)

    return enc


# ─── Node feature extraction ──────────────────────────────────────────────

def extract_node_features(img_rgb: np.ndarray,
                           mask_binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects connected components in mask, extracts per-node features.

    Returns
    -------
    centroids : (N, 2) float32  — (x, y) pixel coordinates
    features  : (N, F) float32  — [mean_R, mean_G, mean_B,
                                    area_norm, eccentricity, solidity,
                                    PE_0..PE_{POS_ENC_DIM-1}]
    """
    h, w = mask_binary.shape
    n_labels, labels, stats, centroids_cv = \
        cv2.connectedComponentsWithStats(mask_binary.astype(np.uint8), 8)

    if n_labels <= 1:
        # Fallback: treat whole image as single node
        centroids = np.array([[w / 2, h / 2]], dtype=np.float32)
        feat      = np.zeros((1, 6 + POS_ENC_DIM), dtype=np.float32)
        feat[0, :3] = img_rgb.mean(axis=(0, 1)) / 255.0
        feat[0, 3]  = 1.0
        return centroids, feat

    # Skip background (label 0)
    node_centroids = []
    node_features  = []

    for lbl in range(1, n_labels):
        comp_mask = (labels == lbl).astype(np.uint8)
        cx, cy    = centroids_cv[lbl]   # (x, y)

        # Colour features
        region_px = img_rgb[comp_mask == 1]
        if region_px.shape[0] == 0:
            continue
        mean_rgb = region_px.mean(axis=0) / 255.0

        # Shape features
        area      = stats[lbl, cv2.CC_STAT_AREA]
        area_norm = area / (h * w + 1e-6)

        contours, _ = cv2.findContours(comp_mask,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        eccentricity = 0.0
        solidity     = 0.0
        if contours and len(contours[0]) >= 5:
            try:
                ellipse = cv2.fitEllipse(contours[0])
                ma, mi  = max(ellipse[1]), min(ellipse[1])
                eccentricity = math.sqrt(1 - (mi / (ma + 1e-6)) ** 2)
            except Exception:
                pass
            hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
            solidity  = area / (hull_area + 1e-6)

        node_centroids.append([cx, cy])
        node_features.append([*mean_rgb, area_norm, eccentricity, solidity])

    if not node_centroids:
        node_centroids = [[w / 2, h / 2]]
        node_features  = [[0, 0, 0, 1, 0, 0]]

    centroids = np.array(node_centroids, dtype=np.float32)
    base_feat = np.array(node_features,  dtype=np.float32)

    # Normalise centroids to [0, 1] for PE
    norm_pos = centroids.copy()
    norm_pos[:, 0] /= (w + 1e-6)
    norm_pos[:, 1] /= (h + 1e-6)
    pe = sinusoidal_pe(norm_pos, POS_ENC_DIM)

    features = np.concatenate([base_feat, pe], axis=1)
    return centroids, features


# ─── rKNN edge construction ───────────────────────────────────────────────

def build_rknn_edges(centroids: np.ndarray,
                     k: int = KNN_K,
                     radius: float = KNN_RADIUS) -> np.ndarray:
    """
    Build edge list for radius-based KNN graph.

    Returns
    -------
    edge_index : (2, E) int64
    """
    N = centroids.shape[0]
    if N <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    edges = []
    for i in range(N):
        diffs = centroids - centroids[i]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        dists[i] = np.inf   # exclude self

        # radius filter
        within = np.where(dists <= radius)[0]
        if len(within) == 0:
            # no neighbours in radius — connect to nearest K anyway
            within = np.argsort(dists)[:k]

        # KNN within radius
        within_dists = dists[within]
        knn_idx      = within[np.argsort(within_dists)[:k]]

        for j in knn_idx:
            edges.append([i, int(j)])
            edges.append([int(j), i])   # undirected → bidirectional

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    edge_index = np.unique(np.array(edges, dtype=np.int64), axis=0).T
    return edge_index


# ─── Full graph builder ───────────────────────────────────────────────────

def image_to_graph(img_bgr:  np.ndarray,
                   mask_bin: np.ndarray,
                   label:    Optional[int] = None):
    """
    Convert one histopathology image + binary mask to a graph.

    Parameters
    ----------
    img_bgr  : (H, W, 3)  BGR image
    mask_bin : (H, W)     binary segmentation mask
    label    : int or None  class index

    Returns
    -------
    torch_geometric.data.Data
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("torch_geometric is required for graph construction. "
                          "Install with: pip install torch_geometric")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    centroids, features = extract_node_features(img_rgb, mask_bin)
    edge_index          = build_rknn_edges(centroids)

    data = Data(
        x          = torch.tensor(features,   dtype=torch.float),
        edge_index = torch.tensor(edge_index, dtype=torch.long),
        pos        = torch.tensor(centroids,  dtype=torch.float),
    )
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)

    return data


# ─── Batch dataset graph builder ──────────────────────────────────────────

def build_graph_dataset(split: str = "train",
                        verbose: bool = True) -> list:
    """
    Build graphs for an entire split and return a list of Data objects.
    """
    import sys
    from tqdm import tqdm

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config.config import SPLIT_DIR, CLASSES, CLASS_TO_IDX

    graphs = []
    for class_label in CLASSES:
        img_dir = SPLIT_DIR / split / class_label / "Image"
        lbl_dir = SPLIT_DIR / split / class_label / "Label"
        if not img_dir.exists():
            continue
        files = sorted(img_dir.glob("*"))
        for p in tqdm(files,
                      desc=f"  Graphs [{split:5s}/{class_label:20s}]",
                      leave=False, disable=not verbose):
            img = cv2.imread(str(p))
            if img is None:
                continue
            mask_p = lbl_dir / p.name
            mask   = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) \
                     if mask_p.exists() else np.zeros(img.shape[:2], np.uint8)
            mask_bin = (mask > 127).astype(np.uint8)
            g = image_to_graph(img, mask_bin, CLASS_TO_IDX[class_label])
            graphs.append(g)

    print(f"  Built {len(graphs)} graphs for split='{split}'")
    return graphs


if __name__ == "__main__":
    print("=" * 60)
    print("  STAGE 5 — GRAPH CONSTRUCTION (rKNN)")
    print("=" * 60)
    train_graphs = build_graph_dataset("train", verbose=True)
    print(f"\n  Example graph: {train_graphs[0]}")
