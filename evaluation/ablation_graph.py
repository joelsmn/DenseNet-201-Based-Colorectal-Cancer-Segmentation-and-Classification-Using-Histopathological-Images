"""
evaluation/ablation_graph.py
─────────────────────────────────────────────
Ablation: Graph Construction Components
  1. No graph          (backbone only)
  2. Graph, no PE      (rKNN graph, no positional encoding)
  3. Graph + PE        (rKNN + sinusoidal PE) ← default
  4. Graph + PE + edge (full cellular interaction graph)

Measures test accuracy and F1 using DenseNet201
backbone + graph transformer readout fusion.
─────────────────────────────────────────────
Run:
    python evaluation/ablation_graph.py
"""

import sys
import csv
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    SPLIT_DIR, CLASSES, CLASS_TO_IDX,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOP_PAT, LR_PATIENCE, NUM_WORKERS,
    MODELS_DIR, LOGS_DIR, RESULTS_DIR,
    IMAGE_SIZE, KNN_K, KNN_RADIUS, POS_ENC_DIM, SEED,
)

NUM_EPOCHS = 15  # Override for graph ablation

from graph.graph_construction import (
    extract_node_features, build_rknn_edges, sinusoidal_pe)
from classifier.neurosymbolic_graph_transformer import FocalCrossEntropyLoss

torch.manual_seed(SEED)
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

import torchvision.transforms as T
_tfm = T.Compose([T.ToTensor(),
                   T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# ── Graph feature builders ────────────────────────────────────────────────

def build_graph_no_pe(img_rgb, mask_bin):
    """rKNN graph without positional encoding."""
    centroids, feats = extract_node_features(img_rgb, mask_bin)
    # Strip PE columns (last POS_ENC_DIM columns) — keep base 6
    feats_no_pe = feats[:, :6]
    edge_index  = build_rknn_edges(centroids)
    return feats_no_pe, centroids, edge_index


def build_graph_with_pe(img_rgb, mask_bin):
    """rKNN graph with sinusoidal PE — pipeline default."""
    centroids, feats = extract_node_features(img_rgb, mask_bin)
    edge_index = build_rknn_edges(centroids)
    return feats, centroids, edge_index


def build_graph_with_pe_edge(img_rgb, mask_bin):
    """rKNN graph with PE + edge features."""
    from graph.cellular_interaction_graph import (
        image_to_cellular_graph)
    # Returns Data object; extract tensors
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    data = image_to_cellular_graph(bgr, mask_bin)
    feats      = data.x.numpy()
    centroids  = data.pos.numpy()
    edge_index = data.edge_index.numpy()
    return feats, centroids, edge_index


GRAPH_CONFIGS = [
    ("No_Graph",           None),
    ("Graph_No_PE",        build_graph_no_pe),
    ("Graph_With_PE",      build_graph_with_pe),
    ("Graph_PE_EdgeFeat",  build_graph_with_pe_edge),
]


# ── Simple graph transformer readout ─────────────────────────────────────

class SimpleGraphReadout(nn.Module):
    """
    Lightweight 2-layer graph transformer for ablation.
    Input : (N, node_in) node features
    Output: (graph_dim,) graph-level embedding
    """
    def __init__(self, node_in: int, graph_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(node_in, graph_dim)
        self.attn = nn.MultiheadAttention(
            graph_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(graph_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N, node_in)
        x = self.proj(x).unsqueeze(0)      # (1, N, D)
        x, _ = self.attn(x, x, x)
        x = self.norm(x)                   # (1, N, D)
        x = x.squeeze(0).T.unsqueeze(0)   # (1, D, N)
        x = self.pool(x).squeeze()         # (D,)
        return x


# ── Combined classifier ───────────────────────────────────────────────────

class BackboneGraphClassifier(nn.Module):
    def __init__(self, node_in: int, graph_dim: int = 128,
                 use_graph: bool = True):
        super().__init__()
        import torchvision.models as tv
        from models.attention import ColourChannelAttention

        self.use_graph = use_graph
        base = tv.densenet201(weights="IMAGENET1K_V1")
        feat_dim = base.classifier.in_features
        base.classifier = nn.Identity()
        fc = base.features.conv0
        base.features.conv0 = nn.Sequential(
            fc, ColourChannelAttention(fc.out_channels))
        self.backbone = base

        if use_graph:
            self.graph_readout = SimpleGraphReadout(
                node_in, graph_dim)
            fused_dim = feat_dim + graph_dim
        else:
            fused_dim = feat_dim

        self.head = nn.Linear(fused_dim, NUM_CLASSES)

    def forward(self, img_tensor: torch.Tensor,
                graph_feat: torch.Tensor = None) -> torch.Tensor:
        feat = self.backbone(img_tensor)          # (B, feat_dim)
        if self.use_graph and graph_feat is not None:
            g = self.graph_readout(graph_feat)    # (graph_dim,)
            g = g.unsqueeze(0).expand(feat.size(0), -1)
            feat = torch.cat([feat, g], dim=-1)
        return self.head(feat)


# ── Dataset ───────────────────────────────────────────────────────────────

class GraphAblationDataset(Dataset):
    def __init__(self, split: str, graph_fn):
        self.graph_fn = graph_fn
        self.samples  = []

        for cls in CLASSES:
            img_dir = SPLIT_DIR / split / cls / "Image"
            lbl_dir = SPLIT_DIR / split / cls / "Label"
            if not img_dir.exists():
                continue
            for p in sorted(img_dir.glob("*")):
                if p.suffix.lower() not in {
                        ".png", ".jpg", ".jpeg",
                        ".tif", ".tiff", ".bmp"}:
                    continue
                lp = lbl_dir / p.name
                self.samples.append(
                    (p, lp if lp.exists() else None,
                     CLASS_TO_IDX[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, lp, label = self.samples[idx]
        img_bgr = cv2.imread(str(p))
        img_rgb = cv2.cvtColor(
            img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
        img_tensor = _tfm(img_rgb)

        graph_feat = None
        if self.graph_fn is not None:
            mask = cv2.imread(str(lp),
                              cv2.IMREAD_GRAYSCALE) \
                   if lp and lp.exists() \
                   else np.zeros(IMAGE_SIZE, np.uint8)
            mask_bin = (mask > 127).astype(np.uint8)
            try:
                feats, _, _ = self.graph_fn(img_rgb, mask_bin)
                graph_feat  = torch.tensor(
                    feats, dtype=torch.float32)
            except Exception:
                graph_feat = None

        return img_tensor, graph_feat, \
               torch.tensor(label, dtype=torch.long)


def _collate(batch):
    imgs    = torch.stack([b[0] for b in batch])
    labels  = torch.stack([b[2] for b in batch])
    # graph_feat is variable-size — keep as list
    graphs  = [b[1] for b in batch]
    return imgs, graphs, labels


# ── Train / eval ──────────────────────────────────────────────────────────

def train_and_eval_graph(config_label: str,
                         graph_fn,
                         device) -> dict:
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from sklearn.metrics import (accuracy_score, f1_score,
                                  precision_score, recall_score)

    use_graph = (graph_fn is not None)

    # Determine node_in dimension
    if graph_fn == build_graph_no_pe:
        node_in = 6
    elif graph_fn == build_graph_with_pe:
        node_in = 6 + POS_ENC_DIM
    elif graph_fn == build_graph_with_pe_edge:
        node_in = 6 + POS_ENC_DIM   # edge feats handled separately
    else:
        node_in = 6

    print(f"\n  ── Config: {config_label} "
          f"(use_graph={use_graph}, node_in={node_in}) ──")

    train_ds = GraphAblationDataset("train", graph_fn)
    val_ds   = GraphAblationDataset("val",   graph_fn)
    test_ds  = GraphAblationDataset("test",  graph_fn)

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=_collate, num_workers=0)
    val_dl   = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=_collate, num_workers=0)
    test_dl  = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=_collate, num_workers=0)

    model = BackboneGraphClassifier(
        node_in=node_in, graph_dim=128,
        use_graph=use_graph).to(device)

    criterion = FocalCrossEntropyLoss(gamma=2.0, smoothing=0.1)
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=LR_PATIENCE)

    best_val_acc = 0.0
    patience_ctr = 0
    ckpt = MODELS_DIR / f"ablation_graph_{config_label}.pth"
    history = []

    def _forward_batch(imgs, graphs, labels):
        imgs   = imgs.to(device)
        labels = labels.to(device)
        # Use first graph in batch (representative)
        g_feat = None
        if use_graph and graphs[0] is not None:
            g_feat = graphs[0].to(device)
        return model(imgs, g_feat), labels

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for imgs, graphs, labels in tqdm(
                train_dl,
                desc=f"  [{config_label}] Ep {epoch:3d}",
                leave=False):
            optimizer.zero_grad()
            out, tgts = _forward_batch(imgs, graphs, labels)
            criterion(out, tgts).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for imgs, graphs, labels in val_dl:
                out, _ = _forward_batch(imgs, graphs, labels)
                val_preds.extend(out.argmax(1).cpu().tolist())
                val_true.extend(labels.tolist())

        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(val_acc)
        pct = epoch / NUM_EPOCHS * 100
        print(f"  [{config_label}] [{pct:5.1f}%]"
              f"  Ep {epoch:3d}  Val Acc: {val_acc*100:.2f}%")

        history.append({"epoch": epoch, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(ckpt))
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= EARLY_STOP_PAT:
            print(f"  [{config_label}] Early stop ep {epoch}")
            break

    model.load_state_dict(
        torch.load(str(ckpt), map_location=device))
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for imgs, graphs, labels in test_dl:
            out, _ = _forward_batch(imgs, graphs, labels)
            test_preds.extend(out.argmax(1).cpu().tolist())
            test_true.extend(labels.tolist())

    return {
        "label":     config_label,
        "use_graph": use_graph,
        "test_acc":  accuracy_score(test_true, test_preds),
        "test_f1":   f1_score(test_true, test_preds,
                               average="macro", zero_division=0),
        "test_prec": precision_score(test_true, test_preds,
                                      average="macro",
                                      zero_division=0),
        "test_rec":  recall_score(test_true, test_preds,
                                   average="macro",
                                   zero_division=0),
        "history":   history,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def run_graph_ablation():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print("=" * 60)
    print("  ABLATION — Graph Construction Components")
    print("=" * 60)

    results = []
    for label, graph_fn in GRAPH_CONFIGS:
        r = train_and_eval_graph(label, graph_fn, device)
        results.append(r)

    # ── Summary ────────────────────────────────────────────────────
    print("\n  ── Graph Ablation Results ───────────────────────")
    print(f"  {'Configuration':26s} {'Acc':>8s} "
          f"{'F1':>8s} {'Prec':>8s} {'Recall':>8s}")
    print("  " + "-" * 62)
    for r in results:
        print(f"  {r['label']:26s} "
              f"{r['test_acc']*100:7.2f}%  "
              f"{r['test_f1']:8.4f}  "
              f"{r['test_prec']:8.4f}  "
              f"{r['test_rec']:8.4f}")

    # ── CSV ────────────────────────────────────────────────────────
    csv_path = LOGS_DIR / "ablation_graph.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "use_graph", "test_acc",
            "test_f1", "test_prec", "test_rec"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\n  CSV saved → {csv_path}")

    _plot_graph_ablation(results)
    return results


def _plot_graph_ablation(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Graph Construction Ablation — DenseNet201",
                 fontsize=13)

    labels = [r["label"].replace("_", "\n") for r in results]
    accs   = [r["test_acc"] * 100 for r in results]
    colors = ["#9E9E9E", "#FF9800", "#2196F3", "#4CAF50"]

    ax = axes[0]
    bars = ax.bar(labels, accs, color=colors,
                  alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f%%", fontsize=8)
    ax.set_ylim(80, 102)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy by Graph Configuration")

    ax2 = axes[1]
    for i, r in enumerate(results):
        ep  = [h["epoch"]   for h in r["history"]]
        acc = [h["val_acc"] for h in r["history"]]
        ax2.plot(ep, acc,
                 label=r["label"].replace("_", " "),
                 color=colors[i], linewidth=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Convergence by Graph Config")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    out = RESULTS_DIR / "ablation_graph.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Plot saved → {out}")


if __name__ == "__main__":
    run_graph_ablation()
