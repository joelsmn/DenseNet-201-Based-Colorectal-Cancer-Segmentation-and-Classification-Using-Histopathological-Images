"""
evaluation/ablation_incremental.py
─────────────────────────────────────────────────────────────────────────────
THE core ablation requested in the ATR:

  Reviewer #1, comments 2 & 6:
    "... a dedicated ablation study that incrementally adds the CCA
     layer, the U-Net-guided graph module, and the KG/PathNet gate on
     top of the DenseNet201 baseline, reporting accuracy and Macro-F1
     at each stage."

  Reviewer #2, comment 4:  identical request, cross-referenced to the
  same ablation table.

Stages (cumulative):
  S0  DenseNet201                                  (baseline)
  S1  + CCA                                         (stem attention)
  S2  + CCA + U-Net-guided graph module              (rKNN + PE, Stage 5)
  S3  + CCA + Graph + KG/PathNet gate                (= full NSGT model)

Each stage's *only* difference from the previous one is the addition of
one module — the rest of the architecture, training budget, optimiser,
and data pipeline are held fixed — so accuracy/Macro-F1 deltas are
attributable to that module alone.

Run:
    python evaluation/ablation_incremental.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import csv
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    SPLIT_DIR, CLASSES, CLASS_TO_IDX,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOP_PAT, LR_PATIENCE, NUM_WORKERS,
    MODELS_DIR, LOGS_DIR, RESULTS_DIR, IMAGE_SIZE,
    KNN_K, KNN_RADIUS, POS_ENC_DIM, SEED,
)

NUM_EPOCHS = 15  # Override for ablation run

from models.attention import ColourChannelAttention, NoAttention
from graph.graph_construction import extract_node_features, build_rknn_edges
from symbolic.knowledge_graph import SymbolicLayer
from classifier.neurosymbolic_graph_transformer import (
    FocalCrossEntropyLoss, GraphTransformer)

from utils.gpu_memory import configure_cudnn_safe_mode, free_gpu_memory

from utils.gpu_memory import configure_cudnn_safe_mode, free_gpu_memory

torch.manual_seed(SEED)
configure_cudnn_safe_mode()
configure_cudnn_safe_mode()
NUM_CLASSES = len(CLASSES)
NODE_IN = 6 + POS_ENC_DIM

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

import torchvision.transforms as T
_tfm = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# ── Stage-configurable model ────────────────────────────────────────────

class IncrementalCRCModel(nn.Module):
    """
    One model class parameterised by which stage's modules are active,
    so every stage shares identical backbone weights initialisation,
    optimiser settings, and forward-pass code paths outside the toggled
    modules.

    stage:
      "S0" -> DenseNet201 only
      "S1" -> + CCA
      "S2" -> + CCA + Graph
      "S3" -> + CCA + Graph + KG/PathNet gate   (full NSGT)
    """

    def __init__(self, stage: str, graph_dim: int = 256, sym_dim: int = 256):
        super().__init__()
        assert stage in ("S0", "S1", "S2", "S3")
        self.stage = stage
        self.use_cca = stage in ("S1", "S2", "S3")
        self.use_graph = stage in ("S2", "S3")
        self.use_symbolic = stage in ("S3",)

        import torchvision.models as tv
        base = tv.densenet201(weights="IMAGENET1K_V1")
        feat_dim = base.classifier.in_features
        base.classifier = nn.Identity()
        fc = base.features.conv0
        attn = ColourChannelAttention(fc.out_channels) if self.use_cca \
            else NoAttention(fc.out_channels)
        base.features.conv0 = nn.Sequential(fc, attn)
        self.backbone = base
        self.feat_dim = feat_dim

        fused_dim = feat_dim
        if self.use_graph:
            self.graph_transformer = GraphTransformer(
                node_in=NODE_IN, node_dim=128, graph_dim=graph_dim,
                n_layers=3, n_heads=4)
            fused_dim += graph_dim
        if self.use_symbolic:
            self.symbolic = SymbolicLayer(feat_dim=feat_dim, node_dim=32,
                                          kg_dim=64, out_dim=sym_dim)
            fused_dim += sym_dim

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))

    def forward(self, img, graph_x=None, graph_pos=None):
        feat = self.backbone(img)                        # (B, feat_dim)
        parts = [feat]

        if self.use_graph and graph_x is not None:
            g = self.graph_transformer(graph_x, graph_pos)   # (graph_dim,)
            g = g.unsqueeze(0).expand(feat.size(0), -1)
            parts.append(g)

        if self.use_symbolic:
            s = self.symbolic(feat)                       # (B, sym_dim)
            parts.append(s)

        fused = torch.cat(parts, dim=-1)
        return self.head(fused)


# ── Dataset (image + optional graph features from U-Net-quality mask) ─────

class IncrementalDataset(Dataset):
    def __init__(self, split: str, use_graph: bool):
        self.use_graph = use_graph
        self.samples = []
        for cls in CLASSES:
            img_dir = SPLIT_DIR / split / cls / "Image"
            lbl_dir = SPLIT_DIR / split / cls / "Label"
            if not img_dir.exists():
                continue
            for p in sorted(img_dir.glob("*")):
                if p.suffix.lower() not in {".png", ".jpg", ".jpeg",
                                            ".tif", ".tiff", ".bmp"}:
                    continue
                lp = lbl_dir / p.name
                self.samples.append(
                    (p, lp if lp.exists() else None, CLASS_TO_IDX[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, lp, label = self.samples[idx]
        img_bgr = cv2.imread(str(p))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
        img_tensor = _tfm(img_rgb)

        graph_feat, graph_pos = None, None
        if self.use_graph:
            mask = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE) \
                if lp and lp.exists() else np.zeros(IMAGE_SIZE, np.uint8)
            mask_bin = (mask > 127).astype(np.uint8)
            try:
                centroids, feats = extract_node_features(img_rgb, mask_bin)
                graph_feat = torch.tensor(feats, dtype=torch.float32)
                graph_pos = torch.tensor(centroids, dtype=torch.float32)
            except Exception:
                graph_feat, graph_pos = None, None

        return img_tensor, graph_feat, graph_pos, \
               torch.tensor(label, dtype=torch.long)


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[3] for b in batch])
    graph_feats = [b[1] for b in batch]
    graph_pos = [b[2] for b in batch]
    return imgs, graph_feats, graph_pos, labels


# ── Train / eval per stage ──────────────────────────────────────────────

def train_and_eval_stage(stage: str, label: str, device,
                         class_weights) -> dict:
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from sklearn.metrics import (accuracy_score, f1_score,
                                  precision_score, recall_score)

    use_graph = stage in ("S2", "S3")
    print(f"\n  ── Stage {stage}: {label} ──")

    train_ds = IncrementalDataset("train", use_graph)
    val_ds = IncrementalDataset("val", use_graph)
    test_ds = IncrementalDataset("test", use_graph)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=_collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=_collate, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=_collate, num_workers=0)

    model = IncrementalCRCModel(stage).to(device)
    criterion = FocalCrossEntropyLoss(gamma=2.0, smoothing=0.1,
                                      weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=LR_PATIENCE)

    best_val_acc, patience_ctr = 0.0, 0
    ckpt = MODELS_DIR / f"ablation_incremental_{stage}.pth"
    history = []

    def _forward_batch(imgs, gfeats, gpos, labels):
        imgs, labels = imgs.to(device), labels.to(device)
        gx = gpos_t = None
        if use_graph and gfeats[0] is not None:
            gx = gfeats[0].to(device)
            gpos_t = gpos[0].to(device)
        return model(imgs, gx, gpos_t), labels

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for imgs, gfeats, gpos, labels in tqdm(
                train_dl, desc=f"  [{stage}] Ep {epoch:3d}", leave=False):
            optimizer.zero_grad()
            out, tgts = _forward_batch(imgs, gfeats, gpos, labels)
            criterion(out, tgts).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for imgs, gfeats, gpos, labels in val_dl:
                out, tgts = _forward_batch(imgs, gfeats, gpos, labels)
                val_preds.extend(out.argmax(1).cpu().tolist())
                val_true.extend(tgts.cpu().tolist())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="macro",
                          zero_division=0)
        scheduler.step(val_acc)

        pct = epoch / NUM_EPOCHS * 100
        print(f"  [{stage}] [{pct:5.1f}%] Ep {epoch:3d}  "
              f"Val Acc: {val_acc*100:.2f}%  Val F1: {val_f1:.4f}")
        history.append({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(ckpt))
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= EARLY_STOP_PAT:
            print(f"  [{stage}] Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for imgs, gfeats, gpos, labels in test_dl:
            out, tgts = _forward_batch(imgs, gfeats, gpos, labels)
            test_preds.extend(out.argmax(1).cpu().tolist())
            test_true.extend(tgts.cpu().tolist())

    test_acc = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    test_prec = precision_score(test_true, test_preds, average="macro",
                                zero_division=0)
    test_rec = recall_score(test_true, test_preds, average="macro",
                            zero_division=0)

    # Single-image inference time (representative graph from val set)
    sample = val_ds[0]
    dummy_img = sample[0].unsqueeze(0).to(device)
    dummy_gx = sample[1].to(device) if use_graph and sample[1] is not None else None
    dummy_gpos = sample[2].to(device) if use_graph and sample[2] is not None else None
    with torch.no_grad():
        for _ in range(10):
            model(dummy_img, dummy_gx, dummy_gpos)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            model(dummy_img, dummy_gx, dummy_gpos)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) / 30 * 1000

    n_params = sum(p.numel() for p in model.parameters())

    result = {
        "stage": stage, "label": label,
        "test_acc": test_acc, "test_f1": test_f1,
        "test_prec": test_prec, "test_rec": test_rec,
        "best_val_acc": best_val_acc,
        "n_params": n_params, "infer_ms": infer_ms,
        "history": history,
    }

    free_gpu_memory(model, optimizer, dummy_img, dummy_gx, dummy_gpos)
    return result

# ── Main ──────────────────────────────────────────────────────────────────

STAGE_LABELS = [
    ("S0", "DenseNet201 (baseline)"),
    ("S1", "+ CCA"),
    ("S2", "+ CCA + U-Net-guided Graph"),
    ("S3", "+ CCA + Graph + KG/PathNet gate (full NSGT)"),
]


def run_incremental_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print("=" * 70)
    print("  INCREMENTAL ABLATION — DenseNet201 -> +CCA -> +Graph -> +KG Gate")
    print("  (ATR Reviewer #1 comments 2 & 6 / Reviewer #2 comment 4)")
    print("=" * 70)

    train_ds = IncrementalDataset("train", use_graph=False)
    counts = {}
    for _, _, label in train_ds.samples:
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    weights = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        weights[i] = total / (NUM_CLASSES * counts.get(i, 1))
    weights = weights.to(device)

    results = []
    for stage, label in STAGE_LABELS:
        r = train_and_eval_stage(stage, label, device, weights)
        results.append(r)

    print("\n  ── Incremental Ablation Results "
          "(cumulative gain per module) ──────────────────")
    print(f"  {'Stage':4s} {'Configuration':42s} {'Acc':>8s} {'F1':>8s} "
          f"{'ΔAcc':>8s} {'ΔF1':>8s} {'Params':>12s} {'Infer(ms)':>10s}")
    print("  " + "-" * 100)
    prev_acc = prev_f1 = None
    for r in results:
        d_acc = "" if prev_acc is None else f"{(r['test_acc']-prev_acc)*100:+.2f}%"
        d_f1 = "" if prev_f1 is None else f"{(r['test_f1']-prev_f1):+.4f}"
        print(f"  {r['stage']:4s} {r['label']:42s} "
              f"{r['test_acc']*100:7.2f}%  {r['test_f1']:8.4f}  "
              f"{d_acc:>8s}  {d_f1:>8s}  {r['n_params']:12,d}  "
              f"{r['infer_ms']:10.2f}")
        prev_acc, prev_f1 = r["test_acc"], r["test_f1"]

    csv_path = LOGS_DIR / "ablation_incremental.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "stage", "label", "test_acc", "test_f1", "test_prec", "test_rec",
            "best_val_acc", "n_params", "infer_ms"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\n  CSV saved -> {csv_path}")

    _plot_incremental(results)
    return results


def _plot_incremental(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Incremental Ablation — DenseNet201 -> +CCA -> +Graph -> "
                 "+KG/PathNet Gate", fontsize=13)

    stages = [r["stage"] for r in results]
    accs = [r["test_acc"] * 100 for r in results]
    f1s = [r["test_f1"] * 100 for r in results]

    ax = axes[0]
    ax.plot(stages, accs, marker="o", label="Accuracy", color="#2196F3",
            linewidth=2)
    ax.plot(stages, f1s, marker="s", label="Macro-F1", color="#F44336",
            linewidth=2)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.annotate(f"{a:.2f}%", (i, a), textcoords="offset points",
                   xytext=(0, 8), ha="center", fontsize=8, color="#2196F3")
        ax.annotate(f"{f:.2f}%", (i, f), textcoords="offset points",
                   xytext=(0, -14), ha="center", fontsize=8, color="#F44336")
    ax.set_ylabel("%"); ax.set_title("Cumulative Test Metrics")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax2 = axes[1]
    deltas_acc = [0] + [
        (results[i]["test_acc"] - results[i-1]["test_acc"]) * 100
        for i in range(1, len(results))]
    ax2.bar(stages, deltas_acc, color="#4CAF50")
    ax2.set_ylabel("Δ Accuracy (pp) vs previous stage")
    ax2.set_title("Per-Module Accuracy Contribution")
    ax2.axhline(0, color="black", linewidth=0.8)

    ax3 = axes[2]
    for r in results:
        epochs = [h["epoch"] for h in r["history"]]
        f1_curve = [h["val_f1"] * 100 for h in r["history"]]
        ax3.plot(epochs, f1_curve, label=r["stage"], linewidth=1.8)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Validation Macro-F1 (%)")
    ax3.set_title("Convergence per Stage"); ax3.legend(fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "ablation_incremental.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Plot saved -> {out}")


if __name__ == "__main__":
    run_incremental_ablation()
