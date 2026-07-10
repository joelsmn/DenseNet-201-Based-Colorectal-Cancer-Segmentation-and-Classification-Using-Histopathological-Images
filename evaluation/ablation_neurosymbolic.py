"""
evaluation/ablation_neurosymbolic.py
─────────────────────────────────────────────────────────────────────────────
Ablation: isolates the contribution of the neurosymbolic Symbolic Layer
(KnowledgeGraphEncoder + PathNetGate) on top of a fixed CCA+Graph
backbone — i.e. holds S2 ("+CCA+Graph", from ablation_incremental.py)
constant and only toggles the symbolic module, plus two internal
variants that isolate *why* the symbolic layer helps:

  A. CCA+Graph  (no symbolic layer)                       — baseline
  B. CCA+Graph + KG embedding, NO PathNet gating           — the KG
     class embeddings are simply concatenated to the fused feature
     instead of gating it (tests whether the *gate*, vs. just having
     the embedding available, is what matters)
  C. CCA+Graph + KG + PathNet gate (full symbolic layer)   — proposed

This directly supports ATR Reviewer #2, comment 1:
  "the novelty lies ... in the differentiable PathNet gating mechanism
   that fuses symbolic Knowledge-Graph embeddings with deep visual
   features at inference time, rather than applying static post-hoc
   rules."
by empirically separating "having KG features" (B) from "gating with
them" (C).

Run:
    python evaluation/ablation_neurosymbolic.py
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
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    CLASSES, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOP_PAT, LR_PATIENCE, MODELS_DIR, LOGS_DIR, RESULTS_DIR,
    POS_ENC_DIM, SEED,
)

NUM_EPOCHS = 15

from models.attention import ColourChannelAttention
from symbolic.knowledge_graph import KnowledgeGraphEncoder, PathNetGate
from classifier.neurosymbolic_graph_transformer import (
    FocalCrossEntropyLoss, GraphTransformer)
from evaluation.ablation_incremental import (
    IncrementalDataset, _collate)

from utils.gpu_memory import configure_cudnn_safe_mode, free_gpu_memory

from utils.gpu_memory import configure_cudnn_safe_mode, free_gpu_memory

torch.manual_seed(SEED)
configure_cudnn_safe_mode()
configure_cudnn_safe_mode()
NUM_CLASSES = len(CLASSES)
NODE_IN = 6 + POS_ENC_DIM


# ── KG-concat-no-gate variant (isolates the gating mechanism) ─────────────

class KGConcatNoGate(nn.Module):
    """Uses the same KnowledgeGraphEncoder as the full symbolic layer, but
    concatenates the (mean-pooled) KG class embedding directly onto the
    feature vector instead of using it as a PathNetGate — no gating,
    no learned modulation."""

    def __init__(self, feat_dim: int, kg_dim: int = 64, out_dim: int = 256):
        super().__init__()
        self.kg_encoder = KnowledgeGraphEncoder(node_dim=32, kg_dim=kg_dim)
        self.proj = nn.Linear(feat_dim + kg_dim, out_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        kg_embed = self.kg_encoder.class_embeddings().mean(0)  # (kg_dim,)
        kg_embed = kg_embed.unsqueeze(0).expand(feat.size(0), -1)
        return torch.relu(self.proj(torch.cat([feat, kg_embed], dim=-1)))


# ── Model ────────────────────────────────────────────────────────────────

class NeurosymbolicVariantModel(nn.Module):
    def __init__(self, variant: str, graph_dim: int = 256, sym_dim: int = 256):
        super().__init__()
        assert variant in ("A_NoSymbolic", "B_KGConcatNoGate", "C_FullPathNetGate")
        self.variant = variant

        import torchvision.models as tv
        base = tv.densenet201(weights="IMAGENET1K_V1")
        feat_dim = base.classifier.in_features
        base.classifier = nn.Identity()
        fc = base.features.conv0
        base.features.conv0 = nn.Sequential(
            fc, ColourChannelAttention(fc.out_channels))
        self.backbone = base

        self.graph_transformer = GraphTransformer(
            node_in=NODE_IN, node_dim=128, graph_dim=graph_dim,
            n_layers=3, n_heads=4)

        fused_dim = feat_dim + graph_dim
        if variant == "B_KGConcatNoGate":
            self.symbolic = KGConcatNoGate(feat_dim, kg_dim=64, out_dim=sym_dim)
            fused_dim += sym_dim
        elif variant == "C_FullPathNetGate":
            self.kg_encoder = KnowledgeGraphEncoder(node_dim=32, kg_dim=64)
            self.path_gate = PathNetGate(feat_dim, 64, sym_dim)
            fused_dim += sym_dim

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))

    def forward(self, img, graph_x, graph_pos):
        feat = self.backbone(img)
        g = self.graph_transformer(graph_x, graph_pos)
        g = g.unsqueeze(0).expand(feat.size(0), -1)
        parts = [feat, g]

        if self.variant == "B_KGConcatNoGate":
            parts.append(self.symbolic(feat))
        elif self.variant == "C_FullPathNetGate":
            kg_embed = self.kg_encoder.class_embeddings()
            parts.append(self.path_gate(feat, kg_embed))

        fused = torch.cat(parts, dim=-1)
        return self.head(fused)


VARIANT_LABELS = [
    ("A_NoSymbolic", "CCA + Graph  (no symbolic layer)"),
    ("B_KGConcatNoGate", "CCA + Graph + KG embed (concat, no gate)"),
    ("C_FullPathNetGate", "CCA + Graph + KG + PathNet gate (proposed)"),
]


def train_and_eval_variant(variant: str, label: str, device,
                           class_weights) -> dict:
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from sklearn.metrics import (accuracy_score, f1_score,
                                  precision_score, recall_score)

    print(f"\n  ── Variant {variant}: {label} ──")

    train_ds = IncrementalDataset("train", use_graph=True)
    val_ds = IncrementalDataset("val", use_graph=True)
    test_ds = IncrementalDataset("test", use_graph=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=_collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=_collate, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=_collate, num_workers=0)

    model = NeurosymbolicVariantModel(variant).to(device)
    criterion = FocalCrossEntropyLoss(gamma=2.0, smoothing=0.1,
                                      weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=LR_PATIENCE)

    best_val_acc, patience_ctr = 0.0, 0
    ckpt = MODELS_DIR / f"ablation_nsym_{variant}.pth"
    history = []

    def _fb(imgs, gfeats, gpos, labels):
        imgs, labels = imgs.to(device), labels.to(device)
        gx = gfeats[0].to(device) if gfeats[0] is not None else None
        gp = gpos[0].to(device) if gpos[0] is not None else None
        return model(imgs, gx, gp), labels

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for imgs, gfeats, gpos, labels in tqdm(
                train_dl, desc=f"  [{variant}] Ep {epoch:3d}", leave=False):
            optimizer.zero_grad()
            out, tgts = _fb(imgs, gfeats, gpos, labels)
            criterion(out, tgts).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for imgs, gfeats, gpos, labels in val_dl:
                out, tgts = _fb(imgs, gfeats, gpos, labels)
                val_preds.extend(out.argmax(1).cpu().tolist())
                val_true.extend(tgts.cpu().tolist())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        scheduler.step(val_acc)
        print(f"  [{variant}] Ep {epoch:3d}  Val Acc: {val_acc*100:.2f}%  "
              f"Val F1: {val_f1:.4f}")
        history.append({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(ckpt))
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= EARLY_STOP_PAT:
            print(f"  [{variant}] Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for imgs, gfeats, gpos, labels in test_dl:
            out, tgts = _fb(imgs, gfeats, gpos, labels)
            test_preds.extend(out.argmax(1).cpu().tolist())
            test_true.extend(tgts.cpu().tolist())

    test_acc = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    test_prec = precision_score(test_true, test_preds, average="macro",
                                zero_division=0)
    test_rec = recall_score(test_true, test_preds, average="macro",
                            zero_division=0)
    n_params = sum(p.numel() for p in model.parameters())

    result = {"variant": variant, "label": label, "test_acc": test_acc,
            "test_f1": test_f1, "test_prec": test_prec, "test_rec": test_rec,
            "best_val_acc": best_val_acc, "n_params": n_params,
            "history": history}

    free_gpu_memory(model, optimizer)
    return result


def run_neurosymbolic_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print("=" * 70)
    print("  NEUROSYMBOLIC ABLATION — isolating the KG/PathNet gate")
    print("  (ATR Reviewer #2, comment 1)")
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
    for variant, label in VARIANT_LABELS:
        r = train_and_eval_variant(variant, label, device, weights)
        results.append(r)

    print("\n  ── Neurosymbolic Ablation Results ──────────────────────")
    print(f"  {'Variant':20s} {'Acc':>8s} {'F1':>8s} {'Params':>12s}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['variant']:20s} {r['test_acc']*100:7.2f}%  "
              f"{r['test_f1']:8.4f}  {r['n_params']:12,d}")

    csv_path = LOGS_DIR / "ablation_neurosymbolic.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "label", "test_acc", "test_f1", "test_prec",
            "test_rec", "best_val_acc", "n_params"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\n  CSV saved -> {csv_path}")

    _plot_neurosymbolic(results)
    return results


def _plot_neurosymbolic(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    labels = [r["variant"] for r in results]
    accs = [r["test_acc"] * 100 for r in results]
    f1s = [r["test_f1"] * 100 for r in results]
    x = range(len(labels))
    ax.bar([xi - 0.2 for xi in x], accs, width=0.4, label="Accuracy",
           color="#2196F3")
    ax.bar([xi + 0.2 for xi in x], f1s, width=0.4, label="Macro-F1",
           color="#F44336")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("%")
    ax.set_title("Neurosymbolic Layer Ablation — Gate vs. Concat vs. None")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "ablation_neurosymbolic.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Plot saved -> {out}")


if __name__ == "__main__":
    run_neurosymbolic_ablation()
