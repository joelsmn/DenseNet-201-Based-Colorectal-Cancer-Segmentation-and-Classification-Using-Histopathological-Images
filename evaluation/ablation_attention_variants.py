"""
evaluation/ablation_attention_variants.py
─────────────────────────────────────────────────────────────────────────────
Ablation: Attention-module comparison on the DenseNet201 stem.

Directly answers ATR Reviewer #1, comment 4:
  "The novelty of the CCA module is not clearly established; comparisons
   with recent attention mechanisms such as SE, CBAM, and ECA are needed."

Trains DenseNet201 with {None, SE, CBAM, ECA, CCA} inserted immediately
after conv0 (the stem), holding every other hyperparameter fixed, and
reports test Accuracy + Macro-F1 + Precision + Recall + parameter
overhead + per-image inference-time overhead for each arm.

Run:
    python evaluation/ablation_attention_variants.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import csv
import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOP_PAT, LR_PATIENCE, NUM_WORKERS,
    MODELS_DIR, LOGS_DIR, RESULTS_DIR, SEED, CLASSES,
)

NUM_EPOCHS = 15  # Override for ablation run — matches ablation_cca.py budget

from utils.datasets import CancerClassificationDataset
from classifier.neurosymbolic_graph_transformer import FocalCrossEntropyLoss
from models.attention import ATTENTION_REGISTRY, build_attention, count_params

from utils.gpu_memory import configure_cudnn_safe_mode, free_gpu_memory

torch.manual_seed(SEED)
configure_cudnn_safe_mode()
NUM_CLASSES = len(CLASSES)


# ── Model builder: DenseNet201 + {attention}@stem ──────────────────────────

def build_densenet_with_attention(attn_name: str) -> nn.Module:
    import torchvision.models as tv
    model = tv.densenet201(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    fc = model.features.conv0
    attn = build_attention(attn_name, fc.out_channels)
    model.features.conv0 = nn.Sequential(fc, attn)
    return model


# ── Train / eval (shared with ablation_cca.py pattern) ─────────────────────

def train_and_eval(model, label: str, train_dl, val_dl, test_dl,
                   device, class_weights) -> dict:
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from sklearn.metrics import (accuracy_score, f1_score,
                                  precision_score, recall_score)

    model.to(device)
    criterion = FocalCrossEntropyLoss(gamma=2.0, smoothing=0.1,
                                      weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=LR_PATIENCE)

    best_val_acc, patience_ctr = 0.0, 0
    ckpt_path = MODELS_DIR / f"ablation_attn_{label}.pth"
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for imgs, labels in tqdm(train_dl,
                                 desc=f"  [{label}] Epoch {epoch:3d} Train",
                                 leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(1).cpu().tolist()
                val_preds.extend(preds)
                val_true.extend(labels.tolist())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="macro",
                          zero_division=0)
        scheduler.step(val_acc)

        pct = epoch / NUM_EPOCHS * 100
        print(f"  [{label}] [{pct:5.1f}%] Ep {epoch:3d}  "
              f"Val Acc: {val_acc*100:.2f}%  Val F1: {val_f1:.4f}")
        history.append({"epoch": epoch, "val_acc": val_acc, "val_f1": val_f1})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(ckpt_path))
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= EARLY_STOP_PAT:
            print(f"  [{label}] Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()

    # ── Test metrics ────────────────────────────────────────────────
    test_preds, test_true = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().tolist()
            test_preds.extend(preds)
            test_true.extend(labels.tolist())

    test_acc = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    test_prec = precision_score(test_true, test_preds, average="macro",
                                zero_division=0)
    test_rec = recall_score(test_true, test_preds, average="macro",
                            zero_division=0)

    # ── Inference-time overhead (single-image, matches ATR comp-cost
    #    methodology: 31 ms/image single pass baseline) ────────────────
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(10):        # warm-up
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) / 50 * 1000

    result = {
        "label": label,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_prec": test_prec,
        "test_rec": test_rec,
        "best_val_acc": best_val_acc,
        "n_params": count_params(model),
        "attn_params": count_params(build_attention(label, 64))
                        if label in ATTENTION_REGISTRY else 0,
        "infer_ms": infer_ms,
        "history": history,
    }

    free_gpu_memory(model, optimizer, dummy)
    return result


# ── Main ──────────────────────────────────────────────────────────────────

def run_attention_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print("=" * 60)
    print("  ABLATION — Attention Module Comparison")
    print("  (None / SE / CBAM / ECA / CCA) on DenseNet201 stem")
    print("=" * 60)

    train_ds = CancerClassificationDataset("train")
    val_ds = CancerClassificationDataset("val")
    test_ds = CancerClassificationDataset("test")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS)

    counts = train_ds.class_counts()
    total = sum(counts.values())
    weights = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        weights[i] = total / (NUM_CLASSES * counts.get(i, 1))
    weights = weights.to(device)

    results = []
    for attn_name in ["None", "SE", "CBAM", "ECA", "CCA"]:
        model = build_densenet_with_attention(attn_name)
        r = train_and_eval(model, attn_name, train_dl, val_dl, test_dl,
                           device, weights)
        results.append(r)

    print("\n  ── Attention Ablation Results ──────────────────────────")
    print(f"  {'Module':8s} {'Acc':>8s} {'F1':>8s} {'Prec':>8s} "
          f"{'Recall':>8s} {'Params':>12s} {'Infer(ms)':>10s}")
    print("  " + "-" * 68)
    for r in results:
        print(f"  {r['label']:8s} {r['test_acc']*100:7.2f}%  "
              f"{r['test_f1']:8.4f}  {r['test_prec']:8.4f}  "
              f"{r['test_rec']:8.4f}  {r['n_params']:12,d}  "
              f"{r['infer_ms']:10.2f}")

    csv_path = LOGS_DIR / "ablation_attention_variants.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "test_acc", "test_f1", "test_prec", "test_rec",
            "best_val_acc", "n_params", "attn_params", "infer_ms"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\n  CSV saved -> {csv_path}")

    _plot_attention_ablation(results)
    return results


def _plot_attention_ablation(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Attention-Module Ablation — DenseNet201 Stem "
                 "(None vs SE vs CBAM vs ECA vs CCA)", fontsize=13)

    labels = [r["label"] for r in results]
    colors = ["#9E9E9E", "#4CAF50", "#FF9800", "#9C27B0", "#2196F3"]

    ax = axes[0]
    accs = [r["test_acc"] * 100 for r in results]
    f1s = [r["test_f1"] * 100 for r in results]
    x = range(len(labels))
    ax.bar([xi - 0.2 for xi in x], accs, width=0.4, label="Accuracy",
           color="#2196F3")
    ax.bar([xi + 0.2 for xi in x], f1s, width=0.4, label="Macro-F1",
           color="#FF5722")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("%"); ax.set_title("Test Accuracy vs Macro-F1")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    for i, r in enumerate(results):
        epochs = [h["epoch"] for h in r["history"]]
        f1_curve = [h["val_f1"] * 100 for h in r["history"]]
        ax2.plot(epochs, f1_curve, label=r["label"], color=colors[i],
                 linewidth=1.8)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation Macro-F1 (%)")
    ax2.set_title("Convergence Curve"); ax2.legend(fontsize=8)

    ax3 = axes[2]
    params = [r["attn_params"] for r in results]
    ax3.bar(labels, params, color=colors)
    ax3.set_ylabel("Attention-module parameters")
    ax3.set_title("Parameter Overhead (64-ch stem)")
    for i, p in enumerate(params):
        ax3.text(i, p, f"{p:,}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "ablation_attention_variants.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Plot saved -> {out}")


if __name__ == "__main__":
    run_attention_ablation()
