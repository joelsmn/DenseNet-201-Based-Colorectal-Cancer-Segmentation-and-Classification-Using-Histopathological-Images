"""
evaluation/ablation_cca.py
─────────────────────────────────────────────
Ablation: Effect of Colour Channel Attention (CCA)
Trains DenseNet201 with and without the CCA layer
and compares accuracy, F1, and convergence speed.
─────────────────────────────────────────────
Run:
    python evaluation/ablation_cca.py
"""

import sys
import csv
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOP_PAT, LR_PATIENCE, NUM_WORKERS,
    MODELS_DIR, LOGS_DIR, SEED, CLASSES,
)

NUM_EPOCHS = 15  # Override for ablation run
from utils.datasets import CancerClassificationDataset
from classifier.neurosymbolic_graph_transformer import FocalCrossEntropyLoss

torch.manual_seed(SEED)
NUM_CLASSES = len(CLASSES)


# ── Build DenseNet201 WITH and WITHOUT CCA ────────────────────────────────

def build_densenet_with_cca():
    import torchvision.models as tv
    from models.attention import ColourChannelAttention
    model = tv.densenet201(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(
        model.classifier.in_features, NUM_CLASSES)
    fc = model.features.conv0
    model.features.conv0 = nn.Sequential(
        fc, ColourChannelAttention(fc.out_channels))
    return model


def build_densenet_without_cca():
    import torchvision.models as tv
    model = tv.densenet201(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(
        model.classifier.in_features, NUM_CLASSES)
    return model


# ── Training loop ─────────────────────────────────────────────────────────

def train_and_eval(model, label: str,
                   train_dl, val_dl, test_dl,
                   device, class_weights) -> dict:
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from sklearn.metrics import (accuracy_score, f1_score,
                                  precision_score, recall_score)

    model.to(device)
    criterion = FocalCrossEntropyLoss(
        gamma=2.0, smoothing=0.1, weight=class_weights)
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=LR_PATIENCE)

    best_val_acc  = 0.0
    patience_ctr  = 0
    ckpt_path     = MODELS_DIR / f"ablation_cca_{label}.pth"
    history       = []

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────
        model.train()
        for imgs, labels in tqdm(
                train_dl,
                desc=f"  [{label}] Epoch {epoch:3d} Train",
                leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(1).cpu().tolist()
                val_preds.extend(preds)
                val_true.extend(labels.tolist())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1  = f1_score(val_true, val_preds,
                            average="macro", zero_division=0)
        scheduler.step(val_acc)

        pct = epoch / NUM_EPOCHS * 100
        print(f"  [{label}] [{pct:5.1f}%] "
              f"Ep {epoch:3d}  Val Acc: {val_acc*100:.2f}%  "
              f"Val F1: {val_f1:.4f}")

        history.append({"epoch": epoch,
                         "val_acc": val_acc,
                         "val_f1":  val_f1})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(ckpt_path))
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= EARLY_STOP_PAT:
            print(f"  [{label}] Early stop at epoch {epoch}")
            break

    # ── Test ───────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(str(ckpt_path), map_location=device))
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().tolist()
            test_preds.extend(preds)
            test_true.extend(labels.tolist())

    test_acc  = accuracy_score(test_true, test_preds)
    test_f1   = f1_score(test_true, test_preds,
                          average="macro", zero_division=0)
    test_prec = precision_score(test_true, test_preds,
                                 average="macro", zero_division=0)
    test_rec  = recall_score(test_true, test_preds,
                              average="macro", zero_division=0)

    return {
        "label":      label,
        "test_acc":   test_acc,
        "test_f1":    test_f1,
        "test_prec":  test_prec,
        "test_rec":   test_rec,
        "best_val_acc": best_val_acc,
        "history":    history,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def run_cca_ablation():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print("=" * 60)
    print("  ABLATION — CCA Layer (With vs Without)")
    print("=" * 60)

    train_ds = CancerClassificationDataset("train")
    val_ds   = CancerClassificationDataset("val")
    test_ds  = CancerClassificationDataset("test")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=NUM_WORKERS)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)

    # Class weights
    counts = train_ds.class_counts()
    total  = sum(counts.values())
    weights = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        weights[i] = total / (NUM_CLASSES * counts.get(i, 1))
    weights = weights.to(device)

    results = []
    for build_fn, label in [
        (build_densenet_with_cca,    "DenseNet201_WITH_CCA"),
        (build_densenet_without_cca, "DenseNet201_WITHOUT_CCA"),
    ]:
        model = build_fn()
        r = train_and_eval(model, label,
                           train_dl, val_dl, test_dl,
                           device, weights)
        results.append(r)

    # ── Print summary ──────────────────────────────────────────────
    print("\n  ── CCA Ablation Results ─────────────────────────")
    print(f"  {'Configuration':30s} {'Acc':>8s} {'F1':>8s} "
          f"{'Prec':>8s} {'Recall':>8s}")
    print("  " + "-" * 64)
    for r in results:
        print(f"  {r['label']:30s} "
              f"{r['test_acc']*100:7.2f}%  "
              f"{r['test_f1']:8.4f}  "
              f"{r['test_prec']:8.4f}  "
              f"{r['test_rec']:8.4f}")

    # ── Save CSV ───────────────────────────────────────────────────
    csv_path = LOGS_DIR / "ablation_cca.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "test_acc", "test_f1",
            "test_prec", "test_rec", "best_val_acc"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\n  CSV saved → {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────
    _plot_cca_ablation(results)
    return results


def _plot_cca_ablation(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from config.config import RESULTS_DIR

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CCA Layer Ablation — DenseNet201", fontsize=13)

    colors = ["#2196F3", "#F44336"]
    metrics = ["test_acc", "test_f1", "test_prec", "test_rec"]
    labels  = ["Accuracy", "Macro-F1", "Precision", "Recall"]

    # Bar chart
    ax = axes[0]
    x  = range(len(metrics))
    for i, r in enumerate(results):
        vals = [r[m] for m in metrics]
        ax.bar([xi + i * 0.35 for xi in x], vals,
               width=0.35, label=r["label"].replace("DenseNet201_", ""),
               color=colors[i], alpha=0.85)
    ax.set_xticks([xi + 0.175 for xi in x])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Test Metrics")
    ax.legend(fontsize=8)

    # Val accuracy curves
    ax2 = axes[1]
    for i, r in enumerate(results):
        epochs = [h["epoch"]   for h in r["history"]]
        accs   = [h["val_acc"] for h in r["history"]]
        ax2.plot(epochs, accs,
                 label=r["label"].replace("DenseNet201_", ""),
                 color=colors[i], linewidth=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Convergence Curve")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "ablation_cca.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Plot saved → {out}")


if __name__ == "__main__":
    run_cca_ablation()
