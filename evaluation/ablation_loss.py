"""
evaluation/ablation_loss.py
─────────────────────────────────────────────
Ablation: Loss Function Comparison
  1. Standard Cross-Entropy
  2. Focal Cross-Entropy  (gamma=2, no smoothing)
  3. Focal CE + Label Smoothing  (gamma=2, eps=0.1)  ← default

All three trained on DenseNet201 (with CCA), same hyperparams.
─────────────────────────────────────────────
Run:
    python evaluation/ablation_loss.py
"""

import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOP_PAT, LR_PATIENCE, NUM_WORKERS,
    MODELS_DIR, LOGS_DIR, RESULTS_DIR, SEED, CLASSES,
)

NUM_EPOCHS = 15  # Override for graph ablation
from utils.datasets import CancerClassificationDataset

torch.manual_seed(SEED)
NUM_CLASSES = len(CLASSES)


# ── Loss definitions ──────────────────────────────────────────────────────

class StandardCELoss(nn.Module):
    """Standard cross-entropy with class weights."""
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.weight)


class FocalCELoss(nn.Module):
    """Focal CE — no label smoothing."""
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        log_p  = F.log_softmax(logits, dim=-1)
        loss   = F.nll_loss(log_p, targets, weight=self.weight,
                             reduction="none")
        pt     = torch.exp(-loss)
        focal  = (1 - pt) ** self.gamma * loss
        return focal.mean()


class FocalCELabelSmoothLoss(nn.Module):
    """Focal CE + Label Smoothing — pipeline default."""
    def __init__(self, gamma: float = 2.0,
                 smoothing: float = 0.1, weight=None):
        super().__init__()
        self.gamma     = gamma
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits, targets):
        log_p  = F.log_softmax(logits, dim=-1)
        smooth = torch.full_like(log_p,
                                 self.smoothing / (NUM_CLASSES - 1))
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss   = -(smooth * log_p).sum(dim=-1)
        pt     = torch.exp(-loss)
        focal  = (1 - pt) ** self.gamma * loss
        if self.weight is not None:
            focal = focal * self.weight[targets]
        return focal.mean()


LOSS_CONFIGS = [
    ("Standard_CE",              StandardCELoss),
    ("Focal_CE",                 FocalCELoss),
    ("Focal_CE_LabelSmooth",     FocalCELabelSmoothLoss),
]


# ── Build model ───────────────────────────────────────────────────────────

def build_model():
    import torchvision.models as tv
    from models.attention import ColourChannelAttention
    model = tv.densenet201(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(
        model.classifier.in_features, NUM_CLASSES)
    fc = model.features.conv0
    model.features.conv0 = nn.Sequential(
        fc, ColourChannelAttention(fc.out_channels))
    return model


# ── Train / eval ──────────────────────────────────────────────────────────

def train_and_eval(loss_label: str, criterion,
                   train_dl, val_dl, test_dl,
                   device) -> dict:
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from sklearn.metrics import (accuracy_score, f1_score,
                                  precision_score, recall_score)

    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=LR_PATIENCE)

    best_val_acc = 0.0
    patience_ctr = 0
    ckpt = MODELS_DIR / f"ablation_loss_{loss_label}.pth"
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for imgs, labels in tqdm(
                train_dl,
                desc=f"  [{loss_label}] Ep {epoch:3d} Train",
                leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(imgs), labels).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs = imgs.to(device)
                val_preds.extend(
                    model(imgs).argmax(1).cpu().tolist())
                val_true.extend(labels.tolist())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1  = f1_score(val_true, val_preds,
                            average="macro", zero_division=0)
        scheduler.step(val_acc)

        pct = epoch / NUM_EPOCHS * 100
        print(f"  [{loss_label}] [{pct:5.1f}%] "
              f"Ep {epoch:3d}  Val Acc: {val_acc*100:.2f}%")

        history.append({"epoch": epoch,
                         "val_acc": val_acc,
                         "val_f1":  val_f1})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(ckpt))
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= EARLY_STOP_PAT:
            print(f"  [{loss_label}] Early stop at epoch {epoch}")
            break

    model.load_state_dict(
        torch.load(str(ckpt), map_location=device))
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs = imgs.to(device)
            test_preds.extend(
                model(imgs).argmax(1).cpu().tolist())
            test_true.extend(labels.tolist())

    return {
        "label":       loss_label,
        "test_acc":    accuracy_score(test_true, test_preds),
        "test_f1":     f1_score(test_true, test_preds,
                                 average="macro", zero_division=0),
        "test_prec":   precision_score(test_true, test_preds,
                                        average="macro",
                                        zero_division=0),
        "test_rec":    recall_score(test_true, test_preds,
                                     average="macro",
                                     zero_division=0),
        "history":     history,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def run_loss_ablation():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print("=" * 60)
    print("  ABLATION — Loss Function Comparison")
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

    counts  = train_ds.class_counts()
    total   = sum(counts.values())
    weights = torch.tensor(
        [total / (NUM_CLASSES * counts.get(i, 1))
         for i in range(NUM_CLASSES)],
        dtype=torch.float32).to(device)

    results = []
    for label, LossCls in LOSS_CONFIGS:
        criterion = LossCls(weight=weights) \
            if "CE" in label else LossCls()
        r = train_and_eval(label, criterion,
                           train_dl, val_dl, test_dl, device)
        results.append(r)

    # ── Summary ────────────────────────────────────────────────────
    print("\n  ── Loss Ablation Results ────────────────────────")
    print(f"  {'Loss Function':28s} {'Acc':>8s} {'F1':>8s} "
          f"{'Prec':>8s} {'Recall':>8s}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['label']:28s} "
              f"{r['test_acc']*100:7.2f}%  "
              f"{r['test_f1']:8.4f}  "
              f"{r['test_prec']:8.4f}  "
              f"{r['test_rec']:8.4f}")

    # ── CSV ────────────────────────────────────────────────────────
    csv_path = LOGS_DIR / "ablation_loss.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "test_acc", "test_f1",
            "test_prec", "test_rec"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\n  CSV saved → {csv_path}")

    _plot_loss_ablation(results)
    return results


def _plot_loss_ablation(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Loss Function Ablation — DenseNet201", fontsize=13)
    colors = ["#F44336", "#FF9800", "#2196F3"]

    metrics = ["test_acc", "test_f1", "test_prec", "test_rec"]
    xlabels = ["Accuracy", "Macro-F1", "Precision", "Recall"]

    ax = axes[0]
    for i, r in enumerate(results):
        vals = [r[m] for m in metrics]
        ax.bar([xi + i * 0.25 for xi in range(len(metrics))],
               vals, width=0.25,
               label=r["label"].replace("_", " "),
               color=colors[i], alpha=0.85)
    ax.set_xticks([xi + 0.25 for xi in range(len(metrics))])
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0.82, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Test Metrics by Loss Function")
    ax.legend(fontsize=7)

    ax2 = axes[1]
    for i, r in enumerate(results):
        ep  = [h["epoch"]   for h in r["history"]]
        acc = [h["val_acc"] for h in r["history"]]
        ax2.plot(ep, acc, label=r["label"].replace("_", " "),
                 color=colors[i], linewidth=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Convergence by Loss Function")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    out = RESULTS_DIR / "ablation_loss.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Plot saved → {out}")


if __name__ == "__main__":
    run_loss_ablation()
