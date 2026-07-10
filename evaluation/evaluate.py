"""
evaluation/evaluate.py
─────────────────────────────────────────────────────────────────────────────
OUTPUT STAGE — Evaluation & Visualisation

For every trained model:
  • Run inference on the TEST split
  • Compute Accuracy, Precision, Recall, F1 (macro + per-class)
  • Plot & save Confusion Matrix
  • Plot & save ROC curves (OvR)
  • Summarise all models in a comparative bar chart + table

Output files written to RESULTS_DIR/
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    CLASSES, IDX_TO_CLASS, MODELS_DIR, RESULTS_DIR,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SEED,
)
from models.backbones import get_model
from utils.datasets  import CancerClassificationDataset

NUM_CLASSES = len(CLASSES)
torch.manual_seed(SEED)


# ─── Inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model_name: str, device: torch.device) -> tuple:
    """
    Load best checkpoint and run inference on test split.
    Returns (y_true, y_pred, y_prob).
    """
    ckpt_path = MODELS_DIR / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        print(f"  [WARN] Checkpoint not found: {ckpt_path}")
        return None, None, None

    model = get_model(model_name, pretrained=False)
    model.load_state_dict(
        torch.load(str(ckpt_path), map_location=device))
    model.to(device).eval()

    test_ds = CancerClassificationDataset("test")
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    y_true, y_pred, y_prob_list = [], [], []
    n_batches = len(test_dl)

    for batch_idx, (imgs, labels) in enumerate(test_dl):
        imgs = imgs.to(device)
        out  = model(imgs)
        prob = torch.softmax(out, dim=1).cpu().numpy()
        pred = out.argmax(dim=1).cpu().numpy()

        y_true.extend(labels.numpy())
        y_pred.extend(pred)
        y_prob_list.append(prob)

        pct = (batch_idx + 1) / n_batches * 100
        print(f"\r  Inferring [{model_name:20s}]  [{pct:5.1f}%]",
              end="", flush=True)

    print()
    y_prob = np.concatenate(y_prob_list, axis=0)
    return np.array(y_true), np.array(y_pred), y_prob


# ─── Confusion Matrix ─────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp.plot(ax=ax, colorbar=True, cmap="Blues",
              xticks_rotation=30)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    out_path = RESULTS_DIR / f"cm_{model_name}.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {out_path}")


# ─── ROC curves ───────────────────────────────────────────────────────────

def plot_roc(y_true, y_prob, model_name: str) -> None:
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=1.5,
                label=f"{cls}  (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (OvR) — {model_name}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_path = RESULTS_DIR / f"roc_{model_name}.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  ROC curve saved       → {out_path}")


# ─── Per-model summary ────────────────────────────────────────────────────

def evaluate_model(model_name: str,
                   device: torch.device) -> dict:
    y_true, y_pred, y_prob = run_inference(model_name, device)
    if y_true is None:
        return {"model_name": model_name, "error": "No checkpoint"}

    report = classification_report(
        y_true, y_pred, target_names=CLASSES,
        output_dict=True, zero_division=0)

    acc       = report["accuracy"]
    macro_f1  = report["macro avg"]["f1-score"]
    macro_prec= report["macro avg"]["precision"]
    macro_rec = report["macro avg"]["recall"]

    print(f"\n  {model_name}")
    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    Macro-F1  : {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASSES,
                                 zero_division=0))

    plot_confusion_matrix(y_true, y_pred, model_name)
    plot_roc(y_true, y_prob, model_name)

    return {
        "model_name":  model_name,
        "accuracy":    acc,
        "macro_f1":    macro_f1,
        "macro_prec":  macro_prec,
        "macro_recall":macro_rec,
        "per_class":   {c: report[c] for c in CLASSES},
    }


# ─── Comparative visualisation ────────────────────────────────────────────

def plot_comparison(results: list) -> None:
    """Bar chart + table comparing all models."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("  No valid results to plot.")
        return

    names    = [r["model_name"]  for r in valid]
    accs     = [r["accuracy"]    for r in valid]
    f1s      = [r["macro_f1"]    for r in valid]
    precs    = [r["macro_prec"]  for r in valid]
    recs     = [r["macro_recall"]for r in valid]

    x    = np.arange(len(names))
    w    = 0.2

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - 1.5*w, accs,  w, label="Accuracy",  color="#4C72B0")
    ax.bar(x - 0.5*w, f1s,   w, label="Macro F1",  color="#55A868")
    ax.bar(x + 0.5*w, precs, w, label="Precision", color="#C44E52")
    ax.bar(x + 1.5*w, recs,  w, label="Recall",    color="#8172B2")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Comparative Model Performance — Colorectal Cancer Detection",
                 fontsize=13)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = RESULTS_DIR / "model_comparison.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"\n  Comparison chart → {path}")

    # ── Text table ─────────────────────────────────────────────────────
    header = f"{'Model':25s} {'Accuracy':>10s} {'Macro-F1':>10s} {'Precision':>11s} {'Recall':>9s}"
    sep    = "-" * len(header)
    rows   = [header, sep]
    for r in valid:
        rows.append(
            f"  {r['model_name']:23s} "
            f"{r['accuracy']*100:9.2f}% "
            f"{r['macro_f1']:10.4f}  "
            f"{r['macro_prec']:10.4f}  "
            f"{r['macro_recall']:8.4f}")

    table_str = "\n".join(rows)
    print("\n" + table_str)

    table_path = RESULTS_DIR / "comparison_table.txt"
    table_path.write_text(table_str)
    print(f"\n  Table saved → {table_path}")


# ─── Training history plots ───────────────────────────────────────────────

def plot_training_history(results: list) -> None:
    """
    For each model that has history, plot loss + accuracy curves.
    """
    for r in results:
        if "history" not in r:
            continue
        h    = r["history"]
        name = r["model_name"]
        ep   = range(1, len(h["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(ep, h["train_loss"], label="Train")
        ax1.plot(ep, h["val_loss"],   label="Val")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title(f"{name} — Loss"); ax1.legend()

        ax2.plot(ep, [a*100 for a in h["train_acc"]], label="Train")
        ax2.plot(ep, [a*100 for a in h["val_acc"]],   label="Val")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
        ax2.set_title(f"{name} — Accuracy"); ax2.legend()

        plt.tight_layout()
        path = RESULTS_DIR / f"history_{name}.png"
        plt.savefig(str(path), dpi=120)
        plt.close()

    print(f"  Training history plots saved → {RESULTS_DIR}")


# ─── Evaluate all models ──────────────────────────────────────────────────

def evaluate_all(model_names: list = None) -> list:
    from config.config import MODEL_NAMES as ALL_NAMES
    if model_names is None:
        model_names = ALL_NAMES

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    print("=" * 60)
    print("  EVALUATION — ALL MODELS")
    print("=" * 60)

    for i, name in enumerate(model_names):
        pct = (i + 1) / len(model_names) * 100
        print(f"\n  ── {i+1}/{len(model_names)}  ({pct:.0f}%) : {name}")
        r = evaluate_model(name, device)
        results.append(r)

    plot_comparison(results)
    return results


# ─── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate_all()


