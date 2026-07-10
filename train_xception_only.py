"""
train_xception_only.py
─────────────────────────────────────────────────────────────────────────────
Trains and evaluates ONLY the Xception backbone (using the fixed,
standalone models/xception.py — no timm dependency), without retraining
the other 14 models whose checkpoints already exist under
config.config.MODELS_DIR.

What this does:
  1. Trains Xception with training/trainer.py's exact train_one_model()
     logic (same optimiser, scheduler, loss, early stopping as the
     other 14 got) — writes SAVED_MODELS/Xception_best.pth and
     LOGS/Xception_log.csv, identical to what train_all_models() would
     have produced for it.
  2. Plots RESULTS/history_Xception.png (loss + accuracy curves) from
     the training run above.
  3. Re-runs evaluation across ALL 15 models (cheap — just test-set
     inference against each existing checkpoint, NOT retraining) so
     RESULTS/model_comparison.png and RESULTS/comparison_table.txt stay
     consistent and now include Xception. Your other 14 checkpoints are
     read-only in this step; nothing is retrained.

Run:
    python train_xception_only.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from config.config import (
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, MODEL_NAMES,
)
from training.trainer import train_one_model, compute_class_weights
from utils.datasets import CancerClassificationDataset
from evaluation.evaluate import (
    evaluate_all, plot_training_history,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print("=" * 60)
    print("  TRAINING — Xception ONLY (other 14 checkpoints untouched)")
    print("=" * 60)

    train_ds = CancerClassificationDataset("train")
    val_ds = CancerClassificationDataset("val")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}")

    class_weights = compute_class_weights(train_ds, device)

    # ── Step 1: train Xception only ────────────────────────────────
    result = train_one_model("Xception", train_dl, val_dl, device,
                             class_weights)
    print(f"\n✔  Xception training complete. "
          f"Best val acc: {result['best_val_acc']*100:.2f}%")

    # ── Step 2: training-history plot for Xception ─────────────────
    plot_training_history([result])

    # ── Step 3: re-evaluate ALL 15 models (inference only — no
    #    retraining — using each model's existing checkpoint) so the
    #    comparison chart/table include Xception alongside the other
    #    14 you've already trained ──
    print("\n" + "=" * 60)
    print("  RE-EVALUATING ALL 15 MODELS (inference only, no retraining)")
    print("  — regenerates model_comparison.png / comparison_table.txt")
    print("=" * 60)
    evaluate_all(MODEL_NAMES)

    print("\n  Done. Check RESULTS_DIR for:")
    print("    - history_Xception.png")
    print("    - cm_Xception.png / roc_Xception.png")
    print("    - model_comparison.png / comparison_table.txt  (updated, all 15)")


if __name__ == "__main__":
    main()