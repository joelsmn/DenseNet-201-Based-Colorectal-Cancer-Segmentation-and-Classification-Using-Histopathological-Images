"""
training/trainer.py
─────────────────────────────────────────────────────────────────────────────
Stage 11 — MODEL TRAINING ENGINE

Features
  • Trains all 15 backbone models sequentially (or a single selected model)
  • Real-time progress: prints % completion at every step
  • Adam optimiser + ReduceLROnPlateau scheduler
  • Cross-Entropy loss (FocalCrossEntropyLoss with class weights)
  • Early stopping
  • Saves best checkpoint per model
  • Logs train/val loss + accuracy per epoch to CSV
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    CLASSES, CLASS_TO_IDX, MODEL_NAMES,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_PATIENCE, EARLY_STOP_PAT, NUM_WORKERS, PIN_MEMORY,
    SEED, MODELS_DIR, LOGS_DIR, USE_CLASS_WEIGHTS,
    MALIGNANT_CLASSES,
)
from models.backbones    import get_model
from utils.datasets      import CancerClassificationDataset
from classifier.neurosymbolic_graph_transformer import FocalCrossEntropyLoss

torch.manual_seed(SEED)
NUM_CLASSES = len(CLASSES)


# ─── Compute class weights ────────────────────────────────────────────────

def compute_class_weights(dataset: CancerClassificationDataset,
                           device: torch.device) -> torch.Tensor:
    counts = dataset.class_counts()
    total  = sum(counts.values())
    weights = torch.zeros(NUM_CLASSES)
    for idx in range(NUM_CLASSES):
        n = counts.get(idx, 1)
        weights[idx] = total / (NUM_CLASSES * n)
    return weights.to(device)


# ─── Single epoch helpers ─────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer,
               device, is_train: bool, epoch: int, n_epochs: int,
               model_name: str) -> tuple:
    model.train() if is_train else model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    split      = "Train" if is_train else "Val  "

    with torch.set_grad_enabled(is_train):
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            # Forward — backbone only (no graph/symbolic for speed during sweep)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            # ── Live % progress ──────────────────────────────────────────
            pct = ((batch_idx + 1) / len(loader)) * 100
            print(f"\r  [{model_name:20s}] Ep {epoch:3d}/{n_epochs} "
                  f"{split} [{pct:5.1f}%]  "
                  f"Loss: {total_loss/(batch_idx+1):.4f}  "
                  f"Acc: {correct/total*100:.2f}%",
                  end="", flush=True)

    print()   # newline after epoch
    return total_loss / len(loader), correct / total


# ─── Train a single backbone ──────────────────────────────────────────────

def train_one_model(model_name: str,
                    train_dl: DataLoader,
                    val_dl:   DataLoader,
                    device:   torch.device,
                    class_weights: torch.Tensor) -> dict:
    """
    Train one backbone for NUM_EPOCHS.
    Returns dict with history and best metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Training : {model_name}")
    print(f"{'='*60}")

    model = get_model(model_name, pretrained=True).to(device)

    criterion = FocalCrossEntropyLoss(
        gamma=2.0, smoothing=0.1,
        weight=class_weights if USE_CLASS_WEIGHTS else None)

    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                  factor=0.5, patience=LR_PATIENCE)

    best_val_acc  = 0.0
    best_path     = MODELS_DIR / f"{model_name}_best.pth"
    patience_ctr  = 0

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    # CSV log
    log_path = LOGS_DIR / f"{model_name}_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss",
                         "train_acc", "val_acc"])

    for epoch in range(1, NUM_EPOCHS + 1):
        t_loss, t_acc = _run_epoch(model, train_dl, criterion, optimizer,
                                   device, True, epoch, NUM_EPOCHS, model_name)
        v_loss, v_acc = _run_epoch(model, val_dl,   criterion, None,
                                   device, False, epoch, NUM_EPOCHS, model_name)

        history["train_loss"].append(t_loss)
        history["val_loss"  ].append(v_loss)
        history["train_acc" ].append(t_acc)
        history["val_acc"   ].append(v_acc)

        scheduler.step(v_acc)

        overall_pct = epoch / NUM_EPOCHS * 100
        print(f"  [Total {overall_pct:5.1f}%]  "
              f"T_Acc: {t_acc*100:.2f}%  V_Acc: {v_acc*100:.2f}%  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, t_loss, v_loss, t_acc, v_acc])

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), str(best_path))
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= EARLY_STOP_PAT:
            print(f"  Early stopping at epoch {epoch}.")
            break

    print(f"\n✔  {model_name} done.  Best val acc: {best_val_acc*100:.2f}%")
    return {"history": history, "best_val_acc": best_val_acc,
            "model_name": model_name, "checkpoint": str(best_path)}


# ─── Train all models ─────────────────────────────────────────────────────

def train_all_models(model_names: list = None,
                     verbose: bool = True) -> list:
    """
    Sequentially train all (or selected) backbones.
    Returns list of result dicts.
    """
    if model_names is None:
        model_names = MODEL_NAMES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    train_ds  = CancerClassificationDataset("train")
    val_ds    = CancerClassificationDataset("val")
    train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_dl    = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}")

    class_weights = compute_class_weights(train_ds, device)

    all_results = []
    for i, name in enumerate(model_names):
        model_pct = (i + 1) / len(model_names) * 100
        print(f"\n  ── Model {i+1}/{len(model_names)}  ({model_pct:.0f}% of sweep) ──")
        try:
            result = train_one_model(name, train_dl, val_dl,
                                     device, class_weights)
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {name} failed: {e}")
            all_results.append({"model_name": name, "error": str(e)})

    return all_results


# ─── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  STAGE 11 — MODEL TRAINING")
    print("=" * 60)
    results = train_all_models()
    print("\n  ── Final Summary ──────────────────────────────────")
    for r in results:
        if "error" in r:
            print(f"  {r['model_name']:25s} : FAILED  — {r['error']}")
        else:
            print(f"  {r['model_name']:25s} : {r['best_val_acc']*100:.2f}%")
