"""
evaluation/ablation_augmentation.py
─────────────────────────────────────────────
Ablation: Augmentation Strategy Comparison
  1. No augmentation       (original counts only)
  2. Geometric only        (flip, rotate, crop, zoom)
  3. Photometric only      (HSV, CLAHE, blur, noise, B/C)
  4. Geometric + Elastic   (adds elastic deformation)
  5. Full pipeline         (all 10 transforms) ← default

Compares test accuracy, F1, and training stability
on DenseNet201 (with CCA, Focal CE + label smooth).
─────────────────────────────────────────────
Run:
    python evaluation/ablation_augmentation.py
"""

import sys
import csv
import random
import shutil
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    PROCESSED_DIR, SPLIT_DIR, CLASSES, CLASS_TO_IDX,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOP_PAT, LR_PATIENCE, NUM_WORKERS,
    MODELS_DIR, LOGS_DIR, RESULTS_DIR,
    IMAGE_SIZE, SEED, TARGET_COUNT,
)

NUM_EPOCHS = 15  # Override for graph ablation


from classifier.neurosymbolic_graph_transformer import FocalCrossEntropyLoss
from augmentation.augment import (
    flip, rotate, random_crop_resize, zoom,
    hsv_jitter, clahe_enhance, elastic_transform,
    gaussian_noise, brightness_contrast,
)
import torchvision.transforms as T

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
_tfm = T.Compose([T.ToTensor(),
                   T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

# ── Augmentation strategies ───────────────────────────────────────────────

def augment_none(img, mask):
    """Return image unchanged."""
    return img.copy(), mask.copy()


def augment_geometric(img, mask):
    """Flip + rotate + crop + zoom only."""
    ops = random.sample(["flip_h", "flip_v",
                          "rotate", "crop", "zoom"],
                         k=random.randint(2, 3))
    if "flip_h"  in ops: img, mask = flip(img, mask, 1)
    if "flip_v"  in ops: img, mask = flip(img, mask, 0)
    if "rotate"  in ops: img, mask = rotate(img, mask)
    if "crop"    in ops: img, mask = random_crop_resize(img, mask)
    if "zoom"    in ops: img, mask = zoom(img, mask)
    return img, mask


def augment_photometric(img, mask):
    """HSV + CLAHE + blur + noise + brightness only."""
    img = hsv_jitter(img)
    img = brightness_contrast(img)
    if random.random() < 0.5:
        img = clahe_enhance(img)
    if random.random() < 0.4:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.3:
        img = gaussian_noise(img)
    return img, mask


def augment_geo_elastic(img, mask):
    """Geometric + elastic deformation."""
    img, mask = augment_geometric(img, mask)
    img, mask = elastic_transform(img, mask)
    return img, mask


def augment_full(img, mask):
    """All 10 transforms — pipeline default."""
    ops = random.sample(
        ["flip_h", "flip_v", "rotate", "crop", "zoom", "elastic"],
        k=random.randint(2, 4))
    if "flip_h"  in ops: img, mask = flip(img, mask, 1)
    if "flip_v"  in ops: img, mask = flip(img, mask, 0)
    if "rotate"  in ops: img, mask = rotate(img, mask)
    if "crop"    in ops: img, mask = random_crop_resize(img, mask)
    if "zoom"    in ops: img, mask = zoom(img, mask)
    if "elastic" in ops: img, mask = elastic_transform(img, mask)
    img = hsv_jitter(img)
    img = brightness_contrast(img)
    if random.random() < 0.5: img = clahe_enhance(img)
    if random.random() < 0.4:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.3: img = gaussian_noise(img)
    return img, mask


AUG_STRATEGIES = [
    ("No_Augmentation",    augment_none,       76),
    ("Geometric_Only",     augment_geometric,  TARGET_COUNT),
    ("Photometric_Only",   augment_photometric, TARGET_COUNT),
    ("Geo_Plus_Elastic",   augment_geo_elastic, TARGET_COUNT),
    ("Full_Pipeline",      augment_full,        TARGET_COUNT),
]


# ── In-memory dataset ─────────────────────────────────────────────────────

class AblationDataset(Dataset):
    def __init__(self, split: str, aug_fn, aug_count: int):
        self.samples   = []
        self.aug_fn    = aug_fn
        self.aug_count = aug_count
        self.is_train  = (split == "train")

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

        # For train: generate augmented copies up to aug_count
        if self.is_train:
            self._build_augmented()

    def _build_augmented(self):
        """Pre-generate augmented images into memory list."""
        self._aug_cache = []
        per_class_target = self.aug_count

        class_samples: dict = {}
        for p, lp, idx in self.samples:
            class_samples.setdefault(idx, []).append((p, lp))

        for idx, items in class_samples.items():
            count = 0
            while count < per_class_target:
                p, lp = random.choice(items)
                img  = cv2.imread(str(p))
                mask = cv2.imread(str(lp),
                                  cv2.IMREAD_GRAYSCALE) \
                       if lp and lp.exists() \
                       else np.zeros(IMAGE_SIZE, np.uint8)
                if img is None:
                    continue
                aug_img, _ = self.aug_fn(img, mask)
                aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                self._aug_cache.append((aug_img, idx))
                count += 1

    def __len__(self):
        if self.is_train:
            return len(self._aug_cache)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_train:
            img_rgb, label = self._aug_cache[idx]
        else:
            p, _, label = self.samples[idx]
            img = cv2.imread(str(p))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor = _tfm(img_rgb.astype(np.uint8))
        return tensor, torch.tensor(label, dtype=torch.long)


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

def train_and_eval_aug(strategy_label: str,
                       aug_fn, aug_count: int,
                       device) -> dict:
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from sklearn.metrics import (accuracy_score, f1_score,
                                  precision_score, recall_score)

    print(f"\n  ── Strategy: {strategy_label} "
          f"(aug_count={aug_count}) ──")

    # Import here to avoid circular import
    from utils.datasets import CancerClassificationDataset

    train_ds = AblationDataset("train", aug_fn, aug_count)
    val_ds   = CancerClassificationDataset("val")
    test_ds  = CancerClassificationDataset("test")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)

    model     = build_model().to(device)
    criterion = FocalCrossEntropyLoss(gamma=2.0, smoothing=0.1)
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=LR_PATIENCE)

    best_val_acc = 0.0
    patience_ctr = 0
    ckpt = MODELS_DIR / f"ablation_aug_{strategy_label}.pth"
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for imgs, labels in tqdm(
                train_dl,
                desc=f"  [{strategy_label}] Ep {epoch:3d}",
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
        scheduler.step(val_acc)
        pct = epoch / NUM_EPOCHS * 100
        print(f"  [{strategy_label}] [{pct:5.1f}%]"
              f"  Ep {epoch:3d}  Val Acc: {val_acc*100:.2f}%")

        history.append({"epoch": epoch, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(ckpt))
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= EARLY_STOP_PAT:
            print(f"  [{strategy_label}] Early stop ep {epoch}")
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
        "label":     strategy_label,
        "aug_count": aug_count,
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

def run_augmentation_ablation():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print("=" * 60)
    print("  ABLATION — Augmentation Strategy")
    print("=" * 60)

    results = []
    for label, aug_fn, aug_count in AUG_STRATEGIES:
        r = train_and_eval_aug(label, aug_fn, aug_count, device)
        results.append(r)

    # ── Summary ────────────────────────────────────────────────────
    print("\n  ── Augmentation Ablation Results ────────────────")
    print(f"  {'Strategy':26s} {'Count':>6s} "
          f"{'Acc':>8s} {'F1':>8s}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['label']:26s} "
              f"{r['aug_count']:6d}  "
              f"{r['test_acc']*100:7.2f}%  "
              f"{r['test_f1']:8.4f}")

    # ── CSV ────────────────────────────────────────────────────────
    csv_path = LOGS_DIR / "ablation_augmentation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "aug_count", "test_acc",
            "test_f1", "test_prec", "test_rec"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\n  CSV saved → {csv_path}")

    _plot_aug_ablation(results)
    return results


def _plot_aug_ablation(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Augmentation Strategy Ablation — DenseNet201",
                 fontsize=13)

    labels = [r["label"].replace("_", "\n") for r in results]
    accs   = [r["test_acc"] * 100 for r in results]
    f1s    = [r["test_f1"]        for r in results]
    colors = ["#9E9E9E", "#FF9800", "#4CAF50",
              "#2196F3", "#9C27B0"]

    ax = axes[0]
    bars = ax.bar(labels, accs, color=colors, alpha=0.85,
                  edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8)
    ax.set_ylim(70, 102)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy by Augmentation Strategy")

    ax2 = axes[1]
    for i, r in enumerate(results):
        ep  = [h["epoch"]   for h in r["history"]]
        acc = [h["val_acc"] for h in r["history"]]
        ax2.plot(ep, acc,
                 label=r["label"].replace("_", " "),
                 color=colors[i], linewidth=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Convergence by Strategy")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    out = RESULTS_DIR / "ablation_augmentation.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Plot saved → {out}")


if __name__ == "__main__":
    run_augmentation_ablation()
