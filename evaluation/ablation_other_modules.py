"""
evaluation/ablation_other_modules.py
─────────────────────────────────────────────────────────────────────────────
Ablation: "all other possible modules or processes" not already covered
by ablation_cca.py / ablation_graph.py / ablation_loss.py /
ablation_augmentation.py / ablation_attention_variants.py /
ablation_incremental.py / ablation_neurosymbolic.py / ablation_unet_variants.py.

Covers, each as an independent sweep on the CCA-DenseNet201 backbone
(same fixed budget as the other ablations, for comparability):

  1. Test-Time Augmentation (TTA)   — 1-view vs 4-view vs 8-view, its
     accuracy/Macro-F1 gain AND its latency cost (directly extends the
     ATR's existing 31 ms/0.98 s figures into a full TTA-view sweep).
  2. Class-weighted vs. unweighted loss — quantifies what the inverse-
     frequency class weighting (used throughout the rest of the
     pipeline to address the imbalanced EBHI-SEG class distribution)
     is actually worth.
  3. Optimiser choice — Adam vs AdamW vs SGD+momentum, same LR schedule.
  4. Backbone architecture — a lighter sweep across representative
     entries from the paper's existing Table-1 comparative list
     (ResNet50, EfficientNetB4, ConvNeXt-Tiny, DenseNet201) with CCA
     attached at each backbone's stem, so the backbone choice itself is
     ablated under the *same* CCA+Focal-CE+full-augmentation recipe as
     the rest of the study.
  5. Dropout-rate sweep in the fusion head (0.0 / 0.3 / 0.5).

Run:
    python evaluation/ablation_other_modules.py
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

NUM_EPOCHS = 15

from utils.datasets import CancerClassificationDataset
from classifier.neurosymbolic_graph_transformer import FocalCrossEntropyLoss
from models.attention import ColourChannelAttention

from utils.gpu_memory import configure_cudnn_safe_mode, free_gpu_memory

torch.manual_seed(SEED)
configure_cudnn_safe_mode()
NUM_CLASSES = len(CLASSES)


# ── Shared backbone builder ────────────────────────────────────────────

def build_cca_backbone(arch: str = "DenseNet201", dropout: float = 0.3):
    import torchvision.models as tv

    if arch == "DenseNet201":
        model = tv.densenet201(weights="IMAGENET1K_V1")
        stem = model.features.conv0
        model.features.conv0 = nn.Sequential(
            stem, ColourChannelAttention(stem.out_channels))
        in_feat = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_feat, NUM_CLASSES))
    elif arch == "ResNet50":
        model = tv.resnet50(weights="IMAGENET1K_V2")
        stem = model.conv1
        model.conv1 = nn.Sequential(
            stem, ColourChannelAttention(stem.out_channels))
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_feat, NUM_CLASSES))
    elif arch == "EfficientNetB4":
        model = tv.efficientnet_b4(weights="IMAGENET1K_V1")
        stem_conv = model.features[0][0]
        model.features[0] = nn.Sequential(
            model.features[0], ColourChannelAttention(stem_conv.out_channels))
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_feat, NUM_CLASSES))
    elif arch == "ConvNeXtTiny":
        model = tv.convnext_tiny(weights="IMAGENET1K_V1")
        stem_conv = model.features[0][0]
        model.features[0] = nn.Sequential(
            model.features[0], ColourChannelAttention(stem_conv.out_channels))
        in_feat = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_feat, NUM_CLASSES))
    else:
        raise ValueError(f"Unknown arch {arch}")

    return model


# ── Generic train/eval routine (shared across all 5 sub-ablations) ────────

def train_and_eval(model, label, train_dl, val_dl, test_dl, device,
                   criterion, optimizer_name="Adam", tta_views=1):
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from sklearn.metrics import accuracy_score, f1_score

    model.to(device)
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                               weight_decay=WEIGHT_DECAY)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE * 10,
                              momentum=0.9, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(optimizer_name)

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=LR_PATIENCE)

    best_val_acc, patience_ctr = 0.0, 0
    ckpt = MODELS_DIR / f"ablation_other_{label}.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for imgs, labels in tqdm(train_dl, desc=f"  [{label}] Ep {epoch}",
                                 leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs = imgs.to(device)
                preds.extend(model(imgs).argmax(1).cpu().tolist())
                trues.extend(labels.tolist())
        val_acc = accuracy_score(trues, preds)
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(ckpt))
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= EARLY_STOP_PAT:
            break

    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.eval()

    def _predict_tta(imgs):
        """tta_views in {1,4,8}: horizontal/vertical flip + 90-rotations."""
        views = [imgs]
        if tta_views >= 4:
            views += [torch.flip(imgs, dims=[3]),
                     torch.flip(imgs, dims=[2]),
                     torch.rot90(imgs, 1, dims=[2, 3])]
        if tta_views >= 8:
            views += [torch.rot90(imgs, 2, dims=[2, 3]),
                     torch.rot90(imgs, 3, dims=[2, 3]),
                     torch.flip(torch.rot90(imgs, 1, dims=[2, 3]), dims=[3]),
                     torch.flip(torch.rot90(imgs, 1, dims=[2, 3]), dims=[2])]
        probs = torch.zeros(imgs.size(0), NUM_CLASSES, device=imgs.device)
        for v in views[:tta_views]:
            probs += torch.softmax(model(v), dim=1)
        return probs / len(views[:tta_views])

    test_preds, test_true = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs = imgs.to(device)
            probs = _predict_tta(imgs)
            test_preds.extend(probs.argmax(1).cpu().tolist())
            test_true.extend(labels.tolist())
    total_time = time.perf_counter() - t0

    test_acc = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    per_image_ms = total_time / max(len(test_true), 1) * 1000

    result = {"label": label, "test_acc": test_acc, "test_f1": test_f1,
            "per_image_ms": per_image_ms,
            "n_params": sum(p.numel() for p in model.parameters())}

    free_gpu_memory(model, optimizer)
    return result


# ── Sub-ablation 1: TTA views ───────────────────────────────────────────

def ablate_tta(train_dl, val_dl, test_dl, device, weights):
    results = []
    for views in [1, 4, 8]:
        model = build_cca_backbone("DenseNet201")
        criterion = FocalCrossEntropyLoss(gamma=2.0, smoothing=0.1,
                                          weight=weights)
        r = train_and_eval(model, f"TTA_{views}view", train_dl, val_dl,
                           test_dl, device, criterion, tta_views=views)
        results.append(r)
    return results


# ── Sub-ablation 2: class weighting ─────────────────────────────────────

def ablate_class_weighting(train_dl, val_dl, test_dl, device, weights):
    results = []
    for use_weights, label in [(True, "Weighted_CE"), (False, "Unweighted_CE")]:
        model = build_cca_backbone("DenseNet201")
        criterion = FocalCrossEntropyLoss(
            gamma=2.0, smoothing=0.1,
            weight=weights if use_weights else None)
        r = train_and_eval(model, label, train_dl, val_dl, test_dl,
                           device, criterion)
        results.append(r)
    return results


# ── Sub-ablation 3: optimiser choice ────────────────────────────────────

def ablate_optimizer(train_dl, val_dl, test_dl, device, weights):
    results = []
    for opt_name in ["Adam", "AdamW", "SGD"]:
        model = build_cca_backbone("DenseNet201")
        criterion = FocalCrossEntropyLoss(gamma=2.0, smoothing=0.1,
                                          weight=weights)
        r = train_and_eval(model, f"Optim_{opt_name}", train_dl, val_dl,
                           test_dl, device, criterion,
                           optimizer_name=opt_name)
        results.append(r)
    return results


# ── Sub-ablation 4: backbone architecture ───────────────────────────────

def ablate_backbone(train_dl, val_dl, test_dl, device, weights):
    results = []
    for arch in ["ResNet50", "EfficientNetB4", "ConvNeXtTiny", "DenseNet201"]:
        model = build_cca_backbone(arch)
        criterion = FocalCrossEntropyLoss(gamma=2.0, smoothing=0.1,
                                          weight=weights)
        r = train_and_eval(model, f"Backbone_{arch}", train_dl, val_dl,
                           test_dl, device, criterion)
        results.append(r)
    return results


# ── Sub-ablation 5: dropout rate ────────────────────────────────────────

def ablate_dropout(train_dl, val_dl, test_dl, device, weights):
    results = []
    for p in [0.0, 0.3, 0.5]:
        model = build_cca_backbone("DenseNet201", dropout=p)
        criterion = FocalCrossEntropyLoss(gamma=2.0, smoothing=0.1,
                                          weight=weights)
        r = train_and_eval(model, f"Dropout_{p}", train_dl, val_dl,
                           test_dl, device, criterion)
        results.append(r)
    return results


# ── Main ──────────────────────────────────────────────────────────────────

def run_other_modules_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print("=" * 70)
    print("  OTHER-MODULES ABLATION — TTA / class-weighting / optimiser /")
    print("  backbone / dropout")
    print("=" * 70)

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

    all_results = {}
    print("\n  [1/5] Test-Time Augmentation sweep ...")
    all_results["tta"] = ablate_tta(train_dl, val_dl, test_dl, device, weights)
    print("\n  [2/5] Class-weighting sweep ...")
    all_results["class_weighting"] = ablate_class_weighting(
        train_dl, val_dl, test_dl, device, weights)
    print("\n  [3/5] Optimiser sweep ...")
    all_results["optimizer"] = ablate_optimizer(
        train_dl, val_dl, test_dl, device, weights)
    print("\n  [4/5] Backbone-architecture sweep ...")
    all_results["backbone"] = ablate_backbone(
        train_dl, val_dl, test_dl, device, weights)
    print("\n  [5/5] Dropout-rate sweep ...")
    all_results["dropout"] = ablate_dropout(
        train_dl, val_dl, test_dl, device, weights)

    csv_path = LOGS_DIR / "ablation_other_modules.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sweep", "label", "test_acc", "test_f1", "per_image_ms",
            "n_params"])
        w.writeheader()
        for sweep, rows in all_results.items():
            for r in rows:
                row = {"sweep": sweep, **r}
                w.writerow({k: row[k] for k in w.fieldnames})
    print(f"\n  CSV saved -> {csv_path}")

    print("\n  ── Other-Modules Ablation Results ──────────────────────")
    for sweep, rows in all_results.items():
        print(f"\n  [{sweep}]")
        for r in rows:
            print(f"    {r['label']:22s} Acc: {r['test_acc']*100:6.2f}%  "
                  f"F1: {r['test_f1']:.4f}  {r['per_image_ms']:.2f} ms/img")

    _plot_other_modules(all_results)
    return all_results


def _plot_other_modules(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle("Other-Modules Ablation Suite", fontsize=14)
    axes_flat = axes.flatten()

    titles = {
        "tta": "TTA View-Count Sweep",
        "class_weighting": "Class-Weighted vs Unweighted Loss",
        "optimizer": "Optimiser Choice",
        "backbone": "Backbone Architecture",
        "dropout": "Fusion-Head Dropout Rate",
    }

    for ax, (sweep, rows) in zip(axes_flat, all_results.items()):
        labels = [r["label"] for r in rows]
        accs = [r["test_acc"] * 100 for r in rows]
        f1s = [r["test_f1"] * 100 for r in rows]
        x = range(len(labels))
        ax.bar([xi - 0.2 for xi in x], accs, width=0.4, label="Accuracy",
               color="#2196F3")
        ax.bar([xi + 0.2 for xi in x], f1s, width=0.4, label="Macro-F1",
               color="#F44336")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(titles.get(sweep, sweep))
        ax.legend(fontsize=7)

    for i in range(len(all_results), len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout()
    out = RESULTS_DIR / "ablation_other_modules.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Plot saved -> {out}")


if __name__ == "__main__":
    run_other_modules_ablation()
