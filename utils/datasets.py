"""
utils/datasets.py
─────────────────────────────────────────────────────────────────────────────
PyTorch Dataset wrappers for:
  • CancerClassificationDataset — (image, class_idx) pairs
  • CancerSegmentationDataset  — (image, binary_mask) pairs
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import numpy as np
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as T

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    SPLIT_DIR, AUGMENTED_DIR, CLASSES,
    CLASS_TO_IDX, IMAGE_SIZE,
)

# ─── Normalisation statistics (ImageNet — works well for H&E after Reinhard) ──
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str):
    """Return torchvision transform for train / val / test."""
    if split == "train":
        return T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.15, contrast=0.15,
                          saturation=0.15, hue=0.05),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ─── Classification Dataset ───────────────────────────────────────────────

class CancerClassificationDataset(Dataset):
    """
    Loads images from SPLIT_DIR/<split>/<class>/ for classification.
    Returns  (tensor_CHW, class_idx).
    """

    def __init__(self, split: str = "train", transform=None):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split     = split
        self.transform = transform or get_transforms(split)
        self.samples   = []   # list of (img_path, label_idx)

        for class_label in CLASSES:
            img_dir = SPLIT_DIR / split / class_label / "Image"
            if not img_dir.exists():
                continue
            for p in sorted(img_dir.glob("*")):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg",
                                        ".tif", ".tiff", ".bmp"}:
                    self.samples.append((p, CLASS_TO_IDX[class_label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = T.ToTensor()(img)

        return img_tensor, torch.tensor(label, dtype=torch.long)

    def class_counts(self):
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        return dict(counts)


# ─── Segmentation Dataset ─────────────────────────────────────────────────

class CancerSegmentationDataset(Dataset):
    """
    Loads (image, binary_mask) pairs from SPLIT_DIR.
    Returns  (tensor_CHW, mask_HW)  where mask values are 0/1.
    """

    def __init__(self, split: str = "train",
                 img_transform=None, mask_transform=None):
        self.split          = split
        self.img_transform  = img_transform or get_transforms(split)
        self.samples        = []   # (img_path, mask_path)

        for class_label in CLASSES:
            img_dir = SPLIT_DIR / split / class_label / "Image"
            lbl_dir = SPLIT_DIR / split / class_label / "Label"
            if not img_dir.exists():
                continue
            for p in sorted(img_dir.glob("*")):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg",
                                        ".tif", ".tiff", ".bmp"}:
                    lbl_p = lbl_dir / p.name
                    self.samples.append((p, lbl_p if lbl_p.exists() else None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        img_tensor = self.img_transform(img)

        if mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros(IMAGE_SIZE, dtype=np.float32)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        return img_tensor, mask_tensor
