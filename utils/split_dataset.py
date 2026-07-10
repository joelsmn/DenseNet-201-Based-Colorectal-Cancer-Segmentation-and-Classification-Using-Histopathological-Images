"""
utils/split_dataset.py
─────────────────────────────────────────────────────────────────────────────
Splits the augmented dataset into train / val / test sets.

Ratio  : 70 % train | 15 % val | 15 % test  (per class, stratified)
Output : SPLIT_DIR/<split>/<class_label>/Image/  &  /Label/
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import shutil
import random
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    AUGMENTED_DIR, SPLIT_DIR, CLASSES,
    SPLIT_RATIOS, SEED,
)

random.seed(SEED)


def split_dataset(verbose: bool = True) -> None:
    print("=" * 60)
    print("  SPLITTING DATASET  (70 / 15 / 15)")
    print("=" * 60)

    for class_label in CLASSES:
        src_img = AUGMENTED_DIR / class_label / "Image"
        src_lbl = AUGMENTED_DIR / class_label / "Label"

        all_imgs = sorted(src_img.glob("*"))
        all_imgs = [f for f in all_imgs
                    if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}]

        random.shuffle(all_imgs)
        n = len(all_imgs)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val   = int(n * SPLIT_RATIOS["val"])

        splits = {
            "train": all_imgs[:n_train],
            "val":   all_imgs[n_train:n_train + n_val],
            "test":  all_imgs[n_train + n_val:],
        }

        for split_name, files in splits.items():
            out_img = SPLIT_DIR / split_name / class_label / "Image"
            out_lbl = SPLIT_DIR / split_name / class_label / "Label"
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

            for f in tqdm(files,
                          desc=f"  {split_name:5s} [{class_label:20s}]",
                          leave=False, disable=not verbose):
                shutil.copy2(str(f), str(out_img / f.name))
                lbl = src_lbl / f.name
                if lbl.exists():
                    shutil.copy2(str(lbl), str(out_lbl / f.name))

        if verbose:
            print(f"  {class_label:20s}  "
                  f"train={len(splits['train'])}  "
                  f"val={len(splits['val'])}  "
                  f"test={len(splits['test'])}")

    print(f"\n✔  Split complete → {SPLIT_DIR}")


if __name__ == "__main__":
    split_dataset(verbose=True)
