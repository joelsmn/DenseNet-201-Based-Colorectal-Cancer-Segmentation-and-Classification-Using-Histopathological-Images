"""
preprocessing/preprocess.py
─────────────────────────────────────────────────────────────────────────────
Stage 1 — PREPROCESSING

Steps performed per image:
  1. Load image (RGB)
  2. Remove black/white border artefacts via adaptive cropping
  3. Reinhard colour normalisation (LAB space)  ← key for H&E slides
  4. Resize to IMAGE_SIZE (224×224) with LANCZOS resampling
  5. Gaussian denoising pass
  6. Save to PROCESSED_DIR preserving folder structure
  7. Copy paired segmentation mask (resize nearest-neighbour, no colour shift)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import shutil
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Allow running as a standalone script
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    DATASET_DIR, PROCESSED_DIR, FOLDER_TO_CLASS,
    IMAGE_SIZE, CLASSES,
)

# ─── Reinhard colour normalisation ────────────────────────────────────────

def _get_lab_stats(img_bgr: np.ndarray):
    """Return (mean_L, mean_a, mean_b, std_L, std_a, std_b) of an BGR image."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    return (l.mean(), a.mean(), b.mean(),
            l.std(),  a.std(),  b.std())


# Reference statistics derived from a high-quality H&E reference image.
# These values approximate a well-stained colorectal histology slide.
REF_STATS = {
    "mean": (148.60, 169.30, 105.97),   # L, a, b
    "std":  ( 41.56,   9.01,   6.67),
}


def reinhard_normalise(img_bgr: np.ndarray,
                       ref_mean=REF_STATS["mean"],
                       ref_std=REF_STATS["std"]) -> np.ndarray:
    """
    Reinhard et al. (2001) colour transfer in LAB space.
    Maps the input image's colour statistics to those of the reference.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    def _normalise_channel(ch, rm, rs, tm_mean, tm_std):
        ch -= rm
        ch *= (tm_std / (rs + 1e-6))
        ch += tm_mean
        return ch

    src_mean = (l.mean(), a.mean(), b.mean())
    src_std  = (l.std(),  a.std(),  b.std())

    l = _normalise_channel(l, src_mean[0], src_std[0], ref_mean[0], ref_std[0])
    a = _normalise_channel(a, src_mean[1], src_std[1], ref_mean[1], ref_std[1])
    b = _normalise_channel(b, src_mean[2], src_std[2], ref_mean[2], ref_std[2])

    lab_norm = cv2.merge([l, a, b])
    lab_norm = np.clip(lab_norm, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)


# ─── Artefact / border cropping ───────────────────────────────────────────

def crop_artefacts(img_bgr: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Remove near-black or near-white border regions that result from
    slide scanning artefacts.  Uses a simple intensity-based bounding box.
    Falls back to the original image if the crop would remove >40% of pixels.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold) & (gray < 245)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img_bgr
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    h, w = img_bgr.shape[:2]
    if (y1 - y0) < 0.6 * h or (x1 - x0) < 0.6 * w:
        return img_bgr            # crop too aggressive — skip
    return img_bgr[y0:y1, x0:x1]


# ─── Gaussian denoising ───────────────────────────────────────────────────

def denoise(img_bgr: np.ndarray) -> np.ndarray:
    """Light Gaussian blur to reduce scanner noise without blurring structure."""
    return cv2.GaussianBlur(img_bgr, (3, 3), sigmaX=0.5)


# ─── Full preprocessing pipeline ──────────────────────────────────────────

def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """Apply full preprocessing pipeline to a single BGR image."""
    img = crop_artefacts(img_bgr)
    img = reinhard_normalise(img)
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
    img = denoise(img)
    return img


def preprocess_mask(mask_bgr: np.ndarray) -> np.ndarray:
    """Resize mask only — no colour transform."""
    if len(mask_bgr.shape) == 3:
        mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_bgr
    mask_resized = cv2.resize(mask_gray, IMAGE_SIZE,
                              interpolation=cv2.INTER_NEAREST)
    return mask_resized


# ─── Dataset-level processing ─────────────────────────────────────────────

def preprocess_dataset(verbose: bool = True) -> None:
    """
    Walk every class folder in DATASET_DIR, preprocess images + masks,
    save to PROCESSED_DIR/<class_label>/Image/ and /Label/.
    Shows a progress bar per class.
    """
    total_processed = 0
    total_skipped   = 0

    for folder_name, class_label in FOLDER_TO_CLASS.items():
        src_img_dir  = DATASET_DIR / folder_name / "Image"
        src_lbl_dir  = DATASET_DIR / folder_name / "Label"

        out_img_dir  = PROCESSED_DIR / class_label / "Image"
        out_lbl_dir  = PROCESSED_DIR / class_label / "Label"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            print(f"  [WARN] Missing: {src_img_dir}  — skipping.")
            continue

        img_files = sorted(src_img_dir.glob("*"))
        img_files = [f for f in img_files
                     if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}]

        desc = f"  Preprocessing [{class_label:20s}]"
        for img_path in tqdm(img_files, desc=desc, leave=True, disable=not verbose):
            out_img_path = out_img_dir / img_path.name
            out_lbl_path = out_lbl_dir / img_path.name

            # ── Image ──────────────────────────────────────────────────
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                total_skipped += 1
                continue
            processed = preprocess_image(img_bgr)
            cv2.imwrite(str(out_img_path), processed)

            # ── Paired mask ────────────────────────────────────────────
            lbl_path = src_lbl_dir / img_path.name
            if lbl_path.exists():
                mask = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask_resized = preprocess_mask(mask)
                    cv2.imwrite(str(out_lbl_path), mask_resized)
            else:
                # No mask available — write blank mask
                blank = np.zeros(IMAGE_SIZE, dtype=np.uint8)
                cv2.imwrite(str(out_lbl_path), blank)

            total_processed += 1

    print(f"\n✔  Preprocessing complete.  "
          f"Processed: {total_processed}   Skipped: {total_skipped}")
    print(f"   Output → {PROCESSED_DIR}")


# ─── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  STAGE 1 — PREPROCESSING")
    print("=" * 60)
    preprocess_dataset(verbose=True)
