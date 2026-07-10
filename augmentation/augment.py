"""
augmentation/augment.py
─────────────────────────────────────────────────────────────────────────────
Stage 2 — AUGMENTATION / ENHANCEMENT

What this module does
─────────────────────
For every class that has < TARGET_COUNT images after preprocessing, we
synthetically generate extra images until each class reaches exactly
TARGET_COUNT images.

Augmentation transforms applied (randomly combined):
  1.  Random horizontal / vertical flip
  2.  Random rotation  (±180°, random fill)
  3.  Random crop + resize  (scale 0.75–1.0, aspect ratio 0.8–1.2)
  4.  Random zoom  (0.8–1.2×)
  5.  Hue / Saturation / Value jitter   (H ±18, S ±40, V ±30)
  6.  Random Gaussian blur               (kernel 3 or 5)
  7.  Random brightness / contrast shift
  8.  CLAHE contrast enhancement         (applied ≈50 % of the time)
  9.  Elastic deformation                (for structural realism)
  10. Gaussian additive noise

Each augmented image has a corresponding augmented binary mask generated
with the same geometric transform (no colour transform on mask).

Classes with more images than TARGET_COUNT are randomly down-sampled so
every class ends up with exactly TARGET_COUNT images.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import random
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import (
    PROCESSED_DIR, AUGMENTED_DIR,
    CLASSES, TARGET_COUNT, IMAGE_SIZE, SEED,
)

random.seed(SEED)
np.random.seed(SEED)

# ─── Individual transform helpers ─────────────────────────────────────────

def flip(img, mask, code):
    return cv2.flip(img, code), cv2.flip(mask, code)


def rotate(img, mask, angle=None):
    if angle is None:
        angle = random.uniform(-180, 180)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_r  = cv2.warpAffine(img,  M, (w, h),
                             flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REFLECT_101)
    mask_r = cv2.warpAffine(mask, M, (w, h),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_REFLECT_101)
    return img_r, mask_r


def random_crop_resize(img, mask):
    h, w = img.shape[:2]
    scale = random.uniform(0.75, 1.0)
    ar    = random.uniform(0.8, 1.2)
    nw    = int(w * scale * ar)
    nh    = int(h * scale / ar)
    nw, nh = max(32, min(nw, w)), max(32, min(nh, h))
    x = random.randint(0, w - nw)
    y = random.randint(0, h - nh)
    img_c  = img[y:y+nh, x:x+nw]
    mask_c = mask[y:y+nh, x:x+nw]
    img_c  = cv2.resize(img_c,  IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
    mask_c = cv2.resize(mask_c, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    return img_c, mask_c


def zoom(img, mask):
    factor = random.uniform(0.8, 1.2)
    h, w = img.shape[:2]
    nw, nh = int(w * factor), int(h * factor)
    img_z  = cv2.resize(img,  (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    mask_z = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    # Crop or pad back to IMAGE_SIZE
    def _fit(arr, target_h, target_w, is_mask=False):
        ch, cw = arr.shape[:2]
        if ch >= target_h and cw >= target_w:
            y = (ch - target_h) // 2
            x = (cw - target_w) // 2
            return arr[y:y+target_h, x:x+target_w]
        pad_h = max(0, target_h - ch)
        pad_w = max(0, target_w - cw)
        if is_mask:
            return np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant')
        return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    img_z  = _fit(img_z,  IMAGE_SIZE[1], IMAGE_SIZE[0], False)
    mask_z = _fit(mask_z, IMAGE_SIZE[1], IMAGE_SIZE[0], True)
    return img_z, mask_z


def hsv_jitter(img):
    """Hue / Saturation / Value jitter for H&E colour variation."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = np.clip(hsv[..., 0] + random.randint(-18, 18), 0, 179)
    hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-40, 40), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-30, 30), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def clahe_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def elastic_transform(img, mask, alpha=34, sigma=4):
    """Elastic deformation — preserves texture topology of glands."""
    h, w = img.shape[:2]
    rng = np.random.RandomState(random.randint(0, 9999))
    dx = cv2.GaussianBlur((rng.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
    img_e  = cv2.remap(img,  map_x, map_y, cv2.INTER_LANCZOS4,
                       borderMode=cv2.BORDER_REFLECT_101)
    mask_e = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_REFLECT_101)
    return img_e, mask_e


def gaussian_noise(img):
    noise = np.random.normal(0, random.uniform(2, 10), img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def brightness_contrast(img):
    alpha = random.uniform(0.8, 1.2)    # contrast
    beta  = random.randint(-20, 20)     # brightness
    return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)


# ─── Compose one augmented pair ───────────────────────────────────────────

def augment_pair(img_bgr: np.ndarray,
                 mask_gray: np.ndarray):
    """
    Apply a random subset of transforms and return
    (augmented_image, augmented_mask).
    """
    img  = img_bgr.copy()
    mask = mask_gray.copy()

    ops = random.sample([
        "flip_h", "flip_v", "rotate",
        "crop", "zoom", "elastic",
    ], k=random.randint(2, 4))

    if "flip_h"  in ops: img, mask = flip(img, mask, 1)
    if "flip_v"  in ops: img, mask = flip(img, mask, 0)
    if "rotate"  in ops: img, mask = rotate(img, mask)
    if "crop"    in ops: img, mask = random_crop_resize(img, mask)
    if "zoom"    in ops: img, mask = zoom(img, mask)
    if "elastic" in ops: img, mask = elastic_transform(img, mask)

    # Colour / intensity ops (image only, never mask)
    img = hsv_jitter(img)
    img = brightness_contrast(img)
    if random.random() < 0.5:
        img = clahe_enhance(img)
    if random.random() < 0.4:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    if random.random() < 0.3:
        img = gaussian_noise(img)

    # Ensure correct sizes
    img  = cv2.resize(img,  IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
    mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    return img, mask


# ─── Class-level augmentation ─────────────────────────────────────────────

def augment_class(class_label: str, verbose: bool = True) -> int:
    """
    Augment one class to TARGET_COUNT.
    Returns number of images written.
    """
    src_img = PROCESSED_DIR / class_label / "Image"
    src_lbl = PROCESSED_DIR / class_label / "Label"
    out_img = AUGMENTED_DIR / class_label / "Image"
    out_lbl = AUGMENTED_DIR / class_label / "Label"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(src_img.glob("*"))
    all_imgs = [f for f in all_imgs
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}]

    n_orig = len(all_imgs)
    if n_orig == 0:
        print(f"  [WARN] No images found for {class_label}")
        return 0

    # ── Copy originals first ───────────────────────────────────────────────
    for img_path in all_imgs:
        img  = cv2.imread(str(img_path))
        lbl_path = src_lbl / img_path.name
        mask = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE) if lbl_path.exists() else \
               np.zeros(IMAGE_SIZE, dtype=np.uint8)
        cv2.imwrite(str(out_img / img_path.name), img)
        cv2.imwrite(str(out_lbl / img_path.name), mask)

    written = n_orig

    if n_orig >= TARGET_COUNT:
        # Down-sample to exactly TARGET_COUNT
        keep = random.sample(all_imgs, TARGET_COUNT)
        # Remove the files NOT in keep list
        for p in (out_img).glob("*"):
            if Path(p.name) not in {k.name for k in keep}:
                p.unlink(missing_ok=True)
                (out_lbl / p.name).unlink(missing_ok=True)
        written = TARGET_COUNT
        if verbose:
            print(f"  [{class_label:20s}] down-sampled  {n_orig} → {TARGET_COUNT}")
        return written

    needed = TARGET_COUNT - n_orig
    aug_counter = 0

    with tqdm(total=needed, desc=f"  Augmenting  [{class_label:20s}]",
              leave=True, disable=not verbose) as pbar:
        while aug_counter < needed:
            src_path = random.choice(all_imgs)
            img  = cv2.imread(str(src_path))
            lbl_path = src_lbl / src_path.name
            mask = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE) \
                   if lbl_path.exists() else np.zeros(IMAGE_SIZE, dtype=np.uint8)

            aug_img, aug_mask = augment_pair(img, mask)

            stem     = src_path.stem
            aug_name = f"{stem}_aug{aug_counter:04d}{src_path.suffix}"
            cv2.imwrite(str(out_img / aug_name), aug_img)
            cv2.imwrite(str(out_lbl / aug_name), aug_mask)
            aug_counter += 1
            written     += 1
            pbar.update(1)

    return written


# ─── Dataset-level augmentation ───────────────────────────────────────────

def augment_dataset(verbose: bool = True) -> None:
    """Augment all classes to TARGET_COUNT images each."""
    print("=" * 60)
    print("  STAGE 2 — AUGMENTATION / ENHANCEMENT")
    print("=" * 60)
    print(f"  Target images per class : {TARGET_COUNT}")
    print(f"  Output directory        : {AUGMENTED_DIR}")
    print()

    totals = {}
    for class_label in CLASSES:
        n = augment_class(class_label, verbose=verbose)
        totals[class_label] = n

    print()
    print("  ── Summary ──────────────────────────────────")
    for cls, cnt in totals.items():
        print(f"  {cls:20s} : {cnt:5d} images")
    print(f"\n✔  Augmentation complete → {AUGMENTED_DIR}")


# ─── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    augment_dataset(verbose=True)
