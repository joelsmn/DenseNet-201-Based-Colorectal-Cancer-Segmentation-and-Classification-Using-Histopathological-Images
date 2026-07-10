"""
config.py — Central configuration for the Colorectal Cancer Detection Pipeline
All paths, hyperparameters, and class definitions live here.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
#  ROOT & DATASET PATHS
# ─────────────────────────────────────────────
ROOT_DIR        = Path(r"X:\P1")
DATASET_DIR     = ROOT_DIR / "EBHI-SEG"
AUGMENTED_DIR   = ROOT_DIR / "EBHI-SEG-AUG"
PROCESSED_DIR   = ROOT_DIR / "EBHI-SEG-PROCESSED"
SPLIT_DIR       = ROOT_DIR / "EBHI-SEG-SPLIT"
SEG_PRED_DIR    = ROOT_DIR / "SEG_PREDICTIONS"
RESULTS_DIR     = ROOT_DIR / "RESULTS"
MODELS_DIR      = ROOT_DIR / "SAVED_MODELS"
LOGS_DIR        = ROOT_DIR / "LOGS"

for d in [AUGMENTED_DIR, PROCESSED_DIR, SPLIT_DIR,
          SEG_PRED_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
#  CLASS DEFINITIONS  (CASE SENSITIVE)
# ─────────────────────────────────────────────
CLASSES = [
    "Adenocarcinoma",
    "Serrated Adenoma",
    "Polyp",
    "Benign",          # folder name in dataset is "Normal"
    "Low Grade IN",
    "High Grade IN",
]

# Map folder names → canonical class labels
FOLDER_TO_CLASS = {
    "Adenocarcinoma": "Adenocarcinoma",
    "Serrated Adenoma": "Serrated Adenoma",
    "Polyp": "Polyp",
    "Normal": "Benign",
    "Low Grade IN": "Low Grade IN",
    "High Grade IN": "High Grade IN",
}

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Malignant vs Benign grouping (for binary output)
MALIGNANT_CLASSES = {
    "Adenocarcinoma", "Serrated Adenoma", "Polyp",
    "Low Grade IN", "High Grade IN",
}
BENIGN_CLASSES = {"Benign"}

# Original image counts per class
ORIGINAL_COUNTS = {
    "Adenocarcinoma": 785,
    "Serrated Adenoma": 53,
    "Polyp": 469,
    "Benign": 71,
    "Low Grade IN": 629,
    "High Grade IN": 181,
}

# ─────────────────────────────────────────────
#  AUGMENTATION
# ─────────────────────────────────────────────
TARGET_COUNT     = 1000   # images per class after augmentation
IMAGE_SIZE       = (224, 224)
IMAGE_CHANNELS   = 3

# ─────────────────────────────────────────────
#  DATASET SPLIT RATIOS
# ─────────────────────────────────────────────
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ─────────────────────────────────────────────
#  TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────
BATCH_SIZE       = 16
NUM_EPOCHS       = 40
LEARNING_RATE    = 1e-4
WEIGHT_DECAY     = 1e-4
LR_PATIENCE      = 5       # ReduceLROnPlateau patience
EARLY_STOP_PAT   = 10      # EarlyStopping patience
NUM_WORKERS      = 4
PIN_MEMORY       = True
SEED             = 42

# Class weights for imbalanced classes (computed dynamically in training)
USE_CLASS_WEIGHTS = True

# ─────────────────────────────────────────────
#  SEGMENTATION (U-Net)
# ─────────────────────────────────────────────
UNET_FILTERS     = [64, 128, 256, 512, 1024]
UNET_EPOCHS      = 30
UNET_LR          = 1e-3
UNET_BATCH       = 8

# ─────────────────────────────────────────────
#  GRAPH CONSTRUCTION (KNN)
# ─────────────────────────────────────────────
KNN_K            = 8       # K neighbours
KNN_RADIUS       = 50.0    # pixel radius for radius-based KNN
POS_ENC_DIM      = 16      # positional encoding dimension

# ─────────────────────────────────────────────
#  MODEL LIST FOR COMPARATIVE ANALYSIS
# ─────────────────────────────────────────────
MODEL_NAMES = [
    "SwinTransformer",
    "VisionTransformer",
    "ConvNeXt",
    "DenseNet121",
    "DenseNet201",
    "EfficientNetB4",
    "EfficientNetV2",
    "InceptionResNetV2",
    "Xception",
    "ResNet50",
    "ResNet101",
    "SEResNet",
    "CBAMResNet",
    "BEiT",
    "DeiT",
]

# ─────────────────────────────────────────────
#  CONFUSION-SIMILAR CLASS PAIRS (for focal loss tuning)
# ─────────────────────────────────────────────
CONFUSABLE_GROUPS = [
    ["Benign", "Polyp", "High Grade IN", "Serrated Adenoma"],
]
