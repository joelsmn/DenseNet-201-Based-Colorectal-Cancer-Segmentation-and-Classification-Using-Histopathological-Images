"""
run_pipeline.py
─────────────────────────────────────────────────────────────────────────────
MASTER PIPELINE — Colorectal Cancer Detection
─────────────────────────────────────────────────────────────────────────────

Run the full end-to-end pipeline from VS Code terminal:

    python run_pipeline.py [--stages all] [--models all] [--skip-done]

Flags
  --stages   Comma-separated list of stage numbers to run, or "all"
             1=preprocess  2=augment  3=split  4=segment
             5=train       6=evaluate
             Default: all

  --models   Comma-separated model names (from config) or "all"
             Example:  --models ResNet50,DenseNet121

  --skip-done  Skip a stage if its output directory already exists and
               contains files (useful for resuming interrupted runs)

Progress is printed as a % at every step so you always know how far along
execution is inside VS Code.
─────────────────────────────────────────────────────────────────────────────
"""


import argparse
import sys
import time
from pathlib import Path




def banner(text: str) -> None:
    sep = "═" * 64
    print(f"\n{sep}")
    print(f"  {text}")
    print(f"{sep}")


def check_skip(directory: Path, label: str, skip: bool) -> bool:
    if skip and directory.exists():
        files = list(directory.rglob("*"))
        if files:
            print(f"  [SKIP] {label} — output dir exists with {len(files)} files.")
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Colorectal Cancer Detection — Full Pipeline")
    parser.add_argument("--stages",    default="all",
                        help="Stage numbers to run (comma-separated or 'all')")
    parser.add_argument("--models",    default="all",
                        help="Model names (comma-separated or 'all')")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip stages whose output already exists")
    args = parser.parse_args()

    # Import after path resolution
    sys.path.append(str(Path(__file__).resolve().parent))
    from config.config import (
        PROCESSED_DIR, AUGMENTED_DIR, SPLIT_DIR,
        MODEL_NAMES, MODELS_DIR,
    )

    # Parse stage list
    if args.stages.lower() == "all":
        stages = list(range(1, 7))
    else:
        stages = [int(s.strip()) for s in args.stages.split(",")]

    # Parse model list
    if args.models.lower() == "all":
        model_names = MODEL_NAMES
    else:
        model_names = [m.strip() for m in args.models.split(",")]

    t0 = time.time()

    # ── Stage 1: Preprocessing ─────────────────────────────────────────
    if 1 in stages:
        banner("STAGE 1 / 6 — PREPROCESSING")
        if not check_skip(PROCESSED_DIR, "Preprocessing", args.skip_done):
            from preprocessing.preprocess import preprocess_dataset
            preprocess_dataset(verbose=True)

    # ── Stage 2: Augmentation ─────────────────────────────────────────
    if 2 in stages:
        banner("STAGE 2 / 6 — AUGMENTATION")
        if not check_skip(AUGMENTED_DIR, "Augmentation", args.skip_done):
            from augmentation.augment import augment_dataset
            augment_dataset(verbose=True)

    # ── Stage 3: Dataset Split ────────────────────────────────────────
    if 3 in stages:
        banner("STAGE 3 / 6 — DATASET SPLIT (70/15/15)")
        if not check_skip(SPLIT_DIR, "Split", args.skip_done):
            from utils.split_dataset import split_dataset
            split_dataset(verbose=True)

    # ── Stage 4: U-Net Segmentation ───────────────────────────────────
    if 4 in stages:
        banner("STAGE 4 / 6 — U-NET SEGMENTATION")
        from segmentation.unet import train_unet
        train_unet(verbose=True)

    # ── Stage 5: Train all backbones ──────────────────────────────────
    if 5 in stages:
        banner("STAGE 5 / 6 — BACKBONE TRAINING")
        from training.trainer import train_all_models
        results = train_all_models(model_names=model_names, verbose=True)
        # Store results for eval stage
        import json
        summary = []
        for r in results:
            if "error" in r:
                summary.append({"model": r["model_name"],
                                 "status": "FAILED",
                                 "error": r["error"]})
            else:
                summary.append({"model": r["model_name"],
                                 "status": "OK",
                                 "best_val_acc": r["best_val_acc"]})
        (MODELS_DIR / "training_summary.json").write_text(
            json.dumps(summary, indent=2))

    # ── Stage 6: Evaluation ───────────────────────────────────────────
    if 6 in stages:
        banner("STAGE 6 / 6 — EVALUATION & VISUALISATION")
        from evaluation.evaluate import evaluate_all, plot_training_history
        # Reload histories if available
        eval_results = evaluate_all(model_names=model_names)
        plot_training_history(eval_results)

    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE   Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
