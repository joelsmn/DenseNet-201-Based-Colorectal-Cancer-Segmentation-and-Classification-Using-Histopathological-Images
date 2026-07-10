# DenseNet-201 Based Colorectal Cancer Segmentation and Classification Using Histopathological Images

This repository contains an end-to-end deep learning pipeline for colorectal cancer analysis from histopathological images. It combines preprocessing, augmentation, segmentation, classification, and evaluation into a single workflow.

## Overview

The project is designed to:

- preprocess histopathological images,
- augment and balance the dataset,
- train a U-Net-based segmentation model,
- train multiple classification backbones, including DenseNet201,
- generate evaluation plots and performance reports.

The dataset is expected to be stored locally outside the repository, so the codebase stays lightweight and Git-friendly.

## Project workflow

1. Preprocess the images
2. Split the data into train/validation/test sets
3. Augment and balance the classes
4. Train a segmentation network
5. Train classification models
6. Evaluate performance and save outputs

## Supported classes

The pipeline is configured for the following classes:

- Adenocarcinoma
- Serrated Adenoma
- Polyp
- Benign
- Low Grade IN
- High Grade IN

## Repository structure

- [config/config.py](config/config.py) — central configuration, dataset paths, and hyperparameters
- [preprocessing/preprocess.py](preprocessing/preprocess.py) — image preprocessing pipeline
- [augmentation/augment.py](augmentation/augment.py) — augmentation and class balancing
- [utils/split_dataset.py](utils/split_dataset.py) — train/validation/test splitting
- [segmentation/unet.py](segmentation/unet.py) — U-Net segmentation training
- [training/trainer.py](training/trainer.py) — backbone training for classification
- [evaluation/evaluate.py](evaluation/evaluate.py) — metrics and visualization
- [run_pipeline.py](run_pipeline.py) — main entry point for the full pipeline
- [launch_gui.py](launch_gui.py) — optional GUI launcher

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
venv\\Scripts\\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the dataset path

Open [config/config.py](config/config.py) and update the root path to match your local dataset location.

```python
ROOT_DIR = Path(r"X:\\P1")
DATASET_DIR = ROOT_DIR / "EBHI-SEG"
```

## Usage

### Run the full pipeline

```bash
python run_pipeline.py
```

### Run specific stages

```bash
python run_pipeline.py --stages 1,2,3
python run_pipeline.py --stages 4
python run_pipeline.py --stages 5,6
```

### Train a specific model

```bash
python run_pipeline.py --stages 5,6 --models DenseNet201
```

### Skip already completed stages

```bash
python run_pipeline.py --skip-done
```

## Optional GUI

```bash
python launch_gui.py
```

## Outputs

The project writes results to the directories configured in [config/config.py](config/config.py), including:

- evaluation plots and confusion matrices,
- ROC curves,
- saved model checkpoints,
- training logs.

## Notes

- This repository does not include the dataset image files.
- The dataset must be available locally before running the pipeline.
- Large data files, model weights, and output folders are ignored by [.gitignore](.gitignore).

## License

This project is intended for academic and research use.
