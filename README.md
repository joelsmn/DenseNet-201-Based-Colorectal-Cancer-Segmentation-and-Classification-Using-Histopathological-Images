# Colorectal Cancer Detection — Neurosymbolic Graph Transformer Pipeline

A full end-to-end deep-learning pipeline for multi-class colorectal cancer
classification from histopathological images.

---

## Classes (case-sensitive)
| Label | Folder in dataset | Count |
|---|---|---|
| Adenocarcinoma | Adenocarcinoma | 795 |
| Serrated Adenoma | Serrated Adenoma | 58 |
| Polyp | Polyp | 474 |
| Benign | Normal | 76 |
| Low Grade IN | Low Grade IN | 637 |
| High Grade IN | High Grade IN | 186 |

---

## Project Structure

```
colorectal_cancer_project/
├── config/
│   └── config.py           ← All paths, hyperparameters, class maps
├── preprocessing/
│   └── preprocess.py       ← Stage 1: crop, Reinhard normalization, resize
├── augmentation/
│   └── augment.py          ← Stage 2: augment to 1000/class
├── segmentation/
│   └── unet.py             ← Stage 3: U-Net binary segmentation
├── models/
│   ├── attention.py        ← CCA, CBAM, SE attention modules
│   └── backbones.py        ← All 15 backbones with CCA injection
├── graph/
│   └── graph_construction.py  ← Stage 5: rKNN graph + positional encoding
├── symbolic/
│   └── knowledge_graph.py  ← Stage 6: KG encoder + PathNet gate
├── classifier/
│   └── neurosymbolic_graph_transformer.py  ← Stages 7-10: full NSGT model
├── training/
│   └── trainer.py          ← Stage 11: training engine
├── evaluation/
│   └── evaluate.py         ← Confusion matrix, ROC, comparison plots
├── utils/
│   ├── datasets.py         ← PyTorch Dataset wrappers
│   └── split_dataset.py    ← 70/15/15 stratified split
├── run_pipeline.py         ← Master entry point
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure paths
Open `config/config.py` and verify:
```python
ROOT_DIR    = Path(r"X:\P1")
DATASET_DIR = ROOT_DIR / "EBHI-SEG"
```

---

## Running the Pipeline

### Full pipeline (all stages)
```bash
python run_pipeline.py
```

### Individual stages
```bash
python run_pipeline.py --stages 1          # preprocess only
python run_pipeline.py --stages 1,2,3      # preprocess + augment + split
python run_pipeline.py --stages 5,6        # train + evaluate
```

### Train specific models only
```bash
python run_pipeline.py --stages 5,6 --models ResNet50,DenseNet121
```

### Skip already-completed stages
```bash
python run_pipeline.py --skip-done
```

### Run individual modules directly
```bash
python preprocessing/preprocess.py
python augmentation/augment.py
python utils/split_dataset.py
python segmentation/unet.py
python training/trainer.py
python evaluation/evaluate.py
```

---

## Pipeline Stages

| # | Stage | Description |
|---|---|---|
| 1 | Preprocessing | Border crop, Reinhard H&E normalisation, 224×224 resize, Gaussian denoise |
| 2 | Augmentation | Augment every class to 1000 images (rotation, flip, HSV jitter, crop, zoom, elastic deformation, CLAHE) |
| 3 | Split | 70/15/15 stratified split into train/val/test |
| 4 | Segmentation | U-Net trained on ground-truth masks (Dice+BCE loss) |
| 5 | Training | All 15 backbones + CCA layer, Adam, focal CE loss, early stopping |
| 6 | Evaluation | Confusion matrices, ROC curves, comparative bar chart, table |

---

## Architecture Notes

### Colour Channel Attention (CCA)
Inserted after the **first convolution** of every backbone. Uses global average
pooling + global max pooling → concat → 2×FC → Sigmoid channel scale.
Specifically addresses the H&E colour overlap between Normal/Polyp/High-grade
IN/Serrated Adenoma.

### Neurosymbolic Graph Transformer (full model)
For use after backbones are individually trained:
```
image → Backbone (CCA) → feat_vec
      → U-Net mask → rKNN graph → Graph Transformer (attention bias) → graph_feat
      → Symbolic Layer (KG + PathNetGate) → sym_feat
      → Fusion → 6-class + binary (Benign/Malignant) output
```

### Confusable Class Strategy
Classes `Benign`, `Polyp`, `High Grade IN`, `Serrated Adenoma` share glandular
structure. Addressed by:
- **Focal Loss** (γ=2) down-weights easy samples
- **Label Smoothing** (ε=0.1) prevents overconfident predictions
- **KG confusable edges** make the symbolic layer aware of confusion pairs
- **CCA** emphasises colour differences early

---

## Output Files

All outputs written to `X:\P1\RESULTS\`:
- `cm_<ModelName>.png` — Confusion matrix per model
- `roc_<ModelName>.png` — ROC curves per model
- `history_<ModelName>.png` — Loss/accuracy training curves
- `model_comparison.png` — Comparative bar chart (all models)
- `comparison_table.txt` — Accuracy / F1 / Precision / Recall table

Model checkpoints: `X:\P1\SAVED_MODELS\<ModelName>_best.pth`
Training logs:     `X:\P1\LOGS\<ModelName>_log.csv`

---

## Hardware Recommendation
- GPU: NVIDIA RTX 3060+ (8GB+ VRAM) for BATCH_SIZE=16
- Reduce BATCH_SIZE in `config.py` if you get CUDA OOM errors
- CPU fallback is supported but training will be slow

---

## Deep Ablation Study (NCVPRIPG ATR revision)

`evaluation/` now contains the full ablation suite required by the
Author's Technical Response, covering the CCA layer, seven segmentation
backbones (U-Net, U-Net++, Attention U-Net, DeepLabV3, TransUNet-lite,
Swin-UNet-lite, nnU-Net-style), the incremental CCA -> Graph -> KG/PathNet
gate study, an SE/CBAM/ECA/CCA attention comparison, neurosymbolic gate
isolation, a computational-cost / resource-utilisation breakdown, and a
consolidated paper-ready report generator.

Run the whole suite with:
```bash
python evaluation/run_all_ablations.py
```

See **`ABLATION_STUDY_GUIDE.md`** for the full mapping from each ATR
reviewer comment to its script, output file, and suggested manuscript
placement.
