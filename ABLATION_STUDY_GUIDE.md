# CRC-AI Deep Ablation Study — Guide

This document maps every ATR (Author's Technical Response) commitment to
the code that fulfils it, the CSV/PNG it produces, and where that output
should land in the revised NCVPRIPG manuscript. Use it as the checklist
for closing out the "implement the manuscript additions" action item.

---

## 0. One-command run

```bash
cd PROJECTX
pip install -r requirements.txt

# Point config/config.py's ROOT_DIR at your EBHI-SEG-SPLIT location first
# (currently set to X:\P1 — update if your dataset lives elsewhere).

python evaluation/run_all_ablations.py            # everything, ~ several hours on one GPU
python evaluation/run_all_ablations.py --list     # see stage names
python evaluation/run_all_ablations.py --only incremental unet cost   # subset
```

Every stage is independently runnable (`python evaluation/ablation_incremental.py`,
etc.) and writes its own CSV to `LOGS/` and PNG to `RESULTS/`. Run
`evaluation/generate_visuals.py` at any time to regenerate the
consolidated `RESULTS/ABLATION_REPORT.md` from whatever CSVs currently
exist — you do not have to re-run the whole suite to refresh the report.

All ablations default to a **15-epoch budget** (see `NUM_EPOCHS`
override at the top of each script) rather than the pipeline's full
50-epoch training run, matching the budget already used by the existing
`ablation_cca.py` / `ablation_graph.py` / `ablation_loss.py` /
`ablation_augmentation.py` scripts, so that all ablation arms are
directly comparable to each other. Increase it if you have compute
headroom before the camera-ready deadline.

---

## 1. Map: ATR comment → script → output → manuscript location

| # | ATR comment | Script(s) | Output | Suggested manuscript placement |
|---|---|---|---|---|
| R1‑2, R1‑6 | Incremental ablation: CCA → Graph → KG/PathNet gate, Acc + Macro‑F1 at each stage | `evaluation/ablation_incremental.py` | `LOGS/ablation_incremental.csv`, `RESULTS/ablation_incremental.png` | New **Table: Incremental Module Ablation** in the Ablation Study section; cite the per‑stage Δ Accuracy / Δ Macro‑F1 in the response‑to‑reviewers table itself |
| R2‑4 | Same request, cross‑referenced from Reviewer #2 | *(same as above)* | *(same as above)* | Cross‑reference the same table; no separate experiment needed |
| R1‑4 | CCA vs SE vs CBAM vs ECA on the DenseNet201 stem | `evaluation/ablation_attention_variants.py` | `ablation_attention_variants.csv/.png` | New **Table: Attention‑Module Comparison**, placed next to the existing Table 1 backbone comparison; the accompanying paragraph already drafted in the ATR (GAP+GMP fusion at the stem vs. SE's GAP‑only vs. CBAM's added spatial branch) can now cite empirical numbers instead of only an architectural argument |
| R1‑7 | Necessity/effectiveness of the U‑Net segmentation step; comparison across segmentation backbones | `evaluation/ablation_unet_variants.py` | `ablation_unet_variants.csv/.png` | New **Table: Segmentation‑Backbone Ablation** (Dice/IoU) in the segmentation subsection; use the Dice‑vs‑params scatter panel of the PNG as a supplementary figure |
| R2‑1 | Novelty of the neurosymbolic module = the *gate*, not just having KG features | `evaluation/ablation_neurosymbolic.py` | `ablation_neurosymbolic.csv/.png` | New **Table: Neurosymbolic Gate Isolation** (A: no symbolic layer, B: KG concat w/o gate, C: full PathNet gate) — this is the strongest empirical evidence for the novelty paragraph already drafted in the ATR |
| R1‑5, R2‑3 | Computational cost / inference time / memory / deployment impact of stacking modules | `evaluation/compute_cost.py` | `compute_cost.csv`, `compute_cost_table.md`, `compute_cost_breakdown.png` | Replace/extend the existing "31 ms/image, 0.98 s TTA, ~20M params" sentence with the full per‑module breakdown table; use the cumulative S0→S3 latency panel to directly answer "impact of integrating multiple modules" |
| R1‑1 | Springer LNCS formatting / page count | *(not a code deliverable)* | — | Formatting-only; addressed at LaTeX/Word template level |
| R1‑3, R2‑2 | External dataset validation (NCT‑CRC‑HE‑100K, TCGA‑COAD/READ) | *(future work — out of scope for this codebase)* | — | Keep as the limitation statement already drafted in the ATR; not an ablation |
| — | "All other possible modules/processes" (open‑ended ask) | `evaluation/ablation_other_modules.py` | `ablation_other_modules.csv/.png` | Optional **Supplementary Table**: TTA view‑count, class‑weighting, optimiser choice, backbone choice, dropout rate — use selectively if reviewers push further at camera‑ready stage |
| — | Loss‑function choice (already existing) | `evaluation/ablation_loss.py` | `ablation_loss.csv` | Already referenced in ATR; no change needed |
| — | Augmentation strategy (already existing) | `evaluation/ablation_augmentation.py` | `ablation_augmentation.csv` | Already referenced in ATR; no change needed |
| — | Graph‑construction ablation (already existing) | `evaluation/ablation_graph.py` | `ablation_graph.csv` | Already referenced in ATR; no change needed |
| — | Original CCA‑only ablation (already existing) | `evaluation/ablation_cca.py` | `ablation_cca.csv` | Already referenced in ATR; superseded in scope by `ablation_incremental.py`'s S0→S1 step, but kept for backward compatibility |

---

## 2. What each new module actually measures (so you can defend it under
further review)

- **`models/attention.py`** implements CCA, SE, CBAM, and ECA behind one
  shared interface so the attention ablation swaps a single line
  (`build_attention(name, channels)`), guaranteeing every other
  hyperparameter is identical across arms — this is what makes the
  Table‑1 comparison in the ATR paragraph *actually* controlled rather
  than just citing SEResNet/CBAMResNet as separately-trained backbones
  from Table 1 (which differ from CCA‑DenseNet201 in more ways than
  just the attention module).

- **`segmentation/unet_variants.py`** implements the six additional
  segmentation backbones on the *same* 5‑level topology and *same*
  `DoubleConv` building block as the existing `segmentation/unet.py`
  baseline wherever architecturally possible (UNet++, Attention U‑Net,
  nnU‑Net‑style), so Dice/IoU deltas are attributable to the specific
  architectural change (nested skips, attention gates, instance‑norm +
  deep supervision) rather than confounded by encoder capacity. For
  DeepLabV3 and the two attention‑bottleneck models (TransUNet‑lite,
  Swin‑UNet‑lite) a same‑topology reproduction isn't architecturally
  meaningful, so those are faithful "lite" reproductions sized for a
  single‑GPU, 224×224 ablation budget rather than the original papers'
  full‑scale configurations — this is noted explicitly in each class's
  docstring so it can be stated accurately in the manuscript's
  experimental‑setup paragraph.

  **Important caveat for nnU‑Net specifically**: `NNUNetStyleUNet`
  reproduces nnU‑Net's *architectural signature* (instance norm, leaky
  ReLU, deep supervision) but **not** its self‑configuring
  preprocessing pipeline, which selects patch size, stage count, and
  augmentation from dataset fingerprinting rather than being a fixed
  architecture at all. State this precisely if a reviewer asks — do not
  describe this arm as "the nnU‑Net framework."

- **`evaluation/ablation_incremental.py`** is the ablation the ATR
  explicitly promised in R1‑2/R1‑6/R2‑4. Each stage (S0→S3) is one
  `IncrementalCRCModel` instance with the *same* backbone
  initialisation, optimiser, LR schedule, and data pipeline; only the
  `use_cca` / `use_graph` / `use_symbolic` flags change, so the
  accuracy/Macro‑F1 deltas reported are attributable to that stage's
  added module.

- **`evaluation/ablation_neurosymbolic.py`** goes one level deeper than
  the incremental ablation's S2→S3 step: it isolates *why* the symbolic
  layer helps by comparing "no symbolic layer" vs. "KG embedding
  concatenated with no gating" vs. "full PathNet gate," which is the
  exact empirical claim the ATR's R2‑1 response makes in prose
  ("novelty lies... in the differentiable PathNet gating mechanism...
  rather than applying static post‑hoc rules").

- **`evaluation/compute_cost.py`** measures every module in isolation
  (attention variants, segmentation backbones, classification
  backbones, graph transformer, symbolic layer) AND the cumulative
  S0→S3 pipeline, including an 8‑view TTA latency figure that extends
  the ATR's existing "31 ms single‑pass / 0.98 s TTA" numbers into a
  full per‑module and per‑stage breakdown.

---

## 3. Known limitations to disclose alongside these results

1. **Single-dataset training throughout** — every ablation here trains
   and evaluates on the same EBHI‑SEG split used elsewhere in the
   paper. This does not address R1‑3/R2‑2 (external‑dataset
   generalisation); that remains a stated limitation / future‑work
   item, not something these scripts can close.
2. **Reduced epoch budget (15 vs. 50)** for ablation runs — chosen for
   parity with the pre‑existing ablation scripts and for tractable
   total compute across ~10 sweeps; if reviewers question convergence,
   re‑run the specific arm(s) in question at the full 50‑epoch budget
   and report both numbers.
3. **TransUNet‑lite / Swin‑UNet‑lite / nnU‑Net‑style are budget‑matched
   reproductions**, not the original papers' full configurations (see
   §2 above) — state this precisely in the experimental setup if these
   comparisons are included in the camera‑ready version.
4. **DeepLabV3 falls back to a from‑scratch ASPP decoder** if
   torchvision's pretrained weights can't be downloaded (e.g. offline
   compute cluster) — check the console output for a fallback notice
   before trusting its absolute numbers; the fallback arm still trains
   from scratch on EBHI‑SEG so it remains a fair comparison, just not
   ImageNet/COCO‑pretrained like the other arms implicitly are via
   their CNN encoders' pretrained weights.
5. **The `unet` ablation's downstream‑classification step is a
   documented template, not a full sweep** (see
   `run_downstream_classification()` in `ablation_unet_variants.py`) —
   it reports Dice/IoU per backbone but does not, by default, retrain
   the full CCA+Graph classifier on every backbone's *predicted* masks.
   Follow the inline instructions there if the paper needs that full
   cross‑product for the camera‑ready version; it was left as a
   template rather than auto‑run given the added ~7× training cost.
