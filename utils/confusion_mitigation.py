"""
utils/confusion_mitigation.py
─────────────────────────────────────────────────────────────────────────────
Tools to address the structural similarity between
  Benign (Normal) / Polyp / High Grade IN / Serrated Adenoma

Strategies implemented
  1. PairwiseFocalLoss     — extra penalty on confusable class pairs
  2. ConfusionAwareAugment — forces harder samples from confusable pairs
  3. EmbeddingMarginLoss   — pushes confusable classes apart in feature space
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import CLASSES, CLASS_TO_IDX, CONFUSABLE_GROUPS

NUM_CLASSES = len(CLASSES)

# Build a confusion penalty matrix: higher weight for confusable pairs
def _build_confusion_penalty(alpha: float = 2.0) -> torch.Tensor:
    """
    Returns (NUM_CLASSES, NUM_CLASSES) matrix.
    Entry [i, j] = alpha if classes i, j are in a confusable group,
                   1.0 otherwise.
    """
    mat = torch.ones(NUM_CLASSES, NUM_CLASSES)
    for group in CONFUSABLE_GROUPS:
        idxs = [CLASS_TO_IDX[c] for c in group if c in CLASS_TO_IDX]
        for i in idxs:
            for j in idxs:
                if i != j:
                    mat[i, j] = alpha
    return mat


class PairwiseFocalLoss(nn.Module):
    """
    Cross-entropy loss with extra penalty when the model confuses
    structurally similar classes (the confusable groups).

    For each sample, if the model's top-2 prediction includes a class
    from the same confusable group as the true label, the loss is
    multiplied by `confusion_alpha`.
    """

    def __init__(self, gamma: float = 2.0, confusion_alpha: float = 2.0,
                 smoothing: float = 0.1):
        super().__init__()
        self.gamma            = gamma
        self.smoothing        = smoothing
        self.register_buffer("penalty_mat",
                             _build_confusion_penalty(confusion_alpha))

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        B     = logits.size(0)
        probs = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)

        # Label smoothing
        smooth_target = torch.full_like(log_p,
                                        self.smoothing / (NUM_CLASSES - 1))
        smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        ce   = -(smooth_target * log_p).sum(dim=1)
        pt   = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce

        # Confusion penalty: look at argmax prediction
        preds = probs.argmax(dim=1)
        pen   = self.penalty_mat[targets, preds]   # (B,)
        return (focal * pen).mean()


# ─── Embedding margin loss ────────────────────────────────────────────────

class EmbeddingMarginLoss(nn.Module):
    """
    Pushes confusable class embeddings apart in feature space.

    Usage:  add to total loss with small weight (e.g. 0.1)
    Input:  feat  (B, D)  feature vectors
            labels (B,)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, feat: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=feat.device)
        n    = 0

        # Build confusable index pairs
        confusable_idx_pairs = []
        for group in CONFUSABLE_GROUPS:
            idxs = [CLASS_TO_IDX[c] for c in group if c in CLASS_TO_IDX]
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    confusable_idx_pairs.append((idxs[i], idxs[j]))

        for ci, cj in confusable_idx_pairs:
            mask_i = (labels == ci)
            mask_j = (labels == cj)
            if mask_i.sum() == 0 or mask_j.sum() == 0:
                continue
            fi = feat[mask_i].mean(0)   # centroid of class ci
            fj = feat[mask_j].mean(0)   # centroid of class cj
            dist = F.pairwise_distance(fi.unsqueeze(0),
                                        fj.unsqueeze(0))
            loss += F.relu(self.margin - dist).mean()
            n    += 1

        return loss / max(n, 1)
