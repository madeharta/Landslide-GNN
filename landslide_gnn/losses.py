"""Class-weighted focal loss, Eqs. (29)-(30)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import NUM_CLASSES


def class_weights(labels: np.ndarray, scheme: str = "effective", beta: float = 0.999,
                  num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Per-class weights normalised so that they sum to ``num_classes``.

    ``effective``: (1 - beta) / (1 - beta**n_c)  (Cui et al., CVPR 2019)
    ``inverse``  : 1 / n_c
    ``none``     : all ones
    Classes absent from the training data receive the largest observed weight.
    """
    labels = labels[labels >= 0]
    n = np.bincount(labels, minlength=num_classes).astype(float)
    if scheme == "none":
        w = np.ones(num_classes)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            if scheme == "effective":
                w = (1.0 - beta) / (1.0 - np.power(beta, n))
            elif scheme == "inverse":
                w = 1.0 / n
            else:
                raise ValueError(f"unknown class_weighting '{scheme}'")
        finite = np.isfinite(w)
        w[~finite] = w[finite].max() if finite.any() else 1.0
    return (w / w.sum() * num_classes).astype(np.float32)


class FocalLoss(nn.Module):
    """Mean over labelled nodes of  -w_y (1 - p_y)^gamma log p_y.

    Nodes with label < 0 or ``mask == False`` are ignored. With gamma = 0
    this is weighted cross-entropy.
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.ones(NUM_CLASSES))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None):
        valid = target >= 0
        if mask is not None:
            valid &= mask
        if not valid.any():
            return logits.sum() * 0.0
        logits, target = logits[valid], target[valid]
        logp = F.log_softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
        p = logp.exp()
        loss = -self.weight[target] * (1.0 - p).pow(self.gamma) * logp
        return loss.mean()
