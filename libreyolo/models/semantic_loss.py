"""Loss helpers shared by the native semantic-segmentation heads.

The Lovász-Softmax surrogate (Berman, Rannen Triki & Blaschko, CVPR 2018)
optimizes the Jaccard/IoU index directly, complementing pixel-wise cross
entropy which only optimizes per-pixel accuracy. Implemented from the paper's
equations (the Lovász extension of the submodular Jaccard loss), not adapted
from any third-party source.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Jaccard loss for a vector of ground-truth labels
    sorted by descending prediction error (paper eq. for the Lovász hinge)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
    return jaccard


def lovasz_softmax_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Multi-class Lovász-Softmax loss over the present classes.

    Args:
        logits: ``[B, C, H, W]`` class logits.
        targets: ``[B, H, W]`` integer labels; ``ignore_index`` pixels excluded.
        ignore_index: label value to skip.

    Returns a scalar that is graph-connected even when every pixel is ignored.
    """
    probs = logits.softmax(dim=1)
    b, c, h, w = probs.shape
    probs = probs.permute(0, 2, 3, 1).reshape(-1, c)  # [P, C]
    labels = targets.reshape(-1)

    valid = labels != ignore_index
    probs = probs[valid]
    labels = labels[valid]
    if probs.numel() == 0:
        # Graph-connected zero, mirroring the all-ignore guard in the heads.
        return logits.sum() * 0.0

    losses = []
    for cls in range(c):
        fg = (labels == cls).float()
        if fg.sum() == 0:
            continue  # class absent in this batch — skip (classes='present')
        errors = (fg - probs[:, cls]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        grad = _lovasz_grad(fg[perm])
        losses.append(torch.dot(errors_sorted, grad))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def semantic_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
    lovasz_weight: float = 1.0,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy plus an optional Lovász-Softmax (IoU-surrogate) term.

    The Lovász term is evaluated at the logits' native resolution (targets are
    nearest-downsampled to match) so it stays cheap regardless of the upsample
    factor, while cross-entropy is computed at full target resolution.
    """
    targets = targets.long()
    if not bool((targets != ignore_index).any()):
        return logits.sum() * 0.0  # all-ignore batch: graph-connected zero

    up_logits = F.interpolate(
        logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
    )
    ce = F.cross_entropy(
        up_logits,
        targets,
        ignore_index=ignore_index,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    if lovasz_weight <= 0.0:
        return ce

    small_targets = (
        F.interpolate(
            targets.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest"
        )
        .squeeze(1)
        .long()
    )
    lovasz = lovasz_softmax_loss(logits, small_targets, ignore_index)
    return ce + lovasz_weight * lovasz
