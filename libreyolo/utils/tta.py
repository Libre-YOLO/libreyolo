"""Shared merge math for flip test-time-augmentation.

Used by the three independent flip-TTA call sites for semantic segmentation
(``BaseModel._predict_augment_semantic``, ``SemanticValidator``'s batched
validation, and ``LibreEoMT.val()``'s bespoke per-image loop) so the merge
policy lives in exactly one place.
"""

from __future__ import annotations

import torch


def average_flip_softmax(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Average softmax probabilities of two spatially-aligned logit views.

    Both tensors must already be in the same spatial orientation (the
    flipped view's logits flipped back before calling this) and share
    ``[..., C, ...]`` shape with softmax taken over ``dim=1``. Averaging
    probabilities rather than raw logits is the standard flip-TTA
    convention (e.g. mmsegmentation): it bounds each view's contribution to
    a proper simplex before combining, so a logit-scale mismatch between the
    two views (e.g. near letterbox-padding borders) can't disproportionately
    dominate the merge.

    The accumulate/divide runs in place on this function's own freshly
    allocated softmax buffer (never on a caller's tensor) so that a dense
    semantic merge holds two full ``[B, C, H, W]`` float tensors rather than
    four: at ADE20K scale (150 classes, 512x512, batch 8) each one is 1.2 GB,
    so the naive ``(a.softmax() + b.softmax()) / 2`` spends several GB on
    temporaries that exist only to be discarded.
    """
    probs = logits_a.softmax(dim=1)
    probs.add_(logits_b.softmax(dim=1)).div_(2)
    return probs
