"""Contrastive denoising for a variable per-image query count.

Ported from Dome-DETR (https://github.com/RicePasteM/Dome-DETR),
commit 2dde3bc1946a3e9fad9abd0612b59fc39bd6b861, Apache License 2.0.
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
Modified from D-FINE (https://github.com/Peterande/D-FINE).
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

The noised query construction is byte-identical to D-FINE's and is reused from
``models/dfine/denoising.py`` rather than duplicated. Only the attention mask
differs, and it has to:

PAQI gives every image in a batch a different number of queries, so the
decoder input is padded to the batch maximum. D-FINE returns one 2D
``(tgt, tgt)`` mask shared by the whole batch, which cannot express "image 0's
real queries end at 264 but image 1's end at 431". Left alone, real queries
attend to another image's padding and the padded rows contribute gradient.

So the mask becomes 3D, ``(bs * num_heads, tgt, tgt)`` — the layout
``nn.MultiheadAttention`` expects — with each image's padding masked out of
both directions. Padded rows are deliberately left able to attend to
themselves: a fully-masked row makes softmax return NaN.

This is upstream's fix for the single-image-per-GPU limitation their README
used to carry.
"""

from __future__ import annotations

import torch

from ..dfine.denoising import (
    get_contrastive_denoising_training_group as _dfine_denoising_group,
)


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    batch_queries_num=None,
    num_heads=8,
):
    """D-FINE's denoising group with a per-image, per-head attention mask.

    ``num_queries`` is the padded width (the batch maximum);
    ``batch_queries_num`` gives each image's real count.
    """
    logits, bbox_unact, attn_mask, dn_meta = _dfine_denoising_group(
        targets,
        num_classes,
        num_queries,
        class_embed,
        num_denoising=num_denoising,
        label_noise_ratio=label_noise_ratio,
        box_noise_scale=box_noise_scale,
    )

    if batch_queries_num is None:
        return logits, bbox_unact, attn_mask, dn_meta

    bs = len(batch_queries_num)
    dn_len = int(dn_meta["dn_num_split"][0])
    tgt_size = dn_len + num_queries
    device = targets[0]["labels"].device

    if attn_mask is None:
        # No ground truth anywhere in the batch: D-FINE skips the denoising
        # group entirely, but the padding still has to be masked.
        attn_mask = torch.zeros((tgt_size, tgt_size), dtype=torch.bool, device=device)

    expanded = attn_mask.unsqueeze(0).repeat(bs * num_heads, 1, 1)

    for b, valid in enumerate(batch_queries_num):
        padding_start = dn_len + int(valid)
        if padding_start >= tgt_size:
            continue
        head_start, head_end = b * num_heads, (b + 1) * num_heads
        # Real queries must not see this image's padding...
        expanded[head_start:head_end, :padding_start, padding_start:] = True
        # ...and padded rows must not leak into the real ones. They keep
        # attending to themselves, which is what stops softmax going NaN.
        expanded[head_start:head_end, padding_start:, :padding_start] = True

    return logits, bbox_unact, expanded, dn_meta
