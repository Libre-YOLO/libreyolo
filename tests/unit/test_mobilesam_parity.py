"""Gated MobileSAM parity against an upstream Apache-2.0 checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.mobilesam.model import MobileSAMNetwork
from libreyolo.models.mobilesam.preprocess import (
    encode_image_and_prompts,
    postprocess_masks,
    preprocess_tensor,
)

pytestmark = [pytest.mark.unit, pytest.mark.sam]


def _load_state_dict(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _upstream_model(upstream_root: Path, checkpoint: Path):
    sys.path.insert(0, str(upstream_root))
    try:
        from mobile_sam import sam_model_registry

        return sam_model_registry["vit_t"](checkpoint=str(checkpoint)).eval()
    finally:
        try:
            sys.path.remove(str(upstream_root))
        except ValueError:
            pass


def _upstream_resize_cls(upstream_root: Path):
    sys.path.insert(0, str(upstream_root))
    try:
        from mobile_sam.utils.transforms import ResizeLongestSide

        return ResizeLongestSide
    finally:
        try:
            sys.path.remove(str(upstream_root))
        except ValueError:
            pass


def _ours(checkpoint: Path):
    model = MobileSAMNetwork()
    model.load_state_dict(_load_state_dict(checkpoint), strict=True)
    return model.eval()


def _assert_exact(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape
    max_abs_diff = (a - b).abs().max().item()
    assert max_abs_diff == 0


@pytest.mark.skipif(
    not os.getenv("LIBREYOLO_MOBILESAM_UPSTREAM")
    or not os.getenv("LIBREYOLO_MOBILESAM_CHECKPOINT"),
    reason=(
        "set LIBREYOLO_MOBILESAM_UPSTREAM and "
        "LIBREYOLO_MOBILESAM_CHECKPOINT to run MobileSAM parity"
    ),
)
def test_mobilesam_upstream_parity():
    upstream_root = Path(os.environ["LIBREYOLO_MOBILESAM_UPSTREAM"])
    checkpoint = Path(os.environ["LIBREYOLO_MOBILESAM_CHECKPOINT"])
    upstream = _upstream_model(upstream_root, checkpoint)
    upstream_resize_cls = _upstream_resize_cls(upstream_root)
    ours = _ours(checkpoint)

    image = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32)
    with torch.no_grad():
        upstream_embedding = upstream.image_encoder(image)
        our_embedding = ours.image_encoder(image)
    _assert_exact(upstream_embedding, our_embedding)

    point_coords = torch.tensor([[[512.0, 512.0]]])
    point_labels = torch.tensor([[1]])
    box = torch.tensor([[128.0, 128.0, 640.0, 640.0]])

    for points, boxes in [((point_coords, point_labels), None), (None, box)]:
        with torch.no_grad():
            up_sparse, up_dense = upstream.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=None,
            )
            our_sparse, our_dense = ours.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=None,
            )
            _assert_exact(up_sparse, our_sparse)
            _assert_exact(up_dense, our_dense)

            up_masks, up_iou = upstream.mask_decoder(
                image_embeddings=upstream_embedding,
                image_pe=upstream.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=up_sparse,
                dense_prompt_embeddings=up_dense,
                multimask_output=True,
            )
            our_masks, our_iou = ours.mask_decoder(
                image_embeddings=our_embedding,
                image_pe=ours.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=our_sparse,
                dense_prompt_embeddings=our_dense,
                multimask_output=True,
            )
        _assert_exact(up_masks, our_masks)
        _assert_exact(up_iou, our_iou)

    raw_image = np.arange(31 * 37 * 3, dtype=np.uint8).reshape(31, 37, 3)
    pil_image = Image.fromarray(raw_image, mode="RGB")
    enc = encode_image_and_prompts(
        pil_image,
        target_length=1024,
        points=[[[18.0, 15.0]]],
        labels=[[1]],
        boxes=[[4.0, 5.0, 30.0, 25.0]],
    )
    upstream_transform = upstream_resize_cls(1024)
    upstream_resized = torch.as_tensor(upstream_transform.apply_image(raw_image))
    our_resized = enc["pixel_values"][0].permute(1, 2, 0).to(torch.uint8)
    assert torch.equal(our_resized, upstream_resized)

    original_size = (31, 37)
    upstream_points = torch.as_tensor(
        upstream_transform.apply_coords(
            np.asarray([[[18.0, 15.0]]], dtype=np.float32),
            original_size,
        ),
        dtype=torch.float32,
    ).unsqueeze(0)
    upstream_boxes = torch.as_tensor(
        upstream_transform.apply_boxes(
            np.asarray([[4.0, 5.0, 30.0, 25.0]], dtype=np.float32),
            original_size,
        ),
        dtype=torch.float32,
    ).unsqueeze(0)
    _assert_exact(enc["input_points"], upstream_points)
    _assert_exact(enc["input_boxes"], upstream_boxes)

    our_preprocessed = preprocess_tensor(
        enc["pixel_values"],
        image_size=1024,
        pixel_mean=ours.pixel_mean,
        pixel_std=ours.pixel_std,
    )[0]
    upstream_preprocessed = upstream.preprocess(enc["pixel_values"][0])
    _assert_exact(our_preprocessed, upstream_preprocessed)

    low_res_masks = torch.linspace(0, 1, 3 * 256 * 256, dtype=torch.float32).view(
        1, 3, 256, 256
    )
    input_size = tuple(int(v) for v in enc["reshaped_input_sizes"][0])
    our_postprocessed = postprocess_masks(
        low_res_masks,
        image_size=1024,
        input_size=input_size,
        original_size=original_size,
        mask_threshold=upstream.mask_threshold,
    )
    upstream_postprocessed = (
        upstream.postprocess_masks(
            low_res_masks,
            input_size=input_size,
            original_size=original_size,
        )
        > upstream.mask_threshold
    )
    assert torch.equal(our_postprocessed, upstream_postprocessed)
