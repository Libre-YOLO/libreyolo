"""Exact raw-tensor parity against the Apache-2.0 ``effdet`` 0.4.1 reference.

Set ``LIBREYOLO_EFFICIENTDET_UPSTREAM_DIR`` to a directory containing the five
official ``tf_efficientdet_d0`` through ``d4`` release checkpoints. The test is
external-data gated because neither the reference dependency nor checkpoints
belong in the PR gate.
"""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

import pytest
import torch

from libreyolo.models.efficientdet.nn import LibreEfficientDetModel
from libreyolo.postprocess.efficientdet import decode_candidates, generate_anchors

UPSTREAM_DIR = os.environ.get("LIBREYOLO_EFFICIENTDET_UPSTREAM_DIR")

pytestmark = [
    pytest.mark.external_data,
    pytest.mark.skipif(
        not UPSTREAM_DIR,
        reason="set LIBREYOLO_EFFICIENTDET_UPSTREAM_DIR to run exact parity",
    ),
]

CHECKPOINTS = {
    "d0": "tf_efficientdet_d0_34-f153e0cf.pth",
    "d1": "tf_efficientdet_d1_40-a30f94af.pth",
    "d2": "tf_efficientdet_d2_43-8107aa99.pth",
    "d3": "tf_efficientdet_d3_47-0b525f35.pth",
    "d4": "tf_efficientdet_d4_49-f56376d9.pth",
}


@pytest.mark.parametrize("size", tuple(CHECKPOINTS))
def test_effdet_raw_outputs_are_bit_exact(size: str) -> None:
    pytest.importorskip("effdet")
    assert importlib.metadata.version("effdet") == "0.4.1"
    from effdet import get_efficientdet_config
    from effdet.anchors import Anchors, decode_box_outputs
    from effdet.bench import _post_process
    from effdet.efficientdet import EfficientDet

    checkpoint_path = Path(UPSTREAM_DIR) / CHECKPOINTS[size]
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reference = EfficientDet(
        get_efficientdet_config(f"tf_efficientdet_{size}"),
        pretrained_backbone=False,
    ).eval()
    actual = LibreEfficientDetModel(size=size, num_classes=90).eval()
    reference.load_state_dict(state_dict, strict=True)
    actual.load_state_dict(state_dict, strict=True)

    expected_anchors = Anchors.from_config(reference.config).boxes
    actual_anchors = generate_anchors(actual.image_size)
    assert torch.equal(expected_anchors, actual_anchors)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference.to(device)
    actual.to(device)
    torch.manual_seed(20260803)
    resolution = actual.image_size
    inputs = torch.randn(1, 3, resolution, resolution, device=device)

    with torch.inference_mode():
        expected_outputs = reference(inputs)
        actual_outputs = actual(inputs)

    maximum = 0.0
    for expected_branch, actual_branch in zip(expected_outputs, actual_outputs):
        assert len(expected_branch) == len(actual_branch) == 5
        for expected, observed in zip(expected_branch, actual_branch):
            assert expected.shape == observed.shape
            maximum = max(maximum, float((expected - observed).abs().max().item()))
    assert maximum == 0.0, f"EfficientDet {size} max_abs_diff={maximum}"

    reference_logits, reference_regression, indices, classes = _post_process(
        *expected_outputs,
        num_levels=5,
        num_classes=90,
        max_detection_points=5000,
    )
    selected_anchors = expected_anchors.to(device)[indices[0]]
    reference_boxes = decode_box_outputs(
        reference_regression[0].float(), selected_anchors, output_xyxy=True
    )
    reference_candidates = torch.cat(
        (
            reference_boxes,
            reference_logits[0].sigmoid(),
            classes[0].to(reference_boxes.dtype).unsqueeze(-1),
        ),
        dim=-1,
    )
    actual_candidates = decode_candidates(
        actual_outputs,
        input_size=actual.image_size,
        max_candidates=5000,
        sparse_coco=False,
    )[0]
    torch.testing.assert_close(actual_candidates, reference_candidates, rtol=0, atol=0)
