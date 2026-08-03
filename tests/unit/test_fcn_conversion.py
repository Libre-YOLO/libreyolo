"""Focused failure-path tests for the FCN checkpoint converter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from libreyolo.models.autoconvert import autoconvert_upstream_checkpoint

pytestmark = pytest.mark.unit

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
if str(WEIGHTS_DIR) not in sys.path:
    sys.path.insert(0, str(WEIGHTS_DIR))

from convert_fcn_weights import convert_weights  # noqa: E402


def _recognizable_state(depth: int, classes: int = 21) -> dict[str, torch.Tensor]:
    last_block = 5 if depth == 50 else 22
    return {
        "backbone.conv1.weight": torch.empty(1),
        "backbone.layer4.0.conv2.weight": torch.empty(1),
        f"backbone.layer3.{last_block}.conv3.weight": torch.empty(1),
        "classifier.0.weight": torch.empty(1),
        "classifier.1.running_mean": torch.empty(1),
        "classifier.4.weight": torch.empty(classes, 1),
        "aux_classifier.0.weight": torch.empty(1),
        "aux_classifier.4.weight": torch.empty(classes, 1),
    }


def test_converter_rejects_wrong_declared_backbone_size(tmp_path):
    source = tmp_path / "fcn.pth"
    torch.save(_recognizable_state(50), source)

    with pytest.raises(ValueError, match="detected 'r50'"):
        convert_weights(str(source), str(tmp_path / "converted.pt"), size="r101")


def test_converter_rejects_noncanonical_head_width(tmp_path):
    source = tmp_path / "fcn.pth"
    torch.save(_recognizable_state(50, classes=7), source)

    with pytest.raises(ValueError, match="21-class"):
        convert_weights(str(source), str(tmp_path / "converted.pt"), size="r50")


def test_bare_upstream_checkpoint_autoconverts_with_voc_names(tmp_path):
    source = tmp_path / "fcn_resnet50_coco.pth"
    torch.save(_recognizable_state(50), source)

    converted = autoconvert_upstream_checkpoint(str(source))

    assert converted is not None
    assert Path(converted).name == "fcn_resnet50_coco-LibreFCNr50-sem.pt"
    checkpoint = torch.load(converted, map_location="cpu", weights_only=True)
    assert checkpoint["model_family"] == "fcn"
    assert checkpoint["size"] == "r50"
    assert checkpoint["task"] == "semantic"
    assert checkpoint["nc"] == 21
    assert checkpoint["names"][0] == "__background__"
    assert checkpoint["names"][20] == "tvmonitor"
