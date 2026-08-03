"""Tests for HRNet upstream checkpoint normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
if str(WEIGHTS_DIR) not in sys.path:
    sys.path.insert(0, str(WEIGHTS_DIR))

import convert_hrnet_weights  # noqa: E402


def test_original_layout_preserves_tensor_keys_and_objects():
    weight = torch.randn(2, 3)

    normalized = convert_hrnet_weights.normalize_state_dict(
        {"conv1.weight": weight},
        "original",
    )

    assert normalized == {"conv1.weight": weight}
    assert normalized["conv1.weight"] is weight


def test_mmpose_layout_remaps_backbone_and_heatmap_head():
    backbone = torch.randn(2, 3)
    head = torch.randn(17, 32, 1, 1)
    raw = {
        "state_dict": {
            "module.backbone.conv1.weight": backbone,
            "module.head.final_layer.weight": head,
            "module.data_preprocessor.mean": torch.zeros(3),
        }
    }

    normalized = convert_hrnet_weights.normalize_state_dict(raw, "mmpose")

    assert set(normalized) == {"conv1.weight", "final_layer.weight"}
    assert normalized["conv1.weight"] is backbone
    assert normalized["final_layer.weight"] is head


def test_checkpoint_payload_must_only_contain_tensors():
    with pytest.raises(TypeError, match="string keys to tensors"):
        convert_hrnet_weights.normalize_state_dict(
            {"conv1.weight": torch.zeros(1), "epoch": 3},
            "original",
        )
