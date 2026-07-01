"""Unit tests for the native RF-DETR-SO (small-object) family."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def so_model():
    from libreyolo import LibreRFDETRSO

    return LibreRFDETRSO(model_path={}, size="s", device="cpu")


@pytest.fixture(scope="module")
def stock_state_dict():
    from libreyolo import LibreRFDETR

    stock = LibreRFDETR(model_path={}, size="s", device="cpu")
    return stock.model.state_dict()


def test_rfdetr_so_is_registered_and_detects_filename():
    from libreyolo import LibreRFDETRSO
    from libreyolo.models.base.model import BaseModel

    assert any(cls.__name__ == "LibreRFDETRSO" for cls in BaseModel._registry)
    assert LibreRFDETRSO.FAMILY == "rfdetr_so"
    assert LibreRFDETRSO.SUPPORTED_TASKS == ("detect",)
    assert LibreRFDETRSO.detect_size_from_filename("LibreRFDETRSOs.pt") == "s"


def test_rfdetr_so_filename_does_not_match_base_family():
    from libreyolo import LibreRFDETR, LibreRFDETRSO

    assert LibreRFDETR.detect_size_from_filename("LibreRFDETRSOs.pt") is None
    assert LibreRFDETRSO.detect_size_from_filename("LibreRFDETRs.pt") is None


def test_rfdetr_so_backbone_produces_three_levels(so_model):
    """The SO backbone must emit a stride-8/16/32 pyramid (64/32/16 at 512)."""
    from libreyolo.models.rfdetr.tensors import NestedTensor

    backbone = so_model.model.model.backbone[0]
    images = torch.zeros(1, 3, 512, 512)
    mask = torch.zeros(1, 512, 512, dtype=torch.bool)
    with torch.no_grad():
        feats = backbone(NestedTensor(images, mask))

    shapes = [tuple(f.tensors.shape) for f in feats]
    assert shapes == [
        (1, 256, 64, 64),
        (1, 256, 32, 32),
        (1, 256, 16, 16),
    ]


def test_rfdetr_so_eval_forward_shapes(so_model):
    """Full model forward returns the standard LW-DETR output dict."""
    so_model.model.eval()
    with torch.no_grad():
        out = so_model.model(torch.zeros(1, 3, 512, 512))

    assert "pred_logits" in out and "pred_boxes" in out
    assert out["pred_boxes"].shape[-1] == 4
    assert out["pred_logits"].shape[1] == out["pred_boxes"].shape[1]


def test_rfdetr_so_encoder_frozen_by_default(so_model):
    backbone = so_model.model.model.backbone[0]
    assert all(not p.requires_grad for p in backbone.encoder.parameters())
    assert any(p.requires_grad for p in backbone.ssa_sde.parameters())
    assert any(p.requires_grad for p in backbone.projector.parameters())


def test_rfdetr_so_can_load_discriminators(so_model, stock_state_dict):
    from libreyolo import LibreRFDETR, LibreRFDETRSO

    so_sd = so_model.model.state_dict()
    assert LibreRFDETRSO.can_load(so_sd) is True
    assert LibreRFDETRSO.can_load(stock_state_dict) is False
    assert LibreRFDETR.can_load(so_sd) is False
    assert LibreRFDETR.can_load(stock_state_dict) is True


def test_rfdetr_so_transfer_remap_from_stock(so_model, stock_state_dict):
    """Every stock RF-DETR-S tensor must land in the SO model: the projector
    P4 stage shifts to index 1, deformable cross-attention expands its level
    axis 1 -> 3, everything else loads 1:1. Missing keys are exactly the new
    small-object modules plus the new projector scale stages."""
    from libreyolo.models.rfdetr_so.nn import remap_stock_state_for_so

    remapped = remap_stock_state_for_so(
        dict(stock_state_dict),
        ca_nheads=so_model.model.args.ca_nheads,
        dec_n_points=so_model.model.args.dec_n_points,
        num_levels=3,
    )
    current = so_model.model.model.state_dict()

    mismatched = [
        key
        for key, value in remapped.items()
        if key not in current or current[key].shape != value.shape
    ]
    assert mismatched == []

    # Projector stage remap: P4 tensors now live at index 1, index 0 is new.
    assert any(k.startswith("backbone.0.projector.stages.1.") for k in remapped)
    assert not any(k.startswith("backbone.0.projector.stages.0.") for k in remapped)

    # Level-axis expansion happened.
    offsets_key = next(
        k for k in remapped if k.endswith("cross_attn.sampling_offsets.weight")
    )
    heads = so_model.model.args.ca_nheads
    points = so_model.model.args.dec_n_points
    assert remapped[offsets_key].shape[0] == heads * 3 * points * 2

    # Fresh modules receive nothing from the checkpoint.
    leftover = set(current) - set(remapped)
    assert any(k.startswith("backbone.0.ssa_sde.") for k in leftover)
    assert any(k.startswith("backbone.0.pbm3.") for k in leftover)
    assert any(k.startswith("backbone.0.projector.stages.0.") for k in leftover)
    assert any(k.startswith("backbone.0.projector.stages.2.") for k in leftover)


def test_rfdetr_so_load_state_dict_accepts_stock_checkpoint(
    so_model, stock_state_dict
):
    """End-to-end loader contract: a stock checkpoint loads with strict=False,
    no unexpected keys, and missing keys limited to the new modules."""
    missing, unexpected = so_model.model.load_state_dict(
        {"model": dict(stock_state_dict)}, strict=False
    )
    assert unexpected == []
    allowed_prefixes = (
        "backbone.0.ssa_sde.",
        "backbone.0.ssa_fuse.",
        "backbone.0.pbm3.",
        "backbone.0.pbm4.",
        "backbone.0.projector.stages.0.",
        "backbone.0.projector.stages.2.",
        "backbone.0.projector.stages_sampling.",
    )
    outside = [k for k in missing if not k.startswith(allowed_prefixes)]
    assert outside == []


def test_rfdetr_so_own_checkpoint_roundtrip(so_model):
    """SO checkpoints must load back without any remap side effects."""
    sd = so_model.model.state_dict()
    missing, unexpected = so_model.model.load_state_dict(
        {"model": dict(sd)}, strict=False
    )
    assert missing == []
    assert unexpected == []


def test_rfdetr_so_trainer_metadata():
    from libreyolo.models.rfdetr_so.config import RFDETRSOConfig
    from libreyolo.models.rfdetr_so.trainer import RFDETRSOTrainer

    assert RFDETRSOTrainer._config_class() is RFDETRSOConfig
    assert RFDETRSOTrainer.artifact_model_families == ("rfdetr_so",)
    cfg = RFDETRSOConfig(size="s")
    assert cfg.warmup_epochs == 1
    assert cfg.mosaic_prob == 0.0
