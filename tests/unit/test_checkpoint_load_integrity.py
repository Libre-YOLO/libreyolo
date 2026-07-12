"""Regression tests for checkpoint tensor-coverage contracts."""

import pytest
import torch

from libreyolo.models.yolox.model import LibreYOLOX
from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.utils.serialization import (
    LEGACY_CHECKPOINT_LOAD_POLICY,
    NATIVE_CHECKPOINT_LOAD_POLICY,
    CheckpointLoadError,
    load_state_dict_checked,
    wrap_libreyolo_checkpoint,
)

pytestmark = pytest.mark.unit


def test_legacy_two_tensor_yolox_checkpoint_is_rejected():
    source = LibreYOLOX(
        model_path=None,
        size="n",
        nb_classes=2,
        device="cpu",
    )
    partial = {
        key: value.detach().clone()
        for key, value in list(source.model.state_dict().items())[:2]
    }

    with pytest.raises(CheckpointLoadError, match="coverage"):
        LibreYOLOX(
            model_path=partial,
            size="n",
            nb_classes=2,
            device="cpu",
        )


def test_complete_native_checkpoint_loads_exact_and_normalizes_names():
    source = LibreYOLOX(
        model_path=None,
        size="n",
        nb_classes=2,
        device="cpu",
    )
    checkpoint = wrap_libreyolo_checkpoint(
        source.model.state_dict(),
        model_family="yolox",
        size="n",
        task="detect",
        nc=2,
        names={0: "cat", 1: "dog"},
        imgsz=416,
    )
    checkpoint["names"] = {"0": "cat", "1": "dog"}

    loaded = LibreYOLOX(
        model_path=checkpoint,
        size="n",
        nb_classes=2,
        device="cpu",
    )

    assert loaded.names == {0: "cat", 1: "dog"}
    assert loaded.model.training is False


def test_native_checkpoint_missing_one_tensor_is_rejected():
    source = LibreYOLOX(
        model_path=None,
        size="n",
        nb_classes=2,
        device="cpu",
    )
    state = dict(source.model.state_dict())
    state.pop(next(iter(state)))
    checkpoint = wrap_libreyolo_checkpoint(
        state,
        model_family="yolox",
        size="n",
        task="detect",
        nc=2,
        names=["cat", "dog"],
        imgsz=416,
    )

    with pytest.raises(CheckpointLoadError, match="required model tensors"):
        LibreYOLOX(
            model_path=checkpoint,
            size="n",
            nb_classes=2,
            device="cpu",
        )


def test_native_checkpoint_rejects_ddp_alias_of_existing_tensor():
    source = LibreYOLOX(
        model_path=None,
        size="n",
        nb_classes=2,
        device="cpu",
    )
    state = dict(source.model.state_dict())
    first_key = next(iter(state))
    state[f"module.{first_key}"] = torch.full_like(state[first_key], 123)
    checkpoint = wrap_libreyolo_checkpoint(
        state,
        model_family="yolox",
        size="n",
        task="detect",
        nc=2,
        names={0: "cat", 1: "dog"},
        imgsz=416,
    )

    with pytest.raises(CheckpointLoadError, match="unexpected checkpoint keys"):
        LibreYOLOX(
            model_path=checkpoint,
            size="n",
            nb_classes=2,
            device="cpu",
        )


def test_legacy_checkpoint_rejects_ddp_normalization_collision():
    source = LibreYOLOX(
        model_path=None,
        size="n",
        nb_classes=2,
        device="cpu",
    )
    state = dict(source.model.state_dict())
    first_key = next(iter(state))
    state[f"module.{first_key}"] = torch.full_like(state[first_key], 123)

    with pytest.raises(ValueError, match="normalization collision"):
        LibreYOLOX(
            model_path=state,
            size="n",
            nb_classes=2,
            device="cpu",
        )


def test_yolo9_e2e_rejects_legacy_head_normalization_collision():
    from libreyolo.models.yolo9_e2e.model import LibreYOLO9E2E

    wrapper = object.__new__(LibreYOLO9E2E)
    state = {
        "head.cv2.0.0.conv.weight": torch.zeros(1),
        "detect.cv2.0.0.conv.weight": torch.ones(1),
    }

    with pytest.raises(ValueError, match="normalization collision"):
        wrapper._prepare_state_dict(state)


def test_explicit_transfer_allowlist_excludes_only_named_head():
    module = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Linear(4, 2),
    )
    state = dict(module.state_dict())
    state.pop("1.weight")
    state.pop("1.bias")
    policy = NATIVE_CHECKPOINT_LOAD_POLICY.allowing(
        name="unit-head-transfer",
        missing=("1.*",),
    )

    report = load_state_dict_checked(
        module,
        state,
        policy=policy,
        context="unit transfer",
    )

    assert report.key_coverage == 1.0
    assert report.allowed_missing_keys == ("1.bias", "1.weight")


def test_legacy_policy_rejects_unexpected_tensors_even_with_high_coverage():
    module = torch.nn.Linear(4, 2)
    state = dict(module.state_dict())
    state["foreign.weight"] = torch.ones(1)

    with pytest.raises(CheckpointLoadError, match="unexpected checkpoint keys"):
        load_state_dict_checked(
            module,
            state,
            policy=LEGACY_CHECKPOINT_LOAD_POLICY,
            context="legacy unit checkpoint",
        )


def test_yolo9_training_transfer_rejects_two_tensor_donor(tmp_path):
    source = LibreYOLO9(
        model_path=None,
        size="t",
        nb_classes=80,
        device="cpu",
    )
    donor = {
        key: value.detach().clone()
        for key, value in list(source.model.state_dict().items())[:2]
    }
    path = tmp_path / "partial.pt"
    torch.save(donor, path)
    target = LibreYOLO9(
        model_path=None,
        size="t",
        nb_classes=3,
        device="cpu",
    )

    with pytest.raises(CheckpointLoadError, match="required model tensors"):
        target._load_transfer_weights(path)


def test_yolo9_training_transfer_allows_only_class_head_shape_drift(tmp_path):
    source = LibreYOLO9(
        model_path=None,
        size="t",
        nb_classes=80,
        device="cpu",
    )
    checkpoint = wrap_libreyolo_checkpoint(
        source.model.state_dict(),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=80,
        imgsz=640,
    )
    path = tmp_path / "source.pt"
    torch.save(checkpoint, path)
    target = LibreYOLO9(
        model_path=None,
        size="t",
        nb_classes=3,
        device="cpu",
    )

    stats = target._load_transfer_weights(path)

    assert stats["loaded"] > 1000
    assert 0 < stats["skipped"] < 20


def test_rfdetr_pretrained_backbone_rejects_negligible_transfer(monkeypatch):
    from transformers import AutoModel

    from libreyolo.models.rfdetr.backbone import DinoV2

    class _Reference:
        @staticmethod
        def state_dict():
            return {"0.weight": torch.ones(4, 4)}

    monkeypatch.setattr(
        AutoModel,
        "from_pretrained",
        lambda name: _Reference(),
    )
    backbone = object.__new__(DinoV2)
    torch.nn.Module.__init__(backbone)
    backbone.encoder = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Linear(4, 4),
    )

    with pytest.raises(CheckpointLoadError, match="coverage"):
        backbone._load_pretrained_dinov2("unit-reference")


def test_rfdetr_native_checkpoint_rejects_alias_and_query_shape_adaptation():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    source = LibreRFDETR(model_path={}, size="n", device="cpu")
    state = dict(source.model.state_dict())
    names = {index: f"class_{index}" for index in range(80)}

    first_key = next(iter(state))
    aliased = dict(state)
    aliased[f"model.{first_key}"] = torch.full_like(state[first_key], 123)
    alias_checkpoint = wrap_libreyolo_checkpoint(
        aliased,
        model_family="rfdetr",
        size="n",
        task="detect",
        nc=80,
        names=names,
        imgsz=384,
    )
    with pytest.raises(RuntimeError, match="unexpected checkpoint keys"):
        LibreRFDETR(
            model_path=alias_checkpoint,
            size="n",
            device="cpu",
        )

    truncated = dict(state)
    query_key = "refpoint_embed.weight"
    truncated[query_key] = truncated[query_key][:100].clone()
    shape_checkpoint = wrap_libreyolo_checkpoint(
        truncated,
        model_family="rfdetr",
        size="n",
        task="detect",
        nc=80,
        names=names,
        imgsz=384,
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        LibreRFDETR(
            model_path=shape_checkpoint,
            size="n",
            device="cpu",
        )

    mismatch_checkpoint = wrap_libreyolo_checkpoint(
        state,
        model_family="rfdetr",
        size="n",
        task="detect",
        nc=79,
        names={index: f"class_{index}" for index in range(79)},
        imgsz=384,
    )
    with pytest.raises(RuntimeError, match="metadata declares nc=79"):
        LibreRFDETR(
            model_path=mismatch_checkpoint,
            size="n",
            device="cpu",
        )
