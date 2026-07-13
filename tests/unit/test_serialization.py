"""Tests for torch checkpoint loading helpers."""

import pytest

from libreyolo.utils import serialization

pytestmark = pytest.mark.unit


def test_untrusted_load_uses_weights_only(monkeypatch):
    calls = {}

    monkeypatch.setattr(serialization, "_supports_weights_only", lambda: True)

    def fake_load(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(serialization.torch, "load", fake_load)

    result = serialization.load_untrusted_torch_file("model.pt")

    assert result == {"ok": True}
    assert calls["kwargs"]["weights_only"] is True
    assert calls["kwargs"]["map_location"] == "cpu"


def test_untrusted_load_uses_explicit_safe_globals(monkeypatch):
    calls = {}

    monkeypatch.setattr(serialization, "_supports_weights_only", lambda: True)

    class SafeGlobalsContext:
        def __init__(self, safe_globals):
            calls["safe_globals"] = safe_globals

        def __enter__(self):
            calls["entered"] = True

        def __exit__(self, exc_type, exc, tb):
            calls["exited"] = True

    class SerializationNamespace:
        @staticmethod
        def safe_globals(safe_globals):
            return SafeGlobalsContext(safe_globals)

    def fake_load(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(serialization.torch, "serialization", SerializationNamespace)
    monkeypatch.setattr(serialization.torch, "load", fake_load)

    result = serialization.load_untrusted_torch_file(
        "model.pt",
        safe_globals=(dict,),
    )

    assert result == {"ok": True}
    assert calls["safe_globals"] == [dict]
    assert calls["entered"] is True
    assert calls["exited"] is True
    assert calls["kwargs"]["weights_only"] is True


def test_trusted_load_uses_full_checkpoint_mode(monkeypatch):
    calls = {}

    monkeypatch.setattr(serialization, "_supports_weights_only", lambda: True)

    def fake_load(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(serialization.torch, "load", fake_load)

    result = serialization.load_trusted_torch_file("last.pt", map_location="cuda:0")

    assert result == {"ok": True}
    assert calls["kwargs"]["weights_only"] is False
    assert calls["kwargs"]["map_location"] == "cuda:0"


def test_untrusted_load_requires_modern_torch(monkeypatch):
    monkeypatch.setattr(serialization, "_supports_weights_only", lambda: False)

    with pytest.raises(RuntimeError, match="weights_only"):
        serialization.load_untrusted_torch_file("model.pt")


def test_wrap_libreyolo_checkpoint_emits_required_v1_metadata(monkeypatch):
    monkeypatch.setattr(serialization, "get_libreyolo_version", lambda: "1.2.3")

    state_dict = {"layer.weight": 1}
    checkpoint = serialization.wrap_libreyolo_checkpoint(
        state_dict,
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "cat", 1: "dog"},
        imgsz=640,
    )

    assert checkpoint == {
        "model": state_dict,
        "schema_version": serialization.SCHEMA_VERSION,
        "libreyolo_version": "1.2.3",
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "nc": 2,
        "names": {0: "cat", 1: "dog"},
        "imgsz": 640,
    }


def test_validate_checkpoint_metadata_requires_all_core_fields():
    checkpoint = {
        "model": {"layer.weight": object()},
        "schema_version": serialization.SCHEMA_VERSION,
        "libreyolo_version": "1.2.3",
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "nc": 1,
        "names": {0: "cat"},
    }

    errors = serialization.validate_checkpoint_metadata(checkpoint)

    assert "missing required key: imgsz" in errors
    with pytest.raises(serialization.CheckpointMetadataError, match="imgsz"):
        serialization.validate_checkpoint_metadata(checkpoint, strict=True)


def test_validate_checkpoint_metadata_accepts_string_name_keys_without_mutation():
    checkpoint = {
        "model": {"layer.weight": object()},
        "schema_version": serialization.SCHEMA_VERSION,
        "libreyolo_version": "1.2.3",
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "nc": 2,
        "names": {"0": "cat", "1": "dog"},
        "imgsz": 640,
    }

    assert serialization.validate_checkpoint_metadata(checkpoint) == []
    assert checkpoint["names"] == {"0": "cat", "1": "dog"}


def test_validate_checkpoint_metadata_accepts_point_task():
    checkpoint = serialization.wrap_libreyolo_checkpoint(
        {"layer.weight": object()},
        model_family="fomo",
        size="s",
        task="point",
        nc=1,
        names={0: "person"},
        imgsz=96,
    )

    assert checkpoint["task"] == "point"
    assert serialization.validate_checkpoint_metadata(checkpoint) == []


def test_validate_checkpoint_metadata_pads_missing_name_indices():
    checkpoint = {
        "model": {"layer.weight": object()},
        "schema_version": serialization.SCHEMA_VERSION,
        "libreyolo_version": "1.2.3",
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "nc": 3,
        "names": {0: "cat", 2: "dog"},
        "imgsz": 640,
    }

    errors = serialization.validate_checkpoint_metadata(checkpoint)
    assert any("missing indices [1]" in error for error in errors)
    with pytest.raises(serialization.CheckpointMetadataError, match="missing indices"):
        serialization.validate_checkpoint_metadata(checkpoint, strict=True)
    with pytest.warns(RuntimeWarning, match="padding"):
        assert serialization.normalize_checkpoint_names(checkpoint["names"], 3) == {
            0: "cat",
            1: "class_1",
            2: "dog",
        }
    assert checkpoint["names"] == {0: "cat", 2: "dog"}


def test_wrap_checkpoint_rejects_sparse_writer_names():
    with pytest.raises(serialization.CheckpointMetadataError, match="missing indices"):
        serialization.wrap_libreyolo_checkpoint(
            {"layer.weight": 1},
            model_family="yolo9",
            size="t",
            task="detect",
            nc=3,
            names={0: "cat", 2: "dog"},
            imgsz=640,
        )


def test_load_parser_rejects_sparse_declared_v1_metadata():
    checkpoint = serialization.wrap_libreyolo_checkpoint(
        {"layer.weight": 1},
        model_family="yolo9",
        size="t",
        task="detect",
        nc=3,
        names={0: "cat", 1: "bird", 2: "dog"},
        imgsz=640,
    )
    checkpoint["names"] = {0: "cat", 2: "dog"}

    with pytest.raises(serialization.CheckpointMetadataError, match="missing indices"):
        serialization.parse_checkpoint_metadata_for_load(checkpoint)


def test_load_parser_warns_and_pads_sparse_legacy_metadata():
    checkpoint = {
        "state_dict": {"layer.weight": 1},
        "task": "detect",
        "nc": 3,
        "names": {0: "cat", 2: "dog"},
    }

    with pytest.warns(RuntimeWarning) as caught:
        parsed, is_native_v1 = serialization.parse_checkpoint_metadata_for_load(
            checkpoint,
            context="legacy unit checkpoint",
        )

    assert is_native_v1 is False
    assert parsed["names"] == {0: "cat", 1: "class_1", 2: "dog"}
    assert any("legacy or incomplete metadata" in str(item.message) for item in caught)
    assert any("padding" in str(item.message) for item in caught)


def test_load_parser_routes_version_only_checkpoint_through_legacy_compatibility():
    checkpoint = {
        "model": {"layer.weight": 1},
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "nc": 1,
        "names": {0: "cat"},
        "imgsz": 640,
        "libreyolo_version": "0.1.0",
    }

    with pytest.warns(RuntimeWarning, match="legacy or incomplete metadata"):
        parsed, is_native_v1 = serialization.parse_checkpoint_metadata_for_load(
            checkpoint,
            context="incomplete unit checkpoint",
        )

    assert is_native_v1 is False
    assert parsed["names"] == {0: "cat"}


def test_load_parser_rejects_schema_only_checkpoint_as_malformed_v1():
    checkpoint = {
        "model": {"layer.weight": 1},
        "schema_version": serialization.SCHEMA_VERSION,
    }

    with pytest.raises(serialization.CheckpointMetadataError, match="missing required"):
        serialization.parse_checkpoint_metadata_for_load(checkpoint)


def test_load_parser_rejects_explicit_unsupported_schema_version():
    checkpoint = serialization.wrap_libreyolo_checkpoint(
        {"layer.weight": 1},
        model_family="yolo9",
        size="t",
        task="detect",
        nc=1,
        names={0: "cat"},
        imgsz=640,
    )
    checkpoint["schema_version"] = "2.0"

    with pytest.raises(serialization.CheckpointMetadataError, match="schema_version"):
        serialization.parse_checkpoint_metadata_for_load(checkpoint)


def test_validate_native_checkpoint_rejects_non_string_name_values():
    checkpoint = serialization.wrap_libreyolo_checkpoint(
        {"layer.weight": 1},
        model_family="yolo9",
        size="t",
        task="detect",
        nc=1,
        names={0: "cat"},
        imgsz=640,
    )
    checkpoint["names"] = {0: 123}

    with pytest.raises(serialization.CheckpointMetadataError, match="must be a string"):
        serialization.validate_checkpoint_metadata(checkpoint, strict=True)


def test_wrap_checkpoint_rejects_required_field_in_extra_metadata():
    with pytest.raises(
        serialization.CheckpointMetadataError,
        match="cannot override required fields: model",
    ):
        serialization.wrap_libreyolo_checkpoint(
            {"layer.weight": 1},
            model_family="yolo9",
            size="t",
            task="detect",
            nc=1,
            names={0: "cat"},
            imgsz=640,
            **{"model": {"other.weight": 2}},
        )


def test_validate_checkpoint_metadata_rejects_out_of_range_names():
    checkpoint = {
        "model": {"layer.weight": object()},
        "schema_version": serialization.SCHEMA_VERSION,
        "libreyolo_version": "1.2.3",
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "nc": 2,
        "names": {0: "cat", 2: "dog"},
        "imgsz": 640,
    }

    with pytest.raises(serialization.CheckpointMetadataError, match="out-of-range"):
        serialization.validate_checkpoint_metadata(checkpoint, strict=True)


def test_warn_on_metadata_schema_version_logs_legacy_metadata(caplog):
    import logging

    logger = logging.getLogger("test-schema")

    with caplog.at_level(logging.WARNING, logger="test-schema"):
        serialization.warn_on_metadata_schema_version(
            {"model_family": "yolo9"},
            artifact="test export",
            logger=logger,
        )

    assert "has no schema_version" in caplog.text


def test_wrap_checkpoint_does_not_fall_back_to_default_size_for_empty_task_map(
    monkeypatch,
):
    from libreyolo.models.base import BaseModel

    class DummyFamily:
        FAMILY = "dummy"
        INPUT_SIZES = {"s": 640}
        TASK_INPUT_SIZES = {"pose": {}}

    monkeypatch.setattr(BaseModel, "_registry", [DummyFamily])

    with pytest.raises(serialization.CheckpointMetadataError, match="imgsz"):
        serialization.wrap_libreyolo_checkpoint(
            {"layer.weight": object()},
            model_family="dummy",
            size="s",
            task="pose",
            nc=1,
            names={0: "person"},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_family", "not-a-family", "public model manifest"),
        ("size", "not-a-size", "declared model artifact"),
        ("task", "semantic", "declared model artifact"),
    ],
)
def test_native_metadata_requires_declared_family_size_task_identity(
    field, value, message
):
    checkpoint = serialization.wrap_libreyolo_checkpoint(
        {"layer.weight": object()},
        model_family="yolo9",
        size="t",
        task="detect",
        nc=1,
        names={0: "object"},
        imgsz=640,
    )
    checkpoint[field] = value

    with pytest.raises(serialization.CheckpointMetadataError, match=message):
        serialization.validate_checkpoint_metadata(checkpoint, strict=True)


def test_checkpoint_wrapper_writes_canonical_identity_values():
    checkpoint = serialization.wrap_libreyolo_checkpoint(
        {"layer.weight": object()},
        model_family=" YOLO9 ",
        size="T",
        task="DETECTION",
        nc=1,
        names={0: "object"},
        imgsz=640,
    )

    assert checkpoint["model_family"] == "yolo9"
    assert checkpoint["size"] == "t"
    assert checkpoint["task"] == "detect"


@pytest.mark.parametrize("imgsz", [True, 640.5, "640", 0, -1])
def test_checkpoint_wrapper_rejects_lossy_or_invalid_imgsz(imgsz):
    with pytest.raises(serialization.CheckpointMetadataError, match="positive int"):
        serialization.wrap_libreyolo_checkpoint(
            {"layer.weight": object()},
            model_family="yolo9",
            size="t",
            task="detect",
            nc=1,
            names={0: "object"},
            imgsz=imgsz,
        )


def test_checkpoint_writer_and_validator_enforce_family_imgsz_contract():
    kwargs = {
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "nc": 1,
        "names": {0: "object"},
        "imgsz": 641,
    }
    with pytest.raises(serialization.CheckpointMetadataError, match="divisible by 32"):
        serialization.wrap_libreyolo_checkpoint(
            {"layer.weight": object()},
            **kwargs,
        )

    checkpoint = serialization.wrap_libreyolo_checkpoint(
        {"layer.weight": object()},
        **{**kwargs, "imgsz": 640},
    )
    checkpoint["imgsz"] = 641
    with pytest.raises(serialization.CheckpointMetadataError, match="divisible by 32"):
        serialization.validate_checkpoint_metadata(checkpoint, strict=True)


def test_checkpoint_writer_uses_rfdetr_patch_window_contract():
    common = {
        "model_family": "rfdetr",
        "size": "n",
        "task": "detect",
        "nc": 1,
        "names": {0: "object"},
    }
    valid = serialization.wrap_libreyolo_checkpoint(
        {"layer.weight": object()},
        imgsz=384,
        **common,
    )
    assert valid["imgsz"] == 384

    with pytest.raises(serialization.CheckpointMetadataError, match="divisible by 32"):
        serialization.wrap_libreyolo_checkpoint(
            {"layer.weight": object()},
            imgsz=641,
            **common,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_family", "YOLO9"),
        ("size", "T"),
        ("task", "DETECTION"),
    ],
)
def test_native_metadata_rejects_noncanonical_identity_values(field, value):
    checkpoint = serialization.wrap_libreyolo_checkpoint(
        {"layer.weight": object()},
        model_family="yolo9",
        size="t",
        task="detect",
        nc=1,
        names={0: "object"},
        imgsz=640,
    )
    checkpoint[field] = value

    with pytest.raises(serialization.CheckpointMetadataError, match="canonical"):
        serialization.validate_checkpoint_metadata(checkpoint, strict=True)


def test_checkpoint_imgsz_inference_is_independent_of_runtime_registry(monkeypatch):
    from libreyolo.models.base import BaseModel

    monkeypatch.setattr(BaseModel, "_registry", [])

    checkpoint = serialization.wrap_libreyolo_checkpoint(
        {"layer.weight": object()},
        model_family="dinov2",
        size="n",
        task="semantic",
        nc=1,
        names={0: "foreground"},
    )

    assert checkpoint["imgsz"] == 518
