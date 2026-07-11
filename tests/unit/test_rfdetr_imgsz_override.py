"""Unit tests for RF-DETR imgsz handling in train() and create_transforms()."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

rfdetr_model = pytest.importorskip("libreyolo.models.rfdetr.model")
rfdetr_trainer = pytest.importorskip("libreyolo.models.rfdetr.trainer")
from libreyolo.models.rfdetr.imgsz import validate_imgsz  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wrapper(input_size: int = 504) -> rfdetr_model.LibreRFDETR:
    wrapper = rfdetr_model.LibreRFDETR.__new__(rfdetr_model.LibreRFDETR)

    class _FakeModel:
        patch_size = 6
        num_windows = 4  # block_size = 24

    wrapper.model = _FakeModel()
    wrapper.size = "l"
    wrapper.nb_classes = 80
    wrapper.input_size = input_size
    wrapper.task = "detect"
    return wrapper


def _make_trainer(imgsz: int, *, multi_scale: bool = False) -> rfdetr_trainer.RFDETRTrainer:
    """Construct a RFDETRTrainer stub without calling __init__."""
    trainer = rfdetr_trainer.RFDETRTrainer.__new__(rfdetr_trainer.RFDETRTrainer)
    trainer.config = rfdetr_trainer.RFDETRConfig(
        data=None,
        imgsz=imgsz,
        multi_scale=multi_scale,
        do_random_resize_via_padding=False,
    )

    class _FakeModel:
        patch_size = 6
        num_windows = 4  # block_size = 24

    class _FakeWrapper:
        task = "detect"

    trainer.model = _FakeModel()
    trainer.wrapper_model = _FakeWrapper()
    return trainer


# ---------------------------------------------------------------------------
# train() kwarg assembly
# ---------------------------------------------------------------------------

def test_explicit_imgsz_is_not_overridden(monkeypatch, tmp_path):
    """imgsz=624 supplied to train() must not be overwritten by model.input_size=504."""
    captured = {}

    class _DummyTrainer:
        def __init__(self, model, wrapper_model=None, **kwargs):
            captured["kwargs"] = kwargs

        def train(self):
            return {"save_dir": str(tmp_path / "exp")}

    monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", _DummyTrainer)

    wrapper = _make_wrapper(input_size=504)
    wrapper.train(data="data.yaml", imgsz=624, output_dir=str(tmp_path / "run"))

    assert captured["kwargs"]["imgsz"] == 624


def test_default_imgsz_applied_when_not_supplied(monkeypatch, tmp_path):
    """When imgsz is not passed, model.input_size is used as the fallback."""
    captured = {}

    class _DummyTrainer:
        def __init__(self, model, wrapper_model=None, **kwargs):
            captured["kwargs"] = kwargs

        def train(self):
            return {"save_dir": str(tmp_path / "exp")}

    monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", _DummyTrainer)

    wrapper = _make_wrapper(input_size=504)
    wrapper.train(data="data.yaml", output_dir=str(tmp_path / "run"))

    assert captured["kwargs"]["imgsz"] == 504


def test_imgsz_none_falls_back_to_model_default(monkeypatch, tmp_path):
    """imgsz=None must not be forwarded; model.input_size is used instead."""
    captured = {}

    class _DummyTrainer:
        def __init__(self, model, wrapper_model=None, **kwargs):
            captured["kwargs"] = kwargs

        def train(self):
            return {"save_dir": str(tmp_path / "exp")}

    monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", _DummyTrainer)

    wrapper = _make_wrapper(input_size=504)
    wrapper.train(data="data.yaml", imgsz=None, output_dir=str(tmp_path / "run"))

    assert captured["kwargs"]["imgsz"] == 504


def test_train_rejects_invalid_explicit_imgsz():
    """train(imgsz=500) fails before constructing the trainer."""
    wrapper = _make_wrapper(input_size=504)
    with pytest.raises(ValueError, match="RF-DETR train imgsz=500 is not divisible by 24"):
        wrapper.train(data="data.yaml", imgsz=500)


def test_inference_rejects_invalid_imgsz_before_loading_image():
    """predict/preprocess paths fail early with the shared RF-DETR message."""
    wrapper = _make_wrapper(input_size=504)
    with pytest.raises(
        ValueError,
        match="RF-DETR inference imgsz=500 is not divisible by 24",
    ):
        wrapper._preprocess("missing.jpg", input_size=500)


def test_val_preprocessor_rejects_invalid_imgsz():
    """Validation preprocessor creation validates the RF-DETR block size."""
    wrapper = _make_wrapper(input_size=504)
    with pytest.raises(
        ValueError,
        match="RF-DETR validation imgsz=500 is not divisible by 24",
    ):
        wrapper._get_val_preprocessor(img_size=500)


def test_export_rejects_invalid_imgsz_before_exporter(monkeypatch):
    """export(imgsz=500) fails before backend exporter setup."""
    wrapper = _make_wrapper(input_size=504)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("BaseModel.export should not run")

    monkeypatch.setattr(rfdetr_model.BaseModel, "export", _should_not_run)
    with pytest.raises(
        ValueError,
        match="RF-DETR export imgsz=500 is not divisible by 24",
    ):
        wrapper.export(imgsz=500)


# ---------------------------------------------------------------------------
# create_transforms() divisibility validation
# ---------------------------------------------------------------------------

def test_imgsz_not_divisible_by_block_size_raises():
    """imgsz=500 with block_size=24 (6*4) and multi_scale=False must raise ValueError."""
    trainer = _make_trainer(imgsz=500, multi_scale=False)
    with pytest.raises(ValueError, match="not divisible by 24"):
        trainer.create_transforms()


def test_imgsz_not_divisible_raises_with_multi_scale_too():
    """Validation always uses literal imgsz, so divisibility is required even with multi_scale=True."""
    trainer = _make_trainer(imgsz=500, multi_scale=True)
    with pytest.raises(ValueError, match="not divisible by 24"):
        trainer.create_transforms()


def test_validator_suggests_adjacent_valid_sizes():
    """The shared message points users to nearby RF-DETR-valid square sizes."""
    with pytest.raises(ValueError, match="Use 544 or 576"):
        validate_imgsz(560, patch_size=16, num_windows=2)


def test_validator_accepts_list_imgsz():
    """CLI/config plumbing may hand a [h, w] list instead of a tuple."""
    assert validate_imgsz([544, 576], patch_size=16, num_windows=2) == (544, 576)


def test_resolve_patch_window_fallback_matches_detection_block():
    """Models without the attrs (mocks, exotic wrappers) fall back to the
    detection block of 32 (patch 16 x 2 windows), so real-world-valid sizes
    such as 544 are never rejected by a guessed block of 64."""
    from libreyolo.models.rfdetr.imgsz import resolve_patch_window

    class _Bare:
        pass

    assert resolve_patch_window(_Bare()) == (16, 2)
    validate_imgsz(544, patch_size=16, num_windows=2)
