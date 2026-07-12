"""Shared prediction, validation, and checkpoint input-size contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from libreyolo.models.base.inference import InferenceRunner
from libreyolo.models.base.model import BaseModel

pytestmark = pytest.mark.unit


class _ContractModel(BaseModel):
    FAMILY = "contract"
    FILENAME_PREFIX = "LibreContract"
    INPUT_SIZES = {"s": 640}

    def _init_model(self):
        return torch.nn.Identity()

    def _get_available_layers(self):
        return {}

    @staticmethod
    def _get_preprocess_numpy():
        return None

    def _preprocess(self, image, color_format="auto", input_size=None):
        raise AssertionError("preprocessing must not run in contract tests")

    def _forward(self, input_tensor):
        raise AssertionError("forward must not run in contract tests")

    def _postprocess(
        self,
        output,
        conf_thres,
        iou_thres,
        original_size,
        max_det=300,
        ratio=1.0,
        **kwargs,
    ):
        raise AssertionError("postprocessing must not run in contract tests")


class _DivisibleContractModel(_ContractModel):
    INPUT_SIZE_DIVISOR = 32
    INPUT_SIZE_MIN = 128


class _RectangularContractModel(_DivisibleContractModel):
    SUPPORTS_RECTANGULAR_INPUT = True


class _FixedContractModel(_ContractModel):
    INPUT_SIZE_FIXED = True


class _CheckpointFixedContractModel(_FixedContractModel):
    CHECKPOINT_INPUT_SIZE_OVERRIDE = True


class _FixedTrainContractModel(_FixedContractModel):
    def train(self, data, *, imgsz=640):
        del data, imgsz
        raise AssertionError("training body must not run after invalid preflight")


class _KwargTrainContractModel(_FixedContractModel):
    def train(self, data, **kwargs):
        del data, kwargs
        raise AssertionError("training body must not run after invalid preflight")


def _stub(model_class=_ContractModel, *, input_size=640):
    model = object.__new__(model_class)
    model.input_size = input_size
    model.task = "detect"
    model.size = "s"
    model.device = torch.device("cpu")
    model.model = torch.nn.Identity()
    return model


def _family_stub(model_class, *, size, task="detect"):
    model = object.__new__(model_class)
    model.size = size
    model.task = task
    task_sizes = getattr(model_class, "TASK_INPUT_SIZES", {})
    sizes = task_sizes.get(task, model_class.INPUT_SIZES)
    model.input_size = sizes[size]
    return model


@pytest.mark.parametrize("value", [True, False, 640.0, "640", 0, -32])
def test_input_size_rejects_non_integer_or_nonpositive_values(value):
    model = _stub()

    with pytest.raises(ValueError, match="imgsz"):
        model._validate_input_size(value, context="inference")


def test_input_size_enforces_minimum_and_divisor():
    model = _stub(_DivisibleContractModel)

    with pytest.raises(ValueError, match="at least 128"):
        model._validate_input_size(96, context="inference")
    with pytest.raises(ValueError, match="divisible by 32"):
        model._validate_input_size(130, context="inference")
    assert model._validate_input_size(160, context="inference") == 160


def test_prediction_input_size_preserves_valid_rectangular_canvas():
    model = _stub(_RectangularContractModel)

    assert model._validate_predict_input_size((640, 672)) == (640, 672)
    assert model._validate_predict_input_size([640, 672]) == (640, 672)
    with pytest.raises(ValueError, match="divisible by 32"):
        model._validate_predict_input_size((640, 650))
    with pytest.raises(ValueError, match="height, width"):
        model._validate_predict_input_size((640,))


def test_scalar_only_prediction_normalizes_square_pair_and_rejects_rectangle():
    model = _stub(_DivisibleContractModel)

    assert model._validate_predict_input_size((640, 640)) == 640
    with pytest.raises(ValueError, match="does not support rectangular"):
        model._validate_predict_input_size((640, 672))


def test_scalar_only_rectangular_rejection_precedes_source_and_device_work():
    model = _stub(_DivisibleContractModel)
    runner = InferenceRunner(model)

    with pytest.raises(ValueError, match="does not support rectangular"):
        runner(
            Path("definitely-missing.jpg"),
            imgsz=(640, 672),
            device="cuda:0",
        )
    assert model.device == torch.device("cpu")


def test_rectangular_tiling_rejection_precedes_source_and_device_work():
    model = _stub(_RectangularContractModel)
    runner = InferenceRunner(model)

    with pytest.raises(ValueError, match="Tiled inference requires a square imgsz"):
        runner(
            Path("definitely-missing.jpg"),
            imgsz=(640, 672),
            tiling=True,
            device="cuda:0",
        )
    assert model.device == torch.device("cpu")


def test_yolo9_declares_verified_rectangular_prediction_contract():
    from libreyolo.models.yolo9.model import LibreYOLO9

    model = _family_stub(LibreYOLO9, size="s")
    assert model._validate_predict_input_size((640, 672)) == (640, 672)


def test_fixed_input_size_rejects_override():
    model = _stub(_FixedContractModel)

    with pytest.raises(ValueError, match="requires imgsz=640"):
        model._validate_input_size(608, context="validation")
    assert model._validate_input_size(640, context="validation") == 640


def test_prediction_preflight_runs_before_device_or_source_work():
    model = _stub(_FixedContractModel)
    runner = InferenceRunner(model)

    with pytest.raises(ValueError, match="requires imgsz=640"):
        runner(
            Path("definitely-missing.jpg"),
            imgsz=608,
            device="cuda:0",
        )
    assert model.device == torch.device("cpu")


def test_validation_preflight_runs_before_dataset_setup():
    model = _stub(_FixedContractModel)

    with pytest.raises(ValueError, match="requires imgsz=640"):
        model.val(data="definitely-missing.yaml", imgsz=608)


def test_export_preflight_runs_before_exporter_creation(monkeypatch):
    from libreyolo.export import BaseExporter

    model = _stub(_FixedContractModel)

    def _unexpected_create(*args, **kwargs):
        raise AssertionError("exporter creation must not run")

    monkeypatch.setattr(BaseExporter, "create", _unexpected_create)
    with pytest.raises(ValueError, match="requires imgsz=640"):
        model.export(imgsz=608)


def test_explicit_none_export_imgsz_keeps_native_default_semantics(monkeypatch):
    from libreyolo.export import BaseExporter

    model = _stub(_FixedContractModel)
    captured = {}

    def _create(*args, **kwargs):
        del args, kwargs

        def _export(**export_kwargs):
            captured.update(export_kwargs)
            return "model.onnx"

        return _export

    monkeypatch.setattr(BaseExporter, "create", _create)

    assert model.export(imgsz=None) == "model.onnx"
    assert captured == {"imgsz": None}


def test_rectangular_export_validates_each_dimension():
    from libreyolo.export.exporter import OnnxExporter

    model = _stub(_FixedContractModel)
    exporter = OnnxExporter(model)

    with pytest.raises(ValueError, match="requires imgsz=640"):
        exporter._resolve_params(
            output_path="model.onnx",
            imgsz=(640, 608),
            device="cpu",
            half=False,
            int8=False,
        )


def test_rectangular_export_preflight_runs_before_exporter_creation(monkeypatch):
    from libreyolo.export import BaseExporter

    model = _stub(_FixedContractModel)

    def _unexpected_create(*args, **kwargs):
        raise AssertionError("exporter creation must not run")

    monkeypatch.setattr(BaseExporter, "create", _unexpected_create)
    with pytest.raises(ValueError, match="requires imgsz=640"):
        model.export(imgsz=(640, 608))


@pytest.mark.parametrize(
    "model_class",
    [_FixedTrainContractModel, _KwargTrainContractModel],
)
def test_training_preflight_runs_before_family_body(model_class):
    model = _stub(model_class)

    with pytest.raises(ValueError, match="requires imgsz=640"):
        model.train("data.yaml", imgsz=608)


def test_checkpoint_imgsz_updates_dynamic_runtime_default():
    model = _stub(_DivisibleContractModel)

    model._apply_checkpoint_input_size({"imgsz": 768}, is_native_v1=True)

    assert model.input_size == 768


def test_fixed_checkpoint_imgsz_requires_explicit_family_opt_in():
    model = _stub(_FixedContractModel)
    with pytest.raises(RuntimeError, match="checkpoint imgsz=768"):
        model._apply_checkpoint_input_size({"imgsz": 768}, is_native_v1=True)

    opted_in = _stub(_CheckpointFixedContractModel)
    opted_in._apply_checkpoint_input_size({"imgsz": 768}, is_native_v1=True)
    assert opted_in.input_size == 768


def _deim_loader_stub():
    from libreyolo.models.deim.model import LibreDEIM

    model = object.__new__(LibreDEIM)
    model.task = "detect"
    model.size = "n"
    model.input_size = 640
    model.nb_classes = 2
    model.names = {0: "left", 1: "right"}
    model.device = torch.device("cpu")
    model.model = torch.nn.Identity()
    return model


def test_fixed_custom_loader_rejects_native_checkpoint_imgsz_mismatch(
    monkeypatch,
    tmp_path,
):
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    checkpoint = wrap_libreyolo_checkpoint(
        {},
        model_family="deim",
        size="n",
        task="detect",
        nc=2,
        names={0: "left", 1: "right"},
        imgsz=640,
    )
    checkpoint["imgsz"] = 672
    checkpoint_path = tmp_path / "LibreDEIMn.pt"
    checkpoint_path.write_bytes(b"checkpoint fixture")
    monkeypatch.setattr(
        "libreyolo.models.deim.model.torch.load",
        lambda *args, **kwargs: checkpoint,
    )

    with pytest.raises(RuntimeError, match=r"checkpoint requires imgsz=640, got 672"):
        _deim_loader_stub()._load_weights(str(checkpoint_path))


def test_custom_loader_rejects_malformed_native_checkpoint_imgsz(
    monkeypatch,
    tmp_path,
):
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    checkpoint = wrap_libreyolo_checkpoint(
        {},
        model_family="deim",
        size="n",
        task="detect",
        nc=2,
        names={0: "left", 1: "right"},
        imgsz=640,
    )
    checkpoint["imgsz"] = "640"
    checkpoint_path = tmp_path / "LibreDEIMn.pt"
    checkpoint_path.write_bytes(b"checkpoint fixture")
    monkeypatch.setattr(
        "libreyolo.models.deim.model.torch.load",
        lambda *args, **kwargs: checkpoint,
    )

    with pytest.raises(RuntimeError, match="imgsz must be a positive int"):
        _deim_loader_stub()._load_weights(str(checkpoint_path))


def test_legacy_raw_checkpoint_does_not_change_runtime_default():
    model = _stub(_DivisibleContractModel)

    model._apply_checkpoint_input_size({"imgsz": 768}, is_native_v1=False)

    assert model.input_size == 640


def test_yolo9_p2_accepts_valid_checkpoint_imgsz_metadata():
    from libreyolo.models.yolo9_p2.model import LibreYOLO9P2

    model = object.__new__(LibreYOLO9P2)
    model.input_size = 640
    model.task = "detect"
    model.size = "s"

    model._apply_checkpoint_input_size({"imgsz": 768}, is_native_v1=True)

    assert model.input_size == 768


def test_yolo9_accepts_only_stride_compatible_overrides():
    from libreyolo.models.yolo9.model import LibreYOLO9

    model = _family_stub(LibreYOLO9, size="s")
    assert model._validate_input_size(672, context="inference") == 672
    with pytest.raises(ValueError, match="divisible by 32"):
        model._validate_input_size(650, context="inference")


def test_dfine_rejects_tiny_or_odd_decoder_canvases():
    from libreyolo.models.dfine.model import LibreDFINE

    model = _family_stub(LibreDFINE, size="n")
    with pytest.raises(ValueError, match="at least 128"):
        model._validate_input_size(96, context="inference")
    with pytest.raises(ValueError, match="divisible by 32"):
        model._validate_input_size(130, context="inference")


@pytest.mark.parametrize(
    ("module_name", "class_name", "size"),
    [
        ("libreyolo.models.birefnet.model", "LibreBiRefNet", "t"),
        ("libreyolo.models.deim.model", "LibreDEIM", "n"),
        ("libreyolo.models.rtdetrv2.model", "LibreRTDETRv2", "r18"),
        ("libreyolo.models.ec.model", "LibreEC", "s"),
        ("libreyolo.models.fomo.model", "LibreFOMO", "s"),
        ("libreyolo.models.l2cs.model", "LibreL2CS", "r18"),
        ("libreyolo.models.clip.model", "LibreCLIP", "b32"),
        ("libreyolo.models.siglip2.model", "LibreSigLIP2", "b16"),
    ],
)
def test_fixed_families_reject_non_native_override(
    module_name,
    class_name,
    size,
):
    module = __import__(module_name, fromlist=[class_name])
    model_class = getattr(module, class_name)
    model = _family_stub(model_class, size=size, task=model_class.DEFAULT_TASK)

    with pytest.raises(ValueError, match="requires imgsz"):
        model._validate_input_size(model.input_size + 32, context="inference")


def test_dinov2_requires_patch_grid_compatible_size_for_every_task():
    from libreyolo.models.dinov2.model import LibreDINOv2

    classify = _family_stub(LibreDINOv2, size="s", task="classify")
    semantic = _family_stub(LibreDINOv2, size="s", task="semantic")
    assert classify._validate_input_size(224, context="inference") == 224
    assert semantic._validate_input_size(518, context="validation") == 518
    with pytest.raises(ValueError, match="divisible by 14"):
        classify._validate_input_size(225, context="inference")


def test_yolonas_accepts_larger_stride_compatible_public_canvas():
    from libreyolo.models.yolonas.model import LibreYOLONAS

    model = _family_stub(LibreYOLONAS, size="s")
    assert model._validate_input_size(672, context="inference") == 672
    with pytest.raises(ValueError, match="at least 640"):
        model._validate_input_size(608, context="inference")
    with pytest.raises(ValueError, match="divisible by 32"):
        model._validate_input_size(650, context="inference")


def test_yolonas_672_preprocess_and_inverse_geometry_for_detect_and_pose():
    import numpy as np

    from libreyolo.models.yolonas.utils import preprocess_numpy
    from libreyolo.postprocess.yolonas import postprocess, postprocess_pose

    original_size = (800, 600)
    image = np.zeros((600, 800, 3), dtype=np.uint8)

    detect_chw, detect_ratio = preprocess_numpy(image, input_size=672)
    detect_box = torch.tensor([[[97.5, 136.75, 336.0, 335.5]]])
    detect = postprocess(
        (detect_box, torch.tensor([[[0.9]]])),
        conf_thres=0.1,
        input_size=672,
        original_size=original_size,
    )
    assert detect_chw.shape == (3, 672, 672)
    assert detect_ratio == pytest.approx(636 / 800)
    assert torch.allclose(
        detect["boxes"],
        torch.tensor([[100.0, 50.0, 400.0, 300.0]]),
    )

    pose_chw, pose_ratio = preprocess_numpy(
        image,
        input_size=672,
        resize_size=640,
        padding_mode="bottom_right",
        pad_value=127,
    )
    pose = postprocess_pose(
        (
            torch.tensor([[[80.0, 40.0, 320.0, 240.0]]]),
            torch.tensor([[[0.9]]]),
            torch.tensor([[[[160.0, 80.0], [400.0, 320.0]]]]),
            torch.tensor([[[0.8, 0.7]]]),
        ),
        conf_thres=0.1,
        input_size=672,
        original_size=original_size,
    )
    assert pose_chw.shape == (3, 672, 672)
    assert pose_ratio == pytest.approx(640 / 800)
    assert torch.allclose(
        pose["boxes"],
        torch.tensor([[100.0, 50.0, 400.0, 300.0]]),
    )
    assert torch.allclose(
        pose["keypoints"][..., :2],
        torch.tensor([[[200.0, 100.0], [500.0, 400.0]]]),
    )


@pytest.mark.parametrize(
    ("module_name", "class_name", "size"),
    [
        ("libreyolo.models.clip.model", "LibreCLIP", "b32"),
        ("libreyolo.models.siglip2.model", "LibreSigLIP2", "b16"),
    ],
)
def test_custom_classifier_exports_enforce_fixed_canvas_before_export_work(
    module_name,
    class_name,
    size,
):
    module = __import__(module_name, fromlist=[class_name])
    model_class = getattr(module, class_name)
    model = _family_stub(model_class, size=size, task="classify")
    model._text_embeds = None

    with pytest.raises(ValueError, match="requires imgsz"):
        model.export(imgsz=model.input_size + 32)

    with pytest.raises(ValueError, match="requires imgsz"):
        model.val(data="missing-image-folder", imgsz=model.input_size + 32)

    with pytest.raises(RuntimeError, match="No classes set"):
        model.export(imgsz=None)
