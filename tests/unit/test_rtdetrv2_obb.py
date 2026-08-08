from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


def _minimal_obb_state_dict(hidden_dim=128, stem_channels=16):
    return {
        "backbone.stages.0.blocks.0.conv1.conv.weight": torch.empty(1),
        "backbone.stem.stem1.conv.weight": torch.empty(stem_channels, 3, 3, 3),
        "encoder.input_proj.0.conv.weight": torch.empty(hidden_dim, 1, 1, 1),
        "decoder.query_pos_head.layers.0.weight": torch.empty(1, 5),
        "decoder.enc_bbox_head.layers.2.weight": torch.empty(5, 1),
        "decoder.decoder.layers.0.cross_attn.num_points_scale": torch.empty(1),
    }


def test_rtdetrv2_obb_checkpoint_recognition_and_nomenclature():
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    state = _minimal_obb_state_dict()

    assert LibreRTDETRv2.can_load(state)
    assert LibreRTDETRv2.detect_checkpoint_task(state) == "obb"
    assert LibreRTDETRv2.detect_size(state) == "n"
    assert LibreRTDETRv2.detect_size_from_filename("LibreRTDETRv2x-obb.pt") == "x"
    assert LibreRTDETRv2.default_checkpoint_names(15)[0] == "plane"
    assert LibreRTDETRv2.get_download_url("LibreRTDETRv2n-obb.pt") == (
        "https://huggingface.co/LibreYOLO/LibreRTDETRv2n-obb/resolve/main/"
        "LibreRTDETRv2n-obb.pt"
    )


@pytest.mark.parametrize(
    ("hidden_dim", "stem_channels", "expected"),
    [(128, 16, "n"), (224, 16, "s"), (256, 24, "m"), (256, 32, "l"), (384, 32, "x")],
)
def test_rtdetrv2_obb_detects_all_sizes(hidden_dim, stem_channels, expected):
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    state = _minimal_obb_state_dict(hidden_dim, stem_channels)
    assert LibreRTDETRv2.detect_size(state) == expected


def test_rtdetrv2_obb_training_is_explicitly_out_of_scope():
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    wrapper = object.__new__(LibreRTDETRv2)
    wrapper.task = "obb"
    with pytest.raises(NotImplementedError, match="inference-only"):
        wrapper.train()


def test_rtdetrv2_obb_preprocess_preserves_aspect_and_pads_bottom():
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    wrapper = object.__new__(LibreRTDETRv2)
    wrapper.task = "obb"
    wrapper.input_size = 1024
    image = Image.fromarray(np.full((100, 200, 3), 255, dtype=np.uint8))

    tensor, _image, original_size, ratio = wrapper._preprocess(image)

    assert tensor.shape == (1, 3, 1024, 1024)
    assert original_size == (200, 100)
    assert ratio == pytest.approx(5.12)
    assert torch.all(tensor[:, :, :512] == 1.0)
    assert torch.all(tensor[:, :, 512:] == 0.0)


def test_rtdetrv2_obb_postprocess_uses_flat_topk_and_original_geometry():
    from libreyolo.postprocess.rtdetr import postprocess_obb

    output = {
        "pred_logits": torch.tensor([[[10.0, 9.0], [-10.0, -10.0]]]),
        "pred_boxes": torch.tensor(
            [[[0.5, 0.25, 0.2, 0.1, 0.25], [0.1, 0.1, 0.1, 0.1, 0.0]]]
        ),
    }

    result = postprocess_obb(
        output,
        conf_thres=0.0,
        iou_thres=0.99,
        original_size=(200, 100),
        max_det=2,
        input_size=1024,
    )

    assert result["classes"].tolist() == [0, 1]
    torch.testing.assert_close(
        result["obb"][:, :5],
        torch.tensor(
            [
                [100.0, 50.0, 40.0, 20.0, math.pi / 4],
                [100.0, 50.0, 40.0, 20.0, math.pi / 4],
            ]
        ),
    )
    extent = 15 * math.sqrt(2)
    expected_xyxy = torch.tensor(
        [[100 - extent, 50 - extent, 100 + extent, 50 + extent]]
    ).repeat(2, 1)
    torch.testing.assert_close(result["boxes"], expected_xyxy)
    assert result["num_detections"] == 2


def test_rtdetrv2_obb_postprocess_thresholds_after_topk():
    from libreyolo.postprocess.rtdetr import postprocess_obb

    output = {
        "pred_logits": torch.tensor([[[0.0, -1.0], [-2.0, -3.0]]]),
        "pred_boxes": torch.full((1, 2, 5), 0.5),
    }
    result = postprocess_obb(
        output,
        conf_thres=0.49,
        iou_thres=0.0,
        original_size=(64, 64),
        max_det=2,
    )

    assert result["num_detections"] == 1
    assert result["classes"].tolist() == [0]


def test_rtdetrv2_obb_wraps_seven_column_results_and_enclosing_boxes():
    from libreyolo.models.base.inference import InferenceRunner

    runner = InferenceRunner(SimpleNamespace(names={0: "plane"}, task="obb"))
    detections = {
        "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        "scores": torch.tensor([0.9]),
        "classes": torch.tensor([0]),
        "obb": torch.tensor([[20.0, 30.0, 20.0, 10.0, 0.5, 0.9, 0.0]]),
        "num_detections": 1,
    }

    result = runner._wrap_results(detections, (100, 80), None, None)

    assert result.obb is not None
    assert result.obb.data.shape == (1, 7)
    torch.testing.assert_close(result.boxes.xyxy, detections["boxes"])
    torch.testing.assert_close(result.obb.conf, detections["scores"])
    torch.testing.assert_close(result.obb.cls, detections["classes"].float())


def test_rtdetrv2_obb_validation_preprocess_matches_inference_geometry():
    from libreyolo.validation.preprocessors import RTDETRv2OBBValPreprocessor

    preprocessor = RTDETRv2OBBValPreprocessor(img_size=(1024, 1024))
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    targets = np.array([[10.0, 20.0, 30.0, 40.0, 2.0]], dtype=np.float32)

    processed, padded_targets = preprocessor(image, targets, (1024, 1024))

    assert preprocessor.uses_letterbox is True
    assert preprocessor.wants_unresized_image is True
    assert processed.shape == (3, 1024, 1024)
    assert np.all(processed[:, :512] == 1.0)
    assert np.all(processed[:, 512:] == 0.0)
    np.testing.assert_allclose(
        padded_targets[0], np.array([51.2, 102.4, 153.6, 204.8, 2.0])
    )


def test_rtdetrv2_obb_exported_preprocess_matches_native():
    from libreyolo.backends.base import BaseBackend
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    wrapper = object.__new__(LibreRTDETRv2)
    wrapper.task = "obb"
    wrapper.input_size = 1024
    image = Image.fromarray(
        np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    )

    native, _image, native_size, native_ratio = wrapper._preprocess(image)
    exported, _image, exported_size, exported_ratio = (
        BaseBackend._preprocess_rtdetrv2_obb(image, 1024, "auto")
    )

    assert torch.equal(native, exported)
    assert native_size == exported_size
    assert native_ratio == exported_ratio


def test_rtdetrv2_obb_exported_parser_matches_native_postprocess():
    from libreyolo.backends.base import BaseBackend
    from libreyolo.postprocess.rtdetr import postprocess_obb

    output = {
        "pred_logits": torch.tensor([[[10.0, 9.0], [-10.0, -10.0]]]),
        "pred_boxes": torch.tensor(
            [[[0.5, 0.25, 0.2, 0.1, 0.25], [0.1, 0.1, 0.1, 0.1, 0.0]]]
        ),
    }
    native = postprocess_obb(
        output,
        conf_thres=0.0,
        iou_thres=0.0,
        original_size=(200, 100),
        max_det=2,
    )
    parsed = BaseBackend._parse_rtdetr_obb(
        [output["pred_logits"].numpy(), output["pred_boxes"].numpy()],
        1024,
        200,
        100,
        0.0,
        max_det=2,
    )
    boxes, scores, classes, masks, obb = parsed

    assert masks is None
    np.testing.assert_allclose(boxes, native["boxes"].numpy(), rtol=0, atol=1e-5)
    np.testing.assert_allclose(scores, native["scores"].numpy(), rtol=0, atol=1e-7)
    np.testing.assert_array_equal(classes, native["classes"].numpy())
    np.testing.assert_allclose(obb, native["obb"].numpy(), rtol=0, atol=1e-5)


def test_rtdetrv2_obb_exported_parser_supports_five_classes():
    from libreyolo.backends.base import BaseBackend
    from libreyolo.postprocess.rtdetr import postprocess_obb

    output = {
        "pred_logits": torch.tensor(
            [[[8.0, 7.0, 6.0, 5.0, 4.0], [-4.0, -5.0, -6.0, -7.0, -8.0]]]
        ),
        "pred_boxes": torch.tensor(
            [[[0.5, 0.25, 0.2, 0.1, 0.25], [0.1, 0.1, 0.1, 0.1, 0.0]]]
        ),
    }
    native = postprocess_obb(
        output,
        conf_thres=0.0,
        iou_thres=0.0,
        original_size=(200, 100),
        max_det=3,
    )
    boxes, scores, classes, masks, obb = BaseBackend._parse_rtdetr_obb(
        [output["pred_logits"].numpy(), output["pred_boxes"].numpy()],
        1024,
        200,
        100,
        0.0,
        max_det=3,
    )

    assert masks is None
    np.testing.assert_allclose(boxes, native["boxes"].numpy(), rtol=0, atol=1e-5)
    np.testing.assert_allclose(scores, native["scores"].numpy(), rtol=0, atol=1e-7)
    np.testing.assert_array_equal(classes, native["classes"].numpy())
    np.testing.assert_allclose(obb, native["obb"].numpy(), rtol=0, atol=1e-5)


@pytest.mark.torchscript
def test_rtdetrv2_obb_torchscript_raw_roundtrip(tmp_path):
    from libreyolo import LibreYOLO
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    torch.manual_seed(1234)
    model = LibreRTDETRv2(
        model_path=None,
        size="n",
        task="obb",
        nb_classes=15,
        device="cpu",
    )
    model.model.eval()
    output_path = tmp_path / "LibreRTDETRv2n-obb.torchscript"
    model.export("torchscript", imgsz=1024, output_path=str(output_path))

    image = torch.randn(1, 3, 1024, 1024, generator=torch.Generator().manual_seed(8))
    with torch.inference_mode():
        expected = model.model(image)
        actual = torch.jit.load(str(output_path)).eval()(image)
    torch.testing.assert_close(actual[0], expected["pred_logits"], rtol=0, atol=2e-6)
    torch.testing.assert_close(actual[1], expected["pred_boxes"], rtol=0, atol=2e-6)

    backend = LibreYOLO(str(output_path), device="cpu")
    assert (backend.model_family, backend.model_size, backend.task) == (
        "rtdetrv2",
        "n",
        "obb",
    )


@pytest.mark.onnx
def test_rtdetrv2_obb_onnx_raw_roundtrip(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from libreyolo import LibreYOLO
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    torch.manual_seed(1234)
    model = LibreRTDETRv2(
        model_path=None,
        size="n",
        task="obb",
        nb_classes=15,
        device="cpu",
    )
    model.model.eval()
    output_path = tmp_path / "LibreRTDETRv2n-obb.onnx"
    model.export(
        "onnx",
        imgsz=1024,
        opset=17,
        simplify=False,
        output_path=str(output_path),
    )

    image = torch.randn(1, 3, 1024, 1024, generator=torch.Generator().manual_seed(8))
    with torch.inference_mode():
        expected = model.model(image)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: image.numpy()})
    assert actual[0].shape == tuple(expected["pred_logits"].shape)
    assert actual[1].shape == tuple(expected["pred_boxes"].shape)
    assert actual[1].shape[-1] == 5
    assert np.isfinite(actual[0]).all()
    assert np.isfinite(actual[1]).all()
    zero_output = session.run(
        None,
        {session.get_inputs()[0].name: np.zeros_like(image.numpy())},
    )
    assert np.max(np.abs(actual[0] - zero_output[0])) > 1e-5

    backend = LibreYOLO(str(output_path), device="cpu")
    assert (backend.model_family, backend.model_size, backend.task) == (
        "rtdetrv2",
        "n",
        "obb",
    )
