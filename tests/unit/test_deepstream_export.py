"""DeepStream export tests: output adapters and nvinfer sidecar generation.

The DeepStream contract is a single ``(batch, num_detections, 6)`` tensor of
``[x1, y1, x2, y2, score, class]`` rows in input-pixel coordinates; the
external parser thresholds and DeepStream clustering suppresses. These tests
validate the adapter math against hand-computed values and the generated
``config_infer_primary`` / labels sidecars.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.export_backend]

_HAS_ORT = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)


class _RawModel(torch.nn.Module):
    def __init__(self, raw: torch.Tensor):
        super().__init__()
        self.register_buffer("raw", raw)

    def forward(self, x):
        return self.raw + x.sum() * 0.0


class _TupleModel(torch.nn.Module):
    def __init__(self, a: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.register_buffer("a", a)
        self.register_buffer("b", b)

    def forward(self, x):
        zero = x.sum() * 0.0
        return self.a + zero, self.b + zero


class _FourOutputModel(torch.nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        zero = x.sum() * 0.0
        return (
            torch.zeros(batch, 2, 4) + zero,
            torch.zeros(batch, 2, 3) + zero,
            torch.zeros(batch, 2, 17, 2) + zero,
            torch.zeros(batch, 2, 17) + zero,
        )


class _ThreeOutputModel(torch.nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        zero = x.sum() * 0.0
        return (
            torch.zeros(batch, 2, 6) + zero,
            torch.zeros(batch, 2) + zero,
            torch.zeros(batch, 2, 17, 3) + zero,
        )


def test_raw_output_adapter_layout_and_argmax():
    from libreyolo.export.deepstream import DeepStreamRawOutput

    # (B, 4 + nc, N): two anchors, three classes.
    raw = torch.zeros(1, 7, 2)
    raw[0, :4, 0] = torch.tensor([0.0, 10.0, 20.0, 30.0])
    raw[0, :4, 1] = torch.tensor([5.0, 15.0, 25.0, 35.0])
    raw[0, 4:, 0] = torch.tensor([0.2, 0.7, 0.1])
    raw[0, 4:, 1] = torch.tensor([0.9, 0.3, 0.4])

    out = DeepStreamRawOutput(_RawModel(raw))(torch.zeros(1, 3, 64, 64))

    assert out.shape == (1, 2, 6)
    np.testing.assert_allclose(out[0, 0].numpy(), [0, 10, 20, 30, 0.7, 1.0])
    np.testing.assert_allclose(out[0, 1].numpy(), [5, 15, 25, 35, 0.9, 0.0], rtol=1e-6)


def test_detr_output_adapter_sigmoid_denorm_and_order():
    from libreyolo.export.deepstream import DeepStreamDETROutput

    # One query, two classes, cxcywh normalized on a 100x200 (h, w) canvas.
    logits = torch.tensor([[[0.0, 2.0]]])  # sigmoid -> [0.5, 0.8808]
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4]]])

    model = _TupleModel(logits, boxes)
    out = DeepStreamDETROutput(model, imgsz=(100, 200), boxes_first=False)(
        torch.zeros(1, 3, 100, 200)
    )

    assert out.shape == (1, 1, 6)
    x1, y1, x2, y2, score, label = out[0, 0].tolist()
    # cx=0.5*200, w=0.2*200 -> x in [80, 120]; cy=0.5*100, h=0.4*100 -> y in [30, 70]
    np.testing.assert_allclose([x1, y1, x2, y2], [80.0, 30.0, 120.0, 70.0], rtol=1e-5)
    assert score == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item(), rel=1e-5)
    assert label == 1.0

    # boxes_first order flips the tuple interpretation.
    model_bf = _TupleModel(boxes, logits)
    out_bf = DeepStreamDETROutput(model_bf, imgsz=(100, 200), boxes_first=True)(
        torch.zeros(1, 3, 100, 200)
    )
    np.testing.assert_allclose(out_bf.numpy(), out.numpy(), rtol=1e-6)


def test_sidecar_files_content(tmp_path):
    from libreyolo.export.deepstream import write_deepstream_sidecars

    onnx_path = tmp_path / "libreyolo9s.onnx"
    onnx_path.write_bytes(b"stub")

    config_path, labels_path = write_deepstream_sidecars(
        str(onnx_path),
        model_family="yolo9",
        class_names=["person", "car"],
        imgsz=(640, 640),
        batch=1,
        precision="fp16",
        conf=0.25,
        iou=0.45,
    )

    labels = (tmp_path / "libreyolo9s_labels.txt").read_text().splitlines()
    assert labels == ["person", "car"]

    config = (tmp_path / "config_infer_primary_libreyolo9s.txt").read_text()
    assert "onnx-file=libreyolo9s.onnx" in config
    assert "model-engine-file=model_b1_gpu0_fp16.engine" in config
    assert "num-detected-classes=2" in config
    assert "network-mode=2" in config
    assert "cluster-mode=2" in config
    assert "parse-bbox-func-name=NvDsInferParseYolo" in config
    assert "pre-cluster-threshold=0.25" in config
    assert "nms-iou-threshold=0.45" in config
    assert config_path.endswith("config_infer_primary_libreyolo9s.txt")
    assert labels_path.endswith("libreyolo9s_labels.txt")


def test_wrap_rejects_unknown_family():
    from libreyolo.export.deepstream import wrap_for_deepstream

    with pytest.raises(NotImplementedError, match="not supported"):
        wrap_for_deepstream(torch.nn.Identity(), model_family="fomo", imgsz=(64, 64))


def test_preflight_unsupported_task_lists_every_supported_task():
    from libreyolo.export.deepstream import deepstream_supported_tasks
    from libreyolo.export.exporter import OnnxExporter

    class _Wrapper:
        task = "obb"

        @staticmethod
        def _get_model_name():
            return "yolo9"

    with pytest.raises(NotImplementedError) as exc_info:
        OnnxExporter(_Wrapper())._preflight(
            half=False, int8=False, data=None, deepstream=True
        )

    message = str(exc_info.value)
    for task in deepstream_supported_tasks():
        assert task in message


@pytest.mark.parametrize("format_name", ["torchscript", "coreml"])
def test_base_preflight_rejects_deepstream_for_non_onnx_formats(format_name):
    from libreyolo.export.exporter import BaseExporter

    class _Exporter:
        pass

    exporter = _Exporter()
    exporter.format_name = format_name

    with pytest.raises(ValueError, match="only for ONNX export"):
        BaseExporter._preflight(
            exporter, half=False, int8=False, data=None, deepstream=True
        )


def test_deepstream_raw_tasks_preserve_output_names_and_dynamic_axes(
    monkeypatch, tmp_path
):
    import libreyolo.export.onnx as onnx_module

    captured = []

    def _capture_export(*args, **kwargs):
        captured.append(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(onnx_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(onnx_module, "_export_onnx_graph", _capture_export)

    onnx_module.export_onnx(
        _FourOutputModel(),
        torch.zeros(1, 3, 32, 32),
        output_path=str(tmp_path / "yolonas_pose.onnx"),
        opset=17,
        simplify=False,
        dynamic=True,
        half=False,
        metadata={"model_family": "yolonas", "task": "pose"},
        deepstream=True,
    )
    onnx_module.export_onnx(
        _ThreeOutputModel(),
        torch.zeros(1, 3, 32, 32),
        output_path=str(tmp_path / "rfdetr_pose.onnx"),
        opset=17,
        simplify=False,
        dynamic=True,
        half=False,
        metadata={"model_family": "rfdetr", "task": "pose"},
        deepstream=True,
    )
    onnx_module.export_onnx(
        _TupleModel(torch.zeros(1, 66), torch.zeros(1, 66)),
        torch.zeros(1, 3, 32, 32),
        output_path=str(tmp_path / "gaze.onnx"),
        opset=17,
        simplify=False,
        dynamic=True,
        half=False,
        metadata={"model_family": "l2cs", "task": "gaze"},
        deepstream=True,
    )

    assert captured[0]["output_names"] == [
        "boxes",
        "scores",
        "keypoints_xy",
        "keypoints_conf",
    ]
    assert captured[0]["dynamic_axes"] == {
        "images": {0: "batch"},
        "boxes": {0: "batch", 1: "anchors"},
        "scores": {0: "batch", 1: "anchors"},
        "keypoints_xy": {0: "batch", 1: "anchors", 2: "keypoints"},
        "keypoints_conf": {0: "batch", 1: "anchors", 2: "keypoints"},
    }
    assert captured[1]["input_names"] == ["input"]
    assert captured[1]["output_names"] == ["dets", "labels", "keypoints"]
    assert captured[1]["dynamic_axes"] == {
        "input": {0: "batch"},
        "dets": {0: "batch"},
        "labels": {0: "batch"},
        "keypoints": {0: "batch"},
    }
    assert captured[2]["output_names"] == ["yaw_logits", "pitch_logits"]
    assert captured[2]["dynamic_axes"] == {
        "images": {0: "faces"},
        "yaw_logits": {0: "faces"},
        "pitch_logits": {0: "faces"},
    }


@pytest.mark.onnx
@pytest.mark.skipif(not _HAS_ORT, reason="requires onnx and onnxruntime")
@pytest.mark.parametrize(
    "adapter_name,make_model,imgsz",
    [
        (
            "raw_channels_first",
            lambda: _RawModel(torch.rand(1, 7, 5)),
            (32, 32),
        ),
        (
            "detr_tuple",
            lambda: _TupleModel(torch.randn(1, 4, 3), torch.rand(1, 4, 4)),
            (64, 48),
        ),
    ],
)
def test_deepstream_graph_matches_torch_through_onnxruntime(
    tmp_path, adapter_name, make_model, imgsz
):
    """The exported graph reproduces the torch adapter output in ORT."""
    import onnxruntime as ort

    from libreyolo.export.deepstream import (
        DeepStreamDETROutput,
        DeepStreamRawOutput,
    )

    inner = make_model()
    if adapter_name == "raw_channels_first":
        wrapped = DeepStreamRawOutput(inner, channels_first=True).eval()
    else:
        wrapped = DeepStreamDETROutput(inner, imgsz, boxes_first=False).eval()

    dummy = torch.randn(1, 3, *imgsz)
    path = tmp_path / f"{adapter_name}.onnx"
    torch.onnx.export(
        wrapped,
        dummy,
        str(path),
        input_names=["images"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )

    with torch.no_grad():
        expected = wrapped(dummy).numpy()

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"images": dummy.numpy()})[0]

    assert got.shape == expected.shape
    assert got.shape[-1] == 6
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "family,expect_cluster,expect_nms",
    [("yolo9", "cluster-mode=2", True), ("rfdetr", "cluster-mode=4", False)],
)
def test_detr_configs_disable_clustering(tmp_path, family, expect_cluster, expect_nms):
    """DETR heads emit one query per object, so DeepStream must not cluster."""
    from libreyolo.export.deepstream import write_deepstream_sidecars

    onnx_path = tmp_path / f"{family}.onnx"
    onnx_path.write_bytes(b"stub")
    config_path, _ = write_deepstream_sidecars(
        str(onnx_path),
        model_family=family,
        class_names=["a"],
        imgsz=(640, 640),
        batch=1,
        precision="fp32",
    )
    config = Path(config_path).read_text()

    assert expect_cluster in config
    assert ("nms-iou-threshold" in config) is expect_nms


class _LogitModel(torch.nn.Module):
    def __init__(self, out: torch.Tensor):
        super().__init__()
        self.register_buffer("out", out)

    def forward(self, x):
        return self.out + x.sum() * 0.0


def test_classifier_adapter_emits_probabilities():
    from libreyolo.export.deepstream import DeepStreamClassifierOutput

    logits = torch.tensor([[1.0, 2.0, 3.0]])
    out = DeepStreamClassifierOutput(_LogitModel(logits))(torch.zeros(1, 3, 8, 8))

    np.testing.assert_allclose(
        out.numpy(), torch.softmax(logits, dim=-1).numpy(), rtol=1e-6
    )
    assert out.sum().item() == pytest.approx(1.0)


def test_semantic_adapter_softmaxes_over_class_axis():
    from libreyolo.export.deepstream import DeepStreamSemanticOutput

    logits = torch.randn(1, 4, 5, 6)
    out = DeepStreamSemanticOutput(_LogitModel(logits))(torch.zeros(1, 3, 8, 8))

    assert out.shape == (1, 4, 5, 6)
    np.testing.assert_allclose(
        out.sum(dim=1).numpy(), np.ones((1, 5, 6), dtype=np.float32), rtol=1e-5
    )


def test_semantic_adapter_passthrough_for_probability_heads():
    """EoMT already emits probabilities; a second softmax would distort them."""
    from libreyolo.export.deepstream import (
        DeepStreamSemanticOutput,
        wrap_for_deepstream,
        _SEMANTIC_ALREADY_PROBABILITIES,
    )

    assert "eomt" in _SEMANTIC_ALREADY_PROBABILITIES

    probs = torch.rand(1, 3, 4, 4)
    out = DeepStreamSemanticOutput(_LogitModel(probs), apply_softmax=False)(
        torch.zeros(1, 3, 8, 8)
    )
    np.testing.assert_allclose(out.numpy(), probs.numpy(), rtol=1e-6)

    wrapped = wrap_for_deepstream(
        torch.nn.Identity(), model_family="eomt", imgsz=(512, 512), task="semantic"
    )
    assert wrapped.apply_softmax is False


def test_segformer_rejected_for_semantic_export():
    """SegFormer has no ONNX export path in this codebase."""
    from libreyolo.export.deepstream import wrap_for_deepstream

    with pytest.raises(NotImplementedError, match="not supported"):
        wrap_for_deepstream(
            torch.nn.Identity(),
            model_family="segformer",
            imgsz=(512, 512),
            task="semantic",
        )


@pytest.mark.parametrize(
    "task,family,expected",
    [
        ("classify", "resnet", "network-type=1"),
        ("semantic", "pidnet", "network-type=2"),
        ("detect", "yolo9", "network-type=0"),
    ],
)
def test_config_network_type_per_task(tmp_path, task, family, expected):
    from libreyolo.export.deepstream import write_deepstream_sidecars

    onnx_path = tmp_path / f"{family}_{task}.onnx"
    onnx_path.write_bytes(b"stub")
    config_path, _ = write_deepstream_sidecars(
        str(onnx_path),
        model_family=family,
        class_names=["a", "b"],
        imgsz=(224, 224),
        batch=1,
        precision="fp32",
        task=task,
    )
    config = Path(config_path).read_text()

    assert expected in config
    # Only detection needs the third-party bbox parser library.
    assert ("custom-lib-path" in config) is (task == "detect")


class _SegModel(torch.nn.Module):
    def __init__(self, logits, boxes, masks):
        super().__init__()
        self.register_buffer("logits", logits)
        self.register_buffer("boxes", boxes)
        self.register_buffer("masks", masks)

    def forward(self, x):
        zero = x.sum() * 0.0
        return self.logits + zero, self.boxes + zero, self.masks + zero


def test_instance_seg_row_is_detection_plus_quarter_canvas_mask():
    """The seg parser hardcodes mask_width=netW/4, mask_height=netH/4."""
    from libreyolo.export.deepstream import DeepStreamInstanceSegOutput

    imgsz = (128, 64)  # h, w -> mask 32 x 16
    logits = torch.tensor([[[0.0, 2.0]]])
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4]]])
    masks = torch.zeros(1, 1, 8, 8)

    out = DeepStreamInstanceSegOutput(
        _SegModel(logits, boxes, masks), imgsz, boxes_first=False
    )(torch.zeros(1, 3, *imgsz))

    mask_size = (imgsz[0] // 4) * (imgsz[1] // 4)
    assert out.shape == (1, 1, 6 + mask_size)

    x1, y1, x2, y2, score, label = out[0, 0, :6].tolist()
    # cx=0.5*64=32, w=0.2*64=12.8 -> x in [25.6, 38.4]
    # cy=0.5*128=64, h=0.4*128=51.2 -> y in [38.4, 89.6]
    np.testing.assert_allclose([x1, y1, x2, y2], [25.6, 38.4, 38.4, 89.6], rtol=1e-5)
    assert label == 1.0
    assert score == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item(), rel=1e-5)

    # Mask values are probabilities: sigmoid(0) == 0.5 everywhere here.
    np.testing.assert_allclose(
        out[0, 0, 6:].numpy(), np.full(mask_size, 0.5, dtype=np.float32), rtol=1e-6
    )


def test_instance_seg_config_uses_mask_parser(tmp_path):
    from libreyolo.export.deepstream import write_deepstream_sidecars

    onnx_path = tmp_path / "seg.onnx"
    onnx_path.write_bytes(b"stub")
    config_path, _ = write_deepstream_sidecars(
        str(onnx_path),
        model_family="dfine",
        class_names=["a"],
        imgsz=(640, 640),
        batch=1,
        precision="fp32",
        task="segment",
    )
    config = Path(config_path).read_text()

    assert "network-type=3" in config
    assert "output-instance-mask=1" in config
    assert "parse-bbox-instance-mask-func-name=NvDsInferParseYoloSeg" in config
    # DETR seg heads emit one query per object.
    assert "cluster-mode=4" in config
    # Masks need the seg build of the parser, not the detection one.
    assert "libnvdsinfer_custom_impl_Yolo_seg.so" in config


def test_instance_seg_rejects_blocked_families():
    """RTMDet-Ins and YOLO9 have no seg export path in libreyolo."""
    from libreyolo.export.deepstream import wrap_for_deepstream

    for family in ("rtmdet", "yolo9"):
        with pytest.raises(NotImplementedError, match="not supported"):
            wrap_for_deepstream(
                torch.nn.Identity(),
                model_family=family,
                imgsz=(640, 640),
                task="segment",
            )


def test_depth_config_uses_raw_tensor_meta_and_no_labels(tmp_path):
    """DeepStream has no depth post-processor: the app reads the tensor."""
    from libreyolo.export.deepstream import write_deepstream_sidecars

    onnx_path = tmp_path / "depth.onnx"
    onnx_path.write_bytes(b"stub")
    config_path, labels_path = write_deepstream_sidecars(
        str(onnx_path),
        model_family="zipdepth",
        class_names=[],
        imgsz=(384, 384),
        batch=1,
        precision="fp32",
        task="depth",
    )
    config = Path(config_path).read_text()

    assert "network-type=100" in config
    assert "output-tensor-meta=1" in config
    # Depth has no classes, so no labels file and no labelfile-path key.
    assert labels_path == ""
    assert "labelfile-path" not in config
    assert not (tmp_path / "depth_labels.txt").exists()


def test_depth_passthrough_keeps_the_map_untouched():
    from libreyolo.export.deepstream import wrap_for_deepstream

    inner = torch.nn.Identity()
    wrapped = wrap_for_deepstream(
        inner, model_family="zipdepth", imgsz=(384, 384), task="depth"
    )
    assert wrapped is inner


@pytest.mark.parametrize(
    "task,family",
    [
        ("pose", "yolonas"),
        ("restore", "realesrgan"),
        ("matte", "birefnet"),
        ("gaze", "l2cs"),
        ("depth", "zipdepth"),
    ],
)
def test_raw_tensor_tasks_share_one_config_shape(tmp_path, task, family):
    """Tasks DeepStream cannot post-process all export the same way."""
    from libreyolo.export.deepstream import write_deepstream_sidecars

    onnx_path = tmp_path / f"{family}_{task}.onnx"
    onnx_path.write_bytes(b"stub")
    config_path, labels_path = write_deepstream_sidecars(
        str(onnx_path),
        model_family=family,
        class_names=[],
        imgsz=(224, 224),
        batch=1,
        precision="fp32",
        task=task,
    )
    config = Path(config_path).read_text()

    assert "network-type=100" in config
    assert "output-tensor-meta=1" in config
    # No parser library, no clustering, no labels: the app decodes.
    assert "custom-lib-path" not in config
    assert "cluster-mode" not in config
    assert labels_path == ""


def test_yolonas_pose_config_matches_native_bgr_bottom_right_preprocess(tmp_path):
    """Graph parity cannot catch a wrong external nvinfer preprocessor."""
    from libreyolo.export.deepstream import write_deepstream_sidecars

    onnx_path = tmp_path / "yolonas_pose.onnx"
    onnx_path.write_bytes(b"stub")
    config_path, _ = write_deepstream_sidecars(
        str(onnx_path),
        model_family="yolonas",
        class_names=[],
        imgsz=(640, 640),
        batch=1,
        precision="fp32",
        task="pose",
    )
    config = Path(config_path).read_text()

    assert "model-color-format=1" in config
    assert "maintain-aspect-ratio=1" in config
    assert "symmetric-padding=0" in config


def test_preprocess_keys_stay_in_property_section(monkeypatch, tmp_path):
    """Future letterboxed seg profiles must not leak keys into class attrs."""
    import libreyolo.export.deepstream as deepstream_module

    monkeypatch.setitem(
        deepstream_module._PREPROCESS_PROFILES,
        "dfine",
        {"maintain_aspect_ratio": 1, "symmetric_padding": 1},
    )
    onnx_path = tmp_path / "dfine_seg.onnx"
    onnx_path.write_bytes(b"stub")
    config_path, _ = deepstream_module.write_deepstream_sidecars(
        str(onnx_path),
        model_family="dfine",
        class_names=["person"],
        imgsz=(640, 640),
        batch=1,
        precision="fp32",
        task="segment",
    )
    property_section, class_attrs = (
        Path(config_path).read_text().split("[class-attrs-all]", maxsplit=1)
    )

    assert "maintain-aspect-ratio=1" in property_section
    assert "symmetric-padding=1" in property_section
    assert "maintain-aspect-ratio" not in class_attrs
    assert "symmetric-padding" not in class_attrs


def test_raw_tensor_normalization_is_per_family():
    """Only families normalizing outside their forward get a graph norm."""
    from libreyolo.export.deepstream import _GraphNorm, wrap_for_deepstream

    # BiRefNet normalizes in its preprocess transform -> bake it in.
    matte = wrap_for_deepstream(
        torch.nn.Identity(), model_family="birefnet", imgsz=(1024, 1024), task="matte"
    )
    assert isinstance(matte, _GraphNorm)

    # Real-ESRGAN takes plain [0, 1]; SwinIR subtracts its own mean inside
    # forward. Neither may be normalized again.
    for family in ("realesrgan", "swinir"):
        inner = torch.nn.Identity()
        assert (
            wrap_for_deepstream(
                inner, model_family=family, imgsz=(256, 256), task="restore"
            )
            is inner
        )


def test_raw_tensor_tasks_reject_unlisted_families():
    from libreyolo.export.deepstream import wrap_for_deepstream

    # picodet is a detector; it has no pose export.
    with pytest.raises(NotImplementedError, match="not supported"):
        wrap_for_deepstream(
            torch.nn.Identity(), model_family="picodet", imgsz=(640, 640), task="pose"
        )


@pytest.mark.parametrize(
    "family,task",
    [
        # No export implementation at all (ADR 0006).
        ("depth_anything3", "depth"),
        # Not wired to the shared semantic export contract.
        ("segformer", "semantic"),
    ],
)
def test_families_without_an_export_path_are_refused(family, task):
    """Never generate a config for a model that cannot produce a graph."""
    from libreyolo.export.deepstream import wrap_for_deepstream

    with pytest.raises(NotImplementedError, match="not supported"):
        wrap_for_deepstream(
            torch.nn.Identity(), model_family=family, imgsz=(518, 518), task=task
        )
