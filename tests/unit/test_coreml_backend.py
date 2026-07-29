"""CoreML backend contract and canonical-input regression tests."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


class _FeatureType:
    def __init__(self, kind: str, *, height: int = 4, width: int = 4):
        self._kind = kind
        self.imageType = SimpleNamespace(height=height, width=width)
        self.multiArrayType = SimpleNamespace(shape=[1, 3, height, width])

    def WhichOneof(self, _name: str) -> str:
        return self._kind


def _feature(name: str, *, kind: str | None = None, height=4, width=4):
    feature_type = (
        _FeatureType(kind, height=height, width=width) if kind is not None else None
    )
    return SimpleNamespace(name=name, type=feature_type)


class _FakeMLModel:
    def __init__(self, metadata, spec, predict_fn=None):
        self.user_defined_metadata = metadata
        self._spec = spec
        self._predict_fn = predict_fn or (
            lambda _inputs: {"prediction": np.zeros((1, 5, 1), dtype=np.float32)}
        )
        self.predict_calls = []

    def get_spec(self):
        return self._spec

    def predict(self, inputs):
        self.predict_calls.append(inputs)
        return self._predict_fn(inputs)


def _io_contract(
    *,
    input_name="image",
    geometry="stretch",
    validation_range="0_255",
    validation_color="rgb",
    validation_mean=None,
    validation_std=None,
    outputs=None,
    interpolation="bilinear",
    resize_backend="pillow",
    resize_long_side=None,
    resize_rounding=None,
    pad_value=114,
    crop_pct=0.875,
):
    validation = {
        "color": validation_color,
        "range": validation_range,
    }
    if validation_mean is not None:
        validation["mean"] = validation_mean
    if validation_std is not None:
        validation["std"] = validation_std
    input_contract = {
        "name": input_name,
        "kind": "image",
        "layout": "nchw",
        "color": "rgb",
        "range": "uint8",
        "geometry": geometry,
        "interpolation": interpolation,
        "resize_backend": resize_backend,
        "pad_value": pad_value,
        "crop_pct": crop_pct,
        "shape_mode": "fixed",
    }
    if resize_long_side is not None:
        input_contract["resize_long_side"] = resize_long_side
    if resize_rounding is not None:
        input_contract["resize_rounding"] = resize_rounding
    return {
        "input": {
            **input_contract,
        },
        "validation": validation,
        "outputs": outputs
        or [
            {
                "name": "prediction",
                "role": "prediction",
            }
        ],
    }


def _add_output_abi(io, *, task):
    io = deepcopy(io)
    rank_by_name = {
        "prediction": 3,
        "pred_logits": 3,
        "pred_boxes": 3,
        "pred_masks": 4,
        "class_logits": 2 if task == "classify" else 3,
        "yaw_logits": 2,
        "pitch_logits": 2,
        "semantic_logits": 4,
        "depth": 4,
        "restored": 4,
        "point_logits": 4,
        "confidence": 2,
        "coordinates": 2,
        "boxes": 3,
        "scores": 3,
        "keypoints_xy": 4,
        "keypoints_conf": 3,
    }
    for output in io["outputs"]:
        output.setdefault("rank", rank_by_name.get(output["name"], 3))
        output.setdefault("dtype", "float32")
    return io


def _profile_io(family, task="detect", size=None, *, nms=False):
    from libreyolo.export.coreml import (
        _input_contract,
        _output_contract,
        _validation_contract,
    )

    return _add_output_abi(
        {
            "input": _input_contract(family, task, size),
            "validation": _validation_contract(family, task),
            "outputs": _output_contract(family, task, nms=nms),
        },
        task=task,
    )


def _metadata(
    *,
    family="yolo9",
    task="detect",
    size="t",
    names=None,
    imgsz=4,
    io=None,
):
    if io is None:
        io = _profile_io(family, task, size)
    else:
        io = _add_output_abi(io, task=task)
    names = names or {0: "object"}
    if isinstance(imgsz, tuple):
        imgsz_h, imgsz_w = imgsz
        legacy_imgsz = max(imgsz)
    else:
        imgsz_h = imgsz_w = legacy_imgsz = imgsz
    metadata = {
        "artifact_format": "coreml",
        "libreyolo_producer": "libreyolo",
        "coreml_io_schema_version": "1",
        "coreml_io": json.dumps(io),
        "schema_version": "1.0",
        "libreyolo_version": "0.test",
        "model_family": family,
        "size": size,
        "model_size": size,
        "task": task,
        "supported_tasks": json.dumps([task]),
        "default_task": task,
        "names": json.dumps({str(key): value for key, value in names.items()}),
        "nc": str(len(names)),
        "imgsz": str(legacy_imgsz),
        "imgsz_h": str(imgsz_h),
        "imgsz_w": str(imgsz_w),
        "dynamic": "false",
        "precision": "fp32",
    }
    if task == "classify":
        metadata["classification_activation"] = "softmax"
    if task == "pose":
        metadata.update(
            {
                "num_keypoints": "17",
                "keypoint_dim": "3",
                "pose_encoding": (
                    "rfdetr_flat_keypoints_v1"
                    if family == "rfdetr"
                    else "yolonas_split_xy_conf_v1"
                    if family == "yolonas"
                    else "ec_normalized_xy_v1"
                    if family == "ec"
                    else "keypoints_v1"
                ),
            }
        )
    return metadata


def _spec(
    *,
    input_name="image",
    input_kind="imageType",
    outputs=("prediction",),
    imgsz=4,
):
    if isinstance(imgsz, tuple):
        imgsz_h, imgsz_w = imgsz
    else:
        imgsz_h = imgsz_w = imgsz
    return SimpleNamespace(
        description=SimpleNamespace(
            input=[
                _feature(
                    input_name,
                    kind=input_kind,
                    height=imgsz_h,
                    width=imgsz_w,
                )
            ],
            output=[_feature(name) for name in outputs],
        )
    )


def _v2_feature(
    name,
    *,
    kind,
    shape=None,
    height=4,
    width=4,
    color_space=20,
    dtype=65568,
    flexible=False,
    optional=False,
):
    image_type = SimpleNamespace(
        height=height,
        width=width,
        colorSpace=color_space,
        WhichOneof=lambda _name: "enumeratedSizes" if flexible else None,
    )
    array_type = SimpleNamespace(
        shape=list(shape or ()),
        dataType=dtype,
        WhichOneof=lambda _name: "shapeRange" if flexible else None,
    )
    return SimpleNamespace(
        name=name,
        isOptional=optional,
        type=SimpleNamespace(
            WhichOneof=lambda _name: kind,
            isOptional=optional,
            imageType=image_type,
            multiArrayType=array_type,
        ),
    )


def _v2_rtdetr_artifact(*, output_flexible=False, output_dtype=65568):
    io = _profile_io("rtdetr", "detect", "r18")
    io["outputs"][0]["shape"] = [1, 2, 1]
    io["outputs"][1]["shape"] = [1, 2, 4]
    metadata = _metadata(family="rtdetr", size="r18", io=io)
    metadata["coreml_io_schema_version"] = "2"
    spec = SimpleNamespace(
        description=SimpleNamespace(
            input=[
                _v2_feature(
                    "image",
                    kind="imageType",
                    height=4,
                    width=4,
                )
            ],
            output=[
                _v2_feature(
                    "pred_logits",
                    kind="multiArrayType",
                    shape=(1, 2, 1),
                    dtype=output_dtype,
                    flexible=output_flexible,
                ),
                _v2_feature(
                    "pred_boxes",
                    kind="multiArrayType",
                    shape=(1, 2, 4),
                ),
            ],
        )
    )
    return metadata, spec


def _v2_picosam3_artifact():
    from libreyolo.export.coreml_picosam3 import (
        picosam3_coreml_component_metadata,
    )

    io = _profile_io("picosam3", "segment", "pico")
    io["outputs"][0]["shape"] = [1, 1, 96, 96]
    metadata = _metadata(
        family="picosam3",
        task="segment",
        size="pico",
        names={0: "object"},
        imgsz=96,
        io=io,
    )
    metadata["coreml_io_schema_version"] = "2"
    metadata["coreml_output_names"] = json.dumps(["mask_logits"])
    metadata.update(
        {
            key: str(value)
            for key, value in picosam3_coreml_component_metadata().items()
        }
    )
    spec = SimpleNamespace(
        description=SimpleNamespace(
            input=[
                _v2_feature(
                    "roi_image",
                    kind="imageType",
                    height=96,
                    width=96,
                )
            ],
            output=[
                _v2_feature(
                    "mask_logits",
                    kind="multiArrayType",
                    shape=(1, 1, 96, 96),
                )
            ],
        )
    )
    return metadata, spec


def _v2_owlv2_artifact():
    from libreyolo.export.coreml_owlv2 import (
        expected_owlv2_coreml_shapes,
        owlv2_coreml_metadata,
    )

    names = {0: "red fox", 1: "fire hydrant"}
    shapes = expected_owlv2_coreml_shapes(size="b16", nc=len(names))
    io = _profile_io("owlv2", "detect", "b16")
    for output in io["outputs"]:
        output["shape"] = list(shapes[output["name"]])
    metadata = _metadata(
        family="owlv2",
        task="detect",
        size="b16",
        names=names,
        imgsz=960,
        io=io,
    )
    metadata["coreml_io_schema_version"] = "2"
    metadata["coreml_output_names"] = json.dumps(
        [output["name"] for output in io["outputs"]]
    )
    metadata.update(
        {
            key: str(value)
            for key, value in owlv2_coreml_metadata(
                size="b16",
                names=names,
            ).items()
        }
    )
    spec = SimpleNamespace(
        description=SimpleNamespace(
            input=[
                _v2_feature(
                    "image",
                    kind="multiArrayType",
                    shape=(1, 3, 960, 960),
                )
            ],
            output=[
                _v2_feature(
                    output["name"],
                    kind="multiArrayType",
                    shape=shapes[output["name"]],
                )
                for output in io["outputs"]
            ],
        )
    )
    return metadata, spec


def _v2_grounding_dino_artifact():
    from libreyolo.export.coreml_grounding_dino import (
        GroundingDinoFrozenText,
        expected_grounding_dino_coreml_shapes,
        grounding_dino_coreml_metadata,
    )

    names = {0: "red fox", 1: "dog"}
    input_ids = torch.tensor(
        [[101, 1037, 2417, 4419, 1012, 1037, 3899, 1012, 102]],
        dtype=torch.long,
    )
    sequence_length = int(input_ids.shape[1])
    frozen = GroundingDinoFrozenText(
        labels=tuple(names.values()),
        prompt="a red fox. a dog.",
        input_ids=input_ids,
        token_type_ids=torch.zeros_like(input_ids),
        attention_mask=torch.ones_like(input_ids),
        text_self_attention_masks=torch.ones(
            (1, sequence_length, sequence_length),
            dtype=torch.bool,
        ),
        position_ids=torch.arange(sequence_length).view(1, -1),
        text_features=torch.zeros((1, sequence_length, 256)),
        token_pieces=(
            "[CLS]",
            "a",
            "red",
            "fox",
            ".",
            "a",
            "dog",
            ".",
            "[SEP]",
        ),
    )
    shapes = expected_grounding_dino_coreml_shapes(
        size="t",
        sequence_length=sequence_length,
    )
    io = _profile_io("grounding_dino", "detect", "t")
    for output in io["outputs"]:
        output["shape"] = list(shapes[output["name"]])
    metadata = _metadata(
        family="grounding_dino",
        task="detect",
        size="t",
        names=names,
        imgsz=800,
        io=io,
    )
    metadata["coreml_io_schema_version"] = "2"
    metadata["coreml_output_names"] = json.dumps(
        [output["name"] for output in io["outputs"]]
    )
    metadata.update(
        {
            key: str(value)
            for key, value in grounding_dino_coreml_metadata(
                size="t",
                names=names,
                frozen=frozen,
            ).items()
        }
    )
    spec = SimpleNamespace(
        description=SimpleNamespace(
            input=[
                _v2_feature(
                    "image",
                    kind="imageType",
                    height=800,
                    width=800,
                )
            ],
            output=[
                _v2_feature(
                    output["name"],
                    kind="multiArrayType",
                    shape=shapes[output["name"]],
                )
                for output in io["outputs"]
            ],
        )
    )
    return metadata, spec


def _v2_omdet_turbo_artifact():
    from libreyolo.export.coreml_omdet_turbo import (
        expected_omdet_turbo_coreml_shapes,
        omdet_turbo_coreml_metadata,
    )

    names = {0: "red fox", 1: "fire hydrant"}
    shapes = expected_omdet_turbo_coreml_shapes(size="t", nc=len(names))
    io = _profile_io("omdet_turbo", "detect", "t")
    for output in io["outputs"]:
        output["shape"] = list(shapes[output["name"]])
    metadata = _metadata(
        family="omdet_turbo",
        task="detect",
        size="t",
        names=names,
        imgsz=640,
        io=io,
    )
    metadata["coreml_io_schema_version"] = "2"
    metadata["coreml_output_names"] = json.dumps(
        [output["name"] for output in io["outputs"]]
    )
    metadata.update(
        {
            key: str(value)
            for key, value in omdet_turbo_coreml_metadata(
                size="t",
                names=names,
            ).items()
        }
    )
    spec = SimpleNamespace(
        description=SimpleNamespace(
            input=[
                _v2_feature(
                    "image",
                    kind="multiArrayType",
                    shape=(1, 3, 640, 640),
                )
            ],
            output=[
                _v2_feature(
                    output["name"],
                    kind="multiArrayType",
                    shape=shapes[output["name"]],
                )
                for output in io["outputs"]
            ],
        )
    )
    return metadata, spec


def _v2_depth_anything3_artifact():
    from libreyolo.export.coreml_depth_anything3 import (
        depth_anything3_coreml_metadata,
        expected_depth_anything3_coreml_shapes,
    )

    shapes = expected_depth_anything3_coreml_shapes(
        batch=1,
        canvas_hw=504,
    )
    io = _profile_io("depth_anything3", "depth", "l")
    for output in io["outputs"]:
        output["shape"] = list(shapes[output["name"]])
    metadata = _metadata(
        family="depth_anything3",
        task="depth",
        size="l",
        names={0: "depth"},
        imgsz=504,
        io=io,
    )
    metadata["coreml_io_schema_version"] = "2"
    metadata["coreml_output_names"] = json.dumps(
        [output["name"] for output in io["outputs"]]
    )
    metadata.update(
        {
            key: str(value)
            for key, value in depth_anything3_coreml_metadata().items()
        }
    )
    spec = SimpleNamespace(
        description=SimpleNamespace(
            input=[
                _v2_feature(
                    "image",
                    kind="imageType",
                    height=504,
                    width=504,
                )
            ],
            output=[
                _v2_feature(
                    output["name"],
                    kind="multiArrayType",
                    shape=shapes[output["name"]],
                )
                for output in io["outputs"]
            ],
        )
    )
    return metadata, spec


def _v2_rtmdet_ins_artifact():
    from libreyolo.export.coreml_rtmdet_ins import (
        expected_rtmdet_ins_coreml_shapes,
        rtmdet_ins_coreml_metadata,
    )

    names = {0: "object"}
    shapes = expected_rtmdet_ins_coreml_shapes(nc=len(names), canvas_hw=(64, 64))
    io = _profile_io("rtmdet", "segment", "t")
    for output in io["outputs"]:
        output["shape"] = list(shapes[output["name"]])
    metadata = _metadata(
        family="rtmdet",
        task="segment",
        size="t",
        names=names,
        imgsz=64,
        io=io,
    )
    metadata["coreml_io_schema_version"] = "2"
    metadata["coreml_output_names"] = json.dumps(
        [output["name"] for output in io["outputs"]]
    )
    metadata.update(
        {
            key: json.dumps(value) if isinstance(value, list) else str(value)
            for key, value in rtmdet_ins_coreml_metadata().items()
        }
    )
    spec = SimpleNamespace(
        description=SimpleNamespace(
            input=[
                _v2_feature(
                    "image",
                    kind="imageType",
                    height=64,
                    width=64,
                )
            ],
            output=[
                _v2_feature(
                    output["name"],
                    kind="multiArrayType",
                    shape=shapes[output["name"]],
                )
                for output in io["outputs"]
            ],
        )
    )
    return metadata, spec


def _v2_eomt_artifact(task="semantic"):
    from libreyolo.export.coreml_eomt import (
        eomt_coreml_metadata,
        expected_eomt_coreml_shapes,
    )

    names = {0: "thing", 1: "stuff"}
    shapes = expected_eomt_coreml_shapes(
        nc=len(names),
        num_queries=4,
        canvas_hw=(32, 32),
    )
    io = _profile_io("eomt", task, "s")
    for output in io["outputs"]:
        output["shape"] = list(shapes[output["name"]])
    metadata = _metadata(
        family="eomt",
        task=task,
        size="s",
        names=names,
        imgsz=32,
        io=io,
    )
    metadata["coreml_io_schema_version"] = "2"
    metadata["coreml_output_names"] = json.dumps(
        [output["name"] for output in io["outputs"]]
    )
    metadata["num_queries"] = "4"
    metadata.update(
        {
            key: json.dumps(value)
            if isinstance(value, (dict, list, tuple))
            else str(value)
            for key, value in eomt_coreml_metadata(
                task=task,
                num_queries=4,
                image_size=32,
            ).items()
        }
    )
    if task == "panoptic":
        metadata.update(
            {
                "thing_class_ids": json.dumps([0]),
                "eomt_panoptic_score_threshold": "0.8",
                "eomt_panoptic_mask_threshold": "0.5",
                "eomt_panoptic_overlap_threshold": "0.8",
            }
        )
    spec = SimpleNamespace(
        description=SimpleNamespace(
            input=[
                _v2_feature(
                    "image",
                    kind="multiArrayType",
                    shape=(1, 3, 32, 32),
                )
            ],
            output=[
                _v2_feature(
                    output["name"],
                    kind="multiArrayType",
                    shape=shapes[output["name"]],
                )
                for output in io["outputs"]
            ],
        )
    )
    return metadata, spec


def _load_backend(
    monkeypatch,
    tmp_path,
    *,
    metadata=None,
    spec=None,
    predict_fn=None,
    backend_kwargs=None,
):
    mlmodel = _FakeMLModel(
        metadata or _metadata(),
        spec or _spec(),
        predict_fn=predict_fn,
    )
    fake_ct = SimpleNamespace(
        ComputeUnit=SimpleNamespace(
            ALL="ALL",
            CPU_AND_GPU="CPU_AND_GPU",
            CPU_AND_NE="CPU_AND_NE",
            CPU_ONLY="CPU_ONLY",
        ),
        models=SimpleNamespace(MLModel=lambda *_args, **_kwargs: mlmodel),
    )
    monkeypatch.setitem(sys.modules, "coremltools", fake_ct)
    monkeypatch.setattr(sys, "platform", "darwin")
    package = tmp_path / "model.mlpackage"
    package.mkdir()

    from libreyolo.backends.coreml import CoreMLBackend

    backend = CoreMLBackend(str(package), **(backend_kwargs or {}))
    return backend, mlmodel


def _rtmdet_ins_runtime_outputs():
    spatial = (8, 4, 2)
    cls = [
        np.full((1, 1, size, size), -20.0, dtype=np.float32)
        for size in spatial
    ]
    boxes = [
        np.full((1, 4, size, size), 4.0, dtype=np.float32)
        for size in spatial
    ]
    kernels = [
        np.zeros((1, 169, size, size), dtype=np.float32)
        for size in spatial
    ]
    cls[0][0, 0, 1, 1] = 10.0
    kernels[0][0, -1, 1, 1] = 10.0
    values = (
        *cls,
        *boxes,
        *kernels,
        np.zeros((1, 8, 8, 8), dtype=np.float32),
    )
    names = [
        "class_logits_s8",
        "class_logits_s16",
        "class_logits_s32",
        "box_distances_s8",
        "box_distances_s16",
        "box_distances_s32",
        "dynamic_kernels_s8",
        "dynamic_kernels_s16",
        "dynamic_kernels_s32",
        "mask_features",
    ]
    # Deliberately reverse insertion order. Runtime interpretation must use
    # semantic names from the artifact contract, never mapping iteration.
    return dict(reversed(list(zip(names, values))))


def test_owlv2_artifact_runs_exact_host_preprocess_and_named_decode(
    monkeypatch,
    tmp_path,
):
    from libreyolo.export.coreml_owlv2 import preprocess_owlv2_coreml_image

    metadata, spec = _v2_owlv2_artifact()
    logits = np.full((1, 3600, 2), -20.0, dtype=np.float32)
    boxes = np.zeros((1, 3600, 4), dtype=np.float32)
    logits[0, 0, 0] = 4.0
    logits[0, 1, 1] = 3.0
    boxes[0, 0] = (0.25, 0.25, 0.2, 0.2)
    boxes[0, 1] = (0.75, 0.25, 0.2, 0.2)

    backend, mlmodel = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        # Deliberately reverse mapping insertion order.
        predict_fn=lambda _inputs: {
            "pred_boxes": boxes,
            "pred_logits": logits,
        },
    )
    source = Image.fromarray(
        np.arange(40 * 80 * 3, dtype=np.uint16)
        .reshape(40, 80, 3)
        .astype(np.uint8),
        mode="RGB",
    )
    result = backend.predict(source)

    assert len(mlmodel.predict_calls) == 1
    runtime_input = mlmodel.predict_calls[0]["image"]
    assert isinstance(runtime_input, np.ndarray)
    expected_input = preprocess_owlv2_coreml_image(
        source,
        image_size=960,
    ).numpy()
    np.testing.assert_allclose(runtime_input, expected_input, rtol=0.0, atol=1e-7)
    assert result.names == {0: "red fox", 1: "fire hydrant"}
    assert result.boxes.cls.tolist() == [0.0, 1.0]
    torch.testing.assert_close(
        result.boxes.xyxy,
        torch.tensor(
            [
                [12.0, 12.0, 28.0, 28.0],
                [52.0, 12.0, 68.0, 28.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("frozen_classes", "false"),
        ("owlv2_vocabulary_sha256", "0" * 64),
        ("owlv2_num_patches", "3599"),
    ],
)
def test_owlv2_artifact_rejects_forged_frozen_metadata(
    monkeypatch,
    tmp_path,
    key,
    value,
):
    metadata, spec = _v2_owlv2_artifact()
    metadata[key] = value
    with pytest.raises(ValueError, match="OWLv2"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=spec,
        )


def test_owlv2_artifact_rejects_forged_output_shape(monkeypatch, tmp_path):
    metadata, spec = _v2_owlv2_artifact()
    io = json.loads(metadata["coreml_io"])
    io["outputs"][0]["shape"] = [1, 3599, 2]
    metadata["coreml_io"] = json.dumps(io)
    spec.description.output[0].type.multiArrayType.shape = [1, 3599, 2]

    with pytest.raises(ValueError, match="OWLv2 CoreML output shapes"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=spec,
        )


def test_grounding_dino_artifact_runs_exact_host_preprocess_and_frozen_decode(
    monkeypatch,
    tmp_path,
):
    from libreyolo.export.coreml_grounding_dino import (
        preprocess_grounding_dino_coreml_image,
    )

    metadata, spec = _v2_grounding_dino_artifact()
    logits = np.full((1, 900, 9), -20.0, dtype=np.float32)
    boxes = np.zeros((1, 900, 4), dtype=np.float32)
    # "red fox." is selected at the default text threshold. The period
    # supplies the detection score while both label pieces remain borderline,
    # making the per-call text_threshold override observable.
    logits[0, 0, 2:4] = 0.0
    logits[0, 0, 4] = 4.0
    boxes[0, 0] = (0.5, 0.5, 0.5, 0.5)

    backend, mlmodel = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        # Deliberately reverse mapping insertion order.
        predict_fn=lambda _inputs: {
            "pred_boxes": boxes,
            "token_logits": logits,
        },
    )
    source = Image.fromarray(
        np.arange(40 * 80 * 3, dtype=np.uint16)
        .reshape(40, 80, 3)
        .astype(np.uint8),
        mode="RGB",
    )
    result = backend.predict(source)

    runtime_input = mlmodel.predict_calls[0]["image"]
    assert isinstance(runtime_input, Image.Image)
    expected = (
        preprocess_grounding_dino_coreml_image(source)
        .mul(255.0)
        .round()
        .to(torch.uint8)[0]
        .permute(1, 2, 0)
        .numpy()
    )
    np.testing.assert_array_equal(np.asarray(runtime_input), expected)
    assert result.names == {0: "red fox", 1: "dog"}
    assert result.boxes.cls.tolist() == [0.0]
    torch.testing.assert_close(
        result.boxes.xyxy,
        torch.tensor([[20.0, 10.0, 60.0, 30.0]]),
    )

    strict = backend.predict(source, text_threshold=0.9)
    assert len(strict.boxes) == 0
    assert backend._grounding_dino_text_threshold == 0.25


def test_grounding_dino_stream_threshold_is_request_local(
    monkeypatch,
    tmp_path,
):
    metadata, spec = _v2_grounding_dino_artifact()
    backend, _ = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
    )
    observed = []

    def fake_predict_video(_source, **_kwargs):
        def frames():
            observed.append(
                backend._current_grounding_dino_text_threshold()
            )
            yield "frame"

        return frames()

    monkeypatch.setattr(
        "libreyolo.backends.base.is_video_file",
        lambda _source: True,
    )
    monkeypatch.setattr(backend, "_predict_video", fake_predict_video)

    first = backend.predict(
        "first.mp4",
        stream=True,
        text_threshold=0.8,
    )
    second = backend.predict(
        "second.mp4",
        stream=True,
        text_threshold=0.6,
    )
    # Merely creating or discarding a stream must not leak its override.
    assert backend._current_grounding_dino_text_threshold() == 0.25
    assert next(second) == "frame"
    assert observed == [0.6]
    assert backend._current_grounding_dino_text_threshold() == 0.25
    assert next(first) == "frame"
    assert observed == [0.6, 0.8]
    assert backend._current_grounding_dino_text_threshold() == 0.25
    first.close()
    second.close()


def test_grounding_dino_filters_classes_before_max_det(
    monkeypatch,
    tmp_path,
):
    metadata, spec = _v2_grounding_dino_artifact()
    logits = np.full((1, 900, 9), -20.0, dtype=np.float32)
    boxes = np.zeros((1, 900, 4), dtype=np.float32)
    # The globally highest query is class 0 ("red fox"), while the requested
    # class 1 ("dog") ranks second. Filtering after max_det=1 would lose it.
    logits[0, 0, 2:5] = (4.0, 4.0, 6.0)
    logits[0, 1, 6:8] = (3.0, 4.0)
    boxes[0, 0] = (0.25, 0.5, 0.2, 0.2)
    boxes[0, 1] = (0.75, 0.5, 0.2, 0.2)
    backend, _ = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        predict_fn=lambda _inputs: {
            "pred_boxes": boxes,
            "token_logits": logits,
        },
    )

    result = backend.predict(
        Image.new("RGB", (100, 50), color="white"),
        classes=[1],
        max_det=1,
    )

    assert result.boxes.cls.tolist() == [1.0]
    torch.testing.assert_close(
        result.boxes.xyxy,
        torch.tensor([[65.0, 20.0, 85.0, 30.0]]),
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("frozen_classes", "false"),
        ("grounding_dino_text_abi_sha256", "0" * 64),
        ("grounding_dino_sequence_length", "8"),
    ],
)
def test_grounding_dino_artifact_rejects_forged_frozen_metadata(
    monkeypatch,
    tmp_path,
    key,
    value,
):
    metadata, spec = _v2_grounding_dino_artifact()
    metadata[key] = value
    with pytest.raises(ValueError, match="Grounding DINO"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=spec,
        )


def test_omdet_turbo_artifact_runs_exact_host_preprocess_and_named_decode(
    monkeypatch,
    tmp_path,
):
    from libreyolo.export.coreml_omdet_turbo import (
        postprocess_omdet_turbo_coreml_outputs,
        preprocess_omdet_turbo_coreml_image,
    )

    metadata, spec = _v2_omdet_turbo_artifact()
    logits = np.full((1, 900, 2), -20.0, dtype=np.float32)
    boxes = np.zeros((1, 900, 4), dtype=np.float32)
    logits[0, 0, 0] = 2.0
    logits[0, 1, 1] = 1.5
    boxes[0, 0] = (0.25, 0.5, 0.2, 0.4)
    boxes[0, 1] = (0.75, 0.5, 0.2, 0.4)

    backend, mlmodel = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        # Deliberately reverse mapping insertion order.
        predict_fn=lambda _inputs: {
            "pred_boxes": boxes,
            "pred_logits": logits,
        },
    )
    source = Image.fromarray(
        np.arange(40 * 80 * 3, dtype=np.uint16)
        .reshape(40, 80, 3)
        .astype(np.uint8),
        mode="RGB",
    )
    result = backend.predict(source)

    runtime_input = mlmodel.predict_calls[0]["image"]
    expected_input = preprocess_omdet_turbo_coreml_image(source).numpy()
    np.testing.assert_array_equal(runtime_input, expected_input)
    expected = postprocess_omdet_turbo_coreml_outputs(
        logits,
        boxes,
        original_size=source.size,
        conf=0.3,
        iou=0.5,
        max_det=300,
    )
    torch.testing.assert_close(result.boxes.xyxy, expected["boxes"])
    torch.testing.assert_close(result.boxes.conf, expected["scores"])
    torch.testing.assert_close(
        result.boxes.cls,
        expected["classes"].to(dtype=torch.float32),
    )

    filtered = backend.predict(source, classes=[1])
    assert filtered.boxes.cls.tolist() == [1.0]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("frozen_classes", "false"),
        ("omdet_turbo_vocabulary_sha256", "0" * 64),
        ("omdet_turbo_num_queries", "899"),
    ],
)
def test_omdet_turbo_artifact_rejects_forged_frozen_metadata(
    monkeypatch,
    tmp_path,
    key,
    value,
):
    metadata, spec = _v2_omdet_turbo_artifact()
    metadata[key] = value
    with pytest.raises(ValueError, match="OMDet-Turbo"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=spec,
        )


def test_depth_anything3_artifact_runs_named_sky_inverse_before_resize(
    monkeypatch,
    tmp_path,
):
    import torch.nn.functional as F

    from libreyolo.export.coreml_depth_anything3 import (
        postprocess_depth_anything3_coreml,
    )

    metadata, spec = _v2_depth_anything3_artifact()
    depth = np.linspace(
        0.2,
        4.0,
        504 * 504,
        dtype=np.float32,
    ).reshape(1, 1, 504, 504)
    sky = np.zeros_like(depth)
    sky[:, :, :400] = 1.0
    backend, mlmodel = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        # Deliberately return reverse insertion order.
        predict_fn=lambda _inputs: {
            "sky_score": sky,
            "relative_depth": depth,
        },
    )
    source = Image.new("RGB", (80, 40), color=(20, 80, 140))
    result = backend.predict(source)

    expected = postprocess_depth_anything3_coreml(
        torch.from_numpy(depth.copy()),
        torch.from_numpy(sky.copy()),
    )
    expected = F.interpolate(
        expected,
        size=(40, 80),
        mode="bilinear",
        align_corners=True,
    )[0, 0]
    assert len(mlmodel.predict_calls) == 1
    assert isinstance(mlmodel.predict_calls[0]["image"], Image.Image)
    assert mlmodel.predict_calls[0]["image"].size == (504, 504)
    assert result.depth_map is not None
    torch.testing.assert_close(result.depth_map.data, expected, rtol=0.0, atol=0.0)


def test_depth_anything3_validator_returns_one_public_inverse_depth_output(
    monkeypatch,
    tmp_path,
):
    metadata, spec = _v2_depth_anything3_artifact()
    depth = np.full((1, 1, 504, 504), 2.0, dtype=np.float32)
    sky = np.zeros_like(depth)
    backend, mlmodel = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        predict_fn=lambda _inputs: {
            "relative_depth": depth,
            "sky_score": sky,
        },
    )

    outputs = backend._forward(torch.zeros(2, 3, 504, 504))
    assert len(mlmodel.predict_calls) == 2
    assert len(outputs) == 1
    assert tuple(outputs[0].shape) == (2, 1, 504, 504)
    torch.testing.assert_close(
        outputs[0],
        torch.full((2, 1, 504, 504), 0.5),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("depth_anything3_sky_threshold", "0.4"),
        ("depth_anything3_sky_sample_limit", "99999"),
        ("depth_anything3_sky_sampling", "without_replacement"),
        ("depth_anything3_position_embedding", "runtime_bicubic"),
    ],
)
def test_depth_anything3_artifact_rejects_forged_host_metadata(
    monkeypatch,
    tmp_path,
    key,
    value,
):
    metadata, spec = _v2_depth_anything3_artifact()
    metadata[key] = value
    with pytest.raises(ValueError, match="Depth Anything 3"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=spec,
        )


def test_depth_anything3_artifact_rejects_forged_raw_shape(
    monkeypatch,
    tmp_path,
):
    metadata, spec = _v2_depth_anything3_artifact()
    io = json.loads(metadata["coreml_io"])
    io["outputs"][1]["shape"] = [1, 1, 252, 252]
    metadata["coreml_io"] = json.dumps(io)
    spec.description.output[1].type.multiArrayType.shape = [1, 1, 252, 252]
    with pytest.raises(ValueError, match="Depth Anything 3 CoreML output shapes"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=spec,
        )


def test_rtmdet_ins_artifact_runs_named_host_mask_decode(monkeypatch, tmp_path):
    metadata, spec = _v2_rtmdet_ins_artifact()
    backend, mlmodel = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        predict_fn=lambda _inputs: _rtmdet_ins_runtime_outputs(),
    )
    result = backend.predict(
        Image.new("RGB", (96, 48), color=(20, 40, 80)),
        conf=0.25,
        iou=0.6,
    )

    assert len(mlmodel.predict_calls) == 1
    assert len(result) == 1
    assert result.masks is not None
    assert result.masks.data.shape == (1, 48, 96)
    assert result.masks.data.all()
    assert result.boxes.cls.tolist() == [0.0]


def test_rtmdet_ins_artifact_does_not_run_backend_nms_twice(
    monkeypatch,
    tmp_path,
):
    metadata, spec = _v2_rtmdet_ins_artifact()
    backend, _ = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        predict_fn=lambda _inputs: _rtmdet_ins_runtime_outputs(),
    )

    def fail_second_nms(*_args, **_kwargs):
        raise AssertionError("RTMDet-Ins host decode already performed NMS")

    monkeypatch.setattr(
        "libreyolo.backends.base._batched_nms_numpy",
        fail_second_nms,
    )
    result = backend.predict(Image.new("RGB", (64, 64), color="white"))
    assert len(result) == 1


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rtmdet_ins_contract", "rtmdet_ins_raw_v2"),
        ("rtmdet_ins_strides", "[8, 16, 64]"),
        ("rtmdet_ins_num_gen_params", "168"),
        ("rtmdet_ins_num_prototypes", "16"),
        ("rtmdet_ins_mask_stride", "4"),
        ("rtmdet_ins_nms_pre", "999"),
        ("rtmdet_ins_max_masks", "101"),
        ("rtmdet_ins_prior_offset", "1"),
        ("rtmdet_ins_dynamic_weight_nums", "[80, 32, 8]"),
        ("rtmdet_ins_dynamic_bias_nums", "[8, 4, 1]"),
        ("rtmdet_ins_dyconv_channels", "4"),
        ("rtmdet_ins_mask_threshold", "0.4"),
    ],
)
def test_rtmdet_ins_artifact_rejects_forged_decode_metadata(
    monkeypatch,
    tmp_path,
    key,
    value,
):
    metadata, spec = _v2_rtmdet_ins_artifact()
    metadata[key] = value
    with pytest.raises(ValueError, match="RTMDet-Ins"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=spec,
        )


def test_rtmdet_ins_artifact_rejects_wrong_kernel_shape(monkeypatch, tmp_path):
    metadata, spec = _v2_rtmdet_ins_artifact()
    io = json.loads(metadata["coreml_io"])
    kernel = next(
        output
        for output in io["outputs"]
        if output["name"] == "dynamic_kernels_s16"
    )
    kernel["shape"] = [1, 168, 4, 4]
    metadata["coreml_io"] = json.dumps(io)

    with pytest.raises(ValueError, match="output shapes"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=spec,
        )


def test_eomt_semantic_artifact_runs_one_named_call_per_split_patch(
    monkeypatch,
    tmp_path,
):
    metadata, spec = _v2_eomt_artifact("semantic")

    def predict(inputs):
        image = np.asarray(inputs["image"], dtype=np.float32)
        level = float(image.mean())
        classes = np.full((1, 4, 3), -5.0, dtype=np.float32)
        classes[0, 0, 0] = 5.0 + level
        classes[0, 1, 1] = 5.0 - level
        masks = np.full((1, 4, 8, 8), -4.0, dtype=np.float32)
        masks[0, 0, :, :4] = 4.0 + level
        masks[0, 1, :, 4:] = 4.0 - level
        # Deliberately return reverse contract order. Names, not dictionary
        # insertion order, define the runtime ABI.
        return {
            "masks_queries_logits": masks,
            "class_queries_logits": classes,
        }

    backend, mlmodel = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        predict_fn=predict,
    )
    result = backend.predict(
        Image.new("RGB", (51, 20), color=(20, 80, 140)),
    )

    assert len(mlmodel.predict_calls) == 3
    assert result.semantic_mask is not None
    assert result.semantic_mask.data.shape == (20, 51)
    assert set(result.semantic_mask.classes).issubset({0, 1})


def test_eomt_panoptic_artifact_builds_dense_result_from_reversed_names(
    monkeypatch,
    tmp_path,
):
    metadata, spec = _v2_eomt_artifact("panoptic")
    classes = np.full((1, 4, 3), -20.0, dtype=np.float32)
    classes[0, 0, 0] = 20.0
    classes[0, 1, 1] = 18.0
    masks = np.full((1, 4, 8, 8), -20.0, dtype=np.float32)
    masks[0, 0, :, :4] = 20.0
    masks[0, 1, :, 4:] = 20.0
    backend, mlmodel = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        predict_fn=lambda _inputs: {
            "masks_queries_logits": masks,
            "class_queries_logits": classes,
        },
    )

    result = backend.predict(
        Image.new("RGB", (47, 23), color=(20, 80, 140)),
        conf=0.99,
        iou=0.01,
        max_det=1,
    )

    assert len(mlmodel.predict_calls) == 1
    assert result.panoptic is not None
    assert result.panoptic.data.shape == (23, 47)
    assert result.panoptic.segment_ids
    assert all(
        segment["category_id"] in {0, 1}
        for segment in result.panoptic.segments_info
    )


def test_picosam3_component_runs_box_prompt_pipeline(monkeypatch, tmp_path):
    metadata, spec = _v2_picosam3_artifact()

    def predict(inputs):
        roi = inputs["roi_image"]
        assert isinstance(roi, Image.Image)
        assert roi.size == (96, 96)
        return {
            "mask_logits": np.ones((1, 1, 96, 96), dtype=np.float32),
        }

    backend, mlmodel = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        predict_fn=predict,
    )
    image = Image.new("RGB", (32, 24), color=(10, 20, 30))
    result = backend.predict(image, bboxes=[[8, 6, 24, 18]])

    assert len(mlmodel.predict_calls) == 1
    assert result.masks is not None
    assert result.masks.data.shape == (1, 24, 32)
    assert result.masks.data[0, 2:21, 6:25].all()
    assert not result.masks.data[0, :2].any()
    assert float(result.boxes.conf[0]) == pytest.approx(
        float(torch.sigmoid(torch.tensor(1.0)))
    )


def test_picosam3_component_supports_cached_image(monkeypatch, tmp_path):
    metadata, spec = _v2_picosam3_artifact()
    backend, _ = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        predict_fn=lambda _inputs: {
            "mask_logits": np.ones((1, 1, 96, 96), dtype=np.float32)
        },
    )

    backend.set_image(Image.new("RGB", (32, 24), color="white"))
    result = backend.predict(bboxes=[[8, 6, 24, 18]])

    assert result.masks is not None and len(result.masks) == 1


def test_picosam3_component_schedules_multiple_rois_sequentially(
    monkeypatch,
    tmp_path,
):
    metadata, spec = _v2_picosam3_artifact()
    calls = 0

    def predict(_inputs):
        nonlocal calls
        calls += 1
        return {
            "mask_logits": np.full(
                (1, 1, 96, 96),
                float(calls),
                dtype=np.float32,
            )
        }

    backend, _ = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        predict_fn=predict,
    )
    result = backend.predict(
        Image.new("RGB", (64, 48), color="white"),
        bboxes=[[4, 4, 20, 20], [36, 20, 56, 42]],
    )

    assert calls == 2
    assert result.masks is not None and len(result.masks) == 2
    assert result.boxes.conf.tolist() == pytest.approx(
        [
            float(torch.sigmoid(torch.tensor(1.0))),
            float(torch.sigmoid(torch.tensor(2.0))),
        ]
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("artifact_scope", "full_model"),
        ("component_contract", "picosam3_roi_v2"),
        ("roi_input_size", "128"),
        ("roi_padding", "0.2"),
        ("roi_batch", "2"),
        ("prompt_type", "points"),
    ],
)
def test_picosam3_component_rejects_forged_orchestration_metadata(
    monkeypatch,
    tmp_path,
    key,
    value,
):
    metadata, spec = _v2_picosam3_artifact()
    metadata[key] = value

    with pytest.raises(ValueError, match="PicoSAM3"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=spec,
        )


def test_picosam3_component_rejects_wrong_mask_shape(monkeypatch, tmp_path):
    metadata, spec = _v2_picosam3_artifact()
    io = json.loads(metadata["coreml_io"])
    io["outputs"][0]["shape"] = [1, 1, 48, 48]
    metadata["coreml_io"] = json.dumps(io)
    spec.description.output[0].type.multiArrayType.shape = [1, 1, 48, 48]

    with pytest.raises(ValueError, match="mask output"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=spec,
        )


def test_strict_contract_orders_outputs_by_metadata(monkeypatch, tmp_path):
    io = _profile_io("rtdetr", "detect", "r18")
    metadata = _metadata(family="rtdetr", io=io)
    spec = _spec(outputs=("pred_boxes", "pred_logits"))

    def predict(_inputs):
        return {
            "pred_boxes": np.full((1, 2, 4), 2.0, dtype=np.float32),
            "pred_logits": np.full((1, 2, 1), 1.0, dtype=np.float32),
        }

    backend, model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        predict_fn=predict,
    )
    output = backend._run_inference(np.zeros((1, 3, 4, 4), dtype=np.float32))

    assert backend.output_names == ["pred_logits", "pred_boxes"]
    assert float(output[0][0, 0, 0]) == 1.0
    assert float(output[1][0, 0, 0]) == 2.0
    assert list(model.predict_calls[0]) == ["image"]


def test_rtdetr_four_class_outputs_use_semantic_contract_order(
    monkeypatch,
    tmp_path,
):
    """Both outputs end in four at nc=4; names/roles must defeat shape guessing."""
    names = {index: f"class_{index}" for index in range(4)}
    io = _profile_io("rtdetr", "detect", "r18")
    metadata = _metadata(
        family="rtdetr",
        size="r18",
        names=names,
        io=io,
    )
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=_spec(outputs=("pred_boxes", "pred_logits")),
    )
    logits = np.array([[[8.0, -8.0, -8.0, -8.0]]], dtype=np.float32)
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)

    parsed_boxes, scores, classes, masks = backend._parse_outputs(
        [logits, boxes],
        effective_imgsz=4,
        original_size=(100, 80),
        conf=0.5,
        max_det=10,
    )

    np.testing.assert_allclose(parsed_boxes, [[40.0, 32.0, 60.0, 48.0]])
    assert scores.tolist() == pytest.approx([1.0 / (1.0 + np.exp(-8.0))])
    assert classes.tolist() == [0]
    assert masks is None


def test_runtime_missing_named_output_fails(monkeypatch, tmp_path):
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        predict_fn=lambda _inputs: {},
    )

    with pytest.raises(RuntimeError, match=r"missing=\['prediction'\]"):
        backend._run_inference(np.zeros((1, 3, 4, 4), dtype=np.float32))


def test_runtime_nonfinite_output_fails(monkeypatch, tmp_path):
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        predict_fn=lambda _inputs: {
            "prediction": np.full((1, 5, 1), np.nan, dtype=np.float32)
        },
    )

    with pytest.raises(RuntimeError, match="NaN or infinity"):
        backend._run_inference(np.zeros((1, 3, 4, 4), dtype=np.float32))


def test_schema_v2_enforces_runtime_exact_output_shape(monkeypatch, tmp_path):
    metadata, spec = _v2_rtdetr_artifact()
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
        predict_fn=lambda _inputs: {
            "pred_logits": np.zeros((1, 3, 1), dtype=np.float32),
            "pred_boxes": np.zeros((1, 2, 4), dtype=np.float32),
        },
    )

    with pytest.raises(RuntimeError, match="shape"):
        backend._run_inference(np.zeros((1, 3, 4, 4), dtype=np.float32))


@pytest.mark.parametrize(
    ("output_flexible", "output_dtype", "message"),
    [
        (True, 65568, "flexible shape"),
        (False, 65552, "dtype disagrees"),
    ],
)
def test_schema_v2_rejects_nonfixed_or_wrong_dtype_spec_outputs(
    monkeypatch,
    tmp_path,
    output_flexible,
    output_dtype,
    message,
):
    metadata, spec = _v2_rtdetr_artifact(
        output_flexible=output_flexible,
        output_dtype=output_dtype,
    )

    with pytest.raises(ValueError, match=message):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=spec,
        )


def test_schema_v2_rejects_non_rgb_image_spec(monkeypatch, tmp_path):
    metadata, spec = _v2_rtdetr_artifact()
    spec.description.input[0].type.imageType.colorSpace = 30

    with pytest.raises(ValueError, match="RGB color"):
        _load_backend(monkeypatch, tmp_path, metadata=metadata, spec=spec)


@pytest.mark.parametrize("target", ["input", "output"])
def test_schema_v2_rejects_optional_spec_features(
    monkeypatch,
    tmp_path,
    target,
):
    metadata, spec = _v2_rtdetr_artifact()
    feature = (
        spec.description.input[0]
        if target == "input"
        else spec.description.output[0]
    )
    feature.type.isOptional = True

    with pytest.raises(ValueError, match="optional"):
        _load_backend(monkeypatch, tmp_path, metadata=metadata, spec=spec)


def test_schema_v2_maps_spec_outputs_by_name_not_protobuf_order(
    monkeypatch,
    tmp_path,
):
    metadata, spec = _v2_rtdetr_artifact()
    spec.description.output.reverse()

    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=spec,
    )

    assert backend.output_names == ["pred_logits", "pred_boxes"]


def test_schema_v2_rejects_forged_dense_candidate_count(
    monkeypatch,
    tmp_path,
):
    io = _profile_io("yolox", "detect", "n")
    io["outputs"][0]["shape"] = [1, 2, 6]
    metadata = _metadata(family="yolox", size="n", io=io)
    metadata["coreml_io_schema_version"] = "2"
    spec = SimpleNamespace(
        description=SimpleNamespace(
            input=[_v2_feature("image", kind="imageType")],
            output=[
                _v2_feature(
                    "prediction",
                    kind="multiArrayType",
                    shape=(1, 2, 6),
                )
            ],
        )
    )

    with pytest.raises(ValueError, match="fixed stride grid"):
        _load_backend(monkeypatch, tmp_path, metadata=metadata, spec=spec)


def test_strict_profile_rejects_tampered_geometry(monkeypatch, tmp_path):
    io = _profile_io("yolo9", "detect", "t")
    io["input"]["geometry"] = "stretch"
    metadata = _metadata(family="yolo9", size="t", io=io)

    with pytest.raises(ValueError, match="input/validation profile"):
        _load_backend(monkeypatch, tmp_path, metadata=metadata)


@pytest.mark.parametrize(
    ("alias", "value"),
    [
        ("crop_pct", "0.5"),
        ("interpolation", "nearest"),
    ],
)
def test_strict_classifier_rejects_preprocess_alias_drift(
    monkeypatch,
    tmp_path,
    alias,
    value,
):
    io = _profile_io("resnet", "classify", "18")
    metadata = _metadata(
        family="resnet",
        task="classify",
        size="18",
        io=io,
    )
    metadata[alias] = value

    with pytest.raises(ValueError, match=alias):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=_spec(outputs=("class_logits",)),
        )


def test_strict_profile_rejects_reordered_semantic_outputs(monkeypatch, tmp_path):
    io = _profile_io("rfdetr", "segment", "n")
    io["outputs"][0], io["outputs"][1] = io["outputs"][1], io["outputs"][0]
    metadata = _metadata(
        family="rfdetr",
        task="segment",
        size="n",
        io=io,
    )

    with pytest.raises(ValueError, match="output profile"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=_spec(outputs=("pred_boxes", "pred_logits", "pred_masks")),
        )


def test_strict_profile_requires_output_rank_and_dtype(monkeypatch, tmp_path):
    metadata = _metadata(family="yolo9", size="t")
    io = json.loads(metadata["coreml_io"])
    io["outputs"][0].pop("rank")
    io["outputs"][0].pop("dtype")
    metadata["coreml_io"] = json.dumps(io)

    with pytest.raises(ValueError, match="rank and dtype"):
        _load_backend(monkeypatch, tmp_path, metadata=metadata)


def test_strict_nms_declaration_must_match_package_outputs(monkeypatch, tmp_path):
    metadata = _metadata(
        family="yolo9",
        size="t",
        io=_profile_io("yolo9", "detect", "t", nms=True),
    )

    with pytest.raises(ValueError, match="NMS metadata does not match"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=_spec(outputs=("confidence", "coordinates")),
        )


def test_strict_deimv2_loader_enforces_license_size_gate(monkeypatch, tmp_path):
    metadata = _metadata(family="deimv2", size="s")

    with pytest.raises(NotImplementedError, match="DINOv3 licensing boundary"):
        _load_backend(monkeypatch, tmp_path, metadata=metadata)


def test_strict_metadata_rejects_conflicting_size_aliases(monkeypatch, tmp_path):
    metadata = _metadata(
        family="realesrgan",
        task="restore",
        size="x4",
    )
    metadata["model_size"] = "x2"

    with pytest.raises(ValueError, match="size=.*model_size"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=_spec(outputs=("restored",)),
        )


def test_sigmoid_classification_activation_is_preserved(monkeypatch, tmp_path):
    io = _profile_io("siglip2", "classify", "b16")
    metadata = _metadata(
        family="siglip2",
        task="classify",
        size="b16",
        names={0: "a", 1: "b"},
        io=io,
    )
    metadata["classification_activation"] = "sigmoid"
    metadata["frozen_classes"] = "true"
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=_spec(outputs=("class_logits",)),
        predict_fn=lambda _inputs: {
            "class_logits": np.array([[0.0, 2.0]], dtype=np.float32)
        },
    )

    outputs = backend._run_inference(
        np.zeros((1, 3, 4, 4), dtype=np.float32)
    )
    probabilities = backend._parse_classify_probs(outputs)

    assert backend.classification_activation == "sigmoid"
    torch.testing.assert_close(
        probabilities,
        torch.sigmoid(torch.tensor([0.0, 2.0])),
    )


def test_strict_classification_requires_valid_activation(monkeypatch, tmp_path):
    io = _profile_io("resnet", "classify", "18")
    metadata = _metadata(
        family="resnet",
        task="classify",
        size="18",
        io=io,
    )
    metadata.pop("classification_activation")

    with pytest.raises(ValueError, match="classification_activation"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=_spec(outputs=("class_logits",)),
        )


def _gaze_io_contract():
    return _profile_io("l2cs", "gaze", "resnet50")


def _gaze_metadata(*, num_bins=66):
    metadata = _metadata(
        family="l2cs",
        task="gaze",
        size="resnet50",
        names={0: "gaze"},
        io=_gaze_io_contract(),
    )
    metadata.update(
        {
            "num_bins": str(num_bins),
            "bin_width_deg": "3.0",
            "offset_deg": "-99.0",
            "gaze_input": "face_crop",
        }
    )
    return metadata


def test_gaze_metadata_preserves_non_default_bin_geometry(monkeypatch, tmp_path):
    logits = np.full((1, 66), -100.0, dtype=np.float32)
    logits[0, 10] = 100.0
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=_gaze_metadata(),
        spec=_spec(outputs=("yaw_logits", "pitch_logits")),
        predict_fn=lambda _inputs: {
            "yaw_logits": logits,
            "pitch_logits": logits,
        },
    )

    outputs = backend._run_inference(np.zeros((1, 3, 4, 4), dtype=np.float32))
    result = backend._build_gaze_result(
        outputs,
        orig_shape=(4, 4),
        image_path=None,
    )

    assert backend.num_bins == 66
    assert backend.bin_width_deg == 3.0
    assert backend.offset_deg == -99.0
    assert float(result.gaze.yaw_deg[0]) == pytest.approx(-69.0, abs=1e-4)
    assert float(result.gaze.pitch_deg[0]) == pytest.approx(-69.0, abs=1e-4)


def test_strict_gaze_artifact_requires_geometry_metadata(monkeypatch, tmp_path):
    metadata = _gaze_metadata()
    metadata.pop("num_bins")

    with pytest.raises(ValueError, match="complete gaze metadata"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=_spec(outputs=("yaw_logits", "pitch_logits")),
        )


def test_gaze_runtime_width_must_match_num_bins(monkeypatch, tmp_path):
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=_gaze_metadata(),
        spec=_spec(outputs=("yaw_logits", "pitch_logits")),
        predict_fn=lambda _inputs: {
            "yaw_logits": np.zeros((1, 90), dtype=np.float32),
            "pitch_logits": np.zeros((1, 90), dtype=np.float32),
        },
    )

    with pytest.raises(RuntimeError, match="width num_bins=66"):
        backend._run_inference(np.zeros((1, 3, 4, 4), dtype=np.float32))


def test_spec_output_mismatch_fails_at_load(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="outputs do not match"):
        _load_backend(
            monkeypatch,
            tmp_path,
            spec=_spec(outputs=("wrong_name",)),
        )


@pytest.mark.parametrize("artifact_format", [None, "onnx"])
def test_strict_artifact_format_is_required(
    monkeypatch,
    tmp_path,
    artifact_format,
):
    metadata = _metadata()
    if artifact_format is None:
        metadata.pop("artifact_format")
    else:
        metadata["artifact_format"] = artifact_format

    match = (
        "missing required metadata" if artifact_format is None else "artifact_format"
    )
    with pytest.raises(ValueError, match=match):
        _load_backend(monkeypatch, tmp_path, metadata=metadata)


def test_unidentified_coreml_package_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="not a recognized LibreYOLO package"):
        _load_backend(monkeypatch, tmp_path, metadata={"author": "someone"})


def test_fixed_artifact_rejects_runtime_imgsz_override(monkeypatch, tmp_path):
    backend, _model = _load_backend(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="fixed input canvas"):
        backend._resolve_predict_imgsz(8)


def test_strict_artifact_rejects_runtime_task_override(monkeypatch, tmp_path):
    metadata = _metadata(task="detect")
    metadata["supported_tasks"] = json.dumps(["detect", "segment"])

    with pytest.raises(ValueError, match="task-specific graph"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            backend_kwargs={"task": "segment"},
        )


def test_coreml_rectangular_validation_requires_declared_geometry(
    monkeypatch,
    tmp_path,
):
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=_metadata(
            family="yolo9",
            size="t",
            imgsz=(2, 4),
            io=_profile_io("yolo9", "detect", "t"),
        ),
        spec=_spec(imgsz=(2, 4)),
    )
    captured = {}

    class _Validator:
        def __init__(self, model, config):
            captured["model"] = model
            captured["imgsz"] = config.imgsz

        def __call__(self):
            return {"metrics/mAP50": 0.5}

    monkeypatch.setattr("libreyolo.validation.DetectionValidator", _Validator)
    metrics = backend.val(
        data="unused.yaml",
        workers=0,
        device="cpu",
        verbose=False,
    )

    assert metrics == {"metrics/mAP50": 0.5}
    assert captured["model"] is backend
    assert captured["imgsz"] == (2, 4)


def test_dense_validation_inherits_exact_artifact_resize_contract(
    monkeypatch,
    tmp_path,
):
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    depth_backend, _model = _load_backend(
        monkeypatch,
        depth_dir,
        metadata=_metadata(
            family="depth_anything",
            task="depth",
            size="s",
            io=_profile_io("depth_anything", "depth", "s"),
        ),
        spec=_spec(outputs=("depth",)),
    )
    assert depth_backend.depth_resize_mode == "stretch"
    assert depth_backend.depth_resize_backend == "opencv"
    assert depth_backend.depth_resize_interpolation == "bicubic"

    semantic_dir = tmp_path / "semantic"
    semantic_dir.mkdir()
    semantic_backend, _model = _load_backend(
        monkeypatch,
        semantic_dir,
        metadata=_metadata(
            family="pidnet",
            task="semantic",
            size="s",
            io=_profile_io("pidnet", "semantic", "s"),
        ),
        spec=_spec(outputs=("semantic_logits",)),
    )
    assert semantic_backend.semantic_resize_mode == "letterbox"
    assert semantic_backend.semantic_resize_backend == "opencv"
    assert semantic_backend.semantic_resize_interpolation == "bilinear"
    assert semantic_backend.semantic_resize_rounding == "floor"


def test_rfdetr_pose_tensor_resize_matches_native_antialiased_float_path(
    monkeypatch,
    tmp_path,
):
    import torch.nn.functional as F

    io = _profile_io("rfdetr", "pose", "n")
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=_metadata(
            family="rfdetr",
            task="pose",
            size="n",
            imgsz=64,
            io=io,
        ),
        spec=_spec(
            input_kind="multiArrayType",
            outputs=("pred_boxes", "pred_logits", "pred_keypoints"),
            imgsz=64,
        ),
    )
    yy, xx = np.mgrid[:37, :61]
    rgb = np.stack(
        (
            (xx * 17 + yy * 3) % 256,
            (xx * 5 + yy * 29) % 256,
            (xx * 31 + yy * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    source = (
        torch.from_numpy(rgb.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
    )
    expected = F.interpolate(
        source,
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    predict_blob = backend._preprocess(Image.fromarray(rgb), 64, "rgb")[0].div(255.0)
    torch.testing.assert_close(predict_blob, expected, rtol=0.0, atol=6e-8)

    preprocessor = backend._get_val_preprocessor(img_size=64)
    val_chw, _targets = preprocessor(
        rgb[:, :, ::-1].copy(),
        np.zeros((0, 5), dtype=np.float32),
        (64, 64),
    )
    torch.testing.assert_close(
        torch.from_numpy(val_chw).unsqueeze(0).div(255.0),
        expected,
        rtol=0.0,
        atol=6e-8,
    )


@pytest.mark.parametrize("missing_key", ["num_keypoints", "keypoint_dim"])
def test_strict_pose_requires_parser_metadata(
    monkeypatch,
    tmp_path,
    missing_key,
):
    metadata = _metadata(
        family="rfdetr",
        task="pose",
        size="n",
        imgsz=64,
        io=_profile_io("rfdetr", "pose", "n"),
    )
    metadata.pop(missing_key)

    with pytest.raises(ValueError, match="pose metadata"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=_spec(
                input_kind="multiArrayType",
                outputs=("pred_boxes", "pred_logits", "pred_keypoints"),
                imgsz=64,
            ),
        )


def test_rfdetr_grouppose_cannot_drop_class_schema(monkeypatch, tmp_path):
    metadata = _metadata(
        family="rfdetr",
        task="pose",
        size="n",
        imgsz=64,
        io=_profile_io("rfdetr", "pose", "n"),
    )
    metadata.update(
        {
            "num_keypoints": "17",
            "keypoint_dim": "8",
            "pose_encoding": "rfdetr_grouppose_padded_v1",
        }
    )
    # If num_keypoints_per_class is absent, this cannot silently fall back to
    # the classic flattened-keypoint parser.
    metadata.pop("num_keypoints_per_class", None)

    with pytest.raises(ValueError, match="pose_encoding"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=_spec(
                input_kind="multiArrayType",
                outputs=("pred_boxes", "pred_logits", "pred_keypoints"),
                imgsz=64,
            ),
        )


@pytest.mark.parametrize(
    "invalid_schema",
    [
        [0, True],
        [0, 17.5],
        [0, {"count": 17}],
        [0, [17]],
    ],
)
def test_rfdetr_grouppose_rejects_noninteger_schema_items(
    monkeypatch,
    tmp_path,
    invalid_schema,
):
    metadata = _metadata(
        family="rfdetr",
        task="pose",
        size="n",
        imgsz=64,
        io=_profile_io("rfdetr", "pose", "n"),
    )
    metadata.update(
        {
            "num_keypoints": "17",
            "keypoint_dim": "8",
            "num_keypoints_per_class": json.dumps(invalid_schema),
            "pose_encoding": "rfdetr_grouppose_padded_v1",
        }
    )

    with pytest.raises(ValueError, match="nonnegative integers"):
        _load_backend(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=_spec(
                input_kind="multiArrayType",
                outputs=("pred_boxes", "pred_logits", "pred_keypoints"),
                imgsz=64,
            ),
        )


def test_yolo9_opencv_resize_matches_native_preprocessor_pixels(
    monkeypatch,
    tmp_path,
):
    input_size = (13, 17)
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=_metadata(
            family="yolo9",
            imgsz=input_size,
            io=_profile_io("yolo9", "detect", "t"),
        ),
        spec=_spec(imgsz=input_size),
    )
    y, x = np.mgrid[:7, :11]
    rgb = np.stack(
        (
            (x * 17 + y * 3) % 256,
            (x * 5 + y * 29) % 256,
            (x * 31 + y * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)

    actual, _original, _size, _ratio = backend._preprocess(
        Image.fromarray(rgb),
        input_size,
        "rgb",
    )

    from libreyolo.models.yolo9.utils import preprocess_numpy

    expected, _native_ratio = preprocess_numpy(rgb, input_size)
    actual_pixels = actual.numpy()[0].astype(np.uint8)
    expected_pixels = np.rint(expected * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(actual_pixels, expected_pixels)


@pytest.mark.parametrize("task", ["detect", "pose"])
def test_yolonas_resize_cap_and_padding_match_native_preprocess(
    monkeypatch,
    tmp_path,
    task,
):
    from libreyolo.export.coreml_yolonas import (
        yolonas_coreml_input_contract,
        yolonas_coreml_output_contract,
        yolonas_coreml_validation_contract,
    )

    output_contract = yolonas_coreml_output_contract(task)
    io = {
        "input": yolonas_coreml_input_contract(task),
        "validation": yolonas_coreml_validation_contract(task),
        "outputs": output_contract,
    }
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=_metadata(
            family="yolonas",
            task=task,
            size="s",
            imgsz=64,
            io=io,
        ),
        spec=_spec(
            input_name=io["input"]["name"],
            outputs=tuple(item["name"] for item in output_contract),
            imgsz=64,
        ),
    )
    yy, xx = np.mgrid[:31, :53]
    rgb = np.stack(
        (
            (xx * 17 + yy * 3) % 256,
            (xx * 5 + yy * 29) % 256,
            (xx * 31 + yy * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)

    actual, _original, _size, actual_ratio = backend._preprocess(
        Image.fromarray(rgb),
        64,
        "rgb",
    )

    if task == "pose":
        from libreyolo.models.yolonas.utils import preprocess_pose_image

        expected, _image, _size, expected_ratio = preprocess_pose_image(
            Image.fromarray(rgb),
            input_size=64,
            color_format="rgb",
        )
        actual_graph_input = actual[:, [2, 1, 0]].div(255.0)
    else:
        from libreyolo.models.yolonas.utils import preprocess_image

        expected, _image, _size, expected_ratio = preprocess_image(
            Image.fromarray(rgb),
            input_size=64,
            color_format="rgb",
        )
        actual_graph_input = actual.div(255.0)

    torch.testing.assert_close(actual_graph_input, expected, rtol=0, atol=0)
    assert actual_ratio == expected_ratio


def test_yolo1_stretch_geometry_uses_independent_xy_inverse(
    monkeypatch,
    tmp_path,
):
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=_metadata(
            family="yolo1",
            size="t",
            io=_profile_io("yolo1", "detect", "t"),
        ),
    )
    prediction = np.zeros((1, 5, 1), dtype=np.float32)
    prediction[0, :4, 0] = [0.0, 1.0, 2.0, 3.0]
    prediction[0, 4, 0] = 0.9

    boxes, scores, classes, masks = backend._parse_outputs(
        [prediction],
        4,
        (8, 2),
        conf=0.25,
    )

    np.testing.assert_allclose(boxes, [[0.0, 0.5, 4.0, 1.5]])
    np.testing.assert_allclose(scores, [0.9])
    np.testing.assert_array_equal(classes, [0])
    assert masks is None


def test_fixed_native_restore_rejects_smaller_source(monkeypatch, tmp_path):
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=_metadata(
            family="nafnet",
            task="restore",
            size="l",
            io=_profile_io("nafnet", "restore", "l"),
        ),
        spec=_spec(outputs=("restored",)),
    )

    with pytest.raises(ValueError, match="match the exported canvas"):
        backend._preprocess(
            Image.new("RGB", (3, 4)),
            4,
            "rgb",
        )


def test_yolo9_embedded_nms_uses_letterbox_inverse_on_non_square_image(
    monkeypatch,
    tmp_path,
):
    io = _profile_io("yolo9", "detect", "t", nms=True)
    metadata = _metadata(family="yolo9", size="t", io=io)
    metadata["nms"] = "true"
    metadata["nms_conf"] = "0.25"
    metadata["nms_iou"] = "0.45"
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=_spec(outputs=("confidence", "coordinates")),
    )
    confidence = np.array([[[0.9]]], dtype=np.float32)
    # Original image is 8x2, fixed canvas is 4x4. The letterboxed content is
    # 4x1 at the top left; this box spans half its width and its full height.
    coordinates = np.array([[[1.0, 0.5, 2.0, 1.0]]], dtype=np.float32)

    boxes, scores, classes, masks = backend._parse_outputs(
        [confidence, coordinates],
        4,
        (8, 2),
        conf=0.25,
    )

    np.testing.assert_allclose(boxes, [[0.0, 0.0, 4.0, 2.0]])
    np.testing.assert_allclose(scores, [0.9])
    np.testing.assert_array_equal(classes, [0])
    assert masks is None


@pytest.mark.parametrize(
    ("family", "size", "crop_pct", "interpolation"),
    [
        ("resnet", "18", 0.95, "bicubic"),
        ("convnext", "t", 0.875, "bicubic"),
        ("dinov2", "n", 0.875, "bilinear"),
    ],
)
def test_center_crop_geometry_matches_native_classify_transform(
    monkeypatch,
    tmp_path,
    family,
    size,
    crop_pct,
    interpolation,
):
    from libreyolo.data.classify_dataset import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        build_classify_transforms,
    )

    io = _profile_io(family, "classify", size)
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=_metadata(
            family=family,
            task="classify",
            size=size,
            names={0: "a", 1: "b"},
            imgsz=17,
            io=io,
        ),
        spec=_spec(outputs=("class_logits",), imgsz=17),
    )
    yy, xx = np.mgrid[:19, :31]
    rgb = np.stack(
        (
            (xx * 7 + yy * 3) % 256,
            (xx * 5 + yy * 11) % 256,
            (xx * 13 + yy * 2) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    actual = backend._preprocess(rgb, 17, "rgb")[0][0].numpy()

    native = build_classify_transforms(
        17,
        augment=False,
        crop_pct=crop_pct,
        interpolation=interpolation,
    )(Image.fromarray(rgb))
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    expected = torch.round((native * std + mean) * 255.0).numpy()

    np.testing.assert_array_equal(actual, expected)


def test_restore_validation_rejects_smaller_sample_hidden_by_batch_padding():
    from libreyolo.validation import RestoreValidator

    validator = object.__new__(RestoreValidator)
    validator.model = SimpleNamespace(
        imgsz=64,
        input_contract=SimpleNamespace(geometry="native"),
    )
    validator.config = SimpleNamespace(imgsz=64)
    images = torch.zeros(2, 3, 64, 64)
    targets = torch.zeros(2, 3, 64, 64)
    img_info = [
        {"orig_shape": (64, 64), "target_shape": (64, 64)},
        {"orig_shape": (48, 40), "target_shape": (48, 40)},
    ]
    batch = (images, targets, img_info, [0, 1])

    with pytest.raises(ValueError, match="every low-resolution input"):
        validator._preprocess_batch(batch)


def test_rectangular_native_restore_routes_to_validator(monkeypatch, tmp_path):
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=_metadata(
            family="nafnet",
            task="restore",
            size="l",
            imgsz=(32, 64),
            io=_profile_io("nafnet", "restore", "l"),
        ),
        spec=_spec(outputs=("restored",), imgsz=(32, 64)),
    )
    captured = {}

    class _Validator:
        def __init__(self, model, config):
            captured["model"] = model
            captured["imgsz"] = config.imgsz

        def __call__(self):
            return {"metrics/PSNR": 30.0}

    monkeypatch.setattr("libreyolo.validation.RestoreValidator", _Validator)
    metrics = backend.val(data="unused.yaml", workers=0, verbose=False)

    assert metrics == {"metrics/PSNR": 30.0}
    assert captured == {"model": backend, "imgsz": (32, 64)}


def test_yolox_validation_forward_uses_canonical_rgb_once(monkeypatch, tmp_path):
    io = _profile_io("yolox", "detect", "n")
    captured = {}

    def predict(inputs):
        image = inputs["image"]
        captured["array"] = np.asarray(image)
        return {"prediction": np.zeros((1, 1, 6), dtype=np.float32)}

    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=_metadata(family="yolox", size="n", io=io),
        predict_fn=predict,
    )
    preprocessor = backend._get_val_preprocessor(img_size=4)
    # Validation datasets hand detection preprocessors BGR uint8.
    bgr = np.full((2, 4, 3), [10, 20, 30], dtype=np.uint8)
    chw, _targets = preprocessor(
        bgr,
        np.zeros((0, 5), dtype=np.float32),
        (4, 4),
    )
    output = backend._forward(torch.from_numpy(chw).unsqueeze(0))

    assert tuple(output[0].shape) == (1, 1, 6)
    np.testing.assert_array_equal(captured["array"][0, 0], [30, 20, 10])
    np.testing.assert_array_equal(captured["array"][3, 0], [114, 114, 114])


def test_backend_val_routes_canonical_tensor_through_forward(
    monkeypatch,
    tmp_path,
):
    captured = {}

    def predict(inputs):
        captured["array"] = np.asarray(inputs["image"])
        return {"prediction": np.zeros((1, 1, 6), dtype=np.float32)}

    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=_metadata(
            family="yolox",
            size="n",
            io=_profile_io("yolox", "detect", "n"),
        ),
        predict_fn=predict,
    )

    class _Validator:
        def __init__(self, model, config):
            self.model = model
            self.config = config

        def __call__(self):
            preprocessor = self.model._get_val_preprocessor(self.config.imgsz)
            bgr = np.full((2, 4, 3), [10, 20, 30], dtype=np.uint8)
            chw, _targets = preprocessor(
                bgr,
                np.zeros((0, 5), dtype=np.float32),
                (4, 4),
            )
            outputs = self.model._forward(torch.from_numpy(chw).unsqueeze(0))
            assert tuple(outputs[0].shape) == (1, 1, 6)
            return {"metrics/mAP50": 0.5}

    monkeypatch.setattr("libreyolo.validation.DetectionValidator", _Validator)
    metrics = backend.val(
        data="unused.yaml",
        batch=1,
        workers=0,
        device="cpu",
        verbose=False,
    )

    assert metrics == {"metrics/mAP50": 0.5}
    np.testing.assert_array_equal(captured["array"][0, 0], [30, 20, 10])


@pytest.mark.parametrize(
    ("family", "size", "validation_range", "mean_values", "std_values"),
    [
        (
            "resnet",
            "18",
            "imagenet",
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
        (
            "clip",
            "b32",
            "standardized",
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ],
)
def test_classifier_validation_unnormalizes_to_canonical_rgb(
    monkeypatch,
    tmp_path,
    family,
    size,
    validation_range,
    mean_values,
    std_values,
):
    io = _profile_io(family, "classify", size)
    captured = {}

    def predict(inputs):
        captured["array"] = np.asarray(inputs["image"])
        return {"class_logits": np.array([[1.0, 2.0]], dtype=np.float32)}

    metadata = _metadata(
        family=family,
        task="classify",
        size=size,
        names={0: "a", 1: "b"},
        io=io,
    )
    if family == "clip":
        metadata["frozen_classes"] = "true"
    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=metadata,
        spec=_spec(outputs=("class_logits",)),
        predict_fn=predict,
    )
    rgb = np.zeros((1, 3, 4, 4), dtype=np.float32)
    rgb[:, 0] = 64.0
    rgb[:, 1] = 128.0
    rgb[:, 2] = 192.0
    mean = np.asarray(mean_values, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(std_values, dtype=np.float32).reshape(1, 3, 1, 1)
    normalized = (rgb / 255.0 - mean) / std

    output = backend._forward(torch.from_numpy(normalized))

    assert tuple(output[0].shape) == (1, 2)
    np.testing.assert_array_equal(captured["array"][0, 0], [64, 128, 192])


def test_fomo_validation_minus_one_to_one_round_trip(monkeypatch, tmp_path):
    io = _profile_io("fomo", "point", "s")
    captured = {}

    def predict(inputs):
        captured["array"] = np.asarray(inputs["image"])
        return {
            "point_logits": np.zeros((1, 2, 2, 2), dtype=np.float32),
        }

    backend, _model = _load_backend(
        monkeypatch,
        tmp_path,
        metadata=_metadata(
            family="fomo",
            task="point",
            size="s",
            names={0: "point"},
            io=io,
        ),
        spec=_spec(outputs=("point_logits",)),
        predict_fn=predict,
    )
    preprocessor = backend._get_val_preprocessor(img_size=4)
    bgr = np.full((4, 4, 3), [255, 127, 0], dtype=np.uint8)
    chw, _targets = preprocessor(
        bgr,
        np.zeros((0, 5), dtype=np.float32),
        (4, 4),
    )
    output = backend._forward(torch.from_numpy(chw).unsqueeze(0))

    assert tuple(output[0].shape) == (1, 2, 2, 2)
    # BGR [255, 127, 0] -> RGB [0, 127, 255], then [-1, 1] inversion.
    np.testing.assert_array_equal(captured["array"][0, 0], [0, 127, 255])
