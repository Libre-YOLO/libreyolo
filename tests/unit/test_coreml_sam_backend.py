"""CPU geometry and fake-runtime coverage for split SAM Core ML packages."""

from __future__ import annotations

import json
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.sam]


def _profile(family="mobilesam", size="tiny", prompt_max_points=4):
    from libreyolo.export.coreml_sam import validate_sam_coreml_profile

    return validate_sam_coreml_profile(
        family=family,
        size=size,
        prompt_max_points=prompt_max_points,
    )


def _image(width=37, height=31):
    values = np.arange(height * width * 3, dtype=np.uint32)
    values = (values.reshape(height, width, 3) % 251).astype(np.uint8)
    return Image.fromarray(values, mode="RGB")


def test_mobilesam_host_geometry_is_exactly_native():
    from libreyolo.backends.coreml_sam import (
        postprocess_sam_coreml_masks,
        prepare_sam_coreml_image,
        transform_sam_coreml_box,
        transform_sam_coreml_points,
    )
    from libreyolo.models.mobilesam.preprocess import (
        encode_image_and_prompts,
        postprocess_masks,
        preprocess_tensor,
    )

    image = _image()
    profile = _profile()
    actual = prepare_sam_coreml_image(image, profile=profile)
    native = encode_image_and_prompts(
        image,
        target_length=1024,
        points=[[[3.25, 4.75], [20.5, 22.25]]],
        labels=[[1, 0]],
        boxes=[[2.5, 3.5, 30.25, 28.75]],
    )
    expected_pixels = preprocess_tensor(
        native["pixel_values"],
        image_size=1024,
        pixel_mean=torch.tensor(
            [123.675, 116.28, 103.53],
            dtype=torch.float32,
        ).reshape(1, 3, 1, 1),
        pixel_std=torch.tensor(
            [58.395, 57.12, 57.375],
            dtype=torch.float32,
        ).reshape(1, 3, 1, 1),
    )
    assert torch.equal(actual.pixel_values, expected_pixels)
    assert actual.original_size == (31, 37)
    assert actual.reshaped_input_size == (858, 1024)

    points = transform_sam_coreml_points(
        [[3.25, 4.75], [20.5, 22.25]],
        encoding=actual,
        profile=profile,
    )
    box = transform_sam_coreml_box(
        [2.5, 3.5, 30.25, 28.75],
        encoding=actual,
        profile=profile,
    )
    assert torch.equal(points, native["input_points"])
    assert torch.equal(box, native["input_boxes"].reshape(1, 1, 4))

    low = torch.linspace(-2.0, 2.0, 256 * 256).reshape(1, 1, 1, 256, 256)
    actual_masks = postprocess_sam_coreml_masks(
        low,
        encoding=actual,
        profile=profile,
    )
    expected_masks = postprocess_masks(
        low[0],
        image_size=1024,
        input_size=(858, 1024),
        original_size=(31, 37),
        mask_threshold=0.0,
    )
    assert torch.equal(actual_masks, expected_masks[0])


def test_fast_sam2_uint8_resize_is_not_edgetam_float_resize():
    from libreyolo.backends.coreml_sam import prepare_sam_coreml_image

    image = _image()
    sam2 = prepare_sam_coreml_image(
        image,
        profile=_profile("sam2", "tiny"),
    )
    edgetam = prepare_sam_coreml_image(
        image,
        profile=_profile("edgetam", "edge"),
    )
    assert sam2.pixel_values.shape == edgetam.pixel_values.shape
    assert not torch.equal(sam2.pixel_values, edgetam.pixel_values)


@pytest.mark.parametrize(
    ("family", "size", "processor_name"),
    [
        ("sam2", "tiny", "Sam2ImageProcessorFast"),
        ("sam3", "large", "Sam3ImageProcessorFast"),
    ],
)
def test_fast_square_pixels_are_bit_exact_with_pinned_processor_contract(
    family,
    size,
    processor_name,
):
    transformers = pytest.importorskip("transformers")
    processor_type = getattr(transformers, processor_name, None)
    if processor_type is None:
        pytest.skip(f"{processor_name} is unavailable in this Transformers build")

    from libreyolo.backends.coreml_sam import prepare_sam_coreml_image

    image = _image()
    actual = prepare_sam_coreml_image(
        image,
        profile=_profile(family, size),
    ).pixel_values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        expected = processor_type()(
            images=image,
            return_tensors="pt",
        )["pixel_values"]
    assert torch.equal(actual, expected)


def test_sam1_pixels_match_transformers_5_3_contract_when_available():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.3.0":
        pytest.skip("SAM-1 v1 is deliberately pinned to Transformers 5.3.0")

    from libreyolo.backends.coreml_sam import prepare_sam_coreml_image

    image = _image()
    actual = prepare_sam_coreml_image(
        image,
        profile=_profile("sam", "base"),
    ).pixel_values
    expected = transformers.SamImageProcessor()(
        images=image,
        return_tensors="pt",
    )["pixel_values"]
    assert torch.equal(actual, expected)


def test_mask_contract_uses_direct_resize_and_strict_positive_threshold():
    from libreyolo.backends.coreml_sam import (
        SAMCoreMLImageEncoding,
        postprocess_sam_coreml_masks,
    )

    profile = _profile("edgetam", "edge")
    encoding = SAMCoreMLImageEncoding(
        pixel_values=torch.zeros(1, 3, 1024, 1024),
        original_size=(3, 5),
        reshaped_input_size=None,
    )
    low = torch.zeros(1, 1, 1, 256, 256)
    assert not postprocess_sam_coreml_masks(
        low,
        encoding=encoding,
        profile=profile,
    ).any()
    low[..., 128, 128] = 1.0
    assert postprocess_sam_coreml_masks(
        low,
        encoding=encoding,
        profile=profile,
    ).any()


def _stringify(metadata):
    return {
        str(key): (
            json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list, tuple))
            else str(value)
        )
        for key, value in metadata.items()
    }


def _metadata():
    from libreyolo.export.coreml_sam import sam_coreml_metadata

    profile = _profile()
    metadata = {
        "schema_version": "1.0",
        "libreyolo_version": "0.test",
        "libreyolo_producer": "libreyolo",
        "artifact_format": "coreml",
        "model_family": "mobilesam",
        "size": "tiny",
        "model_size": "tiny",
        "task": "segment",
        "supported_tasks": ["segment"],
        "default_task": "segment",
        "names": {"0": "object"},
        "nc": 1,
        "nb_classes": 1,
        "imgsz": 1024,
        "imgsz_h": 1024,
        "imgsz_w": 1024,
        "precision": "fp32",
        "dynamic": True,
    }
    metadata.update(sam_coreml_metadata(profile))
    return _stringify(metadata)


def _feature(contract):
    dtype = 131104 if contract["dtype"] == "int32" else 65568
    ranges = []
    has_range = False
    default_shape = []
    for axis in contract["shape"]:
        if axis["kind"] == "fixed":
            lower = upper = default = int(axis["value"])
        else:
            lower = int(axis["lower_bound"])
            upper = int(axis["upper_bound"])
            default = int(axis["default"])
            has_range = True
        ranges.append(
            SimpleNamespace(lowerBound=lower, upperBound=upper)
        )
        default_shape.append(default)
    array = SimpleNamespace(
        shape=[] if has_range else default_shape,
        dataType=dtype,
        shapeRange=SimpleNamespace(sizeRanges=ranges),
        WhichOneof=lambda _name: "shapeRange" if has_range else None,
    )
    return SimpleNamespace(
        name=contract["name"],
        type=SimpleNamespace(multiArrayType=array),
    )


def _spec():
    from libreyolo.export.coreml_sam import sam_coreml_function_contracts

    functions = []
    for name, contract in sam_coreml_function_contracts(_profile()).items():
        functions.append(
            SimpleNamespace(
                name=name,
                input=[_feature(item) for item in contract["inputs"]],
                output=[_feature(item) for item in contract["outputs"]],
            )
        )
    return SimpleNamespace(
        specificationVersion=9,
        description=SimpleNamespace(
            input=[],
            output=[],
            functions=functions,
            defaultFunctionName="encode_image",
        ),
    )


class _Runtime:
    def __init__(self, metadata, spec, function_name, calls):
        self.user_defined_metadata = metadata
        self._spec = spec
        self.function_name = function_name
        self.calls = calls

    def get_spec(self):
        return self._spec

    def predict(self, inputs):
        self.calls.append((self.function_name, inputs))
        if self.function_name == "encode_image":
            signal = float(np.asarray(inputs["pixel_values"]).mean())
            return {
                "image_embedding": np.full(
                    (1, 256, 64, 64),
                    signal,
                    dtype=np.float32,
                )
            }
        mask_count = 3 if self.function_name.endswith("multimask") else 1
        masks = np.ones((1, 1, mask_count, 256, 256), dtype=np.float32)
        scores = np.linspace(
            0.5,
            0.7,
            mask_count,
            dtype=np.float32,
        ).reshape(1, 1, mask_count)
        return {"low_res_masks": masks, "iou_scores": scores}


def _load(monkeypatch, tmp_path, *, metadata=None, spec=None):
    metadata = metadata or _metadata()
    spec = spec or _spec()
    calls = []
    loads = []

    def load_model(_path, **kwargs):
        function_name = kwargs.get("function_name") or "encode_image"
        loads.append(function_name)
        return _Runtime(metadata, spec, function_name, calls)

    fake_ct = SimpleNamespace(
        ComputeUnit=SimpleNamespace(
            ALL="ALL",
            CPU_AND_GPU="CPU_AND_GPU",
            CPU_AND_NE="CPU_AND_NE",
            CPU_ONLY="CPU_ONLY",
        ),
        models=SimpleNamespace(MLModel=load_model),
    )
    monkeypatch.setitem(sys.modules, "coremltools", fake_ct)
    monkeypatch.setattr(sys, "platform", "darwin")
    package = tmp_path / "sam.mlpackage"
    package.mkdir()

    from libreyolo.backends.coreml import CoreMLBackend

    return CoreMLBackend(str(package)), calls, loads


def test_multifunction_backend_caches_encoder_and_loops_queries(
    monkeypatch,
    tmp_path,
):
    backend, calls, loads = _load(monkeypatch, tmp_path)
    assert loads == [
        "encode_image",
        "encode_image",
        "decode_points_single",
        "decode_points_multimask",
        "decode_boxes_single",
        "decode_boxes_multimask",
        "decode_points_boxes_single",
        "decode_points_boxes_multimask",
    ]
    image = _image()
    backend.set_image(image)
    assert [name for name, _inputs in calls] == ["encode_image"]

    result = backend.predict(points=[[3, 4], [10, 12]], labels=[1, 0])
    assert len(result) == 2
    assert [name for name, _inputs in calls[1:]] == [
        "decode_points_single",
        "decode_points_single",
    ]
    for _name, inputs in calls[1:]:
        assert inputs["point_labels"].dtype == np.int32
        assert inputs["point_coords"].dtype == np.float32
        assert inputs["point_coords"].shape == (1, 1, 1, 2)

    backend.predict(bboxes=[1, 2, 20, 25], multimask=True)
    assert calls[-1][0] == "decode_boxes_multimask"
    assert len(calls) == 4
    backend.reset_image()
    with pytest.raises(RuntimeError, match="No image set"):
        backend.predict(points=[3, 4])


def test_invalid_prompt_fails_before_any_vendor_runtime_call(
    monkeypatch,
    tmp_path,
):
    backend, calls, _loads = _load(monkeypatch, tmp_path)
    image = _image()
    with pytest.raises(ValueError, match="source image bounds"):
        backend.predict(image, points=[1000, 2])
    assert calls == []

    too_many = [[[float(index), 2.0] for index in range(5)]]
    with pytest.raises(ValueError, match="prompt_max_points=4"):
        backend.predict(image, points=too_many)
    assert calls == []


def test_explicit_source_does_not_replace_cached_session(monkeypatch, tmp_path):
    backend, calls, _loads = _load(monkeypatch, tmp_path)
    cached = _image()
    backend.set_image(cached)
    other = Image.new("RGB", (19, 17), color=(50, 80, 110))
    backend.predict(other, points=[3, 4])
    backend.predict(points=[3, 4])
    assert [name for name, _inputs in calls] == [
        "encode_image",
        "encode_image",
        "decode_points_single",
        "decode_points_single",
    ]


def test_promptable_sam_package_rejects_generic_dataset_validation(
    monkeypatch,
    tmp_path,
):
    backend, calls, _loads = _load(monkeypatch, tmp_path)
    with pytest.raises(
        NotImplementedError,
        match="promptable.*no fixed class-set validation",
    ):
        backend.val(data="does-not-matter.yaml")
    assert calls == []


def test_unknown_generic_multifunction_marker_is_not_laundered_as_ppocr(
    monkeypatch,
    tmp_path,
):
    metadata = _metadata()
    metadata["model_family"] = "future"
    metadata["component_contract"] = "future_bundle_v1"
    with pytest.raises(ValueError, match="unknown multifunction"):
        _load(monkeypatch, tmp_path, metadata=metadata)
