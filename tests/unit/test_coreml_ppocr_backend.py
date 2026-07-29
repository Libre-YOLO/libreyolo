"""Strict loader/runtime tests for LibrePPOCR Core ML multifunction packages."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


def _stringify(metadata):
    return {
        str(key): (
            json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list, tuple))
            else str(value)
        )
        for key, value in metadata.items()
    }


def _metadata(*, rec_max_width=325, rec_batch_max=2):
    from libreyolo.export.coreml_ppocr import (
        ppocr_coreml_metadata,
        validate_ppocr_coreml_profile,
    )

    profile = validate_ppocr_coreml_profile(
        size="t",
        det_limit_side_len=64,
        rec_batch_max=rec_batch_max,
        rec_max_width=rec_max_width,
    )
    metadata = {
        "schema_version": "1.0",
        "libreyolo_version": "0.test",
        "libreyolo_producer": "libreyolo",
        "artifact_format": "coreml",
        "model_family": "ppocr",
        "size": "t",
        "model_size": "t",
        "task": "ocr",
        "supported_tasks": ["ocr"],
        "default_task": "ocr",
        "names": {"0": "text"},
        "nc": 1,
        "nb_classes": 1,
        "imgsz": 64,
        "imgsz_h": 64,
        "imgsz_w": 64,
        "precision": "fp32",
        "dynamic": True,
    }
    metadata.update(
        ppocr_coreml_metadata(
            profile=profile,
            charset=["blank", "a", "b", " "],
            pipeline={
                "det_limit_side_len": 64,
                "det_db_thresh": 0.3,
                "det_db_box_thresh": 0.6,
                "det_db_unclip_ratio": 1.5,
                "rec_image_shape": [3, 48, 320],
            },
            rec_num_classes=4,
        )
    )
    return _stringify(metadata)


def _feature(name, ranges, *, output=False):
    size_ranges = [
        SimpleNamespace(lowerBound=lower, upperBound=upper)
        for lower, upper in ranges
    ]
    array = SimpleNamespace(
        shape=[] if output else [lower for lower, _upper in ranges],
        dataType=65568,
        shapeRange=SimpleNamespace(sizeRanges=size_ranges),
        WhichOneof=lambda _name: None if output else "shapeRange",
    )
    return SimpleNamespace(
        name=name,
        type=SimpleNamespace(multiArrayType=array),
    )


def _spec(*, detector_input="detector_input", recognizer_output="ctc_probabilities"):
    detector = SimpleNamespace(
        name="detector",
        input=[
            _feature(
                detector_input,
                ((1, 1), (3, 3), (32, 64), (32, 64)),
            )
        ],
        output=[_feature("probability_map", (), output=True)],
    )
    recognizer = SimpleNamespace(
        name="recognizer",
        input=[
            _feature(
                "recognizer_input",
                ((1, 2), (3, 3), (48, 48), (320, 325)),
            )
        ],
        output=[_feature(recognizer_output, (), output=True)],
    )
    return SimpleNamespace(
        specificationVersion=9,
        description=SimpleNamespace(
            input=[],
            output=[],
            functions=[detector, recognizer],
            defaultFunctionName="detector",
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
        if self.function_name == "detector":
            value = np.asarray(inputs["detector_input"], dtype=np.float32)
            return {
                "probability_map": np.zeros(
                    (1, 1, value.shape[-2], value.shape[-1]),
                    dtype=np.float32,
                )
            }
        value = np.asarray(inputs["recognizer_input"], dtype=np.float32)
        timesteps = (value.shape[-1] + 3) // 8
        return {
            "ctc_probabilities": np.full(
                (value.shape[0], timesteps, 4),
                0.25,
                dtype=np.float32,
            )
        }


def _load(monkeypatch, tmp_path, *, metadata=None, spec=None):
    metadata = metadata or _metadata()
    spec = spec or _spec()
    calls = []
    loads = []

    def load_model(_path, **kwargs):
        function_name = kwargs.get("function_name") or "detector"
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
    package = tmp_path / "ocr.mlpackage"
    package.mkdir()

    from libreyolo.backends.coreml import CoreMLBackend

    backend = CoreMLBackend(str(package))
    return backend, calls, loads


def test_multifunction_loader_opens_both_functions_and_runs_named_outputs(
    monkeypatch,
    tmp_path,
):
    backend, calls, loads = _load(monkeypatch, tmp_path)

    assert backend.model_family == "ppocr"
    assert backend.task == "ocr"
    assert backend.charset == ["blank", "a", "b", " "]
    assert backend.rec_num_classes == 4
    assert loads == ["detector", "detector", "recognizer"]

    result = backend.predict(Image.new("RGB", (80, 40), color=(127, 63, 31)))
    assert result.ocr is not None
    assert len(result.ocr) == 0
    assert calls[0][0] == "detector"
    assert set(calls[0][1]) == {"detector_input"}

    recognizer_input = torch.zeros(2, 3, 48, 325, dtype=torch.float32)
    probabilities = backend._ppocr_runner_proxy.model.rec(recognizer_input)
    assert probabilities.shape == (2, 41, 4)
    assert calls[-1][0] == "recognizer"
    assert set(calls[-1][1]) == {"recognizer_input"}


def test_runtime_guards_stride_width_and_preallocation_before_predict(
    monkeypatch,
    tmp_path,
):
    backend, calls, _loads = _load(monkeypatch, tmp_path)
    detector = backend._ppocr_runner_proxy.model.det
    recognizer = backend._ppocr_runner_proxy.model.rec

    with pytest.raises(ValueError, match="stride-32"):
        detector(torch.zeros(1, 3, 33, 64))
    with pytest.raises(ValueError, match="overflow policy is 'error'"):
        recognizer(torch.zeros(1, 3, 48, 326))
    with pytest.raises(ValueError, match="crop 0 exceeds"):
        backend._ppocr_runner_proxy._validate_recognition_crops(
            [np.zeros((10, 200, 3), dtype=np.uint8)],
            1,
        )
    detector_nan = torch.zeros(1, 3, 32, 64)
    detector_nan[..., 0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        detector(detector_nan)
    with pytest.raises(ValueError, match=r"normalized to \[-1, 1\]"):
        recognizer(torch.full((1, 3, 48, 320), 1.01))
    assert calls == []


def test_public_profile_bounds_are_explicit(monkeypatch, tmp_path):
    backend, _calls, _loads = _load(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="rec_batch must be within"):
        backend.predict(Image.new("RGB", (32, 32)), rec_batch=3)
    with pytest.raises(ValueError, match="imgsz must be within"):
        backend.predict(Image.new("RGB", (32, 32)), imgsz=65)
    with pytest.raises(ValueError, match="processes source images sequentially"):
        backend.predict(Image.new("RGB", (32, 32)), batch=2)


def test_exported_ocr_validation_routes_to_ocr_validator(monkeypatch, tmp_path):
    backend, _calls, _loads = _load(monkeypatch, tmp_path)
    captured = {}

    class _Validator:
        def __init__(self, model, config):
            captured["model"] = model
            captured["config"] = config

        def __call__(self):
            return {"fitness": 0.75}

    import libreyolo.validation as validation

    monkeypatch.setattr(validation, "OCRValidator", _Validator)
    assert backend.val(data="ocr-dataset", batch=1) == {"fitness": 0.75}
    assert captured["model"] is backend
    assert captured["config"].data == "ocr-dataset"


def test_public_exporter_routes_bounded_multifunction_profile(
    monkeypatch,
    tmp_path,
):
    from libreyolo.export.exporter import CoreMLExporter
    from libreyolo.models.ppocr.model import LibrePPOCRModel

    class _Wrapper:
        def __init__(self):
            self.model = LibrePPOCRModel("t", num_classes=4).eval()
            self.task = "ocr"
            self.size = "t"
            self.nb_classes = 1
            self.names = {0: "text"}
            self.charset = ["blank", "a", "b", " "]
            self.pipeline_config = {
                "det_limit_side_len": 960,
                "det_db_thresh": 0.3,
                "det_db_box_thresh": 0.6,
                "det_db_unclip_ratio": 1.5,
                "rec_image_shape": [3, 48, 320],
            }
            self.device = torch.device("cpu")
            self.model_path = None

        @staticmethod
        def _get_model_name():
            return "ppocr"

        @staticmethod
        def _get_input_size():
            return 960

    monkeypatch.setitem(sys.modules, "coremltools", SimpleNamespace())
    wrapper = _Wrapper()
    exporter = CoreMLExporter(wrapper)
    captured = {}

    def _export(nn_model, dummy, **kwargs):
        captured["model"] = nn_model
        captured["dummy"] = dummy
        captured.update(kwargs)
        Path(kwargs["output_path"]).mkdir(parents=True)
        return kwargs["output_path"]

    monkeypatch.setattr(exporter, "_export", _export)
    output = tmp_path / "ppocr.mlpackage"
    with pytest.warns(RuntimeWarning, match="experimental"):
        result = exporter(
            imgsz=64,
            output_path=str(output),
            rec_batch_max=2,
            rec_max_width=325,
        )

    assert result == str(output)
    assert captured["dynamic"] is True
    assert captured["dummy"].shape == (1, 3, 64, 64)
    assert captured["rec_batch_max"] == 2
    assert captured["rec_max_width"] == 325
    assert captured["metadata"]["dynamic"] is True
    assert captured["metadata"]["charset"] == wrapper.charset
    assert captured["metadata"]["pipeline"]["det_limit_side_len"] == 64
    assert captured["metadata"]["rec_num_classes"] == 4

    with pytest.raises(ValueError, match="explicit finite rec_max_width"):
        exporter(imgsz=64, output_path=str(tmp_path / "missing.mlpackage"))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda metadata, spec: metadata.__setitem__("dynamic", "False"),
            "dynamic=true",
        ),
        (
            lambda metadata, spec: setattr(
                spec.description.functions[0].input[0],
                "name",
                "image",
            ),
            "invalid input interface",
        ),
        (
            lambda metadata, spec: setattr(spec, "specificationVersion", 8),
            "specification",
        ),
        (
            lambda metadata, spec: metadata.__setitem__(
                "names",
                json.dumps({"0": "not-text"}),
            ),
            r"exactly \{0: 'text'\}",
        ),
    ],
)
def test_loader_rejects_tampered_metadata_or_function_spec(
    monkeypatch,
    tmp_path,
    mutate,
    match,
):
    metadata = _metadata()
    spec = _spec()
    mutate(metadata, spec)
    with pytest.raises(ValueError, match=match):
        _load(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            spec=spec,
        )
