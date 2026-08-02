"""LiteRT FP32 measurements for a second ten-cell converter batch.

Every case uses deterministic LibreYOLO initialization, so this suite tests
conversion behavior rather than pretrained accuracy.  In particular, it does
not download or use the restricted SegFormer checkpoints.  Each case must
export, reload, preserve two input-sensitive raw probes, and match public
``predict()``.
"""

from __future__ import annotations

import gc
import importlib.util
import inspect
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from .test_tensorrt_round8 import (
    _assert_predict_parity as _assert_dense_predict_parity,
)
from .test_tensorrt_round8 import (
    _assert_raw_parity,
    _image,
    _prepare_non_degenerate_model,
)
from .test_tensorrt_round11 import _align_outputs, _rms
from .test_tensorrt_round12 import (
    _assert_predict_parity as _assert_detector_predict_parity,
)
from .test_tflite_round13 import _native_outputs

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("onnx2tf") is None
        or importlib.util.find_spec("ai_edge_litert") is None,
        reason="onnx2tf and ai-edge-litert are required",
    ),
]


@dataclass(frozen=True)
class TFLiteRound14Case:
    class_name: str
    family: str
    size: str
    task: str
    imgsz: int
    nb_classes: int = 3
    weights: str | None = None
    empty_checkpoint: bool = False


_CASES = (
    TFLiteRound14Case("LibreYOLO2", "yolo2", "t", "detect", 416, 5),
    TFLiteRound14Case(
        "LibreYOLO3",
        "yolo3",
        "t",
        "detect",
        416,
        80,
        weights="LibreYOLO3t.pt",
    ),
    TFLiteRound14Case("LibreYOLO4", "yolo4", "t", "detect", 416, 5),
    TFLiteRound14Case("LibreFOMO", "fomo", "s", "point", 96, 2),
    TFLiteRound14Case("LibreNAFNet", "nafnet", "s", "restore", 16, 1),
    TFLiteRound14Case(
        "LibreDepthAnythingV2",
        "depth_anything",
        "s",
        "depth",
        70,
        1,
    ),
    TFLiteRound14Case(
        "LibreSegformer",
        "segformer",
        "b0",
        "semantic",
        64,
        3,
    ),
    TFLiteRound14Case("LibreZipDepth", "zipdepth", "b", "depth", 64, 1),
    TFLiteRound14Case(
        "LibreRFDETR",
        "rfdetr",
        "n",
        "detect",
        384,
        3,
        empty_checkpoint=True,
    ),
    TFLiteRound14Case(
        "LibreRTDETRv4",
        "rtdetrv4",
        "s",
        "detect",
        256,
        3,
    ),
)

_HOLD_REASONS = {
    "yolo2-detect": "LiteRT rejects a 4,225-to-one RESHAPE.",
    "yolo3-detect": "Trained public top-k class membership changes.",
    "yolo4-detect": "Public boxes fall to 0 IoU with 176 px drift.",
    "fomo-point": "LiteRT reports zero DEPTHWISE_CONV_2D input channels.",
    "nafnet-restore": "LiteRT invoke finds an input tensor without data.",
    "depth_anything-depth": "LiteRT rejects incompatible ADD dimensions.",
    "segformer-semantic": "LiteRT rejects an invalid attention reshape.",
    "zipdepth-depth": "onnx2tf cannot lower edge-mode Pad.",
    "rfdetr-detect": "LiteRT rejects a STRIDED_SLICE input above rank five.",
    "rtdetrv4-detect": "onnx2tf crashes while lowering GatherElements.",
}

ROUND14_CASES = tuple(
    pytest.param(
        case,
        marks=pytest.mark.xfail(
            strict=True,
            reason=_HOLD_REASONS[f"{case.family}-{case.task}"],
        ),
    )
    for case in _CASES
)


def _build_model(case: TFLiteRound14Case):
    import libreyolo

    if case.weights:
        return libreyolo.LibreYOLO(case.weights, device="cuda")
    model_cls = getattr(libreyolo, case.class_name)
    parameters = inspect.signature(model_cls).parameters
    kwargs = {
        "model_path": {} if case.empty_checkpoint else None,
        "size": case.size,
        "nb_classes": case.nb_classes,
        "device": "cuda",
    }
    if "task" in parameters:
        kwargs["task"] = case.task
    return model_cls(**kwargs)


def _run_tflite_case(tmp_path, monkeypatch, case: TFLiteRound14Case) -> None:
    from libreyolo import LibreYOLO
    from libreyolo.export.support import SUPPORT, SupportEntry

    torch.manual_seed(14)
    model = _build_model(case)
    model.model.eval()
    _prepare_non_degenerate_model(case, model)
    assert model.FAMILY == case.family
    assert model.task == case.task

    device = next(model.model.parameters()).device
    first = torch.rand(1, 3, case.imgsz, case.imgsz, device=device)
    second = 1.0 - first
    expected_first, expected_second = _native_outputs(
        model,
        first,
        second,
        case.imgsz,
    )

    monkeypatch.setitem(
        SUPPORT,
        (case.family, case.task, "tflite"),
        SupportEntry(
            "experimental",
            "Round 14 measured probe bypasses the recorded conversion hold.",
        ),
    )
    artifact = model.export(
        format="tflite",
        output_path=str(tmp_path / f"{case.family}-{case.task}.tflite"),
        imgsz=case.imgsz,
        dynamic=False,
        half=False,
        simplify=False,
    )
    backend = LibreYOLO(artifact, device="cpu")

    actual_first = _align_outputs(
        expected_first,
        backend._run_inference(first.detach().cpu().numpy()),
    )
    actual_second = _align_outputs(
        expected_second,
        backend._run_inference(second.detach().cpu().numpy()),
    )
    if case.task != "detect":
        _assert_raw_parity(case, expected_first, actual_first)
        _assert_raw_parity(case, expected_second, actual_second)

    expected_signal = max(
        float(np.max(np.abs(first_out - second_out)))
        for first_out, second_out in zip(expected_first, expected_second)
    )
    actual_signal = max(
        float(np.max(np.abs(first_out - second_out)))
        for first_out, second_out in zip(actual_first, actual_second)
    )
    parity_error = max(
        float(np.max(np.abs(expected - actual)))
        for expected, actual in zip(expected_first, actual_first)
    )
    assert expected_signal > 1e-12
    assert actual_signal > max(1e-12, 100.0 * parity_error)
    if case.task == "detect":
        for expected_a, expected_b, actual_a, actual_b in zip(
            expected_first,
            expected_second,
            actual_first,
            actual_second,
        ):
            rms_error = _rms(expected_a - actual_a)
            reference_scale = max(_rms(expected_a), 1e-6)
            relative_error = rms_error / reference_scale
            relative_signal = _rms(expected_a - expected_b) / reference_scale
            signal_to_error = _rms(actual_a - actual_b) / max(rms_error, 1e-12)
            assert relative_error < 1e-3, (
                f"relative raw error {relative_error:.6g}"
            )
            assert relative_signal > 1e-6
            assert signal_to_error > 20.0

    assert backend.model_family == case.family
    assert backend.task == case.task
    assert backend.imgsz == case.imgsz
    if case.task == "detect":
        _assert_detector_predict_parity(case, model, backend)
    else:
        _assert_dense_predict_parity(case, model, backend, _image(case.imgsz))

    del backend, model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "case",
    ROUND14_CASES,
    ids=lambda case: f"{case.family}-{case.task}",
)
def test_tflite_round14_measured_blocks(tmp_path, monkeypatch, case):
    _run_tflite_case(tmp_path, monkeypatch, case)
