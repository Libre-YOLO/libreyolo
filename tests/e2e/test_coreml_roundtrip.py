"""End-to-end CoreML export + load roundtrip. macOS only.

Asserts numerical-ish parity: the CoreML model must produce the same number
of detections as the source PyTorch model on the bundled sample image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.coreml, pytest.mark.e2e]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(autouse=True)
def _macos_only():
    if sys.platform != "darwin":
        pytest.skip("CoreML tests require macOS")
    pytest.importorskip("coremltools")


def _load_or_skip(filename: str):
    """Load a model via the LibreYOLO factory (which auto-downloads on miss).

    Skips the test only if the factory itself errors — e.g. network failure
    or unknown filename pattern.
    """
    from libreyolo import LibreYOLO

    try:
        return LibreYOLO(filename)
    except Exception as e:
        pytest.skip(f"Could not load {filename}: {e}")


def _load_yolox_nano():
    return _load_or_skip("LibreYOLOXn.pt")


def _load_yolo9_tiny():
    return _load_or_skip("LibreYOLO9t.pt")


def _assert_parity(pt_res, cm_res, *, conf_tol: float = 1e-3) -> None:
    """Strong numerical parity check, robust to borderline-threshold detections.

    Compares the top-min(N_pt, N_cm) detections by confidence: each pair must
    agree to within ``conf_tol``. Allows a single missing detection (uint8
    quantization can knock a barely-above-threshold detection under).
    """
    pt_conf = sorted([float(s) for s in pt_res.boxes.conf], reverse=True)
    cm_conf = sorted([float(s) for s in cm_res.boxes.conf], reverse=True)

    assert abs(len(pt_conf) - len(cm_conf)) <= 1, (
        f"detection counts differ too much: pt={len(pt_conf)} cm={len(cm_conf)}"
    )

    n = min(len(pt_conf), len(cm_conf))
    assert n >= 1, "expected at least one matched detection"
    for i in range(n):
        assert abs(pt_conf[i] - cm_conf[i]) < conf_tol, (
            f"confidence mismatch at rank {i}: pt={pt_conf[i]} cm={cm_conf[i]}"
        )


def test_yolox_export_fp32_parity(tmp_path):
    """YOLOX nano: CoreML detections must numerically match PyTorch."""
    from libreyolo import SAMPLE_IMAGE, LibreYOLO

    pt_model = _load_yolox_nano()
    pt_res = pt_model(SAMPLE_IMAGE)
    assert len(pt_res.boxes) >= 1, "PT produced 0 detections — sample image issue"

    out_path = tmp_path / "model.mlpackage"
    pt_model.export(format="coreml", output_path=str(out_path))
    assert out_path.is_dir()

    coreml_model = LibreYOLO(str(out_path))
    cm_res = coreml_model(SAMPLE_IMAGE)
    _assert_parity(pt_res, cm_res)


def test_yolox_export_fp16(tmp_path):
    """fp16 export should still produce a non-empty detection set."""
    from libreyolo import SAMPLE_IMAGE, LibreYOLO

    pt_model = _load_yolox_nano()
    out_path = tmp_path / "model_fp16.mlpackage"
    pt_model.export(format="coreml", output_path=str(out_path), half=True)
    assert out_path.is_dir()

    coreml_model = LibreYOLO(str(out_path))
    assert len(coreml_model(SAMPLE_IMAGE).boxes) >= 1


def test_yolox_export_with_embedded_nms(tmp_path):
    """nms=True should produce a loadable CoreML pipeline."""
    from libreyolo import SAMPLE_IMAGE, LibreYOLO

    pt_model = _load_yolox_nano()
    out_path = tmp_path / "model_nms.mlpackage"
    pt_model.export(format="coreml", output_path=str(out_path), nms=True)
    assert out_path.is_dir()

    coreml_model = LibreYOLO(str(out_path))
    assert len(coreml_model(SAMPLE_IMAGE).boxes) >= 1


def test_yolo9_export_fp32_parity(tmp_path):
    """YOLO9 tiny: CoreML detections must numerically match PyTorch."""
    from libreyolo import SAMPLE_IMAGE, LibreYOLO

    pt_model = _load_yolo9_tiny()
    pt_res = pt_model(SAMPLE_IMAGE)
    assert len(pt_res.boxes) >= 1

    out_path = tmp_path / "model.mlpackage"
    pt_model.export(format="coreml", output_path=str(out_path))
    assert out_path.is_dir()

    coreml_model = LibreYOLO(str(out_path))
    cm_res = coreml_model(SAMPLE_IMAGE)
    _assert_parity(pt_res, cm_res)


def test_compute_units_kwarg_accepted(tmp_path):
    model = _load_yolox_nano()
    out_path = tmp_path / "model_cpu.mlpackage"
    model.export(
        format="coreml",
        output_path=str(out_path),
        compute_units="cpu_only",
    )
    assert out_path.is_dir()


def test_rfdetr_nms_true_raises(tmp_path):
    # RF-DETR auto-detection requires the rfdetr extra to be installed
    # (the LibreYOLORFDETR wrapper is registered lazily via _ensure_rfdetr).
    # Without it, the factory can't resolve size from the filename and the
    # download path fails. Install with: pip install -e ".[coreml,rfdetr]"
    pytest.importorskip("rfdetr")
    rfdetr = _load_or_skip("LibreRFDETRn.pt")
    with pytest.raises(NotImplementedError, match="RF-DETR"):
        rfdetr.export(
            format="coreml",
            output_path=str(tmp_path / "rfdetr.mlpackage"),
            nms=True,
        )


RFDETR_CASES = [
    ("LibreRFDETRn.pt", "detect", 384),
    ("LibreRFDETRn-seg.pt", "segment", 312),
    # Pose currently has one published checkpoint; x is the smallest real case.
    ("LibreRFDETRx-pose.pt", "pose", 576),
]


def _rfdetr_byte_probes(imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:imgsz, :imgsz]
    first = np.stack(
        (
            (3 * xx + 5 * yy + 17) % 256,
            (11 * xx + 7 * yy + 53) % 256,
            (13 * xx + 19 * yy + 101) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    second = np.stack(
        (
            (23 * xx + 2 * yy + 211) % 256,
            (5 * xx + 29 * yy + 37) % 256,
            (17 * xx + 31 * yy + 149) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return first, second


def _rfdetr_tensor(image: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
    return value.float().div_(255.0)


def _rfdetr_flatten(value) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)) and all(
        torch.is_tensor(item) for item in value
    ):
        return list(value)
    raise TypeError(f"Unexpected RF-DETR export output: {type(value).__name__}")


def _rfdetr_prepared_reference(model, task: str, imgsz: int, probes):
    from libreyolo.export.coreml import (
        _prepare_rfdetr_coreml_graph,
        _wrap_for_family,
    )
    from libreyolo.export.exporter import CoreMLExporter

    tensors = tuple(_rfdetr_tensor(probe) for probe in probes)
    exporter = CoreMLExporter(model)
    with exporter._model_context(
        torch.device("cpu"),
        False,
        False,
        1,
        (imgsz, imgsz),
    ) as (nn_model, _):
        wrapped = _wrap_for_family(nn_model, "rfdetr").eval()
        with _prepare_rfdetr_coreml_graph(
            wrapped,
            tensors[0],
            "rfdetr",
            task,
        ), torch.no_grad():
            return [
                [
                    tensor.detach().cpu().numpy()
                    for tensor in _rfdetr_flatten(wrapped(probe))
                ]
                for probe in tensors
            ]


def _rfdetr_artifact_outputs(artifact, probes):
    import coremltools as ct

    runtime = ct.models.MLModel(
        str(artifact),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    spec = runtime.get_spec()
    names = [feature.name for feature in spec.description.output]
    input_feature = next(iter(spec.description.input))
    input_kind = input_feature.type.WhichOneof("Type")
    outputs = []
    for probe in probes:
        if input_kind == "imageType":
            runtime_input = Image.fromarray(probe, mode="RGB")
        else:
            assert input_kind == "multiArrayType"
            runtime_input = np.ascontiguousarray(
                probe.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            )
        result = runtime.predict({input_feature.name: runtime_input})
        outputs.append([np.asarray(result[name]) for name in names])
    return names, outputs


def _assert_rfdetr_raw_parity(names, reference, actual):
    for index, (expected1, expected2, got1, got2) in enumerate(
        zip(reference[0], reference[1], actual[0], actual[1])
    ):
        assert got1.shape == expected1.shape
        assert got2.shape == expected2.shape
        scale = max(
            float(np.abs(expected1).max()),
            float(np.abs(expected2).max()),
            1e-12,
        )
        error = max(
            float(np.abs(got1 - expected1).max()),
            float(np.abs(got2 - expected2).max()),
        ) / scale
        sensitivity = float(np.abs(expected2 - expected1).max()) / scale
        margin = float("inf") if error == 0 else sensitivity / error
        print(
            f"out[{index}] ({names[index]}): error={error:.9e}, "
            f"sensitivity={sensitivity:.9e}, margin={margin:.3f}x"
        )
        assert error <= 3e-4
        assert sensitivity >= 1e-6
        assert margin >= 100.0


@pytest.mark.parametrize("weights,task,imgsz", RFDETR_CASES)
def test_rfdetr_coreml_raw_runtime_parity(weights, task, imgsz, tmp_path):
    from libreyolo import LibreYOLO
    from libreyolo.export.coreml import _rfdetr_output_names

    model = LibreYOLO(weights, device="cpu")
    assert model.task == task
    artifact = model.export(
        format="coreml",
        imgsz=imgsz,
        output_path=str(tmp_path / f"rfdetr-{task}.mlpackage"),
        compute_units="cpu_only",
    )
    probes = _rfdetr_byte_probes(imgsz)
    reference = _rfdetr_prepared_reference(model, task, imgsz, probes)
    names, actual = _rfdetr_artifact_outputs(artifact, probes)
    assert names == _rfdetr_output_names(task)
    _assert_rfdetr_raw_parity(names, reference, actual)
