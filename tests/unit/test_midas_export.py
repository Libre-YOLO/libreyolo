"""Trained-checkpoint MiDaS export parity for both official variants."""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from libreyolo.models.midas.convert import UPSTREAM_SHA256

CHECKPOINT_DIR = os.environ.get("MIDAS_CHECKPOINT_DIR")

pytestmark = [
    pytest.mark.midas,
    pytest.mark.external_data,
    pytest.mark.export_backend,
    pytest.mark.skipif(
        not CHECKPOINT_DIR or not Path(CHECKPOINT_DIR).is_dir(),
        reason="Set MIDAS_CHECKPOINT_DIR to the official MiDaS release assets",
    ),
]

_VARIANTS = {
    "s": ("midas_v21_small_256.pt", 256),
    "l": ("dpt_large_384.pt", 384),
}

_EXPORT_FORMATS = [
    pytest.param("torchscript", id="torchscript"),
    pytest.param("onnx", id="onnx"),
    pytest.param("openvino", marks=pytest.mark.openvino, id="openvino"),
    pytest.param(
        "tensorrt",
        marks=(pytest.mark.tensorrt, pytest.mark.trt),
        id="tensorrt",
    ),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _images(input_size: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:input_size, 0:input_size]
    first = np.stack((x % 256, y % 256, ((x + y) // 2) % 256), axis=-1).astype(np.uint8)
    second = np.stack(
        (255 - (x % 256), (y // 2) % 256, (2 * x + y) % 256), axis=-1
    ).astype(np.uint8)
    return first, second


def _parity_metrics(
    reference: list[np.ndarray],
    actual: list[np.ndarray],
) -> tuple[float, float, list[float]]:
    errors = [
        float(np.mean((expected - observed) ** 2))
        for expected, observed in zip(reference, actual, strict=True)
    ]
    signal = float(np.mean((reference[0] - reference[1]) ** 2))
    peak = max(
        float(np.max(np.abs(reference[0]))),
        float(np.max(np.abs(reference[1]))),
        1e-6,
    )
    worst_error = max(errors)
    psnr = (
        float("inf")
        if worst_error == 0.0
        else 20.0 * np.log10(peak / np.sqrt(worst_error))
    )
    margin = float("inf") if worst_error == 0.0 else signal / worst_error
    return psnr, margin, errors


@pytest.mark.parametrize("format", _EXPORT_FORMATS)
def test_trained_midas_s_and_l_export_parity(tmp_path: Path, format: str):
    pytest.importorskip("timm")
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    elif format == "openvino":
        pytest.importorskip("openvino")
    elif format == "tensorrt":
        pytest.importorskip("tensorrt")
        if not torch.cuda.is_available():
            pytest.skip("TensorRT parity requires CUDA")
    from libreyolo import LibreYOLO
    from libreyolo.models.midas.model import LibreMiDaS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime_device = "cuda" if format == "tensorrt" else "cpu"
    suffix = {
        "torchscript": ".torchscript",
        "onnx": ".onnx",
        "openvino": "_openvino",
        "tensorrt": ".engine",
    }[format]
    for size, (filename, input_size) in _VARIANTS.items():
        checkpoint_path = Path(CHECKPOINT_DIR) / filename
        assert checkpoint_path.is_file()
        assert _sha256(checkpoint_path) == UPSTREAM_SHA256[filename]
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = LibreMiDaS(None, size=size, task="depth", device=device)
        model.model.load_state_dict(state, strict=True)
        del state

        images = _images(input_size)
        reference = [
            model.predict(image, imgsz=input_size).depth_map.data.numpy()
            for image in images
        ]
        export_kwargs = {
            "format": format,
            "output_path": str(tmp_path / f"midas_{size}{suffix}"),
            "imgsz": input_size,
            "batch": 1,
            "dynamic": False,
            "half": False,
            "simplify": False,
        }
        if format == "tensorrt":
            export_kwargs["workspace"] = 1.0
        artifact = model.export(**export_kwargs)
        backend = LibreYOLO(artifact, device=runtime_device)
        actual = [backend.predict(image).depth_map.data.numpy() for image in images]
        psnr, margin, errors = _parity_metrics(reference, actual)
        print(
            f"MiDaS {size} {format}: PSNR={psnr:.3f} dB, "
            f"signal/error={margin:.1f}, MSE={errors}"
        )
        assert psnr > 40.0
        assert margin > 20.0
        assert backend.family == "midas"
        assert backend.size == size
        assert backend.task == "depth"
        assert backend.names == {0: "depth"}
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del model, reference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
