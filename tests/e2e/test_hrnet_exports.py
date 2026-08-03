"""Opt-in real-checkpoint parity gate for every supported HRNet export runtime.

Set ``HRNET_CONVERTED_DIR`` to a directory containing
``LibreHRNetw32-pose.pt`` and ``LibreHRNetw48-pose.pt``. The separate upstream
gate in ``tests/unit/test_hrnet_parity.py`` proves those checkpoints against the
pinned MIT source implementation; this gate proves each exported runtime
against the matching native LibreYOLO graph and public person-crop result.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from libreyolo import LibreYOLO

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.external_data,
    pytest.mark.slow,
    pytest.mark.export_backend,
    pytest.mark.supported_backend,
    pytest.mark.hrnet,
]

CASES = {
    "w32": ("LibreHRNetw32-pose.pt", (256, 192)),
    "w48": ("LibreHRNetw48-pose.pt", (384, 288)),
}
TOLERANCES = {
    "onnx": 3e-6,
    "torchscript": 0.0,
    "openvino": 3e-3,
    "tensorrt": 3e-3,
}


def _weights_dir() -> Path:
    value = os.environ.get("HRNET_CONVERTED_DIR")
    if not value:
        pytest.skip("set HRNET_CONVERTED_DIR to run HRNet export parity")
    path = Path(value)
    if not path.is_dir():
        pytest.fail(f"HRNET_CONVERTED_DIR is not a directory: {path}")
    return path


def _require_runtime(format_name: str) -> str:
    modules = {
        "onnx": ("onnx", "onnxruntime"),
        "torchscript": (),
        "openvino": ("onnx", "openvino"),
        "tensorrt": ("onnx", "tensorrt"),
    }[format_name]
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f"{format_name} parity requires {', '.join(missing)}")
    if format_name == "tensorrt":
        if not torch.cuda.is_available():
            pytest.skip("TensorRT parity requires CUDA")
        return "cuda:0"
    return "cpu"


def _gradient_image(size: str) -> np.ndarray:
    # These dimensions exercise the checkpoint-specific affine canvas while
    # keeping the official heatmap maxima well separated. Near-tied random
    # peaks make coordinate equality an unstable test of otherwise-close raw
    # runtime tensors because argmax can jump to a distant pixel.
    height, width = (317, 181) if size == "w32" else (451, 239)
    y, x = np.mgrid[:height, :width]
    return np.stack(
        ((x * 3 + y) % 256, (x + y * 2) % 256, (x * 7 + y * 5) % 256),
        axis=2,
    ).astype(np.uint8)


def _run_case(output_root: Path, size: str, format_name: str) -> None:
    device = _require_runtime(format_name)
    filename, native_imgsz = CASES[size]
    weights = _weights_dir() / filename
    if not weights.is_file():
        pytest.fail(f"converted HRNet checkpoint is missing: {weights}")

    model = LibreYOLO(str(weights), device=device)
    image = _gradient_image(size)
    tensor, _original, _original_size, _ratio = model._preprocess(
        image,
        color_format="rgb",
    )
    with torch.no_grad():
        native_raw = model._forward(tensor.to(device)).float().cpu().numpy()
    native_result = model(image, cropped=True, color_format="rgb")

    output_path = {
        "onnx": output_root / f"{size}.onnx",
        "torchscript": output_root / f"{size}.torchscript",
        "openvino": output_root / f"{size}_openvino",
        "tensorrt": output_root / f"{size}.engine",
    }[format_name]
    export_kwargs = {
        "output_path": str(output_path),
        "dynamic": False,
        "simplify": False,
        "device": device,
    }
    if format_name == "tensorrt":
        export_kwargs.update(half=False, workspace=2.0)
    artifact = model.export(format_name, **export_kwargs)
    backend = LibreYOLO(artifact, device=device)

    exported_raw = backend._run_inference(tensor.numpy())[0]
    exported_result = backend.predict(image, color_format="rgb")
    tolerance = TOLERANCES[format_name]

    assert backend._get_input_size() == native_imgsz
    assert exported_raw.shape == native_raw.shape
    assert np.allclose(exported_raw, native_raw, rtol=0.0, atol=tolerance)
    assert exported_result.boxes.xyxy.tolist() == [
        [0.0, 0.0, float(image.shape[1]), float(image.shape[0])]
    ]
    keypoint_error = torch.max(
        torch.abs(exported_result.keypoints.data - native_result.keypoints.data)
    ).item()
    assert torch.allclose(
        exported_result.keypoints.data,
        native_result.keypoints.data,
        rtol=0.0,
        atol=tolerance,
    ), f"maximum decoded keypoint error was {keypoint_error}"
    assert torch.allclose(
        exported_result.boxes.conf,
        native_result.boxes.conf,
        rtol=0.0,
        atol=tolerance,
    )

    # The artifacts are hundreds of megabytes. Keep the external gate from
    # filling a shared CI worker when all eight cases run in one invocation.
    del backend, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(output_root)


@pytest.mark.parametrize("size", CASES)
@pytest.mark.parametrize(
    "format_name",
    [
        pytest.param("onnx", marks=pytest.mark.onnx),
        pytest.param("torchscript", marks=pytest.mark.torchscript),
        pytest.param("openvino", marks=pytest.mark.openvino),
        pytest.param("tensorrt", marks=[pytest.mark.tensorrt, pytest.mark.trt]),
    ],
)
def test_real_checkpoint_export_raw_and_public_parity(
    tmp_path: Path,
    size: str,
    format_name: str,
):
    """Run each native runtime in a fresh process.

    ONNX Runtime, OpenVINO, TorchScript, and TensorRT each load native shared
    libraries and large HRNet graphs. Process isolation avoids retaining one
    runtime's allocator and thread pools while the next artifact is serialized.
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(tmp_path),
        size,
        format_name,
    ]
    repo_root = Path(__file__).resolve().parents[2]
    subprocess_env = os.environ.copy()
    subprocess_env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repo_root), subprocess_env.get("PYTHONPATH", "")))
    )
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=subprocess_env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        f"HRNet {size} {format_name} export parity subprocess failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


if __name__ == "__main__":
    _run_case(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
