"""Fast contract tests for HRNet person-crop exports."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.backends.base import BaseBackend
from libreyolo.export.exporter import OnnxExporter, TorchScriptExporter
from libreyolo.export.support import get_support
from libreyolo.models.hrnet.model import LibreHRNet

pytestmark = [pytest.mark.unit, pytest.mark.hrnet]


class _HeatmapBackend(BaseBackend):
    def __init__(self, heatmaps: np.ndarray):
        self._heatmaps = heatmaps
        super().__init__(
            model_path="hrnet.fixture",
            nb_classes=1,
            device="cpu",
            imgsz=(256, 192),
            model_family="hrnet",
            model_size="w32",
            task="pose",
            supported_tasks=("pose",),
            default_task="pose",
            names={0: "person"},
            num_keypoints=17,
            keypoint_dim=3,
        )

    def _run_inference(self, blob: np.ndarray) -> list:
        return [np.repeat(self._heatmaps, blob.shape[0], axis=0)]


def test_exporter_uses_native_rectangular_crop_and_metadata(tmp_path):
    model = LibreHRNet(size="w32", device="cpu")
    exporter = OnnxExporter(model)

    imgsz, device, _ = exporter._resolve_params(
        output_path=str(tmp_path / "hrnet.onnx"),
        imgsz=None,
        device="cpu",
        half=False,
        int8=False,
    )
    metadata = exporter._build_onnx_metadata(
        dynamic=False,
        half=False,
        imgsz=imgsz,
    )

    assert imgsz == (256, 192)
    assert device == torch.device("cpu")
    assert metadata["imgsz"] == "256"
    assert metadata["imgsz_h"] == "256"
    assert metadata["imgsz_w"] == "192"
    assert metadata["pose_input"] == "person_crop"
    assert metadata["num_keypoints"] == "17"
    assert metadata["keypoint_dim"] == "3"


@pytest.mark.parametrize("exporter_cls", [OnnxExporter, TorchScriptExporter])
def test_exporter_rejects_noncanonical_hrnet_canvas(tmp_path, exporter_cls):
    model = LibreHRNet(size="w32", device="cpu")
    with pytest.raises(ValueError, match="fixed person-crop canvas"):
        exporter_cls(model)._resolve_params(
            output_path=str(tmp_path / "artifact"),
            imgsz=(384, 288),
            device="cpu",
            half=False,
            int8=False,
        )


def test_backend_preprocesses_and_decodes_one_full_person_crop():
    heatmaps = np.zeros((1, 17, 64, 48), dtype=np.float32)
    heatmaps[:, :, 20, 10] = 0.9
    backend = _HeatmapBackend(heatmaps)
    image = np.zeros((320, 180, 3), dtype=np.uint8)

    tensor, _original, original_size, _ratio = backend._preprocess(
        image,
        backend.imgsz,
        "rgb",
    )
    result = backend.predict(image, color_format="rgb")

    assert tensor.shape == (1, 3, 256, 192)
    assert original_size == (180, 320)
    assert result.boxes.xyxy.tolist() == [[0.0, 0.0, 180.0, 320.0]]
    assert result.keypoints.data.shape == (1, 17, 3)
    assert torch.allclose(result.keypoints.conf, torch.full((1, 17), 0.9))


def test_hrnet_blocks_unvalidated_export_formats():
    model = LibreHRNet(size="w32", device="cpu")
    with pytest.raises(NotImplementedError, match="ONNX, TorchScript, OpenVINO"):
        model.export("tflite")


def test_support_matrix_claims_only_measured_hrnet_formats():
    for format_name in ("onnx", "torchscript", "openvino", "tensorrt"):
        assert get_support("hrnet", "pose", format_name).tier == "validated"
    for format_name in ("executorch", "ncnn", "tflite", "coreml", "coreai"):
        assert get_support("hrnet", "pose", format_name).tier == "blocked"
