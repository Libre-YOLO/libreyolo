"""Random-weight ONNX export smoke tests for YOLO-NAS pose."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.yolonas]


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx/onnxruntime not installed",
)
def test_yolonas_pose_onnx_export_roundtrip(tmp_path):
    import onnx
    import onnxruntime as ort

    from libreyolo import LibreYOLO, LibreYOLONAS

    model = LibreYOLONAS(None, size="n", task="pose", device="cpu")
    out_path = tmp_path / "LibreYOLONASn-pose.onnx"
    model.export(
        "onnx",
        output_path=str(out_path),
        simplify=False,
        dynamic=False,
        imgsz=64,
        opset=17,
    )

    proto = onnx.load(str(out_path))
    output_names = [o.name for o in proto.graph.output]
    assert output_names == ["boxes", "scores", "keypoints_xy", "keypoints_conf"]

    metadata = {p.key: p.value for p in proto.metadata_props}
    assert metadata.get("model_family") == "yolonas"
    assert metadata.get("task") == "pose"
    assert metadata.get("num_keypoints") == "17"
    assert metadata.get("keypoint_dim") == "3"

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"images": np.random.randn(1, 3, 64, 64).astype(np.float32)})
    assert [tuple(out.shape) for out in outs] == [
        (1, 84, 4),
        (1, 84, 1),
        (1, 84, 17, 2),
        (1, 84, 17),
    ]

    loaded = LibreYOLO(str(out_path), device="cpu")
    assert loaded.model_family == "yolonas"
    assert loaded.task == "pose"
    result = loaded.predict(np.zeros((64, 64, 3), dtype=np.uint8), conf=0.99, imgsz=64)
    assert result.keypoints is not None
