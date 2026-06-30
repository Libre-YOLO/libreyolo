from __future__ import annotations

import importlib.util

import torch
import pytest

pytestmark = pytest.mark.unit


class _FakeEC(torch.nn.Module):
    def deploy(self):
        self.deployed = True
        return self

    def forward(self, x):
        batch = x.shape[0]
        return {
            "pred_logits": torch.zeros(batch, 60, 2),
            "pred_boxes": torch.zeros(batch, 60, 4),
            "pred_masks": torch.zeros(batch, 60, 8, 8),
            "pred_keypoints": torch.zeros(batch, 60, 17, 2),
        }


def test_ec_export_wrapper_returns_detection_tuple():
    from libreyolo.models.ec.nn import ECExportWrapper

    wrapper = ECExportWrapper(_FakeEC(), task="detect")
    out = wrapper(torch.zeros(1, 3, 64, 64))

    assert len(out) == 2
    assert out[0].shape == (1, 60, 2)
    assert out[1].shape == (1, 60, 4)


def test_ec_export_wrapper_returns_segmentation_tuple():
    from libreyolo.models.ec.nn import ECExportWrapper

    wrapper = ECExportWrapper(_FakeEC(), task="segment")
    out = wrapper(torch.zeros(1, 3, 64, 64))

    assert len(out) == 3
    assert out[0].shape == (1, 60, 2)
    assert out[1].shape == (1, 60, 4)
    assert out[2].shape == (1, 60, 8, 8)


def test_ec_export_wrapper_flattens_deployed_pose_keypoints():
    from libreyolo.models.ec.nn import ECExportWrapper

    wrapper = ECExportWrapper(_FakeEC(), task="pose")
    out = wrapper(torch.zeros(1, 3, 64, 64))

    assert len(out) == 2
    assert out[0].shape == (1, 60, 2)
    assert out[1].shape == (1, 60, 34)


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None,
    reason="onnx not installed",
)
def test_ec_segment_export_onnx_output_names(tmp_path):
    import onnx

    from libreyolo.export.onnx import export_onnx

    class SegmentModel(torch.nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            device = x.device
            return (
                torch.zeros(batch, 2, 3, device=device),
                torch.zeros(batch, 2, 4, device=device),
                torch.zeros(batch, 2, 4, 4, device=device),
            )

    path = tmp_path / "ec-seg.onnx"
    export_onnx(
        SegmentModel(),
        torch.zeros(1, 3, 8, 8),
        output_path=str(path),
        opset=17,
        simplify=False,
        dynamic=False,
        half=False,
        metadata={"model_family": "ec", "task": "segment", "segmentation": "true"},
    )

    proto = onnx.load(str(path))
    assert [o.name for o in proto.graph.output] == [
        "pred_logits",
        "pred_boxes",
        "pred_masks",
    ]


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None,
    reason="onnx not installed",
)
def test_ec_pose_export_onnx_output_names(tmp_path):
    import onnx

    from libreyolo.export.onnx import export_onnx

    class PoseModel(torch.nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            device = x.device
            return (
                torch.zeros(batch, 2, 2, device=device),
                torch.zeros(batch, 2, 34, device=device),
            )

    path = tmp_path / "ec-pose.onnx"
    export_onnx(
        PoseModel(),
        torch.zeros(1, 3, 8, 8),
        output_path=str(path),
        opset=17,
        simplify=False,
        dynamic=False,
        half=False,
        metadata={"model_family": "ec", "task": "pose"},
    )

    proto = onnx.load(str(path))
    assert [o.name for o in proto.graph.output] == [
        "pred_logits",
        "pred_keypoints",
    ]
