"""FP16 exported-runtime outputs must be safe on CPU NMS kernels."""

import pytest
import torch

pytestmark = pytest.mark.unit


def test_yolo9_nms_promotes_fp16_inputs(monkeypatch):
    from libreyolo.postprocess import yolo9

    seen = {}

    def fake_batched_nms(boxes, scores, class_ids, _iou):
        seen["boxes"] = boxes.dtype
        seen["scores"] = scores.dtype
        return torch.tensor([0], dtype=torch.long)

    monkeypatch.setattr(yolo9, "batched_nms", fake_batched_nms)
    keep = yolo9._nms_keep_indices(
        torch.tensor([[0, 0, 4, 4]], dtype=torch.float16),
        torch.tensor([0.9], dtype=torch.float16),
        torch.tensor([0]),
        0.5,
        10,
    )

    assert keep.tolist() == [0]
    assert seen == {"boxes": torch.float32, "scores": torch.float32}


def test_yolonas_nms_promotes_fp16_inputs(monkeypatch):
    from libreyolo.postprocess import yolonas

    seen = {}

    def fake_batched_nms(boxes, scores, class_ids, _iou):
        seen["boxes"] = boxes.dtype
        seen["scores"] = scores.dtype
        return torch.tensor([0], dtype=torch.long)

    monkeypatch.setattr(yolonas.torchvision.ops, "batched_nms", fake_batched_nms)
    result = yolonas.postprocess(
        (
            torch.tensor([[[0, 0, 4, 4]]], dtype=torch.float16),
            torch.tensor([[[0.9, 0.1]]], dtype=torch.float16),
        ),
        conf_thres=0.2,
        original_size=None,
    )

    assert result["num_detections"] == 1
    assert seen == {"boxes": torch.float32, "scores": torch.float32}


def test_picodet_nms_promotes_fp16_inputs(monkeypatch):
    from libreyolo.postprocess import picodet
    import torchvision.ops

    monkeypatch.setattr(
        picodet,
        "_per_level_filter_topk",
        lambda *args, **kwargs: (
            torch.tensor([0.9], dtype=torch.float16),
            torch.tensor([0]),
            torch.tensor([[0, 0, 4, 4]], dtype=torch.float16),
        ),
    )
    seen = {}

    def fake_batched_nms(boxes, scores, class_ids, _iou):
        seen["boxes"] = boxes.dtype
        seen["scores"] = scores.dtype
        return torch.tensor([0], dtype=torch.long)

    monkeypatch.setattr(torchvision.ops, "batched_nms", fake_batched_nms)
    result = picodet.postprocess(([], []), original_size=None)

    assert result["num_detections"] == 1
    assert seen == {"boxes": torch.float32, "scores": torch.float32}
