"""Tests for HRNet's two-stage public inference contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from libreyolo.models.hrnet.detector import (
    LibreYOLOPersonDetector,
    PersonBox,
    resolve_person_detector,
)
from libreyolo.models.hrnet.inference import HRNetPoseInferenceRunner
from libreyolo.utils.results import Boxes, Results

pytestmark = pytest.mark.unit


class FixedHeatmapHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        batch, _channels, height, width = images.shape
        heatmaps = torch.zeros(
            (batch, 17, height // 4, width // 4),
            dtype=images.dtype,
            device=images.device,
        )
        heatmaps[:, :, height // 8, width // 8] = 0.9
        return heatmaps


class FakeHRNet:
    def __init__(self):
        self.model = FixedHeatmapHead().eval()
        self.device = torch.device("cpu")
        self.size = "w32"
        self.names = {0: "person"}
        self.person_detector = None

    @staticmethod
    def _get_input_size():
        return (256, 192)


class FakeDetectionModel:
    task = "detect"
    names = {0: "person", 1: "bicycle"}

    def __init__(self, family):
        self.family = family
        self.kwargs = None

    def __call__(self, _image, **kwargs):
        self.kwargs = kwargs
        boxes = Boxes(
            torch.tensor([[1.0, 2.0, 30.0, 50.0], [5.0, 6.0, 20.0, 25.0]]),
            torch.tensor([0.8, 0.9]),
            torch.tensor([0.0, 1.0]),
            orig_shape=(64, 64),
        )
        return Results(boxes=boxes, orig_shape=(64, 64), names=self.names)


class FakePoseModel:
    task = "pose"
    names = {0: "person"}

    def __call__(self, *_args, **_kwargs):
        return None


def _image(width=96, height=128):
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize("family", ["yolo9", "rfdetr"])
def test_libreyolo_person_adapter_covers_flagship_detectors(family):
    model = FakeDetectionModel(family)
    detector = resolve_person_detector(model)

    people = detector(np.zeros((64, 64, 3), dtype=np.uint8))

    assert isinstance(detector, LibreYOLOPersonDetector)
    assert len(people) == 1
    assert people[0].xyxy == (1.0, 2.0, 30.0, 50.0)
    assert people[0].score == pytest.approx(0.8)
    assert model.kwargs["classes"] == [0]


def test_explicit_person_boxes_bypass_detector():
    wrapper = FakeHRNet()
    runner = HRNetPoseInferenceRunner(wrapper)

    def forbidden_detector(_image):
        raise AssertionError("explicit boxes must bypass person detection")

    result = runner(
        _image(),
        person_boxes=[(10, 15, 70, 115, 0.8)],
        person_detector=forbidden_detector,
        conf=0.1,
    )

    assert len(result) == 1
    assert torch.equal(result.boxes.xyxy, torch.tensor([[10.0, 15.0, 70.0, 115.0]]))
    assert result.boxes.conf.item() == pytest.approx(0.72)
    assert result.keypoints.data.shape == (1, 17, 3)
    assert torch.allclose(result.keypoints.conf, torch.full((1, 17), 0.9))


def test_already_cropped_path_uses_full_image_box():
    wrapper = FakeHRNet()
    result = HRNetPoseInferenceRunner(wrapper)(_image(80, 120), cropped=True)

    assert len(result) == 1
    assert torch.equal(result.boxes.xyxy, torch.tensor([[0.0, 0.0, 80.0, 120.0]]))


def test_callable_detector_and_empty_box_bypass():
    wrapper = FakeHRNet()
    runner = HRNetPoseInferenceRunner(wrapper)
    calls = {"count": 0}

    def detector(_image):
        calls["count"] += 1
        return [[4, 5, 60, 100, 0.7]]

    result = runner(_image(), person_detector=detector, conf=0.2)
    assert len(result) == 1
    assert calls["count"] == 1

    empty = runner(_image(), person_boxes=[], person_detector=detector)
    assert len(empty) == 0
    assert empty.keypoints.data.shape == (0, 17, 3)
    assert calls["count"] == 1


def test_default_detector_is_resolved_once_and_cached(monkeypatch):
    from libreyolo.models.hrnet import inference

    wrapper = FakeHRNet()
    runner = HRNetPoseInferenceRunner(wrapper)
    calls = {"factory": 0, "detect": 0}

    def detector(_image):
        calls["detect"] += 1
        return [PersonBox((3, 4, 50, 100), 0.95)]

    def factory(device="auto"):
        del device
        calls["factory"] += 1
        return detector

    monkeypatch.setattr(inference, "default_person_detector", factory)
    runner(_image())
    runner(_image())

    assert calls == {"factory": 1, "detect": 2}
    assert wrapper.person_detector is detector


def test_flip_test_is_explicit_and_runs_second_forward():
    wrapper = FakeHRNet()
    runner = HRNetPoseInferenceRunner(wrapper)
    runner(_image(), cropped=True, flip_test=False)
    assert wrapper.model.calls == 1
    runner(_image(), cropped=True, flip_test=True)
    assert wrapper.model.calls == 3


def test_pose_save_draws_boxes_and_keypoints(tmp_path):
    wrapper = FakeHRNet()
    output = tmp_path / "pose.png"
    result = HRNetPoseInferenceRunner(wrapper)(
        _image(),
        cropped=True,
        save=True,
        output_path=str(output),
    )

    assert output.is_file()
    assert result.saved_path == str(output)


def test_inference_rejects_conflicting_or_detection_shaped_options():
    runner = HRNetPoseInferenceRunner(FakeHRNet())
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner(_image(), cropped=True, person_boxes=[(0, 0, 10, 10)])
    with pytest.raises(ValueError, match="flip_test"):
        runner(_image(), cropped=True, augment=True)
    with pytest.raises(ValueError, match="[Tt]iled"):
        runner(_image(), cropped=True, tiling=True)
    with pytest.raises(ValueError, match="fixed crop size"):
        runner(_image(), cropped=True, imgsz=640)


def test_non_detection_model_cannot_be_person_detector():
    with pytest.raises(ValueError, match="detection model"):
        resolve_person_detector(FakePoseModel())
