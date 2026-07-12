"""Inference source, option, and tiled-payload contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libreyolo.models.base.inference import InferenceRunner
from libreyolo.utils.general import get_slice_bboxes
from libreyolo.utils.image_loader import ImageLoader

pytestmark = pytest.mark.unit


class _ContractStub:
    task = "detect"
    TTA_ENABLED = False
    SUPPORTS_BATCHED_PREDICT = True
    size = "n"
    names = {0: "thing"}
    device = torch.device("cpu")

    def __init__(self):
        self.model = SimpleNamespace(training=False)
        self.forward_shapes: list[tuple[int, ...]] = []

    def _get_input_size(self):
        return 32

    def _get_model_name(self):
        return "contract-stub"

    def _preprocess(self, image, color_format="auto", input_size=None):
        pil = ImageLoader.load(image, color_format=color_format)
        marker = float(np.asarray(pil, dtype=np.float32).mean())
        tensor = torch.full((1, 3, 32, 32), marker)
        return tensor, pil, pil.size, 1.0

    def _forward(self, tensor):
        self.forward_shapes.append(tuple(tensor.shape))
        return tensor.mean(dim=(1, 2, 3), keepdim=True)

    def _postprocess(
        self,
        output,
        conf,
        iou,
        original_size,
        max_det=300,
        ratio=1.0,
        classes=None,
        **kwargs,
    ):
        marker = float(output.reshape(-1)[0])
        return {
            "boxes": [[1.0, 1.0, 5.0, 5.0]],
            "scores": [marker],
            "classes": [0],
            "num_detections": 1,
        }


def test_numpy_4d_source_processes_every_image_in_order():
    model = _ContractStub()
    source = np.stack(
        [
            np.full((8, 12, 3), 10, dtype=np.uint8),
            np.full((8, 12, 3), 200, dtype=np.uint8),
        ]
    )

    results = InferenceRunner(model)(source, batch=2)

    assert model.forward_shapes == [(2, 3, 32, 32)]
    assert [float(result.boxes.conf[0]) for result in results] == [10.0, 200.0]


def test_torch_4d_source_processes_every_image_in_order():
    model = _ContractStub()
    source = torch.stack(
        [
            torch.full((3, 8, 12), 20, dtype=torch.uint8),
            torch.full((3, 8, 12), 180, dtype=torch.uint8),
        ]
    )

    results = InferenceRunner(model)(source, batch=2)

    assert model.forward_shapes == [(2, 3, 32, 32)]
    assert [float(result.boxes.conf[0]) for result in results] == [20.0, 180.0]


def test_empty_4d_source_returns_empty_result_list():
    source = np.empty((0, 8, 12, 3), dtype=np.uint8)
    assert InferenceRunner(_ContractStub())(source) == []


@pytest.mark.parametrize("family", ["yolo9", "rfdetr"])
def test_flagship_models_accept_two_image_4d_source(family):
    if family == "yolo9":
        from libreyolo import LibreYOLO9

        model = LibreYOLO9(None, size="t", device="cpu")
        imgsz = 64
    else:
        from libreyolo import LibreRFDETR

        model = LibreRFDETR(model_path={}, size="n", device="cpu")
        imgsz = model._get_input_size()

    model.model.eval()
    forward_shapes = []

    def fake_forward(tensor):
        forward_shapes.append(tuple(tensor.shape))
        return tensor.mean(dim=(1, 2, 3), keepdim=True)

    def fake_postprocess(output, *args, **kwargs):
        marker = float(output.reshape(-1)[0])
        return {
            "boxes": [[1.0, 1.0, 5.0, 5.0]],
            "scores": [marker],
            "classes": [0],
            "num_detections": 1,
        }

    model._forward = fake_forward
    model._postprocess = fake_postprocess
    source = np.stack(
        [
            np.full((12, 16, 3), 10, dtype=np.uint8),
            np.full((12, 16, 3), 200, dtype=np.uint8),
        ]
    )

    results = model(source, batch=2, imgsz=imgsz)

    assert len(results) == 2
    assert forward_shapes == [(2, 3, imgsz, imgsz)]
    assert float(results[0].boxes.conf[0]) < float(results[1].boxes.conf[0])


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"conf": float("nan")}, ValueError, "conf must be finite"),
        ({"conf": -0.1}, ValueError, "conf must be finite"),
        ({"iou": float("inf")}, ValueError, "iou must be finite"),
        ({"batch": 0}, ValueError, "batch must be positive"),
        ({"batch": 1.5}, TypeError, "batch must be an integer"),
        ({"vid_stride": 0}, ValueError, "vid_stride must be positive"),
        ({"max_det": -1}, ValueError, "max_det must be non-negative"),
        ({"max_det": True}, TypeError, "max_det must be an integer"),
        ({"overlap_ratio": 1.0}, ValueError, "overlap_ratio must be finite"),
        ({"overlap_ratio": float("nan")}, ValueError, "overlap_ratio must be finite"),
        ({"classes": ["0"]}, TypeError, "integer class IDs"),
        ({"classes": [-1]}, ValueError, "non-negative IDs"),
        ({"classes": [2]}, ValueError, "unknown class IDs"),
    ],
)
def test_runner_rejects_invalid_public_inputs(kwargs, error, match):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(error, match=match):
        InferenceRunner(_ContractStub())(image, **kwargs)


def test_runner_normalizes_single_integer_class_filter():
    result = InferenceRunner(_ContractStub())(
        np.zeros((8, 8, 3), dtype=np.uint8), classes=0
    )
    assert len(result) == 1


@pytest.mark.parametrize("task", ["semantic", "panoptic"])
def test_runner_does_not_truncate_fractional_dense_ids(task):
    runner = InferenceRunner(_ContractStub())
    detections = {task: torch.full((8, 12), 1.5)}
    if task == "panoptic":
        detections["segments_info"] = [{"id": 1, "category_id": 0}]

    with pytest.raises(ValueError, match="integer IDs"):
        runner._wrap_results(detections, (12, 8), None, None)


@pytest.mark.parametrize(
    ("task", "payload_name", "payload"),
    [
        ("segment", "masks", torch.zeros((0, 8, 12), dtype=torch.bool)),
        ("pose", "keypoints", torch.zeros((0, 5, 3))),
    ],
)
def test_runner_preserves_empty_task_and_payload_schema(task, payload_name, payload):
    model = _ContractStub()
    model.task = task
    detections = {
        "boxes": [],
        "scores": [],
        "classes": [],
        "num_detections": 0,
        payload_name: payload,
    }

    result = InferenceRunner(model)._wrap_results(detections, (12, 8), None, None)

    assert result.task == task
    assert len(result) == 0
    assert getattr(result, payload_name) is not None


@pytest.mark.parametrize(
    ("declared_task", "detections", "payload_task"),
    [
        ("detect", {"probs": torch.tensor([0.2, 0.8])}, "classify"),
        ("semantic", {"depth": torch.ones((8, 12))}, "depth"),
        ("point", {"probs": torch.tensor([0.2, 0.8])}, "classify"),
    ],
)
def test_runner_rejects_payload_that_conflicts_with_declared_task(
    declared_task, detections, payload_task
):
    model = _ContractStub()
    model.task = declared_task

    with pytest.raises(ValueError, match=rf"conflicts with its '{payload_task}'"):
        InferenceRunner(model)._wrap_results(detections, (12, 8), None, None)


def test_runner_preserves_matching_classification_task():
    model = _ContractStub()
    model.task = "classify"

    result = InferenceRunner(model)._wrap_results(
        {"probs": torch.tensor([0.2, 0.8])}, (12, 8), None, None
    )

    assert result.task == "classify"


def test_native_still_save_renders_gaze_direction(tmp_path):
    from PIL import Image

    from libreyolo.utils.results import Boxes, Gaze, Results

    runner = InferenceRunner(_ContractStub())

    def save_with_yaw(yaw: float, filename: str) -> np.ndarray:
        result = Results(
            Boxes(
                torch.tensor([[8.0, 8.0, 56.0, 56.0]]),
                torch.tensor([0.9]),
                torch.tensor([0.0]),
            ),
            (64, 64),
            names={0: "face"},
            gaze=Gaze(torch.tensor([[0.0, yaw]])),
            task="gaze",
        )
        output = tmp_path / filename
        runner._save_annotated_image(
            result,
            Image.new("RGB", (64, 64), color="white"),
            output,
        )
        return np.asarray(Image.open(output).convert("RGB"))

    left = save_with_yaw(-0.5, "native-left.png")
    right = save_with_yaw(0.5, "native-right.png")

    assert not np.array_equal(left, right)


@pytest.mark.parametrize(
    ("task", "payload"),
    [
        ("pose", "pose keypoints"),
        ("segment", "segmentation masks"),
        ("obb", "oriented boxes"),
        ("point", "point results"),
        ("gaze", "gaze predictions"),
        ("panoptic", "panoptic maps"),
    ],
)
def test_tiling_rejects_non_box_tasks_before_loading_source(task, payload):
    model = _ContractStub()
    model.task = task

    with pytest.raises(ValueError, match=payload):
        InferenceRunner(model)(object(), tiling=True)


def test_tiling_rejects_detect_model_with_segmentation_payload():
    model = _ContractStub()
    model._is_segmentation = True

    with pytest.raises(ValueError, match="segmentation masks"):
        InferenceRunner(model)(object(), tiling=True)


@pytest.mark.parametrize(
    "source",
    [[], np.empty((0, 8, 12, 3), dtype=np.uint8)],
    ids=["empty-list", "empty-4d-array"],
)
def test_tiling_rejects_unsupported_task_even_for_empty_sources(source):
    model = _ContractStub()
    model.task = "pose"

    with pytest.raises(ValueError, match="pose keypoints"):
        InferenceRunner(model)(source, tiling=True)


@pytest.mark.parametrize("max_det", [0, 1])
def test_tiled_result_applies_global_max_det(max_det):
    result = InferenceRunner(_ContractStub())(
        np.zeros((64, 64, 3), dtype=np.uint8),
        tiling=True,
        overlap_ratio=0.0,
        max_det=max_det,
    )

    assert result.tiled is True
    assert result.num_tiles == 4
    assert len(result) == max_det


@pytest.mark.parametrize("overlap", [float("nan"), float("inf"), -0.1, 1.0, 1.1])
def test_slice_generation_rejects_unsafe_overlap(overlap):
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\)"):
        get_slice_bboxes(64, 64, slice_size=32, overlap_ratio=overlap)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"image_width": 0}, ValueError, "image_width must be positive"),
        ({"image_height": 0}, ValueError, "image_height must be positive"),
        ({"slice_size": 0}, ValueError, "slice_size must be positive"),
        ({"slice_size": 32.0}, TypeError, "slice_size must be an integer"),
    ],
)
def test_slice_generation_rejects_invalid_dimensions(kwargs, error, match):
    options = {"image_width": 64, "image_height": 64, "slice_size": 32}
    options.update(kwargs)
    with pytest.raises(error, match=match):
        get_slice_bboxes(**options)


def test_near_full_overlap_still_terminates_with_unique_tiles():
    slices = get_slice_bboxes(33, 33, slice_size=32, overlap_ratio=0.999999)
    assert slices == [
        (0, 0, 32, 32),
        (1, 0, 33, 32),
        (0, 1, 32, 33),
        (1, 1, 33, 33),
    ]
