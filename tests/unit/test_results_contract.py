"""Cross-task contract tests for the public Results containers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.utils.results import (
    Boxes,
    DepthMap,
    Gaze,
    Keypoints,
    Masks,
    Matte,
    OBB,
    OCRRegions,
    PanopticSegmentation,
    Points,
    Probs,
    RestoredImage,
    Results,
    SemanticMask,
)

pytestmark = pytest.mark.unit


def _boxes(n: int = 2) -> Boxes:
    rows = torch.tensor(
        [[0.0, 0.0, 3.0, 3.0], [1.0, 1.0, 4.0, 3.0]],
        dtype=torch.float32,
    )[:n]
    return Boxes(rows, torch.tensor([0.9, 0.8])[:n], torch.tensor([0.0, 1.0])[:n])


def _detect() -> Results:
    return Results(_boxes(), (4, 5), names={0: "a", 1: "b"}, task="detect")


def _segment() -> Results:
    masks = torch.zeros((2, 4, 5), dtype=torch.bool)
    masks[0, :2, :2] = True
    masks[1, 2:, 2:] = True
    return Results(
        _boxes(),
        (4, 5),
        names={0: "a", 1: "b"},
        masks=Masks(masks, (4, 5)),
        task="segment",
    )


def _pose() -> Results:
    keypoints = torch.tensor(
        [
            [[1.0, 1.0, 0.9], [2.0, 2.0, 0.8]],
            [[2.0, 1.0, 0.7], [3.0, 2.0, 0.6]],
        ]
    )
    return Results(
        _boxes(),
        (4, 5),
        names={0: "a", 1: "b"},
        keypoints=Keypoints(keypoints, (4, 5)),
        task="pose",
    )


def _classify() -> Results:
    return Results(
        None,
        (4, 5),
        names={0: "a", 1: "b"},
        probs=Probs(torch.tensor([0.2, 0.8])),
        task="classify",
    )


def _gaze() -> Results:
    return Results(
        _boxes(),
        (4, 5),
        names={0: "face", 1: "face"},
        gaze=Gaze(torch.tensor([[0.1, -0.2], [0.2, 0.1]]), (4, 5)),
        task="gaze",
    )


def _obb() -> Results:
    obb = torch.tensor(
        [
            [1.5, 1.5, 3.0, 2.0, 0.1, 0.9, 0.0],
            [2.5, 2.0, 2.0, 2.0, -0.2, 0.8, 1.0],
        ]
    )
    return Results(
        _boxes(),
        (4, 5),
        names={0: "a", 1: "b"},
        obb=OBB(obb, (4, 5)),
        task="obb",
    )


def _point() -> Results:
    return Results(
        None,
        (4, 5),
        names={0: "a", 1: "b"},
        points=Points(torch.tensor([[1.0, 1.0, 0.0, 0.9], [3.0, 2.0, 1.0, 0.8]])),
        task="point",
    )


def _semantic() -> Results:
    return Results(
        None,
        (4, 5),
        names={0: "a", 1: "b"},
        semantic_mask=SemanticMask(torch.tensor([[0] * 5, [0] * 5, [1] * 5, [1] * 5])),
        task="semantic",
    )


def _panoptic() -> Results:
    data = torch.zeros((4, 5), dtype=torch.int64)
    data[:, :2] = 1
    return Results(
        None,
        (4, 5),
        names={0: "thing"},
        panoptic=PanopticSegmentation(
            data,
            [{"id": 1, "category_id": 0, "isthing": True, "score": 0.9}],
        ),
        task="panoptic",
    )


def _depth() -> Results:
    return Results(
        None,
        (4, 5),
        depth_map=DepthMap(torch.arange(20, dtype=torch.float32).reshape(4, 5)),
        task="depth",
    )


def _restore() -> Results:
    image = torch.zeros((4, 5, 3), dtype=torch.uint8)
    image[..., 0] = 255
    return Results(None, (4, 5), restored=RestoredImage(image), task="restore")


def _matte() -> Results:
    matte = torch.zeros((4, 5), dtype=torch.float32)
    matte[:, :2] = 1.0
    return Results(None, (4, 5), matte=Matte(matte), task="matte")


def _ocr() -> Results:
    polygons = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
            [[1.0, 2.0], [4.0, 2.0], [4.0, 3.0], [1.0, 3.0]],
        ]
    )
    return Results(
        None,
        (4, 5),
        ocr=OCRRegions(polygons, ["one", "two"], [0.9, 0.8], [0.7, 0.6]),
        task="ocr",
    )


TASK_CASES: tuple[tuple[str, Callable[[], Results], int], ...] = (
    ("detect", _detect, 2),
    ("segment", _segment, 2),
    ("pose", _pose, 2),
    ("classify", _classify, 1),
    ("gaze", _gaze, 2),
    ("obb", _obb, 2),
    ("point", _point, 2),
    ("semantic", _semantic, 1),
    ("panoptic", _panoptic, 1),
    ("depth", _depth, 1),
    ("restore", _restore, 1),
    ("matte", _matte, 1),
    ("ocr", _ocr, 2),
)


@pytest.mark.parametrize(("task", "factory", "expected_len"), TASK_CASES)
def test_all_tasks_have_finite_consistent_collection_contract(task, factory, expected_len):
    result = factory()

    assert result.task == task
    assert len(result) == expected_len
    assert len(list(result)) == expected_len
    assert all(len(item) == 1 for item in result)
    assert len(result[:]) == expected_len
    assert len(result[...]) == expected_len
    assert len(result[-1]) == 1
    with pytest.raises(IndexError):
        result[expected_len]


@pytest.mark.parametrize(("task", "factory", "_expected_len"), TASK_CASES)
def test_all_tasks_preserve_schema_metadata_moves_json_and_plot(
    task, factory, _expected_len
):
    result = factory()
    result.frame_idx = 7
    result.saved_path = "runs/result.png"
    result.tiled = True
    result.num_tiles = 4
    result.tiles_path = "runs/tiles"
    result.grid_path = "runs/grid.png"

    for converted in (result.cpu(), result.to("cpu"), result.numpy(), result[:1]):
        assert converted.task == task
        assert converted.orig_shape == result.orig_shape
        assert converted.frame_idx == 7
        assert converted.saved_path == "runs/result.png"
        assert converted.tiled is True
        assert converted.num_tiles == 4
        assert converted.tiles_path == "runs/tiles"
        assert converted.grid_path == "runs/grid.png"

    summary = result.summary()
    assert isinstance(summary, list)
    assert json.loads(result.to_json()) == summary
    source = Image.new("RGB", (5, 4), color=(255, 255, 255))
    plotted = result.plot(image=source)
    assert isinstance(plotted, Image.Image)
    assert plotted.mode == "RGB"


@pytest.mark.parametrize("task", [case[0] for case in TASK_CASES])
def test_empty_results_keep_explicit_task_schema(task):
    empty = Results(None, (4, 5), task=task, frame_idx=3)

    assert empty.task == task
    assert len(empty) == 0
    assert list(empty) == []
    assert empty[:].task == task
    assert empty.cpu().task == task
    assert empty.numpy().frame_idx == 3
    with pytest.raises(IndexError):
        empty[0]


@pytest.mark.parametrize(
    "payload",
    [
        Probs(torch.tensor([0.2, 0.8])),
        SemanticMask(torch.zeros((2, 3), dtype=torch.int64)),
        PanopticSegmentation(torch.zeros((2, 3), dtype=torch.int64), []),
        DepthMap(torch.ones((2, 3))),
        RestoredImage(torch.zeros((2, 3, 3), dtype=torch.uint8)),
        Matte(torch.zeros((2, 3))),
    ],
    ids=["probs", "semantic", "panoptic", "depth", "restore", "matte"],
)
def test_whole_image_payloads_are_finite_singletons(payload):
    assert len(payload) == 1
    assert len(list(payload)) == 1
    assert len(payload[-1]) == 1
    with pytest.raises(IndexError):
        payload[1]
    with pytest.raises(IndexError):
        payload[-2]


@pytest.mark.parametrize(
    "data",
    [
        torch.tensor([[float("nan"), float("inf")]]),
        np.array([[2.0, np.nan]], dtype=np.float32),
    ],
)
def test_constant_or_all_nonfinite_depth_normalizes_to_finite_zeros(data):
    normalized = DepthMap(data).normalized()
    values = normalized.detach().cpu().numpy() if isinstance(normalized, torch.Tensor) else normalized
    assert np.isfinite(values).all()
    assert np.count_nonzero(values) == 0


def test_whole_image_empty_slice_preserves_task_without_payload():
    result = _classify()

    empty = result[1:]

    assert empty.task == "classify"
    assert empty.probs is None
    assert len(empty) == 0


@pytest.mark.parametrize(
    ("payload", "task", "expected"),
    [
        (Masks(torch.zeros((2, 4, 5)), (4, 5)), "segment", 2),
        (Keypoints(torch.zeros((2, 3, 3))), "pose", 2),
        (OBB(torch.zeros((2, 7))), "obb", 2),
        (Gaze(torch.zeros((2, 2))), "gaze", 2),
    ],
)
def test_boxless_instance_payloads_define_results_length(payload, task, expected):
    key = {"segment": "masks", "pose": "keypoints", "obb": "obb", "gaze": "gaze"}[task]
    result = Results(None, (4, 5), task=task, **{key: payload})

    assert len(result) == expected
    assert len(list(result)) == expected


def test_payload_alignment_and_canvas_shapes_fail_at_construction():
    with pytest.raises(ValueError, match="align row-for-row"):
        Results(_boxes(2), (4, 5), masks=Masks(torch.zeros((1, 4, 5)), (4, 5)))
    with pytest.raises(ValueError, match="original canvas"):
        Results(
            None,
            (4, 5),
            semantic_mask=SemanticMask(torch.zeros((3, 5), dtype=torch.int64)),
        )
    with pytest.raises(ValueError, match=r"shape \(N, 4\)"):
        Boxes(torch.zeros((2, 5)), torch.zeros(2), torch.zeros(2))
    with pytest.raises(ValueError, match=r"\(N, H, W\)"):
        Masks(torch.zeros((4, 5)), (4, 5))
    with pytest.raises(ValueError, match=r"\(N, K, 2\|3\)"):
        Keypoints(torch.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="classification vector"):
        Probs(torch.zeros((1, 2)))

    empty = Boxes(torch.tensor([]), torch.tensor([]), torch.tensor([]))
    assert empty.xyxy.shape == (0, 4)
    with pytest.raises(ValueError, match="orig_shape is required"):
        Masks(torch.zeros((0, 2, 2)), None)
    with pytest.raises(ValueError, match="orig_shape is required"):
        Results(None, None)


def test_whole_image_payloads_cannot_mix_with_rows_or_each_other():
    with pytest.raises(ValueError, match="cannot mix per-instance and whole-image"):
        Results(
            _boxes(),
            (4, 5),
            probs=Probs(torch.tensor([0.2, 0.8])),
            task="detect",
        )
    with pytest.raises(ValueError, match="only one whole-image payload"):
        Results(
            None,
            (4, 5),
            probs=Probs(torch.tensor([0.2, 0.8])),
            depth_map=DepthMap(torch.ones((4, 5))),
            task="classify",
        )


def test_ambiguous_or_exclusive_instance_payload_mixes_are_rejected():
    with pytest.raises(ValueError, match="require boxes as their row anchor"):
        Results(
            None,
            (4, 5),
            masks=Masks(torch.zeros((1, 4, 5)), (4, 5)),
            keypoints=Keypoints(torch.zeros((1, 2, 3)), (4, 5)),
            task="detect",
        )
    with pytest.raises(ValueError, match="points is an exclusive"):
        Results(
            _boxes(1),
            (4, 5),
            points=Points(torch.tensor([[1.0, 1.0, 0.0, 0.9]]), (4, 5)),
            task="point",
        )
    with pytest.raises(ValueError, match="ocr is an exclusive"):
        Results(
            _boxes(1),
            (4, 5),
            ocr=OCRRegions(
                torch.zeros((1, 4, 2)),
                ["text"],
                orig_shape=(4, 5),
            ),
            task="ocr",
        )


def test_explicit_task_must_match_single_task_payload():
    with pytest.raises(ValueError, match="conflicts with its 'segment' payload"):
        Results(
            _boxes(1),
            (4, 5),
            masks=Masks(torch.zeros((1, 4, 5)), (4, 5)),
            task="detect",
        )
    with pytest.raises(ValueError, match="conflicts with its 'depth' whole-image"):
        Results(
            None,
            (4, 5),
            depth_map=DepthMap(torch.ones((4, 5))),
            task="semantic",
        )


@pytest.mark.parametrize(
    ("task", "payload_name"),
    [("segment", "masks"), ("pose", "keypoints"), ("obb", "obb"), ("gaze", "gaze")],
)
def test_nonempty_instance_tasks_require_their_task_payload(task, payload_name):
    with pytest.raises(ValueError, match=rf"require an aligned {payload_name} payload"):
        Results(_boxes(1), (4, 5), task=task)

    empty = Results(
        Boxes(torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0)),
        (4, 5),
        task=task,
    )
    assert empty.task == task and len(empty) == 0


def test_empty_tracked_boxes_keep_empty_id_schema_across_moves_and_slices():
    tracked = Boxes(
        torch.zeros((0, 4)),
        torch.zeros(0),
        torch.zeros(0),
        id=torch.zeros(0, dtype=torch.int64),
    )
    result = Results(tracked, (4, 5), task="detect")

    assert result.track_id is not None
    assert result.boxes is not None and result.boxes.is_track
    assert result.track_id.shape == (0,)
    for converted in (result.cpu(), result.numpy(), result[:]):
        assert converted.track_id is not None
        assert converted.boxes is not None and converted.boxes.is_track
        assert converted.track_id.shape == (0,)


def test_update_is_transactional_when_replacement_breaks_contract():
    result = _detect()
    original_boxes = result.boxes
    original_task = result.task

    with pytest.raises(ValueError, match="cannot mix per-instance and whole-image"):
        result.update(probs=Probs(torch.tensor([0.2, 0.8])))

    assert result.boxes is original_boxes
    assert result.probs is None
    assert result.task == original_task
    assert len(result) == 2


def test_boxes_reject_mixed_tensor_and_array_fields_before_data_access():
    with pytest.raises(TypeError, match="same tensor/array container"):
        Boxes(torch.zeros((1, 4)), np.zeros(1), torch.zeros(1))


def test_selector_accepts_scalar_arrays_and_rejects_float_or_boolean_scalars():
    result = _detect()

    assert len(result[np.array(0)]) == 1
    assert len(result[...]) == len(result)
    with pytest.raises(TypeError, match="scalar boolean"):
        result[np.array(True)]
    with pytest.raises(TypeError, match="integers or booleans"):
        result[np.array([], dtype=np.float32)]


def test_parent_orig_shape_propagates_in_place_to_instance_payloads():
    gaze = Gaze(torch.zeros((2, 2)))
    keypoints = Keypoints(torch.zeros((2, 1, 3)))
    obb = OBB(torch.zeros((2, 7)))

    result = Results(
        _boxes(),
        (4, 5),
        task="detect",
        gaze=gaze,
        keypoints=keypoints,
        obb=obb,
    )

    assert result.gaze is gaze and gaze.orig_shape == (4, 5)
    assert result.keypoints is keypoints and keypoints.orig_shape == (4, 5)
    assert result.obb is obb and obb.orig_shape == (4, 5)


def test_ocr_boolean_indexing_and_numpy_keep_every_field_aligned():
    regions = _ocr().ocr
    assert regions is not None

    selected = regions[np.array([True, False])]
    converted = selected.numpy()

    assert selected.texts == ["one"]
    assert np.asarray(selected.conf).shape == (1,)
    assert isinstance(converted.data, np.ndarray)
    assert isinstance(converted.conf, np.ndarray)
    assert converted.texts == ["one"]
    with pytest.raises(IndexError, match="boolean index"):
        regions[np.array([True])]


def test_mask_contours_preserve_components_and_holes_and_are_cached(monkeypatch):
    data = torch.zeros((1, 12, 12), dtype=torch.bool)
    data[0, 1:10, 1:6] = True
    data[0, 3:6, 3:5] = False
    data[0, 2:6, 8:11] = True
    masks = Masks(data, (12, 12))
    calls = 0
    original = cv2.findContours

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "findContours", counted)
    result = Results(_boxes(1), (12, 12), masks=masks, task="segment")

    _ = masks.xy
    _ = masks.xyn
    contours = masks.contours
    row = result.summary()[0]
    _ = result.to_json()

    assert calls == 1
    assert len(contours[0]) >= 3
    assert sum(not contour["is_hole"] for contour in contours[0]) >= 2
    assert any(contour["is_hole"] for contour in contours[0])
    assert len(row["mask_contours"]) == len(contours[0])

    data[0].zero_()
    assert masks.xy[0].shape == (0, 2)
    assert calls == 2
    data.numpy()[0, 2:6, 2:6] = True
    assert masks.xy[0].shape[0] > 0
    assert calls == 3


def test_pose_summary_and_json_include_keypoint_coordinates_and_confidence():
    result = _pose()

    row = result.summary(normalize=True, decimals=3)[0]

    assert row["keypoints"] == {
        "x": [0.2, 0.4],
        "y": [0.25, 0.5],
        "confidence": [0.9, 0.8],
    }
    assert json.loads(result.to_json(normalize=True, decimals=3))[0]["keypoints"] == row[
        "keypoints"
    ]


def test_panoptic_map_and_metadata_ids_must_be_bijective():
    data = torch.ones((2, 2), dtype=torch.int64)

    with pytest.raises(ValueError, match="must match exactly"):
        PanopticSegmentation(data, [])
    with pytest.raises(ValueError, match="must match exactly"):
        PanopticSegmentation(
            data,
            [
                {"id": 1, "category_id": 0},
                {"id": 2, "category_id": 0},
            ],
        )

    payload = PanopticSegmentation(
        torch.ones((2, 2), dtype=torch.int64),
        [{"id": 1, "category_id": 0}],
    )
    row = Results(None, (2, 2), panoptic=payload).summary()[0]
    assert "isthing" not in row


@pytest.mark.parametrize(
    "data",
    [
        torch.full((2, 2), 1.9),
        torch.full((2, 2), float("nan")),
        torch.full((2, 2), float("inf")),
        torch.ones((2, 2), dtype=torch.bool),
        torch.full((2, 2), -1, dtype=torch.int64),
        np.full((2, 2), 1.5, dtype=np.float32),
    ],
    ids=["fractional", "nan", "inf", "bool", "negative", "numpy-fractional"],
)
@pytest.mark.parametrize("payload_type", [SemanticMask, PanopticSegmentation])
def test_dense_id_maps_reject_non_integer_nonfinite_bool_and_negative(data, payload_type):
    with pytest.raises(ValueError, match="integer IDs|negative IDs"):
        payload_type(data)


@pytest.mark.parametrize(
    "segment",
    [
        {"id": 0, "category_id": 0},
        {"id": -1, "category_id": 0},
        {"id": 1.0, "category_id": 0},
        {"id": 1, "category_id": -1},
        {"id": 1, "category_id": 0.0},
        {"id": True, "category_id": 0},
    ],
)
def test_panoptic_segment_metadata_requires_positive_integer_ids(segment):
    with pytest.raises(ValueError, match="positive|non-negative"):
        PanopticSegmentation(torch.ones((2, 2), dtype=torch.int64), [segment])


def test_panoptic_isthing_metadata_must_be_boolean():
    with pytest.raises(ValueError, match="isthing must be boolean"):
        PanopticSegmentation(
            torch.ones((2, 2), dtype=torch.int64),
            [{"id": 1, "category_id": 0, "isthing": "false"}],
        )


def test_tracked_obb_summary_preserves_ids_with_and_without_boxes():
    tracked_obb = OBB(
        torch.tensor([[2.0, 2.0, 2.0, 2.0, 0.0, 42.0, 0.9, 0.0]]),
        (4, 5),
    )
    standalone = Results(None, (4, 5), obb=tracked_obb, task="obb")
    anchored = Results(_boxes(1), (4, 5), obb=tracked_obb, task="obb")

    assert standalone.summary()[0]["track_id"] == 42
    assert anchored.summary()[0]["track_id"] == 42
    assert anchored.track_id is not None and anchored.track_id.tolist() == [42.0]
    assert anchored.boxes is not None and anchored.boxes.id.tolist() == [42.0]

    with pytest.raises(ValueError, match="class ids must match"):
        Results(
            _boxes(1),
            (4, 5),
            obb=OBB(
                torch.tensor([[2.0, 2.0, 2.0, 2.0, 0.0, 0.9, 1.0]]),
                (4, 5),
            ),
            task="obb",
        )


def test_tracked_boxes_drive_untracked_obb_plot_and_summary(monkeypatch):
    boxes = Boxes(
        torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
        torch.tensor([0.9]),
        torch.tensor([0.0]),
        id=torch.tensor([7]),
    )
    obb = OBB(torch.tensor([[2.0, 2.0, 2.0, 2.0, 0.0, 0.9, 0.0]]), (4, 5))
    result = Results(boxes, (4, 5), obb=obb, task="obb")
    captured = {}

    def fake_draw(img, *_args, **kwargs):
        captured["track_ids"] = kwargs["track_ids"]
        return img

    monkeypatch.setattr("libreyolo.utils.drawing.draw_obb", fake_draw)
    result.plot(image=Image.new("RGB", (5, 4)))

    assert result.summary()[0]["track_id"] == 7
    assert captured["track_ids"] == [7]


@pytest.mark.parametrize("value", [1.9, float("nan"), float("inf"), -1.0])
def test_tracking_ids_reject_fractional_nonfinite_and_negative_values(value):
    with pytest.raises(ValueError, match="track ids"):
        Boxes(
            torch.zeros((1, 4)),
            torch.ones(1),
            torch.zeros(1),
            id=torch.tensor([value]),
        )
    with pytest.raises(ValueError, match="OBB track ids"):
        OBB(torch.tensor([[2.0, 2.0, 2.0, 2.0, 0.0, value, 0.9, 0.0]]))


def test_dtype_moves_preserve_or_reject_track_identity_without_rounding():
    result = Results(
        Boxes(
            torch.zeros((1, 4)),
            torch.ones(1),
            torch.zeros(1),
            id=torch.tensor([2049], dtype=torch.int64),
        ),
        (4, 5),
    )
    moved = result.to(dtype=torch.float16)
    assert moved.track_id.dtype == torch.int64
    assert moved.summary()[0]["track_id"] == 2049

    tracked_obb = OBB(
        torch.tensor([[2.0, 2.0, 2.0, 2.0, 0.0, 2049.0, 0.9, 0.0]])
    )
    with pytest.raises(ValueError, match="would change track ids"):
        tracked_obb.to(dtype=torch.float16)


def test_metadata_json_envelope_preserves_empty_task_and_frame_identity():
    result = Results(
        None,
        (4, 5),
        task="semantic",
        frame_idx=11,
        saved_path="runs/frame.png",
        tiled=True,
        num_tiles=4,
        tiles_path="runs/tiles",
        grid_path="runs/grid.png",
    )

    assert json.loads(result.to_json()) == []
    payload = json.loads(result.to_json(include_metadata=True))
    assert payload["task"] == "semantic"
    assert payload["orig_shape"] == [4, 5]
    assert payload["frame_idx"] == 11
    assert payload["saved_path"] == "runs/frame.png"
    assert payload["tiled"] is True and payload["num_tiles"] == 4
    assert payload["results"] == []

    non_native = Results(
        None,
        (4, 5),
        task="semantic",
        path=Path("input.jpg"),
        saved_path=Path("output.png"),
        frame_idx=np.int64(3),
        names={np.int64(0): "class-zero"},
        speed={"preprocess": np.float32(1.2)},
    )
    normalized = json.loads(non_native.to_json(include_metadata=True))
    assert normalized["path"] == "input.jpg"
    assert normalized["saved_path"] == "output.png"
    assert normalized["frame_idx"] == 3
    assert normalized["names"] == {"0": "class-zero"}
    assert normalized["speed"]["preprocess"] == pytest.approx(1.2)


def test_results_plot_uses_matte_checkerboard_and_restore_canvas():
    white = Image.new("RGB", (5, 4), color="white")

    matte_preview = np.asarray(_matte().plot(image=white))
    restored = _restore().plot(image=white)

    assert not np.array_equal(matte_preview, np.asarray(white))
    assert np.array_equal(np.asarray(restored)[..., 0], np.full((4, 5), 255))


def test_boxless_gaze_has_data_contract_but_plot_requires_face_boxes():
    result = Results(None, (4, 5), gaze=Gaze(torch.zeros((1, 2))), task="gaze")

    assert len(result) == 1
    assert result.summary()[0]["gaze"]["pitch_rad"] == 0.0
    with pytest.raises(ValueError, match="requires aligned face boxes"):
        result.plot(image=Image.new("RGB", (5, 4)))

    empty = Results(None, (4, 5), gaze=Gaze(torch.zeros((0, 2))), task="gaze")
    assert empty.plot(image=Image.new("RGB", (5, 4))).size == (5, 4)


def test_direct_matte_save_rejects_non_png_path(tmp_path):
    with pytest.raises(ValueError, match=r"requires a \.png path"):
        _matte().save(tmp_path / "cutout.jpg", image=Image.new("RGB", (5, 4)))
