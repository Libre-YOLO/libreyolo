"""Hermetic regression tests for validator metric-input contracts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libreyolo.validation.classify_validator import ClassifyValidator
from libreyolo.validation.coco_evaluator import COCOEvaluator
from libreyolo.validation.depth_validator import DepthValidator
from libreyolo.validation.detection_validator import DetectionValidator
from libreyolo.validation.fomo_validator import FOMOValidator
from libreyolo.validation.matte_validator import matte_mae, s_measure
from libreyolo.validation.obb_validator import OBBValidator
from libreyolo.validation.ocr_validator import match_image
from libreyolo.validation.panoptic_quality import PanopticQuality
from libreyolo.validation.point_validator import PointValidator
from libreyolo.validation.pose_validator import PoseValidator
from libreyolo.validation.restore_validator import RestoreValidator
from libreyolo.validation.semantic_validator import SemanticValidator

pytestmark = pytest.mark.unit


def _classification_validator(num_classes: int = 2) -> ClassifyValidator:
    validator = ClassifyValidator.__new__(ClassifyValidator)
    validator._num_classes = num_classes
    validator._init_metrics()
    return validator


def test_classification_rejects_batch_broadcasting():
    validator = _classification_validator()

    with pytest.raises(ValueError, match="batch size mismatch"):
        validator._update_metrics(
            torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
            torch.tensor([0]),
            None,
        )


def test_classification_rejects_nonfinite_logits_and_invalid_targets():
    validator = _classification_validator()
    with pytest.raises(ValueError, match="non-finite"):
        validator._update_metrics(
            torch.tensor([[float("nan"), 0.0]]), torch.tensor([0]), None
        )
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        validator._update_metrics(
            torch.tensor([[1.0, 0.0]]), torch.tensor([2]), None
        )


def _depth_validator() -> DepthValidator:
    validator = DepthValidator.__new__(DepthValidator)
    validator._init_metrics()
    return validator


def test_depth_rejects_truncated_batches():
    validator = _depth_validator()
    with pytest.raises(ValueError, match="batch size mismatch"):
        validator._update_metrics(
            torch.ones(1, 2, 2), torch.ones(2, 2, 2), None
        )


def test_depth_rejects_nonfinite_predictions_over_valid_ground_truth():
    validator = _depth_validator()
    prediction = torch.tensor([[[0.5, float("nan")], [0.25, 0.25]]])
    target = torch.tensor([[[2.0, 2.0], [4.0, 4.0]]])

    with pytest.raises(ValueError, match="evaluated region"):
        validator._update_metrics(prediction, target, None)


def _restore_validator() -> RestoreValidator:
    validator = RestoreValidator.__new__(RestoreValidator)
    validator._init_metrics()
    return validator


def test_restoration_rejects_truncated_batches():
    validator = _restore_validator()
    info = [{"orig_shape": (2, 2)}, {"orig_shape": (2, 2)}]
    with pytest.raises(ValueError, match="batch size mismatch"):
        validator._update_metrics(
            torch.zeros(1, 3, 2, 2), torch.zeros(2, 3, 2, 2), info
        )


def test_restoration_rejects_nonfinite_evaluated_pixels():
    validator = _restore_validator()
    prediction = torch.zeros(1, 3, 2, 2)
    prediction[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        validator._update_metrics(
            prediction,
            torch.zeros_like(prediction),
            [{"orig_shape": (2, 2)}],
        )


def test_restoration_does_not_clamp_infinity_into_a_finite_prediction():
    validator = _restore_validator()
    prediction = torch.zeros(1, 3, 2, 2)
    prediction[0, 0, 0, 0] = float("inf")

    with pytest.raises(ValueError, match="non-finite"):
        validator._postprocess_predictions(
            prediction,
            (None, torch.zeros_like(prediction), None, None),
        )


def _semantic_validator() -> SemanticValidator:
    validator = SemanticValidator.__new__(SemanticValidator)
    validator._num_classes = 2
    validator._ignore_index = 255
    validator._init_metrics()
    return validator


def test_semantic_rejects_nonfinite_logits_on_valid_pixels():
    validator = _semantic_validator()
    logits = torch.zeros(1, 2, 2, 2)
    logits[0, 0, 0, 0] = float("nan")
    target = torch.zeros(1, 2, 2, dtype=torch.long)

    with pytest.raises(ValueError, match="non-finite"):
        validator._postprocess_predictions(logits, (None, target, None, None))


def test_semantic_rejects_invalid_prediction_classes_instead_of_clamping():
    validator = _semantic_validator()
    prediction = torch.tensor([[[0, 2], [0, 1]]])
    target = torch.tensor([[[0, 1], [0, 1]]])

    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        validator._update_metrics(prediction, target, None)


@pytest.mark.parametrize(
    ("segment_map", "segments", "message"),
    [
        (np.array([[1, 2]]), [{"id": 1, "category_id": 0}], "missing from"),
        (
            np.array([[1, 1]]),
            [{"id": 1, "category_id": 0}, {"id": 2, "category_id": 0}],
            "absent from the map",
        ),
    ],
)
def test_panoptic_rejects_map_metadata_mismatches(segment_map, segments, message):
    pq = PanopticQuality(num_classes=1)

    with pytest.raises(ValueError, match=message):
        pq.update(segment_map, segments, segment_map, segments)


def test_panoptic_rejects_duplicate_ids_and_invalid_categories():
    segment_map = np.array([[1, 1]])
    with pytest.raises(ValueError, match="duplicate id"):
        PanopticQuality(num_classes=1).update(
            segment_map,
            [{"id": 1, "category_id": 0}, {"id": 1, "category_id": 0}],
            np.zeros_like(segment_map),
            [],
        )
    with pytest.raises(ValueError, match="category_id"):
        PanopticQuality(num_classes=1).update(
            segment_map,
            [{"id": 1, "category_id": 1}],
            np.zeros_like(segment_map),
            [],
        )


def test_panoptic_rejects_nonfinite_segment_ids():
    with pytest.raises(ValueError, match="non-finite"):
        PanopticQuality().update(
            np.array([[float("nan")]]),
            [],
            np.zeros((1, 1), dtype=np.int64),
            [],
        )


def test_panoptic_handles_the_largest_packable_segment_id_without_overflow():
    segment_id = (1 << 32) - 1
    segment_map = np.array([[segment_id]], dtype=np.uint64)
    segments = [{"id": segment_id, "category_id": 0}]
    pq = PanopticQuality(num_classes=1)

    pq.update(segment_map, segments, segment_map, segments)

    assert pq.compute()["metrics/PQ"] == pytest.approx(1.0)


def test_detection_rejects_invalid_prediction_classes():
    validator = DetectionValidator.__new__(DetectionValidator)
    validator.nc = 2
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "scores": torch.tensor([0.9]),
        "classes": torch.tensor([2]),
    }

    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        validator._validate_prediction(prediction, 0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_detection_rejects_nonfinite_prediction_payloads(value):
    validator = DetectionValidator.__new__(DetectionValidator)
    validator.nc = 1
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, value, 1.0]]),
        "scores": torch.tensor([0.9]),
        "classes": torch.tensor([0]),
    }

    with pytest.raises(ValueError, match="non-finite"):
        validator._validate_prediction(prediction, 0)


def test_detection_rejects_malformed_target_rank():
    validator = DetectionValidator.__new__(DetectionValidator)
    validator.nc = 1
    validator.config = SimpleNamespace(save_plots=False)
    validator.coco_evaluator = SimpleNamespace(update=lambda *_args: None)
    prediction = {
        "boxes": torch.zeros((0, 4)),
        "scores": torch.zeros(0),
        "classes": torch.zeros(0, dtype=torch.long),
    }

    with pytest.raises(ValueError, match=r"\[B, N, 5\]"):
        validator._update_metrics(
            [prediction],
            targets=torch.zeros((1, 5)),
            img_info=[(1, 1)],
            img_ids=[1],
        )


def test_detection_rejects_invalid_ground_truth_classes_in_plot_parser():
    validator = DetectionValidator.__new__(DetectionValidator)
    validator.nc = 2
    validator.val_preproc = SimpleNamespace(uses_letterbox=False)
    validator._actual_imgsz = 10
    targets = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 2.0],
        ]
    )

    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        validator._parse_gt_boxes(targets, orig_h=10, orig_w=10)


def test_detection_rejects_prediction_target_batch_mismatch():
    validator = DetectionValidator.__new__(DetectionValidator)
    validator.nc = 1
    validator.config = SimpleNamespace(save_plots=False)
    validator.coco_evaluator = SimpleNamespace(update=lambda *_args: None)
    prediction = {
        "boxes": torch.zeros((0, 4)),
        "scores": torch.zeros(0),
        "classes": torch.zeros(0, dtype=torch.long),
    }

    with pytest.raises(ValueError, match="batch size mismatch"):
        validator._update_metrics(
            [prediction, prediction],
            targets=torch.zeros((1, 0, 5)),
            img_info=[(1, 1), (1, 1)],
            img_ids=[1, 2],
        )


def test_detection_rejects_invalid_active_gt_without_plotting():
    updates = []
    validator = DetectionValidator.__new__(DetectionValidator)
    validator.nc = 1
    validator.config = SimpleNamespace(save_plots=False)
    validator.coco_evaluator = SimpleNamespace(
        update=lambda prediction, image_id: updates.append((prediction, image_id))
    )
    validator.val_preproc = SimpleNamespace(uses_letterbox=False)
    validator._actual_imgsz = 1
    prediction = {
        "boxes": torch.zeros((0, 4)),
        "scores": torch.zeros(0),
        "classes": torch.zeros(0, dtype=torch.long),
    }

    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        validator._update_metrics(
            [prediction],
            targets=torch.tensor([[[0.0, 0.0, 1.0, 1.0, 2.0]]]),
            img_info=[(1, 1)],
            img_ids=[1],
        )

    assert updates == []


def test_coco_evaluator_does_not_fall_back_to_an_invalid_category():
    coco = SimpleNamespace(getCatIds=lambda: [0, 1])
    evaluator = COCOEvaluator(coco)

    with pytest.raises(ValueError, match="not dataset categories"):
        evaluator.update(
            {"boxes": [[0, 0, 1, 1]], "scores": [0.9], "classes": [7]},
            image_id=1,
        )


def test_coco_evaluator_rejects_invalid_mapped_category():
    coco = SimpleNamespace(getCatIds=lambda: [3])
    evaluator = COCOEvaluator(coco, label_to_category_id={0: 7})

    with pytest.raises(ValueError, match="not present in the dataset"):
        evaluator.update(
            {"boxes": [[0, 0, 1, 1]], "scores": [0.9], "classes": [0]},
            image_id=1,
        )


def test_obb_rejects_invalid_classes_and_nonfinite_rows():
    validator = OBBValidator.__new__(OBBValidator)
    validator.nc = 1
    validator._predictions_by_class = {0: []}
    common = {
        "targets": torch.zeros((1, 0, 5)),
        "img_info": [(1, 1)],
        "img_ids": [1],
    }

    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        validator._update_metrics(
            [{"obb": torch.tensor([[0, 0, 1, 1, 0, 0.9, 1]])}],
            **common,
        )
    with pytest.raises(ValueError, match="non-finite"):
        validator._update_metrics(
            [{"obb": torch.tensor([[0, 0, float("nan"), 1, 0, 0.9, 0]])}],
            **common,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        validator._update_metrics(
            [{"obb": torch.zeros((0, 7))}],
            targets=torch.tensor([[[0.0, 0.0, 1.0, 1.0, 1.0]]]),
            img_info=[(1, 1)],
            img_ids=[1],
        )


def test_point_rejects_nonfinite_and_fractional_prediction_payloads():
    validator = PointValidator.__new__(PointValidator)
    validator.nc = 2
    validator._records = []
    validator.val_preproc = None
    validator._actual_imgsz = 1
    common = {
        "targets": torch.zeros((1, 0, 5)),
        "img_info": [(1, 1)],
        "img_ids": [1],
    }

    with pytest.raises(ValueError, match="non-finite"):
        validator._update_metrics(
            [
                {
                    "xy_norm": np.array([[float("nan"), 0.5]]),
                    "scores": np.array([0.9]),
                    "classes": np.array([0]),
                }
            ],
            **common,
        )
    with pytest.raises(ValueError, match="integer-valued"):
        validator._update_metrics(
            [
                {
                    "xy_norm": np.array([[0.5, 0.5]]),
                    "scores": np.array([0.9]),
                    "classes": np.array([0.5]),
                }
            ],
            **common,
        )


def test_point_rejects_invalid_ground_truth_classes():
    validator = PointValidator.__new__(PointValidator)
    validator.nc = 2
    validator._records = []
    validator.val_preproc = None
    validator._actual_imgsz = 1

    class _PointModel:
        @staticmethod
        def _parse_gt_points(gt_row, orig_h, orig_w, validator):
            return validator.parse_gt_points_from_boxes(gt_row, orig_h, orig_w)

    validator.model = _PointModel()
    prediction = {
        "xy_norm": np.zeros((0, 2)),
        "scores": np.zeros(0),
        "classes": np.zeros(0, dtype=np.int64),
    }

    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        validator._update_metrics(
            [prediction],
            torch.tensor([[[0.0, 0.0, 1.0, 1.0, 2.0]]]),
            [(1, 1)],
            [1],
        )


def test_fomo_rejects_nonfinite_logits_before_metric_accumulation():
    validator = FOMOValidator.__new__(FOMOValidator)
    validator.nc = 1
    validator.last_logits = torch.full((1, 2, 2, 2), float("nan"))
    prediction = {
        "xy_norm": np.zeros((0, 2)),
        "scores": np.zeros(0),
        "classes": np.zeros(0, dtype=np.int64),
    }

    with pytest.raises(ValueError, match="non-finite"):
        validator._update_metrics(
            [prediction],
            torch.zeros((1, 0, 5)),
            [(1, 1)],
            [1],
        )


def test_matte_metrics_reject_shape_mismatch_and_nonfinite_values():
    with pytest.raises(ValueError, match="shapes must match"):
        matte_mae(np.zeros((1, 2)), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="non-finite"):
        s_measure(np.array([[float("nan")]]), np.zeros((1, 1)))


def test_ocr_rejects_payload_mismatch_and_nonfinite_polygons():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    with pytest.raises(ValueError, match="1 polygons but 0 transcripts"):
        match_image([square], [], [])
    bad = square.copy()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        match_image([bad], ["text"], [])


class _PoseResult:
    def __init__(self, classes, *, score=0.9, keypoints=None):
        self.boxes = SimpleNamespace(
            conf=torch.tensor([score]), cls=torch.as_tensor(classes)
        )
        self.keypoints = SimpleNamespace(
            data=(
                torch.zeros(1, 2, 3)
                if keypoints is None
                else torch.as_tensor(keypoints)
            )
        )

    def __len__(self):
        return 1


class _PoseModel:
    def __init__(self, result):
        self.result = result

    def __call__(self, *_args, **_kwargs):
        return self.result


def test_pose_rejects_invalid_prediction_classes():
    validator = PoseValidator.__new__(PoseValidator)
    validator.model = _PoseModel(_PoseResult([2]))
    validator.config = SimpleNamespace(
        conf_thres=0.25,
        iou_thres=0.7,
        imgsz=64,
        max_det=10,
        save_plots=False,
    )
    validator._category_ids = [1, 3]
    validator._category_id = 1
    validator._num_keypoints = 2
    validator._predictions = []

    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        validator._predict_image(Path("unused.jpg"), image_id=1)


@pytest.mark.parametrize(
    "result",
    [
        _PoseResult([0], score=float("nan")),
        _PoseResult(
            [0],
            keypoints=[[[float("inf"), 0.0, 1.0], [0.0, 0.0, 1.0]]],
        ),
    ],
)
def test_pose_rejects_nonfinite_prediction_payloads(result):
    validator = PoseValidator.__new__(PoseValidator)
    validator.model = _PoseModel(result)
    validator.config = SimpleNamespace(
        conf_thres=0.25,
        iou_thres=0.7,
        imgsz=64,
        max_det=10,
        save_plots=False,
    )
    validator._category_ids = [1]
    validator._category_id = 1
    validator._num_keypoints = 2
    validator._predictions = []

    with pytest.raises(ValueError, match="non-finite"):
        validator._predict_image(Path("unused.jpg"), image_id=1)


def test_pose_rejects_nonfinite_final_metrics(tmp_path):
    validator = PoseValidator.__new__(PoseValidator)
    validator.config = SimpleNamespace(
        save_plots=False,
        verbose=False,
        to_yaml=lambda path: None,
    )
    validator._setup_paths = lambda: setattr(validator, "save_dir", tmp_path)
    validator._load_coco_gt = lambda: None
    validator._predict_all = lambda: None
    validator._evaluate_oks_ap = lambda: {
        "metrics/keypoints_mAP50-95": float("nan")
    }

    with pytest.raises(ValueError, match="non-finite"):
        validator.run()
