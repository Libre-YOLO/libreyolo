"""
Unit tests for the opt-in faster-coco-eval backend in COCOEvaluator.

Covers:
- Metric parity between pycocotools and faster-coco-eval on synthetic data.
- Graceful fallback to pycocotools when faster-coco-eval is missing.
- LIBREYOLO_FASTER_COCO_EVAL env var overriding the config flag both ways.
- ValidationConfig / TrainConfig exposing the flag (off by default).
"""

import sys

import pytest

pytestmark = pytest.mark.unit

from libreyolo.validation.coco_evaluator import (
    COCOEvaluator,
    FASTER_COCO_EVAL_ENV_VAR,
    resolve_faster_coco_eval,
)


def _make_gt_dataset():
    """Small synthetic COCO GT: 3 images, 2 categories, 6 annotations."""
    images = [
        {"id": i, "file_name": f"img{i}.jpg", "width": 100, "height": 100}
        for i in range(3)
    ]
    categories = [
        {"id": 0, "name": "cat", "supercategory": "object"},
        {"id": 1, "name": "dog", "supercategory": "object"},
    ]
    annotations = []
    ann_id = 1
    boxes = [
        (0, 0, [10, 10, 30, 30]),
        (0, 1, [50, 50, 20, 40]),
        (1, 0, [5, 5, 40, 20]),
        (1, 1, [60, 10, 25, 25]),
        (2, 0, [20, 20, 30, 50]),
        (2, 1, [70, 70, 15, 15]),
    ]
    for img_id, cat_id, bbox in boxes:
        annotations.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
        )
        ann_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}


def _make_coco_gt():
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = _make_gt_dataset()
    coco.createIndex()
    return coco


def _predictions():
    """Predictions per image: one near-perfect match, one shifted, one miss."""
    return [
        # img 0: good matches
        (0, {"boxes": [[10, 10, 41, 41], [51, 49, 69, 91]], "scores": [0.9, 0.8], "classes": [0, 1]}),
        # img 1: one shifted box, one wrong class
        (1, {"boxes": [[10, 8, 48, 27], [58, 12, 84, 33]], "scores": [0.7, 0.6], "classes": [0, 0]}),
        # img 2: one match, one false positive
        (2, {"boxes": [[21, 19, 49, 71], [1, 1, 9, 9]], "scores": [0.85, 0.3], "classes": [0, 1]}),
    ]


def _make_evaluator(faster: bool):
    evaluator = COCOEvaluator(_make_coco_gt(), iou_type="bbox", faster_coco_eval=faster)
    for img_id, preds in _predictions():
        evaluator.update(preds, img_id)
    return evaluator


def _run_evaluator(faster: bool):
    return _make_evaluator(faster).compute()


class TestFasterCocoEvalParity:
    def test_stock_backend_runs(self):
        metrics = _run_evaluator(faster=False)
        assert 0.0 < metrics["mAP"] <= 1.0
        assert metrics["mAP50"] >= metrics["mAP"]

    def test_metrics_match_pycocotools(self):
        pytest.importorskip("faster_coco_eval")
        stock = _run_evaluator(faster=False)
        fast = _run_evaluator(faster=True)
        assert set(stock) == set(fast)
        for key in stock:
            assert fast[key] == pytest.approx(stock[key], abs=1e-9), key

    def test_yolo_coco_api_gt_without_dataset_dict(self, tmp_path):
        """GT objects without a .dataset dict (YOLOCocoAPI-style) work too."""
        pytest.importorskip("faster_coco_eval")

        class MinimalGT:
            def __init__(self, dataset):
                self.imgs = {im["id"]: im for im in dataset["images"]}
                self.anns = {a["id"]: a for a in dataset["annotations"]}
                self.cats = {c["id"]: c for c in dataset["categories"]}

        evaluator = COCOEvaluator(
            MinimalGT(_make_gt_dataset()), iou_type="bbox", faster_coco_eval=True
        )
        for img_id, preds in _predictions():
            evaluator.update(preds, img_id)
        fast = evaluator.compute()
        stock = _run_evaluator(faster=False)
        for key in stock:
            assert fast[key] == pytest.approx(stock[key], abs=1e-9), key


class TestBackendResolution:
    def test_disabled_request_stays_disabled(self, monkeypatch):
        monkeypatch.delenv(FASTER_COCO_EVAL_ENV_VAR, raising=False)
        assert resolve_faster_coco_eval(False) is False

    def test_env_var_forces_on(self, monkeypatch):
        pytest.importorskip("faster_coco_eval")
        monkeypatch.setenv(FASTER_COCO_EVAL_ENV_VAR, "1")
        assert resolve_faster_coco_eval(False) is True

    def test_env_var_forces_off(self, monkeypatch):
        monkeypatch.setenv(FASTER_COCO_EVAL_ENV_VAR, "0")
        assert resolve_faster_coco_eval(True) is False

    def test_fallback_when_package_missing(self, monkeypatch, caplog):
        monkeypatch.delenv(FASTER_COCO_EVAL_ENV_VAR, raising=False)
        # Simulate faster_coco_eval being uninstalled: a None entry in
        # sys.modules makes `import faster_coco_eval` raise ImportError.
        monkeypatch.setitem(sys.modules, "faster_coco_eval", None)
        import libreyolo.validation.coco_evaluator as mod

        monkeypatch.setattr(mod, "_warned_faster_unavailable", False)
        with caplog.at_level("WARNING"):
            assert resolve_faster_coco_eval(True) is False
        assert any("falling back" in r.message for r in caplog.records)

        # And the evaluator still produces stock metrics.
        metrics = _run_evaluator(faster=True)
        assert 0.0 < metrics["mAP"] <= 1.0


class TestConfigFlag:
    def test_validation_config_default_on(self):
        from libreyolo.validation import ValidationConfig

        cfg = ValidationConfig(data="dummy.yaml")
        assert cfg.faster_coco_eval is True
        assert cfg.update(faster_coco_eval=False).faster_coco_eval is False

    def test_train_config_default_on(self):
        from libreyolo.training.config import TrainConfig

        assert TrainConfig().faster_coco_eval is True


class TestBackendProvenance:
    def test_stock_backend_recorded(self):
        evaluator = _make_evaluator(faster=False)
        evaluator.compute()
        assert evaluator.last_backend is not None
        assert evaluator.last_backend.startswith("pycocotools")

    def test_faster_backend_recorded(self):
        pytest.importorskip("faster_coco_eval")
        evaluator = _make_evaluator(faster=True)
        evaluator.compute()
        assert evaluator.last_backend is not None
        assert evaluator.last_backend.startswith("faster-coco-eval")
