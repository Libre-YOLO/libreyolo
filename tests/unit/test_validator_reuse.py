"""Reusing one validator instance across runs (per-epoch validation).

The trainer calls a validator once per epoch. The dataset, dataloader
(worker spawn, pinned buffers), model warmup and parsed ground truth are
per-instance costs that must survive between runs; metrics and speed
counters are per-run state that must reset. These tests pin that split.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from libreyolo.validation.base import BaseValidator
from libreyolo.validation.config import ValidationConfig
from libreyolo.validation.detection_validator import DetectionValidator

pytestmark = pytest.mark.unit


class _CountingValidator(BaseValidator):
    """Stub that counts which template hooks each run() actually pays for."""

    def __init__(self, model, config):
        super().__init__(model, config)
        self.dataloader_builds = 0
        self.warmups = 0
        self.metric_inits = 0
        self.validations = 0

    def _setup_dataloader(self):
        self.dataloader_builds += 1
        return object()  # anything non-None: our _run_validation ignores it

    def _warmup_model(self, n_warmup: int = 3):
        self.warmups += 1

    def _init_metrics(self):
        self.metric_inits += 1

    def _run_validation(self):
        self.validations += 1
        self.seen += 4
        self.speed["total"] += 1.0

    def _preprocess_batch(self, batch):
        raise AssertionError("not used by the stub")

    def _postprocess_predictions(self, preds, batch):
        raise AssertionError("not used by the stub")

    def _update_metrics(self, preds, targets, img_info, img_ids=None):
        raise AssertionError("not used by the stub")

    def _compute_metrics(self):
        return {}


def _counting_validator(tmp_path):
    config = ValidationConfig(
        data="x.yaml",
        device="cpu",
        save_dir=str(tmp_path / "val"),
        verbose=False,
    )
    return _CountingValidator(model=object(), config=config)


def test_second_run_reuses_dataloader_and_warmup(tmp_path):
    v = _counting_validator(tmp_path)
    v.run()
    v.run()

    assert v.validations == 2
    assert v.dataloader_builds == 1, "dataloader must be a per-instance cost"
    assert v.warmups == 1, "warmup must be a per-instance cost"
    assert v.metric_inits == 2, "metrics accumulate and must reset per run"


def test_second_run_resets_per_run_counters(tmp_path):
    v = _counting_validator(tmp_path)
    v.run()
    seen_after_first, total_after_first = v.seen, v.speed["total"]
    v.run()

    assert v.seen == seen_after_first, "seen must restart from zero each run"
    assert v.speed["total"] == total_after_first, "speed must restart each run"


def test_fresh_instance_still_builds_everything(tmp_path):
    first = _counting_validator(tmp_path)
    first.run()
    second = _counting_validator(tmp_path)
    second.run()

    assert second.dataloader_builds == 1
    assert second.warmups == 1


def test_detection_gt_parse_is_cached_across_metric_inits(tmp_path):
    """The COCO GT is immutable per instance; only the evaluator is per-run."""
    annotation_file = tmp_path / "annotations.json"
    annotation_file.write_text("{}")  # never actually parsed: COCO is mocked

    v = object.__new__(DetectionValidator)
    v.config = ValidationConfig(
        data="x.yaml", device="cpu", verbose=False, save_plots=False
    )
    v._coco_annotation_file = annotation_file
    v._coco_label_to_category_id = None
    v._gt_coco_api = None
    v.nc = 3

    with (
        patch("pycocotools.coco.COCO", return_value=MagicMock(imgs={})) as coco,
        patch("libreyolo.validation.COCOEvaluator") as evaluator,
    ):
        v._init_metrics()
        v._init_metrics()

    assert coco.call_count == 1, "GT must be parsed once per instance"
    assert evaluator.call_count == 2, "the evaluator accumulates: one per run"
