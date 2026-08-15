"""Offline tests for the internal VLM confidence quality gate."""

from __future__ import annotations

import math

import pytest

from libreyolo.validation.vlm_confidence import (
    VLMDetection,
    benchmark_manifest_hash,
    binary_ranking_ap,
    build_confidence_run,
    compare_repeats,
    confidence_diagnostics,
    match_detections,
    tie_aware_auroc,
)

pytestmark = pytest.mark.unit


def det(
    box=(0.0, 0.0, 10.0, 10.0),
    *,
    image="image-a",
    cls=0,
    score=None,
):
    return VLMDetection(image, cls, box, score)


def benchmark_config(**overrides):
    config = {
        "family": "qwen3vl",
        "size": "2b",
        "base_repo": "Qwen/Qwen3-VL-2B-Instruct",
        "base_revision": "a" * 40,
        "checkpoint": None,
        "processor": {"source": "base_snapshot", "revision": "a" * 40},
        "class_names": ["boat"],
        "generation_kwargs": {
            "do_sample": False,
            "max_new_tokens": 128,
            "num_beams": 1,
            "repetition_penalty": 1.1,
        },
        "confidence_method": "qwen_generation_policy_label_bbox_geomean_v1",
        "evaluation": {
            "max_det": 100,
            "faster_coco_eval": False,
            "imgsz": 1024,
            "backend": "pycocotools",
        },
        "seed": 0,
        "backend": "transformers",
        "device": "cpu",
        "dtype": "float32",
        "hardware": {"kind": "cpu", "name": "test-cpu"},
        "software": {
            "python": "test",
            "libreyolo": "test",
            "torch": "test",
            "transformers": "test",
            "pycocotools": "test",
        },
    }
    config.update(overrides)
    return config


class TestMatching:
    def test_same_image_and_class_with_inclusive_half_iou(self):
        predictions = [
            det((0, 0, 2, 1), score=0.9),
            det((0, 0, 1, 1), cls=1, score=0.8),
            det((0, 0, 1, 1), image="image-b", score=0.7),
        ]
        targets = [det((0, 0, 1, 1))]

        assert match_detections(predictions, targets) == (True, False, False)

    def test_highest_iou_wins_independently_of_score(self):
        predictions = [
            det((0, 0, 10, 10), score=0.2),
            det((0, 0, 8, 10), score=0.9),
        ]

        assert match_detections(predictions, [det()]) == (True, False)

    def test_highest_iou_is_stable_when_input_order_changes(self):
        exact = det((0, 0, 10, 10), score=0.5)
        partial = det((0, 0, 8, 10), score=0.5)

        assert match_detections([partial, exact], [det()]) == (False, True)
        assert match_detections([exact, partial], [det()]) == (True, False)

    def test_duplicate_boxes_use_deterministic_source_tie_break(self):
        predictions = [det(score=0.1), det(score=0.9)]

        assert match_detections(predictions, [det()]) == (True, False)

    def test_matching_maximizes_valid_pair_count_before_total_iou(self):
        targets = [det((0, 0, 10, 10)), det((3.5, 0, 13.5, 10))]
        predictions = [
            det((0.5, 0, 10.5, 10), score=0.2),
            det((-1, 0, 9, 10), score=0.9),
        ]

        assert match_detections(predictions, targets) == (True, True)

    def test_invalid_detection_and_threshold_fail_loudly(self):
        with pytest.raises(ValueError, match="positive width"):
            det((0, 0, 0, 1))
        with pytest.raises(ValueError, match="greater than zero"):
            match_detections([], [], iou_threshold=0.0)


class TestRankingMetrics:
    def test_auroc_awards_half_credit_to_score_ties(self):
        score = tie_aware_auroc([0.9, 0.9, 0.1], [True, False, False])

        assert score == pytest.approx(0.75)

    def test_auroc_is_undefined_without_both_binary_classes(self):
        assert tie_aware_auroc([0.9], [True]) is None
        assert tie_aware_auroc([], []) is None

    def test_ranking_ap_groups_ties_and_is_order_independent(self):
        first = binary_ranking_ap([0.9, 0.5, 0.5], [True, True, False])
        second = binary_ranking_ap([0.5, 0.9, 0.5], [False, True, True])

        assert first == pytest.approx(5.0 / 6.0)
        assert second == first

    def test_ranking_ap_is_undefined_without_positive_labels(self):
        assert binary_ranking_ap([0.8, 0.2], [False, False]) is None

    def test_metric_inputs_are_strict(self):
        with pytest.raises(ValueError, match="same length"):
            tie_aware_auroc([0.5], [])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            binary_ranking_ap([math.nan], [True])


class TestCoverageAndRetention:
    def test_missing_scores_count_against_coverage_and_use_fallback(self):
        result = confidence_diagnostics(
            [0.9, 0.2, None],
            [True, False, True],
            default_conf=0.25,
            fallback_score=1.0,
        )

        assert result.total_predictions == 3
        assert result.scored_predictions == 2
        assert result.fallback_predictions == 1
        assert result.score_coverage == pytest.approx(2.0 / 3.0)
        assert result.retained_predictions == 2
        assert result.default_conf_retention == pytest.approx(2.0 / 3.0)
        assert result.correct_retention == 1.0
        assert result.incorrect_retention == 0.0

    def test_empty_diagnostics_have_explicit_zero_or_undefined_rates(self):
        result = confidence_diagnostics([], [])

        assert result.score_coverage == 0.0
        assert result.default_conf_retention == 0.0
        assert result.correct_retention is None
        assert result.incorrect_retention is None


class TestManifestAndRepeats:
    def test_manifest_hash_is_canonical_but_preserves_semantics(self):
        first = benchmark_manifest_hash(
            "detect boats",
            {"samples": ({"label": "a.txt", "image": "a.jpg"},), "classes": ["boat"]},
            benchmark_config(),
        )
        reordered_keys = benchmark_manifest_hash(
            "detect boats",
            {"classes": ["boat"], "samples": [{"image": "a.jpg", "label": "a.txt"}]},
            benchmark_config(),
        )

        assert first == reordered_keys
        assert first != benchmark_manifest_hash(
            "detect all boats",
            {"classes": ["boat"], "samples": [{"image": "a.jpg", "label": "a.txt"}]},
            benchmark_config(),
        )
        assert first != benchmark_manifest_hash(
            "detect boats",
            {"classes": ["boat"], "samples": [{"image": "a.jpg", "label": "a.txt"}]},
            benchmark_config(base_revision="b" * 40),
        )

    def test_manifest_rejects_ambiguous_or_nonfinite_values(self):
        with pytest.raises(TypeError, match="keys must be strings"):
            benchmark_manifest_hash("prompt", {1: "image.jpg"}, benchmark_config())
        with pytest.raises(ValueError, match="non-finite"):
            benchmark_manifest_hash("prompt", {"weight": math.inf}, benchmark_config())
        with pytest.raises(ValueError, match="missing required fields"):
            benchmark_manifest_hash("prompt", {}, {})

    @pytest.mark.parametrize(
        ("override", "match"),
        [
            ({"base_revision": "main"}, "40-character"),
            ({"processor": "local/path"}, "processor must be a mapping"),
            ({"generation_kwargs": {}}, "generation_kwargs is missing"),
            ({"hardware": {}}, "hardware must be a non-empty mapping"),
            ({"software": {"torch": "test"}}, "software is missing"),
        ],
    )
    def test_manifest_requires_immutable_complete_run_identity(self, override, match):
        with pytest.raises((TypeError, ValueError), match=match):
            benchmark_manifest_hash("prompt", {}, benchmark_config(**override))

    @staticmethod
    def _run(
        score=0.8,
        *,
        prompt="prompt",
        box=(0, 0, 10, 10),
        generation_hash="a" * 64,
    ):
        return build_confidence_run(
            [det(box, score=score), det((20, 20, 30, 30), score=0.1)],
            [det()],
            prompt=prompt,
            dataset_manifest={"images": ["image-a.jpg"], "sha256": "abc"},
            benchmark_config=benchmark_config(),
            generation_manifest=[{"image_id": "image-a", "sha256": generation_hash}],
        )

    def test_build_run_combines_matching_metrics_and_retention(self):
        run = self._run()

        assert run.matches == (True, False)
        assert run.auroc == 1.0
        assert run.ranking_ap == 1.0
        assert run.diagnostics.score_coverage == 1.0
        assert run.diagnostics.default_conf_retention == 0.5
        assert len(run.manifest_hash) == 64
        assert len(run.prediction_structure_hash) == 64

    def test_repeat_comparison_accepts_bounded_score_drift(self):
        comparison = compare_repeats(
            self._run(0.8), self._run(0.8000001), score_atol=1e-6
        )

        assert comparison.reproducible
        assert comparison.max_abs_score_delta == pytest.approx(1e-7)
        assert comparison.auroc_delta == 0.0
        assert comparison.ranking_ap_delta == 0.0

    def test_repeat_comparison_rejects_rank_reversal_inside_score_tolerance(self):
        def run(scores):
            return build_confidence_run(
                [
                    det(score=scores[0]),
                    det((20, 20, 30, 30), score=scores[1]),
                ],
                [det()],
                prompt="prompt",
                dataset_manifest={"sha256": "abc"},
                benchmark_config=benchmark_config(),
                generation_manifest=[{"image_id": "image-a", "sha256": "a" * 64}],
            )

        comparison = compare_repeats(
            run((0.5000001, 0.5)),
            run((0.5, 0.5000001)),
            score_atol=1e-6,
        )

        assert comparison.scores_within_tolerance
        assert not comparison.metrics_within_tolerance
        assert not comparison.reproducible

    def test_repeat_comparison_rejects_manifest_or_structure_drift(self):
        manifest_drift = compare_repeats(self._run(), self._run(prompt="other"))
        structure_drift = compare_repeats(self._run(), self._run(box=(0, 0, 9, 10)))

        assert not manifest_drift.same_manifest
        assert not manifest_drift.reproducible
        assert not structure_drift.same_prediction_structure
        assert not structure_drift.reproducible

    def test_repeat_comparison_rejects_generated_token_or_text_drift(self):
        comparison = compare_repeats(
            self._run(generation_hash="a" * 64),
            self._run(generation_hash="b" * 64),
        )

        assert not comparison.same_generation
        assert not comparison.reproducible

    def test_repeat_comparison_rejects_model_or_generation_drift(self):
        first = self._run()
        changed = build_confidence_run(
            [det(score=0.8), det((20, 20, 30, 30), score=0.1)],
            [det()],
            prompt="prompt",
            dataset_manifest={"images": ["image-a.jpg"], "sha256": "abc"},
            benchmark_config=benchmark_config(
                generation_kwargs={
                    "do_sample": False,
                    "max_new_tokens": 256,
                    "num_beams": 1,
                    "repetition_penalty": 1.1,
                }
            ),
            generation_manifest=[{"image_id": "image-a", "sha256": "a" * 64}],
        )

        comparison = compare_repeats(first, changed)

        assert not comparison.same_manifest
        assert not comparison.same_configuration
        assert not comparison.reproducible
