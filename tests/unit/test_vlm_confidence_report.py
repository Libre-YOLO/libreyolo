"""Offline tests for strict persisted LibreVLM confidence reports."""

from __future__ import annotations

import json
import math

import pytest

from libreyolo.validation import vlm_confidence_report as report_module
from libreyolo.validation.vlm_confidence import VLMDetection, build_confidence_run
from libreyolo.validation.vlm_confidence_report import (
    VLMConfidenceReportError,
    compare_confidence_reports,
)

pytestmark = pytest.mark.unit


def _benchmark_config(*, hardware_name="test-gpu", class_names=("cat",)):
    return {
        "family": "qwen3vl",
        "size": "2b",
        "base_repo": "Qwen/Qwen3-VL-2B-Instruct",
        "base_revision": "a" * 40,
        "checkpoint": None,
        "processor": {"source": "base_snapshot", "revision": "a" * 40},
        "class_names": list(class_names),
        "generation_kwargs": {
            "do_sample": False,
            "max_new_tokens": 128,
            "num_beams": 1,
            "repetition_penalty": 1.1,
        },
        "confidence_method": "qwen_generation_policy_label_bbox_geomean_v1",
        "confidence_evaluation": {
            "iou_threshold": 0.5,
            "default_conf": 0.25,
            "fallback_score": 1.0,
            "calibration_bins": 10,
            "binning": "uniform_left_closed_v1",
            "population": "scored_postprocessed_predictions",
            "matching": "class_aware_max_cardinality_iou_v1",
        },
        "evaluation": {
            "max_det": 100,
            "faster_coco_eval": False,
            "imgsz": [100, 100],
            "label_to_category_id": None,
            "backend": "offline-stub",
        },
        "seed": 0,
        "backend": "transformers.Qwen3VLForConditionalGeneration",
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
        "hardware": {
            "type": "cuda",
            "name": hardware_name,
            "capability": [8, 9],
            "total_memory": 24_000_000_000,
        },
        "software": {
            "python": "3.10.0",
            "libreyolo": "test",
            "torch": "test",
            "transformers": "test",
            "pycocotools": "test",
        },
    }


def _payload(
    *,
    scores=(0.8, 0.1),
    prompt="detect cats",
    generation_digest="b" * 64,
    hardware_name="test-gpu",
    candidate_map=0.6,
    candidate_map50=0.8,
    constant_map=0.4,
    constant_map50=0.5,
    second_box=(50.0, 50.0, 70.0, 70.0),
    fallback_reason=None,
    artifact=None,
    total_s=2.0,
    empty=False,
    category_id=0,
    class_names=("cat",),
    target_box=(10.0, 10.0, 30.0, 30.0),
    evaluator_bbox=None,
    include_ground_truth=True,
):
    benchmark = _benchmark_config(hardware_name=hardware_name, class_names=class_names)
    if category_id != 0 or len(class_names) > 1:
        benchmark["evaluation"]["label_to_category_id"] = {"0": category_id}
    if evaluator_bbox is None:
        evaluator_bbox = (
            target_box[0],
            target_box[1],
            target_box[2] - target_box[0],
            target_box[3] - target_box[1],
        )
    dataset = {
        "split": "val",
        "class_names": list(class_names),
        "images": [
            {
                "image_id": "1",
                "file_name": "does-not-exist.jpg",
                "sha256": "c" * 64,
                "width": 100,
                "height": 100,
            }
        ],
        "evaluator_ground_truth": {
            "api": "offline.StubCOCO",
            "images": [{"id": 1, "width": 100, "height": 100}],
            "categories": [{"id": category_id, "name": class_names[0]}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": category_id,
                    "bbox": list(evaluator_bbox),
                    "area": float(evaluator_bbox[2] * evaluator_bbox[3]),
                    "iscrowd": 0,
                    "ignore": 0,
                }
            ],
        },
        "ground_truth": (
            [{"image_id": "1", "class_id": 0, "xyxy": list(target_box)}]
            if include_ground_truth
            else []
        ),
    }
    if empty:
        prediction_specs = []
        parsed_items = 0
        fallback_reason = None
    else:
        prediction_specs = [
            (target_box, scores[0]),
            (second_box, scores[1]),
        ]
        parsed_items = 2
    generations = [
        {
            "image_id": "1",
            "sha256": generation_digest,
            "parsed_items": parsed_items,
            "fallback_reason": fallback_reason,
        }
    ]
    predictions = [
        VLMDetection("1", 0, tuple(box), score) for box, score in prediction_specs
    ]
    ground_truth = (
        [VLMDetection("1", 0, tuple(target_box))] if include_ground_truth else []
    )
    evaluator = {
        "candidate_mAP50-95": candidate_map,
        "constant_mAP50-95": constant_map,
        "candidate_mAP50": candidate_map50,
        "constant_mAP50": constant_map50,
    }
    run = build_confidence_run(
        predictions,
        ground_truth,
        prompt=prompt,
        dataset_manifest=dataset,
        benchmark_config=benchmark,
        generation_manifest=generations,
        evaluator_metrics=evaluator,
        iou_threshold=0.5,
        default_conf=0.25,
        fallback_score=1.0,
    )
    scored_response = fallback_reason is None
    response = (
        1,
        int(scored_response),
        parsed_items,
        parsed_items if scored_response else 0,
    )
    metrics = report_module._semantic_metrics(run, response)
    metrics.update(
        {
            "speed/preprocess_ms": 1.0,
            "speed/inference_ms": 2.0,
            "speed/postprocess_ms": 1.0,
            "speed/total_ms": total_s * 1000.0,
            "speed/total_s": total_s,
            "speed/images_seen": 1.0,
        }
    )
    return {
        "schema": "libreyolo.vlm-confidence-report.v2",
        "prompt": prompt,
        "benchmark_config": benchmark,
        "dataset_manifest": dataset,
        "generation_manifest": generations,
        "hashes": {
            "manifest": run.manifest_hash,
            "configuration": run.configuration_hash,
            "generation": run.generation_hash,
            "prediction_structure": run.prediction_structure_hash,
        },
        "confidence": {
            "iou_threshold": run.iou_threshold,
            "default_conf": run.default_conf,
            "fallback_score": run.fallback_score,
        },
        "diagnostics": report_module._diagnostics_surface(run),
        "calibration": report_module._calibration_surface(run),
        "evaluator_metrics": evaluator,
        "fallback_reasons": ({fallback_reason: 1} if fallback_reason else {}),
        "predictions": [
            {
                "image_id": prediction.image_id,
                "class_id": prediction.class_id,
                "xyxy": list(prediction.xyxy),
                "candidate_score": prediction.score,
                "effective_score": (
                    run.fallback_score if prediction.score is None else prediction.score
                ),
                "matched": matched,
            }
            for prediction, matched in zip(predictions, run.matches)
        ],
        "metrics": metrics,
        "artifacts": {"reliability_plot": artifact},
    }


def _write(path, payload):
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def test_valid_separate_reports_ignore_speed_artifact_and_do_not_follow_paths(tmp_path):
    first = _write(tmp_path / "first.json", _payload(total_s=2.0, artifact=None))
    second = _write(
        tmp_path / "second.json",
        _payload(total_s=3.0, artifact="vlm_confidence_reliability.svg"),
    )

    comparison = compare_confidence_reports(first, second)

    assert comparison.reproducible
    assert comparison.core.reproducible
    assert comparison.first_report_sha256 != comparison.second_report_sha256
    assert comparison.differing_fields == ()
    assert not (tmp_path / "does-not-exist.jpg").exists()
    assert not (tmp_path / "vlm_confidence_reliability.svg").exists()


def test_noncontiguous_coco_category_mapping_round_trips(tmp_path):
    path = _write(tmp_path / "mapped.json", _payload(category_id=17))

    assert compare_confidence_reports(path, path).reproducible


def test_score_and_calibration_drift_obey_separate_tolerances(tmp_path):
    first = _write(tmp_path / "first.json", _payload(scores=(0.8, 0.1)))
    second = _write(tmp_path / "second.json", _payload(scores=(0.8000001, 0.1)))

    strict = compare_confidence_reports(first, second)
    tolerant = compare_confidence_reports(
        first, second, score_atol=1e-6, metric_atol=1e-6
    )

    assert not strict.reproducible
    assert tolerant.reproducible
    assert tolerant.core.max_abs_score_delta == pytest.approx(1e-7)


def test_calibration_bin_edge_change_is_never_hidden_by_tolerance(tmp_path):
    first = _write(tmp_path / "first.json", _payload(scores=(0.4999999, 0.1)))
    second = _write(tmp_path / "second.json", _payload(scores=(0.5000001, 0.1)))

    comparison = compare_confidence_reports(
        first, second, score_atol=1.0, metric_atol=1.0, map_atol=1.0
    )

    assert comparison.core.scores_within_tolerance
    assert not comparison.core.same_calibration_bin_assignments
    assert not comparison.reproducible


def test_map_tolerance_is_independent_from_other_metric_tolerance(tmp_path):
    first = _write(tmp_path / "first.json", _payload(candidate_map=0.6))
    second = _write(tmp_path / "second.json", _payload(candidate_map=0.61))

    metric_only = compare_confidence_reports(first, second, metric_atol=1.0)
    map_tolerant = compare_confidence_reports(first, second, map_atol=0.011)

    assert not metric_only.map_metrics_within_tolerance
    assert not metric_only.reproducible
    assert map_tolerant.map_metrics_within_tolerance
    assert map_tolerant.reproducible


@pytest.mark.parametrize(
    ("first_kwargs", "second_kwargs", "flag"),
    [
        ({}, {"prompt": "detect every cat"}, "same_manifest"),
        ({}, {"hardware_name": "other-gpu"}, "same_configuration"),
        ({}, {"generation_digest": "d" * 64}, "same_generation"),
        ({}, {"second_box": (51.0, 50.0, 70.0, 70.0)}, "same_prediction_structure"),
    ],
)
def test_valid_identity_drift_returns_false_not_an_error(
    tmp_path, first_kwargs, second_kwargs, flag
):
    first = _write(tmp_path / "first.json", _payload(**first_kwargs))
    second = _write(tmp_path / "second.json", _payload(**second_kwargs))

    comparison = compare_confidence_reports(first, second)

    assert not getattr(comparison.core, flag)
    assert not comparison.reproducible


def test_response_fallback_and_score_availability_drift_are_visible(tmp_path):
    first = _write(tmp_path / "first.json", _payload())
    second = _write(
        tmp_path / "second.json",
        _payload(scores=(None, None), fallback_reason="token_alignment"),
    )

    comparison = compare_confidence_reports(first, second, map_atol=1.0)

    assert not comparison.same_response_diagnostics
    assert not comparison.same_fallback_reasons
    assert not comparison.core.same_score_availability
    assert not comparison.reproducible


@pytest.mark.parametrize(
    "payload",
    [
        _payload(scores=(0.8, None), fallback_reason=None),
        _payload(scores=(None, None), fallback_reason=None),
        _payload(scores=(0.8, 0.1), fallback_reason="token_alignment"),
    ],
)
def test_report_rejects_scores_that_disagree_with_response_fallback(tmp_path, payload):
    path = _write(tmp_path / "invalid.json", payload)

    with pytest.raises(VLMConfidenceReportError, match="response-wide fallback"):
        compare_confidence_reports(path, path)


def test_report_rejects_more_predictions_than_parsed_items(tmp_path):
    payload = _payload()
    payload["generation_manifest"][0]["parsed_items"] = 1
    path = _write(tmp_path / "invalid.json", payload)

    with pytest.raises(VLMConfidenceReportError, match="retained predictions"):
        compare_confidence_reports(path, path)


def test_empty_prediction_optional_metrics_round_trip(tmp_path):
    first = _write(tmp_path / "first.json", _payload(empty=True))
    second = _write(tmp_path / "second.json", _payload(empty=True, total_s=3.0))

    comparison = compare_confidence_reports(first, second)

    assert comparison.reproducible
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["metrics"]["metrics/vlm_confidence/auroc"] is None
    assert payload["metrics"]["metrics/vlm_confidence/scored_prediction_brier"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["hashes"].__setitem__("manifest", "0" * 64),
        lambda value: value["predictions"][0].__setitem__("effective_score", 0.0),
        lambda value: value["predictions"][0].__setitem__(
            "matched", not value["predictions"][0]["matched"]
        ),
        lambda value: value["diagnostics"].__setitem__("total_predictions", 9),
        lambda value: value["calibration"].__setitem__("brier_score", 0.9),
        lambda value: value["calibration"]["bins"][8].__setitem__("count", 9),
        lambda value: value["evaluator_metrics"].__setitem__("candidate_mAP50-95", 0.9),
        lambda value: value["metrics"].__setitem__(
            "metrics/vlm_confidence/delta_mAP50-95", 0.9
        ),
        lambda value: value["fallback_reasons"].__setitem__("fabricated", 1),
    ],
)
def test_tampered_derived_surfaces_are_rejected(tmp_path, mutate):
    payload = _payload()
    mutate(payload)
    path = _write(tmp_path / "tampered.json", payload)

    with pytest.raises(VLMConfidenceReportError):
        compare_confidence_reports(path, _write(tmp_path / "valid.json", _payload()))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["dataset_manifest"]["evaluator_ground_truth"][
            "annotations"
        ][0].__setitem__("bbox", [60.0, 60.0, 20.0, 20.0]),
        lambda value: value["dataset_manifest"]["evaluator_ground_truth"]["images"][
            0
        ].__setitem__("width", 99),
        lambda value: value["dataset_manifest"]["evaluator_ground_truth"]["categories"][
            0
        ].__setitem__("name", "dog"),
    ],
)
def test_evaluator_and_ordering_ground_truth_must_agree(tmp_path, mutate):
    payload = _payload()
    mutate(payload)
    path = _write(tmp_path / "invalid.json", payload)

    with pytest.raises(VLMConfidenceReportError):
        compare_confidence_reports(path, path)


def test_partial_native_coco_category_map_round_trips(tmp_path):
    path = _write(
        tmp_path / "partial-categories.json",
        _payload(class_names=("cat", "dog"), category_id=5),
    )

    assert compare_confidence_reports(path, path).reproducible


def test_native_coco_bbox_clipping_matches_loader_semantics(tmp_path):
    path = _write(
        tmp_path / "clipped-bbox.json",
        _payload(
            target_box=(0.0, 10.0, 20.0, 30.0),
            evaluator_bbox=(-5.0, 10.0, 20.0, 20.0),
        ),
    )

    assert compare_confidence_reports(path, path).reproducible


def test_native_coco_bbox_with_empty_clipped_extent_is_omitted(tmp_path):
    path = _write(
        tmp_path / "empty-clipped-bbox.json",
        _payload(
            evaluator_bbox=(120.0, 10.0, 20.0, 20.0),
            include_ground_truth=False,
        ),
    )

    assert compare_confidence_reports(path, path).reproducible


def test_skipped_native_coco_bbox_cannot_appear_in_ordering_ground_truth(tmp_path):
    path = _write(
        tmp_path / "mismatched-empty-clipped-bbox.json",
        _payload(evaluator_bbox=(120.0, 10.0, 20.0, 20.0)),
    )

    with pytest.raises(VLMConfidenceReportError, match="same image/class groups"):
        compare_confidence_reports(path, path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["predictions"][0].__setitem__(
            "xyxy", [10.0, 10.0, 130.0, 30.0]
        ),
        lambda value: value["dataset_manifest"]["ground_truth"][0].__setitem__(
            "xyxy", [10.0, 10.0, 130.0, 30.0]
        ),
    ],
)
def test_boxes_must_lie_within_the_recorded_image(tmp_path, mutate):
    payload = _payload()
    mutate(payload)
    path = _write(tmp_path / "invalid.json", payload)

    with pytest.raises(VLMConfidenceReportError, match="within the image"):
        compare_confidence_reports(path, path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("unexpected", {}),
        lambda value: value.__setitem__("schema", "libreyolo.vlm-confidence-report.v3"),
        lambda value: value["metrics"].__setitem__("metrics/hidden", 0.0),
        lambda value: value["metrics"].pop("metrics/vlm_confidence/ranking_ap"),
        lambda value: value["artifacts"].__setitem__("reliability_plot", "../plot.svg"),
        lambda value: value["metrics"].__setitem__("speed/total_s", -1.0),
        lambda value: value["metrics"].__setitem__("speed/total_ms", 1.0),
        lambda value: value["diagnostics"].__setitem__("total_predictions", True),
        lambda value: value["calibration"]["bins"][8].__setitem__("count", True),
        lambda value: value["metrics"].__setitem__("speed/total_s", 10**400),
        lambda value: value["benchmark_config"]["generation_kwargs"].__setitem__(
            "repetition_penalty", 10**400
        ),
        lambda value: value["benchmark_config"]["generation_kwargs"].__setitem__(
            "max_new_tokens", 10**400
        ),
        lambda value: value["predictions"][0].__setitem__("xyxy", [0, 0, 10**400, 1]),
        lambda value: value["dataset_manifest"]["ground_truth"][0].__setitem__(
            "xyxy", [0, 0, 10**400, 1]
        ),
        lambda value: value["generation_manifest"][0].__setitem__(
            "parsed_items", 10**400
        ),
        lambda value: value["dataset_manifest"]["images"][0].__setitem__(
            "sha256", "C" * 64
        ),
        lambda value: (
            value["dataset_manifest"]["images"][0].__setitem__("width", 10**400),
            value["dataset_manifest"]["evaluator_ground_truth"]["images"][
                0
            ].__setitem__("width", 10**400),
        ),
    ],
)
def test_strict_schema_types_metrics_speed_and_artifacts(tmp_path, mutate):
    payload = _payload()
    mutate(payload)
    path = _write(tmp_path / "invalid.json", payload)

    with pytest.raises(VLMConfidenceReportError):
        compare_confidence_reports(path, _write(tmp_path / "valid.json", _payload()))


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema": "x", "schema": "y"}',
        b'{"value": NaN}',
        b"not-json",
        b"\xff\xfe",
    ],
)
def test_untrusted_json_is_rejected_before_schema_use(tmp_path, raw):
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(raw)
    valid = _write(tmp_path / "valid.json", _payload())

    with pytest.raises(VLMConfidenceReportError):
        compare_confidence_reports(invalid, valid)


def test_report_size_is_bounded_without_allocating_a_large_fixture(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(report_module, "_MAX_REPORT_BYTES", 8)
    invalid = tmp_path / "large.json"
    invalid.write_bytes(b"{}" * 5)

    with pytest.raises(VLMConfidenceReportError, match="exceeds"):
        compare_confidence_reports(invalid, invalid)


def test_report_nesting_is_bounded(tmp_path):
    payload = {"leaf": 1}
    for _ in range(70):
        payload = {"nested": payload}
    path = _write(tmp_path / "deep.json", payload)

    with pytest.raises(VLMConfidenceReportError, match="nesting"):
        compare_confidence_reports(path, path)


@pytest.mark.parametrize("value", [True, -1, math.inf, "0.1", 10**400])
def test_tolerances_are_strict(tmp_path, value):
    path = _write(tmp_path / "valid.json", _payload())

    with pytest.raises((TypeError, ValueError)):
        compare_confidence_reports(path, path, map_atol=value)


def test_inputs_are_file_paths_only():
    with pytest.raises(TypeError, match="filesystem path"):
        compare_confidence_reports({}, {})
