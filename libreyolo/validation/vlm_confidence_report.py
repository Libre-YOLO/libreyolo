"""Strict persisted-report comparison for the internal LibreVLM confidence gate.

The comparator accepts only report file paths and never follows paths embedded in
the JSON.  It reconstructs the dependency-free confidence run and verifies every
derived report surface before comparing two runs.  Stored COCO mAP values are
cross-checked against their duplicate metric fields and compared, but are not
independently recomputed.  Consequently this module detects malformed, stale, and
inconsistent reports; it does not authenticate a deliberately forged report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from libreyolo.utils.coco_geometry import clipped_coco_bbox_xyxy

from .vlm_confidence import (
    ConfidenceRun,
    RepeatComparison,
    VLMDetection,
    build_confidence_run,
    compare_repeats,
)

_REPORT_SCHEMA = "libreyolo.vlm-confidence-report.v2"
_MAX_REPORT_BYTES = 256 * 1024 * 1024
_MAX_SAFE_INTEGER = (1 << 53) - 1
_BOX_ABS_TOLERANCE = 1e-4
_BOX_REL_TOLERANCE = 1e-6
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")

_ROOT_FIELDS = {
    "schema",
    "prompt",
    "benchmark_config",
    "dataset_manifest",
    "generation_manifest",
    "hashes",
    "confidence",
    "diagnostics",
    "calibration",
    "evaluator_metrics",
    "fallback_reasons",
    "predictions",
    "metrics",
    "artifacts",
}
_HASH_FIELDS = {
    "manifest",
    "configuration",
    "generation",
    "prediction_structure",
}
_CONFIDENCE_FIELDS = {"iou_threshold", "default_conf", "fallback_score"}
_DIAGNOSTIC_FIELDS = {
    "default_conf",
    "fallback_score",
    "total_predictions",
    "scored_predictions",
    "fallback_predictions",
    "score_coverage",
    "retained_predictions",
    "default_conf_retention",
    "correct_predictions",
    "retained_correct_predictions",
    "correct_retention",
    "incorrect_predictions",
    "retained_incorrect_predictions",
    "incorrect_retention",
}
_CALIBRATION_FIELDS = {
    "method",
    "population",
    "bin_count",
    "total_predictions",
    "scored_predictions",
    "unscored_predictions",
    "score_coverage",
    "brier_score",
    "expected_calibration_error",
    "maximum_calibration_error",
    "bins",
}
_BIN_FIELDS = {
    "index",
    "lower",
    "upper",
    "count",
    "correct",
    "mean_confidence",
    "empirical_accuracy",
    "absolute_gap",
}
_PREDICTION_FIELDS = {
    "image_id",
    "class_id",
    "xyxy",
    "candidate_score",
    "effective_score",
    "matched",
}
_GENERATION_FIELDS = {
    "image_id",
    "sha256",
    "parsed_items",
    "fallback_reason",
}
_EVALUATOR_METRIC_FIELDS = {
    "candidate_mAP50-95",
    "constant_mAP50-95",
    "candidate_mAP50",
    "constant_mAP50",
}
_MAP_METRICS = {
    "metrics/vlm_confidence/candidate_mAP50-95",
    "metrics/vlm_confidence/constant_mAP50-95",
    "metrics/vlm_confidence/delta_mAP50-95",
    "metrics/vlm_confidence/candidate_mAP50",
    "metrics/vlm_confidence/constant_mAP50",
    "metrics/vlm_confidence/delta_mAP50",
}
_QUALITY_METRICS = {
    "metrics/vlm_confidence/auroc",
    "metrics/vlm_confidence/ranking_ap",
    "metrics/vlm_confidence/scored_prediction_brier",
    "metrics/vlm_confidence/scored_prediction_ece",
    "metrics/vlm_confidence/scored_prediction_mce",
}
_EXACT_METRICS = {
    "metrics/vlm_confidence/default_conf_tp_retention",
    "metrics/vlm_confidence/default_conf_fp_retention",
    "metrics/vlm_confidence/default_conf_prediction_retention",
    "metrics/vlm_confidence/response_score_coverage",
    "metrics/vlm_confidence/detection_score_coverage",
    "metrics/vlm_confidence/prediction_score_coverage",
    "metrics/vlm_confidence/responses",
    "metrics/vlm_confidence/scored_responses",
    "metrics/vlm_confidence/parsed_detections",
    "metrics/vlm_confidence/scored_parsed_detections",
    "metrics/vlm_confidence/predictions",
    "metrics/vlm_confidence/correct_predictions",
    "metrics/vlm_confidence/incorrect_predictions",
    "metrics/vlm_confidence/retained_correct_predictions",
    "metrics/vlm_confidence/retained_incorrect_predictions",
}
_SEMANTIC_METRICS = _MAP_METRICS | _QUALITY_METRICS | _EXACT_METRICS
_SPEED_METRICS = {
    "speed/preprocess_ms",
    "speed/inference_ms",
    "speed/postprocess_ms",
    "speed/total_ms",
    "speed/total_s",
    "speed/images_seen",
}


class VLMConfidenceReportError(ValueError):
    """A persisted confidence report is malformed or internally inconsistent."""


@dataclass(frozen=True)
class PersistedRepeatComparison:
    """Comparison of two validated persisted VLM confidence reports."""

    first_report_sha256: str
    second_report_sha256: str
    core: RepeatComparison
    same_response_diagnostics: bool
    same_fallback_reasons: bool
    same_semantic_metric_keys: bool
    map_metrics_within_tolerance: bool
    non_map_metrics_within_tolerance: bool
    semantic_metrics_within_tolerance: bool
    max_abs_semantic_metric_delta: Optional[float]
    differing_fields: tuple[str, ...]
    reproducible: bool


@dataclass(frozen=True)
class _ValidatedReport:
    file_sha256: str
    benchmark_config: dict[str, Any]
    metrics: tuple[tuple[str, Optional[float]], ...]
    run: ConfidenceRun
    response_diagnostics: tuple[int, int, int, int]
    fallback_reasons: tuple[tuple[str, int], ...]
    semantic_metrics: tuple[tuple[str, Optional[float]], ...]


def _error(label: str, path: str, message: str) -> VLMConfidenceReportError:
    return VLMConfidenceReportError(f"{label}:{path}: {message}")


def _mapping(value: Any, label: str, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(label, path, "must be a JSON object")
    return value


def _exact_fields(
    value: Any, expected: set[str], label: str, path: str
) -> Mapping[str, Any]:
    result = _mapping(value, label, path)
    actual = set(result)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unsupported " + ", ".join(extra))
        raise _error(label, path, "; ".join(details))
    return result


def _sequence(value: Any, label: str, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _error(label, path, "must be a JSON array")
    return value


def _string(value: Any, label: str, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        suffix = " non-empty" if nonempty else ""
        raise _error(label, path, f"must be a{suffix} string")
    return value


def _integer(
    value: Any,
    label: str,
    path: str,
    *,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(label, path, "must be an integer")
    if value < minimum:
        raise _error(label, path, f"must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise _error(label, path, f"must be at most {maximum}")
    return value


def _real(
    value: Any,
    label: str,
    path: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, (bool, str, bytes)) or not isinstance(value, (int, float)):
        raise _error(label, path, "must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise _error(label, path, "must be a finite real number") from exc
    if not math.isfinite(result):
        raise _error(label, path, "must be finite")
    if minimum is not None and result < minimum:
        raise _error(label, path, f"must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise _error(label, path, f"must be at most {maximum}")
    return 0.0 if result == 0.0 else result


def _optional_probability(value: Any, label: str, path: str) -> Optional[float]:
    if value is None:
        return None
    return _real(value, label, path, minimum=0.0, maximum=1.0)


def _sha256(value: Any, label: str, path: str) -> str:
    result = _string(value, label, path)
    if not _SHA256.fullmatch(result):
        raise _error(label, path, "must be a 64-character SHA256 digest")
    if result != result.lower():
        raise _error(label, path, "must use lowercase hexadecimal")
    return result


def _box_within_image(
    box: tuple[float, float, float, float], width: int, height: int
) -> bool:
    x1, y1, x2, y2 = box
    return (
        x1 >= -_BOX_ABS_TOLERANCE
        and y1 >= -_BOX_ABS_TOLERANCE
        and x2 <= width + _BOX_ABS_TOLERANCE
        and y2 <= height + _BOX_ABS_TOLERANCE
    )


def _boxes_close(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return all(
        math.isclose(
            left,
            right,
            rel_tol=_BOX_REL_TOLERANCE,
            abs_tol=_BOX_ABS_TOLERANCE,
        )
        for left, right in zip(first, second)
    )


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VLMConfidenceReportError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise VLMConfidenceReportError(f"non-finite JSON constant {value!r} is forbidden")


def _load_json(path_value: Any, label: str) -> tuple[dict[str, Any], str]:
    if isinstance(path_value, bool) or not isinstance(path_value, (str, os.PathLike)):
        raise TypeError(f"{label} must be a filesystem path.")
    path = Path(path_value)
    if not path.is_file():
        raise VLMConfidenceReportError(f"{label}: {path} is not a regular file")
    with path.open("rb") as stream:
        payload = stream.read(_MAX_REPORT_BYTES + 1)
    if len(payload) > _MAX_REPORT_BYTES:
        raise VLMConfidenceReportError(
            f"{label}: report exceeds the {_MAX_REPORT_BYTES}-byte limit"
        )
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise VLMConfidenceReportError(f"{label}: report is not strict UTF-8") from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_json_constant,
        )
    except RecursionError as exc:
        raise VLMConfidenceReportError(
            f"{label}: JSON nesting exceeds the supported depth"
        ) from exc
    except (json.JSONDecodeError, VLMConfidenceReportError) as exc:
        if isinstance(exc, VLMConfidenceReportError):
            raise VLMConfidenceReportError(f"{label}: {exc}") from exc
        raise VLMConfidenceReportError(f"{label}: invalid JSON: {exc.msg}") from exc
    except ValueError as exc:
        raise VLMConfidenceReportError(f"{label}: invalid JSON number") from exc
    if not isinstance(decoded, dict):
        raise VLMConfidenceReportError(f"{label}:$: top level must be a JSON object")
    _validate_json_depth(decoded, label)
    return decoded, hashlib.sha256(payload).hexdigest()


def _validate_json_depth(value: Any, label: str, *, maximum: int = 64) -> None:
    """Reject unusually deep payloads before recursive canonicalization."""

    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > maximum:
            raise _error(label, "$", f"JSON nesting must not exceed {maximum}")
        if isinstance(item, Mapping):
            pending.extend((nested, depth + 1) for nested in item.values())
        elif isinstance(item, list):
            pending.extend((nested, depth + 1) for nested in item)


def _same_json_value(actual: Any, expected: Any) -> bool:
    """Compare reconstructed JSON surfaces without bool/int coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_json_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json_value(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _validate_dataset(
    value: Any,
    benchmark_config: Mapping[str, Any],
    label: str,
) -> tuple[
    dict[str, Any],
    tuple[VLMDetection, ...],
    tuple[str, ...],
    dict[str, tuple[int, int]],
]:
    dataset = _exact_fields(
        value,
        {"split", "class_names", "images", "evaluator_ground_truth", "ground_truth"},
        label,
        "$.dataset_manifest",
    )
    _string(dataset["split"], label, "$.dataset_manifest.split")
    class_names_raw = _sequence(
        dataset["class_names"], label, "$.dataset_manifest.class_names"
    )
    class_names = tuple(
        _string(item, label, f"$.dataset_manifest.class_names[{index}]")
        for index, item in enumerate(class_names_raw)
    )
    if list(class_names) != list(benchmark_config.get("class_names", ())):
        raise _error(
            label,
            "$.dataset_manifest.class_names",
            "does not match benchmark_config.class_names",
        )

    image_rows = _sequence(dataset["images"], label, "$.dataset_manifest.images")
    image_ids = []
    image_sizes = {}
    for index, raw in enumerate(image_rows):
        path = f"$.dataset_manifest.images[{index}]"
        row = _exact_fields(
            raw, {"image_id", "file_name", "sha256", "width", "height"}, label, path
        )
        image_id = _string(row["image_id"], label, f"{path}.image_id")
        file_name = _string(row["file_name"], label, f"{path}.file_name")
        if Path(file_name).name != file_name:
            raise _error(label, f"{path}.file_name", "must be a basename, not a path")
        _sha256(row["sha256"], label, f"{path}.sha256")
        width = _integer(
            row["width"],
            label,
            f"{path}.width",
            minimum=1,
            maximum=_MAX_SAFE_INTEGER,
        )
        height = _integer(
            row["height"],
            label,
            f"{path}.height",
            minimum=1,
            maximum=_MAX_SAFE_INTEGER,
        )
        image_ids.append(image_id)
        image_sizes[image_id] = (width, height)
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise _error(
            label, "$.dataset_manifest.images", "must contain unique image ids"
        )

    evaluator = _exact_fields(
        dataset["evaluator_ground_truth"],
        {"api", "images", "categories", "annotations"},
        label,
        "$.dataset_manifest.evaluator_ground_truth",
    )
    _string(evaluator["api"], label, "$.dataset_manifest.evaluator_ground_truth.api")
    evaluator_image_ids = set()
    evaluator_image_sizes = {}
    for index, raw in enumerate(
        _sequence(
            evaluator["images"],
            label,
            "$.dataset_manifest.evaluator_ground_truth.images",
        )
    ):
        path = f"$.dataset_manifest.evaluator_ground_truth.images[{index}]"
        row = _exact_fields(raw, {"id", "width", "height"}, label, path)
        image_id = _integer(row["id"], label, f"{path}.id")
        width = _integer(
            row["width"],
            label,
            f"{path}.width",
            minimum=1,
            maximum=_MAX_SAFE_INTEGER,
        )
        height = _integer(
            row["height"],
            label,
            f"{path}.height",
            minimum=1,
            maximum=_MAX_SAFE_INTEGER,
        )
        if image_id in evaluator_image_ids:
            raise _error(label, f"{path}.id", "is duplicated")
        evaluator_image_ids.add(image_id)
        evaluator_image_sizes[str(image_id)] = (width, height)
    if {str(value) for value in evaluator_image_ids} != set(image_ids):
        raise _error(
            label,
            "$.dataset_manifest.evaluator_ground_truth.images",
            "image ids do not match dataset_manifest.images",
        )
    if evaluator_image_sizes != image_sizes:
        raise _error(
            label,
            "$.dataset_manifest.evaluator_ground_truth.images",
            "dimensions do not match dataset_manifest.images",
        )

    category_ids = set()
    category_names = {}
    for index, raw in enumerate(
        _sequence(
            evaluator["categories"],
            label,
            "$.dataset_manifest.evaluator_ground_truth.categories",
        )
    ):
        path = f"$.dataset_manifest.evaluator_ground_truth.categories[{index}]"
        row = _exact_fields(raw, {"id", "name"}, label, path)
        category_id = _integer(row["id"], label, f"{path}.id")
        category_name = _string(row["name"], label, f"{path}.name")
        if category_id in category_ids:
            raise _error(label, f"{path}.id", "is duplicated")
        category_ids.add(category_id)
        category_names[category_id] = category_name

    evaluation = _mapping(
        benchmark_config.get("evaluation"), label, "$.benchmark_config.evaluation"
    )
    raw_category_map = evaluation.get("label_to_category_id")
    if raw_category_map is None:
        label_to_category = {}
        for category_id in category_ids:
            if category_id >= len(class_names):
                raise _error(
                    label,
                    "$.dataset_manifest.evaluator_ground_truth.categories",
                    "category id is outside the ordered class vocabulary",
                )
            label_to_category[category_id] = category_id
    else:
        raw_category_map = _mapping(
            raw_category_map,
            label,
            "$.benchmark_config.evaluation.label_to_category_id",
        )
        label_to_category = {}
        for raw_label, raw_category in raw_category_map.items():
            try:
                class_id = int(raw_label)
            except (TypeError, ValueError, OverflowError) as exc:
                raise _error(
                    label,
                    "$.benchmark_config.evaluation.label_to_category_id",
                    "class keys must be canonical non-negative integers",
                ) from exc
            if str(class_id) != raw_label or not 0 <= class_id < len(class_names):
                raise _error(
                    label,
                    "$.benchmark_config.evaluation.label_to_category_id",
                    "class keys must be canonical in-vocabulary integers",
                )
            label_to_category[class_id] = _integer(
                raw_category,
                label,
                f"$.benchmark_config.evaluation.label_to_category_id.{class_id}",
            )
    if len(set(label_to_category.values())) != len(label_to_category):
        raise _error(
            label,
            "$.benchmark_config.evaluation.label_to_category_id",
            "category ids must be unique",
        )
    if set(label_to_category.values()) != category_ids:
        raise _error(
            label,
            "$.dataset_manifest.evaluator_ground_truth.categories",
            "does not match the configured class-to-category mapping",
        )
    for class_id, category_id in label_to_category.items():
        if category_names[category_id] != class_names[class_id]:
            raise _error(
                label,
                "$.dataset_manifest.evaluator_ground_truth.categories",
                "category names do not match the ordered class vocabulary",
            )
    category_to_label = {
        category_id: class_id for class_id, category_id in label_to_category.items()
    }

    annotation_ids = set()
    evaluator_ground_truth = []
    for index, raw in enumerate(
        _sequence(
            evaluator["annotations"],
            label,
            "$.dataset_manifest.evaluator_ground_truth.annotations",
        )
    ):
        path = f"$.dataset_manifest.evaluator_ground_truth.annotations[{index}]"
        row = _exact_fields(
            raw,
            {"id", "image_id", "category_id", "bbox", "area", "iscrowd", "ignore"},
            label,
            path,
        )
        annotation_id = _integer(row["id"], label, f"{path}.id")
        if annotation_id in annotation_ids:
            raise _error(label, f"{path}.id", "is duplicated")
        annotation_ids.add(annotation_id)
        annotation_image_id = _integer(row["image_id"], label, f"{path}.image_id")
        if annotation_image_id not in evaluator_image_ids:
            raise _error(label, f"{path}.image_id", "references an unknown image")
        annotation_category_id = _integer(
            row["category_id"], label, f"{path}.category_id"
        )
        if annotation_category_id not in category_ids:
            raise _error(label, f"{path}.category_id", "references an unknown category")
        bbox = _sequence(row["bbox"], label, f"{path}.bbox")
        if len(bbox) != 4:
            raise _error(label, f"{path}.bbox", "must contain four values")
        bbox = tuple(
            _real(nested, label, f"{path}.bbox[{coordinate}]")
            for coordinate, nested in enumerate(bbox)
        )
        area = _real(row["area"], label, f"{path}.area")
        if _integer(row["iscrowd"], label, f"{path}.iscrowd") != 0:
            raise _error(label, f"{path}.iscrowd", "crowd annotations are unsupported")
        if _integer(row["ignore"], label, f"{path}.ignore") != 0:
            raise _error(label, f"{path}.ignore", "ignored annotations are unsupported")
        if area <= 0.0:
            continue
        image_id = str(annotation_image_id)
        image_width, image_height = image_sizes[image_id]
        clean_bbox = clipped_coco_bbox_xyxy(bbox, image_width, image_height)
        if clean_bbox is None:
            continue
        evaluator_ground_truth.append(
            VLMDetection(
                image_id,
                category_to_label[annotation_category_id],
                clean_bbox,
            )
        )

    ground_truth = []
    for index, raw in enumerate(
        _sequence(dataset["ground_truth"], label, "$.dataset_manifest.ground_truth")
    ):
        path = f"$.dataset_manifest.ground_truth[{index}]"
        row = _exact_fields(raw, {"image_id", "class_id", "xyxy"}, label, path)
        image_id = _string(row["image_id"], label, f"{path}.image_id")
        if image_id not in image_ids:
            raise _error(label, f"{path}.image_id", "references an unknown image")
        class_id = _integer(row["class_id"], label, f"{path}.class_id")
        if class_id >= len(class_names):
            raise _error(label, f"{path}.class_id", "is outside the class vocabulary")
        xyxy = _sequence(row["xyxy"], label, f"{path}.xyxy")
        try:
            detection = VLMDetection(image_id, class_id, tuple(xyxy))
        except (TypeError, ValueError, OverflowError) as exc:
            raise _error(label, f"{path}.xyxy", str(exc)) from exc
        width, height = image_sizes[image_id]
        if not _box_within_image(detection.xyxy, width, height):
            raise _error(label, f"{path}.xyxy", "must lie within the image")
        ground_truth.append(detection)

    evaluator_groups: dict[tuple[str, int], list[VLMDetection]] = {}
    ordering_groups: dict[tuple[str, int], list[VLMDetection]] = {}
    for detection in evaluator_ground_truth:
        evaluator_groups.setdefault(
            (detection.image_id, detection.class_id), []
        ).append(detection)
    for detection in ground_truth:
        ordering_groups.setdefault((detection.image_id, detection.class_id), []).append(
            detection
        )
    if evaluator_groups.keys() != ordering_groups.keys():
        raise _error(
            label,
            "$.dataset_manifest.ground_truth",
            "does not contain the same image/class groups as evaluator ground truth",
        )
    for key, evaluator_detections in evaluator_groups.items():
        remaining = list(ordering_groups[key])
        if len(evaluator_detections) != len(remaining):
            raise _error(
                label,
                "$.dataset_manifest.ground_truth",
                "does not contain the same boxes as evaluator ground truth",
            )
        for evaluator_detection in evaluator_detections:
            match = next(
                (
                    index
                    for index, ordering_detection in enumerate(remaining)
                    if _boxes_close(evaluator_detection.xyxy, ordering_detection.xyxy)
                ),
                None,
            )
            if match is None:
                raise _error(
                    label,
                    "$.dataset_manifest.ground_truth",
                    "does not contain the same boxes as evaluator ground truth",
                )
            remaining.pop(match)
    return dict(dataset), tuple(ground_truth), tuple(image_ids), image_sizes


def _validate_generations(
    value: Any,
    image_ids: tuple[str, ...],
    max_parsed_items: int,
    label: str,
) -> tuple[list[dict[str, Any]], tuple[int, int, int, int], Counter[str]]:
    rows = _sequence(value, label, "$.generation_manifest")
    normalized = []
    response_ids = []
    scored_responses = 0
    parsed_detections = 0
    scored_parsed_detections = 0
    reasons: Counter[str] = Counter()
    for index, raw in enumerate(rows):
        path = f"$.generation_manifest[{index}]"
        row = _exact_fields(raw, _GENERATION_FIELDS, label, path)
        image_id = _string(row["image_id"], label, f"{path}.image_id")
        digest = _sha256(row["sha256"], label, f"{path}.sha256")
        parsed = _integer(row["parsed_items"], label, f"{path}.parsed_items")
        if parsed > max_parsed_items:
            raise _error(
                label,
                f"{path}.parsed_items",
                "cannot exceed generation_kwargs.max_new_tokens",
            )
        reason = row["fallback_reason"]
        if reason is not None:
            reason = _string(reason, label, f"{path}.fallback_reason")
            reasons[reason] += 1
        else:
            scored_responses += 1
            scored_parsed_detections += parsed
        parsed_detections += parsed
        response_ids.append(image_id)
        normalized.append(
            {
                "image_id": image_id,
                "sha256": digest,
                "parsed_items": parsed,
                "fallback_reason": reason,
            }
        )
    if tuple(response_ids) != image_ids:
        raise _error(
            label,
            "$.generation_manifest",
            "must contain exactly one response per dataset image in dataset order",
        )
    return (
        normalized,
        (len(rows), scored_responses, parsed_detections, scored_parsed_detections),
        reasons,
    )


def _validate_predictions(
    value: Any,
    image_sizes: Mapping[str, tuple[int, int]],
    class_count: int,
    fallback_score: float,
    label: str,
) -> tuple[tuple[VLMDetection, ...], tuple[bool, ...]]:
    rows = _sequence(value, label, "$.predictions")
    predictions = []
    matches = []
    geometry = set()
    for index, raw in enumerate(rows):
        path = f"$.predictions[{index}]"
        row = _exact_fields(raw, _PREDICTION_FIELDS, label, path)
        image_id = _string(row["image_id"], label, f"{path}.image_id")
        if image_id not in image_sizes:
            raise _error(label, f"{path}.image_id", "references an unknown image")
        class_id = _integer(row["class_id"], label, f"{path}.class_id")
        if class_id >= class_count:
            raise _error(label, f"{path}.class_id", "is outside the class vocabulary")
        candidate = _optional_probability(
            row["candidate_score"], label, f"{path}.candidate_score"
        )
        effective = _real(
            row["effective_score"],
            label,
            f"{path}.effective_score",
            minimum=0.0,
            maximum=1.0,
        )
        expected_effective = fallback_score if candidate is None else candidate
        if effective != expected_effective:
            raise _error(
                label,
                f"{path}.effective_score",
                "does not match candidate score or configured fallback",
            )
        matched = row["matched"]
        if not isinstance(matched, bool):
            raise _error(label, f"{path}.matched", "must be boolean")
        xyxy = _sequence(row["xyxy"], label, f"{path}.xyxy")
        try:
            prediction = VLMDetection(image_id, class_id, tuple(xyxy), candidate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _error(label, f"{path}.xyxy", str(exc)) from exc
        width, height = image_sizes[image_id]
        if not _box_within_image(prediction.xyxy, width, height):
            raise _error(label, f"{path}.xyxy", "must lie within the image")
        key = (prediction.image_id, prediction.class_id, prediction.xyxy)
        if key in geometry:
            raise _error(label, path, "duplicates an earlier prediction geometry")
        geometry.add(key)
        predictions.append(prediction)
        matches.append(matched)
    return tuple(predictions), tuple(matches)


def _diagnostics_surface(run: ConfidenceRun) -> dict[str, Any]:
    value = run.diagnostics
    return {field: getattr(value, field) for field in _DIAGNOSTIC_FIELDS}


def _calibration_surface(run: ConfidenceRun) -> dict[str, Any]:
    value = run.calibration
    return {
        "method": "equal_width",
        "population": "scored_postprocessed_predictions",
        "bin_count": value.bin_count,
        "total_predictions": value.total_predictions,
        "scored_predictions": value.scored_predictions,
        "unscored_predictions": value.unscored_predictions,
        "score_coverage": value.score_coverage,
        "brier_score": value.brier_score,
        "expected_calibration_error": value.expected_calibration_error,
        "maximum_calibration_error": value.maximum_calibration_error,
        "bins": [
            {field: getattr(item, field) for field in _BIN_FIELDS}
            for item in value.bins
        ],
    }


def _optional(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)


def _semantic_metrics(
    run: ConfidenceRun, response: tuple[int, int, int, int]
) -> dict[str, Optional[float]]:
    evaluator = dict(run.evaluator_metrics)
    diagnostics = run.diagnostics
    responses, scored_responses, parsed, scored_parsed = response
    return {
        "metrics/vlm_confidence/candidate_mAP50-95": evaluator["candidate_mAP50-95"],
        "metrics/vlm_confidence/constant_mAP50-95": evaluator["constant_mAP50-95"],
        "metrics/vlm_confidence/delta_mAP50-95": (
            evaluator["candidate_mAP50-95"] - evaluator["constant_mAP50-95"]
        ),
        "metrics/vlm_confidence/candidate_mAP50": evaluator["candidate_mAP50"],
        "metrics/vlm_confidence/constant_mAP50": evaluator["constant_mAP50"],
        "metrics/vlm_confidence/delta_mAP50": (
            evaluator["candidate_mAP50"] - evaluator["constant_mAP50"]
        ),
        "metrics/vlm_confidence/auroc": _optional(run.auroc),
        "metrics/vlm_confidence/ranking_ap": _optional(run.ranking_ap),
        "metrics/vlm_confidence/scored_prediction_brier": _optional(
            run.calibration.brier_score
        ),
        "metrics/vlm_confidence/scored_prediction_ece": _optional(
            run.calibration.expected_calibration_error
        ),
        "metrics/vlm_confidence/scored_prediction_mce": _optional(
            run.calibration.maximum_calibration_error
        ),
        "metrics/vlm_confidence/default_conf_tp_retention": _optional(
            diagnostics.correct_retention
        ),
        "metrics/vlm_confidence/default_conf_fp_retention": _optional(
            diagnostics.incorrect_retention
        ),
        "metrics/vlm_confidence/default_conf_prediction_retention": float(
            diagnostics.default_conf_retention
        ),
        "metrics/vlm_confidence/response_score_coverage": (
            scored_responses / responses if responses else 0.0
        ),
        "metrics/vlm_confidence/detection_score_coverage": (
            scored_parsed / parsed if parsed else 0.0
        ),
        "metrics/vlm_confidence/prediction_score_coverage": float(
            diagnostics.score_coverage
        ),
        "metrics/vlm_confidence/responses": float(responses),
        "metrics/vlm_confidence/scored_responses": float(scored_responses),
        "metrics/vlm_confidence/parsed_detections": float(parsed),
        "metrics/vlm_confidence/scored_parsed_detections": float(scored_parsed),
        "metrics/vlm_confidence/predictions": float(diagnostics.total_predictions),
        "metrics/vlm_confidence/correct_predictions": float(
            diagnostics.correct_predictions
        ),
        "metrics/vlm_confidence/incorrect_predictions": float(
            diagnostics.incorrect_predictions
        ),
        "metrics/vlm_confidence/retained_correct_predictions": float(
            diagnostics.retained_correct_predictions
        ),
        "metrics/vlm_confidence/retained_incorrect_predictions": float(
            diagnostics.retained_incorrect_predictions
        ),
    }


def _validate_speed(
    metrics: Mapping[str, Any], response_count: int, label: str
) -> None:
    values = {
        key: _real(metrics[key], label, f"$.metrics.{key}", minimum=0.0)
        for key in _SPEED_METRICS
    }
    images_seen = values["speed/images_seen"]
    if not images_seen.is_integer() or int(images_seen) != response_count:
        raise _error(
            label,
            "$.metrics.speed/images_seen",
            "must equal the validated response count",
        )
    expected_ms = values["speed/total_s"] / response_count * 1000.0
    if not math.isclose(
        values["speed/total_ms"], expected_ms, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise _error(
            label,
            "$.metrics.speed/total_ms",
            "does not agree with total_s and images_seen",
        )


def _validate_report(path_value: Any, label: str) -> _ValidatedReport:
    report, file_digest = _load_json(path_value, label)
    report = dict(_exact_fields(report, _ROOT_FIELDS, label, "$"))
    if report["schema"] != _REPORT_SCHEMA:
        raise _error(label, "$.schema", f"must equal {_REPORT_SCHEMA!r}")
    prompt = _string(report["prompt"], label, "$.prompt", nonempty=False)
    benchmark = dict(_mapping(report["benchmark_config"], label, "$.benchmark_config"))
    dataset, ground_truth, image_ids, image_sizes = _validate_dataset(
        report["dataset_manifest"], benchmark, label
    )
    generation_kwargs = _mapping(
        benchmark.get("generation_kwargs"),
        label,
        "$.benchmark_config.generation_kwargs",
    )
    max_new_tokens = _integer(
        generation_kwargs.get("max_new_tokens"),
        label,
        "$.benchmark_config.generation_kwargs.max_new_tokens",
        minimum=1,
    )
    generations, response_diagnostics, fallback_reasons = _validate_generations(
        report["generation_manifest"], image_ids, max_new_tokens, label
    )

    confidence = _exact_fields(
        report["confidence"], _CONFIDENCE_FIELDS, label, "$.confidence"
    )
    iou_threshold = _real(
        confidence["iou_threshold"],
        label,
        "$.confidence.iou_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    if iou_threshold == 0.0:
        raise _error(label, "$.confidence.iou_threshold", "must be positive")
    default_conf = _real(
        confidence["default_conf"],
        label,
        "$.confidence.default_conf",
        minimum=0.0,
        maximum=1.0,
    )
    fallback_score = _real(
        confidence["fallback_score"],
        label,
        "$.confidence.fallback_score",
        minimum=0.0,
        maximum=1.0,
    )
    predictions, recorded_matches = _validate_predictions(
        report["predictions"],
        image_sizes,
        len(dataset["class_names"]),
        fallback_score,
        label,
    )
    predictions_by_image: dict[str, list[VLMDetection]] = {
        image_id: [] for image_id in image_ids
    }
    for prediction in predictions:
        predictions_by_image[prediction.image_id].append(prediction)
    for index, generation in enumerate(generations):
        image_predictions = predictions_by_image[generation["image_id"]]
        if len(image_predictions) > generation["parsed_items"]:
            raise _error(
                label,
                f"$.generation_manifest[{index}].parsed_items",
                "must be at least the number of retained predictions",
            )
        score_available = generation["fallback_reason"] is None
        if any(
            (prediction.score is not None) != score_available
            for prediction in image_predictions
        ):
            raise _error(
                label,
                f"$.predictions[image_id={generation['image_id']!r}]",
                "candidate scores must follow the response-wide fallback state",
            )
    evaluator_metrics = _exact_fields(
        report["evaluator_metrics"],
        _EVALUATOR_METRIC_FIELDS,
        label,
        "$.evaluator_metrics",
    )
    for key, value in evaluator_metrics.items():
        _real(value, label, f"$.evaluator_metrics.{key}", minimum=0.0, maximum=1.0)

    try:
        run = build_confidence_run(
            predictions,
            ground_truth,
            prompt=prompt,
            dataset_manifest=dataset,
            benchmark_config=benchmark,
            generation_manifest=generations,
            evaluator_metrics=evaluator_metrics,
            iou_threshold=iou_threshold,
            default_conf=default_conf,
            fallback_score=fallback_score,
        )
    except (TypeError, ValueError) as exc:
        raise _error(label, "$", f"cannot reconstruct confidence run: {exc}") from exc

    hashes = _exact_fields(report["hashes"], _HASH_FIELDS, label, "$.hashes")
    expected_hashes = {
        "manifest": run.manifest_hash,
        "configuration": run.configuration_hash,
        "generation": run.generation_hash,
        "prediction_structure": run.prediction_structure_hash,
    }
    for key, expected in expected_hashes.items():
        actual = _sha256(hashes[key], label, f"$.hashes.{key}")
        if actual != expected:
            raise _error(label, f"$.hashes.{key}", "does not match reconstructed data")
    if recorded_matches != run.matches:
        raise _error(
            label, "$.predictions[*].matched", "does not match reconstructed matching"
        )

    diagnostics = _exact_fields(
        report["diagnostics"], _DIAGNOSTIC_FIELDS, label, "$.diagnostics"
    )
    if not _same_json_value(dict(diagnostics), _diagnostics_surface(run)):
        raise _error(label, "$.diagnostics", "does not match reconstructed diagnostics")
    calibration = _exact_fields(
        report["calibration"], _CALIBRATION_FIELDS, label, "$.calibration"
    )
    bins = _sequence(calibration["bins"], label, "$.calibration.bins")
    for index, item in enumerate(bins):
        _exact_fields(item, _BIN_FIELDS, label, f"$.calibration.bins[{index}]")
    if not _same_json_value(dict(calibration), _calibration_surface(run)):
        raise _error(label, "$.calibration", "does not match reconstructed calibration")
    if dict(evaluator_metrics) != dict(run.evaluator_metrics):
        raise _error(label, "$.evaluator_metrics", "does not match normalized metrics")

    recorded_fallbacks = _mapping(
        report["fallback_reasons"], label, "$.fallback_reasons"
    )
    normalized_fallbacks = {}
    for key, value in recorded_fallbacks.items():
        reason = _string(key, label, "$.fallback_reasons.<key>")
        normalized_fallbacks[reason] = _integer(
            value, label, f"$.fallback_reasons.{reason}", minimum=1
        )
    if normalized_fallbacks != dict(sorted(fallback_reasons.items())):
        raise _error(
            label,
            "$.fallback_reasons",
            "does not match generation_manifest fallback reasons",
        )

    metrics = _mapping(report["metrics"], label, "$.metrics")
    expected_metric_fields = _SEMANTIC_METRICS | _SPEED_METRICS
    _exact_fields(metrics, expected_metric_fields, label, "$.metrics")
    expected_semantic = _semantic_metrics(run, response_diagnostics)
    for key, expected in expected_semantic.items():
        actual = metrics[key]
        if actual is not None:
            actual = _real(actual, label, f"$.metrics.{key}")
        if actual != expected:
            raise _error(
                label, f"$.metrics.{key}", "does not match reconstructed value"
            )
    _validate_speed(metrics, response_diagnostics[0], label)

    artifacts = _exact_fields(
        report["artifacts"], {"reliability_plot"}, label, "$.artifacts"
    )
    artifact = artifacts["reliability_plot"]
    if artifact not in (None, "vlm_confidence_reliability.svg"):
        raise _error(
            label,
            "$.artifacts.reliability_plot",
            "must be null or the canonical report-local SVG basename",
        )
    return _ValidatedReport(
        file_sha256=file_digest,
        benchmark_config=benchmark,
        metrics=tuple(
            sorted(
                (key, None if value is None else float(value))
                for key, value in metrics.items()
            )
        ),
        run=run,
        response_diagnostics=response_diagnostics,
        fallback_reasons=tuple(sorted(normalized_fallbacks.items())),
        semantic_metrics=tuple(sorted(expected_semantic.items())),
    )


def read_confidence_report_identity(
    path: str | os.PathLike[str], *, label: str = "report"
) -> tuple[str, dict[str, Any], dict[str, Optional[float]]]:
    """Return digest, configuration, and metrics from a validated report."""

    validated = _validate_report(path, label)
    return (
        validated.file_sha256,
        dict(validated.benchmark_config),
        dict(validated.metrics),
    )


def _tolerance(value: Any, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError(f"{name} must be a finite non-negative number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite non-negative number.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _metric_delta(first: Optional[float], second: Optional[float]) -> Optional[float]:
    if first is None and second is None:
        return 0.0
    if first is None or second is None:
        return None
    return abs(first - second)


def compare_confidence_reports(
    first_path: str | os.PathLike[str],
    second_path: str | os.PathLike[str],
    *,
    score_atol: float = 0.0,
    metric_atol: float = 0.0,
    map_atol: float = 0.0,
) -> PersistedRepeatComparison:
    """Validate and compare two persisted confidence-gate reports.

    Valid reports with different identities return ``reproducible=False``.
    Malformed or internally inconsistent reports raise
    :class:`VLMConfidenceReportError`. Timing and artifact-presence differences
    are validated but deliberately excluded from reproducibility. Stored mAP is
    not independently recomputed or cryptographically authenticated.
    """

    score_tolerance = _tolerance(score_atol, "score_atol")
    metric_tolerance = _tolerance(metric_atol, "metric_atol")
    map_tolerance = _tolerance(map_atol, "map_atol")
    first = _validate_report(first_path, "first")
    second = _validate_report(second_path, "second")
    core = compare_repeats(
        first.run,
        second.run,
        score_atol=score_tolerance,
        metric_atol=metric_tolerance,
        map_atol=map_tolerance,
    )

    first_metrics = dict(first.semantic_metrics)
    second_metrics = dict(second.semantic_metrics)
    same_keys = first_metrics.keys() == second_metrics.keys()
    map_ok = True
    non_map_ok = True
    deltas = []
    differing = []
    if same_keys:
        for key in sorted(first_metrics):
            delta = _metric_delta(first_metrics[key], second_metrics[key])
            if delta is None:
                differing.append(f"metrics.{key}")
                if key in _MAP_METRICS:
                    map_ok = False
                else:
                    non_map_ok = False
                continue
            deltas.append(delta)
            if key in _MAP_METRICS:
                within = delta <= map_tolerance
                map_ok &= within
            elif key in _QUALITY_METRICS:
                within = delta <= metric_tolerance
                non_map_ok &= within
            else:
                within = delta == 0.0
                non_map_ok &= within
            if not within:
                differing.append(f"metrics.{key}")
    else:
        map_ok = False
        non_map_ok = False
        differing.append("metrics.<keys>")

    same_response = first.response_diagnostics == second.response_diagnostics
    same_fallbacks = first.fallback_reasons == second.fallback_reasons
    if not same_response:
        differing.append("response_diagnostics")
    if not same_fallbacks:
        differing.append("fallback_reasons")
    identity_flags = (
        (core.same_manifest, "hashes.manifest"),
        (core.same_configuration, "hashes.configuration"),
        (core.same_generation, "hashes.generation"),
        (core.same_prediction_structure, "hashes.prediction_structure"),
        (core.same_matches, "predictions[*].matched"),
        (core.same_score_availability, "predictions[*].candidate_score.availability"),
        (core.scores_within_tolerance, "predictions[*].candidate_score"),
        (core.same_calibration_bin_assignments, "calibration.bins.assignments"),
        (core.calibration_bins_within_tolerance, "calibration.bins.values"),
        (core.evaluator_metrics_within_tolerance, "evaluator_metrics"),
        (core.same_diagnostics, "diagnostics"),
    )
    differing.extend(path for passed, path in identity_flags if not passed)
    semantic_ok = same_keys and map_ok and non_map_ok
    reproducible = all(
        (
            core.reproducible,
            same_response,
            same_fallbacks,
            semantic_ok,
        )
    )
    return PersistedRepeatComparison(
        first_report_sha256=first.file_sha256,
        second_report_sha256=second.file_sha256,
        core=core,
        same_response_diagnostics=same_response,
        same_fallback_reasons=same_fallbacks,
        same_semantic_metric_keys=same_keys,
        map_metrics_within_tolerance=map_ok,
        non_map_metrics_within_tolerance=non_map_ok,
        semantic_metrics_within_tolerance=semantic_ok,
        max_abs_semantic_metric_delta=max(deltas, default=0.0),
        differing_fields=tuple(dict.fromkeys(differing)),
        reproducible=reproducible,
    )
