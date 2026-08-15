"""Offline quality gates for candidate LibreVLM confidence scores.

This module is intentionally internal.  It contains deterministic, dependency-
free metric plumbing for real-data experiments, but it does not enable VLM
``val()``, alter ``predict()``, or promote any family's candidate score.

The binary metrics answer a narrower question than detector mAP: given a fixed
set of generated boxes, do larger candidate scores rank correct boxes above
incorrect ones? A prediction is correct when score-independent IoU matching
pairs it with a ground-truth box from the same image and class.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

_MANIFEST_SCHEMA = "libreyolo.vlm-confidence-manifest.v1"
_CONFIGURATION_SCHEMA = "libreyolo.vlm-confidence-configuration.v1"
_GENERATION_SCHEMA = "libreyolo.vlm-confidence-generations.v1"
_PREDICTION_SCHEMA = "libreyolo.vlm-confidence-predictions.v1"
_REQUIRED_CONFIGURATION_FIELDS = (
    "family",
    "size",
    "base_repo",
    "base_revision",
    "checkpoint",
    "processor",
    "class_names",
    "generation_kwargs",
    "confidence_method",
    "confidence_evaluation",
    "evaluation",
    "seed",
    "backend",
    "device",
    "dtype",
    "hardware",
    "software",
)
_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_REQUIRED_GENERATION_FIELDS = (
    "max_new_tokens",
    "do_sample",
    "num_beams",
    "repetition_penalty",
)
_REQUIRED_SOFTWARE_FIELDS = (
    "python",
    "libreyolo",
    "torch",
    "transformers",
    "pycocotools",
)
_EVALUATOR_METRIC_NAMES = (
    "candidate_mAP50-95",
    "constant_mAP50-95",
    "candidate_mAP50",
    "constant_mAP50",
)


@dataclass(frozen=True)
class VLMDetection:
    """One prediction or ground-truth box in pixel ``xyxy`` coordinates.

    ``score=None`` identifies a prediction for which the candidate confidence
    path could not produce a score. Coverage is measured per prediction;
    callers evaluating an all-or-nothing response fallback must expand that
    response policy before building a run. Ground-truth scores are ignored.
    """

    image_id: str
    class_id: int
    xyxy: tuple[float, float, float, float]
    score: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id:
            raise ValueError("image_id must be a non-empty string.")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            raise TypeError("class_id must be an integer.")
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative.")

        try:
            coords = tuple(float(value) for value in self.xyxy)
        except (TypeError, ValueError) as exc:
            raise ValueError("xyxy must contain four finite numbers.") from exc
        if len(coords) != 4 or not all(math.isfinite(value) for value in coords):
            raise ValueError("xyxy must contain four finite numbers.")
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            raise ValueError("xyxy must have positive width and height.")
        coords = tuple(0.0 if value == 0.0 else value for value in coords)
        object.__setattr__(self, "xyxy", coords)

        if self.score is not None:
            score = _probability(self.score, "score")
            object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class ConfidenceDiagnostics:
    """Coverage and retention behavior at the public confidence threshold."""

    default_conf: float
    fallback_score: float
    total_predictions: int
    scored_predictions: int
    fallback_predictions: int
    score_coverage: float
    retained_predictions: int
    default_conf_retention: float
    correct_predictions: int
    retained_correct_predictions: int
    correct_retention: Optional[float]
    incorrect_predictions: int
    retained_incorrect_predictions: int
    incorrect_retention: Optional[float]


@dataclass(frozen=True)
class ReliabilityBin:
    """One equal-width confidence bin over score-bearing predictions only."""

    index: int
    lower: float
    upper: float
    count: int
    correct: int
    mean_confidence: Optional[float]
    empirical_accuracy: Optional[float]
    absolute_gap: Optional[float]


@dataclass(frozen=True)
class CalibrationDiagnostics:
    """Descriptive calibration errors for the candidate token score.

    These values do not establish that a score is calibrated. Missing scores
    are excluded rather than replacing them with the synthetic fallback;
    coverage remains visible through :class:`ConfidenceDiagnostics`.
    """

    bin_count: int
    total_predictions: int
    scored_predictions: int
    unscored_predictions: int
    score_coverage: float
    brier_score: Optional[float]
    expected_calibration_error: Optional[float]
    maximum_calibration_error: Optional[float]
    bins: tuple[ReliabilityBin, ...]


@dataclass(frozen=True)
class ConfidenceRun:
    """Immutable output of one confidence-quality experiment."""

    manifest_hash: str
    configuration_hash: str
    generation_hash: str
    prediction_structure_hash: str
    iou_threshold: float
    default_conf: float
    fallback_score: float
    scores: tuple[Optional[float], ...]
    matches: tuple[bool, ...]
    auroc: Optional[float]
    ranking_ap: Optional[float]
    diagnostics: ConfidenceDiagnostics
    calibration: CalibrationDiagnostics
    evaluator_metrics: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class RepeatComparison:
    """Determinism comparison for two runs over the same intended benchmark."""

    same_manifest: bool
    same_configuration: bool
    same_generation: bool
    same_prediction_structure: bool
    same_matches: bool
    same_score_availability: bool
    scores_within_tolerance: bool
    metrics_within_tolerance: bool
    same_calibration_bin_assignments: bool
    calibration_bins_within_tolerance: bool
    same_evaluator_metric_keys: bool
    evaluator_metrics_within_tolerance: bool
    same_diagnostics: bool
    max_abs_score_delta: Optional[float]
    max_abs_calibration_bin_delta: Optional[float]
    max_abs_evaluator_metric_delta: Optional[float]
    auroc_delta: Optional[float]
    ranking_ap_delta: Optional[float]
    brier_score_delta: Optional[float]
    expected_calibration_error_delta: Optional[float]
    maximum_calibration_error_delta: Optional[float]
    reproducible: bool


def _probability(value: Any, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError(f"{name} must be a real number in [0, 1].")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number in [0, 1].") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1].")
    return 0.0 if result == 0.0 else result


def _iou(first: VLMDetection, second: VLMDetection) -> float:
    ax1, ay1, ax2, ay2 = first.xyxy
    bx1, by1, bx2, by2 = second.xyxy
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    if intersection == 0.0:
        return 0.0
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    return intersection / (first_area + second_area - intersection)


def _maximum_weight_assignment(weights: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Return one unique column per row for a finite rectangular weight matrix.

    This is an original, deterministic implementation of the assignment
    algorithm. The caller supplies at least as many columns as rows. Lower
    column indices win exact optimization ties.
    """

    row_count = len(weights)
    if row_count == 0:
        return ()
    column_count = len(weights[0])
    if column_count < row_count or any(len(row) != column_count for row in weights):
        raise ValueError(
            "assignment weights must be rectangular with at least one column per row."
        )
    if any(not math.isfinite(value) for row in weights for value in row):
        raise ValueError("assignment weights must be finite.")

    maximum = max(max(row) for row in weights)
    costs = [[maximum - value for value in row] for row in weights]
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        matched_row[0] = row
        current_column = 0
        minimum = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = matched_row[current_column]
            delta = math.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = (
                    costs[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    predecessor[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            if not math.isfinite(delta):
                raise RuntimeError(
                    "assignment optimization found no augmenting column."
                )
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[matched_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while current_column != 0:
            previous = predecessor[current_column]
            matched_row[current_column] = matched_row[previous]
            current_column = previous

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column] != 0:
            assignment[matched_row[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise RuntimeError("assignment optimization left a row unmatched.")
    return tuple(assignment)


def match_detections(
    predictions: Sequence[VLMDetection],
    ground_truth: Sequence[VLMDetection],
    *,
    iou_threshold: float = 0.5,
) -> tuple[bool, ...]:
    """Mark predictions matched by same-image, same-class IoU.

    Matching is independent of candidate confidence. Within each image/class,
    it first maximizes the number of valid one-to-one pairs, then maximizes
    their total IoU. Canonical geometry/source ordering breaks exact ties.
    ``IoU == iou_threshold`` is a match.
    """

    threshold = _probability(iou_threshold, "iou_threshold")
    if threshold == 0.0:
        raise ValueError("iou_threshold must be greater than zero.")
    predictions = tuple(predictions)
    ground_truth = tuple(ground_truth)
    if not all(isinstance(item, VLMDetection) for item in predictions):
        raise TypeError("predictions must contain only VLMDetection values.")
    if not all(isinstance(item, VLMDetection) for item in ground_truth):
        raise TypeError("ground_truth must contain only VLMDetection values.")

    predictions_by_key: dict[tuple[str, int], list[int]] = {}
    for prediction_index, prediction in enumerate(predictions):
        predictions_by_key.setdefault(
            (prediction.image_id, prediction.class_id), []
        ).append(prediction_index)
    targets_by_key: dict[tuple[str, int], list[int]] = {}
    for target_index, target in enumerate(ground_truth):
        targets_by_key.setdefault((target.image_id, target.class_id), []).append(
            target_index
        )

    matches = [False] * len(predictions)
    for key in sorted(predictions_by_key):
        prediction_indices = sorted(
            predictions_by_key[key],
            key=lambda index: (predictions[index].xyxy, index),
        )
        target_indices = sorted(
            targets_by_key.get(key, ()),
            key=lambda index: (ground_truth[index].xyxy, index),
        )
        if not target_indices:
            continue
        cardinality_bonus = float(min(len(prediction_indices), len(target_indices)) + 1)
        forbidden = -cardinality_bonus * float(len(prediction_indices) + 1)
        weights: list[list[float]] = []
        overlaps: list[list[float]] = []
        for prediction_index in prediction_indices:
            row_overlaps = [
                _iou(predictions[prediction_index], ground_truth[target_index])
                for target_index in target_indices
            ]
            overlaps.append(row_overlaps)
            weights.append(
                [
                    cardinality_bonus + overlap if overlap >= threshold else forbidden
                    for overlap in row_overlaps
                ]
                + [0.0] * len(prediction_indices)
            )
        assignment = _maximum_weight_assignment(weights)
        for row, column in enumerate(assignment):
            if column < len(target_indices) and overlaps[row][column] >= threshold:
                matches[prediction_indices[row]] = True
    return tuple(matches)


def _score_label_pairs(
    scores: Sequence[float], labels: Sequence[bool]
) -> list[tuple[float, bool]]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    pairs = []
    for index, (score, label) in enumerate(zip(scores, labels)):
        if not isinstance(label, bool) and label not in (0, 1):
            raise ValueError(f"labels[{index}] must be boolean or 0/1.")
        pairs.append((_probability(score, f"scores[{index}]"), bool(label)))
    return pairs


def tie_aware_auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Return binary AUROC, awarding half credit to equal-score pairs.

    ``None`` means AUROC is undefined because the input contains no positive or
    no negative examples.
    """

    pairs = _score_label_pairs(scores, labels)
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None

    negative_below = 0
    wins = 0.0
    ordered = sorted(pairs, key=lambda pair: pair[0])
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        group = ordered[start:end]
        group_positives = sum(label for _, label in group)
        group_negatives = len(group) - group_positives
        wins += group_positives * negative_below
        wins += 0.5 * group_positives * group_negatives
        negative_below += group_negatives
        start = end
    return wins / (positives * negatives)


def binary_ranking_ap(
    scores: Sequence[float], labels: Sequence[bool]
) -> Optional[float]:
    """Return non-interpolated binary ranking average precision.

    Equal scores are consumed as one threshold group, so input ordering cannot
    improve or degrade the result. ``None`` means there are no positive labels.
    This is a confidence-ranking diagnostic, not COCO detector AP.
    """

    pairs = _score_label_pairs(scores, labels)
    total_positives = sum(label for _, label in pairs)
    if total_positives == 0:
        return None

    true_positives = 0
    seen = 0
    average_precision = 0.0
    ordered = sorted(pairs, key=lambda pair: pair[0], reverse=True)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        group = ordered[start:end]
        group_positives = sum(label for _, label in group)
        true_positives += group_positives
        seen += len(group)
        average_precision += (group_positives / total_positives) * (
            true_positives / seen
        )
        start = end
    return average_precision


def calibration_diagnostics(
    scores: Sequence[Optional[float]],
    labels: Sequence[bool],
    *,
    n_bins: int = 10,
) -> CalibrationDiagnostics:
    """Return Brier/ECE diagnostics and fixed equal-width reliability bins.

    Missing candidate scores are counted for coverage but omitted from the
    errors and bins. They are never replaced by the constant fallback, which
    would describe fallback policy rather than candidate calibration.
    """

    if isinstance(n_bins, bool) or not isinstance(n_bins, int):
        raise TypeError("n_bins must be an integer.")
    if not 1 <= n_bins <= 1000:
        raise ValueError("n_bins must lie in [1, 1000].")
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    pairs = []
    for index, (score, label) in enumerate(zip(scores, labels)):
        if not isinstance(label, bool) and label not in (0, 1):
            raise ValueError(f"labels[{index}] must be boolean or 0/1.")
        if score is not None:
            pairs.append((_probability(score, f"scores[{index}]"), bool(label)))
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    squared_error = 0.0
    for score, label in pairs:
        index = min(int(score * n_bins), n_bins - 1)
        grouped[index].append((score, label))
        squared_error += (score - float(label)) ** 2

    total = len(pairs)
    weighted_gap = 0.0
    maximum_gap: Optional[float] = None
    bins = []
    for index, group in enumerate(grouped):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if group:
            mean_confidence = sum(score for score, _ in group) / len(group)
            correct = sum(label for _, label in group)
            empirical_accuracy = correct / len(group)
            gap = abs(mean_confidence - empirical_accuracy)
            weighted_gap += len(group) * gap
            maximum_gap = gap if maximum_gap is None else max(maximum_gap, gap)
        else:
            correct = 0
            mean_confidence = None
            empirical_accuracy = None
            gap = None
        bins.append(
            ReliabilityBin(
                index=index,
                lower=lower,
                upper=upper,
                count=len(group),
                correct=correct,
                mean_confidence=mean_confidence,
                empirical_accuracy=empirical_accuracy,
                absolute_gap=gap,
            )
        )
    return CalibrationDiagnostics(
        bin_count=n_bins,
        total_predictions=len(scores),
        scored_predictions=total,
        unscored_predictions=len(scores) - total,
        score_coverage=total / len(scores) if scores else 0.0,
        brier_score=squared_error / total if total else None,
        expected_calibration_error=weighted_gap / total if total else None,
        maximum_calibration_error=maximum_gap,
        bins=tuple(bins),
    )


def confidence_diagnostics(
    scores: Sequence[Optional[float]],
    labels: Sequence[bool],
    *,
    default_conf: float = 0.25,
    fallback_score: float = 1.0,
) -> ConfidenceDiagnostics:
    """Measure candidate coverage and box retention at ``default_conf``.

    Missing candidate scores count against per-prediction coverage and use
    ``fallback_score`` for retention. This function has no response grouping;
    an all-or-nothing family fallback must be expanded by its caller first.
    """

    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    threshold = _probability(default_conf, "default_conf")
    fallback = _probability(fallback_score, "fallback_score")

    effective_scores: list[float] = []
    normalized_labels: list[bool] = []
    scored = 0
    for index, (score, label) in enumerate(zip(scores, labels)):
        if not isinstance(label, bool) and label not in (0, 1):
            raise ValueError(f"labels[{index}] must be boolean or 0/1.")
        normalized_labels.append(bool(label))
        if score is None:
            effective_scores.append(fallback)
        else:
            effective_scores.append(_probability(score, f"scores[{index}]"))
            scored += 1

    retained = [score >= threshold for score in effective_scores]
    total = len(scores)
    correct = sum(normalized_labels)
    incorrect = total - correct
    retained_correct = sum(
        keep and label for keep, label in zip(retained, normalized_labels)
    )
    retained_incorrect = sum(
        keep and not label for keep, label in zip(retained, normalized_labels)
    )
    retained_total = sum(retained)
    return ConfidenceDiagnostics(
        default_conf=threshold,
        fallback_score=fallback,
        total_predictions=total,
        scored_predictions=scored,
        fallback_predictions=total - scored,
        score_coverage=scored / total if total else 0.0,
        retained_predictions=retained_total,
        default_conf_retention=retained_total / total if total else 0.0,
        correct_predictions=correct,
        retained_correct_predictions=retained_correct,
        correct_retention=retained_correct / correct if correct else None,
        incorrect_predictions=incorrect,
        retained_incorrect_predictions=retained_incorrect,
        incorrect_retention=(retained_incorrect / incorrect if incorrect else None),
    )


def _canonicalize(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float.")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{path} mapping keys must be strings.")
        normalized = {}
        for key in sorted(value):
            normalized[key] = _canonicalize(value[key], f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _canonicalize(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} must contain only JSON-compatible deterministic values; "
        f"got {type(value).__name__}."
    )


def _payload_hash(payload: Any) -> str:
    encoded = json.dumps(
        _canonicalize(payload, "manifest"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_benchmark_config(benchmark_config: Any) -> dict[str, Any]:
    if not isinstance(benchmark_config, Mapping):
        raise TypeError("benchmark_config must be a mapping.")
    missing = [
        field
        for field in _REQUIRED_CONFIGURATION_FIELDS
        if field not in benchmark_config
    ]
    if missing:
        raise ValueError(
            "benchmark_config is missing required fields: " + ", ".join(missing)
        )

    for field in (
        "family",
        "size",
        "base_repo",
        "confidence_method",
        "backend",
        "device",
        "dtype",
    ):
        value = benchmark_config[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"benchmark_config.{field} must be a non-empty string.")
    base_revision = benchmark_config["base_revision"]
    if not isinstance(base_revision, str) or not _COMMIT_SHA.fullmatch(base_revision):
        raise ValueError(
            "benchmark_config.base_revision must be an immutable 40-character "
            "commit SHA."
        )

    checkpoint = benchmark_config["checkpoint"]
    if checkpoint is not None:
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                "benchmark_config.checkpoint must be null or a mapping with a "
                "content sha256."
            )
        digest = checkpoint.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(
                "benchmark_config.checkpoint.sha256 must be a 64-character digest."
            )

    processor = benchmark_config["processor"]
    if not isinstance(processor, Mapping) or not processor:
        raise ValueError(
            "benchmark_config.processor must be a mapping that identifies an "
            "immutable processor."
        )
    processor_source = processor.get("source")
    processor_revision = processor.get("revision")
    processor_digest = processor.get("sha256")
    if not isinstance(processor_source, str) or not processor_source.strip():
        raise ValueError("benchmark_config.processor.source must be non-empty.")
    immutable_processor = (
        isinstance(processor_revision, str)
        and _COMMIT_SHA.fullmatch(processor_revision)
    ) or (isinstance(processor_digest, str) and _SHA256.fullmatch(processor_digest))
    if not immutable_processor:
        raise ValueError(
            "benchmark_config.processor requires an immutable 40-character "
            "revision or 64-character sha256."
        )
    class_names = benchmark_config["class_names"]
    if (
        not isinstance(class_names, (list, tuple))
        or not class_names
        or any(not isinstance(name, str) or not name.strip() for name in class_names)
    ):
        raise ValueError(
            "benchmark_config.class_names must be an ordered, non-empty sequence "
            "of non-empty strings."
        )
    generation_kwargs = benchmark_config["generation_kwargs"]
    if not isinstance(generation_kwargs, Mapping):
        raise TypeError("benchmark_config.generation_kwargs must be a mapping.")
    missing_generation = [
        field for field in _REQUIRED_GENERATION_FIELDS if field not in generation_kwargs
    ]
    if missing_generation:
        raise ValueError(
            "benchmark_config.generation_kwargs is missing: "
            + ", ".join(missing_generation)
        )
    max_new_tokens = generation_kwargs["max_new_tokens"]
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("generation_kwargs.max_new_tokens must be a positive integer.")
    if generation_kwargs["do_sample"] is not False:
        raise ValueError("generation_kwargs.do_sample must be false for this gate.")
    num_beams = generation_kwargs["num_beams"]
    if isinstance(num_beams, bool) or num_beams != 1:
        raise ValueError("generation_kwargs.num_beams must equal 1 for this gate.")
    repetition_penalty = generation_kwargs["repetition_penalty"]
    if isinstance(repetition_penalty, bool):
        raise TypeError("generation_kwargs.repetition_penalty must be numeric.")
    try:
        repetition_penalty = float(repetition_penalty)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "generation_kwargs.repetition_penalty must be numeric."
        ) from exc
    if not math.isfinite(repetition_penalty) or repetition_penalty <= 0.0:
        raise ValueError(
            "generation_kwargs.repetition_penalty must be finite and positive."
        )
    confidence_evaluation = benchmark_config["confidence_evaluation"]
    if not isinstance(confidence_evaluation, Mapping):
        raise TypeError("benchmark_config.confidence_evaluation must be a mapping.")
    required_confidence_evaluation = (
        "iou_threshold",
        "default_conf",
        "fallback_score",
        "calibration_bins",
        "binning",
        "population",
        "matching",
    )
    missing_confidence = [
        field
        for field in required_confidence_evaluation
        if field not in confidence_evaluation
    ]
    if missing_confidence:
        raise ValueError(
            "benchmark_config.confidence_evaluation is missing: "
            + ", ".join(missing_confidence)
        )
    confidence_iou = _probability(
        confidence_evaluation["iou_threshold"],
        "confidence_evaluation.iou_threshold",
    )
    if confidence_iou <= 0.0:
        raise ValueError("confidence_evaluation.iou_threshold must be positive.")
    _probability(
        confidence_evaluation["default_conf"],
        "confidence_evaluation.default_conf",
    )
    _probability(
        confidence_evaluation["fallback_score"],
        "confidence_evaluation.fallback_score",
    )
    calibration_bins = confidence_evaluation["calibration_bins"]
    if (
        isinstance(calibration_bins, bool)
        or not isinstance(calibration_bins, int)
        or not 1 <= calibration_bins <= 1000
    ):
        raise ValueError(
            "confidence_evaluation.calibration_bins must be an integer in [1, 1000]."
        )
    expected_strings = {
        "binning": "uniform_left_closed_v1",
        "population": "scored_postprocessed_predictions",
        "matching": "class_aware_max_cardinality_iou_v1",
    }
    for field, expected in expected_strings.items():
        if confidence_evaluation[field] != expected:
            raise ValueError(f"confidence_evaluation.{field} must equal {expected!r}.")
    evaluation = benchmark_config["evaluation"]
    if not isinstance(evaluation, Mapping):
        raise TypeError("benchmark_config.evaluation must be a mapping.")
    missing_evaluation = [
        field
        for field in ("max_det", "faster_coco_eval", "imgsz", "backend")
        if field not in evaluation
    ]
    if missing_evaluation:
        raise ValueError(
            "benchmark_config.evaluation is missing: " + ", ".join(missing_evaluation)
        )
    max_det = evaluation["max_det"]
    if isinstance(max_det, bool) or not isinstance(max_det, int) or max_det <= 0:
        raise ValueError("benchmark_config.evaluation.max_det must be positive.")
    if not isinstance(evaluation["faster_coco_eval"], bool):
        raise TypeError("benchmark_config.evaluation.faster_coco_eval must be boolean.")
    if "save_plots" in evaluation and not isinstance(evaluation["save_plots"], bool):
        raise TypeError("benchmark_config.evaluation.save_plots must be boolean.")
    evaluation_backend = evaluation["backend"]
    if not isinstance(evaluation_backend, str) or not evaluation_backend.strip():
        raise ValueError(
            "benchmark_config.evaluation.backend must be a non-empty string."
        )
    seed = benchmark_config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("benchmark_config.seed must be an integer.")
    hardware = benchmark_config["hardware"]
    if not isinstance(hardware, Mapping) or not hardware:
        raise ValueError("benchmark_config.hardware must be a non-empty mapping.")
    software = benchmark_config["software"]
    if not isinstance(software, Mapping) or not software:
        raise ValueError("benchmark_config.software must be a non-empty mapping.")
    missing_software = [
        field for field in _REQUIRED_SOFTWARE_FIELDS if field not in software
    ]
    if missing_software:
        raise ValueError(
            "benchmark_config.software is missing: " + ", ".join(missing_software)
        )
    for field in _REQUIRED_SOFTWARE_FIELDS:
        value = software[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"benchmark_config.software.{field} must be a non-empty string."
            )
    return dict(benchmark_config)


def benchmark_manifest_hash(
    prompt: str, dataset_manifest: Any, benchmark_config: Mapping[str, Any]
) -> str:
    """Hash the complete intended benchmark identity canonically.

    Mapping insertion order and tuples versus lists do not affect the digest.
    Sequence order remains significant because class order and evaluation order
    are part of the benchmark contract. ``benchmark_config`` is mandatory so a
    repeat cannot compare different model, processor, generation, class-order,
    seed, runtime, or software configurations under the same prompt/dataset hash.
    """

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")
    configuration = _validated_benchmark_config(benchmark_config)
    return _payload_hash(
        {
            "schema": _MANIFEST_SCHEMA,
            "prompt": prompt,
            "dataset": dataset_manifest,
            "configuration": configuration,
        }
    )


def _configuration_hash(benchmark_config: Mapping[str, Any]) -> str:
    return _payload_hash(
        {
            "schema": _CONFIGURATION_SCHEMA,
            "configuration": _validated_benchmark_config(benchmark_config),
        }
    )


def _prediction_structure_hash(predictions: Sequence[VLMDetection]) -> str:
    return _payload_hash(
        {
            "schema": _PREDICTION_SCHEMA,
            "predictions": [
                {
                    "image_id": prediction.image_id,
                    "class_id": prediction.class_id,
                    "xyxy": prediction.xyxy,
                }
                for prediction in predictions
            ],
        }
    )


def _normalize_evaluator_metrics(
    metrics: Optional[Mapping[str, Any]],
) -> tuple[tuple[str, float], ...]:
    if metrics is None:
        return ()
    if not isinstance(metrics, Mapping):
        raise TypeError("evaluator_metrics must be null or a mapping.")
    missing = [name for name in _EVALUATOR_METRIC_NAMES if name not in metrics]
    extras = [name for name in metrics if name not in _EVALUATOR_METRIC_NAMES]
    if missing or extras:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extras:
            detail.append("unsupported " + ", ".join(sorted(map(str, extras))))
        raise ValueError("evaluator_metrics has " + "; ".join(detail) + ".")
    normalized = []
    for name in _EVALUATOR_METRIC_NAMES:
        value = _probability(metrics[name], f"evaluator_metrics.{name}")
        normalized.append((name, value))
    return tuple(normalized)


def build_confidence_run(
    predictions: Sequence[VLMDetection],
    ground_truth: Sequence[VLMDetection],
    *,
    prompt: str,
    dataset_manifest: Any,
    benchmark_config: Mapping[str, Any],
    generation_manifest: Any,
    evaluator_metrics: Optional[Mapping[str, Any]] = None,
    iou_threshold: float = 0.5,
    default_conf: float = 0.25,
    fallback_score: float = 1.0,
) -> ConfidenceRun:
    """Evaluate one fixed prediction set without enabling model validation."""

    predictions = tuple(predictions)
    ground_truth = tuple(ground_truth)
    validated_config = _validated_benchmark_config(benchmark_config)
    confidence_evaluation = validated_config["confidence_evaluation"]
    normalized_iou = _probability(iou_threshold, "iou_threshold")
    normalized_default = _probability(default_conf, "default_conf")
    normalized_fallback = _probability(fallback_score, "fallback_score")
    configured_values = {
        "iou_threshold": normalized_iou,
        "default_conf": normalized_default,
        "fallback_score": normalized_fallback,
    }
    for field, actual in configured_values.items():
        configured = float(confidence_evaluation[field])
        if configured != actual:
            raise ValueError(
                f"{field}={actual} does not match benchmark_config "
                f"confidence_evaluation.{field}={configured}."
            )
    matches = match_detections(
        predictions,
        ground_truth,
        iou_threshold=normalized_iou,
    )
    scored_pairs = [
        (prediction.score, matched)
        for prediction, matched in zip(predictions, matches)
        if prediction.score is not None
    ]
    scored_scores = [score for score, _ in scored_pairs]
    scored_labels = [matched for _, matched in scored_pairs]
    scores = tuple(prediction.score for prediction in predictions)
    diagnostics = confidence_diagnostics(
        scores,
        matches,
        default_conf=normalized_default,
        fallback_score=normalized_fallback,
    )
    calibration = calibration_diagnostics(
        scores,
        matches,
        n_bins=int(confidence_evaluation["calibration_bins"]),
    )
    return ConfidenceRun(
        manifest_hash=benchmark_manifest_hash(
            prompt, dataset_manifest, benchmark_config
        ),
        configuration_hash=_configuration_hash(benchmark_config),
        generation_hash=_payload_hash(
            {
                "schema": _GENERATION_SCHEMA,
                "generations": generation_manifest,
            }
        ),
        prediction_structure_hash=_prediction_structure_hash(predictions),
        iou_threshold=normalized_iou,
        default_conf=diagnostics.default_conf,
        fallback_score=diagnostics.fallback_score,
        scores=scores,
        matches=matches,
        auroc=tie_aware_auroc(scored_scores, scored_labels),
        ranking_ap=binary_ranking_ap(scored_scores, scored_labels),
        diagnostics=diagnostics,
        calibration=calibration,
        evaluator_metrics=_normalize_evaluator_metrics(evaluator_metrics),
    )


def _optional_delta(first: Optional[float], second: Optional[float]) -> Optional[float]:
    if first is None and second is None:
        return 0.0
    if first is None or second is None:
        return None
    return abs(first - second)


def compare_repeats(
    first: ConfidenceRun,
    second: ConfidenceRun,
    *,
    score_atol: float = 0.0,
    metric_atol: float = 0.0,
) -> RepeatComparison:
    """Compare two runs and state whether their observable result reproduced."""

    if not isinstance(first, ConfidenceRun) or not isinstance(second, ConfidenceRun):
        raise TypeError("first and second must be ConfidenceRun values.")

    def tolerance(value: Any, name: str) -> float:
        if isinstance(value, (bool, str, bytes)):
            raise TypeError(f"{name} must be a finite non-negative number.")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a finite non-negative number.") from exc
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return result

    score_tolerance = tolerance(score_atol, "score_atol")
    metric_tolerance = tolerance(metric_atol, "metric_atol")

    same_configuration = (
        first.configuration_hash == second.configuration_hash
        and first.iou_threshold == second.iou_threshold
        and first.default_conf == second.default_conf
        and first.fallback_score == second.fallback_score
    )
    same_availability = len(first.scores) == len(second.scores) and all(
        (left is None) == (right is None)
        for left, right in zip(first.scores, second.scores)
    )
    if same_availability:
        deltas = [
            abs(left - right)
            for left, right in zip(first.scores, second.scores)
            if left is not None and right is not None
        ]
        max_score_delta: Optional[float] = max(deltas, default=0.0)
        scores_within_tolerance = max_score_delta <= score_tolerance
    else:
        max_score_delta = None
        scores_within_tolerance = False

    same_manifest = first.manifest_hash == second.manifest_hash
    same_generation = first.generation_hash == second.generation_hash
    same_structure = first.prediction_structure_hash == second.prediction_structure_hash
    same_matches = first.matches == second.matches
    same_diagnostics = first.diagnostics == second.diagnostics
    auroc_delta = _optional_delta(first.auroc, second.auroc)
    ranking_ap_delta = _optional_delta(first.ranking_ap, second.ranking_ap)
    brier_delta = _optional_delta(
        first.calibration.brier_score, second.calibration.brier_score
    )
    ece_delta = _optional_delta(
        first.calibration.expected_calibration_error,
        second.calibration.expected_calibration_error,
    )
    maximum_calibration_error_delta = _optional_delta(
        first.calibration.maximum_calibration_error,
        second.calibration.maximum_calibration_error,
    )
    first_bins = first.calibration.bins
    second_bins = second.calibration.bins
    same_bin_assignments = len(first_bins) == len(second_bins) and all(
        (
            left.index,
            left.lower,
            left.upper,
            left.count,
            left.correct,
        )
        == (
            right.index,
            right.lower,
            right.upper,
            right.count,
            right.correct,
        )
        for left, right in zip(first_bins, second_bins)
    )
    if same_bin_assignments:
        bin_deltas = []
        bins_comparable = True
        for left, right in zip(first_bins, second_bins):
            for field in (
                "mean_confidence",
                "empirical_accuracy",
                "absolute_gap",
            ):
                delta = _optional_delta(getattr(left, field), getattr(right, field))
                if delta is None:
                    bins_comparable = False
                else:
                    bin_deltas.append(delta)
        max_bin_delta: Optional[float] = (
            max(bin_deltas, default=0.0) if bins_comparable else None
        )
        calibration_bins_within_tolerance = (
            max_bin_delta is not None and max_bin_delta <= metric_tolerance
        )
    else:
        max_bin_delta = None
        calibration_bins_within_tolerance = False
    first_evaluator = dict(first.evaluator_metrics)
    second_evaluator = dict(second.evaluator_metrics)
    same_evaluator_keys = first_evaluator.keys() == second_evaluator.keys()
    if same_evaluator_keys:
        evaluator_deltas = [
            abs(first_evaluator[name] - second_evaluator[name])
            for name in first_evaluator
        ]
        if first_evaluator:
            for suffix in ("mAP50-95", "mAP50"):
                first_delta = (
                    first_evaluator[f"candidate_{suffix}"]
                    - first_evaluator[f"constant_{suffix}"]
                )
                second_delta = (
                    second_evaluator[f"candidate_{suffix}"]
                    - second_evaluator[f"constant_{suffix}"]
                )
                evaluator_deltas.append(abs(first_delta - second_delta))
        max_evaluator_delta: Optional[float] = max(evaluator_deltas, default=0.0)
        evaluator_metrics_within_tolerance = max_evaluator_delta <= metric_tolerance
    else:
        max_evaluator_delta = None
        evaluator_metrics_within_tolerance = False
    metrics_within_tolerance = (
        auroc_delta is not None
        and ranking_ap_delta is not None
        and brier_delta is not None
        and ece_delta is not None
        and maximum_calibration_error_delta is not None
        and auroc_delta <= metric_tolerance
        and ranking_ap_delta <= metric_tolerance
        and brier_delta <= metric_tolerance
        and ece_delta <= metric_tolerance
        and maximum_calibration_error_delta <= metric_tolerance
        and calibration_bins_within_tolerance
        and evaluator_metrics_within_tolerance
    )
    reproducible = all(
        (
            same_manifest,
            same_configuration,
            same_generation,
            same_structure,
            same_matches,
            same_availability,
            scores_within_tolerance,
            metrics_within_tolerance,
            same_bin_assignments,
            calibration_bins_within_tolerance,
            same_evaluator_keys,
            evaluator_metrics_within_tolerance,
            same_diagnostics,
        )
    )
    return RepeatComparison(
        same_manifest=same_manifest,
        same_configuration=same_configuration,
        same_generation=same_generation,
        same_prediction_structure=same_structure,
        same_matches=same_matches,
        same_score_availability=same_availability,
        scores_within_tolerance=scores_within_tolerance,
        metrics_within_tolerance=metrics_within_tolerance,
        same_calibration_bin_assignments=same_bin_assignments,
        calibration_bins_within_tolerance=calibration_bins_within_tolerance,
        same_evaluator_metric_keys=same_evaluator_keys,
        evaluator_metrics_within_tolerance=evaluator_metrics_within_tolerance,
        same_diagnostics=same_diagnostics,
        max_abs_score_delta=max_score_delta,
        max_abs_calibration_bin_delta=max_bin_delta,
        max_abs_evaluator_metric_delta=max_evaluator_delta,
        auroc_delta=auroc_delta,
        ranking_ap_delta=ranking_ap_delta,
        brier_score_delta=brier_delta,
        expected_calibration_error_delta=ece_delta,
        maximum_calibration_error_delta=maximum_calibration_error_delta,
        reproducible=reproducible,
    )
