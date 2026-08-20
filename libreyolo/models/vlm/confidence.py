"""Token-to-detection confidence helpers for autoregressive VLM output.

The helpers in this module are deliberately model-agnostic. They align selected
token log-probabilities with the decoded text, locate the label and coordinate
values for each parsed detection, and reduce those values to one ranking score
per box. No score is treated as calibrated probability.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

from .parsing import locate_detection_spans

__all__ = [
    "TokenSpan",
    "decode_token_spans",
    "score_detection_items",
]


@dataclass(frozen=True)
class TokenSpan:
    """Character range and selected-token log-probability in decoded text."""

    start: int
    end: int
    logprob: float


_NUMBER = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _single_row(values, name: str) -> list:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    values = list(values)
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError(f"{name} must contain exactly one generated sequence.")
        values = list(values[0])
    return values


def decode_token_spans(
    token_ids,
    token_logprobs,
    decode: Callable[[Sequence[int]], str],
) -> tuple[str, list[TokenSpan]]:
    """Decode one generated sequence and align each token to character offsets.

    Most tokenizers compose when each token is decoded separately, which gives a
    linear-time path. Tokenizers whose whitespace handling is context-dependent
    fall back to decoding prefixes. If prefix decoding rewrites earlier text, the
    function returns the full text with no spans so callers can safely retain
    their constant-score fallback instead of attaching scores to the wrong box.
    """

    ids = [int(value) for value in _single_row(token_ids, "token_ids")]
    logprobs = [
        float(value) for value in _single_row(token_logprobs, "token_logprobs")
    ]
    if len(ids) != len(logprobs):
        raise ValueError(
            "token_ids and token_logprobs must have the same generated length."
        )

    full_text = decode(ids)
    pieces = [decode([token_id]) for token_id in ids]
    if "".join(pieces) == full_text:
        spans = []
        cursor = 0
        for piece, logprob in zip(pieces, logprobs):
            end = cursor + len(piece)
            spans.append(TokenSpan(cursor, end, logprob))
            cursor = end
        return full_text, spans

    spans = []
    previous = ""
    for index, logprob in enumerate(logprobs, 1):
        current = decode(ids[:index])
        if not current.startswith(previous):
            return full_text, []
        spans.append(TokenSpan(len(previous), len(current), logprob))
        previous = current
    if previous != full_text:
        return full_text, []
    return full_text, spans


def _quoted_end(blob: str, start: int) -> Optional[int]:
    if start >= len(blob) or blob[start] not in {'"', "'"}:
        return None
    quote = blob[start]
    escaped = False
    for index in range(start + 1, len(blob)):
        char = blob[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote:
            return index + 1
    return None


def _value_end(blob: str, start: int) -> Optional[int]:
    if start >= len(blob):
        return None
    if blob[start] in {'"', "'"}:
        return _quoted_end(blob, start)
    pairs = {"[": "]", "{": "}"}
    if blob[start] in pairs:
        stack = [pairs[blob[start]]]
        index = start + 1
        while index < len(blob):
            char = blob[index]
            if char in {'"', "'"}:
                string_end = _quoted_end(blob, index)
                if string_end is None:
                    return None
                index = string_end
                continue
            if char in pairs:
                stack.append(pairs[char])
            elif char in "]}":
                if not stack or char != stack.pop():
                    return None
                if not stack:
                    return index + 1
            index += 1
        return None
    index = start
    while index < len(blob) and blob[index] not in ",}":
        index += 1
    return index


def _object_members(blob: str) -> dict[str, list[tuple[int, int]]]:
    """Return top-level object value spans, or an empty map when ambiguous."""

    if len(blob) < 2 or blob[0] != "{" or blob[-1] != "}":
        return {}
    members: dict[str, list[tuple[int, int]]] = {}
    index = 1
    while index < len(blob) - 1:
        while index < len(blob) - 1 and (blob[index].isspace() or blob[index] == ","):
            index += 1
        if index >= len(blob) - 1:
            break
        key_end = _quoted_end(blob, index)
        if key_end is None:
            return {}
        key = blob[index + 1 : key_end - 1]
        # The scoring keys are ASCII literals. Reject escaped keys rather than
        # guessing how their decoded spelling maps back to source characters.
        if "\\" in key:
            return {}
        index = key_end
        while index < len(blob) - 1 and blob[index].isspace():
            index += 1
        if index >= len(blob) - 1 or blob[index] != ":":
            return {}
        index += 1
        while index < len(blob) - 1 and blob[index].isspace():
            index += 1
        value_start = index
        value_end = _value_end(blob, value_start)
        if value_end is None:
            return {}
        members.setdefault(key, []).append((value_start, value_end))
        index = value_end
        while index < len(blob) - 1 and blob[index].isspace():
            index += 1
        if index < len(blob) - 1 and blob[index] not in ",}":
            return {}
    return members


def _unique_member(
    members: dict[str, list[tuple[int, int]]], key: str
) -> Optional[tuple[int, int]]:
    regions = members.get(key, [])
    return regions[0] if len(regions) == 1 else None


def _string_value_region(
    blob: str,
    members: dict[str, list[tuple[int, int]]],
    key: str,
    offset: int,
) -> list[tuple[int, int]]:
    region = _unique_member(members, key)
    if region is None:
        return []
    start, end = region
    if blob[start] not in {'"', "'"} or end <= start + 1:
        return []
    return [(offset + start + 1, offset + end - 1)]


def _number_value_regions(
    blob: str,
    members: dict[str, list[tuple[int, int]]],
    key: str,
    offset: int,
) -> list[tuple[int, int]]:
    region = _unique_member(members, key)
    if region is None:
        return []
    start, end = region
    if blob[start] != "[" or blob[end - 1] != "]":
        return []
    return [
        (offset + match.start(), offset + match.end())
        for match in _NUMBER.finditer(blob, start + 1, end - 1)
    ]


def _mean_logprob(
    regions: Sequence[tuple[int, int]], token_spans: Sequence[TokenSpan]
) -> Optional[float]:
    selected = {
        index
        for index, token in enumerate(token_spans)
        if token.end > token.start
        and any(token.start < end and token.end > start for start, end in regions)
    }
    values = [token_spans[index].logprob for index in sorted(selected)]
    if not values or any(not math.isfinite(value) for value in values):
        return None
    return sum(values) / len(values)


def score_detection_items(
    text: str,
    items: Sequence[dict],
    token_spans: Sequence[TokenSpan],
    *,
    bbox_key: str,
) -> list[Optional[float]]:
    """Return one token-logprob ranking score for every parsed detection.

    Coordinate-number tokens and label-value tokens are reduced separately by
    their mean log-probability, then given equal weight. This prevents a label's
    score from being drowned out when four coordinates each split into several
    tokens. Both components are required. Missing or ambiguous keys return
    ``None`` so the caller can safely fall back instead of ranking a box using a
    different source value.
    """

    object_spans = locate_detection_spans(text, items)
    scores: list[Optional[float]] = []
    for item, object_span in zip(items, object_spans):
        if object_span is None:
            scores.append(None)
            continue
        object_start, object_end = object_span
        blob = text[object_start:object_end]
        members = _object_members(blob)
        label_regions = _string_value_region(blob, members, "label", object_start)

        # Mirror ``build_detection_dict`` exactly: a present, non-null preferred
        # key wins even when its source mapping is ambiguous. Only a null/missing
        # preferred value permits the builder's first-present alias fallback.
        coord_key = bbox_key if item.get(bbox_key) is not None else None
        if coord_key is None:
            for alias in ("bbox", "bbox_2d"):
                if alias != bbox_key and alias in item:
                    coord_key = alias
                    break
        coord_regions = (
            _number_value_regions(blob, members, coord_key, object_start)
            if coord_key is not None
            else []
        )

        label_logprob = _mean_logprob(label_regions, token_spans)
        coord_logprob = _mean_logprob(coord_regions, token_spans)
        if label_logprob is None or coord_logprob is None:
            scores.append(None)
            continue
        # A normalized geometric mean stays in [0, 1]. Log-probabilities should
        # be <= 0; clamp tiny positive numerical noise before exponentiating.
        score = math.exp(min(0.0, (label_logprob + coord_logprob) / 2.0))
        scores.append(score)
    return scores
