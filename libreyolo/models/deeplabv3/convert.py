"""Strict recognition and conversion of torchvision DeepLabv3 checkpoints."""

from __future__ import annotations

from typing import Optional

import torch


_COMMON_PREFIXES = ("module.", "model.")
_AUX_PREFIX = "aux_classifier."
_RUNTIME_FINGERPRINT = {
    "classifier.0.convs.0.0.weight",
    "classifier.0.convs.1.0.weight",
    "classifier.0.convs.4.1.weight",
    "classifier.0.project.0.weight",
    "classifier.4.weight",
}
_UPSTREAM_AUX_FINGERPRINT = {
    "aux_classifier.0.weight",
    "aux_classifier.4.weight",
}


def _strip_known_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _COMMON_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def _normalized_items(state_dict: dict):
    for raw_key, value in state_dict.items():
        yield _strip_known_prefix(str(raw_key)), value


def is_upstream_deeplabv3_state_dict(state_dict: dict) -> bool:
    """Return whether *state_dict* has the official inference and aux layout.

    The auxiliary FCN head is deliberately part of the fingerprint. Native
    LibreYOLO checkpoints omit it, and broad ASPP-only recognition could claim
    unrelated semantic families during global runtime auto-conversion.
    """
    keys = {key for key, _value in _normalized_items(state_dict)}
    has_backbone = (
        "backbone.conv1.weight" in keys or "backbone.0.0.weight" in keys
    )
    return (
        has_backbone
        and _RUNTIME_FINGERPRINT.issubset(keys)
        and _UPSTREAM_AUX_FINGERPRINT.issubset(keys)
    )


def convert_upstream_deeplabv3_state_dict(
    state_dict: dict,
) -> Optional[dict[str, torch.Tensor]]:
    """Remove only the training-only auxiliary head from recognized weights."""
    if not is_upstream_deeplabv3_state_dict(state_dict):
        return None
    return {
        key: value
        for key, value in _normalized_items(state_dict)
        if not key.startswith(_AUX_PREFIX)
    }


__all__ = [
    "convert_upstream_deeplabv3_state_dict",
    "is_upstream_deeplabv3_state_dict",
]
