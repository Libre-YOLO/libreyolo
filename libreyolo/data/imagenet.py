"""Canonical ImageNet-1k class names.

The list in ``imagenet1k.json`` is index-aligned to the standard wnid-sorted
class ordering used by the timm-derived classification checkpoints, so index
``i`` is the human-readable label for class id ``i`` (for example ``258`` maps
to ``"Samoyed"``).

Classification checkpoints embed this mapping in their ``names`` metadata so
that prediction results expose readable labels: ``result.names[probs.top1]``
returns the class name while ``probs.top1`` stays the integer class id.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

_NAMES_PATH = Path(__file__).resolve().parent / "imagenet1k.json"

IMAGENET1K_NUM_CLASSES = 1000


@lru_cache(maxsize=1)
def imagenet1k_class_list() -> List[str]:
    """Return the 1000 ImageNet-1k class names in class-index order."""
    with _NAMES_PATH.open("r", encoding="utf-8") as f:
        names = json.load(f)
    if len(names) != IMAGENET1K_NUM_CLASSES:
        raise ValueError(
            f"Expected {IMAGENET1K_NUM_CLASSES} ImageNet-1k names, got {len(names)}."
        )
    return list(names)


def imagenet1k_names() -> Dict[int, str]:
    """Return the ImageNet-1k label map ``{index: name}`` for ids ``0..999``.

    A fresh dict is returned on each call so callers may mutate it freely.
    """
    return dict(enumerate(imagenet1k_class_list()))
