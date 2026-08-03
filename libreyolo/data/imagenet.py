"""Canonical ImageNet-1k class names and WordNet ids.

The list in ``imagenet1k.json`` is index-aligned to the standard wnid-sorted
class ordering used by the timm-derived classification checkpoints, so index
``i`` is the human-readable label for class id ``i`` (for example ``258`` maps
to ``"Samoyed"``).

Classification checkpoints embed this mapping in their ``names`` metadata so
that prediction results expose readable labels: ``result.names[probs.top1]``
returns the class name while ``probs.top1`` stays the integer class id.

``imagenet1k_synsets.txt`` is the unmodified Apache-2.0 timm index from
``timm/data/_info/imagenet_synsets.txt`` at commit
``e98c05a5a15e81188ec62dd5380b8f5c3251075a`` (SHA-256
``70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15``).
It lets validation map standard WNID-named ImageFolder directories to the
human-readable checkpoint labels without changing either public surface.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

_NAMES_PATH = Path(__file__).resolve().parent / "imagenet1k.json"
_SYNSETS_PATH = Path(__file__).resolve().parent / "imagenet1k_synsets.txt"

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


@lru_cache(maxsize=1)
def imagenet1k_synset_list() -> List[str]:
    """Return the 1000 ImageNet-1k WNIDs in classifier-index order."""
    with _SYNSETS_PATH.open("r", encoding="utf-8") as stream:
        synsets = [line.strip() for line in stream if line.strip()]
    if len(synsets) != IMAGENET1K_NUM_CLASSES or len(set(synsets)) != len(synsets):
        raise ValueError(
            "Expected 1000 unique ImageNet-1k synsets, "
            f"got {len(synsets)} entries and {len(set(synsets))} unique values."
        )
    return synsets


def imagenet1k_synset_to_index() -> Dict[str, int]:
    """Return the canonical ImageNet-1k mapping ``{wnid: class_id}``."""
    return {synset: index for index, synset in enumerate(imagenet1k_synset_list())}
