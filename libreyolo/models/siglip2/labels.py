"""Default label set + prompt templates for LibreSigLIP2 zero-shot classification.

The shipped ImageNet-1k class names are the **concise** WordNet-lemma names
(e.g. ``"English springer"``, ``"chain saw"``), paired with the standard
index-ordered WordNet ids so validation on a wnid-foldered ImageFolder can
humanize folder names into prompt-able labels. Templates are ``{}``-format
strings, identical to LibreCLIP's ensemble.

Why concise names (and not LibreCLIP's descriptive open_clip list): SigLIP's
sigmoid scoring is more sensitive to verbose class phrasings than CLIP's. The
descriptive open_clip names (e.g. ``"English Springer Spaniel"``) act as
false-positive magnets for SigLIP and drop imagenette top-1 below the release
bar, while the concise names reproduce the published SigLIP zero-shot numbers.
LibreCLIP keeps its own descriptive list unchanged; the two families are
decoupled here on purpose.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

_DATA_DIR = Path(__file__).parent / "data"

# SigLIP's canonical zero-shot prompt; the 80-template ensemble is opt-in.
DEFAULT_TEMPLATES: List[str] = ["a photo of a {}."]


@lru_cache()
def _load_imagenet() -> dict:
    with open(_DATA_DIR / "imagenet1k.json", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache()
def _load_templates() -> tuple:
    with open(_DATA_DIR / "imagenet_templates.json", encoding="utf-8") as fh:
        return tuple(json.load(fh))


# The accessors below return a *fresh* copy each call (the file reads are cached).


def imagenet1k_classnames() -> List[str]:
    """The 1000 concise ImageNet-1k class names (index order)."""
    return list(_load_imagenet()["classnames"])


def imagenet1k_wnids() -> List[str]:
    """The 1000 ImageNet-1k WordNet ids (same index order as the class names)."""
    return list(_load_imagenet()["wnids"])


def wnid_to_classname() -> Dict[str, str]:
    data = _load_imagenet()
    return dict(zip(data["wnids"], data["classnames"]))


def openai_imagenet_templates() -> List[str]:
    """The 80-prompt OpenAI ImageNet template ensemble (``{}``-format strings)."""
    return list(_load_templates())


def humanize_labels(names: List[str]) -> List[str]:
    """Map WordNet-id folder names to readable class names; pass others through.

    Zero-shot accuracy is prompt-sensitive, so a folder literally named
    ``n01440764`` must become ``"tench"`` before it is fed to the text tower.
    Names that are not known wnids are returned unchanged.
    """
    mapping = wnid_to_classname()
    return [mapping.get(name, name) for name in names]


__all__ = [
    "DEFAULT_TEMPLATES",
    "humanize_labels",
    "imagenet1k_classnames",
    "imagenet1k_wnids",
    "openai_imagenet_templates",
    "wnid_to_classname",
]
