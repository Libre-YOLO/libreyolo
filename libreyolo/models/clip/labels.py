"""Default label set + prompt templates for LibreCLIP zero-shot classification.

The shipped ImageNet-1k class names are the canonical open_clip
``IMAGENET_CLASSNAMES`` (the exact wording that reproduces published zero-shot
top-1), paired with the standard index-ordered WordNet ids so validation on a
wnid-foldered ImageFolder (e.g. ImageNet, imagenette) can humanize folder names
into prompt-able labels. Templates are stored as ``{}``-format strings.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

_DATA_DIR = Path(__file__).parent / "data"

# CLIP's default single prompt; the 80-template ensemble is opt-in.
DEFAULT_TEMPLATES: List[str] = ["a photo of a {}."]


@lru_cache()
def _load_imagenet() -> dict:
    with open(_DATA_DIR / "imagenet1k.json", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache()
def imagenet1k_classnames() -> List[str]:
    """The 1000 canonical ImageNet-1k class names (index order)."""
    return list(_load_imagenet()["classnames"])


@lru_cache()
def imagenet1k_wnids() -> List[str]:
    """The 1000 ImageNet-1k WordNet ids (same index order as the class names)."""
    return list(_load_imagenet()["wnids"])


@lru_cache()
def wnid_to_classname() -> Dict[str, str]:
    data = _load_imagenet()
    return dict(zip(data["wnids"], data["classnames"]))


@lru_cache()
def openai_imagenet_templates() -> List[str]:
    """The 80-prompt OpenAI ImageNet template ensemble (``{}``-format strings)."""
    with open(_DATA_DIR / "imagenet_templates.json", encoding="utf-8") as fh:
        return list(json.load(fh))


def humanize_labels(names: List[str]) -> List[str]:
    """Map WordNet-id folder names to readable class names; pass others through.

    Zero-shot accuracy is prompt-sensitive, so a folder literally named
    ``n01440764`` must become ``"tench"`` before it is fed to the text tower.
    Names that are not known wnids are returned unchanged.
    """
    mapping = wnid_to_classname()
    return [mapping.get(name, name) for name in names]
