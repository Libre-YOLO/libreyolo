"""Open-vocabulary detector tier.

``LibreOpenVocab(...)`` is a sibling factory to ``LibreSAM`` and ``LibreVLM``.
It loads discriminative text-conditioned detectors from Hugging Face snapshots
and returns standard detection ``Results``.
"""

from __future__ import annotations

from typing import Dict, Tuple, Type

from .base import LibreOpenVocabDetector
from .grounding_dino import LibreGroundingDINO
from .omdet_turbo import LibreOMDetTurbo
from .owlv2 import LibreOWLv2

_ALIASES: Dict[str, Tuple[Type[LibreOpenVocabDetector], str]] = {
    "grounding-dino": (LibreGroundingDINO, "t"),
    "groundingdino": (LibreGroundingDINO, "t"),
    "grounding-dino-tiny": (LibreGroundingDINO, "t"),
    "groundingdino-tiny": (LibreGroundingDINO, "t"),
    "grounding-dino-t": (LibreGroundingDINO, "t"),
    "groundingdino-t": (LibreGroundingDINO, "t"),
    "grounding-dino-base": (LibreGroundingDINO, "b"),
    "groundingdino-base": (LibreGroundingDINO, "b"),
    "grounding-dino-b": (LibreGroundingDINO, "b"),
    "groundingdino-b": (LibreGroundingDINO, "b"),
    "owlv2": (LibreOWLv2, "b16"),
    "owl-v2": (LibreOWLv2, "b16"),
    "owlv2-base": (LibreOWLv2, "b16"),
    "owl-v2-base": (LibreOWLv2, "b16"),
    "owlv2-b16": (LibreOWLv2, "b16"),
    "owl-v2-b16": (LibreOWLv2, "b16"),
    "owlv2-large": (LibreOWLv2, "l14"),
    "owl-v2-large": (LibreOWLv2, "l14"),
    "owlv2-l14": (LibreOWLv2, "l14"),
    "owl-v2-l14": (LibreOWLv2, "l14"),
    "omdet-turbo": (LibreOMDetTurbo, "t"),
    "omdet": (LibreOMDetTurbo, "t"),
    "omdetturbo": (LibreOMDetTurbo, "t"),
    "omdet-turbo-tiny": (LibreOMDetTurbo, "t"),
    "omdet-turbo-swin-tiny": (LibreOMDetTurbo, "t"),
    "omdet-turbo-t": (LibreOMDetTurbo, "t"),
}

_DEFAULT_MODEL = "grounding-dino-tiny"


def LibreOpenVocab(
    model: str = _DEFAULT_MODEL, **kwargs
) -> LibreOpenVocabDetector:
    """Load an open-vocabulary detector by alias."""
    # Underscores fold to hyphens so the family-qualified names the inventory
    # and `libreyolo models` print (omdet_turbo-t, grounding_dino-t) load as
    # given; the aliases themselves are hyphenated.
    key = str(model).strip().lower().replace("_", "-")
    match = _ALIASES.get(key)
    if match is None:
        raise ValueError(
            f"Unknown open-vocabulary detector {model!r}. Known aliases: "
            f"{', '.join(sorted(_ALIASES))}."
        )
    family_cls, size = match
    return family_cls(size=size, **kwargs)


__all__ = [
    "LibreOpenVocab",
    "LibreOpenVocabDetector",
    "LibreGroundingDINO",
    "LibreOWLv2",
    "LibreOMDetTurbo",
]
