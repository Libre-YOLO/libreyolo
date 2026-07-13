"""Open-vocabulary detector tier.

``LibreOpenVocab(...)`` is a sibling factory to ``LibreSAM`` and ``LibreVLM``.
It loads discriminative text-conditioned detectors from Hugging Face snapshots
and returns standard detection ``Results``.
"""

from __future__ import annotations

from .base import LibreOpenVocabDetector
from .grounding_dino import LibreGroundingDINO
from .owlv2 import LibreOWLv2
from ..manifest import (
    FACTORY_DEFAULT_MODELS,
    FACTORY_MODEL_ALIASES,
    FactoryKind,
    load_family_class,
    resolve_factory_model,
)

_ALIASES = {
    alias: (load_family_class(family), selection.size)
    for (factory, alias), (family, selection) in FACTORY_MODEL_ALIASES.items()
    if factory is FactoryKind.OPENVOCAB
}

_DEFAULT_MODEL = FACTORY_DEFAULT_MODELS[FactoryKind.OPENVOCAB]


def LibreOpenVocab(model: str = _DEFAULT_MODEL, **kwargs) -> LibreOpenVocabDetector:
    """Load an open-vocabulary detector by alias."""
    match = resolve_factory_model(FactoryKind.OPENVOCAB, model)
    if match is None:
        aliases = sorted(
            alias
            for (factory, alias) in FACTORY_MODEL_ALIASES
            if factory is FactoryKind.OPENVOCAB
        )
        raise ValueError(
            f"Unknown open-vocabulary detector {model!r}. Known aliases: "
            f"{', '.join(aliases)}."
        )
    family, selection = match
    family_cls = load_family_class(family)
    return family_cls(size=selection.size, **kwargs)


__all__ = [
    "LibreOpenVocab",
    "LibreOpenVocabDetector",
    "LibreGroundingDINO",
    "LibreOWLv2",
]
