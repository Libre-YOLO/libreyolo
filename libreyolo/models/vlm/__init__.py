"""LibreVLM: vision-language models used as open-vocabulary object detectors.

User-facing entry point is the ``LibreVLM(...)`` factory, a sibling to the
``LibreYOLO(...)`` factory. It returns a model instance that behaves like any
YOLO model (predict on image/folder/video, track, identical ``Results``), but
is backed by a generative VLM, so the class list is open vocabulary.

    from libreyolo import LibreVLM
    model = LibreVLM()                       # defaults to Qwen3-VL-4B, autodownloads
    model.set_classes(["pink car", "wheel"]) # open vocabulary: any words
    results = model.predict("image.jpg")     # same Results as a YOLO model
    results = model.predict("folder/")       # folders, video, track() all work
    text = model.chat("image.jpg", "How many cars are pink?")  # raw escape hatch

See ``docs/librevlm_design.md`` and ``docs/adr/0002-librevlm-contract.md``.
"""

from __future__ import annotations

from .base import LibreVLMModel
from .florence2 import LibreFlorence2
from .internvl3 import LibreInternVL3
from .kosmos2 import LibreKosmos2
from .lfm2 import LibreLFM2VL
from .locateanything import LibreLocateAnything
from .qwen3vl import LibreQwen3VL
from .smolvlm import LibreSmolVLM2
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
    if factory is FactoryKind.VLM
}

_DEFAULT_MODEL = FACTORY_DEFAULT_MODELS[FactoryKind.VLM]


def LibreVLM(model: str = _DEFAULT_MODEL, **kwargs) -> LibreVLMModel:
    """Load a vision-language detector by name.

    Args:
        model: Model alias (e.g. ``"qwen3-vl-4b"``, ``"lfm2-vl-450m"``).
            Defaults to Qwen3-VL-4B (Apache-2.0).
        **kwargs: Forwarded to the family constructor: ``device``, ``names``
            (initial class vocabulary, same as calling ``set_classes`` after
            load), ``prompt`` (override the detection prompt), ``max_new_tokens``.

    Returns:
        A ``LibreVLMModel`` instance with the standard predict/track surface.
    """
    match = resolve_factory_model(FactoryKind.VLM, model)
    if match is None:
        aliases = sorted(
            alias
            for (factory, alias) in FACTORY_MODEL_ALIASES
            if factory is FactoryKind.VLM
        )
        raise ValueError(
            f"Unknown VLM model {model!r}. Known aliases: {', '.join(aliases)}."
        )
    family, selection = match
    family_cls = load_family_class(family)
    return family_cls(size=selection.size, **kwargs)


__all__ = [
    "LibreVLM",
    "LibreVLMModel",
    "LibreLFM2VL",
    "LibreQwen3VL",
    "LibreSmolVLM2",
    "LibreInternVL3",
    "LibreFlorence2",
    "LibreKosmos2",
    "LibreLocateAnything",
]
