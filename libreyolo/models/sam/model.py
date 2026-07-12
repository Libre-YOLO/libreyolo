"""LibreSAM: promptable segmentation models behind a familiar surface.

User-facing entry point is the ``LibreSAM(...)`` factory, a sibling to the
``LibreYOLO(...)`` and ``LibreVLM(...)`` factories. It returns a promptable
segmenter with an interactive, encode-once / prompt-many surface:

    from libreyolo import LibreSAM

    model = LibreSAM("base")                 # autodownloads (Apache-2.0)
    r = model.predict("image.jpg", points=[900, 370], labels=[1])  # point -> mask
    r = model.predict("image.jpg", bboxes=[100, 100, 200, 200])    # box   -> mask
    r = model.predict("image.jpg")                                  # segment everything
    r.masks.xy          # polygons
    r.boxes.xyxy        # tight boxes derived from masks

    model.set_image("image.jpg")             # encode once...
    a = model.predict(points=[500, 375], labels=[1])   # ...prompt cheaply
    b = model.predict(bboxes=[100, 100, 200, 200])
    model.reset_image()

The default family is SAM-1, whose code and weights are Apache-2.0, loaded
through the permissive ``transformers`` model API. See
``docs/adr/0007-libresam-contract.md``.
"""

from __future__ import annotations

from .base import LibreSAMModel
from .sam2 import LibreSAM2
from .sam3 import LibreSAM3
from ..manifest import (
    FACTORY_DEFAULT_MODELS,
    FACTORY_MODEL_ALIASES,
    FactoryKind,
    load_family_class,
    resolve_factory_model,
)


class LibreSAM1(LibreSAMModel):
    """SAM-1 promptable segmenter (ViT-B/L/H image encoder)."""

    FAMILY = "sam"
    FILENAME_PREFIX = "LibreSAM"
    HF_REPOS = {
        "base": "facebook/sam-vit-base",
        "large": "facebook/sam-vit-large",
        "huge": "facebook/sam-vit-huge",
    }
    INPUT_SIZES = {"base": 1024, "large": 1024, "huge": 1024}


_MOBILE_SAM = "mobilesam"
_PICO_SAM3 = "picosam3"


def _compat_alias_target(family, selection):
    if family.family == "mobilesam":
        return _MOBILE_SAM, selection.size
    if family.family == "picosam3":
        return _PICO_SAM3, selection.size
    return load_family_class(family), selection.size


# Backward-compatible private view, generated from the public manifest.
_ALIASES = {
    alias: _compat_alias_target(family, selection)
    for (factory, alias), (family, selection) in FACTORY_MODEL_ALIASES.items()
    if factory is FactoryKind.SAM
}

_DEFAULT_MODEL = FACTORY_DEFAULT_MODELS[FactoryKind.SAM]


def LibreSAM(model: str = _DEFAULT_MODEL, **kwargs) -> LibreSAMModel:
    """Load a promptable segmentation model by name.

    Args:
        model: Size alias — ``"base"`` (default), ``"large"``, or ``"huge"``
            (also ``"sam_b"``/``"sam_l"``/``"sam_h"``, or ``"b"``/``"l"``/``"h"``).
            SAM-2 aliases use an explicit prefix, for example
            ``"sam2-tiny"`` / ``"sam2_t"``.
            SAM 3 uses ``"sam3"`` / ``"sam-3"`` / ``"sam3-large"``.
            MobileSAM aliases resolve to its single ``"tiny"`` size.
        **kwargs: Forwarded to the family constructor: ``device``, and
            ``multimask`` (when True, ``predict`` returns all of SAM's ambiguity
            masks per prompt — 3 whole-vs-part masks — instead of the single
            best one; can also be set per-call on ``predict``).

    Returns:
        A ``LibreSAMModel`` with the interactive ``set_image``/``predict`` surface.
    """
    match = resolve_factory_model(FactoryKind.SAM, model)
    if match is None:
        aliases = sorted(
            alias
            for (factory, alias) in FACTORY_MODEL_ALIASES
            if factory is FactoryKind.SAM
        )
        raise ValueError(
            f"Unknown SAM model {model!r}. Known aliases: {', '.join(aliases)}."
        )
    family, selection = match
    family_cls = load_family_class(family)
    return family_cls(size=selection.size, **kwargs)


__all__ = ["LibreSAM", "LibreSAM1", "LibreSAM2", "LibreSAM3"]
