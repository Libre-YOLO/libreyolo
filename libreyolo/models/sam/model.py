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

from typing import Dict, Tuple, Type

from .base import LibreSAMModel


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


# alias -> (family class, size)
_ALIASES: Dict[str, Tuple[Type[LibreSAMModel], str]] = {
    "base": (LibreSAM1, "base"),
    "large": (LibreSAM1, "large"),
    "huge": (LibreSAM1, "huge"),
    "b": (LibreSAM1, "base"),
    "l": (LibreSAM1, "large"),
    "h": (LibreSAM1, "huge"),
    "sam-base": (LibreSAM1, "base"),
    "sam-large": (LibreSAM1, "large"),
    "sam-huge": (LibreSAM1, "huge"),
    "sam_b": (LibreSAM1, "base"),
    "sam_l": (LibreSAM1, "large"),
    "sam_h": (LibreSAM1, "huge"),
}

_DEFAULT_MODEL = "base"


def LibreSAM(model: str = _DEFAULT_MODEL, **kwargs) -> LibreSAMModel:
    """Load a promptable segmentation model by name.

    Args:
        model: Size alias — ``"base"`` (default), ``"large"``, or ``"huge"``
            (also ``"sam_b"``/``"sam_l"``/``"sam_h"``, or ``"b"``/``"l"``/``"h"``).
        **kwargs: Forwarded to the family constructor: ``device``, and
            ``multimask`` (when True, ``predict`` returns all of SAM's ambiguity
            masks per prompt — 3 whole-vs-part masks — instead of the single
            best one; can also be set per-call on ``predict``).

    Returns:
        A ``LibreSAMModel`` with the interactive ``set_image``/``predict`` surface.
    """
    key = str(model).strip().lower()
    match = _ALIASES.get(key)
    if match is None:
        raise ValueError(
            f"Unknown SAM model {model!r}. Known aliases: "
            f"{', '.join(sorted(_ALIASES))}."
        )
    family_cls, size = match
    return family_cls(size=size, **kwargs)


__all__ = ["LibreSAM", "LibreSAM1"]
