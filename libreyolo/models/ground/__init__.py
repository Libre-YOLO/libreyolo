"""LibreGround: instruction → click point.

Sibling factory to ``LibreVLM`` / ``LibreOpenVocab``. Returns the same
``Results.points`` payload as FOMO / LocateAnything.

    from libreyolo import LibreGround

    r = LibreGround()("screen.png", prompt="Bluetooth")
    r.points.xy
"""

from __future__ import annotations

from typing import Dict, Tuple, Type

from .base import GroundAPIMixin, LibreGroundModel
from .florence import LibreGroundFlorence2
from .holo import LibreHolo
from .locate import LibreGroundLocateAnything
from .qwen3vl import LibreGroundQwen3VL
from .showui import LibreShowUI
from .tinyclick import LibreTinyClick
from .uitars import LibreUITARS

# alias -> (family class, size)
_ALIASES: Dict[str, Tuple[Type, str]] = {
    "showui": (LibreShowUI, "2b"),
    "show-ui": (LibreShowUI, "2b"),
    "showui-2b": (LibreShowUI, "2b"),
    "show-ui-2b": (LibreShowUI, "2b"),
    "holo": (LibreHolo, "7b"),
    "holo1.5": (LibreHolo, "7b"),
    "holo1.5-7b": (LibreHolo, "7b"),
    "holo-7b": (LibreHolo, "7b"),
    "holo1-7b": (LibreHolo, "1-7b"),
    "ui-tars": (LibreUITARS, "7b"),
    "uitars": (LibreUITARS, "7b"),
    "ui-tars-7b": (LibreUITARS, "7b"),
    "uitars-7b": (LibreUITARS, "7b"),
    "ui-tars-1.5": (LibreUITARS, "7b"),
    "ui-tars-1.5-7b": (LibreUITARS, "7b"),
    "florence-2": (LibreGroundFlorence2, "base"),
    "florence2": (LibreGroundFlorence2, "base"),
    "florence-2-base": (LibreGroundFlorence2, "base"),
    "florence-2-large": (LibreGroundFlorence2, "large"),
    "tinyclick": (LibreTinyClick, "b"),
    "tiny-click": (LibreTinyClick, "b"),
    "locate-anything": (LibreGroundLocateAnything, "3b"),
    "locateanything": (LibreGroundLocateAnything, "3b"),
    "locate-anything-3b": (LibreGroundLocateAnything, "3b"),
    "qwen3-vl": (LibreGroundQwen3VL, "4b"),
    "qwen3vl": (LibreGroundQwen3VL, "4b"),
    "qwen3-vl-2b": (LibreGroundQwen3VL, "2b"),
    "qwen3-vl-4b": (LibreGroundQwen3VL, "4b"),
    "qwen3-vl-8b": (LibreGroundQwen3VL, "8b"),
}

_DEFAULT_MODEL = "showui-2b"


def LibreGround(model: str = _DEFAULT_MODEL, **kwargs):
    """Load a grounding model by alias.

    Args:
        model: Alias such as ``"showui-2b"``, ``"holo-7b"``, ``"tinyclick"``.
            Defaults to ShowUI-2B (Apache-2.0).
        **kwargs: Forwarded to the family constructor: ``device``,
            ``query`` / ``prompt`` (initial instruction), ``max_new_tokens``.

    Returns:
        A model with ``set_query`` / ``predict(..., prompt=)`` returning
        ``Results.points``.
    """
    key = str(model).strip().lower().replace("_", "-")
    match = _ALIASES.get(key)
    if match is None:
        raise ValueError(
            f"Unknown grounding model {model!r}. Known aliases: "
            f"{', '.join(sorted(set(_ALIASES)))}."
        )
    family_cls, size = match
    return family_cls(size=size, **kwargs)


__all__ = [
    "LibreGround",
    "LibreGroundModel",
    "GroundAPIMixin",
    "LibreShowUI",
    "LibreHolo",
    "LibreUITARS",
    "LibreTinyClick",
    "LibreGroundFlorence2",
    "LibreGroundLocateAnything",
    "LibreGroundQwen3VL",
]
