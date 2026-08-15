"""Qwen3-VL used as a generalist point grounder.

Same weights as ``LibreVLM``'s Qwen3-VL family. The ask is a single click
on a 0–1000 grid, not a box list. Useful when no GUI-specialized model is
loaded yet; prefer ShowUI / Holo / UI-TARS for screens.
"""

from __future__ import annotations

from typing import ClassVar, Dict

from .base import LibreGroundModel


class LibreGroundQwen3VL(LibreGroundModel):
    FAMILY = "qwen3vl"
    FILENAME_PREFIX = "LibreGroundQwen3VL"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "2b": "Qwen/Qwen3-VL-2B-Instruct",
        "4b": "Qwen/Qwen3-VL-4B-Instruct",
        "8b": "Qwen/Qwen3-VL-8B-Instruct",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "2b": 1024,
        "4b": 1024,
        "8b": 1024,
    }
    COORD_SPACE = "milli"
    MAX_NEW_TOKENS = 128

    def _format_grounding_prompt(self, query: str) -> str:
        return (
            f"Point to: {query}. "
            "Reply with a JSON object "
            '{"point": [x, y], "label": "..."} '
            "where x and y are integers on a 0-1000 scale relative to the image. "
            "If the target is not visible, reply with {}."
        )
