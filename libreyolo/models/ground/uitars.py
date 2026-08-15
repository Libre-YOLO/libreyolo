"""UI-TARS-1.5-7B: Apache-2.0 native GUI agent used here as a grounder.

The family is asked only for a click. Coordinates are the original UI-TARS
0–1000 scheme (``COORD_SPACE="milli"``). The parser also accepts
``click(point='<point>x y</point>')``.
"""

from __future__ import annotations

from typing import ClassVar, Dict

from .base import LibreGroundModel


class LibreUITARS(LibreGroundModel):
    FAMILY = "uitars"
    FILENAME_PREFIX = "LibreUITARS"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "7b": "ByteDance-Seed/UI-TARS-1.5-7B",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "7b": 1344,
    }
    COORD_SPACE = "milli"
    MAX_NEW_TOKENS = 128

    def _format_grounding_prompt(self, query: str) -> str:
        return (
            "You are a GUI grounding model. Output a single click on the "
            "element that matches the instruction.\n"
            "Action: click(start_box='<point>x y</point>')\n"
            "x and y are integers on a 0-1000 scale relative to the screenshot.\n"
            f"Instruction: {query}"
        )
