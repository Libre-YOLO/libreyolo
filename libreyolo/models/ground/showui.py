"""ShowUI-2B: Apache-2.0 Qwen2-VL GUI grounder.

Official output is a bare ``[x, y]`` pair normalized to ``[0, 1]`` relative
to the screenshot. That is the default ``LibreGround`` family because it is
small, permissively licensed, and loads through native transformers.
"""

from __future__ import annotations

from typing import ClassVar, Dict

from .base import LibreGroundModel


class LibreShowUI(LibreGroundModel):
    FAMILY = "showui"
    FILENAME_PREFIX = "LibreShowUI"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "2b": "showlab/ShowUI-2B",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "2b": 1344,
    }
    COORD_SPACE = "unit"
    MAX_NEW_TOKENS = 64

    def _format_grounding_prompt(self, query: str) -> str:
        return (
            "Based on the screenshot of the page, I give a text description "
            "and you give its corresponding location. The coordinate represents "
            "a clickable location [x, y] for an element, which is a relative "
            "coordinate on the screenshot, scaled from 0 to 1.\n"
            f"{query}"
        )
