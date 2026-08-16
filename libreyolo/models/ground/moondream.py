"""Moondream used as a LibreGround family (native ``point()`` skill).

Reuses the existing VLM adapter: Apache-2.0 size 2, BSL size 3, remote code
pinned. Coordinates come back in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ..vlm.moondream import LibreMoondream
from .base import GroundAPIMixin


class LibreGroundMoondream(GroundAPIMixin, LibreMoondream):
    """Moondream locked to ``task="point"`` with the grounding API."""

    FAMILY = "ground_moondream"
    DEFAULT_TASK = "point"
    SUPPORTED_TASKS = ("point",)

    def __init__(self, size: str = "2", **kwargs):
        kwargs.setdefault("task", "point")
        query = kwargs.pop("query", None)
        prompt = kwargs.pop("prompt", None)
        names = kwargs.get("names")
        if query is not None:
            kwargs["names"] = query if not isinstance(query, str) else [query]
        elif names is None and prompt is not None:
            kwargs["names"] = [prompt] if isinstance(prompt, str) else list(prompt)
        super().__init__(size=size, **kwargs)
        if self.names:
            self._queries = [self.names[i] for i in range(len(self.names))]
        else:
            self.names = {}
            self.nb_classes = 0
            self._name_to_id = {}
            self._queries = []

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        return super()._postprocess(
            output,
            conf_thres,
            iou_thres,
            original_size,
            max_det=1 if max_det is None else min(int(max_det), 1),
            ratio=ratio,
            **kwargs,
        )
