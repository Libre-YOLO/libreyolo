"""LocateAnything used as a LibreGround family (point task, NVIDIA NC).

Reuses the existing VLM adapter (remote code, pinned revision, license
notice) and only changes the public surface to ``set_query`` / ``prompt=``.
"""

from __future__ import annotations

from ..vlm.locateanything import LibreLocateAnything
from .base import GroundAPIMixin


class LibreGroundLocateAnything(GroundAPIMixin, LibreLocateAnything):
    """LocateAnything locked to ``task="point"`` with the grounding API."""

    FAMILY = "ground_locateanything"
    DEFAULT_TASK = "point"
    SUPPORTED_TASKS = ("point",)

    def __init__(self, size: str = "3b", **kwargs):
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
