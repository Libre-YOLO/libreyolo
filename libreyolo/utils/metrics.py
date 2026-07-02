"""Validation metrics container returned by ``model.val()``."""

from __future__ import annotations

from typing import Any, Dict


class Metrics(dict):
    """The dict returned by :meth:`val`, with attribute access for headline metrics.

    ``val()`` has always returned a plain ``metrics/*``-keyed dict, and that
    access still works unchanged (``metrics["metrics/mAP50-95"]``,
    ``metrics.get(...)``). On top of that, the common per-task headline numbers
    are reachable as attributes so a result reads naturally:

        >>> m = model.val(data="imagenette160")     # classification
        >>> m.top1, m.top5
        (0.976, 0.999)
        >>> m = model.val(data="coco.yaml")          # detection
        >>> m.map, m.map50
        (0.523, 0.671)

    Only aliases whose underlying key is present resolve; asking for a metric a
    task did not produce (e.g. ``.top1`` on a detector) raises ``AttributeError``
    with the list of metrics that *are* available.
    """

    # attribute -> underlying metrics dict key
    _ALIASES: Dict[str, str] = {
        "top1": "metrics/accuracy_top1",
        "top5": "metrics/accuracy_top5",
        "map": "metrics/mAP50-95",
        "map50": "metrics/mAP50",
        "map75": "metrics/mAP75",
        "precision": "metrics/precision",
        "recall": "metrics/recall",
        "fitness": "fitness",
    }

    def __getattr__(self, name: str) -> Any:
        key = self._ALIASES.get(name, name)
        if key in self:
            return self[key]
        available = sorted(k for k in self.keys() if isinstance(k, str))
        raise AttributeError(
            f"{type(self).__name__!r} object has no metric {name!r}. "
            f"Available metrics: {available}"
        )

    def __repr__(self) -> str:  # keep the dict payload visible
        return f"Metrics({dict.__repr__(self)})"
