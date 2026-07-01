"""LibreSAM tier: promptable segmentation models (SAM family).

See ``model.py`` for the ``LibreSAM(...)`` factory and ``base.py`` for the
``LibreSAMModel`` interactive contract.
"""

from __future__ import annotations

from .base import LibreSAMModel
from .model import LibreSAM, LibreSAM1, LibreSAM2

__all__ = ["LibreSAM", "LibreSAMModel", "LibreSAM1", "LibreSAM2", "LibreMobileSAM"]


def __getattr__(name):
    if name == "LibreMobileSAM":
        from ..mobilesam import LibreMobileSAM

        return LibreMobileSAM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
