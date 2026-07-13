"""LibreSAM tier: promptable segmentation models (SAM family).

See ``model.py`` for the ``LibreSAM(...)`` factory and ``base.py`` for the
``LibreSAMModel`` interactive contract.
"""

from __future__ import annotations

from .base import LibreSAMModel
from .model import LibreEdgeTAM, LibreSAM, LibreSAM1, LibreSAM2, LibreSAM3

__all__ = [
    "LibreSAM",
    "LibreSAMModel",
    "LibreSAM1",
    "LibreSAM2",
    "LibreEdgeTAM",
    "LibreSAM3",
    "LibreMobileSAM",
    "LibrePicoSAM3",
]


def __getattr__(name):
    if name == "LibreMobileSAM":
        from ..mobilesam import LibreMobileSAM

        return LibreMobileSAM
    if name == "LibrePicoSAM3":
        from ..picosam3 import LibrePicoSAM3

        return LibrePicoSAM3
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
