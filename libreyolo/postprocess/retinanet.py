"""RetinaNet postprocessing placeholder for the skeleton commit."""

from __future__ import annotations

import numpy as np


def postprocess(*_args, **_kwargs) -> dict:
    """Return an empty canonical result until the parity gate is complete."""
    return {
        "num_detections": 0,
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int64),
    }


__all__ = ["postprocess"]
