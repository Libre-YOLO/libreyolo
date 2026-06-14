"""Shared array / Results-backend helpers used by every tracker.

These keep track IDs and detection payloads on the same tensor/array backend
(torch or numpy) as the incoming detections, so tracking is backend-agnostic.
"""

from __future__ import annotations

import numpy as np
import torch


def _as_numpy(data) -> np.ndarray:
    """Convert a tensor-like result payload to a NumPy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _make_track_ids(ids: list[int], results):
    """Create track IDs on the same tensor/array backend as the source boxes."""
    boxes = results.boxes.xyxy
    if isinstance(boxes, torch.Tensor):
        return torch.tensor(ids, dtype=torch.int64, device=boxes.device)
    return np.asarray(ids, dtype=np.int64)
