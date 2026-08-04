"""Capture-safety shim for the transformers SAM vision attention.

``SamVisionAttention.get_rel_pos`` builds its relative-position index with
``torch.arange`` on the host, then uses it to index a tensor that lives on the
GPU. That index copies host-to-device on every call, which CUDA graph capture
rejects, so SAM's vision encoder cannot be captured as shipped.

The index depends only on ``(q_size, k_size)``, never on activations, so this
replacement builds it on the embedding's own device and memoises it per
(q_size, k_size, device). The values are identical: the same arithmetic in the
same order, only the allocation moves.

The patch is applied once, defensively: if a future transformers release
renames or restructures the method, the shim declines rather than breaking
imports, and SAM simply stays uncapturable as before.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

_INDEX_CACHE: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}
_PATCHED = False


def _relative_index(q_size: int, k_size: int, device: torch.device) -> torch.Tensor:
    """Memoised relative-position index, allocated on *device*.

    Mirrors the upstream computation exactly, including the float scaling and
    the trailing ``.long()``; only the allocation device differs.
    """
    key = (q_size, k_size, device)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    q_coords = torch.arange(q_size, device=device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    index = relative_coords.long()
    _INDEX_CACHE[key] = index
    return index


def _get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    rel_pos_resized = F.interpolate(
        rel_pos.reshape(1, rel_pos.shape[0], -1).transpose(1, 2),
        size=max_rel_dist,
        mode="linear",
    )
    rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    return rel_pos_resized[_relative_index(q_size, k_size, rel_pos.device)]


def apply() -> bool:
    """Install the shim. Returns True if SAM attention is now capture-safe."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from transformers.models.sam import modeling_sam
    except Exception:  # noqa: BLE001 - transformers absent or restructured
        return False

    attention = getattr(modeling_sam, "SamVisionAttention", None)
    if attention is None or not hasattr(attention, "get_rel_pos"):
        return False

    attention.get_rel_pos = _get_rel_pos
    _PATCHED = True
    return True
