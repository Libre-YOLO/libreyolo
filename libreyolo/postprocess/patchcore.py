"""PatchCore anomaly-map postprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def postprocess(
    output: dict[str, torch.Tensor],
    *,
    original_size: tuple[int, int],
    threshold: float | None,
    sigma: float = 4.0,
) -> dict[str, Any]:
    """Upsample and smooth one PatchCore score map to the original canvas."""
    width, height = original_size
    patch_scores = output["patch_scores"]
    image_scores = output["image_scores"]
    if patch_scores.ndim == 2:
        patch_scores = patch_scores.unsqueeze(0)
    anomaly_map = F.interpolate(
        patch_scores[:, None].float(), size=(height, width), mode="bilinear", align_corners=False
    )[0, 0].detach().cpu().numpy()
    if sigma > 0:
        from scipy.ndimage import gaussian_filter

        anomaly_map = gaussian_filter(anomaly_map, sigma=float(sigma))
    score = float(image_scores.reshape(-1)[0].detach().cpu())
    return {
        "anomaly_score": score,
        "anomaly_map": np.asarray(anomaly_map, dtype=np.float32),
        "is_anomalous": None if threshold is None else bool(score > threshold),
    }


__all__ = ["postprocess"]
