"""Image-restoration validator for LibreYOLO.

Metric protocol: RGB, float data range [0, 1], no border crop. PSNR is computed
per image over all valid original-canvas pixels. SSIM uses an 11x11 Gaussian
window with sigma 1.5 and is averaged over RGB channels.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.restore_dataset import (
    RestoreDataset,
    resolve_restore_data,
    restore_collate_fn,
)
from .base import BaseValidator

logger = logging.getLogger(__name__)

_METRIC_KEYS = ("metrics/PSNR", "metrics/SSIM")

# A perfect reconstruction (mse == 0) has infinite PSNR; cap it at a large but
# finite value so a single perfect image cannot poison the aggregate PSNR /
# fitness (best-checkpoint selection).
_MAX_PSNR = 100.0


def psnr_rgb(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute RGB PSNR for tensors with matching ``[C, H, W]`` shape."""

    mse = torch.mean((pred.float() - target.float()).pow(2))
    if float(mse) == 0.0:
        return _MAX_PSNR
    psnr = float(20.0 * math.log10(data_range) - 10.0 * torch.log10(mse).item())
    return min(psnr, _MAX_PSNR)


def _gaussian_window(
    channels: int,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    device=None,
    dtype=None,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).view(1, 1, window_size, window_size)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def ssim_rgb(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute RGB SSIM with a fixed 11x11 Gaussian window."""

    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError("SSIM expects [C, H, W] tensors.")
    if min(pred.shape[-2:]) < 11:
        return 1.0 if torch.allclose(pred, target) else 0.0
    pred = pred.float().unsqueeze(0)
    target = target.float().unsqueeze(0)
    channels = pred.shape[1]
    window = _gaussian_window(
        channels,
        device=pred.device,
        dtype=pred.dtype,
    )
    padding = 5
    mu1 = F.conv2d(pred, window, padding=padding, groups=channels)
    mu2 = F.conv2d(target, window, padding=padding, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().clamp(0.0, 1.0).item())


class RestoreValidator(BaseValidator):
    """PSNR / SSIM validator for the restore task."""

    task = "restore"

    def _setup_dataloader(self) -> DataLoader:
        if not self.config.data:
            raise ValueError("Restore validation requires data= (a dataset YAML).")
        data_config = resolve_restore_data(
            self.config.data,
            allow_scripts=getattr(self.config, "allow_download_scripts", False),
        )
        # Super-resolution models upscale by an integer factor; the paired GT
        # target must be that many times the input. RestoreDataset enforces the
        # shape relationship and errors clearly on a mismatch.
        scale = int(getattr(self.model, "restore_scale", 1) or 1)
        dataset = RestoreDataset(
            data_config,
            split=self.config.split or "val",
            imgsz=self.config.imgsz,
            augment=False,
            scale=scale,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=restore_collate_fn,
        )

    def _init_metrics(self) -> None:
        self._psnr_values: list[float] = []
        self._ssim_values: list[float] = []

    def _preprocess_batch(self, batch: Any) -> tuple:
        images, targets, img_info, img_ids = batch
        return images, targets, img_info, img_ids

    def _postprocess_predictions(self, preds: Any, batch: Any) -> torch.Tensor:
        output = preds
        if isinstance(output, dict):
            output = output.get("restored", output.get("predictions"))
        if isinstance(output, (list, tuple)):
            output = output[0]
        output = torch.as_tensor(output).float()
        if output.ndim == 3:
            output = output.unsqueeze(0)
        if output.ndim == 4 and output.shape[-1] == 3 and output.shape[1] != 3:
            output = output.permute(0, 3, 1, 2)
        if output.ndim != 4 or output.shape[1] != 3:
            raise ValueError(
                f"Restore validation expects [B, 3, H, W] output, got {tuple(output.shape)}."
            )
        target_hw = tuple(batch[1].shape[-2:])
        output = output[:, :, : target_hw[0], : target_hw[1]]
        return output.clamp(0.0, 1.0)

    def _update_metrics(
        self,
        preds: Any,
        targets: Any,
        img_info: Any,
        img_ids: Any = None,
    ) -> None:
        del img_ids
        preds = preds.detach().cpu().float()
        targets = targets.detach().cpu().float()
        for pred, target, info in zip(preds, targets, img_info):
            # For super-resolution the valid region is the HR target canvas
            # (scale x the input); for deblur/denoise it equals the input shape.
            h, w = info.get("target_shape", info["orig_shape"])
            if pred.shape[-2] < h or pred.shape[-1] < w:
                raise ValueError(
                    "Restore prediction is smaller than the ground-truth target "
                    f"({tuple(pred.shape[-2:])} vs required {(h, w)}). For "
                    "super-resolution ensure the model scale matches the pairs."
                )
            pred = pred[:, :h, :w].clamp(0.0, 1.0)
            target = target[:, :h, :w].clamp(0.0, 1.0)
            self._psnr_values.append(psnr_rgb(pred, target))
            self._ssim_values.append(ssim_rgb(pred, target))

    def _compute_metrics(self) -> Dict[str, float]:
        if not self._psnr_values:
            raise ValueError("Restore validation found no paired images.")
        # PSNR is now capped to a finite value per image (see _MAX_PSNR), but
        # guard against any stray non-finite value so the aggregate / fitness
        # never becomes inf/nan and breaks best-checkpoint selection.
        finite_psnr = [value for value in self._psnr_values if math.isfinite(value)]
        psnr = sum(finite_psnr) / len(finite_psnr) if finite_psnr else 0.0
        ssim = sum(self._ssim_values) / len(self._ssim_values)
        return {
            "metrics/PSNR": psnr,
            "metrics/SSIM": ssim,
            "fitness": psnr,
        }

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("Restore Validation Results")
        logger.info("=" * 50)
        logger.info("  PSNR: %s", metrics.get("metrics/PSNR", 0.0))
        logger.info("  SSIM: %.4f", metrics.get("metrics/SSIM", 0.0))
        logger.info("=" * 50)


__all__ = ["RestoreValidator", "psnr_rgb", "ssim_rgb"]

