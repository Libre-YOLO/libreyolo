"""LibreYOLO network wrapper for the vendored DA3 monocular model."""

from __future__ import annotations

import torch
import torch.nn as nn

from ._vendor.dinov2 import DinoV2
from ._vendor.dpt import DPT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LibreDepthAnything3Net(nn.Module):
    """DA3MONO-LARGE adapted to LibreYOLO's inverse-depth contract.

    The official model consumes a five-dimensional ``(B, views, C, H, W)``
    tensor and emits positive relative depth. LibreYOLO's depth task is
    single-image relative *inverse* depth, so this wrapper adds the singleton
    view, reproduces the official mono sky handling, and returns ``1 / depth``
    as ``(B, 1, H, W)``. ImageNet normalization lives inside ``forward`` so
    callers consistently provide RGB tensors in ``[0, 1]``.

    Learnable module names remain ``backbone.*`` and ``head.*``. Converted
    official tensors therefore load strictly after removing only the outer
    high-level API prefix (``model.``). Normalization buffers are non-persistent.
    """

    PATCH_SIZE = 14
    INVERSE_DEPTH_EPS = 1e-6

    def __init__(self) -> None:
        super().__init__()
        self.backbone = DinoV2(
            name="vitl",
            out_layers=[4, 11, 17, 23],
            alt_start=-1,
            qknorm_start=-1,
            rope_start=-1,
            cat_token=False,
        )
        self.head = DPT(
            dim_in=1024,
            output_dim=1,
            features=256,
            out_channels=[256, 512, 1024, 1024],
        )
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    @staticmethod
    def _apply_mono_sky(depth: torch.Tensor, sky: torch.Tensor) -> torch.Tensor:
        """Match upstream's sky-to-far-depth postprocessing."""
        non_sky_mask = sky < 0.3
        if non_sky_mask.sum() <= 10 or (~non_sky_mask).sum() <= 10:
            return depth

        non_sky_depth = depth[non_sky_mask]
        if non_sky_depth.numel() > 100_000:
            indices = torch.randint(
                0,
                non_sky_depth.numel(),
                (100_000,),
                device=non_sky_depth.device,
            )
            non_sky_depth = non_sky_depth[indices]
        far_depth = torch.quantile(non_sky_depth, 0.99)
        return torch.where(non_sky_mask, depth, far_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "Depth Anything 3 expects input shaped (B, 3, H, W); "
                f"received {tuple(x.shape)}."
            )
        height, width = x.shape[-2:]
        if height % self.PATCH_SIZE or width % self.PATCH_SIZE:
            raise ValueError(
                f"Depth Anything 3 input height and width must be divisible by {self.PATCH_SIZE}."
            )

        x = (x - self.pixel_mean) / self.pixel_std
        features, _ = self.backbone(x.unsqueeze(1), export_feat_layers=[])
        output = self.head(features, height, width, patch_start_idx=0)
        depth = output["depth"]
        if "sky" in output:
            depth = self._apply_mono_sky(depth, output["sky"])
        inverse_depth = torch.reciprocal(depth.clamp_min(self.INVERSE_DEPTH_EPS))
        return inverse_depth[:, 0].unsqueeze(1)
