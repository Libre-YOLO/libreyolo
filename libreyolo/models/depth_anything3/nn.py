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
        self.export = False
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
        """Match upstream's sky-to-far-depth postprocessing, per image.

        Upstream applies this to the views of a single scene. LibreYOLO batches
        are independent images (the depth validator batches them), so the
        far-depth quantile must not mix statistics across batch items.
        """
        result = depth
        for i in range(depth.shape[0]):
            non_sky_mask = sky[i] < 0.3
            if non_sky_mask.sum() <= 10 or (~non_sky_mask).sum() <= 10:
                continue

            non_sky_depth = depth[i][non_sky_mask]
            if non_sky_depth.numel() > 100_000:
                indices = torch.randint(
                    0,
                    non_sky_depth.numel(),
                    (100_000,),
                    device=non_sky_depth.device,
                )
                non_sky_depth = non_sky_depth[indices]
            far_depth = torch.quantile(non_sky_depth, 0.99)
            if result is depth:
                result = depth.clone()
            result[i] = torch.where(non_sky_mask, depth[i], far_depth)
        return result

    def forward_network(self, x: torch.Tensor) -> tuple:
        """Backbone and head only: raw depth, plus sky logits when predicted.

        Split out from ``forward`` because everything here is pure tensor work
        and captures cleanly, whereas the sky step that follows cannot (see
        ``_apply_mono_sky``). Keeping them apart lets the network be replayed
        from a graph while the sky step runs eagerly on the result, which
        leaves the numbers untouched.
        """
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
        sky = output.get("sky")
        return (depth,) if sky is None else (depth, sky)

    def finish_depth(self, depth: torch.Tensor, sky: torch.Tensor | None) -> torch.Tensor:
        """Sky-to-far-depth step and inversion, run eagerly in every path."""
        if sky is not None:
            depth = self._apply_mono_sky(depth, sky)
        inverse_depth = torch.reciprocal(depth.clamp_min(self.INVERSE_DEPTH_EPS))
        return inverse_depth[:, 0].unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.forward_network(x)
        depth = raw[0]
        sky = raw[1] if len(raw) > 1 else None
        if getattr(self, "export", False):
            # The sky heuristic contains tensor-dependent branching and a
            # quantile. Export its two dense inputs and reproduce that exact
            # postprocess in the runtime backend instead of freezing one
            # example's branch into the graph.
            return depth[:, 0].unsqueeze(1), sky[:, 0].unsqueeze(1)
        return self.finish_depth(depth, sky)
