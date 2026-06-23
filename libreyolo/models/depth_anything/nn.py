"""LibreYOLO depth network built on the vendored Depth Anything V2 model."""

from __future__ import annotations

import torch

from ._vendor.dpt import DepthAnythingV2

# Per-encoder DPT configuration, matching upstream ``run.py`` ``model_configs``.
DEPTH_ANYTHING_V2_CONFIGS: dict[str, dict] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LibreDepthAnythingV2Net(DepthAnythingV2):
    """Depth Anything V2 adapted to LibreYOLO's dense-task contract.

    Two changes over the vendored module, both confined to ``forward``:

    - ImageNet normalization happens inside ``forward`` so the network consumes
      ``[0, 1]`` images, matching ``DepthDataset`` and LibreYOLO's other dense
      heads (the depth validator feeds raw ``[0, 1]`` batches straight to
      ``_forward``).
    - The result is returned as ``(B, 1, H, W)`` relative inverse depth so the
      depth predict and validation paths consume every family identically.

    The learnable parameters are untouched, so a converted upstream checkpoint
    (``pretrained.*`` + ``depth_head.*``) loads with ``strict=True``. The
    normalization buffers are non-persistent and never enter the state dict.
    """

    def __init__(self, encoder: str = "vitl"):
        if encoder not in DEPTH_ANYTHING_V2_CONFIGS:
            raise ValueError(
                f"Unknown Depth Anything V2 encoder {encoder!r}; "
                f"expected one of {sorted(DEPTH_ANYTHING_V2_CONFIGS)}."
            )
        super().__init__(**DEPTH_ANYTHING_V2_CONFIGS[encoder])
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        depth = super().forward(x)  # (B, H, W), relative inverse depth (>= 0)
        return depth.unsqueeze(1)

    def infer_image(self, *args, **kwargs):
        # The vendored DepthAnythingV2.infer_image normalizes ImageNet stats in
        # its own transform and then calls forward, which would normalize again
        # here. Route callers through LibreYOLO's predict path instead.
        raise NotImplementedError(
            "Use the LibreYOLO predict API (model.predict(...) / model(...)) "
            "instead of infer_image; the vendored infer_image would double-apply "
            "ImageNet normalization on top of this wrapper's forward."
        )
