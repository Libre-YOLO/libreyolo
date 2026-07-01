"""Spatial modules for RF-DETR-SO: raw-image detail extraction and bi-fusion.

Implemented clean-room from the TinyFormer paper (arXiv 2605.25046):

- ``SpatialDetailExtractor`` (SDE): a short stack of stride-2 convolutions on
  the raw input image. ViT tokenization destroys high-frequency detail before
  any pyramid is built; deconv-upsampled ViT features cannot recover it. The
  SDE re-extracts that detail at strides 4 and 8 so the stride-8 pyramid
  level carries true high-resolution evidence.
- ``BiFusionBlock`` (PBM building block): fuses a pyramid level with both of
  its neighbors in one step. The deeper level enters as an element-wise
  residual ADD (semantic context); the shallower level enters via a stride-2
  convolution and channel CONCAT (independent spatial evidence). The paper
  measured the swapped arrangement to be strictly worse; keep this order.

Both modules reuse the RF-DETR building blocks (``ConvX``, ``C2f``) so their
export behavior matches the rest of the family.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ..rfdetr.backbone import C2f, ConvX, get_norm


class SpatialDetailExtractor(nn.Module):
    """Stride-2 conv stack on the raw image; returns stride-4 and stride-8 maps.

    Channel progression is C -> 2C -> 4C from ``base_channels``. The stride-4
    output stays a pure spatial prior (never mixed with ViT features); the
    stride-8 output is fused into the P3 pyramid level.
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stage1 = ConvX(3, c, kernel=3, stride=2)  # stride 2
        self.stage2 = ConvX(c, 2 * c, kernel=3, stride=2)  # stride 4
        self.stage3 = ConvX(2 * c, 4 * c, kernel=3, stride=2)  # stride 8
        self.out_channels_s4 = 2 * c
        self.out_channels_s8 = 4 * c

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s2 = self.stage1(x)
        f2 = self.stage2(s2)
        f3 = self.stage3(f2)
        return f2, f3


class BiFusionBlock(nn.Module):
    """Parallel bi-fusion of (deeper, current, shallower) pyramid levels.

    ``out = LN(C2f(concat(cur + up2x(1x1(deep)), 3x3s2(shallow))))``

    The C2f + LayerNorm tail mirrors the MultiScaleProjector stages so the
    decoder sees identically normalized features on every level.
    """

    def __init__(
        self,
        hidden_dim: int,
        shallow_channels: int,
        num_blocks: int = 3,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.deep_proj = ConvX(hidden_dim, hidden_dim, kernel=1, layer_norm=layer_norm)
        self.shallow_down = ConvX(
            shallow_channels, hidden_dim, kernel=3, stride=2, layer_norm=layer_norm
        )
        self.fuse = C2f(2 * hidden_dim, hidden_dim, num_blocks, layer_norm=layer_norm)
        self.norm = get_norm("LN", hidden_dim)

    def forward(
        self, deep: torch.Tensor, cur: torch.Tensor, shallow: torch.Tensor
    ) -> torch.Tensor:
        aligned = cur + F.interpolate(
            self.deep_proj(deep),
            size=cur.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        detail = self.shallow_down(shallow)
        return self.norm(self.fuse(torch.cat([aligned, detail], dim=1)))


__all__ = ["SpatialDetailExtractor", "BiFusionBlock"]
