"""RT-DETRv2 hybrid encoder variant used by the released OBB baselines.

The identity BatchNorm branch follows ``engine/rtv4/hybrid_encoder.py`` from
``RicePasteM/RiO-DETR`` commit
``22d5232a4e0df6ac4bc26ed1c8aac8b4060449c7`` (Apache-2.0).  The surrounding
AIFI/FPN/PAN implementation is reused from LibreYOLO's RT-DETR encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..rtdetr.nn import CSPRepLayer, HybridEncoder, RepVggBlock


class OBBRepVggBlock(RepVggBlock):
    """RepVGG block with the identity BN branch present in RT-DETRv2 OBB."""

    def __init__(self, ch_in: int, ch_out: int, act: str = "relu"):
        super().__init__(ch_in, ch_out, act=act)
        self.identity = nn.BatchNorm2d(ch_in) if ch_in == ch_out else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
            if self.identity is not None:
                y = y + self.identity(x)
        return self.act(y)


class OBBCSPRepLayer(CSPRepLayer):
    """CSP fusion layer built from OBB RepVGG blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        bias=None,
        act: str = "silu",
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_blocks=num_blocks,
            expansion=expansion,
            bias=bias,
            act=act,
        )
        hidden_channels = int(out_channels * expansion)
        self.bottlenecks = nn.Sequential(
            *[
                OBBRepVggBlock(hidden_channels, hidden_channels, act=act)
                for _ in range(num_blocks)
            ]
        )


class RTDETRv2OBBHybridEncoder(HybridEncoder):
    """Horizontal RT-DETR encoder with the released OBB fusion blocks."""

    def __init__(self, *args, **kwargs):
        expansion = float(kwargs.get("expansion", 1.0))
        depth_mult = float(kwargs.get("depth_mult", 1.0))
        act = kwargs.get("act", "silu")
        super().__init__(*args, **kwargs)

        block_count = round(3 * depth_mult)
        level_count = len(self.in_channels) - 1
        self.fpn_blocks = nn.ModuleList(
            [
                OBBCSPRepLayer(
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    block_count,
                    expansion=expansion,
                    act=act,
                )
                for _ in range(level_count)
            ]
        )
        self.pan_blocks = nn.ModuleList(
            [
                OBBCSPRepLayer(
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    block_count,
                    expansion=expansion,
                    act=act,
                )
                for _ in range(level_count)
            ]
        )


__all__ = ["RTDETRv2OBBHybridEncoder"]
