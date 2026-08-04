"""Native PyTorch TEED architecture.

Ported from xavysp/TEED at commit
40fa4b1391dc6424f88989d0ca75d5b592c8681d (MIT). See ``NOTICE`` in this
directory and the repository ``THIRD_PARTY_NOTICES.txt``.
"""

from __future__ import annotations

import torch
from torch import nn


def smish(value: torch.Tensor) -> torch.Tensor:
    """Smish activation used by the released TEED checkpoint."""
    return value * torch.tanh(torch.log(1.0 + torch.sigmoid(value)))


class Smish(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return smish(value)


def _initialize(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DoubleFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        del out_channels
        self.DWconv1 = nn.Conv2d(
            in_channels,
            in_channels * 8,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
        )
        # PixelShuffle(1) is an exact identity, but ONNX represents it as
        # DepthToSpace and LiteRT rejects the converter's custom op.
        self.PSconv1 = nn.Identity()
        self.DWconv2 = nn.Conv2d(
            24,
            24,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=24,
        )
        self.AF = Smish()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        attention = self.PSconv1(self.DWconv1(self.AF(value)))
        refined = self.PSconv1(self.DWconv2(self.AF(attention)))
        return smish((refined + attention).sum(dim=1, keepdim=True))


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.add_module(
            "conv1",
            nn.Conv2d(
                input_features,
                output_features,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True,
            ),
        )
        self.add_module("smish1", Smish())
        self.add_module(
            "conv2",
            nn.Conv2d(
                output_features,
                output_features,
                kernel_size=3,
                stride=1,
                bias=True,
            ),
        )

    def forward(
        self,
        values: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        primary, residual = values
        features = super().forward(smish(primary))
        return 0.5 * (features + residual), residual


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        num_layers: int,
        input_features: int,
        output_features: int,
    ):
        super().__init__()
        for index in range(num_layers):
            layer = _DenseLayer(input_features, output_features)
            self.add_module(f"denselayer{index + 1}", layer)
            input_features = output_features


class UpConvBlock(nn.Module):
    _PADDING = (0, 0, 1, 3, 7)

    def __init__(self, input_features: int, up_scale: int):
        super().__init__()
        layers = []
        for index in range(up_scale):
            output_features = 1 if index == up_scale - 1 else 16
            layers.extend(
                [
                    nn.Conv2d(input_features, output_features, kernel_size=1),
                    Smish(),
                    nn.ConvTranspose2d(
                        output_features,
                        output_features,
                        kernel_size=2**up_scale,
                        stride=2,
                        padding=self._PADDING[up_scale],
                    ),
                ]
            )
            input_features = output_features
        self.features = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.features(value)


class SingleConvBlock(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        stride: int,
        use_ac: bool = False,
    ):
        super().__init__()
        self.use_ac = use_ac
        self.conv = nn.Conv2d(
            input_features,
            output_features,
            kernel_size=1,
            stride=stride,
            bias=True,
        )
        if use_ac:
            self.smish = Smish()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv(value)
        return self.smish(value) if self.use_ac else value


class DoubleConvBlock(nn.Module):
    def __init__(
        self,
        input_features: int,
        mid_features: int,
        output_features: int | None = None,
        stride: int = 1,
        use_act: bool = True,
    ):
        super().__init__()
        self.use_act = use_act
        output_features = output_features or mid_features
        self.conv1 = nn.Conv2d(
            input_features,
            mid_features,
            kernel_size=3,
            padding=1,
            stride=stride,
        )
        self.conv2 = nn.Conv2d(
            mid_features,
            output_features,
            kernel_size=3,
            padding=1,
        )
        self.smish = Smish()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.smish(self.conv1(value))
        value = self.conv2(value)
        return self.smish(value) if self.use_act else value


class TEEDCore(nn.Module):
    """Tiny Efficient Edge Detector returning three side logits plus fusion."""

    def __init__(self):
        super().__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)
        self.block_cat = DoubleFusion(3, 3)
        self.apply(_initialize)

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        if value.ndim != 4:
            raise ValueError(f"TEED expects BCHW input, got {tuple(value.shape)}")

        block_1 = self.block_1(value)
        block_1_side = self.side_1(block_1)
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3((block_2_add, block_3_pre_dense))

        outputs = [
            self.up_block_1(block_1),
            self.up_block_2(block_2),
            self.up_block_3(block_3),
        ]
        outputs.append(self.block_cat(torch.cat(outputs, dim=1)))
        return outputs


__all__ = ["Smish", "TEEDCore", "smish"]
