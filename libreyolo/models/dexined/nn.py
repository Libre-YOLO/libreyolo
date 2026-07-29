"""Native PyTorch DexiNed architecture.

Ported from xavysp/DexiNed at commit
08ed67ad0579f3969536a9719cdc1b829fb74fc1 (MIT). See ``NOTICE`` in this
directory and the repository ``THIRD_PARTY_NOTICES.txt``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _initialize(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


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
        self.add_module("norm1", nn.BatchNorm2d(output_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
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
        self.add_module("norm2", nn.BatchNorm2d(output_features))

    def forward(
        self,
        values: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        primary, residual = values
        features = super().forward(F.relu(primary))
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
            self.add_module(
                f"denselayer{index + 1}",
                _DenseLayer(input_features, output_features),
            )
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
                    nn.ReLU(inplace=True),
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
        use_bs: bool = True,
    ):
        super().__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(
            input_features,
            output_features,
            kernel_size=1,
            stride=stride,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(output_features)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv(value)
        return self.bn(value) if self.use_bn else value


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
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(
            mid_features,
            output_features,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(output_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        return self.relu(value) if self.use_act else value


class DexiNedCore(nn.Module):
    """DexiNed base network returning six side logits plus fusion."""

    def __init__(self):
        super().__init__()
        self.block_1 = DoubleConvBlock(3, 32, 64, stride=2)
        self.block_2 = DoubleConvBlock(64, 128, use_act=False)
        self.dblock_3 = _DenseBlock(2, 128, 256)
        self.dblock_4 = _DenseBlock(3, 256, 512)
        self.dblock_5 = _DenseBlock(3, 512, 512)
        self.dblock_6 = _DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.side_1 = SingleConvBlock(64, 128, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1)
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)

        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False)
        self.apply(_initialize)

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        if value.ndim != 4:
            raise ValueError(f"DexiNed expects BCHW input, got {tuple(value.shape)}")

        block_1 = self.block_1(value)
        block_1_side = self.side_1(block_1)

        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3((block_2_add, block_3_pre_dense))
        block_3_down = self.maxpool(block_3)
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)

        block_2_resize_half = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_3_down + block_2_resize_half)
        block_4, _ = self.dblock_4((block_3_add, block_4_pre_dense))
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        block_5_pre_dense = self.pre_dense_5(block_4_down)
        block_5, _ = self.dblock_5((block_4_add, block_5_pre_dense))
        block_5_add = block_5 + block_4_side

        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6((block_5_add, block_6_pre_dense))

        outputs = [
            self.up_block_1(block_1),
            self.up_block_2(block_2),
            self.up_block_3(block_3),
            self.up_block_4(block_4),
            self.up_block_5(block_5),
            self.up_block_6(block_6),
        ]
        outputs.append(self.block_cat(torch.cat(outputs, dim=1)))
        return outputs


__all__ = ["DexiNedCore"]
