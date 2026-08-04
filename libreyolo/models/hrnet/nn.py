"""Native HRNet pose graph.

Adapted from ``lib/models/pose_hrnet.py`` in
``leoxiaobin/deep-high-resolution-net.pytorch`` at commit
``6f69e4676ad8d43d0d61b64b1b9726f0c369e7b1`` (MIT License).
Attribute names and forward arithmetic intentionally match upstream so the
official state dictionaries load strictly and inference is bit-identical.

Copyright (c) Microsoft. Written by Bin Xiao.
"""

from __future__ import annotations

import logging
from typing import TypeAlias

import torch
from torch import nn

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Return the upstream 3x3 convolution."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """Two-convolution residual block used in every parallel HRNet branch."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    """ResNet bottleneck used by the stride-four stem."""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


Block: TypeAlias = type[BasicBlock] | type[Bottleneck]


class HighResolutionModule(nn.Module):
    """Parallel resolution branches plus repeated sum fusion."""

    def __init__(
        self,
        num_branches: int,
        blocks: Block,
        num_blocks: list[int],
        num_inchannels: list[int],
        num_channels: list[int],
        fuse_method: str,
        multi_scale_output: bool = True,
    ) -> None:
        super().__init__()
        self._check_branches(
            num_branches,
            num_blocks,
            num_inchannels,
            num_channels,
        )
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    @staticmethod
    def _check_branches(
        num_branches: int,
        num_blocks: list[int],
        num_inchannels: list[int],
        num_channels: list[int],
    ) -> None:
        fields = {
            "NUM_BLOCKS": num_blocks,
            "NUM_CHANNELS": num_channels,
            "NUM_INCHANNELS": num_inchannels,
        }
        for name, values in fields.items():
            if num_branches != len(values):
                message = f"NUM_BRANCHES({num_branches}) <> {name}({len(values)})"
                logger.error(message)
                raise ValueError(message)

    def _make_one_branch(
        self,
        branch_index: int,
        block: Block,
        num_blocks: list[int],
        num_channels: list[int],
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        out_channels = num_channels[branch_index] * block.expansion
        if stride != 1 or self.num_inchannels[branch_index] != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )

        layers: list[nn.Module] = [
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        ]
        self.num_inchannels[branch_index] = out_channels
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                )
            )
        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches: int,
        block: Block,
        num_blocks: list[int],
        num_channels: list[int],
    ) -> nn.ModuleList:
        return nn.ModuleList(
            [
                self._make_one_branch(i, block, num_blocks, num_channels)
                for i in range(num_branches)
            ]
        )

    def _make_fuse_layers(self) -> nn.ModuleList | None:
        if self.num_branches == 1:
            return None

        fuse_layers: list[nn.ModuleList] = []
        output_branches = self.num_branches if self.multi_scale_output else 1
        for i in range(output_branches):
            fuse_layer: list[nn.Module | None] = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                self.num_inchannels[j],
                                self.num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(self.num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s: list[nn.Module] = []
                    for k in range(i - j):
                        is_last = k == i - j - 1
                        out_channels = (
                            self.num_inchannels[i]
                            if is_last
                            else self.num_inchannels[j]
                        )
                        parts: list[nn.Module] = [
                            nn.Conv2d(
                                self.num_inchannels[j],
                                out_channels,
                                3,
                                2,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(out_channels),
                        ]
                        if not is_last:
                            parts.append(nn.ReLU(True))
                        conv3x3s.append(nn.Sequential(*parts))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> list[int]:
        """Return branch widths after constructing this module."""
        return self.num_inchannels

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        if self.fuse_layers is None:  # pragma: no cover - guarded above
            return x
        x_fuse: list[torch.Tensor] = []
        for i in range(len(self.fuse_layers)):
            first = self.fuse_layers[i][0]
            y = x[0] if i == 0 else first(x[0])
            for j in range(1, self.num_branches):
                layer = self.fuse_layers[i][j]
                y = y + x[j] if i == j else y + layer(x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


BLOCKS: dict[str, Block] = {
    "BASIC": BasicBlock,
    "BOTTLENECK": Bottleneck,
}


def _stage_config(width: int) -> dict[str, dict]:
    return {
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4],
            "NUM_CHANNELS": [width, width * 2],
            "FUSE_METHOD": "SUM",
        },
        "STAGE3": {
            "NUM_MODULES": 4,
            "NUM_BRANCHES": 3,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4],
            "NUM_CHANNELS": [width, width * 2, width * 4],
            "FUSE_METHOD": "SUM",
        },
        "STAGE4": {
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 4,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4, 4],
            "NUM_CHANNELS": [width, width * 2, width * 4, width * 8],
            "FUSE_METHOD": "SUM",
        },
    }


class HRNetPoseModel(nn.Module):
    """High-Resolution Net with a COCO keypoint heatmap head."""

    def __init__(self, width: int, num_keypoints: int = 17) -> None:
        super().__init__()
        if width not in (32, 48):
            raise ValueError(f"HRNet pose width must be 32 or 48, got {width}")
        self.inplanes = 64
        extra = _stage_config(width)

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra["STAGE2"]
        num_channels = self._expanded_channels(self.stage2_cfg)
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg,
            num_channels,
        )

        self.stage3_cfg = extra["STAGE3"]
        num_channels = self._expanded_channels(self.stage3_cfg)
        self.transition2 = self._make_transition_layer(
            pre_stage_channels,
            num_channels,
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg,
            num_channels,
        )

        self.stage4_cfg = extra["STAGE4"]
        num_channels = self._expanded_channels(self.stage4_cfg)
        self.transition3 = self._make_transition_layer(
            pre_stage_channels,
            num_channels,
        )
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multi_scale_output=False,
        )

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    @staticmethod
    def _expanded_channels(stage_cfg: dict) -> list[int]:
        block = BLOCKS[stage_cfg["BLOCK"]]
        return [channel * block.expansion for channel in stage_cfg["NUM_CHANNELS"]]

    @staticmethod
    def _make_transition_layer(
        num_channels_pre_layer: list[int],
        num_channels_cur_layer: list[int],
    ) -> nn.ModuleList:
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers: list[nn.Module | None] = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s: list[nn.Module] = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else in_channels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels,
                                out_channels,
                                3,
                                2,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(
        self,
        block: Block,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    planes * block.expansion,
                    momentum=BN_MOMENTUM,
                ),
            )

        layers: list[nn.Module] = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_stage(
        layer_config: dict,
        num_inchannels: list[int],
        multi_scale_output: bool = True,
    ) -> tuple[nn.Sequential, list[int]]:
        modules: list[nn.Module] = []
        for i in range(layer_config["NUM_MODULES"]):
            output_all_scales = not (
                not multi_scale_output and i == layer_config["NUM_MODULES"] - 1
            )
            module = HighResolutionModule(
                layer_config["NUM_BRANCHES"],
                BLOCKS[layer_config["BLOCK"]],
                layer_config["NUM_BLOCKS"],
                num_inchannels,
                layer_config["NUM_CHANNELS"],
                layer_config["FUSE_METHOD"],
                output_all_scales,
            )
            modules.append(module)
            num_inchannels = module.get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list: list[torch.Tensor] = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            transition = self.transition1[i]
            x_list.append(transition(x) if transition is not None else x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            transition = self.transition2[i]
            x_list.append(transition(y_list[-1]) if transition is not None else y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            transition = self.transition3[i]
            x_list.append(transition(y_list[-1]) if transition is not None else y_list[i])
        y_list = self.stage4(x_list)
        return self.final_layer(y_list[0])


__all__ = [
    "BasicBlock",
    "Bottleneck",
    "HighResolutionModule",
    "HRNetPoseModel",
]
