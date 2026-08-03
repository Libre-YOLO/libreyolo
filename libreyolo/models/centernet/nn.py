"""Native CenterNet detector networks.

The ResNet detector follows xingyizhou/CenterNet at commit
4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c (MIT). The DLA-34 backbone and
deformable-convolution components retain their BSD-3-Clause parameter layout.
LibreYOLO replaces the legacy compiled DCNv2 extension with torchvision's
BSD-3-Clause implementation. See the adjacent NOTICE for complete provenance.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d

BN_MOMENTUM = 0.1


def _fill_up_weights(layer: nn.ConvTranspose2d) -> None:
    """Initialize depthwise transposed convolutions as bilinear upsampling."""
    weight = layer.weight.data
    factor = math.ceil(weight.size(2) / 2)
    center = (2 * factor - 1 - factor % 2) / (2.0 * factor)
    for y in range(weight.size(2)):
        for x in range(weight.size(3)):
            weight[0, 0, y, x] = (1 - abs(y / factor - center)) * (
                1 - abs(x / factor - center)
            )
    for channel in range(1, weight.size(0)):
        weight[channel, 0].copy_(weight[0, 0])


def _fill_head_weights(layers: nn.Module) -> None:
    for layer in layers.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, std=0.001)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def _portable_deform_conv2d(
    x: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> torch.Tensor:
    """Express the one-group 3x3 DCNv2 operation with ONNX-portable ops."""
    batch, channels, height, width = x.shape
    out_height, out_width = offset.shape[-2:]
    kernel_height, kernel_width = weight.shape[-2:]
    kernel_points = kernel_height * kernel_width

    output_y = torch.arange(out_height, dtype=x.dtype, device=x.device)
    output_x = torch.arange(out_width, dtype=x.dtype, device=x.device)
    base_y = output_y * stride[0] - padding[0]
    base_x = output_x * stride[1] - padding[1]
    base_y, base_x = torch.meshgrid(base_y, base_x, indexing="ij")

    grids = []
    for point in range(kernel_points):
        kernel_y = (point // kernel_width) * dilation[0]
        kernel_x = (point % kernel_width) * dilation[1]
        sample_y = base_y + kernel_y + offset[:, point * 2]
        sample_x = base_x + kernel_x + offset[:, point * 2 + 1]
        sample_y = sample_y * (2.0 / (height - 1)) - 1.0
        sample_x = sample_x * (2.0 / (width - 1)) - 1.0
        grids.append(torch.stack((sample_x, sample_y), dim=-1))

    grid = torch.cat(grids, dim=1)
    sampled = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled.reshape(batch, channels, kernel_points, out_height, out_width)
    sampled = sampled * mask.unsqueeze(1)
    sampled = sampled.reshape(batch, channels * kernel_points, out_height, out_width)
    return F.conv2d(sampled, weight.reshape(weight.shape[0], -1, 1, 1), bias)


class DCN(nn.Module):
    """Modulated deformable convolution with the official DCNv2 key layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        deformable_groups: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
        if deformable_groups != 1 or self.kernel_size != (3, 3):
            raise ValueError("CenterNet's portable DCN path supports one 3x3 group")

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.portable = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_mask = self.conv_offset_mask(x)
        offset_first, offset_second, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_first, offset_second), dim=1)
        mask = torch.sigmoid(mask)
        if self.portable:
            return _portable_deform_conv2d(
                x,
                offset,
                mask,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
            )
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
        )


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class CenterNetResDCN(nn.Module):
    """CenterNet with the released ResNet-18 DCN upsampling head."""

    def __init__(self, num_classes: int = 80, head_conv: int = 64) -> None:
        super().__init__()
        self.inplanes = 64
        self.heads = {"hm": num_classes, "wh": 2, "reg": 2}
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.deconv_layers = self._make_deconv_layer()

        for name, channels in self.heads.items():
            head = nn.Sequential(
                nn.Conv2d(64, head_conv, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, channels, 1, bias=True),
            )
            if name == "hm":
                nn.init.constant_(head[-1].bias, -2.19)
            else:
                _fill_head_weights(head)
            setattr(self, name, head)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )
        layers: List[nn.Module] = [
            ResNetBasicBlock(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes
        layers.extend(ResNetBasicBlock(planes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _make_deconv_layer(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        for planes in (256, 128, 64):
            deform = DCN(self.inplanes, planes, 3, stride=1, padding=1)
            upsample = nn.ConvTranspose2d(
                planes, planes, 4, stride=2, padding=1, bias=False
            )
            _fill_up_weights(upsample)
            layers.extend(
                (
                    deform,
                    nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                    upsample,
                    nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                )
            )
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv_layers(x)
        return {name: getattr(self, name)(x) for name in self.heads}


class DLABasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            3,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x if residual is None else residual
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class Root(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, residual: bool
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        out = self.bn(self.conv(torch.cat(inputs, dim=1)))
        if self.residual:
            out = out + inputs[0]
        return self.relu(out)


class Tree(nn.Module):
    def __init__(
        self,
        levels: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        level_root: bool = False,
        root_dim: int = 0,
        root_kernel_size: int = 1,
        dilation: int = 1,
        root_residual: bool = False,
    ) -> None:
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = DLABasicBlock(
                in_channels, out_channels, stride, dilation=dilation
            )
            self.tree2 = DLABasicBlock(out_channels, out_channels, 1, dilation=dilation)
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        else:
            self.tree1 = Tree(
                levels - 1,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
            self.tree2 = Tree(
                levels - 1,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
        self.level_root = level_root
        self.levels = levels
        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else None
        self.project = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )
            if in_channels != out_channels
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        children: List[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del residual
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample is not None else x
        residual = self.project(bottom) if self.project is not None else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            return self.root(x2, x1, *children)
        children.append(x1)
        return self.tree2(x1, children=children)


class DLA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        levels = (1, 1, 1, 2, 2, 1)
        channels = (16, 32, 64, 128, 256, 512)
        self.channels = list(channels)
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], 7, padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2
        )
        self.level2 = Tree(levels[2], channels[1], channels[2], 2)
        self.level3 = Tree(levels[3], channels[2], channels[3], 2, level_root=True)
        self.level4 = Tree(levels[4], channels[3], channels[4], 2, level_root=True)
        self.level5 = Tree(levels[5], channels[4], channels[5], 2, level_root=True)
        # The official detector checkpoint retains this unused ImageNet head.
        self.fc = nn.Conv2d(channels[-1], 1000, 1, bias=True)

    @staticmethod
    def _make_conv_level(
        inplanes: int, planes: int, convs: int, stride: int = 1
    ) -> nn.Sequential:
        modules: List[nn.Module] = []
        for index in range(convs):
            modules.extend(
                (
                    nn.Conv2d(
                        inplanes,
                        planes,
                        3,
                        stride=stride if index == 0 else 1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                )
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        x = self.base_layer(x)
        for level in range(6):
            x = getattr(self, f"level{level}")(x)
            outputs.append(x)
        return outputs


class DeformConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.conv = DCN(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actf(self.conv(x))


class IDAUp(nn.Module):
    def __init__(
        self, out_channels: int, channels: Sequence[int], up_factors: Sequence[int]
    ) -> None:
        super().__init__()
        for index in range(1, len(channels)):
            channels_i = channels[index]
            factor = int(up_factors[index])
            project = DeformConv(channels_i, out_channels)
            node = DeformConv(out_channels, out_channels)
            upsample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                factor * 2,
                stride=factor,
                padding=factor // 2,
                groups=out_channels,
                bias=False,
            )
            _fill_up_weights(upsample)
            setattr(self, f"proj_{index}", project)
            setattr(self, f"up_{index}", upsample)
            setattr(self, f"node_{index}", node)

    def forward(self, layers: List[torch.Tensor], start: int, end: int) -> None:
        for index in range(start + 1, end):
            relative = index - start
            project = getattr(self, f"proj_{relative}")
            upsample = getattr(self, f"up_{relative}")
            node = getattr(self, f"node_{relative}")
            layers[index] = upsample(project(layers[index]))
            layers[index] = node(layers[index] + layers[index - 1])


class DLAUp(nn.Module):
    def __init__(
        self,
        start: int,
        channels: Sequence[int],
        scales: Sequence[int],
        in_channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.startp = start
        mutable_channels = list(channels)
        mutable_inputs = list(channels if in_channels is None else in_channels)
        mutable_scales = list(scales)
        for index in range(len(mutable_channels) - 1):
            source = -index - 2
            relative_scales = [
                scale // mutable_scales[source] for scale in mutable_scales[source:]
            ]
            setattr(
                self,
                f"ida_{index}",
                IDAUp(
                    mutable_channels[source],
                    mutable_inputs[source:],
                    relative_scales,
                ),
            )
            for target in range(source + 1, 0):
                mutable_scales[target] = mutable_scales[source]
                mutable_inputs[target] = mutable_channels[source]

    def forward(self, layers: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = [layers[-1]]
        for index in range(len(layers) - self.startp - 1):
            ida = getattr(self, f"ida_{index}")
            ida(layers, len(layers) - index - 2, len(layers))
            outputs.insert(0, layers[-1])
        return outputs


class CenterNetDLA(nn.Module):
    """CenterNet with the released DLA-34 aggregation head."""

    def __init__(self, num_classes: int = 80, head_conv: int = 256) -> None:
        super().__init__()
        self.first_level = 2
        self.last_level = 5
        self.base = DLA()
        channels = self.base.channels
        scales = [2**index for index in range(len(channels[self.first_level :]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level :], scales)
        self.ida_up = IDAUp(
            channels[self.first_level],
            channels[self.first_level : self.last_level],
            [2**index for index in range(self.last_level - self.first_level)],
        )
        self.heads = {"hm": num_classes, "wh": 2, "reg": 2}
        for name, out_channels in self.heads.items():
            head = nn.Sequential(
                nn.Conv2d(
                    channels[self.first_level], head_conv, 3, padding=1, bias=True
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, out_channels, 1, bias=True),
            )
            if name == "hm":
                nn.init.constant_(head[-1].bias, -2.19)
            else:
                for layer in head.modules():
                    if isinstance(layer, nn.Conv2d) and layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            setattr(self, name, head)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.dla_up(self.base(x))
        fused = [
            features[index].clone()
            for index in range(self.last_level - self.first_level)
        ]
        self.ida_up(fused, 0, len(fused))
        return {name: getattr(self, name)(fused[-1]) for name in self.heads}


def build_centernet(size: str, num_classes: int = 80) -> nn.Module:
    """Build one of the two released CenterNet detection variants."""
    if size == "resdcn18":
        return CenterNetResDCN(num_classes=num_classes, head_conv=64)
    if size == "dla34":
        return CenterNetDLA(num_classes=num_classes, head_conv=256)
    raise ValueError(f"Unsupported CenterNet size {size!r}")


def set_portable_dcn(model: nn.Module, enabled: bool) -> None:
    """Select the export-safe DCN implementation for every CenterNet DCN."""
    for module in model.modules():
        if isinstance(module, DCN):
            module.portable = enabled


class CenterNetExportWrapper(nn.Module):
    """Bake CenterNet peak selection and stride scaling into exported graphs."""

    def __init__(self, model: nn.Module, topk: int = 100) -> None:
        super().__init__()
        self.model = model
        self.topk = topk
        set_portable_dcn(self.model, True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        from ...postprocess.centernet import decode_centernet

        output = self.model(images)
        return decode_centernet(
            output["hm"], output["wh"], output["reg"], topk=self.topk
        ) * output["hm"].new_tensor([4.0, 4.0, 4.0, 4.0, 1.0, 1.0])


__all__ = [
    "CenterNetDLA",
    "CenterNetExportWrapper",
    "CenterNetResDCN",
    "DCN",
    "build_centernet",
    "set_portable_dcn",
]
