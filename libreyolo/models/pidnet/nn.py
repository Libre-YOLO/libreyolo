# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
"""PIDNet network definition for LibreYOLO.

This file ports the MIT-licensed PIDNet PyTorch modules from
https://github.com/XuJiacong/PIDNet. Module names intentionally stay close to
upstream so Cityscapes checkpoints can be converted with a small state-dict
filter instead of a broad key remap.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


BatchNorm2d = nn.BatchNorm2d
BN_MOM = 0.1
ALIGN_CORNERS = False


@dataclass(frozen=True)
class PIDNetConfig:
    m: int
    n: int
    planes: int
    ppm_planes: int
    head_planes: int


SIZE_CONFIGS = {
    "s": PIDNetConfig(m=2, n=3, planes=32, ppm_planes=96, head_planes=128),
    "m": PIDNetConfig(m=2, n=3, planes=64, ppm_planes=96, head_planes=128),
    "l": PIDNetConfig(m=3, n=4, planes=64, ppm_planes=112, head_planes=256),
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        no_relu: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOM)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        if self.no_relu:
            return out
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        no_relu: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=BN_MOM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

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

        out = out + residual
        if self.no_relu:
            return out
        return self.relu(out)


class SegmentHead(nn.Module):
    def __init__(
        self,
        inplanes: int,
        interplanes: int,
        outplanes: int,
        scale_factor: int | None = None,
    ) -> None:
        super().__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=BN_MOM)
        self.conv1 = nn.Conv2d(
            inplanes, interplanes, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(interplanes, momentum=BN_MOM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(
                out,
                size=[height, width],
                mode="bilinear",
                align_corners=ALIGN_CORNERS,
            )

        return out


class DAPPM(nn.Module):
    def __init__(
        self,
        inplanes: int,
        branch_planes: int,
        outplanes: int,
        batch_norm: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            batch_norm(inplanes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            batch_norm(inplanes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            batch_norm(inplanes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            batch_norm(inplanes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            batch_norm(inplanes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            batch_norm(branch_planes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            batch_norm(branch_planes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            batch_norm(branch_planes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            batch_norm(branch_planes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            batch_norm(branch_planes * 5, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            batch_norm(inplanes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = [self.scale0(x)]
        x_list.append(
            self.process1(
                F.interpolate(
                    self.scale1(x),
                    size=[height, width],
                    mode="bilinear",
                    align_corners=ALIGN_CORNERS,
                )
                + x_list[0]
            )
        )
        x_list.append(
            self.process2(
                F.interpolate(
                    self.scale2(x),
                    size=[height, width],
                    mode="bilinear",
                    align_corners=ALIGN_CORNERS,
                )
                + x_list[1]
            )
        )
        x_list.append(
            self.process3(
                F.interpolate(
                    self.scale3(x),
                    size=[height, width],
                    mode="bilinear",
                    align_corners=ALIGN_CORNERS,
                )
                + x_list[2]
            )
        )
        x_list.append(
            self.process4(
                F.interpolate(
                    self.scale4(x),
                    size=[height, width],
                    mode="bilinear",
                    align_corners=ALIGN_CORNERS,
                )
                + x_list[3]
            )
        )

        return self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)


class PAPPM(nn.Module):
    def __init__(
        self,
        inplanes: int,
        branch_planes: int,
        outplanes: int,
        batch_norm: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            batch_norm(inplanes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            batch_norm(inplanes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            batch_norm(inplanes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            batch_norm(inplanes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            batch_norm(inplanes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale_process = nn.Sequential(
            batch_norm(branch_planes * 4, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_planes * 4,
                branch_planes * 4,
                kernel_size=3,
                padding=1,
                groups=4,
                bias=False,
            ),
        )
        self.compression = nn.Sequential(
            batch_norm(branch_planes * 5, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            batch_norm(inplanes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = x.shape[-1]
        height = x.shape[-2]

        x_ = self.scale0(x)
        scale_list = [
            F.interpolate(
                self.scale1(x),
                size=[height, width],
                mode="bilinear",
                align_corners=ALIGN_CORNERS,
            )
            + x_,
            F.interpolate(
                self.scale2(x),
                size=[height, width],
                mode="bilinear",
                align_corners=ALIGN_CORNERS,
            )
            + x_,
            F.interpolate(
                self.scale3(x),
                size=[height, width],
                mode="bilinear",
                align_corners=ALIGN_CORNERS,
            )
            + x_,
            F.interpolate(
                self.scale4(x),
                size=[height, width],
                mode="bilinear",
                align_corners=ALIGN_CORNERS,
            )
            + x_,
        ]
        scale_out = self.scale_process(torch.cat(scale_list, dim=1))
        return self.compression(torch.cat([x_, scale_out], dim=1)) + self.shortcut(x)


class PagFM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        after_relu: bool = False,
        with_channel: bool = False,
        batch_norm: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            batch_norm(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            batch_norm(mid_channels),
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                batch_norm(in_channels),
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(
            y_q, size=[input_size[2], input_size[3]], mode="bilinear", align_corners=False
        )
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(
            y, size=[input_size[2], input_size[3]], mode="bilinear", align_corners=False
        )
        return (1 - sim_map) * x + sim_map * y


class LightBag(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            batch_norm(out_channels),
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            batch_norm(out_channels),
        )

    def forward(self, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        edge_att = torch.sigmoid(d)
        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)
        return p_add + i_add


class Bag(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            batch_norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)


class LibrePIDNetNet(nn.Module):
    """PIDNet S/M/L semantic segmentation network.

    Inputs are expected as RGB float tensors in [0, 1]. The network applies the
    ImageNet normalization used by upstream PIDNet internally; those buffers are
    non-persistent so converted state dict keys stay upstream-compatible.
    """

    def __init__(self, size: str = "s", num_classes: int = 19) -> None:
        super().__init__()
        if size not in SIZE_CONFIGS:
            raise ValueError(f"Unknown PIDNet size {size!r}")
        cfg = SIZE_CONFIGS[size]
        self.size = size
        self.num_classes = int(num_classes)
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        m = cfg.m
        n = cfg.n
        planes = cfg.planes
        ppm_planes = cfg.ppm_planes
        head_planes = cfg.head_planes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=BN_MOM),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=BN_MOM),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes, momentum=BN_MOM),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=BN_MOM),
            )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = LightBag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=BN_MOM),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=BN_MOM),
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        self.final_layer = SegmentHead(planes * 4, head_planes, self.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOM),
            )

        layers: list[nn.Module] = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, no_relu=i == blocks - 1))
        return nn.Sequential(*layers)

    def _make_single_layer(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        inplanes: int,
        planes: int,
        stride: int = 1,
    ) -> nn.Module:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOM),
            )
        return block(inplanes, planes, stride, downsample, no_relu=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean.to(dtype=x.dtype)) / self.pixel_std.to(dtype=x.dtype)
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(
            self.diff3(x),
            size=[height_output, width_output],
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(
            self.diff4(x),
            size=[height_output, width_output],
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(
            self.spp(self.layer5(x)),
            size=[height_output, width_output],
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )

        return self.final_layer(self.dfm(x_, x, x_d))


__all__ = [
    "PIDNetConfig",
    "SIZE_CONFIGS",
    "LibrePIDNetNet",
]
