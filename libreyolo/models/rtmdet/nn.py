"""
RTMDet neural network architecture for LibreYOLO.

Ported from open-mmlab/mmdetection (Apache-2.0). Mirrors the upstream
attribute names so the conversion script is a metadata-wrap rather than a
structural rewrite:

    backbone.stem.0.conv.weight   <-  backbone.stem.0.conv.weight
    neck.reduce_layers.0.conv.*   <-  neck.reduce_layers.0.conv.*
    head.rtm_cls.0.weight         <-  bbox_head.rtm_cls.0.weight   (head only renames bbox_head -> head)

Two divergences vs mmyolo (deliberate, to stay weight-compatible with the published
COCO checkpoints which were trained with mmdet):

1. SPP block uses 3 parallel max-pools (kernels 5, 9, 13). mmyolo's SPPF uses a
   single 5x5 max-pool applied 3 times sequentially: not weight-compatible.
2. RTMDetSepBNHead does not have per-level Scale parameters (those belong to
   the parent RTMDetHead, which is not what the published weights use).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# IMPORTANT: the mmdet RTMDet config sets ``norm_cfg=dict(type='SyncBN')`` and
# does NOT pass ``momentum`` / ``eps``. SyncBN therefore uses PyTorch defaults,
# and the saved checkpoint records BN buffers under those defaults. At inference
# mmdet swaps SyncBN for plain BN but preserves the same eps/momentum. So the
# numerical-parity values are 1e-5 / 0.1, NOT the (1e-3, 0.03) that appear in
# cspnext.py's *docstring default* (which is shadowed by the config override).
# Baking the docstring values cost ~7 mAP at COCO eval before this was caught.
_BN_MOMENTUM = 0.1
_BN_EPS = 1e-5


# =============================================================================
# Building blocks
# =============================================================================


class ConvBNAct(nn.Module):
    """Conv + BN + activation, equivalent to mmcv.cnn.ConvModule for the BN+SiLU path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM, eps=_BN_EPS)
        self.activate = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise + pointwise conv, both BN+SiLU.

    Mirrors mmcv.cnn.DepthwiseSeparableConvModule with norm/act on each branch.
    Not used for the published RTMDet sizes (t/s/m/l/x all set use_depthwise=False)
    but kept for completeness so the head's pred-kernel and depthwise variants work.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.depthwise_conv = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise_conv = ConvBNAct(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class ChannelAttention(nn.Module):
    """Squeeze-and-excite attention used inside CSPLayer when channel_attention=True."""

    def __init__(self, channels: int):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class CSPNeXtBlock(nn.Module):
    """The basic bottleneck used in CSPNeXt: 3x3 then depthwise-separable 5x5."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        kernel_size: int = 5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConv(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv2(self.conv1(x))
        if self.add_identity:
            return out + identity
        return out


class CSPLayer(nn.Module):
    """Cross Stage Partial layer with optional channel attention.

    Same module is used in backbone stages and in PAFPN stages.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        add_identity: bool = True,
        expand_ratio: float = 0.5,
        channel_attention: bool = True,
    ):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvBNAct(in_channels, mid_channels, 1)
        self.short_conv = ConvBNAct(in_channels, mid_channels, 1)
        self.final_conv = ConvBNAct(2 * mid_channels, out_channels, 1)

        self.blocks = nn.Sequential(
            *[
                CSPNeXtBlock(mid_channels, mid_channels, 1.0, add_identity)
                for _ in range(num_blocks)
            ]
        )
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling, mmdet-flavor: 3 parallel max-pools (5, 9, 13).

    Do NOT replace with mmyolo's SPPFBottleneck (sequential single-kernel) -
    the published RTMDet weights are not compatible with that variant.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (5, 9, 13),
    ):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvBNAct(in_channels, mid_channels, 1)
        self.poolings = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvBNAct(conv2_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.poolings], dim=1)
        return self.conv2(x)


# =============================================================================
# Backbone, neck, head
# =============================================================================

# (in_channels, out_channels, num_blocks, add_identity, use_spp) at widen=1.0
_ARCH_P5 = [
    (64, 128, 3, True, False),
    (128, 256, 6, True, False),
    (256, 512, 6, True, False),
    (512, 1024, 3, False, True),
]


class CSPNeXt(nn.Module):
    """CSPNeXt backbone (P5 only — no P6 weights are released for stock RTMDet detect)."""

    def __init__(
        self,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Tuple[int, ...] = (2, 3, 4),
        expand_ratio: float = 0.5,
        channel_attention: bool = True,
    ):
        super().__init__()
        self.out_indices = out_indices

        stem_out = int(_ARCH_P5[0][0] * widen_factor // 2)
        stem_full = int(_ARCH_P5[0][0] * widen_factor)
        self.stem = nn.Sequential(
            ConvBNAct(3, stem_out, 3, stride=2, padding=1),
            ConvBNAct(stem_out, stem_out, 3, stride=1, padding=1),
            ConvBNAct(stem_out, stem_full, 3, stride=1, padding=1),
        )

        self.layer_names = ["stem"]
        for i, (in_ch, out_ch, n_blocks, add_id, use_spp) in enumerate(_ARCH_P5):
            in_ch = int(in_ch * widen_factor)
            out_ch = int(out_ch * widen_factor)
            n_blocks = max(round(n_blocks * deepen_factor), 1)

            stage = []
            stage.append(ConvBNAct(in_ch, out_ch, 3, stride=2, padding=1))
            if use_spp:
                stage.append(SPPBottleneck(out_ch, out_ch, kernel_sizes=(5, 9, 13)))
            stage.append(
                CSPLayer(
                    out_ch,
                    out_ch,
                    num_blocks=n_blocks,
                    add_identity=add_id,
                    expand_ratio=expand_ratio,
                    channel_attention=channel_attention,
                )
            )
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layer_names.append(f"stage{i + 1}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs = []
        for i, name in enumerate(self.layer_names):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class CSPNeXtPAFPN(nn.Module):
    """Path Aggregation FPN with CSPNeXt blocks. Strides 8, 16, 32 by default."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_csp_blocks: int = 3,
        expand_ratio: float = 0.5,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        n = len(self.in_channels)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(n - 1, 0, -1):
            self.reduce_layers.append(
                ConvBNAct(self.in_channels[idx], self.in_channels[idx - 1], 1)
            )
            self.top_down_blocks.append(
                CSPLayer(
                    self.in_channels[idx - 1] * 2,
                    self.in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    expand_ratio=expand_ratio,
                    channel_attention=False,
                )
            )

        # Bottom-up
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(n - 1):
            self.downsamples.append(
                ConvBNAct(
                    self.in_channels[idx], self.in_channels[idx], 3, stride=2, padding=1
                )
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    self.in_channels[idx] * 2,
                    self.in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    expand_ratio=expand_ratio,
                    channel_attention=False,
                )
            )

        self.out_convs = nn.ModuleList(
            [
                ConvBNAct(self.in_channels[i], out_channels, 3, padding=1)
                for i in range(n)
            ]
        )

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.in_channels)
        n = len(self.in_channels)

        # top-down
        inner_outs = [inputs[-1]]
        for i, idx in enumerate(range(n - 1, 0, -1)):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[i](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = self.upsample(feat_high)
            inner_out = self.top_down_blocks[i](
                torch.cat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        # bottom-up
        outs = [inner_outs[0]]
        for idx in range(n - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_high], dim=1)
            )
            outs.append(out)

        # final 3x3 convs to out_channels
        return tuple(self.out_convs[i](outs[i]) for i in range(n))


class RTMDetSepBNHead(nn.Module):
    """Decoupled cls/reg head with per-level BN and (optionally) shared cross-level convs.

    Important details that match the published checkpoints:

    - num_base_priors == 1 (point-based, no anchors).
    - pred_kernel_size == 1.
    - share_conv == True: cross-level cls_convs[n][i].conv aliases cls_convs[0][i].conv.
      Per-level BNs stay distinct. The state_dict only stores conv weights for level 0;
      our converter has to mirror that aliasing pattern.
    - exp_on_reg: tiny/s use False, m/l/x use True. The reg branch outputs ltrb distances
      multiplied by stride; with exp_on_reg the output is `exp(reg) * stride`.
    - No Scale per-level parameters (those belong to the parent RTMDetHead, which is
      NOT what the published weights use).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int,
        strides: Sequence[int] = (8, 16, 32),
        stacked_convs: int = 2,
        share_conv: bool = True,
        exp_on_reg: bool = False,
        pred_kernel_size: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.strides = list(strides)
        self.stacked_convs = stacked_convs
        self.share_conv = share_conv
        self.exp_on_reg = exp_on_reg
        self.pred_kernel_size = pred_kernel_size

        n_levels = len(self.strides)

        # Export-mode flag: when True, forward returns a single flat tensor
        # suitable for ONNX/TorchScript tracing. The exporter (libreyolo/export/exporter.py)
        # flips this on automatically before tracing.
        self.export: bool = False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in range(n_levels):
            cls_per_level = nn.ModuleList()
            reg_per_level = nn.ModuleList()
            for i in range(stacked_convs):
                chn = in_channels if i == 0 else feat_channels
                cls_per_level.append(
                    ConvBNAct(chn, feat_channels, 3, stride=1, padding=1)
                )
                reg_per_level.append(
                    ConvBNAct(chn, feat_channels, 3, stride=1, padding=1)
                )
            self.cls_convs.append(cls_per_level)
            self.reg_convs.append(reg_per_level)

        pad = pred_kernel_size // 2
        self.rtm_cls = nn.ModuleList(
            [
                nn.Conv2d(feat_channels, num_classes, pred_kernel_size, padding=pad)
                for _ in range(n_levels)
            ]
        )
        self.rtm_reg = nn.ModuleList(
            [
                nn.Conv2d(feat_channels, 4, pred_kernel_size, padding=pad)
                for _ in range(n_levels)
            ]
        )

        if share_conv:
            for n in range(1, n_levels):
                for i in range(stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

        self._init_weights()

    def _init_weights(self) -> None:
        # Mirror mmdet RTMDetSepBNHead.init_weights: Normal(std=0.01) convs,
        # norms at 1, and the focal prior bias on rtm_cls so a fresh head
        # scores ~0.01 everywhere instead of ~0.5. Without the prior bias, a
        # head rebuilt for a new class count fires all priors at once and the
        # first QFL batch produces a ~1e5x gradient shock that destroys the
        # pretrained backbone (issue #566). Loading a checkpoint afterwards
        # overwrites all of this, so published weights are unaffected.
        import math

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        bias_cls = -math.log((1 - 0.01) / 0.01)  # bias_init_with_prob(0.01)
        for conv in self.rtm_cls:
            nn.init.constant_(conv.bias, bias_cls)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for idx, feat in enumerate(feats):
            cls_feat = feat
            reg_feat = feat
            for layer in self.cls_convs[idx]:
                cls_feat = layer(cls_feat)
            for layer in self.reg_convs[idx]:
                reg_feat = layer(reg_feat)
            cls_score = self.rtm_cls[idx](cls_feat)
            reg = self.rtm_reg[idx](reg_feat)
            stride = self.strides[idx]
            if self.exp_on_reg:
                reg = reg.exp() * stride
            else:
                reg = reg * stride
            cls_scores.append(cls_score)
            bbox_preds.append(reg)

        if self.export:
            # Export mode: build grid priors inside the graph, decode (l,t,r,b)
            # distances to (x1,y1,x2,y2) boxes, sigmoid the cls scores, and
            # concatenate all levels into a single (B, N, 4 + num_classes) tensor.
            # No NMS in graph; the consumer is expected to apply NMS.
            decoded = []
            for cls_score, reg, stride in zip(cls_scores, bbox_preds, self.strides):
                b, c, h, w = cls_score.shape
                device, dtype = cls_score.device, cls_score.dtype
                # mmdet config: MlvlPointGenerator(offset=0) -> cell corners.
                xx = torch.arange(w, device=device, dtype=dtype) * stride
                yy = torch.arange(h, device=device, dtype=dtype) * stride
                gy, gx = torch.meshgrid(yy, xx, indexing="ij")
                points = torch.stack([gx, gy], dim=-1).reshape(-1, 2)  # (H*W, 2)

                cls_flat = cls_score.permute(0, 2, 3, 1).reshape(b, -1, c).sigmoid()
                reg_flat = reg.permute(0, 2, 3, 1).reshape(b, -1, 4)

                x1 = points[:, 0].unsqueeze(0) - reg_flat[..., 0]
                y1 = points[:, 1].unsqueeze(0) - reg_flat[..., 1]
                x2 = points[:, 0].unsqueeze(0) + reg_flat[..., 2]
                y2 = points[:, 1].unsqueeze(0) + reg_flat[..., 3]
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, H*W, 4)
                decoded.append(torch.cat([boxes, cls_flat], dim=-1))

            return torch.cat(decoded, dim=1)  # (B, total, 4 + num_classes)

        return tuple(cls_scores), tuple(bbox_preds)


class MaskFeatModule(nn.Module):
    """Fuse the three FPN levels into the stride-8 RTMDet-Ins mask feature."""

    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        num_levels: int = 3,
        num_prototypes: int = 8,
        stacked_convs: int = 4,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.fusion_conv = nn.Conv2d(num_levels * in_channels, in_channels, 1)
        convs = []
        for i in range(stacked_convs):
            in_c = in_channels if i == 0 else feat_channels
            convs.append(ConvBNAct(in_c, feat_channels, 3, padding=1))
        self.stacked_convs = nn.Sequential(*convs)
        self.projection = nn.Conv2d(feat_channels, num_prototypes, 1)

    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        size = features[0].shape[-2:]
        fused = [features[0]]
        for feature in features[1 : self.num_levels]:
            fused.append(F.interpolate(feature, size=size, mode="bilinear"))
        return self.projection(
            self.stacked_convs(self.fusion_conv(torch.cat(fused, dim=1)))
        )


class RTMDetInsSepBNHead(nn.Module):
    """RTMDet-Ins box, class, dynamic-kernel, and mask-feature head.

    The module layout follows open-mmlab/mmdetection's Apache-2.0
    ``RTMDetInsSepBNHead`` so official checkpoints need only the syntactic
    ``bbox_head`` to ``head`` key rename.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int,
        strides: Sequence[int] = (8, 16, 32),
        stacked_convs: int = 2,
        share_conv: bool = True,
        pred_kernel_size: int = 1,
        num_prototypes: int = 8,
        num_dyconvs: int = 3,
        dyconv_channels: int = 8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.strides = list(strides)
        self.stacked_convs = stacked_convs
        self.share_conv = share_conv
        self.pred_kernel_size = pred_kernel_size
        self.num_prototypes = num_prototypes
        self.num_dyconvs = num_dyconvs
        self.dyconv_channels = dyconv_channels
        self.export: bool = False

        weight_nums = []
        bias_nums = []
        for i in range(num_dyconvs):
            if i == 0:
                weight_nums.append((num_prototypes + 2) * dyconv_channels)
                bias_nums.append(dyconv_channels)
            elif i == num_dyconvs - 1:
                weight_nums.append(dyconv_channels)
                bias_nums.append(1)
            else:
                weight_nums.append(dyconv_channels * dyconv_channels)
                bias_nums.append(dyconv_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        n_levels = len(self.strides)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        for _ in range(n_levels):
            cls_per_level = nn.ModuleList()
            kernel_per_level = nn.ModuleList()
            for i in range(stacked_convs):
                chn = in_channels if i == 0 else feat_channels
                cls_per_level.append(
                    ConvBNAct(chn, feat_channels, 3, stride=1, padding=1)
                )
                kernel_per_level.append(
                    ConvBNAct(chn, feat_channels, 3, stride=1, padding=1)
                )
            self.cls_convs.append(cls_per_level)
            # The published RTMDet-Ins head intentionally uses the same tower
            # modules for classification and regression.
            self.reg_convs.append(cls_per_level)
            self.kernel_convs.append(kernel_per_level)

        pad = pred_kernel_size // 2
        self.rtm_cls = nn.ModuleList(
            [
                nn.Conv2d(feat_channels, num_classes, pred_kernel_size, padding=pad)
                for _ in range(n_levels)
            ]
        )
        self.rtm_reg = nn.ModuleList(
            [
                nn.Conv2d(feat_channels, 4, pred_kernel_size, padding=pad)
                for _ in range(n_levels)
            ]
        )
        self.rtm_kernel = nn.ModuleList(
            [
                nn.Conv2d(
                    feat_channels,
                    self.num_gen_params,
                    pred_kernel_size,
                    padding=pad,
                )
                for _ in range(n_levels)
            ]
        )

        if share_conv:
            for n in range(1, n_levels):
                for i in range(stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

        self.mask_head = MaskFeatModule(
            in_channels=in_channels,
            feat_channels=feat_channels,
            num_levels=n_levels,
            num_prototypes=num_prototypes,
            stacked_convs=4,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        import math

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        bias_cls = -math.log((1 - 0.01) / 0.01)
        for cls_conv, reg_conv in zip(self.rtm_cls, self.rtm_reg):
            nn.init.constant_(cls_conv.bias, bias_cls)
            nn.init.constant_(reg_conv.bias, 1.0)

    def forward(self, feats):
        if self.export:
            raise NotImplementedError(
                "RTMDet-Ins export is not supported yet; use native PyTorch "
                "inference for task='segment'."
            )

        mask_feat = self.mask_head(feats)
        cls_scores = []
        bbox_preds = []
        kernel_preds = []
        for idx, feat in enumerate(feats):
            cls_feat = feat
            reg_feat = feat
            kernel_feat = feat
            for layer in self.cls_convs[idx]:
                cls_feat = layer(cls_feat)
            for layer in self.reg_convs[idx]:
                reg_feat = layer(reg_feat)
            for layer in self.kernel_convs[idx]:
                kernel_feat = layer(kernel_feat)

            cls_scores.append(self.rtm_cls[idx](cls_feat))
            bbox_preds.append(F.relu(self.rtm_reg[idx](reg_feat)) * self.strides[idx])
            kernel_preds.append(self.rtm_kernel[idx](kernel_feat))

        return (
            tuple(cls_scores),
            tuple(bbox_preds),
            tuple(kernel_preds),
            mask_feat,
        )


# =============================================================================
# Per-size table and assembly
# =============================================================================

# (deepen_factor, widen_factor, neck_in_channels, neck_out_channels, num_csp_blocks, exp_on_reg)
_SIZE_CONFIG = {
    "t": (0.167, 0.375, [96, 192, 384], 96, 1, False),
    "s": (0.33, 0.5, [128, 256, 512], 128, 1, False),
    "m": (0.67, 0.75, [192, 384, 768], 192, 2, True),
    "l": (1.0, 1.0, [256, 512, 1024], 256, 3, True),
    "x": (1.33, 1.25, [320, 640, 1280], 320, 4, True),
}


class LibreRTMDetModel(nn.Module):
    """Top-level RTMDet detection or instance-segmentation model."""

    def __init__(self, size: str = "s", nc: int = 80, enable_mask_head: bool = False):
        super().__init__()
        if size not in _SIZE_CONFIG:
            raise ValueError(
                f"Unknown RTMDet size {size!r}. Must be one of {list(_SIZE_CONFIG)}."
            )
        deepen, widen, neck_in, neck_out, num_csp, exp_on_reg = _SIZE_CONFIG[size]
        self.size = size
        self.nc = nc
        self.enable_mask_head = enable_mask_head

        self.backbone = CSPNeXt(
            deepen_factor=deepen,
            widen_factor=widen,
            out_indices=(2, 3, 4),
            expand_ratio=0.5,
            channel_attention=True,
        )
        self.neck = CSPNeXtPAFPN(
            in_channels=neck_in,
            out_channels=neck_out,
            num_csp_blocks=num_csp,
            expand_ratio=0.5,
        )
        head_cls = RTMDetInsSepBNHead if enable_mask_head else RTMDetSepBNHead
        head_kwargs = dict(
            num_classes=nc,
            in_channels=neck_out,
            feat_channels=neck_out,
            strides=(8, 16, 32),
            stacked_convs=2,
            share_conv=True,
            pred_kernel_size=1,
        )
        if not enable_mask_head:
            head_kwargs["exp_on_reg"] = exp_on_reg
        self.head = head_cls(**head_kwargs)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)
