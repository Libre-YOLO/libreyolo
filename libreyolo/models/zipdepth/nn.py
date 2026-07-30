"""ZipDepth network: lightweight CNN for relative monocular depth estimation.

Ported from the official implementation (https://github.com/fabiotosi92/ZipDepth,
MIT, ECCV 2026, University of Bologna). Module and attribute names mirror
upstream exactly so official checkpoints load with ``strict=True`` and the
inference parity gate can compare layer-by-layer. See the family ``NOTICE``.

Architecture: a four-stage RepVGG-style reparameterizable encoder (3x3 + 1x1 +
identity branches fused into single 3x3 convs for inference) with lightweight
attention (Squeeze-and-Excitation, strip pooling, GCNet-style global context),
an SPPF + cross-scale fusion neck, and an FPN decoder that predicts inverse
depth at half resolution and upsamples it with a convex-combination head.

Two decoder flavors exist, selected by ``upsample_unfold``:

- ``True`` (GPU checkpoint ``LibreZipDepthb-depth``): RAFT-style convex
  upsampling with a predicted 9-way softmax mask over ``F.unfold`` neighbors.
- ``False`` (NPU checkpoint ``LibreZipDepthbnpu-depth``): unfold-free variant
  that blends nearest and bilinear upsampling with a learned per-pixel alpha,
  for NPU/edge compilers without gather/unfold support.

The two flavors are separately trained checkpoints, not a graph rewrite: the
``decoder.convex_up`` weights differ.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Upstream MODEL_CONFIGS. Only 'base' has released checkpoints today; the other
# entries are kept so future upstream releases map onto new size codes without
# re-porting.
MODEL_CONFIGS = {
    "small": {
        "dims": [24, 48, 96, 192],
        "depths": [2, 2, 4, 2],
        "heads": 4,
        "dec_ch": 32,
        "half_dec_ch": 24,
        "use_global": True,
    },
    "base": {
        "dims": [48, 96, 192, 384],
        "depths": [2, 2, 6, 2],
        "heads": 4,
        "dec_ch": 96,
        "half_dec_ch": 32,
        "use_global": True,
    },
    "large": {
        "dims": [64, 128, 256, 384],
        "depths": [2, 4, 10, 4],
        "heads": 8,
        "dec_ch": 192,
        "half_dec_ch": 48,
        "use_global": True,
    },
    "giant": {
        "dims": [96, 192, 384, 512],
        "depths": [2, 4, 14, 6],
        "heads": 8,
        "dec_ch": 288,
        "half_dec_ch": 64,
        "use_global": True,
    },
}

# LibreYOLO size code -> (upstream variant, decoder flavor).
# 'bnpu' is the same 'base' capacity with the unfold-free NPU decoder.
SIZE_CONFIGS = {
    "b": {"variant": "base", "upsample_unfold": True},
    "bnpu": {"variant": "base", "upsample_unfold": False},
}


class ConvBN(nn.Module):
    """Conv2d + BatchNorm2d + optional ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
    ):
        super().__init__()
        if p is None:
            p = (k + (k - 1) * (d - 1)) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class QARepBlock(nn.Module):
    """RepVGG block: 3x3 + 1x1 + identity branches, fused to one 3x3 for inference."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 1, act: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.groups = groups
        self.has_identity = in_ch == out_ch and stride == 1

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, 0, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "fused_conv"):
            return self.act(self.fused_conv(x))

        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.has_identity:
            out = out + x
        return self.act(out)

    def fuse(self):
        if hasattr(self, "fused_conv"):
            return

        k3, b3 = self._fuse_conv_bn(self.branch_3x3[0], self.branch_3x3[1])
        k1, b1 = self._fuse_conv_bn(self.branch_1x1[0], self.branch_1x1[1])
        k1_padded = F.pad(k1, [1, 1, 1, 1])

        kernel = k3 + k1_padded
        bias = b3 + b1

        if self.has_identity:
            k_id = torch.zeros_like(kernel)
            for i in range(self.in_ch):
                k_id[i, i % (self.in_ch // self.groups), 1, 1] = 1.0
            kernel = kernel + k_id

        self.fused_conv = nn.Conv2d(
            self.in_ch, self.out_ch, 3, self.stride, 1, groups=self.groups, bias=True
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias

        del self.branch_3x3, self.branch_1x1

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
        w = conv.weight
        mean, var = bn.running_mean, bn.running_var
        gamma, beta, eps = bn.weight, bn.bias, bn.eps
        std = (var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return w * t, beta - mean * gamma / std


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation with reduced bottleneck."""

    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean reduction is exactly AdaptiveAvgPool2d(1), while also lowering
        # to an operator provided by the ExecuTorch 1.2 portable runtime.
        pooled = x.mean(dim=(-2, -1), keepdim=True)
        return x * self.fc(pooled)


class EfficientGlobalAttention(nn.Module):
    """Global attention via learnable global tokens (only in 'full' global mode)."""

    def __init__(self, dim: int, num_tokens: int = 8, num_heads: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.tokens = nn.Parameter(torch.randn(1, num_tokens, dim))
        nn.init.trunc_normal_(self.tokens, std=0.02)

        self.q_tokens = nn.Linear(dim, dim, bias=False)
        self.kv_spatial = nn.Conv2d(dim, dim * 2, 1, bias=False)

        self.q_spatial = nn.Conv2d(dim, dim, 1, groups=num_heads, bias=False)
        self.k_proj_tokens = nn.Linear(dim, dim, bias=False)

        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        kv = self.kv_spatial(x).reshape(B, 2, C, N).permute(1, 0, 2, 3)
        k_sp, v_sp = kv[0], kv[1]

        k_sp = k_sp.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2)
        v_sp = v_sp.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2)

        q_tok = (
            self.q_tokens(self.tokens)
            .reshape(1, self.num_tokens, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .expand(B, -1, -1, -1)
        )

        attn_t2s = (q_tok @ k_sp.transpose(-2, -1)) * self.scale
        tokens_updated = F.softmax(attn_t2s, dim=-1) @ v_sp

        q_sp = (
            self.q_spatial(x)
            .reshape(B, C, N)
            .view(B, self.num_heads, self.head_dim, N)
            .transpose(-1, -2)
        )

        k_tok = (
            self.k_proj_tokens(self.tokens)
            .reshape(1, self.num_tokens, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .expand(B, -1, -1, -1)
        )
        v_tok = tokens_updated

        attn_s2t = F.softmax((q_sp @ k_tok.transpose(-2, -1)) * self.scale, dim=-1)
        out = attn_s2t @ v_tok

        out = out.transpose(1, 2).reshape(B, N, C).transpose(1, 2).reshape(B, C, H, W)
        return x + self.norm(self.proj_out(out))


class StripPoolingAttention(nn.Module):
    """Horizontal + vertical strip pooling gate."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_strip = x.mean(dim=3, keepdim=True)  # [B, C, H, 1]
        w_strip = x.mean(dim=2, keepdim=True)  # [B, C, 1, W]

        gate = self.gate_conv(h_strip + w_strip)  # broadcast to [B, C, H, W]
        return x * gate


class GlobalContextBlock(nn.Module):
    """GCNet-style global context; BN instead of LN for small-batch stability."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.context_weight = nn.Conv2d(dim, 1, 1)

        hidden = max(dim // reduction, 8)
        self.transform = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        context_mask = self.context_weight(x).view(B, 1, H * W)
        context_mask = F.softmax(context_mask, dim=-1)

        x_flat = x.view(B, C, H * W)
        context = torch.bmm(x_flat, context_mask.transpose(1, 2)).unsqueeze(-1)
        context = self.transform(context)

        return x + context


class MinimalMultiScale(nn.Module):
    """Lightweight multi-scale context with 2 dilation rates."""

    def __init__(self, dim: int):
        super().__init__()
        self.branch1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.branch2 = nn.Conv2d(dim, dim, 3, 1, 2, dilation=2, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bn(self.branch1(x) + self.branch2(x))


def _pick_groups(in_ch: int, out_ch: int, max_g: int = 4) -> int:
    for g in (max_g, 2, 1):
        if in_ch % g == 0 and out_ch % g == 0:
            return g
    return 1


class MinimalCrossScale(nn.Module):
    """Bidirectional stage-3 <-> stage-4 feature exchange."""

    def __init__(self, dim_high: int, dim_low: int):
        super().__init__()
        g_h = _pick_groups(dim_low, dim_high, 4)
        g_l = _pick_groups(dim_high, dim_low, 4)

        self.low_to_high = nn.Conv2d(dim_low, dim_high, 1, groups=g_h, bias=False)
        self.high_to_low = nn.Conv2d(dim_high, dim_low, 1, groups=g_l, bias=False)

    def forward(
        self, x_high: torch.Tensor, x_low: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        low_up = F.interpolate(self.low_to_high(x_low), size=x_high.shape[2:], mode="nearest")
        # Valid ZipDepth inputs keep adjacent stages at an exact 2:1 spatial
        # ratio, making this identical to adaptive pooling to x_low's shape.
        high_down = F.avg_pool2d(self.high_to_low(x_high), kernel_size=2, stride=2)
        return x_high + low_up * 0.3, x_low + high_down * 0.3


class LightweightSPPF(nn.Module):
    """SPPF with reduced hidden channels."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_hidden = c1 // 4
        self.cv1 = ConvBN(c1, c_hidden, 1)
        self.cv2 = ConvBN(c_hidden * 4, c2, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class UltraLightFusion(nn.Module):
    """Project-and-add FPN fusion of a high-res and an upsampled low-res map."""

    def __init__(self, high_ch: int, low_ch: int, out_ch: int):
        super().__init__()
        g_high = _pick_groups(high_ch, out_ch, 4)
        g_low = _pick_groups(low_ch, out_ch, 4)

        self.proj_high = nn.Conv2d(high_ch, out_ch, 1, groups=g_high, bias=False)
        self.proj_low = nn.Conv2d(low_ch, out_ch, 1, groups=g_low, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        x_low = F.interpolate(x_low, size=x_high.shape[2:], mode="bilinear", align_corners=False)
        return self.act(self.bn(self.proj_high(x_high) + self.proj_low(x_low)))


class FastConvexUpsample(nn.Module):
    """Final 2x depth upsampling head; see module docstring for the two flavors."""

    def __init__(
        self,
        feat_ch: int,
        edge_ch: int = 8,
        scale: int = 4,
        temperature: float = 1.0,
        use_unfold: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.temperature = temperature
        self.use_unfold = use_unfold

        in_ch = feat_ch

        if use_unfold:
            hidden = max(feat_ch // 4, 8)
            self.mask_pred = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 9 * scale * scale, 1),
            )
        else:
            where_hidden = max(in_ch // 2, 8)
            self.where_conv = nn.Sequential(
                nn.Conv2d(in_ch, where_hidden, 1, bias=False),
                nn.BatchNorm2d(where_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    where_hidden, where_hidden, 5, padding=2, groups=where_hidden, bias=False
                ),
                nn.BatchNorm2d(where_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(where_hidden, 1, 1, bias=False),
            )

    def forward(self, feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if self.use_unfold:
            return F.relu(self._forward_unfold(feat, depth))

        S = self.scale

        depth_nn = F.interpolate(depth, scale_factor=S, mode="nearest")
        depth_bi = F.interpolate(depth, scale_factor=S, mode="bilinear", align_corners=False)

        alpha = self.where_conv(feat)
        alpha = F.interpolate(alpha, scale_factor=S, mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(alpha)

        out = alpha * depth_nn + (1.0 - alpha) * depth_bi

        return F.relu(out)

    def _forward_unfold(self, feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        B, _, H, W = depth.shape
        S = self.scale

        mask = self.mask_pred(feat)
        mask = mask.view(B, 9, S * S, H, W)
        mask = F.softmax(mask / self.temperature, dim=1)

        depth_pad = F.pad(depth, (1, 1, 1, 1), mode="replicate")
        neighbors = F.unfold(depth_pad, 3).view(B, 9, 1, H, W)

        up = (mask * neighbors).sum(1)
        up = F.pixel_shuffle(up.view(B, S * S, H, W), S)

        return up


class ZipDepthDecoder(nn.Module):
    """FPN decoder: half-resolution inverse-depth head + convex 2x upsampling."""

    def __init__(
        self,
        enc_dims: List[int],
        half_ch: int,
        dec_ch: int,
        half_dec_ch: int = 16,
        upsample_unfold: bool = True,
        use_half_res: bool = True,
    ):
        super().__init__()
        self.use_half_res = use_half_res
        c1, c2, c3, c4 = enc_dims
        ch4 = dec_ch * 3
        ch3 = dec_ch * 2
        ch2 = int(dec_ch * 1.5)
        ch1 = dec_ch
        self.proj4 = ConvBN(c4, ch4, 1)
        self.fuse3 = UltraLightFusion(c3, ch4, ch3)
        self.fuse2 = UltraLightFusion(c2, ch3, ch2)
        self.fuse1 = UltraLightFusion(c1, ch2, ch1)

        if use_half_res:
            ch_half = half_dec_ch
            self.fuse_half = UltraLightFusion(high_ch=half_ch, low_ch=ch1, out_ch=ch_half)
            self.head_half = nn.Conv2d(ch_half, 1, 3, padding=1)
            nn.init.kaiming_normal_(self.head_half.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(self.head_half.bias, 0.5)
            self.convex_up = FastConvexUpsample(
                feat_ch=ch_half, scale=2, use_unfold=upsample_unfold
            )
        else:
            self.head_direct = nn.Conv2d(ch1, 1, 3, padding=1)
            nn.init.kaiming_normal_(self.head_direct.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(self.head_direct.bias, 0.5)

    def forward(
        self,
        s_half: torch.Tensor,
        feats: List[torch.Tensor],
        size: Tuple[int, int],
    ) -> torch.Tensor:
        c1, c2, c3, c4 = feats
        f4 = self.proj4(c4)
        f3 = self.fuse3(c3, f4)
        f2 = self.fuse2(c2, f3)
        f1 = self.fuse1(c1, f2)

        if self.use_half_res:
            f_half = self.fuse_half(s_half, f1)
            depth_half = self.head_half(f_half)
            depth = self.convex_up(f_half, depth_half)
        else:
            depth_lr = self.head_direct(f1)
            depth = F.relu(
                F.interpolate(depth_lr, scale_factor=4, mode="bilinear", align_corners=False)
            )

        return depth

    def fuse(self):
        pass


class ZipDepthEncoder(nn.Module):
    """Four-stage RepVGG encoder with SPPF and cross-scale exchange."""

    def __init__(
        self,
        in_ch: int,
        dims: List[int],
        depths: List[int],
        num_heads: int = 4,
        use_global: bool = True,
        global_mode: str = "balanced",
    ):
        super().__init__()
        self.use_global = use_global
        self.global_mode = global_mode

        self.stem_half = ConvBN(in_ch, dims[0] // 2, k=3, s=2)  # -> H/2
        self.stem_quarter = ConvBN(dims[0] // 2, dims[0], k=3, s=2)  # -> H/4

        self.stage1 = nn.Sequential(*[QARepBlock(dims[0], dims[0]) for _ in range(depths[0])])

        self.down2 = QARepBlock(dims[0], dims[1], stride=2)
        stage2_blocks = []
        for i in range(depths[1]):
            stage2_blocks.append(QARepBlock(dims[1], dims[1]))
            if i == depths[1] - 1:
                stage2_blocks.append(MinimalMultiScale(dims[1]))
                if use_global and global_mode in ["balanced", "full"]:
                    stage2_blocks.append(StripPoolingAttention(dims[1]))
        self.stage2 = nn.Sequential(*stage2_blocks)

        self.down3 = QARepBlock(dims[1], dims[2], stride=2)
        stage3_blocks = []
        for i in range(depths[2]):
            stage3_blocks.append(QARepBlock(dims[2], dims[2]))
            if i == depths[2] - 1:
                stage3_blocks.append(ChannelAttention(dims[2], reduction=8))
                if use_global:
                    stage3_blocks.append(GlobalContextBlock(dims[2]))
        self.stage3 = nn.Sequential(*stage3_blocks)

        self.down4 = QARepBlock(dims[2], dims[3], stride=2)
        stage4_blocks = []
        for i in range(depths[3]):
            stage4_blocks.append(QARepBlock(dims[3], dims[3]))
        if use_global and global_mode == "full":
            stage4_blocks.append(
                EfficientGlobalAttention(dims[3], num_tokens=8, num_heads=num_heads)
            )
        self.stage4 = nn.Sequential(*stage4_blocks)

        self.spp = LightweightSPPF(dims[3], dims[3])
        self.cross_scale = MinimalCrossScale(dims[2], dims[3])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        s_half = self.stem_half(x)
        s_quarter = self.stem_quarter(s_half)

        s1 = self.stage1(s_quarter)
        s2 = self.stage2(self.down2(s1))
        s3 = self.stage3(self.down3(s2))
        s4 = self.stage4(self.down4(s3))

        s4 = self.spp(s4)
        s3, s4 = self.cross_scale(s3, s4)

        return s_half, [s1, s2, s3, s4]

    def fuse(self):
        for m in self.modules():
            if m is not self and hasattr(m, "fuse") and callable(m.fuse):
                m.fuse()


class LibreZipDepthNet(nn.Module):
    """ZipDepth adapted to LibreYOLO's dense depth contract.

    The state-dict layout (``encoder.*``, ``decoder.*``, ``mean``, ``std``) is
    identical to upstream ``ZipDepth``, so official checkpoints load with
    ``strict=True`` after DDP/compile prefix stripping. The ImageNet
    normalization buffers ship inside the checkpoint (persistent, exactly like
    upstream), so ``forward`` consumes plain ``[0, 1]`` RGB and returns a
    ``(B, 1, H, W)`` relative inverse-depth map (higher = closer, >= 0).
    """

    def __init__(self, size: str = "b", global_mode: str = "balanced"):
        super().__init__()
        if size not in SIZE_CONFIGS:
            raise ValueError(
                f"Unknown ZipDepth size {size!r}; expected one of {sorted(SIZE_CONFIGS)}."
            )
        size_cfg = SIZE_CONFIGS[size]
        cfg = MODEL_CONFIGS[size_cfg["variant"]]
        self.size = size
        self.variant = size_cfg["variant"]
        self.global_mode = global_mode
        self.upsample_unfold = size_cfg["upsample_unfold"]

        use_global = cfg.get("use_global", True) and global_mode != "none"

        self.encoder = ZipDepthEncoder(
            in_ch=3,
            dims=cfg["dims"],
            depths=cfg["depths"],
            num_heads=cfg.get("heads", 4),
            use_global=use_global,
            global_mode=global_mode,
        )

        self.decoder = ZipDepthDecoder(
            enc_dims=cfg["dims"],
            half_ch=cfg["dims"][0] // 2,
            dec_ch=cfg["dec_ch"],
            half_dec_ch=cfg["half_dec_ch"],
            upsample_unfold=self.upsample_unfold,
        )

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        x_norm = (x - self.mean) / self.std

        s_half, enc_feats = self.encoder(x_norm)

        return self.decoder(s_half, enc_feats, (H, W))

    def fuse_for_inference(self) -> "LibreZipDepthNet":
        """Reparameterize RepVGG branches into single 3x3 convs (idempotent)."""
        self.eval()
        self.encoder.fuse()
        self.decoder.fuse()
        return self
