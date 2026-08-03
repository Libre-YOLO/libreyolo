"""Text-recognition encoder and CTC head for the LibrePPOCR family.

Ported to PyTorch from PaddleOCR (Apache-2.0):
- ``ppocr/modeling/necks/rnn.py`` (Im2Seq, EncoderWithSVTR, SequenceEncoder)
- ``ppocr/modeling/backbones/rec_svtrnet.py`` (ConvBNLayer, Attention, Block)
- ``ppocr/modeling/heads/rec_ctc_head.py`` (CTCHead)
- ``ppocr/modeling/heads/rec_multi_head.py`` (MultiHead, CTC branch)

Both PP-OCRv5 rec tiers share the same head: a MultiHead whose inference
branch is SequenceEncoder(svtr, dims=120, depth=2, kernel [1, 3]) + CTCHead.
The NRTR GTC branch is train-time-only guidance and is not ported; the
converter drops its weights.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import PPHGNetV2B4, PPLCNetV3


class ConvBNLayer(nn.Module):
    """SVTR conv + BN + activation (attribute names ``conv``/``norm``)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        groups=1,
        act: type = nn.SiLU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer: type = nn.SiLU):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale: Optional[float] = None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if torch.onnx.is_in_onnx_export():
            # LibreYOLO defaults to ONNX opset 13, which has no symbolic for
            # fused SDPA: keep the exact primitive-op equation while tracing.
            attn = (q * self.scale).matmul(k.transpose(-2, -1)).softmax(dim=-1)
            x = attn.matmul(v)
        else:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=self.scale,
            )
        x = x.transpose(1, 2).reshape(b, n, self.dim)
        return self.proj(x)


class Block(nn.Module):
    """SVTR global-mixing transformer block (post-norm variant)."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=2.0,
        qkv_bias=True,
        act_layer: type = nn.SiLU,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=epsilon)
        self.mixer = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=epsilon)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Im2Seq(nn.Module):
    """(B, C, 1, W) feature map to (B, W, C) sequence."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] == 1, f"expected height-1 features, got {tuple(x.shape)}"
        return x.squeeze(2).permute(0, 2, 1)


class EncoderWithSVTR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dims: int = 64,
        depth: int = 2,
        hidden_dims: int = 120,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        kernel_size=(3, 3),
    ):
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv1 = ConvBNLayer(in_channels, in_channels // 8, kernel_size, padding=pad)
        self.conv2 = ConvBNLayer(in_channels // 8, hidden_dims, kernel_size=1)
        self.svtr_block = nn.ModuleList(
            [
                Block(hidden_dims, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvBNLayer(hidden_dims, in_channels, kernel_size=1)
        self.conv4 = ConvBNLayer(2 * in_channels, in_channels // 8, kernel_size, padding=pad)
        self.conv1x1 = ConvBNLayer(in_channels // 8, dims, kernel_size=1)
        self.out_channels = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use_guide only detaches the input during training; inference-identical.
        h = x
        z = self.conv2(self.conv1(x))
        b, c, height, width = z.shape
        z = z.flatten(2).permute(0, 2, 1)
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        z = z.reshape(b, height, width, c).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        return self.conv1x1(self.conv4(z))


class SequenceEncoder(nn.Module):
    """The ``svtr`` sequence encoder used by the PP-OCRv5 CTC branch."""

    def __init__(self, in_channels: int, **kwargs):
        super().__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.encoder = EncoderWithSVTR(in_channels, **kwargs)
        self.out_channels = self.encoder.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_reshape(self.encoder(x))


class CTCHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predicts = self.fc(x)
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
        return predicts


class MultiHead(nn.Module):
    """Inference-side MultiHead: CTC encoder + CTC head only.

    The upstream training MultiHead also carries an NRTR guidance branch
    (``before_gtc`` + ``gtc_head``); it never runs at inference and its
    weights are dropped at conversion.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.ctc_encoder = SequenceEncoder(
            in_channels,
            dims=120,
            depth=2,
            hidden_dims=120,
            kernel_size=(1, 3),
        )
        self.ctc_head = CTCHead(self.ctc_encoder.out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ctc_head(self.ctc_encoder(x))


class PPOCRRecModel(nn.Module):
    """Backbone + CTC MultiHead text recognizer for one tier.

    ``num_classes`` is the CTC alphabet size: blank + dictionary + space.
    """

    def __init__(self, size: str, num_classes: int):
        super().__init__()
        if size == "t":
            self.backbone = PPLCNetV3(scale=0.95, det=False)
        elif size == "l":
            self.backbone = PPHGNetV2B4(text_rec=True)
        else:
            raise ValueError(f"Unknown LibrePPOCR size: {size!r} (expected 't' or 'l')")
        self.head = MultiHead(self.backbone.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-timestep char probabilities ``(B, T, num_classes)``."""
        return self.head(self.backbone(x))


__all__ = ["PPOCRRecModel", "SequenceEncoder", "EncoderWithSVTR", "CTCHead", "MultiHead"]
