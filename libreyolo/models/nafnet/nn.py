"""Native NAFNet modules for image restoration.

This is a compact PyTorch port of NAFNet's plain-convolution U-Net and TLC
test-time local pooling path. Module names intentionally mirror the upstream
architecture so upstream state dicts can be converted with minimal remapping.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Split channels in half and multiply the halves."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class AvgPool2d(nn.Module):
    """TLC local-average replacement for AdaptiveAvgPool2d(1)."""

    def __init__(
        self,
        kernel_size: int | Sequence[int] | None = None,
        base_size: int | Sequence[int] | None = None,
        auto_pad: bool = True,
        fast_imp: bool = False,
        train_size: Sequence[int] | None = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, base_size={self.base_size}, "
            f"stride={self.kernel_size}, fast_imp={self.fast_imp}"
        )

    def _materialize_kernel(self, x: torch.Tensor) -> None:
        if self.kernel_size is not None or self.base_size is None:
            return
        if self.train_size is None:
            raise ValueError("train_size is required when base_size is used.")
        base_size = self.base_size
        if isinstance(base_size, int):
            base_size = (base_size, base_size)
        self.kernel_size = [
            x.shape[2] * int(base_size[0]) // int(self.train_size[-2]),
            x.shape[3] * int(base_size[1]) // int(self.train_size[-1]),
        ]
        self.max_r1 = max(1, self.rs[0] * x.shape[2] // int(self.train_size[-2]))
        self.max_r2 = max(1, self.rs[0] * x.shape[3] // int(self.train_size[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._materialize_kernel(x)
        if self.kernel_size is None:
            return F.adaptive_avg_pool2d(x, 1)
        k1, k2 = int(self.kernel_size[0]), int(self.kernel_size[1])
        if k1 >= x.size(-2) and k2 >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:
            out = self._forward_fast(x, k1, k2)
        else:
            out = self._forward_exact(x, k1, k2)

        if self.auto_pad:
            _, _, h, w = x.shape
            oh, ow = out.shape[2:]
            pad2d = (
                (w - ow) // 2,
                (w - ow + 1) // 2,
                (h - oh) // 2,
                (h - oh + 1) // 2,
            )
            out = F.pad(out, pad2d, mode="replicate")
        return out

    def _forward_fast(self, x: torch.Tensor, k1: int, k2: int) -> torch.Tensor:
        h, w = x.shape[2:]
        r1 = next(r for r in self.rs if h % r == 0)
        r2 = next(r for r in self.rs if w % r == 0)
        r1 = min(self.max_r1, r1)
        r2 = min(self.max_r2, r2)
        s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
        _, _, sh, sw = s.shape
        kk1 = min(sh - 1, k1 // r1)
        kk2 = min(sw - 1, k2 // r2)
        out = (
            s[:, :, :-kk1, :-kk2]
            - s[:, :, :-kk1, kk2:]
            - s[:, :, kk1:, :-kk2]
            + s[:, :, kk1:, kk2:]
        ) / (kk1 * kk2)
        return F.interpolate(out, scale_factor=(r1, r2))

    @staticmethod
    def _forward_exact(x: torch.Tensor, k1: int, k2: int) -> torch.Tensor:
        _, _, h, w = x.shape
        s = x.cumsum(dim=-1).cumsum_(dim=-2)
        s = F.pad(s, (1, 0, 1, 0))
        k1 = min(h, k1)
        k2 = min(w, k2)
        out = (
            s[:, :, k1:, k2:]
            + s[:, :, :-k1, :-k2]
            - s[:, :, :-k1, k2:]
            - s[:, :, k1:, :-k2]
        )
        return out / (k1 * k2)


def replace_adaptive_avg_pool2d(
    module: nn.Module,
    *,
    base_size: Sequence[int],
    train_size: Sequence[int],
    fast_imp: bool = False,
) -> None:
    """Recursively replace AdaptiveAvgPool2d(1) with TLC local pooling."""

    for name, child in module.named_children():
        if len(list(child.children())) > 0:
            replace_adaptive_avg_pool2d(
                child,
                base_size=base_size,
                train_size=train_size,
                fast_imp=fast_imp,
            )
        if isinstance(child, nn.AdaptiveAvgPool2d):
            if child.output_size != 1:
                raise ValueError("TLC only supports AdaptiveAvgPool2d(1).")
            setattr(
                module,
                name,
                AvgPool2d(
                    base_size=base_size,
                    fast_imp=fast_imp,
                    train_size=train_size,
                ),
            )


class NAFBlock(nn.Module):
    """NAFNet residual block."""

    def __init__(
        self,
        c: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channel,
            dw_channel,
            3,
            1,
            1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """NAFNet U-Net for RGB restoration."""

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 1,
        enc_blk_nums: Iterable[int] = (1, 1, 1, 28),
        dec_blk_nums: Iterable[int] = (1, 1, 1, 1),
    ):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inp.shape
        del b, c
        x_in = self.check_image_size(inp)
        x = self.intro(x_in)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + x_in
        return x[:, :, :h, :w]

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_pad_h or mod_pad_w:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(NAFNet):
    """NAFNet with TLC local pooling enabled."""

    def __init__(
        self,
        *args,
        train_size: Sequence[int] = (1, 3, 256, 256),
        fast_imp: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        _, _, h, w = train_size
        base_size = (int(h * 1.5), int(w * 1.5))
        self.convert_to_local(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

    def convert_to_local(
        self,
        *,
        base_size: Sequence[int],
        train_size: Sequence[int],
        fast_imp: bool = False,
    ) -> None:
        replace_adaptive_avg_pool2d(
            self,
            base_size=base_size,
            train_size=train_size,
            fast_imp=fast_imp,
        )
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            self.forward(torch.rand(tuple(train_size), device=device))
        self.train(was_training)


__all__ = [
    "AvgPool2d",
    "LayerNorm2d",
    "NAFBlock",
    "NAFNet",
    "NAFNetLocal",
    "SimpleGate",
    "replace_adaptive_avg_pool2d",
]

