"""Shared neural-network blocks for the Darknet-lineage families (YOLOv2/v3/v4).

Provenance
----------
The reference behaviour reproduced here comes from the Darknet C source, which
is public domain (the "YOLO LICENSE": *"Darknet is public domain. Do whatever
you want with it."*):

    https://github.com/pjreddie/darknet   (YOLOv1/v2/v3)
    https://github.com/AlexeyAB/darknet   (YOLOv4)

Only the *numerical behaviour* is reproduced; no Darknet source is copied.

The one load-bearing subtlety is normalization. Darknet's ``normalize_cpu``
(``src/blas.c``) computes ``(x - mean) / (sqrt(variance) + 1e-6)`` — epsilon is
added to the *standard deviation*, not the variance, and the constant is
``1e-6``. ``torch.nn.BatchNorm2d`` instead computes ``(x - mean) /
sqrt(variance + eps)``. Over Darknet-53's ~75 batch-normed layers that
difference compounds well past detection-parity tolerances, so we use an exact
:class:`DarknetBatchNorm2d` rather than the stock module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Darknet's fixed batchnorm epsilon (src/blas.c::normalize_cpu).
DARKNET_BN_EPS = 1e-6


class DarknetBatchNorm2d(nn.Module):
    """Inference-exact Darknet batch normalization.

    Reproduces ``y = scales * (x - rolling_mean) / (sqrt(rolling_variance) +
    eps) + biases`` with ``eps = 1e-6`` added to the standard deviation. The
    parameter names mirror ``nn.BatchNorm2d`` (``weight`` = Darknet ``scales``,
    ``bias`` = Darknet ``biases``, ``running_mean`` = ``rolling_mean``,
    ``running_var`` = ``rolling_variance``) so converted checkpoints and the
    weight reader stay simple.

    This module is eval-oriented: it always normalizes with the running
    statistics. These families are inference/eval only in LibreYOLO v1, so no
    running-stat update path is implemented.
    """

    def __init__(self, num_features: int, eps: float = DARKNET_BN_EPS):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.running_mean.view(1, -1, 1, 1)
        var = self.running_var.view(1, -1, 1, 1)
        w = self.weight.view(1, -1, 1, 1)
        b = self.bias.view(1, -1, 1, 1)
        return (x - mean) / (torch.sqrt(var) + self.eps) * w + b

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}"


class Mish(nn.Module):
    """Mish activation: ``x * tanh(softplus(x))`` (YOLOv4 backbone)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


def make_activation(name: str) -> nn.Module:
    """Build an activation module from a Darknet ``activation=`` value."""
    if name == "leaky":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "mish":
        return Mish()
    if name in ("linear", "none"):
        return nn.Identity()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "logistic":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported Darknet activation: {name!r}")


class DarknetConv(nn.Module):
    """``[convolutional]`` layer: Conv2d -> (Darknet)BN -> activation.

    With ``batch_normalize=1`` the convolution is bias-free and a
    :class:`DarknetBatchNorm2d` follows; otherwise the convolution carries its
    own bias and no norm is applied. Padding follows Darknet's ``pad=1``
    convention (``size // 2``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int,
        stride: int,
        batch_normalize: bool,
        activation: str,
        pad: bool = True,
    ):
        super().__init__()
        padding = (size // 2) if pad else 0
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=size,
            stride=stride,
            padding=padding,
            bias=not batch_normalize,
        )
        self.bn = DarknetBatchNorm2d(out_channels) if batch_normalize else None
        self.act = make_activation(activation)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)


class DarknetMaxPool(nn.Module):
    """``[maxpool]`` layer with Darknet's same-size stride-1 behaviour.

    For ``size=2, stride=1`` (used by yolov3-tiny / yolov4-tiny to keep spatial
    size), Darknet pads the right/bottom edges before pooling. We reproduce that
    with a ``-inf`` right/bottom pad. For the common ``size=2, stride=2`` case we
    pool with no padding, matching Darknet's downsample-by-two.
    """

    def __init__(self, size: int, stride: int):
        super().__init__()
        self.size = size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1 and self.size == 2:
            # Pad right/bottom before a same-size stride-1 max-pool. Use the
            # most-negative *finite* value rather than -inf: for max-pooling it
            # is numerically identical (no activation is smaller, and no window
            # here is all-padding), but -inf serializes to a literal some export
            # toolchains reject (e.g. ncnn/pnnx `5=-inf` in the .param).
            neg = torch.finfo(x.dtype).min
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=neg)
            return F.max_pool2d(x, kernel_size=2, stride=1)
        padding = (self.size - 1) // 2
        return F.max_pool2d(x, kernel_size=self.size, stride=self.stride, padding=padding)


class Upsample(nn.Module):
    """``[upsample]`` layer: nearest-neighbour upsampling by ``stride``."""

    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.stride, mode="nearest")


class Reorg(nn.Module):
    """``[reorg]`` passthrough layer (YOLOv2 space-to-depth).

    Reproduces Darknet's ``reorg_cpu`` channel ordering, which is *not* a plain
    ``pixel_unshuffle``. For an input of shape ``(B, C, H, W)`` and stride ``s``
    the output is ``(B, C*s*s, H/s, W/s)`` with Darknet's specific interleaving.
    """

    def __init__(self, stride: int = 2):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Darknet's reorg channel interleaving is NOT a plain space-to-depth /
        # pixel_unshuffle. The sequence below reproduces the community-verified
        # darknet2pytorch (marvis) ordering — the order the pretrained 1x1 conv
        # after the YOLOv2 passthrough concat expects. The exact ordering is
        # regression-locked by ``test_reorg_channel_ordering_golden``.
        #
        # Verification status: real LibreYOLO2b weights produce correct
        # detections through this path, so it is functionally faithful. Note
        # OpenCV's dnn ``Reorg`` layer does NOT match Darknet's ordering, so it
        # is not a valid oracle for the passthrough — confirm any bitwise check
        # against compiled Darknet, not OpenCV (see weights/parity_darknet.py).
        s = self.stride
        b, c, h, w = x.shape
        hs = ws = s
        x = x.view(b, c, h // hs, hs, w // ws, ws).transpose(3, 4).contiguous()
        x = x.view(b, c, (h // hs) * (w // ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(b, c, hs * ws, h // hs, w // ws).transpose(1, 2).contiguous()
        x = x.view(b, hs * ws * c, h // hs, w // ws)
        return x


class DarknetDropout(nn.Module):
    """``[dropout]`` layer: identity at inference.

    The Darknet families are eval-only in LibreYOLO, so dropout carries no
    parameters and passes activations straight through. It exists as a distinct
    module (rather than being dropped) so layer indices in the cfg-driven module
    list stay aligned with the ``.weights`` byte stream (dropout writes no bytes).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DarknetConnected(nn.Module):
    """``[connected]`` fully-connected layer: flatten -> Linear -> activation.

    Used by the YOLOv1 head (the v2/v3/v4 detectors are fully convolutional).
    Darknet stores connected weights as an ``(outputs, inputs)`` row-major matrix
    (each output neuron's ``inputs`` weights contiguous) with ``biases`` first,
    which is exactly ``nn.Linear``'s ``(out_features, in_features)`` +
    ``(out_features,)`` layout. No transpose is needed for the v1-era weights
    (Darknet only transposes when ``major > 1000`` or ``minor > 1000``).

    The input is the previous feature map flattened in Darknet's channel-major
    ``(C, H, W)`` order; ``torch.flatten(x, 1)`` on an ``(N, C, H, W)`` tensor
    reproduces that exact order, so the pretrained matrix aligns element-for-
    element. This flatten-order coupling is the load-bearing YOLOv1 port detail.
    """

    def __init__(self, in_features: int, out_features: int, activation: str = "linear"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.act = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        return self.act(self.linear(x))


class DarknetLocal(nn.Module):
    """``[local]`` locally-connected layer (YOLOv1 head, size-3 kernels only).

    Like a convolution but WITHOUT weight sharing: every output spatial location
    has its own filter bank. Reproduces Darknet's ``forward_local_layer`` +
    ``im2col_cpu``: for each of the ``L = out_h * out_w`` locations, an
    independent ``(filters, C*k*k)`` matrix multiplies that location's im2col
    patch, then a per-(channel, location) bias is added.

    Weight layout matches Darknet's ``.weights`` byte order exactly:

    * ``weight`` shape ``(L, filters, C*k*k)`` -> flat index
      ``loc*(filters*C*k*k) + out*(C*k*k) + patch``
    * ``bias``   shape ``(filters, L)``          -> flat index ``out*L + loc``

    ``F.unfold`` yields the ``(N, C*k*k, L)`` patch tensor with the same
    channel-major -> kernel-row -> kernel-col element ordering as Darknet's
    ``im2col_cpu`` and the same row-major ``L`` (h then w), so the per-location
    gemm maps 1:1. Darknet's ``local`` output-size formula only agrees with the
    im2col padding for ``size=3`` (the only case YOLOv1 uses); ``out_h``/``out_w``
    are computed by the caller and passed in.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int,
        stride: int,
        pad: int,
        activation: str,
        out_h: int,
        out_w: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.stride = stride
        self.pad = pad  # padding amount in pixels (size // 2 for "same")
        self.out_h = out_h
        self.out_w = out_w
        locations = out_h * out_w
        patch = in_channels * size * size
        self.weight = nn.Parameter(torch.zeros(locations, out_channels, patch))
        self.bias = nn.Parameter(torch.zeros(out_channels, locations))
        self.act = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cols = F.unfold(x, kernel_size=self.size, padding=self.pad, stride=self.stride)
        # cols: (N, C*k*k, L). out[n, o, l] = sum_k weight[l, o, k] * cols[n, k, l]
        out = torch.einsum("lok,nkl->nol", self.weight, cols)  # (N, out, L)
        out = out + self.bias.unsqueeze(0)  # broadcast (1, out, L)
        out = out.view(out.shape[0], self.out_channels, self.out_h, self.out_w)
        return self.act(out)
