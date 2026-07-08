"""Reader/writer for Darknet ``.weights`` binary files.

The binary layout is reproduced from the public-domain Darknet source
(``src/parser.c``: ``load_weights_upto`` / ``load_convolutional_weights`` /
``save_weights_upto``). No source is copied; only the byte layout is honoured:

Header
    ``int32 major, int32 minor, int32 revision`` then the ``seen`` counter,
    read as ``uint64`` when ``major*10 + minor >= 2`` (and both < 1000), else
    ``uint32``.

Per ``[convolutional]`` layer, in cfg order, as float32 arrays:
    * with ``batch_normalize``: ``biases(n) -> scales(n) -> rolling_mean(n) ->
      rolling_variance(n) -> weights(n*c*h*w)``
    * without: ``biases(n) -> weights(n*c*h*w)``

Per ``[connected]`` (fully-connected) layer (YOLOv1 head):
    * ``biases(outputs) -> weights(outputs*inputs)``, row-major ``(out, in)``.
      Darknet transposes the matrix only when ``major > 1000`` or
      ``minor > 1000``; the v1-era files (major=0) are stored ``(out, in)``,
      matching ``nn.Linear.weight`` directly.

Per ``[local]`` (locally-connected) layer (YOLOv1 head):
    * ``biases(out_h*out_w*n) -> weights(c*n*size*size*out_h*out_w)`` with the
      per-location layout described on :class:`~libreyolo.models.darknet.blocks.DarknetLocal`.

Layers without learned parameters (route/shortcut/maxpool/upsample/reorg/
dropout/yolo/region/detection) contribute nothing.

The reader asserts the file is consumed exactly; a mismatch means the built
network does not match the ``.weights`` file (wrong cfg or wrong build), which
is the single most valuable conversion-correctness check.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import torch

from .blocks import DarknetConnected, DarknetConv, DarknetLocal
from .net import DarknetNet

# Modules that carry ``.weights`` bytes, in the order Darknet serializes them.
_PARAM_TYPES = (DarknetConv, DarknetConnected, DarknetLocal)


def _iter_param_layers(net: DarknetNet):
    for module in net.layers:
        if isinstance(module, _PARAM_TYPES):
            yield module


def read_header(fp: io.BufferedReader) -> tuple[int, int, int, int]:
    """Read the 4-field Darknet header; return ``(major, minor, revision, seen)``."""
    major, minor, revision = struct.unpack("<iii", fp.read(12))
    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        (seen,) = struct.unpack("<Q", fp.read(8))  # uint64
    else:
        (seen,) = struct.unpack("<I", fp.read(4))  # uint32
    return major, minor, revision, seen


def _transpose_connected(major: int, minor: int) -> bool:
    """Darknet transposes ``[connected]`` weights only for very new versions."""
    return major > 1000 or minor > 1000


def load_darknet_weights(net: DarknetNet, data: bytes) -> None:
    """Load a Darknet ``.weights`` byte buffer into a built :class:`DarknetNet`.

    Raises ``ValueError`` if the buffer is not consumed exactly.
    """
    fp = io.BytesIO(data)
    major, minor, _, _ = read_header(fp)
    transpose = _transpose_connected(major, minor)
    payload = fp.read()
    if len(payload) % 4 != 0:
        raise ValueError(
            f"Darknet .weights truncated: payload of {len(payload)} bytes is not "
            f"a whole number of float32 values. The file is corrupt or truncated."
        )
    buf = np.frombuffer(payload, dtype=np.float32)
    ptr = 0

    def take(n: int) -> torch.Tensor:
        nonlocal ptr
        chunk = buf[ptr : ptr + n]
        if chunk.shape[0] != n:
            raise ValueError(
                f"Darknet .weights truncated: needed {n} floats at offset {ptr}, "
                f"only {chunk.shape[0]} remain. The built network does not match "
                f"this .weights file."
            )
        ptr += n
        return torch.from_numpy(chunk.copy())

    for module in _iter_param_layers(net):
        if isinstance(module, DarknetConv):
            conv = module
            c_out = conv.conv.out_channels
            if conv.bn is not None:
                conv.bn.bias.data.copy_(take(c_out))
                conv.bn.weight.data.copy_(take(c_out))
                conv.bn.running_mean.data.copy_(take(c_out))
                conv.bn.running_var.data.copy_(take(c_out))
            else:
                conv.conv.bias.data.copy_(take(c_out))
            w = conv.conv.weight
            conv.conv.weight.data.copy_(take(w.numel()).view_as(w))
        elif isinstance(module, DarknetConnected):
            out_f, in_f = module.out_features, module.in_features
            module.linear.bias.data.copy_(take(out_f))
            w = take(out_f * in_f)
            if transpose:
                w = w.view(in_f, out_f).t().contiguous()
            else:
                w = w.view(out_f, in_f)
            module.linear.weight.data.copy_(w)
        else:  # DarknetLocal
            local = module
            n = local.out_channels
            locations = local.out_h * local.out_w
            patch = local.in_channels * local.size * local.size
            local.bias.data.copy_(take(n * locations).view(n, locations))
            local.weight.data.copy_(take(locations * n * patch).view(locations, n, patch))

    if ptr != buf.shape[0]:
        raise ValueError(
            f"Darknet .weights not fully consumed: read {ptr} of {buf.shape[0]} "
            f"floats ({buf.shape[0] - ptr} left over). The built network has "
            f"fewer parameters than the .weights file provides."
        )


def load_darknet_weights_file(net: DarknetNet, path: str | Path) -> None:
    """Load a Darknet ``.weights`` file from disk into a built network."""
    load_darknet_weights(net, Path(path).read_bytes())


def save_darknet_weights(net: DarknetNet, major: int = 0, minor: int = 2) -> bytes:
    """Serialize a built network back to Darknet ``.weights`` bytes.

    Used to build synthetic fixtures and to round-trip test the reader offline
    (no download needed). The header version defaults to ``0.2`` so ``seen`` is
    written as ``uint64``, matching the modern reader path.
    """
    out = io.BytesIO()
    out.write(struct.pack("<iii", major, minor, 0))
    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        out.write(struct.pack("<Q", 0))
    else:
        out.write(struct.pack("<I", 0))
    transpose = _transpose_connected(major, minor)

    def _w(tensor: torch.Tensor) -> None:
        out.write(tensor.data.cpu().numpy().astype(np.float32).tobytes())

    for module in _iter_param_layers(net):
        if isinstance(module, DarknetConv):
            conv = module
            if conv.bn is not None:
                _w(conv.bn.bias.data)
                _w(conv.bn.weight.data)
                _w(conv.bn.running_mean.data)
                _w(conv.bn.running_var.data)
            else:
                _w(conv.conv.bias.data)
            _w(conv.conv.weight.data)
        elif isinstance(module, DarknetConnected):
            _w(module.linear.bias.data)
            w = module.linear.weight.data
            _w(w.t().contiguous() if transpose else w)
        else:  # DarknetLocal
            _w(module.bias.data)
            _w(module.weight.data)

    return out.getvalue()
