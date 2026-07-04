"""YOLOv7 model — yaml-driven builder for the MIT MultimediaTechLab/YOLO config.

Reproduces the upstream ``yolo/model/yolo.py`` build/forward semantics so that
the bundled ``v7.yaml`` (mirrored from upstream) plus the ported blocks in
:mod:`.blocks` load upstream ``v7.pt`` with no key remapping beyond the flat
``layers.`` prefix. Source-resolution rules match upstream exactly:

* ``output_dim[0]`` is the input (3ch); layers are 1-indexed.
* a ``source`` of ``-1`` means the previous layer; ``source < -1`` is made
  absolute (``+ layer_idx``); a string ``source`` is a tag → the tagged layer's
  index; a list mixes the two (for ``Concat`` / the detection head).

Provenance: MultimediaTechLab/YOLO (MIT). Not the GPL ``WongKinYiu/yolov7``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from .blocks import Conv, MultiheadDetection, Pool, RepConv, SPPCSPConv, UpSample

_V7_YAML = Path(__file__).resolve().parent / "v7.yaml"


class _Concat(nn.Module):
    """Placeholder occupying a layer index; concatenation happens in forward."""


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class YOLOv7Model(nn.Module):
    """YOLOv7 built from ``v7.yaml``. Forward returns the 3 raw head maps."""

    def __init__(self, num_classes: int = 80, yaml_path: Path | None = None):
        super().__init__()
        cfg = _load_yaml(yaml_path or _V7_YAML)
        self.num_classes = num_classes

        anchor_cfg = cfg["anchor"]
        self.anchors = [list(a) for a in anchor_cfg["anchor"]]  # per-head flat w,h
        self.strides = list(anchor_cfg["strides"])

        entries: list[tuple[str, dict]] = []
        for section in ("backbone", "head"):
            for item in cfg["model"][section]:
                (block_type, info), = item.items()
                entries.append((block_type, info or {}))

        self.layers = nn.ModuleList()
        self._block_types: list[str] = []
        self._sources: list[Any] = []
        self._output_flags: list[bool] = []
        tag_index: dict[str, int] = {}
        output_dim = [3]  # index 0 = input channels

        for p, (block_type, info) in enumerate(entries):
            layer_idx = p + 1
            source = self._resolve_source(info.get("source", -1), layer_idx, tag_index)

            module, out_c = self._build(block_type, info.get("args", {}) or {},
                                        source, output_dim)
            self.layers.append(module)
            self._block_types.append(block_type)
            self._sources.append(source)
            self._output_flags.append(bool(info.get("output", False)))
            output_dim.append(out_c)
            if info.get("tags"):
                tag_index[info["tags"]] = layer_idx

    # -- build helpers ------------------------------------------------------
    @staticmethod
    def _resolve_source(source, layer_idx: int, tag_index: dict[str, int]):
        if isinstance(source, list):
            return [YOLOv7Model._resolve_source(s, layer_idx, tag_index) for s in source]
        if isinstance(source, str):
            return tag_index[source]
        if source < -1:
            return source + layer_idx
        return source

    def _build(self, block_type: str, args: dict, source, output_dim: list[int]):
        def in_ch(src):
            return output_dim[src]

        if block_type == "Conv":
            m = Conv(in_ch(source), args["out_channels"], args["kernel_size"],
                     stride=args.get("stride", 1), groups=args.get("groups", 1))
            return m, m.out_channels
        if block_type == "Pool":
            return Pool(kernel_size=args.get("kernel_size", 2),
                        stride=args.get("stride"),
                        padding=args.get("padding")), output_dim[source]
        if block_type == "UpSample":
            return UpSample(scale_factor=args.get("scale_factor", 2)), output_dim[source]
        if block_type == "SPPCSPConv":
            m = SPPCSPConv(in_ch(source), args["out_channels"])
            return m, m.out_channels
        if block_type == "RepConv":
            m = RepConv(in_ch(source), args["out_channels"])
            return m, m.out_channels
        if block_type == "Concat":
            return _Concat(), sum(output_dim[i] for i in source)
        if block_type == "MultiheadDetection":
            in_channels = [output_dim[i] for i in source]
            return MultiheadDetection(in_channels, self.num_classes), 0
        raise ValueError(f"Unsupported v7 block type: {block_type!r}")

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        y: list[Any] = [x]
        head_out = None
        for module, block_type, source in zip(
            self.layers, self._block_types, self._sources
        ):
            if block_type == "Concat":
                out = torch.cat([y[i] for i in source], dim=1)
            elif block_type == "MultiheadDetection":
                out = module([y[i] for i in source])
                head_out = out
            else:
                out = module(y[source])
            y.append(out)
        return head_out
