"""Cfg-driven Darknet network builder (YOLOv2/v3/v4 lineage).

``DarknetNet`` interprets a parsed cfg (see :mod:`.cfg`) into an
``nn.ModuleList`` in file order and runs the Darknet forward pass, resolving
``[route]``/``[shortcut]`` connections by layer index. Detection layers
(``[yolo]`` / ``[region]``) contribute no learned weights; the network returns
their raw feature maps together with the decode metadata (anchors, strides,
class count, decode kind) that :mod:`libreyolo.postprocess.darknet_yolo`
consumes.

Design note
-----------
Darknet architectures are defined *by* their cfg, so a faithful cfg-driven
builder is the single source of truth shared with the weight reader/converter.
This intentionally differs from LibreYOLO's usual hand-written per-family
``nn.py``: transcribing 100+ Darknet layers by hand (and keeping a separate
converter manifest in sync) is far more error-prone than interpreting the
public-domain cfg directly. Module names are stable and index-based
(``layers.<i>...``) so converted checkpoints load deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .blocks import (
    DarknetConnected,
    DarknetConv,
    DarknetDropout,
    DarknetLocal,
    DarknetMaxPool,
    Reorg,
    Upsample,
)
from .cfg import CfgLayer


@dataclass
class DetectionSpec:
    """Decode metadata for one ``[yolo]``/``[region]``/``[detection]`` head."""

    kind: str  # "yolo" (v3/v4), "region" (v2), or "detection" (v1)
    layer_index: int  # index into the module list (for output ordering)
    anchors: list[tuple[float, float]]  # (w, h) pairs selected for this head
    num_classes: int
    stride: int  # network input / feature-map size
    scale_x_y: float = 1.0  # YOLOv4 center scaling (1.0 = no scaling)
    softmax: bool = False  # YOLOv2 region / YOLOv1 detection softmax over classes
    # YOLOv1 ``[detection]`` head (anchorless S*S grid, boxes from a dense FC
    # output rather than a conv feature map):
    side: int = 0  # S: grid is side x side (7 for YOLOv1)
    boxes_per_cell: int = 0  # B: predicted boxes per cell (2 for YOLOv1)
    coords: int = 4  # box coordinates per box (x, y, w, h)
    sqrt: bool = False  # YOLOv1 predicts sqrt(w)/sqrt(h); square them at decode


def _route_indices(layer_idx: int, raw: list[int]) -> list[int]:
    """Resolve cfg route ``layers=`` entries to absolute indices."""
    resolved = []
    for r in raw:
        resolved.append(r if r >= 0 else layer_idx + r)
    return resolved


def _num_anchors_for_head(layer: CfgLayer) -> int:
    """Anchors used by a ``[yolo]``/``[region]`` head (matches decode)."""
    pairs = len(layer.get_float_list("anchors")) // 2
    if layer.type == "yolo":
        mask = layer.get_int_list("mask")
        return len(mask) if mask else pairs
    return pairs  # region uses all anchor pairs


class DarknetNet(nn.Module):
    """Darknet network built from a parsed cfg."""

    def __init__(
        self,
        net_options: dict[str, Any],
        layers_cfg: list[CfgLayer],
        num_classes: int | None = None,
    ):
        super().__init__()
        self.input_channels = int(net_options.get("channels", 3))
        self.input_width = int(net_options.get("width", 416))
        self.input_height = int(net_options.get("height", 416))
        self._layer_meta: list[dict[str, Any]] = []
        self.detections: list[DetectionSpec] = []

        modules = nn.ModuleList()
        # out_channels[i] = channel count produced by module i (or -1 for the
        # input feeding module 0). Index into this list is offset by 1 so that
        # out_channels[-1] is the network input channel count.
        out_channels = [self.input_channels]
        # Spatial size (h, w) produced by each module, same +1 offset convention.
        # ``[connected]``/``[local]`` heads (YOLOv1) need it at build time.
        spatial = [(self.input_height, self.input_width)]

        for i, layer in enumerate(layers_cfg):
            prev_ch = out_channels[-1]
            prev_h, prev_w = spatial[-1]
            next_layer = layers_cfg[i + 1] if i + 1 < len(layers_cfg) else None
            module, ch, (out_h, out_w), meta = self._build_layer(
                i, layer, prev_ch, prev_h, prev_w, out_channels, spatial,
                num_classes, next_layer,
            )
            modules.append(module)
            out_channels.append(ch)
            spatial.append((out_h, out_w))
            self._layer_meta.append(meta)

        self.layers = modules
        self._finalize_detection_strides()

    def _build_layer(
        self,
        i: int,
        layer: CfgLayer,
        prev_ch: int,
        prev_h: int,
        prev_w: int,
        out_channels: list[int],
        spatial: list[tuple[int, int]],
        num_classes_override: int | None,
        next_layer: CfgLayer | None = None,
    ) -> tuple[nn.Module, int, tuple[int, int], dict[str, Any]]:
        t = layer.type
        meta: dict[str, Any] = {"type": t}

        if t == "convolutional":
            filters = layer.get_int("filters")
            size = layer.get_int("size", 1)
            stride = layer.get_int("stride", 1)
            bn = bool(layer.get_int("batch_normalize", 0))
            pad = bool(layer.get_int("pad", 0)) or size == 1
            act = str(layer.get("activation", "linear"))
            # A detection-head conv's channel count is anchors*(5+nc). The cfg
            # hardcodes it for the cfg's own class count; when a caller overrides
            # nc (custom fine-tunes, head rebuild) we must resize the head conv
            # to match, mirroring how Darknet users edit ``filters=`` by hand.
            if (
                num_classes_override is not None
                and next_layer is not None
                and next_layer.type in ("yolo", "region")
            ):
                filters = _num_anchors_for_head(next_layer) * (5 + num_classes_override)
            module = DarknetConv(prev_ch, filters, size, stride, bn, act, pad=pad)
            pad_px = (size // 2) if pad else 0
            out_h = (prev_h + 2 * pad_px - size) // stride + 1
            out_w = (prev_w + 2 * pad_px - size) // stride + 1
            return module, filters, (out_h, out_w), meta

        if t == "maxpool":
            size = layer.get_int("size", 2)
            stride = layer.get_int("stride", 2)
            if stride == 1 and size == 2:
                out_h, out_w = prev_h, prev_w  # same-size stride-1 pool
            else:
                pad_px = (size - 1) // 2
                out_h = (prev_h + 2 * pad_px - size) // stride + 1
                out_w = (prev_w + 2 * pad_px - size) // stride + 1
            return DarknetMaxPool(size, stride), prev_ch, (out_h, out_w), meta

        if t == "upsample":
            stride = layer.get_int("stride", 2)
            return Upsample(stride), prev_ch, (prev_h * stride, prev_w * stride), meta

        if t == "reorg":
            stride = layer.get_int("stride", 2)
            return (
                Reorg(stride),
                prev_ch * stride * stride,
                (prev_h // stride, prev_w // stride),
                meta,
            )

        if t == "route":
            raw = layer.get_int_list("layers")
            indices = _route_indices(i, raw)
            groups = layer.get_int("groups", 1)
            group_id = layer.get_int("group_id", 0)
            meta["indices"] = indices
            meta["groups"] = groups
            meta["group_id"] = group_id
            ch = sum(out_channels[idx + 1] for idx in indices)
            if groups > 1:
                ch = ch // groups
            # route adopts the spatial size of its (first) source layer
            src_h, src_w = spatial[indices[0] + 1]
            return nn.Identity(), ch, (src_h, src_w), meta

        if t == "shortcut":
            frm = layer.get_int("from")
            frm_abs = frm if frm >= 0 else i + frm
            meta["from"] = frm_abs
            act = str(layer.get("activation", "linear"))
            meta["activation"] = act
            # channel/spatial count follows the previous layer (residual add)
            return nn.Identity(), prev_ch, (prev_h, prev_w), meta

        if t == "dropout":
            # No parameters, spatial/channels unchanged. Present so module indices
            # stay aligned with the .weights byte stream (dropout writes nothing).
            return DarknetDropout(), prev_ch, (prev_h, prev_w), meta

        if t == "local":
            filters = layer.get_int("filters")
            size = layer.get_int("size", 1)
            stride = layer.get_int("stride", 1)
            # Unlike [convolutional], Darknet's local layer has no implicit
            # size==1 padding enable; pad comes from the cfg alone. (The im2col
            # output-size formula below is only exercised for size=3, the sole
            # kernel size YOLOv1's [local] uses.)
            pad = bool(layer.get_int("pad", 0))
            act = str(layer.get("activation", "linear"))
            pad_px = (size // 2) if pad else 0
            out_h = (prev_h + 2 * pad_px - size) // stride + 1
            out_w = (prev_w + 2 * pad_px - size) // stride + 1
            module = DarknetLocal(
                prev_ch, filters, size, stride, pad_px, act, out_h, out_w
            )
            return module, filters, (out_h, out_w), meta

        if t == "connected":
            out_features = layer.get_int("output")
            act = str(layer.get("activation", "linear"))
            in_features = prev_ch * prev_h * prev_w
            module = DarknetConnected(in_features, out_features, act)
            # A fully-connected head collapses the spatial grid to 1x1.
            return module, out_features, (1, 1), meta

        if t in ("yolo", "region"):
            nc = (
                num_classes_override
                if num_classes_override is not None
                else layer.get_int("classes", 80)
            )
            all_anchors = layer.get_float_list("anchors")
            pairs = list(zip(all_anchors[0::2], all_anchors[1::2]))
            if t == "yolo":
                mask = layer.get_int_list("mask")
                selected = [pairs[m] for m in mask] if mask else pairs
                spec = DetectionSpec(
                    kind="yolo",
                    layer_index=i,
                    anchors=selected,
                    num_classes=nc,
                    stride=0,  # filled in _finalize_detection_strides
                    scale_x_y=layer.get_float("scale_x_y", 1.0) or 1.0,
                    softmax=False,
                )
            else:  # region (YOLOv2)
                spec = DetectionSpec(
                    kind="region",
                    layer_index=i,
                    anchors=pairs,
                    num_classes=nc,
                    stride=0,
                    scale_x_y=1.0,
                    softmax=bool(layer.get_int("softmax", 0)),
                )
            self.detections.append(spec)
            meta["detection"] = spec
            return nn.Identity(), prev_ch, (prev_h, prev_w), meta

        if t == "detection":
            # YOLOv1 anchorless head: decode a dense S*S*(C + B*(coords+1)) FC
            # vector into a grid. No learned params (the [connected] before it
            # holds the weights); this layer only carries decode metadata.
            nc = (
                num_classes_override
                if num_classes_override is not None
                else layer.get_int("classes", 20)
            )
            spec = DetectionSpec(
                kind="detection",
                layer_index=i,
                anchors=[],
                num_classes=nc,
                stride=0,  # filled in _finalize_detection_strides (input / side)
                softmax=bool(layer.get_int("softmax", 0)),
                side=layer.get_int("side", 7),
                boxes_per_cell=layer.get_int("num", 2),
                coords=layer.get_int("coords", 4),
                sqrt=bool(layer.get_int("sqrt", 0)),
            )
            self.detections.append(spec)
            meta["detection"] = spec
            return nn.Identity(), prev_ch, (prev_h, prev_w), meta

        raise ValueError(f"Unsupported Darknet layer type: {t!r}")

    def _finalize_detection_strides(self) -> None:
        """Fill each detection head's stride from the input/feature-map ratio.

        Strides are computed from the cumulative downsampling implied by the
        cfg, using the network input size as the reference. We derive stride by
        replaying spatial size through the layer metadata rather than assuming
        the canonical 8/16/32, so tiny/spp variants are handled uniformly.
        """
        # Track spatial downsample factor per layer output.
        factors = [1]  # factors[-1] == input (factor 1)
        for i, meta in enumerate(self._layer_meta):
            prev = factors[-1]
            t = meta["type"]
            if t == "convolutional":
                # stride read back from the built module
                stride = self.layers[i].conv.stride[0]
                factors.append(prev * stride)
            elif t == "maxpool":
                stride = self.layers[i].stride
                factors.append(prev * stride)
            elif t == "upsample":
                factors.append(prev // self.layers[i].stride)
            elif t == "reorg":
                factors.append(prev * self.layers[i].stride)
            elif t == "route":
                # route adopts the factor of its (first) source layer
                idx = meta["indices"][0]
                factors.append(factors[idx + 1])
            elif t == "shortcut":
                factors.append(prev)
            else:  # local/connected/dropout/yolo/region/detection: same factor
                factors.append(prev)

        for spec in self.detections:
            if spec.kind == "detection":
                # YOLOv1: the decode grid is fixed at S x S, so the effective
                # stride is input / side regardless of the FC collapse to 1x1.
                spec.stride = int(self.input_width // spec.side)
            else:
                spec.stride = int(factors[spec.layer_index + 1])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the Darknet forward pass.

        Returns the raw feature maps of the detection heads in cfg order. Decode
        metadata is available on :attr:`detections` (aligned to output order).
        """
        outputs: list[torch.Tensor] = []
        detection_outputs: list[torch.Tensor] = []

        for i, (module, meta) in enumerate(zip(self.layers, self._layer_meta)):
            t = meta["type"]
            if t == "route":
                indices = meta["indices"]
                feats = [outputs[idx] for idx in indices]
                y = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
                groups = meta["groups"]
                if groups > 1:
                    chunk = y.shape[1] // groups
                    y = y[:, meta["group_id"] * chunk : (meta["group_id"] + 1) * chunk]
            elif t == "shortcut":
                y = outputs[-1] + outputs[meta["from"]]
                if meta.get("activation") == "leaky":
                    y = torch.nn.functional.leaky_relu(y, 0.1, inplace=True)
            elif t in ("yolo", "region", "detection"):
                y = outputs[-1]
                detection_outputs.append(y)
            else:
                y = module(outputs[-1] if outputs else x)

            outputs.append(y)

        return detection_outputs

    # -- introspection ------------------------------------------------------
    def detection_specs(self) -> list[DetectionSpec]:
        return list(self.detections)
