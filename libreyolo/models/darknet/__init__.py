"""Shared Darknet-lineage engine for the YOLOv2/v3/v4 families.

This package is *not* a registered model family. It provides the cfg-driven
network builder, the ``.weights`` reader, and the shared blocks that the
``yolo2`` / ``yolo3`` / ``yolo4`` families are thin wrappers over.

All behaviour is reproduced from the public-domain Darknet source (pjreddie /
AlexeyAB). See ``blocks.py`` and ``reader.py`` for provenance detail and the
bundled ``cfgs/NOTICE``.
"""

from __future__ import annotations

from .cfg import CfgLayer, load_bundled_cfg, parse_cfg_file, parse_cfg_text
from .net import DarknetNet, DetectionSpec
from .reader import (
    load_darknet_weights,
    load_darknet_weights_file,
    save_darknet_weights,
)


def build_darknet(cfg_name: str, num_classes: int | None = None) -> DarknetNet:
    """Build a :class:`DarknetNet` from a bundled cfg name (e.g. ``"yolov3"``)."""
    net_options, layers = load_bundled_cfg(cfg_name)
    return DarknetNet(net_options, layers, num_classes=num_classes)


__all__ = [
    "CfgLayer",
    "DarknetNet",
    "DetectionSpec",
    "build_darknet",
    "load_bundled_cfg",
    "parse_cfg_file",
    "parse_cfg_text",
    "load_darknet_weights",
    "load_darknet_weights_file",
    "save_darknet_weights",
]
