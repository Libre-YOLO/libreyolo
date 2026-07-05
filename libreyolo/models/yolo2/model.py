"""LibreYOLO2 — YOLOv2 / YOLO9000 detector (Darknet-19 + reorg), cfg-driven.

Source provenance: architecture and pretrained weights derive from
pjreddie/darknet (public-domain "YOLO LICENSE"). See
``libreyolo/models/darknet/`` for the shared engine and provenance detail.

Sizes:
    ``t`` yolov2-tiny (416)
    ``b`` yolov2      (608)

Notes: YOLOv2 uses a single ``[region]`` head with 5 anchors (in grid-cell
units) and **softmax** class probabilities, plus a ``[reorg]`` passthrough
layer whose channel interleaving is the known conversion trap (see
``darknet/blocks.py::Reorg``). Real-weight parity for v2 must be verified
against a Darknet oracle before shipping (Gate B).
"""

from __future__ import annotations

from ..darknet.family import DarknetFamily


class LibreYOLO2(DarknetFamily):
    FAMILY = "yolo2"
    FILENAME_PREFIX = "LibreYOLO2"
    INPUT_SIZES = {"t": 416, "b": 608}

    CFG_BY_SIZE = {"t": "yolov2-tiny", "b": "yolov2"}
    CONV_COUNT_TO_SIZE = {9: "t", 23: "b"}
    ANCHORS_PER_HEAD = 5
    DEFAULT_SIZE = "b"
