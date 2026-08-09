"""LibreYOLO4 — YOLOv4 detector (CSPDarknet-53 + SPP + PANet), cfg-driven.

Source provenance: architecture and pretrained weights derive from
AlexeyAB/darknet, which ships under the public-domain "YOLO LICENSE". See
``libreyolo/models/darknet/`` for the shared engine and provenance detail.

Sizes:
    ``t`` yolov4-tiny (416)
    ``b`` yolov4      (608)

Notes: YOLOv4 uses Mish activations (backbone), an SPP block, a PANet neck, and
per-head ``scale_x_y`` center scaling — all handled by the shared Darknet engine
and decoder.
"""

from __future__ import annotations

from ..darknet.family import DarknetFamily


class LibreYOLO4(DarknetFamily):
    FAMILY = "yolo4"
    FILENAME_PREFIX = "LibreYOLO4"
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    INPUT_SIZES = {"t": 416, "b": 608}

    CFG_BY_SIZE = {"t": "yolov4-tiny", "b": "yolov4"}
    CONV_COUNT_TO_SIZE = {21: "t", 110: "b"}
    ANCHORS_PER_HEAD = 3
    DEFAULT_SIZE = "b"
