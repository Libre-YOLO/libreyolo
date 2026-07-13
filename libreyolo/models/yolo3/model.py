"""LibreYOLO3 — YOLOv3 detector (Darknet-53), cfg-driven.

Source provenance: architecture and pretrained weights derive from
pjreddie/darknet, which is public domain (the "YOLO LICENSE": *"Darknet is
public domain. Do whatever you want with it."*). See
``libreyolo/models/darknet/`` for the shared engine and provenance detail.

Sizes:
    ``t``   yolov3-tiny (416)
    ``b``   yolov3      (416)
    ``spp`` yolov3-spp  (608)
"""

from __future__ import annotations

from ..darknet.family import DarknetFamily


class LibreYOLO3(DarknetFamily):
    FAMILY = "yolo3"
    FILENAME_PREFIX = "LibreYOLO3"
    INPUT_SIZES = {"t": 416, "b": 416, "spp": 608}
    INPUT_SIZE_DIVISOR = 32

    CFG_BY_SIZE = {"t": "yolov3-tiny", "b": "yolov3", "spp": "yolov3-spp"}
    CONV_COUNT_TO_SIZE = {13: "t", 75: "b", 76: "spp"}
    ANCHORS_PER_HEAD = 3
    DEFAULT_SIZE = "b"
