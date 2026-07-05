"""Training config for YOLOv9-P2 (stride-4 small-object variant)."""

from dataclasses import dataclass

from ...training.config import YOLO9Config


@dataclass(kw_only=True)
class YOLO9P2Config(YOLO9Config):
    """YOLOv9-P2 uses the YOLOv9 training defaults."""

    name: str = "yolo9_p2_exp"
