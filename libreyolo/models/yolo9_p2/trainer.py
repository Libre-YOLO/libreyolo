"""YOLOv9-P2 trainer."""

from ..yolo9.trainer import YOLO9Trainer
from .config import YOLO9P2Config


class YOLO9P2Trainer(YOLO9Trainer):
    """Thin trainer subclass for yolo9_p2 family metadata and freeze groups."""

    artifact_model_families = ("yolo9_p2",)

    # Same freeze order as YOLOv9 with the P2 neck extension inserted at its
    # topological position (top-down onto P2, then bottom-up back to P3).
    _NECK_FREEZE_MODULES = (
        "elan_up1",
        "elan_up2",
        "elan_up3",
        "down0",
        "elan_down0",
        "down1",
        "elan_down1",
        "down2",
        "elan_down2",
    )

    @classmethod
    def _config_class(cls):
        return YOLO9P2Config

    def get_model_family(self) -> str:
        return "yolo9_p2"

    def get_model_tag(self) -> str:
        return f"YOLOv9-P2-{self.config.size}"
