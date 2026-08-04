"""YOLOv9 E2E trainer."""

import torch

from .config import YOLO9E2EConfig
from ..yolo9.trainer import YOLO9Trainer


class YOLO9E2ETrainer(YOLO9Trainer):
    """Thin trainer subclass for yolo9_e2e family metadata and defaults."""

    @classmethod
    def _config_class(cls):
        return YOLO9E2EConfig

    def get_model_family(self) -> str:
        return "yolo9_e2e"

    def get_model_tag(self) -> str:
        return f"YOLOv9-E2E-{self.config.size}"

    def validate_validation_loss_config(self) -> None:
        if not getattr(self.config, "val_loss", False):
            return

        from .nn import LibreYOLO9E2EModel, YOLO9E2EDetect

        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        standard_model = (
            type(self.model) is LibreYOLO9E2EModel
            and type(self.model.head) is YOLO9E2EDetect
        )
        if task != "detect" or not standard_model:
            raise ValueError(
                "val_loss=True currently supports YOLO9-E2E detection only; "
                "non-detect tasks are not supported"
            )

    def build_validation_loss_adapter(self, model: torch.nn.Module):
        from .validation_loss import YOLO9E2EValidationLoss

        return YOLO9E2EValidationLoss(
            model,
            max_labels=int(getattr(self.config, "max_labels", 100)),
        )
