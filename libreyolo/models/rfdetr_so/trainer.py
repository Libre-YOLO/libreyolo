"""RF-DETR-SO trainer."""

from typing import Type

from ...training.config import TrainConfig
from ..rfdetr.trainer import RFDETRTrainer
from .config import RFDETRSOConfig


class RFDETRSOTrainer(RFDETRTrainer):
    """Thin trainer subclass for rfdetr_so family metadata and defaults."""

    artifact_model_families = ("rfdetr_so",)

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return RFDETRSOConfig

    def get_model_family(self) -> str:
        return "rfdetr_so"

    def get_model_tag(self) -> str:
        return f"LibreRFDETRSO-{self.config.size}"
