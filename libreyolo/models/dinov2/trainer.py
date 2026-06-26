"""DINOv2 trainer — inherits RF-DETR's trainer with dinov2 family metadata."""

from __future__ import annotations

from ..rfdetr.trainer import RFDETRTrainer


class DINOv2Trainer(RFDETRTrainer):
    """Trainer for the LibreDINOv2 semantic-segmentation family.

    Inherits all training logic from RFDETRTrainer (which handles the
    semantic task path through its ``on_setup``, ``on_forward``, and
    ``get_loss_components`` semantic branches). Only the model-family
    metadata is overridden so saved checkpoints carry
    ``model_family="dinov2"`` instead of ``"rfdetr"``.
    """

    artifact_model_families = ("dinov2",)

    def get_model_family(self) -> str:
        return "dinov2"

    def get_model_tag(self) -> str:
        return f"LibreDINOv2-{self.config.size}"
