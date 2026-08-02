"""Mask R-CNN architecture placeholder for the family skeleton commit."""

from __future__ import annotations

from ..faster_rcnn.nn import LibreFasterRCNNModel


class LibreMaskRCNNModel(LibreFasterRCNNModel):
    """Temporary shared two-stage graph, replaced by the mask graph next commit."""

    def __init__(self, size: str = "r50", num_classes: int = 91) -> None:
        if size != "r50":
            raise ValueError("Mask R-CNN currently ships only size 'r50'.")
        super().__init__(size="l", num_classes=num_classes)
        self.size = size
