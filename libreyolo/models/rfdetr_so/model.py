"""LibreRFDETRSO inference and training wrapper.

RF-DETR small-object variant (detect only). Compared to stock RF-DETR it
detects on a 3-level pyramid (strides 8/16/32) and carries the SSA raw-image
detail branch plus a PBM bi-fusion neck; see ``rfdetr_so/nn.py`` for the
architecture and the TinyFormer paper grounding.

Loading behavior: RF-DETR-SO checkpoints load as-is. Stock RF-DETR detect
checkpoints (including the upstream pretrained weights the size config
points at) are accepted as transfer sources and remapped onto the 3-level
layout at load time; the SSA/PBM modules and the new projector stages start
freshly initialized. The DINOv2 encoder is frozen by default
(``freeze_encoder=False`` unfreezes it for a final polish phase).

Sizes: s (resolution 512).
"""

from typing import Any, ClassVar

import torch.nn as nn

from ..rfdetr.model import LibreRFDETR
from .config import RFDETRSOConfig
from .nn import RFDETRSO_CONFIGS, LibreRFDETRSOModel, is_so_state_dict


class LibreRFDETRSO(LibreRFDETR):
    """RF-DETR-SO: RF-DETR specialized for small objects (SSA + PBM).

    Args:
        model_path: Path to weights, pre-loaded state_dict, or None (None
            transfer-initializes from the stock pretrained detect weights).
        size: Model size variant ("s").
        nb_classes: Number of classes (default: 80).
        device: Device for inference.
        freeze_encoder: Freeze the DINOv2 encoder (default: True).

    Example::

        >>> model = LibreRFDETRSO(None, size="s")
        >>> model.train(data="coco.yaml", epochs=30)
    """

    FAMILY: ClassVar[str] = "rfdetr_so"
    FILENAME_PREFIX: ClassVar[str] = "LibreRFDETRSO"
    INPUT_SIZES: ClassVar[dict[str, int]] = {"s": 512}
    SUPPORTED_TASKS: ClassVar[tuple[str, ...]] = ("detect",)
    TASK_INPUT_SIZES: ClassVar[dict[str, dict[str, int]]] = {"detect": INPUT_SIZES}
    TRAIN_CONFIG = RFDETRSOConfig
    # Stock RF-DETR detect checkpoints are valid initialization sources
    # (remapped by LibreRFDETRSOModel.load_state_dict).
    TRANSFER_COMPATIBLE_FAMILIES: ClassVar[tuple[str, ...]] = ("rfdetr",)

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Match checkpoints that contain the SSA/PBM small-object modules."""
        return is_so_state_dict(weights_dict)

    @classmethod
    def detect_size(
        cls, weights_dict: dict, state_dict: dict | None = None
    ) -> str | None:
        size = super().detect_size(weights_dict, state_dict)
        return size if size in RFDETRSO_CONFIGS else None

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        model_path: str | dict[str, Any] | None = None,
        size: str | None = None,
        nb_classes: int = 80,
        device: str = "auto",
        freeze_encoder: bool = True,
        **kwargs,
    ):
        self._freeze_encoder = bool(freeze_encoder)
        kwargs.pop("segmentation", None)
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task="detect",
            **kwargs,
        )

    def _init_model(self) -> nn.Module:
        return LibreRFDETRSOModel(
            config=self.size,
            nb_classes=self._model_num_classes,
            device=str(self.device),
            freeze_encoder=self._freeze_encoder,
        )

    def _trainer_class(self):
        from .trainer import RFDETRSOTrainer

        return RFDETRSOTrainer
