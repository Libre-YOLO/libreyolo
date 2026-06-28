"""LibreResNet: BaseModel subclass wiring ResNet classification into the factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...postprocess.resnet import postprocess as _resnet_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .config import ResNetConfig
from .nn import ResNet
from .utils import preprocess_image as _resnet_preprocess

_TRAIN_DEFAULTS = ResNetConfig()


class LibreResNet(BaseModel):
    """ResNet image classifier (18/34/50/101).

    Examples::

        >>> model = LibreYOLO("LibreResNet50-cls.pt")
        >>> result = model.predict("cat.jpg")[0]
        >>> result.probs.top1, result.probs.top5

        >>> model = LibreResNet(size="50")
        >>> model.train(data="imagenette160", epochs=5)
    """

    FAMILY = "resnet"
    FILENAME_PREFIX = "LibreResNet"
    INPUT_SIZES = {"18": 224, "34": 224, "50": 224, "101": 224}
    SUPPORTED_TASKS = ("classify",)
    DEFAULT_TASK = "classify"
    TRAIN_CONFIG = ResNetConfig

    # timm a1 eval crop_pct (matches the upstream benchmark preprocessing).
    CROP_PCT = {"18": 0.95, "34": 0.95, "50": 0.95, "101": 0.95}

    # ---- registry --------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # A *standalone* ResNet classifier: top-level conv1 + fc + first stage.
        # ResNet also appears as a BACKBONE inside other families (l2cs gaze is a
        # ResNet; rt-detr uses a ResNet backbone) — but there its keys are
        # ``backbone.conv1.*`` and the head is not a plain ``fc``. Requiring
        # top-level ``conv1.weight`` + ``fc.weight`` + ``layer1.0.conv1.weight``
        # matches only the standalone classifier and rejects those embeds.
        return (
            "conv1.weight" in weights_dict
            and "fc.weight" in weights_dict
            and "layer1.0.conv1.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        if "conv1.weight" not in weights_dict:
            return None
        # Bottleneck (50/101) has a conv3 in each block; BasicBlock (18/34) does not.
        bottleneck = "layer1.0.conv3.weight" in weights_dict
        if bottleneck:
            # layer3 depth: resnet50 has 6 blocks (0-5), resnet101 has 23 (0-22).
            return "101" if "layer3.6.conv1.weight" in weights_dict else "50"
        # layer3 depth: resnet18 has 2 blocks (0-1), resnet34 has 6 (0-5).
        return "34" if "layer3.2.conv1.weight" in weights_dict else "18"

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "fc.weight"
        if key not in weights_dict:
            return None
        return int(weights_dict[key].shape[0])

    # ---- init ------------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "50",
        nb_classes: int = 1000,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task,
            **kwargs,
        )
        self.crop_pct = self.CROP_PCT[self.size]
        self.interpolation = "bicubic"
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        return ResNet(size=self.size, num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "conv1": self.model.conv1,
            "layer1": self.model.layer1,
            "layer2": self.model.layer2,
            "layer3": self.model.layer3,
            "layer4": self.model.layer4,
            "fc": self.model.fc,
        }

    def _rebuild_for_new_classes(self, new_nb_classes: int) -> None:
        """Swap the final Linear for a new class count (backbone preserved)."""
        self.nb_classes = new_nb_classes
        self.names = {i: f"class_{i}" for i in range(new_nb_classes)}
        self.model.reset_classifier(new_nb_classes)
        self.model.to(self.device)

    # ---- inference -------------------------------------------------------

    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy

        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        eff = input_size if input_size is not None else self.input_size
        return _resnet_preprocess(
            image, input_size=eff, crop_pct=self.crop_pct, color_format=color_format
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        return _resnet_postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
            ratio=ratio,
        )

    # ---- training --------------------------------------------------------

    def train(
        self,
        data: str,
        *,
        epochs: int = _TRAIN_DEFAULTS.epochs,
        batch: int = _TRAIN_DEFAULTS.batch,
        imgsz: int | None = None,
        lr0: float = _TRAIN_DEFAULTS.lr0,
        optimizer: str = _TRAIN_DEFAULTS.optimizer,
        device: str = "",
        workers: int = _TRAIN_DEFAULTS.workers,
        seed: int = _TRAIN_DEFAULTS.seed,
        project: str = _TRAIN_DEFAULTS.project,
        name: str = _TRAIN_DEFAULTS.name,
        exist_ok: bool = _TRAIN_DEFAULTS.exist_ok,
        resume: bool = _TRAIN_DEFAULTS.resume,
        amp: bool = _TRAIN_DEFAULTS.amp,
        patience: int = _TRAIN_DEFAULTS.patience,
        **kwargs: Any,
    ) -> dict:
        """Fine-tune the classifier on an ImageFolder-style dataset.

        ``data`` is a dataset root (``train/`` + ``val/`` folder-per-class), a
        known name (e.g. ``"imagenette160"``), or a ``.zip`` URL. The head is
        rebuilt to the dataset's class count automatically. Cross-entropy +
        AdamW + cosine; the ImageNet-pretrained backbone transfers cleanly.
        """
        from .trainer import ResNetTrainer

        if imgsz is None:
            imgsz = self.input_size

        trainer = ResNetTrainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            optimizer=optimizer.lower(),
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            patience=patience,
            **kwargs,
        )

        results = trainer.train()
        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self._load_weights(best_ckpt)
        return results
