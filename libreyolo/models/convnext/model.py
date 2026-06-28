"""LibreConvNeXt: BaseModel subclass wiring ConvNeXt classification into the factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...postprocess.convnext import postprocess as _cnx_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .config import ConvNeXtConfig
from .nn import ConvNeXt
from .utils import preprocess_image as _cnx_preprocess

_TRAIN_DEFAULTS = ConvNeXtConfig()


class LibreConvNeXt(BaseModel):
    """ConvNeXt V1 image classifier (tiny/small/base).

    Examples::

        >>> model = LibreYOLO("LibreConvNeXtt-cls.pt")
        >>> result = model.predict("cat.jpg")[0]
        >>> result.probs.top1, result.probs.top5

        >>> model = LibreConvNeXt(size="t")
        >>> model.train(data="imagenette160", epochs=5)
    """

    FAMILY = "convnext"
    FILENAME_PREFIX = "LibreConvNeXt"
    INPUT_SIZES = {"t": 224, "s": 224, "b": 224}
    SUPPORTED_TASKS = ("classify",)
    DEFAULT_TASK = "classify"
    TRAIN_CONFIG = ConvNeXtConfig

    # timm eval crop_pct per checkpoint — convnext_*.fb_in1k all use 0.875.
    CROP_PCT = {"t": 0.875, "s": 0.875, "b": 0.875}

    # ---- registry --------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # ConvNeXt signature: patch stem + layer-scale gamma + LayerNorm head fc.
        # The gamma layer-scale parameter is unique to ConvNeXt among all
        # families (detectors and MobileNetV4 have no per-block ``gamma``).
        return (
            "stem.0.weight" in weights_dict
            and "head.fc.weight" in weights_dict
            and any(k.endswith(".gamma") and k.startswith("stages.") for k in weights_dict)
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        key = "stem.0.weight"
        if key not in weights_dict:
            return None
        dim0 = int(weights_dict[key].shape[0])
        if dim0 == 128:
            return "b"  # base widens to 128/256/512/1024
        # tiny and small share dims (96..768); they differ in stage-2 depth
        # (tiny=9 blocks -> max index 8; small=27 -> index 9 exists).
        if "stages.2.blocks.9.gamma" in weights_dict:
            return "s"
        return "t"

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "head.fc.weight"
        if key not in weights_dict:
            return None
        return int(weights_dict[key].shape[0])

    # ---- init ------------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "t",
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
        return ConvNeXt(size=self.size, num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "stem": self.model.stem,
            "stages": self.model.stages,
            "head_norm": self.model.head.norm,
            "classifier": self.model.head.fc,
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
        return _cnx_preprocess(
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
        return _cnx_postprocess(
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
        from .trainer import ConvNeXtTrainer

        if imgsz is None:
            imgsz = self.input_size

        trainer = ConvNeXtTrainer(
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
