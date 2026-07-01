"""LibreNAFNet image-restoration model family."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.nafnet import postprocess as _nafnet_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .config import NAFNetConfig
from .nn import NAFNetLocal
from .utils import preprocess_image, preprocess_numpy


NAFNET_SIZE_CONFIGS: dict[str, dict[str, object]] = {
    "s": {
        "width": 32,
        "middle_blk_num": 1,
        "enc_blk_nums": [1, 1, 1, 28],
        "dec_blk_nums": [1, 1, 1, 1],
    },
    "l": {
        "width": 64,
        "middle_blk_num": 1,
        "enc_blk_nums": [1, 1, 1, 28],
        "dec_blk_nums": [1, 1, 1, 1],
    },
}
_TRAIN_DEFAULTS = NAFNetConfig()


class LibreNAFNet(BaseModel):
    """NAFNet RGB image restoration.

    Predict runs at native image resolution, pads to the network downsample
    factor, and returns ``Results.restored`` on the original canvas.
    """

    FAMILY = "nafnet"
    FILENAME_PREFIX = "LibreNAFNet"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": 256, "l": 256}
    SUPPORTED_TASKS = ("restore",)
    DEFAULT_TASK = "restore"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = NAFNetConfig
    SUPPORTS_BATCHED_PREDICT = True
    TTA_ENABLED = False

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return (
            "intro.weight" in weights_dict
            and "ending.weight" in weights_dict
            and any(k.startswith("encoders.") for k in weights_dict)
            and cls.detect_size(weights_dict) is not None
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        intro = weights_dict.get("intro.weight")
        if intro is None or getattr(intro, "ndim", 0) != 4:
            return None
        return {32: "s", 64: "l"}.get(int(intro.shape[0]))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "restore" if cls.can_load(state_dict) else None

    def __init__(
        self,
        model_path=None,
        size: str = "s",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.nb_classes = 1
        self.names = {0: "image"}

    def _init_model(self) -> nn.Module:
        cfg = NAFNET_SIZE_CONFIGS[self.size]
        return NAFNetLocal(
            img_channel=3,
            width=int(cfg["width"]),
            middle_blk_num=int(cfg["middle_blk_num"]),
            enc_blk_nums=cfg["enc_blk_nums"],
            dec_blk_nums=cfg["dec_blk_nums"],
            train_size=(1, 3, self.input_size, self.input_size),
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "intro": self.model.intro,
            "encoders": self.model.encoders,
            "middle_blks": self.model.middle_blks,
            "decoders": self.model.decoders,
            "ending": self.model.ending,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        del input_size
        return preprocess_image(
            image,
            pad_multiple=int(getattr(self.model, "padder_size", 16)),
            color_format=color_format,
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
        del conf_thres, iou_thres, max_det, ratio, kwargs
        return {"restored": _nafnet_postprocess(output, original_size)}

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
        degradation: str | None = None,
        dataset: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Fine-tune NAFNet on paired degraded/target RGB images."""

        from .trainer import NAFNetTrainer

        if imgsz is None:
            imgsz = self.input_size

        trainer = NAFNetTrainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=1,
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
            degradation=degradation,
            dataset=dataset,
            **kwargs,
        )

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreNAFNet('path/to/last.pt', size='s'); "
                    "model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()
        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self._load_weights(best_ckpt)
        return results


__all__ = ["LibreNAFNet", "NAFNET_SIZE_CONFIGS"]
