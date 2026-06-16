"""LibreFOMO — FOMO point-localizer family wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.model import BaseModel
from .nn import CONFIGS, LibreFOMOModel, detect_size_from_state_dict
from .utils import postprocess as postprocess_fomo
from ...training.config import FOMOConfig
from ...training.ddp_spawn import ddp_aware
from ...validation.preprocessors import FOMOValPreprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LibreFOMO — BaseModel subclass
# ---------------------------------------------------------------------------


class LibreFOMO(BaseModel):
    """LibreFOMO point localizer: image → (x, y, class, confidence) keypoint predictions."""

    FAMILY = "fomo"
    FILENAME_PREFIX = "LibreFOMO"
    WEIGHT_EXT = ".pt"

    # Input sizes keyed by variant code (matches nn.CONFIGS)
    INPUT_SIZES: ClassVar[Dict[str, int]] = {k: int(v["imgsz"]) for k, v in CONFIGS.items()}

    SUPPORTED_TASKS = ("point",)
    DEFAULT_TASK = "point"
    TRAIN_CONFIG = FOMOConfig
    val_preprocessor_class = FOMOValPreprocessor

    # TTA is not meaningful for heatmap point models
    TTA_ENABLED = False

    # HuggingFace weights repo for future weight releases
    _HF_REPO = "LibreYOLO/LibreFOMO"
    _LICENSE_NOTICE_SHOWN = False
    _WEIGHTS_REPO = "fomo-edge-ai/FOMO"

    # -------------------------------------------------------------------------
    # Registry / can_load interface
    # -------------------------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return (
            "head.weight" in weights_dict
            and any(k.startswith("backbone.block_6_expand") for k in weights_dict)
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return detect_size_from_state_dict(weights_dict)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        weight = weights_dict.get("head.weight")
        if weight is None:
            return None
        # head output channels = nc + 1 (background channel)
        return max(int(weight.shape[0]) - 1, 1)

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        size = cls.detect_size_from_filename(filename)
        if size is None:
            return None
        # Map LibreFOMO{size}.pt to FOMO{size}.pt for the external repository
        fomo_filename = f"FOMO{size}.pt"
        return f"https://huggingface.co/{cls._WEIGHTS_REPO}/resolve/main/weights/{fomo_filename}"

    @classmethod
    def _notify_license_once(cls) -> None:
        """Print the FOMO weights license notice once."""
        if cls._LICENSE_NOTICE_SHOWN:
            return
        cls._LICENSE_NOTICE_SHOWN = True
        print(
            "\n"
            "----------------------------------------------------------------\n"
            f"LibreFOMO weights are hosted externally at huggingface.co/{cls._WEIGHTS_REPO}\n"
            "under cc-by-nc-4.0.\n"
            "They are treated as externally hosted, ImageNet-derived weights and\n"
            "are not redistributed by LibreYOLO. By downloading them you accept\n"
            "the Hugging Face repository license terms.\n"
            "----------------------------------------------------------------\n"
        )

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def __init__(
        self,
        model_path: str | Path | None = None,
        size: str = "m",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task,
            **kwargs,
        )
        if isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))

    def _load_weights(self, model_path: str) -> None:
        self._notify_license_once()
        super()._load_weights(model_path)

    # -------------------------------------------------------------------------
    # BaseModel abstract surface
    # -------------------------------------------------------------------------

    def _init_model(self) -> nn.Module:
        return LibreFOMOModel(size=self.size, nc=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"backbone": self.model.backbone, "head": self.model.head}

    @staticmethod
    def _get_preprocess_numpy():
        return FOMOValPreprocessor

    def _preprocess(
        self,
        image: Any,
        color_format: str = "auto",
        input_size: int | None = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        import numpy as np
        from PIL import Image as _Image

        from ...utils.image_loader import ImageLoader

        img = ImageLoader.load(image, color_format=color_format)
        pil_img = img.convert("RGB")
        isz = int(input_size or self._get_input_size())
        pil_resized = pil_img.resize((isz, isz), _Image.Resampling.BILINEAR)
        arr = np.asarray(pil_resized, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        chw = np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)
        tensor = torch.from_numpy(chw).unsqueeze(0)
        return tensor, pil_img, pil_img.size, 1.0

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if output.dim() == 3:
            output = output.unsqueeze(0)
        return postprocess_fomo(
            output,
            conf_thres=conf_thres,
            input_size=self._get_input_size(),
            original_size=original_size,
            nms_radius=int(kwargs.get("nms_radius", 1)),
            max_det=max_det,
        )

    # -------------------------------------------------------------------------
    # PointValidator integration hooks
    # -------------------------------------------------------------------------

    def _parse_gt_points(
        self,
        gt_row: Any,
        orig_h: int,
        orig_w: int,
        validator: Any,
    ) -> Tuple[Any, Any]:
        return validator.parse_gt_points_from_boxes(gt_row, orig_h, orig_w)

    # -------------------------------------------------------------------------
    # Training (experimental — requires allow_experimental=True)
    # -------------------------------------------------------------------------

    @ddp_aware(experimental_key="allow_experimental")
    def train(self, allow_experimental: bool = False, **kwargs: Any) -> Dict:
        """Train LibreFOMO."""
        if not allow_experimental:
            raise NotImplementedError(
                "LibreFOMO training is experimental in this version. "
                "Pass allow_experimental=True to model.train() to enable it."
            )

        from .trainer import FOMOTrainer

        if "imgsz" not in kwargs:
            kwargs["imgsz"] = self._get_input_size()
        if "size" not in kwargs:
            kwargs["size"] = self.size
        if "num_classes" not in kwargs:
            kwargs["num_classes"] = self.nb_classes

        trainer = FOMOTrainer(model=self.model, wrapper_model=self, **kwargs)
        results = trainer.train()

        best_ckpt = results.get("best_checkpoint", "")
        last_ckpt = results.get("last_checkpoint", "")
        reload_path = best_ckpt if Path(best_ckpt).exists() else last_ckpt
        if reload_path and Path(reload_path).exists():
            self._load_weights(reload_path)

        return results
