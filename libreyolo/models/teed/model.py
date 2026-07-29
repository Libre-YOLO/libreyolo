"""LibreYOLO wrapper for the MIT-licensed TEED edge architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from ..edge_common import (
    EdgeInferenceNet,
    prefix_upstream_state_dict,
    preprocess_numpy,
    unprefixed_keys,
)
from .nn import TEEDCore


class LibreTEED(BaseModel):
    """TEED tiny specialist: RGB image to dense edge probabilities."""

    FAMILY = "teed"
    FILENAME_PREFIX = "LibreTEED"
    WEIGHT_EXT = ".pt"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"t": 352}
    SUPPORTED_TASKS = ("edge",)
    DEFAULT_TASK = "edge"
    REQUIRE_TASK_SUFFIX = True
    TTA_ENABLED = False
    IMGSZ_DIVISOR = 4
    edge_imgsz_divisor = 4
    edge_resize_mode = "stretch"

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys = unprefixed_keys(weights_dict)
        return {
            "block_1.conv1.weight",
            "dblock_3.denselayer1.conv2.weight",
            "block_cat.DWconv1.weight",
            "block_cat.DWconv2.weight",
        }.issubset(keys)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return "t" if cls.can_load(weights_dict) else None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "edge" if cls.can_load(state_dict) else None

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        if not cls.can_load(state_dict):
            return None
        return prefix_upstream_state_dict(state_dict)

    @classmethod
    def get_download_url(cls, filename: str) -> None:
        # Released checkpoints were trained on BIPED, whose terms are
        # non-commercial. LibreYOLO intentionally does not mirror them.
        if cls.detect_size_from_filename(filename) is not None:
            raise ValueError(
                "LibreYOLO does not mirror the upstream BIPED-trained TEED "
                "checkpoint. Provide a checkpoint you are licensed to use, "
                "or convert a local one with weights/convert_teed_weights.py."
            )
        return None

    def __init__(
        self,
        model_path,
        size: str = "t",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        self.nb_classes = 1
        self.names = {0: "edge"}
        self.model.eval()
        if model_path is not None and isinstance(model_path, (str, Path)):
            resolved = Path(self._resolve_weights_path(str(model_path)))
            if not resolved.exists():
                raise FileNotFoundError(
                    f"TEED weights not found at {resolved}. LibreYOLO does not "
                    "mirror the upstream BIPED-trained checkpoint; provide a "
                    "checkpoint you are licensed to use, or convert a local one "
                    "with weights/convert_teed_weights.py."
                )
            self._load_weights(str(resolved))

    def _init_model(self) -> nn.Module:
        return EdgeInferenceNet(TEEDCore())

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"core": self.model.core}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        return prefix_upstream_state_dict(state_dict)

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        image_pil = ImageLoader.load(image, color_format=color_format).convert("RGB")
        original_size = image_pil.size
        size = int(input_size or self._get_input_size())
        if size % self.edge_imgsz_divisor:
            raise ValueError(
                f"TEED imgsz={size} must be divisible by {self.edge_imgsz_divisor}."
            )
        chw, ratio = preprocess_numpy(np.asarray(image_pil), size)
        tensor = torch.from_numpy(chw).unsqueeze(0)
        return tensor, image_pil, original_size, ratio

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        del conf_thres, iou_thres, max_det, kwargs
        if isinstance(output, dict):
            output = output.get("edges", output.get("predictions"))
        if isinstance(output, (list, tuple)):
            output = output[-1]
        original_width, original_height = original_size
        edges = F.interpolate(
            torch.as_tensor(output).float(),
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )
        return {"edges": edges[0, 0].clamp(0.0, 1.0).cpu()}

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "TEED training is not integrated. Train with a compatible local "
            "recipe and convert the checkpoint with "
            "weights/convert_teed_weights.py."
        )


__all__ = ["LibreTEED"]
