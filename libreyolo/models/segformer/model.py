"""LibreYOLO wrapper for SegFormer semantic segmentation.

SegFormer ("Simple and Efficient Design for Semantic Segmentation with
Transformers", Xie et al., NeurIPS 2021) is a lightweight ViT-style encoder
(MiT, Mix Transformer) paired with an all-MLP decode head. This family covers
all six standard sizes, b0 through b5.

LibreSegformer ships NO pretrained weights: the upstream NVIDIA checkpoints
(``nvidia/segformer-b0..b5-*``) are under a non-permissive, non-commercial
license and are never downloaded, converted, or redistributed by LibreYOLO.
Train from scratch on your own semantic dataset via
``LibreSegformer(...).train(...)``. See ``NOTICE`` in this directory for the
full licensing rationale.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...tasks import normalize_task
from ...training.callbacks import TrainCallbacks
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import load_untrusted_torch_file, validate_checkpoint_metadata
from ..base.model import BaseModel
from .nn import SIZE_CONFIGS, LibreSegformerNet

logger = logging.getLogger(__name__)


def _input_size_hw(input_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(input_size, int):
        return input_size, input_size
    if len(input_size) != 2:
        raise ValueError(f"input_size must be int or (height, width), got {input_size!r}")
    return int(input_size[0]), int(input_size[1])


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int | tuple[int, int] = 512,
) -> tuple[np.ndarray, float]:
    """Letterbox RGB image to SegFormer's canvas as CHW float32 in [0, 1]."""
    orig_h, orig_w = img_rgb_hwc.shape[:2]
    input_h, input_w = _input_size_hw(input_size)
    ratio = min(input_h / orig_h, input_w / orig_w)
    new_h = max(int(orig_h * ratio), 1)
    new_w = max(int(orig_w * ratio), 1)

    resized = cv2.resize(img_rgb_hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    arr = np.ascontiguousarray(padded, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), ratio


class LibreSegformer(BaseModel):
    """SegFormer b0-b5 family for dense semantic segmentation, trained from scratch."""

    FAMILY: ClassVar[str] = "segformer"
    FILENAME_PREFIX: ClassVar[str] = "LibreSegformer"
    WEIGHT_EXT: ClassVar[str] = ".pt"
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("semantic",)
    DEFAULT_TASK: ClassVar[str] = "semantic"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    INPUT_SIZES: ClassVar[Dict[str, int]] = {size: 512 for size in SIZE_CONFIGS}

    semantic_resize_mode: ClassVar[str] = "letterbox"
    semantic_imgsz_divisor: ClassVar[int] = 32

    # ------------------------------------------------------------------
    # Registry / can_load interface
    # ------------------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys = set(weights_dict)
        return (
            "decode_head.linear_fuse.weight" in keys
            and "decode_head.classifier.weight" in keys
            and "encoder.stages.0.patch_embeddings.proj.weight" in keys
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        stem = weights_dict.get("encoder.stages.0.patch_embeddings.proj.weight")
        if stem is None or getattr(stem, "ndim", 0) < 1:
            return None
        embed_dim0 = int(stem.shape[0])
        if embed_dim0 == 32:
            return "b0"
        if embed_dim0 != 64:
            return None

        depth_by_size = {2: "b1", 6: "b2", 18: "b3", 27: "b4", 40: "b5"}
        stage2_prefix = "encoder.stages.2.blocks."
        block_indices = set()
        for key in weights_dict:
            if key.startswith(stage2_prefix):
                remainder = key[len(stage2_prefix) :]
                idx_str = remainder.split(".", 1)[0]
                if idx_str.isdigit():
                    block_indices.add(int(idx_str))
        depth = len(block_indices)
        return depth_by_size.get(depth)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        head = weights_dict.get("decode_head.classifier.weight")
        if head is not None and getattr(head, "ndim", 0) >= 1:
            return int(head.shape[0])
        return None

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        return None

    @classmethod
    def get_download_url(cls, _filename: str) -> Optional[str]:
        # LibreSegformer ships no pretrained weights of any size (b0-b5): the
        # upstream nvidia/segformer checkpoints are under a non-permissive,
        # non-commercial license and are never mirrored or auto-downloaded.
        # Train from scratch. See NOTICE in this directory.
        return None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "b0",
        nb_classes: int = 150,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else "semantic"
        if resolved_task != "semantic":
            raise ValueError(f"LibreSegformer supports only task='semantic'; got {task!r}.")
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )
        self.model.eval()
        if self.model_path is not None:
            self._load_weights(str(self.model_path))

    def _init_model(self) -> nn.Module:
        return LibreSegformerNet(size=self.size, num_classes=self.nb_classes)

    def _rebuild_for_new_classes(self, new_nb_classes: int) -> None:
        decode_head = self.model.decode_head
        in_channels = decode_head.classifier.in_channels
        decode_head.classifier = nn.Conv2d(in_channels, new_nb_classes, kernel_size=1)
        self.model.num_classes = new_nb_classes
        self.nb_classes = new_nb_classes
        self.names = {i: f"class_{i}" for i in range(new_nb_classes)}
        self.model.to(self.device)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"encoder": self.model.encoder, "decode_head": self.model.decode_head}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_res = input_size if input_size is not None else self._get_input_size()
        if effective_res % self.semantic_imgsz_divisor:
            raise ValueError(
                f"LibreSegformer semantic imgsz={effective_res} must be divisible "
                f"by {self.semantic_imgsz_divisor} (encoder stride product)."
            )
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), effective_res)
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

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
        logits = output
        if isinstance(logits, dict):
            logits = logits.get("semantic_logits", logits.get("predictions"))
        orig_w, orig_h = original_size
        input_size = kwargs.get("input_size", self._get_input_size())
        input_h, input_w = _input_size_hw(input_size)
        scale_y = logits.shape[-2] / input_h
        scale_x = logits.shape[-1] / input_w
        valid_h = min(logits.shape[-2], max(int(round(orig_h * ratio * scale_y)), 1))
        valid_w = min(logits.shape[-1], max(int(round(orig_w * ratio * scale_x)), 1))
        logits = logits[..., :valid_h, :valid_w]
        logits = F.interpolate(logits.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return {"semantic": logits.argmax(dim=1)[0].cpu()}

    def _strict_loading(self) -> bool:
        return True

    def _validate_loaded_state_dict_for_task(
        self,
        state_dict: dict,
        checkpoint: dict | None = None,
    ) -> None:
        if not self.can_load(state_dict):
            raise RuntimeError("Checkpoint does not look like a SegFormer semantic segmentation model.")

    def _load_weights(self, model_path: str | dict[str, Any]) -> None:
        if isinstance(model_path, str):
            if not Path(model_path).exists():
                from ...utils.download import download_weights

                download_weights(model_path, self.size)
            loaded = load_untrusted_torch_file(
                model_path, map_location="cpu", context="SegFormer semantic weights"
            )
        else:
            loaded = model_path

        if not isinstance(loaded, dict):
            raise TypeError("LibreSegformer checkpoints must be dictionaries")
        metadata_errors = validate_checkpoint_metadata(loaded, strict=False)
        if metadata_errors:
            raise ValueError(
                "LibreSegformer only loads its own checkpoints (train from scratch "
                "via model.train(...)); no upstream checkpoint format is supported."
            )

        ckpt_family = loaded.get("model_family")
        if isinstance(ckpt_family, str) and ckpt_family and ckpt_family != self.FAMILY:
            raise RuntimeError(
                f"Checkpoint was trained with model_family='{ckpt_family}' "
                f"but is being loaded into '{self.FAMILY}'."
            )

        ckpt_task = loaded.get("task")
        if isinstance(ckpt_task, str) and normalize_task(ckpt_task) != "semantic":
            raise RuntimeError(
                f"Checkpoint was trained for task={normalize_task(ckpt_task)!r}, "
                "but LibreSegformer is semantic-only."
            )

        if isinstance(loaded.get("model"), dict):
            state = loaded["model"]
        elif isinstance(loaded.get("state_dict"), dict):
            state = loaded["state_dict"]
        else:
            state = loaded

        ckpt_nc = loaded.get("nc") or self.detect_nb_classes(state)
        if ckpt_nc is not None and int(ckpt_nc) != self.nb_classes:
            self._rebuild_for_new_classes(int(ckpt_nc))

        if not self.can_load(state):
            raise RuntimeError("Checkpoint does not look like a SegFormer semantic segmentation model.")
        self.model.load_state_dict(state, strict=True)

        ckpt_names = loaded.get("names")
        if ckpt_names is not None:
            self.names = self._sanitize_names(ckpt_names, self.nb_classes)
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Training — the point of this port: no pretrained weights exist, so
    # LibreSegformer must be trainable from scratch.
    # ------------------------------------------------------------------

    def train(
        self,
        data: str,
        *,
        epochs: int = 160,
        batch: int = 8,
        imgsz: Optional[int] = None,
        lr0: Optional[float] = None,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "segformer_exp",
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = True,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> Dict:
        """Train LibreSegformer from scratch (no pretrained backbone exists)."""
        from .trainer import SegformerTrainer

        train_kwargs = dict(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz if imgsz is not None else self.input_size,
            size=self.size,
            num_classes=self.nb_classes,
            device=device,
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            **kwargs,
        )
        if lr0 is not None:
            train_kwargs["lr0"] = lr0

        trainer = SegformerTrainer(
            model=self.model,
            wrapper_model=self,
            callbacks=callbacks,
            loggers=loggers,
            **train_kwargs,
        )
        result = trainer.train()
        self._restore_after_training(result)
        return result

    def _restore_after_training(self, result: dict) -> None:
        checkpoint = None
        for key in ("best_checkpoint", "last_checkpoint"):
            path = result.get(key)
            if path and Path(path).exists():
                checkpoint = str(path)
                break
        if checkpoint is not None:
            self.model_path = checkpoint
            self._load_weights(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def export(self, format: str = "onnx", **kwargs) -> str:
        raise NotImplementedError("Export is not implemented for LibreSegformer yet.")


__all__ = ["LibreSegformer", "preprocess_numpy"]
