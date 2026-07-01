"""LibreEoMT semantic segmentation wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ...tasks import normalize_task
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import load_untrusted_torch_file
from ..base.model import BaseModel
from .nn import LibreEoMTNet, normalize_eomt_state_dict

logger = logging.getLogger(__name__)


def _extract_state(loaded: dict[str, Any]) -> dict[str, Any]:
    for key in ("model", "state_dict"):
        if key in loaded and isinstance(loaded[key], dict):
            return loaded[key]
    return loaded


def _eomt_keys(weights_dict: dict[str, Any]) -> set[str]:
    return set(normalize_eomt_state_dict(weights_dict))


class LibreEoMT(BaseModel):
    """Encoder-only Mask Transformer for semantic segmentation."""

    FAMILY: ClassVar[str] = "eomt"
    FILENAME_PREFIX: ClassVar[str] = "LibreEoMT"
    WEIGHT_EXT: ClassVar[str] = ".pt"

    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("semantic",)
    DEFAULT_TASK: ClassVar[str] = "semantic"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"l": 512}

    semantic_resize_mode: ClassVar[str] = "stretch"
    semantic_imgsz_divisor: ClassVar[int] = 16

    _EMBED_DIM_TO_SIZE: ClassVar[Dict[int, str]] = {1024: "l"}
    _UPSTREAM_URL: ClassVar[str] = "https://github.com/tue-mps/eomt"

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys = _eomt_keys(weights_dict)
        return {
            "query.weight",
            "mask_head.fc1.weight",
            "mask_head.fc2.weight",
            "mask_head.fc3.weight",
            "class_predictor.weight",
            "embeddings.patch_embeddings.projection.weight",
        }.issubset(keys)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        state = normalize_eomt_state_dict(weights_dict)
        for key in ("query.weight", "class_predictor.weight"):
            tensor = state.get(key)
            if tensor is not None and getattr(tensor, "ndim", 0) >= 2:
                return cls._EMBED_DIM_TO_SIZE.get(int(tensor.shape[-1]))
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        state = normalize_eomt_state_dict(weights_dict)
        weight = state.get("class_predictor.weight")
        if weight is not None and getattr(weight, "ndim", 0) >= 1:
            return max(1, int(weight.shape[0]) - 1)
        return None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "semantic" if cls.can_load(state_dict) else None

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        # Raw HF EoMT checkpoints must go through weights/convert_eomt_weights.py
        # so DINOv2-only provenance and ADE20K metadata are enforced.
        return None

    def __init__(
        self,
        model_path=None,
        size: str = "l",
        nb_classes: int = 150,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else "semantic"
        if resolved_task != "semantic":
            raise ValueError(f"LibreEoMT supports only task='semantic'; got {task!r}.")
        if size is None:
            size = "l"

        if isinstance(model_path, dict) and not model_path:
            weight_source = None
        elif isinstance(model_path, str):
            weight_source = self._resolve_weights_path(model_path)
        else:
            weight_source = model_path

        super().__init__(
            model_path=None,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )

        if weight_source is not None:
            self._load_weights(weight_source)
            # BaseModel.__init__ received model_path=None (EoMT loads its own
            # weights above), so it left self.model_path unset. Restore the
            # resolved path so direct ``LibreEoMT("...")`` construction matches
            # the factory path, which sets model_path post-construction.
            if isinstance(weight_source, (str, Path)):
                self.model_path = str(weight_source)
        self.model.eval()

    def _init_model(self) -> nn.Module:
        return LibreEoMTNet(
            config=self.size,
            nb_classes=self.nb_classes,
            image_size=self.input_size,
        )

    def _strict_loading(self) -> bool:
        return True

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        self.nb_classes = int(new_nb_classes)
        self.names = {i: f"class_{i}" for i in range(self.nb_classes)}
        self.model = self._init_model()
        self.model.to(self.device)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        core = getattr(self.model, "eomt", self.model)
        layers: Dict[str, nn.Module] = {}
        for name in ("embeddings", "layers", "mask_head", "class_predictor"):
            module = getattr(core, name, None)
            if module is not None:
                layers[name] = module
        return layers

    @staticmethod
    def _get_preprocess_numpy():
        import cv2
        import numpy as _np

        def _preprocess_numpy(img_rgb_hwc, input_size=512):
            h = input_size if isinstance(input_size, int) else input_size[0]
            w = input_size if isinstance(input_size, int) else input_size[1]
            resized = cv2.resize(img_rgb_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
            arr = _np.ascontiguousarray(resized, dtype=_np.float32) / 255.0
            return arr.transpose(2, 0, 1), 1.0

        return _preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_res = input_size if input_size is not None else self.input_size
        if effective_res % self.semantic_imgsz_divisor:
            raise ValueError(
                f"LibreEoMT semantic imgsz={effective_res} must be divisible "
                f"by {self.semantic_imgsz_divisor} (EoMT patch grid)."
            )
        if effective_res != self.input_size:
            raise ValueError(
                f"LibreEoMT requires imgsz={self.input_size}; got imgsz="
                f"{effective_res}. The HF EoMT-L checkpoint uses fixed "
                "position embeddings."
            )
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        resized = img.resize((effective_res, effective_res), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return img_tensor, img, (orig_w, orig_h), 1.0

    def _forward(self, input_tensor: torch.Tensor) -> Any:
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
        logits = output
        if isinstance(logits, dict):
            logits = logits.get("semantic_logits", logits.get("logits"))
        if logits is None:
            raise ValueError("LibreEoMT forward output did not include semantic logits.")
        orig_w, orig_h = original_size
        logits = torch.nn.functional.interpolate(
            logits.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        return {"semantic": logits.argmax(dim=1)[0].cpu()}

    def _load_weights(self, model_path: str | dict[str, Any]) -> None:
        if isinstance(model_path, (str, Path)):
            path = Path(model_path)
            if not path.exists():
                from ...utils.download import download_weights

                download_weights(str(model_path), self.size)
                path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model weights not found at {model_path}")
            loaded = load_untrusted_torch_file(
                path,
                map_location="cpu",
                context="EoMT weights",
            )
        else:
            loaded = model_path

        if not isinstance(loaded, dict):
            raise TypeError("LibreEoMT checkpoints must be dictionaries.")

        has_libreyolo_metadata = isinstance(loaded.get("model"), dict) and all(
            key in loaded
            for key in (
                "schema_version",
                "libreyolo_version",
                "model_family",
                "size",
                "task",
                "nc",
                "names",
                "imgsz",
            )
        )
        if not has_libreyolo_metadata:
            raise RuntimeError(
                "Raw EoMT state dicts are not loaded directly. Convert the "
                "approved DINOv2 ADE20K checkpoint with "
                "weights/convert_eomt_weights.py so LibreYOLO metadata and "
                "DINOv2-only provenance checks are applied."
            )

        ckpt_family = loaded.get("model_family", "")
        if ckpt_family and ckpt_family != self.FAMILY:
            raise RuntimeError(
                f"Checkpoint was trained with model_family='{ckpt_family}' "
                f"but is being loaded into '{self.FAMILY}'."
            )

        ckpt_task = loaded.get("task")
        if isinstance(ckpt_task, str) and normalize_task(ckpt_task) != "semantic":
            raise RuntimeError(
                f"Checkpoint was trained for task={normalize_task(ckpt_task)!r}, "
                "but is being loaded into a LibreEoMT semantic model."
            )

        state = _extract_state(loaded)
        state = normalize_eomt_state_dict(state)
        if not self.can_load(state):
            raise RuntimeError(
                "Checkpoint does not look like a LibreEoMT model "
                "(missing EoMT query, mask head, class head, or patch embedding keys)."
            )

        ckpt_nc = loaded.get("nc")
        if ckpt_nc is None:
            names = loaded.get("names")
            ckpt_nc = len(names) if names else None
        if ckpt_nc is None:
            ckpt_nc = self.detect_nb_classes(state)
        if ckpt_nc is not None and int(ckpt_nc) != self.nb_classes:
            self._rebuild_for_new_classes(int(ckpt_nc))

        result = self.model.load_state_dict(state, strict=self._strict_loading())
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if missing:
            logger.debug("LibreEoMT missing checkpoint keys: %s", missing[:8])
        if unexpected:
            logger.debug("LibreEoMT unexpected checkpoint keys: %s", unexpected[:8])

        ckpt_names = loaded.get("names")
        if ckpt_names is not None:
            self.names = self._sanitize_names(ckpt_names, self.nb_classes)
        self.model.to(self.device)

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training LibreEoMT is out of scope for LibreYOLO v1. "
            f"Fine-tune upstream at {self._UPSTREAM_URL} and convert the result "
            "with weights/convert_eomt_weights.py."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        raise NotImplementedError(
            "Export is not implemented for LibreEoMT yet. Semantic export needs "
            "a dense-logits runtime contract before this family can be exported."
        )

    def val(self, *args, imgsz: int | None = None, **kwargs):
        effective_imgsz = self.input_size if imgsz is None else int(imgsz)
        if effective_imgsz != self.input_size:
            raise ValueError(
                f"LibreEoMT validation requires imgsz={self.input_size}; got "
                f"imgsz={effective_imgsz}. The HF EoMT-L checkpoint uses fixed "
                "position embeddings."
            )
        return super().val(*args, imgsz=effective_imgsz, **kwargs)


__all__ = ["LibreEoMT"]
