"""LibreEoMT semantic and instance segmentation wrapper."""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.ops import batched_nms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TVF

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
    """Encoder-only Mask Transformer for semantic and instance segmentation."""

    FAMILY: ClassVar[str] = "eomt"
    FILENAME_PREFIX: ClassVar[str] = "LibreEoMT"
    WEIGHT_EXT: ClassVar[str] = ".pt"

    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("semantic", "segment")
    DEFAULT_TASK: ClassVar[str] = "semantic"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": 512, "b": 512, "l": 512}
    TASK_INPUT_SIZES: ClassVar[Dict[str, Dict[str, int]]] = {
        "semantic": {"s": 512, "b": 512, "l": 512},
        "segment": {"s": 640, "b": 640, "l": 640},
    }

    semantic_resize_mode: ClassVar[str] = "split"
    semantic_imgsz_divisor: ClassVar[int] = 16

    _EMBED_DIM_TO_SIZE: ClassVar[Dict[int, str]] = {384: "s", 768: "b", 1024: "l"}
    _UPSTREAM_URL: ClassVar[str] = "https://github.com/tue-mps/eomt"
    SUPPORTS_BATCHED_PREDICT: ClassVar[bool] = False

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
    def detect_num_queries(cls, weights_dict: dict) -> Optional[int]:
        state = normalize_eomt_state_dict(weights_dict)
        weight = state.get("query.weight")
        if weight is not None and getattr(weight, "ndim", 0) >= 1:
            return int(weight.shape[0])
        return None

    @classmethod
    def detect_image_size(cls, weights_dict: dict, patch_size: int = 16) -> Optional[int]:
        """Infer image_size from position embedding shape: sqrt(num_positions) * patch_size."""
        state = normalize_eomt_state_dict(weights_dict)
        weight = state.get("embeddings.position_embeddings.weight")
        if weight is not None and getattr(weight, "ndim", 0) >= 1:
            num_positions = int(weight.shape[0])
            side = int(round(num_positions ** 0.5))
            if side * side == num_positions:
                return side * patch_size
        return None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        if not cls.can_load(state_dict):
            return None
        nc = cls.detect_nb_classes(state_dict)
        # nc==80 is the COCO instance checkpoint convention; all other class
        # counts (ADE20K 150, panoptic 133, custom) default to semantic.
        if nc is not None and nc == 80:
            return "segment"
        return "semantic"

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
        num_queries: int = 100,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        if size is None:
            size = "l"
        self.num_queries = int(num_queries)

        if isinstance(model_path, dict) and not model_path:
            weight_source = None
        elif isinstance(model_path, str):
            weight_source = self._resolve_weights_path(model_path)
        else:
            weight_source = model_path

        # BaseModel._resolve_task validates task against SUPPORTED_TASKS.
        super().__init__(
            model_path=None,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task,
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
            num_queries=getattr(self, "num_queries", 100),
        )

    def _strict_loading(self) -> bool:
        return True

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        self.nb_classes = int(new_nb_classes)
        self.names = {i: f"class_{i}" for i in range(self.nb_classes)}
        self.model = self._init_model()
        self.model.to(self.device)

    def _rebuild_for_new_queries(self, new_num_queries: int):
        self.num_queries = int(new_num_queries)
        self.model = self._init_model()
        self.model.to(self.device)

    def _rebuild_for_new_image_size(self, new_image_size: int):
        self.input_size = int(new_image_size)
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

    @staticmethod
    def _shortest_edge_size(height: int, width: int, size: int) -> tuple[int, int]:
        """Match HF EoMT shortest-edge resize integer semantics."""
        if (height <= width and height == size) or (width <= height and width == size):
            return height, width
        if width < height:
            return int(size * height / width), size
        return size, int(size * width / height)

    @staticmethod
    def _split_resized_tensor(
        tensor: torch.Tensor,
        patch_size: int,
    ) -> tuple[list[torch.Tensor], list[tuple[int, int, int]]]:
        """Split a shortest-edge-resized image into HF-style EoMT patches."""
        _, height, width = tensor.shape
        longer_side = max(height, width)
        num_patches = int(math.ceil(longer_side / patch_size))
        total_overlap = num_patches * patch_size - longer_side
        overlap_per_patch = total_overlap / (num_patches - 1) if num_patches > 1 else 0

        patches: list[torch.Tensor] = []
        offsets: list[tuple[int, int, int]] = []
        for i in range(num_patches):
            start = int(i * (patch_size - overlap_per_patch))
            end = start + patch_size
            patch = tensor[:, start:end, :] if height > width else tensor[:, :, start:end]
            if patch.shape[-2:] != (patch_size, patch_size):
                raise RuntimeError(
                    "LibreEoMT split preprocessing produced a non-square patch "
                    f"{patch.shape[-2:]} for resized image {(height, width)}."
                )
            patches.append(patch)
            offsets.append((0, start, end))
        return patches, offsets

    def _preprocess_pil_split(
        self,
        img: Image.Image,
        input_size: int,
    ) -> tuple[torch.Tensor, tuple[int, int], list[tuple[int, int, int]]]:
        orig_w, orig_h = img.size
        resized_h, resized_w = self._shortest_edge_size(orig_h, orig_w, input_size)
        resized = TVF.resize(
            TVF.pil_to_tensor(img).unsqueeze(0),
            [resized_h, resized_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )[0].float()
        resized.div_(255.0)
        patches, offsets = self._split_resized_tensor(resized, input_size)
        tensors = [patch.contiguous() for patch in patches]
        return torch.stack(tensors, dim=0), (resized_h, resized_w), offsets

    @staticmethod
    def _stitch_patch_logits(
        logits: torch.Tensor,
        patch_offsets: list[tuple[int, int, int]],
        *,
        resized_shape: tuple[int, int],
        original_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Merge per-patch semantic logits using HF EoMT overlap averaging."""
        resized_h, resized_w = resized_shape
        orig_h, orig_w = original_shape
        num_classes = int(logits.shape[1])
        merged = torch.zeros(
            (num_classes, resized_h, resized_w),
            dtype=logits.dtype,
            device=logits.device,
        )
        counts = torch.zeros(
            (1, resized_h, resized_w),
            dtype=logits.dtype,
            device=logits.device,
        )

        vertical = resized_h > resized_w
        for patch_idx, (_, start, end) in enumerate(patch_offsets):
            if vertical:
                merged[:, start:end, :] += logits[patch_idx]
                counts[:, start:end, :] += 1
            else:
                merged[:, :, start:end] += logits[patch_idx]
                counts[:, :, start:end] += 1

        averaged = merged / counts.clamp(min=1)
        return F.interpolate(
            averaged.unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )[0]

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_res = input_size if input_size is not None else self.input_size
        if effective_res % self.semantic_imgsz_divisor:
            raise ValueError(
                f"LibreEoMT imgsz={effective_res} must be divisible "
                f"by {self.semantic_imgsz_divisor} (EoMT patch grid)."
            )
        if effective_res != self.input_size:
            raise ValueError(
                f"LibreEoMT requires imgsz={self.input_size}; got imgsz="
                f"{effective_res}. The EoMT checkpoint uses fixed "
                "position embeddings."
            )
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        img_tensor, resized_shape, patch_offsets = self._preprocess_pil_split(
            img,
            effective_res,
        )
        self._last_eomt_resized_shape = resized_shape
        self._last_eomt_patch_offsets = patch_offsets
        return img_tensor, img, (orig_w, orig_h), 1.0

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    @staticmethod
    def _masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
        """Convert (K, H, W) masks to (K, 4) xyxy boxes in absolute pixel coords."""
        k = masks.shape[0]
        if k == 0:
            return torch.zeros((0, 4), dtype=torch.float32, device=masks.device)
        boxes = []
        for mask in masks:
            nonzero = mask.nonzero()
            if len(nonzero) == 0:
                boxes.append(torch.zeros(4, dtype=torch.float32, device=masks.device))
            else:
                y1, x1 = nonzero.min(0).values.float()
                y2, x2 = nonzero.max(0).values.float()
                boxes.append(torch.stack([x1, y1, x2, y2]))
        return torch.stack(boxes)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        if self.task == "segment":
            return self._postprocess_segment(
                output, conf_thres, iou_thres, original_size, max_det
            )
        return self._postprocess_semantic(output, original_size)

    def _postprocess_semantic(self, output: Any, original_size: Tuple[int, int]) -> Dict:
        logits = output
        if isinstance(logits, dict):
            logits = logits.get("semantic_logits", logits.get("logits"))
        if logits is None:
            raise ValueError("LibreEoMT forward output did not include semantic logits.")
        orig_w, orig_h = original_size
        patch_offsets = getattr(self, "_last_eomt_patch_offsets", None)
        resized_shape = getattr(self, "_last_eomt_resized_shape", None)
        if (
            patch_offsets
            and resized_shape
            and len(patch_offsets) == int(logits.shape[0])
        ):
            logits_hw = self._stitch_patch_logits(
                logits.float(),
                patch_offsets,
                resized_shape=resized_shape,
                original_shape=(orig_h, orig_w),
            )
            return {"semantic": logits_hw.argmax(dim=0).cpu()}

        logits = F.interpolate(
            logits.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        return {"semantic": logits.argmax(dim=1)[0].cpu()}

    def _postprocess_segment(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
    ) -> Dict:
        """Decode instance segmentation from per-query class and mask logits."""
        if not isinstance(output, dict):
            raise ValueError("LibreEoMT segment forward must return a dict.")
        class_logits = output.get("class_queries_logits")  # (B, Q, C+1)
        mask_logits = output.get("masks_queries_logits")   # (B, Q, H, W)
        if class_logits is None or mask_logits is None:
            raise ValueError(
                "LibreEoMT segment forward did not include class_queries_logits "
                "and masks_queries_logits."
            )
        orig_w, orig_h = original_size
        patch_offsets = getattr(self, "_last_eomt_patch_offsets", None)
        resized_shape = getattr(self, "_last_eomt_resized_shape", None)
        num_patches = int(class_logits.shape[0])
        resized_h, resized_w = resized_shape if resized_shape else (orig_h, orig_w)
        has_patches = (
            patch_offsets is not None
            and resized_shape is not None
            and len(patch_offsets) == num_patches
        )

        all_scores: list[torch.Tensor] = []
        all_classes: list[torch.Tensor] = []
        all_boxes: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []

        for patch_idx in range(num_patches):
            cls_logit = class_logits[patch_idx]  # (Q, C+1)
            msk_logit = mask_logits[patch_idx]   # (Q, H, W)

            # DETR-style decoding: softmax, exclude null/background class
            scores_per_query = cls_logit.softmax(-1)[..., :-1]  # (Q, C)
            scores, labels = scores_per_query.max(-1)           # (Q,), (Q,)

            keep = scores > conf_thres
            if not keep.any():
                continue

            scores = scores[keep]
            labels = labels[keep]
            binary_masks = (msk_logit[keep].sigmoid() > 0.5).float()  # (K, H, W)

            # Place patch masks into full resized-image canvas
            if has_patches:
                _, start, end = patch_offsets[patch_idx]
                vertical = resized_h > resized_w
                full = torch.zeros(
                    (len(binary_masks), resized_h, resized_w),
                    dtype=binary_masks.dtype,
                    device=binary_masks.device,
                )
                if vertical:
                    full[:, start:end, :] = binary_masks
                else:
                    full[:, :, start:end] = binary_masks
                binary_masks = full

            # Upsample masks to original image size
            masks_orig = F.interpolate(
                binary_masks.unsqueeze(0),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )[0]
            masks_orig = (masks_orig > 0.5).float()

            boxes = self._masks_to_boxes(masks_orig)

            all_scores.append(scores.cpu())
            all_classes.append(labels.cpu())
            all_boxes.append(boxes.cpu())
            all_masks.append(masks_orig.cpu())

        if not all_scores:
            return {
                "boxes": [],
                "scores": [],
                "classes": [],
                "num_detections": 0,
                "masks": torch.zeros((0, orig_h, orig_w), dtype=torch.float32),
            }

        boxes_t = torch.cat(all_boxes, dim=0)    # (N, 4)
        scores_t = torch.cat(all_scores, dim=0)  # (N,)
        labels_t = torch.cat(all_classes, dim=0) # (N,)
        masks_t = torch.cat(all_masks, dim=0)    # (N, H, W)

        if num_patches > 1:
            # Multi-patch: NMS merges predictions of the same object detected
            # in overlapping patches (cross-patch duplicates are expected).
            keep_idx = batched_nms(boxes_t.float(), scores_t.float(), labels_t, iou_thres)
            if len(keep_idx) > max_det:
                keep_idx = keep_idx[:max_det]
        else:
            # Single-patch: EoMT is DETR-style (queries are uniquely assigned by
            # bipartite matching); applying NMS can suppress valid overlapping
            # detections. Use top-k by confidence score instead.
            keep_idx = scores_t.argsort(descending=True)[:max_det]

        return {
            "boxes": boxes_t[keep_idx].tolist(),
            "scores": scores_t[keep_idx].tolist(),
            "classes": labels_t[keep_idx].tolist(),
            "num_detections": int(len(keep_idx)),
            "masks": masks_t[keep_idx],
        }

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
        if isinstance(ckpt_task, str):
            normalized_ckpt_task = normalize_task(ckpt_task)
            if normalized_ckpt_task != self.task:
                raise RuntimeError(
                    f"Checkpoint task={normalized_ckpt_task!r} does not match "
                    f"model task={self.task!r}. Pass task={normalized_ckpt_task!r} "
                    "when constructing LibreEoMT, or use the correct checkpoint."
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

        ckpt_num_queries = self.detect_num_queries(state)
        if ckpt_num_queries is not None and ckpt_num_queries != self.num_queries:
            self._rebuild_for_new_queries(ckpt_num_queries)

        # State-dict detection is authoritative: position embeddings encode the
        # native resolution and cannot be wrong. Metadata imgsz is a hint only.
        ckpt_imgsz = self.detect_image_size(state)
        if ckpt_imgsz is None:
            ckpt_imgsz = loaded.get("imgsz") if isinstance(loaded, dict) else None
        if ckpt_imgsz is not None and int(ckpt_imgsz) != self.input_size:
            self._rebuild_for_new_image_size(int(ckpt_imgsz))

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

    def val(
        self,
        data: str | None = None,
        batch: int = 1,
        imgsz: int | None = None,
        conf: float = 0.001,
        iou: float = 0.6,
        workers: int = 0,
        allow_download_scripts: bool = False,
        device: str | None = None,
        split: str = "val",
        augment: bool = False,
        save_json: bool = False,
        verbose: bool = True,
        *args,
        plots: bool | None = None,
        save_plots: bool = False,
        save_dir: str | None = None,
        half: bool = False,
        **kwargs,
    ):
        if self.task == "segment":
            return super().val(
                data=data,
                batch=batch,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                workers=workers,
                allow_download_scripts=allow_download_scripts,
                device=device,
                split=split,
                augment=augment,
                save_json=save_json,
                verbose=verbose,
                plots=plots,
                **kwargs,
            )

        conf_thres = float(conf)
        iou_thres = float(iou)
        if not 0 <= conf_thres < 1:
            raise ValueError(f"conf must be in [0, 1), got {conf_thres}.")
        if not 0 < iou_thres < 1:
            raise ValueError(f"iou must be in (0, 1), got {iou_thres}.")
        if args:
            raise TypeError("LibreEoMT.val() does not accept extra positional arguments.")
        data_dir = kwargs.pop("data_dir", None)
        max_det = kwargs.pop("max_det", 300)
        iou_thresholds = kwargs.pop("iou_thresholds", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                f"LibreEoMT.val() got unexpected keyword argument(s): {names}."
            )
        if data_dir is not None:
            raise ValueError(
                "LibreEoMT validation requires data= (a semantic dataset YAML); "
                "data_dir is not supported."
            )
        if max_det != 300:
            logger.warning("LibreEoMT semantic validation ignores max_det=%s.", max_det)
        if iou_thresholds is not None:
            logger.warning("LibreEoMT semantic validation ignores iou_thresholds.")
        if int(batch) != 1:
            logger.warning(
                "LibreEoMT validation processes split-image inference one image "
                "at a time; batch=%s is ignored.",
                batch,
            )
        if int(workers) != 0:
            logger.warning(
                "LibreEoMT validation does not use dataloader workers; workers=%s "
                "is ignored.",
                workers,
            )
        if save_json:
            raise ValueError(
                "LibreEoMT semantic validation does not support save_json output."
            )
        if plots is not None and not save_plots:
            save_plots = bool(plots)
        if save_plots:
            logger.warning("LibreEoMT validation does not generate plots yet.")
        if data is None:
            raise ValueError("LibreEoMT validation requires data= (a dataset YAML).")
        if augment:
            raise ValueError(
                "Augmented validation does not support semantic segmentation yet. "
                "Use augment=False for semantic models."
            )
        effective_imgsz = self.input_size if imgsz is None else int(imgsz)
        if effective_imgsz != self.input_size:
            raise ValueError(
                f"LibreEoMT validation requires imgsz={self.input_size}; got "
                f"imgsz={effective_imgsz}. The EoMT checkpoint uses fixed "
                "position embeddings."
            )

        from ...data.semantic_dataset import (
            _apply_label_mapping,
            _load_mask_image,
            img2mask_paths,
            resolve_semantic_data,
        )
        from ...data.utils import get_img_files

        if device is not None and str(device).lower() != "auto":
            device_str = f"cuda:{device}" if str(device).isdigit() else str(device)
            self.device = torch.device(device_str)
            self.model.to(self.device)

        data_config = resolve_semantic_data(
            data,
            allow_scripts=allow_download_scripts,
        )
        split_value = data_config.get(split)
        if not split_value:
            raise ValueError(f"Semantic dataset config has no '{split}' split.")
        img_files = data_config.get(f"{split}_img_files") or get_img_files(split_value)
        if not img_files:
            raise FileNotFoundError(
                f"No images found for semantic split '{split}' at {split_value}."
            )
        masks_dir = data_config.get("masks_dir")
        if not masks_dir:
            raise ValueError("LibreEoMT validation requires dense PNG masks.")
        mask_files = img2mask_paths(img_files, str(masks_dir))
        missing = [str(path) for path in mask_files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} semantic mask file(s) missing for split "
                f"{split!r}; first missing: {missing[0]}"
            )

        nc = int(data_config.get("nc") or len(data_config.get("names") or {}))
        if int(self.nb_classes) != nc:
            raise ValueError(
                f"Semantic dataset has {nc} classes but LibreEoMT predicts "
                f"{self.nb_classes}."
            )
        names = data_config.get("names") or {}
        if isinstance(names, list):
            names = {i: name for i, name in enumerate(names)}
        names = {int(k): str(v) for k, v in names.items()}

        ignore_index = int(data_config.get("ignore_index", 255))
        raw_mapping = data_config.get("label_mapping") or None
        label_mapping = (
            {int(k): int(v) for k, v in raw_mapping.items()} if raw_mapping else None
        )

        confusion = torch.zeros((nc, nc), dtype=torch.int64)
        self.model.eval()
        start_time = time.time()
        preprocess_time = 0.0
        inference_time = 0.0
        postprocess_time = 0.0
        iterator = tqdm(
            list(zip(img_files, mask_files)),
            desc="Validating",
            total=len(img_files),
            disable=not verbose or not sys.stderr.isatty(),
            file=sys.stderr,
        )

        with torch.no_grad():
            for img_path, mask_path in iterator:
                with Image.open(img_path) as img_pil:
                    orig_shape = (img_pil.height, img_pil.width)

                target_np = _load_mask_image(mask_path)
                if target_np.shape != orig_shape:
                    raise ValueError(
                        f"Semantic mask {mask_path} shape {target_np.shape} "
                        f"does not match image shape {orig_shape}."
                    )
                if label_mapping:
                    target_np = _apply_label_mapping(
                        target_np,
                        label_mapping,
                        ignore_index,
                    )
                invalid = (target_np != ignore_index) & (
                    (target_np < 0) | (target_np >= nc)
                )
                if bool(invalid.any()):
                    bad = sorted(np.unique(target_np[invalid]).tolist())[:5]
                    raise ValueError(
                        f"Semantic mask {mask_path} contains class IDs {bad} "
                        f"outside 0..{nc - 1} (ignore={ignore_index}). "
                        "Use label_mapping to remap source IDs."
                    )

                t1 = time.time()
                tensor, _, original_size, _ = self._preprocess(
                    img_path,
                    color_format="rgb",
                    input_size=effective_imgsz,
                )
                preprocess_time += time.time() - t1

                t2 = time.time()
                if half and self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        output = self._forward(tensor.to(self.device))
                else:
                    output = self._forward(tensor.to(self.device))
                inference_time += time.time() - t2

                t3 = time.time()
                pred = self._postprocess(
                    output,
                    conf_thres=0.0,
                    iou_thres=0.0,
                    original_size=original_size,
                )["semantic"].long().view(-1)
                postprocess_time += time.time() - t3

                target = torch.from_numpy(np.ascontiguousarray(target_np)).long().view(-1)
                valid = target != ignore_index
                if not bool(valid.any()):
                    continue
                target_valid = target[valid]
                pred_valid = pred[valid].clamp_(0, nc - 1)
                index = target_valid * nc + pred_valid
                counts = torch.bincount(index, minlength=nc**2)
                confusion += counts.reshape(nc, nc)

        total = confusion.sum()
        true_positive = confusion.diag().double()
        union = confusion.sum(dim=0).double() + confusion.sum(dim=1).double() - true_positive
        per_class_iou = torch.full((nc,), float("nan"), dtype=torch.float64)
        present = union > 0
        per_class_iou[present] = true_positive[present] / union[present]
        observed = ~torch.isnan(per_class_iou)
        miou = float(per_class_iou[observed].mean()) if bool(observed.any()) else 0.0
        accuracy = float(confusion.diag().sum() / total) if total > 0 else 0.0

        if verbose:
            logger.info("=" * 50)
            logger.info("LibreEoMT Semantic Segmentation Validation Results")
            logger.info("=" * 50)
            for class_id, value in enumerate(per_class_iou):
                if torch.isnan(value):
                    continue
                logger.info("  IoU %-20s %.4f", names.get(class_id, str(class_id)), float(value))
            logger.info("  mIoU:           %.4f", miou)
            logger.info("  pixel accuracy: %.4f", accuracy)
            logger.info("=" * 50)

        elapsed = max(time.time() - start_time, 0.0)
        seen = len(img_files)
        metrics = {
            "metrics/mIoU": miou,
            "metrics/pixel_accuracy": accuracy,
            "fitness": miou,
            "speed/preprocess_ms": preprocess_time / seen * 1000,
            "speed/inference_ms": inference_time / seen * 1000,
            "speed/postprocess_ms": postprocess_time / seen * 1000,
            "speed/total_ms": elapsed / seen * 1000,
            "speed/total_s": elapsed,
            "speed/images_seen": seen,
        }
        if save_dir is not None:
            import yaml

            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            config = {
                "data": data,
                "split": split,
                "batch_size": int(batch),
                "imgsz": effective_imgsz,
                "conf_thres": conf_thres,
                "iou_thres": iou_thres,
                "num_workers": int(workers),
                "allow_download_scripts": bool(allow_download_scripts),
                "device": str(self.device),
                "augment": bool(augment),
                "save_json": bool(save_json),
                "save_plots": bool(save_plots),
                "half": bool(half),
            }
            with open(save_path / "config.yaml", "w") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
        return metrics


__all__ = ["LibreEoMT"]
