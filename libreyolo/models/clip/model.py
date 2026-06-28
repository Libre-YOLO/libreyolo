"""LibreCLIP — native CLIP zero-shot (open-vocabulary) classification.

LibreCLIP is a classifier that needs **no training and no fixed label set**::

    from libreyolo import LibreCLIP
    model = LibreCLIP(size="b32")                    # LAION-2B ViT-B/32 (MIT)
    model.set_classes(["a forklift", "an empty aisle", "a spill"])
    r = model.predict("warehouse.jpg")[0]
    print(model.names[r.probs.top1], float(r.probs.top1conf))

It reuses LibreYOLO's existing ``classify`` plumbing — ``Results.probs`` /
``top1`` / ``top5`` and the (CLIP-preprocessing) ``ClassifyValidator`` — and adds
the one open-vocabulary primitive, :meth:`set_classes`, which re-derives the
classifier head from text on the fly (cached, not recomputed per image).

The towers are a native ``torch`` re-implementation (see :mod:`.nn`); the BPE
tokenizer is vendored (see :mod:`.tokenizer`). open_clip is **not** a runtime
dependency. Weights are the MIT-redistributable OpenCLIP LAION-2B checkpoints
(see the family ``NOTICE`` for the LAION data-provenance note).

Zero-shot only: ``train()`` raises. ONNX export bakes the *current* label set
into a fixed ``[B, K]`` classifier graph (see :meth:`export`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...tasks import normalize_task
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import load_trusted_torch_file
from ..base.model import BaseModel
from .labels import (
    DEFAULT_TEMPLATES,
    humanize_labels,
    imagenet1k_classnames,
    openai_imagenet_templates,
)
from .nn import CLIP_CONFIGS, build_clip_model

logger = logging.getLogger(__name__)

# CLIP-specific preprocessing (NOT ImageNet stats) — bicubic + these mean/std.
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


class LibreCLIP(BaseModel):
    """Open-vocabulary zero-shot image classifier (inference only)."""

    FAMILY: ClassVar[str] = "clip"
    FILENAME_PREFIX: ClassVar[str] = "LibreCLIP"
    WEIGHT_EXT: ClassVar[str] = ".pt"

    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        size: cfg.image_size for size, cfg in CLIP_CONFIGS.items()
    }
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("classify",)
    DEFAULT_TASK: ClassVar[str] = "classify"
    TRAIN_CONFIG = None

    # The text->image attention pooling makes multi-scale TTA meaningless and
    # the model resizes to a fixed square; keep predict to a single forward.
    TTA_ENABLED: ClassVar[bool] = False

    validator_class: ClassVar[Optional[type]] = None  # set lazily (see _resolve_validator)

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """CLIP checkpoints carry a learned ``logit_scale`` + ``text_projection``
        alongside the ``visual.conv1`` patch-embed — a signature no other family
        shares."""
        return (
            "logit_scale" in weights_dict
            and "text_projection" in weights_dict
            and "visual.conv1.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        w = weights_dict.get("visual.conv1.weight")
        if w is None:
            return None
        patch = int(w.shape[-1])
        width = int(w.shape[0])
        if patch == 32:
            return "b32"
        if patch == 16:
            return "b16"
        if patch == 14 and width == 1024:
            return "l14"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # Open-vocabulary: there is no fixed head; the class count is set by
        # set_classes() (defaults to ImageNet-1k on construction).
        return None

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        model_path: str | dict | None = None,
        size: str | None = None,
        nb_classes: int | None = None,
        device: str = "auto",
        task: str | None = None,
        templates: Optional[Sequence[str]] = None,
        classes: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else "classify"
        if resolved_task != "classify":
            raise ValueError(f"LibreCLIP only supports task='classify'; got {task!r}.")

        # Resolve the weight source: explicit dict/path, or default per-size
        # checkpoint name for zero-config autodownload.
        if isinstance(model_path, dict):
            weight_source: str | dict = model_path
            if size is None:
                size = self.detect_size(self._extract_state(model_path))
        elif isinstance(model_path, str):
            weight_source = self._resolve_weights_path(model_path)
            if size is None:
                size = self.detect_size_from_filename(model_path)
        else:
            # Zero-config: pick a default size and autodownload its checkpoint.
            size = size or "b32"
            weight_source = self._resolve_weights_path(f"{self.FILENAME_PREFIX}{size}-cls.pt")
        size = size or "b32"

        self._default_templates = list(templates) if templates else list(DEFAULT_TEMPLATES)
        self._text_embeds: Optional[torch.Tensor] = None
        self.tokenizer = None  # built after super().__init__

        # Build the (random) towers via BaseModel, then load real weights.
        super().__init__(
            model_path=None,
            size=size,
            nb_classes=1000,
            device=device,
            task=resolved_task,
            **kwargs,
        )

        self._load_weights(weight_source)
        self.model.eval()

        from .tokenizer import SimpleTokenizer

        self.tokenizer = SimpleTokenizer(context_length=self.model.context_length)

        # Default to ImageNet-1k so predict() works zero-config; or honor the
        # caller's initial class list.
        if classes is not None:
            self.set_classes(list(classes), templates=self._default_templates)
        else:
            self.set_classes(imagenet1k_classnames(), templates=self._default_templates)

    @staticmethod
    def _extract_state(ckpt: dict) -> dict:
        for key in ("model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt

    # =========================================================================
    # Open-vocabulary head — the headline primitive
    # =========================================================================

    @torch.no_grad()
    def _encode_texts(self, texts: List[str], chunk: int = 256) -> torch.Tensor:
        """Run the text tower over many prompts, returning L2-normalized [N, D]."""
        out: List[torch.Tensor] = []
        for start in range(0, len(texts), chunk):
            batch = texts[start : start + chunk]
            tokens = self.tokenizer(batch).to(self.device)
            feats = self.model.encode_text(tokens)
            out.append(F.normalize(feats, dim=-1))
        return torch.cat(out, dim=0)

    def set_classes(
        self,
        labels: Sequence[str],
        templates: Optional[Sequence[str]] = None,
    ) -> "LibreCLIP":
        """Define the (open) class set for zero-shot classification.

        For each label, every prompt template is rendered, encoded by the text
        tower, L2-normalized, averaged across templates, then re-normalized. The
        resulting ``[K, D]`` text-embedding matrix is cached on the model — it is
        the classifier head and is **not** recomputed per image.

        Args:
            labels: Class label strings (e.g. ``["a forklift", "a spill"]``).
            templates: Optional ``{}``-format prompt templates; defaults to the
                templates set at construction (``["a photo of a {}."]``). Pass
                ``LibreCLIP.imagenet_ensemble()`` for the 80-prompt ensemble.
        """
        labels = [str(lbl) for lbl in labels]
        if not labels:
            raise ValueError("set_classes() requires at least one label.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized; weights must load first.")
        templates = list(templates) if templates else list(self._default_templates)

        prompts = [tmpl.format(label) for label in labels for tmpl in templates]
        feats = self._encode_texts(prompts)  # [K*T, D], normalized
        feats = feats.view(len(labels), len(templates), -1).mean(dim=1)  # [K, D]
        feats = F.normalize(feats, dim=-1)

        self._text_embeds = feats.to(self.device)
        self.nb_classes = len(labels)
        self.names = {i: label for i, label in enumerate(labels)}
        return self

    @staticmethod
    def imagenet_ensemble() -> List[str]:
        """The 80-prompt OpenAI ImageNet template ensemble (a few extra points)."""
        return openai_imagenet_templates()

    # =========================================================================
    # BaseModel abstract surface
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return build_clip_model(self.size)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "image_tower": self.model.visual,
            "text_tower": self.model.transformer,
        }

    def _build_transform(self, imgsz: int):
        from ...data.classify_dataset import build_classify_transforms
        from torchvision.transforms import InterpolationMode

        return build_classify_transforms(
            imgsz,
            augment=False,
            mean=CLIP_MEAN,
            std=CLIP_STD,
            interpolation=InterpolationMode.BICUBIC,
            crop_pct=1.0,
        )

    @staticmethod
    def _get_preprocess_numpy():
        from ...data.classify_dataset import build_classify_transforms
        from torchvision.transforms import InterpolationMode
        import numpy as _np

        def _preprocess_numpy(img_rgb_hwc, input_size=224):
            res = input_size if isinstance(input_size, int) else input_size[0]
            transform = build_classify_transforms(
                res, augment=False, mean=CLIP_MEAN, std=CLIP_STD,
                interpolation=InterpolationMode.BICUBIC, crop_pct=1.0,
            )
            pil = Image.fromarray(_np.asarray(img_rgb_hwc).astype("uint8"))
            return transform(pil).numpy(), 1.0

        return _preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        res = input_size if input_size is not None else self.input_size
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        transform = self._build_transform(res)
        return transform(img).unsqueeze(0), img, (orig_w, orig_h), 1.0

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self._text_embeds is None:
            raise RuntimeError("No classes set; call set_classes() first.")
        image_features = self.model.encode_image(input_tensor.to(self.device))
        image_features = F.normalize(image_features, dim=-1)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * image_features @ self._text_embeds.t().to(image_features.dtype)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        logits = output[0] if isinstance(output, (list, tuple)) else output
        probs = torch.softmax(logits.float(), dim=1)[0]
        return {"probs": probs.cpu()}

    # =========================================================================
    # Weights I/O
    # =========================================================================

    def _strict_loading(self) -> bool:
        return True

    def _load_weights(self, model_path: str | dict) -> None:
        if isinstance(model_path, dict):
            loaded = model_path
        else:
            path = Path(model_path)
            if not path.exists():
                from ...utils.download import download_weights

                download_weights(str(path), self.size)
            loaded = load_trusted_torch_file(
                str(model_path), map_location="cpu", context="LibreCLIP weights"
            )

        if not isinstance(loaded, dict):
            raise TypeError("LibreCLIP checkpoints must be dictionaries.")

        ckpt_family = loaded.get("model_family", "")
        if ckpt_family and ckpt_family != self.FAMILY:
            raise RuntimeError(
                f"Checkpoint was trained with model_family='{ckpt_family}' but is "
                f"being loaded into '{self.FAMILY}'."
            )
        ckpt_task = loaded.get("task")
        if isinstance(ckpt_task, str) and normalize_task(ckpt_task) != "classify":
            raise RuntimeError(
                f"Checkpoint task={normalize_task(ckpt_task)!r} is not 'classify'."
            )

        state = self._extract_state(loaded)
        if "logit_scale" not in state or "text_projection" not in state:
            raise RuntimeError(
                "Checkpoint does not look like a LibreCLIP model (missing "
                "'logit_scale'/'text_projection')."
            )
        self.model.load_state_dict(state, strict=self._strict_loading())
        self.model.to(self.device).eval()

    # =========================================================================
    # Validation — reuse ClassifyValidator with CLIP preprocessing + open vocab
    # =========================================================================

    def _resolve_validator(self):
        from ...validation.clip_validator import CLIPClassifyValidator

        return CLIPClassifyValidator

    def val(self, data: str | None = None, **kwargs) -> Dict:
        """Zero-shot top-1/top-5 on an ImageFolder split.

        Reads the train-split folder names, humanizes wnid folders to readable
        labels, calls :meth:`set_classes`, then runs the CLIP-preprocessing
        validator. Zero-shot accuracy depends on the label *wording*.
        """
        from ...data.classify_dataset import get_class_names, resolve_classify_data

        if data is None:
            raise ValueError("LibreCLIP.val() requires data= (an ImageFolder root).")
        root = resolve_classify_data(data)
        folder_names = get_class_names(root, split="train")
        self.set_classes(humanize_labels(folder_names))
        self.validator_class = self._resolve_validator()
        return super().val(data=data, **kwargs)

    # =========================================================================
    # Training is out of scope (zero-shot)
    # =========================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreCLIP is zero-shot / inference-only; there is nothing to train. "
            "Use set_classes([...]) to define classes, then predict()/val(). "
            "CLIP fine-tuning (linear-probe / full) is a separate future feature."
        )

    # =========================================================================
    # Export — frozen-class ONNX
    # =========================================================================

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export a **frozen-class** ONNX classifier for the current labels.

        The current ``set_classes`` text embeddings are baked into a final
        ``Linear`` (weight = ``logit_scale.exp() * text_embeds``), giving a
        standard ``[B, K]`` image-classifier graph (no text tower / tokenizer at
        inference). The ONNX is fixed to the labels set at export time and to a
        fixed input resolution; re-export to change either.
        """
        if format.lower() != "onnx":
            raise NotImplementedError(
                f"LibreCLIP export to {format!r} is not implemented; only 'onnx' "
                "(frozen-class) is supported. Open-vocabulary export (two towers "
                "+ tokenizer) is out of scope for v1."
            )
        if self._text_embeds is None:
            raise RuntimeError("No classes set; call set_classes() before export().")

        from .export import export_frozen_onnx

        return export_frozen_onnx(self, **kwargs)
