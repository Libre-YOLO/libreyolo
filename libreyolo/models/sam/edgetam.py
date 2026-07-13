"""EdgeTAM on-device promptable segmenter for the LibreSAM tier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from .base import _INSTALL_HINT, LibreSAMModel


def _load_edgetam_video_config(config_path: str | Path):
    """Build EdgeTAM's fully embedded config without a default Hub lookup.

    EdgeTAM's complete ``TimmWrapperConfig`` is embedded in ``config.json``.
    Current Transformers releases nevertheless instantiate a *default*
    ``EdgeTamVisionConfig`` while validating/rendering the supplied config;
    that default performs an unrelated Hub lookup for ``timm/repvit_m1``.

    Constructing each nested config first avoids the dict-to-config assignment
    that triggers that rendering path. The serialization hints are instance
    attributes, so concurrent model loads never mutate Transformers globals.
    """
    from transformers import EdgeTamVideoConfig
    from transformers.models.edgetam.configuration_edgetam import (
        EdgeTamVisionConfig,
    )
    from transformers.models.timm_wrapper.configuration_timm_wrapper import (
        TimmWrapperConfig,
    )

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config_data = json.load(handle)
    if not isinstance(config_data, dict):
        raise ValueError("EdgeTAM config.json must contain a JSON object")

    vision_data = config_data.pop("vision_config", None)
    if not isinstance(vision_data, dict):
        raise ValueError("EdgeTAM config.json is missing an embedded vision_config")
    backbone_data = vision_data.pop("backbone_config", None)
    if not isinstance(backbone_data, dict):
        raise ValueError("EdgeTAM config.json is missing an embedded backbone_config")

    backbone_config = TimmWrapperConfig(**backbone_data)
    vision_config = EdgeTamVisionConfig(backbone_config=backbone_config, **vision_data)
    object.__setattr__(vision_config, "has_no_defaults_at_init", True)
    config = EdgeTamVideoConfig(vision_config=vision_config, **config_data)
    object.__setattr__(config, "has_no_defaults_at_init", True)
    return config


def _clear_edgetam_config_serialization_hints(config) -> None:
    """Remove the temporary instance hints before config serialization."""
    config.__dict__.pop("has_no_defaults_at_init", None)
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        vision_config.__dict__.pop("has_no_defaults_at_init", None)


class LibreEdgeTAM(LibreSAMModel):
    """EdgeTAM promptable segmenter, with image inference only in v1."""

    FAMILY = "edgetam"
    FILENAME_PREFIX = "LibreEdgeTAM"
    HF_REPOS: ClassVar[Dict[str, str]] = {
        "edge": "LibreYOLO/LibreEdgeTAM",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {size: 1024 for size in HF_REPOS}
    SNAPSHOT_IGNORE_PATTERNS: ClassVar[tuple[str, ...]] = (
        *LibreSAMModel.SNAPSHOT_IGNORE_PATTERNS,
        "*.pt",
    )

    def __init__(self, size: str = "edge", **kwargs):
        super().__init__(size=size, **kwargs)

    def _load_pretrained(self, snapshot_dir: str) -> tuple:
        try:
            from transformers import EdgeTamModel, Sam2Processor
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        # Load the already-local video config explicitly. EdgeTamModel consumes
        # its image-inference subset, while avoiding Transformers' implicit
        # config construction and its hidden network/cache dependency.
        config = _load_edgetam_video_config(Path(snapshot_dir) / "config.json")
        model = None
        try:
            model = EdgeTamModel.from_pretrained(
                snapshot_dir,
                config=config,
                dtype=self._resolve_dtype(),
                local_files_only=True,
            )
        finally:
            _clear_edgetam_config_serialization_hints(config)
            if model is not None:
                _clear_edgetam_config_serialization_hints(model.config)
        if model is None:  # Defensive: Transformers always returns a model or raises.
            raise RuntimeError("Transformers returned no EdgeTAM model")
        processor = Sam2Processor.from_pretrained(snapshot_dir, local_files_only=True)
        return model, processor

    def _encode(self, img, *, points=None, labels=None, boxes=None) -> Dict[str, Any]:
        """Use EdgeTAM's exact image transform and processor prompt geometry.

        ``Sam2Processor`` supplies original-size metadata and point labels. Its
        image and coordinate arithmetic differ slightly from the official
        EdgeTAM transform, however, so replace the pixels and prompt coordinates
        with the upstream square-resize and divide-then-multiply operations.

        Transform provenance: facebookresearch/EdgeTAM, commit
        7711e012a30a2402c4eaab637bdb00a521302c91, Apache-2.0,
        ``sam2/utils/transforms.py``.
        """
        enc = super()._encode(img, points=points, labels=labels, boxes=boxes)
        size = self.INPUT_SIZES[self.size]
        pixels = vision_functional.to_tensor(img)
        pixels = vision_functional.resize(
            pixels,
            [size, size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        pixels = vision_functional.normalize(
            pixels,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        enc["pixel_values"] = pixels.unsqueeze(0)

        # Preserve upstream float32 operation order. Sam2Processor computes an
        # equivalent ratio first, which can shift 1024-frame coordinates by a
        # few ULPs and needlessly breaks exact preprocessing parity.
        width, height = img.size
        if points is not None:
            point_tensor = torch.as_tensor(points, dtype=torch.float32)
            point_tensor[..., 0] /= width
            point_tensor[..., 1] /= height
            enc["input_points"] = (point_tensor * size).unsqueeze(0)
        if boxes is not None:
            box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 2, 2)
            box_tensor[..., 0] /= width
            box_tensor[..., 1] /= height
            enc["input_boxes"] = (box_tensor * size).reshape(-1, 4).unsqueeze(0)
        return enc

    def _move_embeddings(self, embeddings, device: torch.device):
        """EdgeTAM caches a list of FPN feature tensors."""
        if isinstance(embeddings, list):
            return [embedding.to(device) for embedding in embeddings]
        return super()._move_embeddings(embeddings, device)

    def _post_process_masks(self, pred_masks, enc):
        """Upscale EdgeTAM masks with the SAM-2 processor signature."""
        fn = getattr(self.processor, "post_process_masks", None)
        if fn is None:
            fn = self.processor.image_processor.post_process_masks
        return fn(pred_masks.cpu(), enc["original_sizes"])

    def track(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreEdgeTAM image inference is implemented; video tracking via "
            "EdgeTamVideoModel is a follow-up. Use predict() per frame for now."
        )


__all__ = ["LibreEdgeTAM"]
