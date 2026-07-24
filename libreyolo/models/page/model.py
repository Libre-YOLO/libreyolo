"""LibreYOLO wrapper for PaGE gaze-target estimation (inference only).

PaGE ("PAGE: Towards Practical Human-level Gaze Target Estimation",
arXiv:2607.04860; upstream https://github.com/OctopusWen/PaGE, MIT) is a
two-stage gaze-target estimator: a head detector locates people's heads,
and a dual DINOv3-tower cross-attention decoder predicts, per head, a
64x64 heatmap of where that person is looking plus an in/out-of-frame
probability. LibreYOLO embeds the gaze-target decoder and reuses the
pluggable face/head-detector protocol from the L2CS gaze family.
Training and ground-truth-dataset validation are deliberately out of
scope here — train upstream at PaGE.

Weight licensing: the decoder weights are MIT, but the DINOv3 tower
weights bundled in every checkpoint are derivatives of Meta's DINOv3 and
remain governed by the DINOv3 License (commercial use permitted;
redistribution must carry the license text). The LibreYOLO HF repos ship
``DINOv3_LICENSE.md`` alongside the weights accordingly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Dict, Optional

import torch.nn as nn

from ..base.model import BaseModel
from .nn import PAGE_CONFIGS, LibrePAGEModel, detect_size_from_state_dict

logger = logging.getLogger(__name__)


class LibrePAGE(BaseModel):
    """PaGE gaze-target estimator: image -> per-person head box + gaze point."""

    FAMILY = "page"
    FILENAME_PREFIX = "LibrePAGE"
    WEIGHT_EXT = ".pt"
    # Size codes mirror the upstream DINOv3 tower: s (ViT-S), sp (ViT-S+),
    # b (ViT-B), hp (ViT-H+). The scene input is 512x512 for every size.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "s": 512,
        "sp": 512,
        "b": 512,
        "hp": 512,
    }
    SUPPORTED_TASKS = ("gazetarget",)
    DEFAULT_TASK = "gazetarget"

    # Two-stage per-head inference: TTA, tiling and the stacked batched
    # predict path make no sense here (same rationale as L2CS gaze).
    TTA_ENABLED = False
    SUPPORTS_BATCHED_PREDICT = False

    # State-dict fingerprint unique to the PaGE decoder.
    _SIGNATURE_KEYS = (
        "scene_head_interaction_layers.0.cross_attn_scene.attn.q.weight",
        "heatmap_head.0.weight",
    )

    # =========================================================================
    # Detection of weights belonging to this family
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return all(key in weights_dict for key in cls._SIGNATURE_KEYS)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return detect_size_from_state_dict(weights_dict)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # PaGE does not classify objects; report the single "person" class so
        # the surrounding factory plumbing has a sensible value.
        return 1

    @classmethod
    def format_weight_filename(cls, size_code: str) -> str:
        # Canonical filenames carry the task suffix (LibreZipDepth precedent).
        return f"{cls.FILENAME_PREFIX}{size_code}-gazetarget{cls.WEIGHT_EXT}"

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        # "sp"/"hp" contain "s"/"p" prefixes of each other's codes, so match
        # longest-first (the RT-DETR multi-char size-code precedent).
        import re

        stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        for size in sorted(cls.INPUT_SIZES, key=len, reverse=True):
            if re.fullmatch(
                rf"{cls.FILENAME_PREFIX}{size}(-gazetarget)?{cls.WEIGHT_EXT}", stem
            ):
                return size
        return None

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        model_path,
        size: str = "s",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        head_detector=None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,  # always 1 ("person"); ignore caller's nb_classes
            device=device,
            task=task,
            **kwargs,
        )
        self.names = {0: "person"}
        # Resolve the optional default head detector once at construction
        # time; the runner falls back to the shared face-detector default.
        from ..l2cs.face import resolve_face_detector

        self.head_detector = (
            resolve_face_detector(head_detector) if head_detector is not None else None
        )
        # Inference-only family: keep eval() unconditionally.
        self.model.eval()

        # BaseModel.__init__ only loads dict checkpoints; file paths are the
        # subclass's job (same as LibreL2CS).
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))

    # =========================================================================
    # BaseModel abstract surface — gaze-target uses its own runner
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return LibrePAGEModel(self.size)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {name: module for name, module in self.model.named_modules() if name}

    @staticmethod
    def _get_preprocess_numpy():
        raise NotImplementedError(
            "LibrePAGE preprocesses scene + head crops inside "
            "PageInferenceRunner; see libreyolo.models.page.utils."
        )

    def _preprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "LibrePAGE does not use the detection-shaped _preprocess hook; "
            "PageInferenceRunner orchestrates head detection and cropping."
        )

    def _forward(self, *args, **kwargs):
        raise NotImplementedError(
            "LibrePAGE does not use the detection-shaped _forward hook; "
            "PageInferenceRunner calls the underlying network directly."
        )

    def _postprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "LibrePAGE does not use the detection-shaped _postprocess hook; "
            "see libreyolo.models.page.utils.decode_heatmaps."
        )

    def _strict_loading(self) -> bool:
        # Tower keys are remapped at load time between transformers naming
        # conventions (4.56.x vs 5.x); the remap covers every parameter, but
        # non-persistent RoPE buffers stay out of the state dict by design.
        return False

    # =========================================================================
    # Upstream checkpoint recognition (runtime auto-conversion)
    # =========================================================================

    @classmethod
    def convert_upstream_state_dict(cls, weights_dict: dict) -> Optional[dict]:
        from .convert import convert_upstream, is_upstream_state_dict

        if not is_upstream_state_dict(weights_dict):
            return None
        return convert_upstream(weights_dict)

    # =========================================================================
    # Override the runner
    # =========================================================================

    @property
    def _runner(self):
        if getattr(self, "_runner_instance", None) is None:
            from .inference import PageInferenceRunner

            self._runner_instance = PageInferenceRunner(self)
        return self._runner_instance

    # =========================================================================
    # Train / val are explicitly out of scope
    # =========================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training is out of scope for LibrePAGE in LibreYOLO. "
            "Train upstream at https://github.com/OctopusWen/PaGE and load "
            "the resulting state dict here."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            "Validation against gaze-target ground-truth datasets (GazeFollow, "
            "VideoAttentionTarget) is out of scope for LibrePAGE. Evaluate "
            "upstream."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        if format.lower() != "onnx":
            raise NotImplementedError(
                f"LibrePAGE export to {format!r} is not implemented. "
                "The v1 gaze-target export contract supports ONNX only."
            )
        from .export import export_page_onnx

        return export_page_onnx(self, **kwargs)


__all__ = ["LibrePAGE", "PAGE_CONFIGS"]
