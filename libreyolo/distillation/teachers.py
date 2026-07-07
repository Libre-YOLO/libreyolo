"""Foundation-model teachers for feature distillation.

These teachers are *not* LibreYOLO detectors: they are frozen, pretrained
feature extractors (e.g. a DINOv2 ViT) used to supervise a student backbone
via a plain feature-MSE loss (see :class:`~libreyolo.distillation.losses.FeatureMSELoss`).
They plug into :class:`~libreyolo.distillation.distiller.Distiller` through its
``teacher_feature_fn`` seam rather than forward hooks, because a ViT emits token
sequences on a single /patch grid instead of hookable multi-scale BCHW maps.

DINOv2 is Apache-2.0 (Meta). The distillation objective (match student features
to a frozen teacher with an L2 loss) is FitNets (Romero et al., ICLR 2015); this
adapter is written from that public method, independently of any third-party
distillation implementation.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Friendly aliases -> HuggingFace hub id (all Apache-2.0, facebook/*).
DINOV2_ALIASES = {
    "dinov2": "facebook/dinov2-base",
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
}

# ImageNet-1k normalization DINOv2 was trained with. The student batch arrives
# in [0, 1] (YOLO9 scales by 1/255 with no mean/std), so we renormalize here.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def is_foundation_teacher(identifier: str) -> bool:
    """True if ``identifier`` names a supported foundation-model teacher."""
    if not isinstance(identifier, str):
        return False
    key = identifier.lower()
    return key in DINOV2_ALIASES or key.startswith("facebook/dinov2")


class DINOv2Teacher(nn.Module):
    """Frozen DINOv2 ViT teacher that emits a BCHW patch-feature grid.

    Args:
        identifier: Alias (``"dinov2"``, ``"dinov2_vitb14"``, ...) or a
            ``facebook/dinov2-*`` hub id.
        patch_size: ViT patch size (DINOv2 uses 14).

    The teacher is single-scale: its :meth:`get_distill_config` reports one tap
    at stride ``patch_size`` with ``hidden_size`` channels. Feature extraction
    goes through :meth:`extract_features`, not forward hooks.
    """

    def __init__(self, identifier: str = "dinov2", patch_size: int = 14):
        super().__init__()
        try:
            from transformers import Dinov2Model
        except ImportError as e:  # pragma: no cover - dependency guard
            raise ImportError(
                "DINOv2 distillation requires 'transformers'. "
                "Install with: pip install transformers"
            ) from e

        hub_id = DINOV2_ALIASES.get(identifier.lower(), identifier)
        self.hub_id = hub_id
        self.patch_size = patch_size

        # Load the model directly (NOT via AutoBackbone). Meta/HF DINOv2
        # `from_pretrained` has silently returned a randomly-initialized model
        # in some transformers versions; `output_loading_info` lets us verify
        # the pretrained weights actually landed and fail loudly if not.
        model, loading_info = Dinov2Model.from_pretrained(
            hub_id, output_loading_info=True
        )
        missing = loading_info.get("missing_keys", [])
        if missing:
            raise RuntimeError(
                f"DINOv2 teacher '{hub_id}' loaded with {len(missing)} missing "
                f"weight keys (e.g. {missing[:5]}). The backbone would be partly "
                f"random, so distillation targets would be meaningless. Aborting."
            )

        self.dino = model.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        self.hidden_size = int(self.dino.config.hidden_size)
        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )
        logger.info(
            f"DINOv2 teacher ready: {hub_id} "
            f"(hidden_size={self.hidden_size}, patch={patch_size})"
        )

    def get_distill_config(self) -> Dict:
        """Single-scale teacher config: one tap, stride=patch, hidden channels."""
        return {
            "tap_points": [],  # unused: features come via extract_features()
            "channels": [self.hidden_size],
            "strides": [self.patch_size],
        }

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Return ``[BCHW]`` patch features for a batch of ``[0, 1]`` images.

        The image is first resized to the nearest multiple of ``patch_size`` so
        the patch grid covers the *whole* frame. A raw ``h // patch`` grid would
        crop the trailing ``h % patch`` pixels (e.g. 640 -> 45*14 = 630, a 10 px
        border), and the loss would then map that partial grid onto the full
        student feature map, supervising border cells with spatially-shifted
        targets. Resizing to full coverage keeps teacher and student aligned.

        The token sequence's last ``Hp*Wp`` entries are the patch tokens
        (slicing from the end is robust to CLS and register prefix tokens),
        reshaped to a ``(N, hidden_size, Hp, Wp)`` grid.
        """
        images = (images - self._mean) / self._std
        n, _, h, w = images.shape

        # Nearest multiple of patch_size on each axis => full-frame coverage.
        hp = max(1, round(h / self.patch_size))
        wp = max(1, round(w / self.patch_size))
        th, tw = hp * self.patch_size, wp * self.patch_size
        if (th, tw) != (h, w):
            images = torch.nn.functional.interpolate(
                images, size=(th, tw), mode="bilinear", align_corners=False
            )

        out = self.dino(pixel_values=images, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state  # (N, prefix + Hp*Wp, C)
        patch_tokens = tokens[:, -(hp * wp):, :]  # drop CLS/register prefixes
        grid = patch_tokens.reshape(n, hp, wp, self.hidden_size).permute(0, 3, 1, 2)
        return [grid.contiguous()]
