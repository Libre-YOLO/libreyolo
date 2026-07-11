"""LibreEoMT network wrapper around the Hugging Face EoMT implementation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


_EOMT_BASE_CONFIG: dict[str, Any] = {
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-6,
    "image_size": 512,
    "patch_size": 16,
    "num_channels": 3,
    "mlp_ratio": 4,
    "layerscale_value": 1e-5,
    "drop_path_rate": 0.0,
    "num_upscale_blocks": 2,
    "attention_dropout": 0.0,
    "use_swiglu_ffn": False,
    "no_object_weight": 0.1,
    "class_weight": 2.0,
    "mask_weight": 5.0,
    "dice_weight": 5.0,
    "train_num_points": 12544,
    "oversample_ratio": 3.0,
    "importance_sample_ratio": 0.75,
    "num_queries": 100,
    "num_register_tokens": 4,
}

EOMT_S_CONFIG: dict[str, Any] = {
    **_EOMT_BASE_CONFIG,
    "hidden_size": 384,
    "num_hidden_layers": 12,
    "num_attention_heads": 6,
    "num_blocks": 3,
}

EOMT_B_CONFIG: dict[str, Any] = {
    **_EOMT_BASE_CONFIG,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_blocks": 3,
}

EOMT_L_ADE20K_CONFIG: dict[str, Any] = {
    **_EOMT_BASE_CONFIG,
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_blocks": 4,
}

_SIZE_TO_CONFIG: dict[str, dict[str, Any]] = {
    "s": EOMT_S_CONFIG,
    "b": EOMT_B_CONFIG,
    "l": EOMT_L_ADE20K_CONFIG,
}


def _load_transformers_eomt():
    try:
        from transformers import EomtConfig, EomtForUniversalSegmentation
    except ImportError as exc:  # pragma: no cover - dependency message
        raise ModuleNotFoundError(
            "LibreEoMT requires Hugging Face Transformers with EoMT support. "
            "Install with: pip install 'libreyolo[eomt]' or 'libreyolo[rfdetr]'."
        ) from exc
    return EomtConfig, EomtForUniversalSegmentation


def normalize_eomt_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Return tensor keys in the raw HF EoMT layout.

    Strips wrapper prefixes repeatedly so compound prefixes (e.g. a
    ``torch.compile`` + DDP ``_orig_mod.module.`` combination) are fully removed
    in one call, matching LibrePIDNet's while-changed loop. A single pass would
    strip only the outer prefix and leave the inner one behind.
    """
    prefixes = ("module.", "_orig_mod.", "model.", "eomt.")
    normalized = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        normalized[key] = value
    return normalized


class LibreEoMTNet(nn.Module):
    """EoMT-L model adapted to LibreYOLO dense-task conventions."""

    def __init__(
        self,
        config: str = "l",
        nb_classes: int = 150,
        image_size: int = 512,
        num_queries: int = 100,
    ) -> None:
        super().__init__()
        if config not in _SIZE_TO_CONFIG:
            raise ValueError(
                f"LibreEoMT size {config!r} is not supported. "
                f"Valid sizes: {sorted(_SIZE_TO_CONFIG)}."
            )
        size_cfg = _SIZE_TO_CONFIG[config]
        self.nb_classes = int(nb_classes)
        self.image_size = int(image_size)
        self.num_queries = int(num_queries)
        self.patch_size = int(size_cfg["patch_size"])

        EomtConfig, EomtForUniversalSegmentation = _load_transformers_eomt()
        id2label = {i: f"class_{i}" for i in range(self.nb_classes)}
        label2id = {label: i for i, label in id2label.items()}
        hf_config = EomtConfig(
            **{
                **size_cfg,
                "image_size": self.image_size,
                "num_queries": self.num_queries,
                "id2label": id2label,
                "label2id": label2id,
            }
        )
        self.eomt = EomtForUniversalSegmentation(hf_config)

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def state_dict(self, *args, **kwargs):  # noqa: D401
        """Expose the raw HF EoMT key layout for converted checkpoints."""
        return self.eomt.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = False):
        state = normalize_eomt_state_dict(state_dict)
        return self.eomt.load_state_dict(state, strict=strict)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _, _, height, width = x.shape
        x = (x - self.pixel_mean) / self.pixel_std
        outputs = self.eomt(pixel_values=x)

        class_logits = outputs.class_queries_logits.float()
        mask_logits = outputs.masks_queries_logits.float()
        if tuple(mask_logits.shape[-2:]) != (height, width):
            mask_logits = F.interpolate(
                mask_logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        class_scores = class_logits.softmax(dim=-1)[..., : self.nb_classes]
        mask_scores = mask_logits.sigmoid()
        semantic_scores = torch.einsum("bqc,bqhw->bchw", class_scores, mask_scores)
        return {
            "semantic_logits": semantic_scores,
            "class_queries_logits": class_logits,
            "masks_queries_logits": mask_logits,
        }


__all__ = [
    "EOMT_S_CONFIG",
    "EOMT_B_CONFIG",
    "EOMT_L_ADE20K_CONFIG",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LibreEoMTNet",
    "normalize_eomt_state_dict",
]
