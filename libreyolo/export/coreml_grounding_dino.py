"""Frozen-vocabulary Grounding DINO contract fragments for Core ML.

Grounding DINO cannot be reduced to one static embedding per class: its text
tokens continue through every cross-modal encoder layer and the decoder attends
to the resulting text sequence.  A finite, image-only export can nevertheless
freeze the part that is independent of the image:

* the exact prompt and BERT token sequence;
* token attention, block-attention, and position-id tensors; and
* the projected output of the BERT tower, immediately before cross-modal
  fusion.

The adapter below deliberately retains the cross-modal text path.  It removes
only the tokenizer and BERT tower from the deployment graph. The frozen
features come from the loaded Hugging Face source BERT (including its selected
attention implementation), and the translated cross-modal graph is checked
numerically against that source model.

Prompt handling, WordPiece decoding, image preprocessing, and grounded
postprocessing are adapted from Hugging Face Transformers v5.12.1 source commit
``ddb849abe009d1089e6c691bfc897f27211c663c`` (Apache-2.0).  Exact provenance
is recorded in ``libreyolo/models/grounding_dino/NOTICE``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..models.grounding_dino.nn import (
    GroundingDinoDetectionModel,
    encode_sine_pos,
    generate_masks_with_special_tokens_and_transfer_map,
)

GROUNDING_DINO_COREML_CONTRACT = "grounding_dino_frozen_vocabulary_v1"
GROUNDING_DINO_COREML_PREPROCESS = "rgb_uint8_torchvision_bilinear_antialias_stretch_v1"
GROUNDING_DINO_COREML_POSTPROCESS = "grounded_token_threshold_wordpiece_v1"
GROUNDING_DINO_COREML_INPUT_NAME = "image"
GROUNDING_DINO_COREML_OUTPUT_NAMES = ("token_logits", "pred_boxes")
GROUNDING_DINO_COREML_CANVAS = 800
GROUNDING_DINO_COREML_MEAN = (0.485, 0.456, 0.406)
GROUNDING_DINO_COREML_STD = (0.229, 0.224, 0.225)
GROUNDING_DINO_COREML_SPECIAL_TOKENS = (101, 102, 1012, 1029)
GROUNDING_DINO_COREML_PAD_TOKEN = 0
GROUNDING_DINO_COREML_PROMPT_PREFIX = "a "
GROUNDING_DINO_COREML_PROMPT_SEPARATOR = ". "
GROUNDING_DINO_COREML_PROMPT_SUFFIX = "."

_LABEL_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ARTICLES = frozenset({"a", "an", "the"})


@dataclass(frozen=True)
class GroundingDinoCoreMLProfile:
    """One fixed Grounding DINO deployment profile."""

    size: str
    canvas: int
    num_queries: int
    max_text_len: int
    backbone_embed_dim: int
    backbone_depths: tuple[int, ...]
    backbone_heads: tuple[int, ...]
    backbone_window: int


GROUNDING_DINO_COREML_PROFILES = {
    "t": GroundingDinoCoreMLProfile(
        size="t",
        canvas=800,
        num_queries=900,
        max_text_len=256,
        backbone_embed_dim=96,
        backbone_depths=(2, 2, 6, 2),
        backbone_heads=(3, 6, 12, 24),
        backbone_window=7,
    ),
    "b": GroundingDinoCoreMLProfile(
        size="b",
        canvas=800,
        num_queries=900,
        max_text_len=256,
        backbone_embed_dim=128,
        backbone_depths=(2, 2, 18, 2),
        backbone_heads=(4, 8, 16, 32),
        backbone_window=12,
    ),
}


@dataclass(frozen=True)
class GroundingDinoFrozenText:
    """Exact prompt tensors and text features frozen before cross-modal fusion."""

    labels: tuple[str, ...]
    prompt: str
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    text_self_attention_masks: torch.Tensor
    position_ids: torch.Tensor
    text_features: torch.Tensor
    token_pieces: tuple[str, ...]

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[1])


def _ordered_names(
    names: Mapping[int, str] | Sequence[str],
) -> list[str]:
    """Return a strict finite vocabulary in class-id order."""
    if isinstance(names, Mapping):
        try:
            keys = sorted(int(key) for key in names)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Grounding DINO Core ML class ids must be integers."
            ) from exc
        if keys != list(range(len(keys))):
            raise ValueError(
                "Grounding DINO Core ML class ids must be contiguous from "
                f"zero; got {keys!r}."
            )
        values = [names[key] if key in names else names[str(key)] for key in keys]
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        values = list(names)
    else:
        raise TypeError(
            "Grounding DINO Core ML classes must be a mapping or a finite "
            "label sequence."
        )

    if not values:
        raise ValueError(
            "Grounding DINO Core ML export requires at least one frozen class. "
            "Call set_classes([...]) before export."
        )
    if not all(isinstance(value, str) for value in values):
        raise TypeError("Grounding DINO Core ML class labels must all be strings.")
    labels = [value.strip() for value in values]
    if any(not label for label in labels):
        raise ValueError("Grounding DINO Core ML class labels must not be blank.")
    if any("." in label or "?" in label for label in labels):
        raise ValueError(
            "Grounding DINO Core ML class labels must not contain '.' or '?': "
            "those tokens delimit text-attention blocks."
        )
    normalized = [" ".join(_label_tokens(label)) for label in labels]
    if any(not label for label in normalized):
        raise ValueError(
            "Grounding DINO Core ML class labels must contain letters or digits."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Grounding DINO Core ML class labels must be unique after "
            "case/punctuation normalization."
        )
    return labels


def grounding_dino_coreml_prompt(
    names: Mapping[int, str] | Sequence[str],
) -> str:
    """Render the exact prompt used by LibreGroundingDINO."""
    labels = _ordered_names(names)
    phrases = [
        f"{GROUNDING_DINO_COREML_PROMPT_PREFIX}{label.lower()}" for label in labels
    ]
    return (
        GROUNDING_DINO_COREML_PROMPT_SEPARATOR.join(phrases)
        + GROUNDING_DINO_COREML_PROMPT_SUFFIX
    )


def grounding_dino_coreml_vocabulary_hash(
    names: Mapping[int, str] | Sequence[str],
) -> str:
    """Hash labels plus the exact prompt grammar."""
    payload = {
        "labels": _ordered_names(names),
        "prefix": GROUNDING_DINO_COREML_PROMPT_PREFIX,
        "separator": GROUNDING_DINO_COREML_PROMPT_SEPARATOR,
        "suffix": GROUNDING_DINO_COREML_PROMPT_SUFFIX,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_abi_payload(
    frozen: GroundingDinoFrozenText,
) -> dict[str, Any]:
    return {
        "labels": list(frozen.labels),
        "prompt": frozen.prompt,
        "input_ids": frozen.input_ids.reshape(-1).tolist(),
        "token_type_ids": frozen.token_type_ids.reshape(-1).tolist(),
        "attention_mask": frozen.attention_mask.reshape(-1).tolist(),
        "token_pieces": list(frozen.token_pieces),
    }


def grounding_dino_coreml_text_abi_hash(
    frozen: GroundingDinoFrozenText,
) -> str:
    """Hash all host-visible frozen text fields."""
    encoded = json.dumps(
        _text_abi_payload(frozen),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_grounding_dino_coreml_profile(
    *,
    size: str | None,
    canvas_hw: tuple[int, int] | None = None,
    config: Any | None = None,
) -> GroundingDinoCoreMLProfile:
    """Resolve a released fixed profile and reject silent architecture drift."""
    key = str(size or "").strip().lower()
    profile = GROUNDING_DINO_COREML_PROFILES.get(key)
    if profile is None:
        raise NotImplementedError(
            "Grounding DINO Core ML export supports only the t and b "
            f"checkpoints; got size={size!r}."
        )
    if canvas_hw is not None:
        height, width = (int(value) for value in canvas_hw)
        expected = (profile.canvas, profile.canvas)
        if (height, width) != expected:
            raise NotImplementedError(
                "Grounding DINO Core ML v1 uses a fixed square stretch canvas "
                f"of {expected[1]}x{expected[0]}; got {width}x{height}."
            )
    if config is not None:
        backbone = getattr(config, "backbone_config", None)
        actual = {
            "num_queries": int(getattr(config, "num_queries", -1)),
            "max_text_len": int(getattr(config, "max_text_len", -1)),
            "backbone_embed_dim": int(getattr(backbone, "embed_dim", -1)),
            "backbone_depths": tuple(
                int(value) for value in getattr(backbone, "depths", ())
            ),
            "backbone_heads": tuple(
                int(value) for value in getattr(backbone, "num_heads", ())
            ),
            "backbone_window": int(getattr(backbone, "window_size", -1)),
        }
        expected = {
            "num_queries": profile.num_queries,
            "max_text_len": profile.max_text_len,
            "backbone_embed_dim": profile.backbone_embed_dim,
            "backbone_depths": profile.backbone_depths,
            "backbone_heads": profile.backbone_heads,
            "backbone_window": profile.backbone_window,
        }
        if actual != expected:
            raise RuntimeError(
                "Grounding DINO Core ML checkpoint/config does not match the "
                f"released {key!r} profile: expected {expected}, got {actual}."
            )
    return profile


def grounding_dino_coreml_input_contract() -> dict[str, Any]:
    """Declare the fixed canonical image boundary."""
    return {
        "name": GROUNDING_DINO_COREML_INPUT_NAME,
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "stretch",
        "interpolation": "bilinear",
        "resize_backend": "torchvision",
        "antialias": True,
        "pad_value": 0,
    }


def grounding_dino_coreml_validation_contract() -> dict[str, str]:
    """Describe the canonical raw-image tensors supplied by validators."""
    return {"color": "rgb", "range": "0_255"}


def grounding_dino_coreml_output_contract() -> list[dict[str, Any]]:
    """Return the fixed raw detector ABI in graph order."""
    return [
        {
            "name": "token_logits",
            "role": "text_token_logits",
            "encoding": "raw_logits_compact_frozen_sequence",
            "rank": 3,
        },
        {
            "name": "pred_boxes",
            "role": "boxes",
            "encoding": "cxcywh_normalized_stretched_canvas",
            "rank": 3,
        },
    ]


def expected_grounding_dino_coreml_shapes(
    *,
    size: str,
    sequence_length: int,
) -> dict[str, tuple[int, ...]]:
    """Return exact fixed output shapes."""
    profile = validate_grounding_dino_coreml_profile(size=size)
    if (
        isinstance(sequence_length, bool)
        or int(sequence_length) <= 2
        or int(sequence_length) > profile.max_text_len
    ):
        raise ValueError(
            "Grounding DINO Core ML sequence length must be in "
            f"[3, {profile.max_text_len}]."
        )
    return {
        "token_logits": (
            1,
            profile.num_queries,
            int(sequence_length),
        ),
        "pred_boxes": (1, profile.num_queries, 4),
    }


def grounding_dino_coreml_metadata(
    *,
    size: str,
    names: Mapping[int, str] | Sequence[str],
    frozen: GroundingDinoFrozenText,
) -> dict[str, Any]:
    """Return the complete self-contained frozen text manifest."""
    labels = _ordered_names(names)
    if labels != list(frozen.labels):
        raise ValueError(
            "Grounding DINO frozen text labels disagree with export names."
        )
    profile = validate_grounding_dino_coreml_profile(size=size)
    if frozen.sequence_length > profile.max_text_len:
        raise ValueError("Grounding DINO frozen prompt exceeds max_text_len.")
    payload = _text_abi_payload(frozen)
    return {
        "frozen_classes": True,
        "grounding_dino_contract": GROUNDING_DINO_COREML_CONTRACT,
        "grounding_dino_preprocess": GROUNDING_DINO_COREML_PREPROCESS,
        "grounding_dino_postprocess": GROUNDING_DINO_COREML_POSTPROCESS,
        "grounding_dino_vocabulary_sha256": (
            grounding_dino_coreml_vocabulary_hash(labels)
        ),
        "grounding_dino_text_abi_sha256": (grounding_dino_coreml_text_abi_hash(frozen)),
        "grounding_dino_prompt": frozen.prompt,
        "grounding_dino_input_ids_json": json.dumps(
            payload["input_ids"], separators=(",", ":")
        ),
        "grounding_dino_token_type_ids_json": json.dumps(
            payload["token_type_ids"], separators=(",", ":")
        ),
        "grounding_dino_attention_mask_json": json.dumps(
            payload["attention_mask"], separators=(",", ":")
        ),
        "grounding_dino_token_pieces_json": json.dumps(
            payload["token_pieces"],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "grounding_dino_sequence_length": frozen.sequence_length,
        "grounding_dino_max_text_len": profile.max_text_len,
        "grounding_dino_num_queries": profile.num_queries,
        "grounding_dino_canvas_height": profile.canvas,
        "grounding_dino_canvas_width": profile.canvas,
        "grounding_dino_non_square_geometry": (
            "fixed_stretch_differs_from_native_keep_aspect"
        ),
    }


def _strict_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "1"}:
        return True
    if token in {"false", "0"}:
        return False
    raise ValueError(f"Grounding DINO Core ML metadata {key!r} must be true or false.")


def _strict_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Grounding DINO Core ML metadata {key!r} must be an integer.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise ValueError(f"Grounding DINO Core ML metadata {key!r} must be an integer.")
    if parsed <= 0:
        raise ValueError(f"Grounding DINO Core ML metadata {key!r} must be positive.")
    return parsed


def _json_list(
    metadata: Mapping[str, Any],
    key: str,
    *,
    element_type: type,
) -> list[Any]:
    try:
        value = json.loads(str(metadata[key]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Grounding DINO Core ML metadata {key!r} must be valid JSON."
        ) from exc
    if not isinstance(value, list) or not all(
        isinstance(item, element_type) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(
            f"Grounding DINO Core ML metadata {key!r} has invalid elements."
        )
    return value


def frozen_grounding_dino_text_from_metadata(
    metadata: Mapping[str, Any],
    *,
    names: Mapping[int, str] | Sequence[str],
) -> dict[str, Any]:
    """Parse the host-visible text ABI without importing a tokenizer."""
    labels = _ordered_names(names)
    input_ids = _json_list(
        metadata,
        "grounding_dino_input_ids_json",
        element_type=int,
    )
    token_type_ids = _json_list(
        metadata,
        "grounding_dino_token_type_ids_json",
        element_type=int,
    )
    attention_mask = _json_list(
        metadata,
        "grounding_dino_attention_mask_json",
        element_type=int,
    )
    token_pieces = _json_list(
        metadata,
        "grounding_dino_token_pieces_json",
        element_type=str,
    )
    sequence_length = _strict_int(
        metadata.get("grounding_dino_sequence_length"),
        key="grounding_dino_sequence_length",
    )
    if not (
        len(input_ids)
        == len(token_type_ids)
        == len(attention_mask)
        == len(token_pieces)
        == sequence_length
    ):
        raise ValueError(
            "Grounding DINO Core ML frozen text metadata has inconsistent "
            "sequence lengths."
        )
    if input_ids[0] != 101 or input_ids[-1] != 102:
        raise ValueError(
            "Grounding DINO Core ML frozen text must start with BERT [CLS] "
            "and end with [SEP]."
        )
    if any(value != 0 for value in token_type_ids):
        raise ValueError("Grounding DINO Core ML v1 supports only BERT segment zero.")
    if any(value != 1 for value in attention_mask):
        raise ValueError(
            "Grounding DINO Core ML v1 does not permit padding inside the "
            "frozen prompt."
        )
    return {
        "labels": labels,
        "prompt": str(metadata.get("grounding_dino_prompt", "")),
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "token_pieces": token_pieces,
    }


def validate_grounding_dino_coreml_metadata(
    metadata: Mapping[str, Any],
    *,
    size: str,
    names: Mapping[int, str] | Sequence[str],
) -> dict[str, Any]:
    """Validate every field that changes host interpretation."""
    profile = validate_grounding_dino_coreml_profile(size=size)
    if not _strict_bool(
        metadata.get("frozen_classes"),
        key="frozen_classes",
    ):
        raise ValueError(
            "Grounding DINO Core ML artifacts must declare frozen_classes=true."
        )
    expected_strings = {
        "grounding_dino_contract": GROUNDING_DINO_COREML_CONTRACT,
        "grounding_dino_preprocess": GROUNDING_DINO_COREML_PREPROCESS,
        "grounding_dino_postprocess": GROUNDING_DINO_COREML_POSTPROCESS,
        "grounding_dino_vocabulary_sha256": (
            grounding_dino_coreml_vocabulary_hash(names)
        ),
        "grounding_dino_non_square_geometry": (
            "fixed_stretch_differs_from_native_keep_aspect"
        ),
    }
    for key, expected in expected_strings.items():
        actual = str(metadata.get(key, ""))
        if actual != expected:
            raise ValueError(
                f"Grounding DINO Core ML metadata {key!r} was modified: "
                f"expected {expected!r}, got {actual!r}."
            )
    expected_integers = {
        "grounding_dino_max_text_len": profile.max_text_len,
        "grounding_dino_num_queries": profile.num_queries,
        "grounding_dino_canvas_height": profile.canvas,
        "grounding_dino_canvas_width": profile.canvas,
    }
    for key, expected in expected_integers.items():
        actual = _strict_int(metadata.get(key), key=key)
        if actual != expected:
            raise ValueError(
                f"Grounding DINO Core ML metadata {key!r} must equal "
                f"{expected}, got {actual}."
            )

    parsed = frozen_grounding_dino_text_from_metadata(
        metadata,
        names=names,
    )
    if parsed["prompt"] != grounding_dino_coreml_prompt(names):
        raise ValueError(
            "Grounding DINO Core ML prompt does not match the frozen classes."
        )
    if len(parsed["input_ids"]) > profile.max_text_len:
        raise ValueError("Grounding DINO Core ML prompt exceeds max_text_len.")
    hash_payload = {
        "labels": parsed["labels"],
        "prompt": parsed["prompt"],
        "input_ids": parsed["input_ids"],
        "token_type_ids": parsed["token_type_ids"],
        "attention_mask": parsed["attention_mask"],
        "token_pieces": parsed["token_pieces"],
    }
    actual_hash = hashlib.sha256(
        json.dumps(
            hash_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    declared_hash = str(metadata.get("grounding_dino_text_abi_sha256", ""))
    if actual_hash != declared_hash:
        raise ValueError(
            "Grounding DINO Core ML frozen text ABI hash does not match its "
            "serialized tensors."
        )
    return parsed


def _swin_state_dict_from_hf(
    state_dict: Mapping[str, torch.Tensor],
    *,
    config: Any,
) -> dict[str, torch.Tensor]:
    """Map the Transformers Swin state tree to LibreYOLO's native tree."""
    output: dict[str, torch.Tensor] = {}
    backbone = config.backbone_config
    depths = tuple(int(value) for value in backbone.depths)

    def require(key: str) -> torch.Tensor:
        try:
            return state_dict[key]
        except KeyError as exc:
            raise RuntimeError(
                "Grounding DINO Hugging Face checkpoint is missing required "
                f"Swin tensor {key!r}."
            ) from exc

    for stage, depth in enumerate(depths):
        for block in range(depth):
            source = f"swin.encoder.layers.{stage}.blocks.{block}."
            target = f"backbone_conv._backbone.layers.{stage}.blocks.{block}."
            output[target + "norm1.weight"] = require(
                source + "layernorm_before.weight"
            )
            output[target + "norm1.bias"] = require(source + "layernorm_before.bias")
            output[target + "norm2.weight"] = require(source + "layernorm_after.weight")
            output[target + "norm2.bias"] = require(source + "layernorm_after.bias")
            output[target + "attn.qkv.weight"] = torch.cat(
                [
                    require(source + "attention.q_proj.weight"),
                    require(source + "attention.k_proj.weight"),
                    require(source + "attention.v_proj.weight"),
                ],
                dim=0,
            )
            output[target + "attn.qkv.bias"] = torch.cat(
                [
                    require(source + "attention.q_proj.bias"),
                    require(source + "attention.k_proj.bias"),
                    require(source + "attention.v_proj.bias"),
                ],
                dim=0,
            )
            output[target + "attn.proj.weight"] = require(
                source + "attention.o_proj.weight"
            )
            output[target + "attn.proj.bias"] = require(
                source + "attention.o_proj.bias"
            )
            output[target + "attn.relative_position_bias_table"] = require(
                source + "attention.relative_position_bias.relative_position_bias_table"
            )
            for layer in ("fc1", "fc2"):
                output[target + f"mlp.{layer}.weight"] = require(
                    source + f"mlp.{layer}.weight"
                )
                output[target + f"mlp.{layer}.bias"] = require(
                    source + f"mlp.{layer}.bias"
                )

    output["backbone_conv._backbone.patch_embed.proj.weight"] = require(
        "swin.embeddings.patch_embeddings.projection.weight"
    )
    output["backbone_conv._backbone.patch_embed.proj.bias"] = require(
        "swin.embeddings.patch_embeddings.projection.bias"
    )
    output["backbone_conv._backbone.patch_embed.norm.weight"] = require(
        "swin.embeddings.norm.weight"
    )
    output["backbone_conv._backbone.patch_embed.norm.bias"] = require(
        "swin.embeddings.norm.bias"
    )
    for stage in range(len(depths) - 1):
        source = f"swin.encoder.layers.{stage}.downsample."
        target = f"backbone_conv._backbone.layers.{stage + 1}.downsample."
        output[target + "norm.weight"] = require(source + "norm.weight")
        output[target + "norm.bias"] = require(source + "norm.bias")
        output[target + "reduction.weight"] = require(source + "reduction.weight")
    return output


def grounding_dino_hf_to_native_state_dict(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Translate a compatible Transformers detector state dict."""
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) not in {
        None,
        "grounding-dino",
    }:
        raise TypeError(
            "Grounding DINO Core ML export requires a "
            "GroundingDinoForObjectDetection-compatible model."
        )
    full = model.state_dict()
    output: dict[str, torch.Tensor] = {}
    swin: dict[str, torch.Tensor] = {}
    backbone_prefix = "model.backbone.conv_encoder.model."
    norm_prefix = backbone_prefix + "hidden_states_norms.stage"

    for key, value in full.items():
        if key.startswith(backbone_prefix + "swin."):
            swin[key[len(backbone_prefix) :]] = value
            continue
        if key.startswith(norm_prefix):
            remainder = key[len(norm_prefix) :]
            stage_text, separator, suffix = remainder.partition(".")
            if not separator or not stage_text.isdigit():
                raise RuntimeError(
                    "Grounding DINO checkpoint has an invalid backbone "
                    f"normalization key {key!r}."
                )
            stage = int(stage_text)
            output[f"backbone_conv.hidden_states_norms.{stage - 2}.{suffix}"] = value
            continue
        if key.startswith("model.decoder.bbox_embed."):
            output[key[len("model.") :]] = value
            continue
        if key.startswith("model.decoder.class_embed") or key.startswith("class_embed"):
            continue
        if key.startswith("model.backbone.position_embedding"):
            continue
        if key.startswith("model."):
            output[key[len("model.") :]] = value
        else:
            output[key] = value

    output.update(_swin_state_dict_from_hf(swin, config=config))
    return output


def build_native_grounding_dino_from_hf(
    model: nn.Module,
) -> GroundingDinoDetectionModel:
    """Build the trace-oriented native graph and transfer every used tensor."""
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("Grounding DINO Core ML export requires model.config.")
    native = GroundingDinoDetectionModel(config).to(device="cpu").eval()
    translated = grounding_dino_hf_to_native_state_dict(model)
    result = native.load_state_dict(translated, strict=False)
    ignored = ("relative_position_index", "num_batches_tracked")
    missing = [
        key for key in result.missing_keys if not any(token in key for token in ignored)
    ]
    unexpected = [
        key
        for key in result.unexpected_keys
        if not any(token in key for token in ignored)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Grounding DINO Hugging Face/native state trees are "
            f"incompatible: missing={missing!r}, unexpected={unexpected!r}."
        )
    return native


def freeze_grounding_dino_text(
    source_model: nn.Module,
    processor: Any,
    names: Mapping[int, str] | Sequence[str],
) -> GroundingDinoFrozenText:
    """Freeze the source model's exact BERT output before cross-modal fusion.

    The loaded Transformers model is the semantic authority.  In particular,
    its BERT implementation may use SDPA while LibreYOLO's translated detector
    uses eager attention for its trace-oriented image/cross-modal graph.
    Freezing from the translated BERT would silently change the public source
    model's outputs even though all text parameters transferred correctly.
    """
    native_model = (
        source_model
        if isinstance(source_model, GroundingDinoDetectionModel)
        else None
    )
    if native_model is not None:
        text_owner: nn.Module = native_model
        config = getattr(native_model, "config", None)
    else:
        text_owner = getattr(source_model, "model", None)
        config = getattr(source_model, "config", None)
        if (
            not isinstance(text_owner, nn.Module)
            or not isinstance(
                getattr(text_owner, "text_backbone", None),
                nn.Module,
            )
            or not isinstance(
                getattr(text_owner, "text_projection", None),
                nn.Module,
            )
        ):
            raise TypeError(
                "Grounding DINO text freezing requires either LibreYOLO's "
                "native detector or the loaded Transformers "
                "GroundingDinoForObjectDetection model."
            )
    max_text_len = int(
        getattr(
            native_model if native_model is not None else config,
            "max_text_len",
            0,
        )
    )
    d_model = int(
        getattr(
            native_model if native_model is not None else config,
            "d_model",
            0,
        )
    )
    if max_text_len <= 0 or d_model <= 0:
        raise TypeError(
            "Grounding DINO text freezing requires positive max_text_len "
            "and d_model configuration values."
        )
    labels = tuple(_ordered_names(names))
    prompt = grounding_dino_coreml_prompt(labels)
    tokenizer = getattr(processor, "tokenizer", processor)
    if not callable(tokenizer):
        raise TypeError(
            "Grounding DINO Core ML export requires the loaded BERT tokenizer."
        )
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
        return_tensors="pt",
    )
    if not isinstance(encoded, Mapping):
        try:
            encoded = dict(encoded)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Grounding DINO tokenizer returned an invalid payload."
            ) from exc
    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    token_type_ids = encoded.get("token_type_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise RuntimeError("Grounding DINO tokenizer must return rank-two input_ids.")
    if input_ids.shape[0] != 1:
        raise RuntimeError("Grounding DINO frozen Core ML prompt must have batch one.")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    if (
        not torch.is_tensor(attention_mask)
        or not torch.is_tensor(token_type_ids)
        or attention_mask.shape != input_ids.shape
        or token_type_ids.shape != input_ids.shape
    ):
        raise RuntimeError(
            "Grounding DINO frozen input_ids, attention_mask, and "
            "token_type_ids must have identical shapes."
        )

    input_ids = input_ids.detach().to(device="cpu", dtype=torch.long)
    attention_mask = attention_mask.detach().to(
        device="cpu",
        dtype=torch.long,
    )
    token_type_ids = token_type_ids.detach().to(
        device="cpu",
        dtype=torch.long,
    )
    if input_ids.shape[1] > max_text_len:
        raise ValueError(
            "Grounding DINO Core ML cannot freeze a chunked vocabulary: "
            f"prompt length {int(input_ids.shape[1])} exceeds "
            f"max_text_len={max_text_len}. Export a smaller "
            "class set."
        )
    ids = input_ids.reshape(-1).tolist()
    if ids[0] != 101 or ids[-1] != 102:
        raise ValueError(
            "Grounding DINO Core ML v1 requires the released uncased BERT "
            "tokenizer with [CLS]=101 and [SEP]=102."
        )
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if unknown_id is not None and int(unknown_id) in ids:
        raise ValueError(
            "Grounding DINO Core ML class labels produced a BERT [UNK] token; "
            "rename the class vocabulary before export."
        )
    inner_ids = ids[1:-1]
    if (
        inner_ids.count(1012) != len(labels)
        or 1029 in inner_ids
        or any(value in {0, 101, 102} for value in inner_ids)
    ):
        raise ValueError(
            "Grounding DINO Core ML prompt does not contain exactly one "
            "period-delimited BERT attention block per class."
        )
    if not bool((attention_mask == 1).all()):
        raise ValueError("Grounding DINO Core ML v1 does not freeze padded prompts.")
    if not bool((token_type_ids == 0).all()):
        raise ValueError("Grounding DINO Core ML v1 supports only BERT segment zero.")

    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert):
        raise TypeError(
            "Grounding DINO tokenizer must expose convert_ids_to_tokens() "
            "so the artifact remains self-contained."
        )
    pieces = convert(ids)
    if not isinstance(pieces, (list, tuple)) or len(pieces) != len(ids):
        raise RuntimeError(
            "Grounding DINO tokenizer returned an invalid token-piece ABI."
        )
    token_pieces = tuple(str(piece) for piece in pieces)
    if any(not piece for piece in token_pieces):
        raise RuntimeError("Grounding DINO tokenizer returned a blank token piece.")

    text_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(
        input_ids
    )
    text_backbone = text_owner.text_backbone
    text_projection = text_owner.text_projection
    backbone_training = text_backbone.training
    projection_training = text_projection.training
    try:
        reference_parameter = next(text_backbone.parameters())
    except StopIteration as exc:
        raise TypeError(
            "Grounding DINO text backbone must be parameterized."
        ) from exc
    text_device = reference_parameter.device
    text_backbone.eval()
    text_projection.eval()
    try:
        with torch.inference_mode():
            device_input_ids = input_ids.to(device=text_device)
            device_text_masks = text_masks[:, None, :, :].to(
                device=text_device
            )
            device_token_type_ids = token_type_ids.to(device=text_device)
            device_position_ids = position_ids.to(device=text_device)
            if native_model is not None:
                features = text_backbone(
                    device_input_ids,
                    device_text_masks,
                    device_token_type_ids,
                    device_position_ids,
                )
            else:
                text_outputs = text_backbone(
                    device_input_ids,
                    device_text_masks,
                    device_token_type_ids,
                    device_position_ids,
                    return_dict=True,
                )
                features = text_outputs.last_hidden_state
            features = text_projection(features)
    finally:
        text_backbone.train(backbone_training)
        text_projection.train(projection_training)
    if features.shape != (
        1,
        input_ids.shape[1],
        d_model,
    ):
        raise RuntimeError(
            "Grounding DINO frozen BERT feature shape disagrees with the "
            f"detector: got {tuple(features.shape)}."
        )
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError(
            "Grounding DINO frozen BERT features contain NaN or infinity."
        )
    return GroundingDinoFrozenText(
        labels=labels,
        prompt=prompt,
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        text_self_attention_masks=text_masks.detach().to(device="cpu"),
        position_ids=position_ids.detach().to(device="cpu"),
        text_features=features.detach().to(
            device="cpu",
            dtype=torch.float32,
        ),
        token_pieces=token_pieces,
    )


class GroundingDinoFrozenCoreMLAdapter(nn.Module):
    """Image-only native detector retaining the full cross-modal text path."""

    def __init__(
        self,
        native_model: GroundingDinoDetectionModel,
        frozen_text: GroundingDinoFrozenText,
        *,
        canvas_hw: tuple[int, int],
    ) -> None:
        super().__init__()
        if not isinstance(native_model, GroundingDinoDetectionModel):
            raise TypeError("Grounding DINO Core ML adapter requires the native model.")
        height, width = (int(value) for value in canvas_hw)
        if height <= 0 or width <= 0:
            raise ValueError(
                "Grounding DINO Core ML canvas dimensions must be positive."
            )
        if frozen_text.text_features.shape[-1] != native_model.d_model:
            raise ValueError(
                "Grounding DINO frozen text feature width disagrees with the detector."
            )
        sequence_length = frozen_text.sequence_length
        if sequence_length <= 2 or sequence_length > native_model.max_text_len:
            raise ValueError("Grounding DINO frozen prompt length is invalid.")
        expected_square = (1, sequence_length, sequence_length)
        if tuple(frozen_text.text_self_attention_masks.shape) != expected_square:
            raise ValueError(
                "Grounding DINO frozen self-attention mask has the wrong shape."
            )
        if tuple(frozen_text.position_ids.shape) != (1, sequence_length):
            raise ValueError("Grounding DINO frozen position ids have the wrong shape.")

        self.backbone_conv = native_model.backbone_conv
        self.position_embedding = native_model.position_embedding
        self.input_proj_vision = native_model.input_proj_vision
        self.encoder = native_model.encoder
        self.decoder = native_model.decoder
        self.level_embed = native_model.level_embed
        self.enc_output = native_model.enc_output
        self.enc_output_norm = native_model.enc_output_norm
        self.encoder_output_bbox_embed = native_model.encoder_output_bbox_embed
        self.query_position_embeddings = native_model.query_position_embeddings
        self.max_text_len = int(native_model.max_text_len)
        self.num_queries = int(native_model.config.num_queries)
        self.canvas_hw = (height, width)
        self.frozen_text_contract = frozen_text

        self.register_buffer(
            "text_features",
            frozen_text.text_features.detach().clone(),
            persistent=True,
        )
        self.register_buffer(
            "text_token_mask",
            frozen_text.attention_mask.detach().bool().clone(),
            persistent=True,
        )
        self.register_buffer(
            "text_self_attention_masks",
            frozen_text.text_self_attention_masks.detach().bool().clone(),
            persistent=True,
        )
        self.register_buffer(
            "position_ids",
            frozen_text.position_ids.detach().long().clone(),
            persistent=True,
        )
        self.register_buffer(
            "pixel_mask",
            torch.ones((1, height, width), dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor(
                GROUNDING_DINO_COREML_MEAN,
                dtype=torch.float32,
            ).view(1, 3, 1, 1)
            * 255.0,
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(
                GROUNDING_DINO_COREML_STD,
                dtype=torch.float32,
            ).view(1, 3, 1, 1)
            * 255.0,
            persistent=False,
        )
        static_shapes = _grounding_dino_static_spatial_shapes(height, width)
        self.spatial_shapes_list = static_shapes
        spatial_shapes = torch.tensor(static_shapes, dtype=torch.long)
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(dim=1).cumsum(dim=0)[:-1],
            )
        )
        valid_ratios = torch.ones(
            (1, len(static_shapes), 2),
            dtype=torch.float32,
        )
        reference_points = []
        proposal_logits = []
        proposal_valid = []
        for level, (shape_height, shape_width) in enumerate(static_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(shape_height, dtype=torch.float32) + 0.5,
                torch.arange(shape_width, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            normalized = torch.stack(
                [
                    grid_x.reshape(-1) / shape_width,
                    grid_y.reshape(-1) / shape_height,
                ],
                dim=-1,
            )
            reference_points.append(normalized)
            width_height = torch.full_like(
                normalized,
                0.05 * (2.0**level),
            )
            proposal = torch.cat(
                [normalized, width_height],
                dim=-1,
            )
            valid = ((proposal > 0.01) & (proposal < 0.99)).all(
                dim=-1,
                keepdim=True,
            )
            proposal_valid.append(valid)
            proposal = torch.log(proposal / (1.0 - proposal))
            proposal_logits.append(proposal.masked_fill(~valid, float("inf")))
        reference_points_tensor = torch.cat(
            reference_points,
            dim=0,
        ).view(1, -1, 1, 2)
        reference_points_tensor = reference_points_tensor * valid_ratios[:, None]
        self.register_buffer(
            "spatial_shapes",
            spatial_shapes,
            persistent=False,
        )
        self.register_buffer(
            "level_start_index",
            level_start_index,
            persistent=False,
        )
        self.register_buffer(
            "valid_ratios",
            valid_ratios,
            persistent=False,
        )
        self.register_buffer(
            "encoder_reference_points",
            reference_points_tensor,
            persistent=False,
        )
        self.register_buffer(
            "proposal_logits",
            torch.cat(proposal_logits, dim=0).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "proposal_valid",
            torch.cat(proposal_valid, dim=0).unsqueeze(0),
            persistent=False,
        )

    def _contrastive(
        self,
        vision_hidden: torch.Tensor,
        text_hidden: torch.Tensor,
    ) -> torch.Tensor:
        output = vision_hidden @ text_hidden.transpose(-1, -2)
        output = output.masked_fill(
            ~self.text_token_mask[:, None, :],
            float("-inf"),
        )
        return output

    def _proposals(
        self,
        encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        object_query = encoded.masked_fill(~self.proposal_valid, 0.0)
        object_query = self.enc_output_norm(self.enc_output(object_query))
        return object_query, self.proposal_logits

    @staticmethod
    def _logit(value: torch.Tensor) -> torch.Tensor:
        """Conversion-safe, bit-identical ``torch.special.logit(eps=1e-5)``."""
        clamped = value.clamp(1e-5, 1.0 - 1e-5)
        return torch.log(clamped / (1.0 - clamped))

    def _decode(
        self,
        target: torch.Tensor,
        reference_points: torch.Tensor,
        vision_encoded: torch.Tensor,
        vision_mask: torch.Tensor,
        text_encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the native decoder with Core ML-lowerable logit arithmetic."""
        hidden_states = target
        dtype = text_encoded.dtype
        text_mask = (~self.text_token_mask)[:, None, None, :].repeat(
            1,
            self.decoder.config.decoder_attention_heads,
            self.num_queries,
            1,
        ).to(dtype) * torch.finfo(dtype).min
        intermediate = []
        intermediate_reference = []
        for index, layer in enumerate(self.decoder.layers):
            reference_input = (
                reference_points[:, :, None]
                * torch.cat(
                    [self.valid_ratios, self.valid_ratios],
                    dim=-1,
                )[:, None]
            )
            query_position = encode_sine_pos(
                reference_input[:, :, 0, :],
                num_pos_feats=self.decoder.config.d_model // 2,
            )
            query_position = self.decoder.reference_points_head(query_position)
            hidden_states = layer(
                hidden_states,
                query_position,
                reference_input,
                self.spatial_shapes,
                self.spatial_shapes_list,
                vision_encoded,
                vision_mask,
                text_encoded,
                text_mask,
            )
            delta = self.decoder.bbox_embed[index](hidden_states)
            reference_points = (
                (delta + self._logit(reference_points)).sigmoid().detach()
            )
            intermediate.append(self.decoder.layer_norm(hidden_states))
            intermediate_reference.append(reference_points)
        return (
            torch.stack(intermediate, dim=1),
            torch.stack(intermediate_reference, dim=1),
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one canonical RGB ``[0,1]`` image at the fixed canvas."""
        # Transformers fuses rescale and normalization as
        # (uint8 - mean*255) / (std*255). Multiplying canonical RGB[0,1]
        # back to 0-255 preserves that operation order bit-for-bit.
        pixel_values = (image * 255.0 - self.pixel_mean) / self.pixel_std
        pixel_mask = self.pixel_mask
        feature_maps = self.backbone_conv(pixel_values)
        masks = [
            F.interpolate(
                pixel_mask[None].float(),
                size=feature.shape[-2:],
            ).to(torch.bool)[0]
            for feature in feature_maps
        ]
        sources = [
            self.input_proj_vision[index](feature)
            for index, feature in enumerate(feature_maps)
        ]
        positions = [
            self.position_embedding(mask).to(feature.dtype)
            for mask, feature in zip(masks, sources)
        ]

        source = self.input_proj_vision[3](feature_maps[-1])
        mask = F.interpolate(
            pixel_mask[None].float(),
            size=source.shape[-2:],
        ).to(torch.bool)[0]
        position = self.position_embedding(mask).to(source.dtype)
        sources.append(source)
        masks.append(mask)
        positions.append(position)

        source_flatten = []
        mask_flatten = []
        level_positions = []
        for level, (source, mask, position) in enumerate(
            zip(sources, masks, positions)
        ):
            source_flatten.append(source.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            level_positions.append(
                position.flatten(2).transpose(1, 2)
                + self.level_embed[level].view(1, 1, -1)
            )
        source_flatten_tensor = torch.cat(source_flatten, dim=1)
        mask_flatten_tensor = torch.cat(mask_flatten, dim=1)
        level_positions_tensor = torch.cat(level_positions, dim=1)
        vision_encoded = source_flatten_tensor
        text_encoded = self.text_features
        for layer in self.encoder.layers:
            vision_encoded, text_encoded = layer(
                vision_encoded,
                text_encoded,
                level_positions_tensor,
                self.spatial_shapes,
                self.spatial_shapes_list,
                self.level_start_index,
                ~mask_flatten_tensor,
                self.encoder_reference_points,
                ~self.text_token_mask,
                ~self.text_self_attention_masks,
                self.position_ids,
            )
        object_query, output_proposals = self._proposals(vision_encoded)
        encoder_class = self._contrastive(
            object_query,
            text_encoded,
        )
        encoder_coordinates = (
            self.encoder_output_bbox_embed(object_query) + output_proposals
        )
        topk_indices = torch.topk(
            encoder_class.max(dim=-1)[0],
            self.num_queries,
            dim=1,
        )[1]
        topk_coordinates = torch.gather(
            encoder_coordinates,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4),
        ).detach()
        reference_points = topk_coordinates.sigmoid()
        initial_reference = reference_points
        target = self.query_position_embeddings.weight.unsqueeze(0)

        intermediate_hidden, intermediate_reference = self._decode(
            target,
            reference_points,
            vision_encoded,
            mask_flatten_tensor,
            text_encoded,
        )
        logits = None
        boxes = None
        for level in range(intermediate_hidden.shape[1]):
            reference = (
                initial_reference
                if level == 0
                else intermediate_reference[:, level - 1]
            )
            reference = self._logit(reference)
            logits = self._contrastive(
                intermediate_hidden[:, level],
                text_encoded,
            )
            delta = self.decoder.bbox_embed[level](intermediate_hidden[:, level])
            boxes = (delta + reference).sigmoid()
        if logits is None or boxes is None:  # pragma: no cover - config gate
            raise RuntimeError("Grounding DINO decoder returned no prediction stages.")
        return logits.float(), boxes.float()


def _grounding_dino_static_spatial_shapes(
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    """Derive Swin stage and extra-level shapes for one fixed canvas."""

    def ceil_div(value: int, divisor: int) -> int:
        return (value + divisor - 1) // divisor

    stage_height = ceil_div(height, 4)
    stage_width = ceil_div(width, 4)
    shapes = []
    for _ in range(3):
        stage_height = ceil_div(stage_height, 2)
        stage_width = ceil_div(stage_width, 2)
        shapes.append((stage_height, stage_width))
    shapes.append(
        (
            ceil_div(stage_height, 2),
            ceil_div(stage_width, 2),
        )
    )
    return shapes


def build_grounding_dino_frozen_coreml_adapter(
    model: nn.Module,
    processor: Any,
    *,
    size: str,
    names: Mapping[int, str] | Sequence[str],
) -> GroundingDinoFrozenCoreMLAdapter:
    """Build a CPU image-only graph from the loaded Transformers detector."""
    profile = validate_grounding_dino_coreml_profile(
        size=size,
        config=getattr(model, "config", None),
    )
    native = build_native_grounding_dino_from_hf(model)
    frozen = freeze_grounding_dino_text(model, processor, names)
    adapter = GroundingDinoFrozenCoreMLAdapter(
        native,
        frozen,
        canvas_hw=(profile.canvas, profile.canvas),
    )
    return adapter.eval()


def prepare_grounding_dino_coreml_export(
    model: Any,
    kwargs: Mapping[str, Any],
    *,
    default_output: str = "grounding_dino_coreml.mlpackage",
) -> tuple[
    GroundingDinoFrozenCoreMLAdapter,
    str,
    dict[str, Any],
    str,
    str,
]:
    """Validate a direct frozen-vocabulary export request."""
    from .exporter import CoreMLExporter

    options = dict(kwargs)
    imgsz = options.pop("imgsz", None)
    output_path = options.pop("output_path", None)
    output_alias = options.pop("output", None)
    if (
        output_path not in (None, "")
        and output_alias not in (None, "")
        and str(output_path) != str(output_alias)
    ):
        raise ValueError("Pass only one Core ML destination: output_path= or output=.")
    output_path = output_path or output_alias or default_output

    half = bool(options.pop("half", False))
    int8 = bool(options.pop("int8", False))
    data = options.pop("data", None)
    dynamic = bool(options.pop("dynamic", False))
    batch = int(options.pop("batch", 1))
    nms = bool(options.pop("nms", False))
    device = options.pop("device", None)
    compute_units = str(options.pop("compute_units", "all")).lower()
    conf = options.pop("conf", 0.25)
    iou = options.pop("iou", 0.45)
    max_det = options.pop("max_det", 300)

    for name in (
        "opset",
        "simplify",
        "verbose",
        "fraction",
        "allow_download_scripts",
        "_pre_trace_hook",
    ):
        options.pop(name, None)
    if options:
        names_text = ", ".join(sorted(options))
        raise TypeError(
            f"Unsupported Grounding DINO Core ML export options: {names_text}"
        )
    if dynamic:
        raise NotImplementedError(
            "Frozen-vocabulary Grounding DINO Core ML export uses fixed image "
            "and text shapes; dynamic=True is not supported."
        )
    if batch != 1:
        raise ValueError(
            "Frozen-vocabulary Grounding DINO Core ML export requires "
            f"batch=1; got batch={batch}."
        )
    if nms:
        raise NotImplementedError(
            "Grounding DINO does not run NMS. Core ML export preserves raw "
            "token logits and boxes; nms=True is not applicable."
        )
    if device not in (None, "auto", "cpu", torch.device("cpu")):
        raise NotImplementedError(
            "Core ML conversion traces on CPU; pass device='cpu', "
            "device='auto', or omit device."
        )

    size = str(getattr(model, "size", "")).strip().lower()
    profile = validate_grounding_dino_coreml_profile(
        size=size,
        config=getattr(getattr(model, "model", None), "config", None),
    )
    if imgsz is None:
        requested = (profile.canvas, profile.canvas)
    elif isinstance(imgsz, (tuple, list)):
        if len(imgsz) != 2:
            raise ValueError(f"imgsz must be an int or (height, width), got {imgsz}")
        requested = (int(imgsz[0]), int(imgsz[1]))
    else:
        requested = (int(imgsz), int(imgsz))
    validate_grounding_dino_coreml_profile(
        size=size,
        canvas_hw=requested,
    )

    labels = _ordered_names(getattr(model, "names", {}))
    if int(getattr(model, "nb_classes", 0)) != len(labels):
        raise RuntimeError(
            "Grounding DINO class metadata is inconsistent: nb_classes must "
            "match names."
        )
    exporter = CoreMLExporter(model)
    half, int8 = exporter._validate(half, int8, data)
    exporter._preflight(
        half=half,
        int8=int8,
        data=data,
        nms=False,
        compute_units=compute_units,
        conf=conf,
        iou=iou,
        max_det=max_det,
    )
    adapter = build_grounding_dino_frozen_coreml_adapter(
        model.model,
        model.processor,
        size=size,
        names=model.names,
    )
    precision = "fp16" if half else "fp32"
    metadata = exporter._build_metadata(
        precision,
        False,
        None,
        imgsz=requested,
    )
    metadata.update(
        grounding_dino_coreml_metadata(
            size=size,
            names=model.names,
            frozen=adapter.frozen_text_contract,
        )
    )
    destination = Path(output_path)
    if destination.suffix != ".mlpackage":
        destination = destination.with_suffix(".mlpackage")
    return (
        adapter,
        str(destination),
        metadata,
        precision,
        compute_units,
    )


def export_grounding_dino_coreml(
    model: Any,
    kwargs: Mapping[str, Any],
) -> str:
    """Export the current finite Grounding DINO class vocabulary."""
    adapter, output_path, metadata, precision, compute_units = (
        prepare_grounding_dino_coreml_export(model, kwargs)
    )
    height, width = adapter.canvas_hw
    dummy = torch.zeros(1, 3, height, width, dtype=torch.float32)
    from .coreml import export_coreml

    return export_coreml(
        adapter,
        dummy,
        output_path=output_path,
        precision=precision,
        compute_units=compute_units,
        nms=False,
        metadata=metadata,
        model_family="grounding_dino",
        model_task="detect",
        model_size=model.size,
    )


def preprocess_grounding_dino_coreml_image(
    image: Image.Image | np.ndarray,
    *,
    canvas_hw: int | Sequence[int] = GROUNDING_DINO_COREML_CANVAS,
) -> torch.Tensor:
    """Return the exact fixed-stretch canonical RGB ``[0,1]`` tensor."""
    if isinstance(canvas_hw, int):
        height = width = int(canvas_hw)
    else:
        values = tuple(int(value) for value in canvas_hw)
        if len(values) != 2:
            raise ValueError("Grounding DINO canvas must be an int or (height, width).")
        height, width = values
    if height <= 0 or width <= 0:
        raise ValueError("Grounding DINO canvas dimensions must be positive.")
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        array = np.array(rgb, dtype=np.uint8, copy=True)
    else:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError(
                "Grounding DINO Core ML input must be an HWC RGB/RGBA image."
            )
        if array.dtype != np.uint8:
            if not np.issubdtype(array.dtype, np.number):
                raise TypeError("Grounding DINO Core ML image values must be numeric.")
            if not bool(np.isfinite(array).all()):
                raise ValueError(
                    "Grounding DINO Core ML image contains NaN or infinity."
                )
            if float(array.min()) < 0 or float(array.max()) > 255:
                raise ValueError(
                    "Grounding DINO Core ML image values must be in [0,255]."
                )
            array = np.rint(array).astype(np.uint8)
        if array.shape[2] == 4:
            array = np.asarray(
                Image.fromarray(array, mode="RGBA").convert("RGB"),
                dtype=np.uint8,
            )
        else:
            array = np.array(array, dtype=np.uint8, copy=True)

    try:
        from torchvision.transforms.v2 import functional as tvf
        from torchvision.transforms.v2.functional import InterpolationMode
    except ImportError as exc:  # pragma: no cover - torch depends on it here
        raise ImportError(
            "Grounding DINO exact Core ML preprocessing requires torchvision."
        ) from exc
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    tensor = tvf.resize(
        tensor,
        [height, width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return tensor.unsqueeze(0).to(dtype=torch.float32).div(255.0)


def _label_tokens(text: str) -> list[str]:
    tokens = _LABEL_TOKEN_RE.findall(str(text).lower())
    while tokens and tokens[0] in _ARTICLES:
        tokens.pop(0)
    return tokens


def _contains_subsequence(
    haystack: list[str],
    needle: list[str],
) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    size = len(needle)
    return any(
        haystack[index : index + size] == needle
        for index in range(len(haystack) - size + 1)
    )


def _decode_wordpiece(pieces: Sequence[str]) -> str:
    """Reproduce the released BERT WordPiece decoder and cleanup."""
    text = " ".join(str(piece) for piece in pieces)
    text = text.replace(" ##", "").strip()
    return (
        text.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )


def _phrase_to_class_id(
    phrase: str,
    labels: Sequence[str],
) -> int | None:
    phrase_tokens = _label_tokens(phrase)
    if not phrase_tokens:
        return None
    normalized = " ".join(phrase_tokens)
    exact = {
        " ".join(_label_tokens(label)): index for index, label in enumerate(labels)
    }.get(normalized)
    if exact is not None:
        return exact
    matches = []
    for index, label in enumerate(labels):
        label_tokens = _label_tokens(label)
        if _contains_subsequence(
            phrase_tokens,
            label_tokens,
        ) or _contains_subsequence(label_tokens, phrase_tokens):
            matches.append(index)
    return matches[0] if len(matches) == 1 else None


def validate_grounding_dino_coreml_outputs(
    token_logits: Any,
    pred_boxes: Any,
    *,
    size: str,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the exact runtime tensor ABI before postprocessing."""
    logits = torch.as_tensor(token_logits, dtype=torch.float32)
    boxes = torch.as_tensor(pred_boxes, dtype=torch.float32)
    expected = expected_grounding_dino_coreml_shapes(
        size=size,
        sequence_length=sequence_length,
    )
    if tuple(logits.shape) != expected["token_logits"]:
        raise ValueError(
            "Grounding DINO Core ML token_logits shape mismatch: expected "
            f"{expected['token_logits']}, got {tuple(logits.shape)}."
        )
    if tuple(boxes.shape) != expected["pred_boxes"]:
        raise ValueError(
            "Grounding DINO Core ML pred_boxes shape mismatch: expected "
            f"{expected['pred_boxes']}, got {tuple(boxes.shape)}."
        )
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Grounding DINO Core ML token logits contain NaN or infinity.")
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("Grounding DINO Core ML boxes contain NaN or infinity.")
    return logits, boxes


def postprocess_grounding_dino_coreml_outputs(
    token_logits: Any,
    pred_boxes: Any,
    *,
    size: str,
    names: Mapping[int, str] | Sequence[str],
    text_contract: Mapping[str, Any],
    original_size: tuple[int, int],
    conf: float = 0.25,
    text_threshold: float = 0.25,
    max_det: int = 300,
    classes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Reproduce LibreGroundingDINO's grounded host postprocessing."""
    labels = _ordered_names(names)
    width, height = (int(value) for value in original_size)
    if width <= 0 or height <= 0:
        raise ValueError(
            "Grounding DINO original_size must be positive (width, height)."
        )
    for key, value in {
        "conf": conf,
        "text_threshold": text_threshold,
    }.items():
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"Grounding DINO {key} must be finite and in [0,1].")
    if isinstance(max_det, bool) or int(max_det) <= 0:
        raise ValueError("Grounding DINO max_det must be a positive integer.")

    token_pieces = text_contract.get("token_pieces")
    input_ids = text_contract.get("input_ids")
    if not isinstance(token_pieces, (list, tuple)) or not isinstance(
        input_ids,
        (list, tuple),
    ):
        raise ValueError(
            "Grounding DINO frozen text contract is missing token pieces or input ids."
        )
    if len(token_pieces) != len(input_ids):
        raise ValueError(
            "Grounding DINO frozen token pieces and ids have different lengths."
        )
    logits, boxes = validate_grounding_dino_coreml_outputs(
        token_logits,
        pred_boxes,
        size=size,
        sequence_length=len(input_ids),
    )
    probabilities = torch.sigmoid(logits[0])
    scores = probabilities.max(dim=-1)[0]
    keep = scores > float(conf)
    if not bool(keep.any()):
        return _empty_grounding_dino_detections()

    scores = scores[keep]
    boxes = boxes[0, keep]
    selected_probabilities = probabilities[keep, : len(input_ids)]
    class_ids = []
    keep_indices = []
    for index, probability in enumerate(selected_probabilities):
        positions = (probability > float(text_threshold)).nonzero(as_tuple=True)[0]
        positions = [
            int(position)
            for position in positions.tolist()
            if 0 < int(position) < len(input_ids) - 1
        ]
        phrase = _decode_wordpiece([token_pieces[position] for position in positions])
        class_id = _phrase_to_class_id(phrase, labels)
        if class_id is not None:
            keep_indices.append(index)
            class_ids.append(class_id)
    if not keep_indices:
        return _empty_grounding_dino_detections()

    indices = torch.as_tensor(keep_indices, dtype=torch.long)
    boxes = boxes[indices]
    scores = scores[indices]
    class_tensor = torch.as_tensor(class_ids, dtype=torch.int64)
    center_x, center_y, box_width, box_height = boxes.unbind(dim=-1)
    boxes = torch.stack(
        [
            center_x - 0.5 * box_width,
            center_y - 0.5 * box_height,
            center_x + 0.5 * box_width,
            center_y + 0.5 * box_height,
        ],
        dim=-1,
    )
    boxes = boxes * torch.tensor(
        [width, height, width, height],
        dtype=torch.float32,
    )
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, float(width))
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(height))
    valid = (
        torch.isfinite(boxes).all(dim=1)
        & torch.isfinite(scores)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    if classes is not None:
        if isinstance(classes, (str, bytes)):
            raise TypeError("Grounding DINO classes filter must be integer ids.")
        allowed_values = list(classes)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in allowed_values
        ):
            raise TypeError("Grounding DINO classes filter must contain integer ids.")
        allowed = torch.as_tensor(allowed_values, dtype=torch.int64)
        valid &= (class_tensor[:, None] == allowed[None, :]).any(dim=1)
    boxes = boxes[valid]
    scores = scores[valid]
    class_tensor = class_tensor[valid]
    if boxes.numel() == 0:
        return _empty_grounding_dino_detections()
    order = scores.argsort(descending=True)[: int(max_det)]
    boxes = boxes[order].cpu()
    scores = scores[order].cpu()
    class_tensor = class_tensor[order].cpu()
    return {
        "boxes": boxes,
        "scores": scores,
        "classes": class_tensor,
        "num_detections": int(boxes.shape[0]),
    }


def _empty_grounding_dino_detections() -> dict[str, Any]:
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "scores": torch.zeros((0,), dtype=torch.float32),
        "classes": torch.zeros((0,), dtype=torch.int64),
        "num_detections": 0,
    }


__all__ = [
    "GROUNDING_DINO_COREML_CANVAS",
    "GROUNDING_DINO_COREML_CONTRACT",
    "GROUNDING_DINO_COREML_INPUT_NAME",
    "GROUNDING_DINO_COREML_OUTPUT_NAMES",
    "GROUNDING_DINO_COREML_POSTPROCESS",
    "GROUNDING_DINO_COREML_PREPROCESS",
    "GROUNDING_DINO_COREML_PROFILES",
    "GroundingDinoCoreMLProfile",
    "GroundingDinoFrozenCoreMLAdapter",
    "GroundingDinoFrozenText",
    "build_grounding_dino_frozen_coreml_adapter",
    "build_native_grounding_dino_from_hf",
    "expected_grounding_dino_coreml_shapes",
    "export_grounding_dino_coreml",
    "freeze_grounding_dino_text",
    "frozen_grounding_dino_text_from_metadata",
    "grounding_dino_coreml_input_contract",
    "grounding_dino_coreml_metadata",
    "grounding_dino_coreml_output_contract",
    "grounding_dino_coreml_prompt",
    "grounding_dino_coreml_text_abi_hash",
    "grounding_dino_coreml_validation_contract",
    "grounding_dino_coreml_vocabulary_hash",
    "grounding_dino_hf_to_native_state_dict",
    "postprocess_grounding_dino_coreml_outputs",
    "prepare_grounding_dino_coreml_export",
    "preprocess_grounding_dino_coreml_image",
    "validate_grounding_dino_coreml_metadata",
    "validate_grounding_dino_coreml_outputs",
    "validate_grounding_dino_coreml_profile",
]
