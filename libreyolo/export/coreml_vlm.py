"""Stateful multifunction Core ML contracts for generative vision-language models.

This module is intentionally separate from :mod:`libreyolo.export.coreml`.
Generative VLM inference is a host-orchestrated tokenizer -> vision encoder ->
prefill/decode loop, not LibreYOLO's usual one-image/one-prediction graph.

The first deliberately narrow profile is SmolVLM2-500M:

* the exact Apache-2.0 checkpoint snapshot is pinned;
* the host processor is hash-pinned and never executes remote code;
* one fixed-stretch square image expands to 17 all-valid 512x512 crops;
* ``encode_image`` and ``embed_tokens`` are stateless functions;
* one stateful ``decode`` function handles both multi-token prefill and
  single-token decoding with an aggregated FP16 KV cache;
* all flexible axes have finite Core ML ``RangeDim`` upper bounds.

Conversion wrapper provenance
-----------------------------
The fixed-grid vision and Llama decoder equations compose modules from
``huggingface/transformers`` 5.12.1, source commit
``ddb849abe009d1089e6c691bfc897f27211c663c`` (Apache-2.0), specifically the
SmolVLM and Llama implementations.  The wrappers remove data-dependent image
filtering/mask interpolation for the exact all-valid square profile and replace
``DynamicCache`` with Apple's documented aggregate slice-update state layout.
No checkpoint code is executed (``trust_remote_code=False``).
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .coreml_profiles import (
    coreml_execution_profile_metadata,
    resolve_coreml_export_compute_units,
)


COREML_VLM_SCHEMA_VERSION = 2
COREML_VLM_ARTIFACT_SCOPE = "host_orchestrated_generative_vlm_multifunction"
COREML_VLM_MINIMUM_DEPLOYMENT_TARGETS = ("iOS18", "macOS15")
COREML_VLM_REQUIRED_COREMLTOOLS_MAJOR = 9
COREML_VLM_TRANSFORMERS_VERSION = "5.12.1"
COREML_VLM_TRANSFORMERS_COMMIT = "ddb849abe009d1089e6c691bfc897f27211c663c"
COREML_VLM_SOURCE_REL_TOL = 3e-4

COREML_VLM_ENCODE_IMAGE_FUNCTION = "encode_image"
COREML_VLM_EMBED_TOKENS_FUNCTION = "embed_tokens"
COREML_VLM_DECODE_FUNCTION = "decode"
COREML_VLM_FUNCTION_NAMES = (
    COREML_VLM_ENCODE_IMAGE_FUNCTION,
    COREML_VLM_EMBED_TOKENS_FUNCTION,
    COREML_VLM_DECODE_FUNCTION,
)

COREML_VLM_PIXEL_VALUES_INPUT = "pixel_values"
COREML_VLM_INPUT_IDS_INPUT = "input_ids"
COREML_VLM_TOKEN_EMBEDDINGS_INPUT = "token_embeddings"
COREML_VLM_CAUSAL_MASK_INPUT = "causal_mask"
COREML_VLM_POSITION_IDS_INPUT = "position_ids"
COREML_VLM_IMAGE_EMBEDDINGS_OUTPUT = "image_embeddings"
COREML_VLM_TOKEN_EMBEDDINGS_OUTPUT = "token_embeddings"
COREML_VLM_LAST_LOGITS_OUTPUT = "last_logits"
COREML_VLM_KEY_CACHE_STATE = "key_cache"
COREML_VLM_VALUE_CACHE_STATE = "value_cache"

SMOLVLM2_500M_COMPONENT_CONTRACT = (
    "smolvlm2_500m_fixed_square_fp32vision_fp16embed_fp32decode_state16_v2"
)
SMOLVLM2_500M_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SMOLVLM2_500M_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
SMOLVLM2_500M_CONTEXT_CHOICES = (2048, 4096, 8192)
SMOLVLM2_500M_DEFAULT_CONTEXT = 4096
SMOLVLM2_500M_MAX_NEW_TOKENS = 1024
SMOLVLM2_500M_SOURCE_IMAGE_SIZE = 2048
SMOLVLM2_500M_CONTEXT_MAX_NEW_TOKENS = {
    2048: 512,
    4096: SMOLVLM2_500M_MAX_NEW_TOKENS,
    8192: SMOLVLM2_500M_MAX_NEW_TOKENS,
}
SMOLVLM2_500M_IMAGE_TOKEN_ID = 49190
SMOLVLM2_500M_BOS_TOKEN_ID = 0
SMOLVLM2_500M_EOS_TOKEN_ID = 49279
SMOLVLM2_500M_PAD_TOKEN_ID = 2
SMOLVLM2_500M_REPETITION_PENALTY = 1.1
SMOLVLM2_500M_WEIGHTS_FILENAME = "model.safetensors"
SMOLVLM2_500M_WEIGHTS_SIZE = 2_029_990_624
SMOLVLM2_500M_WEIGHTS_SHA256 = (
    "b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3"
)

# Exact non-weight assets consumed by AutoProcessor / generation setup at the
# pinned snapshot. Extra files in a complete local snapshot are allowed, but
# every file below must exist and match byte-for-byte.
SMOLVLM2_500M_REQUIRED_ASSETS: dict[str, str] = {
    "added_tokens.json": "74135b8664b56088c0006f1c8e848d79a8eba003411f72ebf1dc2ee96227be3a",
    "chat_template.json": "b585e3598909a5687f9f9d738d35223724dedef256b9b274e1cbfb32b13c74bf",
    "config.json": "ea6bc1237e96247f6258de3e202e2e62b93d6f386dc47e7b36b5588bf3a15e17",
    "generation_config.json": "34835060c9f0f74d1acb456cc72ca32746d3843d9eb5f578f9cbffac1d2eb840",
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "preprocessor_config.json": "149e315d9410368e5491455bb06e0f763426e9e56cca731c13b24404a29b6374",
    "processor_config.json": "f3ad45028447b3562b4752be0d5916d6806c1ef589091a469608dcf0faa1737c",
    "special_tokens_map.json": "2dfea2a426162316ff1567c82bc6d36d9690cd9f90455f075c77daca78b45c60",
    "tokenizer.json": "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a",
    "tokenizer_config.json": "dd9ce2ab89a3dd881bd9378f1a79b943a064b9275a7e1706d5b7b47b68977913",
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
}

COREML_VLM_HOST_OPERATIONS = (
    "fixed_stretch_rgb_preprocess",
    "pinned_processor_tokenize",
    "image_placeholder_count_validation",
    "image_embedding_merge",
    "append_only_causal_control_construction",
    "fresh_decode_state_per_request",
    "greedy_generation_loop",
    "repetition_penalty",
    "eos_termination",
    "detokenize",
    "family_result_parse",
)


def _strict_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}.")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_axis(name: str, value: int) -> dict[str, Any]:
    return {"name": name, "kind": "fixed", "value": int(value)}


def _range_axis(
    name: str,
    lower_bound: int,
    upper_bound: int,
    default: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "range",
        "lower_bound": int(lower_bound),
        "upper_bound": int(upper_bound),
        "default": int(default),
    }


@dataclass(frozen=True)
class CoreMLVLMProfile:
    """Exact tensor/state dimensions for one multifunction VLM package."""

    family: str
    size: str
    context_length: int
    image_crops: int
    image_channels: int
    image_height: int
    image_width: int
    image_tokens_per_crop: int
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        for name in (
            "context_length",
            "image_crops",
            "image_channels",
            "image_height",
            "image_width",
            "image_tokens_per_crop",
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_new_tokens",
        ):
            _strict_positive_int(getattr(self, name), name=name)
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads."
            )
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim.")
        if self.image_token_count >= self.context_length:
            raise ValueError(
                "context_length must leave room beyond the fixed image tokens."
            )
        if self.max_new_tokens >= self.context_length:
            raise ValueError("max_new_tokens must be smaller than context_length.")
        if self.image_token_count + self.max_new_tokens >= self.context_length:
            raise ValueError(
                "context_length must leave room for text beyond the fixed "
                "image tokens and maximum generation budget."
            )

    @property
    def image_token_count(self) -> int:
        return self.image_crops * self.image_tokens_per_crop

    @property
    def cache_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.num_hidden_layers,
            1,
            self.num_key_value_heads,
            self.context_length,
            self.head_dim,
        )

    @property
    def cache_bytes_fp16(self) -> int:
        return 2 * math.prod(self.cache_shape) * 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "size": self.size,
            "context_length": self.context_length,
            "image_crops": self.image_crops,
            "image_channels": self.image_channels,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "image_tokens_per_crop": self.image_tokens_per_crop,
            "image_token_count": self.image_token_count,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_new_tokens": self.max_new_tokens,
            "cache_shape": list(self.cache_shape),
            "cache_bytes_fp16": self.cache_bytes_fp16,
        }


def smolvlm2_500m_coreml_profile(
    context_length: int = SMOLVLM2_500M_DEFAULT_CONTEXT,
) -> CoreMLVLMProfile:
    """Build the only currently specified bounded VLM graph profile."""

    context = _strict_positive_int(context_length, name="context_length")
    if context not in SMOLVLM2_500M_CONTEXT_CHOICES:
        raise ValueError(
            "SmolVLM2-500M Core ML context_length must be one of "
            f"{list(SMOLVLM2_500M_CONTEXT_CHOICES)}, got {context}."
        )
    return CoreMLVLMProfile(
        family="smolvlm2",
        size="500m",
        context_length=context,
        image_crops=17,
        image_channels=3,
        image_height=512,
        image_width=512,
        image_tokens_per_crop=64,
        hidden_size=960,
        vocab_size=49280,
        num_hidden_layers=32,
        num_attention_heads=15,
        num_key_value_heads=5,
        head_dim=64,
        max_new_tokens=SMOLVLM2_500M_CONTEXT_MAX_NEW_TOKENS[context],
    )


def resolve_smolvlm2_500m_coreml_export_compute_units(
    compute_units: Any,
) -> str:
    """Resolve this unvalidated VLM export without implying Apple evidence."""

    resolved, execution_profile = resolve_coreml_export_compute_units(
        compute_units,
        family="smolvlm2",
        task="detect",
        size="500m",
        canvas=SMOLVLM2_500M_SOURCE_IMAGE_SIZE,
        precision="fp32",
        nms=False,
    )
    if execution_profile is not None:
        raise RuntimeError(
            "SmolVLM2 Core ML is still experimental, but an exact hardware "
            "execution profile was unexpectedly registered. Update the "
            "specialized bundle contract before enabling validated routing."
        )
    return resolved


def validate_coreml_vlm_decode_bounds(
    profile: CoreMLVLMProfile,
    *,
    query_length: int,
    end_step: int,
) -> tuple[int, int]:
    """Validate the relationship Core ML RangeDim cannot express (Q <= E)."""

    query = _strict_positive_int(query_length, name="query_length")
    end = _strict_positive_int(end_step, name="end_step")
    if query > end:
        raise ValueError(
            f"query_length must not exceed end_step, got Q={query}, E={end}."
        )
    if end > profile.context_length:
        raise ValueError(
            f"end_step {end} exceeds Core ML context {profile.context_length}."
        )
    return query, end


def build_coreml_vlm_decode_controls(
    profile: CoreMLVLMProfile,
    *,
    query_length: int,
    end_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the only valid append-only FP16 mask and INT32 position IDs."""

    query, end = validate_coreml_vlm_decode_bounds(
        profile,
        query_length=query_length,
        end_step=end_step,
    )
    begin = end - query
    position_ids = np.arange(begin, end, dtype=np.int32)[None, :]
    columns = np.arange(end, dtype=np.int32)[None, :]
    allowed = columns <= position_ids.reshape(query, 1)
    causal_mask = np.where(
        allowed,
        np.float16(0),
        np.float16(np.finfo(np.float16).min),
    )[None, None, :, :]
    return np.ascontiguousarray(causal_mask), position_ids


def validate_coreml_vlm_decode_controls(
    profile: CoreMLVLMProfile,
    *,
    causal_mask: Any,
    position_ids: Any,
) -> tuple[int, int]:
    """Reject non-contiguous positions or a non-canonical additive mask."""

    actual_mask = np.asarray(causal_mask)
    actual_positions = np.asarray(position_ids)
    if actual_mask.dtype != np.float16:
        raise ValueError(
            f"Core ML VLM causal_mask must use float16, got {actual_mask.dtype}."
        )
    if actual_positions.dtype != np.int32:
        raise ValueError(
            f"Core ML VLM position_ids must use int32, got {actual_positions.dtype}."
        )
    if actual_mask.ndim != 4 or actual_mask.shape[:2] != (1, 1):
        raise ValueError(
            "Core ML VLM causal_mask must have shape [1, 1, Q, E], got "
            f"{actual_mask.shape}."
        )
    query = int(actual_mask.shape[2])
    end = int(actual_mask.shape[3])
    validate_coreml_vlm_decode_bounds(
        profile,
        query_length=query,
        end_step=end,
    )
    if actual_positions.shape != (1, query):
        raise ValueError(
            "Core ML VLM position_ids must have shape [1, Q], got "
            f"{actual_positions.shape}."
        )
    expected_mask, expected_positions = build_coreml_vlm_decode_controls(
        profile,
        query_length=query,
        end_step=end,
    )
    if not np.array_equal(actual_positions, expected_positions):
        raise ValueError(
            "Core ML VLM decode is append-only: position_ids must equal "
            "arange(E - Q, E)."
        )
    if not np.array_equal(actual_mask, expected_mask):
        raise ValueError(
            "Core ML VLM causal_mask must be the canonical append-only additive mask."
        )
    return query, end


class CoreMLVLMDecodeCursor:
    """Track the append position paired with one fresh Core ML decode state.

    ``Q <= E`` and canonical masks are necessary but not sufficient: the
    state itself has a cursor.  This host object rejects first-call skips,
    rewrites, and gaps before they can read zero or stale cache slots.
    Call :meth:`commit` only after the matching Core ML prediction succeeds;
    discard both this cursor and its paired ``MLState`` after any runtime
    failure.
    """

    def __init__(self, profile: CoreMLVLMProfile) -> None:
        self.profile = profile
        self._end_step = 0

    @property
    def end_step(self) -> int:
        return self._end_step

    def controls(
        self,
        *,
        query_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = _strict_positive_int(query_length, name="query_length")
        return build_coreml_vlm_decode_controls(
            self.profile,
            query_length=query,
            end_step=self._end_step + query,
        )

    def commit(
        self,
        *,
        causal_mask: Any,
        position_ids: Any,
    ) -> int:
        query, end = validate_coreml_vlm_decode_controls(
            self.profile,
            causal_mask=causal_mask,
            position_ids=position_ids,
        )
        begin = end - query
        if begin != self._end_step:
            raise ValueError(
                "Core ML VLM decode controls do not append to the paired "
                f"state cursor: expected begin={self._end_step}, got {begin}."
            )
        self._end_step = end
        return end


def merge_coreml_vlm_image_embeddings(
    profile: CoreMLVLMProfile,
    *,
    input_ids: Any,
    token_embeddings: Any,
    image_embeddings: Any,
    image_token_id: int = SMOLVLM2_500M_IMAGE_TOKEN_ID,
) -> np.ndarray:
    """Replace exactly the fixed image placeholders in host-side FP16 data."""

    ids = np.asarray(input_ids)
    tokens = np.asarray(token_embeddings)
    images = np.asarray(image_embeddings)
    if ids.dtype != np.int32:
        raise ValueError(f"Core ML VLM input_ids must use int32, got {ids.dtype}.")
    if tokens.dtype != np.float16 or images.dtype != np.float16:
        raise ValueError(
            "Core ML VLM token and image embeddings must both use float16."
        )
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(
            f"Core ML VLM input_ids must have shape [1, Q], got {ids.shape}."
        )
    expected_tokens = (1, int(ids.shape[1]), profile.hidden_size)
    if tokens.shape != expected_tokens:
        raise ValueError(
            "Core ML VLM token embeddings have the wrong shape: expected "
            f"{expected_tokens}, got {tokens.shape}."
        )
    expected_images = (1, profile.image_token_count, profile.hidden_size)
    if images.shape != expected_images:
        raise ValueError(
            "Core ML VLM image embeddings have the wrong shape: expected "
            f"{expected_images}, got {images.shape}."
        )
    placeholders = ids == int(image_token_id)
    count = int(np.count_nonzero(placeholders))
    if count != profile.image_token_count:
        raise ValueError(
            "Core ML VLM prompt has the wrong image-placeholder count: "
            f"expected {profile.image_token_count}, got {count}."
        )
    merged = np.array(tokens, copy=True, order="C")
    merged[placeholders] = images.reshape(
        profile.image_token_count,
        profile.hidden_size,
    )
    return merged


def validate_coreml_vlm_context_budget(
    profile: CoreMLVLMProfile,
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    image_tokens: int,
) -> tuple[int, int]:
    """Fail before runtime when processor output cannot fit the package state."""

    prompt = _strict_positive_int(prompt_tokens, name="prompt_tokens")
    generated = _strict_positive_int(max_new_tokens, name="max_new_tokens")
    images = _strict_positive_int(image_tokens, name="image_tokens")
    if images != profile.image_token_count:
        raise ValueError(
            "SmolVLM2 fixed-square processor/image ABI mismatch: expected "
            f"{profile.image_token_count} image placeholders, got {images}."
        )
    if generated > profile.max_new_tokens:
        raise ValueError(
            f"max_new_tokens {generated} exceeds artifact maximum "
            f"{profile.max_new_tokens}."
        )
    if prompt + generated > profile.context_length:
        raise ValueError(
            "Prompt plus generation budget exceeds Core ML KV state: "
            f"{prompt} + {generated} > {profile.context_length}."
        )
    return prompt, generated


def preprocess_smolvlm2_500m_coreml_image(
    image: Image.Image | np.ndarray,
    *,
    image_size: int = SMOLVLM2_500M_SOURCE_IMAGE_SIZE,
) -> Image.Image:
    """Return the canonical 2048-square RGB image passed to the processor.

    The pinned processor splits this square into sixteen 512-square local
    crops plus one global crop. All other source sizes are bilinearly
    stretched, so this experimental deployment profile intentionally differs
    from native aspect-preserving preprocessing and may affect accuracy.
    """

    size = _strict_positive_int(image_size, name="image_size")
    if size != SMOLVLM2_500M_SOURCE_IMAGE_SIZE:
        raise ValueError(
            "SmolVLM2-500M Core ML raw-image preprocessing is fixed at "
            f"{SMOLVLM2_500M_SOURCE_IMAGE_SIZE}x"
            f"{SMOLVLM2_500M_SOURCE_IMAGE_SIZE}, got {size}."
        )
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    else:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError(
                "SmolVLM2 Core ML preprocessing requires an HWC RGB/RGBA "
                "image."
            )
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(
                "SmolVLM2 Core ML preprocessing requires a non-empty image."
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("SmolVLM2 Core ML image values must be numeric.")
        if not bool(np.isfinite(array).all()):
            raise ValueError(
                "SmolVLM2 Core ML image contains NaN or infinity."
            )
        array = array[..., :3]
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < 0:
            raise ValueError(
                "SmolVLM2 Core ML image values must be non-negative."
            )
        if np.issubdtype(array.dtype, np.floating):
            if maximum <= 1.0:
                array = array * 255.0
                maximum *= 255.0
        if maximum > 255.0:
            raise ValueError(
                "SmolVLM2 Core ML image values must be in [0, 1] or "
                "[0, 255]."
            )
        array = array.astype(np.uint8)
        rgb = Image.fromarray(array, mode="RGB")
    if rgb.size != (size, size):
        rgb = rgb.resize(
            (size, size),
            resample=Image.Resampling.BILINEAR,
        )
    return rgb


def prepare_smolvlm2_500m_coreml_processor_batch(
    profile: CoreMLVLMProfile,
    processor_batch: Mapping[str, Any],
    *,
    max_new_tokens: int | None = None,
) -> dict[str, np.ndarray]:
    """Validate a pinned square processor result and cast Core ML host inputs."""

    _validate_exact_smolvlm2_profile(profile)
    if not isinstance(processor_batch, Mapping):
        raise TypeError("SmolVLM2 processor output must be a mapping.")
    required = (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "pixel_attention_mask",
    )
    missing = [name for name in required if name not in processor_batch]
    if missing:
        raise ValueError(
            f"SmolVLM2 processor output is missing required keys {missing}."
        )

    def host_array(name: str) -> np.ndarray:
        value = processor_batch[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    input_ids = host_array("input_ids")
    attention_mask = host_array("attention_mask")
    pixel_values = host_array("pixel_values")
    pixel_attention_mask = host_array("pixel_attention_mask")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            "SmolVLM2 processor input_ids must have shape [1, Q], got "
            f"{input_ids.shape}."
        )
    if not np.issubdtype(input_ids.dtype, np.integer):
        raise ValueError("SmolVLM2 processor input_ids must be integral.")
    if attention_mask.shape != input_ids.shape or not np.all(attention_mask == 1):
        raise ValueError(
            "SmolVLM2 fixed prompt requires an all-valid attention_mask "
            "matching input_ids."
        )
    expected_pixels = (
        1,
        profile.image_crops,
        profile.image_channels,
        profile.image_height,
        profile.image_width,
    )
    if pixel_values.shape != expected_pixels:
        raise ValueError(
            "SmolVLM2 processor did not produce the fixed-square crop ABI: "
            f"expected {expected_pixels}, got {pixel_values.shape}."
        )
    if not np.issubdtype(pixel_values.dtype, np.floating):
        raise ValueError("SmolVLM2 processor pixel_values must be floating point.")
    expected_pixel_mask = (
        1,
        profile.image_crops,
        profile.image_height,
        profile.image_width,
    )
    if pixel_attention_mask.shape != expected_pixel_mask or not np.all(
        pixel_attention_mask
    ):
        raise ValueError(
            "SmolVLM2 fixed-square pixel_attention_mask must be all-valid."
        )
    placeholders = int(np.count_nonzero(input_ids == SMOLVLM2_500M_IMAGE_TOKEN_ID))
    budget = profile.max_new_tokens if max_new_tokens is None else max_new_tokens
    validate_coreml_vlm_context_budget(
        profile,
        prompt_tokens=int(input_ids.shape[1]),
        max_new_tokens=budget,
        image_tokens=placeholders,
    )
    if np.any(input_ids < 0) or np.any(input_ids >= profile.vocab_size):
        raise ValueError("SmolVLM2 processor emitted an out-of-vocabulary token.")
    return {
        COREML_VLM_INPUT_IDS_INPUT: np.ascontiguousarray(
            input_ids.astype(np.int32, copy=False)
        ),
        COREML_VLM_PIXEL_VALUES_INPUT: np.ascontiguousarray(
            pixel_values.astype(np.float16, copy=False)
        ),
    }


def validate_smolvlm2_500m_processor_assets(
    processor_dir: str | os.PathLike[str],
    *,
    revision: str,
    transformers_version: str = COREML_VLM_TRANSFORMERS_VERSION,
) -> dict[str, Any]:
    """Hash and validate the exact processor snapshot without loading code."""

    if revision != SMOLVLM2_500M_REVISION:
        raise ValueError(
            "SmolVLM2-500M Core ML export requires processor revision "
            f"{SMOLVLM2_500M_REVISION}, got {revision!r}."
        )
    if transformers_version != COREML_VLM_TRANSFORMERS_VERSION:
        raise ValueError(
            "SmolVLM2-500M Core ML processor semantics are pinned to "
            f"transformers {COREML_VLM_TRANSFORMERS_VERSION}, got "
            f"{transformers_version!r}."
        )
    root = Path(processor_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"SmolVLM2-500M processor directory does not exist: {root}."
        )
    for relative_name, expected_hash in SMOLVLM2_500M_REQUIRED_ASSETS.items():
        path = root / relative_name
        if not path.is_file():
            raise FileNotFoundError(
                "SmolVLM2-500M processor snapshot is incomplete; missing "
                f"{relative_name!r}."
            )
        actual_hash = _file_sha256(path)
        if not hmac.compare_digest(actual_hash, expected_hash):
            raise ValueError(
                "SmolVLM2-500M processor asset failed SHA-256 validation: "
                f"{relative_name!r}, expected {expected_hash}, got {actual_hash}."
            )
    return smolvlm2_500m_processor_manifest()


def smolvlm2_500m_processor_manifest() -> dict[str, Any]:
    """Return the immutable processor/tokenizer ABI recorded in metadata."""

    return {
        "repo": SMOLVLM2_500M_REPO,
        "revision": SMOLVLM2_500M_REVISION,
        "trust_remote_code": False,
        "transformers_version": COREML_VLM_TRANSFORMERS_VERSION,
        "required_assets": dict(SMOLVLM2_500M_REQUIRED_ASSETS),
        "chat_template_sha256": SMOLVLM2_500M_REQUIRED_ASSETS["chat_template.json"],
        "generation_config_sha256": SMOLVLM2_500M_REQUIRED_ASSETS[
            "generation_config.json"
        ],
    }


def validate_smolvlm2_500m_weight_asset(
    checkpoint_dir: str | os.PathLike[str],
    *,
    revision: str,
) -> dict[str, Any]:
    """Validate the exact official learned-tensor payload byte-for-byte."""

    if revision != SMOLVLM2_500M_REVISION:
        raise ValueError(
            "SmolVLM2-500M Core ML export requires weight revision "
            f"{SMOLVLM2_500M_REVISION}, got {revision!r}."
        )
    root = Path(checkpoint_dir)
    path = root / SMOLVLM2_500M_WEIGHTS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            "SmolVLM2-500M checkpoint snapshot is incomplete; missing "
            f"{SMOLVLM2_500M_WEIGHTS_FILENAME!r}."
        )
    actual_size = path.stat().st_size
    if actual_size != SMOLVLM2_500M_WEIGHTS_SIZE:
        raise ValueError(
            "SmolVLM2-500M weight asset has the wrong byte length: "
            f"expected {SMOLVLM2_500M_WEIGHTS_SIZE}, got {actual_size}."
        )
    actual_hash = _file_sha256(path)
    if not hmac.compare_digest(actual_hash, SMOLVLM2_500M_WEIGHTS_SHA256):
        raise ValueError(
            "SmolVLM2-500M weight asset failed SHA-256 validation: "
            f"expected {SMOLVLM2_500M_WEIGHTS_SHA256}, got {actual_hash}."
        )
    return smolvlm2_500m_weights_manifest()


def smolvlm2_500m_weights_manifest() -> dict[str, Any]:
    """Return exact source and integrity data for the supported checkpoint."""

    return {
        "repo": SMOLVLM2_500M_REPO,
        "revision": SMOLVLM2_500M_REVISION,
        "filename": SMOLVLM2_500M_WEIGHTS_FILENAME,
        "size_bytes": SMOLVLM2_500M_WEIGHTS_SIZE,
        "sha256": SMOLVLM2_500M_WEIGHTS_SHA256,
        "license": "Apache-2.0",
    }


def validate_smolvlm2_500m_model_weight_values(
    model: nn.Module,
    checkpoint_dir: str | os.PathLike[str],
) -> None:
    """Prove every in-memory tensor equals the hash-pinned safetensors file."""

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "SmolVLM2 Core ML weight validation requires safetensors."
        ) from exc
    path = Path(checkpoint_dir) / SMOLVLM2_500M_WEIGHTS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"SmolVLM2 weight payload does not exist: {path}.")
    actual_state = model.state_dict()
    with safe_open(path, framework="pt", device="cpu") as source:
        source_keys = set(source.keys())
        actual_keys = set(actual_state)
        if actual_keys != source_keys:
            missing = sorted(source_keys - actual_keys)
            unexpected = sorted(actual_keys - source_keys)
            raise ValueError(
                "In-memory SmolVLM2 state keys differ from the pinned payload: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}."
            )
        for name in sorted(source_keys):
            expected = source.get_tensor(name)
            actual = actual_state[name].detach()
            if actual.device.type != "cpu":
                raise ValueError(
                    "SmolVLM2 weight comparison requires a CPU model, got "
                    f"{name!r} on {actual.device}."
                )
            if (
                actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or not torch.equal(actual, expected)
            ):
                raise ValueError(
                    "In-memory SmolVLM2 tensor differs from the hash-pinned "
                    f"checkpoint: {name!r}."
                )


def coreml_vlm_function_contracts(
    profile: CoreMLVLMProfile,
) -> dict[str, dict[str, Any]]:
    """Return exact named function IO including finite shared Q/E bounds."""

    q_axis = _range_axis("Q", 1, profile.context_length, 1)
    e_axis = _range_axis("E", 1, profile.context_length, 1)
    return {
        COREML_VLM_ENCODE_IMAGE_FUNCTION: {
            "stateful": False,
            "inputs": [
                {
                    "name": COREML_VLM_PIXEL_VALUES_INPUT,
                    "dtype": "float16",
                    "shape": [
                        _fixed_axis("N", 1),
                        _fixed_axis("I", profile.image_crops),
                        _fixed_axis("C", profile.image_channels),
                        _fixed_axis("H", profile.image_height),
                        _fixed_axis("W", profile.image_width),
                    ],
                }
            ],
            "outputs": [
                {
                    "name": COREML_VLM_IMAGE_EMBEDDINGS_OUTPUT,
                    "dtype": "float16",
                    "shape": [
                        _fixed_axis("N", 1),
                        _fixed_axis("T_image", profile.image_token_count),
                        _fixed_axis("D_model", profile.hidden_size),
                    ],
                }
            ],
            "capture": "torch_jit_trace_fixed",
        },
        COREML_VLM_EMBED_TOKENS_FUNCTION: {
            "stateful": False,
            "inputs": [
                {
                    "name": COREML_VLM_INPUT_IDS_INPUT,
                    "dtype": "int32",
                    "shape": [_fixed_axis("N", 1), dict(q_axis)],
                }
            ],
            "outputs": [
                {
                    "name": COREML_VLM_TOKEN_EMBEDDINGS_OUTPUT,
                    "dtype": "float16",
                    "shape": [
                        _fixed_axis("N", 1),
                        dict(q_axis),
                        _fixed_axis("D_model", profile.hidden_size),
                    ],
                }
            ],
            "capture": "torch_jit_trace_bounded_q",
        },
        COREML_VLM_DECODE_FUNCTION: {
            "stateful": True,
            "inputs": [
                {
                    "name": COREML_VLM_TOKEN_EMBEDDINGS_INPUT,
                    "dtype": "float16",
                    "shape": [
                        _fixed_axis("N", 1),
                        dict(q_axis),
                        _fixed_axis("D_model", profile.hidden_size),
                    ],
                },
                {
                    "name": COREML_VLM_CAUSAL_MASK_INPUT,
                    "dtype": "float16",
                    "shape": [
                        _fixed_axis("N", 1),
                        _fixed_axis("heads", 1),
                        dict(q_axis),
                        dict(e_axis),
                    ],
                    "semantic_constraint": (
                        "canonical append-only additive mask; Q <= E"
                    ),
                },
                {
                    "name": COREML_VLM_POSITION_IDS_INPUT,
                    "dtype": "int32",
                    "shape": [_fixed_axis("N", 1), dict(q_axis)],
                    "semantic_constraint": "position_ids == arange(E - Q, E)",
                },
            ],
            "outputs": [
                {
                    "name": COREML_VLM_LAST_LOGITS_OUTPUT,
                    "dtype": "float16",
                    "shape": [
                        _fixed_axis("N", 1),
                        _fixed_axis("V", profile.vocab_size),
                    ],
                }
            ],
            "states": [
                {
                    "name": COREML_VLM_KEY_CACHE_STATE,
                    "dtype": "float16",
                    "shape": list(profile.cache_shape),
                },
                {
                    "name": COREML_VLM_VALUE_CACHE_STATE,
                    "dtype": "float16",
                    "shape": list(profile.cache_shape),
                },
            ],
            "capture": "torch_jit_trace_stateful_bounded_q_e",
        },
    }


def smolvlm2_500m_coreml_metadata(
    profile: CoreMLVLMProfile | None = None,
) -> dict[str, Any]:
    """Build the integrity-checked artifact/host contract."""

    resolved = profile or smolvlm2_500m_coreml_profile()
    _validate_exact_smolvlm2_profile(resolved)
    functions = coreml_vlm_function_contracts(resolved)
    processor = smolvlm2_500m_processor_manifest()
    weights = smolvlm2_500m_weights_manifest()
    generation = {
        "mode": "greedy",
        "do_sample": False,
        "repetition_penalty": SMOLVLM2_500M_REPETITION_PENALTY,
        "max_new_tokens": resolved.max_new_tokens,
        "bos_token_id": SMOLVLM2_500M_BOS_TOKEN_ID,
        "eos_token_ids": [SMOLVLM2_500M_EOS_TOKEN_ID],
        "pad_token_id": SMOLVLM2_500M_PAD_TOKEN_ID,
    }
    image_profile = {
        "mode": "fixed_stretch_square_all_valid_crops",
        "color": "rgb",
        "input_range": "uint8_0_255",
        "geometry": "stretch",
        "interpolation": "bilinear",
        "resize_backend": "pillow",
        "source_size": [
            SMOLVLM2_500M_SOURCE_IMAGE_SIZE,
            SMOLVLM2_500M_SOURCE_IMAGE_SIZE,
        ],
        "crop_size": [resolved.image_height, resolved.image_width],
        "processed_shape": [
            1,
            resolved.image_crops,
            resolved.image_channels,
            resolved.image_height,
            resolved.image_width,
        ],
        "image_token_id": SMOLVLM2_500M_IMAGE_TOKEN_ID,
        "image_token_count": resolved.image_token_count,
        "dynamic_crop_count": False,
        "pixel_attention_mask": "implicit_all_ones",
    }
    execution_profile = coreml_execution_profile_metadata(
        None,
        conversion_compute_units="cpu_only",
    )
    integrity_surface = {
        "profile": resolved.as_dict(),
        "functions": functions,
        "processor": processor,
        "weights": weights,
        "generation": generation,
        "image_profile": image_profile,
        "host_operations": list(COREML_VLM_HOST_OPERATIONS),
        "execution_profile": execution_profile,
    }
    return {
        "artifact_scope": COREML_VLM_ARTIFACT_SCOPE,
        "component_contract": SMOLVLM2_500M_COMPONENT_CONTRACT,
        "coreml_vlm_schema_version": COREML_VLM_SCHEMA_VERSION,
        "coreml_multifunction": True,
        "coreml_default_function": COREML_VLM_ENCODE_IMAGE_FUNCTION,
        "coreml_function_names": list(COREML_VLM_FUNCTION_NAMES),
        "coreml_stateful_functions": [COREML_VLM_DECODE_FUNCTION],
        "coreml_minimum_deployment_targets": list(
            COREML_VLM_MINIMUM_DEPLOYMENT_TARGETS
        ),
        "coremltools_major": COREML_VLM_REQUIRED_COREMLTOOLS_MAJOR,
        "model_family": "smolvlm2",
        "size": "500m",
        "task": "detect",
        "precision": "mixed",
        "vision_compute_precision": "fp32",
        "token_embedding_compute_precision": "fp16",
        "decoder_compute_precision": "fp32",
        "function_io_precision": "fp16",
        "state_precision": "fp16",
        "conversion_source_precision": "fp32",
        "batch": 1,
        "dynamic": True,
        "weights_license": "apache-2.0",
        "artifact_redistributable": True,
        **execution_profile,
        "vlm_profile": resolved.as_dict(),
        "vlm_functions": functions,
        "processor": processor,
        "weights": weights,
        "generation": generation,
        "image_profile": image_profile,
        "host_operations": list(COREML_VLM_HOST_OPERATIONS),
        "transformers_source": {
            "repo": "https://github.com/huggingface/transformers",
            "commit": COREML_VLM_TRANSFORMERS_COMMIT,
            "license": "Apache-2.0",
        },
        "coreml_vlm_contract_sha256": _canonical_sha256(integrity_surface),
    }


def _validate_exact_smolvlm2_profile(profile: CoreMLVLMProfile) -> None:
    expected = smolvlm2_500m_coreml_profile(profile.context_length)
    if profile != expected:
        raise ValueError("Profile conflicts with the exact SmolVLM2-500M Core ML ABI.")


def _decode_metadata_value(value: Any, expected: Any, *, name: str) -> Any:
    if not isinstance(value, str):
        return value
    if isinstance(expected, (dict, list, bool)) or expected is None:
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} must contain valid JSON.") from exc
    if isinstance(expected, int):
        try:
            return int(value)
        except ValueError:
            return value
    if isinstance(expected, float):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def validate_coreml_vlm_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate native or Core-ML-stringified SmolVLM metadata fail-closed."""

    if not isinstance(metadata, Mapping):
        raise ValueError("Core ML VLM metadata must be a mapping.")
    expected_keys = set(
        smolvlm2_500m_coreml_metadata(
            smolvlm2_500m_coreml_profile()
        )
    )
    if set(metadata) != expected_keys:
        missing = sorted(expected_keys - set(metadata))
        extra = sorted(set(metadata) - expected_keys)
        raise ValueError(
            "Core ML VLM metadata keys changed: "
            f"missing={missing}, extra={extra}."
        )
    raw_profile = metadata.get("vlm_profile")
    if isinstance(raw_profile, str):
        try:
            raw_profile = json.loads(raw_profile)
        except json.JSONDecodeError as exc:
            raise ValueError("vlm_profile must contain valid JSON.") from exc
    if not isinstance(raw_profile, dict):
        raise ValueError("vlm_profile must be a mapping.")
    try:
        profile = smolvlm2_500m_coreml_profile(
            context_length=raw_profile["context_length"]
        )
    except KeyError as exc:
        raise ValueError(
            f"vlm_profile is missing required field {exc.args[0]!r}."
        ) from exc
    expected = smolvlm2_500m_coreml_metadata(profile)
    for name, expected_value in expected.items():
        if name not in metadata:
            raise ValueError(f"Core ML VLM metadata is missing {name!r}.")
        actual = _decode_metadata_value(metadata[name], expected_value, name=name)
        if actual != expected_value:
            raise ValueError(f"{name} conflicts with the strict Core ML VLM contract.")
    return expected


def stringify_coreml_vlm_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Convert validated metadata to Core ML's string-only user dictionary."""

    validate_coreml_vlm_metadata(metadata)
    result: dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list, bool)) or value is None:
            result[str(key)] = _canonical_json(value)
        else:
            result[str(key)] = str(value)
    return result


def _nested_attr(value: Any, path: str) -> Any:
    current = value
    for name in path.split("."):
        current = getattr(current, name)
    return current


def validate_smolvlm2_500m_model(model: nn.Module) -> None:
    """Reject lookalike configs before wrapping or converting weights."""

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("SmolVLM2 Core ML export requires model.config.")
    expected = {
        "model_type": "smolvlm",
        "image_token_id": SMOLVLM2_500M_IMAGE_TOKEN_ID,
        "scale_factor": 4,
        "text_config.hidden_size": 960,
        "text_config.intermediate_size": 2560,
        "text_config.num_hidden_layers": 32,
        "text_config.num_attention_heads": 15,
        "text_config.num_key_value_heads": 5,
        "text_config.head_dim": 64,
        "text_config.vocab_size": 49280,
        "text_config.max_position_embeddings": 8192,
        "text_config.hidden_act": "silu",
        "text_config.rms_norm_eps": 1e-5,
        "text_config.rope_parameters": {
            "rope_theta": 100000.0,
            "rope_type": "default",
        },
        "text_config.attention_bias": False,
        "text_config.mlp_bias": False,
        "vision_config.hidden_size": 768,
        "vision_config.image_size": 512,
        "vision_config.patch_size": 16,
        "vision_config.num_hidden_layers": 12,
        "vision_config.num_attention_heads": 12,
        "vision_config.hidden_act": "gelu_pytorch_tanh",
        "vision_config.layer_norm_eps": 1e-6,
        "vision_config.num_channels": 3,
    }
    for path, wanted in expected.items():
        try:
            actual = _nested_attr(config, path)
        except AttributeError as exc:
            raise ValueError(
                f"SmolVLM2-500M model config is missing {path!r}."
            ) from exc
        if actual != wanted:
            raise ValueError(
                "Model does not match the exact SmolVLM2-500M Core ML "
                f"profile: {path}={actual!r}, expected {wanted!r}."
            )
    base = getattr(model, "model", None)
    if (
        base is None
        or not hasattr(base, "vision_model")
        or not hasattr(base, "connector")
        or not hasattr(base, "text_model")
        or not hasattr(model, "lm_head")
    ):
        raise ValueError(
            "SmolVLM2-500M model is missing native vision/text/LM components."
        )


class SmolVLM2FixedSquareVisionEncoder(nn.Module):
    """Conversion-only all-valid fixed-grid SmolVLM vision encoder."""

    def __init__(self, model: nn.Module, *, image_crops: int = 17):
        super().__init__()
        base = getattr(model, "model", model)
        vision_model = base.vision_model
        connector = base.connector
        vision_config = vision_model.config
        self.image_crops = _strict_positive_int(image_crops, name="image_crops")
        self.image_size = int(vision_config.image_size)
        self.patch_size = int(vision_config.patch_size)
        self.scale_factor = int(connector.scale_factor)
        self.hidden_size = int(connector.modality_projection.proj.out_features)
        if self.image_size % self.patch_size:
            raise ValueError("SmolVLM fixed image size must divide by patch size.")
        patch_side = self.image_size // self.patch_size
        if patch_side % self.scale_factor:
            raise ValueError(
                "SmolVLM patch grid must divide by connector scale factor."
            )
        self.tokens_per_crop = (patch_side // self.scale_factor) ** 2
        self.patch_embedding = vision_model.embeddings.patch_embedding
        self.position_embedding = vision_model.embeddings.position_embedding
        self.encoder = vision_model.encoder
        self.post_layernorm = vision_model.post_layernorm
        self.connector = connector
        self.register_buffer(
            "fixed_position_ids",
            torch.arange(patch_side * patch_side, dtype=torch.long),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixels = pixel_values.reshape(
            self.image_crops,
            3,
            self.image_size,
            self.image_size,
        )
        patch_embeddings = self.patch_embedding(pixels)
        hidden_states = patch_embeddings.flatten(2).transpose(1, 2)
        positions = self.position_embedding(self.fixed_position_ids).unsqueeze(0)
        hidden_states = hidden_states + positions
        encoded = self.encoder(inputs_embeds=hidden_states, attention_mask=None)
        hidden_states = encoded.last_hidden_state
        hidden_states = self.post_layernorm(hidden_states)
        image_embeddings = self.connector(hidden_states)
        return image_embeddings.reshape(
            1,
            self.image_crops * self.tokens_per_crop,
            self.hidden_size,
        )


class CoreMLVLMTokenEmbedding(nn.Module):
    """Named stateless token-embedding component."""

    def __init__(self, embedding: nn.Module):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first = value[..., : value.shape[-1] // 2]
    second = value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


class CoreMLVLMStatefulLlamaDecoder(nn.Module):
    """Llama decoder with aggregate slice-update K/V buffers as Core ML state."""

    def __init__(
        self,
        *,
        text_model: nn.Module,
        lm_head: nn.Module,
        context_length: int,
        state_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        config = text_model.config
        self.layers = text_model.layers
        self.final_norm = text_model.norm
        self.lm_head = lm_head
        self.context_length = _strict_positive_int(
            context_length, name="context_length"
        )
        self.hidden_size = int(config.hidden_size)
        self.num_hidden_layers = int(config.num_hidden_layers)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(
                config,
                "head_dim",
                self.hidden_size // self.num_attention_heads,
            )
        )
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("Unsupported Llama hidden/head layout.")
        parameter = next(text_model.parameters())
        dtype = state_dtype or parameter.dtype
        if dtype not in (torch.float16, torch.float32):
            raise ValueError(
                "Core ML VLM decoder state preparation supports float16 or "
                f"float32 eager probes, got {dtype}."
            )
        cache_shape = (
            self.num_hidden_layers,
            1,
            self.num_key_value_heads,
            self.context_length,
            self.head_dim,
        )
        self.register_buffer(
            COREML_VLM_KEY_CACHE_STATE,
            torch.zeros(cache_shape, dtype=dtype),
        )
        self.register_buffer(
            COREML_VLM_VALUE_CACHE_STATE,
            torch.zeros(cache_shape, dtype=dtype),
        )
        positions = torch.arange(self.context_length, dtype=torch.long).unsqueeze(0)
        probe = torch.zeros(
            (1, self.context_length, self.hidden_size),
            dtype=dtype,
            device=parameter.device,
        )
        positions = positions.to(parameter.device)
        with torch.no_grad():
            rope_cos, rope_sin = text_model.rotary_emb(
                probe,
                position_ids=positions,
            )
        self.register_buffer(
            "rope_cos",
            rope_cos[0].to(dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "rope_sin",
            rope_sin[0].to(dtype=dtype),
            persistent=False,
        )

    def reset_state(self) -> None:
        self.key_cache.zero_()
        self.value_cache.zero_()

    def _repeat_kv(self, value: torch.Tensor) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return value
        batch, key_heads, sequence, head_dim = value.shape
        expanded = value[:, :, None, :, :].expand(
            batch,
            key_heads,
            self.num_key_value_groups,
            sequence,
            head_dim,
        )
        return expanded.reshape(
            batch,
            key_heads * self.num_key_value_groups,
            sequence,
            head_dim,
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = token_embeddings
        query_length = hidden_states.shape[1]
        end_step = causal_mask.shape[-1]
        begin_step = end_step - query_length
        cos = F.embedding(position_ids, self.rope_cos).unsqueeze(1)
        sin = F.embedding(position_ids, self.rope_sin).unsqueeze(1)

        for layer_index, layer in enumerate(self.layers):
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            attention = layer.self_attn
            query_states = attention.q_proj(normalized).reshape(
                1,
                query_length,
                self.num_attention_heads,
                self.head_dim,
            )
            key_states = attention.k_proj(normalized).reshape(
                1,
                query_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            value_states = attention.v_proj(normalized).reshape(
                1,
                query_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            query_states = (query_states * cos) + (_rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (_rotate_half(key_states) * sin)

            self.key_cache[layer_index, :, :, begin_step:end_step, :] = key_states
            self.value_cache[layer_index, :, :, begin_step:end_step, :] = value_states
            cached_keys = self.key_cache[layer_index, :, :, :end_step, :]
            cached_values = self.value_cache[layer_index, :, :, :end_step, :]
            cached_keys = self._repeat_kv(cached_keys)
            cached_values = self._repeat_kv(cached_values)

            attention_weights = torch.matmul(
                query_states,
                cached_keys.transpose(2, 3),
            ) * float(attention.scaling)
            attention_weights = attention_weights + causal_mask
            attention_weights = F.softmax(
                attention_weights,
                dim=-1,
                dtype=torch.float32,
            ).to(query_states.dtype)
            attention_output = torch.matmul(attention_weights, cached_values)
            attention_output = attention_output.transpose(1, 2).reshape(
                1,
                query_length,
                self.hidden_size,
            )
            hidden_states = residual + attention.o_proj(attention_output)
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states[:, -1, :])


def wrap_smolvlm2_500m_coreml_components(
    model: nn.Module,
    *,
    profile: CoreMLVLMProfile | None = None,
) -> dict[str, nn.Module]:
    """Prepare the three exact Smol500 component graphs."""

    resolved = profile or smolvlm2_500m_coreml_profile()
    _validate_exact_smolvlm2_profile(resolved)
    validate_smolvlm2_500m_model(model)
    base = model.model
    decoder = CoreMLVLMStatefulLlamaDecoder(
        text_model=base.text_model,
        lm_head=model.lm_head,
        context_length=resolved.context_length,
        state_dtype=next(base.text_model.parameters()).dtype,
    )
    return {
        COREML_VLM_ENCODE_IMAGE_FUNCTION: SmolVLM2FixedSquareVisionEncoder(
            model,
            image_crops=resolved.image_crops,
        ).eval(),
        COREML_VLM_EMBED_TOKENS_FUNCTION: CoreMLVLMTokenEmbedding(
            model.get_input_embeddings()
        ).eval(),
        COREML_VLM_DECODE_FUNCTION: decoder.eval(),
    }


def assert_smolvlm2_fixed_vision_eager_parity(
    model: nn.Module,
    wrapper: SmolVLM2FixedSquareVisionEncoder,
    pixel_values: torch.Tensor,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Prove the fixed-grid rewrite against the stock all-valid image path."""

    if tuple(pixel_values.shape) != (
        1,
        wrapper.image_crops,
        3,
        wrapper.image_size,
        wrapper.image_size,
    ):
        raise ValueError("Pixel probe does not match the fixed vision wrapper.")
    mask = torch.ones(
        (
            1,
            wrapper.image_crops,
            wrapper.image_size,
            wrapper.image_size,
        ),
        dtype=torch.bool,
        device=pixel_values.device,
    )
    with torch.inference_mode():
        stock = model.get_image_features(
            pixel_values=pixel_values,
            pixel_attention_mask=mask,
            return_dict=True,
        ).pooler_output
        stock = stock.reshape(
            1,
            wrapper.image_crops * wrapper.tokens_per_crop,
            wrapper.hidden_size,
        )
        prepared = wrapper(pixel_values)
    torch.testing.assert_close(prepared, stock, rtol=rtol, atol=atol)


def assert_smolvlm2_decoder_source_parity(
    model: nn.Module,
    wrapper: CoreMLVLMStatefulLlamaDecoder,
    *,
    profile: CoreMLVLMProfile,
    relative_tolerance: float = COREML_VLM_SOURCE_REL_TOL,
) -> dict[str, float]:
    """Gate prefill/decode against the loaded source attention implementation."""

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as exc:
        raise ImportError(
            "SmolVLM2 source parity requires transformers "
            f"{COREML_VLM_TRANSFORMERS_VERSION}."
        ) from exc
    if (
        not math.isfinite(float(relative_tolerance))
        or float(relative_tolerance) <= 0
    ):
        raise ValueError("relative_tolerance must be finite and positive.")
    text_model = model.model.text_model
    source_cache = DynamicCache(config=model.config.text_config)
    cursor = CoreMLVLMDecodeCursor(profile)
    metrics: dict[str, float] = {}
    wrapper.reset_state()
    try:
        for label, input_ids in (
            ("prefill", torch.tensor([[3, 4, 5]], dtype=torch.long)),
            ("decode", torch.tensor([[6]], dtype=torch.long)),
        ):
            embeddings = model.get_input_embeddings()(input_ids)
            causal_mask, position_ids = cursor.controls(
                query_length=int(input_ids.shape[1])
            )
            end_step = int(causal_mask.shape[-1])
            with torch.inference_mode():
                source = text_model(
                    inputs_embeds=embeddings,
                    attention_mask=torch.ones(
                        (1, end_step),
                        dtype=torch.long,
                        device=embeddings.device,
                    ),
                    position_ids=torch.from_numpy(position_ids)
                    .to(device=embeddings.device, dtype=torch.long),
                    past_key_values=source_cache,
                    use_cache=True,
                )
                expected = model.lm_head(
                    source.last_hidden_state[:, -1, :]
                )
                actual = wrapper(
                    embeddings,
                    torch.from_numpy(causal_mask).to(
                        device=embeddings.device,
                        dtype=embeddings.dtype,
                    ),
                    torch.from_numpy(position_ids).to(
                        device=embeddings.device,
                    ),
                )
            scale = max(float(expected.abs().max()), 1e-12)
            error = float((actual - expected).abs().max()) / scale
            metrics[f"{label}_relative_max_error"] = error
            if error > float(relative_tolerance):
                raise RuntimeError(
                    "SmolVLM2 Core ML decoder diverges from the loaded "
                    f"source {label} path: relative max error {error:.3e} "
                    f"exceeds {float(relative_tolerance):.0e}."
                )
            cursor.commit(
                causal_mask=causal_mask,
                position_ids=position_ids,
            )
    finally:
        wrapper.reset_state()
    return metrics


def _parse_major(version: Any) -> int:
    text = str(version)
    try:
        return int(text.split(".", 1)[0])
    except ValueError as exc:
        raise RuntimeError(f"Invalid coremltools version {text!r}.") from exc


def require_coreml_vlm_toolchain(coremltools_module: Any) -> None:
    """Require the converter generation used to validate state+multifunction."""

    version = getattr(coremltools_module, "__version__", "")
    major = _parse_major(version)
    if major != COREML_VLM_REQUIRED_COREMLTOOLS_MAJOR:
        raise RuntimeError(
            "Core ML VLM export is pinned to coremltools 9.x while its "
            f"stateful ABI is validated; found {version!r}."
        )


def require_coreml_vlm_transformers_toolchain(
    transformers_module: Any | None = None,
) -> None:
    """Require the exact Transformers implementation used for clean adaptation."""

    if transformers_module is None:
        try:
            import transformers as transformers_module
        except ImportError as exc:
            raise ImportError(
                "SmolVLM2 Core ML export requires transformers "
                f"{COREML_VLM_TRANSFORMERS_VERSION}."
            ) from exc
    version = str(getattr(transformers_module, "__version__", ""))
    if version != COREML_VLM_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "SmolVLM2 Core ML export is pinned to transformers "
            f"{COREML_VLM_TRANSFORMERS_VERSION}, found {version!r}."
        )


def _feature_shape_ranges(feature: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for axis in feature["shape"]:
        if isinstance(axis, Integral):
            value = int(axis)
            ranges.append((value, value))
        elif axis["kind"] == "fixed":
            value = int(axis["value"])
            ranges.append((value, value))
        elif axis["kind"] == "range":
            lower = int(axis["lower_bound"])
            upper = int(axis["upper_bound"])
            if lower <= 0 or upper < lower:
                raise ValueError(f"Invalid bounded axis {axis!r}.")
            ranges.append((lower, upper))
        else:
            raise ValueError(f"Unknown Core ML VLM axis kind {axis!r}.")
    return tuple(ranges)


def _coreml_tensor_type(
    ct: Any,
    feature: Mapping[str, Any],
    *,
    symbols: dict[str, Any],
) -> Any:
    dtype = {
        "float16": np.float16,
        "int32": np.int32,
    }[str(feature["dtype"])]
    shape: list[Any] = []
    for axis in feature["shape"]:
        if isinstance(axis, Integral):
            shape.append(int(axis))
        elif axis["kind"] == "fixed":
            shape.append(int(axis["value"]))
        else:
            symbol = str(axis["name"])
            dimension = symbols.get(symbol)
            if dimension is None:
                dimension = ct.RangeDim(
                    lower_bound=int(axis["lower_bound"]),
                    upper_bound=int(axis["upper_bound"]),
                    default=int(axis["default"]),
                    symbol=symbol,
                )
                symbols[symbol] = dimension
            shape.append(dimension)
    return ct.TensorType(name=feature["name"], shape=tuple(shape), dtype=dtype)


def _capture_coreml_vlm_component(
    component: nn.Module,
    *,
    function_name: str,
    profile: CoreMLVLMProfile,
) -> tuple[Any, tuple[torch.Tensor, ...]]:
    dtype = next(component.parameters()).dtype
    if function_name == COREML_VLM_ENCODE_IMAGE_FUNCTION:
        probe = (
            torch.full(
                (
                    1,
                    profile.image_crops,
                    profile.image_channels,
                    profile.image_height,
                    profile.image_width,
                ),
                0.125,
                dtype=dtype,
            ),
        )
        captured = torch.jit.trace(component, probe, check_trace=True)
    elif function_name == COREML_VLM_EMBED_TOKENS_FUNCTION:
        probe = (torch.zeros((1, 2), dtype=torch.int32),)
        captured = torch.jit.trace(component, probe, check_trace=True)
    elif function_name == COREML_VLM_DECODE_FUNCTION:
        if not isinstance(component, CoreMLVLMStatefulLlamaDecoder):
            raise TypeError("decode component must expose explicit VLM state.")
        component.reset_state()
        probe = (
            torch.zeros((1, 2, profile.hidden_size), dtype=dtype),
            torch.tensor(
                [[[[0.0, torch.finfo(dtype).min], [0.0, 0.0]]]],
                dtype=dtype,
            ),
            torch.tensor([[0, 1]], dtype=torch.int32),
        )
        # Stateful tracing necessarily mutates registered buffers. Validate
        # eager sequential parity independently; ordinary trace checking would
        # compare different state histories.
        captured = torch.jit.trace(component, probe, check_trace=False)
        if hasattr(captured, COREML_VLM_KEY_CACHE_STATE):
            getattr(captured, COREML_VLM_KEY_CACHE_STATE).zero_()
            getattr(captured, COREML_VLM_VALUE_CACHE_STATE).zero_()
        component.reset_state()
    else:
        raise ValueError(f"Unknown Core ML VLM function {function_name!r}.")
    return captured, probe


def _to_compute_unit(ct: Any, value: str) -> Any:
    normalized = str(value).strip().lower()
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    if normalized not in mapping:
        raise ValueError(
            f"Invalid Core ML compute_units {value!r}; expected one of "
            f"{sorted(mapping)}."
        )
    return mapping[normalized]


def _bind_coreml_vlm_output_bounds(
    ct: Any,
    converted: Any,
    *,
    function_name: str,
    profile: CoreMLVLMProfile,
    compute_units: str,
) -> Any:
    """Serialize inferred flexible-output bounds that CT9 leaves empty.

    Core ML Tools 9 preserves the dynamic input RangeDim in the MIL program,
    but its PyTorch frontend emits an empty protobuf shape for outputs whose
    extent depends on that input (notably ``embed_tokens``). Output shapes
    cannot be passed to ``ct.convert``. Bind the already-declared finite
    contract after conversion and reject any non-empty inference that
    disagrees with it.
    """

    contract = coreml_vlm_function_contracts(profile)[function_name]
    spec = converted.get_spec()
    outputs = list(spec.description.output)
    if [str(value.name) for value in outputs] != [
        item["name"] for item in contract["outputs"]
    ]:
        raise RuntimeError(f"{function_name!r} output names changed.")

    changed = False
    for actual, expected in zip(outputs, contract["outputs"]):
        expected_ranges = _feature_shape_ranges(expected)
        actual_ranges = _protobuf_feature_ranges(actual)
        if actual_ranges:
            if actual_ranges != expected_ranges:
                raise RuntimeError(
                    f"{function_name!r} output {expected['name']!r} changed "
                    f"shape: {actual_ranges!r} != {expected_ranges!r}."
                )
            continue
        array = actual.type.multiArrayType
        array.ClearField("shape")
        array.ClearField("shapeRange")
        for lower, upper in expected_ranges:
            axis = array.shapeRange.sizeRanges.add()
            axis.lowerBound = lower
            axis.upperBound = upper
        changed = True

    if not changed:
        return converted
    weights_dir = getattr(converted, "weights_dir", None)
    if not weights_dir:
        raise RuntimeError(
            f"{function_name!r} conversion has no reusable ML Program weights."
        )
    return ct.models.MLModel(
        spec,
        weights_dir=weights_dir,
        skip_model_load=True,
        compute_units=_to_compute_unit(ct, compute_units),
    )


def _convert_coreml_vlm_component(
    ct: Any,
    component: nn.Module,
    *,
    function_name: str,
    profile: CoreMLVLMProfile,
    compute_units: str,
) -> Any:
    contracts = coreml_vlm_function_contracts(profile)
    contract = contracts[function_name]
    captured, _ = _capture_coreml_vlm_component(
        component,
        function_name=function_name,
        profile=profile,
    )
    symbols: dict[str, Any] = {}
    inputs = [
        _coreml_tensor_type(ct, feature, symbols=symbols)
        for feature in contract["inputs"]
    ]
    outputs = [
        ct.TensorType(name=feature["name"], dtype=np.float16)
        for feature in contract["outputs"]
    ]
    kwargs: dict[str, Any] = {
        "inputs": inputs,
        "outputs": outputs,
        "convert_to": "mlprogram",
        # The complete SmolVLM2 vision tower and recurrent decoder exceed their
        # hardware parity gates when every intermediate is lowered to FP16.
        # Preserve FP32 compute for both graphs while retaining the bounded
        # FP16 public/state ABI. The embedding lookup is exact with FP16 compute.
        "compute_precision": (
            ct.precision.FLOAT16
            if function_name == COREML_VLM_EMBED_TOKENS_FUNCTION
            else ct.precision.FLOAT32
        ),
        "minimum_deployment_target": ct.target.iOS18,
        "compute_units": _to_compute_unit(ct, compute_units),
        "skip_model_load": True,
    }
    if function_name == COREML_VLM_DECODE_FUNCTION:
        kwargs["states"] = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=tuple(state["shape"]),
                    dtype=np.float16,
                ),
                name=state["name"],
            )
            for state in contract["states"]
        ]
    converted = ct.convert(captured, **kwargs)
    del captured
    return _bind_coreml_vlm_output_bounds(
        ct,
        converted,
        function_name=function_name,
        profile=profile,
        compute_units=compute_units,
    )


def _protobuf_feature_ranges(feature: Any) -> tuple[tuple[int, int], ...]:
    array = feature.type.multiArrayType
    if array.WhichOneof("ShapeFlexibility") == "shapeRange":
        return tuple(
            (int(axis.lowerBound), int(axis.upperBound))
            for axis in array.shapeRange.sizeRanges
        )
    return tuple((int(value), int(value)) for value in array.shape)


def validate_coreml_vlm_function_description(
    description: Any,
    *,
    function_name: str,
    profile: CoreMLVLMProfile,
) -> None:
    """Validate serialized names, dtypes, finite bounds, and state shapes."""

    contract = coreml_vlm_function_contracts(profile)[function_name]
    inputs = list(getattr(description, "input", ()) or ())
    outputs = list(getattr(description, "output", ()) or ())
    if [str(value.name) for value in inputs] != [
        item["name"] for item in contract["inputs"]
    ]:
        raise RuntimeError(f"{function_name!r} input names changed.")
    if [str(value.name) for value in outputs] != [
        item["name"] for item in contract["outputs"]
    ]:
        raise RuntimeError(f"{function_name!r} output names changed.")
    dtype_codes = {"float16": 65552, "int32": 131104}
    for actual, expected in zip(inputs, contract["inputs"]):
        array = actual.type.multiArrayType
        if int(array.dataType) != dtype_codes[expected["dtype"]]:
            raise RuntimeError(
                f"{function_name!r} input {expected['name']!r} changed dtype."
            )
        actual_ranges = _protobuf_feature_ranges(actual)
        expected_ranges = _feature_shape_ranges(expected)
        if actual_ranges != expected_ranges:
            raise RuntimeError(
                f"{function_name!r} input {expected['name']!r} changed bounds: "
                f"{actual_ranges!r} != {expected_ranges!r}."
            )
    for actual, expected in zip(outputs, contract["outputs"]):
        if int(actual.type.multiArrayType.dataType) != dtype_codes[expected["dtype"]]:
            raise RuntimeError(
                f"{function_name!r} output {expected['name']!r} changed dtype."
            )
        actual_ranges = _protobuf_feature_ranges(actual)
        expected_ranges = _feature_shape_ranges(expected)
        if actual_ranges != expected_ranges:
            raise RuntimeError(
                f"{function_name!r} output {expected['name']!r} changed shape: "
                f"{actual_ranges!r} != {expected_ranges!r}."
            )

    expected_states = contract.get("states", [])
    actual_states = list(getattr(description, "state", ()) or ())
    if [str(value.name) for value in actual_states] != [
        item["name"] for item in expected_states
    ]:
        raise RuntimeError(f"{function_name!r} state names changed.")
    for actual, expected in zip(actual_states, expected_states):
        array = actual.type.stateType.arrayType
        if int(array.dataType) != dtype_codes["float16"]:
            raise RuntimeError(
                f"{function_name!r} state {expected['name']!r} changed dtype."
            )
        if tuple(int(value) for value in array.shape) != tuple(expected["shape"]):
            raise RuntimeError(
                f"{function_name!r} state {expected['name']!r} changed shape."
            )


def validate_coreml_vlm_multifunction_spec(
    spec: Any,
    *,
    profile: CoreMLVLMProfile,
) -> None:
    """Validate the complete native function table after weight deduplication."""

    description = getattr(spec, "description", None)
    if list(getattr(description, "input", ()) or ()) or list(
        getattr(description, "output", ()) or ()
    ):
        raise RuntimeError(
            "Core ML VLM multifunction package exposes a false top-level ABI."
        )
    if str(getattr(description, "defaultFunctionName", "")) != (
        COREML_VLM_ENCODE_IMAGE_FUNCTION
    ):
        raise RuntimeError("Core ML VLM package has the wrong default function.")
    functions = list(getattr(description, "functions", ()) or ())
    names = [str(value.name) for value in functions]
    if names != list(COREML_VLM_FUNCTION_NAMES):
        raise RuntimeError(
            f"Core ML VLM package function order/names changed: {names!r}."
        )
    for function in functions:
        validate_coreml_vlm_function_description(
            function,
            function_name=str(function.name),
            profile=profile,
        )


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a staged package without replacing any node."""

    if os.name == "nt":
        source.rename(destination)
        return

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    function = None
    arguments: tuple[Any, ...] = ()
    if sys.platform == "darwin":
        function = getattr(libc, "renameatx_np", None)
        if function is not None:
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            function.restype = ctypes.c_int
            arguments = (
                -2,
                source_bytes,
                -2,
                destination_bytes,
                0x00000004,
            )
    elif sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        if function is not None:
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            function.restype = ctypes.c_int
            arguments = (
                -100,
                source_bytes,
                -100,
                destination_bytes,
                0x00000001,
            )
    if function is not None:
        ctypes.set_errno(0)
        if function(*arguments) == 0:
            return
        error = ctypes.get_errno()
        if error in {errno.EEXIST, getattr(errno, "ENOTEMPTY", errno.EEXIST)}:
            raise FileExistsError(
                error,
                "Core ML VLM artifact destination already exists",
                str(destination),
            )
        unsupported = {
            errno.EINVAL,
            errno.ENOSYS,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if error not in unsupported:
            raise OSError(
                error,
                "Failed to publish Core ML VLM artifact",
                str(destination),
            )
    raise RuntimeError(
        "The destination filesystem lacks atomic no-replace directory "
        "publication. Refusing an unsafe Core ML VLM artifact rename."
    )


def _write_coreml_vlm_metadata_in_place(
    ct: Any,
    model: Any,
    package_path: Path,
    metadata: Mapping[str, str],
) -> None:
    """Atomically replace only the package protobuf, never its large weights."""

    spec = model.get_spec()
    spec.description.metadata.userDefined.update(dict(metadata))
    model_files = list(package_path.rglob("*.mlmodel"))
    if len(model_files) != 1:
        raise RuntimeError(
            "Merged Core ML VLM package must contain exactly one model "
            f"protobuf, found {len(model_files)}."
        )
    model_file = model_files[0]
    temporary = model_file.with_name(".libreyolo-metadata.mlmodel")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(
            f"Refusing to replace existing Core ML metadata staging file: {temporary}."
        )
    try:
        ct.utils.save_spec(spec, str(temporary))
        os.replace(temporary, model_file)
    finally:
        temporary.unlink(missing_ok=True)


def build_coreml_vlm_multifunction_package(
    components: Mapping[str, nn.Module],
    *,
    output_path: str | os.PathLike[str],
    profile: CoreMLVLMProfile,
    metadata: Mapping[str, Any],
    compute_units: str = "validated",
    coremltools_module: Any | None = None,
) -> str:
    """Convert and merge three components into one native stateful package.

    The destination must not already exist. Publication uses a same-filesystem
    atomic no-replace rename and fails closed when the platform or filesystem
    cannot provide that primitive.
    """

    resolved_compute_units = (
        resolve_smolvlm2_500m_coreml_export_compute_units(compute_units)
    )
    expected_names = list(COREML_VLM_FUNCTION_NAMES)
    if list(components) != expected_names:
        raise ValueError(
            "Core ML VLM components must be ordered exactly as "
            f"{expected_names}, got {list(components)}."
        )
    _validate_exact_smolvlm2_profile(profile)
    validated_metadata = validate_coreml_vlm_metadata(metadata)
    if validated_metadata["vlm_profile"] != profile.as_dict():
        raise ValueError(
            "Core ML VLM package profile conflicts with vlm_profile metadata."
        )
    destination = Path(output_path)
    if destination.suffix.lower() != ".mlpackage":
        raise ValueError("Core ML VLM output must end in .mlpackage.")
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Core ML VLM package: {destination}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    if coremltools_module is None:
        try:
            import coremltools as ct
        except ImportError as exc:
            raise ImportError("Core ML VLM export requires coremltools 9.x.") from exc
    else:
        ct = coremltools_module
    require_coreml_vlm_toolchain(ct)
    string_metadata = stringify_coreml_vlm_metadata(metadata)

    with tempfile.TemporaryDirectory(
        prefix=".libreyolo-coreml-vlm-",
        dir=str(destination.parent),
    ) as temporary_root:
        workspace = Path(temporary_root)
        descriptor = ct.utils.MultiFunctionDescriptor()
        for index, function_name in enumerate(COREML_VLM_FUNCTION_NAMES):
            converted = _convert_coreml_vlm_component(
                ct,
                components[function_name],
                function_name=function_name,
                profile=profile,
                compute_units=resolved_compute_units,
            )
            validate_coreml_vlm_function_description(
                converted.get_spec().description,
                function_name=function_name,
                profile=profile,
            )
            component_path = workspace / f"{index:02d}-{function_name}.mlpackage"
            converted.save(str(component_path))
            descriptor.add_function(
                str(component_path),
                "main",
                function_name,
            )
            del converted
        descriptor.default_function_name = COREML_VLM_ENCODE_IMAGE_FUNCTION
        merged_path = workspace / "merged.mlpackage"
        ct.utils.save_multifunction(descriptor, str(merged_path))
        merged = ct.models.MLModel(str(merged_path), skip_model_load=True)
        _write_coreml_vlm_metadata_in_place(
            ct,
            merged,
            merged_path,
            string_metadata,
        )
        del merged
        staged = ct.models.MLModel(str(merged_path), skip_model_load=True)
        validate_coreml_vlm_multifunction_spec(
            staged.get_spec(),
            profile=profile,
        )
        validate_coreml_vlm_metadata(dict(staged.user_defined_metadata))
        del staged
        _publish_directory_no_replace(merged_path, destination)

    return str(destination)


def export_smolvlm2_500m_coreml_package(
    model: nn.Module,
    *,
    processor_dir: str | os.PathLike[str],
    processor_revision: str,
    output_path: str | os.PathLike[str],
    context_length: int = SMOLVLM2_500M_DEFAULT_CONTEXT,
    compute_units: str = "validated",
) -> str:
    """Strict internal Smol500 package path; not yet a public model hook."""

    resolved_compute_units = (
        resolve_smolvlm2_500m_coreml_export_compute_units(compute_units)
    )
    require_coreml_vlm_transformers_toolchain()
    profile = smolvlm2_500m_coreml_profile(context_length)
    validate_smolvlm2_500m_model(model)
    devices = {
        value.device.type
        for value in (*tuple(model.parameters()), *tuple(model.buffers()))
    }
    if devices != {"cpu"}:
        raise NotImplementedError(
            "SmolVLM2 Core ML conversion requires a CPU model, found "
            f"{sorted(devices)}."
        )
    floating_dtypes = {
        value.dtype
        for value in (*tuple(model.parameters()), *tuple(model.buffers()))
        if value.is_floating_point()
    }
    # Core ML Tools 9.0 lowers this FP32 capture to an FP16 ML Program. Feeding
    # it an already-half PyTorch graph makes LayerNorm's FP32 epsilon conflict
    # with FP16 gamma during frontend conversion. Requiring FP32 here is both
    # deterministic and covered by the Linux conversion probe.
    if floating_dtypes != {torch.float32}:
        raise NotImplementedError(
            "SmolVLM2 Core ML conversion requires an FP32-loaded model; found "
            f"{sorted(str(value) for value in floating_dtypes)}."
        )
    validate_smolvlm2_500m_processor_assets(
        processor_dir,
        revision=processor_revision,
        transformers_version=COREML_VLM_TRANSFORMERS_VERSION,
    )
    validate_smolvlm2_500m_weight_asset(
        processor_dir,
        revision=processor_revision,
    )
    validate_smolvlm2_500m_model_weight_values(model, processor_dir)
    training_states = tuple(
        (module, module.training) for module in model.modules()
    )
    try:
        components = wrap_smolvlm2_500m_coreml_components(
            model.eval(),
            profile=profile,
        )
        y_axis = torch.linspace(
            0.0,
            1.0,
            profile.image_height,
            dtype=torch.float32,
        ).view(1, 1, 1, profile.image_height, 1)
        x_axis = torch.linspace(
            0.0,
            1.0,
            profile.image_width,
            dtype=torch.float32,
        ).view(1, 1, 1, 1, profile.image_width)
        channels = torch.tensor(
            [0.0, 0.25, 0.5],
            dtype=torch.float32,
        ).view(1, 1, profile.image_channels, 1, 1)
        vision_probe = (
            (x_axis + y_axis + channels)
            .remainder(1.0)
            .expand(
                1,
                profile.image_crops,
                profile.image_channels,
                profile.image_height,
                profile.image_width,
            )
            .contiguous()
        )
        assert_smolvlm2_fixed_vision_eager_parity(
            model,
            components[COREML_VLM_ENCODE_IMAGE_FUNCTION],
            vision_probe,
        )
        assert_smolvlm2_decoder_source_parity(
            model,
            components[COREML_VLM_DECODE_FUNCTION],
            profile=profile,
        )
        metadata = smolvlm2_500m_coreml_metadata(profile)
        return build_coreml_vlm_multifunction_package(
            components,
            output_path=output_path,
            profile=profile,
            metadata=metadata,
            compute_units=resolved_compute_units,
        )
    finally:
        for module, training in training_states:
            module.training = training


__all__ = [
    "COREML_VLM_ARTIFACT_SCOPE",
    "COREML_VLM_DECODE_FUNCTION",
    "COREML_VLM_EMBED_TOKENS_FUNCTION",
    "COREML_VLM_ENCODE_IMAGE_FUNCTION",
    "COREML_VLM_FUNCTION_NAMES",
    "COREML_VLM_KEY_CACHE_STATE",
    "COREML_VLM_LAST_LOGITS_OUTPUT",
    "COREML_VLM_REQUIRED_COREMLTOOLS_MAJOR",
    "COREML_VLM_SCHEMA_VERSION",
    "COREML_VLM_SOURCE_REL_TOL",
    "COREML_VLM_TRANSFORMERS_COMMIT",
    "COREML_VLM_TRANSFORMERS_VERSION",
    "COREML_VLM_VALUE_CACHE_STATE",
    "CoreMLVLMProfile",
    "CoreMLVLMDecodeCursor",
    "CoreMLVLMStatefulLlamaDecoder",
    "CoreMLVLMTokenEmbedding",
    "SMOLVLM2_500M_COMPONENT_CONTRACT",
    "SMOLVLM2_500M_CONTEXT_MAX_NEW_TOKENS",
    "SMOLVLM2_500M_REQUIRED_ASSETS",
    "SMOLVLM2_500M_REVISION",
    "SMOLVLM2_500M_SOURCE_IMAGE_SIZE",
    "SMOLVLM2_500M_WEIGHTS_FILENAME",
    "SMOLVLM2_500M_WEIGHTS_SHA256",
    "SMOLVLM2_500M_WEIGHTS_SIZE",
    "SmolVLM2FixedSquareVisionEncoder",
    "assert_smolvlm2_fixed_vision_eager_parity",
    "assert_smolvlm2_decoder_source_parity",
    "build_coreml_vlm_decode_controls",
    "build_coreml_vlm_multifunction_package",
    "coreml_vlm_function_contracts",
    "export_smolvlm2_500m_coreml_package",
    "merge_coreml_vlm_image_embeddings",
    "preprocess_smolvlm2_500m_coreml_image",
    "prepare_smolvlm2_500m_coreml_processor_batch",
    "require_coreml_vlm_toolchain",
    "require_coreml_vlm_transformers_toolchain",
    "resolve_smolvlm2_500m_coreml_export_compute_units",
    "smolvlm2_500m_coreml_metadata",
    "smolvlm2_500m_coreml_profile",
    "smolvlm2_500m_processor_manifest",
    "smolvlm2_500m_weights_manifest",
    "stringify_coreml_vlm_metadata",
    "validate_coreml_vlm_context_budget",
    "validate_coreml_vlm_decode_controls",
    "validate_coreml_vlm_decode_bounds",
    "validate_coreml_vlm_function_description",
    "validate_coreml_vlm_metadata",
    "validate_coreml_vlm_multifunction_spec",
    "validate_smolvlm2_500m_model",
    "validate_smolvlm2_500m_model_weight_values",
    "validate_smolvlm2_500m_processor_assets",
    "validate_smolvlm2_500m_weight_asset",
    "wrap_smolvlm2_500m_coreml_components",
]
