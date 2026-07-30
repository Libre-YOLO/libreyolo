"""Pinned Core ML 9 export contract for Florence-2-base.

Florence-2 is an encoder-decoder generative detector.  Its deployment artifact
therefore consists of two named ML Program functions rather than a conventional
one-shot detector:

``encode``
    Runs the fixed 768-pixel DaViT vision tower, multimodal projector, and
    1024-position BART encoder, then returns the six packed cross-attention
    key/value tensors.

``decode``
    Runs one token of exact three-beam BART decoding.  Request-local self and
    cross key/value tensors are Core ML state.  ``beam_parent_indices`` reorders
    the self cache before the selected token is appended.

Adaptation provenance
---------------------
The static DaViT/projector and BART equations in this file are adapted from
``huggingface/transformers`` 5.12.1 at commit
``ddb849abe009d1089e6c691bfc897f27211c663c`` (Apache-2.0), specifically:

* ``src/transformers/models/florence2/modeling_florence2.py``;
* ``src/transformers/models/bart/modeling_bart.py``.

The adapted wrappers remove only data-dependent shape construction,
``masked_scatter``, and Transformers cache objects.  They retain the exact
fixed-profile equations and learned modules.  No remote checkpoint code is
executed.  The converted checkpoint is the MIT-licensed
``florence-community/Florence-2-base`` snapshot pinned below.
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
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .coreml_profiles import (
    coreml_execution_profile_metadata,
    resolve_coreml_export_compute_units,
)

FLORENCE_COREML_SCHEMA_VERSION = 2
FLORENCE_COREML_ARTIFACT_SCOPE = "host_orchestrated_florence2_beam_multifunction"
FLORENCE_COREML_COMPONENT_CONTRACT = (
    "florence2_base_768_enc1024_dec1024_beam3_fp32enc_fp16dec_state32_v2"
)
FLORENCE_COREML_REQUIRED_COREMLTOOLS_MAJOR = 9
FLORENCE_COREML_TRANSFORMERS_VERSION = "5.12.1"
FLORENCE_COREML_TRANSFORMERS_COMMIT = "ddb849abe009d1089e6c691bfc897f27211c663c"
FLORENCE_COREML_MINIMUM_DEPLOYMENT_TARGETS = ("iOS18", "macOS15")
FLORENCE_COREML_SOURCE_REL_TOL = 3e-4

FLORENCE2_BASE_REPO = "florence-community/Florence-2-base"
FLORENCE2_BASE_REVISION = "00921df66db728a9ceb750f5eca43e5c203a2051"
FLORENCE2_ORIGINAL_REPO = "microsoft/Florence-2-base"
FLORENCE2_ORIGINAL_LICENSE_REVISION = "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
FLORENCE2_BASE_WEIGHTS_FILENAME = "model.safetensors"
FLORENCE2_BASE_WEIGHTS_SIZE = 463_178_864
FLORENCE2_BASE_WEIGHTS_SHA256 = (
    "62f3e696da74f8869a68ddb529a9b3e14eb25b21c592cb3dea6179bf944df6a0"
)
FLORENCE2_BASE_UNIQUE_PARAMETER_COUNT = 231_443_968

FLORENCE2_TASK = "<OPEN_VOCABULARY_DETECTION>"
FLORENCE2_IMAGE_TOKEN_ID = 51_289
FLORENCE2_BOS_TOKEN_ID = 0
FLORENCE2_PAD_TOKEN_ID = 1
FLORENCE2_EOS_TOKEN_ID = 2
FLORENCE2_DECODER_START_TOKEN_ID = 2
FLORENCE2_FORCED_BOS_TOKEN_ID = 0
FLORENCE2_FORCED_EOS_TOKEN_ID = 2
FLORENCE2_NUM_BEAMS = 3
FLORENCE2_NO_REPEAT_NGRAM_SIZE = 3
FLORENCE2_LENGTH_PENALTY = 1.0

FLORENCE_ENCODE_FUNCTION = "encode"
FLORENCE_DECODE_FUNCTION = "decode"
FLORENCE_FUNCTION_NAMES = (
    FLORENCE_ENCODE_FUNCTION,
    FLORENCE_DECODE_FUNCTION,
)

FLORENCE_PIXEL_VALUES_INPUT = "pixel_values"
FLORENCE_ENCODER_INPUT_IDS_INPUT = "encoder_input_ids"
FLORENCE_ENCODER_ATTENTION_MASK_INPUT = "encoder_attention_mask"
FLORENCE_CROSS_KEY_OUTPUT = "cross_key_values"
FLORENCE_CROSS_VALUE_OUTPUT = "cross_value_values"

FLORENCE_DECODER_INPUT_IDS_INPUT = "decoder_input_ids"
FLORENCE_CAUSAL_MASK_INPUT = "causal_mask"
FLORENCE_CROSS_ATTENTION_MASK_INPUT = "cross_attention_mask"
FLORENCE_POSITION_IDS_INPUT = "position_ids"
FLORENCE_BEAM_PARENT_INDICES_INPUT = "beam_parent_indices"
FLORENCE_LAST_LOGITS_OUTPUT = "last_logits"

FLORENCE_SELF_KEY_CACHE_STATE = "self_key_cache"
FLORENCE_SELF_VALUE_CACHE_STATE = "self_value_cache"
FLORENCE_CROSS_KEY_CACHE_STATE = "cross_key_cache"
FLORENCE_CROSS_VALUE_CACHE_STATE = "cross_value_cache"


FLORENCE2_BASE_REQUIRED_ASSETS: dict[str, str] = {
    "added_tokens.json": (
        "1d75deda84dfa81fb6c09301f3fed00f9695059568bbb1403a6bf299cd84fc37"
    ),
    "config.json": ("2a4dfe9885d183da121f239ada734bd700ee6908b650a7a25ab32eb104fe1d7a"),
    "generation_config.json": (
        "0251459c49cc358ac033b5d4b8569e61ac22bde5d76be404b97a44cfb33fb12e"
    ),
    "merges.txt": ("1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
    "preprocessor_config.json": (
        "1396ec5a0a7adfe1c04fb777b09e8ba753be6dbb5868212ab3c3ef39d91fe031"
    ),
    "processor_config.json": (
        "cd0e3bf41a39b1276503fbd273bc03b9afc70d7a15ea681a92a1b4b77f858ee6"
    ),
    "special_tokens_map.json": (
        "72ff172dc769bc1551b1b4211628ce3271643bc60379e4da45d85a9be9332c39"
    ),
    "tokenizer.json": (
        "3ad7001f773409abe6bba33eac92662611a73d72f459bda2f00d2a221dd31ce4"
    ),
    "tokenizer_config.json": (
        "cb5f80bd9afa767bb1bb798ee5f0a79eae45239b8e72506166b4218175a16723"
    ),
    "vocab.json": ("ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e"),
}

_FLORENCE_TIED_WEIGHT_ALIASES = frozenset(
    {
        "lm_head.weight",
        "model.language_model.decoder.embed_tokens.weight",
        "model.language_model.encoder.embed_tokens.weight",
    }
)


def _strict_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
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
class FlorenceCoreMLProfile:
    """Exact, finite Core ML tensor and state dimensions."""

    family: str = "florence2"
    size: str = "base"
    image_size: int = 768
    image_channels: int = 3
    image_token_count: int = 577
    encoder_context_length: int = 1024
    decoder_context_length: int = 1024
    hidden_size: int = 768
    vocab_size: int = 51_328
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    head_dim: int = 64
    num_beams: int = 3
    max_new_tokens: int = 1024

    def __post_init__(self) -> None:
        for name in (
            "image_size",
            "image_channels",
            "image_token_count",
            "encoder_context_length",
            "decoder_context_length",
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "num_attention_heads",
            "head_dim",
            "num_beams",
            "max_new_tokens",
        ):
            _strict_positive_int(getattr(self, name), name=name)
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal attention heads times head_dim.")
        if self.image_token_count >= self.encoder_context_length:
            raise ValueError("encoder context must leave room beyond image tokens.")
        if self.max_new_tokens > self.decoder_context_length:
            raise ValueError(
                "max_new_tokens cannot exceed the decoder context.  The last "
                "token at the full budget is forced EOS and is not fed back."
            )

    @property
    def single_cross_cache_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.num_hidden_layers,
            1,
            self.num_attention_heads,
            self.encoder_context_length,
            self.head_dim,
        )

    @property
    def self_cache_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.num_hidden_layers,
            self.num_beams,
            self.num_attention_heads,
            self.decoder_context_length,
            self.head_dim,
        )

    @property
    def cross_cache_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.num_hidden_layers,
            self.num_beams,
            self.num_attention_heads,
            self.encoder_context_length,
            self.head_dim,
        )

    @property
    def total_state_bytes_fp16(self) -> int:
        return (
            2
            * (math.prod(self.self_cache_shape) + math.prod(self.cross_cache_shape))
            * 2
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "size": self.size,
            "image_size": self.image_size,
            "image_channels": self.image_channels,
            "image_token_count": self.image_token_count,
            "encoder_context_length": self.encoder_context_length,
            "decoder_context_length": self.decoder_context_length,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "single_cross_cache_shape": list(self.single_cross_cache_shape),
            "self_cache_shape": list(self.self_cache_shape),
            "cross_cache_shape": list(self.cross_cache_shape),
            "total_state_bytes_fp16": self.total_state_bytes_fp16,
        }


def florence2_base_coreml_profile() -> FlorenceCoreMLProfile:
    """Return the only bounded Florence Core ML graph profile."""

    return FlorenceCoreMLProfile()


def resolve_florence2_base_coreml_export_compute_units(
    compute_units: Any,
) -> str:
    """Resolve this unvalidated VLM export without implying Apple evidence."""

    resolved, execution_profile = resolve_coreml_export_compute_units(
        compute_units,
        family="florence2",
        task="detect",
        size="base",
        canvas=768,
        precision="fp32",
        nms=False,
    )
    if execution_profile is not None:
        raise RuntimeError(
            "Florence-2 Core ML is still experimental, but an exact hardware "
            "execution profile was unexpectedly registered. Update the "
            "specialized bundle contract before enabling validated routing."
        )
    return resolved


def _validate_exact_profile(profile: FlorenceCoreMLProfile) -> None:
    if profile != florence2_base_coreml_profile():
        raise ValueError(
            "Profile conflicts with the exact Florence-2-base Core ML ABI."
        )


def florence2_base_weights_manifest() -> dict[str, Any]:
    return {
        "repo": FLORENCE2_BASE_REPO,
        "revision": FLORENCE2_BASE_REVISION,
        "filename": FLORENCE2_BASE_WEIGHTS_FILENAME,
        "size_bytes": FLORENCE2_BASE_WEIGHTS_SIZE,
        "sha256": FLORENCE2_BASE_WEIGHTS_SHA256,
        "license": "MIT",
        "original_repo": FLORENCE2_ORIGINAL_REPO,
        "original_license_revision": FLORENCE2_ORIGINAL_LICENSE_REVISION,
    }


def florence2_base_processor_manifest() -> dict[str, Any]:
    return {
        "repo": FLORENCE2_BASE_REPO,
        "revision": FLORENCE2_BASE_REVISION,
        "trust_remote_code": False,
        "transformers_version": FLORENCE_COREML_TRANSFORMERS_VERSION,
        "required_assets": dict(FLORENCE2_BASE_REQUIRED_ASSETS),
    }


def validate_florence2_base_processor_assets(
    processor_dir: str | os.PathLike[str],
    *,
    revision: str,
    transformers_version: str = FLORENCE_COREML_TRANSFORMERS_VERSION,
) -> dict[str, Any]:
    """Validate the exact offline processor snapshot byte-for-byte."""

    if revision != FLORENCE2_BASE_REVISION:
        raise ValueError(
            "Florence-2-base Core ML requires processor revision "
            f"{FLORENCE2_BASE_REVISION}, got {revision!r}."
        )
    if transformers_version != FLORENCE_COREML_TRANSFORMERS_VERSION:
        raise ValueError(
            "Florence-2-base processor semantics are pinned to transformers "
            f"{FLORENCE_COREML_TRANSFORMERS_VERSION}, got "
            f"{transformers_version!r}."
        )
    root = Path(processor_dir)
    if root.is_symlink():
        raise ValueError("Florence processor root must not be a symbolic link.")
    if not root.is_dir():
        raise FileNotFoundError(f"Florence processor directory is missing: {root}.")
    for name, expected_hash in FLORENCE2_BASE_REQUIRED_ASSETS.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"Florence processor snapshot is missing regular file {name!r}."
            )
        actual_hash = _file_sha256(path)
        if not hmac.compare_digest(actual_hash, expected_hash):
            raise ValueError(
                "Florence processor asset failed SHA-256 validation: "
                f"{name!r}, expected {expected_hash}, got {actual_hash}."
            )
    return florence2_base_processor_manifest()


def validate_florence2_base_weight_asset(
    checkpoint_dir: str | os.PathLike[str],
    *,
    revision: str,
) -> dict[str, Any]:
    """Validate the only accepted source checkpoint payload."""

    if revision != FLORENCE2_BASE_REVISION:
        raise ValueError(
            "Florence-2-base Core ML requires weight revision "
            f"{FLORENCE2_BASE_REVISION}, got {revision!r}."
        )
    path = Path(checkpoint_dir) / FLORENCE2_BASE_WEIGHTS_FILENAME
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            f"Florence checkpoint is missing {FLORENCE2_BASE_WEIGHTS_FILENAME!r}."
        )
    actual_size = int(path.stat().st_size)
    if actual_size != FLORENCE2_BASE_WEIGHTS_SIZE:
        raise ValueError(
            "Florence weight asset has the wrong byte length: "
            f"expected {FLORENCE2_BASE_WEIGHTS_SIZE}, got {actual_size}."
        )
    actual_hash = _file_sha256(path)
    if not hmac.compare_digest(actual_hash, FLORENCE2_BASE_WEIGHTS_SHA256):
        raise ValueError(
            "Florence weight asset failed SHA-256 validation: "
            f"expected {FLORENCE2_BASE_WEIGHTS_SHA256}, got {actual_hash}."
        )
    return florence2_base_weights_manifest()


def build_florence_encoder_masks(
    attention_mask: Any,
    *,
    profile: FlorenceCoreMLProfile | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build canonical additive encoder and three-beam cross masks."""

    resolved = profile or florence2_base_coreml_profile()
    values = np.asarray(attention_mask)
    expected = (1, resolved.encoder_context_length)
    if values.shape != expected:
        raise ValueError(
            f"Florence attention_mask must have shape {expected}, got {values.shape}."
        )
    if values.dtype not in (np.bool_, np.int32, np.int64):
        raise ValueError("Florence attention_mask must be boolean or integer.")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("Florence attention_mask must contain only zero and one.")
    if not np.any(values):
        raise ValueError("Florence attention_mask cannot mask every encoder token.")
    flat = values.astype(bool, copy=False)
    # Padding must be one contiguous suffix.  This makes both host semantics and
    # the fixed image-prefix contract unambiguous.
    first_zero = np.flatnonzero(~flat[0])
    if first_zero.size and np.any(flat[0, int(first_zero[0]) :]):
        raise ValueError("Florence encoder padding must be a contiguous suffix.")
    minimum = np.float16(np.finfo(np.float16).min)
    encoder = np.where(flat[:, None, None, :], np.float16(0), minimum)
    cross = np.repeat(encoder, resolved.num_beams, axis=0)
    return (
        np.ascontiguousarray(encoder, dtype=np.float16),
        np.ascontiguousarray(cross, dtype=np.float16),
    )


def prepare_florence2_base_processor_batch(
    processor: Any,
    image: Any,
    class_names: list[str] | tuple[str, ...],
    *,
    profile: FlorenceCoreMLProfile | None = None,
) -> dict[str, np.ndarray]:
    """Tokenize one detection request and enforce the fixed image-token prefix."""

    resolved = profile or florence2_base_coreml_profile()
    _validate_exact_profile(resolved)
    labels = [str(value).strip() for value in class_names]
    if not labels or any(not value for value in labels):
        raise ValueError("Florence Core ML requires at least one non-empty class.")
    prompt = FLORENCE2_TASK + ", ".join(labels)
    batch = processor(
        text=prompt,
        images=image,
        return_tensors="np",
    )
    try:
        raw_ids = np.asarray(batch["input_ids"])
        raw_pixels = np.asarray(batch["pixel_values"])
        raw_attention = np.asarray(batch["attention_mask"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Pinned Florence processor did not return the required inputs."
        ) from exc
    if raw_ids.ndim != 2 or raw_ids.shape[0] != 1:
        raise ValueError(f"Florence input_ids have invalid shape {raw_ids.shape}.")
    if raw_ids.shape[1] > resolved.encoder_context_length:
        raise ValueError(
            "Florence prompt exceeds the fixed 1024-position encoder context."
        )
    if raw_attention.shape != raw_ids.shape:
        raise ValueError("Florence processor attention mask shape changed.")
    expected_pixels = (
        1,
        resolved.image_channels,
        resolved.image_size,
        resolved.image_size,
    )
    if raw_pixels.shape != expected_pixels:
        raise ValueError(
            f"Florence pixel_values must have shape {expected_pixels}, "
            f"got {raw_pixels.shape}."
        )
    if not np.isfinite(raw_pixels).all():
        raise ValueError("Florence processor emitted non-finite pixels.")
    ids64 = raw_ids.astype(np.int64, copy=False)
    if np.any(ids64 < 0) or np.any(ids64 >= resolved.vocab_size):
        raise ValueError("Florence processor emitted an out-of-vocabulary token.")
    image_positions = np.flatnonzero(ids64[0] == FLORENCE2_IMAGE_TOKEN_ID)
    expected_positions = np.arange(resolved.image_token_count)
    if not np.array_equal(image_positions, expected_positions):
        raise ValueError(
            "Florence processor must emit exactly 577 contiguous image tokens "
            "at the beginning of the encoder sequence."
        )
    padded_ids = np.full(
        (1, resolved.encoder_context_length),
        FLORENCE2_PAD_TOKEN_ID,
        dtype=np.int32,
    )
    padded_attention = np.zeros(
        (1, resolved.encoder_context_length),
        dtype=np.int32,
    )
    length = int(raw_ids.shape[1])
    padded_ids[:, :length] = ids64.astype(np.int32, copy=False)
    padded_attention[:, :length] = raw_attention.astype(np.int32, copy=False)
    encoder_mask, cross_mask = build_florence_encoder_masks(
        padded_attention,
        profile=resolved,
    )
    return {
        FLORENCE_ENCODER_INPUT_IDS_INPUT: np.ascontiguousarray(padded_ids),
        FLORENCE_PIXEL_VALUES_INPUT: np.ascontiguousarray(
            raw_pixels.astype(np.float16, copy=False)
        ),
        FLORENCE_ENCODER_ATTENTION_MASK_INPUT: encoder_mask,
        FLORENCE_CROSS_ATTENTION_MASK_INPUT: cross_mask,
        "unpadded_input_ids": np.ascontiguousarray(ids64.astype(np.int32, copy=False)),
        "prompt_length": np.asarray(length, dtype=np.int32),
    }


class FlorenceDecodeCursor:
    """Append-only single-token cursor paired with one fresh decoder state."""

    def __init__(self, profile: FlorenceCoreMLProfile | None = None):
        self.profile = profile or florence2_base_coreml_profile()
        _validate_exact_profile(self.profile)
        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    def controls(self) -> tuple[np.ndarray, np.ndarray]:
        if self._position >= self.profile.decoder_context_length:
            raise ValueError("Florence decoder context is exhausted.")
        end = self._position + 1
        mask = np.zeros(
            (self.profile.num_beams, 1, 1, end),
            dtype=np.float16,
        )
        positions = np.full(
            (self.profile.num_beams, 1),
            self._position,
            dtype=np.int32,
        )
        return np.ascontiguousarray(mask), np.ascontiguousarray(positions)

    def commit(self, *, causal_mask: Any, position_ids: Any) -> None:
        expected_mask, expected_positions = self.controls()
        actual_mask = np.asarray(causal_mask)
        actual_positions = np.asarray(position_ids)
        if not np.array_equal(actual_mask, expected_mask):
            raise ValueError("Florence causal mask is not append-only canonical.")
        if not np.array_equal(actual_positions, expected_positions):
            raise ValueError("Florence position IDs do not match the state cursor.")
        self._position += 1


def florence_coreml_function_contracts(
    profile: FlorenceCoreMLProfile | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the exact two-function Core ML ABI."""

    resolved = profile or florence2_base_coreml_profile()
    _validate_exact_profile(resolved)
    e_axis = _range_axis(
        "E_decoder",
        1,
        resolved.decoder_context_length,
        1,
    )
    single_cross = list(resolved.single_cross_cache_shape)
    states = [
        {
            "name": FLORENCE_SELF_KEY_CACHE_STATE,
            "dtype": "float16",
            "shape": list(resolved.self_cache_shape),
        },
        {
            "name": FLORENCE_SELF_VALUE_CACHE_STATE,
            "dtype": "float16",
            "shape": list(resolved.self_cache_shape),
        },
        {
            "name": FLORENCE_CROSS_KEY_CACHE_STATE,
            "dtype": "float16",
            "shape": list(resolved.cross_cache_shape),
        },
        {
            "name": FLORENCE_CROSS_VALUE_CACHE_STATE,
            "dtype": "float16",
            "shape": list(resolved.cross_cache_shape),
        },
    ]
    return {
        FLORENCE_ENCODE_FUNCTION: {
            "stateful": False,
            "inputs": [
                {
                    "name": FLORENCE_PIXEL_VALUES_INPUT,
                    "dtype": "float16",
                    "shape": [
                        _fixed_axis("N", 1),
                        _fixed_axis("C", resolved.image_channels),
                        _fixed_axis("H", resolved.image_size),
                        _fixed_axis("W", resolved.image_size),
                    ],
                },
                {
                    "name": FLORENCE_ENCODER_INPUT_IDS_INPUT,
                    "dtype": "int32",
                    "shape": [
                        _fixed_axis("N", 1),
                        _fixed_axis(
                            "E_encoder",
                            resolved.encoder_context_length,
                        ),
                    ],
                },
                {
                    "name": FLORENCE_ENCODER_ATTENTION_MASK_INPUT,
                    "dtype": "float16",
                    "shape": [
                        _fixed_axis("N", 1),
                        _fixed_axis("heads", 1),
                        _fixed_axis("Q_mask", 1),
                        _fixed_axis(
                            "E_encoder",
                            resolved.encoder_context_length,
                        ),
                    ],
                },
            ],
            "outputs": [
                {
                    "name": FLORENCE_CROSS_KEY_OUTPUT,
                    "dtype": "float16",
                    "shape": [
                        _fixed_axis(f"axis_{i}", value)
                        for i, value in enumerate(single_cross)
                    ],
                },
                {
                    "name": FLORENCE_CROSS_VALUE_OUTPUT,
                    "dtype": "float16",
                    "shape": [
                        _fixed_axis(f"axis_{i}", value)
                        for i, value in enumerate(single_cross)
                    ],
                },
            ],
            "capture": "torch_jit_trace_fixed",
        },
        FLORENCE_DECODE_FUNCTION: {
            "stateful": True,
            "inputs": [
                {
                    "name": FLORENCE_DECODER_INPUT_IDS_INPUT,
                    "dtype": "int32",
                    "shape": [
                        _fixed_axis("beams", resolved.num_beams),
                        _fixed_axis("Q", 1),
                    ],
                },
                {
                    "name": FLORENCE_CAUSAL_MASK_INPUT,
                    "dtype": "float16",
                    "shape": [
                        _fixed_axis("beams", resolved.num_beams),
                        _fixed_axis("heads", 1),
                        _fixed_axis("Q", 1),
                        dict(e_axis),
                    ],
                },
                {
                    "name": FLORENCE_CROSS_ATTENTION_MASK_INPUT,
                    "dtype": "float16",
                    "shape": [
                        _fixed_axis("beams", resolved.num_beams),
                        _fixed_axis("heads", 1),
                        _fixed_axis("Q", 1),
                        _fixed_axis(
                            "E_encoder",
                            resolved.encoder_context_length,
                        ),
                    ],
                },
                {
                    "name": FLORENCE_POSITION_IDS_INPUT,
                    "dtype": "int32",
                    "shape": [
                        _fixed_axis("beams", resolved.num_beams),
                        _fixed_axis("Q", 1),
                    ],
                },
                {
                    "name": FLORENCE_BEAM_PARENT_INDICES_INPUT,
                    "dtype": "int32",
                    "shape": [_fixed_axis("beams", resolved.num_beams)],
                },
            ],
            "outputs": [
                {
                    "name": FLORENCE_LAST_LOGITS_OUTPUT,
                    "dtype": "float16",
                    "shape": [
                        _fixed_axis("beams", resolved.num_beams),
                        _fixed_axis("V", resolved.vocab_size),
                    ],
                }
            ],
            "states": states,
            "capture": "torch_jit_trace_stateful_beam3_bounded_e",
        },
    }


def florence2_base_coreml_metadata(
    profile: FlorenceCoreMLProfile | None = None,
) -> dict[str, Any]:
    """Build the hash-bound conversion and host-generation contract."""

    resolved = profile or florence2_base_coreml_profile()
    _validate_exact_profile(resolved)
    functions = florence_coreml_function_contracts(resolved)
    generation = {
        "mode": "beam_search",
        "do_sample": False,
        "num_beams": FLORENCE2_NUM_BEAMS,
        "num_return_sequences": 1,
        "decoder_start_token_id": FLORENCE2_DECODER_START_TOKEN_ID,
        "bos_token_id": FLORENCE2_BOS_TOKEN_ID,
        "eos_token_id": FLORENCE2_EOS_TOKEN_ID,
        "pad_token_id": FLORENCE2_PAD_TOKEN_ID,
        "forced_bos_token_id": FLORENCE2_FORCED_BOS_TOKEN_ID,
        "forced_eos_token_id": FLORENCE2_FORCED_EOS_TOKEN_ID,
        "no_repeat_ngram_size": FLORENCE2_NO_REPEAT_NGRAM_SIZE,
        "early_stopping": True,
        "length_penalty": FLORENCE2_LENGTH_PENALTY,
        "max_new_tokens": resolved.max_new_tokens,
    }
    processor = florence2_base_processor_manifest()
    weights = florence2_base_weights_manifest()
    image_profile = {
        "mode": "pinned_florence_processor_fixed_square",
        "color": "rgb",
        "processed_shape": [
            1,
            resolved.image_channels,
            resolved.image_size,
            resolved.image_size,
        ],
        "image_token_id": FLORENCE2_IMAGE_TOKEN_ID,
        "image_token_count": resolved.image_token_count,
        "image_token_layout": "contiguous_prefix",
    }
    host_operations = [
        "pinned_processor_tokenize_and_preprocess",
        "image_placeholder_prefix_validation",
        "encoder_padding_and_additive_mask",
        "fresh_decode_state_per_request",
        "cross_cache_repeat_and_state_write",
        "exact_three_beam_search",
        "self_cache_parent_reorder",
        "forced_bos_and_eos",
        "no_repeat_ngram_3",
        "early_stopping_and_length_penalty",
        "processor_batch_decode",
        "processor_post_process_generation",
    ]
    execution_profile = coreml_execution_profile_metadata(
        None,
        conversion_compute_units="cpu_only",
    )
    integrity = {
        "profile": resolved.as_dict(),
        "functions": functions,
        "generation": generation,
        "processor": processor,
        "weights": weights,
        "image_profile": image_profile,
        "host_operations": host_operations,
        "execution_profile": execution_profile,
    }
    return {
        "artifact_scope": FLORENCE_COREML_ARTIFACT_SCOPE,
        "component_contract": FLORENCE_COREML_COMPONENT_CONTRACT,
        "coreml_florence_schema_version": FLORENCE_COREML_SCHEMA_VERSION,
        "coreml_multifunction": True,
        "coreml_default_function": FLORENCE_ENCODE_FUNCTION,
        "coreml_function_names": list(FLORENCE_FUNCTION_NAMES),
        "coreml_stateful_functions": [FLORENCE_DECODE_FUNCTION],
        "coreml_minimum_deployment_targets": list(
            FLORENCE_COREML_MINIMUM_DEPLOYMENT_TARGETS
        ),
        "coremltools_major": FLORENCE_COREML_REQUIRED_COREMLTOOLS_MAJOR,
        "model_family": "florence2",
        "size": "base",
        "task": "detect",
        "precision": "mixed",
        "encoder_compute_precision": "fp32",
        "decoder_compute_precision": "fp16",
        "function_io_precision": "fp16",
        "runtime_state_materialization_precision": "fp32",
        "conversion_source_precision": "fp32",
        "batch": 1,
        "beam_batch": resolved.num_beams,
        "dynamic": True,
        "weights_license": "mit",
        "artifact_redistributable": True,
        **execution_profile,
        "florence_profile": resolved.as_dict(),
        "florence_functions": functions,
        "processor": processor,
        "weights": weights,
        "generation": generation,
        "image_profile": image_profile,
        "host_operations": host_operations,
        "transformers_source": {
            "repo": "https://github.com/huggingface/transformers",
            "commit": FLORENCE_COREML_TRANSFORMERS_COMMIT,
            "version": FLORENCE_COREML_TRANSFORMERS_VERSION,
            "license": "Apache-2.0",
            "adapted_files": [
                "src/transformers/models/florence2/modeling_florence2.py",
                "src/transformers/models/bart/modeling_bart.py",
                "src/transformers/generation/utils.py",
                "src/transformers/generation/logits_process.py",
                "src/transformers/generation/stopping_criteria.py",
            ],
        },
        "coreml_florence_contract_sha256": _canonical_sha256(integrity),
    }


def stringify_florence_coreml_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    expected = florence2_base_coreml_metadata()
    if set(metadata) != set(expected):
        raise ValueError("Florence Core ML metadata keys changed.")
    values: dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            values[key] = value
        elif isinstance(value, bool):
            values[key] = json.dumps(value)
        elif isinstance(value, (dict, list)):
            values[key] = _canonical_json(value)
        else:
            values[key] = str(value)
    return values


def validate_florence_coreml_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    expected = florence2_base_coreml_metadata()
    if set(metadata) != set(expected):
        missing = sorted(set(expected) - set(metadata))
        extra = sorted(set(metadata) - set(expected))
        raise ValueError(
            f"Florence Core ML metadata keys changed: missing={missing}, extra={extra}."
        )
    result: dict[str, Any] = {}
    for key, expected_value in expected.items():
        actual = metadata[key]
        if isinstance(actual, str) and not isinstance(expected_value, str):
            try:
                if isinstance(expected_value, (dict, list, bool)):
                    actual = json.loads(actual)
                elif isinstance(expected_value, int):
                    actual = int(actual)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Florence metadata field {key!r} is malformed."
                ) from exc
        if actual != expected_value:
            raise ValueError(f"Florence metadata field {key!r} changed.")
        result[key] = actual
    return result


def _unique_parameter_count(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        total += int(parameter.numel())
    return total


def validate_florence2_base_model(model: nn.Module) -> None:
    """Reject every architecture other than the pinned native base model."""

    config = getattr(model, "config", None)
    text = getattr(config, "text_config", None)
    vision = getattr(config, "vision_config", None)
    exact_values = {
        "model_type": (getattr(config, "model_type", None), "florence2"),
        "image_token_id": (
            getattr(config, "image_token_id", None),
            FLORENCE2_IMAGE_TOKEN_ID,
        ),
        "text.vocab_size": (getattr(text, "vocab_size", None), 51_328),
        "text.d_model": (getattr(text, "d_model", None), 768),
        "text.encoder_layers": (getattr(text, "encoder_layers", None), 6),
        "text.decoder_layers": (getattr(text, "decoder_layers", None), 6),
        "text.encoder_attention_heads": (
            getattr(text, "encoder_attention_heads", None),
            12,
        ),
        "text.decoder_attention_heads": (
            getattr(text, "decoder_attention_heads", None),
            12,
        ),
        "text.encoder_ffn_dim": (
            getattr(text, "encoder_ffn_dim", None),
            3072,
        ),
        "text.decoder_ffn_dim": (
            getattr(text, "decoder_ffn_dim", None),
            3072,
        ),
        "text.max_position_embeddings": (
            getattr(text, "max_position_embeddings", None),
            1024,
        ),
        "vision.window_size": (getattr(vision, "window_size", None), 12),
        "vision.projection_dim": (
            getattr(vision, "projection_dim", None),
            768,
        ),
    }
    for name, (actual, expected) in exact_values.items():
        if actual != expected:
            raise ValueError(
                f"Florence-2-base Core ML expected {name}={expected!r}, got {actual!r}."
            )
    list_values = {
        "vision.depths": (getattr(vision, "depths", None), [1, 1, 9, 1]),
        "vision.dim_embed": (
            getattr(vision, "dim_embed", None),
            [128, 256, 512, 1024],
        ),
        "vision.num_heads": (
            getattr(vision, "num_heads", None),
            [4, 8, 16, 32],
        ),
        "vision.num_groups": (
            getattr(vision, "num_groups", None),
            [4, 8, 16, 32],
        ),
        "vision.patch_stride": (
            getattr(vision, "patch_stride", None),
            [4, 2, 2, 2],
        ),
    }
    for name, (actual, expected) in list_values.items():
        if list(actual or ()) != expected:
            raise ValueError(
                f"Florence-2-base Core ML expected {name}={expected!r}, got {actual!r}."
            )
    base = getattr(model, "model", None)
    language = getattr(base, "language_model", None)
    shared = getattr(language, "shared", None)
    encoder = getattr(language, "encoder", None)
    decoder = getattr(language, "decoder", None)
    if any(value is None for value in (base, language, shared, encoder, decoder)):
        raise ValueError("Florence model is missing the native module tree.")
    shared_weight = shared.weight
    tied = (
        getattr(model, "lm_head", None).weight,
        encoder.embed_tokens.weight,
        decoder.embed_tokens.weight,
    )
    if any(value is not shared_weight for value in tied):
        raise ValueError("Florence BART embedding/language-head weights are not tied.")
    parameter_count = _unique_parameter_count(model)
    if parameter_count != FLORENCE2_BASE_UNIQUE_PARAMETER_COUNT:
        raise ValueError(
            "Florence-2-base unique parameter count changed: "
            f"expected {FLORENCE2_BASE_UNIQUE_PARAMETER_COUNT}, "
            f"got {parameter_count}."
        )


def validate_florence2_base_model_weight_values(
    model: nn.Module,
    checkpoint_dir: str | os.PathLike[str],
) -> None:
    """Compare every source tensor and the three explicit tied aliases."""

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "Florence Core ML weight validation requires safetensors."
        ) from exc
    path = Path(checkpoint_dir) / FLORENCE2_BASE_WEIGHTS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Florence weight payload does not exist: {path}.")
    actual_state = model.state_dict()
    with safe_open(path, framework="pt", device="cpu") as source:
        source_keys = set(source.keys())
        actual_keys = set(actual_state)
        if source_keys - actual_keys:
            raise ValueError(
                "In-memory Florence model is missing source tensors: "
                f"{sorted(source_keys - actual_keys)[:5]}."
            )
        extras = actual_keys - source_keys
        if extras != _FLORENCE_TIED_WEIGHT_ALIASES:
            raise ValueError(
                f"In-memory Florence state has unexpected aliases: {sorted(extras)}."
            )
        for name in sorted(source_keys):
            expected = source.get_tensor(name)
            actual = actual_state[name].detach()
            if actual.device.type != "cpu":
                raise ValueError(
                    "Florence weight validation requires a CPU model, "
                    f"got {name!r} on {actual.device}."
                )
            if (
                actual.shape != expected.shape
                or expected.dtype != torch.float16
                or actual.dtype != torch.float32
                # The pinned safetensors payload is entirely FP16. Core ML
                # conversion deliberately loads it as FP32; half-to-float
                # widening is exact, so require bit-exact widened values.
                or not torch.equal(actual, expected.to(dtype=torch.float32))
            ):
                raise ValueError(
                    "In-memory Florence tensor differs from the pinned "
                    f"checkpoint: {name!r}."
                )
    shared = actual_state["model.language_model.shared.weight"]
    for name in _FLORENCE_TIED_WEIGHT_ALIASES:
        if not torch.equal(actual_state[name], shared):
            raise ValueError(f"Florence tied alias {name!r} differs from shared.")


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apache-2.0 Transformers eager attention with dropout disabled."""

    weights = torch.matmul(query, key.transpose(2, 3)) * float(scaling)
    if attention_mask is not None:
        weights = weights + attention_mask
    probabilities = F.softmax(weights, dim=-1)
    return torch.matmul(probabilities, value).transpose(1, 2).contiguous()


class Florence2StaticVisionProjector(nn.Module):
    """Static 768-square DaViT and 577-token multimodal projector."""

    def __init__(
        self,
        vision_tower: nn.Module,
        projector: nn.Module,
        *,
        image_size: int,
    ):
        super().__init__()
        self.convs = vision_tower.convs
        self.blocks = vision_tower.blocks
        self.image_projection = projector.image_projection
        self.image_proj_norm = projector.image_proj_norm
        self.row_embeddings = projector.image_position_embed.row_embeddings
        self.column_embeddings = projector.image_position_embed.column_embeddings
        self.image_size = _strict_positive_int(image_size, name="image_size")

        sizes: list[int] = []
        current = self.image_size
        for conv in self.convs:
            kernel = int(conv.conv.kernel_size[0])
            stride = int(conv.conv.stride[0])
            padding = int(conv.conv.padding[0])
            current = (current + 2 * padding - kernel) // stride + 1
            sizes.append(current)
        self.stage_sizes = tuple(sizes)
        self.stage_channels = tuple(int(conv.conv.out_channels) for conv in self.convs)
        for stage_size, stage in zip(self.stage_sizes, self.blocks):
            for block in stage:
                window_size = int(block.spatial_block.window_attn.window_size)
                if stage_size % window_size:
                    raise ValueError(
                        "Static Florence stage is not divisible by its window."
                    )
        final_size = self.stage_sizes[-1]
        if final_size > int(self.row_embeddings.num_embeddings):
            raise ValueError("Florence position embedding table is too short.")
        positions = torch.arange(final_size, dtype=torch.long)
        self.register_buffer("row_positions", positions, persistent=False)
        self.register_buffer(
            "column_positions",
            positions.clone(),
            persistent=False,
        )
        temporal = projector.visual_temporal_embed.pos_idx_to_embed[:1, :]
        self.register_buffer(
            "temporal_embedding",
            temporal.detach().clone(),
            persistent=False,
        )
        self.image_token_count = final_size * final_size + 1
        self.hidden_size = int(self.image_projection.out_features)

    @staticmethod
    def _conv_embed(conv: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        if getattr(conv, "norm", None) is not None and bool(conv.pre_norm):
            hidden = conv.norm(hidden.permute(0, 2, 3, 1))
            hidden = hidden.permute(0, 3, 1, 2)
        hidden = conv.conv(hidden)
        if getattr(conv, "norm", None) is not None and not bool(conv.pre_norm):
            hidden = conv.norm(hidden.permute(0, 2, 3, 1))
            hidden = hidden.permute(0, 3, 1, 2)
        return hidden

    @staticmethod
    def _window_attention(
        attention: nn.Module,
        hidden: torch.Tensor,
        *,
        height: int,
        width: int,
        channels: int,
    ) -> torch.Tensor:
        window = int(attention.window_size)
        windows = (height // window) * (width // window)
        tokens = window * window
        values = hidden.view(
            1,
            height // window,
            window,
            width // window,
            window,
            channels,
        )
        values = values.permute(0, 1, 3, 2, 4, 5).contiguous()
        values = values.view(windows, tokens, channels)
        heads = int(attention.num_heads)
        head_dim = channels // heads
        qkv = attention.qkv(values).reshape(
            windows,
            tokens,
            3,
            heads,
            head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        attended = _attention(
            qkv[0],
            qkv[1],
            qkv[2],
            scaling=float(attention.scale),
        )
        attended = attention.proj(attended.reshape(windows, tokens, channels))
        attended = attended.view(
            1,
            height // window,
            width // window,
            window,
            window,
            channels,
        )
        attended = attended.permute(0, 1, 3, 2, 4, 5).contiguous()
        return attended.view(1, height * width, channels)

    @staticmethod
    def _channel_attention(
        attention: nn.Module,
        hidden: torch.Tensor,
        *,
        tokens: int,
        channels: int,
    ) -> torch.Tensor:
        groups = int(attention.groups)
        group_width = channels // groups
        qkv = attention.qkv(hidden).reshape(
            1,
            tokens,
            3,
            groups,
            group_width,
        )
        qkv = qkv.permute(2, 0, 3, 4, 1)
        attended = _attention(
            qkv[0],
            qkv[1],
            qkv[2],
            scaling=float(tokens**-0.5),
        )
        attended = attended.permute(0, 3, 2, 1)
        return attention.proj(attended.reshape(1, tokens, channels))

    def _spatial_block(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
        *,
        size: int,
        channels: int,
    ) -> torch.Tensor:
        hidden = block.conv1(hidden) + hidden
        sequence = hidden.flatten(2).transpose(1, 2)
        residual = sequence
        normalized = block.norm1(sequence).view(1, size, size, channels)
        attended = self._window_attention(
            block.window_attn,
            normalized,
            height=size,
            width=size,
            channels=channels,
        )
        sequence = residual + attended
        hidden = sequence.transpose(1, 2).view(1, channels, size, size)
        hidden = block.conv2(hidden) + hidden
        sequence = hidden.flatten(2).transpose(1, 2)
        residual = sequence
        sequence = residual + block.ffn(block.norm2(sequence))
        return sequence.transpose(1, 2).view(1, channels, size, size)

    def _channel_block(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
        *,
        size: int,
        channels: int,
    ) -> torch.Tensor:
        tokens = size * size
        hidden = block.conv1(hidden) + hidden
        sequence = hidden.flatten(2).transpose(1, 2)
        residual = sequence
        attended = self._channel_attention(
            block.channel_attn,
            block.norm1(sequence),
            tokens=tokens,
            channels=channels,
        )
        sequence = residual + attended
        hidden = sequence.transpose(1, 2).view(1, channels, size, size)
        hidden = block.conv2(hidden) + hidden
        sequence = hidden.flatten(2).transpose(1, 2)
        residual = sequence
        sequence = residual + block.ffn(block.norm2(sequence))
        return sequence.transpose(1, 2).view(1, channels, size, size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = pixel_values
        for stage_index, (conv, stage) in enumerate(zip(self.convs, self.blocks)):
            size = self.stage_sizes[stage_index]
            channels = self.stage_channels[stage_index]
            hidden = self._conv_embed(conv, hidden)
            for block in stage:
                hidden = self._spatial_block(
                    block.spatial_block,
                    hidden,
                    size=size,
                    channels=channels,
                )
                hidden = self._channel_block(
                    block.channel_block,
                    hidden,
                    size=size,
                    channels=channels,
                )

        size = self.stage_sizes[-1]
        x_embedding = self.column_embeddings(self.column_positions)
        y_embedding = self.row_embeddings(self.row_positions)
        position = torch.cat(
            (
                x_embedding.unsqueeze(0).repeat(size, 1, 1),
                y_embedding.unsqueeze(1).repeat(1, size, 1),
            ),
            dim=-1,
        )
        position = position.permute(2, 0, 1).unsqueeze(0)
        position_features = (hidden + position).flatten(2).transpose(1, 2)
        visual = position_features + self.temporal_embedding.unsqueeze(1)
        visual = visual.unsqueeze(1)
        spatial = visual.mean(dim=2)
        temporal = visual.mean(dim=1)
        image_features = torch.cat((spatial, temporal), dim=1)
        return self.image_proj_norm(self.image_projection(image_features))


class Florence2StaticBartEncoder(nn.Module):
    """BART encoder with explicit fixed-shape eager attention."""

    def __init__(self, encoder: nn.Module, *, sequence_length: int):
        super().__init__()
        self.embed_positions = encoder.embed_positions
        self.layernorm_embedding = encoder.layernorm_embedding
        self.layers = encoder.layers
        self.sequence_length = _strict_positive_int(
            sequence_length,
            name="sequence_length",
        )
        self.hidden_size = int(encoder.config.d_model)
        self.num_heads = int(encoder.config.encoder_attention_heads)
        self.head_dim = self.hidden_size // self.num_heads
        positions = torch.arange(self.sequence_length, dtype=torch.long) + int(
            self.embed_positions.offset
        )
        self.register_buffer("position_ids", positions.unsqueeze(0), persistent=False)

    def _self_attention(
        self,
        attention: nn.Module,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        query = attention.q_proj(hidden).reshape(
            1,
            self.sequence_length,
            self.num_heads,
            self.head_dim,
        )
        key = attention.k_proj(hidden).reshape(
            1,
            self.sequence_length,
            self.num_heads,
            self.head_dim,
        )
        value = attention.v_proj(hidden).reshape(
            1,
            self.sequence_length,
            self.num_heads,
            self.head_dim,
        )
        attended = _attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            scaling=float(attention.scaling),
            attention_mask=mask,
        )
        attended = attended.reshape(
            1,
            self.sequence_length,
            self.hidden_size,
        )
        return attention.out_proj(attended)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = F.embedding(self.position_ids, self.embed_positions.weight)
        hidden = self.layernorm_embedding(inputs_embeds + positions)
        for layer in self.layers:
            residual = hidden
            hidden = layer.self_attn_layer_norm(
                residual
                + self._self_attention(
                    layer.self_attn,
                    hidden,
                    attention_mask,
                )
            )
            residual = hidden
            hidden = layer.final_layer_norm(
                residual + layer.fc2(layer.activation_fn(layer.fc1(hidden)))
            )
        return hidden


class Florence2CoreMLEncoder(nn.Module):
    """Complete fixed Florence encoder returning packed cross-attention K/V."""

    def __init__(
        self,
        model: nn.Module,
        *,
        profile: FlorenceCoreMLProfile,
    ):
        super().__init__()
        base = model.model
        language = base.language_model
        self.profile = profile
        self.vision = Florence2StaticVisionProjector(
            base.vision_tower,
            base.multi_modal_projector,
            image_size=profile.image_size,
        )
        self.embed_tokens = language.shared
        self.encoder = Florence2StaticBartEncoder(
            language.encoder,
            sequence_length=profile.encoder_context_length,
        )
        self.cross_attentions = nn.ModuleList(
            [layer.encoder_attn for layer in language.decoder.layers]
        )
        if self.vision.image_token_count != profile.image_token_count:
            raise ValueError("Florence visual token count conflicts with profile.")

    def forward(
        self,
        pixel_values: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.vision(pixel_values)
        token_features = self.embed_tokens(encoder_input_ids)
        hidden = torch.cat(
            (
                image_features,
                token_features[:, self.profile.image_token_count :, :],
            ),
            dim=1,
        )
        encoded = self.encoder(hidden, encoder_attention_mask)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for attention in self.cross_attentions:
            key = attention.k_proj(encoded).reshape(
                1,
                self.profile.encoder_context_length,
                self.profile.num_attention_heads,
                self.profile.head_dim,
            )
            value = attention.v_proj(encoded).reshape(
                1,
                self.profile.encoder_context_length,
                self.profile.num_attention_heads,
                self.profile.head_dim,
            )
            keys.append(key.transpose(1, 2))
            values.append(value.transpose(1, 2))
        return torch.stack(keys, dim=0), torch.stack(values, dim=0)


class Florence2CoreMLStatefulDecoder(nn.Module):
    """One-token, three-beam BART decoder with four aggregate Core ML states."""

    def __init__(
        self,
        model: nn.Module,
        *,
        profile: FlorenceCoreMLProfile,
        state_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        decoder = model.model.language_model.decoder
        self.embed_tokens = decoder.embed_tokens
        self.embed_positions = decoder.embed_positions
        self.layernorm_embedding = decoder.layernorm_embedding
        self.layers = decoder.layers
        self.lm_head = model.lm_head
        self.profile = profile
        dtype = state_dtype or next(decoder.parameters()).dtype
        if dtype not in (torch.float16, torch.float32):
            raise ValueError("Florence decoder state must be float16 or float32.")
        self.register_buffer(
            FLORENCE_SELF_KEY_CACHE_STATE,
            torch.zeros(profile.self_cache_shape, dtype=dtype),
        )
        self.register_buffer(
            FLORENCE_SELF_VALUE_CACHE_STATE,
            torch.zeros(profile.self_cache_shape, dtype=dtype),
        )
        self.register_buffer(
            FLORENCE_CROSS_KEY_CACHE_STATE,
            torch.zeros(profile.cross_cache_shape, dtype=dtype),
        )
        self.register_buffer(
            FLORENCE_CROSS_VALUE_CACHE_STATE,
            torch.zeros(profile.cross_cache_shape, dtype=dtype),
        )

    def reset_state(self) -> None:
        self.self_key_cache.zero_()
        self.self_value_cache.zero_()
        self.cross_key_cache.zero_()
        self.cross_value_cache.zero_()

    def initialize_cross_cache(
        self,
        key_values: torch.Tensor,
        value_values: torch.Tensor,
    ) -> None:
        expected = self.profile.single_cross_cache_shape
        if tuple(key_values.shape) != expected or tuple(value_values.shape) != expected:
            raise ValueError("Florence single-beam cross cache shape changed.")
        repeated_keys = key_values.repeat(1, self.profile.num_beams, 1, 1, 1)
        repeated_values = value_values.repeat(
            1,
            self.profile.num_beams,
            1,
            1,
            1,
        )
        self.cross_key_cache.copy_(repeated_keys)
        self.cross_value_cache.copy_(repeated_values)

    def _bart_attention(
        self,
        attention: nn.Module,
        hidden: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        query = attention.q_proj(hidden).reshape(
            self.profile.num_beams,
            1,
            self.profile.num_attention_heads,
            self.profile.head_dim,
        )
        attended = _attention(
            query.transpose(1, 2),
            key,
            value,
            scaling=float(attention.scaling),
            attention_mask=mask,
        )
        attended = attended.reshape(
            self.profile.num_beams,
            1,
            self.profile.hidden_size,
        )
        return attention.out_proj(attended)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        causal_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        beam_parent_indices: torch.Tensor,
    ) -> torch.Tensor:
        parent_indices = beam_parent_indices.to(dtype=torch.long)
        reordered_keys = torch.index_select(
            self.self_key_cache,
            1,
            parent_indices,
        )
        reordered_values = torch.index_select(
            self.self_value_cache,
            1,
            parent_indices,
        )
        # Core ML Tools recognizes full-slice assignment as a state update.
        # A raw buffer.copy_() is rejected by its tensor-assignment pass.
        self.self_key_cache[:, :, :, :, :] = reordered_keys
        self.self_value_cache[:, :, :, :, :] = reordered_values

        end_step = causal_mask.shape[-1]
        begin_step = end_step - 1
        positions = F.embedding(
            position_ids + int(self.embed_positions.offset),
            self.embed_positions.weight,
        )
        hidden = self.layernorm_embedding(
            self.embed_tokens(decoder_input_ids) + positions
        )
        for layer_index, layer in enumerate(self.layers):
            residual = hidden
            attention = layer.self_attn
            key = attention.k_proj(hidden).reshape(
                self.profile.num_beams,
                1,
                self.profile.num_attention_heads,
                self.profile.head_dim,
            )
            value = attention.v_proj(hidden).reshape(
                self.profile.num_beams,
                1,
                self.profile.num_attention_heads,
                self.profile.head_dim,
            )
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            self.self_key_cache[
                layer_index,
                :,
                :,
                begin_step:end_step,
                :,
            ] = key
            self.self_value_cache[
                layer_index,
                :,
                :,
                begin_step:end_step,
                :,
            ] = value
            cached_key = self.self_key_cache[
                layer_index,
                :,
                :,
                :end_step,
                :,
            ]
            cached_value = self.self_value_cache[
                layer_index,
                :,
                :,
                :end_step,
                :,
            ]
            hidden = layer.self_attn_layer_norm(
                residual
                + self._bart_attention(
                    attention,
                    hidden,
                    cached_key,
                    cached_value,
                    causal_mask,
                )
            )

            residual = hidden
            hidden = layer.encoder_attn_layer_norm(
                residual
                + self._bart_attention(
                    layer.encoder_attn,
                    hidden,
                    self.cross_key_cache[layer_index],
                    self.cross_value_cache[layer_index],
                    cross_attention_mask,
                )
            )
            residual = hidden
            hidden = layer.final_layer_norm(
                residual + layer.fc2(layer.activation_fn(layer.fc1(hidden)))
            )
        return self.lm_head(hidden[:, -1, :])


def wrap_florence2_base_coreml_components(
    model: nn.Module,
    *,
    profile: FlorenceCoreMLProfile | None = None,
) -> dict[str, nn.Module]:
    """Prepare the exact two production component graphs."""

    resolved = profile or florence2_base_coreml_profile()
    _validate_exact_profile(resolved)
    validate_florence2_base_model(model)
    decoder = Florence2CoreMLStatefulDecoder(
        model,
        profile=resolved,
        state_dtype=next(model.parameters()).dtype,
    )
    return {
        FLORENCE_ENCODE_FUNCTION: Florence2CoreMLEncoder(
            model,
            profile=resolved,
        ).eval(),
        FLORENCE_DECODE_FUNCTION: decoder.eval(),
    }


def assert_florence2_static_encoder_parity(
    model: nn.Module,
    wrapper: Florence2CoreMLEncoder,
    pixel_values: torch.Tensor,
    encoder_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    relative_tolerance: float = FLORENCE_COREML_SOURCE_REL_TOL,
) -> dict[str, float]:
    """Compare the full static vision/encoder/cross-projection path."""

    profile = wrapper.profile
    expected_pixels = (
        1,
        profile.image_channels,
        profile.image_size,
        profile.image_size,
    )
    if tuple(pixel_values.shape) != expected_pixels:
        raise ValueError(f"Florence encoder parity pixels must be {expected_pixels}.")
    expected_ids = (1, profile.encoder_context_length)
    if tuple(encoder_input_ids.shape) != expected_ids:
        raise ValueError(f"Florence encoder parity IDs must be {expected_ids}.")
    if tuple(attention_mask.shape) != expected_ids:
        raise ValueError("Florence encoder parity attention mask shape changed.")
    encoder_mask, _ = build_florence_encoder_masks(
        attention_mask.detach().cpu().numpy(),
        profile=profile,
    )
    additive = torch.from_numpy(encoder_mask).to(
        device=pixel_values.device,
        dtype=pixel_values.dtype,
    )
    language = model.model.language_model
    with torch.inference_mode():
        image_features = model.model.get_image_features(
            pixel_values,
        ).pooler_output
        token_features = language.shared(encoder_input_ids)
        source_embeddings = torch.cat(
            (
                image_features,
                token_features[:, profile.image_token_count :, :],
            ),
            dim=1,
        )
        source_hidden = language.encoder(
            inputs_embeds=source_embeddings,
            attention_mask=attention_mask,
        ).last_hidden_state
        source_keys: list[torch.Tensor] = []
        source_values: list[torch.Tensor] = []
        for layer in language.decoder.layers:
            key = layer.encoder_attn.k_proj(source_hidden).reshape(
                1,
                profile.encoder_context_length,
                profile.num_attention_heads,
                profile.head_dim,
            )
            value = layer.encoder_attn.v_proj(source_hidden).reshape(
                1,
                profile.encoder_context_length,
                profile.num_attention_heads,
                profile.head_dim,
            )
            source_keys.append(key.transpose(1, 2))
            source_values.append(value.transpose(1, 2))
        expected_key = torch.stack(source_keys, dim=0)
        expected_value = torch.stack(source_values, dim=0)
        actual_key, actual_value = wrapper(
            pixel_values,
            encoder_input_ids,
            additive,
        )
    metrics: dict[str, float] = {}
    for name, expected, actual in (
        ("cross_key", expected_key, actual_key),
        ("cross_value", expected_value, actual_value),
    ):
        scale = max(float(expected.abs().max()), 1e-12)
        error = float((actual - expected).abs().max()) / scale
        metrics[f"{name}_relative_max_error"] = error
        if error > float(relative_tolerance):
            raise RuntimeError(
                f"Florence static encoder {name} relative error {error:.3e} "
                f"exceeds {float(relative_tolerance):.0e}."
            )
    return metrics


def assert_florence2_decoder_source_parity(
    model: nn.Module,
    wrapper: Florence2CoreMLStatefulDecoder,
    *,
    profile: FlorenceCoreMLProfile,
    relative_tolerance: float = FLORENCE_COREML_SOURCE_REL_TOL,
) -> dict[str, float]:
    """Compare two stateful steps, including duplicated parent-beam reorder."""

    try:
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache
    except ImportError as exc:
        raise ImportError(
            "Florence decoder parity requires transformers "
            f"{FLORENCE_COREML_TRANSFORMERS_VERSION}."
        ) from exc
    parameter = next(model.parameters())
    generator = torch.Generator(device=parameter.device)
    generator.manual_seed(731)
    encoder_hidden = torch.randn(
        (
            1,
            profile.encoder_context_length,
            profile.hidden_size,
        ),
        generator=generator,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for layer in model.model.language_model.decoder.layers:
        key = layer.encoder_attn.k_proj(encoder_hidden).reshape(
            1,
            profile.encoder_context_length,
            profile.num_attention_heads,
            profile.head_dim,
        )
        value = layer.encoder_attn.v_proj(encoder_hidden).reshape(
            1,
            profile.encoder_context_length,
            profile.num_attention_heads,
            profile.head_dim,
        )
        keys.append(key.transpose(1, 2))
        values.append(value.transpose(1, 2))
    packed_key = torch.stack(keys, dim=0)
    packed_value = torch.stack(values, dim=0)
    encoder_hidden = encoder_hidden.repeat(profile.num_beams, 1, 1)
    encoder_mask = torch.ones(
        (profile.num_beams, profile.encoder_context_length),
        dtype=torch.long,
        device=parameter.device,
    )
    _, cross_mask = build_florence_encoder_masks(
        np.ones((1, profile.encoder_context_length), dtype=np.int32),
        profile=profile,
    )
    cross_mask_tensor = torch.from_numpy(cross_mask).to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    source_cache = EncoderDecoderCache(
        DynamicCache(config=model.config.text_config),
        DynamicCache(config=model.config.text_config),
    )
    wrapper.reset_state()
    wrapper.initialize_cross_cache(packed_key, packed_value)
    metrics: dict[str, float] = {}
    try:
        steps = (
            (
                "initial",
                torch.full(
                    (profile.num_beams, 1),
                    FLORENCE2_DECODER_START_TOKEN_ID,
                    dtype=torch.long,
                    device=parameter.device,
                ),
                torch.arange(
                    profile.num_beams,
                    dtype=torch.long,
                    device=parameter.device,
                ),
            ),
            (
                "reordered",
                torch.full(
                    (profile.num_beams, 1),
                    FLORENCE2_FORCED_BOS_TOKEN_ID,
                    dtype=torch.long,
                    device=parameter.device,
                ),
                torch.tensor(
                    [0, 0, profile.num_beams - 1],
                    dtype=torch.long,
                    device=parameter.device,
                ),
            ),
        )
        for step_index, (label, tokens, parents) in enumerate(steps):
            if step_index:
                source_cache.reorder_cache(parents)
            source_attention = torch.ones(
                (profile.num_beams, step_index + 1),
                dtype=torch.long,
                device=parameter.device,
            )
            with torch.inference_mode():
                source = model.model.language_model.decoder(
                    input_ids=tokens,
                    attention_mask=source_attention,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=encoder_mask,
                    past_key_values=source_cache,
                    use_cache=True,
                ).last_hidden_state
                expected = model.lm_head(source[:, -1, :])
                causal = torch.zeros(
                    (profile.num_beams, 1, 1, step_index + 1),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                positions = torch.full(
                    (profile.num_beams, 1),
                    step_index,
                    dtype=torch.int32,
                    device=parameter.device,
                )
                actual = wrapper(
                    tokens,
                    causal,
                    cross_mask_tensor,
                    positions,
                    parents,
                )
            scale = max(float(expected.abs().max()), 1e-12)
            error = float((actual - expected).abs().max()) / scale
            metrics[f"{label}_relative_max_error"] = error
            if error > float(relative_tolerance):
                raise RuntimeError(
                    f"Florence decoder {label} relative error {error:.3e} "
                    f"exceeds {float(relative_tolerance):.0e}."
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


def require_florence_coreml_toolchain(coremltools_module: Any) -> None:
    version = getattr(coremltools_module, "__version__", "")
    if _parse_major(version) != FLORENCE_COREML_REQUIRED_COREMLTOOLS_MAJOR:
        raise RuntimeError(
            f"Florence Core ML export requires coremltools 9.x, found {version!r}."
        )


def require_florence_transformers_toolchain(
    transformers_module: Any | None = None,
) -> Any:
    if transformers_module is None:
        try:
            import transformers as transformers_module
        except ImportError as exc:
            raise ImportError(
                "Florence Core ML export requires transformers "
                f"{FLORENCE_COREML_TRANSFORMERS_VERSION}."
            ) from exc
    version = str(getattr(transformers_module, "__version__", ""))
    if version != FLORENCE_COREML_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Florence Core ML graph semantics are pinned to transformers "
            f"{FLORENCE_COREML_TRANSFORMERS_VERSION}, found {version!r}."
        )
    return transformers_module


def _feature_ranges(feature: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for axis in feature["shape"]:
        if axis["kind"] == "fixed":
            value = int(axis["value"])
            ranges.append((value, value))
        elif axis["kind"] == "range":
            ranges.append(
                (
                    int(axis["lower_bound"]),
                    int(axis["upper_bound"]),
                )
            )
        else:
            raise ValueError(f"Unknown Florence Core ML axis {axis!r}.")
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
    }[feature["dtype"]]
    shape: list[Any] = []
    for axis in feature["shape"]:
        if axis["kind"] == "fixed":
            shape.append(int(axis["value"]))
        else:
            name = str(axis["name"])
            dimension = symbols.get(name)
            if dimension is None:
                dimension = ct.RangeDim(
                    lower_bound=int(axis["lower_bound"]),
                    upper_bound=int(axis["upper_bound"]),
                    default=int(axis["default"]),
                    symbol=name,
                )
                symbols[name] = dimension
            shape.append(dimension)
    return ct.TensorType(
        name=feature["name"],
        shape=tuple(shape),
        dtype=dtype,
    )


def _capture_florence_component(
    component: nn.Module,
    *,
    function_name: str,
    profile: FlorenceCoreMLProfile,
) -> Any:
    dtype = next(component.parameters()).dtype
    if function_name == FLORENCE_ENCODE_FUNCTION:
        ids = torch.full(
            (1, profile.encoder_context_length),
            FLORENCE2_PAD_TOKEN_ID,
            dtype=torch.int32,
        )
        ids[:, : profile.image_token_count] = FLORENCE2_IMAGE_TOKEN_ID
        probes = (
            torch.full(
                (
                    1,
                    profile.image_channels,
                    profile.image_size,
                    profile.image_size,
                ),
                0.125,
                dtype=dtype,
            ),
            ids,
            torch.zeros(
                (1, 1, 1, profile.encoder_context_length),
                dtype=dtype,
            ),
        )
        captured = torch.jit.trace(component, probes, check_trace=True)
        if any(node.kind() == "aten::Int" for node in captured.inlined_graph.nodes()):
            raise RuntimeError(
                "Florence fixed encoder trace retained a dynamic aten::Int."
            )
        return captured
    if function_name != FLORENCE_DECODE_FUNCTION:
        raise ValueError(f"Unknown Florence function {function_name!r}.")
    if not isinstance(component, Florence2CoreMLStatefulDecoder):
        raise TypeError("Florence decode component must expose explicit state.")
    component.reset_state()
    probes = (
        torch.full(
            (profile.num_beams, 1),
            FLORENCE2_DECODER_START_TOKEN_ID,
            dtype=torch.int32,
        ),
        torch.zeros((profile.num_beams, 1, 1, 2), dtype=dtype),
        torch.zeros(
            (
                profile.num_beams,
                1,
                1,
                profile.encoder_context_length,
            ),
            dtype=dtype,
        ),
        torch.ones((profile.num_beams, 1), dtype=torch.int32),
        torch.arange(profile.num_beams, dtype=torch.int32),
    )
    captured = torch.jit.trace(
        component,
        probes,
        check_trace=False,
        strict=False,
    )
    for name in (
        FLORENCE_SELF_KEY_CACHE_STATE,
        FLORENCE_SELF_VALUE_CACHE_STATE,
        FLORENCE_CROSS_KEY_CACHE_STATE,
        FLORENCE_CROSS_VALUE_CACHE_STATE,
    ):
        if hasattr(captured, name):
            getattr(captured, name).zero_()
    component.reset_state()
    return captured


def _compute_unit(ct: Any, value: str) -> Any:
    normalized = str(value).strip().lower()
    choices = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    if normalized not in choices:
        raise ValueError(
            f"Invalid Core ML compute_units {value!r}; "
            f"expected one of {sorted(choices)}."
        )
    return choices[normalized]


def _protobuf_ranges(feature: Any) -> tuple[tuple[int, int], ...]:
    array = feature.type.multiArrayType
    if array.WhichOneof("ShapeFlexibility") == "shapeRange":
        return tuple(
            (int(axis.lowerBound), int(axis.upperBound))
            for axis in array.shapeRange.sizeRanges
        )
    return tuple((int(value), int(value)) for value in array.shape)


def validate_florence_function_description(
    description: Any,
    *,
    function_name: str,
    profile: FlorenceCoreMLProfile,
) -> None:
    contract = florence_coreml_function_contracts(profile)[function_name]
    inputs = list(getattr(description, "input", ()) or ())
    outputs = list(getattr(description, "output", ()) or ())
    if [str(value.name) for value in inputs] != [
        value["name"] for value in contract["inputs"]
    ]:
        raise RuntimeError(f"Florence {function_name!r} input names changed.")
    if [str(value.name) for value in outputs] != [
        value["name"] for value in contract["outputs"]
    ]:
        raise RuntimeError(f"Florence {function_name!r} output names changed.")
    dtype_codes = {"float16": 65552, "int32": 131104}
    for actual, expected in zip(inputs, contract["inputs"]):
        if int(actual.type.multiArrayType.dataType) != dtype_codes[expected["dtype"]]:
            raise RuntimeError(f"Florence {function_name!r} input dtype changed.")
        if _protobuf_ranges(actual) != _feature_ranges(expected):
            raise RuntimeError(f"Florence {function_name!r} input bounds changed.")
    for actual, expected in zip(outputs, contract["outputs"]):
        if int(actual.type.multiArrayType.dataType) != dtype_codes[expected["dtype"]]:
            raise RuntimeError(f"Florence {function_name!r} output dtype changed.")
        if _protobuf_ranges(actual) != _feature_ranges(expected):
            raise RuntimeError(f"Florence {function_name!r} output bounds changed.")
    expected_states = contract.get("states", [])
    actual_states = list(getattr(description, "state", ()) or ())
    if [str(value.name) for value in actual_states] != [
        value["name"] for value in expected_states
    ]:
        raise RuntimeError(f"Florence {function_name!r} state names changed.")
    for actual, expected in zip(actual_states, expected_states):
        array = actual.type.stateType.arrayType
        if int(array.dataType) != dtype_codes["float16"]:
            raise RuntimeError(f"Florence {function_name!r} state dtype changed.")
        if tuple(int(value) for value in array.shape) != tuple(expected["shape"]):
            raise RuntimeError(f"Florence {function_name!r} state shape changed.")


def validate_florence_multifunction_spec(
    spec: Any,
    *,
    profile: FlorenceCoreMLProfile | None = None,
) -> None:
    resolved = profile or florence2_base_coreml_profile()
    _validate_exact_profile(resolved)
    description = getattr(spec, "description", None)
    if list(getattr(description, "input", ()) or ()) or list(
        getattr(description, "output", ()) or ()
    ):
        raise RuntimeError("Florence package exposes a false top-level ABI.")
    if str(getattr(description, "defaultFunctionName", "")) != (
        FLORENCE_ENCODE_FUNCTION
    ):
        raise RuntimeError("Florence package has the wrong default function.")
    functions = list(getattr(description, "functions", ()) or ())
    names = [str(value.name) for value in functions]
    if names != list(FLORENCE_FUNCTION_NAMES):
        raise RuntimeError(f"Florence package functions changed: {names!r}.")
    for function in functions:
        validate_florence_function_description(
            function,
            function_name=str(function.name),
            profile=resolved,
        )


def _convert_florence_component(
    ct: Any,
    component: nn.Module,
    *,
    function_name: str,
    profile: FlorenceCoreMLProfile,
    compute_units: str,
) -> Any:
    contract = florence_coreml_function_contracts(profile)[function_name]
    captured = _capture_florence_component(
        component,
        function_name=function_name,
        profile=profile,
    )
    symbols: dict[str, Any] = {}
    inputs = [
        _coreml_tensor_type(ct, value, symbols=symbols) for value in contract["inputs"]
    ]
    outputs = [
        ct.TensorType(name=value["name"], dtype=np.float16)
        for value in contract["outputs"]
    ]
    arguments: dict[str, Any] = {
        "inputs": inputs,
        "outputs": outputs,
        "convert_to": "mlprogram",
        # The full Florence vision/BART encoder is numerically unstable when
        # Core ML lowers every intermediate to FP16. Keep that function in
        # FP32, while preserving the decoder's explicit FP16 state ABI.
        "compute_precision": (
            ct.precision.FLOAT32
            if function_name == FLORENCE_ENCODE_FUNCTION
            else ct.precision.FLOAT16
        ),
        "minimum_deployment_target": ct.target.iOS18,
        "compute_units": _compute_unit(ct, compute_units),
        "skip_model_load": True,
    }
    if function_name == FLORENCE_DECODE_FUNCTION:
        arguments["states"] = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=tuple(value["shape"]),
                    dtype=np.float16,
                ),
                name=value["name"],
            )
            for value in contract["states"]
        ]
    converted = ct.convert(captured, **arguments)
    del captured
    return converted


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        source.rename(destination)
        return
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
                os.fsencode(source),
                -2,
                os.fsencode(destination),
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
                os.fsencode(source),
                -100,
                os.fsencode(destination),
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
                "Florence Core ML destination already exists",
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
                "Failed to publish Florence Core ML package",
                str(destination),
            )
    raise RuntimeError(
        "The destination filesystem lacks atomic no-replace directory "
        "publication. Refusing an unsafe Florence package rename."
    )


def build_florence_coreml_multifunction_package(
    components: Mapping[str, nn.Module],
    *,
    output_path: str | os.PathLike[str],
    profile: FlorenceCoreMLProfile | None = None,
    metadata: Mapping[str, Any] | None = None,
    compute_units: str = "validated",
    coremltools_module: Any | None = None,
) -> str:
    """Convert, deduplicate, validate, and atomically publish both functions."""

    resolved_compute_units = (
        resolve_florence2_base_coreml_export_compute_units(compute_units)
    )
    resolved = profile or florence2_base_coreml_profile()
    _validate_exact_profile(resolved)
    if list(components) != list(FLORENCE_FUNCTION_NAMES):
        raise ValueError(
            "Florence components must be ordered exactly as "
            f"{list(FLORENCE_FUNCTION_NAMES)}."
        )
    expected_metadata = florence2_base_coreml_metadata(resolved)
    supplied_metadata = (
        expected_metadata if metadata is None else metadata
    )
    validated_metadata = validate_florence_coreml_metadata(supplied_metadata)
    destination = Path(output_path)
    if destination.suffix.lower() != ".mlpackage":
        raise ValueError("Florence Core ML output must end in .mlpackage.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite Florence Core ML package: {destination}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if coremltools_module is None:
        try:
            import coremltools as ct
        except ImportError as exc:
            raise ImportError(
                "Florence Core ML export requires coremltools 9.x."
            ) from exc
    else:
        ct = coremltools_module
    require_florence_coreml_toolchain(ct)
    serialized_metadata = stringify_florence_coreml_metadata(validated_metadata)
    with tempfile.TemporaryDirectory(
        prefix=".libreyolo-coreml-florence-",
        dir=str(destination.parent),
    ) as temporary:
        workspace = Path(temporary)
        descriptor = ct.utils.MultiFunctionDescriptor()
        for index, function_name in enumerate(FLORENCE_FUNCTION_NAMES):
            converted = _convert_florence_component(
                ct,
                components[function_name],
                function_name=function_name,
                profile=resolved,
                compute_units=resolved_compute_units,
            )
            validate_florence_function_description(
                converted.get_spec().description,
                function_name=function_name,
                profile=resolved,
            )
            component_path = workspace / f"{index:02d}-{function_name}.mlpackage"
            converted.save(str(component_path))
            descriptor.add_function(
                str(component_path),
                "main",
                function_name,
            )
            del converted
        descriptor.default_function_name = FLORENCE_ENCODE_FUNCTION
        merged_path = workspace / "merged.mlpackage"
        ct.utils.save_multifunction(descriptor, str(merged_path))
        merged = ct.models.MLModel(str(merged_path), skip_model_load=True)
        merged.user_defined_metadata.update(serialized_metadata)
        staged_path = workspace / "staged.mlpackage"
        merged.save(str(staged_path))
        del merged
        staged = ct.models.MLModel(str(staged_path), skip_model_load=True)
        validate_florence_multifunction_spec(
            staged.get_spec(),
            profile=resolved,
        )
        validate_florence_coreml_metadata(dict(staged.user_defined_metadata))
        del staged
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite Florence package: {destination}."
            )
        _publish_directory_no_replace(staged_path, destination)
    return str(destination)


def export_florence2_base_coreml_package(
    model: nn.Module,
    *,
    checkpoint_dir: str | os.PathLike[str],
    processor_revision: str,
    output_path: str | os.PathLike[str],
    compute_units: str = "validated",
    run_source_parity: bool = True,
) -> str:
    """Strict conversion entry point for the one pinned base checkpoint."""

    resolved_compute_units = (
        resolve_florence2_base_coreml_export_compute_units(compute_units)
    )
    require_florence_transformers_toolchain()
    profile = florence2_base_coreml_profile()
    validate_florence2_base_model(model)
    devices = {
        value.device.type
        for value in (*tuple(model.parameters()), *tuple(model.buffers()))
    }
    if devices != {"cpu"}:
        raise NotImplementedError(
            "Florence Core ML conversion requires a CPU model, "
            f"found {sorted(devices)}."
        )
    floating_dtypes = {
        value.dtype
        for value in (*tuple(model.parameters()), *tuple(model.buffers()))
        if value.is_floating_point()
    }
    if floating_dtypes != {torch.float32}:
        raise NotImplementedError(
            "Florence Core ML conversion requires an FP32-loaded model, "
            f"found {sorted(str(value) for value in floating_dtypes)}."
        )
    validate_florence2_base_processor_assets(
        checkpoint_dir,
        revision=processor_revision,
    )
    validate_florence2_base_weight_asset(
        checkpoint_dir,
        revision=processor_revision,
    )
    validate_florence2_base_model_weight_values(model, checkpoint_dir)
    training_states = tuple((module, module.training) for module in model.modules())
    try:
        components = wrap_florence2_base_coreml_components(
            model.eval(),
            profile=profile,
        )
        if run_source_parity:
            y_axis = torch.linspace(
                0.0,
                1.0,
                profile.image_size,
                dtype=torch.float32,
            ).view(1, 1, profile.image_size, 1)
            x_axis = torch.linspace(
                0.0,
                1.0,
                profile.image_size,
                dtype=torch.float32,
            ).view(1, 1, 1, profile.image_size)
            channels = torch.tensor(
                [0.0, 0.25, 0.5],
                dtype=torch.float32,
            ).view(1, profile.image_channels, 1, 1)
            pixels = (x_axis + y_axis + channels).remainder(1.0)
            input_ids = torch.full(
                (1, profile.encoder_context_length),
                4,
                dtype=torch.long,
            )
            input_ids[:, : profile.image_token_count] = FLORENCE2_IMAGE_TOKEN_ID
            attention_mask = torch.ones_like(input_ids)
            assert_florence2_static_encoder_parity(
                model,
                components[FLORENCE_ENCODE_FUNCTION],
                pixels,
                input_ids,
                attention_mask,
            )
            assert_florence2_decoder_source_parity(
                model,
                components[FLORENCE_DECODE_FUNCTION],
                profile=profile,
            )
        return build_florence_coreml_multifunction_package(
            components,
            output_path=output_path,
            profile=profile,
            metadata=florence2_base_coreml_metadata(profile),
            compute_units=resolved_compute_units,
        )
    finally:
        for module, training in training_states:
            module.training = training


__all__ = [
    "FLORENCE2_BASE_REPO",
    "FLORENCE2_BASE_REQUIRED_ASSETS",
    "FLORENCE2_BASE_REVISION",
    "FLORENCE2_BASE_WEIGHTS_FILENAME",
    "FLORENCE2_BASE_WEIGHTS_SHA256",
    "FLORENCE2_BASE_WEIGHTS_SIZE",
    "FLORENCE2_DECODER_START_TOKEN_ID",
    "FLORENCE2_EOS_TOKEN_ID",
    "FLORENCE2_IMAGE_TOKEN_ID",
    "FLORENCE2_NUM_BEAMS",
    "FLORENCE2_PAD_TOKEN_ID",
    "FLORENCE_CAUSAL_MASK_INPUT",
    "FLORENCE_COREML_COMPONENT_CONTRACT",
    "FLORENCE_COREML_TRANSFORMERS_COMMIT",
    "FLORENCE_COREML_TRANSFORMERS_VERSION",
    "FLORENCE_CROSS_ATTENTION_MASK_INPUT",
    "FLORENCE_CROSS_KEY_CACHE_STATE",
    "FLORENCE_CROSS_KEY_OUTPUT",
    "FLORENCE_CROSS_VALUE_CACHE_STATE",
    "FLORENCE_CROSS_VALUE_OUTPUT",
    "FLORENCE_DECODER_INPUT_IDS_INPUT",
    "FLORENCE_DECODE_FUNCTION",
    "FLORENCE_ENCODER_ATTENTION_MASK_INPUT",
    "FLORENCE_ENCODER_INPUT_IDS_INPUT",
    "FLORENCE_ENCODE_FUNCTION",
    "FLORENCE_FUNCTION_NAMES",
    "FLORENCE_LAST_LOGITS_OUTPUT",
    "FLORENCE_PIXEL_VALUES_INPUT",
    "FLORENCE_POSITION_IDS_INPUT",
    "FLORENCE_SELF_KEY_CACHE_STATE",
    "FLORENCE_SELF_VALUE_CACHE_STATE",
    "Florence2CoreMLEncoder",
    "Florence2CoreMLStatefulDecoder",
    "Florence2StaticBartEncoder",
    "Florence2StaticVisionProjector",
    "FlorenceCoreMLProfile",
    "FlorenceDecodeCursor",
    "assert_florence2_decoder_source_parity",
    "assert_florence2_static_encoder_parity",
    "build_florence_coreml_multifunction_package",
    "build_florence_encoder_masks",
    "export_florence2_base_coreml_package",
    "florence2_base_coreml_metadata",
    "florence2_base_coreml_profile",
    "florence2_base_processor_manifest",
    "florence2_base_weights_manifest",
    "florence_coreml_function_contracts",
    "prepare_florence2_base_processor_batch",
    "require_florence_coreml_toolchain",
    "require_florence_transformers_toolchain",
    "resolve_florence2_base_coreml_export_compute_units",
    "stringify_florence_coreml_metadata",
    "validate_florence2_base_model",
    "validate_florence2_base_model_weight_values",
    "validate_florence2_base_processor_assets",
    "validate_florence2_base_weight_asset",
    "validate_florence_coreml_metadata",
    "validate_florence_function_description",
    "validate_florence_multifunction_spec",
    "wrap_florence2_base_coreml_components",
]
