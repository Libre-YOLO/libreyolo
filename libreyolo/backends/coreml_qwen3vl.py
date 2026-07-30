"""Portable Qwen3-VL-2B Core ML bundle loader and greedy runtime.

The host-side MRoPE construction follows the Apache-2.0 Transformers 5.12.1
Qwen3-VL implementation pinned by the bundle manifest. It is intentionally
fixed to one 448-square image and a 512-token left-padded decoder prefix.
"""

from __future__ import annotations

import importlib.metadata
import itertools
import json
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..export.coreml_qwen3vl import (
    QWEN3VL_COREML_COMPONENTS,
    QWEN3VL_COREML_COMPONENT_CONTRACT,
    QWEN3VL_COREML_CONTEXT_LENGTH,
    QWEN3VL_COREML_EOS_TOKEN_IDS,
    QWEN3VL_COREML_HEAD_DIM,
    QWEN3VL_COREML_HIDDEN_SIZE,
    QWEN3VL_COREML_IMAGE_SIZE,
    QWEN3VL_COREML_IMAGE_TOKEN_ID,
    QWEN3VL_COREML_IMAGE_TOKENS,
    QWEN3VL_COREML_MAX_NEW_TOKENS,
    QWEN3VL_COREML_PAD_TOKEN_ID,
    QWEN3VL_COREML_PATCH_COUNT,
    QWEN3VL_COREML_PATCH_WIDTH,
    QWEN3VL_COREML_REPETITION_PENALTY,
    QWEN3VL_COREML_REVISION,
    QWEN3VL_COREML_TRANSFORMERS_VERSION,
    QWEN3VL_COREML_VOCAB_SIZE,
    qwen3vl_bundle_manifest,
    resolve_qwen3vl_coreml_compute_units,
    validate_qwen3vl_processor_assets,
)

COREML_QWEN3VL_BUNDLE_FORMAT = "libreyolo_coreml_qwen3vl_bundle"
COREML_QWEN3VL_BUNDLE_SUFFIX = ".coremlvlm"
_ROPE_THETA = 5_000_000.0
_MROPE_SECTION = (24, 20, 20)


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"Qwen3-VL bundle manifest repeats key {key!r}."
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Qwen3-VL bundle manifest is not valid UTF-8 JSON."
        ) from exc
    if not isinstance(value, dict):
        raise TypeError("Qwen3-VL bundle manifest must be a JSON object.")
    return value


def _validate_package_spec(model, *, component: str) -> None:
    expected = {
        "decoder": (
            {
                "input_embeddings": (
                    1,
                    QWEN3VL_COREML_CONTEXT_LENGTH,
                    QWEN3VL_COREML_HIDDEN_SIZE,
                ),
                "causal_mask": (
                    1,
                    1,
                    QWEN3VL_COREML_CONTEXT_LENGTH,
                    QWEN3VL_COREML_CONTEXT_LENGTH,
                ),
                "rope_cos": (
                    1,
                    QWEN3VL_COREML_CONTEXT_LENGTH,
                    QWEN3VL_COREML_HEAD_DIM,
                ),
                "rope_sin": (
                    1,
                    QWEN3VL_COREML_CONTEXT_LENGTH,
                    QWEN3VL_COREML_HEAD_DIM,
                ),
                "deepstack_embeddings": (
                    3,
                    1,
                    QWEN3VL_COREML_CONTEXT_LENGTH,
                    QWEN3VL_COREML_HIDDEN_SIZE,
                ),
            },
            {"last_logits": (1, QWEN3VL_COREML_VOCAB_SIZE)},
        ),
        "token_embedding": (
            {"input_ids": (1, QWEN3VL_COREML_CONTEXT_LENGTH)},
            {
                "token_embeddings": (
                    1,
                    QWEN3VL_COREML_CONTEXT_LENGTH,
                    QWEN3VL_COREML_HIDDEN_SIZE,
                )
            },
        ),
        "vision": (
            {
                "patch_values": (
                    QWEN3VL_COREML_PATCH_COUNT,
                    QWEN3VL_COREML_PATCH_WIDTH,
                )
            },
            {
                "image_embeddings": (
                    QWEN3VL_COREML_IMAGE_TOKENS,
                    QWEN3VL_COREML_HIDDEN_SIZE,
                ),
                "deepstack_0": (
                    QWEN3VL_COREML_IMAGE_TOKENS,
                    QWEN3VL_COREML_HIDDEN_SIZE,
                ),
                "deepstack_1": (
                    QWEN3VL_COREML_IMAGE_TOKENS,
                    QWEN3VL_COREML_HIDDEN_SIZE,
                ),
                "deepstack_2": (
                    QWEN3VL_COREML_IMAGE_TOKENS,
                    QWEN3VL_COREML_HIDDEN_SIZE,
                ),
            },
        ),
    }[component]
    spec = model.get_spec()

    def shapes(features) -> dict[str, tuple[int, ...]]:
        result = {}
        for feature in features:
            if feature.type.WhichOneof("Type") != "multiArrayType":
                raise ValueError(
                    f"Qwen3-VL {component} feature {feature.name!r} is not a tensor."
                )
            result[feature.name] = tuple(feature.type.multiArrayType.shape)
        return result

    if shapes(spec.description.input) != expected[0]:
        raise ValueError(f"Qwen3-VL {component} input ABI does not match.")
    if shapes(spec.description.output) != expected[1]:
        raise ValueError(f"Qwen3-VL {component} output ABI does not match.")
    precision = "fp32" if component == "vision" else "fp16"
    required = {
        "libreyolo_producer": "libreyolo",
        "artifact_format": "coreml",
        "family": "qwen3vl",
        "task": "detect",
        "component": component,
        "component_contract": QWEN3VL_COREML_COMPONENT_CONTRACT,
        "source_revision": QWEN3VL_COREML_REVISION,
        "context_length": str(QWEN3VL_COREML_CONTEXT_LENGTH),
        "precision": precision,
        "compute_units": "cpu_only",
    }
    metadata = dict(model.user_defined_metadata or {})
    for key, expected_value in required.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"Qwen3-VL {component} metadata field {key!r} does not match."
            )


class CoreMLQwen3VLRuntime:
    """Execute the three fixed components in a Qwen3-VL Core ML bundle."""

    def __init__(
        self,
        bundle_path: str,
        *,
        compute_units: str = "cpu_only",
    ) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("Qwen3-VL Core ML bundles require macOS.")
        resolved_units = resolve_qwen3vl_coreml_compute_units(compute_units)
        if resolved_units != "cpu_only":
            raise ValueError("Qwen3-VL Core ML requires CPU_ONLY.")
        if importlib.metadata.version("transformers") != (
            QWEN3VL_COREML_TRANSFORMERS_VERSION
        ):
            raise RuntimeError(
                "Qwen3-VL Core ML runtime requires transformers "
                f"{QWEN3VL_COREML_TRANSFORMERS_VERSION}."
            )
        try:
            import coremltools as ct
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL Core ML runtime requires coremltools and transformers."
            ) from exc

        root = Path(bundle_path)
        if root.is_symlink() or not root.is_dir():
            raise FileNotFoundError(
                f"Qwen3-VL Core ML bundle does not exist: {root}."
            )
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise FileNotFoundError("Qwen3-VL bundle manifest is missing.")
        manifest = _load_json_object(manifest_path)
        if manifest != qwen3vl_bundle_manifest():
            raise ValueError("Qwen3-VL bundle manifest does not match the contract.")

        processor_dir = root / manifest["processor_path"]
        validate_qwen3vl_processor_assets(processor_dir)
        pixels = QWEN3VL_COREML_IMAGE_SIZE**2
        self.processor = AutoProcessor.from_pretrained(
            str(processor_dir),
            local_files_only=True,
            trust_remote_code=False,
            min_pixels=pixels,
            max_pixels=pixels,
        )
        self._models = {}
        for component, relative_name in QWEN3VL_COREML_COMPONENTS.items():
            package = root / relative_name
            if package.is_symlink() or not package.is_dir():
                raise FileNotFoundError(
                    f"Qwen3-VL Core ML component is missing: {relative_name}."
                )
            loaded = ct.models.MLModel(
                str(package),
                compute_units=ct.ComputeUnit.CPU_ONLY,
            )
            _validate_package_spec(loaded, component=component)
            self._models[component] = loaded
        self.bundle_path = str(root)
        self.context_length = QWEN3VL_COREML_CONTEXT_LENGTH
        self.max_new_tokens = QWEN3VL_COREML_MAX_NEW_TOKENS
        self._future_mask = np.triu(
            np.full(
                (self.context_length, self.context_length),
                -1e4,
                dtype=np.float16,
            ),
            k=1,
        )[None, None, :, :]
        self._lock = threading.RLock()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        with self._lock:
            self._models.clear()
            self.processor = None
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Qwen3-VL Core ML runtime is closed.")

    @staticmethod
    def _active_position_ids(token_types: np.ndarray) -> np.ndarray:
        """Reproduce the pinned one-image Qwen3-VL 3D position contract."""

        values = np.asarray(token_types)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Qwen3-VL token types must be a non-empty vector.")
        if np.any((values != 0) & (values != 1)):
            raise ValueError("Qwen3-VL Core ML supports text and one image only.")
        groups = []
        for modality, group in itertools.groupby(
            enumerate(values.tolist()),
            key=lambda item: item[1],
        ):
            entries = list(group)
            groups.append((modality, entries[0][0], entries[-1][0] + 1))
        current = 0
        pieces = []
        image_groups = 0
        for modality, start, end in groups:
            length = end - start
            if modality == 0:
                position = np.arange(
                    current,
                    current + length,
                    dtype=np.int32,
                )
                pieces.append(np.repeat(position[None, :], 3, axis=0))
                current += length
                continue
            image_groups += 1
            if image_groups > 1 or length != QWEN3VL_COREML_IMAGE_TOKENS:
                raise ValueError(
                    "Qwen3-VL Core ML requires exactly one 196-token image group."
                )
            side = 14
            temporal = np.full(side * side, current, dtype=np.int32)
            height = np.repeat(np.arange(side, dtype=np.int32), side) + current
            width = np.tile(np.arange(side, dtype=np.int32), side) + current
            pieces.append(np.stack((temporal, height, width), axis=0))
            current += side
        if image_groups != 1:
            raise ValueError("Qwen3-VL Core ML requires exactly one image.")
        return np.concatenate(pieces, axis=1)

    @staticmethod
    def _rope_tables(position_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(position_ids, dtype=np.float32)
        if (
            positions.ndim != 2
            or positions.shape[0] != 3
            or positions.shape[1] != QWEN3VL_COREML_CONTEXT_LENGTH
        ):
            raise ValueError("Qwen3-VL fixed position tensor has the wrong shape.")
        frequency_ids = np.arange(
            0,
            QWEN3VL_COREML_HEAD_DIM,
            2,
            dtype=np.float32,
        )
        inverse = 1.0 / np.power(
            np.float32(_ROPE_THETA),
            frequency_ids / np.float32(QWEN3VL_COREML_HEAD_DIM),
        )
        frequencies = positions[:, :, None] * inverse[None, None, :]
        interleaved = frequencies[0].copy()
        for dimension, offset in enumerate((1, 2), start=1):
            length = _MROPE_SECTION[dimension] * 3
            interleaved[:, offset:length:3] = frequencies[
                dimension,
                :,
                offset:length:3,
            ]
        embedding = np.concatenate((interleaved, interleaved), axis=-1)
        return (
            np.cos(embedding)[None, :, :].astype(np.float16),
            np.sin(embedding)[None, :, :].astype(np.float16),
        )

    def _prepare_processor_batch(
        self,
        image: Image.Image,
        prompt: str,
    ) -> tuple[Mapping[str, Any], Image.Image]:
        if not isinstance(image, Image.Image):
            raise TypeError("Qwen3-VL Core ML expects a PIL image.")
        square = image.convert("RGB").resize(
            (QWEN3VL_COREML_IMAGE_SIZE, QWEN3VL_COREML_IMAGE_SIZE),
            resample=Image.Resampling.BICUBIC,
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": square},
                    {"type": "text", "text": str(prompt)},
                ],
            }
        ]
        batch = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="np",
            return_dict=True,
            tokenize=True,
        )
        required = {
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "image_grid_thw",
        }
        if not isinstance(batch, Mapping) or not required.issubset(batch):
            raise ValueError("Qwen3-VL processor output is incomplete.")
        return batch, square

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> np.ndarray:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer.")
        if max_new_tokens <= 0 or max_new_tokens > self.max_new_tokens:
            raise ValueError(
                "max_new_tokens must be between 1 and "
                f"{self.max_new_tokens} for this Qwen3-VL profile."
            )
        with self._lock:
            self._require_open()
            batch, _square = self._prepare_processor_batch(image, prompt)
            sequence = np.asarray(batch["input_ids"], dtype=np.int64)
            attention = np.asarray(batch["attention_mask"])
            token_types = np.asarray(batch["mm_token_type_ids"], dtype=np.int32)
            grid = np.asarray(batch["image_grid_thw"], dtype=np.int64)
            pixels = np.asarray(batch["pixel_values"], dtype=np.float32)
            if sequence.ndim != 2 or sequence.shape[0] != 1:
                raise ValueError("Qwen3-VL processor must emit one token sequence.")
            if attention.shape != sequence.shape or not np.all(attention == 1):
                raise ValueError("Qwen3-VL processor emitted an invalid mask.")
            if token_types.shape != sequence.shape:
                raise ValueError("Qwen3-VL token-type tensor has the wrong shape.")
            if grid.shape != (1, 3) or not np.array_equal(
                grid,
                np.asarray([[1, 28, 28]], dtype=np.int64),
            ):
                raise ValueError("Qwen3-VL processor emitted the wrong image grid.")
            if pixels.shape != (
                QWEN3VL_COREML_PATCH_COUNT,
                QWEN3VL_COREML_PATCH_WIDTH,
            ):
                raise ValueError("Qwen3-VL processor emitted the wrong patch tensor.")
            sequence_list = [int(value) for value in sequence[0]]
            token_type_list = [int(value) for value in token_types[0]]
            prompt_length = len(sequence_list)
            if prompt_length + max_new_tokens > self.context_length:
                raise ValueError(
                    "Qwen3-VL prompt plus generation budget exceeds the fixed "
                    f"{self.context_length}-token context."
                )
            image_positions = np.flatnonzero(
                np.asarray(sequence_list) == QWEN3VL_COREML_IMAGE_TOKEN_ID
            )
            type_positions = np.flatnonzero(np.asarray(token_type_list) == 1)
            if (
                image_positions.size != QWEN3VL_COREML_IMAGE_TOKENS
                or not np.array_equal(image_positions, type_positions)
            ):
                raise ValueError(
                    "Qwen3-VL processor emitted invalid image-token positions."
                )
            vision_output = self._models["vision"].predict(
                {"patch_values": np.ascontiguousarray(pixels)}
            )
            image_embeddings = np.asarray(
                vision_output["image_embeddings"],
                dtype=np.float16,
            )
            deepstack = [
                np.asarray(
                    vision_output[f"deepstack_{index}"],
                    dtype=np.float16,
                )
                for index in range(3)
            ]
            expected_vision_shape = (
                QWEN3VL_COREML_IMAGE_TOKENS,
                QWEN3VL_COREML_HIDDEN_SIZE,
            )
            if image_embeddings.shape != expected_vision_shape or any(
                value.shape != expected_vision_shape for value in deepstack
            ):
                raise RuntimeError("Qwen3-VL vision output is malformed.")

            from transformers import RepetitionPenaltyLogitsProcessor

            repetition = RepetitionPenaltyLogitsProcessor(
                QWEN3VL_COREML_REPETITION_PENALTY
            )
            generated = []
            for _ in range(max_new_tokens):
                length = len(sequence_list)
                padding = self.context_length - length
                input_ids = np.full(
                    (1, self.context_length),
                    QWEN3VL_COREML_PAD_TOKEN_ID,
                    dtype=np.int32,
                )
                input_ids[0, padding:] = np.asarray(
                    sequence_list,
                    dtype=np.int32,
                )
                embedding_output = self._models["token_embedding"].predict(
                    {"input_ids": input_ids}
                )
                embeddings = np.asarray(
                    embedding_output["token_embeddings"],
                    dtype=np.float16,
                )
                if embeddings.shape != (
                    1,
                    self.context_length,
                    QWEN3VL_COREML_HIDDEN_SIZE,
                ):
                    raise RuntimeError(
                        "Qwen3-VL token-embedding output is malformed."
                    )
                deployed_image_positions = padding + image_positions
                embeddings[0, deployed_image_positions, :] = image_embeddings
                deepstack_input = np.zeros(
                    (
                        3,
                        1,
                        self.context_length,
                        QWEN3VL_COREML_HIDDEN_SIZE,
                    ),
                    dtype=np.float16,
                )
                for index, value in enumerate(deepstack):
                    deepstack_input[
                        index,
                        0,
                        deployed_image_positions,
                        :,
                    ] = value

                active_positions = self._active_position_ids(
                    np.asarray(token_type_list, dtype=np.int32)
                )
                fixed_positions = np.zeros(
                    (3, self.context_length),
                    dtype=np.int32,
                )
                fixed_positions[:, padding:] = active_positions
                rope_cos, rope_sin = self._rope_tables(fixed_positions)
                invalid_keys = np.zeros(
                    (1, 1, 1, self.context_length),
                    dtype=np.float16,
                )
                invalid_keys[:, :, :, :padding] = np.float16(-1e4)
                causal_mask = self._future_mask + invalid_keys
                decoder_output = self._models["decoder"].predict(
                    {
                        "input_embeddings": np.ascontiguousarray(embeddings),
                        "causal_mask": np.ascontiguousarray(causal_mask),
                        "rope_cos": np.ascontiguousarray(rope_cos),
                        "rope_sin": np.ascontiguousarray(rope_sin),
                        "deepstack_embeddings": np.ascontiguousarray(
                            deepstack_input
                        ),
                    }
                )
                logits = np.asarray(
                    decoder_output["last_logits"],
                    dtype=np.float32,
                )
                if logits.shape != (1, QWEN3VL_COREML_VOCAB_SIZE):
                    raise RuntimeError("Qwen3-VL decoder output is malformed.")
                processed = repetition(
                    torch.tensor([sequence_list], dtype=torch.long),
                    torch.from_numpy(logits.copy()),
                )
                next_token = int(torch.argmax(processed[0]).item())
                sequence_list.append(next_token)
                token_type_list.append(0)
                generated.append(next_token)
                if next_token in QWEN3VL_COREML_EOS_TOKEN_IDS:
                    break
            if not generated:
                raise RuntimeError("Qwen3-VL Core ML generation produced no token.")
            return np.asarray([generated], dtype=np.int64)


__all__ = [
    "COREML_QWEN3VL_BUNDLE_FORMAT",
    "COREML_QWEN3VL_BUNDLE_SUFFIX",
    "CoreMLQwen3VLRuntime",
]
