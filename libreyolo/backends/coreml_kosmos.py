"""Portable Kosmos-2 Core ML bundle loader and stateless greedy runtime."""

from __future__ import annotations

import importlib.metadata
import json
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..export.coreml_kosmos import (
    KOSMOS2_COREML_COMPONENTS,
    KOSMOS2_COREML_COMPONENT_CONTRACT,
    KOSMOS2_COREML_CONTEXT_LENGTH,
    KOSMOS2_COREML_EOS_TOKEN_ID,
    KOSMOS2_COREML_HIDDEN_SIZE,
    KOSMOS2_COREML_IMAGE_SIZE,
    KOSMOS2_COREML_IMAGE_TOKENS,
    KOSMOS2_COREML_MAX_NEW_TOKENS,
    KOSMOS2_COREML_NO_REPEAT_NGRAM_SIZE,
    KOSMOS2_COREML_PAD_TOKEN_ID,
    KOSMOS2_COREML_REVISION,
    KOSMOS2_COREML_TRANSFORMERS_VERSION,
    KOSMOS2_COREML_VOCAB_SIZE,
    kosmos2_bundle_manifest,
    resolve_kosmos2_coreml_compute_units,
    validate_kosmos2_processor_assets,
)

COREML_KOSMOS2_BUNDLE_FORMAT = "libreyolo_coreml_kosmos2_bundle"
COREML_KOSMOS2_BUNDLE_SUFFIX = ".coremlvlm"


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"Kosmos-2 bundle manifest repeats key {key!r}."
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
            "Kosmos-2 bundle manifest is not valid UTF-8 JSON."
        ) from exc
    if not isinstance(value, dict):
        raise TypeError("Kosmos-2 bundle manifest must be a JSON object.")
    return value


def _validate_package_spec(model, *, component: str) -> None:
    expected = {
        "decoder": (
            {
                "input_embeddings": (1, KOSMOS2_COREML_CONTEXT_LENGTH, 2048),
                "attention_mask": (1, KOSMOS2_COREML_CONTEXT_LENGTH),
                "position_ids": (1, KOSMOS2_COREML_CONTEXT_LENGTH),
            },
            {"last_logits": (1, KOSMOS2_COREML_VOCAB_SIZE)},
        ),
        "token_embedding": (
            {"input_ids": (1, KOSMOS2_COREML_CONTEXT_LENGTH)},
            {
                "token_embeddings": (
                    1,
                    KOSMOS2_COREML_CONTEXT_LENGTH,
                    KOSMOS2_COREML_HIDDEN_SIZE,
                )
            },
        ),
        "vision": (
            {
                "pixel_values": (
                    1,
                    3,
                    KOSMOS2_COREML_IMAGE_SIZE,
                    KOSMOS2_COREML_IMAGE_SIZE,
                )
            },
            {
                "image_embeddings": (
                    1,
                    KOSMOS2_COREML_IMAGE_TOKENS,
                    KOSMOS2_COREML_HIDDEN_SIZE,
                )
            },
        ),
    }[component]
    spec = model.get_spec()

    def shapes(features) -> dict[str, tuple[int, ...]]:
        result = {}
        for feature in features:
            if feature.type.WhichOneof("Type") != "multiArrayType":
                raise ValueError(
                    f"Kosmos-2 {component} feature {feature.name!r} is not a tensor."
                )
            result[feature.name] = tuple(feature.type.multiArrayType.shape)
        return result

    if shapes(spec.description.input) != expected[0]:
        raise ValueError(f"Kosmos-2 {component} input ABI does not match.")
    if shapes(spec.description.output) != expected[1]:
        raise ValueError(f"Kosmos-2 {component} output ABI does not match.")
    metadata = dict(model.user_defined_metadata or {})
    required = {
        "libreyolo_producer": "libreyolo",
        "artifact_format": "coreml",
        "family": "kosmos2",
        "task": "detect",
        "component": component,
        "component_contract": KOSMOS2_COREML_COMPONENT_CONTRACT,
        "source_revision": KOSMOS2_COREML_REVISION,
        "context_length": str(KOSMOS2_COREML_CONTEXT_LENGTH),
        "precision": "fp32",
        "compute_units": "cpu_only",
    }
    for key, expected_value in required.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"Kosmos-2 {component} metadata field {key!r} does not match."
            )


class CoreMLKosmos2Runtime:
    """Execute a three-component fixed-prefix Kosmos-2 bundle."""

    def __init__(
        self,
        bundle_path: str,
        *,
        compute_units: str = "cpu_only",
    ) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("Kosmos-2 Core ML bundles require macOS.")
        resolved_units = resolve_kosmos2_coreml_compute_units(compute_units)
        if resolved_units != "cpu_only":  # defensive; resolver already enforces this.
            raise ValueError("Kosmos-2 Core ML requires CPU_ONLY.")
        if importlib.metadata.version("transformers") != (
            KOSMOS2_COREML_TRANSFORMERS_VERSION
        ):
            raise RuntimeError(
                "Kosmos-2 Core ML runtime requires transformers "
                f"{KOSMOS2_COREML_TRANSFORMERS_VERSION}."
            )
        try:
            import coremltools as ct
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Kosmos-2 Core ML runtime requires coremltools and transformers."
            ) from exc

        root = Path(bundle_path)
        if root.is_symlink() or not root.is_dir():
            raise FileNotFoundError(
                f"Kosmos-2 Core ML bundle does not exist: {root}."
            )
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise FileNotFoundError("Kosmos-2 bundle manifest is missing.")
        manifest = _load_json_object(manifest_path)
        if manifest != kosmos2_bundle_manifest():
            raise ValueError("Kosmos-2 bundle manifest does not match the contract.")

        processor_dir = root / manifest["processor_path"]
        validate_kosmos2_processor_assets(processor_dir)
        self.processor = AutoProcessor.from_pretrained(
            str(processor_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
        self._models = {}
        for component, relative_name in KOSMOS2_COREML_COMPONENTS.items():
            package = root / relative_name
            if package.is_symlink() or not package.is_dir():
                raise FileNotFoundError(
                    f"Kosmos-2 Core ML component is missing: {relative_name}."
                )
            loaded = ct.models.MLModel(
                str(package),
                compute_units=ct.ComputeUnit.CPU_ONLY,
            )
            _validate_package_spec(loaded, component=component)
            self._models[component] = loaded
        self.bundle_path = str(root)
        self.context_length = KOSMOS2_COREML_CONTEXT_LENGTH
        self.max_new_tokens = KOSMOS2_COREML_MAX_NEW_TOKENS
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
            raise RuntimeError("Kosmos-2 Core ML runtime is closed.")

    @staticmethod
    def _left_pad_prefix(sequence: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = len(sequence)
        if length <= 0 or length > KOSMOS2_COREML_CONTEXT_LENGTH:
            raise ValueError(
                "Kosmos-2 token prefix is empty or exceeds the fixed context."
            )
        padding = KOSMOS2_COREML_CONTEXT_LENGTH - length
        input_ids = np.full(
            (1, KOSMOS2_COREML_CONTEXT_LENGTH),
            KOSMOS2_COREML_PAD_TOKEN_ID,
            dtype=np.int32,
        )
        input_ids[0, padding:] = np.asarray(sequence, dtype=np.int32)
        attention_mask = np.zeros(
            (1, KOSMOS2_COREML_CONTEXT_LENGTH),
            dtype=np.float32,
        )
        attention_mask[0, padding:] = 1.0
        position_ids = np.full(
            (1, KOSMOS2_COREML_CONTEXT_LENGTH),
            KOSMOS2_COREML_PAD_TOKEN_ID,
            dtype=np.int32,
        )
        position_ids[0, padding:] = np.arange(2, length + 2, dtype=np.int32)
        return input_ids, attention_mask, position_ids

    def generate(
        self,
        image,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> np.ndarray:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer.")
        if max_new_tokens <= 0 or max_new_tokens > self.max_new_tokens:
            raise ValueError(
                "max_new_tokens must be between 1 and "
                f"{self.max_new_tokens} for this Kosmos-2 profile."
            )
        with self._lock:
            self._require_open()
            batch = self.processor(
                text=str(prompt),
                images=image,
                return_tensors="np",
            )
            required = {
                "input_ids",
                "attention_mask",
                "image_embeds_position_mask",
                "pixel_values",
            }
            if not isinstance(batch, Mapping) or not required.issubset(batch):
                raise ValueError("Kosmos-2 processor output is incomplete.")
            sequence = np.asarray(batch["input_ids"], dtype=np.int64)
            if sequence.ndim != 2 or sequence.shape[0] != 1:
                raise ValueError("Kosmos-2 processor must emit one token sequence.")
            sequence_list = [int(value) for value in sequence[0]]
            if len(sequence_list) + max_new_tokens > self.context_length:
                raise ValueError(
                    "Kosmos-2 prompt plus generation budget exceeds the fixed "
                    f"{self.context_length}-token context."
                )
            image_mask = np.asarray(
                batch["image_embeds_position_mask"],
                dtype=bool,
            )
            if image_mask.shape != sequence.shape:
                raise ValueError("Kosmos-2 image-position mask has the wrong shape.")
            image_positions = np.flatnonzero(image_mask[0])
            if image_positions.size != KOSMOS2_COREML_IMAGE_TOKENS:
                raise ValueError(
                    "Kosmos-2 processor emitted the wrong number of image slots."
                )
            pixel_values = np.asarray(batch["pixel_values"], dtype=np.float32)
            if pixel_values.shape != (
                1,
                3,
                KOSMOS2_COREML_IMAGE_SIZE,
                KOSMOS2_COREML_IMAGE_SIZE,
            ):
                raise ValueError("Kosmos-2 processor emitted the wrong image tensor.")
            vision_output = self._models["vision"].predict(
                {"pixel_values": np.ascontiguousarray(pixel_values)}
            )
            image_embeddings = np.asarray(
                vision_output["image_embeddings"],
                dtype=np.float32,
            )
            if image_embeddings.shape != (
                1,
                KOSMOS2_COREML_IMAGE_TOKENS,
                KOSMOS2_COREML_HIDDEN_SIZE,
            ):
                raise RuntimeError("Kosmos-2 vision component output is malformed.")

            from transformers import NoRepeatNGramLogitsProcessor

            no_repeat = NoRepeatNGramLogitsProcessor(
                KOSMOS2_COREML_NO_REPEAT_NGRAM_SIZE
            )
            prompt_length = len(sequence_list)
            for _ in range(max_new_tokens):
                input_ids, attention_mask, position_ids = self._left_pad_prefix(
                    sequence_list
                )
                embedding_output = self._models["token_embedding"].predict(
                    {"input_ids": input_ids}
                )
                token_embeddings = np.asarray(
                    embedding_output["token_embeddings"],
                    dtype=np.float32,
                )
                padding = self.context_length - len(sequence_list)
                deployed_positions = padding + image_positions
                token_embeddings[0, deployed_positions, :] = image_embeddings[0]
                decoder_output = self._models["decoder"].predict(
                    {
                        "input_embeddings": np.ascontiguousarray(token_embeddings),
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                    }
                )
                logits = np.asarray(
                    decoder_output["last_logits"],
                    dtype=np.float32,
                )
                if logits.shape != (1, KOSMOS2_COREML_VOCAB_SIZE):
                    raise RuntimeError(
                        "Kosmos-2 decoder component output is malformed."
                    )
                processed = no_repeat(
                    torch.tensor([sequence_list], dtype=torch.long),
                    torch.from_numpy(logits.copy()),
                )
                next_token = int(torch.argmax(processed[0]).item())
                sequence_list.append(next_token)
                if next_token == KOSMOS2_COREML_EOS_TOKEN_ID:
                    break
            if len(sequence_list) == prompt_length:
                raise RuntimeError("Kosmos-2 Core ML generation produced no token.")
            return np.asarray([sequence_list], dtype=np.int64)


__all__ = [
    "COREML_KOSMOS2_BUNDLE_FORMAT",
    "COREML_KOSMOS2_BUNDLE_SUFFIX",
    "CoreMLKosmos2Runtime",
]
