"""Bounded Qwen3-VL-2B Core ML components and portable bundle contract.

The source is the exact Apache-2.0 ``Qwen/Qwen3-VL-2B-Instruct`` snapshot
recorded below. Learned equations are composed from the public Apache-2.0
Transformers 5.12.1 Qwen3-VL modules. LibreYOLO fixes the image grid, moves
deterministic MRoPE table construction to the host, and uses a stateless
fixed-prefix decoder to keep the first deployment profile small and auditable.
"""

from __future__ import annotations

import gc
import hashlib
import hmac
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .coreml_profiles import normalize_coreml_compute_units

QWEN3VL_COREML_REPO = "Qwen/Qwen3-VL-2B-Instruct"
QWEN3VL_COREML_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
QWEN3VL_COREML_TRANSFORMERS_VERSION = "5.12.1"
QWEN3VL_COREML_TRANSFORMERS_COMMIT = (
    "ddb849abe009d1089e6c691bfc897f27211c663c"
)
QWEN3VL_COREML_COMPONENT_CONTRACT = (
    "qwen3vl_2b_448_stateless_prefix512_mixed_v1"
)
QWEN3VL_COREML_CONTEXT_LENGTH = 512
QWEN3VL_COREML_MAX_NEW_TOKENS = 48
QWEN3VL_COREML_IMAGE_SIZE = 448
QWEN3VL_COREML_PATCH_COUNT = 784
QWEN3VL_COREML_PATCH_WIDTH = 1536
QWEN3VL_COREML_IMAGE_TOKENS = 196
QWEN3VL_COREML_HIDDEN_SIZE = 2048
QWEN3VL_COREML_HEAD_DIM = 128
QWEN3VL_COREML_VOCAB_SIZE = 151936
QWEN3VL_COREML_IMAGE_TOKEN_ID = 151655
QWEN3VL_COREML_PAD_TOKEN_ID = 151643
QWEN3VL_COREML_EOS_TOKEN_IDS = (151645, 151643)
QWEN3VL_COREML_REPETITION_PENALTY = 1.1
QWEN3VL_COREML_PARAMETER_COUNT = 2_127_532_032
QWEN3VL_COREML_WEIGHTS_FILENAME = "model.safetensors"
QWEN3VL_COREML_WEIGHTS_SIZE = 4_255_140_312
QWEN3VL_COREML_WEIGHTS_SHA256 = (
    "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
)

QWEN3VL_COREML_PROCESSOR_ASSETS = {
    "chat_template.json": (
        5_502,
        "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
    ),
    "config.json": (
        1_505,
        "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
    ),
    "generation_config.json": (
        269,
        "1e241830b48b397cb0900101421df5450baddc7adf01e5fc86b5615865f3bae4",
    ),
    "merges.txt": (
        1_671_839,
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
    "preprocessor_config.json": (
        390,
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    ),
    "tokenizer.json": (
        7_032_403,
        "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    ),
    "tokenizer_config.json": (
        10_868,
        "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    ),
    "video_preprocessor_config.json": (
        385,
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    ),
    "vocab.json": (
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
}

QWEN3VL_COREML_COMPONENTS = {
    "decoder": "Decoder.mlpackage",
    "token_embedding": "TokenEmbedding.mlpackage",
    "vision": "Vision.mlpackage",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_qwen3vl_coreml_toolchain() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("Qwen3-VL Core ML export requires macOS.")
    try:
        import coremltools  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Qwen3-VL Core ML export requires coremltools 9.0."
        ) from exc
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(
            "Qwen3-VL Core ML export requires transformers 5.12.1."
        ) from exc
    if version != QWEN3VL_COREML_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Qwen3-VL Core ML export is pinned to transformers "
            f"{QWEN3VL_COREML_TRANSFORMERS_VERSION}; found {version}."
        )


def resolve_qwen3vl_coreml_compute_units(value: Any) -> str:
    resolved = normalize_coreml_compute_units(value)
    if resolved != "cpu_only":
        raise ValueError(
            "Qwen3-VL Core ML currently has hardware parity only for "
            "compute_units='cpu_only'."
        )
    return resolved


def validate_qwen3vl_processor_assets(root: str | os.PathLike[str]) -> None:
    directory = Path(root)
    for name, (expected_size, expected_hash) in (
        QWEN3VL_COREML_PROCESSOR_ASSETS.items()
    ):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"Qwen3-VL processor snapshot is missing {name!r}."
            )
        if path.stat().st_size != expected_size:
            raise ValueError(
                f"Qwen3-VL processor asset {name!r} has the wrong size."
            )
        actual_hash = _file_sha256(path)
        if not hmac.compare_digest(actual_hash, expected_hash):
            raise ValueError(
                f"Qwen3-VL processor asset {name!r} failed SHA-256 validation."
            )


def validate_qwen3vl_weight_asset(root: str | os.PathLike[str]) -> None:
    path = Path(root) / QWEN3VL_COREML_WEIGHTS_FILENAME
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("The pinned Qwen3-VL safetensors file is missing.")
    if path.stat().st_size != QWEN3VL_COREML_WEIGHTS_SIZE:
        raise ValueError("The pinned Qwen3-VL safetensors file has the wrong size.")
    if not hmac.compare_digest(
        _file_sha256(path),
        QWEN3VL_COREML_WEIGHTS_SHA256,
    ):
        raise ValueError("The pinned Qwen3-VL safetensors file failed SHA-256.")


def validate_qwen3vl_source_model(source: nn.Module) -> None:
    config = getattr(source, "config", None)
    text = getattr(config, "text_config", None)
    vision = getattr(config, "vision_config", None)
    actual = {
        "hidden": getattr(text, "hidden_size", None),
        "intermediate": getattr(text, "intermediate_size", None),
        "layers": getattr(text, "num_hidden_layers", None),
        "heads": getattr(text, "num_attention_heads", None),
        "kv_heads": getattr(text, "num_key_value_heads", None),
        "head_dim": getattr(text, "head_dim", None),
        "vocab": getattr(text, "vocab_size", None),
        "vision_depth": getattr(vision, "depth", None),
        "vision_hidden": getattr(vision, "hidden_size", None),
        "vision_heads": getattr(vision, "num_heads", None),
        "patch": getattr(vision, "patch_size", None),
        "merge": getattr(vision, "spatial_merge_size", None),
        "deepstack": list(getattr(vision, "deepstack_visual_indexes", ())),
    }
    expected = {
        "hidden": QWEN3VL_COREML_HIDDEN_SIZE,
        "intermediate": 6144,
        "layers": 28,
        "heads": 16,
        "kv_heads": 8,
        "head_dim": QWEN3VL_COREML_HEAD_DIM,
        "vocab": QWEN3VL_COREML_VOCAB_SIZE,
        "vision_depth": 24,
        "vision_hidden": 1024,
        "vision_heads": 16,
        "patch": 16,
        "merge": 2,
        "deepstack": [5, 11, 17],
    }
    if actual != expected:
        raise ValueError(
            "Qwen3-VL source architecture does not match the fixed profile: "
            f"{actual}."
        )
    parameters = sum(value.numel() for value in source.parameters())
    if parameters != QWEN3VL_COREML_PARAMETER_COUNT:
        raise ValueError(
            "Qwen3-VL source parameter count does not match the pinned checkpoint."
        )
    floating_dtypes = {
        value.dtype
        for value in (*tuple(source.parameters()), *tuple(source.buffers()))
        if value.is_floating_point()
    }
    if floating_dtypes != {torch.float32}:
        raise ValueError(
            "Qwen3-VL Core ML export requires an entirely FP32 source model."
        )


class Qwen3VLCoreMLVision(nn.Module):
    """Fixed 448-square vision tower with precomputed positional tensors."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        from transformers.vision_utils import (
            get_vision_bilinear_indices_and_weights,
            get_vision_cu_seqlens,
            get_vision_position_ids,
        )

        self.visual = source.model.visual
        grid = torch.tensor([[1, 28, 28]], dtype=torch.int64)
        indices, weights = get_vision_bilinear_indices_and_weights(
            grid,
            num_grid_per_side=self.visual.num_grid_per_side,
            spatial_merge_size=self.visual.spatial_merge_size,
        )
        self.register_buffer("bilinear_indices", indices)
        self.register_buffer("bilinear_weights", weights)
        self.register_buffer(
            "position_ids",
            get_vision_position_ids(grid, self.visual.spatial_merge_size),
        )
        self.register_buffer("cu_seqlens", get_vision_cu_seqlens(grid))

    def forward(self, patch_values: torch.Tensor):
        hidden_states = self.visual.patch_embed(patch_values)
        position = (
            self.visual.pos_embed(self.bilinear_indices)
            * self.bilinear_weights[:, :, None]
        ).sum(0)
        hidden_states = hidden_states + position.to(hidden_states.dtype)
        rotary = self.visual.rotary_pos_emb(self.position_ids)
        rotary = rotary.reshape(hidden_states.shape[0], -1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())
        deepstack = []
        for layer_index, block in enumerate(self.visual.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_index in self.visual.deepstack_visual_indexes:
                merger_index = self.visual.deepstack_visual_indexes.index(
                    layer_index
                )
                deepstack.append(
                    self.visual.deepstack_merger_list[merger_index](hidden_states)
                )
        merged = self.visual.merger(hidden_states)
        return merged, deepstack[0], deepstack[1], deepstack[2]


class Qwen3VLCoreMLTokenEmbedding(nn.Module):
    """Expose the tied Qwen token embedding at the fixed prefix width."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.embedding = source.get_input_embeddings()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids.to(dtype=torch.long))


class Qwen3VLCoreMLDecoder(nn.Module):
    """Stateless Qwen text graph with host-provided causal and MRoPE inputs."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        text = source.model.language_model
        self.layers = text.layers
        self.final_norm = text.norm
        self.lm_head = source.lm_head

    def forward(
        self,
        input_embeddings: torch.Tensor,
        causal_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        deepstack_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = input_embeddings
        position_embeddings = (rope_cos, rope_sin)
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            if layer_index < 3:
                hidden_states = (
                    hidden_states + deepstack_embeddings[layer_index]
                )
        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states[:, -1, :])


def _component_metadata(component: str) -> dict[str, str]:
    precision = "fp32" if component == "vision" else "fp16"
    return {
        "libreyolo_producer": "libreyolo",
        "artifact_format": "coreml",
        "family": "qwen3vl",
        "task": "detect",
        "component": component,
        "component_contract": QWEN3VL_COREML_COMPONENT_CONTRACT,
        "source_repo": QWEN3VL_COREML_REPO,
        "source_revision": QWEN3VL_COREML_REVISION,
        "context_length": str(QWEN3VL_COREML_CONTEXT_LENGTH),
        "precision": precision,
        "compute_units": "cpu_only",
    }


def _convert_component(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    *,
    inputs: list[Any],
    outputs: list[Any],
    component: str,
    output_path: Path,
    compute_precision: Any,
) -> None:
    import coremltools as ct

    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(
            module,
            example_inputs,
            check_trace=True,
            strict=True,
        )
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=compute_precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    converted.user_defined_metadata.update(_component_metadata(component))
    converted.save(str(output_path))
    del traced, converted
    gc.collect()


def export_qwen3vl_coreml_components(
    source: nn.Module,
    *,
    checkpoint_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    compute_units: str,
) -> dict[str, Path]:
    """Export the decoder first because it owns the largest transient graph."""

    require_qwen3vl_coreml_toolchain()
    resolve_qwen3vl_coreml_compute_units(compute_units)
    validate_qwen3vl_source_model(source)
    validate_qwen3vl_processor_assets(checkpoint_dir)
    validate_qwen3vl_weight_asset(checkpoint_dir)

    import coremltools as ct

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite Qwen3-VL component directory: {destination}."
        )
    destination.mkdir(parents=True)
    context = QWEN3VL_COREML_CONTEXT_LENGTH

    decoder_path = destination / QWEN3VL_COREML_COMPONENTS["decoder"]
    decoder = Qwen3VLCoreMLDecoder(source).eval()
    generator = torch.Generator(device="cpu").manual_seed(32512)
    embeddings = torch.randn(
        (1, context, QWEN3VL_COREML_HIDDEN_SIZE),
        generator=generator,
    )
    future = torch.triu(
        torch.full((context, context), -1e4, dtype=torch.float32),
        diagonal=1,
    )[None, None, :, :]
    rope_cos = torch.ones((1, context, QWEN3VL_COREML_HEAD_DIM))
    rope_sin = torch.zeros((1, context, QWEN3VL_COREML_HEAD_DIM))
    deepstack = torch.randn(
        (3, 1, context, QWEN3VL_COREML_HIDDEN_SIZE),
        generator=generator,
    )
    _convert_component(
        decoder,
        (embeddings, future, rope_cos, rope_sin, deepstack),
        inputs=[
            ct.TensorType(
                name="input_embeddings",
                shape=tuple(embeddings.shape),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="causal_mask",
                shape=tuple(future.shape),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="rope_cos",
                shape=tuple(rope_cos.shape),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="rope_sin",
                shape=tuple(rope_sin.shape),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="deepstack_embeddings",
                shape=tuple(deepstack.shape),
                dtype=np.float16,
            ),
        ],
        outputs=[ct.TensorType(name="last_logits", dtype=np.float16)],
        component="decoder",
        output_path=decoder_path,
        compute_precision=ct.precision.FLOAT16,
    )
    del decoder
    gc.collect()

    vision_path = destination / QWEN3VL_COREML_COMPONENTS["vision"]
    vision = Qwen3VLCoreMLVision(source).eval()
    patch_values = torch.linspace(
        -1.0,
        1.0,
        steps=QWEN3VL_COREML_PATCH_COUNT * QWEN3VL_COREML_PATCH_WIDTH,
    ).reshape(
        QWEN3VL_COREML_PATCH_COUNT,
        QWEN3VL_COREML_PATCH_WIDTH,
    )
    _convert_component(
        vision,
        (patch_values,),
        inputs=[
            ct.TensorType(
                name="patch_values",
                shape=tuple(patch_values.shape),
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(name="image_embeddings", dtype=np.float32),
            ct.TensorType(name="deepstack_0", dtype=np.float32),
            ct.TensorType(name="deepstack_1", dtype=np.float32),
            ct.TensorType(name="deepstack_2", dtype=np.float32),
        ],
        component="vision",
        output_path=vision_path,
        compute_precision=ct.precision.FLOAT32,
    )
    del vision
    gc.collect()

    embedding_path = destination / QWEN3VL_COREML_COMPONENTS["token_embedding"]
    token_embedding = Qwen3VLCoreMLTokenEmbedding(source).eval()
    input_ids = (
        torch.arange(context, dtype=torch.int32) % QWEN3VL_COREML_VOCAB_SIZE
    ).unsqueeze(0)
    _convert_component(
        token_embedding,
        (input_ids,),
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=tuple(input_ids.shape),
                dtype=np.int32,
            )
        ],
        outputs=[
            ct.TensorType(name="token_embeddings", dtype=np.float16),
        ],
        component="token_embedding",
        output_path=embedding_path,
        compute_precision=ct.precision.FLOAT16,
    )
    del token_embedding
    gc.collect()
    return {
        "decoder": decoder_path,
        "token_embedding": embedding_path,
        "vision": vision_path,
    }


def qwen3vl_bundle_manifest() -> dict[str, Any]:
    return {
        "bundle_format": "libreyolo_coreml_qwen3vl_bundle",
        "bundle_schema_version": 1,
        "component_contract": QWEN3VL_COREML_COMPONENT_CONTRACT,
        "components": dict(QWEN3VL_COREML_COMPONENTS),
        "context_length": QWEN3VL_COREML_CONTEXT_LENGTH,
        "max_new_tokens": QWEN3VL_COREML_MAX_NEW_TOKENS,
        "image_size": QWEN3VL_COREML_IMAGE_SIZE,
        "patch_count": QWEN3VL_COREML_PATCH_COUNT,
        "patch_width": QWEN3VL_COREML_PATCH_WIDTH,
        "image_tokens": QWEN3VL_COREML_IMAGE_TOKENS,
        "hidden_size": QWEN3VL_COREML_HIDDEN_SIZE,
        "head_dim": QWEN3VL_COREML_HEAD_DIM,
        "vocab_size": QWEN3VL_COREML_VOCAB_SIZE,
        "image_token_id": QWEN3VL_COREML_IMAGE_TOKEN_ID,
        "pad_token_id": QWEN3VL_COREML_PAD_TOKEN_ID,
        "eos_token_ids": list(QWEN3VL_COREML_EOS_TOKEN_IDS),
        "repetition_penalty": QWEN3VL_COREML_REPETITION_PENALTY,
        "processor_path": "Processor",
        "processor_assets": {
            name: {"size_bytes": size, "sha256": digest}
            for name, (size, digest) in QWEN3VL_COREML_PROCESSOR_ASSETS.items()
        },
        "source": {
            "repo": QWEN3VL_COREML_REPO,
            "revision": QWEN3VL_COREML_REVISION,
            "weights_filename": QWEN3VL_COREML_WEIGHTS_FILENAME,
            "weights_size_bytes": QWEN3VL_COREML_WEIGHTS_SIZE,
            "weights_sha256": QWEN3VL_COREML_WEIGHTS_SHA256,
            "license": "Apache-2.0",
        },
        "transformers": {
            "version": QWEN3VL_COREML_TRANSFORMERS_VERSION,
            "commit": QWEN3VL_COREML_TRANSFORMERS_COMMIT,
            "license": "Apache-2.0",
        },
        "precision": {
            "vision": "fp32",
            "token_embedding": "fp16",
            "decoder": "fp16",
        },
        "compute_units": "cpu_only",
        "decode": "greedy_stateless_full_prefix_left_pad",
        "image_geometry": "host_stretch_448_square",
    }


def build_qwen3vl_coreml_bundle(
    component_dir: str | os.PathLike[str],
    *,
    processor_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> str:
    components = Path(component_dir)
    processor = Path(processor_dir)
    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite Qwen3-VL Core ML bundle: {destination}."
        )
    validate_qwen3vl_processor_assets(processor)
    for relative_name in QWEN3VL_COREML_COMPONENTS.values():
        path = components / relative_name
        if path.is_symlink() or not path.is_dir():
            raise FileNotFoundError(
                f"Qwen3-VL Core ML component is missing: {relative_name}."
            )

    bundle = components.parent / f"{destination.name}.building"
    if bundle.exists() or bundle.is_symlink():
        raise FileExistsError(
            f"Qwen3-VL bundle staging path already exists: {bundle}."
        )
    bundle.mkdir()
    try:
        for relative_name in QWEN3VL_COREML_COMPONENTS.values():
            shutil.move(
                str(components / relative_name),
                str(bundle / relative_name),
            )
        processor_output = bundle / "Processor"
        processor_output.mkdir()
        for name in QWEN3VL_COREML_PROCESSOR_ASSETS:
            shutil.copy2(processor / name, processor_output / name)
        licenses = bundle / "LICENSES"
        licenses.mkdir()
        apache = Path(__file__).resolve().parents[2] / "licenses" / "Apache-2.0.txt"
        shutil.copy2(apache, licenses / "Apache-2.0.txt")
        (bundle / "NOTICE.txt").write_text(
            "LibreYOLO Qwen3-VL Core ML bundle\n\n"
            f"Model: {QWEN3VL_COREML_REPO}\n"
            f"Revision: {QWEN3VL_COREML_REVISION}\n"
            "Model license: Apache-2.0\n"
            "Runtime reference: Hugging Face Transformers "
            f"{QWEN3VL_COREML_TRANSFORMERS_VERSION} at "
            f"{QWEN3VL_COREML_TRANSFORMERS_COMMIT}\n"
            "Runtime reference license: Apache-2.0\n"
            "The bundle does not contain model.safetensors.\n",
            encoding="utf-8",
        )
        (bundle / "manifest.json").write_text(
            json.dumps(qwen3vl_bundle_manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(bundle, destination)
    except Exception:
        shutil.rmtree(bundle, ignore_errors=True)
        raise
    return str(destination)


__all__ = [
    "QWEN3VL_COREML_COMPONENTS",
    "QWEN3VL_COREML_COMPONENT_CONTRACT",
    "QWEN3VL_COREML_CONTEXT_LENGTH",
    "QWEN3VL_COREML_EOS_TOKEN_IDS",
    "QWEN3VL_COREML_HEAD_DIM",
    "QWEN3VL_COREML_HIDDEN_SIZE",
    "QWEN3VL_COREML_IMAGE_SIZE",
    "QWEN3VL_COREML_IMAGE_TOKEN_ID",
    "QWEN3VL_COREML_IMAGE_TOKENS",
    "QWEN3VL_COREML_MAX_NEW_TOKENS",
    "QWEN3VL_COREML_PAD_TOKEN_ID",
    "QWEN3VL_COREML_PATCH_COUNT",
    "QWEN3VL_COREML_PATCH_WIDTH",
    "QWEN3VL_COREML_PROCESSOR_ASSETS",
    "QWEN3VL_COREML_REPETITION_PENALTY",
    "QWEN3VL_COREML_REPO",
    "QWEN3VL_COREML_REVISION",
    "QWEN3VL_COREML_TRANSFORMERS_COMMIT",
    "QWEN3VL_COREML_TRANSFORMERS_VERSION",
    "QWEN3VL_COREML_VOCAB_SIZE",
    "Qwen3VLCoreMLDecoder",
    "Qwen3VLCoreMLTokenEmbedding",
    "Qwen3VLCoreMLVision",
    "build_qwen3vl_coreml_bundle",
    "export_qwen3vl_coreml_components",
    "qwen3vl_bundle_manifest",
    "require_qwen3vl_coreml_toolchain",
    "resolve_qwen3vl_coreml_compute_units",
    "validate_qwen3vl_processor_assets",
    "validate_qwen3vl_source_model",
    "validate_qwen3vl_weight_asset",
]
