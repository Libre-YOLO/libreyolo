"""Bounded Kosmos-2 Core ML components and portable bundle contract.

The source checkpoint is Microsoft's MIT-licensed
``microsoft/kosmos-2-patch14-224`` snapshot at
``e91cfbcb4ce051b6a55bfb5f96165a3bbf5eb82c``. The component boundaries call
the public Apache-2.0 Transformers 5.12.1 Kosmos-2 modules; the fixed-prefix
host contract and bundle format are LibreYOLO code under MIT.

This first profile deliberately favors a small, auditable implementation over
decode speed. The host left-pads one bounded 128-token prefix and recomputes the
stateless language graph for every greedy token. A future stateful profile can
replace it without changing the public ``LibreVLM`` result contract.
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
import torch.nn.functional as F

from .coreml_profiles import normalize_coreml_compute_units

KOSMOS2_COREML_REPO = "microsoft/kosmos-2-patch14-224"
KOSMOS2_COREML_REVISION = "e91cfbcb4ce051b6a55bfb5f96165a3bbf5eb82c"
KOSMOS2_COREML_TRANSFORMERS_VERSION = "5.12.1"
KOSMOS2_COREML_TRANSFORMERS_COMMIT = "ddb849abe009d1089e6c691bfc897f27211c663c"
KOSMOS2_COREML_COMPONENT_CONTRACT = "kosmos2_224_stateless_prefix128_fp32_v1"
KOSMOS2_COREML_CONTEXT_LENGTH = 128
KOSMOS2_COREML_MAX_NEW_TOKENS = 48
KOSMOS2_COREML_IMAGE_SIZE = 224
KOSMOS2_COREML_IMAGE_TOKENS = 64
KOSMOS2_COREML_HIDDEN_SIZE = 2048
KOSMOS2_COREML_VOCAB_SIZE = 65037
KOSMOS2_COREML_PAD_TOKEN_ID = 1
KOSMOS2_COREML_EOS_TOKEN_ID = 2
KOSMOS2_COREML_NO_REPEAT_NGRAM_SIZE = 3
KOSMOS2_COREML_PARAMETER_COUNT = 1_664_485_376
KOSMOS2_COREML_WEIGHTS_FILENAME = "model.safetensors"
KOSMOS2_COREML_WEIGHTS_SIZE = 6_658_052_808
KOSMOS2_COREML_WEIGHTS_SHA256 = (
    "051bf4b62a25429f4d542d11ec0c07a4ac1aac91003d3bf301133c6913008cbf"
)

KOSMOS2_COREML_PROCESSOR_ASSETS = {
    "added_tokens.json": (
        32_001,
        "b3eb1ec0cef1678c73b11cfd7b41c69d3b79e40a4400e857e466c6aefb039e95",
    ),
    "config.json": (
        4_452,
        "131c5e1eb60cb445f04efc39e61d027cc49636be72ae8cd1d3946bf272232c52",
    ),
    "generation_config.json": (
        137,
        "63c089d4157f49e33d15f8bbed5442922198f9a618be68af064f934d8e916fc5",
    ),
    "preprocessor_config.json": (
        534,
        "6ab68e439b4b5aee971db5477eb57431579c0b740a9bb0d10038efb704fd8eb3",
    ),
    "sentencepiece.bpe.model": (
        1_363_614,
        "3a60b4d1d1d8f70c8b2569c94540d4d9b7c694fd32e7a428ad0dcffaafaa3beb",
    ),
    "special_tokens_map.json": (
        1_064,
        "276d9449066f1fa93b3542c17129a52470b4b46e5d807ebc100e93985487e4ca",
    ),
    "tokenizer.json": (
        4_698_210,
        "3deef0657b5cb05b87d962e5a4489d5572ea0aa04d056a794c3c26a6f287b1d4",
    ),
    "tokenizer_config.json": (
        190_658,
        "fb8c7370fcd02e4a26fb26a2bbf937ec6f3ac17ab3ee6f975a15a0b50af11a60",
    ),
}

KOSMOS2_COREML_COMPONENTS = {
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


def require_kosmos2_coreml_toolchain() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("Kosmos-2 Core ML export requires macOS.")
    try:
        import coremltools  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Kosmos-2 Core ML export requires coremltools 9.0."
        ) from exc
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(
            "Kosmos-2 Core ML export requires transformers 5.12.1."
        ) from exc
    if version != KOSMOS2_COREML_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Kosmos-2 Core ML export is pinned to transformers "
            f"{KOSMOS2_COREML_TRANSFORMERS_VERSION}; found {version}."
        )


def resolve_kosmos2_coreml_compute_units(value: Any) -> str:
    resolved = normalize_coreml_compute_units(value)
    if resolved != "cpu_only":
        raise ValueError(
            "Kosmos-2 Core ML currently has hardware parity only for "
            "compute_units='cpu_only'."
        )
    return resolved


def validate_kosmos2_processor_assets(root: str | os.PathLike[str]) -> None:
    directory = Path(root)
    for name, (expected_size, expected_hash) in KOSMOS2_COREML_PROCESSOR_ASSETS.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"Kosmos-2 processor snapshot is missing {name!r}."
            )
        if path.stat().st_size != expected_size:
            raise ValueError(f"Kosmos-2 processor asset {name!r} has the wrong size.")
        actual_hash = _file_sha256(path)
        if not hmac.compare_digest(actual_hash, expected_hash):
            raise ValueError(
                f"Kosmos-2 processor asset {name!r} failed SHA-256 validation."
            )


def validate_kosmos2_weight_asset(root: str | os.PathLike[str]) -> None:
    path = Path(root) / KOSMOS2_COREML_WEIGHTS_FILENAME
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("The pinned Kosmos-2 safetensors file is missing.")
    if path.stat().st_size != KOSMOS2_COREML_WEIGHTS_SIZE:
        raise ValueError("The pinned Kosmos-2 safetensors file has the wrong size.")
    actual_hash = _file_sha256(path)
    if not hmac.compare_digest(actual_hash, KOSMOS2_COREML_WEIGHTS_SHA256):
        raise ValueError("The pinned Kosmos-2 safetensors file failed SHA-256.")


def validate_kosmos2_source_model(source: nn.Module) -> None:
    config = getattr(source, "config", None)
    text_config = getattr(config, "text_config", None)
    vision_config = getattr(config, "vision_config", None)
    actual = {
        "context": getattr(text_config, "max_position_embeddings", None),
        "hidden": getattr(text_config, "embed_dim", None),
        "image": getattr(vision_config, "image_size", None),
        "image_tokens": getattr(config, "latent_query_num", None),
        "layers": getattr(text_config, "layers", None),
        "vocab": getattr(text_config, "vocab_size", None),
    }
    expected = {
        "context": 2048,
        "hidden": KOSMOS2_COREML_HIDDEN_SIZE,
        "image": KOSMOS2_COREML_IMAGE_SIZE,
        "image_tokens": KOSMOS2_COREML_IMAGE_TOKENS,
        "layers": 24,
        "vocab": KOSMOS2_COREML_VOCAB_SIZE,
    }
    if actual != expected:
        raise ValueError(
            f"Kosmos-2 source architecture does not match the fixed profile: {actual}."
        )
    parameters = sum(parameter.numel() for parameter in source.parameters())
    if parameters != KOSMOS2_COREML_PARAMETER_COUNT:
        raise ValueError(
            "Kosmos-2 source parameter count does not match the pinned checkpoint."
        )
    floating_dtypes = {
        value.dtype
        for value in (*tuple(source.parameters()), *tuple(source.buffers()))
        if value.is_floating_point()
    }
    if floating_dtypes != {torch.float32}:
        raise ValueError(
            "Kosmos-2 Core ML export requires an entirely FP32 source model."
        )


class Kosmos2CoreMLVision(nn.Module):
    """Public-module wrapper for the fixed image-to-language projection."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.vision_model = source.vision_model
        self.image_to_text_projection = source.image_to_text_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_output = self.vision_model(
            pixel_values=pixel_values,
            return_dict=False,
        )
        image_embeddings = self.vision_model.model.post_layernorm(vision_output[0])
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        image_embeddings, _ = self.image_to_text_projection(image_embeddings)
        return image_embeddings


class Kosmos2CoreMLTokenEmbedding(nn.Module):
    """Expose the tied token embedding as one bounded Core ML component."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.embedding = source.get_input_embeddings()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids.to(dtype=torch.long))


class Kosmos2CoreMLDecoder(nn.Module):
    """Return logits for the last token of one left-padded fixed prefix."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.transformer = source.text_model.model
        self.lm_head = source.text_model.lm_head

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = self.transformer(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids.to(dtype=torch.long),
            use_cache=False,
            return_dict=False,
        )
        return self.lm_head(output[0][:, -1, :])


def _component_metadata(component: str) -> dict[str, str]:
    return {
        "libreyolo_producer": "libreyolo",
        "artifact_format": "coreml",
        "family": "kosmos2",
        "task": "detect",
        "component": component,
        "component_contract": KOSMOS2_COREML_COMPONENT_CONTRACT,
        "source_repo": KOSMOS2_COREML_REPO,
        "source_revision": KOSMOS2_COREML_REVISION,
        "context_length": str(KOSMOS2_COREML_CONTEXT_LENGTH),
        "precision": "fp32",
        "compute_units": "cpu_only",
    }


def _convert_component(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    *,
    inputs: list[Any],
    output_name: str,
    component: str,
    output_path: Path,
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
        outputs=[ct.TensorType(name=output_name, dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    converted.user_defined_metadata.update(_component_metadata(component))
    converted.save(str(output_path))
    del traced, converted
    gc.collect()


def export_kosmos2_coreml_components(
    source: nn.Module,
    *,
    checkpoint_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    compute_units: str,
) -> dict[str, Path]:
    """Export decoder first to keep transient disk usage bounded."""

    require_kosmos2_coreml_toolchain()
    resolve_kosmos2_coreml_compute_units(compute_units)
    validate_kosmos2_source_model(source)
    validate_kosmos2_processor_assets(checkpoint_dir)
    validate_kosmos2_weight_asset(checkpoint_dir)

    import coremltools as ct

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite Kosmos-2 component directory: {destination}."
        )
    destination.mkdir(parents=True)
    context = KOSMOS2_COREML_CONTEXT_LENGTH
    generator = torch.Generator(device="cpu").manual_seed(240224)

    decoder_path = destination / KOSMOS2_COREML_COMPONENTS["decoder"]
    decoder = Kosmos2CoreMLDecoder(source).eval()
    input_embeddings = torch.randn(
        (1, context, KOSMOS2_COREML_HIDDEN_SIZE),
        generator=generator,
        dtype=torch.float32,
    )
    attention_mask = torch.ones((1, context), dtype=torch.float32)
    position_ids = torch.arange(2, context + 2, dtype=torch.int32).unsqueeze(0)
    _convert_component(
        decoder,
        (input_embeddings, attention_mask, position_ids),
        inputs=[
            ct.TensorType(
                name="input_embeddings",
                shape=tuple(input_embeddings.shape),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=tuple(attention_mask.shape),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="position_ids",
                shape=tuple(position_ids.shape),
                dtype=np.int32,
            ),
        ],
        output_name="last_logits",
        component="decoder",
        output_path=decoder_path,
    )
    del decoder
    gc.collect()

    vision_path = destination / KOSMOS2_COREML_COMPONENTS["vision"]
    vision = Kosmos2CoreMLVision(source).eval()
    pixels = torch.linspace(
        -1.0,
        1.0,
        steps=3 * KOSMOS2_COREML_IMAGE_SIZE * KOSMOS2_COREML_IMAGE_SIZE,
        dtype=torch.float32,
    ).reshape(1, 3, KOSMOS2_COREML_IMAGE_SIZE, KOSMOS2_COREML_IMAGE_SIZE)
    _convert_component(
        vision,
        (pixels,),
        inputs=[
            ct.TensorType(
                name="pixel_values",
                shape=tuple(pixels.shape),
                dtype=np.float32,
            )
        ],
        output_name="image_embeddings",
        component="vision",
        output_path=vision_path,
    )
    del vision
    gc.collect()

    embedding_path = destination / KOSMOS2_COREML_COMPONENTS["token_embedding"]
    embedding = Kosmos2CoreMLTokenEmbedding(source).eval()
    input_ids = torch.arange(context, dtype=torch.int32).unsqueeze(0)
    _convert_component(
        embedding,
        (input_ids,),
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=tuple(input_ids.shape),
                dtype=np.int32,
            )
        ],
        output_name="token_embeddings",
        component="token_embedding",
        output_path=embedding_path,
    )
    del embedding
    gc.collect()
    return {
        "decoder": decoder_path,
        "token_embedding": embedding_path,
        "vision": vision_path,
    }


def kosmos2_bundle_manifest() -> dict[str, Any]:
    return {
        "bundle_format": "libreyolo_coreml_kosmos2_bundle",
        "bundle_schema_version": 1,
        "component_contract": KOSMOS2_COREML_COMPONENT_CONTRACT,
        "components": dict(KOSMOS2_COREML_COMPONENTS),
        "context_length": KOSMOS2_COREML_CONTEXT_LENGTH,
        "max_new_tokens": KOSMOS2_COREML_MAX_NEW_TOKENS,
        "image_size": KOSMOS2_COREML_IMAGE_SIZE,
        "image_tokens": KOSMOS2_COREML_IMAGE_TOKENS,
        "hidden_size": KOSMOS2_COREML_HIDDEN_SIZE,
        "vocab_size": KOSMOS2_COREML_VOCAB_SIZE,
        "pad_token_id": KOSMOS2_COREML_PAD_TOKEN_ID,
        "eos_token_id": KOSMOS2_COREML_EOS_TOKEN_ID,
        "no_repeat_ngram_size": KOSMOS2_COREML_NO_REPEAT_NGRAM_SIZE,
        "processor_path": "Processor",
        "processor_assets": {
            name: {"size_bytes": size, "sha256": digest}
            for name, (size, digest) in KOSMOS2_COREML_PROCESSOR_ASSETS.items()
        },
        "source": {
            "repo": KOSMOS2_COREML_REPO,
            "revision": KOSMOS2_COREML_REVISION,
            "weights_filename": KOSMOS2_COREML_WEIGHTS_FILENAME,
            "weights_size_bytes": KOSMOS2_COREML_WEIGHTS_SIZE,
            "weights_sha256": KOSMOS2_COREML_WEIGHTS_SHA256,
            "license": "MIT",
        },
        "transformers": {
            "version": KOSMOS2_COREML_TRANSFORMERS_VERSION,
            "commit": KOSMOS2_COREML_TRANSFORMERS_COMMIT,
            "license": "Apache-2.0",
        },
        "precision": "fp32",
        "compute_units": "cpu_only",
        "decode": "greedy_stateless_full_prefix_left_pad",
    }


def _microsoft_mit_license() -> bytes:
    return (
        "MIT License\n\n"
        "Copyright (c) Microsoft Corporation.\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining "
        "a copy\nof this software and associated documentation files (the "
        '"Software"), to deal\nin the Software without restriction, including '
        "without limitation the rights\nto use, copy, modify, merge, publish, "
        "distribute, sublicense, and/or sell\ncopies of the Software, and to "
        "permit persons to whom the Software is\nfurnished to do so, subject "
        "to the following conditions:\n\nThe above copyright notice and this "
        "permission notice shall be included in all\ncopies or substantial "
        "portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", "
        "WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT "
        "LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A "
        "PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS "
        "OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, "
        "ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE "
        "OR OTHER DEALINGS IN THE\nSOFTWARE.\n"
    ).encode("ascii")


def build_kosmos2_coreml_bundle(
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
            f"Refusing to overwrite Kosmos-2 Core ML bundle: {destination}."
        )
    validate_kosmos2_processor_assets(processor)
    for relative_name in KOSMOS2_COREML_COMPONENTS.values():
        path = components / relative_name
        if path.is_symlink() or not path.is_dir():
            raise FileNotFoundError(
                f"Kosmos-2 Core ML component is missing: {relative_name}."
            )

    bundle = components.parent / f"{destination.name}.building"
    if bundle.exists() or bundle.is_symlink():
        raise FileExistsError(f"Kosmos-2 bundle staging path already exists: {bundle}.")
    bundle.mkdir()
    try:
        for relative_name in KOSMOS2_COREML_COMPONENTS.values():
            shutil.move(str(components / relative_name), str(bundle / relative_name))
        processor_output = bundle / "Processor"
        processor_output.mkdir()
        for name in KOSMOS2_COREML_PROCESSOR_ASSETS:
            shutil.copy2(processor / name, processor_output / name)
        licenses = bundle / "LICENSES"
        licenses.mkdir()
        (licenses / "MIT-Kosmos2.txt").write_bytes(_microsoft_mit_license())
        apache = Path(__file__).resolve().parents[2] / "licenses" / "Apache-2.0.txt"
        shutil.copy2(apache, licenses / "Apache-2.0.txt")
        (bundle / "NOTICE.txt").write_text(
            "LibreYOLO Kosmos-2 Core ML bundle\n\n"
            f"Model: {KOSMOS2_COREML_REPO}\n"
            f"Revision: {KOSMOS2_COREML_REVISION}\n"
            "Model license: MIT, Microsoft Corporation\n"
            "Runtime reference: Hugging Face Transformers "
            f"{KOSMOS2_COREML_TRANSFORMERS_VERSION} at "
            f"{KOSMOS2_COREML_TRANSFORMERS_COMMIT}\n"
            "Runtime reference license: Apache-2.0\n"
            "The bundle does not contain model.safetensors.\n",
            encoding="utf-8",
        )
        (bundle / "manifest.json").write_text(
            json.dumps(kosmos2_bundle_manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(bundle, destination)
    except Exception:
        shutil.rmtree(bundle, ignore_errors=True)
        raise
    return str(destination)


__all__ = [
    "KOSMOS2_COREML_COMPONENTS",
    "KOSMOS2_COREML_COMPONENT_CONTRACT",
    "KOSMOS2_COREML_CONTEXT_LENGTH",
    "KOSMOS2_COREML_EOS_TOKEN_ID",
    "KOSMOS2_COREML_HIDDEN_SIZE",
    "KOSMOS2_COREML_IMAGE_SIZE",
    "KOSMOS2_COREML_IMAGE_TOKENS",
    "KOSMOS2_COREML_MAX_NEW_TOKENS",
    "KOSMOS2_COREML_NO_REPEAT_NGRAM_SIZE",
    "KOSMOS2_COREML_PAD_TOKEN_ID",
    "KOSMOS2_COREML_PROCESSOR_ASSETS",
    "KOSMOS2_COREML_REPO",
    "KOSMOS2_COREML_REVISION",
    "KOSMOS2_COREML_TRANSFORMERS_COMMIT",
    "KOSMOS2_COREML_TRANSFORMERS_VERSION",
    "KOSMOS2_COREML_VOCAB_SIZE",
    "Kosmos2CoreMLDecoder",
    "Kosmos2CoreMLTokenEmbedding",
    "Kosmos2CoreMLVision",
    "build_kosmos2_coreml_bundle",
    "export_kosmos2_coreml_components",
    "kosmos2_bundle_manifest",
    "require_kosmos2_coreml_toolchain",
    "resolve_kosmos2_coreml_compute_units",
    "validate_kosmos2_processor_assets",
    "validate_kosmos2_source_model",
    "validate_kosmos2_weight_asset",
]
