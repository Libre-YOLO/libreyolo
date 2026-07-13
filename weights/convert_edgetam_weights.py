"""Convert the official EdgeTAM checkpoint into a Transformers snapshot.

The conversion is deliberately reproducible: both input repositories are pinned
to immutable revisions, every downloaded artifact is checked by SHA-256, and the
converted tensors must match the independently published reference snapshot
exactly before a payload is written.

The state-dict key conversion logic below is adapted from:

* https://github.com/huggingface/transformers/blob/
  bd37c453544e83eb875ed3608980a1660376007a/src/transformers/models/
  edgetam_video/convert_edgetam_video_to_hf.py
* Copyright 2025 The HuggingFace Inc. team
* Apache License 2.0

The upstream converter commit and the official EdgeTAM source are recorded in
the generated README and NOTICE. The model weights are independently converted;
the non-weight Transformers configuration, processor JSON, and .gitattributes
files are copied byte-for-byte from the separately hash-pinned Apache-2.0
reference snapshot, and that reuse is disclosed in both generated surfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

OFFICIAL_REPO = "facebook/EdgeTAM"
OFFICIAL_REVISION = "14d7ecc48c656b94e5184519f698cd5386c5a2bf"
OFFICIAL_CHECKPOINT = "edgetam.pt"
OFFICIAL_CHECKPOINT_SHA256 = (
    "ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df"
)

OFFICIAL_SOURCE_REPO = "https://github.com/facebookresearch/EdgeTAM"
OFFICIAL_SOURCE_COMMIT = "7711e012a30a2402c4eaab637bdb00a521302c91"
OFFICIAL_LICENSE_URL = (
    "https://raw.githubusercontent.com/facebookresearch/EdgeTAM/"
    f"{OFFICIAL_SOURCE_COMMIT}/LICENSE"
)
OFFICIAL_LICENSE_SHA256 = (
    "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
)

REFERENCE_REPO = "yonigozlan/EdgeTAM-hf"
REFERENCE_REVISION = "c266ce53b3fc00f0f495b583f6a116c4e57f53bb"
REFERENCE_FILE_SHA256 = {
    ".gitattributes": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    "config.json": "0827d003132f67efc496928f4fe5dd4125fcac518950c4eabe1d1f85e245c82a",
    "model.safetensors": "8858f8e4757b0b96dab8763f296ecffd845efbbbf698f64163cfa20a63d5fff4",
    "preprocessor_config.json": (
        "247f7be8f3979e98bc8e89b94c3e6542f6f1506070c97fbcc983f53a64a9bf86"
    ),
    "processor_config.json": (
        "f8a68e865cfad115c1c2763f3d93eca7b1c622da06da2a9273eb437fb2389b6d"
    ),
    "video_preprocessor_config.json": (
        "0b05a4bdcd7bb31e7bc12d5882e6a4f4ea7fcc8063d2240ab225e57561658233"
    ),
}

TRANSFORMERS_REPO = "https://github.com/huggingface/transformers"
TRANSFORMERS_CONVERTER_COMMIT = "bd37c453544e83eb875ed3608980a1660376007a"
TRANSFORMERS_CONVERTER_PATH = (
    "src/transformers/models/edgetam_video/convert_edgetam_video_to_hf.py"
)

TARGET_REPO = "LibreYOLO/LibreEdgeTAM"
PAYLOAD_FILES = frozenset(
    {
        ".gitattributes",
        "README.md",
        "LICENSE",
        "NOTICE",
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    }
)


# Adapted losslessly from the pinned Apache-2.0 Transformers converter above.
KEYS_TO_MODIFY_MAPPING = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "dwconv": "depthwise_conv",
    "pwconv": "pointwise_conv",
    "fuser": "memory_fuser",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": (
        "shared_embedding.positional_embedding"
    ),
    "obj_ptr_tpos_proj": "temporal_positional_encoding_projection_layer",
    "no_obj_embed_spatial": "occlusion_spatial_embedding_parameter",
    "sam_prompt_encoder": "prompt_encoder",
    "sam_mask_decoder": "mask_decoder",
    "maskmem_tpos_enc": "memory_temporal_positional_encoding",
    "gamma": "scale",
    "image_encoder.neck": "vision_encoder.neck",
    "image_encoder": "vision_encoder.backbone",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "pix_feat_proj": "feature_projection",
    "patch_embed.proj": "patch_embed.projection",
    "no_mem_embed": "no_memory_embedding",
    "no_mem_pos_enc": "no_memory_positional_encoding",
    "obj_ptr": "object_pointer",
    ".norm": ".layer_norm",
    "trunk.": "",
    "out_proj": "o_proj",
    "body.": "timm_model.",
    "ff.0": "mlp.layer_norm",
    "ff.1": "mlp.up_proj",
    "ff.3": "mlp.down_proj",
}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(
    path: str | Path, expected: str, *, label: str | None = None
) -> Path:
    """Reject an artifact whose bytes do not match the pinned digest."""
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Missing {label or 'artifact'}: {artifact}")
    actual = sha256_file(artifact)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {label or artifact.name}: expected {expected}, got {actual}"
        )
    return artifact


def validate_reference_dir(path: str | Path) -> Path:
    """Validate every reference artifact used by the conversion."""
    directory = Path(path).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Reference snapshot directory does not exist: {directory}"
        )
    for name, expected in REFERENCE_FILE_SHA256.items():
        require_sha256(directory / name, expected, label=f"reference {name}")
    return directory


def download_official_checkpoint(cache_dir: str | Path | None = None) -> Path:
    """Download and verify the immutable official EdgeTAM checkpoint."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=OFFICIAL_REPO,
        filename=OFFICIAL_CHECKPOINT,
        revision=OFFICIAL_REVISION,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return require_sha256(path, OFFICIAL_CHECKPOINT_SHA256, label="official edgetam.pt")


def download_reference_snapshot(cache_dir: str | Path | None = None) -> Path:
    """Download and verify the pinned Transformers reference snapshot."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=REFERENCE_REPO,
        revision=REFERENCE_REVISION,
        allow_patterns=list(REFERENCE_FILE_SHA256),
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return validate_reference_dir(path)


def get_official_license(path: str | Path | None = None) -> bytes:
    """Read the verbatim Apache-2.0 license from a local or pinned remote source."""
    if path is not None:
        license_path = require_sha256(
            path, OFFICIAL_LICENSE_SHA256, label="official EdgeTAM LICENSE"
        )
        return license_path.read_bytes()

    with urllib.request.urlopen(OFFICIAL_LICENSE_URL, timeout=30) as response:  # noqa: S310
        contents = response.read()
    actual = hashlib.sha256(contents).hexdigest()
    if actual != OFFICIAL_LICENSE_SHA256:
        raise ValueError(
            "SHA-256 mismatch for downloaded official EdgeTAM LICENSE: "
            f"expected {OFFICIAL_LICENSE_SHA256}, got {actual}"
        )
    return contents


def convert_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Perform the pinned lossless key/tensor conversion.

    This intentionally follows the operation order in the pinned Transformers
    converter. That order is part of the reproducible conversion contract.
    """
    converted: dict[str, torch.Tensor] = {}

    output_hypernetworks_mlps_pattern = (
        r".*.output_hypernetworks_mlps.(\d+).layers.(\d+).*"
    )
    output_mask_decoder_mlps_pattern = (
        r"mask_decoder.transformer.layers.(\d+).mlp.layers.(\d+).*"
    )
    output_mask_decoder_score_head_pattern = (
        r"mask_decoder.pred_obj_score_head.layers.(\d+).*"
    )
    output_vision_encoder_mlps_pattern = (
        r"vision_encoder.backbone.blocks.(\d+).mlp.layers.(\d+).*"
    )
    output_vision_encoder_neck_pattern = r"vision_encoder.neck.convs.(\d+).conv"
    output_memory_encoder_projection_pattern = r"memory_encoder.o_proj.*"
    memory_attention_pattern = r"memory_attention.*"
    output_object_pointer_proj_pattern = r"object_pointer_proj.layers.(\d+).*"
    output_memory_encoder_mask_downsampler_pattern = (
        r"memory_encoder.mask_downsampler.encoder.(\d+).*"
    )
    perceiver_resampler_patterns = {
        r"spatial_perceiver.latents": r"spatial_perceiver.latents_1d",
        r"spatial_perceiver.latents_1d_2d": r"spatial_perceiver.latents_2d",
        r"spatial_perceiver.layers.(\d+).attn.layer_norm_x": (
            r"spatial_perceiver.layers.\1.layer_norm_input"
        ),
        r"spatial_perceiver.layers.(\d+).attn.layer_norm_latents": (
            r"spatial_perceiver.layers.\1.layer_norm_latents"
        ),
        r"spatial_perceiver.layers.(\d+).self_attn.layer_norm": (
            r"spatial_perceiver.layers.\1.layer_norm_self"
        ),
        r"spatial_perceiver.layers.(\d+).attn.to_q": (
            r"spatial_perceiver.layers.\1.cross_attention.q_proj"
        ),
        r"spatial_perceiver.layers.(\d+).attn.to_kv": (
            r"spatial_perceiver.layers.\1.cross_attention.kv_proj_combined"
        ),
        r"spatial_perceiver.layers.(\d+).attn.to_out": (
            r"spatial_perceiver.layers.\1.cross_attention.o_proj"
        ),
        r"spatial_perceiver.layers.(\d+).self_attn.to_q": (
            r"spatial_perceiver.layers.\1.self_attention.q_proj"
        ),
        r"spatial_perceiver.layers.(\d+).self_attn.to_kv": (
            r"spatial_perceiver.layers.\1.self_attention.kv_proj_combined"
        ),
        r"spatial_perceiver.layers.(\d+).self_attn.to_out": (
            r"spatial_perceiver.layers.\1.self_attention.o_proj"
        ),
        r"spatial_perceiver.layers.(\d+).attn": (
            r"spatial_perceiver.layers.\1.cross_attention"
        ),
        r"spatial_perceiver.layers.(\d+).self_attn": (
            r"spatial_perceiver.layers.\1.self_attention"
        ),
    }

    for source_key, tensor in state_dict.items():
        key = source_key
        for old, new in KEYS_TO_MODIFY_MAPPING.items():
            if old in key:
                key = key.replace(old, new)

        for pattern, replacement in perceiver_resampler_patterns.items():
            if re.match(pattern, key):
                key = re.sub(pattern, replacement, key)

        match = re.match(output_vision_encoder_mlps_pattern, key)
        if match:
            layer_number = int(match.group(2))
            if layer_number == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_number == 1:
                key = key.replace("layers.1", "proj_out")

        if re.match(memory_attention_pattern, key):
            key = key.replace("linear1", "mlp.up_proj")
            key = key.replace("linear2", "mlp.down_proj")

        match = re.match(output_mask_decoder_mlps_pattern, key)
        if match:
            layer_number = int(match.group(2))
            if layer_number == 0:
                key = key.replace("mlp.layers.0", "mlp.proj_in")
            elif layer_number == 1:
                key = key.replace("mlp.layers.1", "mlp.proj_out")

        match = re.match(output_mask_decoder_score_head_pattern, key)
        if match:
            layer_number = int(match.group(1))
            if layer_number == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_number == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_number == 2:
                key = key.replace("layers.2", "proj_out")

        match = re.match(output_hypernetworks_mlps_pattern, key)
        if match:
            layer_number = int(match.group(2))
            if layer_number == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_number == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_number == 2:
                key = key.replace("layers.2", "proj_out")

        if re.match(output_vision_encoder_neck_pattern, key):
            key = key.replace(".conv.", ".")

        if re.match(output_memory_encoder_projection_pattern, key):
            key = key.replace(".o_proj.", ".projection.")

        match = re.match(output_object_pointer_proj_pattern, key)
        if match:
            layer_number = int(match.group(1))
            if layer_number == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_number == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_number == 2:
                key = key.replace("layers.2", "proj_out")
            key = key.replace("layers.2", "proj_out")

        match = re.match(output_memory_encoder_mask_downsampler_pattern, key)
        if match:
            layer_number = int(match.group(1))
            if layer_number == 12:
                key = key.replace(f"encoder.{layer_number}", "final_conv")
            elif layer_number % 3 == 0:
                key = key.replace(
                    f"encoder.{layer_number}", f"layers.{layer_number // 3}.conv"
                )
            elif layer_number % 3 == 1:
                key = key.replace(
                    f"encoder.{layer_number}",
                    f"layers.{layer_number // 3}.layer_norm",
                )

        if "kv_proj_combined" in key:
            key_projection, value_projection = torch.chunk(tensor, 2, dim=0)
            key_key = key.replace("kv_proj_combined", "k_proj")
            value_key = key.replace("kv_proj_combined", "v_proj")
            if key_key in converted or value_key in converted:
                raise KeyError(
                    f"Duplicate converted key while splitting {source_key!r}"
                )
            converted[key_key] = key_projection
            converted[value_key] = value_projection
            continue

        if key in converted:
            raise KeyError(f"Duplicate converted key {key!r} from {source_key!r}")
        converted[key] = tensor

    positional_key = "prompt_encoder.shared_embedding.positional_embedding"
    shared_positional_key = "shared_image_embedding.positional_embedding"
    if positional_key not in converted:
        raise KeyError(f"Converted checkpoint is missing {positional_key!r}")
    converted[shared_positional_key] = converted[positional_key]

    point_prefix = "prompt_encoder.point_embed."
    point_keys = [f"{point_prefix}{index}.weight" for index in range(4)]
    missing = [key for key in point_keys if key not in converted]
    if missing:
        raise KeyError(f"Converted checkpoint is missing point embeddings: {missing}")
    point_embeddings = [converted.pop(key) for key in point_keys]
    converted[f"{point_prefix}weight"] = torch.cat(point_embeddings, dim=0)
    return converted


def load_official_state_dict(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    """Safely load and validate the official checkpoint envelope."""
    from libreyolo.utils.serialization import load_untrusted_torch_file

    checkpoint_path = require_sha256(
        checkpoint, OFFICIAL_CHECKPOINT_SHA256, label="official edgetam.pt"
    )
    envelope = load_untrusted_torch_file(
        checkpoint_path,
        map_location="cpu",
        context="official EdgeTAM checkpoint conversion",
    )
    if not isinstance(envelope, dict) or set(envelope) != {"model"}:
        raise ValueError(
            "Unexpected official checkpoint envelope; expected exactly one 'model' entry"
        )
    state_dict = envelope["model"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            "Official checkpoint 'model' entry is not a non-empty state dict"
        )
    invalid = [
        key
        for key, value in state_dict.items()
        if not isinstance(key, str) or not torch.is_tensor(value)
    ]
    if invalid:
        raise TypeError(
            f"Official checkpoint contains non-tensor state entries: {invalid[:5]}"
        )
    return state_dict


def load_safetensors_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a local safetensors state dict onto CPU."""
    from safetensors.torch import load_file

    return load_file(str(Path(path).resolve()), device="cpu")


def compare_state_dicts(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    *,
    actual_label: str = "converted",
    expected_label: str = "reference",
) -> int:
    """Require identical key sets, tensor metadata, and tensor values."""
    actual_keys = set(actual)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise AssertionError(
            f"State-dict keys differ ({actual_label} vs {expected_label}); "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )

    for key in sorted(expected_keys):
        actual_tensor = actual[key].detach().cpu()
        expected_tensor = expected[key].detach().cpu()
        if actual_tensor.shape != expected_tensor.shape:
            raise AssertionError(
                f"Shape mismatch for {key}: {actual_tensor.shape} != {expected_tensor.shape}"
            )
        if actual_tensor.dtype != expected_tensor.dtype:
            raise AssertionError(
                f"Dtype mismatch for {key}: {actual_tensor.dtype} != {expected_tensor.dtype}"
            )
        if not torch.equal(actual_tensor, expected_tensor):
            if actual_tensor.is_floating_point():
                maximum = (actual_tensor - expected_tensor).abs().max().item()
                detail = f"; max absolute difference={maximum}"
            else:
                detail = ""
            raise AssertionError(f"Tensor mismatch for {key}{detail}")
    return len(expected_keys)


def strict_load_model(
    config_path: str | Path, state_dict: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Strict-load the converted state into the installed Transformers model."""
    from libreyolo.models.sam.edgetam import (
        _clear_edgetam_config_serialization_hints,
        _load_edgetam_video_config,
    )
    from transformers import EdgeTamVideoModel

    config = _load_edgetam_video_config(Path(config_path).resolve())
    model = None
    try:
        model = EdgeTamVideoModel(config)
    finally:
        _clear_edgetam_config_serialization_hints(config)
        if model is not None:
            _clear_edgetam_config_serialization_hints(model.config)
    if model is None:  # Defensive: Transformers always returns a model or raises.
        raise RuntimeError("Transformers returned no EdgeTamVideoModel")
    incompatible = model.load_state_dict(dict(state_dict), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"Strict model load failed: {incompatible}")
    model.eval()
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def validate_payload(directory: str | Path) -> dict[str, Any]:
    """Perform offline structural validation of a generated mirror payload."""
    payload = Path(directory).resolve()
    if not payload.is_dir():
        raise FileNotFoundError(f"Payload directory does not exist: {payload}")
    actual_files = {
        path.relative_to(payload).as_posix()
        for path in payload.rglob("*")
        if path.is_file()
    }
    if actual_files != PAYLOAD_FILES:
        missing = sorted(PAYLOAD_FILES - actual_files)
        unexpected = sorted(actual_files - PAYLOAD_FILES)
        raise ValueError(
            f"Invalid payload file set; missing={missing}, unexpected={unexpected}"
        )
    if list(payload.rglob("*.pt")):
        raise ValueError(
            "Raw .pt checkpoints must not be included in the mirror payload"
        )

    require_sha256(
        payload / "LICENSE", OFFICIAL_LICENSE_SHA256, label="payload LICENSE"
    )
    for name in REFERENCE_FILE_SHA256:
        require_sha256(
            payload / name,
            REFERENCE_FILE_SHA256[name],
            label=f"payload {name}",
        )

    with (payload / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("architectures") != ["EdgeTamVideoModel"]:
        raise ValueError("Payload config does not declare EdgeTamVideoModel")
    if config.get("model_type") != "edgetam_video":
        raise ValueError("Payload config does not declare model_type=edgetam_video")
    if config.get("image_size") != 1024:
        raise ValueError("Payload config does not declare image_size=1024")

    state_dict = load_safetensors_state(payload / "model.safetensors")
    if not state_dict:
        raise ValueError("Payload model.safetensors contains no tensors")
    return {
        "directory": str(payload),
        "files": len(actual_files),
        "tensors": len(state_dict),
        "model_sha256": sha256_file(payload / "model.safetensors"),
    }


def _render_readme(model_sha256: str, tensor_count: int) -> str:
    return f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: mask-generation
tags:
- libreyolo
- edgetam
- promptable-segmentation
- image-segmentation
base_model: facebook/EdgeTAM
---

# LibreEdgeTAM

This is LibreYOLO's Transformers-compatible mirror of the official EdgeTAM
checkpoint. The raw pickle-based `.pt` file is deliberately not included.

## Source

- Official checkpoint: [`{OFFICIAL_REPO}`](https://huggingface.co/{OFFICIAL_REPO})
  at revision `{OFFICIAL_REVISION}`
- `edgetam.pt` SHA-256: `{OFFICIAL_CHECKPOINT_SHA256}`
- Official source: [{OFFICIAL_SOURCE_REPO}]({OFFICIAL_SOURCE_REPO}) at commit
  `{OFFICIAL_SOURCE_COMMIT}`
- Reference Transformers snapshot: [`{REFERENCE_REPO}`](https://huggingface.co/{REFERENCE_REPO})
  at revision `{REFERENCE_REVISION}`; its model card declares Apache-2.0
- Conversion implementation: [`huggingface/transformers`]({TRANSFORMERS_REPO}) at
  commit `{TRANSFORMERS_CONVERTER_COMMIT}`, file `{TRANSFORMERS_CONVERTER_PATH}`

## Modifications

LibreYOLO independently converted the model weights from the safely loaded
official checkpoint using the pinned Apache-2.0 Transformers conversion (key
remapping, lossless key/value tensor splitting, and point-embedding
concatenation). It strict-loaded `EdgeTamVideoModel` and checked all
{tensor_count} resulting tensors for exact equality with the pinned reference.
No learned numeric parameter was changed, and `model.safetensors` was not copied
from the reference.

The non-weight `.gitattributes`, `config.json`, `preprocessor_config.json`,
`processor_config.json`, and `video_preprocessor_config.json` files are copied
byte-for-byte from the hash-pinned reference revision above. Every copied file
is SHA-256 verified. The reference repository declares Apache-2.0 in its model
card.

Generated `model.safetensors` SHA-256: `{model_sha256}`.

## Usage

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("{TARGET_REPO}", trust_remote_code=False)
```

LibreYOLO users can load it through `LibreEdgeTAM` once the EdgeTAM integration is
installed.

## License

EdgeTAM code and model checkpoints are licensed under Apache License 2.0. The
verbatim upstream license is included in `LICENSE`; provenance and modification
details are included in `NOTICE`.
"""


def _render_notice(model_sha256: str, tensor_count: int) -> str:
    return f"""LibreEdgeTAM provenance notice

Official model
--------------
Repository: https://huggingface.co/{OFFICIAL_REPO}
Revision: {OFFICIAL_REVISION}
File: {OFFICIAL_CHECKPOINT}
SHA-256: {OFFICIAL_CHECKPOINT_SHA256}

Official source and license
---------------------------
Repository: {OFFICIAL_SOURCE_REPO}
Commit: {OFFICIAL_SOURCE_COMMIT}
License: Apache License 2.0
LICENSE SHA-256: {OFFICIAL_LICENSE_SHA256}

Conversion implementation
-------------------------
Repository: {TRANSFORMERS_REPO}
Commit: {TRANSFORMERS_CONVERTER_COMMIT}
File: {TRANSFORMERS_CONVERTER_PATH}
Copyright 2025 The HuggingFace Inc. team
License: Apache License 2.0

Hash-pinned snapshot metadata source and conversion reference
-------------------------------------------------------------
Repository: https://huggingface.co/{REFERENCE_REPO}
Revision: {REFERENCE_REVISION}
License declared by reference model card: Apache License 2.0
model.safetensors SHA-256: {REFERENCE_FILE_SHA256["model.safetensors"]}
Files copied byte-for-byte after SHA-256 verification:
  .gitattributes
  config.json
  preprocessor_config.json
  processor_config.json
  video_preprocessor_config.json

LibreYOLO modifications
-----------------------
The official checkpoint weights were independently safe-loaded and converted
through lossless key remapping, key/value tensor splitting, and point-embedding
concatenation; model.safetensors was not copied from the reference. The result
was strict-loaded into EdgeTamVideoModel. All {tensor_count} state tensors were
checked for exact equality with the pinned reference snapshot. The five
non-weight files listed above were copied from that reference revision. The raw
.pt checkpoint was omitted from this mirror.

Generated model.safetensors SHA-256: {model_sha256}
"""


def write_payload(
    output: str | Path,
    *,
    reference_dir: str | Path,
    model_state: Mapping[str, torch.Tensor],
    license_contents: bytes,
) -> dict[str, Any]:
    """Atomically write and validate the complete local mirror payload."""
    from safetensors.torch import save_file

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {destination}. Choose a new directory."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    reference = validate_reference_dir(reference_dir)

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=str(destination.parent)
    ) as temporary:
        staging = Path(temporary)
        for name in REFERENCE_FILE_SHA256:
            if name == "model.safetensors":
                continue
            shutil.copyfile(reference / name, staging / name)

        (staging / "LICENSE").write_bytes(license_contents)
        require_sha256(
            staging / "LICENSE", OFFICIAL_LICENSE_SHA256, label="staged LICENSE"
        )

        # Clone shared tensors so safetensors records both model keys explicitly.
        serializable = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in model_state.items()
        }
        save_file(
            serializable, str(staging / "model.safetensors"), metadata={"format": "pt"}
        )
        saved_state = load_safetensors_state(staging / "model.safetensors")
        tensor_count = compare_state_dicts(
            saved_state,
            model_state,
            actual_label="saved payload",
            expected_label="strict-loaded model",
        )
        require_sha256(
            staging / "model.safetensors",
            REFERENCE_FILE_SHA256["model.safetensors"],
            label="staged model.safetensors",
        )
        model_sha256 = REFERENCE_FILE_SHA256["model.safetensors"]

        (staging / "README.md").write_text(
            _render_readme(model_sha256, tensor_count), encoding="utf-8", newline="\n"
        )
        (staging / "NOTICE").write_text(
            _render_notice(model_sha256, tensor_count), encoding="utf-8", newline="\n"
        )
        staging.rename(destination)

    return validate_payload(destination)


def convert_checkpoint(
    checkpoint: str | Path,
    reference_dir: str | Path,
    output: str | Path,
    *,
    upstream_license: str | Path | None = None,
) -> dict[str, Any]:
    """Run safe loading, exact reference parity, strict loading, and serialization."""
    reference = validate_reference_dir(reference_dir)
    source_state = load_official_state_dict(checkpoint)
    converted = convert_state_dict(source_state)
    reference_state = load_safetensors_state(reference / "model.safetensors")
    converted_count = compare_state_dicts(converted, reference_state)

    model_state = strict_load_model(reference / "config.json", converted)
    loaded_count = compare_state_dicts(
        model_state,
        reference_state,
        actual_label="strict-loaded model",
        expected_label="reference",
    )
    if converted_count != loaded_count:
        raise AssertionError("Tensor counts changed during strict model loading")

    result = write_payload(
        output,
        reference_dir=reference,
        model_state=model_state,
        license_contents=get_official_license(upstream_license),
    )
    result.update(
        {
            "checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
            "reference_revision": REFERENCE_REVISION,
            "exact_reference_tensors": loaded_count,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Local official edgetam.pt. Downloads the pinned file when omitted.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Local pinned EdgeTAM-hf snapshot. Downloads it when omitted.",
    )
    parser.add_argument(
        "--upstream-license",
        type=Path,
        help="Local verbatim official LICENSE. Downloads the pinned file when omitted.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Optional Hugging Face download cache directory."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_hf_build_LibreEdgeTAM"),
        help="New output directory (default: %(default)s). Existing paths are refused.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require all three local input arguments; perform no network requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offline:
        missing = [
            name
            for name, value in (
                ("--checkpoint", args.checkpoint),
                ("--reference-dir", args.reference_dir),
                ("--upstream-license", args.upstream_license),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(f"--offline requires local inputs: {', '.join(missing)}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    checkpoint = args.checkpoint or download_official_checkpoint(args.cache_dir)
    reference_dir = args.reference_dir or download_reference_snapshot(args.cache_dir)
    result = convert_checkpoint(
        checkpoint,
        reference_dir,
        args.output,
        upstream_license=args.upstream_license,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
