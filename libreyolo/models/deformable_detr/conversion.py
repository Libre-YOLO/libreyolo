# SPDX-License-Identifier: Apache-2.0
# Native target layout: fundamentalvision/Deformable-DETR at
# 11169a60c33333af00a4849f1808023eba96a931 (Apache-2.0).
# Transformers source layout and conversion mapping: huggingface/transformers
# at 4a224b1e2182d1f8f27d1d76fb8de6ab40b7ff62 (Apache-2.0).
"""Convert official Transformers Deformable DETR weights to native keys."""

from __future__ import annotations

import re

import torch


_HF_DECODER_QKV = re.compile(
    r"^model\.decoder\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<projection>[qkv])_proj\.(?P<parameter>weight|bias)$"
)


def is_hf_deformable_detr_state_dict(state_dict: dict) -> bool:
    """Return whether keys identify the official Transformers architecture."""
    has_head = (
        "class_embed.0.weight" in state_dict
        or "model.decoder.class_embed.0.weight" in state_dict
    )
    return (
        "model.backbone.conv_encoder.model.conv1.weight" in state_dict
        and "model.encoder.layers.0.self_attn.sampling_offsets.weight" in state_dict
        and "model.input_proj.0.0.weight" in state_dict
        and "model.level_embed" in state_dict
        and has_head
    )


def _map_encoder_key(key: str) -> str:
    key = key.replace("model.encoder.layers.", "transformer.encoder.layers.", 1)
    return (
        key.replace(".self_attn_layer_norm.", ".norm1.")
        .replace(".final_layer_norm.", ".norm2.")
        .replace(".fc1.", ".linear1.")
        .replace(".fc2.", ".linear2.")
    )


def _map_decoder_key(key: str) -> str:
    key = key.replace("model.decoder.layers.", "transformer.decoder.layers.", 1)
    return (
        key.replace(".encoder_attn_layer_norm.", ".norm1.")
        .replace(".self_attn_layer_norm.", ".norm2.")
        .replace(".final_layer_norm.", ".norm3.")
        .replace(".encoder_attn.", ".cross_attn.")
        .replace(".fc1.", ".linear1.")
        .replace(".fc2.", ".linear2.")
    )


def _copy_aliases(state_dict: dict[str, torch.Tensor], source: str, target: str):
    for key, value in list(state_dict.items()):
        if key.startswith(source):
            state_dict[target + key[len(source) :]] = value


def _head_indices(state_dict: dict[str, torch.Tensor], prefix: str) -> set[int]:
    indices = set()
    for key in state_dict:
        match = re.match(re.escape(prefix) + r"(\d+)\.", key)
        if match:
            indices.add(int(match.group(1)))
    return indices


def _restore_head_aliases(state_dict: dict[str, torch.Tensor]) -> None:
    decoder_bbox = _head_indices(state_dict, "transformer.decoder.bbox_embed.")
    decoder_class = _head_indices(state_dict, "transformer.decoder.class_embed.")

    if decoder_bbox:
        _copy_aliases(state_dict, "transformer.decoder.bbox_embed.", "bbox_embed.")
    if decoder_class:
        _copy_aliases(state_dict, "transformer.decoder.class_embed.", "class_embed.")

    # In the non-refinement releases, all six top-level prediction heads are
    # the same module. Safe serialization stores that tied module only once.
    if not decoder_bbox:
        class_indices = _head_indices(state_dict, "class_embed.")
        bbox_indices = _head_indices(state_dict, "bbox_embed.")
        if class_indices == {0} and bbox_indices == {0}:
            for index in range(1, 6):
                _copy_aliases(state_dict, "class_embed.0.", f"class_embed.{index}.")
                _copy_aliases(state_dict, "bbox_embed.0.", f"bbox_embed.{index}.")


def convert_hf_deformable_detr_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map official Hugging Face keys to the pinned native architecture.

    Decoder Q/K/V projections are concatenated into PyTorch
    ``MultiheadAttention.in_proj_*`` tensors. Tied top-level/decoder prediction
    modules omitted by safetensors are restored so strict loading remains a
    meaningful integrity check.
    """
    if not is_hf_deformable_detr_state_dict(state_dict):
        raise ValueError("State dict is not an official Transformers Deformable DETR")

    converted: dict[str, torch.Tensor] = {}
    qkv: dict[tuple[int, str], dict[str, torch.Tensor]] = {}

    for key, value in state_dict.items():
        if key.endswith(".num_batches_tracked"):
            # Transformers uses ordinary BatchNorm in the ResNet wrapper;
            # upstream's FrozenBatchNorm has no tracking counter.
            continue

        qkv_match = _HF_DECODER_QKV.match(key)
        if qkv_match:
            group = (
                int(qkv_match.group("layer")),
                qkv_match.group("parameter"),
            )
            qkv.setdefault(group, {})[qkv_match.group("projection")] = value
            continue

        if key.startswith("model.backbone.conv_encoder.model."):
            native_key = key.replace(
                "model.backbone.conv_encoder.model.", "backbone.0.body.", 1
            )
        elif key.startswith("model.encoder.layers."):
            native_key = _map_encoder_key(key)
        elif key.startswith("model.decoder.layers."):
            native_key = _map_decoder_key(key)
        elif key.startswith("model.decoder.bbox_embed."):
            native_key = key.replace(
                "model.decoder.bbox_embed.",
                "transformer.decoder.bbox_embed.",
                1,
            )
        elif key.startswith("model.decoder.class_embed."):
            native_key = key.replace(
                "model.decoder.class_embed.",
                "transformer.decoder.class_embed.",
                1,
            )
        elif key.startswith("model.input_proj."):
            native_key = key.removeprefix("model.")
        elif key == "model.level_embed":
            native_key = "transformer.level_embed"
        elif key.startswith("model.query_position_embeddings."):
            native_key = key.replace(
                "model.query_position_embeddings.", "query_embed.", 1
            )
        elif key.startswith("model.reference_points."):
            native_key = key.replace(
                "model.reference_points.", "transformer.reference_points.", 1
            )
        elif key.startswith("model.enc_output_norm."):
            native_key = key.replace(
                "model.enc_output_norm.", "transformer.enc_output_norm.", 1
            )
        elif key.startswith("model.enc_output."):
            native_key = key.replace("model.enc_output.", "transformer.enc_output.", 1)
        elif key.startswith("model.pos_trans_norm."):
            native_key = key.replace(
                "model.pos_trans_norm.", "transformer.pos_trans_norm.", 1
            )
        elif key.startswith("model.pos_trans."):
            native_key = key.replace("model.pos_trans.", "transformer.pos_trans.", 1)
        elif key.startswith(("class_embed.", "bbox_embed.")):
            native_key = key
        else:
            raise ValueError(f"Unrecognized Transformers parameter key: {key}")
        converted[native_key] = value

    expected_groups = {
        (layer, kind) for layer in range(6) for kind in ("weight", "bias")
    }
    if set(qkv) != expected_groups:
        missing = sorted(expected_groups - set(qkv))
        extra = sorted(set(qkv) - expected_groups)
        raise ValueError(
            "Incomplete decoder Q/K/V projection groups: "
            f"missing={missing}, extra={extra}"
        )
    for (layer, parameter), projections in qkv.items():
        if set(projections) != {"q", "k", "v"}:
            raise ValueError(
                f"Decoder layer {layer} {parameter} has projections "
                f"{sorted(projections)}, expected q/k/v"
            )
        converted[
            f"transformer.decoder.layers.{layer}.self_attn.in_proj_{parameter}"
        ] = torch.cat([projections[name] for name in ("q", "k", "v")], dim=0)

    _restore_head_aliases(converted)
    return converted


__all__ = [
    "convert_hf_deformable_detr_state_dict",
    "is_hf_deformable_detr_state_dict",
]
