"""Transformers-to-native conversion contracts for Deformable DETR."""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.deformable_detr.conversion import (
    convert_hf_deformable_detr_state_dict,
    is_hf_deformable_detr_state_dict,
)

pytestmark = pytest.mark.unit


def _hf_base_fixture() -> dict[str, torch.Tensor]:
    state = {
        "model.backbone.conv_encoder.model.conv1.weight": torch.zeros(2, 3, 1, 1),
        "model.backbone.conv_encoder.model.layer1.0.downsample.1.num_batches_tracked": torch.tensor(
            0
        ),
        "model.encoder.layers.0.self_attn.sampling_offsets.weight": torch.zeros(2, 2),
        "model.encoder.layers.0.self_attn_layer_norm.weight": torch.ones(2),
        "model.encoder.layers.0.final_layer_norm.weight": torch.ones(2),
        "model.encoder.layers.0.fc1.weight": torch.zeros(4, 2),
        "model.encoder.layers.0.fc2.weight": torch.zeros(2, 4),
        "model.decoder.layers.0.encoder_attn.sampling_offsets.weight": torch.zeros(
            2, 2
        ),
        "model.decoder.layers.0.encoder_attn_layer_norm.weight": torch.ones(2),
        "model.decoder.layers.0.self_attn_layer_norm.weight": torch.ones(2),
        "model.decoder.layers.0.final_layer_norm.weight": torch.ones(2),
        "model.decoder.layers.0.fc1.weight": torch.zeros(4, 2),
        "model.decoder.layers.0.fc2.weight": torch.zeros(2, 4),
        "model.input_proj.0.0.weight": torch.zeros(2, 2, 1, 1),
        "model.level_embed": torch.zeros(4, 2),
        "model.query_position_embeddings.weight": torch.zeros(3, 4),
        "model.reference_points.weight": torch.zeros(2, 2),
        "class_embed.0.weight": torch.zeros(91, 2),
        "class_embed.0.bias": torch.zeros(91),
        "bbox_embed.0.layers.0.weight": torch.zeros(2, 2),
    }
    for layer in range(6):
        for parameter, shape in (("weight", (2, 2)), ("bias", (2,))):
            for value, projection in enumerate(("q", "k", "v"), start=1):
                state[
                    f"model.decoder.layers.{layer}.self_attn."
                    f"{projection}_proj.{parameter}"
                ] = torch.full(shape, float(value))
    return state


def test_hf_layout_is_recognized_and_unrelated_layout_is_rejected():
    state = _hf_base_fixture()
    assert is_hf_deformable_detr_state_dict(state) is True

    state.pop("model.backbone.conv_encoder.model.conv1.weight")
    assert is_hf_deformable_detr_state_dict(state) is False
    with pytest.raises(ValueError, match="not an official Transformers"):
        convert_hf_deformable_detr_state_dict(state)


def test_hf_keys_map_to_native_names_and_tracking_counters_are_dropped():
    converted = convert_hf_deformable_detr_state_dict(_hf_base_fixture())

    expected = {
        "backbone.0.body.conv1.weight",
        "transformer.encoder.layers.0.self_attn.sampling_offsets.weight",
        "transformer.encoder.layers.0.norm1.weight",
        "transformer.encoder.layers.0.norm2.weight",
        "transformer.encoder.layers.0.linear1.weight",
        "transformer.encoder.layers.0.linear2.weight",
        "transformer.decoder.layers.0.cross_attn.sampling_offsets.weight",
        "transformer.decoder.layers.0.norm1.weight",
        "transformer.decoder.layers.0.norm2.weight",
        "transformer.decoder.layers.0.norm3.weight",
        "transformer.decoder.layers.0.linear1.weight",
        "transformer.decoder.layers.0.linear2.weight",
        "input_proj.0.0.weight",
        "transformer.level_embed",
        "query_embed.weight",
        "transformer.reference_points.weight",
    }
    assert expected <= set(converted)
    assert not any(key.endswith("num_batches_tracked") for key in converted)
    assert not any(key.startswith("model.") for key in converted)


def test_decoder_qkv_is_concatenated_in_pytorch_order():
    converted = convert_hf_deformable_detr_state_dict(_hf_base_fixture())

    weight = converted["transformer.decoder.layers.0.self_attn.in_proj_weight"]
    bias = converted["transformer.decoder.layers.0.self_attn.in_proj_bias"]
    torch.testing.assert_close(
        weight, torch.cat([torch.ones(2, 2) * n for n in (1, 2, 3)])
    )
    torch.testing.assert_close(bias, torch.cat([torch.ones(2) * n for n in (1, 2, 3)]))


def test_tied_base_heads_are_restored_for_strict_loading():
    converted = convert_hf_deformable_detr_state_dict(_hf_base_fixture())

    for index in range(6):
        assert f"class_embed.{index}.weight" in converted
        assert f"class_embed.{index}.bias" in converted
        assert f"bbox_embed.{index}.layers.0.weight" in converted
        assert (
            converted[f"class_embed.{index}.weight"].data_ptr()
            == converted["class_embed.0.weight"].data_ptr()
        )


def test_refinement_and_two_stage_decoder_head_aliases_are_restored():
    refine = _hf_base_fixture()
    refine.pop("bbox_embed.0.layers.0.weight")
    for index in range(6):
        refine[f"class_embed.{index}.weight"] = torch.full((91, 2), index)
        refine[f"model.decoder.bbox_embed.{index}.layers.0.weight"] = torch.full(
            (2, 2), index
        )
    converted_refine = convert_hf_deformable_detr_state_dict(refine)
    assert "bbox_embed.5.layers.0.weight" in converted_refine
    assert "transformer.decoder.bbox_embed.5.layers.0.weight" in converted_refine
    assert "transformer.decoder.class_embed.0.weight" not in converted_refine

    two_stage = _hf_base_fixture()
    two_stage.pop("class_embed.0.weight")
    two_stage.pop("class_embed.0.bias")
    two_stage.pop("bbox_embed.0.layers.0.weight")
    two_stage.pop("model.query_position_embeddings.weight")
    two_stage["model.enc_output.weight"] = torch.zeros(2, 2)
    for index in range(7):
        two_stage[f"model.decoder.class_embed.{index}.weight"] = torch.zeros(91, 2)
        two_stage[f"model.decoder.bbox_embed.{index}.layers.0.weight"] = torch.zeros(
            2, 2
        )
    converted_two_stage = convert_hf_deformable_detr_state_dict(two_stage)
    assert "class_embed.6.weight" in converted_two_stage
    assert "bbox_embed.6.layers.0.weight" in converted_two_stage
    assert "transformer.decoder.class_embed.6.weight" in converted_two_stage
    assert "transformer.decoder.bbox_embed.6.layers.0.weight" in converted_two_stage


def test_incomplete_decoder_projection_group_fails_closed():
    state = _hf_base_fixture()
    state.pop("model.decoder.layers.5.self_attn.v_proj.bias")
    with pytest.raises(ValueError, match="projections.*expected q/k/v"):
        convert_hf_deformable_detr_state_dict(state)


def test_family_runtime_hook_converts_hf_layout():
    from libreyolo import LibreDeformableDETR

    converted = LibreDeformableDETR.convert_upstream_state_dict(_hf_base_fixture())
    assert converted is not None
    assert LibreDeformableDETR.can_load(converted) is True
    assert "transformer.decoder.layers.0.self_attn.in_proj_weight" in converted


def test_runtime_autoconvert_uses_hf_repository_directory_for_dc5(tmp_path):
    from libreyolo.models.autoconvert import autoconvert_upstream_checkpoint

    source = (
        tmp_path
        / "models--SenseTime--deformable-detr-single-scale-dc5"
        / "snapshots"
        / "revision"
        / "model.safetensors"
    )
    source.parent.mkdir(parents=True)
    source.write_bytes(b"loaded state is supplied directly")

    native = {
        "backbone.0.body.conv1.weight": torch.zeros(64, 3, 7, 7),
        "transformer.encoder.layers.0.self_attn.sampling_offsets.weight": torch.zeros(
            64, 256
        ),
        "transformer.level_embed": torch.zeros(1, 256),
        "input_proj.0.0.weight": torch.zeros(256, 2048, 1, 1),
        "class_embed.0.weight": torch.zeros(91, 256),
        "bbox_embed.0.layers.0.weight": torch.zeros(256, 256),
    }
    output_path = autoconvert_upstream_checkpoint(str(source), loaded=native)

    assert output_path is not None
    checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    assert checkpoint["model_family"] == "deformable_detr"
    assert checkpoint["size"] == "r50ssdc5"
    assert checkpoint["task"] == "detect"
    assert checkpoint["nc"] == 80
    assert checkpoint["imgsz"] == 800
