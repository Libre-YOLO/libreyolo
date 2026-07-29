"""Offline contract tests for split promptable SAM Core ML components."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.unit, pytest.mark.sam]


def _profile(
    family: str = "edgetam",
    size: str | None = None,
    *,
    prompt_max_points: int = 16,
):
    from libreyolo.export.coreml_sam import validate_sam_coreml_profile

    default_sizes = {
        "edgetam": "edge",
        "mobilesam": "tiny",
        "sam": "base",
        "sam2": "tiny",
        "sam3": "large",
    }
    return validate_sam_coreml_profile(
        family=family,
        size=size or default_sizes[family],
        prompt_max_points=prompt_max_points,
    )


def _stringify_metadata(metadata: dict) -> dict[str, str]:
    result = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list, tuple)):
            result[key] = json.dumps(value, ensure_ascii=False)
        else:
            result[key] = str(value)
    return result


@pytest.mark.parametrize(
    ("family", "sizes", "embedding_shapes", "mask_size"),
    [
        (
            "edgetam",
            ("edge",),
            (
                (1, 32, 256, 256),
                (1, 64, 128, 128),
                (1, 256, 64, 64),
            ),
            256,
        ),
        ("mobilesam", ("tiny",), ((1, 256, 64, 64),), 256),
        (
            "sam",
            ("base", "large", "huge"),
            ((1, 256, 64, 64),),
            256,
        ),
        (
            "sam2",
            ("tiny", "small", "base-plus", "large"),
            (
                (1, 32, 256, 256),
                (1, 64, 128, 128),
                (1, 256, 64, 64),
            ),
            256,
        ),
        (
            "sam3",
            ("large",),
            (
                (1, 32, 288, 288),
                (1, 64, 144, 144),
                (1, 256, 72, 72),
            ),
            288,
        ),
    ],
)
def test_profiles_pin_every_audited_visual_family(
    family,
    sizes,
    embedding_shapes,
    mask_size,
):
    for size in sizes:
        profile = _profile(family, size)
        assert profile.embedding_shapes == embedding_shapes
        assert profile.low_res_mask_size == mask_size
        assert profile.precision == "fp32"
        assert profile.prompt_max_points == 16


def test_profile_rejects_unbounded_static_lossy_and_nonvisual_variants():
    from libreyolo.export.coreml_sam import validate_sam_coreml_profile

    with pytest.raises(NotImplementedError, match="EdgeTAM"):
        validate_sam_coreml_profile(
            family="picosam3",
            size="pico",
            prompt_max_points=16,
        )
    with pytest.raises(NotImplementedError, match="does not support size"):
        validate_sam_coreml_profile(
            family="sam2",
            size="medium",
            prompt_max_points=16,
        )
    with pytest.raises(NotImplementedError, match="FP32-only"):
        validate_sam_coreml_profile(
            family="edgetam",
            size="edge",
            prompt_max_points=16,
            precision="fp16",
        )
    with pytest.raises(ValueError, match="at least 2"):
        validate_sam_coreml_profile(
            family="edgetam",
            size="edge",
            prompt_max_points=1,
        )
    with pytest.raises(ValueError, match="at most 64"):
        validate_sam_coreml_profile(
            family="edgetam",
            size="edge",
            prompt_max_points=65,
        )
    with pytest.raises(ValueError, match="integer"):
        validate_sam_coreml_profile(
            family="edgetam",
            size="edge",
            prompt_max_points=True,
        )


def test_edgetam_function_manifest_has_seven_distinct_exact_abis():
    from libreyolo.export.coreml_sam import (
        SAM_COREML_FUNCTION_NAMES,
        sam_coreml_function_contracts,
    )

    profile = _profile()
    functions = sam_coreml_function_contracts(profile)
    assert tuple(functions) == SAM_COREML_FUNCTION_NAMES
    assert functions["encode_image"]["inputs"][0] == {
        "name": "pixel_values",
        "kind": "tensor",
        "dtype": "float32",
        "layout": "NCHW",
        "color": "rgb",
        "range": "family_native_standardized",
        "shape": [
            {"axis": "N", "kind": "fixed", "value": 1},
            {"axis": "C", "kind": "fixed", "value": 3},
            {"axis": "H", "kind": "fixed", "value": 1024},
            {"axis": "W", "kind": "fixed", "value": 1024},
        ],
        "preprocess_owner": "host",
        "preprocess_contract": "edgetam_square_imagenet_v1",
    }
    assert [
        item["name"] for item in functions["encode_image"]["outputs"]
    ] == [
        "image_embedding_s4",
        "image_embedding_s8",
        "image_embedding_s16",
    ]

    points = functions["decode_points_single"]
    assert [item["name"] for item in points["inputs"]] == [
        "image_embedding_s4",
        "image_embedding_s8",
        "image_embedding_s16",
        "point_coords",
        "point_labels",
    ]
    point_axis = points["inputs"][-2]["shape"][2]
    assert point_axis == {
        "axis": "P",
        "kind": "range",
        "lower_bound": 1,
        "upper_bound": 16,
        "default": 1,
        "padding": "forbidden",
    }
    assert points["inputs"][-1]["dtype"] == "int32"
    assert points["outputs"][0]["shape"][2]["value"] == 1

    boxes = functions["decode_boxes_multimask"]
    assert [item["name"] for item in boxes["inputs"]][-1] == "boxes"
    assert boxes["outputs"][0]["shape"][2]["value"] == 3

    combined = functions["decode_points_boxes_multimask"]
    assert [item["name"] for item in combined["inputs"]][-3:] == [
        "point_coords",
        "point_labels",
        "boxes",
    ]
    assert combined["capture"] == "torch_export_dynamic_points"

    from libreyolo.export.coreml_sam import sam_coreml_metadata

    assert sam_coreml_metadata(profile)["native_outputs_omitted"] == [
        "object_score_logits"
    ]


def test_sam3_profile_keeps_288_grid_and_local_only_license():
    from libreyolo.export.coreml_sam import (
        sam_coreml_function_contracts,
        sam_coreml_metadata,
    )

    profile = _profile("sam3")
    functions = sam_coreml_function_contracts(profile)
    output = functions["decode_points_multimask"]["outputs"][0]
    assert [axis["value"] for axis in output["shape"] if axis["kind"] == "fixed"] == [
        1,
        1,
        3,
        288,
        288,
    ]

    metadata = sam_coreml_metadata(profile)
    assert metadata["artifact_redistributable"] is False
    assert metadata["sam3_visual_only"] is True
    assert metadata["sam3_pcs_included"] is False
    assert "custom" in metadata["weights_license"]


def test_profiles_serialize_operation_order_sensitive_host_math():
    profiles = {
        family: _profile(family).as_dict()["host_contract"]
        for family in ("edgetam", "mobilesam", "sam", "sam2", "sam3")
    }
    assert "float01" in profiles["edgetam"]["image_resize"]
    assert "uint8" in profiles["sam2"]["image_resize"]
    assert profiles["edgetam"]["image_resize"] != profiles["sam2"]["image_resize"]
    assert "normalized_zero" in profiles["mobilesam"]["padding"]
    assert "float64_div255_then_fp32" in profiles["sam"]["normalization"]
    assert "low256_to_1024_crop" in profiles["sam"]["mask_postprocess"]
    assert "low288_to_original" in profiles["sam3"]["mask_postprocess"]


@pytest.mark.parametrize(
    ("family", "expected_gap"),
    [
        ("edgetam", None),
        ("sam", None),
        ("sam2", None),
        ("mobilesam", "MobileSAM"),
    ],
)
def test_release_notice_gates_are_explicit(family, expected_gap):
    from libreyolo.export.coreml_sam import sam_coreml_metadata

    gap = sam_coreml_metadata(_profile(family))["release_notice_gap"]
    if expected_gap is None:
        assert gap is None
    else:
        assert expected_gap in gap


def test_metadata_round_trip_is_hash_checked_and_string_safe():
    from libreyolo.export.coreml_sam import (
        sam_coreml_metadata,
        validate_sam_coreml_metadata,
    )

    metadata = sam_coreml_metadata(_profile("sam2", "base-plus"))
    assert validate_sam_coreml_metadata(metadata) == metadata
    assert validate_sam_coreml_metadata(_stringify_metadata(metadata)) == metadata

    tampered = copy.deepcopy(metadata)
    tampered["sam_coreml_functions"]["decode_points_single"]["inputs"][-1][
        "dtype"
    ] = "int64"
    with pytest.raises(ValueError, match="exact graph ABI"):
        validate_sam_coreml_metadata(tampered)

    bad_hash = copy.deepcopy(metadata)
    bad_hash["sam_coreml_functions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        validate_sam_coreml_metadata(bad_hash)

    wrong_license = copy.deepcopy(metadata)
    wrong_license["artifact_redistributable"] = False
    with pytest.raises(ValueError, match="license contract"):
        validate_sam_coreml_metadata(wrong_license)


def _valid_decoder_io(profile, *, point_count: int = 4):
    embeddings = {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in zip(profile.embedding_names, profile.embedding_shapes)
    }
    inputs = {
        **embeddings,
        "point_coords": torch.full(
            (1, 1, point_count, 2),
            profile.image_size / 2,
            dtype=torch.float32,
        ),
        "point_labels": torch.ones(
            (1, 1, point_count),
            dtype=torch.int32,
        ),
        "boxes": torch.tensor(
            [[[1.0, 2.0, 100.0, 200.0]]],
            dtype=torch.float32,
        ),
    }
    size = profile.low_res_mask_size
    outputs = {
        "low_res_masks": torch.zeros(
            (1, 1, 3, size, size),
            dtype=torch.float32,
        ),
        "iou_scores": torch.full(
            (1, 1, 3),
            0.5,
            dtype=torch.float32,
        ),
    }
    return inputs, outputs


def test_named_runtime_io_validator_enforces_dynamic_p_and_semantics():
    from libreyolo.export.coreml_sam import validate_sam_coreml_function_io

    profile = _profile(prompt_max_points=16)
    inputs, outputs = _valid_decoder_io(profile)
    validate_sam_coreml_function_io(
        "decode_points_boxes_multimask",
        inputs,
        outputs,
        profile=profile,
    )

    bad_dtype = {**inputs, "point_labels": inputs["point_labels"].long()}
    with pytest.raises(ValueError, match="torch.int32"):
        validate_sam_coreml_function_io(
            "decode_points_boxes_multimask",
            bad_dtype,
            outputs,
            profile=profile,
        )

    padded, padded_outputs = _valid_decoder_io(profile, point_count=17)
    with pytest.raises(ValueError, match="sentinel padding"):
        validate_sam_coreml_function_io(
            "decode_points_boxes_multimask",
            padded,
            padded_outputs,
            profile=profile,
        )

    sentinel = {
        **inputs,
        "point_labels": torch.tensor([[[1, 0, -1, 1]]], dtype=torch.int32),
    }
    with pytest.raises(ValueError, match="only 0 or 1"):
        validate_sam_coreml_function_io(
            "decode_points_boxes_multimask",
            sentinel,
            outputs,
            profile=profile,
        )

    reversed_box = {
        **inputs,
        "boxes": torch.tensor([[[10.0, 10.0, 1.0, 2.0]]]),
    }
    with pytest.raises(ValueError, match="ordered"):
        validate_sam_coreml_function_io(
            "decode_points_boxes_multimask",
            reversed_box,
            outputs,
            profile=profile,
        )

    wrong_order = dict(inputs)
    point_labels = wrong_order.pop("point_labels")
    wrong_order["point_labels"] = point_labels
    with pytest.raises(ValueError, match="names/order"):
        validate_sam_coreml_function_io(
            "decode_points_boxes_multimask",
            wrong_order,
            outputs,
            profile=profile,
        )


def test_encoder_io_and_sam3_probability_gate_are_strict():
    from libreyolo.export.coreml_sam import validate_sam_coreml_function_io

    profile = _profile("sam3")
    image = torch.zeros((1, 3, 1008, 1008), dtype=torch.float32)
    embeddings = {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in zip(profile.embedding_names, profile.embedding_shapes)
    }
    validate_sam_coreml_function_io(
        "encode_image",
        {"pixel_values": image},
        embeddings,
        profile=profile,
    )

    inputs, outputs = _valid_decoder_io(profile)
    outputs["iou_scores"].fill_(1.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_sam_coreml_function_io(
            "decode_points_boxes_multimask",
            inputs,
            outputs,
            profile=profile,
        )


def test_converted_graph_signature_must_preserve_range_descriptor():
    from libreyolo.export.coreml_sam import (
        sam_coreml_function_contracts,
        validate_sam_coreml_graph_signature,
    )

    profile = _profile()
    contract = sam_coreml_function_contracts(profile)["decode_points_single"]

    def normalized(items):
        return [
            {
                "name": item["name"],
                "dtype": item["dtype"],
                "shape": copy.deepcopy(item["shape"]),
            }
            for item in items
        ]

    input_specs = normalized(contract["inputs"])
    output_specs = normalized(contract["outputs"])
    validate_sam_coreml_graph_signature(
        "decode_points_single",
        input_specs=input_specs,
        output_specs=output_specs,
        profile=profile,
    )

    input_specs[-2]["shape"][2] = {
        "axis": "P",
        "kind": "fixed",
        "value": 1,
    }
    with pytest.raises(ValueError, match="strict manifest"):
        validate_sam_coreml_graph_signature(
            "decode_points_single",
            input_specs=input_specs,
            output_specs=output_specs,
            profile=profile,
        )


class _ToyMaskDecoder(nn.Module):
    num_multimask_outputs = 3

    def forward(
        self,
        *,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
    ):
        del image_pe, dense_prompt_embeddings
        mask_count = 3 if multimask_output else 1
        signal = image_embeddings.mean() + sparse_prompt_embeddings.mean(dim=(1, 2))
        masks = signal.reshape(-1, 1, 1, 1).expand(-1, mask_count, 4, 4)
        scores = signal.sigmoid().reshape(-1, 1).expand(-1, mask_count)
        return masks, scores


class _ToyMobileSAM(nn.Module):
    def __init__(self):
        super().__init__()
        from libreyolo.models.mobilesam.prompt_encoder import PromptEncoder

        self.image_encoder = nn.Identity()
        self.prompt_encoder = PromptEncoder(
            embed_dim=8,
            image_embedding_size=(2, 2),
            input_image_size=(32, 32),
            mask_in_chans=4,
        )
        self.mask_decoder = _ToyMaskDecoder()


@pytest.mark.parametrize(
    ("prompt_mode", "point_count"),
    [("points", 1), ("points", 5), ("boxes", 0), ("points_boxes", 3)],
)
def test_mobilesam_functional_prompt_rewrite_is_eager_exact(
    prompt_mode,
    point_count,
):
    from libreyolo.export.coreml_sam import SAMCoreMLDecoder

    torch.manual_seed(20260729)
    model = _ToyMobileSAM().eval()
    decoder = SAMCoreMLDecoder(
        model,
        _profile("mobilesam"),
        prompt_mode=prompt_mode,
        mask_mode="multimask",
    ).eval()
    points = labels = boxes = None
    if prompt_mode in ("points", "points_boxes"):
        points = torch.linspace(
            1.0,
            24.0,
            point_count * 2,
        ).reshape(1, 1, point_count, 2)
        labels = (torch.arange(point_count) % 2).reshape(1, 1, -1).to(torch.int32)
    if prompt_mode in ("boxes", "points_boxes"):
        boxes = torch.tensor([[[2.0, 3.0, 20.0, 24.0]]])

    actual = decoder._mobile_prompt_embeddings(points, labels, boxes)
    expected = model.prompt_encoder(
        points=None if points is None else (points[0], labels[0]),
        boxes=None if boxes is None else boxes[0],
        masks=None,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)


def test_mobilesam_dynamic_point_graph_exports_without_index_put():
    from libreyolo.export.coreml_sam import (
        SAMCoreMLDecoder,
        sam_coreml_decoder_dynamic_shapes,
    )

    torch.manual_seed(20260729)
    profile = _profile("mobilesam", prompt_max_points=16)
    model = _ToyMobileSAM().eval()
    decoder = SAMCoreMLDecoder(
        model,
        profile,
        prompt_mode="points",
        mask_mode="single",
    ).eval()
    embedding = torch.randn(1, 8, 2, 2)
    # A symbolic dimension cannot be discovered from a size-1 example because
    # PyTorch specializes 0/1 dimensions.  Capture at P=2 while retaining the
    # declared runtime lower bound P=1.
    point = torch.tensor([[[[8.0, 9.0], [12.0, 14.0]]]])
    label = torch.tensor([[[1, 0]]], dtype=torch.int32)
    many_points = torch.linspace(2.0, 28.0, 10).reshape(1, 1, 5, 2)
    many_labels = torch.tensor([[[1, 0, 1, 0, 1]]], dtype=torch.int32)

    exported = torch.export.export(
        decoder,
        (embedding, point, label),
        dynamic_shapes=sam_coreml_decoder_dynamic_shapes(
            profile,
            "decode_points_single",
        ),
        strict=False,
    ).run_decompositions({})
    with torch.no_grad():
        expected = decoder(embedding, many_points, many_labels)
        actual = exported.module()(embedding, many_points, many_labels)
        expected_one = decoder(embedding, point[:, :, :1], label[:, :, :1])
        actual_one = exported.module()(
            embedding,
            point[:, :, :1],
            label[:, :, :1],
        )
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_one[0], expected_one[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_one[1], expected_one[1], rtol=0.0, atol=0.0)
    assert "index_put" not in str(exported.graph).lower()


@pytest.mark.parametrize("mask_mode", ["single", "multimask"])
def test_real_mobilesam_decoder_adapter_matches_native_exactly(mask_mode):
    from libreyolo.export.coreml_sam import SAMCoreMLDecoder
    from libreyolo.models.mobilesam.model import MobileSAMNetwork

    torch.manual_seed(20260729)
    model = MobileSAMNetwork().eval()
    decoder = SAMCoreMLDecoder(
        model,
        _profile("mobilesam"),
        prompt_mode="points_boxes",
        mask_mode=mask_mode,
    ).eval()
    embedding = torch.randn(1, 256, 64, 64)
    points = torch.tensor([[[[100.0, 200.0], [300.0, 400.0]]]])
    labels = torch.tensor([[[1, 0]]], dtype=torch.int32)
    boxes = torch.tensor([[[50.0, 60.0, 700.0, 800.0]]])

    with torch.inference_mode():
        actual_masks, actual_scores = decoder(embedding, points, labels, boxes)
        native = model(
            image_embeddings=embedding,
            input_points=points,
            input_labels=labels,
            input_boxes=boxes,
            multimask_output=mask_mode == "multimask",
        )
    torch.testing.assert_close(
        actual_masks,
        native.pred_masks,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual_scores,
        native.iou_scores,
        rtol=0.0,
        atol=0.0,
    )


def test_tiny_transformers_sam1_adapter_delegates_with_exact_parity():
    pytest.importorskip("transformers", reason="SAM-1 adapter requires transformers")
    from transformers import SamConfig, SamModel
    from transformers.models.sam.configuration_sam import (
        SamMaskDecoderConfig,
        SamPromptEncoderConfig,
        SamVisionConfig,
    )

    from libreyolo.export.coreml_sam import SAMCoreMLDecoder

    vision = SamVisionConfig(
        hidden_size=16,
        output_channels=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        mlp_dim=32,
        image_size=32,
        patch_size=4,
        global_attn_indexes=[1],
        window_size=4,
        num_pos_feats=4,
    )
    prompt = SamPromptEncoderConfig(
        hidden_size=8,
        image_size=32,
        image_embedding_size=8,
        patch_size=4,
        mask_input_channels=4,
    )
    mask = SamMaskDecoderConfig(
        hidden_size=8,
        mlp_dim=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        attention_downsample_rate=1,
        iou_head_hidden_dim=8,
    )
    model = SamModel(
        SamConfig(
            vision_config=vision,
            prompt_encoder_config=prompt,
            mask_decoder_config=mask,
        )
    ).eval()
    decoder = SAMCoreMLDecoder(
        model,
        _profile("sam"),
        prompt_mode="points_boxes",
        mask_mode="multimask",
    ).eval()
    embedding = torch.randn(1, 8, 8, 8)
    points = torch.tensor([[[[5.0, 6.0], [10.0, 12.0]]]])
    labels = torch.tensor([[[1, 0]]], dtype=torch.int32)
    boxes = torch.tensor([[[2.0, 3.0, 20.0, 24.0]]])

    with torch.inference_mode():
        actual = decoder(embedding, points, labels, boxes)
        native = model(
            image_embeddings=embedding,
            input_points=points,
            input_labels=labels,
            input_boxes=boxes,
            multimask_output=True,
        )
    torch.testing.assert_close(actual[0], native.pred_masks, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], native.iou_scores, rtol=0.0, atol=0.0)


class _FakeHieraBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.scale = nn.Parameter(torch.tensor(2.0))

    def _get_pos_embed(self, hw):
        self.calls += 1
        height, width = hw
        return self.scale.detach() * torch.ones((1, height, width, 2))


def test_sam2_position_rewrite_freezes_each_loaded_models_native_tensor():
    from libreyolo.export.coreml_sam import (
        freeze_sam2_coreml_position_embedding,
    )

    graph = nn.Module()
    graph.vision_encoder = nn.Module()
    graph.vision_encoder.backbone = _FakeHieraBackbone()
    profile = _profile("sam2", "large")

    freeze_sam2_coreml_position_embedding(graph, profile=profile)
    backbone = graph.vision_encoder.backbone
    assert backbone.calls == 1
    first = backbone._get_pos_embed((256, 256))
    second = backbone._get_pos_embed((8, 8))
    assert backbone.calls == 1
    assert tuple(first.shape) == (1, 256, 256, 2)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert float(first[0, 0, 0, 0]) == 2.0


class _SignatureMaskDecoder(nn.Module):
    num_multimask_outputs = 3


class _SignaturePromptEncoder(nn.Module):
    pass


class _SignatureSAM2(nn.Module):
    def __init__(self, *, model_type="sam2", feature_sizes=None):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.prompt_encoder = _SignaturePromptEncoder()
        self.mask_decoder = _SignatureMaskDecoder()
        self.backbone_feature_sizes = feature_sizes or [
            [256, 256],
            [128, 128],
            [64, 64],
        ]

    def get_image_embeddings(self, pixel_values):
        return pixel_values


def test_model_signature_rejects_wrong_family_and_feature_pyramid():
    from libreyolo.export.coreml_sam import inspect_sam_coreml_model

    profile = _profile("sam2")
    signature = inspect_sam_coreml_model(_SignatureSAM2(), profile=profile)
    assert signature.family == "sam2"
    assert signature.num_multimask_outputs == 3

    with pytest.raises(ValueError, match="model_type"):
        inspect_sam_coreml_model(
            _SignatureSAM2(model_type="sam3_tracker"),
            profile=profile,
        )
    with pytest.raises(ValueError, match="feature sizes"):
        inspect_sam_coreml_model(
            _SignatureSAM2(feature_sizes=[[1, 1], [2, 2], [3, 3]]),
            profile=profile,
        )
