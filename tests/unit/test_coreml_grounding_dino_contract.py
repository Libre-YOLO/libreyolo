from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from libreyolo.export.coreml_grounding_dino import (
    GROUNDING_DINO_COREML_CONTRACT,
    GROUNDING_DINO_COREML_MEAN,
    GROUNDING_DINO_COREML_POSTPROCESS,
    GROUNDING_DINO_COREML_PREPROCESS,
    GROUNDING_DINO_COREML_STD,
    GroundingDinoFrozenCoreMLAdapter,
    build_native_grounding_dino_from_hf,
    expected_grounding_dino_coreml_shapes,
    freeze_grounding_dino_text,
    frozen_grounding_dino_text_from_metadata,
    grounding_dino_coreml_input_contract,
    grounding_dino_coreml_metadata,
    grounding_dino_coreml_output_contract,
    grounding_dino_coreml_prompt,
    grounding_dino_coreml_validation_contract,
    grounding_dino_coreml_vocabulary_hash,
    postprocess_grounding_dino_coreml_outputs,
    preprocess_grounding_dino_coreml_image,
    validate_grounding_dino_coreml_metadata,
    validate_grounding_dino_coreml_outputs,
    validate_grounding_dino_coreml_profile,
)
from libreyolo.models.grounding_dino.nn import GDMSDeformAttn

pytestmark = pytest.mark.unit


class _TinyBertTokenizer:
    input_ids = torch.tensor(
        [[101, 1037, 4937, 1012, 1037, 3899, 1012, 102]],
        dtype=torch.long,
    )
    pieces = ("[CLS]", "a", "cat", ".", "a", "dog", ".", "[SEP]")

    def __call__(self, text, **kwargs):
        assert text == "a cat. a dog."
        assert kwargs == {
            "add_special_tokens": True,
            "truncation": False,
            "return_tensors": "pt",
        }
        return {
            "input_ids": self.input_ids.clone(),
            "token_type_ids": torch.zeros_like(self.input_ids),
            "attention_mask": torch.ones_like(self.input_ids),
        }

    def convert_ids_to_tokens(self, input_ids):
        assert input_ids == self.input_ids[0].tolist()
        return list(self.pieces)


@pytest.fixture(scope="module")
def tiny_grounding_dino():
    transformers = pytest.importorskip("transformers")
    bert_config = transformers.BertConfig(
        vocab_size=30522,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    backbone_config = transformers.SwinConfig(
        image_size=128,
        patch_size=4,
        embed_dim=32,
        depths=[1, 1, 1, 1],
        num_heads=[1, 2, 4, 8],
        window_size=4,
        out_features=["stage2", "stage3", "stage4"],
        drop_path_rate=0.0,
    )
    config = transformers.GroundingDinoConfig(
        backbone_config=backbone_config,
        text_config=bert_config,
        d_model=32,
        num_queries=4,
        encoder_layers=1,
        decoder_layers=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        max_text_len=32,
        num_feature_levels=4,
        dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        text_enhancer_dropout=0.0,
        fusion_dropout=0.0,
        fusion_droppath=0.0,
    )
    torch.manual_seed(17)
    hf_model = transformers.GroundingDinoForObjectDetection(config).eval()
    native = build_native_grounding_dino_from_hf(hf_model)
    processor = SimpleNamespace(tokenizer=_TinyBertTokenizer())
    frozen = freeze_grounding_dino_text(
        hf_model,
        processor,
        {0: "cat", 1: "dog"},
    )
    adapter = GroundingDinoFrozenCoreMLAdapter(
        native,
        frozen,
        canvas_hw=(128, 128),
    ).eval()
    return SimpleNamespace(
        hf_model=hf_model,
        native=native,
        frozen=frozen,
        adapter=adapter,
    )


def _gradient(width: int, height: int) -> Image.Image:
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    red = (x + 3 * y) % 256
    green = (2 * x + y) % 256
    blue = (x + 5 * y) % 256
    return Image.fromarray(
        np.stack([red, green, blue], axis=-1).astype(np.uint8),
        mode="RGB",
    )


def test_profile_and_raw_abi_are_strict():
    tiny = validate_grounding_dino_coreml_profile(
        size="t",
        canvas_hw=(800, 800),
    )
    base = validate_grounding_dino_coreml_profile(size="b")
    assert tiny.backbone_depths == (2, 2, 6, 2)
    assert base.backbone_depths == (2, 2, 18, 2)
    assert expected_grounding_dino_coreml_shapes(
        size="t",
        sequence_length=8,
    ) == {
        "token_logits": (1, 900, 8),
        "pred_boxes": (1, 900, 4),
    }
    assert grounding_dino_coreml_input_contract() == {
        "name": "image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "stretch",
        "interpolation": "bilinear",
        "resize_backend": "torchvision",
        "antialias": True,
        "pad_value": 0,
    }
    assert grounding_dino_coreml_validation_contract() == {
        "color": "rgb",
        "range": "0_255",
    }
    assert grounding_dino_coreml_output_contract() == [
        {
            "name": "token_logits",
            "role": "text_token_logits",
            "encoding": "raw_logits_compact_frozen_sequence",
            "rank": 3,
        },
        {
            "name": "pred_boxes",
            "role": "boxes",
            "encoding": "cxcywh_normalized_stretched_canvas",
            "rank": 3,
        },
    ]


@pytest.mark.parametrize(
    ("size", "canvas", "error"),
    [
        ("s", None, "only the t and b"),
        ("t", (800, 1333), "fixed square stretch"),
        ("b", (799, 799), "fixed square stretch"),
    ],
)
def test_profile_rejects_unknown_or_flexible_shapes(size, canvas, error):
    with pytest.raises(NotImplementedError, match=error):
        validate_grounding_dino_coreml_profile(
            size=size,
            canvas_hw=canvas,
        )


def test_frozen_vocabulary_prompt_is_ordered_hashed_and_strict():
    names = {1: "Remote Control", 0: "Cat"}
    assert grounding_dino_coreml_prompt(names) == ("a cat. a remote control.")
    assert grounding_dino_coreml_vocabulary_hash(names) == (
        grounding_dino_coreml_vocabulary_hash(["Cat", "Remote Control"])
    )
    assert grounding_dino_coreml_vocabulary_hash(names) != (
        grounding_dino_coreml_vocabulary_hash(["Remote Control", "Cat"])
    )

    invalid = [
        ({}, "at least one frozen class"),
        ({1: "cat"}, "contiguous from zero"),
        ({0: ""}, "must not be blank"),
        ({0: "the"}, "contain letters or digits"),
        ({0: "Cat", 1: "a cat"}, "unique after"),
        ({0: "cat. dog"}, "must not contain"),
        ({0: "which cat?"}, "must not contain"),
    ]
    for classes, error in invalid:
        with pytest.raises((TypeError, ValueError), match=error):
            grounding_dino_coreml_prompt(classes)


def test_fixed_stretch_preprocessing_matches_pinned_transformers():
    transformers = pytest.importorskip("transformers")
    image = _gradient(119, 73)
    canonical = preprocess_grounding_dino_coreml_image(
        image,
        canvas_hw=(128, 128),
    )
    mean = torch.tensor(GROUNDING_DINO_COREML_MEAN).view(1, 3, 1, 1).mul(255.0)
    std = torch.tensor(GROUNDING_DINO_COREML_STD).view(1, 3, 1, 1).mul(255.0)
    prepared = (canonical * 255.0 - mean) / std
    reference = transformers.GroundingDinoImageProcessor()(
        images=image,
        size={"height": 128, "width": 128},
        do_pad=False,
        return_tensors="pt",
    )["pixel_values"]
    assert torch.equal(prepared, reference)


@pytest.mark.parametrize("coordinate_width", [2, 4])
def test_rank_five_deformable_attention_is_bit_exact(coordinate_width):
    config = SimpleNamespace(d_model=32, num_feature_levels=4)
    torch.manual_seed(11)
    module = GDMSDeformAttn(config, num_heads=4, n_points=4).eval()
    shapes = [(4, 4), (2, 2), (1, 1), (1, 1)]
    spatial_shapes = torch.tensor(shapes, dtype=torch.long)
    hidden = torch.randn(1, 10, 32)
    encoded = torch.randn(1, 22, 32)
    attention_mask = torch.ones(1, 22, dtype=torch.bool)
    reference_points = torch.rand(1, 10, 4, coordinate_width)

    def original_rank_six_forward():
        batch, queries, _ = hidden.shape
        _, sequence, _ = encoded.shape
        value = module.value_proj(encoded)
        value = value.masked_fill(~attention_mask[..., None], 0.0)
        value = value.view(batch, sequence, 4, 8)
        offsets = module.sampling_offsets(hidden).view(
            batch,
            queries,
            4,
            4,
            4,
            2,
        )
        weights = module.attention_weights(hidden).view(
            batch,
            queries,
            4,
            16,
        )
        weights = F.softmax(weights, dim=-1).view(
            batch,
            queries,
            4,
            4,
            4,
        )
        if coordinate_width == 2:
            normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]],
                dim=-1,
            )
            locations = (
                reference_points[:, :, None, :, None, :]
                + offsets / normalizer[None, None, None, :, None, :]
            )
        else:
            locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets / 4 * reference_points[:, :, None, :, None, 2:] * 0.5
            )

        values = value.split(
            [height * width for height, width in shapes],
            dim=1,
        )
        grids = 2 * locations - 1
        sampled = []
        for level, (height, width) in enumerate(shapes):
            level_value = (
                values[level]
                .flatten(2)
                .transpose(1, 2)
                .reshape(batch * 4, 8, height, width)
            )
            grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
            sampled.append(
                F.grid_sample(
                    level_value,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            )
        weights = weights.transpose(1, 2).reshape(
            batch * 4,
            1,
            queries,
            16,
        )
        output = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(dim=-1)
        output = output.view(batch, 32, queries).transpose(1, 2)
        return module.output_proj(output.contiguous())

    with torch.inference_mode():
        reference = original_rank_six_forward()
        actual = module(
            hidden,
            encoded,
            attention_mask,
            reference_points,
            spatial_shapes,
            shapes,
        )
    assert torch.equal(actual, reference)


def test_frozen_pre_fusion_text_graph_matches_source_and_is_traceable(
    tiny_grounding_dino,
):
    bundle = tiny_grounding_dino
    torch.manual_seed(23)
    image = torch.rand(1, 3, 128, 128)
    mean = torch.tensor(GROUNDING_DINO_COREML_MEAN).view(1, 3, 1, 1).mul(255.0)
    std = torch.tensor(GROUNDING_DINO_COREML_STD).view(1, 3, 1, 1).mul(255.0)
    normalized = (image * 255.0 - mean) / std
    frozen = bundle.frozen
    with torch.inference_mode():
        reference = bundle.hf_model(
            pixel_values=normalized,
            pixel_mask=torch.ones(
                normalized.shape[0],
                normalized.shape[2],
                normalized.shape[3],
                dtype=torch.long,
            ),
            input_ids=frozen.input_ids,
            token_type_ids=frozen.token_type_ids,
            attention_mask=frozen.attention_mask,
        )
        actual = bundle.adapter(image)
    torch.testing.assert_close(
        reference["logits"][..., : frozen.sequence_length],
        actual[0],
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        reference["pred_boxes"],
        actual[1],
        rtol=1e-4,
        atol=1e-5,
    )

    traced = torch.jit.trace(
        bundle.adapter,
        image,
        check_trace=True,
        check_inputs=[(1.0 - image,)],
    )
    with torch.inference_mode():
        traced_outputs = traced(image)
    assert torch.equal(actual[0], traced_outputs[0])
    assert torch.equal(actual[1], traced_outputs[1])
    assert all(
        "text_backbone" not in name for name, _ in bundle.adapter.named_modules()
    )
    assert all("text_backbone" not in key for key in bundle.adapter.state_dict())
    assert bundle.frozen.text_features.shape == (1, 8, 32)
    assert bundle.frozen.prompt == "a cat. a dog."


def test_metadata_is_self_contained_and_consistency_checked(
    tiny_grounding_dino,
):
    frozen = tiny_grounding_dino.frozen
    names = {0: "cat", 1: "dog"}
    metadata = grounding_dino_coreml_metadata(
        size="t",
        names=names,
        frozen=frozen,
    )
    assert metadata["frozen_classes"] is True
    assert metadata["grounding_dino_contract"] == (GROUNDING_DINO_COREML_CONTRACT)
    assert metadata["grounding_dino_preprocess"] == (GROUNDING_DINO_COREML_PREPROCESS)
    assert metadata["grounding_dino_postprocess"] == (GROUNDING_DINO_COREML_POSTPROCESS)
    parsed = validate_grounding_dino_coreml_metadata(
        metadata,
        size="t",
        names=names,
    )
    assert parsed == frozen_grounding_dino_text_from_metadata(
        metadata,
        names=names,
    )
    assert parsed["token_pieces"] == list(_TinyBertTokenizer.pieces)
    assert parsed["input_ids"] == _TinyBertTokenizer.input_ids[0].tolist()

    changed = copy.deepcopy(metadata)
    changed["grounding_dino_token_pieces_json"] = (
        '["[CLS]","a","fox",".","a","dog",".","[SEP]"]'
    )
    with pytest.raises(ValueError, match="ABI hash"):
        validate_grounding_dino_coreml_metadata(
            changed,
            size="t",
            names=names,
        )

    changed = copy.deepcopy(metadata)
    changed["grounding_dino_canvas_width"] = "1333"
    with pytest.raises(ValueError, match="canvas_width"):
        validate_grounding_dino_coreml_metadata(
            changed,
            size="t",
            names=names,
        )


def test_host_postprocess_matches_grounded_phrase_and_class_rules():
    logits = torch.full((1, 900, 8), -10.0)
    boxes = torch.zeros((1, 900, 4), dtype=torch.float32)
    boxes[..., :2] = 0.5
    boxes[..., 2:] = 0.2
    # cat, dog, ambiguous "cat dog", and article-only (unmappable).
    for query, positions in [
        (0, [1, 2]),
        (1, [4, 5]),
        (2, [1, 2, 4, 5]),
        (3, [1]),
    ]:
        logits[0, query, positions] = 3.0
    text_contract = {
        "input_ids": _TinyBertTokenizer.input_ids[0].tolist(),
        "token_pieces": list(_TinyBertTokenizer.pieces),
    }
    result = postprocess_grounding_dino_coreml_outputs(
        logits,
        boxes,
        size="t",
        names={0: "cat", 1: "dog"},
        text_contract=text_contract,
        original_size=(100, 50),
        conf=0.25,
        text_threshold=0.25,
    )
    assert result["num_detections"] == 2
    assert result["classes"].tolist() == [0, 1]
    assert torch.allclose(
        result["boxes"],
        torch.tensor(
            [
                [40.0, 20.0, 60.0, 30.0],
                [40.0, 20.0, 60.0, 30.0],
            ]
        ),
    )
    assert torch.allclose(
        result["scores"],
        torch.full((2,), torch.sigmoid(torch.tensor(3.0))),
    )

    dog_only = postprocess_grounding_dino_coreml_outputs(
        logits,
        boxes,
        size="t",
        names={0: "cat", 1: "dog"},
        text_contract=text_contract,
        original_size=(100, 50),
        classes=[1],
    )
    assert dog_only["classes"].tolist() == [1]


def test_runtime_output_validation_rejects_bad_shapes_and_values():
    logits = torch.full((1, 900, 8), -10.0)
    boxes = torch.full((1, 900, 4), 0.5)
    validate_grounding_dino_coreml_outputs(
        logits,
        boxes,
        size="t",
        sequence_length=8,
    )

    with pytest.raises(ValueError, match="token_logits shape"):
        validate_grounding_dino_coreml_outputs(
            logits[:, :899],
            boxes,
            size="t",
            sequence_length=8,
        )
    changed = logits.clone()
    changed[0, 0, 1] = math.nan
    with pytest.raises(ValueError, match="token logits"):
        validate_grounding_dino_coreml_outputs(
            changed,
            boxes,
            size="t",
            sequence_length=8,
        )
    changed = logits.clone()
    changed[0, 0, 7] = math.inf
    with pytest.raises(ValueError, match="token logits"):
        validate_grounding_dino_coreml_outputs(
            changed,
            boxes,
            size="t",
            sequence_length=8,
        )
    changed_boxes = boxes.clone()
    changed_boxes[0, 0, 0] = math.inf
    with pytest.raises(ValueError, match="boxes contain"):
        validate_grounding_dino_coreml_outputs(
            logits,
            changed_boxes,
            size="t",
            sequence_length=8,
        )


def test_public_model_exposes_only_frozen_coreml_export(monkeypatch):
    from libreyolo.export import coreml_grounding_dino
    from libreyolo.models.openvocab.grounding_dino import (
        LibreGroundingDINO,
    )

    captured = {}

    def fake_export(model, kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "frozen.mlpackage"

    monkeypatch.setattr(
        coreml_grounding_dino,
        "export_grounding_dino_coreml",
        fake_export,
    )
    model = object.__new__(LibreGroundingDINO)
    assert model.export(format="COREML", half=True) == "frozen.mlpackage"
    assert captured == {"model": model, "kwargs": {"half": True}}
    with pytest.raises(NotImplementedError, match="out of scope"):
        model.export(format="onnx")
