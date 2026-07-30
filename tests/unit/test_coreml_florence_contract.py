"""Hermetic contract and source-parity tests for Florence-2 Core ML."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libreyolo.export import coreml_florence
from libreyolo.export.coreml_florence import (
    FLORENCE2_BASE_REQUIRED_ASSETS,
    FLORENCE2_BASE_REVISION,
    FLORENCE2_BASE_WEIGHTS_SHA256,
    FLORENCE2_BASE_WEIGHTS_SIZE,
    FLORENCE2_IMAGE_TOKEN_ID,
    FLORENCE2_TASK,
    FLORENCE_BEAM_PARENT_INDICES_INPUT,
    FLORENCE_CAUSAL_MASK_INPUT,
    FLORENCE_COREML_COMPONENT_CONTRACT,
    FLORENCE_COREML_TRANSFORMERS_COMMIT,
    FLORENCE_COREML_TRANSFORMERS_VERSION,
    FLORENCE_CROSS_ATTENTION_MASK_INPUT,
    FLORENCE_CROSS_KEY_CACHE_STATE,
    FLORENCE_CROSS_KEY_OUTPUT,
    FLORENCE_CROSS_VALUE_CACHE_STATE,
    FLORENCE_CROSS_VALUE_OUTPUT,
    FLORENCE_DECODE_FUNCTION,
    FLORENCE_DECODER_INPUT_IDS_INPUT,
    FLORENCE_ENCODE_FUNCTION,
    FLORENCE_ENCODER_ATTENTION_MASK_INPUT,
    FLORENCE_ENCODER_INPUT_IDS_INPUT,
    FLORENCE_LAST_LOGITS_OUTPUT,
    FLORENCE_PIXEL_VALUES_INPUT,
    FLORENCE_POSITION_IDS_INPUT,
    Florence2CoreMLEncoder,
    Florence2CoreMLStatefulDecoder,
    FlorenceCoreMLProfile,
    FlorenceDecodeCursor,
    assert_florence2_decoder_source_parity,
    assert_florence2_static_encoder_parity,
    build_florence_encoder_masks,
    build_florence_coreml_multifunction_package,
    florence2_base_coreml_metadata,
    florence2_base_coreml_profile,
    florence_coreml_function_contracts,
    prepare_florence2_base_processor_batch,
    require_florence_coreml_toolchain,
    require_florence_transformers_toolchain,
    stringify_florence_coreml_metadata,
    validate_florence2_base_model_weight_values,
    validate_florence2_base_processor_assets,
    validate_florence_coreml_metadata,
)

pytestmark = [pytest.mark.unit, pytest.mark.experimental_backend]


def test_profile_pins_full_base_state_and_function_abi():
    profile = florence2_base_coreml_profile()

    assert profile == FlorenceCoreMLProfile()
    assert profile.as_dict() == {
        "family": "florence2",
        "size": "base",
        "image_size": 768,
        "image_channels": 3,
        "image_token_count": 577,
        "encoder_context_length": 1024,
        "decoder_context_length": 1024,
        "hidden_size": 768,
        "vocab_size": 51328,
        "num_hidden_layers": 6,
        "num_attention_heads": 12,
        "head_dim": 64,
        "num_beams": 3,
        "max_new_tokens": 1024,
        "single_cross_cache_shape": [6, 1, 12, 1024, 64],
        "self_cache_shape": [6, 3, 12, 1024, 64],
        "cross_cache_shape": [6, 3, 12, 1024, 64],
        "total_state_bytes_fp16": 113246208,
    }

    contracts = florence_coreml_function_contracts(profile)
    assert list(contracts) == [FLORENCE_ENCODE_FUNCTION, FLORENCE_DECODE_FUNCTION]
    assert contracts[FLORENCE_ENCODE_FUNCTION]["stateful"] is False
    assert [
        value["name"] for value in contracts[FLORENCE_ENCODE_FUNCTION]["inputs"]
    ] == [
        FLORENCE_PIXEL_VALUES_INPUT,
        FLORENCE_ENCODER_INPUT_IDS_INPUT,
        FLORENCE_ENCODER_ATTENTION_MASK_INPUT,
    ]
    assert [
        value["name"] for value in contracts[FLORENCE_ENCODE_FUNCTION]["outputs"]
    ] == [FLORENCE_CROSS_KEY_OUTPUT, FLORENCE_CROSS_VALUE_OUTPUT]
    decode = contracts[FLORENCE_DECODE_FUNCTION]
    assert decode["stateful"] is True
    assert [value["name"] for value in decode["inputs"]] == [
        FLORENCE_DECODER_INPUT_IDS_INPUT,
        FLORENCE_CAUSAL_MASK_INPUT,
        FLORENCE_CROSS_ATTENTION_MASK_INPUT,
        FLORENCE_POSITION_IDS_INPUT,
        FLORENCE_BEAM_PARENT_INDICES_INPUT,
    ]
    assert [value["name"] for value in decode["outputs"]] == [
        FLORENCE_LAST_LOGITS_OUTPUT
    ]
    assert [value["name"] for value in decode["states"]] == [
        "self_key_cache",
        "self_value_cache",
        FLORENCE_CROSS_KEY_CACHE_STATE,
        FLORENCE_CROSS_VALUE_CACHE_STATE,
    ]
    assert decode["inputs"][1]["shape"][-1] == {
        "name": "E_decoder",
        "kind": "range",
        "lower_bound": 1,
        "upper_bound": 1024,
        "default": 1,
    }


def test_metadata_is_hash_bound_to_processor_weights_generation_and_host_ops():
    metadata = florence2_base_coreml_metadata()
    assert metadata["component_contract"] == FLORENCE_COREML_COMPONENT_CONTRACT
    assert metadata["task"] == "detect"
    assert metadata["precision"] == "mixed"
    assert metadata["encoder_compute_precision"] == "fp32"
    assert metadata["decoder_compute_precision"] == "fp16"
    assert metadata["function_io_precision"] == "fp16"
    assert metadata["runtime_state_materialization_precision"] == "fp32"
    assert metadata["coreml_execution_profile_status"] == "experimental"
    assert "coreml_execution_profile" not in metadata
    assert metadata["generation"]["num_beams"] == 3
    assert metadata["generation"]["max_new_tokens"] == 1024
    assert metadata["weights"]["revision"] == FLORENCE2_BASE_REVISION
    assert metadata["weights"]["size_bytes"] == FLORENCE2_BASE_WEIGHTS_SIZE
    assert metadata["weights"]["sha256"] == FLORENCE2_BASE_WEIGHTS_SHA256
    assert metadata["processor"]["trust_remote_code"] is False
    assert metadata["transformers_source"] == {
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
    }
    assert "cross_cache_repeat_and_state_write" in metadata["host_operations"]
    serialized = stringify_florence_coreml_metadata(metadata)
    assert validate_florence_coreml_metadata(serialized) == metadata

    modified = dict(serialized)
    modified["task"] = "caption"
    with pytest.raises(ValueError, match="task"):
        validate_florence_coreml_metadata(modified)


def test_package_builder_rejects_explicit_empty_metadata(tmp_path):
    components = {
        FLORENCE_ENCODE_FUNCTION: torch.nn.Identity(),
        FLORENCE_DECODE_FUNCTION: torch.nn.Identity(),
    }
    with pytest.raises(ValueError, match="metadata keys changed"):
        build_florence_coreml_multifunction_package(
            components,
            output_path=tmp_path / "florence.mlpackage",
            metadata={},
            compute_units="cpu_only",
        )


def test_processor_allowlist_is_exact_and_hash_validation_is_fail_closed(
    monkeypatch,
    tmp_path,
):
    assert len(FLORENCE2_BASE_REQUIRED_ASSETS) == 10
    assert set(FLORENCE2_BASE_REQUIRED_ASSETS) == {
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    for digest in FLORENCE2_BASE_REQUIRED_ASSETS.values():
        assert len(digest) == 64
        int(digest, 16)

    values = {
        "config.json": b'{"model_type":"florence2"}',
        "tokenizer.json": b'{"version":"1.0"}',
    }
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in values.items()}
    monkeypatch.setattr(
        coreml_florence,
        "FLORENCE2_BASE_REQUIRED_ASSETS",
        hashes,
    )
    for name, value in values.items():
        (tmp_path / name).write_bytes(value)

    manifest = validate_florence2_base_processor_assets(
        tmp_path,
        revision=FLORENCE2_BASE_REVISION,
    )
    assert manifest["trust_remote_code"] is False

    (tmp_path / "config.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_florence2_base_processor_assets(
            tmp_path,
            revision=FLORENCE2_BASE_REVISION,
        )
    with pytest.raises(ValueError, match="revision"):
        validate_florence2_base_processor_assets(
            tmp_path,
            revision="main",
        )


class _TinyTiedFlorenceWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        shared = torch.nn.Embedding(2, 2, dtype=torch.float32)
        with torch.no_grad():
            shared.weight.copy_(
                torch.tensor(
                    [[0.25, -1.5], [2.0, 0.125]],
                    dtype=torch.float32,
                )
            )
        self.model.language_model.shared = shared
        self.model.language_model.encoder = torch.nn.Module()
        self.model.language_model.encoder.embed_tokens = shared
        self.model.language_model.decoder = torch.nn.Module()
        self.model.language_model.decoder.embed_tokens = shared
        self.lm_head = torch.nn.Linear(2, 2, bias=False, dtype=torch.float32)
        self.lm_head.weight = shared.weight


def test_weight_values_require_exact_lossless_fp16_to_fp32_widening(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    model = _TinyTiedFlorenceWeights()
    source = {
        "model.language_model.shared.weight": (
            model.model.language_model.shared.weight.detach().to(torch.float16)
        )
    }
    safetensors.save_file(
        source,
        str(tmp_path / "model.safetensors"),
    )

    validate_florence2_base_model_weight_values(model, tmp_path)

    with torch.no_grad():
        model.model.language_model.shared.weight[0, 0] += 1.0e-4
    with pytest.raises(ValueError, match="differs from the pinned checkpoint"):
        validate_florence2_base_model_weight_values(model, tmp_path)


class _Processor:
    def __init__(self, *, invalid_prefix: bool = False):
        self.calls = []
        self.invalid_prefix = invalid_prefix

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = np.asarray(
            [[FLORENCE2_IMAGE_TOKEN_ID] * 577 + [0, 17, 2]],
            dtype=np.int64,
        )
        if self.invalid_prefix:
            input_ids[0, 0] = 17
        return {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids),
            "pixel_values": np.linspace(
                -1.0,
                1.0,
                3 * 768 * 768,
                dtype=np.float32,
            ).reshape(1, 3, 768, 768),
        }


def test_processor_batch_pads_exactly_and_builds_both_additive_masks():
    processor = _Processor()
    image = object()

    prepared = prepare_florence2_base_processor_batch(
        processor,
        image,
        ["Cat", "traffic light"],
    )

    assert processor.calls == [
        {
            "text": FLORENCE2_TASK + "Cat, traffic light",
            "images": image,
            "return_tensors": "np",
        }
    ]
    assert prepared[FLORENCE_PIXEL_VALUES_INPUT].shape == (1, 3, 768, 768)
    assert prepared[FLORENCE_PIXEL_VALUES_INPUT].dtype == np.float16
    assert prepared[FLORENCE_ENCODER_INPUT_IDS_INPUT].shape == (1, 1024)
    assert prepared[FLORENCE_ENCODER_INPUT_IDS_INPUT].dtype == np.int32
    assert prepared[FLORENCE_ENCODER_ATTENTION_MASK_INPUT].shape == (
        1,
        1,
        1,
        1024,
    )
    assert prepared[FLORENCE_CROSS_ATTENTION_MASK_INPUT].shape == (
        3,
        1,
        1,
        1024,
    )
    assert np.all(prepared[FLORENCE_ENCODER_ATTENTION_MASK_INPUT][..., :580] == 0)
    assert np.all(
        prepared[FLORENCE_ENCODER_ATTENTION_MASK_INPUT][..., 580:]
        == np.finfo(np.float16).min
    )
    assert int(prepared["prompt_length"]) == 580


def test_processor_batch_rejects_prefix_drift_and_masks_reject_holes():
    with pytest.raises(ValueError, match="577 contiguous"):
        prepare_florence2_base_processor_batch(
            _Processor(invalid_prefix=True),
            object(),
            ["cat"],
        )
    attention = np.ones((1, 1024), dtype=np.int32)
    attention[0, 700] = 0
    with pytest.raises(ValueError, match="contiguous suffix"):
        build_florence_encoder_masks(attention)


def test_decode_cursor_is_append_only_and_bounded():
    cursor = FlorenceDecodeCursor()
    first_mask, first_positions = cursor.controls()
    assert first_mask.shape == (3, 1, 1, 1)
    assert np.array_equal(first_positions, np.zeros((3, 1), dtype=np.int32))
    cursor.commit(causal_mask=first_mask, position_ids=first_positions)
    second_mask, second_positions = cursor.controls()
    assert second_mask.shape == (3, 1, 1, 2)
    assert np.array_equal(second_positions, np.ones((3, 1), dtype=np.int32))
    with pytest.raises(ValueError, match="append-only"):
        cursor.commit(
            causal_mask=np.ones_like(second_mask),
            position_ids=second_positions,
        )


def test_toolchains_are_pinned_exactly():
    require_florence_coreml_toolchain(SimpleNamespace(__version__="9.0"))
    with pytest.raises(RuntimeError, match="9.x"):
        require_florence_coreml_toolchain(SimpleNamespace(__version__="8.3"))
    with pytest.raises(RuntimeError, match="9.x"):
        require_florence_coreml_toolchain(SimpleNamespace(__version__="10.0"))
    require_florence_transformers_toolchain(
        SimpleNamespace(__version__=FLORENCE_COREML_TRANSFORMERS_VERSION)
    )
    with pytest.raises(RuntimeError, match="5.12.1"):
        require_florence_transformers_toolchain(SimpleNamespace(__version__="5.13.0"))


def test_export_default_rejects_before_toolchain_or_conversion(
    tmp_path,
    monkeypatch,
):
    touched = []
    monkeypatch.setattr(
        coreml_florence,
        "require_florence_transformers_toolchain",
        lambda: touched.append("toolchain"),
    )

    with pytest.raises(NotImplementedError, match="exact Apple-M4"):
        coreml_florence.export_florence2_base_coreml_package(
            torch.nn.Identity(),
            checkpoint_dir=tmp_path,
            processor_revision=FLORENCE2_BASE_REVISION,
            output_path=tmp_path / "blocked.mlpackage",
        )

    assert touched == []


def _tiny_florence():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != FLORENCE_COREML_TRANSFORMERS_VERSION:
        pytest.skip("tiny source parity requires pinned Transformers 5.12.1")
    text_config = transformers.BartConfig(
        vocab_size=64,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=8,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        classifier_dropout=0.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
        forced_bos_token_id=0,
        forced_eos_token_id=2,
        use_cache=True,
        attn_implementation="eager",
    )
    config = transformers.Florence2Config(
        text_config=text_config,
        vision_config={
            "in_channels": 3,
            "depths": (1, 1, 1, 1),
            "patch_size": (7, 3, 3, 3),
            "patch_stride": (4, 2, 2, 2),
            "patch_padding": (3, 1, 1, 1),
            "patch_prenorm": (False, True, True, True),
            "embed_dim": (8, 8, 8, 8),
            "num_heads": (1, 1, 1, 1),
            "num_groups": (1, 1, 1, 1),
            "window_size": 1,
            "drop_path_rate": 0.0,
            "mlp_ratio": 2.0,
            "projection_dim": 16,
            "max_temporal_embeddings": 4,
            "max_position_embeddings": 4,
        },
        image_token_id=63,
    )
    torch.manual_seed(947)
    model = transformers.Florence2ForConditionalGeneration(config).eval()
    profile = FlorenceCoreMLProfile(
        image_size=32,
        image_token_count=2,
        encoder_context_length=8,
        decoder_context_length=8,
        hidden_size=16,
        vocab_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=8,
        max_new_tokens=8,
    )
    return model, profile


def test_tiny_native_model_matches_static_encoder_and_two_stateful_steps():
    model, profile = _tiny_florence()
    encoder = Florence2CoreMLEncoder(model, profile=profile).eval()
    decoder = Florence2CoreMLStatefulDecoder(
        model,
        profile=profile,
        state_dtype=torch.float32,
    ).eval()
    pixels = torch.linspace(
        -0.75,
        0.75,
        3 * 32 * 32,
        dtype=torch.float32,
    ).reshape(1, 3, 32, 32)
    input_ids = torch.tensor([[63, 63, 4, 5, 6, 2, 1, 1]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])

    encoder_metrics = assert_florence2_static_encoder_parity(
        model,
        encoder,
        pixels,
        input_ids,
        attention_mask,
        relative_tolerance=1e-5,
    )
    decoder_metrics = assert_florence2_decoder_source_parity(
        model,
        decoder,
        profile=profile,
        relative_tolerance=1e-5,
    )

    assert max(encoder_metrics.values()) <= 1e-5
    assert max(decoder_metrics.values()) <= 1e-5
