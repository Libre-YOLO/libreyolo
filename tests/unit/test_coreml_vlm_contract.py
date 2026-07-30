"""Focused tests for the isolated stateful Core ML VLM vertical slice."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from libreyolo.export import coreml_vlm
from libreyolo.export.coreml_vlm import (
    COREML_VLM_DECODE_FUNCTION,
    COREML_VLM_EMBED_TOKENS_FUNCTION,
    COREML_VLM_ENCODE_IMAGE_FUNCTION,
    COREML_VLM_FUNCTION_NAMES,
    COREML_VLM_TRANSFORMERS_VERSION,
    CoreMLVLMDecodeCursor,
    CoreMLVLMProfile,
    CoreMLVLMStatefulLlamaDecoder,
    CoreMLVLMTokenEmbedding,
    SmolVLM2FixedSquareVisionEncoder,
    assert_smolvlm2_fixed_vision_eager_parity,
    build_coreml_vlm_multifunction_package,
    coreml_vlm_function_contracts,
    require_coreml_vlm_toolchain,
    smolvlm2_500m_coreml_metadata,
    smolvlm2_500m_coreml_profile,
    stringify_coreml_vlm_metadata,
    validate_coreml_vlm_context_budget,
    validate_coreml_vlm_decode_bounds,
    validate_coreml_vlm_function_description,
    validate_coreml_vlm_metadata,
    validate_smolvlm2_500m_processor_assets,
)

pytestmark = [pytest.mark.unit, pytest.mark.experimental_backend]


def _tiny_profile() -> CoreMLVLMProfile:
    return CoreMLVLMProfile(
        family="test_vlm",
        size="tiny",
        context_length=16,
        image_crops=3,
        image_channels=3,
        image_height=32,
        image_width=32,
        image_tokens_per_crop=4,
        hidden_size=32,
        vocab_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_new_tokens=3,
    )


def _tiny_smol_model(*, dtype: torch.dtype = torch.float32):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != COREML_VLM_TRANSFORMERS_VERSION:
        pytest.skip(
            "Eager wrapper provenance test requires transformers "
            f"{COREML_VLM_TRANSFORMERS_VERSION}."
        )
    from transformers import SmolVLMConfig, SmolVLMForConditionalGeneration

    config = SmolVLMConfig(
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 32,
            "patch_size": 8,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "attention_dropout": 0.0,
        },
        text_config={
            "model_type": "llama",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "pad_token_id": 2,
            "attention_bias": False,
            "mlp_bias": False,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "rope_theta": 100000.0,
        },
        scale_factor=2,
        image_token_id=63,
        pad_token_id=2,
        use_cache=True,
    )
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    torch.manual_seed(7)
    return SmolVLMForConditionalGeneration(config).to(dtype=dtype).eval()


def _causal_mask(length: int, *, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full(
        (1, 1, length, length),
        torch.finfo(dtype).min,
        dtype=dtype,
    )
    return torch.triu(mask, diagonal=1)


def test_smolvlm2_500m_profile_pins_state_and_image_abi():
    profile = smolvlm2_500m_coreml_profile()

    assert profile.image_token_count == 1088
    assert profile.cache_shape == (32, 1, 5, 4096, 64)
    assert profile.cache_bytes_fp16 == 160 * 1024 * 1024
    contracts = coreml_vlm_function_contracts(profile)
    assert list(contracts) == list(COREML_VLM_FUNCTION_NAMES)
    assert contracts[COREML_VLM_DECODE_FUNCTION]["stateful"] is True
    assert [state["name"] for state in contracts["decode"]["states"]] == [
        "key_cache",
        "value_cache",
    ]
    mask = contracts["decode"]["inputs"][1]
    assert "Q <= E" in mask["semantic_constraint"]
    assert mask["shape"][2]["upper_bound"] == 4096
    assert mask["shape"][3]["upper_bound"] == 4096
    assert (
        contracts["decode"]["inputs"][2]["semantic_constraint"]
        == "position_ids == arange(E - Q, E)"
    )


@pytest.mark.parametrize("context", [2048, 4096, 8192])
def test_smolvlm2_500m_profile_accepts_only_finite_reviewed_contexts(context):
    profile = smolvlm2_500m_coreml_profile(context)
    assert profile.context_length == context
    assert profile.max_new_tokens == (512 if context == 2048 else 1024)


def test_profile_rejects_impossible_image_plus_generation_budget():
    with pytest.raises(ValueError, match="leave room for text"):
        CoreMLVLMProfile(
            family="impossible",
            size="tiny",
            context_length=16,
            image_crops=3,
            image_channels=3,
            image_height=32,
            image_width=32,
            image_tokens_per_crop=4,
            hidden_size=32,
            vocab_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_new_tokens=4,
        )


@pytest.mark.parametrize("context", [0, 1024, 4097, 16384, True])
def test_smolvlm2_500m_profile_rejects_unknown_contexts(context):
    with pytest.raises(ValueError, match="context_length"):
        smolvlm2_500m_coreml_profile(context)


def test_decode_and_context_relational_guards_fail_before_runtime():
    profile = smolvlm2_500m_coreml_profile()
    compact_profile = smolvlm2_500m_coreml_profile(2048)

    assert validate_coreml_vlm_decode_bounds(profile, query_length=8, end_step=16) == (
        8,
        16,
    )
    with pytest.raises(ValueError, match="must not exceed"):
        validate_coreml_vlm_decode_bounds(profile, query_length=9, end_step=8)
    with pytest.raises(ValueError, match="exceeds Core ML context"):
        validate_coreml_vlm_decode_bounds(profile, query_length=1, end_step=4097)

    assert validate_coreml_vlm_context_budget(
        profile,
        prompt_tokens=1500,
        max_new_tokens=1024,
        image_tokens=1088,
    ) == (1500, 1024)
    with pytest.raises(ValueError, match="image placeholders"):
        validate_coreml_vlm_context_budget(
            profile,
            prompt_tokens=1500,
            max_new_tokens=1024,
            image_tokens=1087,
        )
    with pytest.raises(ValueError, match="generation budget"):
        validate_coreml_vlm_context_budget(
            profile,
            prompt_tokens=3500,
            max_new_tokens=1024,
            image_tokens=1088,
        )
    assert validate_coreml_vlm_context_budget(
        compact_profile,
        prompt_tokens=1137,
        max_new_tokens=512,
        image_tokens=1088,
    ) == (1137, 512)
    with pytest.raises(ValueError, match="artifact maximum"):
        validate_coreml_vlm_context_budget(
            compact_profile,
            prompt_tokens=1137,
            max_new_tokens=513,
            image_tokens=1088,
        )


def test_append_only_decode_controls_are_canonical_and_fail_closed():
    profile = _tiny_profile()
    mask, positions = coreml_vlm.build_coreml_vlm_decode_controls(
        profile,
        query_length=3,
        end_step=5,
    )

    assert mask.dtype == np.float16
    assert positions.dtype == np.int32
    assert mask.shape == (1, 1, 3, 5)
    assert positions.tolist() == [[2, 3, 4]]
    assert mask[0, 0, 0].tolist() == [
        0.0,
        0.0,
        0.0,
        -65504.0,
        -65504.0,
    ]
    assert coreml_vlm.validate_coreml_vlm_decode_controls(
        profile,
        causal_mask=mask,
        position_ids=positions,
    ) == (3, 5)

    bad_positions = positions.copy()
    bad_positions[0, 0] = 1
    with pytest.raises(ValueError, match="append-only"):
        coreml_vlm.validate_coreml_vlm_decode_controls(
            profile,
            causal_mask=mask,
            position_ids=bad_positions,
        )
    bad_mask = mask.copy()
    bad_mask[0, 0, 0, 4] = 0
    with pytest.raises(ValueError, match="canonical"):
        coreml_vlm.validate_coreml_vlm_decode_controls(
            profile,
            causal_mask=bad_mask,
            position_ids=positions,
        )
    with pytest.raises(ValueError, match="float16"):
        coreml_vlm.validate_coreml_vlm_decode_controls(
            profile,
            causal_mask=mask.astype(np.float32),
            position_ids=positions,
        )


def test_decode_cursor_rejects_skips_rewrites_and_gaps():
    profile = _tiny_profile()
    cursor = CoreMLVLMDecodeCursor(profile)
    mask, positions = cursor.controls(query_length=3)
    assert positions.tolist() == [[0, 1, 2]]
    assert cursor.end_step == 0
    assert cursor.commit(
        causal_mask=mask,
        position_ids=positions,
    ) == 3

    skipped_mask, skipped_positions = coreml_vlm.build_coreml_vlm_decode_controls(
        profile,
        query_length=1,
        end_step=5,
    )
    with pytest.raises(ValueError, match="paired state cursor"):
        cursor.commit(
            causal_mask=skipped_mask,
            position_ids=skipped_positions,
        )
    rewrite_mask, rewrite_positions = coreml_vlm.build_coreml_vlm_decode_controls(
        profile,
        query_length=1,
        end_step=3,
    )
    with pytest.raises(ValueError, match="paired state cursor"):
        cursor.commit(
            causal_mask=rewrite_mask,
            position_ids=rewrite_positions,
        )

    next_mask, next_positions = cursor.controls(query_length=1)
    assert next_positions.tolist() == [[3]]
    assert cursor.commit(
        causal_mask=next_mask,
        position_ids=next_positions,
    ) == 4
    # A fresh host cursor must be created together with a fresh Core ML
    # MLState; rewinding only one side would expose stale KV slots.
    cursor = CoreMLVLMDecodeCursor(profile)
    fresh_mask, fresh_positions = cursor.controls(query_length=1)
    assert fresh_positions.tolist() == [[0]]
    assert cursor.commit(
        causal_mask=fresh_mask,
        position_ids=fresh_positions,
    ) == 1


def test_raw_image_preprocess_is_strict_and_deterministic():
    source = np.zeros((3, 5, 3), dtype=np.uint8)
    source[..., 0] = np.arange(5, dtype=np.uint8)
    source[..., 1] = np.arange(3, dtype=np.uint8)[:, None]

    actual = coreml_vlm.preprocess_smolvlm2_500m_coreml_image(source)
    expected = Image.fromarray(source, mode="RGB").resize(
        (2048, 2048),
        resample=Image.Resampling.BILINEAR,
    )
    assert actual.mode == "RGB"
    assert actual.size == (2048, 2048)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    unit_float = source.astype(np.float32) / 255.0
    np.testing.assert_array_equal(
        np.asarray(
            coreml_vlm.preprocess_smolvlm2_500m_coreml_image(unit_float)
        ),
        np.asarray(expected),
    )

    for bad in (
        np.full((2, 2, 3), -1, dtype=np.int16),
        np.full((2, 2, 3), 256, dtype=np.uint16),
    ):
        with pytest.raises(ValueError, match=r"\[0, 1\] or \[0, 255\]|non-negative"):
            coreml_vlm.preprocess_smolvlm2_500m_coreml_image(bad)

    with pytest.raises(ValueError, match="fixed at 2048x2048"):
        coreml_vlm.preprocess_smolvlm2_500m_coreml_image(
            source,
            image_size=512,
        )


def test_host_image_embedding_merge_requires_exact_placeholders_and_abi():
    profile = _tiny_profile()
    input_ids = np.array(
        [[49190] * 6 + [7] + [49190] * 6],
        dtype=np.int32,
    )
    token_embeddings = np.zeros((1, 13, 32), dtype=np.float16)
    image_embeddings = np.arange(12 * 32, dtype=np.float16).reshape(1, 12, 32)

    merged = coreml_vlm.merge_coreml_vlm_image_embeddings(
        profile,
        input_ids=input_ids,
        token_embeddings=token_embeddings,
        image_embeddings=image_embeddings,
    )
    np.testing.assert_array_equal(
        merged[input_ids == 49190],
        image_embeddings.reshape(12, 32),
    )
    np.testing.assert_array_equal(merged[0, 6], np.zeros(32, dtype=np.float16))
    assert merged.flags.c_contiguous

    bad_ids = input_ids.copy()
    bad_ids[0, 0] = 7
    with pytest.raises(ValueError, match="placeholder count"):
        coreml_vlm.merge_coreml_vlm_image_embeddings(
            profile,
            input_ids=bad_ids,
            token_embeddings=token_embeddings,
            image_embeddings=image_embeddings,
        )
    with pytest.raises(ValueError, match="float16"):
        coreml_vlm.merge_coreml_vlm_image_embeddings(
            profile,
            input_ids=input_ids,
            token_embeddings=token_embeddings.astype(np.float32),
            image_embeddings=image_embeddings,
        )


def test_fixed_square_processor_bridge_validates_and_casts_host_inputs():
    profile = smolvlm2_500m_coreml_profile(2048)
    input_ids = np.array(
        [[49190] * profile.image_token_count + [7] * 49],
        dtype=np.int64,
    )
    batch = {
        "input_ids": input_ids,
        "attention_mask": np.ones_like(input_ids),
        "pixel_values": np.zeros(
            (
                1,
                profile.image_crops,
                profile.image_channels,
                profile.image_height,
                profile.image_width,
            ),
            dtype=np.float16,
        ),
        "pixel_attention_mask": np.ones(
            (
                1,
                profile.image_crops,
                profile.image_height,
                profile.image_width,
            ),
            dtype=bool,
        ),
    }

    prepared = coreml_vlm.prepare_smolvlm2_500m_coreml_processor_batch(
        profile,
        batch,
    )
    assert prepared["input_ids"].dtype == np.int32
    assert prepared["pixel_values"].dtype == np.float16
    assert prepared["input_ids"].flags.c_contiguous
    assert prepared["pixel_values"].flags.c_contiguous

    rectangular = dict(batch)
    rectangular["pixel_values"] = batch["pixel_values"][:, :13]
    rectangular["pixel_attention_mask"] = batch["pixel_attention_mask"][:, :13]
    with pytest.raises(ValueError, match="fixed-square crop ABI"):
        coreml_vlm.prepare_smolvlm2_500m_coreml_processor_batch(
            profile,
            rectangular,
        )
    batch["pixel_attention_mask"][0, 0, 0, 0] = False
    with pytest.raises(ValueError, match="all-valid"):
        coreml_vlm.prepare_smolvlm2_500m_coreml_processor_batch(
            profile,
            batch,
        )
    batch["pixel_attention_mask"][0, 0, 0, 0] = True
    with pytest.raises(ValueError, match="artifact maximum"):
        coreml_vlm.prepare_smolvlm2_500m_coreml_processor_batch(
            profile,
            batch,
            max_new_tokens=513,
        )


def test_metadata_roundtrip_and_hash_tamper_fail_closed():
    metadata = smolvlm2_500m_coreml_metadata()
    assert metadata["coreml_execution_profile_status"] == "experimental"
    assert "coreml_execution_profile" not in metadata
    stringified = stringify_coreml_vlm_metadata(metadata)

    validated = validate_coreml_vlm_metadata(stringified)
    assert validated == metadata
    assert json.loads(stringified["processor"])["trust_remote_code"] is False
    weights = json.loads(stringified["weights"])
    assert weights["revision"] == coreml_vlm.SMOLVLM2_500M_REVISION
    assert weights["size_bytes"] == 2_029_990_624
    assert (
        weights["sha256"]
        == "b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3"
    )
    assert stringified["precision"] == "mixed"
    assert stringified["vision_compute_precision"] == "fp32"
    assert stringified["token_embedding_compute_precision"] == "fp16"
    assert stringified["decoder_compute_precision"] == "fp32"
    assert stringified["function_io_precision"] == "fp16"
    assert stringified["state_precision"] == "fp16"
    assert stringified["conversion_source_precision"] == "fp32"

    bad = dict(stringified)
    functions = json.loads(bad["vlm_functions"])
    functions["decode"]["inputs"][1]["shape"][3]["upper_bound"] = 8192
    bad["vlm_functions"] = json.dumps(functions)
    with pytest.raises(ValueError, match="vlm_functions"):
        validate_coreml_vlm_metadata(bad)

    missing = dict(stringified)
    del missing["processor"]
    with pytest.raises(ValueError, match=r"missing=\['processor'\]"):
        validate_coreml_vlm_metadata(missing)

    extra = dict(stringified)
    extra["unreviewed_metadata"] = "must-not-survive"
    with pytest.raises(ValueError, match="extra=.*unreviewed_metadata"):
        validate_coreml_vlm_metadata(extra)


def test_processor_snapshot_revision_version_missing_and_hash_guards(
    tmp_path,
    monkeypatch,
):
    contents = {
        "chat_template.json": b"template",
        "generation_config.json": b"generation",
        "processor.json": b"strict",
        "tokenizer.json": b"tokens",
    }
    hashes = {
        name: hashlib.sha256(value).hexdigest() for name, value in contents.items()
    }
    monkeypatch.setattr(coreml_vlm, "SMOLVLM2_500M_REQUIRED_ASSETS", hashes)
    for name, value in contents.items():
        (tmp_path / name).write_bytes(value)

    manifest = validate_smolvlm2_500m_processor_assets(
        tmp_path,
        revision=coreml_vlm.SMOLVLM2_500M_REVISION,
    )
    assert manifest["required_assets"] == hashes

    with pytest.raises(ValueError, match="processor revision"):
        validate_smolvlm2_500m_processor_assets(
            tmp_path,
            revision="0" * 40,
        )
    with pytest.raises(ValueError, match="processor semantics"):
        validate_smolvlm2_500m_processor_assets(
            tmp_path,
            revision=coreml_vlm.SMOLVLM2_500M_REVISION,
            transformers_version="5.13.0",
        )

    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        validate_smolvlm2_500m_processor_assets(
            tmp_path,
            revision=coreml_vlm.SMOLVLM2_500M_REVISION,
        )
    (tmp_path / "tokenizer.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_smolvlm2_500m_processor_assets(
            tmp_path,
            revision=coreml_vlm.SMOLVLM2_500M_REVISION,
        )


def test_weight_snapshot_revision_missing_size_and_hash_guards(
    tmp_path,
    monkeypatch,
):
    payload = b"weights"
    monkeypatch.setattr(
        coreml_vlm,
        "SMOLVLM2_500M_WEIGHTS_FILENAME",
        "model.safetensors",
    )
    monkeypatch.setattr(
        coreml_vlm,
        "SMOLVLM2_500M_WEIGHTS_SIZE",
        len(payload),
    )
    monkeypatch.setattr(
        coreml_vlm,
        "SMOLVLM2_500M_WEIGHTS_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    path = tmp_path / "model.safetensors"
    path.write_bytes(payload)

    manifest = coreml_vlm.validate_smolvlm2_500m_weight_asset(
        tmp_path,
        revision=coreml_vlm.SMOLVLM2_500M_REVISION,
    )
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()

    with pytest.raises(ValueError, match="weight revision"):
        coreml_vlm.validate_smolvlm2_500m_weight_asset(
            tmp_path,
            revision="0" * 40,
        )
    path.unlink()
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        coreml_vlm.validate_smolvlm2_500m_weight_asset(
            tmp_path,
            revision=coreml_vlm.SMOLVLM2_500M_REVISION,
        )
    path.write_bytes(b"short")
    with pytest.raises(ValueError, match="byte length"):
        coreml_vlm.validate_smolvlm2_500m_weight_asset(
            tmp_path,
            revision=coreml_vlm.SMOLVLM2_500M_REVISION,
        )
    path.write_bytes(b"WEIGHTS")
    with pytest.raises(ValueError, match="SHA-256"):
        coreml_vlm.validate_smolvlm2_500m_weight_asset(
            tmp_path,
            revision=coreml_vlm.SMOLVLM2_500M_REVISION,
        )


def test_in_memory_weights_must_equal_hash_pinned_safetensors(
    tmp_path,
    monkeypatch,
):
    from safetensors.torch import save_file

    model = nn.Linear(2, 3).eval()
    path = tmp_path / "model.safetensors"
    save_file(model.state_dict(), path)
    monkeypatch.setattr(
        coreml_vlm,
        "SMOLVLM2_500M_WEIGHTS_FILENAME",
        path.name,
    )

    coreml_vlm.validate_smolvlm2_500m_model_weight_values(model, tmp_path)
    with torch.no_grad():
        model.weight[0, 0].add_(1)
    with pytest.raises(ValueError, match="differs"):
        coreml_vlm.validate_smolvlm2_500m_model_weight_values(model, tmp_path)


def test_toolchain_pin_is_exact_major():
    require_coreml_vlm_toolchain(SimpleNamespace(__version__="9.0"))
    with pytest.raises(RuntimeError, match="9.x"):
        require_coreml_vlm_toolchain(SimpleNamespace(__version__="8.3"))
    with pytest.raises(RuntimeError, match="9.x"):
        require_coreml_vlm_toolchain(SimpleNamespace(__version__="10.0"))
    coreml_vlm.require_coreml_vlm_transformers_toolchain(
        SimpleNamespace(__version__="5.12.1")
    )
    with pytest.raises(RuntimeError, match="transformers 5.12.1"):
        coreml_vlm.require_coreml_vlm_transformers_toolchain(
            SimpleNamespace(__version__="5.13.0")
        )


def test_exact_model_config_rejects_numerically_relevant_drift():
    config = SimpleNamespace(
        model_type="smolvlm",
        image_token_id=49190,
        scale_factor=4,
        text_config=SimpleNamespace(
            hidden_size=960,
            intermediate_size=2560,
            num_hidden_layers=32,
            num_attention_heads=15,
            num_key_value_heads=5,
            head_dim=64,
            vocab_size=49280,
            max_position_embeddings=8192,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            rope_parameters={
                "rope_theta": 100000.0,
                "rope_type": "default",
            },
            attention_bias=False,
            mlp_bias=False,
        ),
        vision_config=SimpleNamespace(
            hidden_size=768,
            image_size=512,
            patch_size=16,
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_act="gelu_pytorch_tanh",
            layer_norm_eps=1e-6,
            num_channels=3,
        ),
    )
    model = SimpleNamespace(
        config=config,
        model=SimpleNamespace(
            vision_model=object(),
            connector=object(),
            text_model=object(),
        ),
        lm_head=object(),
    )
    coreml_vlm.validate_smolvlm2_500m_model(model)

    config.text_config.rms_norm_eps = 1e-6
    with pytest.raises(ValueError, match="rms_norm_eps"):
        coreml_vlm.validate_smolvlm2_500m_model(model)


def test_export_default_rejects_before_toolchain_or_conversion(
    tmp_path,
    monkeypatch,
):
    touched = []
    monkeypatch.setattr(
        coreml_vlm,
        "require_coreml_vlm_transformers_toolchain",
        lambda: touched.append("toolchain"),
    )

    with pytest.raises(NotImplementedError, match="exact Apple-M4"):
        coreml_vlm.export_smolvlm2_500m_coreml_package(
            nn.Identity(),
            processor_dir=tmp_path,
            processor_revision=coreml_vlm.SMOLVLM2_500M_REVISION,
            output_path=tmp_path / "blocked.mlpackage",
        )

    assert touched == []


@pytest.mark.parametrize("conversion_fails", [False, True])
def test_export_restores_every_module_training_flag(
    tmp_path,
    monkeypatch,
    conversion_fails,
):
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
    model.train()
    model[1].eval()
    expected = tuple(module.training for module in model.modules())

    monkeypatch.setattr(
        coreml_vlm,
        "require_coreml_vlm_transformers_toolchain",
        lambda: None,
    )
    monkeypatch.setattr(
        coreml_vlm,
        "smolvlm2_500m_coreml_profile",
        lambda _context: _tiny_profile(),
    )
    for name in (
        "validate_smolvlm2_500m_model",
        "validate_smolvlm2_500m_processor_assets",
        "validate_smolvlm2_500m_weight_asset",
        "validate_smolvlm2_500m_model_weight_values",
        "assert_smolvlm2_fixed_vision_eager_parity",
        "assert_smolvlm2_decoder_source_parity",
    ):
        monkeypatch.setattr(coreml_vlm, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        coreml_vlm,
        "wrap_smolvlm2_500m_coreml_components",
        lambda *_args, **_kwargs: {
            name: nn.Identity() for name in COREML_VLM_FUNCTION_NAMES
        },
    )
    monkeypatch.setattr(
        coreml_vlm,
        "smolvlm2_500m_coreml_metadata",
        lambda _profile: {},
    )

    def fake_build(*_args, **_kwargs):
        if conversion_fails:
            raise RuntimeError("conversion failed")
        return str(tmp_path / "smol.mlpackage")

    monkeypatch.setattr(
        coreml_vlm,
        "build_coreml_vlm_multifunction_package",
        fake_build,
    )

    if conversion_fails:
        with pytest.raises(RuntimeError, match="conversion failed"):
            coreml_vlm.export_smolvlm2_500m_coreml_package(
                model,
                processor_dir=tmp_path,
                processor_revision=coreml_vlm.SMOLVLM2_500M_REVISION,
                output_path=tmp_path / "smol.mlpackage",
                compute_units="cpu_only",
            )
    else:
        coreml_vlm.export_smolvlm2_500m_coreml_package(
            model,
            processor_dir=tmp_path,
            processor_revision=coreml_vlm.SMOLVLM2_500M_REVISION,
            output_path=tmp_path / "smol.mlpackage",
            compute_units="cpu_only",
        )

    assert tuple(module.training for module in model.modules()) == expected


def test_export_rejects_half_source_graph_before_coremltools(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        coreml_vlm,
        "validate_smolvlm2_500m_processor_assets",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        coreml_vlm,
        "validate_smolvlm2_500m_model",
        lambda _model: None,
    )
    monkeypatch.setattr(
        coreml_vlm,
        "validate_smolvlm2_500m_weight_asset",
        lambda *_args, **_kwargs: {},
    )
    model = nn.Linear(1, 1).half().eval()
    with pytest.raises(NotImplementedError, match="FP32-loaded"):
        coreml_vlm.export_smolvlm2_500m_coreml_package(
            model,
            processor_dir=tmp_path,
            processor_revision=coreml_vlm.SMOLVLM2_500M_REVISION,
            output_path=tmp_path / "blocked.mlpackage",
            compute_units="cpu_only",
        )


def test_fixed_square_vision_wrapper_is_exact_on_all_valid_grid():
    model = _tiny_smol_model()
    wrapper = SmolVLM2FixedSquareVisionEncoder(
        model,
        image_crops=3,
    ).eval()
    pixels = torch.rand((1, 3, 3, 32, 32)) + 0.01

    assert_smolvlm2_fixed_vision_eager_parity(model, wrapper, pixels)
    assert tuple(wrapper(pixels).shape) == (1, 12, 32)


def test_stateful_decoder_matches_stock_prefill_and_incremental_decode():
    from transformers.cache_utils import DynamicCache

    model = _tiny_smol_model()
    decoder = CoreMLVLMStatefulLlamaDecoder(
        text_model=model.model.text_model,
        lm_head=model.lm_head,
        context_length=16,
        state_dtype=torch.float32,
    ).eval()
    stock_cache = DynamicCache(config=model.config.text_config)

    input_ids = torch.tensor([[3, 4, 5]], dtype=torch.long)
    embeddings = model.get_input_embeddings()(input_ids)
    positions = torch.arange(3, dtype=torch.int32).unsqueeze(0)
    with torch.inference_mode():
        stock = model.model.text_model(
            inputs_embeds=embeddings,
            attention_mask=torch.ones((1, 3)),
            position_ids=positions.long(),
            past_key_values=stock_cache,
            use_cache=True,
        )
        expected = model.lm_head(stock.last_hidden_state[:, -1, :])
        actual = decoder(
            embeddings,
            _causal_mask(3, dtype=torch.float32),
            positions,
        )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    next_ids = torch.tensor([[6]], dtype=torch.long)
    next_embeddings = model.get_input_embeddings()(next_ids)
    next_positions = torch.tensor([[3]], dtype=torch.int32)
    with torch.inference_mode():
        stock_next = model.model.text_model(
            inputs_embeds=next_embeddings,
            attention_mask=torch.ones((1, 4)),
            position_ids=next_positions.long(),
            past_key_values=stock_cache,
            use_cache=True,
        )
        expected_next = model.lm_head(stock_next.last_hidden_state[:, -1, :])
        actual_next = decoder(
            next_embeddings,
            torch.zeros((1, 1, 1, 4)),
            next_positions,
        )
    torch.testing.assert_close(
        actual_next,
        expected_next,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(decoder.key_cache[:, :, :, :4, :]) > 0
    decoder.reset_state()
    assert torch.count_nonzero(decoder.key_cache) == 0
    assert torch.count_nonzero(decoder.value_cache) == 0


def test_traced_decoder_keeps_q_and_e_dynamic_across_prefill_and_decode():
    model = _tiny_smol_model()
    profile = _tiny_profile()
    traced_source = CoreMLVLMStatefulLlamaDecoder(
        text_model=model.model.text_model,
        lm_head=model.lm_head,
        context_length=profile.context_length,
        state_dtype=torch.float32,
    ).eval()
    reference = CoreMLVLMStatefulLlamaDecoder(
        text_model=model.model.text_model,
        lm_head=model.lm_head,
        context_length=profile.context_length,
        state_dtype=torch.float32,
    ).eval()
    captured, _ = coreml_vlm._capture_coreml_vlm_component(
        traced_source,
        function_name=COREML_VLM_DECODE_FUNCTION,
        profile=profile,
    )

    for input_ids, end_step in (
        (torch.tensor([[3, 4, 5]]), 3),
        (torch.tensor([[6]]), 4),
    ):
        embeddings = model.get_input_embeddings()(input_ids)
        mask, positions = coreml_vlm.build_coreml_vlm_decode_controls(
            profile,
            query_length=input_ids.shape[1],
            end_step=end_step,
        )
        torch_mask = torch.from_numpy(mask).to(torch.float32)
        torch_positions = torch.from_numpy(positions)
        with torch.inference_mode():
            expected = reference(embeddings, torch_mask, torch_positions)
            actual = captured(embeddings, torch_mask, torch_positions)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class _FakeArray:
    def __init__(self, dtype, ranges):
        self.dataType = dtype
        self.shape = []
        self.shapeRange = SimpleNamespace(
            sizeRanges=[
                SimpleNamespace(lowerBound=lower, upperBound=upper)
                for lower, upper in ranges
            ]
        )

    def WhichOneof(self, _):
        return "shapeRange"


def _fake_feature(name, dtype, ranges):
    return SimpleNamespace(
        name=name,
        type=SimpleNamespace(multiArrayType=_FakeArray(dtype, ranges)),
    )


def _fake_state(name, shape):
    array = SimpleNamespace(dataType=65552, shape=list(shape))
    return SimpleNamespace(
        name=name,
        type=SimpleNamespace(
            stateType=SimpleNamespace(arrayType=array),
        ),
    )


def _fake_function_description(function_name, profile):
    contract = coreml_vlm_function_contracts(profile)[function_name]
    dtype_codes = {"float16": 65552, "int32": 131104}
    return SimpleNamespace(
        input=[
            _fake_feature(
                item["name"],
                dtype_codes[item["dtype"]],
                coreml_vlm._feature_shape_ranges(item),
            )
            for item in contract["inputs"]
        ],
        output=[
            _fake_feature(
                item["name"],
                dtype_codes[item["dtype"]],
                coreml_vlm._feature_shape_ranges(item),
            )
            for item in contract["outputs"]
        ],
        state=[
            _fake_state(item["name"], item["shape"])
            for item in contract.get("states", [])
        ],
    )


def test_function_description_validation_pins_ranges_dtypes_and_state():
    profile = smolvlm2_500m_coreml_profile()
    description = _fake_function_description(
        COREML_VLM_DECODE_FUNCTION,
        profile,
    )
    validate_coreml_vlm_function_description(
        description,
        function_name=COREML_VLM_DECODE_FUNCTION,
        profile=profile,
    )

    description.input[1].type.multiArrayType.shapeRange.sizeRanges[3].upperBound = 8192
    with pytest.raises(RuntimeError, match="changed bounds"):
        validate_coreml_vlm_function_description(
            description,
            function_name=COREML_VLM_DECODE_FUNCTION,
            profile=profile,
        )

    description = _fake_function_description(
        COREML_VLM_DECODE_FUNCTION,
        profile,
    )
    description.output[0].type.multiArrayType.shapeRange.sizeRanges.clear()
    with pytest.raises(RuntimeError, match="changed shape"):
        validate_coreml_vlm_function_description(
            description,
            function_name=COREML_VLM_DECODE_FUNCTION,
            profile=profile,
        )


def test_package_publication_fails_closed_without_atomic_rename(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "staged.mlpackage"
    destination = tmp_path / "published.mlpackage"
    source.mkdir()
    monkeypatch.setattr(coreml_vlm.os, "name", "posix")
    monkeypatch.setattr(coreml_vlm.sys, "platform", "unsupported")
    monkeypatch.setattr(
        coreml_vlm.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="atomic no-replace"):
        coreml_vlm._publish_directory_no_replace(source, destination)

    assert source.is_dir()
    assert not destination.exists()


def test_package_builder_uses_native_multifunction_and_no_overwrite(
    tmp_path,
    monkeypatch,
):
    profile = smolvlm2_500m_coreml_profile()
    metadata = smolvlm2_500m_coreml_metadata(profile)
    converted_names = []
    metadata_store = {}

    class FakeConverted:
        def __init__(self, function_name):
            self.function_name = function_name

        def get_spec(self):
            return SimpleNamespace(
                description=_fake_function_description(
                    self.function_name,
                    profile,
                )
            )

        def save(self, path):
            Path(path).mkdir()

    def fake_convert(_ct, _component, *, function_name, **_kwargs):
        converted_names.append(function_name)
        return FakeConverted(function_name)

    class FakeDescriptor:
        def __init__(self):
            self.functions = []
            self.default_function_name = None

        def add_function(self, path, source_name, target_name):
            self.functions.append((path, source_name, target_name))

    def fake_save_multifunction(descriptor, path):
        assert [item[2] for item in descriptor.functions] == list(
            COREML_VLM_FUNCTION_NAMES
        )
        assert descriptor.default_function_name == COREML_VLM_ENCODE_IMAGE_FUNCTION
        Path(path).mkdir()

    class FakeMLModel:
        def __init__(self, path, skip_model_load=True):
            del skip_model_load
            self.path = Path(path)
            self.user_defined_metadata = dict(
                metadata_store.get(
                    str(self.path),
                    metadata_store.get("latest", {}),
                )
            )

        def save(self, path):
            target = Path(path)
            target.mkdir()
            metadata_store[str(target)] = dict(self.user_defined_metadata)
            metadata_store["latest"] = dict(self.user_defined_metadata)

        def get_spec(self):
            return SimpleNamespace()

    fake_ct = SimpleNamespace(
        __version__="9.0",
        utils=SimpleNamespace(
            MultiFunctionDescriptor=FakeDescriptor,
            save_multifunction=fake_save_multifunction,
        ),
        models=SimpleNamespace(MLModel=FakeMLModel),
    )
    monkeypatch.setattr(
        coreml_vlm,
        "_convert_coreml_vlm_component",
        fake_convert,
    )
    monkeypatch.setattr(
        coreml_vlm,
        "validate_coreml_vlm_multifunction_spec",
        lambda *_args, **_kwargs: None,
    )

    def fake_write_metadata(_ct, _model, package_path, values):
        metadata_store[str(package_path)] = dict(values)
        metadata_store["latest"] = dict(values)

    monkeypatch.setattr(
        coreml_vlm,
        "_write_coreml_vlm_metadata_in_place",
        fake_write_metadata,
    )
    # The fake model has no real protobuf; individual descriptions above cover
    # exact ABI validation.
    output = tmp_path / "smol.mlpackage"
    components = {
        COREML_VLM_ENCODE_IMAGE_FUNCTION: nn.Identity(),
        COREML_VLM_EMBED_TOKENS_FUNCTION: nn.Identity(),
        COREML_VLM_DECODE_FUNCTION: nn.Identity(),
    }
    with pytest.raises(ValueError, match="profile conflicts"):
        build_coreml_vlm_multifunction_package(
            components,
            output_path=tmp_path / "mismatched.mlpackage",
            profile=smolvlm2_500m_coreml_profile(2048),
            metadata=metadata,
            compute_units="cpu_only",
            coremltools_module=fake_ct,
        )
    result = build_coreml_vlm_multifunction_package(
        components,
        output_path=output,
        profile=profile,
        metadata=metadata,
        compute_units="cpu_only",
        coremltools_module=fake_ct,
    )

    assert result == str(output)
    assert output.is_dir()
    assert converted_names == list(COREML_VLM_FUNCTION_NAMES)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_coreml_vlm_multifunction_package(
            components,
            output_path=output,
            profile=profile,
            metadata=metadata,
            compute_units="cpu_only",
            coremltools_module=fake_ct,
        )

    raced = tmp_path / "raced.mlpackage"
    sentinel = raced / "winner.txt"

    def lose_publication_race(_source, destination):
        destination.mkdir()
        sentinel.write_text("competitor", encoding="utf-8")
        raise FileExistsError("simulated package publication race")

    monkeypatch.setattr(
        coreml_vlm,
        "_publish_directory_no_replace",
        lose_publication_race,
    )
    with pytest.raises(FileExistsError, match="publication race"):
        build_coreml_vlm_multifunction_package(
            components,
            output_path=raced,
            profile=profile,
            metadata=metadata,
            compute_units="cpu_only",
            coremltools_module=fake_ct,
        )
    assert sentinel.read_text(encoding="utf-8") == "competitor"


def test_metadata_update_replaces_only_model_protobuf(tmp_path):
    package = tmp_path / "model.mlpackage"
    model_file = package / "Data" / "com.apple.CoreML" / "model.mlmodel"
    weight_file = package / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
    model_file.parent.mkdir(parents=True)
    weight_file.parent.mkdir()
    model_file.write_bytes(b"old protobuf")
    weight_file.write_bytes(b"weight sentinel")
    spec = SimpleNamespace(
        description=SimpleNamespace(
            metadata=SimpleNamespace(userDefined={}),
        )
    )
    model = SimpleNamespace(get_spec=lambda: spec)

    def save_spec(value, path):
        Path(path).write_text(
            json.dumps(dict(value.description.metadata.userDefined)),
            encoding="utf-8",
        )

    fake_ct = SimpleNamespace(utils=SimpleNamespace(save_spec=save_spec))
    coreml_vlm._write_coreml_vlm_metadata_in_place(
        fake_ct,
        model,
        package,
        {"contract": "v1"},
    )

    assert json.loads(model_file.read_text(encoding="utf-8")) == {
        "contract": "v1"
    }
    assert weight_file.read_bytes() == b"weight sentinel"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="coremltools conversion is unsupported on Windows",
)
def test_coremltools9_converts_and_merges_tiny_stateful_vlm(tmp_path):
    ct = pytest.importorskip("coremltools")
    if str(ct.__version__).split(".", 1)[0] != "9":
        pytest.skip("This conversion probe is pinned to coremltools 9.x.")
    profile = _tiny_profile()
    model = _tiny_smol_model(dtype=torch.float32)
    components = {
        COREML_VLM_ENCODE_IMAGE_FUNCTION: SmolVLM2FixedSquareVisionEncoder(
            model,
            image_crops=profile.image_crops,
        ).eval(),
        COREML_VLM_EMBED_TOKENS_FUNCTION: CoreMLVLMTokenEmbedding(
            model.get_input_embeddings()
        ).eval(),
        COREML_VLM_DECODE_FUNCTION: CoreMLVLMStatefulLlamaDecoder(
            text_model=model.model.text_model,
            lm_head=model.lm_head,
            context_length=profile.context_length,
            state_dtype=torch.float32,
        ).eval(),
    }
    descriptor = ct.utils.MultiFunctionDescriptor()
    for index, function_name in enumerate(COREML_VLM_FUNCTION_NAMES):
        converted = coreml_vlm._convert_coreml_vlm_component(
            ct,
            components[function_name],
            function_name=function_name,
            profile=profile,
            compute_units="cpu_and_gpu",
        )
        validate_coreml_vlm_function_description(
            converted.get_spec().description,
            function_name=function_name,
            profile=profile,
        )
        component_path = tmp_path / f"{index}-{function_name}.mlpackage"
        converted.save(str(component_path))
        descriptor.add_function(
            str(component_path),
            "main",
            function_name,
        )
    descriptor.default_function_name = COREML_VLM_ENCODE_IMAGE_FUNCTION
    merged_path = tmp_path / "tiny.mlpackage"
    ct.utils.save_multifunction(descriptor, str(merged_path))
    merged = ct.models.MLModel(str(merged_path), skip_model_load=True)
    coreml_vlm.validate_coreml_vlm_multifunction_spec(
        merged.get_spec(),
        profile=profile,
    )
