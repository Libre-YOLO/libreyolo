"""Offline contract tests for the bounded Qwen3-VL-2B Core ML path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from libreyolo.backends.coreml_qwen3vl import CoreMLQwen3VLRuntime
from libreyolo.export.coreml_qwen3vl import (
    QWEN3VL_COREML_COMPONENT_CONTRACT,
    QWEN3VL_COREML_CONTEXT_LENGTH,
    QWEN3VL_COREML_HEAD_DIM,
    QWEN3VL_COREML_HIDDEN_SIZE,
    QWEN3VL_COREML_IMAGE_SIZE,
    QWEN3VL_COREML_IMAGE_TOKENS,
    QWEN3VL_COREML_MAX_NEW_TOKENS,
    QWEN3VL_COREML_PATCH_COUNT,
    QWEN3VL_COREML_PATCH_WIDTH,
    QWEN3VL_COREML_REVISION,
    QWEN3VL_COREML_VOCAB_SIZE,
    Qwen3VLCoreMLDecoder,
    Qwen3VLCoreMLTokenEmbedding,
    qwen3vl_bundle_manifest,
    require_qwen3vl_coreml_toolchain,
    resolve_qwen3vl_coreml_compute_units,
)

pytestmark = pytest.mark.unit


def test_manifest_is_exact_bounded_cpu_profile():
    manifest = qwen3vl_bundle_manifest()
    assert manifest["component_contract"] == QWEN3VL_COREML_COMPONENT_CONTRACT
    assert manifest["context_length"] == QWEN3VL_COREML_CONTEXT_LENGTH == 512
    assert manifest["max_new_tokens"] == QWEN3VL_COREML_MAX_NEW_TOKENS == 48
    assert manifest["image_size"] == QWEN3VL_COREML_IMAGE_SIZE == 448
    assert manifest["patch_count"] == QWEN3VL_COREML_PATCH_COUNT == 784
    assert manifest["patch_width"] == QWEN3VL_COREML_PATCH_WIDTH == 1536
    assert manifest["image_tokens"] == QWEN3VL_COREML_IMAGE_TOKENS == 196
    assert manifest["hidden_size"] == QWEN3VL_COREML_HIDDEN_SIZE == 2048
    assert manifest["head_dim"] == QWEN3VL_COREML_HEAD_DIM == 128
    assert manifest["vocab_size"] == QWEN3VL_COREML_VOCAB_SIZE == 151936
    assert manifest["source"]["revision"] == QWEN3VL_COREML_REVISION
    assert manifest["compute_units"] == "cpu_only"
    assert manifest["precision"] == {
        "vision": "fp32",
        "token_embedding": "fp16",
        "decoder": "fp16",
    }


def test_compute_unit_profile_fails_closed():
    assert resolve_qwen3vl_coreml_compute_units("cpu_only") == "cpu_only"
    for value in ("validated", "all", "cpu_and_ne"):
        with pytest.raises(ValueError, match="cpu_only"):
            resolve_qwen3vl_coreml_compute_units(value)


def test_non_macos_toolchain_fails_before_import(monkeypatch):
    monkeypatch.setattr("libreyolo.export.coreml_qwen3vl.sys.platform", "win32")
    with pytest.raises(RuntimeError, match="macOS"):
        require_qwen3vl_coreml_toolchain()


def test_one_image_position_ids_follow_qwen_mrope_groups():
    token_types = np.asarray([0, 0, *([1] * 196), 0, 0, 0], dtype=np.int32)
    positions = CoreMLQwen3VLRuntime._active_position_ids(token_types)
    assert positions.shape == (3, token_types.size)
    np.testing.assert_array_equal(positions[:, :2], [[0, 1], [0, 1], [0, 1]])
    np.testing.assert_array_equal(positions[:, 2], [2, 2, 2])
    np.testing.assert_array_equal(positions[:, 15], [2, 2, 15])
    np.testing.assert_array_equal(positions[:, 16], [2, 3, 2])
    np.testing.assert_array_equal(
        positions[:, -3:],
        [[16, 17, 18], [16, 17, 18], [16, 17, 18]],
    )


@pytest.mark.parametrize(
    "token_types",
    [
        np.zeros(5, dtype=np.int32),
        np.ones(195, dtype=np.int32),
        np.asarray([*([1] * 196), 0, *([1] * 196)]),
        np.asarray([0, 2, *([1] * 196)]),
    ],
)
def test_position_ids_reject_unsupported_modal_layout(token_types):
    with pytest.raises(ValueError):
        CoreMLQwen3VLRuntime._active_position_ids(token_types)


def test_host_rope_tables_are_finite_and_preserve_text_axis():
    active = np.arange(512, dtype=np.int32)
    positions = np.repeat(active[None, :], 3, axis=0)
    cosine, sine = CoreMLQwen3VLRuntime._rope_tables(positions)
    assert cosine.shape == sine.shape == (1, 512, 128)
    assert cosine.dtype == sine.dtype == np.float16
    assert np.isfinite(cosine).all()
    assert np.isfinite(sine).all()
    np.testing.assert_allclose(cosine[0, 0], 1.0)
    np.testing.assert_allclose(sine[0, 0], 0.0)


class _FakeTextLayer(nn.Module):
    def forward(
        self,
        hidden_states,
        *,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        position_embeddings,
    ):
        assert position_ids is None
        assert past_key_values is None
        assert use_cache is False
        assert attention_mask.ndim == 4
        cosine, sine = position_embeddings
        return hidden_states + cosine[..., :2] + sine[..., :2]


class _FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 2)
        language = SimpleNamespace(
            layers=nn.ModuleList([_FakeTextLayer() for _ in range(4)]),
            norm=nn.Identity(),
        )
        self.model = SimpleNamespace(language_model=language)
        self.lm_head = nn.Linear(2, 7, bias=False)

    def get_input_embeddings(self):
        return self.embedding


def test_text_component_wrappers_have_tensor_only_contracts():
    source = _FakeQwen().eval()
    embedding = Qwen3VLCoreMLTokenEmbedding(source)
    assert embedding(torch.tensor([[1, 2]], dtype=torch.int32)).shape == (1, 2, 2)

    decoder = Qwen3VLCoreMLDecoder(source)
    output = decoder(
        torch.randn(1, 3, 2),
        torch.zeros(1, 1, 3, 3),
        torch.ones(1, 3, 2),
        torch.zeros(1, 3, 2),
        torch.zeros(3, 1, 3, 2),
    )
    assert output.shape == (1, 7)
