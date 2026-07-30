"""Offline contract tests for the bounded Kosmos-2 Core ML path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from libreyolo.backends.coreml_kosmos import CoreMLKosmos2Runtime
from libreyolo.export.coreml_kosmos import (
    KOSMOS2_COREML_COMPONENT_CONTRACT,
    KOSMOS2_COREML_CONTEXT_LENGTH,
    KOSMOS2_COREML_HIDDEN_SIZE,
    KOSMOS2_COREML_IMAGE_SIZE,
    KOSMOS2_COREML_IMAGE_TOKENS,
    KOSMOS2_COREML_MAX_NEW_TOKENS,
    KOSMOS2_COREML_REVISION,
    KOSMOS2_COREML_VOCAB_SIZE,
    Kosmos2CoreMLDecoder,
    Kosmos2CoreMLTokenEmbedding,
    Kosmos2CoreMLVision,
    kosmos2_bundle_manifest,
    require_kosmos2_coreml_toolchain,
    resolve_kosmos2_coreml_compute_units,
)

pytestmark = pytest.mark.unit


def test_manifest_is_exact_bounded_cpu_profile():
    manifest = kosmos2_bundle_manifest()
    assert manifest["component_contract"] == KOSMOS2_COREML_COMPONENT_CONTRACT
    assert manifest["context_length"] == KOSMOS2_COREML_CONTEXT_LENGTH == 128
    assert manifest["max_new_tokens"] == KOSMOS2_COREML_MAX_NEW_TOKENS == 48
    assert manifest["image_size"] == KOSMOS2_COREML_IMAGE_SIZE == 224
    assert manifest["image_tokens"] == KOSMOS2_COREML_IMAGE_TOKENS == 64
    assert manifest["hidden_size"] == KOSMOS2_COREML_HIDDEN_SIZE == 2048
    assert manifest["vocab_size"] == KOSMOS2_COREML_VOCAB_SIZE == 65037
    assert manifest["source"]["revision"] == KOSMOS2_COREML_REVISION
    assert manifest["compute_units"] == "cpu_only"
    assert manifest["precision"] == "fp32"


def test_compute_unit_profile_fails_closed():
    assert resolve_kosmos2_coreml_compute_units("cpu_only") == "cpu_only"
    with pytest.raises(ValueError, match="cpu_only"):
        resolve_kosmos2_coreml_compute_units("all")


def test_non_macos_toolchain_fails_before_import(monkeypatch):
    monkeypatch.setattr("libreyolo.export.coreml_kosmos.sys.platform", "win32")
    with pytest.raises(RuntimeError, match="macOS"):
        require_kosmos2_coreml_toolchain()


def test_left_pad_prefix_owns_attention_and_positions():
    ids, attention, positions = CoreMLKosmos2Runtime._left_pad_prefix([7, 8, 9])
    assert ids.shape == attention.shape == positions.shape == (1, 128)
    np.testing.assert_array_equal(ids[0, -3:], [7, 8, 9])
    np.testing.assert_array_equal(attention[0, -3:], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(positions[0, -3:], [2, 3, 4])
    assert np.all(ids[0, :-3] == 1)
    assert np.all(attention[0, :-3] == 0)
    assert np.all(positions[0, :-3] == 1)


@pytest.mark.parametrize("sequence", [[], list(range(129))])
def test_left_pad_prefix_rejects_out_of_bounds(sequence):
    with pytest.raises(ValueError, match="context"):
        CoreMLKosmos2Runtime._left_pad_prefix(sequence)


class _FakeVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(post_layernorm=nn.Identity())

    def forward(self, *, pixel_values, return_dict):
        assert return_dict is False
        pooled = pixel_values.mean(dim=(-2, -1), keepdim=False)
        return (pooled.unsqueeze(1).repeat(1, 2, 1),)


class _FakeProjection(nn.Module):
    def forward(self, features):
        return features.repeat(1, 1, 2), None


class _FakeTransformer(nn.Module):
    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        position_ids,
        use_cache,
        return_dict,
    ):
        assert use_cache is False
        assert return_dict is False
        value = inputs_embeds + attention_mask.unsqueeze(-1)
        value = value + position_ids.to(value.dtype).unsqueeze(-1)
        return (value,)


class _FakeKosmos(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = _FakeVisionModel()
        self.image_to_text_projection = _FakeProjection()
        self.embedding = nn.Embedding(16, 6)
        self.text_model = SimpleNamespace(
            model=_FakeTransformer(),
            lm_head=nn.Linear(6, 5, bias=False),
        )

    def get_input_embeddings(self):
        return self.embedding


def test_component_wrappers_have_tensor_only_contracts():
    source = _FakeKosmos().eval()
    vision = Kosmos2CoreMLVision(source)
    vision_output = vision(torch.randn(1, 3, 4, 4))
    assert vision_output.shape == (1, 2, 6)
    np.testing.assert_allclose(
        torch.linalg.vector_norm(vision_output[..., :3], dim=-1).detach().numpy(),
        np.ones((1, 2)),
        rtol=1e-6,
    )

    embedding = Kosmos2CoreMLTokenEmbedding(source)
    assert embedding(torch.tensor([[1, 2]], dtype=torch.int32)).shape == (1, 2, 6)

    decoder = Kosmos2CoreMLDecoder(source)
    output = decoder(
        torch.randn(1, 3, 6),
        torch.ones(1, 3),
        torch.tensor([[2, 3, 4]], dtype=torch.int32),
    )
    assert output.shape == (1, 5)
