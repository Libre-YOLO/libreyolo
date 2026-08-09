"""Structural unit tests for the V-JEPA 2 port.

These run offline with no checkpoints and no Transformers oracle. The exact
``max_abs_diff == 0.0`` parity gate against pinned ``transformers==5.1.0``
lives in ``weights/parity_vjepa2.py`` because it needs both the network and a
pinned out-of-tree oracle install.

Each test here locks in an invariant that was found to be load-bearing while
porting: getting any of them wrong still builds a model, but breaks the strict
load of a released checkpoint or silently changes the numerics.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.vjepa2.nn import (
    VJEPA2_CONFIGS,
    LibreVJEPA2Classifier,
    LibreVJEPA2Encoder,
    VJEPA2Config,
)

pytestmark = [pytest.mark.unit, pytest.mark.vjepa2]


# Transcribed from the pinned Hugging Face config.json of each released
# snapshot. Depth/heads/mlp_ratio are not inferable from the size label.
EXPECTED_ARCH = {
    "l256": (1024, 16, 24, 4.0, 256),
    "h256": (1280, 16, 32, 4.0, 256),
    "g256": (1408, 22, 40, 4.363636363636363, 256),
    "g384": (1408, 22, 40, 4.363636363636363, 384),
}


@pytest.mark.parametrize("size", sorted(EXPECTED_ARCH))
def test_architecture_table_matches_pinned_configs(size):
    hidden, heads, layers, ratio, crop = EXPECTED_ARCH[size]
    cfg = VJEPA2_CONFIGS[size]
    assert cfg["hidden_size"] == hidden
    assert cfg["num_attention_heads"] == heads
    assert cfg["num_hidden_layers"] == layers
    assert cfg["mlp_ratio"] == pytest.approx(ratio)
    assert cfg["crop_size"] == crop


def test_g_sizes_differ_only_by_crop():
    """g256 and g384 share a width; only the crop distinguishes them."""
    g256, g384 = VJEPA2_CONFIGS["g256"], VJEPA2_CONFIGS["g384"]
    assert g256["hidden_size"] == g384["hidden_size"] == 1408
    assert g256["crop_size"] != g384["crop_size"]


def test_unknown_size_rejected():
    with pytest.raises(ValueError, match="unknown V-JEPA 2 size"):
        VJEPA2Config.for_size("vitl")


def _tiny(**overrides) -> VJEPA2Config:
    params = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        mlp_ratio=4.363636363636363,
        crop_size=32,
        patch_size=16,
        tubelet_size=2,
        frames_per_clip=8,
    )
    params.update(overrides)
    return VJEPA2Config(**params)


def test_token_count_and_order():
    """(F/2) * (H/16) * (W/16) tokens, time-major flattening."""
    cfg = _tiny()
    model = LibreVJEPA2Encoder(cfg).eval()
    x = torch.randn(2, 8, 3, 32, 32)
    with torch.no_grad():
        tokens = model(x)
    # 8/2 * 32/16 * 32/16 = 4 * 2 * 2 = 16
    assert tokens.shape == (2, 16, 64)


@pytest.mark.parametrize("size,frames,expected", [("l256", 64, 8192), ("g384", 64, 18432)])
def test_released_token_counts(size, frames, expected):
    cfg = VJEPA2_CONFIGS[size]
    grid = cfg["crop_size"] // 16
    assert (frames // 2) * grid * grid == expected


def test_pooler_mlp_ratio_is_four_not_encoder_ratio():
    """The probe's MLP ratio is fixed at 4.0 even when the encoder's is not.

    Upstream builds the pooler MLP without passing ``mlp_ratio``, so it keeps
    the 4.0 default. Using the encoder's 4.3636 for a g-size probe would build
    fc1 wider than every released checkpoint provides.
    """
    cfg = _tiny()  # encoder ratio 4.3636...
    model = LibreVJEPA2Classifier(cfg, nc=5)
    assert model.encoder.layer[0].mlp.fc1.out_features == int(64 * 4.363636363636363)
    assert model.pooler.cross_attention_layer.mlp.fc1.out_features == int(64 * 4.0)


def test_pooler_cross_attention_has_no_output_projection():
    """Upstream's pooler cross-attention omits out_proj; adding one breaks load."""
    cfg = _tiny()
    model = LibreVJEPA2Classifier(cfg, nc=5)
    cross = model.pooler.cross_attention_layer.cross_attn
    assert not hasattr(cross, "out_proj")
    assert hasattr(model.pooler.self_attention_layers[0].self_attn, "out_proj")


def test_pooler_depth_is_three():
    model = LibreVJEPA2Classifier(_tiny(), nc=5)
    assert len(model.pooler.self_attention_layers) == 3


def test_classifier_logit_shape():
    model = LibreVJEPA2Classifier(_tiny(), nc=7).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 8, 3, 32, 32))
    assert logits.shape == (2, 7)


def test_temporal_order_changes_tokens():
    """A clip model that ignores frame order is a failed port."""
    torch.manual_seed(0)
    model = LibreVJEPA2Encoder(_tiny()).eval()
    x = torch.randn(1, 8, 3, 32, 32)
    with torch.no_grad():
        forward = model(x)
        backward = model(x.flip(dims=[1]))
    assert (forward - backward).abs().max().item() > 1e-4


def test_sdpa_and_eager_agree_closely():
    """Both attention paths implement the same math."""
    torch.manual_seed(0)
    x = torch.randn(1, 8, 3, 32, 32)
    sdpa = LibreVJEPA2Encoder(_tiny(attn_implementation="sdpa")).eval()
    eager = LibreVJEPA2Encoder(_tiny(attn_implementation="eager")).eval()
    eager.load_state_dict(sdpa.state_dict(), strict=True)
    with torch.no_grad():
        assert (sdpa(x) - eager(x)).abs().max().item() < 1e-5


def test_public_layout_is_batch_frames_channels():
    """Channels must be dim 2; a (B, C, F, H, W) input is a different picture."""
    model = LibreVJEPA2Encoder(_tiny()).eval()
    with torch.no_grad():
        ok = model(torch.randn(1, 8, 3, 32, 32))
    assert ok.shape[-1] == 64
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 3, 8, 32, 32))
