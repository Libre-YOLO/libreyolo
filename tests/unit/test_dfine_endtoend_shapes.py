"""Random-weight forward-pass shape tests for backbone + encoder + decoder.

This exercises the full LibreYOLO-ported pipeline without loading any
checkpoint. Purpose: catch shape/wiring mistakes before we attempt parity
against a real D-FINE checkpoint.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.dfine.backbone import HGNetv2
from libreyolo.models.dfine.decoder import (
    DFINETransformer,
    EVAL_CONSTANT_CACHE_LIMIT as DECODER_CACHE_LIMIT,
)
from libreyolo.models.dfine.encoder import (
    HybridEncoder,
    EVAL_CONSTANT_CACHE_LIMIT as ENCODER_CACHE_LIMIT,
)
from libreyolo.models.dfine.nn import LibreDFINEModel, SIZE_CONFIGS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("size", ["n", "s", "x"])
def test_full_pipeline_random_weights(size):
    """Build backbone -> encoder -> decoder, run one batch, check output shapes."""
    cfg = SIZE_CONFIGS[size]

    backbone = HGNetv2(
        name=cfg["backbone"],
        use_lab=cfg["use_lab"],
        return_idx=cfg["return_idx"],
        freeze_at=-1,
        freeze_norm=False,
    )
    encoder = HybridEncoder(
        in_channels=cfg["enc_in_channels"],
        feat_strides=cfg["enc_feat_strides"],
        hidden_dim=cfg["enc_hidden_dim"],
        dim_feedforward=cfg["enc_dim_feedforward"],
        expansion=cfg["enc_expansion"],
        depth_mult=cfg["enc_depth_mult"],
        use_encoder_idx=cfg["enc_use_encoder_idx"],
        eval_spatial_size=(640, 640),
    )
    decoder = DFINETransformer(
        num_classes=80,
        hidden_dim=cfg["dec_hidden_dim"],
        num_queries=300,
        feat_channels=cfg["dec_feat_channels"],
        feat_strides=cfg["dec_feat_strides"],
        num_levels=cfg["dec_num_levels"],
        num_points=cfg["dec_num_points"],
        num_layers=cfg["dec_num_layers"],
        dim_feedforward=cfg["dec_dim_feedforward"],
        eval_spatial_size=(640, 640),
        eval_idx=cfg["dec_eval_idx"],
        reg_scale=cfg["reg_scale"],
    )

    backbone.eval()
    encoder.eval()
    decoder.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        feats = backbone(x)
        enc_feats = encoder(feats)
        out = decoder(enc_feats)

    assert set(out.keys()) == {"pred_logits", "pred_boxes"}
    assert out["pred_logits"].shape == (1, 300, 80)
    assert out["pred_boxes"].shape == (1, 300, 4)
    # Boxes should be in [0, 1] after sigmoid.
    assert (out["pred_boxes"] >= 0).all() and (out["pred_boxes"] <= 1).all()


def test_eval_forward_accepts_runtime_size_different_from_eval_spatial_size():
    """Eval must rebuild fixed-size encoder/decoder constants for runtime imgsz."""
    model = LibreDFINEModel(
        "n", nb_classes=2, eval_spatial_size=(640, 640)
    ).eval()

    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        out = model(x)

    assert set(out.keys()) == {"pred_logits", "pred_boxes"}
    assert out["pred_logits"].shape == (1, 300, 2)
    assert out["pred_boxes"].shape == (1, 300, 4)


def test_eval_rectangular_size_with_native_token_count_matches_dynamic():
    """A rect grid can share the native buffer's token count (16x25 == 20x20).

    The pos-embed must be rebuilt from the grid shape, not the token count,
    or the encoder silently applies a wrong-grid embedding.
    """
    fixed = LibreDFINEModel("n", nb_classes=2, eval_spatial_size=(640, 640)).eval()
    dynamic = LibreDFINEModel("n", nb_classes=2, eval_spatial_size=None).eval()
    dynamic.load_state_dict(fixed.state_dict(), strict=False)

    for hw in ((512, 800), (320, 1280)):
        x = torch.randn(1, 3, *hw)
        with torch.no_grad():
            out_fixed = fixed(x)
            out_dynamic = dynamic(x)
        for key in ("pred_logits", "pred_boxes"):
            torch.testing.assert_close(
                out_fixed[key], out_dynamic[key], rtol=1e-4, atol=1e-5
            )


def test_eval_pos_embed_cache_is_bounded_and_reuses_moved_tensor():
    encoder = HybridEncoder(
        in_channels=(16,),
        feat_strides=(16,),
        hidden_dim=16,
        nhead=4,
        dim_feedforward=32,
        use_encoder_idx=(0,),
        eval_spatial_size=(64, 64),
    )
    src_flatten = torch.empty(1, 25, 16)

    first = encoder._get_pos_embed(0, 5, 5, src_flatten)
    second = encoder._get_pos_embed(0, 5, 5, src_flatten)

    assert first.data_ptr() == second.data_ptr()

    for size in range(6, 6 + ENCODER_CACHE_LIMIT + 4):
        src_flatten = torch.empty(1, size * size, 16)
        encoder._get_pos_embed(0, size, size, src_flatten)

    assert len(encoder._pos_embed_cache) <= ENCODER_CACHE_LIMIT


def test_eval_anchor_cache_is_bounded_and_reuses_moved_tensors():
    decoder = DFINETransformer(
        num_classes=2,
        hidden_dim=16,
        num_queries=4,
        feat_channels=(16,),
        feat_strides=(16,),
        num_levels=1,
        num_points=2,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        eval_spatial_size=(64, 64),
        reg_max=4,
    )
    memory = torch.empty(1, 25, 16)

    first_anchors, first_mask = decoder._get_anchors_for_spatial_shapes(
        [[5, 5]], memory
    )
    second_anchors, second_mask = decoder._get_anchors_for_spatial_shapes(
        [[5, 5]], memory
    )

    assert first_anchors.data_ptr() == second_anchors.data_ptr()
    assert first_mask.data_ptr() == second_mask.data_ptr()

    for size in range(6, 6 + DECODER_CACHE_LIMIT + 4):
        memory = torch.empty(1, size * size, 16)
        decoder._get_anchors_for_spatial_shapes([[size, size]], memory)

    assert len(decoder._anchor_cache) <= DECODER_CACHE_LIMIT
