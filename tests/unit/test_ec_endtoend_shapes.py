"""Random-weight forward-pass shape tests for EC (backbone + encoder + decoder)."""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.ec.backbone import ViTAdapter
from libreyolo.models.ec.decoder import ECTransformer
from libreyolo.models.ec.encoder import HybridEncoder
from libreyolo.models.ec.nn import SIZE_CONFIGS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("size", ["s", "m", "l", "x"])
def test_full_pipeline_random_weights(size):
    cfg = SIZE_CONFIGS[size]

    backbone = ViTAdapter(
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        proj_dim=cfg["proj_dim"],
        ffn_ratio=cfg.get("ffn_ratio", 4.0),
        interaction_indexes=(10, 11),
    )
    encoder = HybridEncoder(
        in_channels=cfg["enc_in_channels"],
        hidden_dim=cfg["enc_hidden_dim"],
        dim_feedforward=cfg["enc_dim_feedforward"],
        expansion=cfg["enc_expansion"],
        depth_mult=cfg["enc_depth_mult"],
        eval_spatial_size=[640, 640],
    )
    decoder = ECTransformer(
        num_classes=80,
        hidden_dim=cfg["dec_hidden_dim"],
        feat_channels=cfg["dec_feat_channels"],
        dim_feedforward=cfg["dec_dim_feedforward"],
        num_layers=4,
        num_points=(3, 6, 3),
        eval_idx=-1,
        reg_max=32,
        reg_scale=4.0,
        eval_spatial_size=[640, 640],
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


def _load_twin(fixed, cls, **kwargs):
    from libreyolo.models.ec import nn as ec_nn

    dynamic = getattr(ec_nn, cls)(config="s", eval_spatial_size=None, **kwargs)
    dynamic.load_state_dict(fixed.state_dict(), strict=False)
    return dynamic.eval()


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        ("LibreECModel", {"nb_classes": 2}),
        ("LibreECSegModel", {"nb_classes": 2}),
        ("LibreECPoseModel", {}),
    ],
)
def test_eval_forward_accepts_runtime_size_different_from_eval_spatial_size(
    cls, kwargs
):
    """Eval must rebuild fixed-size encoder/decoder constants for runtime imgsz."""
    from libreyolo.models.ec import nn as ec_nn

    model = getattr(ec_nn, cls)(
        config="s", eval_spatial_size=(640, 640), **kwargs
    ).eval()

    for hw in ((320, 320), (800, 800)):
        x = torch.randn(1, 3, *hw)
        with torch.no_grad():
            model(x)


def test_eval_rectangular_size_with_native_token_count_matches_dynamic():
    """A rect grid can share the native buffer's token count (16x25 == 20x20).

    Constants must be rebuilt from the grid shape, not the token count, or
    the encoder/decoder silently apply wrong-grid embeddings and anchors.
    """
    from libreyolo.models.ec.nn import LibreECModel

    fixed = LibreECModel(
        config="s", nb_classes=2, eval_spatial_size=(640, 640)
    ).eval()
    dynamic = _load_twin(fixed, "LibreECModel", nb_classes=2)

    x = torch.randn(1, 3, 512, 800)
    with torch.no_grad():
        out_fixed = fixed(x)
        out_dynamic = dynamic(x)
    for key in ("pred_logits", "pred_boxes"):
        torch.testing.assert_close(
            out_fixed[key], out_dynamic[key], rtol=1e-4, atol=1e-5
        )
