"""Unit tests for the LibrePAGE gaze-target family."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

# The page family's decoder blocks build on timm layers, imported at module
# scope below; slim CI environments (e.g. the distributed job) lack timm.
pytest.importorskip("timm", reason="LibrePAGE decoder blocks build on timm layers")

from libreyolo.models.page.convert import convert_upstream, is_upstream_state_dict
from libreyolo.models.page.model import LibrePAGE
from libreyolo.models.page.utils import (
    decode_heatmaps,
    head_rects_grid,
    pil_to_tensor,
)
from libreyolo.utils.results import Boxes, GazeTargets, Results

pytestmark = pytest.mark.unit


def _page_signature_dict() -> dict:
    return {
        "scene_head_interaction_layers.0.cross_attn_scene.attn.q.weight": torch.zeros(256, 256),
        "heatmap_head.0.weight": torch.zeros(256, 256, 2, 2),
        "scene_branch_backbone.model.embeddings.patch_embeddings.weight": torch.zeros(384, 3, 16, 16),
    }


# =========================================================================
# GazeTargets payload
# =========================================================================


def test_gazetargets_shapes_and_props():
    data = torch.tensor([[100.0, 50.0, 0.9], [10.0, 20.0, 0.2]])
    hm = torch.rand(2, 64, 64)
    gt = GazeTargets(data, hm, orig_shape=(480, 640))
    assert len(gt) == 2
    assert gt.xy.shape == (2, 2)
    assert gt.inout.tolist() == pytest.approx([0.9, 0.2])
    xyn = gt.xyn
    assert float(xyn[0, 0]) == pytest.approx(100.0 / 640)
    assert float(xyn[0, 1]) == pytest.approx(50.0 / 480)


def test_gazetargets_slicing_carries_heatmaps():
    data = torch.tensor([[1.0, 2.0, 0.5], [3.0, 4.0, 0.6], [5.0, 6.0, 0.7]])
    hm = torch.rand(3, 64, 64)
    gt = GazeTargets(data, hm, orig_shape=(100, 100))
    sub = gt[torch.tensor([0, 2])]
    assert len(sub) == 2
    assert sub.heatmaps.shape == (2, 64, 64)
    assert torch.equal(sub.heatmaps[1], hm[2])
    np_gt = gt.numpy()
    assert isinstance(np_gt.data, np.ndarray)
    assert isinstance(np_gt.heatmaps, np.ndarray)


def test_gazetargets_rejects_bad_shapes():
    with pytest.raises(ValueError):
        GazeTargets(torch.zeros(2, 2))
    with pytest.raises(ValueError):
        GazeTargets(torch.zeros(2, 3), torch.zeros(1, 64, 64))


def test_results_select_slices_gazetarget_with_boxes():
    boxes = Boxes(
        torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 20.0]]),
        torch.tensor([0.9, 0.8]),
        torch.tensor([0.0, 0.0]),
    )
    gt = GazeTargets(
        torch.tensor([[50.0, 60.0, 0.9], [70.0, 80.0, 0.1]]),
        torch.rand(2, 64, 64),
    )
    r = Results(boxes=boxes, orig_shape=(100, 100), gazetarget=gt, names={0: "person"})
    r0 = r[0]
    assert len(r0.boxes) == 1
    assert len(r0.gazetarget) == 1
    assert float(r0.gazetarget.data[0, 0]) == 50.0
    row = r.summary()[0]
    assert row["gaze_target"]["in_frame"] == pytest.approx(0.9)


# =========================================================================
# Preprocessing / decoding helpers
# =========================================================================


def test_decode_heatmaps_known_answer():
    hm = torch.zeros(1, 64, 64)
    hm[0, 10, 20] = 1.0
    pts = decode_heatmaps(hm, orig_w=640, orig_h=480)
    assert float(pts[0, 0]) == pytest.approx((20 + 0.5) / 64 * 640)
    assert float(pts[0, 1]) == pytest.approx((10 + 0.5) / 64 * 480)


def test_head_rects_grid_matches_upstream_rounding():
    rects = head_rects_grid([(0.30, 0.12, 0.48, 0.40)])
    # upstream: [round(ymin*32), round(xmin*32), round(ymax*32), round(xmax*32)]
    assert rects.tolist() == [[round(0.12 * 32), round(0.30 * 32), round(0.40 * 32), round(0.48 * 32)]]


def test_pil_to_tensor_normalization():
    img = Image.new("RGB", (100, 80), (124, 116, 104))  # ~ImageNet mean
    t = pil_to_tensor(img, (256, 256))
    assert t.shape == (3, 256, 256)
    assert float(t.abs().mean()) < 0.05


# =========================================================================
# Upstream conversion
# =========================================================================


def test_convert_upstream_flattens_nested_layers():
    raw = {
        "scene_branch_backbone.model.model.layer.0.attention.q_proj.weight": torch.zeros(2, 2),
        "scene_branch_backbone.model.embeddings.cls_token": torch.zeros(1, 1, 2),
        "scene_head_interaction_layers.0.cross_attn_scene.attn.q.weight": torch.zeros(2, 2),
    }
    assert is_upstream_state_dict(raw)
    flat = convert_upstream(raw)
    assert "scene_branch_backbone.model.layer.0.attention.q_proj.weight" in flat
    assert not is_upstream_state_dict(flat)


def test_is_upstream_rejects_canonical_and_foreign_dicts():
    assert not is_upstream_state_dict(_page_signature_dict())
    assert not is_upstream_state_dict({"backbone.stem.weight": torch.zeros(1)})


# =========================================================================
# Family recognition
# =========================================================================


def test_filename_detection_all_sizes():
    for size in ("s", "sp", "b", "hp"):
        assert LibrePAGE.detect_size_from_filename(f"LibrePAGE{size}-gazetarget.pt") == size
        assert LibrePAGE.detect_size_from_filename(f"LibrePAGE{size}.pt") == size
    assert LibrePAGE.detect_size_from_filename("LibrePAGEx.pt") is None
    assert LibrePAGE.detect_size_from_filename("LibreYOLOXs.pt") is None


def test_download_url_includes_task_suffix():
    url = LibrePAGE.get_download_url("LibrePAGEsp-gazetarget.pt")
    assert url == (
        "https://huggingface.co/LibreYOLO/LibrePAGEsp-gazetarget/"
        "resolve/main/LibrePAGEsp-gazetarget.pt"
    )


def test_can_load_signature():
    assert LibrePAGE.can_load(_page_signature_dict())
    assert not LibrePAGE.can_load({"fc_yaw_gaze.weight": torch.zeros(90, 512)})
    assert not LibrePAGE.can_load({"backbone.stem.conv.weight": torch.zeros(8, 3, 3, 3)})


def test_can_load_rejection_is_bidirectional_with_l2cs():
    from libreyolo.models.l2cs.model import LibreL2CS

    assert not LibreL2CS.can_load(_page_signature_dict())
    l2cs_sig = {
        "fc_yaw_gaze.weight": torch.zeros(90, 512),
        "fc_pitch_gaze.weight": torch.zeros(90, 512),
    }
    assert not LibrePAGE.can_load(l2cs_sig)


def test_detect_size_discriminates_gated_towers():
    from libreyolo.models.page.nn import detect_size_from_state_dict

    base = {
        "scene_branch_backbone.model.embeddings.patch_embeddings.weight": torch.zeros(384, 3, 16, 16)
    }
    assert detect_size_from_state_dict(base) == "s"
    gated = dict(base)
    gated["scene_branch_backbone.model.layer.0.mlp.gate_proj.weight"] = torch.zeros(1)
    assert detect_size_from_state_dict(gated) == "sp"
    big = {
        "scene_branch_backbone.model.embeddings.patch_embeddings.weight": torch.zeros(768, 3, 16, 16)
    }
    assert detect_size_from_state_dict(big) == "b"


# =========================================================================
# Drawing
# =========================================================================


def test_draw_gaze_targets_smoke():
    from libreyolo.utils.drawing import draw_gaze_targets

    img = Image.new("RGB", (200, 150), (30, 30, 30))
    out = draw_gaze_targets(
        img,
        boxes=[(10, 10, 50, 60)],
        targets=[(150.0, 100.0)],
        inout=[0.92],
        heatmaps=np.random.rand(1, 64, 64).astype(np.float32) * 0.5,
    )
    assert out.size == img.size
    assert np.asarray(out).sum() != np.asarray(img).sum()


# =========================================================================
# Model build (needs transformers)
# =========================================================================


def test_model_builds_and_runs():
    pytest.importorskip("transformers")
    from libreyolo.models.page.nn import LibrePAGEModel

    m = LibrePAGEModel("s").eval()
    with torch.no_grad():
        hm, io = m(
            torch.zeros(1, 3, 512, 512),
            torch.zeros(2, 3, 256, 256),
            torch.tensor([[8.0, 8.0, 16.0, 16.0], [4.0, 20.0, 10.0, 28.0]]),
        )
    assert hm.shape == (2, 64, 64)
    assert io.shape == (2,)
