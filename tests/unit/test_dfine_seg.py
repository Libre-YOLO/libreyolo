"""Unit tests for D-FINE instance-segmentation support."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from libreyolo.models.dfine.loss import DFINECriterion
from libreyolo.models.dfine.model import LibreDFINE
from libreyolo.models.dfine.nn import DFINEExportWrapper, LibreDFINEModel
from libreyolo.postprocess.dfine import postprocess_seg

pytestmark = pytest.mark.unit


def test_dfine_seg_suffix_and_state_dict_task_detection():
    assert LibreDFINE.detect_task_from_filename("LibreDFINEn-seg.pt") == "segment"
    assert LibreDFINE.detect_task_from_filename("LibreDFINEs.pt") is None
    assert (
        LibreDFINE.detect_checkpoint_task(
            {"decoder.mask_head.layers.0.weight": torch.zeros(1)}
        )
        == "segment"
    )


def test_factory_detects_dfine_segment_state_dict(tmp_path):
    from libreyolo import LibreYOLO

    model = LibreDFINEModel(
        "n",
        nb_classes=2,
        eval_spatial_size=(640, 640),
        enable_mask_head=True,
    )
    state = model.state_dict()
    partial = {
        "encoder.lateral_convs.0.conv.weight": state[
            "encoder.lateral_convs.0.conv.weight"
        ],
        "decoder.dec_score_head.0.bias": state["decoder.dec_score_head.0.bias"],
        "decoder.pre_bbox_head.layers.0.weight": state[
            "decoder.pre_bbox_head.layers.0.weight"
        ],
        "decoder.mask_head.layers.0.weight": state[
            "decoder.mask_head.layers.0.weight"
        ],
    }
    for key, value in state.items():
        if key.startswith("encoder.input_proj.") and "conv.weight" in key:
            partial[key] = value

    path = tmp_path / "LibreDFINEn.pt"
    torch.save(partial, path)

    loaded = LibreYOLO(str(path), device="cpu")

    assert isinstance(loaded, LibreDFINE)
    assert loaded.task == "segment"
    assert hasattr(loaded.model.decoder, "mask_head")


def test_dfine_segment_forward_shapes_nano():
    model = LibreDFINEModel(
        "n",
        nb_classes=2,
        eval_spatial_size=(256, 256),
        enable_mask_head=True,
    )
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))

    assert out["pred_logits"].shape == (1, 300, 2)
    assert out["pred_boxes"].shape == (1, 300, 4)
    assert out["pred_masks"].shape == (1, 300, 64, 64)
    assert out["pred_masks"].min() >= 0
    assert out["pred_masks"].max() <= 1


def test_dfine_segment_export_wrapper_returns_three_tensors():
    model = LibreDFINEModel(
        "n",
        nb_classes=2,
        eval_spatial_size=(256, 256),
        enable_mask_head=True,
    )
    wrapper = DFINEExportWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        out = wrapper(torch.randn(1, 3, 256, 256))

    assert isinstance(out, tuple) and len(out) == 3
    pred_logits, pred_boxes, pred_masks = out
    assert pred_logits.shape == (1, 300, 2)
    assert pred_boxes.shape == (1, 300, 4)
    assert pred_masks.shape == (1, 300, 64, 64)


def test_dfine_segment_postprocess_emits_cropped_masks():
    outputs = {
        "pred_logits": torch.tensor([[[10.0], [-10.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]]]),
        "pred_masks": torch.ones(1, 2, 4, 4),
    }

    det = postprocess_seg(
        outputs,
        conf_thres=0.5,
        iou_thres=0.0,
        original_size=(8, 8),
        max_det=2,
    )

    assert det["num_detections"] == 1
    assert det["masks"].shape == (1, 8, 8)
    assert det["masks"].dtype == torch.bool
    assert det["masks"][0, 4, 4]
    assert not det["masks"][0, 0, 0]


def test_dfine_segment_postprocess_matches_validation_mask_resize_path():
    mask = torch.tensor(
        [
            [
                [
                    [0.4963, 0.7682, 0.0885],
                    [0.1320, 0.3074, 0.6341],
                    [0.4901, 0.8964, 0.4556],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    outputs = {
        "pred_logits": torch.tensor([[[10.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 1.0, 1.0]]]),
        "pred_masks": mask,
    }

    det = postprocess_seg(
        outputs,
        conf_thres=0.5,
        original_size=(5, 7),
        input_size=6,
        max_det=1,
    )

    two_step = F.interpolate(
        F.interpolate(mask, size=(6, 6), mode="bilinear", align_corners=False),
        size=(7, 5),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    direct = F.interpolate(
        mask,
        size=(7, 5),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    assert not torch.equal(two_step >= 0.5, direct >= 0.5)
    assert torch.equal(det["masks"][0], two_step >= 0.5)


def test_upstream_seg_filename_size_detection():
    """ArgoSA release filenames (dfine_seg_n_coco.pt) must resolve a size so
    the D-FINE/DEIM tensor-tie disambiguation can claim them."""
    assert LibreDFINE.detect_size_from_filename("dfine_seg_n_coco.pt") == "n"
    assert LibreDFINE.detect_size_from_filename("dfine_seg_x_coco.pt") == "x"
    assert LibreDFINE.detect_size_from_filename("deim_hgnetv2_n_coco.pth") is None


def test_rtdetrv4_stays_detect_only():
    """RT-DETRv4 inherits from LibreDFINE but has no mask head."""
    from libreyolo.models.rtdetrv4.model import LibreRTDETRv4

    assert LibreRTDETRv4.SUPPORTED_TASKS == ("detect",)
    with pytest.raises(ValueError, match="not supported"):
        LibreRTDETRv4(None, size="s", nb_classes=2, device="cpu", task="segment")


def _detect_checkpoint(tmp_path):
    det = LibreDFINEModel("n", nb_classes=2, eval_spatial_size=(640, 640))
    ckpt = {
        "model": det.state_dict(),
        "model_family": "dfine",
        "size": "n",
        "nc": 2,
        "task": "detect",
        "metadata_version": "1.0",
    }
    path = tmp_path / "LibreDFINEn.pt"
    torch.save(ckpt, path)
    return path


def test_detect_checkpoint_as_segment_requires_transfer_flag(tmp_path):
    path = _detect_checkpoint(tmp_path)

    with pytest.raises(RuntimeError, match="no mask head"):
        LibreDFINE(str(path), size="n", nb_classes=2, device="cpu", task="segment")

    model = LibreDFINE(
        str(path),
        size="n",
        nb_classes=2,
        device="cpu",
        task="segment",
        allow_detect_to_segment_transfer=True,
    )
    assert model.task == "segment"
    assert hasattr(model.model.decoder, "mask_head")


def test_factory_rejects_detect_checkpoint_as_segment(tmp_path):
    from libreyolo import LibreYOLO

    path = _detect_checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="no mask head"):
        LibreYOLO(str(path), device="cpu", task="segment")


def test_dfine_mask_loss_no_match_zero_stays_connected():
    criterion = DFINECriterion(
        matcher=None,
        weight_dict={},
        losses=["masks"],
        num_classes=1,
    )
    pred_masks = torch.randn(2, 5, 8, 8, requires_grad=True)
    targets = [
        {
            "labels": torch.zeros(0, dtype=torch.long),
            "boxes": torch.zeros(0, 4),
            "masks": torch.zeros(0, 8, 8, dtype=torch.bool),
        }
        for _ in range(2)
    ]
    indices = [
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        for _ in targets
    ]

    losses = criterion.loss_masks({"pred_masks": pred_masks}, targets, indices, 1.0)
    loss = losses["loss_mask_bce"] + losses["loss_mask_dice"]
    loss.backward()

    assert loss.item() == 0.0
    assert pred_masks.grad is not None
    assert pred_masks.grad.abs().sum().item() == 0.0
