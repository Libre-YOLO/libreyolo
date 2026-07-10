"""PICODET training-path smoke test.

Verifies the loss + assigner stack is callable end-to-end and produces
a finite scalar loss with a non-empty backward graph. Does *not* check
mAP; that's the e2e ``test_val_coco128`` / ``test_rf1_training`` job.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.picodet.loss import PICODETLoss, SimOTAAssigner, bbox_iou_xyxy
from libreyolo.models.picodet.nn import LibrePICODETModel

pytestmark = [pytest.mark.unit, pytest.mark.picodet]


def test_loss_runs_and_backprops():
    torch.manual_seed(0)
    m = LibrePICODETModel(size="s", nb_classes=80).train()
    loss_fn = PICODETLoss(num_classes=80)

    x = torch.randn(2, 3, 320, 320)
    cs, bp = m(x)

    gts = [
        torch.tensor([[10.0, 20.0, 100.0, 150.0], [50.0, 60.0, 200.0, 220.0]]),
        torch.zeros((0, 4)),
    ]
    gtl = [
        torch.tensor([3, 17], dtype=torch.long),
        torch.zeros((0,), dtype=torch.long),
    ]

    out = loss_fn(cs, bp, gts, gtl)
    assert torch.isfinite(out["total_loss"])
    assert out["num_pos"] > 0, "SimOTA must match at least one prior to a real GT"
    out["total_loss"].backward()

    # All parameters that received gradient should have a real .grad
    has_grad = sum(1 for p in m.parameters() if p.grad is not None)
    assert has_grad > 0, "no parameters received gradient — broken graph"


def test_simota_handles_empty_gt():
    a = SimOTAAssigner()
    priors = torch.zeros((100, 4))
    priors[:, 2:] = 8.0  # stride
    decoded = torch.zeros((100, 4))
    cls_pred = torch.zeros((100, 80))
    gt_boxes = torch.zeros((0, 4))
    gt_labels = torch.zeros((0,), dtype=torch.long)
    inds, lab, ovl, pos = a.assign(priors, decoded, cls_pred, gt_boxes, gt_labels)
    assert pos.sum() == 0


def test_simota_bce_cost_disables_autocast(monkeypatch):
    """Raw probability BCE must not run inside CUDA AMP autocast.

    ``torch.amp.autocast("cuda")`` refuses to enable on CUDA-less hosts (CI),
    so set the thread-local flag directly to simulate an active CUDA autocast
    region; the guard's disabled "cuda" context must still flip it off.
    """
    from libreyolo.models.picodet import loss as loss_module

    a = SimOTAAssigner()
    priors = torch.tensor(
        [[5.0, 5.0, 8.0, 8.0], [20.0, 20.0, 8.0, 8.0]],
        dtype=torch.float32,
    )
    decoded = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [16.0, 16.0, 24.0, 24.0]],
        dtype=torch.float32,
    )
    cls_pred = torch.full((2, 80), 0.5, dtype=torch.float32)
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    gt_labels = torch.tensor([3], dtype=torch.long)
    seen = {}
    real_bce = loss_module.F.binary_cross_entropy

    def wrapped_bce(*args, **kwargs):
        seen["cuda_autocast"] = torch.is_autocast_enabled("cuda")
        return real_bce(*args, **kwargs)

    monkeypatch.setattr(loss_module.F, "binary_cross_entropy", wrapped_bce)

    torch.set_autocast_enabled("cuda", True)
    try:
        assert torch.is_autocast_enabled("cuda")
        a.assign(priors, decoded, cls_pred, gt_boxes, gt_labels)
    finally:
        torch.set_autocast_enabled("cuda", False)

    assert seen["cuda_autocast"] is False


def test_iou_pairwise_correct():
    a = torch.tensor([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=torch.float32)
    b = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    iou = bbox_iou_xyxy(a, b)
    assert iou.shape == (2, 1)
    assert abs(iou[0, 0].item() - 1.0) < 1e-6
    assert abs(iou[1, 0].item() - 25 / 100) < 1e-6
