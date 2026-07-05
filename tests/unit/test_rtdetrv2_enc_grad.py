"""RT-DETRv2 encoder query-selection heads must receive gradient in training.

Finding #7: the v2 decoder emits its encoder top-k predictions under
``enc_aux_outputs`` (not ``aux_outputs`` like v1), and the shared v1
``SetCriterion`` never reads that key. With the v1 loss the encoder
``enc_score_head``/``enc_bbox_head`` get **zero** gradient and stay random.
``RTDETRv2Loss`` (``RTDETRCriterionv2``) restores that supervision.

These tests build a tiny CPU decoder, run one forward + backward, and assert:
  1. the v2 criterion gives ``enc_score_head``/``enc_bbox_head`` non-None,
     non-zero gradients and a finite loss;
  2. the v1 criterion leaves those same heads with **no** gradient (proving the
     bug the fix addresses, and that the fix is what closes it).
"""

import pytest
import torch

from libreyolo.models.rtdetr.loss import RTDETRLoss
from libreyolo.models.rtdetrv2.decoder import RTDETRTransformerv2
from libreyolo.models.rtdetrv2.loss import RTDETRv2Loss

NUM_CLASSES = 4
HIDDEN_DIM = 32


def _build_tiny_decoder():
    torch.manual_seed(0)
    decoder = RTDETRTransformerv2(
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_queries=20,
        feat_channels=(HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM),
        feat_strides=(8, 16, 32),
        num_levels=3,
        num_points=4,
        nhead=2,
        num_layers=2,
        dim_feedforward=64,
        num_denoising=0,  # isolate encoder supervision from the denoising path
        eval_spatial_size=None,
    )
    decoder.train()
    return decoder


def _dummy_feats(batch_size=2):
    torch.manual_seed(1)
    # Feature maps for strides 8/16/32 at a 64px input: 8x8, 4x4, 2x2.
    return [
        torch.randn(batch_size, HIDDEN_DIM, 8, 8),
        torch.randn(batch_size, HIDDEN_DIM, 4, 4),
        torch.randn(batch_size, HIDDEN_DIM, 2, 2),
    ]


def _dummy_targets(batch_size=2):
    # Two normalized cxcywh boxes per image.
    targets = []
    for _ in range(batch_size):
        targets.append(
            {
                "labels": torch.tensor([1, 2], dtype=torch.int64),
                "boxes": torch.tensor(
                    [[0.5, 0.5, 0.4, 0.4], [0.3, 0.7, 0.2, 0.2]],
                    dtype=torch.float32,
                ),
            }
        )
    return targets


@pytest.mark.unit
def test_rtdetrv2_enc_heads_receive_gradient():
    decoder = _build_tiny_decoder()
    feats = _dummy_feats()
    targets = _dummy_targets()

    outputs = decoder(feats, targets=targets)

    # The v2 decoder must actually emit the encoder-aux supervision key.
    assert "enc_aux_outputs" in outputs
    assert "enc_meta" in outputs

    criterion = RTDETRv2Loss(num_classes=NUM_CLASSES)
    loss_dict = criterion(outputs, targets)

    total = loss_dict["total_loss"]
    assert torch.isfinite(total), f"loss not finite: {total}"

    # An encoder-aux term must be present in the breakdown.
    assert any(k.endswith("_enc_0") for k in loss_dict), sorted(loss_dict)

    decoder.zero_grad(set_to_none=True)
    total.backward()

    def _grad_is_live(module):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None for g in grads), "some grads are None"
        return any(g.abs().sum().item() > 0 for g in grads)

    assert _grad_is_live(decoder.enc_score_head), "enc_score_head got no gradient"
    assert _grad_is_live(decoder.enc_bbox_head), "enc_bbox_head got no gradient"


@pytest.mark.unit
def test_v1_criterion_leaves_enc_heads_unsupervised():
    """Regression guard: the v1 loss ignores enc_aux_outputs (the bug)."""
    decoder = _build_tiny_decoder()
    feats = _dummy_feats()
    targets = _dummy_targets()

    outputs = decoder(feats, targets=targets)

    criterion = RTDETRLoss(num_classes=NUM_CLASSES)
    loss_dict = criterion(outputs, targets)

    # v1 emits no encoder-aux term.
    assert not any(k.endswith("_enc_0") for k in loss_dict)

    decoder.zero_grad(set_to_none=True)
    loss_dict["total_loss"].backward()

    # enc_score_head has no non-detached path to the loss under the v1 criterion,
    # so its gradients stay None.
    enc_score_grads = [p.grad for p in decoder.enc_score_head.parameters()]
    assert all(g is None for g in enc_score_grads), (
        "v1 criterion unexpectedly produced enc_score_head gradients"
    )
