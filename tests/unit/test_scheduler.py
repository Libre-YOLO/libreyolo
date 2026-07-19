"""Unit tests for libreyolo.training.scheduler.

Focuses on edge cases: zero warmup, LR monotonicity, boundary values.
"""

from __future__ import annotations

import pytest

from libreyolo.training.scheduler import (
    ConstantLRScheduler,
    CosineAnnealingScheduler,
    FlatCosineScheduler,
    LinearLRScheduler,
    WarmupCosineScheduler,
)

pytestmark = pytest.mark.unit


# =============================================================================
# WarmupCosineScheduler — warmup_epochs=0 regression (ZeroDivisionError fix)
# =============================================================================


def test_warmup_cosine_zero_warmup_does_not_raise_at_iter_zero():
    """warmup_epochs=0: update_lr(0) must not divide by zero."""
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=100, total_epochs=100, warmup_epochs=0
    )
    lr = sched.update_lr(0)
    assert isinstance(lr, float)
    assert 0.0 <= lr <= 0.01


def test_warmup_cosine_zero_warmup_full_lr_at_iter_zero():
    """With no warmup and no plateau, iter=0 is the peak of the cosine → lr."""
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=100, total_epochs=100, warmup_epochs=0, plateau_epochs=0
    )
    assert sched.update_lr(0) == pytest.approx(0.01, abs=1e-9)


def test_warmup_cosine_zero_warmup_monotone_decay():
    """With no warmup the LR must be non-increasing for every iteration."""
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=10, total_epochs=10, warmup_epochs=0, plateau_epochs=0
    )
    lrs = [sched.update_lr(i) for i in range(100)]
    for prev, curr in zip(lrs, lrs[1:]):
        assert prev >= curr - 1e-10, f"LR increased: {prev} → {curr}"


# =============================================================================
# WarmupCosineScheduler — normal warmup sanity checks
# =============================================================================


def test_warmup_cosine_starts_at_warmup_lr_start():
    """iter=0 with warmup active must equal warmup_lr_start (quadratic: 0^2 = 0)."""
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=100, total_epochs=100, warmup_epochs=5, warmup_lr_start=0.0
    )
    assert sched.update_lr(0) == pytest.approx(0.0, abs=1e-9)


def test_warmup_cosine_peaks_at_end_of_warmup():
    """At the last warmup iter the LR should equal lr (quadratic reaches 1.0)."""
    iters_per_epoch = 100
    warmup_epochs = 5
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=iters_per_epoch, total_epochs=100, warmup_epochs=warmup_epochs
    )
    warmup_iters = iters_per_epoch * warmup_epochs
    assert sched.update_lr(warmup_iters) == pytest.approx(0.01, abs=1e-9)


def test_warmup_cosine_plateau_is_min_lr():
    """Iterations past the plateau boundary must return min_lr."""
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=100, total_epochs=100, warmup_epochs=0,
        plateau_epochs=10, min_lr_ratio=0.05
    )
    min_lr = 0.01 * 0.05
    total_iters = 100 * 100
    plateau_iters = 100 * 10
    # Last iteration is firmly in plateau
    assert sched.update_lr(total_iters - plateau_iters) == pytest.approx(min_lr, abs=1e-9)
    assert sched.update_lr(total_iters) == pytest.approx(min_lr, abs=1e-9)


def test_warmup_cosine_cosine_midpoint():
    """At the midpoint of cosine annealing the LR should be midway between lr and min_lr."""
    iters_per_epoch = 100
    total_epochs = 100
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=iters_per_epoch, total_epochs=total_epochs,
        warmup_epochs=0, plateau_epochs=0, min_lr_ratio=0.0
    )
    mid = (iters_per_epoch * total_epochs) // 2
    # cos(pi * 0.5) = 0 → lr = min_lr + 0.5*(lr - min_lr)*(1+0) = lr/2
    assert sched.update_lr(mid) == pytest.approx(0.005, abs=1e-3)


@pytest.mark.parametrize(
    "scheduler_cls,kwargs",
    [
        (WarmupCosineScheduler, {"plateau_epochs": 0}),
        (LinearLRScheduler, {}),
        (ConstantLRScheduler, {}),
        (FlatCosineScheduler, {"no_aug_epochs": 2}),
        (CosineAnnealingScheduler, {}),
    ],
)
def test_warmup_start_is_clamped_to_target_lr_for_low_lr_finetunes(
    scheduler_cls,
    kwargs,
):
    sched = scheduler_cls(
        lr=1e-5,
        iters_per_epoch=10,
        total_epochs=10,
        warmup_epochs=2,
        warmup_lr_start=1e-4,
        **kwargs,
    )

    warmup_lrs = [sched.update_lr(i) for i in range(0, 21)]

    assert max(warmup_lrs) <= 1e-5 + 1e-12
    assert sched.update_lr(0) == pytest.approx(1e-5)


def test_flat_cosine_tail_cannot_extend_into_warmup_on_short_runs():
    """A run shorter than the configured no-aug tail must still warm up
    first and end at min lr, not cosine-decay from iteration zero."""
    sched = FlatCosineScheduler(
        lr=1e-4, iters_per_epoch=10, total_epochs=8,
        warmup_epochs=2, no_aug_epochs=12, min_lr_ratio=0.05,
    )
    # cosine window is clamped to the post-warmup budget
    assert sched.cosine_iters == 8 * 10 - 2 * 10
    # end of run sits at the floor
    assert sched.update_lr(80) == pytest.approx(1e-4 * 0.05, rel=1e-6)
    # warmup still ramps up from the start value
    assert sched.update_lr(1) < 1e-4


def test_deimv2_flat_cosine_clamps_recipe_phases_to_short_budgets():
    """DEIMv2 recipes use iteration-based warmup (2000) and epoch-based flat
    (64) sized for COCO; a 15-epoch small-dataset run must not spend its
    whole budget warming up."""
    from libreyolo.models.deimv2.trainer import DEIMv2FlatCosineScheduler

    ips, total = 56, 15
    sched = DEIMv2FlatCosineScheduler(
        lr=5e-4, iters_per_epoch=ips, total_epochs=total,
        warmup_iters=2000, flat_epochs=64, no_aug_epochs=12, min_lr_ratio=0.5,
    )
    budget = ips * total
    assert sched.warmup_iters <= int(0.1 * budget)
    assert sched.no_aug_iters <= int(0.2 * budget)
    assert sched.flat_iters < budget - sched.no_aug_iters
    # base LR is actually reached during the flat phase
    mid_flat = (sched.warmup_iters + sched.flat_iters) // 2
    assert sched.update_lr(mid_flat) == pytest.approx(5e-4, rel=1e-6)
    # and the run ends at the floor
    assert sched.update_lr(budget) == pytest.approx(5e-4 * 0.5, rel=1e-6)


def test_deimv2_flat_cosine_leaves_coco_scale_recipes_untouched():
    """None of the budget guards may alter a recipe that fits its budget."""
    from libreyolo.models.deimv2.trainer import DEIMv2FlatCosineScheduler

    ips, total = 3600, 132
    sched = DEIMv2FlatCosineScheduler(
        lr=5e-4, iters_per_epoch=ips, total_epochs=total,
        warmup_iters=2000, flat_epochs=64, no_aug_epochs=12, min_lr_ratio=0.5,
    )
    assert sched.warmup_iters == 2000
    assert sched.flat_iters == 64 * ips
    assert sched.no_aug_iters == 12 * ips


def test_aug_stop_epoch_leaves_no_aug_tail_clean_on_short_runs():
    from libreyolo.data.augment.detr import resolve_aug_stop_epoch

    # Published recipes are unchanged (equality for DEIM/DEIMv2, earlier for
    # D-FINE's independent 0.85 ratio).
    assert resolve_aug_stop_epoch(132, 0.91, 12) == 120
    assert resolve_aug_stop_epoch(500, 468 / 500, 32) == 468
    assert resolve_aug_stop_epoch(72, 0.85, 4) == 61
    # Short fine-tunes stop strong augs before the no-aug LR tail begins.
    assert resolve_aug_stop_epoch(15, 0.91, 12) == 3
    assert resolve_aug_stop_epoch(8, 0.91, 12) == 1
    # No tail configured: pure ratio semantics.
    assert resolve_aug_stop_epoch(100, 0.85, 0) == 85
    # ratio=None (a config whose size recipe did not set it) must not crash;
    # treated as 1.0, then clamped by the no-aug tail.
    assert resolve_aug_stop_epoch(60, None, 12) == 48
