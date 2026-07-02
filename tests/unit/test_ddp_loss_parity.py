"""DDP↔single-GPU numerical parity tests (issue #484, complaint #3).

The existing ``test_ddp_distributed.py`` smoke tests only assert that all
ranks hold *identical* parameters after a step — which is true regardless of
loss scaling, because DDP always all-reduces gradients. They do **not** check
that N-rank training produces the **same** gradient as single-GPU training on
the same global batch. That equivalence is exactly what the ``× world_size``
loss scaling used to break: every LibreYOLO loss is mean/ratio-normalized, so
DDP's built-in gradient averaging already yields single-GPU-equivalent
gradients; pre-multiplying by ``world_size`` over-counted by ~N and inflated
the effective learning rate — the root cause of the multi-GPU accuracy gap in
issue #484.

Method: build the *same* deterministic model and a fixed 2-sample global
batch. Compute the reference gradient once, single-process, over the whole
batch (``batch=2``). Then spawn 2 gloo ranks, give rank ``r`` sample ``r``
(``batch=1`` each), run forward → ``scale_loss_for_ddp`` → backward (DDP
averages the grads), and compare rank 0's synced gradient to the reference.

Correct behavior ⇒ ``‖g_ddp‖ / ‖g_ref‖ ≈ 1``. The old ``× world_size`` bug
makes the ratio ≈ ``world_size`` (≈ 2 here). BatchNorm is frozen (running
stats) so the loss is separable across the batch and the only cross-sample
coupling is the (globally reduced) classification normalizer — which is what
makes the parity exact rather than merely approximate.

Gloo/CPU only — no GPU required. ~20-40s.
"""

from __future__ import annotations

import contextlib
import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

pytestmark = pytest.mark.unit


# =============================================================================
# Shared deterministic model + batch (module-level so spawn() can pickle refs)
# =============================================================================


def _freeze_batchnorm(model: nn.Module) -> None:
    """Put every BatchNorm into eval() so it uses fixed running stats.

    In train mode BN normalizes over the batch, coupling the two samples and
    making the per-sample loss non-separable — which would make single-vs-DDP
    gradients differ for a reason unrelated to loss scaling. Freezing BN
    isolates the loss-normalization math we're actually testing. The detection
    head stays in train mode (raw outputs for the loss path).
    """
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()


def _build_yolo9():
    """Deterministic yolo9-t on CPU, train mode, BN frozen. Identical weights
    in every process because the seed and constructor are fixed."""
    from libreyolo import LibreYOLO9

    torch.manual_seed(0)
    wrapper = LibreYOLO9(None, size="t", device="cpu")
    model = wrapper.model
    model.train()
    _freeze_batchnorm(model)
    return model


def _global_batch():
    """Fixed 2-sample batch: images plus normalized [cls, x1, y1, x2, y2] boxes.

    Deterministic so sample ``r`` is byte-identical in the reference run and in
    rank ``r``'s worker. Padding rows are all-zero (a degenerate box that
    matches no anchor, contributing nothing to the loss)."""
    g = torch.Generator().manual_seed(1234)
    imgs = torch.rand(2, 3, 320, 320, generator=g)
    targets = torch.zeros(2, 8, 5)
    targets[0, 0] = torch.tensor([0.0, 0.20, 0.20, 0.50, 0.60])
    targets[0, 1] = torch.tensor([1.0, 0.55, 0.50, 0.80, 0.85])
    targets[1, 0] = torch.tensor([2.0, 0.10, 0.15, 0.45, 0.50])
    return imgs, targets


def _empty_global_batch():
    """All-background 2-sample batch: zero positive mass on every rank.

    Exercises the normalizer clamp: with clamp-after-divide DDP would train
    this batch at exactly 1/world_size gradient (measured ratio 0.5000 during
    review); clamp-before-divide keeps it exact."""
    g = torch.Generator().manual_seed(4321)
    imgs = torch.rand(2, 3, 320, 320, generator=g)
    targets = torch.zeros(2, 8, 5)
    return imgs, targets


def _flat_grad(model: nn.Module) -> torch.Tensor:
    """Concatenate all parameter gradients in named-parameter order.

    None grads become zeros so the vector layout is identical across runs
    (the same params are used in the reference and DDP runs)."""
    parts = []
    for _name, p in model.named_parameters():
        if p.grad is None:
            parts.append(torch.zeros_like(p).reshape(-1))
        else:
            parts.append(p.grad.detach().reshape(-1))
    return torch.cat(parts)


def _reference_grad(empty: bool = False) -> torch.Tensor:
    """Single-process gradient over the full 2-sample batch — the ground truth
    that DDP over 2 ranks must reproduce."""
    model = _build_yolo9()
    imgs, targets = _empty_global_batch() if empty else _global_batch()
    out = model(imgs, targets=targets)
    model.zero_grad(set_to_none=True)
    out["total_loss"].backward()
    return _flat_grad(model)


# =============================================================================
# DDP worker
# =============================================================================


def _setup_pg(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _yolo9_parity_worker(
    rank: int, world_size: int, port: int, out_dir: str, empty: bool = False
) -> None:
    """One rank: forward its own sample, DDP-averaged backward, rank 0 saves the
    synced gradient for the parent to compare against the reference."""
    out_path = Path(out_dir) / f"rank_{rank}.txt"
    try:
        _setup_pg(rank, world_size, port)

        from libreyolo.training.distributed import scale_loss_for_ddp, unwrap_model

        model = _build_yolo9()
        ddp_model = nn.parallel.DistributedDataParallel(model)

        imgs, targets = _empty_global_batch() if empty else _global_batch()
        my_img = imgs[rank : rank + 1]
        my_tgt = targets[rank : rank + 1]

        out = ddp_model(my_img, targets=my_tgt)
        loss = out["total_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss on rank {rank}: {loss.item()}")
        loss = scale_loss_for_ddp(loss)
        model.zero_grad(set_to_none=True)
        loss.backward()

        flat = _flat_grad(unwrap_model(ddp_model))
        if rank == 0:
            torch.save(flat, Path(out_dir) / "ddp_grad.pt")
        dist.barrier()
        out_path.write_text(
            f"ok grad_norm={float(flat.norm().item()):.6f}\n"
        )
    except Exception as exc:
        out_path.write_text(f"error: {type(exc).__name__}: {exc}\n")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# =============================================================================
# Helpers
# =============================================================================


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _spawn_and_check(worker, n_ranks: int, tmp_path, extra_args: tuple = ()) -> dict:
    port = _free_port()
    out_dir = str(tmp_path)
    try:
        mp.spawn(
            worker,
            args=(n_ranks, port, out_dir) + extra_args,
            nprocs=n_ranks,
            join=True,
        )
    except Exception as exc:
        outputs = {
            rank: (tmp_path / f"rank_{rank}.txt").read_text()
            if (tmp_path / f"rank_{rank}.txt").exists()
            else "<no output>"
            for rank in range(n_ranks)
        }
        raise AssertionError(
            f"spawn failed: {exc}\nper-rank outputs: {outputs}"
        ) from exc
    return {
        rank: (tmp_path / f"rank_{rank}.txt").read_text() for rank in range(n_ranks)
    }


# =============================================================================
# Test
# =============================================================================


@pytest.mark.skipif(
    sys.platform == "win32" and sys.version_info < (3, 8),
    reason="mp.spawn on Windows needs Python 3.8+",
)
def test_yolo9_ddp_gradient_matches_single_gpu(tmp_path):
    """2-rank DDP gradient must match single-GPU gradient on the same global
    batch. Fails with ratio ≈ world_size if the ``× world_size`` loss-scaling
    bug (issue #484) is present."""
    ref = _reference_grad()

    outputs = _spawn_and_check(_yolo9_parity_worker, n_ranks=2, tmp_path=tmp_path)
    for rank, text in outputs.items():
        assert text.startswith("ok "), f"rank {rank} did not finish ok: {text!r}"

    ddp = torch.load(tmp_path / "ddp_grad.pt", weights_only=False)

    ref_norm = float(ref.norm().item())
    ddp_norm = float(ddp.norm().item())
    ratio = ddp_norm / max(ref_norm, 1e-12)

    assert 0.9 < ratio < 1.1, (
        f"DDP gradient norm is {ratio:.3f}× the single-GPU gradient "
        f"(ref={ref_norm:.4f}, ddp={ddp_norm:.4f}). A ratio near world_size (2) "
        f"means the loss is being multiplied by world_size — the issue #484 "
        f"multi-GPU learning-rate bug."
    )
    # The elementwise allclose is the LOAD-BEARING assertion: a local (not
    # globally-reduced) cls_norm passes the norm-ratio band above (~0.98) but
    # fails here decisively (max_abs_diff ~1e-1). Do not loosen the tolerance.
    assert torch.allclose(ddp, ref, rtol=2e-3, atol=1e-5), (
        "DDP gradient diverged from single-GPU beyond fp tolerance: "
        f"max_abs_diff={float((ddp - ref).abs().max().item()):.3e}"
    )


@pytest.mark.skipif(
    sys.platform == "win32" and sys.version_info < (3, 8),
    reason="mp.spawn on Windows needs Python 3.8+",
)
def test_yolo9_ddp_gradient_matches_single_gpu_all_background(tmp_path):
    """Same parity check on an all-background global batch (zero positive
    mass). Pins the clamp-BEFORE-divide order in ``all_reduce_avg_scalar``:
    with clamp-after-divide, every rank's normalizer clamps to 1 instead of
    sharing global ``max(sum,1)/world_size``, and DDP trains the batch at
    exactly 1/world_size gradient (measured ratio 0.5000 during review) —
    worst exactly on the background-heavy datasets from issue #484."""
    ref = _reference_grad(empty=True)

    outputs = _spawn_and_check(
        _yolo9_parity_worker, n_ranks=2, tmp_path=tmp_path, extra_args=(True,)
    )
    for rank, text in outputs.items():
        assert text.startswith("ok "), f"rank {rank} did not finish ok: {text!r}"

    ddp = torch.load(tmp_path / "ddp_grad.pt", weights_only=False)

    ref_norm = float(ref.norm().item())
    ddp_norm = float(ddp.norm().item())
    ratio = ddp_norm / max(ref_norm, 1e-12)

    assert 0.9 < ratio < 1.1, (
        f"all-background DDP gradient is {ratio:.3f}× single-GPU "
        f"(ref={ref_norm:.4f}, ddp={ddp_norm:.4f}). A ratio near "
        f"1/world_size (0.5) means the normalizer clamps after dividing."
    )
    assert torch.allclose(ddp, ref, rtol=2e-3, atol=1e-5), (
        "all-background DDP gradient diverged from single-GPU: "
        f"max_abs_diff={float((ddp - ref).abs().max().item()):.3e}"
    )


def _helper_value_worker(rank: int, world_size: int, port: int, out_dir: str) -> None:
    """Exercise ``all_reduce_avg_scalar`` itself under a real process group —
    including the non-tensor + ``device=`` form RT-DETR uses, which no other
    test runs distributed."""
    out_path = Path(out_dir) / f"rank_{rank}.txt"
    try:
        _setup_pg(rank, world_size, port)

        from libreyolo.training.distributed import all_reduce_avg_scalar

        # Python-scalar branch (the RT-DETR num_boxes form): ranks contribute
        # 3 and 5 → both must see max(8, 1) / 2 = 4.
        got_scalar = all_reduce_avg_scalar(3.0 if rank == 0 else 5.0)
        # Tensor branch (the yolo9 cls_norm form): same values as tensors.
        got_tensor = all_reduce_avg_scalar(torch.tensor(3.0 if rank == 0 else 5.0))
        # Zero global mass: clamp the GLOBAL sum first → max(0, 1) / 2 = 0.5.
        # (Clamp-after-divide would return 1.0 and break all-background parity.)
        got_zero = all_reduce_avg_scalar(torch.tensor(0.0))

        if got_scalar != 4.0:
            raise RuntimeError(f"scalar branch: expected 4.0, got {got_scalar}")
        if got_tensor != 4.0:
            raise RuntimeError(f"tensor branch: expected 4.0, got {got_tensor}")
        if got_zero != 0.5:
            raise RuntimeError(f"zero-mass: expected 0.5, got {got_zero}")

        out_path.write_text("ok\n")
    except Exception as exc:
        out_path.write_text(f"error: {type(exc).__name__}: {exc}\n")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "win32" and sys.version_info < (3, 8),
    reason="mp.spawn on Windows needs Python 3.8+",
)
def test_all_reduce_avg_scalar_2_ranks(tmp_path):
    """Distributed values of the reduction helper: global-sum / world_size with
    the clamp applied to the global sum. Covers the python-scalar + ``device=``
    branch that RT-DETR's ``num_boxes`` uses (previously only exercised
    single-process)."""
    outputs = _spawn_and_check(_helper_value_worker, n_ranks=2, tmp_path=tmp_path)
    for rank, text in outputs.items():
        assert text.startswith("ok"), f"rank {rank} did not finish ok: {text!r}"


# =============================================================================
# Deterministic unit tests for the reduction primitives (no spawn needed).
# These pin single-GPU behavior so the DDP fix cannot regress it, and prove the
# loss is no longer multiplied by world_size — the shared mechanism every
# normalized-loss family (yolo9, YOLOX, DETR num_boxes, …) relies on.
# =============================================================================


def test_scale_loss_for_ddp_is_identity():
    """The DDP loss scaler must pass the loss through untouched (no
    ``× world_size``). Returned tensor is the same object."""
    from libreyolo.training.distributed import scale_loss_for_ddp

    loss = torch.tensor(3.14159)
    assert scale_loss_for_ddp(loss) is loss


def test_all_reduce_avg_scalar_single_process_matches_legacy_max():
    """Outside DDP, ``all_reduce_avg_scalar`` must equal the old
    ``max(sum, 1)`` normalizer exactly, so single-GPU yolo9 loss values are
    unchanged. Accepts tensor, python float, and int inputs; clamps to 1."""
    from libreyolo.training.distributed import all_reduce_avg_scalar

    assert all_reduce_avg_scalar(torch.tensor(5.0)) == 5.0
    assert all_reduce_avg_scalar(torch.tensor(0.0)) == 1.0  # clamp to min 1
    assert all_reduce_avg_scalar(0.0) == 1.0
    assert all_reduce_avg_scalar(7) == 7.0
    # A multi-element tensor is summed (matches ``targets_cls.sum()`` usage).
    assert all_reduce_avg_scalar(torch.tensor([1.0, 2.0, 3.0])) == 6.0
