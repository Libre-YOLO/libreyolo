"""Distributed training utilities for LibreYOLO.

Thin helpers around ``torch.distributed`` so the rest of the trainer can stay
backend-agnostic. All helpers degrade to no-ops when distributed is not
initialised — single-GPU code paths continue to work unchanged.

User-facing surface accepts ``device=[0, 1]`` (or ``device="0,1"``) and
launches with ``torchrun --nproc_per_node=N``. Inside each child process
``init_distributed()`` is called by the trainer; outside DDP everything is a
no-op.
"""

from __future__ import annotations

import os
import pickle
import socket
import traceback
from datetime import timedelta
from typing import Callable, List, Optional, Tuple, TypeVar, Union

import torch
import torch.distributed as dist
import torch.nn as nn

DeviceArg = Union[str, int, List[int], None]
PhaseResult = TypeVar("PhaseResult")


class RankZeroPhaseError(RuntimeError):
    """A rank-zero-only phase failed and the failure was shared with all ranks.

    ``root_exception_type``, ``root_exception_message``, and
    ``rank_zero_traceback`` describe the original exception on rank 0. On rank
    0, the original exception is also retained as ``__cause__`` by
    :func:`run_rank_zero_phase`.
    """

    def __init__(
        self,
        phase: str,
        root_exception_type: str,
        root_exception_message: str,
        rank_zero_traceback: str,
        failure_stage: str = "execution",
    ) -> None:
        self.phase = phase
        self.root_exception_type = root_exception_type
        self.root_exception_message = root_exception_message
        self.rank_zero_traceback = rank_zero_traceback
        self.failure_stage = failure_stage
        cause = root_exception_type
        if root_exception_message:
            cause = f"{cause}: {root_exception_message}"
        super().__init__(
            f"Rank-zero phase {phase!r} failed during {failure_stage} "
            f"on rank 0 with {cause}"
        )


# =============================================================================
# Distributed state queries
# =============================================================================


def is_distributed() -> bool:
    """True iff a process group is initialised."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Global rank of this process, or 0 outside DDP."""
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    """Local rank from ``LOCAL_RANK`` env (set by torchrun), or 0."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Number of processes participating, or 1 outside DDP."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """True on rank 0 (always True outside DDP)."""
    return get_rank() == 0


def has_torchrun_env() -> bool:
    """True iff this process was spawned by torchrun (LOCAL_RANK is set)."""
    return "LOCAL_RANK" in os.environ


def barrier() -> None:
    """Synchronisation barrier; no-op outside DDP."""
    if is_distributed():
        dist.barrier()


def _phase_name(phase: str) -> str:
    """Validate and normalize a rank-zero phase label."""
    if not isinstance(phase, str):
        raise TypeError(f"phase must be a string, got {type(phase).__name__}")
    name = phase.strip()
    if not name:
        raise ValueError("phase must be a non-empty string")
    return name


def _exception_type_name(exc: BaseException) -> str:
    """Return a stable, readable type name for a serialized exception."""
    cls = type(exc)
    if cls.__module__ == "builtins":
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


def _failure_payload(phase: str, stage: str, exc: BaseException) -> bytes:
    """Serialize exception metadata without requiring the exception to pickle."""
    try:
        message = str(exc)
    except BaseException:
        message = "<exception message could not be rendered>"
    try:
        formatted_traceback = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    except BaseException:
        formatted_traceback = "<rank-zero traceback could not be rendered>"
    outcome = {
        "ok": False,
        "phase": phase,
        "failure_stage": stage,
        "root_exception_type": _exception_type_name(exc),
        "root_exception_message": message,
        "rank_zero_traceback": formatted_traceback,
    }
    return pickle.dumps(outcome, protocol=pickle.HIGHEST_PROTOCOL)


def _collective_device() -> torch.device:
    """Choose storage accepted by the active collective backend."""
    backend = str(dist.get_backend()).lower()
    if "nccl" in backend:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _broadcast_bytes_from_rank_zero(payload: Optional[bytes]) -> bytes:
    """Broadcast a variable-length byte payload from rank 0."""
    device = _collective_device()
    size = torch.tensor(
        [len(payload) if payload is not None else 0],
        dtype=torch.int64,
        device=device,
    )
    dist.broadcast(size, src=0)
    payload_size = int(size.item())
    if payload_size <= 0:
        raise RuntimeError("rank-zero phase produced an empty collective payload")

    if get_rank() == 0:
        data = torch.tensor(bytearray(payload or b""), dtype=torch.uint8, device=device)
    else:
        data = torch.empty(payload_size, dtype=torch.uint8, device=device)
    dist.broadcast(data, src=0)
    return bytes(data.cpu().tolist())


def run_rank_zero_phase(
    phase: str,
    function: Callable[[], PhaseResult],
) -> PhaseResult:
    """Run one callable on rank 0 and share its result or failure with all ranks.

    Every rank must call this collective in the same order. Rank 0 executes
    ``function``; the other ranks never call it. A successful, pickleable
    return value is broadcast and returned on every rank. If the callable
    raises, or its result cannot be serialized, every rank raises the same
    :class:`RankZeroPhaseError` after receiving the rank-zero exception type,
    message, and traceback. Rank 0 chains the original exception as the error's
    cause.

    Outside distributed mode, this calls ``function`` directly and preserves
    its normal return value and exception behavior.

    Args:
        phase: Short name used to identify the setup, validation, checkpoint,
            callback, or other rank-zero-only phase in failures.
        function: Zero-argument callable to execute on rank 0.

    Returns:
        The callable's return value on every rank.

    Raises:
        RankZeroPhaseError: The rank-zero callable failed or returned a value
            that could not be serialized while distributed is initialized.
    """
    if not is_distributed():
        _phase_name(phase)
        return function()

    wire_payload: Optional[bytes] = None
    root_exception: Optional[BaseException] = None
    if get_rank() == 0:
        try:
            name = _phase_name(phase)
        except BaseException as exc:
            root_exception = exc
            wire_payload = _failure_payload("<invalid phase>", "phase validation", exc)
        else:
            try:
                value = function()
            except BaseException as exc:
                root_exception = exc
                wire_payload = _failure_payload(name, "execution", exc)
            else:
                try:
                    outcome = {"ok": True, "phase": name, "value": value}
                    wire_payload = pickle.dumps(
                        outcome, protocol=pickle.HIGHEST_PROTOCOL
                    )
                except BaseException as exc:
                    root_exception = exc
                    wire_payload = _failure_payload(name, "result serialization", exc)

    outcome = pickle.loads(_broadcast_bytes_from_rank_zero(wire_payload))
    if outcome["ok"]:
        return outcome["value"]

    error = RankZeroPhaseError(
        phase=outcome["phase"],
        root_exception_type=outcome["root_exception_type"],
        root_exception_message=outcome["root_exception_message"],
        rank_zero_traceback=outcome["rank_zero_traceback"],
        failure_stage=outcome["failure_stage"],
    )
    if root_exception is not None:
        raise error from root_exception
    raise error


# =============================================================================
# Device argument parsing
# =============================================================================


def parse_device_arg(device: DeviceArg) -> List[int]:
    """Parse a user-facing device argument into a list of CUDA device indices.

    Returns an empty list for CPU / MPS / auto-no-cuda.

    Accepts:
      - ``0`` or ``"0"`` → ``[0]``
      - ``[0, 1]`` or ``"0,1"`` → ``[0, 1]``
      - ``"cpu"``, ``"mps"``, ``"auto"``, ``""`` → ``[]``
      - ``"cuda:0"`` → ``[0]``
    """
    if device is None:
        return []
    if isinstance(device, int):
        return [device] if device >= 0 else []
    if isinstance(device, (list, tuple)):
        return [int(d) for d in device if isinstance(d, int) and d >= 0]
    s = str(device).strip().lower()
    if s in ("", "auto", "cpu", "mps"):
        return []
    if "," in s:
        return [
            int(x.strip())
            for x in s.split(",")
            if x.strip().lstrip("-").isdigit() and int(x.strip()) >= 0
        ]
    if s.startswith("cuda:"):
        s = s.split(":", 1)[1]
    if s.lstrip("-").isdigit():
        idx = int(s)
        return [idx] if idx >= 0 else []
    return []


def wants_distributed(device: DeviceArg) -> bool:
    """True iff the device argument names more than one GPU.

    This is a *user intent* check, separate from whether torchrun launched
    the process. A user calling ``model.train(device=[0, 1])`` from a plain
    Python script (no torchrun) signals intent to do DDP; the trainer can
    then raise a clear error pointing them at torchrun.
    """
    return len(parse_device_arg(device)) > 1


# =============================================================================
# Process-group lifecycle
# =============================================================================


def _select_backend() -> str:
    """Pick NCCL when CUDA + NCCL are available, else Gloo.

    NCCL is the fast GPU backend but isn't built on Windows. Gloo works
    everywhere (CPU and GPU) so it's the safe fallback. Windows users
    get Gloo automatically.
    """
    if torch.cuda.is_available() and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def init_distributed(timeout_seconds: int = 10800) -> None:
    """Initialise the default process group from env vars set by torchrun.

    Safe to call multiple times — second and later calls are no-ops.
    Expects ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE`` to be set in the
    environment (which torchrun does automatically).
    """
    import inspect

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build")
    if dist.is_initialized():
        return
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "init_distributed() called without LOCAL_RANK env var. "
            "Multi-GPU training requires launching with torchrun, e.g. "
            "`torchrun --nproc_per_node=2 your_script.py`."
        )
    backend = _select_backend()
    local_rank = int(os.environ["LOCAL_RANK"])
    init_kwargs: dict = {
        "backend": backend,
        "timeout": timedelta(seconds=timeout_seconds),
        "rank": int(os.environ["RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
    }
    # device_id was added in PyTorch 2.0; guard so we stay compatible with older builds.
    # inspect.signature() can raise ValueError/TypeError on some C-extension builds,
    # so wrap defensively and omit the kwarg on failure.
    try:
        pg_sig = inspect.signature(dist.init_process_group)
        if (
            "device_id" in pg_sig.parameters
            and backend == "nccl"
            and torch.cuda.is_available()
        ):
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
    except (TypeError, ValueError):
        pass
    dist.init_process_group(**init_kwargs)


def shutdown_distributed() -> None:
    """Tear down the default process group if it was initialised."""
    if is_distributed():
        dist.destroy_process_group()


# =============================================================================
# Model unwrapping
# =============================================================================


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP / DataParallel / torch.compile wrappers from a module.

    Idempotent. Returns ``model`` unchanged if no wrappers are present.
    Required when reading ``model.named_parameters()`` for optimizer setup
    after DDP wrap, for state-dict saving, and when model-specific hooks
    need to read attributes that live on the unwrapped module.
    """
    parallel_types = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    while True:
        if isinstance(model, parallel_types):
            model = model.module
            continue
        # torch.compile() wraps modules with an _orig_mod attribute
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        return model


# =============================================================================
# Loss scaling for DDP
# =============================================================================


def all_reduce_avg_scalar(
    value: Union[float, torch.Tensor],
    *,
    device: Optional[torch.device] = None,
    min_value: float = 1.0,
) -> float:
    """Average a per-rank scalar count across ranks: ``sum`` then ``/ world_size``.

    Mirrors the DETR ``num_boxes`` reduction. A mean/ratio-normalized loss must
    divide by the *global* count of positives to be numerically equivalent to
    single-GPU training on the same global batch: with each rank dividing by
    ``global_count / world_size`` and DDP averaging the gradients, the two
    cancel to reproduce the single-GPU gradient exactly (see
    ``scale_loss_for_ddp``).

    The global sum is clamped to ``min_value`` BEFORE dividing by world_size.
    Clamping after the divide (the upstream-DETR order) breaks exactness
    whenever the global positive mass is below world_size: an all-background
    batch would train at exactly 1/world_size gradient — worst on the
    background-heavy datasets from issue #484. Clamp-first keeps the
    single-GPU-equivalence exact for every batch, including empty ones.

    Outside DDP this is just ``max(value, min_value)`` — single-GPU behavior is
    unchanged (identical numeric value to the previous ``max(sum, 1)``).

    This is a COLLECTIVE under DDP: every rank must reach it or the callers
    deadlock. Only call it from code that runs symmetrically on all ranks
    (the training loss); never from rank-0-only paths like validation.
    """
    if isinstance(value, torch.Tensor):
        # .sum() materializes a fresh tensor even for 0-dim input, so the
        # in-place all_reduce below cannot touch the caller's storage.
        v = value.detach().float().sum().reshape(())
    else:
        v = torch.as_tensor(float(value), dtype=torch.float32, device=device)
    if is_distributed():
        if v.device.type == "cpu" and dist.get_backend() == "nccl":
            raise ValueError(
                "all_reduce_avg_scalar got a CPU scalar under the NCCL "
                "backend; pass device= (or a tensor already on the right "
                "GPU) so the collective can run."
            )
        dist.all_reduce(v)
        v = v.clamp_min(min_value) / float(get_world_size())
        return float(v.item())
    return float(v.clamp_min(min_value).item())


def scale_loss_for_ddp(loss: torch.Tensor) -> torch.Tensor:
    """Pass the loss through unchanged — DDP needs no per-loss rescaling here.

    DDP all-reduces gradients during ``backward()`` and **averages** them
    (divides by world_size). Every LibreYOLO loss is mean/ratio-normalized:
    each rank already produces a "full-batch magnitude" gradient (normalized by
    a per-positive / per-box count — globally reduced for yolo9's ``cls_norm``
    and the DETR ``num_boxes``), so DDP's averaging composes them into the
    single-GPU-equivalent gradient. Multiplying by world_size on top of that
    over-counts by ~N and inflates the effective learning rate — this was the
    root cause of the multi-GPU accuracy gap in issue #484 (4-GPU trained at
    ~4× LR vs single-GPU).

    This is a CONTRACT each family's loss must satisfy, not a given: a loss
    that normalizes by a globally-summed count WITHOUT dividing by world_size
    (as RT-DETR's ``num_boxes`` did before #484 fixed it) under-scales by 1/N
    once the identity is in place. New families must either use a local
    normalizer or the ``all_reduce_avg_scalar`` reduction (global sum /
    world_size) — never a bare global sum.

    Kept as the single, documented seam where a *genuinely sum-reduced* loss
    (no normalizer) could opt back into ``loss * world_size``. No family
    currently uses one, so this is the identity in every case, DDP or not.
    """
    return loss


# =============================================================================
# Seeding
# =============================================================================


def seed_for_rank(base_seed: int) -> int:
    """Per-rank seed: ``base_seed + 1 + rank``.

    Ensures different augmentation / dataloader shuffling across ranks while
    keeping the run reproducible when ``base_seed`` and ``world_size`` are
    fixed.
    """
    return base_seed + 1 + get_rank()


# =============================================================================
# Auto-spawn DDP helpers
# =============================================================================


def _find_free_port() -> tuple:
    """Bind to port 0 and return ``(port, socket)``.

    The caller is responsible for closing the socket.  Keeping it open
    until just before ``mp.spawn`` is called minimises the TOCTOU window
    between OS port selection and torch.distributed's TCPStore binding.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", 0))
    port = s.getsockname()[1]
    return port, s


def spawn_ddp_train(
    worker_fn: Callable,
    spawn_args: Tuple,
    nprocs: int,
    result_path: str,
    master_addr: str = "127.0.0.1",
    master_port: Optional[int] = None,
    devices: Optional[List[int]] = None,
) -> None:
    """Spawn *nprocs* DDP workers via :func:`torch.multiprocessing.spawn`.

    Each worker is called as::

        worker_fn(rank, nprocs, master_addr, master_port, result_path, *spawn_args)

    The worker is responsible for setting RANK/LOCAL_RANK/WORLD_SIZE/MASTER_*
    env vars, initialising the process group, running training, and writing a
    result JSON to *result_path* (rank 0 only).

    This is the internal engine behind the auto-spawn path triggered when a
    user calls ``model.train(device="0,1")`` from a plain Python script (no
    torchrun). The model's ``train()`` method calls this helper, collects the
    result JSON from *result_path*, and returns it to the caller — so the user
    gets a clean blocking call without any subprocess plumbing.

    When *devices* is provided, ``CUDA_VISIBLE_DEVICES`` is set to the
    comma-joined device indices before spawning so that ``cuda:N`` inside each
    worker maps to the N-th requested physical GPU.  The original value is
    restored after spawning completes.
    """
    import multiprocessing
    import torch.multiprocessing as mp

    if multiprocessing.current_process().name != "MainProcess":
        raise RuntimeError(
            "spawn_ddp_train() was called from inside a spawned subprocess. "
            "This usually means your script calls model.train(device=...) at "
            "the top level without a 'if __name__ == \"__main__\":' guard. "
            "Each spawned worker re-imports __main__, which re-launches "
            "training and causes infinite recursion. Wrap your training call:\n\n"
            "    if __name__ == '__main__':\n"
            "        model.train(device='0,1')\n"
        )

    port_sock = None
    if master_port is None:
        master_port, port_sock = _find_free_port()

    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        if prev_cvd is not None:
            # devices are logical indices into the existing mask — translate to
            # physical GPU IDs so the new mask is correct inside spawned workers.
            existing = [x.strip() for x in prev_cvd.split(",") if x.strip()]
            try:
                new_cvd = ",".join(existing[d] for d in devices)
            except IndexError:
                new_cvd = ",".join(str(d) for d in devices)
        else:
            new_cvd = ",".join(str(d) for d in devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = new_cvd
    try:
        # Close the reservation socket as late as possible — just before
        # spawning — so the OS cannot hand the port to another process in the
        # gap between our bind(0) call and torch.distributed's TCPStore bind.
        if port_sock is not None:
            port_sock.close()
            port_sock = None
        mp.spawn(
            worker_fn,
            args=(nprocs, master_addr, master_port, result_path) + spawn_args,
            nprocs=nprocs,
            join=True,
        )
    finally:
        if port_sock is not None:
            port_sock.close()
        if devices:
            if prev_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd


__all__ = [
    "DeviceArg",
    "RankZeroPhaseError",
    "all_reduce_avg_scalar",
    "barrier",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "has_torchrun_env",
    "init_distributed",
    "is_distributed",
    "is_main_process",
    "parse_device_arg",
    "run_rank_zero_phase",
    "scale_loss_for_ddp",
    "seed_for_rank",
    "shutdown_distributed",
    "spawn_ddp_train",
    "unwrap_model",
    "wants_distributed",
]
