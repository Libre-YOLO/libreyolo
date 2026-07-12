"""Internal validation helpers for metric input contracts."""

from __future__ import annotations

from typing import Any

import torch


def _batch_size(value: Any, name: str, context: str) -> int:
    """Return an input's leading dimension or fail with a metric-level error."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0:
            raise ValueError(f"{context} {name} must have a batch dimension.")
        return int(shape[0])
    try:
        return len(value)
    except TypeError as exc:
        raise ValueError(f"{context} {name} must have a batch dimension.") from exc


def require_matching_batch_sizes(context: str, **batches: Any) -> int:
    """Require all supplied values to have the same leading dimension."""
    if not batches:
        raise ValueError(f"{context} did not receive any batched values.")
    sizes = {
        name: _batch_size(value, name, context) for name, value in batches.items()
    }
    if len(set(sizes.values())) != 1:
        details = ", ".join(f"{name}={size}" for name, size in sizes.items())
        raise ValueError(f"{context} batch size mismatch: {details}.")
    return next(iter(sizes.values()))


def require_finite(
    values: Any,
    context: str,
    *,
    where: Any | None = None,
) -> None:
    """Reject non-finite metric inputs, optionally only in an evaluated region."""
    try:
        tensor = torch.as_tensor(values)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{context} must contain numeric values.") from exc
    if tensor.is_complex():
        raise ValueError(f"{context} must contain real values.")
    if tensor.numel() == 0 or not tensor.is_floating_point():
        return
    finite = torch.isfinite(tensor)
    if where is not None:
        mask = torch.as_tensor(where, dtype=torch.bool, device=tensor.device)
        if mask.shape != tensor.shape:
            raise ValueError(
                f"{context} validity mask shape {tuple(mask.shape)} does not match "
                f"values shape {tuple(tensor.shape)}."
            )
        finite = finite[mask]
    if finite.numel() and not bool(finite.all()):
        suffix = " in the evaluated region" if where is not None else ""
        raise ValueError(f"{context} contains non-finite values{suffix}.")


def require_class_ids(values: Any, num_classes: int, context: str) -> torch.Tensor:
    """Return class ids as int64 after enforcing an integral, finite class space."""
    if int(num_classes) <= 0:
        raise ValueError(f"{context} requires a positive class count.")
    try:
        tensor = torch.as_tensor(values)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{context} class ids must be numeric.") from exc
    if tensor.dtype == torch.bool:
        raise ValueError(f"{context} class ids must be integers, not booleans.")
    if tensor.numel() == 0:
        return tensor.to(dtype=torch.int64)
    require_finite(tensor, context)
    if tensor.is_floating_point():
        if not bool((tensor == tensor.round()).all()):
            raise ValueError(f"{context} class ids must be integer-valued.")
    class_ids = tensor.to(dtype=torch.int64)
    valid = (class_ids >= 0) & (class_ids < int(num_classes))
    if not bool(valid.all()):
        invalid = torch.unique(class_ids[~valid]).cpu().tolist()
        raise ValueError(
            f"{context} class ids must lie in [0, {int(num_classes)}); "
            f"got {invalid}."
        )
    return class_ids


__all__ = [
    "require_class_ids",
    "require_finite",
    "require_matching_batch_sizes",
]
