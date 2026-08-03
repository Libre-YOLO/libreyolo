"""Shared target conversion for training-time validation-loss adapters.

The validator hands every adapter the same padded ``[batch, labels, >=5]``
tensor of ``xyxy`` pixel boxes plus a class column, because that is what the
COCO metric path already needs. Each model family's criterion wants something
else, so the conversions live here instead of being copied into one
``validation_loss.py`` per family.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch
from torch import nn

__all__ = [
    "check_padded_targets",
    "emit_loss_outputs",
    "loss_output_modules",
    "padded_targets_to_detr",
    "padded_targets_to_flat_pixels",
    "valid_target_mask",
]


def loss_output_modules(model: nn.Module) -> list[nn.Module]:
    """Return the submodules that gate training-shaped outputs on a flag.

    The DETR-line decoders score only the ``eval_idx`` layer and assemble a
    two-key inference dict in eval, which is not enough for the auxiliary and
    encoder loss terms. Each such module declares an ``emit_loss_outputs``
    attribute; validation flips them all for the duration of the pass.
    """
    return [
        module
        for module in model.modules()
        if isinstance(getattr(module, "emit_loss_outputs", None), bool)
    ]


@contextlib.contextmanager
def emit_loss_outputs(model: nn.Module) -> Iterator[None]:
    """Make ``model`` produce training-shaped outputs inside the block.

    The flag stays off outside the block so predict, export, and standalone
    ``val()`` keep seeing the inference-shaped output dict.
    """
    modules = loss_output_modules(model)
    if not modules:
        raise TypeError(
            "Validation loss needs a model whose decoder declares "
            "emit_loss_outputs; none was found"
        )
    for module in modules:
        module.emit_loss_outputs = True
    try:
        yield
    finally:
        for module in modules:
            module.emit_loss_outputs = False


def check_padded_targets(targets: torch.Tensor, *, family: str) -> None:
    """Reject a target tensor the validator could not have produced."""
    if targets.ndim != 3 or targets.shape[-1] < 5:
        raise ValueError(
            f"{family} validation targets must have shape [batch, labels, >=5]"
        )


def valid_target_mask(source: torch.Tensor) -> torch.Tensor:
    """True where a padded row holds a real box (padding rows are all zero)."""
    return (source[..., 2] > source[..., 0]) & (source[..., 3] > source[..., 1])


def _check_labels(labels: torch.Tensor, *, num_classes: int, family: str) -> None:
    """Reject class ids the criterion would index out of bounds with."""
    invalid = (labels < 0) | (labels >= num_classes) | (labels != labels.round())
    if invalid.any():
        bad_label = float(labels[invalid][0].item())
        raise ValueError(
            f"{family} validation target class {bad_label:g} is outside "
            f"[0, {num_classes - 1}]"
        )


def _to_device(
    targets: torch.Tensor, device: torch.device, family: str
) -> torch.Tensor:
    check_padded_targets(targets, family=family)
    return targets[..., :5].to(
        device=device,
        dtype=torch.float32,
        non_blocking=device.type == "cuda",
    )


def padded_targets_to_detr(
    targets: torch.Tensor,
    *,
    image_size: tuple[int, int],
    num_classes: int,
    device: torch.device,
    family: str,
) -> list[dict[str, torch.Tensor]]:
    """Convert to the DETR-line ``[{labels, boxes}]`` list.

    ``boxes`` are normalized ``cxcywh`` clamped to ``[0, 1]``, matching what
    the family trainers build for the same criterion during training.
    """
    source = _to_device(targets, device, family)
    height, width = image_size
    scale = torch.tensor(
        [width, height, width, height], dtype=torch.float32, device=device
    )

    target_list: list[dict[str, torch.Tensor]] = []
    for image_targets in source:
        rows = image_targets[valid_target_mask(image_targets)]
        labels = rows[:, 4]
        _check_labels(labels, num_classes=num_classes, family=family)

        xyxy = (rows[:, :4] / scale).clamp(0.0, 1.0)
        boxes = torch.empty_like(xyxy)
        boxes[:, :2] = (xyxy[:, :2] + xyxy[:, 2:]) * 0.5
        boxes[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
        target_list.append({"labels": labels.long(), "boxes": boxes})
    return target_list


def padded_targets_to_flat_pixels(
    targets: torch.Tensor,
    *,
    num_classes: int,
    device: torch.device,
    family: str,
) -> torch.Tensor:
    """Convert to the flat ``[N, 6]`` ``image_index, class, cx, cy, w, h`` form.

    Coordinates stay in pixels, which is the space YOLO-NAS assigns in.
    """
    source = _to_device(targets, device, family)
    rows = valid_target_mask(source).nonzero(as_tuple=False)
    if rows.numel() == 0:
        return torch.zeros((0, 6), dtype=torch.float32, device=device)

    selected = source[rows[:, 0], rows[:, 1]]
    _check_labels(selected[:, 4], num_classes=num_classes, family=family)

    image_index = rows[:, 0].to(dtype=torch.float32)
    xyxy = selected[:, :4]
    centers = (xyxy[:, :2] + xyxy[:, 2:]) * 0.5
    sizes = xyxy[:, 2:] - xyxy[:, :2]
    return torch.cat(
        [
            image_index.unsqueeze(1),
            selected[:, 4:5],
            centers,
            sizes,
        ],
        dim=1,
    )
