"""Flat result containers for LibreYOLO."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch

from libreyolo.tasks import normalize_task


TensorLike = Union[torch.Tensor, np.ndarray]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        converted = value.detach().cpu()
        return converted.item() if converted.ndim == 0 else converted.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_key(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value.detach().cpu().item()
    return value


def _require_tensorlike(data: Any, name: str) -> TensorLike:
    if not isinstance(data, (torch.Tensor, np.ndarray)):
        raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray")
    return data


def _validate_nonnegative_id_map(data: TensorLike, name: str) -> None:
    if isinstance(data, torch.Tensor):
        is_integer = (
            data.dtype != torch.bool
            and not data.dtype.is_floating_point
            and not data.dtype.is_complex
        )
        if not is_integer:
            raise ValueError(f"{name} must contain non-negative integer IDs")
        if torch.iinfo(data.dtype).min < 0 and bool((data < 0).any().item()):
            raise ValueError(f"{name} must not contain negative IDs")
        return

    if data.dtype == np.bool_ or not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"{name} must contain non-negative integer IDs")
    if bool(np.any(data < 0)):
        raise ValueError(f"{name} must not contain negative IDs")


def _validate_track_ids(data: TensorLike, name: str = "track ids") -> None:
    if isinstance(data, torch.Tensor):
        if data.dtype == torch.bool or data.dtype.is_complex:
            raise ValueError(f"{name} must contain finite non-negative integers")
        if data.dtype.is_floating_point:
            if not bool(torch.isfinite(data).all().item()) or not bool(
                torch.equal(data, data.trunc())
            ):
                raise ValueError(f"{name} must contain finite integer-valued IDs")
        is_signed = data.dtype.is_floating_point or torch.iinfo(data.dtype).min < 0
        if is_signed and bool((data < 0).any().item()):
            raise ValueError(f"{name} must contain non-negative IDs")
        return

    if data.dtype == np.bool_ or np.issubdtype(data.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain finite non-negative integers")
    if np.issubdtype(data.dtype, np.floating):
        if not bool(np.isfinite(data).all()) or not bool(np.equal(data, np.trunc(data)).all()):
            raise ValueError(f"{name} must contain finite integer-valued IDs")
    elif not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"{name} must contain finite non-negative integers")
    if bool(np.any(data < 0)):
        raise ValueError(f"{name} must contain non-negative IDs")


def _validate_orig_shape(
    orig_shape: Tuple[int, int] | None,
    *,
    name: str = "orig_shape",
) -> Tuple[int, int] | None:
    if orig_shape is None:
        return None
    if not isinstance(orig_shape, (tuple, list)) or len(orig_shape) != 2:
        raise ValueError(f"{name} must be a (height, width) pair")
    h, w = orig_shape
    if isinstance(h, bool) or isinstance(w, bool):
        raise ValueError(f"{name} values must be positive integers")
    if int(h) != h or int(w) != w or int(h) <= 0 or int(w) <= 0:
        raise ValueError(f"{name} values must be positive integers, got {orig_shape!r}")
    return int(h), int(w)


def _validate_integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if converted != value or converted < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return converted


def _selector_indices(idx: Any, length: int) -> Tuple[List[int], bool]:
    """Resolve a public first-axis selector to finite, normalized row indices."""
    if idx is Ellipsis:
        return list(range(length)), False
    if isinstance(idx, (int, np.integer)) and not isinstance(idx, (bool, np.bool_)):
        value = int(idx)
        if value < 0:
            value += length
        if value < 0 or value >= length:
            raise IndexError(f"index {int(idx)} is out of bounds for result of length {length}")
        return [value], True

    if isinstance(idx, slice):
        return list(range(length))[idx], False

    if isinstance(idx, torch.Tensor):
        if idx.ndim == 0:
            if idx.dtype == torch.bool:
                raise TypeError("a scalar boolean is not a valid result index")
            if idx.dtype.is_floating_point or idx.dtype.is_complex:
                raise TypeError("result indices must be integers or booleans")
            return _selector_indices(int(idx.item()), length)
        values = idx.detach().cpu().numpy()
    else:
        values = np.asarray(idx)

    if values.ndim == 0:
        if np.issubdtype(values.dtype, np.bool_):
            raise TypeError("a scalar boolean is not a valid result index")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("result indices must be integers or booleans")
        return _selector_indices(int(values.item()), length)
    if values.ndim != 1:
        raise IndexError("result indices must be one-dimensional")
    if values.size == 0:
        if np.issubdtype(values.dtype, np.integer) or np.issubdtype(
            values.dtype, np.bool_
        ):
            return [], False
        if isinstance(idx, (list, tuple)) and len(idx) == 0:
            return [], False
        raise TypeError("result indices must be integers or booleans")
    if np.issubdtype(values.dtype, np.bool_):
        if int(values.size) != length:
            raise IndexError(
                f"boolean index has length {int(values.size)}, expected {length}"
            )
        return np.flatnonzero(values).astype(int).tolist(), False
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("result indices must be integers or booleans")

    indices: List[int] = []
    for raw in values.tolist():
        value = int(raw)
        if value < 0:
            value += length
        if value < 0 or value >= length:
            raise IndexError(f"index {int(raw)} is out of bounds for result of length {length}")
        indices.append(value)
    return indices, False


def _take_first(data: TensorLike, indices: List[int]) -> TensorLike:
    if isinstance(data, torch.Tensor):
        selector = torch.tensor(indices, dtype=torch.long, device=data.device)
        return data.index_select(0, selector)
    return data[np.asarray(indices, dtype=np.intp)]


def _move(data: TensorLike | None, *args, **kwargs):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data).to(*args, **kwargs)
    return data


def _move_ids_like(data: TensorLike | None, reference: TensorLike) -> TensorLike | None:
    """Move identity data to a reference backend/device without changing dtype."""
    if data is None:
        return None
    if isinstance(reference, torch.Tensor):
        if isinstance(data, torch.Tensor):
            return data.to(device=reference.device)
        return torch.as_tensor(data, device=reference.device)
    return _numpy(data)


def _cpu(data: TensorLike | None):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    return data


def _cuda(data: TensorLike | None):
    if isinstance(data, torch.Tensor):
        return data.cuda()
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data).cuda()
    return data


def _numpy(data: TensorLike | None):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def _slice_first(data: TensorLike | None, idx):
    if data is None:
        return None
    indices, _ = _selector_indices(idx, int(data.shape[0]))
    return _take_first(data, indices)


class Boxes:
    """Wrap detection boxes for a single image."""

    def __init__(
        self,
        boxes: TensorLike,
        conf: TensorLike,
        cls: TensorLike,
        id: TensorLike | None = None,
        orig_shape: Tuple[int, int] | None = None,
    ):
        _require_tensorlike(boxes, "boxes")
        _require_tensorlike(conf, "conf")
        _require_tensorlike(cls, "cls")
        if boxes.ndim == 1 and int(boxes.shape[0]) == 0:
            boxes = boxes.reshape(0, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"expected boxes with shape (N, 4), got {tuple(boxes.shape)}")
        n = int(boxes.shape[0])
        for value, name in ((conf, "conf"), (cls, "cls"), (id, "id")):
            if value is None:
                continue
            _require_tensorlike(value, name)
            if value.ndim != 1 or int(value.shape[0]) != n:
                raise ValueError(
                    f"expected {name} with shape ({n},), got {tuple(value.shape)}"
                )
            if isinstance(value, torch.Tensor) != isinstance(boxes, torch.Tensor):
                raise TypeError(
                    f"boxes and {name} must use the same tensor/array container"
                )
            if (
                isinstance(boxes, torch.Tensor)
                and isinstance(value, torch.Tensor)
                and value.device != boxes.device
            ):
                raise ValueError(
                    f"boxes and {name} must be on the same device, got "
                    f"{boxes.device} and {value.device}"
                )
            if name == "id":
                _validate_track_ids(value)
        self._boxes = boxes
        self._conf = conf
        self._cls = cls
        self._id = id
        self.orig_shape = _validate_orig_shape(orig_shape)

    @property
    def xyxy(self) -> TensorLike:
        return self._boxes

    @property
    def conf(self) -> TensorLike:
        return self._conf

    @property
    def cls(self) -> TensorLike:
        return self._cls

    @property
    def id(self) -> TensorLike | None:
        return self._id

    @property
    def is_track(self) -> bool:
        return self._id is not None

    @property
    def xywh(self) -> TensorLike:
        b = self._boxes
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        if isinstance(b, torch.Tensor):
            return torch.stack([cx, cy, w, h], dim=1)
        return np.stack([cx, cy, w, h], axis=1)

    @property
    def xyxyn(self) -> TensorLike:
        """Normalized xyxy boxes."""
        return self._normalize_boxes(self.xyxy)

    @property
    def xywhn(self) -> TensorLike:
        """Normalized xywh boxes."""
        return self._normalize_boxes(self.xywh)

    def _normalize_boxes(self, boxes: TensorLike) -> TensorLike:
        if self.orig_shape is None:
            raise ValueError("orig_shape is required for normalized box coordinates")
        h, w = self.orig_shape
        if isinstance(boxes, torch.Tensor):
            scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        else:
            scale = np.array([w, h, w, h], dtype=boxes.dtype)
        return boxes / scale

    def with_id(self, id: TensorLike | None) -> "Boxes":
        return Boxes(self._boxes, self._conf, self._cls, id, self.orig_shape)

    def with_orig_shape(self, orig_shape: Tuple[int, int] | None) -> "Boxes":
        return Boxes(self._boxes, self._conf, self._cls, self._id, orig_shape)

    @property
    def data(self) -> TensorLike:
        parts = [self._boxes]
        if self._id is not None:
            parts.append(self._id.reshape(-1, 1))
        parts.extend([self._conf.reshape(-1, 1), self._cls.reshape(-1, 1)])
        if isinstance(self._boxes, torch.Tensor):
            return torch.cat(parts, dim=1)
        return np.concatenate(parts, axis=1)

    def to(self, *args, **kwargs) -> "Boxes":
        moved_boxes = _move(self._boxes, *args, **kwargs)
        return Boxes(
            moved_boxes,
            _move(self._conf, *args, **kwargs),
            _move(self._cls, *args, **kwargs),
            _move_ids_like(self._id, moved_boxes),
            self.orig_shape,
        )

    def cpu(self) -> "Boxes":
        return Boxes(
            _cpu(self._boxes),
            _cpu(self._conf),
            _cpu(self._cls),
            _cpu(self._id),
            self.orig_shape,
        )

    def cuda(self) -> "Boxes":
        return Boxes(
            _cuda(self._boxes),
            _cuda(self._conf),
            _cuda(self._cls),
            _cuda(self._id),
            self.orig_shape,
        )

    def numpy(self) -> "Boxes":
        return Boxes(
            _numpy(self._boxes),
            _numpy(self._conf),
            _numpy(self._cls),
            _numpy(self._id),
            self.orig_shape,
        )

    def __getitem__(self, idx) -> "Boxes":
        return Boxes(
            _slice_first(self._boxes, idx),
            _slice_first(self._conf, idx),
            _slice_first(self._cls, idx),
            _slice_first(self._id, idx),
            self.orig_shape,
        )

    def __len__(self) -> int:
        return int(self._boxes.shape[0])

    def __iter__(self) -> Iterator["Boxes"]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return (
            f"Boxes(n={len(self)}, "
            f"xyxy={tuple(self._boxes.shape)}, "
            f"conf={tuple(self._conf.shape)}, "
            f"cls={tuple(self._cls.shape)}, "
            f"is_track={self.is_track})"
        )


class Masks:
    """Wrap instance masks for a single image."""

    def __init__(
        self,
        masks: TensorLike,
        orig_shape: Tuple[int, int],
    ):
        _require_tensorlike(masks, "masks")
        if masks.ndim != 3:
            raise ValueError(f"expected masks with shape (N, H, W), got {tuple(masks.shape)}")
        self._masks = masks
        validated_shape = _validate_orig_shape(orig_shape)
        if validated_shape is None:
            raise ValueError("orig_shape is required for Masks")
        self.orig_shape = validated_shape
        self._contours_cache: Optional[List[List[Dict[str, Any]]]] = None
        self._contours_cache_token: Any = None

    @property
    def data(self) -> TensorLike:
        return self._masks

    @property
    def xy(self) -> List[np.ndarray]:
        """Largest outer contour per mask, kept for API compatibility."""
        return [self._primary_contour(records, normalize=False) for records in self._contours()]

    @property
    def xyn(self) -> List[np.ndarray]:
        """Normalized largest outer contour per mask."""
        return [self._primary_contour(records, normalize=True) for records in self._contours()]

    @property
    def contours(self) -> List[List[Dict[str, Any]]]:
        """All contour components and holes for every mask in pixel coordinates."""
        return self._public_contours(normalize=False)

    @property
    def contours_normalized(self) -> List[List[Dict[str, Any]]]:
        """All contour components and holes normalized to the original canvas."""
        return self._public_contours(normalize=True)

    def _contours(self) -> List[List[Dict[str, Any]]]:
        raw_masks_np = _numpy(self._masks)
        token = self._contour_token(raw_masks_np)
        if self._contours_cache is not None and token == self._contours_cache_token:
            return self._contours_cache

        import cv2

        masks_np = raw_masks_np.astype(np.uint8)
        contours_list: List[List[Dict[str, Any]]] = []
        for mask in masks_np:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            records: List[Dict[str, Any]] = []
            hierarchy_rows = hierarchy[0] if hierarchy is not None else []
            for contour_index, contour in enumerate(contours):
                points = contour.reshape(-1, 2).astype(np.float64)
                points.setflags(write=False)
                parent = int(hierarchy_rows[contour_index][3])
                depth = 0
                ancestor = parent
                while ancestor >= 0:
                    depth += 1
                    ancestor = int(hierarchy_rows[ancestor][3])
                records.append(
                    {
                        "points": points,
                        "is_hole": bool(depth % 2),
                        "parent": parent if parent >= 0 else None,
                        "_area": float(abs(cv2.contourArea(contour))),
                    }
                )
            contours_list.append(records)
        self._contours_cache = contours_list
        self._contours_cache_token = token
        return contours_list

    def _contour_token(self, masks_np: np.ndarray) -> Any:
        content_hash = hash(np.ascontiguousarray(masks_np).tobytes())
        if isinstance(self._masks, torch.Tensor):
            return (
                self._masks.data_ptr(),
                self._masks._version,
                tuple(self._masks.shape),
                tuple(self._masks.stride()),
                self._masks.dtype,
                self._masks.device,
                content_hash,
            )
        return (
            tuple(self._masks.shape),
            self._masks.strides,
            self._masks.dtype.str,
            content_hash,
        )

    def _primary_contour(
        self,
        records: List[Dict[str, Any]],
        *,
        normalize: bool,
    ) -> np.ndarray:
        outer = [record for record in records if not record["is_hole"]]
        if not outer:
            return np.empty((0, 2), dtype=np.float64)
        points = np.asarray(max(outer, key=lambda record: record["_area"])["points"]).copy()
        if normalize:
            h, w = self.orig_shape
            points /= np.array([w, h], dtype=np.float64)
        return points

    def _public_contours(self, *, normalize: bool) -> List[List[Dict[str, Any]]]:
        h, w = self.orig_shape
        scale = np.array([w, h], dtype=np.float64)
        output: List[List[Dict[str, Any]]] = []
        for records in self._contours():
            public_records: List[Dict[str, Any]] = []
            for record in records:
                points = np.asarray(record["points"]).copy()
                if normalize:
                    points /= scale
                public_records.append(
                    {
                        "points": points,
                        "is_hole": record["is_hole"],
                        "parent": record["parent"],
                    }
                )
            output.append(public_records)
        return output

    def to(self, *args, **kwargs) -> "Masks":
        moved = _move(self._masks, *args, **kwargs)
        if moved is self._masks and not isinstance(moved, torch.Tensor):
            return self
        return Masks(moved, self.orig_shape)

    def cpu(self) -> "Masks":
        if isinstance(self._masks, torch.Tensor):
            return Masks(self._masks.cpu(), self.orig_shape)
        return self

    def cuda(self) -> "Masks":
        return Masks(_cuda(self._masks), self.orig_shape)

    def numpy(self) -> "Masks":
        if isinstance(self._masks, torch.Tensor):
            return Masks(self._masks.detach().cpu().numpy(), self.orig_shape)
        return self

    def __getitem__(self, idx) -> "Masks":
        return Masks(_slice_first(self._masks, idx), self.orig_shape)

    def __len__(self) -> int:
        return int(self._masks.shape[0])

    def __iter__(self) -> Iterator["Masks"]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return (
            f"Masks(n={len(self)}, "
            f"shape={tuple(self._masks.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class _TensorPayload:
    """Small wrapper used for future flat result slots."""

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        _require_tensorlike(data, "data")
        self.data = data
        self.orig_shape = _validate_orig_shape(orig_shape)

    def to(self, *args, **kwargs):
        return self.__class__(_move(self.data, *args, **kwargs), self.orig_shape)

    def cpu(self):
        return self.__class__(_cpu(self.data), self.orig_shape)

    def cuda(self):
        return self.__class__(_cuda(self.data), self.orig_shape)

    def numpy(self):
        return self.__class__(_numpy(self.data), self.orig_shape)

    def __getitem__(self, idx):
        return self.__class__(_slice_first(self.data, idx), self.orig_shape)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class _WholeImagePayload(_TensorPayload):
    """A single image-level value whose internal dimensions are not rows."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx):
        indices, _ = _selector_indices(idx, 1)
        if indices != [0]:
            raise IndexError("whole-image payload selection must contain index 0")
        return self.__class__(self.data, self.orig_shape)


class Keypoints(_TensorPayload):
    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        _require_tensorlike(data, "keypoints")
        if data.ndim != 3 or int(data.shape[-1]) not in (2, 3):
            raise ValueError(
                f"expected keypoints with shape (N, K, 2|3), got {tuple(data.shape)}"
            )
        super().__init__(data, orig_shape)

    @property
    def xy(self) -> TensorLike:
        return self.data[..., :2]

    @property
    def xyn(self) -> TensorLike:
        if self.orig_shape is None:
            raise ValueError("orig_shape is required for normalized keypoints")
        h, w = self.orig_shape
        xy = self.xy
        if isinstance(xy, torch.Tensor):
            scale = torch.tensor([w, h], dtype=xy.dtype, device=xy.device)
        else:
            scale = np.array([w, h], dtype=xy.dtype)
        return xy / scale

    @property
    def conf(self) -> TensorLike | None:
        if self.data.shape[-1] < 3:
            return None
        return self.data[..., 2]

    @property
    def has_visible(self) -> TensorLike:
        conf = self.conf
        if conf is None:
            if isinstance(self.data, torch.Tensor):
                return torch.ones(self.data.shape[:-1], dtype=torch.bool, device=self.data.device)
            return np.ones(self.data.shape[:-1], dtype=bool)
        return conf > 0


class Points(_TensorPayload):
    """Wrap point-localization predictions for a single image.

    Data shape is ``(N, 4)`` with rows ``x, y, class, confidence``.
    Coordinates are absolute image pixels unless accessed through ``xyn``.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        _require_tensorlike(data, "points")
        if data.ndim == 1:
            if isinstance(data, torch.Tensor):
                data = data.unsqueeze(0)
            else:
                data = data[None, :]
        if data.ndim != 2 or data.shape[-1] != 4:
            raise ValueError(
                f"expected (N, 4) point rows but got shape {tuple(data.shape)}: "
                "x, y, class, confidence"
            )
        super().__init__(data, orig_shape)

    @property
    def xy(self) -> TensorLike:
        return self.data[:, :2]

    @property
    def xyn(self) -> TensorLike:
        if self.orig_shape is None:
            raise ValueError("orig_shape is required for normalized point coordinates")
        h, w = self.orig_shape
        xy = self.xy
        if isinstance(xy, torch.Tensor):
            scale = torch.tensor([w, h], dtype=xy.dtype, device=xy.device)
        else:
            scale = np.array([w, h], dtype=xy.dtype)
        return xy / scale

    @property
    def cls(self) -> TensorLike:
        return self.data[:, 2]

    @property
    def conf(self) -> TensorLike:
        return self.data[:, 3]

    def __repr__(self) -> str:
        return (
            f"Points(n={len(self)}, "
            f"shape={tuple(self.data.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class Probs(_WholeImagePayload):
    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        _require_tensorlike(data, "probabilities")
        if data.ndim != 1 or int(data.shape[0]) == 0:
            raise ValueError(
                f"expected a non-empty classification vector with shape (C,), got {tuple(data.shape)}"
            )
        super().__init__(data, orig_shape)

    @property
    def top1(self) -> int:
        values = _numpy(self.data)
        return int(np.argmax(values))

    @property
    def top5(self) -> List[int]:
        values = _numpy(self.data)
        return np.argsort(values)[-5:][::-1].astype(int).tolist()

    @property
    def top1conf(self):
        return self.data[self.top1]

    @property
    def top5conf(self):
        indices = self.top5
        if isinstance(self.data, torch.Tensor):
            return self.data[torch.tensor(indices, device=self.data.device)]
        return self.data[indices]


class SemanticMask(_WholeImagePayload):
    """Dense semantic segmentation map for a single image.

    Data shape is ``(H, W)`` integer class IDs on the original image canvas.
    ``255`` is the ignore/void value and never counts as a class.
    """

    IGNORE_INDEX = 255

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        _require_tensorlike(data, "semantic mask")
        if data.ndim != 2:
            raise ValueError(
                f"expected (H, W) semantic class map but got shape {tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
        if int(data.shape[0]) <= 0 or int(data.shape[1]) <= 0:
            raise ValueError("semantic class map dimensions must be positive")
        _validate_nonnegative_id_map(data, "semantic class map")
        super().__init__(data, orig_shape)

    @property
    def classes(self) -> List[int]:
        """Sorted class IDs present in the map, excluding the ignore value."""
        values = np.unique(_numpy(self.data))
        return [int(v) for v in values if int(v) != self.IGNORE_INDEX]

    def class_mask(self, class_id: int) -> TensorLike:
        """Boolean ``(H, W)`` mask selecting the pixels of one class."""
        return self.data == class_id

    def __repr__(self) -> str:
        return (
            f"SemanticMask(shape={tuple(self.data.shape)}, "
            f"classes={len(self.classes)}, orig_shape={self.orig_shape})"
        )


class PanopticSegmentation(_WholeImagePayload):
    """Panoptic segmentation result for a single image.

    Panoptic segmentation assigns every pixel exactly one non-overlapping
    segment, unifying "stuff" (amorphous background regions) and "things"
    (countable object instances). ``data`` is a ``(H, W)`` integer segment-id
    map on the original image canvas; ``segments_info`` describes each segment
    id that appears in the map.

    ``segments_info`` is a list of dicts, one per segment, each with at least::

        {"id": int, "category_id": int}

    where ``id`` matches a value in the map and ``category_id`` is the class
    index in the model's ``names``.

    thing-vs-stuff is a *per-category* property of the label set (mirroring the
    COCO-panoptic GT, where ``isthing`` lives on the ``categories`` list, not on
    per-segment ``segments_info`` entries), so the category metadata is the
    source of truth. As a convenience a prediction payload MAY denormalize it
    onto each segment (``"isthing": bool``, derived from ``category_id``); it is
    optional and, when present, must agree with the category-level map. This
    keeps the payload consistent with the GT contract in
    ``docs/dataset_schema.md`` and puts the derive-from-category responsibility
    on the producer (a model's ``_postprocess_predictions`` /
    ``PanopticValidator``), not on downstream consumers.

    ``predict`` populates this slot whenever a model family's ``_postprocess``
    returns a ``panoptic`` segment-id map plus ``segments_info``; evaluation is
    ``PanopticValidator`` (Panoptic Quality) over a ``PanopticDataset``.
    ``predict(save=True)`` renders the map via ``draw_panoptic`` and
    ``Results.summary`` reports one row per segment.
    """

    IGNORE_INDEX = 0  # COCO panoptic convention: segment id 0 is unlabeled/void.

    def __init__(
        self,
        data: TensorLike,
        segments_info: Optional[List[dict]] = None,
        orig_shape: Tuple[int, int] | None = None,
    ):
        _require_tensorlike(data, "panoptic map")
        if data.ndim != 2:
            raise ValueError(
                f"expected (H, W) panoptic segment-id map but got shape "
                f"{tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
        if int(data.shape[0]) <= 0 or int(data.shape[1]) <= 0:
            raise ValueError("panoptic map dimensions must be positive")
        _validate_nonnegative_id_map(data, "panoptic map")
        super().__init__(data, orig_shape)
        # Plain-Python metadata; carried verbatim across device/array moves.
        self.segments_info: List[dict] = list(segments_info or [])
        self._validate_segments_info()

    def _validate_segments_info(self) -> None:
        map_ids = set(self.segment_ids)
        info_ids: List[int] = []
        for index, segment in enumerate(self.segments_info):
            if not isinstance(segment, dict):
                raise ValueError(f"segments_info[{index}] must be a mapping")
            if "id" not in segment or "category_id" not in segment:
                raise ValueError(
                    f"segments_info[{index}] must contain 'id' and 'category_id'"
                )
            segment_id = segment["id"]
            category_id = segment["category_id"]
            if isinstance(segment_id, (bool, np.bool_)) or not isinstance(
                segment_id, (int, np.integer)
            ):
                raise ValueError(f"segments_info[{index}].id must be a positive integer")
            if isinstance(category_id, (bool, np.bool_)) or not isinstance(
                category_id, (int, np.integer)
            ):
                raise ValueError(
                    f"segments_info[{index}].category_id must be a non-negative integer"
                )
            if int(segment_id) <= self.IGNORE_INDEX:
                raise ValueError(f"segments_info[{index}].id must be positive")
            if int(category_id) < 0:
                raise ValueError(
                    f"segments_info[{index}].category_id must be non-negative"
                )
            if "isthing" in segment and not isinstance(
                segment["isthing"], (bool, np.bool_)
            ):
                raise ValueError(f"segments_info[{index}].isthing must be boolean")
            info_ids.append(int(segment_id))
        if len(info_ids) != len(set(info_ids)):
            raise ValueError("segments_info contains duplicate segment ids")
        if set(info_ids) != map_ids:
            missing = sorted(map_ids - set(info_ids))
            extra = sorted(set(info_ids) - map_ids)
            raise ValueError(
                "panoptic map and segments_info ids must match exactly; "
                f"missing metadata={missing}, absent from map={extra}"
            )

    @property
    def segment_ids(self) -> List[int]:
        """Sorted segment ids present in the map, excluding the void id."""
        values = np.unique(_numpy(self.data))
        return [int(v) for v in values if int(v) != self.IGNORE_INDEX]

    def segment_mask(self, segment_id: int) -> TensorLike:
        """Boolean ``(H, W)`` mask selecting the pixels of one segment id."""
        return self.data == segment_id

    # segments_info is not tensor data, so the base _TensorPayload moves (which
    # rebuild via ``self.__class__(data, orig_shape)``) would drop it. Override
    # the move/slice methods to carry it through.
    def to(self, *args, **kwargs):
        return self.__class__(_move(self.data, *args, **kwargs), self.segments_info, self.orig_shape)

    def cpu(self):
        return self.__class__(_cpu(self.data), self.segments_info, self.orig_shape)

    def cuda(self):
        return self.__class__(_cuda(self.data), self.segments_info, self.orig_shape)

    def numpy(self):
        return self.__class__(_numpy(self.data), self.segments_info, self.orig_shape)

    def __getitem__(self, idx):
        indices, _ = _selector_indices(idx, 1)
        if indices != [0]:
            raise IndexError("whole-image payload selection must contain index 0")
        return self.__class__(self.data, self.segments_info, self.orig_shape)

    def __repr__(self) -> str:
        return (
            f"PanopticSegmentation(shape={tuple(self.data.shape)}, "
            f"segments={len(self.segment_ids)}, orig_shape={self.orig_shape})"
        )


class DepthMap(_WholeImagePayload):
    """Dense relative inverse-depth map for a single image.

    Data shape is ``(H, W)`` float values on the original image canvas. Higher
    values mean closer to the camera. Values are relative, not metric meters.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        _require_tensorlike(data, "depth map")
        if data.ndim != 2:
            raise ValueError(
                f"expected (H, W) depth map but got shape {tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
        if int(data.shape[0]) <= 0 or int(data.shape[1]) <= 0:
            raise ValueError("depth map dimensions must be positive")
        super().__init__(data, orig_shape)

    def _finite_values(self) -> np.ndarray:
        values = np.asarray(_numpy(self.data), dtype=np.float32)
        return values[np.isfinite(values)]

    @property
    def min(self) -> float:
        values = self._finite_values()
        return float(values.min()) if values.size else 0.0

    @property
    def max(self) -> float:
        values = self._finite_values()
        return float(values.max()) if values.size else 0.0

    @property
    def mean(self) -> float:
        values = self._finite_values()
        return float(values.mean()) if values.size else 0.0

    def normalized(self) -> TensorLike:
        """Depth map rescaled to ``[0, 1]`` over finite values."""
        data = self.data
        lo, hi = self.min, self.max
        if hi - lo <= 0:
            return (
                torch.zeros_like(data)
                if isinstance(data, torch.Tensor)
                else np.zeros_like(data)
            )
        normalized = (data - lo) / (hi - lo)
        if isinstance(normalized, torch.Tensor):
            return torch.where(torch.isfinite(normalized), normalized, torch.zeros_like(normalized))
        return np.where(np.isfinite(normalized), normalized, np.zeros_like(normalized))

    def __repr__(self) -> str:
        return (
            f"DepthMap(shape={tuple(self.data.shape)}, "
            f"range=({self.min:.4g}, {self.max:.4g}), "
            f"orig_shape={self.orig_shape})"
        )


class RestoredImage(_WholeImagePayload):
    """Dense restored RGB image for a single input.

    Data shape is ``(H, W, 3)`` uint8 RGB on the original image canvas.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        _require_tensorlike(data, "restored image")
        if data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError(
                f"expected (H, W, 3) restored RGB image but got shape {tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
        if int(data.shape[0]) <= 0 or int(data.shape[1]) <= 0:
            raise ValueError("restored image dimensions must be positive")
        super().__init__(data, orig_shape)

    @property
    def array(self) -> np.ndarray:
        """Return the raw HWC uint8 RGB ndarray."""

        arr = np.asarray(_numpy(self.data))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def save(self, path: str | Path) -> None:
        """Write the restored RGB image to disk."""

        from PIL import Image

        path = Path(path)
        if path.parent and path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.array, mode="RGB").save(path)

    def __repr__(self) -> str:
        return (
            f"RestoredImage(shape={tuple(self.data.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class Matte(_WholeImagePayload):
    """Dense soft alpha matte for a single image.

    Data shape is ``(H, W)`` float32 in ``[0, 1]`` on the original image canvas.
    ``1`` is fully foreground (opaque), ``0`` is fully background (transparent).
    A soft matte subsumes a hard background-removal mask (threshold at 0.5) and
    carries the anti-aliased edges (hair, fur) that binary masks discard.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        _require_tensorlike(data, "alpha matte")
        if data.ndim != 2:
            raise ValueError(
                f"expected (H, W) alpha matte but got shape {tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
        if int(data.shape[0]) <= 0 or int(data.shape[1]) <= 0:
            raise ValueError("alpha matte dimensions must be positive")
        super().__init__(data, orig_shape)

    @property
    def array(self) -> np.ndarray:
        """Return the raw ``(H, W)`` float32 alpha matte clamped to ``[0, 1]``."""
        arr = np.asarray(_numpy(self.data), dtype=np.float32)
        return np.clip(arr, 0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"Matte(shape={tuple(self.data.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class OCRRegions(_TensorPayload):
    """Located text regions with transcripts for a single image.

    ``data`` is an ``(N, 4, 2)`` float array of 4-point polygons in
    original-image pixel coordinates, ordered top-left, top-right,
    bottom-right, bottom-left per region. Regions are in reading order
    (top to bottom, then left to right). ``texts`` is the list of N
    transcripts; ``confidence`` is the per-region recognition score and
    ``det_confidence`` the detection score, both ``(N,)`` float arrays.

    Detection quads are genuine polygons (rotated text), so they do not
    populate ``Results.boxes``; use :attr:`xyxy` for axis-aligned hulls.
    """

    def __init__(
        self,
        data: TensorLike,
        texts: Optional[List[str]] = None,
        confidence: TensorLike | None = None,
        det_confidence: TensorLike | None = None,
        orig_shape: Tuple[int, int] | None = None,
    ):
        _require_tensorlike(data, "OCR polygons")
        if int(data.numel() if isinstance(data, torch.Tensor) else data.size) == 0:
            data = data.reshape(0, 4, 2)
        if data.ndim != 3 or data.shape[-2:] != (4, 2):
            raise ValueError(
                f"expected (N, 4, 2) OCR polygons but got shape {tuple(data.shape)}"
            )
        super().__init__(data, orig_shape)
        n = int(data.shape[0])
        self.texts: List[str] = list(texts) if texts is not None else [""] * n
        if len(self.texts) != n:
            raise ValueError(
                f"expected {n} transcripts to match {n} polygons, got {len(self.texts)}"
            )

        def _as_scores(values):
            if values is None:
                if isinstance(data, torch.Tensor):
                    return torch.zeros(n, dtype=torch.float32, device=data.device)
                return np.zeros(n, dtype=np.float32)
            if isinstance(values, torch.Tensor):
                values = values.reshape(-1).float()
                if isinstance(data, torch.Tensor):
                    values = values.to(device=data.device)
                else:
                    values = values.detach().cpu().numpy()
            else:
                values = np.asarray(values, dtype=np.float32).reshape(-1)
                if isinstance(data, torch.Tensor):
                    values = torch.as_tensor(values, dtype=torch.float32, device=data.device)
            if int(values.shape[0]) != n:
                raise ValueError(
                    f"expected {n} scores to match {n} polygons, got {int(values.shape[0])}"
                )
            return values

        self._conf = _as_scores(confidence)
        self._det_conf = _as_scores(det_confidence)

    @property
    def polygons(self) -> TensorLike:
        return self.data

    @property
    def conf(self) -> TensorLike:
        return self._conf

    @property
    def det_conf(self) -> TensorLike:
        return self._det_conf

    @property
    def xyxy(self) -> TensorLike:
        """Axis-aligned bounding boxes of the polygons, ``(N, 4)``."""
        polys = self.data
        if isinstance(polys, torch.Tensor):
            if len(self) == 0:
                return polys.new_zeros((0, 4))
            x = polys[..., 0]
            y = polys[..., 1]
            return torch.stack(
                [
                    x.min(dim=1).values,
                    y.min(dim=1).values,
                    x.max(dim=1).values,
                    y.max(dim=1).values,
                ],
                dim=1,
            )
        if len(self) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x = polys[..., 0]
        y = polys[..., 1]
        return np.stack(
            [x.min(axis=1), y.min(axis=1), x.max(axis=1), y.max(axis=1)], axis=1
        )

    # texts/scores are extra payload the base _TensorPayload moves (which
    # rebuild via ``self.__class__(data, orig_shape)``) would drop. Override
    # the move/slice methods to carry them through, mirroring
    # PanopticSegmentation.segments_info.
    def to(self, *args, **kwargs):
        return self.__class__(
            _move(self.data, *args, **kwargs),
            self.texts,
            _move(self._conf, *args, **kwargs),
            _move(self._det_conf, *args, **kwargs),
            self.orig_shape,
        )

    def cpu(self):
        return self.__class__(
            _cpu(self.data), self.texts, _cpu(self._conf), _cpu(self._det_conf), self.orig_shape
        )

    def cuda(self):
        return self.__class__(
            _cuda(self.data), self.texts, _cuda(self._conf), _cuda(self._det_conf), self.orig_shape
        )

    def numpy(self):
        return self.__class__(
            _numpy(self.data), self.texts, _numpy(self._conf), _numpy(self._det_conf), self.orig_shape
        )

    def __getitem__(self, idx):
        indices, _ = _selector_indices(idx, len(self))
        return self.__class__(
            _take_first(self.data, indices),
            [self.texts[i] for i in indices],
            _take_first(self._conf, indices),
            _take_first(self._det_conf, indices),
            self.orig_shape,
        )

    def __repr__(self) -> str:
        return (
            f"OCRRegions(n={len(self)}, "
            f"shape={tuple(self.data.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class OBB(_TensorPayload):
    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        _require_tensorlike(data, "oriented boxes")
        if data.ndim == 1:
            data = data[None, :]
        if data.ndim != 2:
            raise ValueError(
                f"expected OBB rows with shape (N, 7|8), got {tuple(data.shape)}"
            )
        n = data.shape[-1]
        if n not in {7, 8}:
            raise ValueError(
                f"expected 7 or 8 OBB values but got {n}: "
                "xywhr, optional track_id, conf, cls"
            )
        super().__init__(data, orig_shape)
        if self.is_track:
            _validate_track_ids(self.data[:, -3], "OBB track ids")

    def to(self, *args, **kwargs):
        converted = self.__class__(_move(self.data, *args, **kwargs), self.orig_shape)
        if self.id is not None and converted.id is not None:
            before = np.asarray(_numpy(self.id), dtype=np.float64)
            after = np.asarray(_numpy(converted.id), dtype=np.float64)
            if not np.array_equal(before, after):
                raise ValueError("requested OBB dtype conversion would change track ids")
        return converted

    @property
    def xywhr(self) -> TensorLike:
        return self.data[:, :5]

    @property
    def is_track(self) -> bool:
        return self.data.shape[-1] == 8

    @property
    def id(self) -> TensorLike | None:
        return self.data[:, -3] if self.is_track else None

    @property
    def conf(self) -> TensorLike:
        return self.data[:, -2]

    @property
    def cls(self) -> TensorLike:
        return self.data[:, -1]

    @property
    def xyxyxyxy(self) -> TensorLike:
        box = self.xywhr
        if isinstance(box, torch.Tensor):
            xy = box[:, :2]
            w = box[:, 2] / 2
            h = box[:, 3] / 2
            angle = box[:, 4]
            cos = torch.cos(angle)
            sin = torch.sin(angle)
            corners = torch.stack(
                [
                    torch.stack([-w, -h], dim=1),
                    torch.stack([w, -h], dim=1),
                    torch.stack([w, h], dim=1),
                    torch.stack([-w, h], dim=1),
                ],
                dim=1,
            )
            rot = torch.stack(
                [
                    torch.stack([cos, -sin], dim=1),
                    torch.stack([sin, cos], dim=1),
                ],
                dim=1,
            )
            return torch.matmul(corners, rot.transpose(1, 2)) + xy[:, None, :]

        xy = box[:, :2]
        w = box[:, 2] / 2
        h = box[:, 3] / 2
        angle = box[:, 4]
        cos = np.cos(angle)
        sin = np.sin(angle)
        corners = np.stack(
            [
                np.stack([-w, -h], axis=1),
                np.stack([w, -h], axis=1),
                np.stack([w, h], axis=1),
                np.stack([-w, h], axis=1),
            ],
            axis=1,
        )
        rot = np.stack(
            [
                np.stack([cos, -sin], axis=1),
                np.stack([sin, cos], axis=1),
            ],
            axis=1,
        )
        return np.matmul(corners, np.swapaxes(rot, 1, 2)) + xy[:, None, :]

    @property
    def xyxyxyxyn(self) -> TensorLike:
        if self.orig_shape is None:
            raise ValueError("orig_shape is required for normalized OBB coordinates")
        h, w = self.orig_shape
        corners = self.xyxyxyxy
        if isinstance(corners, torch.Tensor):
            scale = torch.tensor([w, h], dtype=corners.dtype, device=corners.device)
        else:
            scale = np.array([w, h], dtype=corners.dtype)
        return corners / scale

    @property
    def xyxy(self) -> TensorLike:
        corners = self.xyxyxyxy
        x = corners[..., 0]
        y = corners[..., 1]
        if isinstance(corners, torch.Tensor):
            return torch.stack(
                [x.min(dim=1).values, y.min(dim=1).values, x.max(dim=1).values, y.max(dim=1).values],
                dim=1,
            )
        return np.stack([x.min(axis=1), y.min(axis=1), x.max(axis=1), y.max(axis=1)], axis=1)


class Gaze(_TensorPayload):
    """Per-face gaze angles in radians.

    Data shape: (N, 2) where column 0 is pitch and column 1 is yaw.
    Aligned row-by-row with the parent Results.boxes (face boxes).
    The L2CS convention is used: positive yaw rotates the gaze toward
    the subject's left, positive pitch rotates it downward.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        _require_tensorlike(data, "gaze angles")
        if data.ndim == 1:
            if isinstance(data, torch.Tensor):
                data = data.unsqueeze(0)
            else:
                data = data[None, :]
        if data.ndim != 2 or data.shape[-1] != 2:
            raise ValueError(
                f"expected (N, 2) pitch/yaw, got shape {tuple(data.shape)}"
            )
        super().__init__(data, orig_shape)

    @property
    def pitch(self) -> TensorLike:
        return self.data[..., 0]

    @property
    def yaw(self) -> TensorLike:
        return self.data[..., 1]

    @property
    def pitch_deg(self) -> TensorLike:
        return self.pitch * (180.0 / math.pi)

    @property
    def yaw_deg(self) -> TensorLike:
        return self.yaw * (180.0 / math.pi)

    @property
    def direction_3d(self) -> TensorLike:
        """Unit gaze direction in the camera frame: (N, 3), columns (x, y, z).

        Matches upstream L2CS-Net ``gazeto3d``: (-cos(p)*sin(y), -sin(p), -cos(p)*cos(y)).
        """
        p, y = self.pitch, self.yaw
        if isinstance(self.data, torch.Tensor):
            cos_p, sin_p = torch.cos(p), torch.sin(p)
            cos_y, sin_y = torch.cos(y), torch.sin(y)
            return torch.stack([-cos_p * sin_y, -sin_p, -cos_p * cos_y], dim=-1)
        cos_p, sin_p = np.cos(p), np.sin(p)
        cos_y, sin_y = np.cos(y), np.sin(y)
        return np.stack([-cos_p * sin_y, -sin_p, -cos_p * cos_y], axis=-1)

    def __repr__(self) -> str:
        return (
            f"Gaze(n={len(self)}, "
            f"shape={tuple(self.data.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class Results:
    """Single-image result with flat detection/segmentation slots."""

    _keys = (
        "boxes",
        "masks",
        "probs",
        "keypoints",
        "obb",
        "gaze",
        "points",
        "semantic_mask",
        "panoptic",
        "depth_map",
        "restored",
        "matte",
        "ocr",
    )
    _instance_keys = ("boxes", "masks", "keypoints", "obb", "gaze", "points", "ocr")
    _whole_image_keys = (
        "probs",
        "semantic_mask",
        "panoptic",
        "depth_map",
        "restored",
        "matte",
    )

    def __init__(
        self,
        boxes: Optional[Boxes],
        orig_shape: Tuple[int, int],
        path: Optional[str] = None,
        names: Optional[Dict[int, str]] = None,
        masks: Optional[Masks] = None,
        keypoints: Optional[Keypoints] = None,
        probs: Optional[Probs] = None,
        obb: Optional[OBB] = None,
        gaze: Optional[Gaze] = None,
        points: Optional[Points] = None,
        semantic_mask: Optional[SemanticMask] = None,
        panoptic: Optional[PanopticSegmentation] = None,
        depth_map: Optional[DepthMap] = None,
        restored: Optional[RestoredImage] = None,
        matte: Optional[Matte] = None,
        ocr: Optional[OCRRegions] = None,
        restore_scale: int = 1,
        speed: Optional[Dict[str, float]] = None,
        track_id: Optional[TensorLike] = None,
        frame_idx: Optional[int] = None,
        task: Optional[str] = None,
        saved_path: Optional[str] = None,
        tiled: bool = False,
        num_tiles: Optional[int] = None,
        tiles_path: Optional[str] = None,
        grid_path: Optional[str] = None,
    ):
        validated_shape = _validate_orig_shape(orig_shape)
        if validated_shape is None:
            raise ValueError("orig_shape is required for Results")
        orig_shape = validated_shape
        if boxes is not None and boxes.orig_shape is None:
            boxes.orig_shape = orig_shape
        if boxes is not None and track_id is not None:
            boxes = boxes.with_id(track_id)
        if boxes is not None and obb is not None and obb.id is not None:
            effective_track_id = track_id if track_id is not None else boxes.id
            if effective_track_id is None:
                track_id = obb.id
                boxes = boxes.with_id(track_id)
        if masks is not None and masks.orig_shape is None:
            masks.orig_shape = orig_shape
        if keypoints is not None and keypoints.orig_shape is None:
            keypoints.orig_shape = orig_shape
        if probs is not None and probs.orig_shape is None:
            probs.orig_shape = orig_shape
        if obb is not None and obb.orig_shape is None:
            obb.orig_shape = orig_shape
        if gaze is not None and gaze.orig_shape is None:
            gaze.orig_shape = orig_shape
        if points is not None and points.orig_shape is None:
            points.orig_shape = orig_shape
        if semantic_mask is not None and semantic_mask.orig_shape is None:
            semantic_mask.orig_shape = orig_shape
        if panoptic is not None and panoptic.orig_shape is None:
            panoptic.orig_shape = orig_shape
        if depth_map is not None and depth_map.orig_shape is None:
            depth_map.orig_shape = orig_shape
        if matte is not None and matte.orig_shape is None:
            matte.orig_shape = orig_shape
        if ocr is not None and ocr.orig_shape is None:
            ocr.orig_shape = orig_shape

        self.boxes = boxes
        self.masks = masks
        self.keypoints = keypoints
        self.probs = probs
        self.obb = obb
        self.gaze = gaze
        self.points = points
        self.semantic_mask = semantic_mask
        self.panoptic = panoptic
        self.depth_map = depth_map
        self.restored = restored
        self.matte = matte
        self.ocr = ocr
        # Integer upscale factor of a restore/super-resolution result: the
        # restored canvas is ``restore_scale`` times the input. 1 for
        # deblur/denoise and every non-restore task.
        self.restore_scale = _validate_integer(restore_scale, "restore_scale", minimum=1)
        self.orig_shape = orig_shape
        self.path = path
        self.names = names or {}
        self.speed = speed or {}
        self.track_id = (
            track_id
            if track_id is not None
            else (boxes.id if boxes is not None else None)
        )
        self.frame_idx = frame_idx
        self.saved_path = saved_path
        self.tiled = bool(tiled)
        self.num_tiles = (
            _validate_integer(num_tiles, "num_tiles", minimum=0)
            if num_tiles is not None
            else None
        )
        self.tiles_path = tiles_path
        self.grid_path = grid_path
        self.task = self._resolve_task(task)
        self._validate_contract()

    def _resolve_task(self, requested_task: Optional[str]) -> str:
        specific = []
        for key, payload_task in (
            ("masks", "segment"),
            ("keypoints", "pose"),
            ("probs", "classify"),
            ("obb", "obb"),
            ("gaze", "gaze"),
            ("points", "point"),
            ("semantic_mask", "semantic"),
            ("panoptic", "panoptic"),
            ("depth_map", "depth"),
            ("restored", "restore"),
            ("matte", "matte"),
            ("ocr", "ocr"),
        ):
            if getattr(self, key) is not None:
                specific.append((key, payload_task))
        task_values = {payload_task for _, payload_task in specific}
        explicit = normalize_task(requested_task) if requested_task is not None else None
        if explicit is not None:
            return str(explicit)
        if len(task_values) > 1:
            if self.boxes is not None:
                return "detect"
            details = ", ".join(f"{key}={value}" for key, value in specific)
            raise ValueError(
                "Results without boxes has ambiguous task payloads; pass task= explicitly: "
                f"{details}"
            )
        inferred = next(iter(task_values), None)
        return str(explicit or inferred or "detect")

    def _validate_contract(self) -> None:
        present_instance = [
            key for key in self._instance_keys if getattr(self, key) is not None
        ]
        present_whole_image = [
            key for key in self._whole_image_keys if getattr(self, key) is not None
        ]
        if present_instance and present_whole_image:
            raise ValueError(
                "Results cannot mix per-instance and whole-image payloads; "
                f"instance={present_instance}, whole_image={present_whole_image}"
            )
        if len(present_whole_image) > 1:
            raise ValueError(
                "Results may contain only one whole-image payload; "
                f"got {present_whole_image}"
            )
        for exclusive_key in ("points", "ocr"):
            if exclusive_key in present_instance and len(present_instance) > 1:
                raise ValueError(
                    f"{exclusive_key} is an exclusive Results payload and cannot be "
                    f"combined with {present_instance}"
                )
        if len(present_instance) > 1 and "boxes" not in present_instance:
            raise ValueError(
                "multiple per-instance Results payloads require boxes as their row anchor; "
                f"got {present_instance}"
            )

        whole_image_tasks = {
            "probs": "classify",
            "semantic_mask": "semantic",
            "panoptic": "panoptic",
            "depth_map": "depth",
            "restored": "restore",
            "matte": "matte",
        }
        if present_whole_image:
            expected_task = whole_image_tasks[present_whole_image[0]]
            if self.task != expected_task:
                raise ValueError(
                    f"Results task {self.task!r} conflicts with its {expected_task!r} "
                    f"whole-image payload"
                )

        instance_tasks = {
            "masks": "segment",
            "keypoints": "pose",
            "obb": "obb",
            "gaze": "gaze",
            "points": "point",
            "ocr": "ocr",
        }
        specific_instance_tasks = {
            instance_tasks[key] for key in present_instance if key != "boxes"
        }
        if len(specific_instance_tasks) == 1:
            expected_task = next(iter(specific_instance_tasks))
            if self.task != expected_task:
                raise ValueError(
                    f"Results task {self.task!r} conflicts with its {expected_task!r} payload"
                )
        elif len(specific_instance_tasks) > 1 and self.task != "detect":
            raise ValueError(
                "combined aligned instance payloads use the generic 'detect' task; "
                f"got task={self.task!r} and payload tasks={sorted(specific_instance_tasks)}"
            )

        required_instance_payload = {
            "segment": "masks",
            "pose": "keypoints",
            "obb": "obb",
            "gaze": "gaze",
        }.get(self.task)
        if (
            required_instance_payload is not None
            and self.boxes is not None
            and len(self.boxes) > 0
            and getattr(self, required_instance_payload) is None
        ):
            raise ValueError(
                f"non-empty {self.task} results require an aligned "
                f"{required_instance_payload} payload"
            )

        if self.task in {
            "classify",
            "semantic",
            "panoptic",
            "point",
            "depth",
            "restore",
            "matte",
            "ocr",
        } and self.boxes is not None:
            raise ValueError(f"{self.task} results must not fabricate or carry boxes")

        per_instance = {
            key: len(value)
            for key in self._instance_keys
            if (value := getattr(self, key)) is not None
        }
        if per_instance:
            expected = next(iter(per_instance.values()))
            mismatched = {key: count for key, count in per_instance.items() if count != expected}
            if mismatched:
                counts = ", ".join(f"{key}={count}" for key, count in per_instance.items())
                raise ValueError(f"per-instance Results payloads must align row-for-row; {counts}")

        if self.track_id is not None:
            _require_tensorlike(self.track_id, "track_id")
            expected = len(self.boxes) if self.boxes is not None else None
            if self.track_id.ndim != 1 or expected is None or len(self.track_id) != expected:
                shape = tuple(self.track_id.shape)
                raise ValueError(
                    "track_id requires boxes and must have one value per box; "
                    f"got shape {shape}"
                )
        if (
            self.boxes is not None
            and self.obb is not None
            and self.obb.id is not None
            and self.track_id is not None
            and not np.array_equal(_numpy(self.obb.id), _numpy(self.track_id))
        ):
            raise ValueError("OBB track ids must match Results/Boxes track ids")
        if self.task == "obb" and self.boxes is not None and self.obb is not None:
            boxes_conf = np.asarray(_numpy(self.boxes.conf), dtype=np.float64)
            obb_conf = np.asarray(_numpy(self.obb.conf), dtype=np.float64)
            boxes_cls = np.asarray(_numpy(self.boxes.cls))
            obb_cls = np.asarray(_numpy(self.obb.cls))
            if not np.allclose(boxes_conf, obb_conf, rtol=1e-5, atol=1e-7):
                raise ValueError("OBB confidence values must match aligned Boxes confidence")
            if not np.array_equal(boxes_cls, obb_cls):
                raise ValueError("OBB class ids must match aligned Boxes class ids")

        for key in ("boxes", "masks", "keypoints", "probs", "obb", "gaze", "points", "ocr"):
            value = getattr(self, key)
            if value is not None and value.orig_shape != self.orig_shape:
                raise ValueError(
                    f"{key}.orig_shape {value.orig_shape} does not match Results.orig_shape "
                    f"{self.orig_shape}"
                )
        if self.masks is not None and tuple(self.masks.data.shape[-2:]) != self.orig_shape:
            raise ValueError(
                f"masks canvas {tuple(self.masks.data.shape[-2:])} does not match "
                f"Results.orig_shape {self.orig_shape}"
            )
        for key in ("semantic_mask", "panoptic", "depth_map", "matte"):
            value = getattr(self, key)
            if value is not None:
                if value.orig_shape != self.orig_shape or tuple(value.data.shape) != self.orig_shape:
                    raise ValueError(
                        f"{key} must use the Results original canvas {self.orig_shape}; "
                        f"got data {tuple(value.data.shape)} and orig_shape {value.orig_shape}"
                    )
        if self.restored is not None:
            restored_shape = tuple(self.restored.data.shape[:2])
            expected_restored = (
                self.orig_shape[0] * self.restore_scale,
                self.orig_shape[1] * self.restore_scale,
            )
            if restored_shape != expected_restored or self.restored.orig_shape != restored_shape:
                raise ValueError(
                    "restored image canvas must equal Results.orig_shape multiplied by "
                    f"restore_scale={self.restore_scale}; expected {expected_restored}, "
                    f"got {restored_shape}"
                )
        if self.task != "restore" and self.restore_scale != 1:
            raise ValueError("restore_scale may differ from 1 only for restore results")

    def _new(self, **overrides) -> "Results":
        data = {
            "boxes": self.boxes,
            "orig_shape": self.orig_shape,
            "path": self.path,
            "names": self.names,
            "masks": self.masks,
            "keypoints": self.keypoints,
            "probs": self.probs,
            "obb": self.obb,
            "gaze": self.gaze,
            "points": self.points,
            "semantic_mask": self.semantic_mask,
            "panoptic": self.panoptic,
            "depth_map": self.depth_map,
            "restored": self.restored,
            "matte": self.matte,
            "ocr": self.ocr,
            "restore_scale": self.restore_scale,
            "speed": dict(self.speed),
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "task": self.task,
            "saved_path": self.saved_path,
            "tiled": self.tiled,
            "num_tiles": self.num_tiles,
            "tiles_path": self.tiles_path,
            "grid_path": self.grid_path,
        }
        data.update(overrides)
        return Results(**data)

    def to(self, *args, **kwargs) -> "Results":
        return self._apply("to", *args, **kwargs)

    def cpu(self) -> "Results":
        return self._apply("cpu")

    def cuda(self) -> "Results":
        return self.to("cuda")

    def numpy(self) -> "Results":
        return self._apply("numpy")

    def _apply(self, method: str, *args, **kwargs) -> "Results":
        overrides = {}
        for key in self._keys:
            value = getattr(self, key)
            overrides[key] = getattr(value, method)(*args, **kwargs) if value is not None else None

        if method == "cpu":
            overrides["track_id"] = _cpu(self.track_id)
        elif method == "numpy":
            overrides["track_id"] = _numpy(self.track_id)
        elif method == "to":
            moved_boxes = overrides.get("boxes")
            if moved_boxes is not None and moved_boxes.id is not None:
                overrides["track_id"] = moved_boxes.id
            elif moved_boxes is not None:
                overrides["track_id"] = _move_ids_like(
                    self.track_id, moved_boxes.xyxy
                )
            else:
                overrides["track_id"] = self.track_id
        elif method == "__getitem__":
            overrides["track_id"] = _slice_first(self.track_id, args[0])

        return self._new(**overrides)

    def _select(self, idx) -> "Results":
        indices, _ = _selector_indices(idx, len(self))
        has_whole_image_payload = any(
            getattr(self, key) is not None for key in self._whole_image_keys
        )
        if has_whole_image_payload and not indices:
            overrides = {key: None for key in self._keys}
            overrides["track_id"] = None
            return self._new(**overrides)
        return self._apply("__getitem__", indices)

    def __getitem__(self, idx) -> "Results":
        return self._select(idx)

    def __iter__(self) -> Iterator["Results"]:
        for index in range(len(self)):
            yield self[index]

    def update(
        self,
        boxes: Optional[Boxes] = None,
        masks: Optional[Masks] = None,
        probs: Optional[Probs] = None,
        keypoints: Optional[Keypoints] = None,
        obb: Optional[OBB] = None,
        gaze: Optional[Gaze] = None,
        points: Optional[Points] = None,
        semantic_mask: Optional[SemanticMask] = None,
        panoptic: Optional[PanopticSegmentation] = None,
        depth_map: Optional[DepthMap] = None,
        restored: Optional[RestoredImage] = None,
        matte: Optional[Matte] = None,
        ocr: Optional[OCRRegions] = None,
        restore_scale: Optional[int] = None,
        track_id: Optional[TensorLike] = None,
        task: Optional[str] = None,
    ) -> "Results":
        candidate_data = {
            "boxes": self.boxes,
            "orig_shape": self.orig_shape,
            "path": self.path,
            "names": self.names,
            "masks": self.masks,
            "keypoints": self.keypoints,
            "probs": self.probs,
            "obb": self.obb,
            "gaze": self.gaze,
            "points": self.points,
            "semantic_mask": self.semantic_mask,
            "panoptic": self.panoptic,
            "depth_map": self.depth_map,
            "restored": self.restored,
            "matte": self.matte,
            "ocr": self.ocr,
            "restore_scale": self.restore_scale,
            "speed": dict(self.speed),
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "saved_path": self.saved_path,
            "tiled": self.tiled,
            "num_tiles": self.num_tiles,
            "tiles_path": self.tiles_path,
            "grid_path": self.grid_path,
        }
        replacements = {
            "boxes": boxes,
            "masks": masks,
            "probs": probs,
            "keypoints": keypoints,
            "obb": obb,
            "gaze": gaze,
            "points": points,
            "semantic_mask": semantic_mask,
            "panoptic": panoptic,
            "depth_map": depth_map,
            "restored": restored,
            "matte": matte,
            "ocr": ocr,
            "restore_scale": restore_scale,
            "track_id": track_id,
        }
        candidate_data.update(
            {key: value for key, value in replacements.items() if value is not None}
        )

        requested_task = task
        if requested_task is None and self.task != "detect":
            has_specific_payload = any(
                candidate_data[key] is not None for key in self._keys if key != "boxes"
            )
            if not has_specific_payload:
                requested_task = self.task
        candidate_data["task"] = requested_task
        candidate = Results(**candidate_data)
        for key, value in candidate.__dict__.items():
            setattr(self, key, value)
        return self

    def cutout(self, image: Any = None) -> np.ndarray:
        """Return an RGBA ``(H, W, 4)`` uint8 cutout: source RGB + matte alpha.

        The alpha channel is the soft matte scaled to ``[0, 255]``. The RGB is
        taken from ``image`` when given (a PIL image or ``HxWx3`` array), else
        reloaded from ``self.path``. Only valid for matte results.
        """
        if self.matte is None:
            raise ValueError("cutout() is only defined for matte results (Results.matte is None).")
        alpha = self.matte.array  # (H, W) float32 in [0, 1]
        h, w = alpha.shape
        rgb = self._source_rgb(image, (h, w))
        alpha_u8 = np.rint(alpha * 255.0).astype(np.uint8)
        return np.dstack([rgb, alpha_u8])

    def _source_rgb(self, image: Any, hw: Tuple[int, int]) -> np.ndarray:
        """Load the source image as an ``HxWx3`` uint8 RGB array on the matte canvas."""
        from PIL import Image

        from libreyolo.utils.image_loader import ImageLoader

        h, w = hw
        if image is None:
            if not self.path:
                raise ValueError(
                    "cutout()/save() needs the source image but Results.path is unset; "
                    "pass image=<PIL.Image or HxWx3 array>."
                )
            rgb = np.asarray(ImageLoader.load(self.path))
        elif isinstance(image, Image.Image):
            rgb = np.asarray(ImageLoader.load(image))
        else:
            rgb = np.asarray(ImageLoader.load(image, color_format="rgb"))
        if rgb.shape[:2] != (h, w):
            rgb = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((w, h), Image.BILINEAR))
        return rgb.astype(np.uint8)

    def plot(self, image: Any = None):
        """Render every supported task to a new RGB :class:`PIL.Image.Image`.

        ``image`` accepts the same local, remote, PIL, NumPy, and tensor inputs
        as :class:`libreyolo.utils.image_loader.ImageLoader`. When no source is
        available, a black canvas of ``orig_shape`` is used. Restore results
        return their restored RGB canvas; matte results require a source so the
        alpha preview represents the actual cutout.
        """
        from PIL import Image, ImageDraw

        from libreyolo.utils.drawing import (
            draw_boxes,
            draw_depth_map,
            draw_gaze_arrows,
            draw_keypoints,
            draw_masks,
            draw_matte,
            draw_obb,
            draw_ocr_regions,
            draw_panoptic,
            draw_points,
            draw_semantic_mask,
        )
        from libreyolo.utils.image_loader import ImageLoader

        if self.restored is not None:
            return Image.fromarray(self.restored.array, mode="RGB")

        h, w = self.orig_shape
        source = image if image is not None else self.path
        if self.matte is not None:
            rgb = self._source_rgb(image, self.orig_shape)
            return draw_matte(Image.fromarray(rgb, mode="RGB"), self.matte.array)
        if source is None:
            rendered = Image.new("RGB", (w, h), color=(0, 0, 0))
        else:
            rendered = ImageLoader.load(source).resize((w, h), Image.BILINEAR)

        if self.semantic_mask is not None:
            return draw_semantic_mask(rendered, _numpy(self.semantic_mask.data))
        if self.panoptic is not None:
            return draw_panoptic(
                rendered,
                _numpy(self.panoptic.data),
                self.panoptic.segments_info,
                class_names=self.names,
            )
        if self.depth_map is not None:
            return draw_depth_map(rendered, _numpy(self.depth_map.data))
        if self.ocr is not None:
            ocr = self.ocr.numpy()
            return draw_ocr_regions(rendered, ocr.data, ocr.texts, ocr.conf)
        if self.points is not None:
            points = self.points.numpy()
            return draw_points(
                rendered,
                points.xy,
                points.conf.tolist(),
                points.cls.tolist(),
                class_names=self.names,
            )
        if self.probs is not None:
            row = self.summary(decimals=2)[0]
            label = f"{row['name']}: {row['confidence']:.2f}"
            draw = ImageDraw.Draw(rendered)
            bbox = draw.textbbox((8, 8), label)
            draw.rectangle((4, 4, bbox[2] + 4, bbox[3] + 4), fill=(15, 23, 42))
            draw.text((8, 8), label, fill="white")
            return rendered

        classes = None
        if self.boxes is not None:
            boxes = self.boxes.numpy()
            classes = boxes.cls.tolist()
            if self.masks is not None:
                rendered = draw_masks(
                    rendered,
                    np.asarray(_numpy(self.masks.data)),
                    classes,
                )
            if self.obb is None:
                track_ids = (
                    np.asarray(_numpy(self.track_id)).tolist()
                    if self.track_id is not None
                    else None
                )
                rendered = draw_boxes(
                    rendered,
                    boxes.xyxy.tolist(),
                    boxes.conf.tolist(),
                    classes,
                    class_names=self.names,
                    track_ids=track_ids,
                )
        elif self.masks is not None:
            classes = [0] * len(self.masks)
            rendered = draw_masks(rendered, np.asarray(_numpy(self.masks.data)), classes)

        if self.obb is not None:
            obb = self.obb.numpy()
            obb_track_ids = (
                np.asarray(_numpy(self.track_id)).tolist()
                if self.track_id is not None
                else (obb.id.tolist() if obb.id is not None else None)
            )
            rendered = draw_obb(
                rendered,
                obb.xywhr.tolist(),
                obb.conf.tolist(),
                obb.cls.tolist(),
                class_names=self.names,
                track_ids=obb_track_ids,
            )
        if self.keypoints is not None:
            rendered = draw_keypoints(rendered, np.asarray(_numpy(self.keypoints.data)))
        if self.gaze is not None:
            if self.boxes is None and len(self.gaze) > 0:
                raise ValueError("gaze plotting requires aligned face boxes")
            if self.boxes is None:
                return rendered
            gaze = self.gaze.numpy()
            rendered = draw_gaze_arrows(
                rendered,
                self.boxes.numpy().xyxy.tolist(),
                gaze.pitch.tolist(),
                gaze.yaw.tolist(),
            )
        return rendered

    def save(self, path: str, image: Any = None) -> str:
        """Save a matte result as a transparent-background RGBA PNG cutout.

        Returns the written path. Requires the source image (via ``image`` or
        ``self.path``).
        """
        from PIL import Image

        if self.matte is None:
            raise NotImplementedError(
                "Results.save() writes a transparent-PNG cutout and is defined for "
                "matte results only. Use result.plot()/CLI --save for other tasks."
            )
        rgba = self.cutout(image=image)
        out = Path(path)
        if out.suffix.lower() != ".png":
            raise ValueError(
                "Matte Results.save() requires a .png path so transparency is preserved."
            )
        if out.parent and str(out.parent) not in (".", ""):
            out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").save(out)
        return str(out)

    def summary(self, normalize: bool = False, decimals: int = 5) -> List[Dict[str, Any]]:
        keypoints_np = self.keypoints.numpy() if self.keypoints is not None else None
        mask_primary = None
        mask_contours = None
        if self.masks is not None:
            mask_primary = self.masks.xyn if normalize else self.masks.xy
            mask_contours = (
                self.masks.contours_normalized if normalize else self.masks.contours
            )

        def _keypoint_fields(index: int) -> Dict[str, Any]:
            if keypoints_np is None:
                return {}
            xy = keypoints_np.xyn[index] if normalize else keypoints_np.xy[index]
            payload: Dict[str, Any] = {
                "x": [round(float(value), decimals) for value in xy[:, 0]],
                "y": [round(float(value), decimals) for value in xy[:, 1]],
            }
            if keypoints_np.conf is not None:
                payload["confidence"] = [
                    round(float(value), decimals) for value in keypoints_np.conf[index]
                ]
            return {"keypoints": payload}

        def _mask_fields(index: int) -> Dict[str, Any]:
            if mask_primary is None or mask_contours is None:
                return {}
            primary = mask_primary[index]
            contours = []
            for contour in mask_contours[index]:
                points = contour["points"]
                contours.append(
                    {
                        "x": [round(float(value), decimals) for value in points[:, 0]],
                        "y": [round(float(value), decimals) for value in points[:, 1]],
                        "is_hole": bool(contour["is_hole"]),
                        "parent": contour["parent"],
                    }
                )
            return {
                "segments": {
                    "x": [round(float(value), decimals) for value in primary[:, 0]],
                    "y": [round(float(value), decimals) for value in primary[:, 1]],
                },
                "mask_contours": contours,
            }

        if self.boxes is None:
            if self.ocr is not None:
                ocr_np = self.ocr.numpy()
                h, w = self.orig_shape
                rows = []
                for i in range(len(ocr_np)):
                    polygon = np.asarray(ocr_np.data[i], dtype=float)
                    if normalize:
                        polygon = polygon / np.array([w, h], dtype=float)
                    rows.append(
                        {
                            "name": "text",
                            "text": ocr_np.texts[i],
                            "confidence": round(float(ocr_np.conf[i]), decimals),
                            "det_confidence": round(float(ocr_np.det_conf[i]), decimals),
                            "polygon": {
                                "x": [round(float(x), decimals) for x in polygon[:, 0]],
                                "y": [round(float(y), decimals) for y in polygon[:, 1]],
                            },
                        }
                    )
                return rows
            if self.points is not None:
                points_np = self.points.numpy()
                xy_values = points_np.xyn if normalize else points_np.xy
                rows = []
                for i in range(len(points_np)):
                    cls_id = int(points_np.cls[i])
                    rows.append(
                        {
                            "name": self.names.get(cls_id, str(cls_id)),
                            "class": cls_id,
                            "confidence": round(float(points_np.conf[i]), decimals),
                            "point": {
                                "x": round(float(xy_values[i, 0]), decimals),
                                "y": round(float(xy_values[i, 1]), decimals),
                            },
                        }
                    )
                return rows
            if self.obb is not None:
                obb_np = self.obb.numpy()
                rows = []
                for i in range(len(obb_np)):
                    cls_id = int(obb_np.cls[i])
                    xywhr = np.asarray(obb_np.xywhr[i], dtype=float).copy()
                    corners = np.asarray(
                        obb_np.xyxyxyxyn[i] if normalize else obb_np.xyxyxyxy[i],
                        dtype=float,
                    )
                    if normalize:
                        h, w = self.orig_shape
                        xywhr[:4] /= np.array([w, h, w, h], dtype=float)
                    rows.append(
                        {
                            "name": self.names.get(cls_id, str(cls_id)),
                            "class": cls_id,
                            "confidence": round(float(obb_np.conf[i]), decimals),
                            "obb": {
                                "x_center": round(float(xywhr[0]), decimals),
                                "y_center": round(float(xywhr[1]), decimals),
                                "width": round(float(xywhr[2]), decimals),
                                "height": round(float(xywhr[3]), decimals),
                                "rotation": round(float(xywhr[4]), decimals),
                            },
                            "corners": {
                                "x": [round(float(x), decimals) for x in corners[:, 0]],
                                "y": [round(float(y), decimals) for y in corners[:, 1]],
                            },
                            **(
                                {"track_id": int(obb_np.id[i])}
                                if obb_np.id is not None
                                else {}
                            ),
                        }
                    )
                return rows
            if self.keypoints is not None:
                return [
                    _keypoint_fields(index)
                    for index in range(len(self.keypoints))
                ]
            if self.masks is not None:
                return [
                    _mask_fields(index)
                    for index in range(len(self.masks))
                ]
            if self.gaze is not None:
                gaze_np = self.gaze.numpy()
                return [
                    {
                        "gaze": {
                            "pitch_rad": round(float(gaze_np.data[index, 0]), decimals),
                            "yaw_rad": round(float(gaze_np.data[index, 1]), decimals),
                            "pitch_deg": round(
                                float(gaze_np.data[index, 0]) * 180.0 / math.pi,
                                decimals,
                            ),
                            "yaw_deg": round(
                                float(gaze_np.data[index, 1]) * 180.0 / math.pi,
                                decimals,
                            ),
                        },
                    }
                    for index in range(len(self.gaze))
                ]
            if self.panoptic is not None:
                pan_np = _numpy(self.panoptic.data)
                total = int(pan_np.size)
                rows = []
                for seg in self.panoptic.segments_info:
                    cat_id = int(seg["category_id"])
                    count = int((pan_np == int(seg["id"])).sum())
                    row = {
                        "name": self.names.get(cat_id, str(cat_id)),
                        "class": cat_id,
                        "segment_id": int(seg["id"]),
                        "pixel_count": count,
                        "pixel_fraction": round(count / total, decimals) if total else 0.0,
                    }
                    if "isthing" in seg:
                        row["isthing"] = bool(seg["isthing"])
                    if "score" in seg:
                        row["confidence"] = round(float(seg["score"]), decimals)
                    rows.append(row)
                return rows
            if self.semantic_mask is not None:
                mask_np = _numpy(self.semantic_mask.data)
                total = int(mask_np.size)
                rows = []
                for cls_id in self.semantic_mask.classes:
                    count = int((mask_np == cls_id).sum())
                    rows.append(
                        {
                            "name": self.names.get(cls_id, str(cls_id)),
                            "class": cls_id,
                            "pixel_count": count,
                            "pixel_fraction": round(count / total, decimals) if total else 0.0,
                        }
                    )
                return rows
            if self.depth_map is not None:
                return [
                    {
                        "name": "depth_map",
                        "min": round(self.depth_map.min, decimals),
                        "max": round(self.depth_map.max, decimals),
                        "mean": round(self.depth_map.mean, decimals),
                    }
                ]
            if self.restored is not None:
                h, w = self.restored.array.shape[:2]
                return [
                    {
                        "name": "restored",
                        "shape": [int(h), int(w), 3],
                        "scale": int(self.restore_scale),
                    }
                ]
            if self.matte is not None:
                matte_np = self.matte.array
                h, w = matte_np.shape[:2]
                fg = float((matte_np >= 0.5).mean())
                return [
                    {
                        "name": "matte",
                        "shape": [int(h), int(w)],
                        "coverage": round(fg, decimals),
                    }
                ]
            if self.probs is None:
                return []
            probs_np = _numpy(self.probs.data)
            rows = []
            for cls_id in self.probs.top5:
                rows.append(
                    {
                        "name": self.names.get(cls_id, str(cls_id)),
                        "class": int(cls_id),
                        "confidence": round(float(probs_np[cls_id]), decimals),
                    }
                )
            return rows

        boxes_np = self.boxes.numpy()
        obb_np = None
        if self.obb is not None:
            obb_np = self.obb.numpy() if isinstance(self.obb.data, torch.Tensor) else self.obb
        track_ids = _numpy(self.track_id)
        if track_ids is None and obb_np is not None and obb_np.id is not None:
            track_ids = _numpy(obb_np.id)
        rows = []
        for i in range(len(boxes_np)):
            cls_id = int(boxes_np.cls[i])
            box_values = boxes_np.xyxyn[i] if normalize else boxes_np.xyxy[i]
            row = {
                "name": self.names.get(cls_id, str(cls_id)),
                "class": cls_id,
                "confidence": round(float(boxes_np.conf[i]), decimals),
                "box": {
                    "x1": round(float(box_values[0]), decimals),
                    "y1": round(float(box_values[1]), decimals),
                    "x2": round(float(box_values[2]), decimals),
                    "y2": round(float(box_values[3]), decimals),
                },
            }
            if obb_np is not None and i < len(obb_np):
                xywhr = np.asarray(obb_np.xywhr[i], dtype=float).copy()
                corners = np.asarray(
                    obb_np.xyxyxyxyn[i] if normalize else obb_np.xyxyxyxy[i],
                    dtype=float,
                )
                if normalize:
                    h, w = self.orig_shape
                    xywhr[0] /= w
                    xywhr[1] /= h
                    xywhr[2] /= w
                    xywhr[3] /= h
                row["obb"] = {
                    "x_center": round(float(xywhr[0]), decimals),
                    "y_center": round(float(xywhr[1]), decimals),
                    "width": round(float(xywhr[2]), decimals),
                    "height": round(float(xywhr[3]), decimals),
                    "rotation": round(float(xywhr[4]), decimals),
                }
                row["corners"] = {
                    "x": [round(float(x), decimals) for x in corners[:, 0]],
                    "y": [round(float(y), decimals) for y in corners[:, 1]],
                }
            if self.masks is not None:
                row.update(_mask_fields(i))
            if self.keypoints is not None:
                row.update(_keypoint_fields(i))
            if self.gaze is not None and i < len(self.gaze):
                gaze_np = self.gaze.numpy() if isinstance(self.gaze.data, torch.Tensor) else self.gaze
                row["gaze"] = {
                    "pitch_rad": round(float(gaze_np.data[i, 0]), decimals),
                    "yaw_rad": round(float(gaze_np.data[i, 1]), decimals),
                    "pitch_deg": round(float(gaze_np.data[i, 0]) * 180.0 / math.pi, decimals),
                    "yaw_deg": round(float(gaze_np.data[i, 1]) * 180.0 / math.pi, decimals),
                }
            if track_ids is not None:
                row["track_id"] = int(track_ids[i])
            rows.append(row)
        return rows

    def to_json(self, *, include_metadata: bool = False, **kwargs) -> str:
        """Serialize summary rows, optionally wrapped with image/task metadata.

        The default remains the established JSON row list. Set
        ``include_metadata=True`` when an empty result must retain its task,
        source canvas, frame, save, or tiling identity in the serialized form.
        """
        rows = self.summary(**kwargs)
        if not include_metadata:
            return json.dumps(rows, default=_json_default)
        return json.dumps(
            {
                "task": self.task,
                "orig_shape": list(self.orig_shape),
                "path": self.path,
                "names": {_json_key(key): value for key, value in self.names.items()},
                "frame_idx": self.frame_idx,
                "saved_path": self.saved_path,
                "tiled": self.tiled,
                "num_tiles": self.num_tiles,
                "tiles_path": self.tiles_path,
                "grid_path": self.grid_path,
                "restore_scale": self.restore_scale,
                "speed": self.speed,
                "results": rows,
            },
            default=_json_default,
        )

    def __len__(self) -> int:
        for key in self._instance_keys:
            value = getattr(self, key)
            if value is not None:
                return len(value)
        for key in self._whole_image_keys:
            if getattr(self, key) is not None:
                return 1
        return 0

    def __repr__(self) -> str:
        parts = [
            f"path='{self.path}'",
            f"task='{self.task}'",
            f"orig_shape={self.orig_shape}",
            f"boxes={self.boxes}",
        ]
        if self.points is not None:
            parts.append(f"points={self.points}")
        if self.masks is not None:
            parts.append(f"masks={self.masks}")
        if self.keypoints is not None:
            parts.append(f"keypoints={self.keypoints}")
        if self.probs is not None:
            parts.append(f"probs={self.probs}")
        if self.obb is not None:
            parts.append(f"obb={self.obb}")
        if self.gaze is not None:
            parts.append(f"gaze={self.gaze}")
        if self.semantic_mask is not None:
            parts.append(f"semantic_mask={self.semantic_mask}")
        if self.panoptic is not None:
            parts.append(f"panoptic={self.panoptic}")
        if self.depth_map is not None:
            parts.append(f"depth_map={self.depth_map}")
        if self.restored is not None:
            parts.append(f"restored={self.restored}")
            if self.restore_scale != 1:
                parts.append(f"restore_scale={self.restore_scale}")
        if self.matte is not None:
            parts.append(f"matte={self.matte}")
        if self.ocr is not None:
            parts.append(f"ocr={self.ocr}")
        if self.track_id is not None:
            parts.append(f"track_ids={len(self.track_id)}")
        if self.frame_idx is not None:
            parts.append(f"frame_idx={self.frame_idx}")
        return f"Results({', '.join(parts)})"
