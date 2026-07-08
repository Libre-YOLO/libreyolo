"""Flat result containers for LibreYOLO."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


TensorLike = Union[torch.Tensor, np.ndarray]


def _move(data: TensorLike | None, *args, **kwargs):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data).to(*args, **kwargs)
    return data


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
    sliced = data[idx]
    if isinstance(sliced, torch.Tensor):
        if sliced.ndim == data.ndim - 1:
            sliced = sliced.unsqueeze(0)
    elif isinstance(sliced, np.ndarray):
        if sliced.ndim == data.ndim - 1:
            sliced = np.expand_dims(sliced, axis=0)
    else:
        sliced = np.asarray([sliced])
    return sliced


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
        self._boxes = boxes
        self._conf = conf
        self._cls = cls
        self._id = id
        self.orig_shape = orig_shape

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
        return Boxes(
            _move(self._boxes, *args, **kwargs),
            _move(self._conf, *args, **kwargs),
            _move(self._cls, *args, **kwargs),
            _move(self._id, *args, **kwargs),
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
        self._masks = masks
        self.orig_shape = orig_shape

    @property
    def data(self) -> TensorLike:
        return self._masks

    @property
    def xy(self) -> List[np.ndarray]:
        return self._masks_to_contours(normalize=False)

    @property
    def xyn(self) -> List[np.ndarray]:
        return self._masks_to_contours(normalize=True)

    def _masks_to_contours(self, normalize: bool) -> List[np.ndarray]:
        import cv2

        masks_np = _numpy(self._masks).astype(np.uint8)
        h, w = self.orig_shape
        contours_list = []
        for mask in masks_np:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                contour = max(contours, key=cv2.contourArea).squeeze(1).astype(np.float64)
                if normalize:
                    contour[:, 0] /= w
                    contour[:, 1] /= h
                contours_list.append(contour)
            else:
                contours_list.append(np.empty((0, 2), dtype=np.float64))
        return contours_list

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
        if isinstance(self._masks, torch.Tensor):
            return Masks(self._masks.cuda(), self.orig_shape)
        return self

    def numpy(self) -> "Masks":
        if isinstance(self._masks, torch.Tensor):
            return Masks(self._masks.detach().cpu().numpy(), self.orig_shape)
        return self

    def __getitem__(self, idx) -> "Masks":
        return Masks(_slice_first(self._masks, idx), self.orig_shape)

    def __len__(self) -> int:
        return int(self._masks.shape[0])

    def __repr__(self) -> str:
        return (
            f"Masks(n={len(self)}, "
            f"shape={tuple(self._masks.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class _TensorPayload:
    """Small wrapper used for future flat result slots."""

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        self.data = data
        self.orig_shape = orig_shape

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


class Keypoints(_TensorPayload):
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


class Probs(_TensorPayload):
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

    def __getitem__(self, idx):
        # A classification probability vector is whole-image, not per-detection;
        # keep it intact so shared Results slicing (e.g. ``result[0]``) cannot
        # truncate it to a single class. Mirrors SemanticMask/DepthMap.
        return self.__class__(self.data, self.orig_shape)


class SemanticMask(_TensorPayload):
    """Dense semantic segmentation map for a single image.

    Data shape is ``(H, W)`` integer class IDs on the original image canvas.
    ``255`` is the ignore/void value and never counts as a class.
    """

    IGNORE_INDEX = 255

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if data.ndim != 2:
            raise ValueError(
                f"expected (H, W) semantic class map but got shape {tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
        super().__init__(data, orig_shape)

    @property
    def classes(self) -> List[int]:
        """Sorted class IDs present in the map, excluding the ignore value."""
        values = np.unique(_numpy(self.data))
        return [int(v) for v in values if int(v) != self.IGNORE_INDEX]

    def class_mask(self, class_id: int) -> TensorLike:
        """Boolean ``(H, W)`` mask selecting the pixels of one class."""
        return self.data == class_id

    def __getitem__(self, idx):
        # Instance indexing does not apply to a dense map; keep it intact so
        # shared Results slicing paths cannot corrupt the (H, W) layout.
        return self.__class__(self.data, self.orig_shape)

    def __repr__(self) -> str:
        return (
            f"SemanticMask(shape={tuple(self.data.shape)}, "
            f"classes={len(self.classes)}, orig_shape={self.orig_shape})"
        )


class DepthMap(_TensorPayload):
    """Dense relative inverse-depth map for a single image.

    Data shape is ``(H, W)`` float values on the original image canvas. Higher
    values mean closer to the camera. Values are relative, not metric meters.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if data.ndim != 2:
            raise ValueError(
                f"expected (H, W) depth map but got shape {tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
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
            return data * 0
        normalized = (data - lo) / (hi - lo)
        if isinstance(normalized, torch.Tensor):
            return torch.where(torch.isfinite(normalized), normalized, torch.zeros_like(normalized))
        return np.where(np.isfinite(normalized), normalized, np.zeros_like(normalized))

    def __getitem__(self, idx):
        # Instance indexing does not apply to a dense map; keep it intact so
        # shared Results slicing paths cannot corrupt the (H, W) layout.
        return self.__class__(self.data, self.orig_shape)

    def __repr__(self) -> str:
        return (
            f"DepthMap(shape={tuple(self.data.shape)}, "
            f"range=({self.min:.4g}, {self.max:.4g}), "
            f"orig_shape={self.orig_shape})"
        )


class RestoredImage(_TensorPayload):
    """Dense restored RGB image for a single input.

    Data shape is ``(H, W, 3)`` uint8 RGB on the original image canvas.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError(
                f"expected (H, W, 3) restored RGB image but got shape {tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
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

    def __getitem__(self, idx):
        # Instance indexing does not apply to a dense restored image; keep it
        # intact so shared Results slicing paths cannot corrupt the HWC layout.
        return self.__class__(self.data, self.orig_shape)

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return (
            f"RestoredImage(shape={tuple(self.data.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class Matte(_TensorPayload):
    """Dense soft alpha matte for a single image.

    Data shape is ``(H, W)`` float32 in ``[0, 1]`` on the original image canvas.
    ``1`` is fully foreground (opaque), ``0`` is fully background (transparent).
    A soft matte subsumes a hard background-removal mask (threshold at 0.5) and
    carries the anti-aliased edges (hair, fur) that binary masks discard.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if data.ndim != 2:
            raise ValueError(
                f"expected (H, W) alpha matte but got shape {tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
        super().__init__(data, orig_shape)

    @property
    def array(self) -> np.ndarray:
        """Return the raw ``(H, W)`` float32 alpha matte clamped to ``[0, 1]``."""
        arr = np.asarray(_numpy(self.data), dtype=np.float32)
        return np.clip(arr, 0.0, 1.0)

    def __getitem__(self, idx):
        # A dense matte is whole-image, not per-detection; keep it intact so
        # shared Results slicing paths cannot corrupt the (H, W) layout.
        return self.__class__(self.data, self.orig_shape)

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return (
            f"Matte(shape={tuple(self.data.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class OBB(_TensorPayload):
    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if data.ndim == 1:
            data = data[None, :]
        n = data.shape[-1]
        if n not in {7, 8}:
            raise ValueError(
                f"expected 7 or 8 OBB values but got {n}: "
                "xywhr, optional track_id, conf, cls"
            )
        super().__init__(data, orig_shape)

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
        if data.ndim == 1:
            if isinstance(data, torch.Tensor):
                data = data.unsqueeze(0)
            else:
                data = data[None, :]
        if data.shape[-1] != 2:
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
        depth_map: Optional[DepthMap] = None,
        restored: Optional[RestoredImage] = None,
        matte: Optional[Matte] = None,
        speed: Optional[Dict[str, float]] = None,
        track_id: Optional[TensorLike] = None,
        frame_idx: Optional[int] = None,
    ):
        if boxes is not None and boxes.orig_shape is None:
            boxes = boxes.with_orig_shape(orig_shape)
        if boxes is not None and track_id is not None:
            boxes = boxes.with_id(track_id)
        if points is not None and points.orig_shape is None:
            points = Points(points.data, orig_shape)
        if depth_map is not None and depth_map.orig_shape is None:
            depth_map = DepthMap(depth_map.data, orig_shape)
        if restored is not None and restored.orig_shape is None:
            restored = RestoredImage(restored.data, orig_shape)
        if matte is not None and matte.orig_shape is None:
            matte = Matte(matte.data, orig_shape)

        self.boxes = boxes
        self.masks = masks
        self.keypoints = keypoints
        self.probs = probs
        self.obb = obb
        self.gaze = gaze
        self.points = points
        self.semantic_mask = semantic_mask
        self.depth_map = depth_map
        self.restored = restored
        self.matte = matte
        self.orig_shape = orig_shape
        self.path = path
        self.names = names or {}
        self.speed = speed or {}
        self.track_id = track_id if track_id is not None else (boxes.id if boxes else None)
        self.frame_idx = frame_idx

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
            "depth_map": self.depth_map,
            "restored": self.restored,
            "matte": self.matte,
            "speed": dict(self.speed),
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
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
            overrides["track_id"] = _move(self.track_id, *args, **kwargs)
        elif method == "__getitem__":
            overrides["track_id"] = _slice_first(self.track_id, args[0])

        return self._new(**overrides)

    def _select(self, idx) -> "Results":
        return self._apply("__getitem__", idx)

    def __getitem__(self, idx) -> "Results":
        return self._select(idx)

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
        depth_map: Optional[DepthMap] = None,
        restored: Optional[RestoredImage] = None,
        matte: Optional[Matte] = None,
        track_id: Optional[TensorLike] = None,
    ) -> "Results":
        if boxes is not None:
            self.boxes = boxes.with_orig_shape(self.orig_shape)
        if masks is not None:
            self.masks = masks
        if probs is not None:
            self.probs = probs
        if keypoints is not None:
            self.keypoints = keypoints
        if obb is not None:
            self.obb = obb
        if gaze is not None:
            self.gaze = gaze
        if points is not None:
            self.points = points if points.orig_shape is not None else Points(points.data, self.orig_shape)
        if semantic_mask is not None:
            self.semantic_mask = semantic_mask
        if depth_map is not None:
            self.depth_map = depth_map
        if restored is not None:
            self.restored = restored
        if matte is not None:
            self.matte = matte if matte.orig_shape is not None else Matte(matte.data, self.orig_shape)
        if track_id is not None:
            self.track_id = track_id
            if self.boxes is not None:
                self.boxes = self.boxes.with_id(track_id)
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

        h, w = hw
        if image is None:
            if not self.path:
                raise ValueError(
                    "cutout()/save() needs the source image but Results.path is unset; "
                    "pass image=<PIL.Image or HxWx3 array>."
                )
            rgb = np.asarray(Image.open(self.path).convert("RGB"))
        elif isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
        else:
            rgb = np.asarray(image)
            if rgb.ndim == 2:
                rgb = np.stack([rgb] * 3, axis=-1)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
        if rgb.shape[:2] != (h, w):
            rgb = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((w, h), Image.BILINEAR))
        return rgb.astype(np.uint8)

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
        if out.parent and str(out.parent) not in (".", ""):
            out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").save(out)
        return str(out)

    def summary(self, normalize: bool = False, decimals: int = 5) -> List[Dict[str, Any]]:
        if self.boxes is None:
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
                            "pixel_fraction": round(count / total, decimals),
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
                segment = self.masks.xyn[i] if normalize else self.masks.xy[i]
                row["segments"] = {
                    "x": [round(float(x), decimals) for x in segment[:, 0]],
                    "y": [round(float(y), decimals) for y in segment[:, 1]],
                }
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

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.summary(**kwargs))

    def __len__(self) -> int:
        if self.boxes is not None:
            return len(self.boxes)
        if self.points is not None:
            return len(self.points)
        if self.probs is not None:
            return 1
        if self.semantic_mask is not None:
            return 1
        if self.depth_map is not None:
            return 1
        if self.restored is not None:
            return 1
        if self.matte is not None:
            return 1
        return 0

    def __repr__(self) -> str:
        parts = [
            f"path='{self.path}'",
            f"orig_shape={self.orig_shape}",
            f"boxes={self.boxes}",
        ]
        if self.points is not None:
            parts.append(f"points={self.points}")
        if self.masks is not None:
            parts.append(f"masks={self.masks}")
        if self.semantic_mask is not None:
            parts.append(f"semantic_mask={self.semantic_mask}")
        if self.depth_map is not None:
            parts.append(f"depth_map={self.depth_map}")
        if self.restored is not None:
            parts.append(f"restored={self.restored}")
        if self.matte is not None:
            parts.append(f"matte={self.matte}")
        if self.track_id is not None:
            parts.append(f"track_ids={len(self.track_id)}")
        if self.frame_idx is not None:
            parts.append(f"frame_idx={self.frame_idx}")
        return f"Results({', '.join(parts)})"
