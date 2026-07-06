"""Golden parity tests for every train-time augmentation pipeline.

These tests pin the exact numerical behavior of each family's augmentation
pipeline (seeded RNG, fixed synthetic inputs) against committed ``.npz``
fixtures. They exist so the shared-augmentation refactor cannot silently
change training behavior: any change to RNG call order, BGR/RGB handling,
normalization, padding, or label packing flips them red.

Regenerate fixtures (only when a behavior change is intentional):

    python tests/unit/test_augment_parity.py --generate

Fixtures embed the generation environment (platform, python, numpy, cv2,
torch, torchvision). Exact-equality assertions run only when the current
environment matches the reference one; otherwise the test skips loudly with
the env diff — never silently, and never falling back to ``allclose``.
``--generate`` refuses to run under CI so fixtures can only be re-pinned
deliberately, on a developer machine.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "augment_golden"
SEED = 20260610
ENV_KEY = "__env__"


def _env_fingerprint() -> dict:
    import cv2
    import torch
    import torchvision

    return {
        "platform": f"{platform.system()}-{platform.machine()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cv2": cv2.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }


def _check_reference_env(fixture_env: dict) -> None:
    """Skip (loudly) on environments that differ from the fixture's."""
    actual = _env_fingerprint()
    diffs = {
        k: (fixture_env.get(k), v) for k, v in actual.items() if fixture_env.get(k) != v
    }
    if diffs:
        pytest.skip(
            "augment parity fixtures were generated on a different environment; "
            "byte-exact comparison is only meaningful on the reference platform. "
            f"Mismatches (fixture, here): {diffs}"
        )


def _seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch

        torch.manual_seed(SEED)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Deterministic synthetic inputs
# ---------------------------------------------------------------------------


def _image(h=97, w=131, key=0):
    rng = np.random.RandomState(1000 + key)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _boxes_xyxy_cls(key=0):
    """Pixel xyxy + class column, incl. a near-border and a small box."""
    boxes = {
        0: np.array(
            [
                [10.0, 12.0, 60.0, 70.0, 1.0],
                [0.5, 0.5, 25.0, 30.0, 0.0],
                [100.0, 60.0, 130.0, 96.0, 2.0],
                [50.0, 50.0, 52.5, 53.0, 1.0],
            ],
            dtype=np.float32,
        ),
        1: np.array(
            [
                [5.0, 20.0, 80.0, 90.0, 0.0],
                [40.0, 10.0, 120.0, 50.0, 2.0],
            ],
            dtype=np.float32,
        ),
        2: np.array([[20.0, 30.0, 110.0, 85.0, 1.0]], dtype=np.float32),
        3: np.array(
            [
                [15.0, 5.0, 45.0, 40.0, 0.0],
                [60.0, 55.0, 125.0, 92.0, 1.0],
                [30.0, 60.0, 75.0, 95.0, 2.0],
            ],
            dtype=np.float32,
        ),
    }
    return boxes[key].copy()


def _obb_targets():
    """Pixel xyxy + class + angle (radians)."""
    t = _boxes_xyxy_cls(0)
    angles = np.array([[0.3], [-0.9], [1.2], [0.0]], dtype=np.float32)
    return np.hstack([t, angles]).astype(np.float32)


def _segments(key=0, with_dense=False):
    """Per-instance polygon rings matching _boxes_xyxy_cls(key) rows."""
    boxes = _boxes_xyxy_cls(key)
    segments = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b[:4]
        # A pentagon-ish ring inside the box so rasterization is nontrivial.
        ring = np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, (y1 + y2) / 2],
                [(x1 + x2) / 2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )
        if with_dense and i == 0:
            import cv2

            from libreyolo.data.dataset import DenseMaskRing

            mask = np.zeros((97, 131), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(ring).astype(np.int32)], color=1)
            segments.append([DenseMaskRing(ring, mask)])
        else:
            segments.append([ring])
    return segments


def _pose_inputs(num_keypoints=5):
    """Normalized cxcywh boxes, classes, and (N, K, 3) normalized keypoints."""
    bboxes = np.array(
        [
            [0.30, 0.40, 0.25, 0.35],
            [0.70, 0.55, 0.30, 0.40],
            [0.50, 0.85, 0.20, 0.22],
        ],
        dtype=np.float32,
    )
    cls = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rng = np.random.RandomState(7)
    kpts = np.zeros((3, num_keypoints, 3), dtype=np.float32)
    for i, b in enumerate(bboxes):
        cx, cy, w, h = b
        kpts[i, :, 0] = np.clip(cx + (rng.rand(num_keypoints) - 0.5) * w, 0, 1)
        kpts[i, :, 1] = np.clip(cy + (rng.rand(num_keypoints) - 0.5) * h, 0, 1)
        kpts[i, :, 2] = rng.choice([0.0, 1.0, 2.0], size=num_keypoints, p=[0.15, 0.25, 0.6])
    return bboxes, cls, kpts


FLIP_IDX_5 = [0, 2, 1, 4, 3]


class _StubDetDataset:
    """Minimal dataset exposing the pull_item/load_anno contract."""

    def __init__(self, n=6, with_segments=False, dense_first=False):
        self.n = n
        self.with_segments = with_segments
        self.dense_first = dense_first

    def __len__(self):
        return self.n

    def load_anno(self, idx):
        return _boxes_xyxy_cls(idx % 4)

    def pull_item(self, idx):
        img = _image(key=idx % 4)
        label = _boxes_xyxy_cls(idx % 4)
        if self.with_segments:
            segments = _segments(idx % 4, with_dense=self.dense_first and idx % 4 == 0)
            return img, label, (img.shape[0], img.shape[1]), idx, segments
        return img, label, (img.shape[0], img.shape[1]), idx


# ---------------------------------------------------------------------------
# Pipeline cases. Each returns a dict of arrays to pin.
# ---------------------------------------------------------------------------


def _case_yolox_train():
    from libreyolo.training.augment import TrainTransform

    t = TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_yolox_train_empty():
    from libreyolo.training.augment import TrainTransform

    t = TrainTransform(max_labels=50)
    _seed_all()
    img, labels = t(_image(), np.zeros((0, 5), dtype=np.float32), (64, 64))
    return {"img": img, "labels": labels}


def _case_yolox_val():
    from libreyolo.training.augment import ValTransform

    t = ValTransform()
    _seed_all()
    img, labels = t(_image(), None, (64, 64))
    return {"img": img, "labels": labels}


def _case_yolox_mosaic_mixup():
    from libreyolo.training.augment import MosaicMixupDataset, TrainTransform

    ds = MosaicMixupDataset(
        _StubDetDataset(),
        img_size=(64, 64),
        mosaic=True,
        preproc=TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0),
        degrees=10.0,
        translate=0.1,
        mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5),
        shear=2.0,
        enable_mixup=True,
        mosaic_prob=1.0,
        mixup_prob=1.0,
    )
    _seed_all()
    img, labels, _info, _id = ds[1]
    return {"img": img, "labels": labels}


def _case_perspective_random_affine():
    from libreyolo.data.augment.geometry import random_affine

    _seed_all()
    img = _image(h=64, w=64)
    boxes = _boxes_xyxy_cls(0)
    out_img, out_boxes = random_affine(
        img,
        boxes,
        target_size=(64, 64),
        degrees=10.0,
        translate=0.1,
        scales=0.1,
        shear=10.0,
        perspective=0.001,
    )
    return {"img": out_img, "labels": out_boxes}


def _case_yolox_flipud():
    from libreyolo.training.augment import TrainTransform

    t = TrainTransform(max_labels=50, flip_prob=0.0, hsv_prob=0.0, flipud=1.0)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_yolonas_flipud():
    from libreyolo.models.yolonas.transforms import YOLONASTrainTransform

    t = YOLONASTrainTransform(max_labels=50, flip_prob=0.0, hsv_prob=0.0, flipud=1.0)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_yolo9_rot90_obb():
    from libreyolo.models.yolo9.transforms import YOLO9TrainTransform

    t = YOLO9TrainTransform(
        max_labels=50,
        flip_prob=0.0,
        vertical_flip_prob=0.0,
        hsv_prob=0.0,
        output_label_dim=6,
        rot90_prob=1.0,
    )
    _seed_all()
    img, labels = t(_image(), _obb_targets(), (64, 64))
    return {"img": img, "labels": labels}


def _case_yolonas_train():
    from libreyolo.models.yolonas.transforms import YOLONASTrainTransform

    t = YOLONASTrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=0.5)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_yolonas_affine_mixup():
    from libreyolo.models.yolonas.transforms import (
        YOLONASAffineMixupDataset,
        YOLONASTrainTransform,
    )

    ds = YOLONASAffineMixupDataset(
        _StubDetDataset(),
        img_size=(64, 64),
        preproc=YOLONASTrainTransform(max_labels=50),
        degrees=0.0,
        translate=0.25,
        mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5),
        shear=0.0,
        enable_mixup=True,
        mixup_prob=1.0,
    )
    _seed_all()
    img, labels, _info, _id = ds[2]
    return {"img": img, "labels": labels}


def _case_yolo9_train_det():
    from libreyolo.models.yolo9.transforms import YOLO9TrainTransform

    t = YOLO9TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_yolo9_train_obb():
    from libreyolo.models.yolo9.transforms import YOLO9TrainTransform

    t = YOLO9TrainTransform(
        max_labels=50,
        flip_prob=0.5,
        vertical_flip_prob=0.5,
        hsv_prob=1.0,
        output_label_dim=6,
    )
    _seed_all()
    img, labels = t(_image(), _obb_targets(), (64, 64))
    return {"img": img, "labels": labels}


def _case_yolo9_train_seg():
    from libreyolo.models.yolo9.transforms import YOLO9TrainTransform

    t = YOLO9TrainTransform(max_labels=50, flip_prob=1.0, hsv_prob=1.0)
    _seed_all()
    img, labels, masks = t(
        _image(), _boxes_xyxy_cls(0), (64, 64), segments=_segments(0, with_dense=True)
    )
    return {"img": img, "labels": labels, "masks": masks}


def _case_yolo9_mosaic_det():
    from libreyolo.models.yolo9.transforms import (
        YOLO9MosaicMixupDataset,
        YOLO9TrainTransform,
    )

    ds = YOLO9MosaicMixupDataset(
        _StubDetDataset(),
        img_size=(64, 64),
        preproc=YOLO9TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0),
        mosaic_prob=1.0,
    )
    _seed_all()
    img, labels, _info, _id = ds[0]
    return {"img": img, "labels": labels}


def _case_yolo9_mosaic_seg():
    from libreyolo.models.yolo9.transforms import (
        YOLO9MosaicMixupDataset,
        YOLO9TrainTransform,
    )

    ds = YOLO9MosaicMixupDataset(
        _StubDetDataset(with_segments=True, dense_first=True),
        img_size=(64, 64),
        preproc=YOLO9TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0),
        mosaic_prob=1.0,
    )
    _seed_all()
    img, labels, _info, _id, masks = ds[0]
    return {"img": img, "labels": labels, "masks": masks}


def _case_yolo9_val():
    from libreyolo.models.yolo9.transforms import YOLO9ValTransform

    t = YOLO9ValTransform()
    _seed_all()
    img, labels = t(_image(), None, (64, 64))
    return {"img": img, "labels": labels}


def _case_deim_train():
    from libreyolo.models.deim.transforms import DEIMTrainTransform

    t = DEIMTrainTransform(max_labels=20, flip_prob=0.5, imgsz=64)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_deim_train_weak_only():
    from libreyolo.models.deim.transforms import DEIMTrainTransform

    t = DEIMTrainTransform(max_labels=20, flip_prob=0.5, imgsz=64, strong_augs=False)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(1), (64, 64))
    return {"img": img, "labels": labels}


def _case_deim_imagenet_norm():
    from libreyolo.models.deim.transforms import DEIMTrainTransform

    t = DEIMTrainTransform(max_labels=20, imgsz=64, imagenet_norm=True)
    _seed_all()
    img, labels = t(_image(key=2), _boxes_xyxy_cls(2), (64, 64))
    return {"img": img, "labels": labels}


def _case_deim_multiscale_collate():
    from libreyolo.models.deim.transforms import (
        DEIMMultiScaleCollate,
        DEIMTrainTransform,
    )

    t = DEIMTrainTransform(max_labels=20, imgsz=64)
    _seed_all()
    item0 = t(_image(key=0), _boxes_xyxy_cls(0), (64, 64))
    item1 = t(_image(key=1), _boxes_xyxy_cls(1), (64, 64))
    collate = DEIMMultiScaleCollate(base_size=64, base_size_repeat=3, stop_epoch=10)
    collate.set_epoch(0)
    imgs, labels, _infos, _ids = collate(
        [(item0[0], item0[1], (97, 131), 0), (item1[0], item1[1], (97, 131), 1)]
    )
    return {"imgs": imgs.numpy(), "labels": labels.numpy()}


def _case_deim_sanitize_min_size():
    from libreyolo.models.deim.transforms import DEIMTrainTransform

    t = DEIMTrainTransform(max_labels=20, imgsz=64, sanitize_min_size=8)
    _seed_all()
    img, labels = t(_image(key=3), _boxes_xyxy_cls(3), (64, 64))
    return {"img": img, "labels": labels}


def _case_dfine_train():
    from libreyolo.models.dfine.transforms import DFINETrainTransform

    t = DFINETrainTransform(max_labels=20, flip_prob=0.5, imgsz=64)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_dfine_train_weak_only():
    from libreyolo.models.dfine.transforms import DFINETrainTransform

    t = DFINETrainTransform(max_labels=20, flip_prob=0.5, imgsz=64, strong_augs=False)
    _seed_all()
    img, labels = t(_image(key=1), _boxes_xyxy_cls(1), (64, 64))
    return {"img": img, "labels": labels}


def _case_dfine_multiscale_collate():
    from libreyolo.models.dfine.transforms import (
        DFINEMultiScaleCollate,
        DFINETrainTransform,
    )

    t = DFINETrainTransform(max_labels=20, imgsz=64)
    _seed_all()
    item0 = t(_image(key=0), _boxes_xyxy_cls(0), (64, 64))
    item1 = t(_image(key=1), _boxes_xyxy_cls(1), (64, 64))
    collate = DFINEMultiScaleCollate(base_size=64, base_size_repeat=3, stop_epoch=10)
    collate.set_epoch(0)
    imgs, labels, _infos, _ids = collate(
        [(item0[0], item0[1], (97, 131), 0), (item1[0], item1[1], (97, 131), 1)]
    )
    return {"imgs": imgs.numpy(), "labels": labels.numpy()}


class _SniffingDataset:
    """Replicates ``YOLODataset.pull_item``'s ``wants_unresized_image`` branch.

    Mirrors ``libreyolo/data/dataset.py``: stored labels are scaled to the
    resized canvas; the unresized branch hands back the ORIGINAL image and
    divides labels by the letterbox ratio, the default branch hands back the
    pre-resized image with stored labels. Records which branch fired so the
    fixture pins the DEIM/D-FINE input-path split through the twin merge.
    """

    def __init__(self, preproc, img_size=(64, 64), n=4):
        import cv2

        self.preproc = preproc
        self.img_size = img_size
        self.n = n
        self._cv2 = cv2
        self.branch_taken = -1

    def __len__(self):
        return self.n

    def _ratio(self, orig_hw):
        return min(self.img_size[0] / orig_hw[0], self.img_size[1] / orig_hw[1])

    def pull_item(self, index):
        orig = _image(key=index % 4)
        orig_hw = orig.shape[:2]
        r = self._ratio(orig_hw)
        stored_label = _boxes_xyxy_cls(index % 4)
        stored_label[:, :4] = stored_label[:, :4] * r  # as stored post-resize
        if getattr(self.preproc, "wants_unresized_image", False):
            self.branch_taken = 1
            label = stored_label.copy()
            if label.shape[0] > 0 and r > 0:
                label[:, :4] = label[:, :4] / r
            return orig, label, orig_hw, index
        self.branch_taken = 0
        resized = self._cv2.resize(
            orig,
            (int(orig_hw[1] * r), int(orig_hw[0] * r)),
            interpolation=self._cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized, stored_label.copy(), orig_hw, index


def _case_deim_pull_item_unresized():
    from libreyolo.models.deim.transforms import (
        DEIMPassThroughDataset,
        DEIMTrainTransform,
    )

    t = DEIMTrainTransform(max_labels=20, imgsz=64)
    stub = _SniffingDataset(preproc=t)
    ds = DEIMPassThroughDataset(stub, img_size=(64, 64), preproc=t)
    _seed_all()
    img, labels, _info, _id = ds[0]
    assert stub.branch_taken == 1, "DEIM must take the unresized pull_item branch"
    return {
        "img": img,
        "labels": labels,
        "branch": np.array([stub.branch_taken], dtype=np.int64),
    }


def _case_dfine_pull_item_letterboxed():
    from libreyolo.models.dfine.transforms import (
        DFINEPassThroughDataset,
        DFINETrainTransform,
    )

    t = DFINETrainTransform(max_labels=20, imgsz=64)
    stub = _SniffingDataset(preproc=t)
    ds = DFINEPassThroughDataset(stub, img_size=(64, 64), preproc=t)
    _seed_all()
    img, labels, _info, _id = ds[0]
    assert stub.branch_taken == 0, "D-FINE must take the pre-resized pull_item branch"
    return {
        "img": img,
        "labels": labels,
        "branch": np.array([stub.branch_taken], dtype=np.int64),
    }


def _case_rfdetr_seg():
    from libreyolo.models.rfdetr.seg_transforms import RFDETRSegTransform

    t = RFDETRSegTransform(
        max_labels=20,
        flip_prob=1.0,
        imgsz=64,
        crop_resize_prob=1.0,
        crop_intermediate_sizes=(80, 96),
        crop_min_size=48,
        crop_max_size=96,
    )
    _seed_all()
    img, labels, masks = t(
        _image(), _boxes_xyxy_cls(0), (64, 64), segments=_segments(0, with_dense=True)
    )
    return {"img": img, "labels": labels, "masks": masks}


def _case_rfdetr_seg_no_crop():
    from libreyolo.models.rfdetr.seg_transforms import RFDETRSegTransform

    t = RFDETRSegTransform(max_labels=20, flip_prob=0.5, imgsz=64)
    _seed_all()
    img, labels, masks = t(
        _image(key=3), _boxes_xyxy_cls(3), (64, 64), segments=_segments(3)
    )
    return {"img": img, "labels": labels, "masks": masks}


def _case_rfdetr_det():
    from libreyolo.models.rfdetr.seg_transforms import RFDETRDetTransform

    t = RFDETRDetTransform(max_labels=20, flip_prob=0.5, imgsz=64)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_rfdetr_det_obb():
    from libreyolo.models.rfdetr.seg_transforms import RFDETRDetTransform

    t = RFDETRDetTransform(max_labels=20, flip_prob=1.0, imgsz=64, target_dim=6)
    _seed_all()
    img, labels = t(_image(), _obb_targets(), (64, 64))
    return {"img": img, "labels": labels}


def _case_rfdetr_pose():
    from libreyolo.models.rfdetr.pose_transforms import RFDETRPoseTransform

    bboxes, cls, kpts = _pose_inputs()
    t = RFDETRPoseTransform(
        num_keypoints=5,
        flip_idx=FLIP_IDX_5,
        max_labels=20,
        flip_prob=1.0,
        imgsz=64,
        crop_resize_prob=1.0,
        crop_intermediate_sizes=(80, 96),
        crop_min_size=48,
        crop_max_size=96,
    )
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


def _case_yolonas_pose_train():
    from libreyolo.models.yolonas.pose_transforms import YOLONASPoseTrainTransform

    bboxes, cls, kpts = _pose_inputs()
    t = YOLONASPoseTrainTransform(
        num_keypoints=5, flip_idx=FLIP_IDX_5, max_labels=20
    )
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


def _case_yolonas_pose_val():
    from libreyolo.models.yolonas.pose_transforms import YOLONASPoseValTransform

    bboxes, cls, kpts = _pose_inputs()
    t = YOLONASPoseValTransform(num_keypoints=5, max_labels=20)
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


def _case_ec_pose_train():
    from libreyolo.models.ec.pose_transforms import ECPoseTrainTransform

    bboxes, cls, kpts = _pose_inputs()
    t = ECPoseTrainTransform(num_keypoints=5, flip_idx=FLIP_IDX_5, max_labels=20)
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


def _case_ec_pose_val():
    from libreyolo.models.ec.pose_transforms import ECPoseValTransform

    bboxes, cls, kpts = _pose_inputs()
    t = ECPoseValTransform(num_keypoints=5, max_labels=20)
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


# Forced-gate cases: every probabilistic branch (flip, HSV, brightness,
# affine) is guaranteed to fire so a regression in those branches cannot
# hide behind a seed that happens not to roll them.


def _case_yolonas_pose_train_forced():
    from libreyolo.models.yolonas.pose_transforms import YOLONASPoseTrainTransform

    bboxes, cls, kpts = _pose_inputs()
    t = YOLONASPoseTrainTransform(
        num_keypoints=5,
        flip_idx=FLIP_IDX_5,
        max_labels=20,
        flip_prob=1.0,
        hsv_prob=1.0,
        brightness_contrast_prob=1.0,
        affine_prob=1.0,
    )
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


def _case_yolonas_pose_train_nonsquare():
    from libreyolo.models.yolonas.pose_transforms import YOLONASPoseTrainTransform

    bboxes, cls, kpts = _pose_inputs()
    t = YOLONASPoseTrainTransform(num_keypoints=5, flip_idx=FLIP_IDX_5, max_labels=20)
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (48, 80))
    return {"img": img, "target": target}


def _case_ec_pose_train_forced():
    from libreyolo.models.ec.pose_transforms import ECPoseTrainTransform

    bboxes, cls, kpts = _pose_inputs()
    t = ECPoseTrainTransform(
        num_keypoints=5,
        flip_idx=FLIP_IDX_5,
        max_labels=20,
        flip_prob=1.0,
        hsv_prob=1.0,
        brightness_contrast_prob=1.0,
        affine_prob=1.0,
    )
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


CASES = {
    "yolox_train": _case_yolox_train,
    "yolox_train_empty": _case_yolox_train_empty,
    "yolox_val": _case_yolox_val,
    "yolox_mosaic_mixup": _case_yolox_mosaic_mixup,
    "perspective_random_affine": _case_perspective_random_affine,
    "yolox_flipud": _case_yolox_flipud,
    "yolonas_flipud": _case_yolonas_flipud,
    "yolo9_rot90_obb": _case_yolo9_rot90_obb,
    "yolonas_train": _case_yolonas_train,
    "yolonas_affine_mixup": _case_yolonas_affine_mixup,
    "yolo9_train_det": _case_yolo9_train_det,
    "yolo9_train_obb": _case_yolo9_train_obb,
    "yolo9_train_seg": _case_yolo9_train_seg,
    "yolo9_mosaic_det": _case_yolo9_mosaic_det,
    "yolo9_mosaic_seg": _case_yolo9_mosaic_seg,
    "yolo9_val": _case_yolo9_val,
    "deim_train": _case_deim_train,
    "deim_train_weak_only": _case_deim_train_weak_only,
    "deim_imagenet_norm": _case_deim_imagenet_norm,
    "deim_multiscale_collate": _case_deim_multiscale_collate,
    "deim_sanitize_min_size": _case_deim_sanitize_min_size,
    "deim_pull_item_unresized": _case_deim_pull_item_unresized,
    "dfine_train": _case_dfine_train,
    "dfine_train_weak_only": _case_dfine_train_weak_only,
    "dfine_multiscale_collate": _case_dfine_multiscale_collate,
    "dfine_pull_item_letterboxed": _case_dfine_pull_item_letterboxed,
    "rfdetr_seg": _case_rfdetr_seg,
    "rfdetr_seg_no_crop": _case_rfdetr_seg_no_crop,
    "rfdetr_det": _case_rfdetr_det,
    "rfdetr_det_obb": _case_rfdetr_det_obb,
    "rfdetr_pose": _case_rfdetr_pose,
    "yolonas_pose_train": _case_yolonas_pose_train,
    "yolonas_pose_val": _case_yolonas_pose_val,
    "ec_pose_train": _case_ec_pose_train,
    "ec_pose_val": _case_ec_pose_val,
    "yolonas_pose_train_forced": _case_yolonas_pose_train_forced,
    "yolonas_pose_train_nonsquare": _case_yolonas_pose_train_nonsquare,
    "ec_pose_train_forced": _case_ec_pose_train_forced,
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_augment_parity(name):
    fixture = FIXTURE_DIR / f"{name}.npz"
    if not fixture.exists():
        pytest.fail(
            f"Missing golden fixture {fixture}. Run "
            f"`python tests/unit/test_augment_parity.py --generate`."
        )
    expected = np.load(fixture, allow_pickle=True)
    fixture_env = expected[ENV_KEY].item() if ENV_KEY in expected.files else {}
    _check_reference_env(fixture_env)
    actual = CASES[name]()
    data_keys = [k for k in expected.files if k != ENV_KEY]
    assert set(data_keys) == set(actual), (
        f"Output keys changed for {name}: {sorted(actual)} vs {sorted(data_keys)}"
    )
    for key in data_keys:
        # strict=True also pins dtype and shape — augmentation output dtype is
        # part of the training contract, not an implementation detail.
        np.testing.assert_array_equal(
            np.asarray(actual[key]),
            expected[key],
            err_msg=f"{name}/{key} diverged from golden fixture",
            strict=True,
        )


def _generate():
    if os.environ.get("CI"):
        raise SystemExit(
            "Refusing to regenerate golden fixtures under CI. Fixtures must be "
            "re-pinned deliberately on a developer machine."
        )
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    env = np.array(_env_fingerprint(), dtype=object)
    for name, fn in sorted(CASES.items()):
        out = fn()
        np.savez_compressed(FIXTURE_DIR / f"{name}.npz", **out, **{ENV_KEY: env})
        shapes = {k: tuple(np.asarray(v).shape) for k, v in out.items()}
        print(f"wrote {name}.npz {shapes}")


if __name__ == "__main__":
    if "--generate" in sys.argv:
        _generate()
    else:
        print(__doc__)
