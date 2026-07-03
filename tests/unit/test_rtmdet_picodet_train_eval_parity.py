"""Train/eval input-distribution parity for RTMDet and PicoDet (Finding #8).

These families previously trained on a different input distribution than they
infer/validate on:

* RTMDet infers/validates on a BGR letterbox with mmdet mean/std applied in
  0-255 space, but training emitted a raw (unnormalized) 0-255 BGR letterbox.
* PicoDet infers/validates on an RGB **stretch** resize with ImageNet mean/std
  in 0-255 space, but training emitted a raw 0-255 **BGR letterbox** — a
  three-axis mismatch (letterbox vs stretch, BGR vs RGB, raw vs normalized).

Each test runs the same synthetic image (and box) through the training transform
and through the canonical validation preprocessor and asserts the two produce
the same tensor (same color order + geometry + normalization) and the same box
mapping — proving TRAIN now matches EVAL without a full training run.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from libreyolo.data.augment.boxes import cxcywh2xyxy
from libreyolo.models.picodet.trainer import PICODETTrainTransform
from libreyolo.models.rtmdet.trainer import RTMDetTrainTransform
from libreyolo.validation.preprocessors import (
    PICODETValPreprocessor,
    RTMDetValPreprocessor,
)

pytestmark = pytest.mark.unit


# Non-square so letterbox vs stretch geometry is actually exercised, and random
# per-channel content so a BGR/RGB color mix-up cannot pass by symmetry.
_ORIG_H, _ORIG_W = 48, 80
_INPUT = 64


def _synthetic_bgr() -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, size=(_ORIG_H, _ORIG_W, 3), dtype=np.uint8)


def _train_labels_to_xyxy(padded_labels: np.ndarray) -> np.ndarray:
    """[class, cx, cy, w, h] padded rows -> Nx4 xyxy for the valid rows."""
    valid = padded_labels[(padded_labels[:, 3] > 0) & (padded_labels[:, 4] > 0)]
    cxcywh = valid[:, 1:5].astype(np.float64).copy()
    return cxcywh2xyxy(cxcywh)


def test_rtmdet_train_matches_val_image_distribution():
    random.seed(0)
    img = _synthetic_bgr()

    train_tf = RTMDetTrainTransform(max_labels=120, flip_prob=0.0, hsv_prob=0.0)
    val_tf = RTMDetValPreprocessor((_INPUT, _INPUT))

    box = np.array([[10.0, 8.0, 60.0, 40.0, 3.0]], dtype=np.float32)
    train_img, _ = train_tf(img.copy(), box.copy(), (_INPUT, _INPUT))
    val_img, _ = val_tf(img.copy(), box.copy(), (_INPUT, _INPUT))

    assert train_img.shape == val_img.shape == (3, _INPUT, _INPUT)
    # Same BGR color order + top-left letterbox geometry + mmdet mean/std norm.
    assert np.allclose(train_img, val_img, atol=1e-4), (
        f"train/val max abs diff = {np.abs(train_img - val_img).max()}"
    )
    # Normalization is genuinely applied (raw 0-255 would never go negative and
    # would peak near 255, not ~3 after dividing by std ~57).
    assert train_img.min() < -0.5
    assert train_img.max() < 50.0


def test_picodet_train_matches_val_image_distribution():
    random.seed(0)
    img = _synthetic_bgr()

    train_tf = PICODETTrainTransform(max_labels=50, flip_prob=0.0, hsv_prob=0.0)
    val_tf = PICODETValPreprocessor((_INPUT, _INPUT))

    box = np.array([[10.0, 8.0, 60.0, 40.0, 3.0]], dtype=np.float32)
    train_img, _ = train_tf(img.copy(), box.copy(), (_INPUT, _INPUT))
    val_img, _ = val_tf(img.copy(), box.copy(), (_INPUT, _INPUT))

    assert train_img.shape == val_img.shape == (3, _INPUT, _INPUT)
    # Same RGB color order + full-canvas stretch geometry + ImageNet mean/std.
    assert np.allclose(train_img, val_img, atol=1e-4), (
        f"train/val max abs diff = {np.abs(train_img - val_img).max()}"
    )
    assert train_img.min() < -0.5
    assert train_img.max() < 50.0


def test_picodet_train_is_stretch_not_letterbox():
    """Stretch resize fills the whole canvas; a letterbox would leave a constant
    114-padded band on the short axis. Prove the geometry axis of the fix."""
    random.seed(0)
    img = _synthetic_bgr()  # 48x80 -> letterbox to 64 leaves padded bottom rows
    train_tf = PICODETTrainTransform(max_labels=50, flip_prob=0.0, hsv_prob=0.0)
    box = np.array([[10.0, 8.0, 60.0, 40.0, 3.0]], dtype=np.float32)
    out, _ = train_tf(img.copy(), box.copy(), (_INPUT, _INPUT))

    # r = min(64/48, 64/80) = 0.8 -> a letterbox would pad rows 38..63.
    padded_band = out[:, 40:, :]
    # Content varies across the band (stretch); a constant pad would have ~0 std.
    assert padded_band.std() > 0.1


def test_rtmdet_train_box_mapping_matches_val():
    random.seed(0)
    img = _synthetic_bgr()
    train_tf = RTMDetTrainTransform(max_labels=120, flip_prob=0.0, hsv_prob=0.0)
    val_tf = RTMDetValPreprocessor((_INPUT, _INPUT))

    box_orig = np.array([[10.0, 8.0, 60.0, 40.0, 3.0]], dtype=np.float32)
    # RTMDet letterbox ratio (top-left pad) — the coordinate frame both agree on.
    r = min(_INPUT / _ORIG_H, _INPUT / _ORIG_W)
    expected_xyxy = box_orig[:, :4] * r

    # Train scales original-coord boxes internally by the same letterbox ratio.
    _, train_labels = train_tf(img.copy(), box_orig.copy(), (_INPUT, _INPUT))
    train_xyxy = _train_labels_to_xyxy(train_labels)

    # Val's contract: labels arrive already in letterbox coords and are copied.
    val_in = np.hstack([expected_xyxy, box_orig[:, 4:5]]).astype(np.float32)
    _, val_labels = val_tf(img.copy(), val_in.copy(), (_INPUT, _INPUT))
    val_xyxy = val_labels[val_labels[:, 2] > 0][:, :4]

    assert train_xyxy.shape[0] == 1
    assert np.allclose(train_xyxy, expected_xyxy, atol=1e-3)
    assert np.allclose(val_xyxy, expected_xyxy, atol=1e-3)


def test_picodet_train_box_mapping_matches_val():
    random.seed(0)
    img = _synthetic_bgr()
    train_tf = PICODETTrainTransform(max_labels=50, flip_prob=0.0, hsv_prob=0.0)
    val_tf = PICODETValPreprocessor((_INPUT, _INPUT))

    box_orig = np.array([[10.0, 8.0, 60.0, 40.0, 3.0]], dtype=np.float32)
    # Independent x/y stretch scaling — both paths map original -> resized.
    scale_x = _INPUT / _ORIG_W
    scale_y = _INPUT / _ORIG_H
    expected = box_orig[:, :4].copy()
    expected[:, 0::2] *= scale_x
    expected[:, 1::2] *= scale_y

    _, train_labels = train_tf(img.copy(), box_orig.copy(), (_INPUT, _INPUT))
    train_xyxy = _train_labels_to_xyxy(train_labels)

    _, val_labels = val_tf(img.copy(), box_orig.copy(), (_INPUT, _INPUT))
    val_xyxy = val_labels[val_labels[:, 2] > 0][:, :4]

    assert train_xyxy.shape[0] == 1
    assert np.allclose(train_xyxy, expected, atol=1e-3)
    assert np.allclose(val_xyxy, expected, atol=1e-3)
