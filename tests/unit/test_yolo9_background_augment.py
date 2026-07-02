"""Regression tests: background (empty-label) images must be augmented too.

``YOLO9TrainTransform`` used to early-return on images with no labels,
skipping HSV and flips entirely. On background-heavy datasets (the issue #484
reporter's is >80% background) most of the training data was therefore never
augmented: the model saw the exact same negative images every epoch while
labeled images varied. Backgrounds must get the same photometric/flip
augmentation as labeled images — there are no labels to keep in sync, so only
the image changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from libreyolo.data.augment.yolo9 import YOLO9TrainTransform, preproc

pytestmark = pytest.mark.unit

_EMPTY = np.zeros((0, 5), dtype=np.float32)


def _asymmetric_image(h=48, w=64):
    """Left half dark, right half bright — any flip is detectable."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, w // 2 :] = 200
    return img


def test_background_image_is_horizontally_flipped():
    """flip_prob=1 must flip an empty-label image (old code returned it
    unflipped)."""
    t = YOLO9TrainTransform(
        max_labels=10, flip_prob=1.0, vertical_flip_prob=0.0, hsv_prob=0.0
    )
    img = _asymmetric_image()

    out, labels = t(img.copy(), _EMPTY, (48, 64))
    expected, _ = preproc(img[:, ::-1], (48, 64))

    assert np.array_equal(out, expected), (
        "empty-label image was not flipped — background images are skipping "
        "augmentation (issue #484)"
    )
    # Labels stay an all-padding block.
    assert labels.shape == (10, 5)
    assert np.all(labels[:, 0] == -1)


def test_background_image_gets_hsv_augmentation(monkeypatch):
    """hsv_prob=1 must invoke ``augment_hsv`` on an empty-label image.

    Structural spy check rather than a pixel diff: ``augment_hsv``'s random
    draws are legitimately a visible no-op ~1/3 of the time (each gain is
    zeroed with p=0.5, sub-unit draws truncate to 0, and hue shifts are
    invisible on achromatic pixels), so asserting a pixel change is flaky."""
    calls = []
    monkeypatch.setattr(
        "libreyolo.data.augment.yolo9.augment_hsv",
        lambda image: calls.append(image.shape),
    )
    t = YOLO9TrainTransform(
        max_labels=10, flip_prob=0.0, vertical_flip_prob=0.0, hsv_prob=1.0
    )

    t(_asymmetric_image(), _EMPTY, (48, 64))

    assert calls, (
        "augment_hsv was not invoked on the empty-label path — background "
        "images are skipping photometric augmentation (issue #484)"
    )


def test_background_image_unchanged_when_augs_disabled():
    """With all probabilities 0 the empty path must reduce to plain
    letterbox preprocessing — pinning that the fix adds no unconditional
    behavior change."""
    t = YOLO9TrainTransform(
        max_labels=10, flip_prob=0.0, vertical_flip_prob=0.0, hsv_prob=0.0
    )
    img = _asymmetric_image()

    out, _ = t(img.copy(), _EMPTY, (48, 64))
    expected, _ = preproc(img, (48, 64))

    assert np.array_equal(out, expected)


def test_background_image_with_segments_returns_empty_masks():
    """The mask variant of the empty path still returns all-zero masks."""
    t = YOLO9TrainTransform(
        max_labels=10, flip_prob=1.0, vertical_flip_prob=0.0, hsv_prob=1.0
    )
    img = _asymmetric_image()

    out, labels, masks = t(img.copy(), _EMPTY, (48, 64), segments=[])

    assert masks.shape == (10, 48 // 4, 64 // 4)
    assert not masks.any()
    assert np.all(labels[:, 0] == -1)
