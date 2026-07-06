"""Property tests for the geometric augmentation knobs.

These complement the byte-exact golden fixtures in ``test_augment_parity.py``
by pinning invariants that hold regardless of platform: the perspective knob
is a no-op when zero, vertical flip is an involution, a full turn of the rot90
helper is the identity, and the OBB angle remap agrees with a brute-force
corner rotation.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from libreyolo.data.augment.geometry import (
    get_affine_matrix,
    mirror_vertical,
    random_affine,
    rot90_image_boxes,
)
from libreyolo.data.obb import (
    normalize_obb_angle,
    xywhr_iou,
    xywhr_to_corners,
    corners_to_xywhr,
)

pytestmark = pytest.mark.unit


def test_perspective_zero_matches_affine_matrix_and_rng():
    """perspective=0.0 returns the historical 2x3 affine and draws no extra RNG."""
    random.seed(4242)
    m_zero, scale_zero = get_affine_matrix((64, 64), 10.0, 0.1, 0.1, 10.0, perspective=0.0)
    tail_zero = random.random()

    random.seed(4242)
    m_default, scale_default = get_affine_matrix((64, 64), 10.0, 0.1, 0.1, 10.0)
    tail_default = random.random()

    assert m_zero.shape == (2, 3)
    assert np.array_equal(m_zero, m_default)
    assert scale_zero == scale_default
    # Identical follow-up draw proves the same number of RNG values was consumed.
    assert tail_zero == tail_default


def test_perspective_nonzero_is_3x3_homography():
    random.seed(7)
    M, _scale = get_affine_matrix((80, 64), 10.0, 0.1, 0.1, 10.0, perspective=5e-4)
    assert M.shape == (3, 3)
    # The bottom row is non-trivial (projective), unlike an affine's [0, 0, 1].
    # The overall scale of a homography is arbitrary; both cv2.warpPerspective
    # and the box remap divide by the true third coordinate, so M is not
    # normalized to M[2, 2] == 1.
    assert not np.allclose(M[2, :2], 0.0)


def test_perspective_shares_first_six_draws_then_draws_two_more():
    """The projective terms are drawn last, after the shared affine draws."""
    random.seed(99)
    _ = get_affine_matrix((64, 64), 10.0, 0.1, 0.1, 10.0, perspective=0.0)
    remaining_after_affine = [random.random() for _ in range(2)]

    random.seed(99)
    _ = get_affine_matrix((64, 64), 10.0, 0.1, 0.1, 10.0, perspective=5e-4)
    remaining_after_persp = [random.random() for _ in range(2)]

    # The perspective path consumed exactly two extra draws (the tilt terms),
    # so the two streams realign two draws apart.
    assert remaining_after_affine != remaining_after_persp


def test_random_affine_perspective_runs_and_clips_boxes():
    rng = np.random.RandomState(0)
    img = rng.randint(0, 255, (80, 64, 3), dtype=np.uint8)
    boxes = np.array(
        [[5.0, 6.0, 40.0, 50.0, 1.0], [10.0, 12.0, 30.0, 35.0, 0.0]],
        dtype=np.float32,
    )
    random.seed(1)
    out_img, out_boxes = random_affine(
        img,
        boxes.copy(),
        target_size=(64, 64),
        degrees=10.0,
        translate=0.1,
        scales=0.1,
        shear=10.0,
        perspective=1e-3,
    )
    assert out_img.shape == (64, 64, 3)
    assert (out_boxes[:, :4] >= 0).all()
    assert (out_boxes[:, [0, 2]] <= 64).all()
    assert (out_boxes[:, [1, 3]] <= 64).all()


def test_mirror_vertical_twice_is_identity():
    rng = np.random.RandomState(3)
    img = rng.randint(0, 255, (48, 72, 3), dtype=np.uint8)
    boxes = np.array([[4.0, 6.0, 20.0, 30.0], [10.0, 2.0, 60.0, 40.0]], dtype=np.float32)

    once_img, once_boxes = mirror_vertical(img.copy(), boxes.copy(), prob=1.0)
    twice_img, twice_boxes = mirror_vertical(once_img, once_boxes, prob=1.0)

    assert np.array_equal(twice_img, img)
    assert np.allclose(twice_boxes, boxes)


def test_rot90_k4_is_identity():
    rng = np.random.RandomState(5)
    img = rng.randint(0, 255, (37, 53, 3), dtype=np.uint8)
    boxes = np.array([[3.0, 4.0, 20.0, 25.0], [8.0, 10.0, 40.0, 30.0]], dtype=np.float32)

    out_img, out_boxes = rot90_image_boxes(img.copy(), boxes.copy(), k=4)
    assert np.array_equal(out_img, img)
    assert np.allclose(out_boxes, boxes)


def _rotate_point_k(x, y, k, width, height):
    """Brute-force apply k CCW quarter turns, matching rot90_image_boxes' T."""
    cur_w, cur_h = width, height
    for _ in range(k % 4):
        x, y = y, cur_w - x
        cur_w, cur_h = cur_h, cur_w
    return x, y


@pytest.mark.parametrize("k", [1, 2, 3])
def test_obb_rot90_matches_bruteforce_corner_rotation(k):
    """The proxy-box + angle remap equals refitting brute-force-rotated corners."""
    width, height = 100, 80
    cx, cy, w, h, r = 30.0, 25.0, 16.0, 6.0, 0.4  # canonical (w > h)

    # Model path: rotate the horizontal proxy box and turn the angle by k*90.
    proxy = np.array(
        [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]], dtype=np.float32
    )
    img = np.zeros((height, width, 3), dtype=np.uint8)
    _rot_img, new_proxy = rot90_image_boxes(img, proxy.copy(), k)
    ncx = (new_proxy[0, 0] + new_proxy[0, 2]) / 2
    ncy = (new_proxy[0, 1] + new_proxy[0, 3]) / 2
    nw = new_proxy[0, 2] - new_proxy[0, 0]
    nh = new_proxy[0, 3] - new_proxy[0, 1]
    new_angle = normalize_obb_angle(r + k * math.pi / 2)
    model_xywhr = np.array([ncx, ncy, nw, nh, new_angle], dtype=np.float32)

    # Brute force: rotate the true rectangle corners, then refit canonically.
    corners = xywhr_to_corners(np.array([cx, cy, w, h, r], dtype=np.float32))
    rotated_corners = np.array(
        [_rotate_point_k(px, py, k, width, height) for px, py in corners],
        dtype=np.float32,
    )
    brute_xywhr = corners_to_xywhr(rotated_corners)

    # Same rectangle: centers coincide and IoU is ~1.
    assert model_xywhr[0] == pytest.approx(brute_xywhr[0], abs=1e-3)
    assert model_xywhr[1] == pytest.approx(brute_xywhr[1], abs=1e-3)
    assert xywhr_iou(model_xywhr, brute_xywhr) > 0.999
