"""RT-DETR train and eval must use the SAME resize geometry (stretch-to-square).

Finding #31: the RT-DETR trainer letterboxed (aspect-preserving + padding) while
validation/inference stretch each image to the square input. That train/eval
geometry mismatch is fixed by ``RTDETRTrainTransform`` (stretch-to-square).

This test maps a known box on a non-square image through both the training
transform and the RT-DETR val preprocessors and asserts they land on the same
(normalized) coordinates — i.e. identical resize geometry — and that the mapping
is a pure stretch (``coord / original_dimension``), not a letterbox.
"""

import numpy as np
import pytest
from PIL import Image

from libreyolo.models.rtdetrv2.model import LibreRTDETRv2
from libreyolo.models.rtdetr.transforms import RTDETRTrainTransform
from libreyolo.validation.preprocessors import (
    RTDETRv2ValPreprocessor,
    RTDETRValPreprocessor,
)

# Non-square image: H=480, W=800 (4:2.4). BGR uint8, cv2 convention.
ORIG_H, ORIG_W = 480, 800
TARGET = 640
# Known box in original pixel coords: [x1, y1, x2, y2, class].
BOX = [100.0, 50.0, 300.0, 200.0, 0.0]


def _image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(ORIG_H, ORIG_W, 3), dtype=np.uint8)


def _train_norm_box():
    """Normalized [x1, y1, x2, y2] from the training transform (aug disabled)."""
    transform = RTDETRTrainTransform(
        max_labels=10, flip_prob=0.0, vertical_flip_prob=0.0, hsv_prob=0.0
    )
    targets = np.array([BOX], dtype=np.float32)
    img_out, padded = transform(_image(), targets, (TARGET, TARGET))

    # Output image is the square model canvas with no padding rows.
    assert img_out.shape == (3, TARGET, TARGET)
    assert 0.0 <= float(img_out.min()) and float(img_out.max()) <= 1.0

    # Output row is [class, x1, y1, x2, y2] normalized.
    assert padded[0, 0] == BOX[4]
    return padded[0, 1:5].astype(np.float64)


def _val_norm_box(preproc_cls):
    """Normalized [x1, y1, x2, y2] from a val preprocessor's stretch geometry."""
    preproc = preproc_cls(img_size=(TARGET, TARGET), max_labels=10)
    targets = np.array([BOX], dtype=np.float32)
    _img_out, padded = preproc(_image(), targets, (TARGET, TARGET))
    # Val preprocessors emit pixel [x1, y1, x2, y2, class] on the resized canvas.
    return padded[0, :4].astype(np.float64) / TARGET


@pytest.mark.unit
def test_train_matches_val_stretch_geometry():
    train_box = _train_norm_box()

    # Analytic pure-stretch mapping: normalized coord == pixel / original dim.
    expected = np.array(
        [
            BOX[0] / ORIG_W,
            BOX[1] / ORIG_H,
            BOX[2] / ORIG_W,
            BOX[3] / ORIG_H,
        ],
        dtype=np.float64,
    )

    # Train transform is a genuine stretch (matches the analytic mapping),
    # which by construction differs from letterbox on a non-square image.
    np.testing.assert_allclose(train_box, expected, atol=1e-4)

    # Train geometry == v1 and v2 val preprocessor geometry.
    np.testing.assert_allclose(
        train_box, _val_norm_box(RTDETRValPreprocessor), atol=1e-4
    )
    np.testing.assert_allclose(
        train_box, _val_norm_box(RTDETRv2ValPreprocessor), atol=1e-4
    )


@pytest.mark.unit
def test_train_transform_declares_unresized_optin():
    # The stretch geometry relies on receiving the original (un-letterboxed)
    # image from the dataset; guard that opt-in so a refactor can't silently
    # reintroduce a letterbox-then-stretch double resize.
    assert RTDETRTrainTransform.wants_unresized_image is True


@pytest.mark.unit
def test_rtdetrv2_predict_pixels_match_validation_pil_resize():
    """Protect the PIL kernel shared by prediction, validation, and CoreML."""
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(37, 61, 3), dtype=np.uint8)
    target = 53

    model = object.__new__(LibreRTDETRv2)
    predicted, *_ = LibreRTDETRv2._preprocess(
        model,
        Image.fromarray(rgb, mode="RGB"),
        color_format="rgb",
        input_size=target,
    )
    validated, _ = RTDETRv2ValPreprocessor(
        img_size=(target, target),
        max_labels=1,
    )(
        rgb[:, :, ::-1].copy(),
        np.empty((0, 5), dtype=np.float32),
        (target, target),
    )

    np.testing.assert_array_equal(predicted[0].numpy(), validated)
