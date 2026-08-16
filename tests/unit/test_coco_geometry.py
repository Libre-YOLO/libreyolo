"""Unit tests for shared native-COCO bounding-box geometry."""

import math

import pytest

from libreyolo.utils.coco_geometry import clipped_coco_bbox_xyxy

pytestmark = pytest.mark.unit


def test_clip_preserves_the_visible_extent_of_negative_origin_box():
    assert clipped_coco_bbox_xyxy((-10, 5, 30, 20), 100, 50) == (
        0.0,
        5.0,
        20.0,
        25.0,
    )


@pytest.mark.parametrize(
    "bbox",
    [
        (math.nan, 0, 20, 20),
        (0, math.inf, 20, 20),
        (0, 0, -math.inf, 20),
        (True, 0, 20, 20),
        ("0", 0, 20, 20),
        (0, 0, 20),
    ],
)
def test_nonfinite_or_malformed_bbox_is_rejected(bbox):
    with pytest.raises(ValueError, match="finite"):
        clipped_coco_bbox_xyxy(bbox, 100, 100)


@pytest.mark.parametrize("dimensions", [(math.inf, 100), (100, 0), (True, 100)])
def test_invalid_image_dimensions_are_rejected(dimensions):
    with pytest.raises(ValueError, match="dimensions|finite"):
        clipped_coco_bbox_xyxy((0, 0, 20, 20), *dimensions)
