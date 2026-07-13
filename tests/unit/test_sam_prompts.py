"""Unit tests for LibreSAM prompt normalization (offline, no model weights)."""

import numpy as np
import pytest

from libreyolo.models.sam.prompts import (
    build_point_grid,
    normalize_boxes,
    normalize_labels,
    normalize_points,
    validate_max_det,
)

pytestmark = pytest.mark.unit


class TestNormalizePoints:
    def test_none_passthrough(self):
        assert normalize_points(None) is None

    def test_single_point(self):
        # Documented: points=[x, y] -> one object, one point.
        assert normalize_points([900, 370]) == [[[900, 370]]]

    def test_multiple_points_are_separate_objects(self):
        # Documented: points=[[x, y], [x, y]] -> two objects, one point each.
        assert normalize_points([[400, 370], [900, 370]]) == [
            [[400, 370]],
            [[900, 370]],
        ]

    def test_grouped_points_per_object(self):
        # Documented: points=[[[x, y], [x, y]]] -> one object, two points.
        assert normalize_points([[[400, 370], [900, 370]]]) == [
            [[400, 370], [900, 370]]
        ]

    def test_tuples_accepted(self):
        assert normalize_points((900, 370)) == [[[900, 370]]]

    def test_bad_point_length_raises(self):
        with pytest.raises(ValueError):
            normalize_points([[1, 2, 3]])

    def test_numpy_1d_point(self):
        # numpy is the natural carrier of click coords; must be accepted.
        assert normalize_points(np.array([900, 370])) == [[[900, 370]]]

    def test_numpy_2d_points_are_separate_objects(self):
        assert normalize_points(np.array([[400, 370], [900, 370]])) == [
            [[400, 370]],
            [[900, 370]],
        ]

    def test_ragged_point_counts_raise_clearly(self):
        # One 1-click object + one 2-click object would fail deep in the
        # processor with an opaque numpy error; reject it up front.
        with pytest.raises(ValueError, match="same number of points"):
            normalize_points([[[400, 370]], [[500, 370], [600, 370]]])

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "x", True])
    def test_nonfinite_or_nonnumeric_coordinates_raise(self, value):
        with pytest.raises(ValueError, match="finite numbers"):
            normalize_points([value, 1])

    @pytest.mark.parametrize("point", [[-1, 2], [10, 2], [2, -1], [2, 8]])
    def test_out_of_image_points_raise(self, point):
        with pytest.raises(ValueError, match="outside image bounds"):
            normalize_points(point, image_size=(10, 8))

    def test_points_on_last_pixel_are_valid(self):
        assert normalize_points([9, 7], image_size=(10, 8)) == [[[9.0, 7.0]]]


class TestNormalizeLabels:
    def test_default_all_positive(self):
        pts = normalize_points([[400, 370], [900, 370]])
        assert normalize_labels(None, pts) == [[1], [1]]

    def test_default_for_grouped(self):
        pts = normalize_points([[[400, 370], [900, 370]]])
        assert normalize_labels(None, pts) == [[1, 1]]

    def test_explicit_flat_labels_map_per_object(self):
        pts = normalize_points([[400, 370], [900, 370]])
        assert normalize_labels([1, 0], pts) == [[1], [0]]

    def test_explicit_grouped_labels(self):
        pts = normalize_points([[[400, 370], [900, 370]]])
        assert normalize_labels([[1, 0]], pts) == [[1, 0]]

    def test_flat_labels_on_single_object_are_per_point(self):
        # The natural positive/negative click on ONE object: points grouped,
        # labels flat [1, 0] should map to that object's two points.
        pts = normalize_points([[[400, 370], [900, 370]]])
        assert normalize_labels([1, 0], pts) == [[1, 0]]

    def test_mixed_nesting_labels_raise_value_error(self):
        # [[1], 0] reports depth 2 but isn't uniformly nested — must be a clean
        # ValueError, not a TypeError from flattening a scalar.
        pts = normalize_points([[400, 370], [900, 370]])
        with pytest.raises(ValueError, match="mixed nesting"):
            normalize_labels([[1], 0], pts)

    def test_out_of_range_label_raises(self):
        # A typo'd label (2) would silently produce a wrong mask; reject it.
        pts = normalize_points([900, 370])
        with pytest.raises(ValueError, match="1 .positive. or 0 .negative."):
            normalize_labels([2], pts)

    def test_truncating_float_label_raises(self):
        pts = normalize_points([900, 370])
        with pytest.raises(ValueError, match="1 .positive. or 0 .negative."):
            normalize_labels([1.9], pts)

    def test_float_one_point_zero_is_accepted(self):
        pts = normalize_points([900, 370])
        assert normalize_labels([1.0], pts) == [[1]]

    def test_shape_mismatch_raises(self):
        pts = normalize_points([[400, 370], [900, 370]])
        with pytest.raises(ValueError):
            normalize_labels([1], pts)

    def test_labels_without_points_raises(self):
        with pytest.raises(ValueError):
            normalize_labels([1], None)


class TestNormalizeBoxes:
    def test_none_passthrough(self):
        assert normalize_boxes(None) is None

    def test_single_box(self):
        assert normalize_boxes([100, 100, 200, 200]) == [[100, 100, 200, 200]]

    def test_multiple_boxes(self):
        assert normalize_boxes([[10, 10, 20, 20], [30, 30, 40, 40]]) == [
            [10, 10, 20, 20],
            [30, 30, 40, 40],
        ]

    def test_bad_box_length_raises(self):
        with pytest.raises(ValueError):
            normalize_boxes([1, 2, 3])

    def test_numpy_single_box(self):
        assert normalize_boxes(np.array([100, 100, 200, 200])) == [
            [100, 100, 200, 200]
        ]

    @pytest.mark.parametrize(
        "box",
        [
            [4, 1, 2, 5],
            [1, 4, 5, 2],
            [1, 1, float("nan"), 5],
            [1, 1, float("inf"), 5],
        ],
    )
    def test_reversed_or_nonfinite_boxes_raise(self, box):
        with pytest.raises(ValueError):
            normalize_boxes(box)

    @pytest.mark.parametrize(
        "box", [[-1, 0, 5, 5], [0, -1, 5, 5], [0, 0, 11, 5], [0, 0, 5, 9]]
    )
    def test_out_of_image_boxes_raise(self, box):
        with pytest.raises(ValueError, match="outside image bounds"):
            normalize_boxes(box, image_size=(10, 8))

    def test_full_image_box_is_valid(self):
        assert normalize_boxes([0, 0, 10, 8], image_size=(10, 8)) == [
            [0.0, 0.0, 10.0, 8.0]
        ]


class TestBuildPointGrid:
    def test_count(self):
        assert len(build_point_grid(16)) == 256

    def test_normalized_range_and_symmetry(self):
        grid = build_point_grid(2)
        # 2x2 grid centered in each quadrant: 0.25 and 0.75 on each axis.
        assert grid == [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
        ]

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            build_point_grid(0)


@pytest.mark.parametrize("value", [-1, 0, 1.5, True, "3"])
def test_validate_max_det_rejects_nonpositive_or_noninteger_values(value):
    with pytest.raises(ValueError, match="integer >= 1"):
        validate_max_det(value)


def test_validate_max_det_accepts_positive_integer():
    assert validate_max_det(np.int64(3)) == 3
