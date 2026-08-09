"""Unit tests for the surface-normal result contract."""

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.base.inference import InferenceRunner
from libreyolo.utils.drawing import draw_normal_map
from libreyolo.utils.results import NormalMap, Results

pytestmark = pytest.mark.unit


def _front_wall(height: int = 3, width: int = 4) -> torch.Tensor:
    normals = torch.zeros((height, width, 3))
    normals[..., 2] = -1.0
    return normals


def test_normal_map_validates_shape_and_casts_float32():
    normal_map = NormalMap(np.zeros((3, 4, 3), dtype=np.float64))

    assert normal_map.data.dtype == np.float32
    assert normal_map.orig_shape == (3, 4)
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        NormalMap(torch.zeros((3, 4)))
    with pytest.raises(ValueError, match="does not match original image shape"):
        NormalMap(_front_wall(), orig_shape=(30, 40))


def test_assert_normalized_accepts_unit_vectors_and_rejects_bad_values():
    NormalMap(_front_wall()).assert_normalized()

    with pytest.raises(AssertionError, match="unit-normalized"):
        NormalMap(torch.zeros((2, 2, 3))).assert_normalized()
    contaminated = _front_wall(2, 2)
    contaminated[0, 0, 0] = float("nan")
    with pytest.raises(AssertionError, match="non-finite"):
        NormalMap(contaminated).assert_normalized()
    with pytest.raises(ValueError, match="non-negative"):
        NormalMap(_front_wall()).assert_normalized(atol=-1)


def test_dense_slicing_and_device_moves_preserve_whole_map():
    normal_map = NormalMap(_front_wall())

    sliced = normal_map[0]
    moved = normal_map.to("cpu")
    as_numpy = normal_map.numpy()

    assert sliced.data.shape == (3, 4, 3)
    assert moved.data.device.type == "cpu"
    assert isinstance(as_numpy.data, np.ndarray)
    assert as_numpy.data.dtype == np.float32


def test_results_wires_normal_map_through_summary_moves_and_slicing():
    result = Results(
        boxes=None,
        orig_shape=(3, 4),
        path="wall.png",
        normal_map=NormalMap(_front_wall()),
    )

    assert len(result) == 1
    assert result.summary() == [
        {
            "name": "normal_map",
            "shape": [3, 4, 3],
            "frame": "opencv",
            "orientation": "camera-facing",
        }
    ]
    assert result.cpu().normal_map.data.shape == (3, 4, 3)
    assert result.numpy().normal_map.data.shape == (3, 4, 3)
    assert result[0].normal_map.data.shape == (3, 4, 3)
    assert "normal_map=NormalMap" in repr(result)


def test_plot_uses_rgb_visualization_without_mutating_payload():
    normals = _front_wall(1, 2)
    normals[0, 1] = torch.tensor([1.0, 0.0, 0.0])
    original = normals.clone()
    result = Results(
        boxes=None,
        orig_shape=(1, 2),
        normal_map=NormalMap(normals),
    )

    rendered = np.asarray(result.plot())

    assert rendered.tolist() == [[[128, 128, 0], [255, 128, 128]]]
    torch.testing.assert_close(result.normal_map.data, original)


def test_draw_normal_map_renormalizes_after_resize_and_handles_invalid():
    image = Image.new("RGB", (4, 2), color="black")
    normals = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=np.float32,
    )

    rendered = draw_normal_map(image, normals)

    assert rendered.size == image.size
    assert rendered.mode == "RGB"
    invalid = normals.copy()
    invalid[0, 0] = np.nan
    pixels = np.asarray(draw_normal_map(Image.new("RGB", (2, 1)), invalid))
    assert pixels[0, 0].tolist() == [0, 0, 0]


def test_inference_runner_wraps_normal_payload():
    class DummyModel:
        names = {0: "normal"}

    result = InferenceRunner(DummyModel())._wrap_results(
        {"normal": _front_wall(3, 4)},
        original_size=(4, 3),
        image_path=None,
        classes=None,
    )

    assert result.boxes is None
    assert result.normal_map is not None
    assert result.normal_map.data.dtype == torch.float32
    result.normal_map.assert_normalized()
