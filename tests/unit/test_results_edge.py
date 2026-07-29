"""Unit tests for the dense edge result contract."""

import numpy as np
import pytest
import torch

from libreyolo.utils.results import EdgeMap, NormalMap, Results

pytestmark = pytest.mark.unit


def test_edge_map_casts_float32_and_validates_shape_and_range():
    edge_map = EdgeMap(np.array([[0.0, 0.5, 1.0]], dtype=np.float64))

    assert edge_map.data.dtype == np.float32
    assert edge_map.orig_shape == (1, 3)
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        EdgeMap(torch.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        EdgeMap(torch.tensor([[1.01]]))
    with pytest.raises(ValueError, match="non-finite"):
        EdgeMap(np.array([[np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match="does not match original image shape"):
        EdgeMap(torch.zeros((2, 3)), orig_shape=(3, 2))


def test_binary_preserves_backend_and_validates_threshold():
    tensor_map = EdgeMap(torch.tensor([[0.49, 0.5]]))
    numpy_map = tensor_map.numpy()

    assert tensor_map.binary().tolist() == [[False, True]]
    assert tensor_map.binary().dtype == torch.bool
    assert numpy_map.binary().tolist() == [[False, True]]
    assert numpy_map.binary().dtype == np.bool_
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tensor_map.binary(1.1)


def test_results_wires_edges_and_normal_plural_alias():
    edge_map = EdgeMap(torch.tensor([[0.0, 0.5, 1.0]]))
    result = Results(boxes=None, orig_shape=(1, 3), edges=edge_map)

    assert len(result) == 1
    assert result.summary() == [
        {
            "name": "edges",
            "shape": [1, 3],
            "min": 0.0,
            "max": 1.0,
            "mean": 0.5,
        }
    ]
    assert result.cpu().edges.data.dtype == torch.float32
    assert result.numpy().edges.data.dtype == np.float32
    assert result[0].edges.data.shape == (1, 3)
    assert "edges=EdgeMap" in repr(result)

    normals = torch.zeros((1, 3, 3), dtype=torch.float32)
    normals[..., 2] = -1.0
    normal_result = Results(
        boxes=None,
        orig_shape=(1, 3),
        normal_map=NormalMap(normals),
    )
    assert normal_result.normals is normal_result.normal_map


def test_edge_plot_is_inverted_grayscale_without_mutating_payload():
    values = torch.tensor([[0.0, 0.5, 1.0]])
    result = Results(
        boxes=None,
        orig_shape=(1, 3),
        edges=EdgeMap(values.clone()),
    )

    rendered = np.asarray(result.plot())

    assert rendered.tolist() == [[[255, 255, 255], [128, 128, 128], [0, 0, 0]]]
    torch.testing.assert_close(result.edges.data, values)
