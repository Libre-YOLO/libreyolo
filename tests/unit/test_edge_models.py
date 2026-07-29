"""Unit and ONNX parity tests for TEED and DexiNed edge specialists."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo import LibreDexiNed, LibreTEED, LibreYOLO
from libreyolo.models.dexined.nn import DexiNedCore
from libreyolo.models.edge_common import EdgeInferenceNet, preprocess_numpy
from libreyolo.models.teed.nn import TEEDCore

pytestmark = pytest.mark.unit


def test_teed_architecture_shape_parameter_count_and_detection():
    core = TEEDCore().eval()
    with torch.no_grad():
        outputs = core(torch.randn(2, 3, 32, 48))

    assert [tuple(output.shape) for output in outputs] == [
        (2, 1, 32, 48),
        (2, 1, 32, 48),
        (2, 1, 32, 48),
        (2, 1, 32, 48),
    ]
    assert sum(parameter.numel() for parameter in core.parameters()) == 58_910
    assert LibreTEED.can_load(core.state_dict())
    assert LibreTEED.detect_size(core.state_dict()) == "t"


def test_dexined_architecture_shape_and_detection():
    core = DexiNedCore().eval()
    with torch.no_grad():
        outputs = core(torch.randn(1, 3, 32, 48))

    assert len(outputs) == 7
    assert all(tuple(output.shape) == (1, 1, 32, 48) for output in outputs)
    assert LibreDexiNed.can_load(core.state_dict())
    assert LibreDexiNed.detect_size(core.state_dict()) == "b"
    assert not LibreTEED.can_load(core.state_dict())


def test_edge_inference_graph_owns_bgr_mean_and_sigmoid():
    core = TEEDCore().eval()
    runtime = EdgeInferenceNet(core).eval()
    canonical_rgb = torch.rand(1, 3, 32, 32)
    mean_bgr = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)

    with torch.no_grad():
        actual = runtime(canonical_rgb)
        expected = torch.sigmoid(
            core(canonical_rgb[:, [2, 1, 0]] * 255.0 - mean_bgr)[-1]
        )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert float(actual.min()) >= 0.0
    assert float(actual.max()) <= 1.0


def test_canonical_preprocess_is_rgb_float32_stretch():
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    image[..., 0] = 255

    chw, ratio = preprocess_numpy(image, 32)

    assert chw.shape == (3, 32, 32)
    assert chw.dtype == np.float32
    assert ratio == 1.0
    assert np.all(chw[0] == 1.0)
    assert np.all(chw[1:] == 0.0)


@pytest.mark.parametrize(
    ("model_class", "size", "divisor"),
    [(LibreTEED, "t", 4), (LibreDexiNed, "b", 16)],
)
def test_native_predict_returns_original_canvas_edge_map(model_class, size, divisor):
    model = model_class(None, size=size, device="cpu")
    model.model.eval()
    image = np.full((24, 40, 3), 127, dtype=np.uint8)

    result = model.predict(image, imgsz=32)

    assert result.edges is not None
    assert result.edges.data.shape == (24, 40)
    assert result.edges.data.dtype == torch.float32
    assert model.edge_imgsz_divisor == divisor


def test_public_factory_loads_teed_metadata_checkpoint(tmp_path):
    model = LibreTEED(None, device="cpu")
    checkpoint_path = tmp_path / "LibreTEEDt-edge.pt"
    torch.save(
        {
            "schema_version": "1.0",
            "libreyolo_version": "1.4.0",
            "model_family": "teed",
            "size": "t",
            "task": "edge",
            "nc": 1,
            "names": {0: "edge"},
            "imgsz": 352,
            "model": model.model.state_dict(),
        },
        checkpoint_path,
    )

    loaded = LibreYOLO(str(checkpoint_path), device="cpu")

    assert isinstance(loaded, LibreTEED)
    assert loaded.task == "edge"
    assert loaded.names == {0: "edge"}


@pytest.mark.parametrize(
    ("filename", "converter"),
    [
        ("LibreTEEDt-edge.pt", "convert_teed_weights.py"),
        ("LibreDexiNedb-edge.pt", "convert_dexined_weights.py"),
    ],
)
def test_public_factory_does_not_download_restricted_edge_weights(
    tmp_path, filename, converter
):
    with pytest.raises(FileNotFoundError, match=converter):
        LibreYOLO(str(tmp_path / filename), device="cpu")


@pytest.mark.parametrize(
    ("model_class", "size"),
    [(LibreTEED, "t"), (LibreDexiNed, "b")],
)
def test_onnx_edge_runtime_parity(tmp_path, model_class, size):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(7)
    model = model_class(None, size=size, device="cpu")
    model.model.eval()
    image = np.random.default_rng(7).integers(0, 256, (24, 40, 3), dtype=np.uint8)
    expected = model.predict(image, imgsz=32).edges.data.numpy()

    artifact = model.export(
        format="onnx",
        output_path=str(tmp_path / f"{model.FAMILY}.onnx"),
        imgsz=32,
        dynamic=False,
        simplify=False,
    )
    actual = LibreYOLO(artifact, device="cpu").predict(image).edges.data.numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("model_class", "size"),
    [(LibreTEED, "t"), (LibreDexiNed, "b")],
)
def test_edge_export_rejects_batch_and_non_onnx_formats(tmp_path, model_class, size):
    model = model_class(None, size=size, device="cpu")
    with pytest.raises(ValueError, match="batch-1"):
        model.export(format="onnx", batch=2, imgsz=32)
    with pytest.raises(NotImplementedError, match="ONNX-only"):
        model.export(
            format="torchscript",
            output_path=str(tmp_path / "edge.torchscript"),
            imgsz=32,
            dynamic=False,
        )
