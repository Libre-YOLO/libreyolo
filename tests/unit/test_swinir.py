"""Unit tests for the LibreSwinIR super-resolution family."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreSwinIR
from libreyolo.models import LibreYOLO
from libreyolo.models.autoconvert import autoconvert_upstream_checkpoint
from libreyolo.postprocess.swinir import postprocess
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit


def test_public_contract_and_download_url():
    assert LibreSwinIR.FAMILY == "swinir"
    assert LibreSwinIR.SUPPORTED_TASKS == ("restore",)
    assert LibreSwinIR.detect_size_from_filename("LibreSwinIRm-restore.pt") == "m"
    assert LibreSwinIR.detect_size_from_filename("LibreSwinIRm.pt") is None
    assert LibreSwinIR.get_download_url("LibreSwinIRm-restore.pt") == (
        "https://huggingface.co/LibreYOLO/LibreSwinIRm-restore/resolve/main/"
        "LibreSwinIRm-restore.pt"
    )


@pytest.mark.parametrize("size,expected_entries", [("s", 366), ("m", 552), ("l", 859)])
def test_state_dict_detection(size, expected_entries):
    model = LibreSwinIR(size=size, device="cpu")
    state = model.model.state_dict()
    assert len(state) == expected_entries
    assert LibreSwinIR.can_load(state)
    assert LibreSwinIR.detect_size(state) == size
    assert model.restore_scale == 4


def test_rejects_foreign_checkpoints():
    realesrgan_like = {
        "conv_first.weight": torch.zeros(64, 3, 3, 3),
        "body.0.rdb1.conv1.weight": torch.zeros(32, 64, 3, 3),
    }
    yolo_like = {
        "model.0.conv.weight": torch.zeros(16, 3, 3, 3),
        "model.22.cv3.0.0.conv.weight": torch.zeros(16, 16, 3, 3),
    }
    assert not LibreSwinIR.can_load(realesrgan_like)
    assert not LibreSwinIR.can_load(yolo_like)


def test_forward_and_predict_crop_padding():
    model = LibreSwinIR(size="s", device="cpu")
    model.model.eval()
    with torch.no_grad():
        output = model.model(torch.zeros(1, 3, 8, 16))
    assert tuple(output.shape) == (1, 3, 32, 64)

    image = Image.fromarray(np.zeros((7, 9, 3), dtype=np.uint8), mode="RGB")
    result = model.predict(image)
    assert result.restore_scale == 4
    assert result.restored.array.shape == (28, 36, 3)
    assert result.restored.array.dtype == np.uint8


def test_tiled_predict_shape():
    model = LibreSwinIR(size="s", device="cpu")
    image = Image.fromarray(np.zeros((16, 24, 3), dtype=np.uint8), mode="RGB")
    result = model.predict(image, tile=8, tile_pad=8)
    assert result.restored.array.shape == (64, 96, 3)


def test_concurrent_predictions_keep_per_call_tiling(monkeypatch):
    model = LibreSwinIR(size="s", device="cpu")
    first_ready = threading.Event()
    second_forward = threading.Event()
    release_second = threading.Event()
    observed = {}
    observed_lock = threading.Lock()
    original_preprocess = model._preprocess

    def synchronized_preprocess(image, *args, **kwargs):
        prepared = original_preprocess(image, *args, **kwargs)
        is_second = np.asarray(image).mean() > 127
        if is_second:
            assert first_ready.wait(timeout=5)
        else:
            first_ready.set()
            assert second_forward.wait(timeout=5)
        return prepared

    def recording_forward(network, image, *, scale, tile_size, tile_pad):  # noqa: ARG001
        is_second = image.mean().item() > 0.5
        label = "second" if is_second else "first"
        with observed_lock:
            observed[label] = (tile_size, tile_pad)
        if is_second:
            second_forward.set()
            assert release_second.wait(timeout=5)
        else:
            release_second.set()
        return image.new_zeros(
            image.shape[0], 3, image.shape[2] * scale, image.shape[3] * scale
        )

    monkeypatch.setattr(model, "_preprocess", synchronized_preprocess)
    monkeypatch.setattr(
        "libreyolo.models.swinir.model.forward_with_optional_tiling",
        recording_forward,
    )

    first_image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
    second_image = Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8), mode="RGB")
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(model.predict, first_image, tile=8, tile_pad=4)
        second = executor.submit(model.predict, second_image, tile=16, tile_pad=12)
        first.result(timeout=10)
        second.result(timeout=10)

    assert observed == {"first": (8, 4), "second": (16, 12)}


def test_postprocess_crop_scale_and_quantization():
    output = torch.zeros(1, 3, 32, 48)
    output[:, 2] = 0.5
    restored = postprocess(output, original_size=(5, 3), scale=4)
    assert restored.shape == (12, 20, 3)
    assert restored.dtype == np.uint8
    assert np.all(restored[..., 2] == 128)


def test_factory_loads_wrapped_checkpoint(tmp_path):
    source = LibreSwinIR(size="s", device="cpu")
    checkpoint = wrap_libreyolo_checkpoint(
        source.model.state_dict(),
        model_family="swinir",
        size="s",
        task="restore",
        nc=1,
        names={0: "image"},
        imgsz=64,
        scale=4,
        degradation="super-resolution",
    )
    path = tmp_path / "LibreSwinIRs-restore.pt"
    torch.save(checkpoint, path)
    loaded = LibreYOLO(str(path), device="cpu")
    assert isinstance(loaded, LibreSwinIR)
    assert loaded.size == "s"
    assert loaded.task == "restore"


def test_upstream_params_wrapper_autoconverts(tmp_path):
    source = LibreSwinIR(size="s", device="cpu")
    raw_path = tmp_path / "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
    torch.save({"params": source.model.state_dict()}, raw_path)
    converted = autoconvert_upstream_checkpoint(str(raw_path))
    assert converted is not None
    assert converted.endswith("-LibreSwinIRs-restore.pt")
    loaded = LibreYOLO(converted, device="cpu")
    assert isinstance(loaded, LibreSwinIR)


def test_training_is_explicitly_out_of_scope():
    model = LibreSwinIR(size="s", device="cpu")
    with pytest.raises(NotImplementedError, match="inference and validation only"):
        model.train(data="unused.yaml")


@pytest.mark.parametrize("format", ["onnx", "torchscript", "openvino", "tflite"])
def test_fixed_canvas_export_predict_parity(tmp_path, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "openvino":
        pytest.importorskip("openvino")
    if format == "tflite":
        pytest.importorskip("onnx2tf")
        pytest.importorskip("ai_edge_litert")

    from libreyolo.export.exporter import OnnxExporter

    torch.manual_seed(0)
    model = LibreSwinIR(size="s", device="cpu")
    model.model.eval()
    tensor = torch.rand(1, 3, 64, 64)
    with OnnxExporter(model)._model_context(
        "cpu", False, False, 1, (64, 64)
    ) as (wrapped, _):
        with torch.no_grad():
            expected = wrapped(tensor).detach().cpu().numpy()

    output_path = tmp_path / {
        "onnx": "swinir.onnx",
        "torchscript": "swinir.torchscript",
        "openvino": "swinir_openvino",
        "tflite": "swinir.tflite",
    }[format]
    artifact = model.export(
        format=format,
        imgsz=64,
        dynamic=False,
        simplify=False,
        output_path=str(output_path),
    )
    backend = LibreYOLO(artifact, device="cpu")
    actual = backend._run_inference(tensor.numpy())[0]
    rtol, atol = (
        (2e-3, 2e-2)
        if format in {"onnx", "openvino", "tflite"}
        else (1e-3, 1e-3)
    )
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    image = np.random.default_rng(64).integers(
        0, 256, size=(64, 64, 3), dtype=np.uint8
    )
    native = model.predict(image).restored.array
    exported = backend.predict(image).restored.array
    mse = np.mean(
        (native.astype(np.float64) - exported.astype(np.float64)) ** 2
    )
    psnr = float("inf") if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))
    assert psnr > 40.0
    assert backend.restore_scale == 4
    assert exported.shape == (256, 256, 3)
    assert exported.dtype == np.uint8
