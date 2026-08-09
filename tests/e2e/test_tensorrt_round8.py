"""TensorRT FP32 parity for the Round 8 edge-export families.

The nine promoted cases cover the complete double-tick contract:

1. export a full LibreYOLO architecture;
2. reload the engine through the public factory;
3. compare raw runtime tensors against the prepared PyTorch export graph;
4. reject constant/disconnected graphs with a second input; and
5. compare the public ``predict()`` result using the task threshold.

PIDNet is the tenth measured case. It gets an explicit available status because
floor because repeated TensorRT builds can straddle the stricter promotion
threshold. The models use deterministic random weights so this suite validates
conversion and runtime behavior, not task accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from .conftest import requires_tensorrt

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.extended_backend,
    pytest.mark.tensorrt,
    pytest.mark.trt,
]


@dataclass(frozen=True)
class TensorRTRound8Case:
    class_name: str
    size: str
    task: str
    imgsz: int
    nb_classes: int


ROUND8_VALIDATED_CASES = (
    TensorRTRound8Case("LibreMobileNetV4", "s", "classify", 224, 5),
    TensorRTRound8Case("LibreConvNeXt", "t", "classify", 224, 5),
    TensorRTRound8Case("LibreEfficientNetV2", "b0", "classify", 224, 5),
    TensorRTRound8Case("LibreResNet", "18", "classify", 224, 5),
    TensorRTRound8Case("LibreFOMO", "s", "point", 96, 2),
    TensorRTRound8Case("LibreRealESRGAN", "x4t", "restore", 16, 1),
    TensorRTRound8Case("LibreNAFNet", "s", "restore", 16, 1),
    TensorRTRound8Case("LibreSwinIR", "s", "restore", 64, 1),
    TensorRTRound8Case("LibreDepthAnythingV2", "s", "depth", 70, 1),
)

ROUND8_AVAILABLE_PIDNET = TensorRTRound8Case(
    "LibrePIDNet",
    "s",
    "semantic",
    64,
    3,
)


def _build_model(case: TensorRTRound8Case):
    import libreyolo

    model_cls = getattr(libreyolo, case.class_name)
    return model_cls(
        model_path=None,
        size=case.size,
        task=case.task,
        nb_classes=case.nb_classes,
        device="cuda",
    )


def _tensor_outputs(output) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, dict):
        values = output.values()
    elif isinstance(output, (tuple, list)):
        values = output
    else:
        raise TypeError(f"Unsupported export output type: {type(output)!r}")
    tensors = []
    for value in values:
        tensors.extend(_tensor_outputs(value))
    return tensors


def _raw_outputs(model, tensor: torch.Tensor, imgsz: int) -> list[np.ndarray]:
    from libreyolo.export.exporter import TensorRTExporter

    with TensorRTExporter(model)._model_context(
        torch.device("cuda"),
        False,
        False,
        1,
        (imgsz, imgsz),
    ) as (wrapped, _), torch.inference_mode():
        output = wrapped(tensor)
    return [value.detach().float().cpu().numpy() for value in _tensor_outputs(output)]


def _image(imgsz: int) -> np.ndarray:
    image = np.random.default_rng(81).integers(
        0,
        256,
        size=(imgsz, imgsz, 3),
        dtype=np.uint8,
    )
    if imgsz == 70:
        image[..., 0] = np.arange(imgsz, dtype=np.uint8)[None, :]
        image[..., 1] = np.arange(imgsz, dtype=np.uint8)[:, None]
        image[..., 2] = 127
    return image


def _prepare_non_degenerate_model(case, model) -> None:
    if case.task != "point":
        return

    from libreyolo.models.fomo.utils import preprocess_image

    training_image = _image(case.imgsz)
    tensor, *_ = preprocess_image(training_image, case.imgsz)
    tensor = tensor.cuda()
    network = model.model.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    targets = torch.zeros(1, 12, 12, dtype=torch.long, device="cuda")
    targets[0, 2, 3] = 1
    targets[0, 7, 8] = 2
    targets[0, 4, 9] = 1
    targets[0, 9, 2] = 2
    class_weights = torch.tensor([0.02, 1.0, 1.0], device="cuda")
    for _ in range(80):
        logits = network(tensor)
        loss = torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=class_weights,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    network.eval()


def _psnr(expected: np.ndarray, actual: np.ndarray, peak: float) -> float:
    mse = float(
        np.mean(
            (expected.astype(np.float64) - actual.astype(np.float64)) ** 2
        )
    )
    return float("inf") if mse == 0 else 20.0 * np.log10(peak / np.sqrt(mse))


def _assert_raw_parity(
    case,
    expected_outputs: list[np.ndarray],
    actual_outputs: list[np.ndarray],
) -> None:
    assert len(actual_outputs) == len(expected_outputs)
    for expected, actual in zip(expected_outputs, actual_outputs):
        assert actual.shape == expected.shape
        if case.task == "classify":
            expected_tensor = torch.from_numpy(expected).flatten()
            actual_tensor = torch.from_numpy(actual).flatten()
            cosine = torch.nn.functional.cosine_similarity(
                expected_tensor[None],
                actual_tensor[None],
            )
            assert float(cosine) > 0.999
            assert int(expected_tensor.argmax()) == int(actual_tensor.argmax())
        elif case.task == "semantic":
            expected_tensor = torch.from_numpy(expected).flatten()
            actual_tensor = torch.from_numpy(actual).flatten()
            cosine = torch.nn.functional.cosine_similarity(
                expected_tensor[None],
                actual_tensor[None],
            )
            assert float(cosine) > 0.999
            agreement = np.mean(
                np.argmax(expected, axis=1) == np.argmax(actual, axis=1)
            )
            assert float(agreement) > 0.95
        elif case.task in {"restore", "depth"}:
            peak = max(float(np.max(np.abs(expected))), 1e-6)
            assert _psnr(expected, actual, peak) > 40.0
        elif case.task == "point":
            expected_tensor = torch.from_numpy(expected).flatten()
            actual_tensor = torch.from_numpy(actual).flatten()
            cosine = torch.nn.functional.cosine_similarity(
                expected_tensor[None],
                actual_tensor[None],
            )
            assert float(cosine) > 0.999
            expected_peaks = expected[0].reshape(expected.shape[1], -1).argmax(axis=1)
            actual_peaks = actual[0].reshape(actual.shape[1], -1).argmax(axis=1)
            width = expected.shape[-1]
            expected_y, expected_x = np.divmod(expected_peaks, width)
            actual_y, actual_x = np.divmod(actual_peaks, width)
            assert np.max(np.abs(expected_x - actual_x)) <= 1
            assert np.max(np.abs(expected_y - actual_y)) <= 1
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


def _assert_predict_parity(case, native_model, backend, image) -> None:
    conf = 0.1 if case.task == "point" else 0.0
    native = native_model.predict(image, imgsz=case.imgsz, conf=conf, max_det=25)
    actual = backend.predict(image, conf=conf, max_det=25)

    if case.task == "classify":
        expected_probs = native.probs.data.cpu()
        actual_probs = actual.probs.data.cpu()
        cosine = torch.nn.functional.cosine_similarity(
            expected_probs[None], actual_probs[None]
        )
        assert float(cosine) > 0.999
        assert int(expected_probs.argmax()) == int(actual_probs.argmax())
        return

    if case.task == "point":
        torch.testing.assert_close(
            actual.points.data.cpu(),
            native.points.data.cpu(),
            rtol=1e-3,
            atol=1.0,
        )
        return

    if case.task == "semantic":
        agreement = (
            native.semantic_mask.data.cpu() == actual.semantic_mask.data.cpu()
        ).float().mean()
        assert float(agreement) > 0.95
        return

    if case.task == "restore":
        expected_rgb = native.restored.array.astype(np.float64)
        actual_rgb = actual.restored.array.astype(np.float64)
        assert actual_rgb.shape == expected_rgb.shape
        assert _psnr(expected_rgb, actual_rgb, 255.0) > 40.0
        return

    if case.task == "depth":
        expected_depth = native.depth_map.data.cpu().numpy()
        actual_depth = actual.depth_map.data.cpu().numpy()
        peak = max(float(np.max(np.abs(expected_depth))), 1e-6)
        assert _psnr(expected_depth, actual_depth, peak) > 40.0
        return

    raise AssertionError(f"Unhandled Round 8 task: {case.task}")


def _run_tensorrt_case(tmp_path, case):
    from libreyolo import LibreYOLO

    torch.manual_seed(0)
    model = _build_model(case)
    model.model.eval()
    _prepare_non_degenerate_model(case, model)

    first = torch.rand(1, 3, case.imgsz, case.imgsz, device="cuda")
    second = 1.0 - first
    expected_first = _raw_outputs(model, first, case.imgsz)
    expected_second = _raw_outputs(model, second, case.imgsz)

    engine_path = tmp_path / f"{model.FAMILY}.engine"
    artifact = model.export(
        format="tensorrt",
        output_path=str(engine_path),
        imgsz=case.imgsz,
        dynamic=False,
        half=False,
        simplify=False,
    )
    backend = LibreYOLO(artifact, device="cuda")

    actual_first = backend._run_inference(first.cpu().numpy())
    actual_second = backend._run_inference(second.cpu().numpy())
    _assert_raw_parity(case, expected_first, actual_first)
    _assert_raw_parity(case, expected_second, actual_second)

    expected_signal = max(
        float(np.max(np.abs(first_out - second_out)))
        for first_out, second_out in zip(expected_first, expected_second)
    )
    actual_signal = max(
        float(np.max(np.abs(first_out - second_out)))
        for first_out, second_out in zip(actual_first, actual_second)
    )
    parity_error = max(
        float(np.max(np.abs(expected - actual)))
        for expected, actual in zip(expected_first, actual_first)
    )
    assert expected_signal > 1e-12
    assert actual_signal > max(1e-12, 100.0 * parity_error)

    assert backend.model_family == model.FAMILY
    assert backend.task == case.task
    assert backend.imgsz == case.imgsz
    _assert_predict_parity(case, model, backend, _image(case.imgsz))

    del backend, model
    torch.cuda.empty_cache()


@requires_tensorrt
@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    ROUND8_VALIDATED_CASES,
    ids=lambda case: case.class_name,
)
def test_tensorrt_round8_raw_and_predict_parity(tmp_path, case):
    _run_tensorrt_case(tmp_path, case)


@requires_tensorrt
@pytest.mark.slow
def test_tensorrt_round8_pidnet_available_parity_floor(tmp_path):
    """Keep PIDNet runnable while its build-to-build drift blocks promotion."""
    from libreyolo import LibreYOLO

    case = ROUND8_AVAILABLE_PIDNET
    torch.manual_seed(0)
    model = _build_model(case)
    model.model.eval()
    first = torch.rand(1, 3, case.imgsz, case.imgsz, device="cuda")
    second = 1.0 - first
    expected_outputs = (
        _raw_outputs(model, first, case.imgsz),
        _raw_outputs(model, second, case.imgsz),
    )

    engine_path = tmp_path / f"{model.FAMILY}.engine"
    artifact = model.export(
        format="tensorrt",
        output_path=str(engine_path),
        imgsz=case.imgsz,
        dynamic=False,
        half=False,
        simplify=False,
    )
    backend = LibreYOLO(artifact, device="cuda")

    for tensor, expected in zip((first, second), expected_outputs):
        actual = backend._run_inference(tensor.cpu().numpy())
        assert len(actual) == len(expected)
        for expected_tensor, actual_tensor in zip(expected, actual):
            cosine = torch.nn.functional.cosine_similarity(
                torch.from_numpy(expected_tensor).flatten()[None],
                torch.from_numpy(actual_tensor).flatten()[None],
            )
            assert float(cosine) > 0.99
            agreement = np.mean(
                np.argmax(expected_tensor, axis=1)
                == np.argmax(actual_tensor, axis=1)
            )
            assert float(agreement) > 0.95

    assert backend.model_family == model.FAMILY
    assert backend.task == case.task
    assert backend.imgsz == case.imgsz
    _assert_predict_parity(case, model, backend, _image(case.imgsz))
