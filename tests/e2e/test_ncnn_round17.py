"""Round 17 NCNN export, reload, raw-output, and public-predict parity.

The ten cases span classification, point detection, semantic segmentation,
restoration, depth, and the MIT-licensed YOLO9 flagship. The YOLO9 checkpoint
is runtime evidence only and is not committed.
"""

from __future__ import annotations

import gc
import importlib.util
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.extended_backend,
    pytest.mark.ncnn,
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("pnnx") is None
        or importlib.util.find_spec("ncnn") is None,
        reason="PNNX and NCNN are required",
    ),
]


@dataclass(frozen=True)
class NCNNRound17Case:
    class_name: str
    family: str
    size: str
    task: str
    imgsz: int
    nb_classes: int
    weights: str | None = None


ROUND17_CASES = (
    NCNNRound17Case("LibreMobileNetV4", "mobilenetv4", "s", "classify", 224, 5),
    NCNNRound17Case("LibreConvNeXt", "convnext", "t", "classify", 224, 5),
    NCNNRound17Case(
        "LibreEfficientNetV2",
        "efficientnetv2",
        "b0",
        "classify",
        224,
        5,
    ),
    NCNNRound17Case("LibreResNet", "resnet", "18", "classify", 224, 5),
    NCNNRound17Case("LibreFOMO", "fomo", "s", "point", 96, 2),
    NCNNRound17Case("LibrePIDNet", "pidnet", "s", "semantic", 64, 3),
    NCNNRound17Case("LibreRealESRGAN", "realesrgan", "x4t", "restore", 16, 1),
    NCNNRound17Case("LibreNAFNet", "nafnet", "s", "restore", 16, 1),
    NCNNRound17Case("LibreZipDepth", "zipdepth", "b", "depth", 64, 1),
    pytest.param(
        NCNNRound17Case(
            "LibreYOLO9",
            "yolo9",
            "t",
            "detect",
            64,
            80,
            "weights/LibreYOLO9t.pt",
        ),
        marks=pytest.mark.network,
    ),
)


def _build_model(case: NCNNRound17Case):
    import libreyolo

    if case.weights is not None:
        model = libreyolo.LibreYOLO(case.weights, device="cpu")
    else:
        model_cls = getattr(libreyolo, case.class_name)
        model = model_cls(
            model_path=None,
            size=case.size,
            task=case.task,
            nb_classes=case.nb_classes,
            device="cpu",
        )
    model.model.eval()
    assert model.FAMILY == case.family
    return model


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


def _native_outputs(
    model,
    tensor: torch.Tensor,
    imgsz: int,
) -> list[np.ndarray]:
    from libreyolo.export.exporter import NcnnExporter

    with (
        NcnnExporter(model)._model_context(
            torch.device("cpu"),
            False,
            False,
            1,
            (imgsz, imgsz),
        ) as (wrapped, _),
        torch.inference_mode(),
    ):
        output = wrapped(tensor)
    return [value.detach().float().cpu().numpy() for value in _tensor_outputs(output)]


def _prepare_input_sensitive_classifier(
    model,
    first: torch.Tensor,
    second: torch.Tensor,
) -> None:
    linear_layers = [
        module
        for module in model.model.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    assert linear_layers
    classifier = linear_layers[-1]
    features = []

    def capture_features(_module, inputs):
        features.append(inputs[0].detach())

    handle = classifier.register_forward_pre_hook(capture_features)
    try:
        with torch.inference_mode():
            model.model(first)
            model.model(second)
    finally:
        handle.remove()

    delta = (features[1] - features[0]).flatten()
    norm = delta.norm()
    assert float(norm) > 1e-12
    direction = delta / norm
    with torch.no_grad():
        classifier.weight.zero_()
        classifier.weight[0].copy_(direction * 1e5)
        if classifier.out_features > 1:
            classifier.weight[1].copy_(-direction * 1e5)
        if classifier.bias is not None:
            classifier.bias.copy_(
                torch.linspace(
                    0.0,
                    0.04,
                    classifier.out_features,
                    dtype=classifier.bias.dtype,
                )
            )


def _psnr(expected: np.ndarray, actual: np.ndarray, peak: float) -> float:
    error = expected.astype(np.float64) - actual.astype(np.float64)
    mse = float(np.mean(np.square(error)))
    return float("inf") if mse == 0 else 20.0 * np.log10(peak / np.sqrt(mse))


def _assert_raw_parity(
    case: NCNNRound17Case,
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
        elif case.task == "detect":
            matches = np.isclose(actual, expected, rtol=2e-3, atol=2e-2)
            assert float(matches.mean()) > 0.95
        else:
            raise AssertionError(f"Unhandled Round 17 task: {case.task}")


def _image(case: NCNNRound17Case) -> np.ndarray:
    if case.task == "restore":
        height, width = case.imgsz - 4, case.imgsz - 2
    else:
        height, width = case.imgsz + 8, case.imgsz + 16
    return np.random.default_rng(117).integers(
        0,
        256,
        size=(height, width, 3),
        dtype=np.uint8,
    )


def _assert_detect_predict_parity(native, actual) -> None:
    expected = native.boxes.data.cpu().numpy()
    converted = actual.boxes.data.cpu().numpy()
    assert converted.shape == expected.shape
    if expected.shape[0] == 0:
        return
    cost = np.square(converted[:, None, :4] - expected[None, :, :4]).sum(axis=-1)
    converted_indices, expected_indices = linear_sum_assignment(cost)
    converted = converted[converted_indices[np.argsort(expected_indices)]]
    box_match = np.isclose(
        converted[:, :4],
        expected[:, :4],
        rtol=2e-3,
        atol=1.0,
    ).all(axis=-1)
    score_match = np.isclose(
        converted[:, 4],
        expected[:, 4],
        rtol=2e-3,
        atol=2e-2,
    )
    class_match = converted[:, 5] == expected[:, 5]
    assert float(box_match.mean()) > 0.95
    assert float(score_match.mean()) > 0.95
    assert float(class_match.mean()) > 0.95


def _assert_predict_parity(case: NCNNRound17Case, model, backend) -> None:
    image = _image(case)
    conf = 0.1 if case.task == "point" else 0.0
    native = model.predict(image, imgsz=case.imgsz, conf=conf, max_det=100)
    actual = backend.predict(image, conf=conf, max_det=100)

    if case.task == "classify":
        expected_probs = native.probs.data.cpu()
        actual_probs = actual.probs.data.cpu()
        cosine = torch.nn.functional.cosine_similarity(
            expected_probs[None],
            actual_probs[None],
        )
        assert float(cosine) > 0.999
        assert int(expected_probs.argmax()) == int(actual_probs.argmax())
    elif case.task == "point":
        torch.testing.assert_close(
            actual.points.data.cpu(),
            native.points.data.cpu(),
            rtol=1e-3,
            atol=1.0,
        )
    elif case.task == "semantic":
        agreement = (
            native.semantic_mask.data.cpu() == actual.semantic_mask.data.cpu()
        ).float().mean()
        assert float(agreement) > 0.95
    elif case.task == "restore":
        expected_rgb = native.restored.array.astype(np.float64)
        actual_rgb = actual.restored.array.astype(np.float64)
        assert actual_rgb.shape == expected_rgb.shape
        assert _psnr(expected_rgb, actual_rgb, 255.0) > 40.0
    elif case.task == "depth":
        expected_depth = native.depth_map.data.cpu().numpy()
        actual_depth = actual.depth_map.data.cpu().numpy()
        peak = max(float(np.max(np.abs(expected_depth))), 1e-6)
        assert _psnr(expected_depth, actual_depth, peak) > 40.0
    elif case.task == "detect":
        _assert_detect_predict_parity(native, actual)
    else:
        raise AssertionError(f"Unhandled Round 17 task: {case.task}")


def _run_case(tmp_path, case: NCNNRound17Case) -> None:
    from libreyolo import LibreYOLO

    torch.manual_seed(17)
    model = _build_model(case)
    if case.task == "classify":
        first = torch.zeros(1, 3, case.imgsz, case.imgsz)
        second = torch.ones(1, 3, case.imgsz, case.imgsz)
    else:
        first = torch.rand(1, 3, case.imgsz, case.imgsz)
        second = 1.0 - first
    if case.task == "classify":
        _prepare_input_sensitive_classifier(model, first, second)
    expected_first = _native_outputs(model, first, case.imgsz)
    expected_second = _native_outputs(model, second, case.imgsz)

    artifact = model.export(
        format="ncnn",
        output_path=str(tmp_path / f"{case.family}_ncnn"),
        imgsz=case.imgsz,
        dynamic=False,
        half=False,
        simplify=False,
    )
    backend = LibreYOLO(artifact, device="cpu")
    actual_first = backend._run_inference(first.numpy())
    actual_second = backend._run_inference(second.numpy())
    _assert_raw_parity(case, expected_first, actual_first)
    _assert_raw_parity(case, expected_second, actual_second)

    expected_signal = max(
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        first_output.astype(np.float64)
                        - second_output.astype(np.float64)
                    )
                )
            )
        )
        for first_output, second_output in zip(expected_first, expected_second)
    )
    actual_signal = max(
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        first_output.astype(np.float64)
                        - second_output.astype(np.float64)
                    )
                )
            )
        )
        for first_output, second_output in zip(actual_first, actual_second)
    )
    parity_error = max(
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        expected.astype(np.float64) - actual.astype(np.float64)
                    )
                )
            )
        )
        for expected, actual in zip(expected_first, actual_first)
    )
    assert expected_signal > 1e-8
    assert actual_signal > 20.0 * max(parity_error, 1e-12)

    assert backend.model_family == case.family
    assert backend.task == case.task
    assert backend.imgsz == case.imgsz
    _assert_predict_parity(case, model, backend)
    del backend, model
    gc.collect()


@pytest.mark.parametrize(
    "case",
    ROUND17_CASES,
    ids=lambda case: case.family,
)
def test_ncnn_round17_raw_and_predict_parity(tmp_path, case):
    _run_case(tmp_path, case)
