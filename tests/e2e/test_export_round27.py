"""Round 27 parity probes for frozen CLIP-family classifiers."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from PIL import Image

from .test_export_round23 import _assert_raw_parity, _outputs
from .test_export_round25 import Round25Case
from .test_export_round25 import _build_model as _build_embed_model
from .test_tensorrt_round11 import _align_outputs

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
]


@dataclass(frozen=True)
class Round27Case:
    family: str
    format: str
    imgsz: int


def _case(family: str, format: str):
    marks = []
    if family == "clip" and format == "tflite":
        marks.append(
            pytest.mark.xfail(
                strict=True,
                reason="LiteRT rejects a rank-5 transpose permutation",
            )
        )
    if format == "executorch":
        marks.append(pytest.mark.executorch)
    elif format == "tensorrt":
        marks.extend((pytest.mark.tensorrt, pytest.mark.trt))
    elif format == "openvino":
        marks.append(pytest.mark.openvino)
    elif format == "torchscript":
        marks.append(pytest.mark.torchscript)
    return pytest.param(
        Round27Case(family, format, 32),
        marks=marks,
        id=f"{family}-classify-{format}",
    )


ROUND27_CASES = tuple(
    _case(family, format)
    for family in ("clip", "siglip2")
    for format in (
        "torchscript",
        "openvino",
        "tensorrt",
        "executorch",
        "tflite",
    )
)


def _build_model(case: Round27Case, device: str, monkeypatch):
    model = _build_embed_model(
        Round25Case(case.family, case.format, case.imgsz),
        device,
        monkeypatch,
    )
    model.task = "classify"
    dimensions = (
        model.model.config.embed_dim
        if case.family == "clip"
        else model.model.config.projection_size
    )
    generator = torch.Generator().manual_seed(270)
    model._text_embeds = torch.nn.functional.normalize(
        torch.randn(3, dimensions, generator=generator),
        dim=1,
    ).to(device)
    model.nb_classes = 3
    model.names = {0: "red object", 1: "green object", 2: "blue object"}
    return model


def _native_outputs(case: Round27Case, model, tensor, device):
    from libreyolo.export.exporter import BaseExporter

    with (
        BaseExporter.create(case.format, model)._model_context(
            device,
            False,
            False,
            1,
            (case.imgsz, case.imgsz),
        ) as (wrapped, _),
        torch.no_grad(),
    ):
        outputs = _outputs(wrapped(tensor.to(device)))
    return tuple(output.detach().float().cpu().numpy() for output in outputs)


def _assert_public_parity(model, runtime, imgsz: int):
    array = np.random.default_rng(27).integers(
        0,
        256,
        size=(imgsz + 11, imgsz + 7, 3),
        dtype=np.uint8,
    )
    image = Image.fromarray(array)
    expected = model(image)
    actual = runtime(image)
    torch.testing.assert_close(
        actual.probs.data.cpu(),
        expected.probs.data.cpu(),
        rtol=2e-3,
        atol=5e-3,
    )
    assert actual.probs.top1 == expected.probs.top1


@pytest.mark.parametrize("case", ROUND27_CASES)
def test_round27_export_predict_parity(tmp_path, monkeypatch, case):
    if case.format == "openvino":
        pytest.importorskip("openvino")
    elif case.format == "tensorrt":
        pytest.importorskip("tensorrt")
        if not torch.cuda.is_available():
            pytest.skip("TensorRT parity requires CUDA")
    elif case.format == "executorch":
        if importlib.util.find_spec("executorch") is None:
            pytest.skip("ExecuTorch is required")
    elif (
        importlib.util.find_spec("onnx2tf") is None
        or importlib.util.find_spec("ai_edge_litert") is None
    ):
        pytest.skip("onnx2tf and ai-edge-litert are required")

    from libreyolo import LibreYOLO
    from libreyolo.export.support import SUPPORT, SupportEntry

    monkeypatch.setitem(
        SUPPORT,
        (case.family, "classify", case.format),
        SupportEntry("experimental", "Round 27 frozen-class classifier probe."),
    )
    torch.manual_seed(27)
    device = torch.device("cuda" if case.format == "tensorrt" else "cpu")
    model = _build_model(case, str(device), monkeypatch)
    model.model.eval()
    first = torch.rand(
        1,
        3,
        case.imgsz,
        case.imgsz,
        generator=torch.Generator().manual_seed(27),
    )
    second = 1.0 - first
    expected_first = _native_outputs(case, model, first, device)
    expected_second = _native_outputs(case, model, second, device)

    suffix = {
        "torchscript": ".torchscript",
        "openvino": "_openvino",
        "tensorrt": ".engine",
        "executorch": ".pte",
        "tflite": ".tflite",
    }[case.format]
    if case.family == "clip" and case.format == "tflite":
        from libreyolo.models.base.model import BaseModel

        def export(format, **kwargs):
            return BaseModel.export(model, format=format, **kwargs)

    else:
        export = model.export
    artifact = export(
        case.format,
        output_path=str(tmp_path / f"{case.family}-classify{suffix}"),
        imgsz=case.imgsz,
        batch=1,
        dynamic=False,
        half=False,
        simplify=False,
    )
    runtime = LibreYOLO(artifact, device=str(device))
    actual_first = _align_outputs(
        expected_first,
        runtime._run_inference(first.numpy()),
    )
    actual_second = _align_outputs(
        expected_second,
        runtime._run_inference(second.numpy()),
    )
    _assert_raw_parity(
        expected_first,
        expected_second,
        actual_first,
        actual_second,
    )
    _assert_public_parity(model, runtime, case.imgsz)
    assert runtime.model_family == case.family
    assert runtime.task == "classify"
    assert runtime.names == model.names
    assert runtime.imgsz == case.imgsz
