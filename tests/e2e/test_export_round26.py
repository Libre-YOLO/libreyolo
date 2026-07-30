"""Round 26 parity probes for remaining embedding, semantic, and edge runtimes."""

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
    pytest.mark.slow,
]


@dataclass(frozen=True)
class Round26Case:
    family: str
    task: str
    format: str
    imgsz: int


def _case(family: str, task: str, format: str, imgsz: int):
    hold_reasons = {
        ("clip", "embed", "ncnn"): "PNNX leaves unsupported Expression nodes",
        ("clip", "embed", "tflite"): "LiteRT rejects a rank-5 transpose permutation",
        ("siglip2", "embed", "ncnn"): "PNNX leaves unsupported Expression nodes",
        ("dinov2", "embed", "ncnn"): "PNNX cannot lower batch-axis broadcasts",
        ("dinov2", "semantic", "ncnn"): "PNNX cannot lower batch-axis broadcasts",
        (
            "dinov2",
            "semantic",
            "tflite",
        ): "cubic Resize retains dynamic C/H/W",
        ("teed", "edge", "ncnn"): "PNNX leaves an unsupported Tensor.index node",
        ("dexined", "edge", "ncnn"): "PNNX leaves an unsupported Tensor.index node",
    }
    marks = []
    if (family, task, format) in hold_reasons:
        marks.append(
            pytest.mark.xfail(
                strict=True,
                reason=hold_reasons[(family, task, format)],
            )
        )
    return pytest.param(
        Round26Case(family, task, format, imgsz),
        marks=marks,
        id=f"{family}-{task}-{format}",
    )


ROUND26_CASES = (
    _case("clip", "embed", "ncnn", 32),
    _case("clip", "embed", "tflite", 32),
    _case("siglip2", "embed", "ncnn", 32),
    _case("siglip2", "embed", "tflite", 32),
    _case("dinov2", "embed", "ncnn", 56),
    _case("dinov2", "embed", "tflite", 56),
    _case("dinov2", "semantic", "ncnn", 56),
    _case("dinov2", "semantic", "tflite", 56),
    _case("teed", "edge", "ncnn", 64),
    _case("dexined", "edge", "ncnn", 64),
)


def _build_model(case: Round26Case, monkeypatch):
    if case.family in {"clip", "siglip2"}:
        return _build_embed_model(
            Round25Case(case.family, case.format, case.imgsz),
            "cpu",
            monkeypatch,
        )

    from libreyolo import LibreDexiNed, LibreDINOv2, LibreTEED

    if case.family == "dinov2":
        return LibreDINOv2(
            None,
            size="n",
            task=case.task,
            nb_classes=3,
            device="cpu",
        )
    if case.family == "teed":
        return LibreTEED(None, size="t", device="cpu")
    return LibreDexiNed(None, size="b", device="cpu")


def _native_outputs(case: Round26Case, model, tensor):
    from libreyolo.export.exporter import BaseExporter

    with (
        BaseExporter.create(case.format, model)._model_context(
            torch.device("cpu"),
            False,
            False,
            1,
            (case.imgsz, case.imgsz),
        ) as (wrapped, _),
        torch.no_grad(),
    ):
        outputs = _outputs(wrapped(tensor))
    return tuple(output.detach().float().cpu().numpy() for output in outputs)


def _assert_public_parity(case: Round26Case, model, runtime):
    array = np.random.default_rng(26).integers(
        0,
        256,
        size=(case.imgsz + 11, case.imgsz + 7, 3),
        dtype=np.uint8,
    )
    image = Image.fromarray(array)
    expected = model(image, imgsz=case.imgsz)
    actual = runtime(image)
    if case.task == "embed":
        torch.testing.assert_close(
            actual.embeddings.data.cpu(),
            expected.embeddings.data.cpu(),
            rtol=2e-3,
            atol=5e-3,
        )
        return
    if case.task == "semantic":
        expected_mask = expected.semantic_mask.data.cpu().numpy()
        actual_mask = actual.semantic_mask.data.cpu().numpy()
        assert actual_mask.shape == expected_mask.shape
        assert float(np.mean(actual_mask == expected_mask)) > 0.95
        return

    expected_edge = expected.edge_map.data.cpu().numpy().astype(np.float64)
    actual_edge = actual.edge_map.data.cpu().numpy().astype(np.float64)
    assert expected_edge.shape == actual_edge.shape
    assert float(np.mean(np.abs(expected_edge - actual_edge))) < 2e-3


@pytest.mark.parametrize("case", ROUND26_CASES)
def test_round26_export_predict_parity(tmp_path, monkeypatch, case):
    if case.format == "ncnn":
        pytest.importorskip("ncnn")
    elif (
        importlib.util.find_spec("onnx2tf") is None
        or importlib.util.find_spec("ai_edge_litert") is None
    ):
        pytest.skip("onnx2tf and ai-edge-litert are required")

    from libreyolo import LibreYOLO
    from libreyolo.export.support import SUPPORT, SupportEntry
    from libreyolo.models.base.model import BaseModel

    monkeypatch.setitem(
        SUPPORT,
        (case.family, case.task, case.format),
        SupportEntry("experimental", "Round 26 measured edge-runtime probe."),
    )
    torch.manual_seed(26)
    model = _build_model(case, monkeypatch)
    model.model.eval()
    first = torch.rand(
        1,
        3,
        case.imgsz,
        case.imgsz,
        generator=torch.Generator().manual_seed(26),
    )
    second = 1.0 - first
    expected_first = _native_outputs(case, model, first)
    expected_second = _native_outputs(case, model, second)

    suffix = "_ncnn" if case.format == "ncnn" else ".tflite"
    artifact = BaseModel.export(
        model,
        format=case.format,
        output_path=str(
            tmp_path / f"{case.family}-{case.task}{suffix}"
        ),
        imgsz=case.imgsz,
        batch=1,
        dynamic=False,
        half=False,
        simplify=False,
        opset=17,
    )
    runtime = LibreYOLO(artifact, device="cpu")
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
    _assert_public_parity(case, model, runtime)
    assert runtime.model_family == case.family
    assert runtime.task == case.task
    assert runtime.imgsz == case.imgsz
