"""Round 25 parity probes for CLIP-family whole-image embedding exports."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from PIL import Image

from .test_export_round23 import _assert_raw_parity, _outputs
from .test_tensorrt_round11 import _align_outputs

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.extended_backend,
]


@dataclass(frozen=True)
class Round25Case:
    family: str
    format: str
    imgsz: int


def _case(family: str, format: str, imgsz: int):
    marks = []
    if format == "executorch":
        marks.append(pytest.mark.executorch)
    elif format == "tensorrt":
        marks.extend((pytest.mark.tensorrt, pytest.mark.trt))
    elif format == "openvino":
        marks.append(pytest.mark.openvino)
    elif format == "torchscript":
        marks.append(pytest.mark.torchscript)
    case = Round25Case(family, format, imgsz)
    return pytest.param(case, marks=marks, id=f"{family}-embed-{format}")


ROUND25_CASES = tuple(
    _case(family, format, 32)
    for family in ("clip", "siglip2")
    for format in ("onnx", "torchscript", "openvino", "tensorrt", "executorch")
)


def _build_model(case: Round25Case, device: str, monkeypatch):
    if case.family == "clip":
        from libreyolo.models.clip import nn as clip_nn
        from libreyolo.models.clip.model import LibreCLIP

        config = clip_nn.CLIPConfig(
            embed_dim=16,
            image_size=case.imgsz,
            patch_size=16,
            vision_width=64,
            vision_layers=1,
            text_width=32,
            text_heads=2,
            text_layers=1,
        )
        monkeypatch.setitem(clip_nn.CLIP_CONFIGS, "round25", config)
        monkeypatch.setattr(
            LibreCLIP,
            "INPUT_SIZES",
            {**LibreCLIP.INPUT_SIZES, "round25": case.imgsz},
        )
        state = clip_nn.CLIPModel(config).state_dict()
        return LibreCLIP(
            state,
            size="round25",
            task="embed",
            device=device,
        )

    from libreyolo.models.siglip2 import nn as siglip2_nn
    from libreyolo.models.siglip2.model import LibreSigLIP2

    config = siglip2_nn.SigLIP2Config(
        vision_width=64,
        vision_layers=1,
        vision_heads=2,
        vision_intermediate=128,
        image_size=case.imgsz,
        patch_size=16,
        text_width=32,
        text_layers=1,
        text_heads=2,
        text_intermediate=64,
        vocab_size=100,
        max_position_embeddings=16,
        projection_size=64,
    )
    monkeypatch.setitem(siglip2_nn.SIGLIP2_CONFIGS, "round25", config)
    monkeypatch.setattr(
        LibreSigLIP2,
        "INPUT_SIZES",
        {**LibreSigLIP2.INPUT_SIZES, "round25": case.imgsz},
    )
    state = siglip2_nn.SigLIP2Model(config).state_dict()
    return LibreSigLIP2(
        state,
        size="round25",
        task="embed",
        device=device,
    )


def _native_outputs(case: Round25Case, model, tensor, device):
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
    array = np.random.default_rng(25).integers(
        0,
        256,
        size=(imgsz + 11, imgsz + 7, 3),
        dtype=np.uint8,
    )
    image = Image.fromarray(array)
    expected = model(image)
    actual = runtime(image)
    torch.testing.assert_close(
        actual.embeddings.data.cpu(),
        expected.embeddings.data.cpu(),
        rtol=2e-3,
        atol=5e-3,
    )
    torch.testing.assert_close(
        actual.embeddings.data.norm(dim=1).cpu(),
        torch.ones(1),
        rtol=0.0,
        atol=1e-5,
    )


@pytest.mark.parametrize("case", ROUND25_CASES)
def test_round25_export_predict_parity(tmp_path, monkeypatch, case):
    if case.format == "openvino":
        pytest.importorskip("openvino")
    elif case.format == "tensorrt":
        pytest.importorskip("tensorrt")
        if not torch.cuda.is_available():
            pytest.skip("TensorRT parity requires CUDA")
    elif case.format == "executorch" and importlib.util.find_spec("executorch") is None:
        pytest.skip("ExecuTorch is required")

    from libreyolo import LibreYOLO
    from libreyolo.export.support import SUPPORT, SupportEntry

    monkeypatch.setitem(
        SUPPORT,
        (case.family, "embed", case.format),
        SupportEntry("available", "Round 25 measured embedding probe."),
    )
    torch.manual_seed(25)
    device = torch.device("cuda" if case.format == "tensorrt" else "cpu")
    model = _build_model(case, str(device), monkeypatch)
    model.model.eval()
    first = torch.rand(
        1,
        3,
        case.imgsz,
        case.imgsz,
        generator=torch.Generator().manual_seed(25),
    )
    second = 1.0 - first
    expected_first = _native_outputs(case, model, first, device)
    expected_second = _native_outputs(case, model, second, device)

    suffix = {
        "onnx": ".onnx",
        "torchscript": ".torchscript",
        "openvino": "_openvino",
        "tensorrt": ".engine",
        "executorch": ".pte",
    }[case.format]
    artifact = model.export(
        case.format,
        output_path=str(tmp_path / f"{case.family}-embed{suffix}"),
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
    assert runtime.task == "embed"
    assert runtime.imgsz == case.imgsz
