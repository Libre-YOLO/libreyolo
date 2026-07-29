"""Offline contract tests for PicoSAM3's Core ML ROI component."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.sam]


def _bare_picosam3():
    from libreyolo.models.picosam3.model import LibrePicoSAM3
    from libreyolo.models.picosam3.nn import PicoSAM3Network

    model = object.__new__(LibrePicoSAM3)
    model.model = PicoSAM3Network().eval()
    model.device = torch.device("cpu")
    model._model_dtype = torch.float32
    model.names = {0: "object"}
    model.nb_classes = 1
    model.size = "pico"
    model.task = "segment"
    model.input_size = 96
    return model


def test_adapter_matches_native_roi_preprocessing_exactly():
    from libreyolo.export.coreml_picosam3 import PicoSAM3CoreMLAdapter
    from libreyolo.models.picosam3.preprocess import (
        padded_square_roi,
        preprocess_roi,
    )

    yy, xx = np.mgrid[:37, :53]
    image = Image.fromarray(
        np.stack(
            (
                (7 * xx + 3 * yy) % 256,
                (11 * xx + 5 * yy) % 256,
                (13 * xx + 17 * yy) % 256,
            ),
            axis=-1,
        ).astype(np.uint8),
        mode="RGB",
    )
    roi = padded_square_roi([8, 4, 41, 31], *image.size)
    resized = image.crop(roi).resize((96, 96), Image.Resampling.BILINEAR)
    canonical = (
        torch.from_numpy(np.asarray(resized, dtype=np.float32).copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .div(255.0)
    )

    actual = PicoSAM3CoreMLAdapter(torch.nn.Identity()).eval()(canonical)[0]
    expected = preprocess_roi(image, roi)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_component_contract_is_fixed_and_semantic():
    from libreyolo.export.coreml_picosam3 import (
        picosam3_coreml_component_metadata,
        picosam3_coreml_input_contract,
        picosam3_coreml_output_contract,
        picosam3_coreml_validation_contract,
        validate_picosam3_coreml_profile,
    )

    validate_picosam3_coreml_profile(size="pico", canvas_hw=(96, 96))
    assert picosam3_coreml_input_contract() == {
        "name": "roi_image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": "native",
        "interpolation": "bilinear",
        "resize_backend": "pillow",
        "pad_value": 0,
    }
    assert picosam3_coreml_validation_contract() == {
        "color": "rgb",
        "range": "imagenet",
    }
    assert picosam3_coreml_output_contract() == [
        {
            "name": "mask_logits",
            "role": "mask_logits",
            "encoding": "raw_logits",
            "rank": 4,
        }
    ]
    assert picosam3_coreml_component_metadata()["artifact_scope"] == "roi_component"

    with pytest.raises(NotImplementedError, match="96x96"):
        validate_picosam3_coreml_profile(size="pico", canvas_hw=(128, 128))


def test_real_component_two_probe_trace_is_exact_and_sensitive():
    from libreyolo.export.coreml_picosam3 import PicoSAM3CoreMLAdapter
    from libreyolo.models.picosam3.nn import PicoSAM3Network

    torch.manual_seed(20260729)
    graph = PicoSAM3CoreMLAdapter(PicoSAM3Network().eval()).eval()
    first = torch.linspace(0.0, 1.0, 3 * 96 * 96).reshape(1, 3, 96, 96)
    second = 1.0 - first
    traced = torch.jit.trace(
        graph,
        first,
        check_trace=True,
        check_inputs=[(second,)],
    )

    with torch.no_grad():
        expected_first = graph(first)
        expected_second = graph(second)
        actual_first = traced(first)
        actual_second = traced(second)
    torch.testing.assert_close(actual_first, expected_first, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_second, expected_second, rtol=0.0, atol=0.0)
    assert float((expected_second - expected_first).abs().max()) > 1e-6


def test_public_export_routes_coreml_to_shared_exporter(monkeypatch, tmp_path):
    from libreyolo.export import BaseExporter

    model = _bare_picosam3()
    exporter = MagicMock(return_value=str(tmp_path / "pico.mlpackage"))
    monkeypatch.setattr(
        BaseExporter,
        "create",
        classmethod(lambda _cls, export_format, wrapper: exporter),
    )

    result = model.export(
        "coreml",
        output=tmp_path / "pico.mlpackage",
        compute_units="cpu_only",
    )

    assert result == str(tmp_path / "pico.mlpackage")
    exporter.assert_called_once_with(
        output_path=str(Path(tmp_path / "pico.mlpackage")),
        imgsz=96,
        dynamic=False,
        compute_units="cpu_only",
    )


def test_public_export_rejects_dynamic_coreml(tmp_path):
    model = _bare_picosam3()

    with pytest.raises(NotImplementedError, match="fixed input shape"):
        model.export(
            "coreml",
            output=tmp_path / "pico.mlpackage",
            dynamic=True,
        )


def test_public_export_rejects_wrong_coreml_canvas(tmp_path):
    model = _bare_picosam3()

    with pytest.raises(NotImplementedError, match="96x96"):
        model.export(
            "coreml",
            output=tmp_path / "pico.mlpackage",
            imgsz=128,
        )


def test_public_export_rejects_two_destination_aliases(tmp_path):
    model = _bare_picosam3()

    with pytest.raises(ValueError, match="only one"):
        model.export(
            "coreml",
            output=tmp_path / "one.mlpackage",
            output_path=tmp_path / "two.mlpackage",
        )
