from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit
runner = CliRunner()


def _build_app() -> typer.Typer:
    from libreyolo.cli.commands import export

    app = typer.Typer(add_completion=False)
    app.command("export", cls=KeyValueCommand)(export.export_cmd)
    return app


def _parse_json_output(output: str) -> dict:
    for line in output.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"No JSON found in output:\n{output}")


class _LoadedModel:
    FAMILY = "yolo9"
    size = "t"
    INPUT_SIZES = {"t": 128}

    def __init__(self, output_path, captured):
        self.output_path = output_path
        self.captured = captured

    def _get_input_size(self):
        return 128

    def export(self, format, **kwargs):
        self.captured["format"] = format
        self.captured["kwargs"] = kwargs
        self.output_path.mkdir()
        return str(self.output_path)


class _LoadedPPOCR(_LoadedModel):
    FAMILY = "ppocr"
    size = "t"
    INPUT_SIZES = {"t": 960}

    def _get_input_size(self):
        return 960


class _LoadedFace(_LoadedModel):
    FAMILY = "facerec"
    size = "l"
    INPUT_SIZES = {"l": 112}
    cfg = type("Cfg", (), {"layout": "NHWC"})()

    def _get_input_size(self):
        return 112


def test_export_cli_allows_coreml_embedded_nms(monkeypatch, tmp_path):
    from libreyolo.cli.commands import export

    captured = {}
    monkeypatch.setattr(export, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        export,
        "load_model_or_exit",
        lambda out, model, model_path, device: _LoadedModel(
            tmp_path / "model.mlpackage", captured
        ),
    )

    result = runner.invoke(
        _build_app(),
        [
            "model=dummy.pt",
            "format=coreml",
            "nms=true",
            "conf=0.2",
            "iou=0.4",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = _parse_json_output(result.output)
    assert data["format"] == "coreml"
    assert captured["format"] == "coreml"
    assert captured["kwargs"]["nms"] is True
    assert captured["kwargs"]["conf"] == 0.2
    assert captured["kwargs"]["iou"] == 0.4
    assert "max_det" not in captured["kwargs"]


def test_export_cli_rejects_coreml_max_det(monkeypatch):
    from libreyolo.cli.commands import export

    monkeypatch.setattr(export, "resolve_model_or_exit", lambda out, model: model)

    result = runner.invoke(
        _build_app(),
        [
            "model=dummy.pt",
            "format=coreml",
            "nms=true",
            "max_det=12",
            "--json",
        ],
    )

    assert result.exit_code == 2
    data = _parse_json_output(result.output)
    assert data["error"] == "config_unsupported"
    assert "max_det is only supported for ONNX" in data["message"]


def test_export_cli_forwards_ppocr_bounds_and_reports_effective_dynamic(
    monkeypatch,
    tmp_path,
):
    from libreyolo.cli.commands import export

    captured = {}
    monkeypatch.setattr(export, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        export,
        "load_model_or_exit",
        lambda out, model, model_path, device: _LoadedPPOCR(
            tmp_path / "ocr.mlpackage",
            captured,
        ),
    )

    result = runner.invoke(
        _build_app(),
        [
            "model=LibrePPOCRt-ocr.pt",
            "format=coreml",
            "rec_max_width=2048",
            "rec_batch_max=4",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = _parse_json_output(result.output)
    assert captured["kwargs"]["rec_max_width"] == 2048
    assert captured["kwargs"]["rec_batch_max"] == 4
    assert captured["kwargs"]["dynamic"] is True
    assert data["dynamic"] is True


def test_export_cli_does_not_forward_ocr_options_to_face(
    monkeypatch,
    tmp_path,
):
    from libreyolo.cli.commands import export

    captured = {}
    monkeypatch.setattr(export, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        export,
        "load_model_or_exit",
        lambda out, model, model_path, device: _LoadedFace(
            tmp_path / "face.mlpackage",
            captured,
        ),
    )

    result = runner.invoke(
        _build_app(),
        [
            "model=librefacerec-l.onnx",
            "format=coreml",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = _parse_json_output(result.output)
    assert "rec_batch_max" not in captured["kwargs"]
    assert "rec_max_width" not in captured["kwargs"]
    assert "simplify" not in captured["kwargs"]
    assert "opset" not in captured["kwargs"]
    assert "verbose" not in captured["kwargs"]
    assert data["input_shape"] == [1, 112, 112, 3]
