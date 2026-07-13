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


@pytest.mark.parametrize("format_name", ["torchscript", "ncnn", "coreml"])
def test_export_cli_reports_effective_static_dynamic(
    monkeypatch, tmp_path, format_name
):
    from libreyolo.cli.commands import export

    captured = {}
    monkeypatch.setattr(export, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        export,
        "load_model_or_exit",
        lambda out, model, model_path, device: _LoadedModel(
            tmp_path / f"model_{format_name}", captured
        ),
    )

    result = runner.invoke(
        _build_app(),
        [
            "model=dummy.pt",
            f"format={format_name}",
            "dynamic=true",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = _parse_json_output(result.output)
    assert captured["kwargs"]["dynamic"] is False
    assert data["dynamic"] is False


def test_export_cli_keeps_tflite_dynamic_for_explicit_rejection(monkeypatch, tmp_path):
    from libreyolo.cli.commands import export

    captured = {}
    monkeypatch.setattr(export, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        export,
        "load_model_or_exit",
        lambda out, model, model_path, device: _LoadedModel(
            tmp_path / "model.tflite", captured
        ),
    )

    result = runner.invoke(
        _build_app(),
        ["model=dummy.pt", "format=tflite", "dynamic=true", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["dynamic"] is True
    assert _parse_json_output(result.output)["dynamic"] is True


def test_export_cli_custom_classifier_reports_int8_before_cli_only_kwargs(monkeypatch):
    from libreyolo.cli.commands import export
    from libreyolo.models.clip.model import LibreCLIP

    model = object.__new__(LibreCLIP)
    model.size = "b32"
    model.task = "classify"
    monkeypatch.setattr(export, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        export,
        "load_model_or_exit",
        lambda out, model, model_path, device: model_instance,
    )
    model_instance = model

    result = runner.invoke(
        _build_app(),
        [
            "model=dummy.pt",
            "format=onnx",
            "int8=true",
            "fraction=0.5",
            "--json",
        ],
    )

    assert result.exit_code == 5
    data = _parse_json_output(result.output)
    assert data["error"] == "format_precision_unsupported"
    assert "int8=True" in data["message"]
    assert "Unsupported LibreCLIP export arguments" not in data["message"]
