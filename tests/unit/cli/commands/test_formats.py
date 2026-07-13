"""Tests for family-aware export format reporting."""

from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands import special
from libreyolo.cli.parsing import KeyValueCommand


pytestmark = pytest.mark.unit
runner = CliRunner()


def _build_app() -> typer.Typer:
    app = typer.Typer(add_completion=False)
    app.command("formats", cls=KeyValueCommand)(special.formats_cmd)
    return app


def _format(output: dict, name: str) -> dict:
    return next(item for item in output["formats"] if item["name"] == name)


def test_formats_reports_family_specific_capabilities():
    result = runner.invoke(
        _build_app(), ["family=yolox", "task=detect", "--json", "--quiet"]
    )

    assert result.exit_code == 0, result.output
    output = json.loads(result.stdout)
    assert _format(output, "onnx")["int8"] is False
    assert _format(output, "onnx")["dynamic"] is True
    assert _format(output, "torchscript")["dynamic"] is False
    assert _format(output, "ncnn")["fp16"] is False


def test_formats_reports_yolo9_onnx_int8_support():
    result = runner.invoke(
        _build_app(), ["family=yolo9", "task=detect", "--json", "--quiet"]
    )

    assert result.exit_code == 0, result.output
    output = json.loads(result.stdout)
    assert _format(output, "onnx")["int8"] is True
