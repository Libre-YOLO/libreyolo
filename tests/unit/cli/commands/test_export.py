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


class _LoadedSAM(_LoadedModel):
    FAMILY = "sam2"
    size = "tiny"
    INPUT_SIZES = {"tiny": 1024}

    def _get_input_size(self):
        return 1024


class _LoadedVLM(_LoadedModel):
    def __init__(self, output_path, captured, *, family, size, input_size):
        super().__init__(output_path, captured)
        self.FAMILY = family
        self.size = size
        self.INPUT_SIZES = {size: input_size}
        self._input_size = input_size

    def _get_input_size(self):
        return self._input_size


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


def test_export_cli_rejects_invalid_or_irrelevant_compute_units():
    for args, message in (
        (
            ["model=unused.pt", "format=coreml", "compute_units=magic", "--json"],
            "Invalid Core ML compute_units",
        ),
        (
            ["model=unused.pt", "format=onnx", "compute_units=cpu_only", "--json"],
            "applies only to CoreML",
        ),
    ):
        result = runner.invoke(_build_app(), args)
        assert result.exit_code == 2
        data = _parse_json_output(result.output)
        assert data["error"] == "config_unsupported"
        assert message in data["message"]


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
    assert captured["kwargs"]["compute_units"] == "cpu_only"
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


def test_export_cli_uses_supported_sibling_factory(
    monkeypatch,
    tmp_path,
):
    from libreyolo.cli.commands import export

    captured = {}
    sibling_factory = object()
    monkeypatch.setattr(
        export,
        "resolve_export_sibling_factory",
        lambda model: sibling_factory if model == "sam2-tiny" else None,
    )

    def fail_generic_resolve(_out, _model):
        raise AssertionError("sibling aliases must bypass the native resolver")

    monkeypatch.setattr(export, "resolve_model_or_exit", fail_generic_resolve)

    def fake_load(
        _out,
        *,
        model,
        model_path,
        device,
        model_factory,
    ):
        captured.update(
            {
                "model": model,
                "model_path": model_path,
                "device": device,
                "model_factory": model_factory,
            }
        )
        return _LoadedSAM(tmp_path / "sam.mlpackage", captured)

    monkeypatch.setattr(export, "load_model_or_exit", fake_load)

    result = runner.invoke(
        _build_app(),
        [
            "model=sam2-tiny",
            "format=coreml",
            "device=cpu",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["model"] == "sam2-tiny"
    assert captured["model_path"] == "sam2-tiny"
    assert captured["device"] == "cpu"
    assert captured["model_factory"] is sibling_factory
    assert captured["kwargs"]["dynamic"] is False


@pytest.mark.parametrize(
    ("alias", "family", "size", "input_size"),
    [
        ("smolvlm2-500m", "smolvlm2", "500m", 512),
        ("florence-2-base", "florence2", "base", 768),
        ("qwen3-vl-2b", "qwen3vl", "2b", 448),
    ],
)
def test_export_cli_vlm_factory_receives_only_specialized_kwargs(
    monkeypatch,
    tmp_path,
    alias,
    family,
    size,
    input_size,
):
    from libreyolo.cli.commands import export

    captured = {}

    def fail_generic_resolve(_out, _model):
        raise AssertionError("supported VLM aliases must bypass the native resolver")

    monkeypatch.setattr(export, "resolve_model_or_exit", fail_generic_resolve)

    def fake_load(
        _out,
        *,
        model,
        model_path,
        device,
        model_factory,
    ):
        captured.update(
            {
                "model": model,
                "model_path": model_path,
                "device": device,
                "model_factory": model_factory,
            }
        )
        return _LoadedVLM(
            tmp_path / f"{family}.coremlbundle",
            captured,
            family=family,
            size=size,
            input_size=input_size,
        )

    monkeypatch.setattr(export, "load_model_or_exit", fake_load)

    result = runner.invoke(
        _build_app(),
        [
            f"model={alias}",
            "format=coreml",
            "device=cpu",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["model_factory"].__name__ == "LibreVLM"
    assert captured["model_path"] == alias
    assert captured["device"] == "cpu"
    assert captured["kwargs"] == {}


def test_export_cli_vlm_forces_auto_to_cpu_and_rejects_accelerator_source(
    monkeypatch,
    tmp_path,
):
    from libreyolo.cli.commands import export

    captured = {}

    def fake_load(
        _out,
        *,
        model,
        model_path,
        device,
        model_factory,
    ):
        del model, model_path, model_factory
        captured["device"] = device
        return _LoadedVLM(
            tmp_path / "florence.coremlvlm",
            captured,
            family="florence2",
            size="base",
            input_size=768,
        )

    monkeypatch.setattr(export, "load_model_or_exit", fake_load)
    result = runner.invoke(
        _build_app(),
        ["model=florence-2", "format=coreml", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert captured["device"] == "cpu"

    captured.clear()
    result = runner.invoke(
        _build_app(),
        [
            "model=florence-2",
            "format=coreml",
            "device=cuda",
            "--json",
        ],
    )
    assert result.exit_code == 2, result.output
    assert _parse_json_output(result.output)["error"] == "config_unsupported"
    assert captured == {}


@pytest.mark.parametrize("option", ["half=true", "batch=2", "simplify=false"])
def test_export_cli_vlm_rejects_generic_graph_options(
    monkeypatch,
    tmp_path,
    option,
):
    from libreyolo.cli.commands import export

    captured = {}

    def fake_load(
        _out,
        *,
        model,
        model_path,
        device,
        model_factory,
    ):
        del model, model_path, device, model_factory
        return _LoadedVLM(
            tmp_path / "smol.coremlvlm",
            captured,
            family="smolvlm2",
            size="500m",
            input_size=512,
        )

    monkeypatch.setattr(export, "load_model_or_exit", fake_load)

    result = runner.invoke(
        _build_app(),
        [
            "model=smolvlm2-500m",
            "format=coreml",
            option,
            "--json",
        ],
    )

    assert result.exit_code == 2, result.output
    data = _parse_json_output(result.output)
    assert data["error"] == "config_unsupported"
    assert option.split("=", 1)[0] in data["message"]
    assert "format" not in captured


@pytest.mark.parametrize(
    ("alias", "factory_name"),
    [
        ("sam2-tiny", "LibreSAM"),
        ("SAM2-TINY", "LibreSAM"),
        ("b", "LibreSAM"),
        ("picosam3", "LibreSAM"),
        ("owlv2-b16", "LibreOpenVocab"),
        ("grounding_dino-t", "LibreOpenVocab"),
        ("smolvlm2-500m", "LibreVLM"),
        ("florence-2", "LibreVLM"),
        ("florence-2-base", "LibreVLM"),
        ("florence2", "LibreVLM"),
        ("qwen3-vl-2b", "LibreVLM"),
    ],
)
def test_export_sibling_factory_alias_resolution(alias, factory_name):
    from libreyolo.cli.command_utils import resolve_export_sibling_factory

    factory = resolve_export_sibling_factory(alias)
    assert factory is not None
    assert factory.__name__ == factory_name


@pytest.mark.parametrize(
    "reference",
    [
        "yolo9-s",
        "sample-model",
        "owl-house",
        "sam2-tiny.pt",
        "smolvlm2",
        "smolvlm2-2.2b",
        "florence-2-large",
        "unknown-model",
    ],
)
def test_export_sibling_factory_does_not_capture_near_collisions(reference):
    from libreyolo.cli.command_utils import resolve_export_sibling_factory

    assert resolve_export_sibling_factory(reference) is None


def test_export_sibling_factory_preserves_filesystem_paths(tmp_path):
    from libreyolo.cli.command_utils import resolve_export_sibling_factory

    path = tmp_path / "sam3"
    path.mkdir()
    assert resolve_export_sibling_factory(str(path)) is None
