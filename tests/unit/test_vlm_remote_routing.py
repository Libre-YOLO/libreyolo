"""Offline routing tests for immutable remote LibreVLM artifacts."""

from __future__ import annotations

import json
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import typer
from PIL import Image
from typer.testing import CliRunner

from libreyolo.cli.commands import export, predict, quantize, special, train, val
from libreyolo.cli.parsing import KeyValueCommand
from libreyolo.utils.results import Boxes, Results

pytestmark = pytest.mark.unit

_REVISION = "0123456789abcdef0123456789abcdef01234567"
_REMOTE = f"hf+vlm://libreyolo/qwen3-vl-detect@{_REVISION}"
_RUNNER = CliRunner()


def _make_app(name: str, command) -> typer.Typer:
    app = typer.Typer(add_completion=False, no_args_is_help=True)
    app.command(name, cls=KeyValueCommand)(command)
    # Keep Typer in command-group mode so invocations consistently include
    # the command name even when this focused test registers one target.
    app.command("version", cls=KeyValueCommand)(special.version_cmd)
    return app


def test_remote_reference_inspection_is_typed_and_has_no_transport(monkeypatch):
    from libreyolo.models import vlm
    from libreyolo.models.vlm import hub

    monkeypatch.setattr(
        hub,
        "download_vlm_artifact",
        lambda *_args, **_kwargs: pytest.fail("inspection attempted a download"),
    )

    reference = vlm.inspect_vlm_reference(_REMOTE)

    assert reference is not None
    assert reference.remote is True
    assert reference.checkpoint is True
    assert reference.family is None
    assert reference.size is None
    assert reference.trainable is False
    assert reference.hub == hub.VLMHubRef(
        repo_id="libreyolo/qwen3-vl-detect",
        revision=_REVISION,
    )


def test_vlm_reference_public_type_hints_are_resolvable():
    from libreyolo.models.vlm import VLMReference
    from libreyolo.models.vlm.hub import VLMHubRef

    assert typing.get_type_hints(VLMReference)["hub"] == VLMHubRef | None


@pytest.mark.parametrize(
    "reference",
    [
        "hf+vlm://libreyolo/qwen3-vl-detect",
        "hf+vlm://libreyolo/qwen3-vl-detect@main",
        "hf+vlm://libreyolo/qwen3-vl-detect@01234567",
        f"hf+vlm://libreyolo/qwen3-vl-detect@{_REVISION.upper()}",
        f"hf+vlm://libreyolo/qwen3-vl-detect@{_REVISION}/adapter_model.safetensors",
    ],
)
def test_malformed_remote_vlm_references_fail_closed(reference):
    from libreyolo.models.vlm import inspect_vlm_reference

    with pytest.raises(ValueError):
        inspect_vlm_reference(reference)


def test_generic_hub_reference_spellings_remain_outside_vlm_routing():
    from libreyolo.models.vlm import inspect_vlm_reference

    assert inspect_vlm_reference(f"hf://libreyolo/model@{_REVISION}") is None
    assert inspect_vlm_reference("libreyolo/model") is None


def test_librevlm_remote_downloads_to_isolated_directory_and_loads(monkeypatch):
    from libreyolo.models import vlm
    from libreyolo.models.vlm import artifact as artifact_module
    from libreyolo.models.vlm import hub

    calls = {}
    events = []
    expected_base = {"schema": "test-base-identity"}
    expected_manifest = {"schema": "test-artifact"}

    def fake_download(source, output_dir, *, token=None, local_files_only=False):
        output = Path(output_dir)
        assert output.parent.is_dir()
        assert not output.exists()
        output.mkdir()
        events.append("download")
        calls["download"] = (source, output, token, local_files_only)
        return SimpleNamespace(
            root=output,
            base_snapshot=expected_base,
            aggregate_sha256="a" * 64,
            files=("adapter_model.safetensors",),
            manifest=expected_manifest,
        )

    def fake_ensure(info, *, token=None, local_files_only=False):
        assert info.root == calls["download"][1]
        base_root = info.root.parent / "base"
        base_root.mkdir()
        events.append("ensure")
        calls["ensure"] = (base_root, token, local_files_only)
        return SimpleNamespace(root=base_root, identity=expected_base)

    def fake_validate(root, expected):
        assert root == calls["ensure"][0]
        assert expected is expected_base
        events.append("validate")
        return expected

    def fake_validate_artifact(root):
        assert root == calls["download"][1]
        events.append("validate_artifact")
        return SimpleNamespace(
            root=root,
            aggregate_sha256="a" * 64,
            files=("adapter_model.safetensors",),
            manifest=expected_manifest,
        )

    class Loaded:
        pass

    def fake_load(path, **kwargs):
        root = Path(path)
        assert root.is_dir()
        events.append("load")
        calls["load"] = (root, kwargs)
        return Loaded()

    monkeypatch.setattr(hub, "download_vlm_artifact", fake_download)
    monkeypatch.setattr(hub, "ensure_vlm_base_snapshot", fake_ensure)
    monkeypatch.setattr(
        artifact_module,
        "validate_vlm_base_snapshot",
        fake_validate,
    )
    monkeypatch.setattr(
        artifact_module,
        "validate_vlm_artifact",
        fake_validate_artifact,
    )
    monkeypatch.setattr(vlm, "_load_checkpoint", fake_load)

    loaded = vlm.LibreVLM(
        _REMOTE,
        device="cpu",
        names=["forklift"],
        token="private-token",
        local_files_only=True,
    )

    source, artifact_root, token, local_only = calls["download"]
    assert source == _REMOTE
    assert token == "private-token"
    assert local_only is True
    assert calls["ensure"][1:] == ("private-token", True)
    assert calls["load"] == (
        artifact_root,
        {"device": "cpu", "names": ["forklift"]},
    )
    assert events == [
        "download",
        "ensure",
        "validate",
        "load",
        "validate_artifact",
        "validate",
    ]
    assert loaded._vlm_remote_source == _REMOTE
    assert artifact_root.is_dir()

    loaded._vlm_remote_artifact.cleanup()
    assert not artifact_root.parent.exists()


def test_librevlm_remote_rechecks_base_after_model_construction(monkeypatch):
    from libreyolo.models import vlm
    from libreyolo.models.vlm import artifact as artifact_module
    from libreyolo.models.vlm import hub

    locations = {}
    expected_base = {"schema": "test-base-identity"}
    expected_manifest = {"schema": "test-artifact"}

    def fake_download(_source, output_dir, **_kwargs):
        root = Path(output_dir)
        root.mkdir()
        locations["temporary"] = root.parent
        return SimpleNamespace(
            root=root,
            base_snapshot=expected_base,
            aggregate_sha256="a" * 64,
            files=("adapter_model.safetensors",),
            manifest=expected_manifest,
        )

    def fake_ensure(info, **_kwargs):
        base = info.root.parent / "base"
        base.mkdir()
        return SimpleNamespace(root=base, identity=expected_base)

    validations = 0

    def validate_then_detect_mutation(_root, _expected):
        nonlocal validations
        validations += 1
        if validations == 2:
            raise ValueError("base snapshot changed after construction")
        return expected_base

    monkeypatch.setattr(hub, "download_vlm_artifact", fake_download)
    monkeypatch.setattr(hub, "ensure_vlm_base_snapshot", fake_ensure)
    monkeypatch.setattr(
        artifact_module,
        "validate_vlm_base_snapshot",
        validate_then_detect_mutation,
    )
    monkeypatch.setattr(
        artifact_module,
        "validate_vlm_artifact",
        lambda root: SimpleNamespace(
            root=root,
            aggregate_sha256="a" * 64,
            files=("adapter_model.safetensors",),
            manifest=expected_manifest,
        ),
    )
    monkeypatch.setattr(vlm, "_load_checkpoint", lambda *_args, **_kwargs: object())

    with pytest.raises(ValueError, match="changed after construction"):
        vlm.LibreVLM(_REMOTE)

    assert validations == 2
    assert not locations["temporary"].exists()


def test_librevlm_remote_rechecks_artifact_after_model_construction(monkeypatch):
    from libreyolo.models import vlm
    from libreyolo.models.vlm import artifact as artifact_module
    from libreyolo.models.vlm import hub

    locations = {}
    expected_base = {"schema": "test-base-identity"}
    initial_manifest = {"schema": "test-artifact"}

    def fake_download(_source, output_dir, **_kwargs):
        root = Path(output_dir)
        root.mkdir()
        locations["temporary"] = root.parent
        return SimpleNamespace(
            root=root,
            base_snapshot=expected_base,
            aggregate_sha256="a" * 64,
            files=("adapter_model.safetensors",),
            manifest=initial_manifest,
        )

    def fake_ensure(info, **_kwargs):
        base = info.root.parent / "base"
        base.mkdir()
        return SimpleNamespace(root=base, identity=expected_base)

    monkeypatch.setattr(hub, "download_vlm_artifact", fake_download)
    monkeypatch.setattr(hub, "ensure_vlm_base_snapshot", fake_ensure)
    monkeypatch.setattr(
        artifact_module,
        "validate_vlm_base_snapshot",
        lambda _root, expected: expected,
    )
    monkeypatch.setattr(
        artifact_module,
        "validate_vlm_artifact",
        lambda root: SimpleNamespace(
            root=root,
            aggregate_sha256="b" * 64,
            files=("adapter_model.safetensors",),
            manifest=initial_manifest,
        ),
    )
    monkeypatch.setattr(vlm, "_load_checkpoint", lambda *_args, **_kwargs: object())

    with pytest.raises(ValueError, match="artifact changed during model construction"):
        vlm.LibreVLM(_REMOTE)

    assert not locations["temporary"].exists()


def test_librevlm_remote_cleans_isolated_directory_after_download_failure(monkeypatch):
    from libreyolo.models import vlm
    from libreyolo.models.vlm import hub

    attempted = {}

    def fail_download(_source, output_dir, **_kwargs):
        output = Path(output_dir)
        attempted["parent"] = output.parent
        assert output.parent.is_dir()
        raise RuntimeError("offline transport failure")

    monkeypatch.setattr(hub, "download_vlm_artifact", fail_download)
    monkeypatch.setattr(
        vlm,
        "_load_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("failed download reached loader"),
    )

    with pytest.raises(RuntimeError, match="offline transport failure"):
        vlm.LibreVLM(_REMOTE)

    assert not attempted["parent"].exists()


class _FakeRemoteVLM:
    FAMILY = "qwen3vl"
    size = "2b"
    task = "detect"
    device = torch.device("cpu")

    def __init__(self):
        self.names = {0: "person"}
        self.vocabularies = []

    def set_classes(self, names):
        self.vocabularies.append(list(names))
        self.names = dict(enumerate(names))
        return self

    def _get_input_size(self):
        return 448

    def __call__(self, source, **_kwargs):
        boxes = Boxes(
            torch.tensor([[1.0, 2.0, 8.0, 9.0]]),
            torch.tensor([0.9]),
            torch.tensor([0]),
        )
        return Results(
            boxes=boxes,
            orig_shape=(10, 12),
            path=str(source),
            names=self.names,
        )


def test_predict_routes_remote_vlm_and_applies_names(monkeypatch, tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    fake = _FakeRemoteVLM()
    load_calls = []

    def fake_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return fake

    monkeypatch.setattr(predict, "load_model_or_exit", fake_load)
    result = _RUNNER.invoke(
        _make_app("predict", predict.predict_cmd),
        [
            "predict",
            f"source={source}",
            f"model={_REMOTE}",
            'names=["forklift","worker"]',
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert load_calls[0][1]["model_path"] == _REMOTE
    assert load_calls[0][1]["names"] == ["forklift", "worker"]
    assert fake.vocabularies == [["forklift", "worker"]]
    payload = json.loads(result.stdout)
    assert payload["model_family"] == "qwen3vl"
    assert payload["results"][0]["detections"][0]["class"] == "forklift"


@pytest.mark.parametrize(
    "option",
    [
        "imgsz=640",
        "face_detector=yolox-s",
        "gallery=faces.npz",
        "gallery_threshold=0.5",
    ],
)
def test_predict_rejects_remote_vlm_incompatible_options_before_load(
    option, monkeypatch, tmp_path
):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    monkeypatch.setattr(
        predict,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("incompatible option loaded artifact"),
    )

    result = _RUNNER.invoke(
        _make_app("predict", predict.predict_cmd),
        ["predict", f"source={source}", f"model={_REMOTE}", option, "--json"],
    )

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"] == "config_unsupported"


@pytest.mark.parametrize(
    ("name", "command", "extra_args"),
    [
        ("train", train.train_cmd, ["data=coco8.yaml", "--dry-run"]),
        ("val", val.val_cmd, ["data=coco8.yaml"]),
        ("export", export.export_cmd, []),
        ("quantize", quantize.quantize_cmd, []),
        ("info", special.info_cmd, []),
    ],
)
def test_non_predict_commands_reject_remote_vlm_before_load(
    name, command, extra_args, monkeypatch
):
    module = __import__(command.__module__, fromlist=["load_model_or_exit"])
    monkeypatch.setattr(
        module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("unsupported command loaded artifact"),
    )

    result = _RUNNER.invoke(
        _make_app(name, command),
        [name, f"model={_REMOTE}", *extra_args, "--json"],
    )

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"] == "config_unsupported"
