"""Behavior tests for the predict command."""

import json
from pathlib import Path

import pytest
import torch
import typer
from PIL import Image
from typer.testing import CliRunner

from libreyolo.cli.commands import predict as predict_module
from libreyolo.cli.commands.predict import predict_cmd
from libreyolo.cli.parsing import KeyValueCommand
from libreyolo.utils.results import (
    Matte,
    PanopticSegmentation,
    Points,
    Probs,
    RestoredImage,
    Results,
    SemanticMask,
)

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer()
    app.command("predict", cls=KeyValueCommand)(predict_cmd)
    return app


class _FakeClassifyModel:
    FAMILY = "yolo9"
    task = "classify"
    size = "t"
    device = "cpu"

    def _get_input_size(self) -> int:
        return 224

    def __call__(self, source, **kwargs):
        return Results(
            boxes=None,
            orig_shape=(10, 12),
            path=str(source),
            names={0: "cat", 1: "dog"},
            probs=Probs(torch.tensor([0.2, 0.8])),
        )


class _FakePointModel:
    FAMILY = "librefomo"
    task = "point"
    size = "s"
    device = "cpu"

    def _get_input_size(self) -> int:
        return 96

    def __call__(self, source, **kwargs):
        return Results(
            boxes=None,
            orig_shape=(10, 12),
            path=str(source),
            names={0: "person"},
            points=Points(torch.tensor([[6.0, 5.0, 0.0, 0.9]])),
        )


class _FakeRestoreModel:
    FAMILY = "nafnet"
    task = "restore"
    size = "s"
    device = "cpu"

    def _get_input_size(self) -> int:
        return 256

    def __call__(self, source, **kwargs):
        del kwargs
        return Results(
            boxes=None,
            orig_shape=(10, 12),
            path=str(source),
            names={0: "image"},
            restored=RestoredImage(torch.zeros((10, 12, 3), dtype=torch.uint8)),
        )


class _FailingModel:
    FAMILY = "yolo9"
    task = "detect"
    size = "t"
    device = "cpu"

    def _get_input_size(self) -> int:
        return 640

    def __call__(self, source, **kwargs):
        print("library diagnostic that must not reach stdout")
        raise RuntimeError("synthetic prediction failure")


class _FakeSavingModel(_FakeClassifyModel):
    def __call__(self, source, **kwargs):
        requested = Path(kwargs["output_path"])
        if requested.suffix:
            actual = requested
            actual.parent.mkdir(parents=True, exist_ok=True)
        else:
            requested.mkdir(parents=True, exist_ok=True)
            actual = requested / "resolved-result.png"
        actual.write_bytes(b"saved")
        result = super().__call__(source, **kwargs)
        result.saved_path = str(actual)
        return result


class _FakeStaticResultModel:
    FAMILY = "dense"
    size = "s"
    device = "cpu"

    def __init__(self, result):
        self.result = result
        self.task = result.task

    def _get_input_size(self) -> int:
        return 8

    def __call__(self, source, **kwargs):
        del source, kwargs
        return self.result


def test_predict_formats_classification_probs(monkeypatch, tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    fake_model = _FakeClassifyModel()

    monkeypatch.setattr(
        predict_module,
        "resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=fake-cls.pt",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    item = data["results"][0]
    assert item["detections"] == []
    assert item["classification"]["name"] == "dog"
    assert item["classification"]["class"] == 1
    assert item["top5"][0]["name"] == "dog"


def test_predict_formats_point_results(monkeypatch, tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    fake_model = _FakePointModel()

    monkeypatch.setattr(
        predict_module,
        "resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=fake-point.pt",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    det = data["results"][0]["detections"][0]
    assert det["class"] == "person"
    assert det["class_id"] == 0
    assert det["confidence"] == 0.9
    assert det["point_xy"] == [6.0, 5.0]


def test_predict_formats_restore_results(monkeypatch, tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    fake_model = _FakeRestoreModel()

    monkeypatch.setattr(
        predict_module,
        "resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    json_result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=fake-restore.pt",
            "--json",
        ],
    )

    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    item = data["results"][0]
    assert item["detections"] == []
    assert item["restored"] == {"shape": [10, 12, 3], "dtype": "uint8"}

    human_result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=fake-restore.pt",
        ],
    )

    assert human_result.exit_code == 0
    assert "restored" in human_result.stdout


@pytest.mark.parametrize("task", ["semantic", "panoptic", "matte"])
def test_predict_formats_dense_primary_payloads(monkeypatch, tmp_path, task):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8)).save(source)
    kwargs = {}
    if task == "semantic":
        mask = torch.zeros((8, 8), dtype=torch.int64)
        mask[:, 4:] = 1
        kwargs["semantic_mask"] = SemanticMask(mask)
    elif task == "panoptic":
        kwargs["panoptic"] = PanopticSegmentation(
            torch.ones((8, 8), dtype=torch.int64),
            [{"id": 1, "category_id": 0, "isthing": True}],
        )
    else:
        kwargs["matte"] = Matte(torch.ones((8, 8), dtype=torch.float32))
    fake_model = _FakeStaticResultModel(
        Results(
            boxes=None,
            orig_shape=(8, 8),
            path=str(source),
            names={0: "thing", 1: "other"},
            **kwargs,
        )
    )
    monkeypatch.setattr(predict_module, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    result = runner.invoke(
        _make_app(),
        [f"source={source}", "model=fake-dense.pt", "--json"],
    )

    assert result.exit_code == 0
    item = json.loads(result.stdout)["results"][0]
    assert item["task"] == task
    assert item[task] is not None
    assert item["detections"] == []


def test_predict_json_preserves_empty_pose_task_schema(monkeypatch, tmp_path):
    from libreyolo.utils.results import Boxes, Keypoints

    source = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8)).save(source)
    fake_model = _FakeStaticResultModel(
        Results(
            boxes=Boxes(torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0)),
            keypoints=Keypoints(torch.zeros((0, 5, 3))),
            orig_shape=(8, 8),
            task="pose",
        )
    )
    monkeypatch.setattr(predict_module, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    result = runner.invoke(
        _make_app(),
        [f"source={source}", "model=fake-pose.pt", "--json", "--quiet"],
    )

    assert result.exit_code == 0
    item = json.loads(result.stdout)["results"][0]
    assert item["task"] == "pose"
    assert item["detections"] == []


@pytest.mark.parametrize(
    "source",
    ["https://example.com/image.jpg", "s3://bucket/image.jpg", "gs://bucket/image.jpg"],
)
def test_predict_accepts_same_remote_schemes_as_python_loader(monkeypatch, source):
    fake_model = _FakeClassifyModel()
    monkeypatch.setattr(predict_module, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    result = runner.invoke(
        _make_app(),
        [f"source={source}", "model=fake-cls.pt", "--json", "--quiet"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["source"] == source


def test_predict_rejects_unsupported_remote_scheme_as_structured_error():
    source = "ftp://example.com/image.jpg"

    result = runner.invoke(
        _make_app(),
        [f"source={source}", "model=fake.pt", "--json", "--quiet"],
    )

    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["error"] == "source_not_found"
    assert payload["stage"] == "source_validation"
    assert payload["source"] == source


@pytest.mark.parametrize("save", [False, True])
def test_predict_failure_is_one_structured_quiet_json_object(
    monkeypatch, tmp_path, save
):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    monkeypatch.setattr(predict_module, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: _FailingModel(),
    )
    args = [
        f"source={source}",
        "model=failing.pt",
        "--json",
        "--quiet",
    ]
    if save:
        args.append("save")

    result = runner.invoke(_make_app(), args)

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error"] == "inference_failed"
    assert payload["stage"] == ("inference/save" if save else "inference")
    assert payload["model"] == "failing.pt"
    assert payload["source"] == str(source)
    assert "library diagnostic" not in result.stdout
    assert result.stderr == ""


def test_predict_reports_resolved_saved_paths(monkeypatch, tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    requested = tmp_path / "requested-output"
    fake_model = _FakeSavingModel()
    monkeypatch.setattr(predict_module, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=fake-cls.pt",
            "save",
            f"output_path={requested}",
            "--json",
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    actual = requested / "resolved-result.png"
    assert payload["output_path"] == str(actual)
    assert payload["saved_paths"] == [str(actual)]
    assert payload["requested_output_path"] == str(requested)
    assert actual.exists()
    assert payload["results"][0]["saved_path"] == str(actual)
