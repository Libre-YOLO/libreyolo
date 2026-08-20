"""Behavior tests for the predict command."""

import json

import pytest
import torch
import typer
from PIL import Image
from typer.testing import CliRunner

from libreyolo.cli.commands import predict as predict_module
from libreyolo.cli.commands.predict import predict_cmd
from libreyolo.cli.parsing import KeyValueCommand
from libreyolo.utils.results import Boxes, Points, Probs, RestoredImage, Results

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


class _FakeGuidedRestoreModel(_FakeRestoreModel):
    def __init__(self):
        self.calls = []

    def __call__(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return super().__call__(source, **kwargs)


class _FakeStreamModel:
    FAMILY = "yolo9"
    task = "classify"
    size = "t"
    device = "cpu"

    def __init__(self):
        self.calls = []

    def _get_input_size(self) -> int:
        return 224

    def __call__(self, source, **kwargs):
        self.calls.append((source, kwargs))

        def generate():
            for frame_idx in range(2):
                yield Results(
                    boxes=None,
                    orig_shape=(10, 12),
                    path=str(source),
                    names={0: "cat", 1: "dog"},
                    probs=Probs(torch.tensor([0.2, 0.8])),
                    frame_idx=frame_idx,
                )

        return generate()


class _FakeVLM:
    FAMILY = "qwen3vl"
    task = "detect"
    size = "2b"
    device = "cpu"

    def __init__(self):
        self.names = {0: "person"}
        self.calls = []
        self.vocabularies = []

    def _get_input_size(self) -> int:
        return 1024

    def set_classes(self, names):
        self.vocabularies.append(list(names))
        self.names = {index: name for index, name in enumerate(names)}
        return self

    def __call__(self, source, **kwargs):
        self.calls.append((source, kwargs))
        boxes = Boxes(
            torch.tensor([[1.0, 2.0, 8.0, 9.0]]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            orig_shape=(10, 12),
        )
        return Results(
            boxes=boxes,
            orig_shape=(10, 12),
            path=str(source),
            names=self.names,
        )


class _FakeGazeModel:
    FAMILY = "l2cs"
    task = "gaze"
    size = "r50"
    device = "cpu"


def test_predict_vlm_names_set_open_vocabulary_and_classes_remain_filter(
    monkeypatch, tmp_path
):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    fake_model = _FakeVLM()
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=qwen3-vl-2b",
            'names=["forklift","worker"]',
            "classes=[1]",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert fake_model.vocabularies == [["forklift", "worker"]]
    assert fake_model.calls[0][1]["classes"] == [1]
    data = json.loads(result.stdout)
    assert data["model_family"] == "qwen3vl"
    assert data["results"][0]["detections"][0]["class"] == "worker"


def test_predict_vlm_names_are_trimmed(monkeypatch, tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    fake_model = _FakeVLM()
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=qwen3-vl-2b",
            'names=[" forklift ","worker"]',
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_model.vocabularies == [["forklift", "worker"]]


def test_predict_vlm_classes_only_remains_numeric_filter(monkeypatch, tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    fake_model = _FakeVLM()
    fake_model.names = {0: "person", 1: "vehicle"}
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=qwen3-vl-2b",
            "classes=[1]",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_model.vocabularies == []
    assert fake_model.calls[0][1]["classes"] == [1]
    assert (
        json.loads(result.stdout)["results"][0]["detections"][0]["class"] == "vehicle"
    )


@pytest.mark.parametrize(
    "names",
    ["not-json", "[]", '["cat",""]', '["Cat","cat"]', '{"cat": 1}'],
)
def test_predict_rejects_invalid_vlm_names_before_loading(names, monkeypatch, tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: pytest.fail("invalid names must not load a model"),
    )

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=qwen3-vl-2b",
            f"names={names}",
            "--json",
        ],
    )

    assert result.exit_code != 0
    assert json.loads(result.stdout)["error"] == "config_type_error"


def test_predict_rejects_names_for_detector_before_loading(monkeypatch, tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: pytest.fail("unsupported names must not load a model"),
    )

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=yolox-s",
            'names=["cat"]',
            "--json",
        ],
    )

    assert result.exit_code != 0
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"
    assert "classes=" in data["message"]


def test_predict_rejects_vlm_imgsz_before_loading(monkeypatch, tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: pytest.fail("ignored imgsz must not load a VLM"),
    )

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=qwen3-vl-2b",
            "imgsz=640",
            "--json",
        ],
    )

    assert result.exit_code == 2, result.output
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"
    assert "processor owns" in data["message"]


@pytest.mark.parametrize(
    "option",
    [
        "face_detector=yolox-s",
        "gallery=faces.npz",
        "gallery_threshold=0.5",
    ],
)
def test_predict_rejects_face_options_for_vlm_before_loading(
    option, monkeypatch, tmp_path
):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: pytest.fail("face option loaded a VLM"),
    )

    result = runner.invoke(
        _make_app(),
        [f"source={source}", "model=qwen3-vl-2b", option, "--json"],
    )

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"] == "config_unsupported"


def test_predict_rejects_vlm_as_gaze_face_detector_before_auxiliary_load(
    monkeypatch, tmp_path
):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    loads = []

    def fake_load(*args, **kwargs):
        loads.append((args, kwargs))
        if len(loads) > 1:
            pytest.fail("unsupported VLM face detector reached model loading")
        return _FakeGazeModel()

    monkeypatch.setattr(predict_module, "load_model_or_exit", fake_load)

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=l2cs-r50",
            "face_detector=qwen3-vl-2b",
            "--json",
        ],
    )

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"] == "config_unsupported"
    assert len(loads) == 1


@pytest.mark.parametrize(
    ("option", "error"),
    [
        ("color_format=xyz", "config_type_error"),
        ("output_file_format=tiff", "config_type_error"),
        ("batch=0", "config_range_error"),
        ("max_det=0", "config_range_error"),
    ],
)
def test_predict_rejects_invalid_runtime_config_before_loading(
    option, error, monkeypatch, tmp_path
):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: pytest.fail("invalid config loaded a model"),
    )

    result = runner.invoke(
        _make_app(),
        [f"source={source}", "model=qwen3-vl-2b", option, "--json"],
    )

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"] == error


@pytest.mark.parametrize("classes", ["not-json", '["cat"]', "[true]", "[-1]"])
def test_predict_rejects_invalid_classes_before_vlm_loading(
    classes, monkeypatch, tmp_path
):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: pytest.fail("invalid classes must not load a VLM"),
    )

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=qwen3-vl-2b",
            f"classes={classes}",
            "--json",
        ],
    )

    assert result.exit_code != 0
    assert json.loads(result.stdout)["error"] == "config_type_error"


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


@pytest.mark.parametrize("guide_name", ["mask", "trimap"])
def test_predict_forwards_guided_image_option(monkeypatch, tmp_path, guide_name):
    source = tmp_path / "image.jpg"
    guide = tmp_path / f"{guide_name}.png"
    Image.new("RGB", (12, 10)).save(source)
    Image.new("L", (12, 10)).save(guide)
    fake_model = _FakeGuidedRestoreModel()

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
            "model=fake-guided.pt",
            f"{guide_name}={guide}",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert fake_model.calls[0][1][guide_name] == str(guide)


def test_predict_webcam_source_auto_streams_as_ndjson(monkeypatch):
    fake_model = _FakeStreamModel()
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
            "source=0",
            "model=fake-stream.pt",
            "stream_buffer=true",
            "vid_stride=2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["frame_index"] for row in rows] == [0, 1]
    assert all(row["results"][0]["predictions"][0]["name"] == "dog" for row in rows)
    source, kwargs = fake_model.calls[0]
    assert source == 0
    assert kwargs["stream"] is True
    assert kwargs["stream_buffer"] is True
    assert kwargs["vid_stride"] == 2


def test_predict_rtsp_source_bypasses_local_path_validation(monkeypatch):
    fake_model = _FakeStreamModel()
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

    source = "rtsp://127.0.0.1:8554/camera"
    result = runner.invoke(
        _make_app(),
        [f"source={source}", "model=fake-stream.pt", "--json"],
    )

    assert result.exit_code == 0
    assert fake_model.calls[0][0] == source
