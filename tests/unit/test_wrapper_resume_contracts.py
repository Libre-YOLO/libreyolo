from __future__ import annotations

import importlib

import pytest
import torch

pytestmark = pytest.mark.unit


_CASES = [
    (
        "dfine",
        "libreyolo.models.dfine.model",
        "LibreDFINE",
        "libreyolo.models.dfine.trainer",
        "DFINETrainer",
    ),
    (
        "deim",
        "libreyolo.models.deim.model",
        "LibreDEIM",
        "libreyolo.models.deim.trainer",
        "DEIMTrainer",
    ),
    (
        "deimv2",
        "libreyolo.models.deimv2.model",
        "LibreDEIMv2",
        "libreyolo.models.deimv2.trainer",
        "DEIMv2Trainer",
    ),
    (
        "ec",
        "libreyolo.models.ec.model",
        "LibreEC",
        "libreyolo.models.ec.trainer",
        "ECTrainer",
    ),
    (
        "rtdetrv4",
        "libreyolo.models.rtdetrv4.model",
        "LibreRTDETRv4",
        "libreyolo.models.rtdetrv4.trainer",
        "RTDETRv4Trainer",
    ),
    (
        "yolonas",
        "libreyolo.models.yolonas.model",
        "LibreYOLONAS",
        "libreyolo.models.yolonas.trainer",
        "YOLONASTrainer",
    ),
    (
        "segformer",
        "libreyolo.models.segformer.model",
        "LibreSegformer",
        "libreyolo.models.segformer.trainer",
        "SegformerTrainer",
    ),
    (
        "dinov2",
        "libreyolo.models.dinov2.model",
        "LibreDINOv2",
        "libreyolo.models.dinov2.trainer",
        "DINOv2Trainer",
    ),
    (
        "rfdetr",
        "libreyolo.models.rfdetr.model",
        "LibreRFDETR",
        "libreyolo.models.rfdetr.model",
        "RFDETRTrainer",
    ),
]


class _TrackingModel:
    def __init__(self):
        self.to_calls = []
        self.eval_calls = 0

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self


def _make_wrapper(model_module, wrapper_name, model_path):
    wrapper_cls = getattr(model_module, wrapper_name)
    wrapper = wrapper_cls.__new__(wrapper_cls)
    wrapper.model = _TrackingModel()
    wrapper.model_path = str(model_path) if model_path is not None else None
    wrapper.device = torch.device("cpu")
    wrapper.size = "s"
    wrapper.nb_classes = 2
    wrapper.input_size = 640
    wrapper.task = "detect"
    wrapper.names = {0: "zero", 1: "one"}
    loaded = []
    wrapper._load_weights = lambda path: loaded.append(path)
    if wrapper_name == "LibreRFDETR":
        wrapper._resume_checkpoint_uses_lora = lambda _path: False
    return wrapper, loaded


def _install_trainer(monkeypatch, module_name, trainer_name, result):
    trainer_module = importlib.import_module(module_name)
    events = []
    captured = {}

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            events.append("init")

        def setup(self):
            events.append("setup")

        def resume(self, checkpoint_path):
            captured["resume"] = checkpoint_path
            events.append("resume")

        def train(self):
            events.append("train")
            return dict(result)

    monkeypatch.setattr(trainer_module, trainer_name, _DummyTrainer)
    return events, captured


@pytest.mark.parametrize(
    ("family", "model_module_name", "wrapper_name", "trainer_module", "trainer_name"),
    _CASES,
    ids=[case[0] for case in _CASES],
)
@pytest.mark.parametrize("resume_kind", ["loaded", "explicit"])
@pytest.mark.parametrize("saved_kind", ["best", "last"])
def test_wrapper_resume_runs_training_and_restores_checkpoint(
    monkeypatch,
    tmp_path,
    family,
    model_module_name,
    wrapper_name,
    trainer_module,
    trainer_name,
    resume_kind,
    saved_kind,
):
    data_module = importlib.import_module("libreyolo.data")
    monkeypatch.setattr(
        data_module,
        "load_data_config",
        lambda *_args, **_kwargs: {
            "yaml_file": "data.yaml",
            "nc": 2,
        },
    )

    loaded_checkpoint = tmp_path / f"{family}-loaded.pt"
    explicit_checkpoint = tmp_path / f"{family}-explicit.pt"
    best_checkpoint = tmp_path / f"{family}-best.pt"
    last_checkpoint = tmp_path / f"{family}-last.pt"
    loaded_checkpoint.touch()
    explicit_checkpoint.touch()
    best_checkpoint.touch()
    last_checkpoint.touch()

    model_module = importlib.import_module(model_module_name)
    wrapper, loaded_paths = _make_wrapper(
        model_module,
        wrapper_name,
        loaded_checkpoint,
    )
    events, captured = _install_trainer(
        monkeypatch,
        trainer_module,
        trainer_name,
        {
            "best_checkpoint": (
                str(best_checkpoint) if saved_kind == "best" else None
            ),
            "last_checkpoint": str(last_checkpoint),
        },
    )

    resume = True if resume_kind == "loaded" else explicit_checkpoint
    expected_resume = loaded_checkpoint if resume is True else explicit_checkpoint
    kwargs = {"allow_experimental": True} if family == "ec" else {}
    wrapper.train("data.yaml", resume=resume, **kwargs)

    assert events == ["init", "resume", "train"]
    assert captured["resume"] == str(expected_resume)
    assert captured["kwargs"]["resume"] is True
    expected_saved = best_checkpoint if saved_kind == "best" else last_checkpoint
    assert wrapper.model_path == str(expected_saved.resolve())
    assert loaded_paths == [str(expected_saved.resolve())]
    assert wrapper.model.to_calls[-1] == wrapper.device
    assert wrapper.model.eval_calls == 1


@pytest.mark.parametrize(
    ("family", "model_module_name", "wrapper_name", "trainer_module", "trainer_name"),
    _CASES,
    ids=[case[0] for case in _CASES],
)
def test_wrapper_resume_true_requires_loaded_checkpoint(
    monkeypatch,
    family,
    model_module_name,
    wrapper_name,
    trainer_module,
    trainer_name,
):
    data_module = importlib.import_module("libreyolo.data")
    monkeypatch.setattr(
        data_module,
        "load_data_config",
        lambda *_args, **_kwargs: {
            "yaml_file": "data.yaml",
            "nc": 2,
        },
    )
    model_module = importlib.import_module(model_module_name)
    wrapper, _ = _make_wrapper(model_module, wrapper_name, None)
    events, _ = _install_trainer(
        monkeypatch,
        trainer_module,
        trainer_name,
        {},
    )

    kwargs = {"allow_experimental": True} if family == "ec" else {}
    with pytest.raises(ValueError, match="resume=True requires a checkpoint"):
        wrapper.train("data.yaml", resume=True, **kwargs)

    assert events == []


@pytest.mark.parametrize("family", ["ec", "yolonas"])
@pytest.mark.parametrize("resume_kind", ["loaded", "explicit"])
def test_pose_wrapper_resume_runs_training_and_restores_last_checkpoint(
    monkeypatch,
    tmp_path,
    family,
    resume_kind,
):
    if family == "ec":
        model_module_name = "libreyolo.models.ec.model"
        wrapper_name = "LibreEC"
        trainer_module = "libreyolo.models.ec.pose_trainer"
        trainer_name = "ECPoseTrainer"
        data_config = {
            "yaml_file": "data.yaml",
            "nc": 1,
            "names": ["person"],
            "kpt_shape": [17, 3],
        }
    else:
        model_module_name = "libreyolo.models.yolonas.model"
        wrapper_name = "LibreYOLONAS"
        trainer_module = "libreyolo.models.yolonas.pose_trainer"
        trainer_name = "YOLONASPoseTrainer"
        data_config = {
            "yaml_file": "data.yaml",
            "nc": 2,
            "names": ["zero", "one"],
            "kpt_shape": [17, 3],
        }

    data_module = importlib.import_module("libreyolo.data")
    monkeypatch.setattr(
        data_module,
        "load_data_config",
        lambda *_args, **_kwargs: data_config,
    )

    loaded_checkpoint = tmp_path / f"{family}-pose-loaded.pt"
    explicit_checkpoint = tmp_path / f"{family}-pose-explicit.pt"
    last_checkpoint = tmp_path / f"{family}-pose-last.pt"
    loaded_checkpoint.touch()
    explicit_checkpoint.touch()
    last_checkpoint.touch()

    model_module = importlib.import_module(model_module_name)
    wrapper, loaded_paths = _make_wrapper(
        model_module,
        wrapper_name,
        loaded_checkpoint,
    )
    wrapper.task = "pose"
    wrapper.num_keypoints = 17
    if family == "ec":
        wrapper.nb_classes = 1
        wrapper.names = {0: "person"}
    events, captured = _install_trainer(
        monkeypatch,
        trainer_module,
        trainer_name,
        {
            "best_checkpoint": None,
            "last_checkpoint": str(last_checkpoint),
        },
    )

    resume = True if resume_kind == "loaded" else explicit_checkpoint
    expected_resume = loaded_checkpoint if resume is True else explicit_checkpoint
    kwargs = {"allow_experimental": True} if family == "ec" else {}
    wrapper.train("data.yaml", resume=resume, **kwargs)

    assert events == ["init", "resume", "train"]
    assert captured["resume"] == str(expected_resume)
    assert captured["kwargs"]["resume"] is True
    assert wrapper.model_path == str(last_checkpoint)
    assert loaded_paths == [str(last_checkpoint)]
    assert wrapper.model.eval_calls == 1
