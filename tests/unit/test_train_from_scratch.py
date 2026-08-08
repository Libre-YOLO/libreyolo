"""G0/G1 train-from-scratch contracts."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands.train import train_cmd
from libreyolo.cli.config import get_model_class
from libreyolo.cli.parsing import KeyValueCommand
from libreyolo.models.base.model import _wrap_train_with_cfg
from libreyolo.models.registry import families_in

pytestmark = pytest.mark.unit

SCRATCH_FAMILIES = families_in("g0") + families_in("g1")
CLI_CASES = (
    ("yolo9-t", "yolo9", "t"),
    ("rfdetr-n", "rfdetr", "n"),
    ("yolo9_e2e-t", "yolo9_e2e", "t"),
    ("yolo9_p2-t", "yolo9_p2", "t"),
    ("ec-s", "ec", "s"),
    ("rtdetr-r18", "rtdetr", "r18"),
    ("rtdetrv2-r18", "rtdetrv2", "r18"),
    ("rtdetrv4-s", "rtdetrv4", "s"),
    ("dfine-n", "dfine", "n"),
    ("deim-n", "deim", "n"),
    ("deimv2-atto", "deimv2", "atto"),
    ("yolonas-s", "yolonas", "s"),
)


class _ScratchReached(RuntimeError):
    pass


@pytest.mark.parametrize("family", SCRATCH_FAMILIES)
def test_pretrained_false_reaches_scratch_reset_for_every_family(family):
    """Intercept the flag before any family-specific training code runs."""
    model_cls = get_model_class(family)
    assert model_cls is not None
    model = model_cls.__new__(model_cls)

    def reset(*, seed):
        assert seed == 37
        raise _ScratchReached

    model._reset_for_scratch = reset
    with pytest.raises(_ScratchReached):
        model.train("unused.yaml", pretrained=False, seed=37)


@pytest.mark.parametrize("family", SCRATCH_FAMILIES)
def test_pretrained_false_rejects_resume_before_reset(family):
    model_cls = get_model_class(family)
    assert model_cls is not None
    model = model_cls.__new__(model_cls)
    model._reset_for_scratch = lambda **_kwargs: pytest.fail(
        "resume conflict must be rejected before rebuilding"
    )

    with pytest.raises(ValueError, match="cannot be combined with resume"):
        model.train("unused.yaml", pretrained=False, resume=True)


def test_scratch_handling_does_not_change_g2_behavior():
    captured = {}

    class Wrapper:
        FAMILY = "yolox"

        def _reset_for_scratch(self, **_kwargs):
            pytest.fail("G2 behavior must remain unchanged")

    def train(_self, data, **kwargs):
        captured.update(data=data, kwargs=kwargs)

    _wrap_train_with_cfg(train)(Wrapper(), "data.yaml", pretrained=False)
    assert captured == {"data": "data.yaml", "kwargs": {"pretrained": False}}


def test_explicit_pretrained_parameter_remains_false_after_reset():
    captured = {}

    class Wrapper:
        FAMILY = "yolo9"

        def _reset_for_scratch(self, *, seed):
            captured["seed"] = seed

    def train(_self, data, *, pretrained=True, seed=0):
        captured.update(data=data, pretrained=pretrained)

    _wrap_train_with_cfg(train)(
        Wrapper(), "data.yaml", pretrained=False, seed=17
    )
    assert captured == {"seed": 17, "data": "data.yaml", "pretrained": False}


def test_scratch_flag_and_seed_can_come_from_cfg(tmp_path):
    captured = {}
    cfg = tmp_path / "scratch.yaml"
    cfg.write_text("pretrained: false\nseed: 19\n")

    class Wrapper:
        FAMILY = "dfine"

        def _reset_for_scratch(self, *, seed):
            captured["seed"] = seed

    def train(_self, data, **kwargs):
        captured.update(data=data, kwargs=kwargs)

    _wrap_train_with_cfg(train)(Wrapper(), "data.yaml", cfg=cfg)
    assert captured == {"seed": 19, "data": "data.yaml", "kwargs": {"seed": 19}}


def test_ddp_workers_rebuild_with_the_scratch_policy():
    from libreyolo.training.ddp_spawn import _build_init_kw

    class Wrapper:
        def __init__(self, size, **kwargs):
            pass

    model = Wrapper.__new__(Wrapper)
    model.size = "s"
    model._training_from_scratch = True

    assert _build_init_kw(model)["_scratch_init"] is True


def test_yolo9_scratch_reset_is_seeded_and_discards_loaded_state():
    from libreyolo.models.yolo9.model import LibreYOLO9

    model = LibreYOLO9._from_scratch(
        size="t", nb_classes=3, device="cpu", seed=11
    )
    expected = next(model.model.parameters()).detach().clone()
    with torch.no_grad():
        next(model.model.parameters()).fill_(123)

    model._reset_for_scratch(seed=11)

    assert torch.equal(next(model.model.parameters()), expected)
    assert model.model_path is None
    assert model.model.training
    assert model._training_from_scratch is True
    assert model.names == {0: "class_0", 1: "class_1", 2: "class_2"}


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("libreyolo.models.rtdetr.model", "LibreRTDETR"),
        ("libreyolo.models.rtdetrv2.model", "LibreRTDETRv2"),
    ],
)
def test_rtdetr_scratch_backbone_stays_trainable_after_rehead(
    monkeypatch, module_name, class_name
):
    module = __import__(module_name, fromlist=[class_name])
    model_cls = getattr(module, class_name)
    monkeypatch.setattr(
        torch.hub,
        "load_state_dict_from_url",
        lambda *_args, **_kwargs: pytest.fail("scratch build downloaded weights"),
    )

    wrapper = model_cls._from_scratch(
        size="r18", nb_classes=2, device="cpu", seed=5
    )
    wrapper._rebuild_for_new_classes(3)

    from libreyolo.models.rtdetr.backbone import FrozenBatchNorm2d

    backbone = wrapper.model.backbone
    assert all(parameter.requires_grad for parameter in backbone.parameters())
    assert not any(isinstance(layer, FrozenBatchNorm2d) for layer in backbone.modules())


def test_rfdetr_scratch_clears_checkpoint_derived_class_layout(monkeypatch):
    import libreyolo.models.rfdetr.model as rfdetr_module

    built_classes = []

    def build_model(*, nb_classes, **_kwargs):
        built_classes.append(nb_classes)
        return nn.Identity()

    monkeypatch.setattr(rfdetr_module, "LibreRFDETRModel", build_model)
    model = rfdetr_module.LibreRFDETR._from_scratch(
        size="n", nb_classes=5, device="cpu", seed=3
    )
    model._model_num_classes = 90
    model._weight_source = "checkpoint.pt"

    model._reset_for_scratch(seed=3)

    assert built_classes == [5, 5]
    assert model._model_num_classes == 5
    assert model._weight_source is None


def test_rfdetr_ddp_scratch_worker_loads_temporary_state(monkeypatch):
    import libreyolo.models.rfdetr.model as rfdetr_module

    loaded = []
    monkeypatch.setattr(
        rfdetr_module,
        "LibreRFDETRModel",
        lambda **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        rfdetr_module.LibreRFDETR,
        "_load_weights",
        lambda _self, source: loaded.append(source),
    )

    model = rfdetr_module.LibreRFDETR(
        "scratch-worker.pt",
        size="n",
        nb_classes=5,
        device="cpu",
        _scratch_init=True,
    )

    assert loaded and str(loaded[0]).endswith("scratch-worker.pt")
    assert model._training_from_scratch is True


@pytest.mark.parametrize(
    "module_name,class_name,decoder_name",
    [
        ("libreyolo.models.dfine.nn", "LibreDFINEModel", "DFINETransformer"),
        ("libreyolo.models.deim.nn", "LibreDEIMModel", "DEIMTransformer"),
    ],
)
@pytest.mark.parametrize("scratch", [False, True], ids=["finetune", "scratch"])
def test_hgnet_scratch_build_only_disables_finetune_freezes(
    monkeypatch, module_name, class_name, decoder_name, scratch
):
    module = __import__(module_name, fromlist=[class_name])
    captured = {}

    def build_backbone(**kwargs):
        captured.update(kwargs)
        return nn.Identity()

    monkeypatch.setattr(module, "HGNetv2", build_backbone)
    monkeypatch.setattr(module, "HybridEncoder", lambda **_kwargs: nn.Identity())
    monkeypatch.setattr(module, decoder_name, lambda **_kwargs: nn.Identity())

    getattr(module, class_name)(config="l", train_from_scratch=scratch)

    cfg = module.SIZE_CONFIGS["l"]
    expected = (
        (False, -1, False)
        if scratch
        else (cfg["freeze_stem_only"], cfg["freeze_at"], cfg["freeze_norm"])
    )
    assert (
        captured["freeze_stem_only"],
        captured["freeze_at"],
        captured["freeze_norm"],
    ) == expected


@pytest.mark.parametrize(
    "module_name,class_name,model_name",
    [
        ("libreyolo.models.dfine.model", "LibreDFINE", "LibreDFINEModel"),
        ("libreyolo.models.deim.model", "LibreDEIM", "LibreDEIMModel"),
        ("libreyolo.models.rtdetrv4.model", "LibreRTDETRv4", "LibreDFINEModel"),
    ],
)
def test_hgnet_wrappers_preserve_scratch_policy_on_rebuild(
    monkeypatch, module_name, class_name, model_name
):
    module = __import__(module_name, fromlist=[class_name])
    captured = {}
    monkeypatch.setattr(
        module,
        model_name,
        lambda **kwargs: captured.update(kwargs) or nn.Identity(),
    )
    wrapper_cls = getattr(module, class_name)
    wrapper = wrapper_cls.__new__(wrapper_cls)
    wrapper.size = "s"
    wrapper.nb_classes = 3
    wrapper.input_size = 640
    wrapper.task = "detect"
    wrapper._training_from_scratch = True

    wrapper._init_model()

    assert captured["train_from_scratch"] is True


@pytest.mark.parametrize(
    "alias,family,size",
    CLI_CASES,
    ids=[case[1] for case in CLI_CASES],
)
def test_cli_known_alias_builds_scratch_without_loading_weights(
    monkeypatch, tmp_path, alias, family, size
):
    import libreyolo.cli.commands.train as train_module

    captured = {}

    class ScratchModel:
        device = torch.device("cpu")

        def train(self, data, **kwargs):
            captured["train"] = {"data": data, "kwargs": kwargs}
            return {"output_dir": str(tmp_path / "run")}

    ScratchModel.FAMILY = family

    class ScratchClass:
        DEFAULT_TASK = "detect"

        @classmethod
        def detect_size_from_filename(cls, _filename):
            return size

        @classmethod
        def detect_task_from_filename(cls, _filename):
            return None

        @classmethod
        def _from_scratch(cls, **kwargs):
            captured["init"] = kwargs
            return ScratchModel()

    monkeypatch.setattr(train_module, "get_model_class", lambda _family: ScratchClass)
    monkeypatch.setattr(
        train_module,
        "load_model_or_exit",
        lambda *_args, **_kwargs: pytest.fail("scratch aliases must not load weights"),
    )

    app = typer.Typer()
    app.command("train", cls=KeyValueCommand)(train_cmd)
    result = CliRunner().invoke(
        app,
        [
            "data=unused.yaml",
            f"model={alias}",
            "pretrained=false",
            "epochs=1",
            "seed=29",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["model_family"] == family
    assert captured["init"] == {
        "size": size,
        "task": "detect",
        "device": "auto",
        "seed": 29,
    }
    assert captured["train"]["data"] == "unused.yaml"
    assert "pretrained" not in captured["train"]["kwargs"]
