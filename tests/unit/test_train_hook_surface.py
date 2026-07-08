"""Regression coverage for the public training hook surface."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from libreyolo.models.deim.model import LibreDEIM
from libreyolo.models.deimv2.model import LibreDEIMv2
from libreyolo.models.dfine.model import LibreDFINE
from libreyolo.models.ec.model import LibreEC
from libreyolo.models.fomo.model import LibreFOMO
from libreyolo.models.picodet.model import LibrePICODET
from libreyolo.models.rfdetr.model import LibreRFDETR
from libreyolo.models.rtmdet.model import LibreRTMDet
from libreyolo.models.rtdetr.model import LibreRTDETR
from libreyolo.models.rtdetrv2.model import LibreRTDETRv2
from libreyolo.models.rtdetrv4.model import LibreRTDETRv4
from libreyolo.models.yolo7.model import LibreYOLO7
from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.models.yolo9_e2e.model import LibreYOLO9E2E
from libreyolo.models.yolonas.model import LibreYOLONAS
from libreyolo.models.yolox.model import LibreYOLOX

pytestmark = pytest.mark.unit


TRAINABLE_MODEL_CLASSES = [
    LibreYOLOX,
    LibreYOLO7,
    LibreYOLO9,
    LibreYOLO9E2E,
    LibreYOLONAS,
    LibreRTDETR,
    LibreRTDETRv2,
    LibreRTDETRv4,
    LibreRFDETR,
    LibreDFINE,
    LibreDEIM,
    LibreDEIMv2,
    LibreRTMDet,
    LibrePICODET,
    LibreFOMO,
    LibreEC,
]


@pytest.mark.parametrize(
    "model_cls",
    TRAINABLE_MODEL_CLASSES,
    ids=lambda cls: cls.__name__,
)
def test_trainable_models_expose_callbacks_and_loggers(model_cls: type) -> None:
    signature = inspect.signature(model_cls.train)

    for name in ("callbacks", "loggers"):
        assert name in signature.parameters
        assert signature.parameters[name].default is None


@pytest.mark.parametrize(
    "module_path,class_name",
    [
        ("libreyolo.models.sam.model", "LibreSAM1"),
        ("libreyolo.models.sam.sam2", "LibreSAM2"),
        ("libreyolo.models.mobilesam.model", "LibreMobileSAM"),
        ("libreyolo.models.l2cs.model", "LibreL2CS"),
        ("libreyolo.models.depth_anything.model", "LibreDepthAnythingV2"),
        ("libreyolo.models.vlm.base", "LibreVLMModel"),
    ],
)
def test_inference_only_models_keep_train_out_of_scope(
    module_path: str,
    class_name: str,
) -> None:
    module = pytest.importorskip(module_path)
    model_cls = getattr(module, class_name)
    instance = model_cls.__new__(model_cls)

    with pytest.raises(NotImplementedError):
        model_cls.train(instance, "dummy.yaml")


class _StopTraining(RuntimeError):
    """Sentinel raised by patched trainer constructors."""


@dataclass(frozen=True)
class ForwardingCase:
    model_factory: Callable[[], Any]
    trainer_target: str
    train_kwargs: dict[str, Any] | None = None
    data_config: dict[str, Any] | None = None


def _detect_data_config(nc: int = 2) -> dict[str, Any]:
    return {
        "yaml_file": "dummy.yaml",
        "nc": nc,
        "names": {i: f"class_{i}" for i in range(nc)},
    }


def _pose_data_config() -> dict[str, Any]:
    return {
        "yaml_file": "pose.yaml",
        "nc": 1,
        "names": {0: "person"},
        "kpt_shape": [17, 3],
        "flip_idx": list(range(17)),
    }


FORWARDING_CASES = {
    "yolox": ForwardingCase(
        lambda: LibreYOLOX(None, size="n", nb_classes=2, device="cpu"),
        "libreyolo.models.yolox.trainer.YOLOXTrainer",
    ),
    "yolo7": ForwardingCase(
        lambda: LibreYOLO7(None, size="b", nb_classes=2, device="cpu"),
        "libreyolo.models.yolo7.trainer.YOLOv7Trainer",
    ),
    "yolo9": ForwardingCase(
        lambda: LibreYOLO9(None, size="t", nb_classes=2, device="cpu"),
        "libreyolo.models.yolo9.trainer.YOLO9Trainer",
    ),
    "yolo9_e2e": ForwardingCase(
        lambda: LibreYOLO9E2E(None, size="t", nb_classes=2, device="cpu"),
        "libreyolo.models.yolo9_e2e.trainer.YOLO9E2ETrainer",
    ),
    "yolonas_detect": ForwardingCase(
        lambda: LibreYOLONAS(None, size="s", nb_classes=2, device="cpu"),
        "libreyolo.models.yolonas.trainer.YOLONASTrainer",
    ),
    "yolonas_pose": ForwardingCase(
        lambda: LibreYOLONAS(None, size="s", device="cpu", task="pose"),
        "libreyolo.models.yolonas.pose_trainer.YOLONASPoseTrainer",
        data_config=_pose_data_config(),
    ),
    "rtdetr": ForwardingCase(
        lambda: LibreRTDETR(None, size="r18", nb_classes=2, device="cpu"),
        "libreyolo.models.rtdetr.trainer.RTDETRTrainer",
    ),
    "rtdetrv2": ForwardingCase(
        lambda: LibreRTDETRv2(None, size="r18", nb_classes=2, device="cpu"),
        "libreyolo.models.rtdetrv2.trainer.RTDETRv2Trainer",
    ),
    "rtdetrv4": ForwardingCase(
        lambda: LibreRTDETRv4(None, size="s", nb_classes=2, device="cpu"),
        "libreyolo.models.rtdetrv4.trainer.RTDETRv4Trainer",
    ),
    "rfdetr": ForwardingCase(
        lambda: LibreRFDETR({}, size="n", nb_classes=2, device="cpu"),
        "libreyolo.models.rfdetr.model.RFDETRTrainer",
    ),
    "dfine": ForwardingCase(
        lambda: LibreDFINE(None, size="n", nb_classes=2, device="cpu"),
        "libreyolo.models.dfine.trainer.DFINETrainer",
    ),
    "deim": ForwardingCase(
        lambda: LibreDEIM(None, size="n", nb_classes=2, device="cpu"),
        "libreyolo.models.deim.trainer.DEIMTrainer",
    ),
    "deimv2": ForwardingCase(
        lambda: LibreDEIMv2(None, size="atto", nb_classes=2, device="cpu"),
        "libreyolo.models.deimv2.trainer.DEIMv2Trainer",
    ),
    "rtmdet": ForwardingCase(
        lambda: LibreRTMDet(None, size="t", nb_classes=2, device="cpu"),
        "libreyolo.models.rtmdet.trainer.RTMDetTrainer",
        train_kwargs={"allow_experimental": True},
    ),
    "picodet": ForwardingCase(
        lambda: LibrePICODET(None, size="s", nb_classes=2, device="cpu"),
        "libreyolo.models.picodet.trainer.PICODETTrainer",
        train_kwargs={"allow_experimental": True},
    ),
    "fomo": ForwardingCase(
        lambda: LibreFOMO(None, size="s", nb_classes=2, device="cpu"),
        "libreyolo.models.fomo.trainer.FOMOTrainer",
        train_kwargs={"allow_experimental": True},
    ),
    "ec_detect": ForwardingCase(
        lambda: LibreEC(None, size="s", nb_classes=2, device="cpu"),
        "libreyolo.models.ec.trainer.ECTrainer",
        train_kwargs={"allow_experimental": True},
    ),
    "ec_segment": ForwardingCase(
        lambda: LibreEC(None, size="s", nb_classes=2, device="cpu", task="segment"),
        "libreyolo.models.ec.seg_trainer.ECSegTrainer",
        train_kwargs={"allow_experimental": True},
    ),
    "ec_pose": ForwardingCase(
        lambda: LibreEC(None, size="s", device="cpu", task="pose"),
        "libreyolo.models.ec.pose_trainer.ECPoseTrainer",
        train_kwargs={"allow_experimental": True},
        data_config=_pose_data_config(),
    ),
}


@pytest.mark.parametrize(
    "case",
    FORWARDING_CASES.values(),
    ids=FORWARDING_CASES.keys(),
)
def test_train_hooks_reach_family_trainer(
    case: ForwardingCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = object()
    captured: dict[str, Any] = {}

    class CapturingTrainer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs
            raise _StopTraining

    monkeypatch.setattr(case.trainer_target, CapturingTrainer)

    data_config = case.data_config or _detect_data_config()

    def fake_load_data_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return dict(data_config)

    monkeypatch.setattr("libreyolo.data.load_data_config", fake_load_data_config)
    monkeypatch.setattr(
        "libreyolo.models.rfdetr.model.load_data_config",
        fake_load_data_config,
    )

    model = case.model_factory()
    train_kwargs = dict(case.train_kwargs or {})

    with pytest.raises(_StopTraining):
        model.train(
            "dummy.yaml",
            epochs=1,
            batch=1,
            device="cpu",
            callbacks=callback,
            loggers="mlflow",
            **train_kwargs,
        )

    assert captured["kwargs"]["callbacks"] is callback
    assert captured["kwargs"]["loggers"] == "mlflow"
