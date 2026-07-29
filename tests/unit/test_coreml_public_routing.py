"""Public model routing for task-specific Core ML export paths."""

from importlib import import_module
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("family", ["l2cs", "dinov2"])
def test_custom_model_export_routes_coreml_to_shared_export(monkeypatch, family):
    if family == "l2cs":
        from libreyolo.models.l2cs.model import LibreL2CS as model_cls
    else:
        from libreyolo.models.dinov2.model import LibreDINOv2 as model_cls

    model = object.__new__(model_cls)
    model.task = "gaze" if family == "l2cs" else "classify"
    shared_export = MagicMock(return_value="model.mlpackage")
    monkeypatch.setattr(
        "libreyolo.models.base.model.BaseModel.export",
        shared_export,
    )

    assert model.export(format="coreml", imgsz=224) == "model.mlpackage"
    shared_export.assert_called_once_with(
        format="coreml",
        imgsz=224,
        **({"opset": 17} if family == "dinov2" else {}),
    )


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("libreyolo.models.sam.model", "LibreSAM1"),
        ("libreyolo.models.sam.sam2", "LibreSAM2"),
        ("libreyolo.models.sam.edgetam", "LibreEdgeTAM"),
        ("libreyolo.models.sam.sam3", "LibreSAM3"),
        ("libreyolo.models.mobilesam.model", "LibreMobileSAM"),
    ],
)
def test_promptable_sam_export_routes_coreml_to_shared_exporter(
    monkeypatch,
    module_name,
    class_name,
):
    from libreyolo.export.exporter import BaseExporter

    model_cls = getattr(import_module(module_name), class_name)
    model = object.__new__(model_cls)
    exporter = MagicMock(return_value="sam.mlpackage")
    create = MagicMock(return_value=exporter)
    monkeypatch.setattr(BaseExporter, "create", create)

    assert (
        model.export(
            format=" CoreML ",
            output_path="sam.mlpackage",
            prompt_max_points=32,
        )
        == "sam.mlpackage"
    )
    create.assert_called_once_with("coreml", model)
    exporter.assert_called_once_with(
        output_path="sam.mlpackage",
        prompt_max_points=32,
    )


def test_libresam_factory_routes_mlpackage_to_coreml_backend(monkeypatch, tmp_path):
    from libreyolo.backends import coreml as coreml_module
    from libreyolo.models.sam.model import LibreSAM

    backend = object()
    backend_cls = MagicMock(return_value=backend)
    monkeypatch.setattr(coreml_module, "CoreMLBackend", backend_cls)
    package = Path(tmp_path, "Interactive.MLPACKAGE")

    assert LibreSAM(package, device="cpu") is backend
    backend_cls.assert_called_once_with(str(package), device="cpu")


def test_promptable_sam_non_coreml_export_fails_before_factory(monkeypatch):
    from libreyolo.export.exporter import BaseExporter
    from libreyolo.models.sam.model import LibreSAM1

    model = object.__new__(LibreSAM1)
    create = MagicMock()
    monkeypatch.setattr(BaseExporter, "create", create)

    with pytest.raises(NotImplementedError, match="format='coreml'"):
        model.export(format="onnx")
    create.assert_not_called()
