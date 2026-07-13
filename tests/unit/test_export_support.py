"""Regression tests for the centralized export support matrix."""

from __future__ import annotations

import json
import importlib
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from libreyolo.export.exporter import NcnnExporter, OnnxExporter
from libreyolo.export.support import (
    EXPORT_FORMATS,
    SUPPORT,
    get_export_capabilities,
    get_support,
)
from libreyolo.models.inventory import collect_model_inventory
from libreyolo.tasks import TASKS


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_SNAPSHOT = REPO_ROOT / "reports" / "export_inventory.json"


def _wrapper(family: str, task: str = "detect") -> MagicMock:
    model = MagicMock()
    model._get_model_name.return_value = family
    model.task = task
    return model


def test_matrix_keys_use_canonical_registry_values():
    inventory = json.loads(INVENTORY_SNAPSHOT.read_text(encoding="utf-8"))
    families = set(inventory)
    for family, task, fmt in SUPPORT:
        assert family in families
        assert task in TASKS
        assert fmt in EXPORT_FORMATS


def test_matrix_rejects_duplicate_explicit_keys():
    from libreyolo.export import support

    key = ("yolo9", "detect", "onnx")
    original = support.SUPPORT[key]
    with pytest.raises(ValueError, match="Duplicate export support entries"):
        support._add("validated", (key[0],), (key[1],), (key[2],))
    assert support.SUPPORT[key] is original


@pytest.mark.parametrize(
    "family",
    ["dfine", "deim", "deimv2", "rtdetr", "rtdetrv2", "rtdetrv4", "rfdetr", "ec"],
)
def test_ncnn_detr_families_fail_in_preflight(family):
    exporter = NcnnExporter(_wrapper(family))
    with pytest.raises(NotImplementedError, match="NCNN"):
        exporter._preflight(half=False, int8=False, data=None)


def test_experimental_export_warns_in_preflight():
    exporter = OnnxExporter(_wrapper("deim"))
    with pytest.warns(RuntimeWarning, match="experimental"):
        exporter._preflight(half=False, int8=False, data=None)


def test_tflite_support_keys_use_canonical_tasks():
    from libreyolo.export.tflite import supported_tflite_exports

    assert all(task in TASKS for _, task in supported_tflite_exports())
    assert get_support("yolo3", "detect", "tflite").tier == "blocked"
    assert get_support("rfdetr", "detect", "tflite").tier == "experimental"
    assert get_support("rfdetr", "segment", "tflite").tier == "blocked"


def test_dinov2_classify_routes_to_base_onnx_export(monkeypatch):
    from libreyolo.models.base.model import BaseModel
    from libreyolo.models.dinov2.model import LibreDINOv2

    model = object.__new__(LibreDINOv2)
    model.task = "classify"
    captured = {}

    def fake_export(self, format="onnx", **kwargs):
        captured.update(format=format, **kwargs)
        return "dinov2.onnx"

    monkeypatch.setattr(BaseModel, "export", fake_export)
    assert model.export("onnx", dynamic=False) == "dinov2.onnx"
    assert captured == {"format": "onnx", "opset": 17, "dynamic": False}


def test_dinov2_semantic_routes_to_shared_export(monkeypatch):
    from libreyolo.models.base.model import BaseModel
    from libreyolo.models.dinov2.model import LibreDINOv2

    model = object.__new__(LibreDINOv2)
    model.task = "semantic"
    captured = {}

    def fake_export(self, format="onnx", **kwargs):
        captured.update(format=format, **kwargs)
        return "dinov2-semantic.onnx"

    monkeypatch.setattr(BaseModel, "export", fake_export)
    assert model.export("onnx", dynamic=False) == "dinov2-semantic.onnx"
    assert captured == {"format": "onnx", "opset": 17, "dynamic": False}


def test_observed_cpu_toolchain_blocks_are_explicit():
    depth_ncnn = get_support("depth_anything", "depth", "ncnn")
    fomo_tflite = get_support("fomo", "point", "tflite")
    assert depth_ncnn.tier == "blocked" and "reshape" in depth_ncnn.reason
    assert fomo_tflite.tier == "blocked" and "depthwise" in fomo_tflite.reason


def test_fallback_reasons_describe_project_support_not_developer_environment():
    semantic = get_support("unwired_family", "semantic", "onnx")
    tensorrt = get_support("unwired_family", "detect", "tensorrt")
    eomt_segment = get_support("eomt", "segment", "onnx")

    assert "not wired" in semantic.reason
    assert "project has not yet recorded" in tensorrt.reason
    assert "this environment" not in tensorrt.reason
    assert "instance and panoptic" in eomt_segment.reason
    assert "semantic" not in eomt_segment.reason


def test_readme_generator_reports_missing_compatibility_anchor():
    from tools.gen_compat_table import _replace_readme

    with pytest.raises(ValueError, match="neither export-support markers"):
        _replace_readme("# README without compatibility", "generated")


def test_compat_table_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    from tools import gen_compat_table

    monkeypatch.chdir(tmp_path)
    assert gen_compat_table.INVENTORY_PATH.exists()
    rows, _ = gen_compat_table._rows()
    assert rows


def test_dump_inventory_refuses_partial_overwrite(tmp_path):
    from tools.dump_model_inventory import write_inventory

    output = tmp_path / "export_inventory.json"
    fake = {"zzz_fake_family": {"tasks": ["detect"]}}
    output.write_text(json.dumps(fake), encoding="utf-8")

    with pytest.raises(SystemExit, match="zzz_fake_family"):
        write_inventory(output)
    assert json.loads(output.read_text(encoding="utf-8")) == fake

    inventory = write_inventory(output, allow_family_removal=True)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert "zzz_fake_family" not in written
    assert written == inventory


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="the canonical inventory snapshot includes transformer-backed families",
)
def test_committed_inventory_matches_runtime_inventory():
    committed = json.loads(INVENTORY_SNAPSHOT.read_text(encoding="utf-8"))
    runtime = collect_model_inventory()
    # Dependency availability is intentionally local-environment state; every
    # declarative capability/publication field must still match the snapshot.
    for inventory in (committed, runtime):
        for metadata in inventory.values():
            metadata.pop("available", None)
    assert committed == runtime


def test_partial_exporters_are_custom_not_blocked():
    """A family that exports some formats and raises for the rest is custom.

    PicoSAM3 ships a validated ONNX export and raises for every other format.
    Reporting it as ``blocked`` would tell inventory consumers to reject an
    export the support matrix marks validated.
    """
    inventory = collect_model_inventory()
    assert inventory["picosam3"]["export_override"] == "custom"
    assert get_support("picosam3", "segment", "onnx").tier == "validated"

    for family, metadata in inventory.items():
        if metadata["export_override"] != "blocked":
            continue
        for task in metadata["tasks"]:
            for format in EXPORT_FORMATS:
                assert get_support(family, task, format).tier != "validated", (
                    f"{family}/{task}/{format} is validated in the support "
                    "matrix but the inventory reports export as blocked"
                )


def test_export_capabilities_are_family_and_task_aware():
    yolo9_onnx = get_export_capabilities("yolo9", "detect", "onnx")
    yolox_onnx = get_export_capabilities("yolox", "detect", "onnx")
    yolo9_torchscript = get_export_capabilities("yolo9", "detect", "torchscript")
    yolo9_ncnn = get_export_capabilities("yolo9", "detect", "ncnn")

    assert yolo9_onnx.supports_int8 is True
    assert yolox_onnx.supports_int8 is False
    assert yolo9_torchscript.supports_dynamic is False
    assert yolo9_ncnn.supports_dynamic is False
    assert yolo9_ncnn.supports_fp16 is False


@pytest.mark.parametrize(
    ("family", "task"),
    [
        ("depth_anything", "depth"),
        ("birefnet", "matte"),
        ("nafnet", "restore"),
        ("swinir", "restore"),
    ],
)
def test_fixed_runtime_contracts_do_not_advertise_dynamic_onnx(family, task):
    assert get_export_capabilities(family, task, "onnx").supports_dynamic is False


def test_realesrgan_keeps_dynamic_onnx_capability():
    assert get_export_capabilities("realesrgan", "restore", "onnx").supports_dynamic


@pytest.mark.parametrize(
    ("family", "task"),
    [
        ("clip", "classify"),
        ("siglip2", "classify"),
        ("picosam3", "segment"),
    ],
)
def test_custom_onnx_exporters_do_not_advertise_unsupported_fp16(family, task):
    assert get_export_capabilities(family, task, "onnx").supports_fp16 is False


def test_blocked_export_has_no_effective_option_capabilities():
    capabilities = get_export_capabilities("rfdetr", "detect", "ncnn")

    assert capabilities.support.tier == "blocked"
    assert capabilities.supports_dynamic is False
    assert capabilities.supports_fp16 is False
    assert capabilities.supports_int8 is False


def test_manifest_blocked_override_prevents_generic_fallback(monkeypatch):
    from types import SimpleNamespace

    from libreyolo.export import support

    manifest_entry = SimpleNamespace(
        tasks=(SimpleNamespace(task="detect"),), export_override="blocked"
    )
    monkeypatch.setattr(support, "get_family_spec", lambda family: manifest_entry)

    entry = support.get_support("future_family", "detect", "onnx")

    assert entry.tier == "blocked"
    assert "public model manifest" in entry.reason


def test_concrete_exporters_are_publicly_importable():
    from libreyolo.export import (
        CoreMLExporter,
        NcnnExporter,
        OnnxExporter,
        OpenVINOExporter,
        TensorRTExporter,
        TFLiteExporter,
        TorchScriptExporter,
    )

    assert {
        exporter.format_name
        for exporter in (
            CoreMLExporter,
            NcnnExporter,
            OnnxExporter,
            OpenVINOExporter,
            TensorRTExporter,
            TFLiteExporter,
            TorchScriptExporter,
        )
    } == set(EXPORT_FORMATS)


def test_default_download_urls_keep_task_repo_suffixes():
    from libreyolo.models.base.model import BaseModel
    from libreyolo.models.manifest import ARTIFACT_BY_FILENAME

    for metadata in collect_model_inventory().values():
        if not metadata["generic_cli"] or not metadata["available"]:
            continue
        module_name, class_name = metadata["class"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        if "get_download_url" in cls.__dict__:
            continue
        artifacts = [
            artifact
            for artifact in ARTIFACT_BY_FILENAME.values()
            if artifact.family == cls.FAMILY and artifact.download_kind == "hf"
        ]
        for artifact in artifacts:
            filename = artifact.canonical_filename
            url = BaseModel.get_download_url.__func__(cls, filename)
            assert url == artifact.download_url
            assert f"/{filename.removesuffix(cls.WEIGHT_EXT)}/resolve/main/" in url


def test_generated_export_docs_are_current():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "gen_compat_table.py"), "--check"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
