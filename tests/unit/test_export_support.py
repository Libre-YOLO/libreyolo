"""Regression tests for the centralized export support matrix."""

from __future__ import annotations

import importlib
import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from libreyolo.export.exporter import NcnnExporter, OnnxExporter
from libreyolo.export.support import (
    CHECKPOINT_GATES,
    EXPORT_FORMATS,
    SUPPORT,
    get_support,
)
from libreyolo.models.inventory import (
    OPTIONAL_MODELS,
    collect_model_inventory,
    iter_model_cases,
)
from libreyolo.tasks import TASKS, task_to_suffix

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


def test_public_lazy_model_classes_are_in_the_inventory_contract():
    """Keep public optional families from silently disappearing from reports."""
    tree = ast.parse((REPO_ROOT / "libreyolo" / "__init__.py").read_text("utf-8"))
    lazy_mapping = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "_lazy"
            for target in node.targets
        ):
            lazy_mapping = ast.literal_eval(node.value)
            break
    assert lazy_mapping is not None

    facade_classes = {
        "FaceGallery",
        "LibreOpenVocab",
        "LibreSAM",
        "LibreVLM",
    }
    public_lazy_models = {
        class_name
        for module_name, class_name in lazy_mapping.values()
        if module_name.startswith(".models.") and class_name not in facade_classes
    }
    inventory_classes = {
        metadata["class"].rsplit(".", 1)[-1]
        for metadata in json.loads(
            INVENTORY_SNAPSHOT.read_text(encoding="utf-8")
        ).values()
    }
    optional_classes = {class_name for _, class_name, _, _ in OPTIONAL_MODELS}

    assert public_lazy_models <= inventory_classes
    assert public_lazy_models - {"LibreDINOv2", "LibreRFDETR"} <= optional_classes


def test_canonical_cases_use_each_tasks_size_map():
    inventory = json.loads(INVENTORY_SNAPSHOT.read_text(encoding="utf-8"))
    cases = list(iter_model_cases(inventory))
    keys = [(family, task, size) for family, task, size, _ in cases]
    assert len(keys) == len(set(keys))

    for family, metadata in inventory.items():
        for task in metadata["tasks"]:
            expected = metadata["task_sizes"].get(task) or metadata["default_imgsz"]
            actual = {
                size: imgsz
                for case_family, case_task, size, imgsz in cases
                if (case_family, case_task) == (family, task)
            }
            assert actual == expected


def test_every_inventory_row_has_an_explicit_coreml_disposition():
    inventory = json.loads(INVENTORY_SNAPSHOT.read_text(encoding="utf-8"))
    expected = {
        (family, task, "coreml")
        for family, metadata in inventory.items()
        for task in metadata["tasks"]
    }
    assert expected <= SUPPORT.keys()


def test_checkpoint_gates_are_canonical_and_independent_of_technical_tier():
    inventory = json.loads(INVENTORY_SNAPSHOT.read_text(encoding="utf-8"))
    for family, task in CHECKPOINT_GATES:
        assert family in inventory
        assert task in inventory[family]["tasks"]

    assert get_support("sensenovavision", "detect", "coreml").tier == "blocked"
    assert CHECKPOINT_GATES[("sensenovavision", "detect")]
    assert get_support("depth_anything", "depth", "onnx").tier == "validated"
    assert CHECKPOINT_GATES[("depth_anything", "depth")]


def test_coreml_matrix_is_explicit_and_never_overclaims_validation():
    from libreyolo.export.coreml import supported_coreml_exports

    inventory = json.loads(INVENTORY_SNAPSHOT.read_text(encoding="utf-8"))
    entries = {
        (family, task): get_support(family, task, "coreml")
        for family, metadata in inventory.items()
        for task in metadata["tasks"]
    }
    assert entries
    assert all(entry.tier in {"experimental", "blocked"} for entry in entries.values())
    assert not any(entry.tier == "validated" for entry in entries.values())
    dedicated_component_routes = {
        ("facerec", "embed"),
        ("smolvlm2", "detect"),
    }
    assert {
        key for key, entry in entries.items() if entry.tier == "experimental"
    } == supported_coreml_exports() | (dedicated_component_routes & entries.keys())
    assert not (dedicated_component_routes & supported_coreml_exports())

    deimv2 = get_support("deimv2", "detect", "coreml")
    assert deimv2.tier == "experimental"
    assert deimv2.constraint
    assert all(size in deimv2.constraint for size in ("atto", "femto", "pico", "n"))
    assert all(size in deimv2.constraint for size in ("s", "m", "l", "x"))


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


def test_coreai_validated_tier_has_hardware_parity_coverage():
    validated = {
        (family, task)
        for (family, task, fmt), entry in SUPPORT.items()
        if fmt == "coreai" and entry.tier == "validated"
    }
    assert validated == {
        ("clip", "classify"),
        ("convnext", "classify"),
        ("deim", "detect"),
        ("deimv2", "detect"),
        ("depth_anything", "depth"),
        ("dfine", "detect"),
        ("ec", "detect"),
        ("efficientnetv2", "classify"),
        ("fomo", "point"),
        ("lingbotvision", "semantic"),
        ("mobilenetv4", "classify"),
        ("nafnet", "restore"),
        ("picodet", "detect"),
        ("pidnet", "semantic"),
        ("realesrgan", "restore"),
        ("resnet", "classify"),
        ("rfdetr", "detect"),
        ("rtdetr", "detect"),
        ("rtdetrv2", "detect"),
        ("rtdetrv4", "detect"),
        ("rtmdet", "detect"),
        ("siglip2", "classify"),
        ("yolo1", "detect"),
        ("yolo2", "detect"),
        ("yolo3", "detect"),
        ("yolo4", "detect"),
        ("yolo7", "detect"),
        ("yolo9", "detect"),
        ("yolo9_e2e", "detect"),
        ("yolo9_p2", "detect"),
        ("yolonas", "detect"),
        ("yolox", "detect"),
        ("zipdepth", "depth"),
    }


def test_fallback_reasons_describe_project_support_not_developer_environment():
    semantic = get_support("unwired_family", "semantic", "onnx")
    tensorrt = get_support("unwired_family", "detect", "tensorrt")
    eomt_segment = get_support("eomt", "segment", "onnx")

    assert "not wired" in semantic.reason
    assert "project has not yet recorded" in tensorrt.reason
    assert "this environment" not in tensorrt.reason
    assert "instance and panoptic" in eomt_segment.reason
    assert "semantic" not in eomt_segment.reason


def test_compat_table_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    from tools import gen_compat_table

    monkeypatch.chdir(tmp_path)
    assert gen_compat_table.INVENTORY_PATH.exists()
    rows, _, _ = gen_compat_table._rows()
    assert rows
    # The full matrix lives in docs/export_support.md; the README is curated.
    assert gen_compat_table.render_docs().startswith("# Export support")


def test_compat_table_contains_every_inventory_row_once():
    from tools import gen_compat_table

    inventory = json.loads(INVENTORY_SNAPSHOT.read_text(encoding="utf-8"))
    expected = [
        (family, task)
        for family, metadata in inventory.items()
        for task in metadata["tasks"]
    ]
    rows, _, _ = gen_compat_table._rows()
    actual = []
    for row in rows[2:]:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        actual.append((cells[0], cells[1]))
    assert actual == expected


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


def test_dump_inventory_refuses_partial_task_size_overwrite(tmp_path, monkeypatch):
    from libreyolo.models import inventory as inventory_module
    from tools.dump_model_inventory import write_inventory

    complete = collect_model_inventory()
    partial = copy.deepcopy(complete)
    family, task = next(
        (family, task)
        for family, metadata in partial.items()
        for task, sizes in metadata["task_sizes"].items()
        if len(sizes) > 1
    )
    removed_size = next(iter(partial[family]["task_sizes"][task]))
    del partial[family]["task_sizes"][task][removed_size]

    output = tmp_path / "export_inventory.json"
    output.write_text(json.dumps(complete), encoding="utf-8")
    monkeypatch.setattr(inventory_module, "collect_model_inventory", lambda: partial)

    with pytest.raises(SystemExit, match=rf"{family}/{task}/{removed_size}"):
        write_inventory(output)
    assert json.loads(output.read_text(encoding="utf-8")) == complete


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="the canonical inventory snapshot includes transformer-backed families",
)
def test_committed_inventory_matches_runtime_inventory():
    committed = json.loads(INVENTORY_SNAPSHOT.read_text(encoding="utf-8"))
    assert committed == collect_model_inventory()


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


def test_default_download_urls_keep_task_repo_suffixes():
    from libreyolo.models.base.model import BaseModel

    for metadata in collect_model_inventory().values():
        module_name, class_name = metadata["class"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        if "get_download_url" in cls.__dict__:
            continue
        for task in metadata["tasks"]:
            sizes = metadata["task_sizes"].get(task) or metadata["default_imgsz"]
            if not sizes or not cls.FILENAME_PREFIX:
                continue
            size = next(iter(sizes))
            suffix = task_to_suffix(task)
            filename = f"{cls.FILENAME_PREFIX}{size}"
            if suffix:
                filename += f"-{suffix}"
            filename += cls.WEIGHT_EXT
            url = BaseModel.get_download_url.__func__(cls, filename)
            assert url is not None
            expected_repo = f"/{cls.FILENAME_PREFIX}{size}"
            if suffix:
                expected_repo += f"-{suffix}"
            assert expected_repo + "/resolve/main/" in url


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


def test_generated_docs_expose_validated_constraints():
    from tools.gen_compat_table import render_docs

    docs = render_docs()
    assert "## Validated constraints" in docs
    assert "`yolonas` / `detect` / `coreai`" in docs
    assert "raw-image preprocessing" in docs
    assert "## Experimental constraints" in docs
    assert "`deimv2` / `detect` / `coreml`" in docs
    assert "## Checkpoint and artifact gates" in docs
    assert "`sensenovavision` / `detect`" in docs
