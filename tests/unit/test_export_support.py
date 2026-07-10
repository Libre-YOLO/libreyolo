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
from libreyolo.export.support import EXPORT_FORMATS, SUPPORT, get_support
from libreyolo.models.inventory import collect_model_inventory
from libreyolo.tasks import TASKS
from libreyolo.tasks import task_to_suffix


pytestmark = pytest.mark.unit


def _wrapper(family: str, task: str = "detect") -> MagicMock:
    model = MagicMock()
    model._get_model_name.return_value = family
    model.task = task
    return model


def test_matrix_keys_use_canonical_registry_values():
    families = set(collect_model_inventory())
    for family, task, fmt in SUPPORT:
        assert family in families
        assert task in TASKS
        assert fmt in EXPORT_FORMATS


@pytest.mark.parametrize(
    "family",
    ["dfine", "deim", "deimv2", "rtdetr", "rtdetrv2", "rtdetrv4", "rfdetr", "ec"],
)
def test_ncnn_detr_families_fail_in_preflight(family):
    exporter = NcnnExporter(_wrapper(family))
    with pytest.raises(NotImplementedError, match="NCNN"):
        exporter._preflight(half=False, int8=False, data=None)


def test_experimental_export_warns_in_preflight():
    exporter = OnnxExporter(_wrapper("yolox"))
    with pytest.warns(RuntimeWarning, match="experimental"):
        exporter._preflight(half=False, int8=False, data=None)


def test_tflite_support_keys_use_canonical_tasks():
    from libreyolo.export.tflite import supported_tflite_exports

    assert all(task in TASKS for _, task in supported_tflite_exports())
    assert get_support("rfdetr", "segment", "tflite").tier == "experimental"


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
    assert captured == {"format": "onnx", "dynamic": False}


def test_dinov2_semantic_remains_blocked():
    from libreyolo.models.dinov2.model import LibreDINOv2

    model = object.__new__(LibreDINOv2)
    model.task = "semantic"
    with pytest.raises(NotImplementedError, match="dense-logits"):
        model.export("onnx")


def test_committed_inventory_matches_runtime_inventory():
    path = Path("reports/export_inventory_2026-07-10.json")
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert committed == collect_model_inventory()


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
    env["PYTHONPATH"] = str(Path.cwd())
    result = subprocess.run(
        [sys.executable, "tools/gen_compat_table.py", "--check"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
