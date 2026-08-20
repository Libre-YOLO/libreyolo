"""Tests for declared dependency floors."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import pytest

pytestmark = pytest.mark.unit


def test_rfdetr_extra_uses_native_dependencies():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["optional-dependencies"]["rfdetr"]
    assert "transformers>=5.1.0" in deps
    assert "scipy>=1.7.0" not in deps
    assert all(not dep.startswith("rfdetr") for dep in deps)


def test_all_extra_includes_triton_serving_dependencies():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    all_deps = pyproject["project"]["optional-dependencies"]["all"]
    assert "libreyolo[triton]" in all_deps


def test_core_dependencies_include_import_chain_requirements():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    assert "Pillow>=9.1.0" in deps
    assert "scipy>=1.7.0" in deps
    assert "torchvision>=0.19.0" in deps


def test_torch_floor_supports_amp_grad_scaler():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    assert "torch>=2.4.0" in deps


def test_paddle_extra_pins_the_measured_converter_stack():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["optional-dependencies"]["paddle"]
    assert "libreyolo[onnx]" in deps
    assert "onnx>=1.14.0,<1.18" in deps
    assert any(dep.startswith("paddlepaddle==2.6.2") for dep in deps)
    assert any(dep.startswith("x2paddle==1.6.0") for dep in deps)
    assert any(dep.startswith("six>=1.16.0") for dep in deps)


def test_openvocab_extra_covers_clip_tokenizer_runtime():
    """OV-DEIM always embeds prompts with the vendored CLIP BPE tokenizer.

    That tokenizer imports ftfy and regex at predict time, so a clean
    ``pip install libreyolo[openvocab]`` must ship them or the first
    LibreOVDEIM prediction raises ImportError (v1.4.0 release blocker).
    """
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["optional-dependencies"]["openvocab"]
    names = {dep.split(">=")[0].split("==")[0].strip() for dep in deps}
    assert "ftfy" in names
    assert "regex" in names


def test_hf_extra_supports_bounded_vlm_download_preflight():
    """Hub 1.0 introduced ``hf_hub_download(dry_run=True)``."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["optional-dependencies"]["hf"]
    assert "huggingface_hub>=1.0.0" in deps
