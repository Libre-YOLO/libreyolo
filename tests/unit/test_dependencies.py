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


def test_coreml_extra_matches_parity_gated_converter_frontends():
    """Keep CT9 below NumPy's 2.4 removal of one-element scalar coercion."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["optional-dependencies"]["coreml"]
    assert "numpy<2.4" in deps
    assert "coremltools>=9.0,<10" in deps
    assert "transformers==5.12.1" in deps

    vlm_deps = pyproject["project"]["optional-dependencies"]["coreml-vlm"]
    assert "numpy<2.4" in vlm_deps
