"""GPU parity tests for Triton quantization kernels.

Benchmarks deliberately stay out of pytest.  Each landed implementation is
added to ``LANDED_KERNELS`` and checked against every canonical shape.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch

from libreyolo.quant import kernels

from .harness import check_parity, load_shapes


pytestmark = pytest.mark.unit

HAS_TRITON_CUDA = (
    importlib.util.find_spec("triton") is not None and torch.cuda.is_available()
)

# (registry slot, module, public implementation)
LANDED_KERNELS: tuple[tuple[str, str, str], ...] = (
    (
        "fake_quant_nvfp4_weight",
        "libreyolo.quant.kernels.triton.nvfp4_weight",
        "fake_quant_nvfp4_weight",
    ),
    (
        "fake_quant_nvfp4_dynamic",
        "libreyolo.quant.kernels.triton.nvfp4_dynamic",
        "fake_quant_nvfp4_dynamic",
    ),
    (
        "fake_quant_int8_per_channel",
        "libreyolo.quant.kernels.triton.int8_per_channel",
        "fake_quant_int8_per_channel",
    ),
    (
        "fake_quant_int_grouped",
        "libreyolo.quant.kernels.triton.int_grouped",
        "fake_quant_int_grouped",
    ),
    (
        "fake_quant_fp8",
        "libreyolo.quant.kernels.triton.fp8",
        "fake_quant_fp8",
    ),
    (
        "fake_quant_mxfp4_weight",
        "libreyolo.quant.kernels.triton.mxfp4",
        "fake_quant_mxfp4_weight",
    ),
    (
        "fake_quant_mxfp4_dynamic",
        "libreyolo.quant.kernels.triton.mxfp4",
        "fake_quant_mxfp4_dynamic",
    ),
)


@pytest.mark.skipif(not HAS_TRITON_CUDA, reason="requires CUDA and Triton")
@pytest.mark.parametrize(
    ("op", "module_name", "implementation_name"),
    LANDED_KERNELS,
)
def test_triton_kernel_parity(
    op: str, module_name: str, implementation_name: str
) -> None:
    module = importlib.import_module(module_name)
    implementation = getattr(module, implementation_name)
    check_parity(op, implementation, load_shapes())


@pytest.mark.skipif(not HAS_TRITON_CUDA, reason="requires CUDA and Triton")
@pytest.mark.parametrize(
    ("op", "module_name", "implementation_name"),
    LANDED_KERNELS,
)
def test_triton_kernel_registration_and_escape_hatch(
    op: str,
    module_name: str,
    implementation_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(module_name)
    implementation = getattr(module, implementation_name)
    monkeypatch.delenv("LIBREYOLO_QUANT_KERNELS", raising=False)
    kernels.clear_cache()
    assert kernels.resolve(op) is implementation

    monkeypatch.setenv("LIBREYOLO_QUANT_KERNELS", "off")
    kernels.clear_cache()
    assert kernels.resolve(op) is not implementation
