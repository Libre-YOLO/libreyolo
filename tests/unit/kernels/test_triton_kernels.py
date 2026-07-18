"""GPU parity tests for Triton quantization kernels.

Benchmarks deliberately stay out of pytest.  Each landed implementation is
added to ``LANDED_KERNELS`` and checked against every canonical shape.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch

from .harness import check_parity, load_shapes


pytestmark = pytest.mark.unit

HAS_TRITON_CUDA = (
    importlib.util.find_spec("triton") is not None and torch.cuda.is_available()
)

# (registry slot, module, public implementation)
LANDED_KERNELS: tuple[tuple[str, str, str], ...] = ()


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
