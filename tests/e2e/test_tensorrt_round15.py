"""TensorRT Round 15 measurement for the last unmeasured detector cell.

The round also rechecked ten existing measured holds after explicitly
disabling TF32 during engine construction. None crossed the raw and public
``predict()`` parity gates, so that global precision experiment was reverted.
RT-DETRv4 uses deterministic LibreYOLO initialization and verifies export,
factory reload, two input-sensitive raw probes, metadata, and public detection
parity. The fixture validates conversion behavior, not task accuracy.
"""

from __future__ import annotations

import pytest

from .conftest import requires_tensorrt
from .test_tensorrt_round12 import TensorRTRound12Case, _run_tensorrt_case

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
    pytest.mark.tensorrt,
    pytest.mark.trt,
]


_RTDETRV4 = TensorRTRound12Case(
    "LibreRTDETRv4",
    "rtdetrv4",
    "s",
    "detect",
    256,
)


@requires_tensorrt
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Repeated builds change public top-k class membership or box geometry."
    ),
)
def test_tensorrt_round15_rtdetrv4_measured_experimental(tmp_path):
    _run_tensorrt_case(tmp_path, _RTDETRV4)
