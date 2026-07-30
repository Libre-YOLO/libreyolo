"""Core ML regression tests for DEIM-family distribution projection."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

_CASES = (
    pytest.param("deim", "n", 640, id="deim-n"),
    pytest.param("deimv2", "atto", 320, id="deimv2-atto"),
)


def _prepared_wrapper(family: str, size: str):
    torch.manual_seed(20260729)
    if family == "deim":
        from libreyolo import LibreDEIM
        from libreyolo.models.deim.nn import DEIMExportWrapper

        model = LibreDEIM(None, size=size, device="cpu")
        return DEIMExportWrapper(model.model).eval()

    from libreyolo import LibreDEIMv2
    from libreyolo.models.deimv2.nn import DEIMv2ExportWrapper

    model = LibreDEIMv2(None, size=size, device="cpu")
    return DEIMv2ExportWrapper(model.model).eval()


def _trace_with_two_probes(family: str, size: str, canvas: int):
    wrapper = _prepared_wrapper(family, size)
    count = 3 * canvas * canvas
    first = torch.linspace(0.0, 1.0, count).reshape(1, 3, canvas, canvas)
    second = torch.linspace(1.0, 0.0, count).reshape(1, 3, canvas, canvas)
    with torch.inference_mode():
        references = (wrapper(first), wrapper(second))
        traced = torch.jit.trace(
            wrapper,
            first,
            check_trace=True,
            check_inputs=[(second,)],
        )
        actuals = (traced(first), traced(second))
    return traced, references, actuals


@pytest.mark.parametrize(("family", "size", "canvas"), _CASES)
def test_deim_integral_projection_has_exact_eager_trace_parity(
    family,
    size,
    canvas,
):
    traced, references, actuals = _trace_with_two_probes(family, size, canvas)

    for reference_outputs, actual_outputs in zip(references, actuals):
        assert len(reference_outputs) == len(actual_outputs) == 2
        for reference, actual in zip(reference_outputs, actual_outputs):
            torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)

    integral_weights = [
        node.inputsAt(1).type().sizes()
        for node in traced.inlined_graph.nodes()
        if node.kind() == "aten::linear" and "decoder.integral" in node.scopeName()
    ]
    assert integral_weights
    assert integral_weights == [[1, 33]] * len(integral_weights)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="coremltools conversion is unsupported on Windows",
)
@pytest.mark.parametrize(("family", "size", "canvas"), _CASES)
def test_coremltools9_converts_deim_integral_projection(
    family,
    size,
    canvas,
):
    ct = pytest.importorskip("coremltools")
    if str(ct.__version__).split(".", 1)[0] != "9":
        pytest.skip("This conversion regression is pinned to coremltools 9.x.")

    traced, _references, _actuals = _trace_with_two_probes(family, size, canvas)
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="images",
                shape=(1, 3, canvas, canvas),
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(name="pred_logits"),
            ct.TensorType(name="pred_boxes"),
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True,
    )

    assert converted.get_spec().WhichOneof("Type") == "mlProgram"
