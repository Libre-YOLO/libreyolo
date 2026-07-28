"""Trained-weight Core AI parity on real Apple hardware.

The test compares the public ``.aimodel`` export with the exact fixed-canvas
PyTorch graph prepared by the exporter. Two probes establish both numeric
agreement and meaningful input sensitivity. Multi-output DETR tensors are
compared in the graph's declared order without independently sorting or
matching rows, so the gate cannot hide a broken box/logit association.
"""

from __future__ import annotations

import asyncio
import inspect
import sys

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.general_nightly,
    pytest.mark.export_backend,
    pytest.mark.supported_backend,
]

coreai_runtime = pytest.importorskip(
    "coreai.runtime",
    reason="Core AI export requires the coreai toolchain (macOS only)",
)

if sys.platform != "darwin":  # pragma: no cover - platform gate
    pytest.skip("Core AI artifacts only run on macOS", allow_module_level=True)


REL_TOL = 3e-4
MIN_SENSITIVITY_MARGIN = 100.0
CASES = [
    ("LibreYOLO9t.pt", "yolo9", 640),
    ("LibreDFINEn.pt", "dfine", 640),
    ("LibreRFDETRn.pt", "rfdetr", 384),
]


def _run(value):
    return asyncio.run(value) if inspect.isawaitable(value) else value


def _flatten(value):
    if torch.is_tensor(value):
        return [value.detach().cpu().numpy()]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _flatten(item)]
    if isinstance(value, dict):
        return [tensor for key in sorted(value) for tensor in _flatten(value[key])]
    return []


def _input_name(function) -> str:
    desc = getattr(function, "desc", None)
    for attr in ("inputs", "input_names", "input_descriptors"):
        values = getattr(desc, attr, None) if desc is not None else None
        if not values:
            continue
        first = next(iter(values))
        return str(getattr(first, "name", first))
    return "x"


def _prepared_reference(model, family, imgsz, x1, x2):
    from coreai_torch import get_decomp_table

    from libreyolo.export.coreai import (
        _exported_output_names,
        _prepare_coreai_graph,
    )
    from libreyolo.export.coreml import _wrap_for_family
    from libreyolo.export.exporter import CoreAIExporter

    exporter = CoreAIExporter(model)
    with exporter._model_context(
        torch.device("cpu"),
        False,
        False,
        1,
        (imgsz, imgsz),
    ) as (nn_model, _):
        wrapped = _wrap_for_family(nn_model, family).eval()
        with _prepare_coreai_graph(wrapped, x1, family):
            # The eager prepared graph is the semantic reference. Running the
            # decomposed ExportedProgram as a module is not equivalent for
            # these DETR graphs: functionalization replays mutation-sensitive
            # buffers differently and can be nearly 1.0 relative off before
            # Core AI conversion is involved.
            with torch.no_grad():
                ref1 = _flatten(wrapped(x1))
                ref2 = _flatten(wrapped(x2))
            exported = torch.export.export(wrapped, args=(x1,))
            from torch._decomp import get_decompositions

            table = dict(get_decomp_table())
            table.update(get_decompositions([torch.ops.aten.grid_sampler_2d]))
            exported = exported.run_decompositions(table)
    return _exported_output_names(exported), ref1, ref2


@pytest.mark.parametrize("weights,family,imgsz", CASES)
def test_coreai_artifact_matches_prepared_trained_model(
    weights, family, imgsz, tmp_path
):
    from libreyolo import LibreYOLO

    if family == "rfdetr":
        pytest.importorskip(
            "transformers",
            reason="RF-DETR parity requires the rfdetr extra",
        )

    model = LibreYOLO(weights, device="cpu")
    artifact = model.export(
        format="coreai",
        imgsz=imgsz,
        output_path=str(tmp_path / family),
    )

    generator = torch.Generator().manual_seed(20260728)
    # The public export contract is canonical RGB float input in [0, 1].
    # Out-of-contract Gaussian/extreme probes exercise unspecified runtime
    # behaviour and previously produced false failures in DETR activations.
    x1 = torch.rand(1, 3, imgsz, imgsz, generator=generator)
    x2 = torch.rand(1, 3, imgsz, imgsz, generator=generator)
    output_names, ref1, ref2 = _prepared_reference(model, family, imgsz, x1, x2)
    assert len(output_names) == len(ref1)

    loaded = _run(coreai_runtime.AIModel.load(artifact))
    function = _run(loaded.load_function(next(iter(loaded.function_names))))
    input_name = _input_name(function)

    def call(tensor):
        result = _run(
            function(
                {input_name: coreai_runtime.NDArray(tensor.detach().cpu().numpy())}
            )
        )
        assert isinstance(result, dict), "Core AI output contract must be named"
        assert set(output_names) == set(result), (
            f"runtime names {sorted(result)} != graph names {sorted(output_names)}"
        )
        return [
            np.asarray(
                result[name].numpy() if hasattr(result[name], "numpy") else result[name]
            )
            for name in output_names
        ]

    got1 = call(x1)
    got2 = call(x2)
    assert [array.shape for array in got1] == [array.shape for array in ref1]

    for index, (expected1, expected2, actual1, actual2) in enumerate(
        zip(ref1, ref2, got1, got2)
    ):
        scale = max(
            float(np.abs(expected1).max()),
            float(np.abs(expected2).max()),
            1e-12,
        )
        error = (
            max(
                float(np.abs(actual1 - expected1).max()),
                float(np.abs(actual2 - expected2).max()),
            )
            / scale
        )
        sensitivity = float(np.abs(expected2 - expected1).max()) / scale
        margin = float("inf") if error == 0 else sensitivity / error
        assert error <= REL_TOL, (
            f"out[{index}] ({output_names[index]}) relative error "
            f"{error:.3e} exceeds {REL_TOL:.0e}"
        )
        assert margin >= MIN_SENSITIVITY_MARGIN, (
            f"out[{index}] parity margin {margin:.1f}x is below "
            f"{MIN_SENSITIVITY_MARGIN:.0f}x "
            f"(error={error:.3e}, sensitivity={sensitivity:.3e})"
        )
