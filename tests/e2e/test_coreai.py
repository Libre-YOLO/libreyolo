"""Core AI artifact parity on real Apple hardware.

The test compares the public ``.aimodel`` export with the exact fixed-canvas
PyTorch graph prepared by the exporter. Two probes establish both numeric
agreement and meaningful input sensitivity. Multi-output DETR tensors are
compared in the graph's declared output order. RT-DETRv2 alone permits a
single whole-row assignment shared by every output tensor because its query
rows are an unordered set; outputs are never sorted or matched independently,
so the gate cannot hide a broken box/logit association.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

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
MIN_REL_SENSITIVITY = 1e-6
CASES = [
    ("LibreYOLO9t.pt", "yolo9", 640),
    ("LibreDFINEn.pt", "dfine", 640),
    ("LibreRFDETRn.pt", "rfdetr", 384),
    ("LibreRFDETRn-seg.pt", "rfdetr", 312),
    # Pose currently has one published checkpoint; x is therefore the
    # smallest trained variant available for a real runtime-parity probe.
    ("LibreRFDETRx-pose.pt", "rfdetr", 576),
    ("LibreYOLOXn.pt", "yolox", 416),
    ("LibreDEIMn.pt", "deim", 640),
    ("LibreDEIMv2atto.pt", "deimv2", 320),
    ("LibreECs.pt", "ec", 640),
    ("LibrePICODETs.pt", "picodet", 320),
    ("LibreRTDETRr18.pt", "rtdetr", 640),
    ("LibreRTDETRv2r18.pt", "rtdetrv2", 640),
    ("LibreRTDETRv4s.pt", "rtdetrv4", 640),
    ("LibreRTMDett.pt", "rtmdet", 640),
    ("LibreYOLO9E2Et.pt", "yolo9_e2e", 640),
    ("LibreYOLO1b.pt", "yolo1", 448),
    ("LibreYOLO2b.pt", "yolo2", 608),
    ("LibreYOLO3b.pt", "yolo3", 416),
    ("LibreYOLO4b.pt", "yolo4", 608),
    ("LibreYOLO7b.pt", "yolo7", 640),
    ("LibrePIDNets-sem.pt", "pidnet", 1024),
    ("LibreLingBotVisions-sem.pt", "lingbotvision", 512),
    ("LibreResNet18-cls.pt", "resnet", 224),
    ("LibreMobileNetV4s-cls.pt", "mobilenetv4", 224),
    ("LibreEfficientNetV2b0-cls.pt", "efficientnetv2", 224),
    ("LibreConvNeXtt-cls.pt", "convnext", 224),
    ("LibreDepthAnythingV2s-depth.pt", "depth_anything", 518),
    ("LibreZipDepthb-depth.pt", "zipdepth", 384),
    ("LibreRealESRGANx4t-restore.pt", "realesrgan", 64),
    ("LibreNAFNetl-restore-sidd.pt", "nafnet", 256),
]
FROZEN_CLASS_CASES = [
    ("LibreCLIPb32-cls.pt", "clip", 224),
    ("LibreSigLIP2b16-cls.pt", "siglip2", 256),
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
        _wrap_coreai_contract,
    )
    from libreyolo.export.exporter import CoreAIExporter

    exporter = CoreAIExporter(model)
    with exporter._model_context(
        torch.device("cpu"),
        False,
        False,
        1,
        (imgsz, imgsz),
    ) as (nn_model, _):
        wrapped = _wrap_coreai_contract(nn_model, family)
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


def _artifact_outputs(artifact, tensors, output_names=None):
    loaded = _run(coreai_runtime.AIModel.load(artifact))
    function = _run(loaded.load_function(next(iter(loaded.function_names))))
    input_name = _input_name(function)
    names = list(output_names) if output_names is not None else None
    outputs = []
    for tensor in tensors:
        result = _run(
            function(
                {input_name: coreai_runtime.NDArray(tensor.detach().cpu().numpy())}
            )
        )
        assert isinstance(result, dict), "Core AI output contract must be named"
        if names is None:
            names = list(result)
        assert set(names) == set(result), (
            f"runtime names {sorted(result)} != graph names {sorted(names)}"
        )
        outputs.append(
            [
                np.asarray(
                    result[name].numpy()
                    if hasattr(result[name], "numpy")
                    else result[name]
                )
                for name in names
            ]
        )
    return names, outputs


def _assert_parity(output_names, ref1, ref2, got1, got2):
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
        print(
            f"out[{index}] ({output_names[index]}): error={error:.9e}, "
            f"sensitivity={sensitivity:.9e}, margin={margin:.3f}x"
        )
        assert error <= REL_TOL, (
            f"out[{index}] ({output_names[index]}) relative error "
            f"{error:.3e} exceeds {REL_TOL:.0e}"
        )
        assert sensitivity >= MIN_REL_SENSITIVITY, (
            f"out[{index}] ({output_names[index]}) relative input sensitivity "
            f"{sensitivity:.3e} is below {MIN_REL_SENSITIVITY:.0e}"
        )
        assert margin >= MIN_SENSITIVITY_MARGIN, (
            f"out[{index}] parity margin {margin:.1f}x is below "
            f"{MIN_SENSITIVITY_MARGIN:.0f}x "
            f"(error={error:.3e}, sensitivity={sensitivity:.3e})"
        )


def _align_unordered_queries(reference, candidate):
    """Apply one whole-row assignment shared by every output tensor."""
    assert len(reference) >= 2
    assert all(array.ndim == 3 for array in reference + candidate)
    assert len({array.shape[1] for array in reference + candidate}) == 1

    ref_rows = []
    got_rows = []
    for expected, actual in zip(reference, candidate):
        scale = max(float(np.abs(expected).max()), 1e-12)
        ref_rows.append(expected[0].reshape(expected.shape[1], -1) / scale)
        got_rows.append(actual[0].reshape(actual.shape[1], -1) / scale)
    ref_key = np.concatenate(ref_rows, axis=1)
    got_key = np.concatenate(got_rows, axis=1)
    cost = np.max(
        np.abs(ref_key[:, None, :] - got_key[None, :, :]),
        axis=2,
    )
    rows, columns = linear_sum_assignment(cost)
    order = columns[np.argsort(rows)]
    return [array[:, order, ...] for array in candidate]


def _assert_model_artifact_parity(model, family, imgsz, tmp_path):
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

    _, (got1, got2) = _artifact_outputs(artifact, (x1, x2), output_names)
    if family == "rtdetrv2":
        # Core AI may choose a different order for equal/near-equal top-k
        # encoder queries. DETR query rows are an unordered set, but boxes and
        # logits must remain paired, so derive exactly one joint assignment
        # from every output and apply it wholesale to every tensor.
        got1 = _align_unordered_queries(ref1, got1)
        got2 = _align_unordered_queries(ref2, got2)
    _assert_parity(output_names, ref1, ref2, got1, got2)


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
    _assert_model_artifact_parity(model, family, imgsz, tmp_path)


def test_coreai_fomo_synthetic_trained_parity(tmp_path):
    from libreyolo import LibreFOMO

    torch.manual_seed(20260728)
    model = LibreFOMO(None, size="s", nb_classes=2, device="cpu")
    network = model.model.train()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.02, momentum=0.9)

    for step in range(8):
        images = torch.rand(4, 3, 96, 96)
        logits = network(images)
        targets = torch.zeros(
            logits.shape[0],
            *logits.shape[-2:],
            dtype=torch.long,
        )
        targets[:, 2 + step % 4, 3 + step % 5] = 1
        targets[:, 7 - step % 4, 8 - step % 5] = 2
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    network.eval()
    _assert_model_artifact_parity(model, "fomo", 96, tmp_path)


def test_coreai_yolonas_synthetic_trained_parity(tmp_path):
    from libreyolo import LibreYOLONAS
    from libreyolo.models.yolonas.loss import PPYoloELoss

    torch.manual_seed(20260728)
    model = LibreYOLONAS(None, size="s", nb_classes=2, device="cpu")
    network = model.model.train()
    loss_fn = PPYoloELoss(num_classes=2)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

    for step in range(12):
        images = torch.rand(2, 3, 96, 96)
        targets = torch.zeros(2, 10, 5)
        targets[0, 0] = torch.tensor([float(step % 2), 36.0 + step, 42.0, 24.0, 30.0])
        targets[1, 0] = torch.tensor([float((step + 1) % 2), 64.0, 52.0, 20.0, 26.0])
        outputs = network(images)
        loss, _ = loss_fn(outputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    network.eval()
    # A randomly initialized detector's decoded boxes are dominated by its
    # fixed anchor grid. Scaling only the owned synthetic regression heads
    # makes the conversion gate sensitive to both outputs without asserting
    # anything about detection accuracy.
    with torch.no_grad():
        for head in (
            network.heads.head1,
            network.heads.head2,
            network.heads.head3,
        ):
            head.reg_pred.weight.mul_(20.0)

    _assert_model_artifact_parity(model, "yolonas", 96, tmp_path)


def test_coreai_yolo9_p2_permissive_transfer_parity(tmp_path):
    from libreyolo import LibreYOLO9P2
    from libreyolo.utils.download import download_weights

    torch.manual_seed(20260728)
    model = LibreYOLO9P2(None, size="t", device="cpu")
    weights_path = Path(model._resolve_weights_path("LibreYOLO9t.pt"))
    if not weights_path.exists():
        download_weights(str(weights_path), model.size)
    with weights_path.open("rb") as weights_file:
        digest = hashlib.file_digest(weights_file, "sha256").hexdigest()
    assert digest == "b4d7e93f9e0393830fb42e6135c0e3464b2673b05e5ecf4b7f2374ec18e39eb2"
    model._load_transfer_weights(weights_path)
    _assert_model_artifact_parity(model, "yolo9_p2", 640, tmp_path)


@pytest.mark.parametrize("weights,family,imgsz", FROZEN_CLASS_CASES)
def test_coreai_frozen_classifier_matches_trained_model(
    weights, family, imgsz, tmp_path
):
    from libreyolo import LibreYOLO

    model = LibreYOLO(weights, device="cpu")
    model.set_classes(
        ["cat", "dog", "car"],
        templates=["a photo of a {}."],
    )
    artifact = model.export(
        format="coreai",
        imgsz=imgsz,
        output_path=str(tmp_path / family),
    )

    if family == "clip":
        from libreyolo.models.clip.export import _FrozenCLIPClassifier

        scale = float(model.model.logit_scale.exp().detach().cpu())
        weight = (scale * model._text_embeds).detach().cpu()
        frozen = _FrozenCLIPClassifier(model.model.visual, weight).eval()
    else:
        from libreyolo.models.siglip2.export import _FrozenSigLIP2Classifier

        scale = float(model.model.logit_scale.exp().detach().cpu())
        weight = (scale * model._text_embeds).detach().cpu()
        bias = model.model.logit_bias.detach().to("cpu", torch.float32).reshape(())
        frozen = _FrozenSigLIP2Classifier(
            model.model.vision_model,
            weight,
            bias,
        ).eval()

    generator = torch.Generator().manual_seed(20260728)
    x1 = torch.rand(1, 3, imgsz, imgsz, generator=generator)
    x2 = torch.rand(1, 3, imgsz, imgsz, generator=generator)
    with torch.no_grad():
        ref1 = _flatten(frozen(x1))
        ref2 = _flatten(frozen(x2))
    output_names, (got1, got2) = _artifact_outputs(artifact, (x1, x2))
    assert len(output_names) == 1
    _assert_parity(output_names, ref1, ref2, got1, got2)
