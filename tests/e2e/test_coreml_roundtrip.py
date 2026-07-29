"""Core ML conversion and runtime parity on real Apple hardware.

This gate compares a saved ``.mlpackage`` with the exact fixed-canvas PyTorch
graph prepared by the exporter. It deliberately does more than confirm that
conversion finishes:

* two deterministic RGB byte probes must produce named outputs;
* every output must match the prepared graph numerically;
* every output must change meaningfully between probes; and
* output lookup uses the artifact's declared semantic names, never dict order.

The test proves graph fidelity. Public preprocessing/postprocessing parity is
covered separately because an accurate converted graph can still be paired
with an incorrect host-side image transform.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

pytestmark = [
    pytest.mark.coreml,
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
]

if sys.platform != "darwin":  # pragma: no cover - platform gate
    pytest.skip("Core ML artifacts only run on macOS", allow_module_level=True)

ct = pytest.importorskip(
    "coremltools",
    reason="Core ML parity requires the coremltools runtime",
)


REL_TOL = 3e-4
MIN_SENSITIVITY_MARGIN = 100.0
MIN_REL_SENSITIVITY = 1e-6

# These checkpoints already form the trained Core AI hardware set. Reusing
# them makes differences attributable to the Apple runtime/export path rather
# than to a different model fixture. Additional task-specific and component
# contracts have dedicated tests below as they become conversion-capable.
TRAINED_CASES = [
    ("LibreYOLO9t.pt", "yolo9", "detect", 640),
    ("LibreDFINEn.pt", "dfine", "detect", 640),
    ("LibreRFDETRn.pt", "rfdetr", "detect", 384),
    ("LibreYOLOXn.pt", "yolox", "detect", 416),
    ("LibreDEIMn.pt", "deim", "detect", 640),
    ("LibreDEIMv2atto.pt", "deimv2", "detect", 320),
    ("LibreECs.pt", "ec", "detect", 640),
    ("LibrePICODETs.pt", "picodet", "detect", 320),
    ("LibreRTDETRr18.pt", "rtdetr", "detect", 640),
    ("LibreRTDETRv2r18.pt", "rtdetrv2", "detect", 640),
    ("LibreRTDETRv4s.pt", "rtdetrv4", "detect", 640),
    ("LibreRTMDett.pt", "rtmdet", "detect", 640),
    ("LibreRTMDett-seg.pt", "rtmdet", "segment", 640),
    ("LibreYOLO9E2Et.pt", "yolo9_e2e", "detect", 640),
    ("LibreYOLO1b.pt", "yolo1", "detect", 448),
    ("LibreYOLO2b.pt", "yolo2", "detect", 608),
    ("LibreYOLO3b.pt", "yolo3", "detect", 416),
    ("LibreYOLO4b.pt", "yolo4", "detect", 608),
    ("LibreYOLO7b.pt", "yolo7", "detect", 640),
    ("LibrePIDNets-sem.pt", "pidnet", "semantic", 1024),
    ("LibreLingBotVisions-sem.pt", "lingbotvision", "semantic", 512),
    ("LibreEoMTl-sem.pt", "eomt", "semantic", 512),
    ("LibreEoMTl-seg.pt", "eomt", "segment", 640),
    ("LibreEoMTl-seg-1280.pt", "eomt", "segment", 1280),
    ("LibreEoMTs-panoptic.pt", "eomt", "panoptic", 640),
    ("LibreEoMTb-panoptic.pt", "eomt", "panoptic", 640),
    ("LibreEoMTl-panoptic.pt", "eomt", "panoptic", 640),
    ("LibreResNet18-cls.pt", "resnet", "classify", 224),
    ("LibreMobileNetV4s-cls.pt", "mobilenetv4", "classify", 224),
    ("LibreEfficientNetV2b0-cls.pt", "efficientnetv2", "classify", 224),
    ("LibreConvNeXtt-cls.pt", "convnext", "classify", 224),
    ("LibreDepthAnythingV2s-depth.pt", "depth_anything", "depth", 518),
    ("LibreZipDepthb-depth.pt", "zipdepth", "depth", 384),
    ("LibreRealESRGANx4t-restore.pt", "realesrgan", "restore", 64),
    ("LibreNAFNetl-restore-sidd.pt", "nafnet", "restore", 256),
    ("LibreDINOv2n.pt", "dinov2", "semantic", 518),
    ("LibreDINOv2n-cls.pt", "dinov2", "classify", 224),
    ("LibreDFINEn-seg.pt", "dfine", "segment", 640),
    ("LibreECs-pose.pt", "ec", "pose", 640),
    ("LibreECs-seg.pt", "ec", "segment", 640),
    ("LibreRFDETRn-seg.pt", "rfdetr", "segment", 312),
    ("LibreRFDETRx-pose.pt", "rfdetr", "pose", 576),
    ("LibreRFDETRn-obb.pt", "rfdetr", "obb", 384),
    ("LibreSegformerb0-sem.pt", "segformer", "semantic", 512),
    ("LibreSwinIRs-restore.pt", "swinir", "restore", 64),
    ("LibreSwinIRm-restore.pt", "swinir", "restore", 64),
    ("LibreSwinIRl-restore.pt", "swinir", "restore", 64),
]

FROZEN_CLASS_CASES = [
    ("LibreCLIPb32-cls.pt", "clip", 224),
    ("LibreSigLIP2b16-cls.pt", "siglip2", 256),
]


def _canonical_byte_probes(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two deterministic, chromatic, non-degenerate RGB byte images."""
    yy, xx = np.mgrid[:height, :width]
    first = np.stack(
        (
            (3 * xx + 5 * yy + 17) % 256,
            (11 * xx + 7 * yy + 53) % 256,
            (13 * xx + 19 * yy + 101) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    second = np.stack(
        (
            (23 * xx + 2 * yy + 211) % 256,
            (5 * xx + 29 * yy + 37) % 256,
            (17 * xx + 31 * yy + 149) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return first, second


def _as_canonical_tensor(image: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
    return value.float().div_(255.0)


def _prepared_reference(model, family: str, task: str, imgsz: int, probes):
    from libreyolo.export.coreai import _prepare_coreai_graph
    from libreyolo.export.coreml import (
        _CoreMLOutputAdapter,
        _flatten_tensor_outputs,
        _output_contract,
        _wrap_coreml_contract,
    )
    from libreyolo.export.exporter import CoreMLExporter

    exporter = CoreMLExporter(model)
    tensors = tuple(_as_canonical_tensor(probe) for probe in probes)
    with exporter._model_context(
        torch.device("cpu"),
        False,
        False,
        1,
        (imgsz, imgsz),
    ) as (nn_model, _):
        declared = _output_contract(family, task, nms=False)
        names = [item["name"] for item in declared]
        wrapped = _wrap_coreml_contract(nn_model, family, task)
        wrapped = _CoreMLOutputAdapter(wrapped, names).eval()
        with _prepare_coreai_graph(wrapped, tensors[0], family):
            with torch.no_grad():
                outputs = [
                    [
                        tensor.detach().cpu().numpy()
                        for tensor in _flatten_tensor_outputs(wrapped(probe))
                    ]
                    for probe in tensors
                ]
    assert all(len(values) == len(names) for values in outputs)
    return names, outputs


def _load_artifact(path: str | Path):
    model = ct.models.MLModel(
        str(path),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    metadata = dict(model.user_defined_metadata or {})
    assert metadata.get("libreyolo_producer") == "libreyolo"
    assert metadata.get("artifact_format") == "coreml"
    assert metadata.get("coreml_io_schema_version") == "2"
    contract = json.loads(metadata["coreml_io"])
    names = [item["name"] for item in contract["outputs"]]
    assert names == json.loads(metadata["coreml_output_names"])
    spec = model.get_spec()
    assert names == [feature.name for feature in spec.description.output]
    spec_inputs = list(spec.description.input)
    assert len(spec_inputs) == 1
    spec_kind = spec_inputs[0].type.WhichOneof("Type")
    assert {
        "imageType": "image",
        "multiArrayType": "tensor",
    }[spec_kind] == contract["input"]["kind"]
    return model, contract, names


def _artifact_outputs(model, contract, names, probes):
    input_name = contract["input"]["name"]
    input_contract = contract["input"]
    results = []
    for probe in probes:
        if input_contract["kind"] == "image":
            runtime_input = Image.fromarray(probe, mode="RGB")
        else:
            assert input_contract["kind"] == "tensor"
            assert input_contract["layout"] == "nchw"
            assert input_contract["color"] == "rgb"
            tensor = probe.astype(np.float32).transpose(2, 0, 1)[None]
            if input_contract["range"] == "0_1":
                tensor = tensor / 255.0
            else:
                assert input_contract["range"] == "0_255"
            runtime_input = np.ascontiguousarray(tensor)
        output = model.predict({input_name: runtime_input})
        assert isinstance(output, dict)
        assert set(output) == set(names)
        results.append([np.asarray(output[name]) for name in names])
    return results


def _align_unordered_queries(reference, candidate):
    """Apply one whole-query assignment shared by every output tensor."""
    assert len(reference) >= 2
    assert all(array.ndim == 3 for array in reference + candidate)
    assert len({array.shape[1] for array in reference + candidate}) == 1

    reference_rows = []
    candidate_rows = []
    for expected, actual in zip(reference, candidate):
        scale = max(float(np.abs(expected).max()), 1e-12)
        reference_rows.append(expected[0].reshape(expected.shape[1], -1) / scale)
        candidate_rows.append(actual[0].reshape(actual.shape[1], -1) / scale)
    reference_key = np.concatenate(reference_rows, axis=1)
    candidate_key = np.concatenate(candidate_rows, axis=1)
    cost = np.max(
        np.abs(reference_key[:, None, :] - candidate_key[None, :, :]),
        axis=2,
    )
    rows, columns = linear_sum_assignment(cost)
    order = columns[np.argsort(rows)]
    return [array[:, order, ...] for array in candidate]


def _assert_parity(names, reference, actual):
    reference_first, reference_second = reference
    actual_first, actual_second = actual
    assert [array.shape for array in actual_first] == [
        array.shape for array in reference_first
    ]
    assert [array.shape for array in actual_second] == [
        array.shape for array in reference_second
    ]

    for index, (expected1, expected2, got1, got2) in enumerate(
        zip(reference_first, reference_second, actual_first, actual_second)
    ):
        assert np.isfinite(got1).all() and np.isfinite(got2).all()
        scale = max(
            float(np.abs(expected1).max()),
            float(np.abs(expected2).max()),
            1e-12,
        )
        error = (
            max(
                float(np.abs(got1 - expected1).max()),
                float(np.abs(got2 - expected2).max()),
            )
            / scale
        )
        sensitivity = float(np.abs(expected2 - expected1).max()) / scale
        margin = float("inf") if error == 0 else sensitivity / error
        assert error <= REL_TOL, (
            f"out[{index}] ({names[index]}) relative error "
            f"{error:.3e} exceeds {REL_TOL:.0e}"
        )
        assert sensitivity >= MIN_REL_SENSITIVITY, (
            f"out[{index}] ({names[index]}) relative input sensitivity "
            f"{sensitivity:.3e} is below {MIN_REL_SENSITIVITY:.0e}"
        )
        assert margin >= MIN_SENSITIVITY_MARGIN, (
            f"out[{index}] ({names[index]}) parity margin {margin:.1f}x is "
            f"below {MIN_SENSITIVITY_MARGIN:.0f}x "
            f"(error={error:.3e}, sensitivity={sensitivity:.3e})"
        )


def _assert_model_artifact_parity(
    model,
    family,
    task,
    imgsz,
    tmp_path,
    *,
    half=False,
):
    output_path = tmp_path / f"{family}-{task}.mlpackage"
    artifact = model.export(
        format="coreml",
        imgsz=imgsz,
        output_path=str(output_path),
        compute_units="cpu_only",
        half=half,
    )
    assert Path(artifact).is_dir()

    probes = _canonical_byte_probes(imgsz, imgsz)
    names, reference = _prepared_reference(model, family, task, imgsz, probes)
    runtime, contract, artifact_names = _load_artifact(artifact)
    assert artifact_names == names
    actual = _artifact_outputs(runtime, contract, artifact_names, probes)
    if family == "rtdetrv2":
        actual[0] = _align_unordered_queries(reference[0], actual[0])
        actual[1] = _align_unordered_queries(reference[1], actual[1])
    _assert_parity(names, reference, actual)
    return artifact


@pytest.mark.parametrize("weights,family,task,imgsz", TRAINED_CASES)
def test_coreml_artifact_matches_prepared_trained_model(
    weights,
    family,
    task,
    imgsz,
    tmp_path,
):
    from libreyolo import LibreYOLO

    model = LibreYOLO(weights, device="cpu")
    assert model.task == task
    _assert_model_artifact_parity(model, family, task, imgsz, tmp_path)


def test_coreml_birefnet_l_trained_matte_parity_and_public_path(tmp_path):
    """Gate the post-9.0 deform-conv lowering on real permissive weights."""
    from libreyolo import LibreYOLO
    from libreyolo.export.coreml_birefnet import (
        BIREFNET_COREML_DEFORM_CONV_MERGE,
        has_birefnet_coreml_lowering,
    )

    if not has_birefnet_coreml_lowering():
        pytest.skip(
            "Installed coremltools predates Apple's BiRefNet deform-conv "
            f"lowering merge {BIREFNET_COREML_DEFORM_CONV_MERGE}."
        )

    model = LibreYOLO("LibreBiRefNetl-matte.pt", device="cpu")
    artifact = _assert_model_artifact_parity(
        model,
        "birefnet",
        "matte",
        1024,
        tmp_path,
    )

    # Exercise Apple's non-CPU-only planning path independently. Passing the
    # parity gate does not assert that every operation was placed on the ANE.
    probes = _canonical_byte_probes(1024, 1024)
    names, reference = _prepared_reference(
        model,
        "birefnet",
        "matte",
        1024,
        probes[:1],
    )
    all_runtime = ct.models.MLModel(
        str(artifact),
        compute_units=ct.ComputeUnit.ALL,
    )
    _runtime, contract, artifact_names = _load_artifact(artifact)
    all_actual = _artifact_outputs(
        all_runtime,
        contract,
        artifact_names,
        probes[:1],
    )[0]
    assert artifact_names == names
    for expected, actual in zip(reference[0], all_actual):
        scale = max(float(np.abs(expected).max()), 1e-12)
        assert float(np.abs(actual - expected).max()) / scale <= REL_TOL

    # The public backend must preserve native stretch/ImageNet preprocessing
    # and sigmoid/bilinear matte placement for a non-square source image.
    yy, xx = np.mgrid[:173, :257]
    odd_rgb = np.stack(
        (
            (7 * xx + 3 * yy + 11) % 256,
            (5 * xx + 13 * yy + 29) % 256,
            (17 * xx + 19 * yy + 47) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    source = Image.fromarray(odd_rgb, mode="RGB")
    native = model.predict(source, verbose=False)[0].matte.array
    deployed = LibreYOLO(artifact).predict(source, verbose=False)[0].matte.array
    assert native.shape == deployed.shape == (173, 257)
    assert np.isfinite(deployed).all()
    assert float(np.max(np.abs(native - deployed))) <= 3e-4


def test_coreml_depth_anything3_trained_raw_and_public_depth_path(tmp_path):
    """Validate DA3's converted raw component and host sky/inverse contract."""
    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreDepthAnything3l-depth.pt", device="cpu")
    artifact = _assert_model_artifact_parity(
        model,
        "depth_anything3",
        "depth",
        504,
        tmp_path,
        half=True,
    )

    # Use an already-native square so this verifies host orchestration without
    # conflating the documented non-square fixed-stretch approximation with
    # DA3's native keep-aspect geometry.
    source = Image.fromarray(
        _canonical_byte_probes(504, 504)[0],
        mode="RGB",
    )
    torch.manual_seed(9182)
    native = model.predict(source, verbose=False)[0].depth_map.array
    torch.manual_seed(9182)
    deployed = LibreYOLO(artifact).predict(source, verbose=False)[0].depth_map.array
    assert native.shape == deployed.shape == (504, 504)
    assert np.isfinite(deployed).all()
    scale = max(float(np.abs(native).max()), 1e-8)
    assert float(np.abs(native - deployed).max()) / scale <= 3e-4


def test_coreml_owlv2_b16_frozen_vocabulary_raw_and_public_path(tmp_path):
    """Gate the text-free graph, exact host geometry, and named raw outputs."""
    from libreyolo import LibreOpenVocab, LibreYOLO
    from libreyolo.export.coreml_owlv2 import (
        build_owlv2_frozen_coreml_adapter,
        postprocess_owlv2_coreml_outputs,
        preprocess_owlv2_coreml_image,
    )

    labels = ["person", "dog", "remote control"]
    model = LibreOpenVocab("owlv2-b16", device="cpu")
    model.set_classes(labels)
    artifact = model.export(
        format="coreml",
        half=True,
        output_path=str(tmp_path / "owlv2-b16.mlpackage"),
        compute_units="cpu_only",
    )
    adapter = build_owlv2_frozen_coreml_adapter(
        model.model,
        model.processor,
        size=model.size,
        names=model.names,
    ).eval()

    probes = _canonical_byte_probes(960, 960)
    tensors = tuple(_as_canonical_tensor(probe) for probe in probes)
    with torch.no_grad():
        reference = [
            [
                tensor.detach().cpu().numpy()
                for tensor in adapter(value)
            ]
            for value in tensors
        ]

    runtime, contract, names = _load_artifact(artifact)
    assert names == ["pred_logits", "pred_boxes"]
    actual = _artifact_outputs(runtime, contract, names, probes)
    actual = [
        _align_unordered_queries(expected, deployed)
        for expected, deployed in zip(reference, actual)
    ]
    _assert_parity(names, reference, actual)

    # Exercise the default ALL compute-unit route and the non-square public
    # preprocessing/postprocessing contract independently of the raw CPU gate.
    source = Image.fromarray(
        _canonical_byte_probes(317, 191)[0],
        mode="RGB",
    )
    prepared = preprocess_owlv2_coreml_image(source, image_size=960)
    with torch.no_grad():
        logits, boxes = adapter(prepared)
    expected = postprocess_owlv2_coreml_outputs(
        logits,
        boxes,
        original_size=source.size,
        conf=0.1,
        max_det=300,
    )
    deployed = LibreYOLO(artifact, compute_units="all").predict(
        source,
        conf=0.1,
        verbose=False,
    )[0]
    assert deployed.names == dict(enumerate(labels))
    assert len(deployed.boxes) == expected["num_detections"]
    torch.testing.assert_close(
        deployed.boxes.xyxy.cpu(),
        expected["boxes"].cpu(),
        rtol=3e-4,
        atol=3e-4,
    )
    torch.testing.assert_close(
        deployed.boxes.conf.cpu(),
        expected["scores"].cpu(),
        rtol=3e-4,
        atol=3e-4,
    )
    torch.testing.assert_close(
        deployed.boxes.cls.cpu(),
        expected["classes"].to(dtype=torch.float32).cpu(),
        rtol=0.0,
        atol=0.0,
    )


def test_coreml_grounding_dino_t_frozen_prompt_raw_and_public_path(tmp_path):
    """Gate the frozen BERT boundary, named outputs, and host phrase decode."""
    from libreyolo import LibreOpenVocab, LibreYOLO
    from libreyolo.export.coreml_grounding_dino import (
        GROUNDING_DINO_COREML_MEAN,
        GROUNDING_DINO_COREML_STD,
        build_grounding_dino_frozen_coreml_adapter,
        postprocess_grounding_dino_coreml_outputs,
        preprocess_grounding_dino_coreml_image,
    )

    labels = ["person", "dog", "remote control"]
    model = LibreOpenVocab("grounding-dino-tiny", device="cpu")
    model.set_classes(labels)
    adapter = build_grounding_dino_frozen_coreml_adapter(
        model.model,
        model.processor,
        size=model.size,
        names=model.names,
    ).eval()

    probes = _canonical_byte_probes(800, 800)
    tensors = tuple(_as_canonical_tensor(probe) for probe in probes)
    with torch.no_grad():
        reference = [
            [
                tensor.detach().cpu().numpy()
                for tensor in adapter(value)
            ]
            for value in tensors
        ]
        mean = torch.tensor(GROUNDING_DINO_COREML_MEAN).view(
            1,
            3,
            1,
            1,
        )
        std = torch.tensor(GROUNDING_DINO_COREML_STD).view(
            1,
            3,
            1,
            1,
        )
        frozen = adapter.frozen_text_contract
        source_reference = []
        for value in tensors:
            normalized = (value - mean) / std
            source_outputs = model.model(
                pixel_values=normalized,
                pixel_mask=torch.ones(
                    normalized.shape[0],
                    normalized.shape[2],
                    normalized.shape[3],
                    dtype=torch.long,
                ),
                input_ids=frozen.input_ids,
                token_type_ids=frozen.token_type_ids,
                attention_mask=frozen.attention_mask,
            )
            source_reference.append(
                [
                    source_outputs.logits[
                        ..., : frozen.sequence_length
                    ]
                    .detach()
                    .cpu()
                    .numpy(),
                    source_outputs.pred_boxes.detach().cpu().numpy(),
                ]
            )
    _assert_parity(
        ["token_logits", "pred_boxes"],
        source_reference,
        reference,
    )

    artifact = model.export(
        format="coreml",
        half=True,
        output_path=str(tmp_path / "grounding-dino-t.mlpackage"),
        compute_units="cpu_only",
    )

    runtime, contract, names = _load_artifact(artifact)
    assert names == ["token_logits", "pred_boxes"]
    actual = _artifact_outputs(runtime, contract, names, probes)
    _assert_parity(names, reference, actual)

    source = Image.fromarray(
        _canonical_byte_probes(317, 191)[0],
        mode="RGB",
    )
    prepared = preprocess_grounding_dino_coreml_image(source)
    with torch.no_grad():
        logits, boxes = adapter(prepared)
    from libreyolo.export.coreml_grounding_dino import (
        frozen_grounding_dino_text_from_metadata,
    )

    metadata = dict(runtime.user_defined_metadata)
    text_contract = frozen_grounding_dino_text_from_metadata(
        metadata,
        names=model.names,
    )
    expected = postprocess_grounding_dino_coreml_outputs(
        logits,
        boxes,
        size=model.size,
        names=model.names,
        text_contract=text_contract,
        original_size=source.size,
        conf=0.25,
        text_threshold=0.25,
        max_det=300,
    )
    deployed = LibreYOLO(artifact, compute_units="all").predict(
        source,
        conf=0.25,
        text_threshold=0.25,
        verbose=False,
    )[0]
    assert deployed.names == dict(enumerate(labels))
    assert len(deployed.boxes) == expected["num_detections"]
    torch.testing.assert_close(
        deployed.boxes.xyxy.cpu(),
        expected["boxes"].cpu(),
        rtol=3e-4,
        atol=3e-4,
    )
    torch.testing.assert_close(
        deployed.boxes.conf.cpu(),
        expected["scores"].cpu(),
        rtol=3e-4,
        atol=3e-4,
    )
    torch.testing.assert_close(
        deployed.boxes.cls.cpu(),
        expected["classes"].to(dtype=torch.float32).cpu(),
        rtol=0.0,
        atol=0.0,
    )


def test_coreml_omdet_turbo_t_frozen_vocabulary_raw_and_public_path(tmp_path):
    """Gate the complete pinned graph, exact uint8 resize, and host NMS."""
    from libreyolo import LibreOpenVocab, LibreYOLO
    from libreyolo.export.coreml_omdet_turbo import (
        OmDetTurboFrozenCoreMLAdapter,
        build_omdet_turbo_frozen_coreml_adapter,
        postprocess_omdet_turbo_coreml_outputs,
        preprocess_omdet_turbo_coreml_image,
    )

    labels = ["person", "dog", "remote control"]
    model = LibreOpenVocab("omdet-turbo", device="cpu")
    model.set_classes(labels)
    adapter = build_omdet_turbo_frozen_coreml_adapter(
        model.model,
        model.processor,
        size=model.size,
        names=model.names,
    ).eval()

    probes = _canonical_byte_probes(640, 640)
    tensors = tuple(
        torch.from_numpy(probe.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        for probe in probes
    )
    with torch.no_grad():
        reference = [
            [
                tensor.detach().cpu().numpy()
                for tensor in adapter(value)
            ]
            for value in tensors
        ]
        source_adapter = OmDetTurboFrozenCoreMLAdapter(
            model.model,
            adapter.class_features,
            adapter.task_features,
            adapter.task_mask,
        ).eval()
        source_reference = [
            [
                tensor.detach().cpu().numpy()
                for tensor in source_adapter(value)
            ]
            for value in tensors
        ]
    _assert_parity(
        ["pred_logits", "pred_boxes"],
        source_reference,
        reference,
    )

    artifact = model.export(
        format="coreml",
        half=True,
        output_path=str(tmp_path / "omdet-turbo-t.mlpackage"),
        compute_units="cpu_only",
    )

    runtime, contract, names = _load_artifact(artifact)
    assert names == ["pred_logits", "pred_boxes"]
    actual = _artifact_outputs(runtime, contract, names, probes)
    actual = [
        _align_unordered_queries(expected, deployed)
        for expected, deployed in zip(reference, actual)
    ]
    _assert_parity(names, reference, actual)

    source = Image.fromarray(
        _canonical_byte_probes(317, 191)[0],
        mode="RGB",
    )
    prepared = preprocess_omdet_turbo_coreml_image(source)
    with torch.no_grad():
        logits, boxes = adapter(prepared)
    expected = postprocess_omdet_turbo_coreml_outputs(
        logits,
        boxes,
        original_size=source.size,
        conf=0.3,
        iou=0.5,
        max_det=300,
    )
    deployed = LibreYOLO(artifact, compute_units="all").predict(
        source,
        verbose=False,
    )[0]
    assert deployed.names == dict(enumerate(labels))
    assert len(deployed.boxes) == expected["num_detections"]
    torch.testing.assert_close(
        deployed.boxes.xyxy.cpu(),
        expected["boxes"].cpu(),
        rtol=3e-4,
        atol=3e-4,
    )
    torch.testing.assert_close(
        deployed.boxes.conf.cpu(),
        expected["scores"].cpu(),
        rtol=3e-4,
        atol=3e-4,
    )
    torch.testing.assert_close(
        deployed.boxes.cls.cpu(),
        expected["classes"].to(dtype=torch.float32).cpu(),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("weights,family,imgsz", FROZEN_CLASS_CASES)
def test_coreml_frozen_classifier_matches_trained_model(
    weights,
    family,
    imgsz,
    tmp_path,
):
    """Exercise the custom two-tower-to-frozen-class Core ML export route."""
    from libreyolo import LibreYOLO
    from libreyolo.export.coreml import (
        _CoreMLOutputAdapter,
        _flatten_tensor_outputs,
        _output_contract,
        _wrap_coreml_contract,
    )

    model = LibreYOLO(weights, device="cpu")
    model.set_classes(
        ["cat", "dog", "car"],
        templates=["a photo of a {}."],
    )
    artifact = model.export(
        format="coreml",
        imgsz=imgsz,
        output_path=str(tmp_path / f"{family}-classify.mlpackage"),
        compute_units="cpu_only",
    )

    scale = float(model.model.logit_scale.exp().detach().cpu())
    weight = (scale * model._text_embeds).detach().cpu()
    if family == "clip":
        from libreyolo.models.clip.export import _FrozenCLIPClassifier

        frozen = _FrozenCLIPClassifier(model.model.visual, weight).eval()
    else:
        from libreyolo.models.siglip2.export import _FrozenSigLIP2Classifier

        bias = model.model.logit_bias.detach().cpu().float().reshape(())
        frozen = _FrozenSigLIP2Classifier(
            model.model.vision_model,
            weight,
            bias,
        ).eval()

    probes = _canonical_byte_probes(imgsz, imgsz)
    tensors = tuple(_as_canonical_tensor(probe) for probe in probes)
    declared = _output_contract(family, "classify", nms=False)
    names = [item["name"] for item in declared]
    prepared = _CoreMLOutputAdapter(
        _wrap_coreml_contract(frozen, family, "classify"),
        names,
    ).eval()
    with torch.no_grad():
        reference = [
            [
                tensor.detach().cpu().numpy()
                for tensor in _flatten_tensor_outputs(prepared(value))
            ]
            for value in tensors
        ]

    runtime, contract, artifact_names = _load_artifact(artifact)
    assert artifact_names == names
    actual = _artifact_outputs(runtime, contract, names, probes)
    _assert_parity(names, reference, actual)


def test_coreml_l2cs_gaze_parity_when_nonredistributable_weights_are_staged(
    tmp_path,
):
    """Queue the gaze ABI without pretending its restricted weights can ship."""
    candidates = (
        Path("LibreL2CSr50.pt"),
        Path("weights") / "LibreL2CSr50.pt",
    )
    weights = next((path for path in candidates if path.is_file()), None)
    if weights is None:
        pytest.skip(
            "L2CS/Gaze360 weights are non-redistributable; stage "
            "LibreL2CSr50.pt locally to run this Core ML gate."
        )

    from libreyolo import LibreYOLO

    model = LibreYOLO(str(weights), device="cpu")
    assert model.task == "gaze"
    _assert_model_artifact_parity(model, "l2cs", "gaze", 448, tmp_path)


def test_coreml_fomo_synthetic_trained_parity(tmp_path):
    """Use generated tensors so point export does not depend on hosted weights."""
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
    _assert_model_artifact_parity(model, "fomo", "point", 96, tmp_path)


def test_coreml_picosam3_trained_roi_component_parity(tmp_path):
    """Validate the honest one-ROI component rather than a full-image fiction."""
    from libreyolo import LibrePicoSAM3

    model = LibrePicoSAM3(size="pico", device="cpu")
    _assert_model_artifact_parity(
        model,
        "picosam3",
        "segment",
        96,
        tmp_path,
    )


@pytest.mark.parametrize(
    "weights",
    ["LibrePPOCRt-ocr.pt", "LibrePPOCRl-ocr.pt"],
)
def test_coreml_ppocr_trained_multifunction_parity(weights, tmp_path):
    """Run both bounded-flexible OCR functions by their declared names."""
    from libreyolo import LibreYOLO
    from libreyolo.export.coreml_ppocr import (
        PPOCR_COREML_DETECTOR_INPUT,
        PPOCR_COREML_DETECTOR_OUTPUT,
        PPOCR_COREML_RECOGNIZER_INPUT,
        PPOCR_COREML_RECOGNIZER_OUTPUT,
        validate_ppocr_coreml_metadata,
    )

    model = LibreYOLO(weights, device="cpu")
    artifact = model.export(
        format="coreml",
        imgsz=960,
        rec_batch_max=6,
        rec_max_width=2048,
        output_path=str(tmp_path / f"{Path(weights).stem}.mlpackage"),
        compute_units="cpu_only",
    )
    detector = ct.models.MLModel(
        artifact,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        function_name="detector",
    )
    recognizer = ct.models.MLModel(
        artifact,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        function_name="recognizer",
    )
    metadata = dict(detector.user_defined_metadata)
    validate_ppocr_coreml_metadata(metadata)
    spec = detector.get_spec()
    assert spec.specificationVersion >= 9
    assert spec.description.defaultFunctionName == "detector"
    assert [function.name for function in spec.description.functions] == [
        "detector",
        "recognizer",
    ]

    det_inputs = [
        torch.linspace(-2.0, 2.0, 3 * 64 * 96).reshape(1, 3, 64, 96),
        torch.linspace(2.0, -2.0, 3 * 96 * 64).reshape(1, 3, 96, 64),
    ]
    rec_inputs = [
        torch.linspace(-1.0, 1.0, 3 * 48 * 320).reshape(1, 3, 48, 320),
        torch.linspace(1.0, -1.0, 2 * 3 * 48 * 641).reshape(2, 3, 48, 641),
    ]
    with torch.inference_mode():
        expected_det = [model.model.det(value).cpu().numpy() for value in det_inputs]
        expected_rec = [model.model.rec(value).cpu().numpy() for value in rec_inputs]
    actual_det = [
        np.asarray(
            detector.predict(
                {
                    PPOCR_COREML_DETECTOR_INPUT: np.ascontiguousarray(
                        value.numpy(),
                    )
                }
            )[PPOCR_COREML_DETECTOR_OUTPUT]
        )
        for value in det_inputs
    ]
    actual_rec = [
        np.asarray(
            recognizer.predict(
                {
                    PPOCR_COREML_RECOGNIZER_INPUT: np.ascontiguousarray(
                        value.numpy(),
                    )
                }
            )[PPOCR_COREML_RECOGNIZER_OUTPUT]
        )
        for value in rec_inputs
    ]
    for expected, actual in zip(expected_det + expected_rec, actual_det + actual_rec):
        assert expected.shape == actual.shape
        assert np.isfinite(actual).all()
        scale = max(float(np.abs(expected).max()), 1e-12)
        assert float(np.abs(expected - actual).max()) / scale <= REL_TOL

    # Same-shape inversions make constant-output false passes impossible.
    detector_sensitivity = float(
        np.abs(
            actual_det[0]
            - np.asarray(
                detector.predict(
                    {
                        PPOCR_COREML_DETECTOR_INPUT: np.ascontiguousarray(
                            (-det_inputs[0]).numpy(),
                        )
                    }
                )[PPOCR_COREML_DETECTOR_OUTPUT]
            )
        ).max()
    )
    recognizer_sensitivity = float(
        np.abs(
            actual_rec[0]
            - np.asarray(
                recognizer.predict(
                    {
                        PPOCR_COREML_RECOGNIZER_INPUT: np.ascontiguousarray(
                            (-rec_inputs[0]).numpy(),
                        )
                    }
                )[PPOCR_COREML_RECOGNIZER_OUTPUT]
            )
        ).max()
    )
    assert detector_sensitivity > MIN_REL_SENSITIVITY
    assert recognizer_sensitivity > MIN_REL_SENSITIVITY


def test_coreml_yolonas_synthetic_trained_parity(tmp_path):
    """Exercise the decoded detection contract without hosted weights."""
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
        targets[0, 0] = torch.tensor(
            [float(step % 2), 36.0 + step, 42.0, 24.0, 30.0]
        )
        targets[1, 0] = torch.tensor(
            [float((step + 1) % 2), 64.0, 52.0, 20.0, 26.0]
        )
        outputs = network(images)
        loss, _ = loss_fn(outputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    network.eval()
    # A randomly initialized detector's decoded boxes are dominated by its
    # fixed anchor grid. Scaling only the synthetic regression heads keeps
    # both exported outputs meaningfully input-sensitive.
    with torch.no_grad():
        for head in (
            network.heads.head1,
            network.heads.head2,
            network.heads.head3,
        ):
            head.reg_pred.weight.mul_(20.0)

    _assert_model_artifact_parity(model, "yolonas", "detect", 96, tmp_path)


def test_coreml_yolo9_p2_permissive_transfer_parity(tmp_path):
    """Cover the extra stride-4 branch with a verified compatible checkpoint."""
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

    _assert_model_artifact_parity(model, "yolo9_p2", "detect", 640, tmp_path)
