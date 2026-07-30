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

import gc
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
UNORDERED_QUERY_FAMILIES = {
    "deim",
    "deimv2",
    "dfine",
    "ec",
    "rfdetr",
    "rtdetr",
    "rtdetrv2",
    "rtdetrv4",
}
FLAGSHIP_PUBLIC_DETECT_CASES = {
    ("LibreYOLO9t.pt", "yolo9", "detect", 640),
    ("LibreRFDETRn.pt", "rfdetr", "detect", 384),
}
PUBLIC_DETECT_CONF = 1e-5
PUBLIC_DETECT_MAX_DET = 50
PUBLIC_GENERIC_CONF = 1e-5
PUBLIC_GENERIC_MAX_DET = 25
PUBLIC_SEGMENT_MAX_DET = 12

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


def _public_non_square_source() -> Image.Image:
    """Return a bounded real-image probe with odd, non-square dimensions."""
    from libreyolo import SAMPLE_IMAGE

    with Image.open(SAMPLE_IMAGE) as image:
        return image.convert("RGB").resize(
            (257, 173),
            Image.Resampling.BILINEAR,
        )


def _public_fixed_canvas_source(imgsz: int) -> Image.Image:
    """Derive an exact square canvas from the same public real-image probe."""
    return _public_non_square_source().resize(
        (imgsz, imgsz),
        Image.Resampling.BILINEAR,
    )


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
        with _prepare_coreai_graph(wrapped, tensors[0], family), torch.no_grad():
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
            assert str(input_contract["layout"]).lower() == "nchw"
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
    assert all(array.ndim >= 3 for array in reference + candidate)
    assert len({array.shape[1] for array in reference + candidate}) == 1

    reference_rows = []
    candidate_rows = []
    for expected, actual in zip(reference, candidate):
        # Boxes/logits are compact assignment keys. Dense per-query masks and
        # keypoint grids receive the resulting permutation but must not enter
        # the QxQ cost tensor, which would make the parity gate needlessly huge.
        if expected.ndim != 3:
            continue
        scale = max(float(np.abs(expected).max()), 1e-12)
        reference_rows.append(expected[0].reshape(expected.shape[1], -1) / scale)
        candidate_rows.append(actual[0].reshape(actual.shape[1], -1) / scale)
    assert reference_rows
    reference_key = np.concatenate(reference_rows, axis=1)
    candidate_key = np.concatenate(candidate_rows, axis=1)
    cost = np.max(
        np.abs(reference_key[:, None, :] - candidate_key[None, :, :]),
        axis=2,
    )
    rows, columns = linear_sum_assignment(cost)
    order = columns[np.argsort(rows)]
    return [array[:, order, ...] for array in candidate]


def _result_detections(result):
    """Return validated public ``Results`` detection fields as NumPy arrays."""
    from libreyolo.utils.results import Results

    assert isinstance(result, Results)
    assert result.boxes is not None
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float64)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float64)
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.float64)
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert scores.shape == classes.shape == (boxes.shape[0],)
    assert boxes.shape[0] > 0, "public Core ML parity requires detections"
    assert np.isfinite(boxes).all()
    assert np.isfinite(scores).all()
    assert np.isfinite(classes).all()
    np.testing.assert_array_equal(classes, np.rint(classes))
    return boxes, scores, classes.astype(np.int64)


def _pairwise_detection_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compute the complete pairwise IoU matrix for two xyxy box arrays."""
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_wh = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    first_wh = np.maximum(first[:, 2:] - first[:, :2], 0.0)
    second_wh = np.maximum(second[:, 2:] - second[:, :2], 0.0)
    first_area = first_wh[:, 0] * first_wh[:, 1]
    second_area = second_wh[:, 0] * second_wh[:, 1]
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _pairwise_obb_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compute semantic IoU for every pair of rotated xywhr boxes."""
    import cv2

    result = np.empty((first.shape[0], second.shape[0]), dtype=np.float64)
    for first_index, first_box in enumerate(first):
        first_rect = (
            (float(first_box[0]), float(first_box[1])),
            (max(float(first_box[2]), 0.0), max(float(first_box[3]), 0.0)),
            float(np.rad2deg(first_box[4])),
        )
        first_area = first_rect[1][0] * first_rect[1][1]
        for second_index, second_box in enumerate(second):
            second_rect = (
                (float(second_box[0]), float(second_box[1])),
                (
                    max(float(second_box[2]), 0.0),
                    max(float(second_box[3]), 0.0),
                ),
                float(np.rad2deg(second_box[4])),
            )
            second_area = second_rect[1][0] * second_rect[1][1]
            _, intersection_polygon = cv2.rotatedRectangleIntersection(
                first_rect,
                second_rect,
            )
            intersection = (
                abs(float(cv2.contourArea(intersection_polygon)))
                if intersection_polygon is not None
                else 0.0
            )
            union = first_area + second_area - intersection
            result[first_index, second_index] = (
                intersection / union if union > 0.0 else 0.0
            )
    return result


def _pairwise_mask_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU without materializing an N x M x H x W tensor."""
    result = np.empty((first.shape[0], second.shape[0]), dtype=np.float64)
    for first_index, first_mask in enumerate(first):
        for second_index, second_mask in enumerate(second):
            intersection = np.logical_and(first_mask, second_mask).sum()
            union = np.logical_or(first_mask, second_mask).sum()
            result[first_index, second_index] = (
                float(intersection) / float(union) if union else 1.0
            )
    return result


def _match_public_detections(
    native,
    deployed,
    repeated,
    *,
    minimum_iou: float,
    maximum_score_error: float,
) -> np.ndarray:
    """Return one whole-detection assignment after repeat/parity checks."""
    assert native.orig_shape == deployed.orig_shape == repeated.orig_shape
    assert native.names == deployed.names == repeated.names
    native_boxes, native_scores, native_classes = _result_detections(native)
    deployed_boxes, deployed_scores, deployed_classes = _result_detections(deployed)
    repeat_boxes, repeat_scores, repeat_classes = _result_detections(repeated)

    np.testing.assert_array_equal(deployed_classes, repeat_classes)
    np.testing.assert_allclose(deployed_boxes, repeat_boxes, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        deployed_scores,
        repeat_scores,
        rtol=0.0,
        atol=1e-7,
    )

    assert native_boxes.shape[0] == deployed_boxes.shape[0]
    ious = _pairwise_detection_iou(native_boxes, deployed_boxes)
    score_delta = np.abs(native_scores[:, None] - deployed_scores[None, :])
    class_mismatch = native_classes[:, None] != deployed_classes[None, :]
    cost = 1.0 - ious + 2.0 * class_mismatch + 0.01 * score_delta
    rows, columns = linear_sum_assignment(cost)
    order = columns[np.argsort(rows)]
    assert np.array_equal(np.sort(rows), np.arange(native_boxes.shape[0]))
    np.testing.assert_array_equal(native_classes, deployed_classes[order])
    matched_ious = ious[np.arange(native_boxes.shape[0]), order]
    assert float(matched_ious.min()) >= minimum_iou
    matched_score_error = np.abs(native_scores - deployed_scores[order])
    assert float(matched_score_error.max()) <= maximum_score_error
    return order


def _segmentation_payload(result, *, allow_empty_masks: bool = False):
    """Return validated, row-aligned public instance-segmentation fields."""
    from libreyolo.utils.results import Results

    assert isinstance(result, Results)
    assert result.boxes is not None
    assert result.masks is not None
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float64)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float64)
    raw_classes = result.boxes.cls.detach().cpu().numpy().astype(np.float64)
    masks = result.masks.data.detach().cpu().numpy().astype(bool)
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert scores.shape == raw_classes.shape == (boxes.shape[0],)
    assert masks.shape == (boxes.shape[0], *result.orig_shape)
    assert boxes.shape[0] > 0, "public segmentation parity requires masks"
    assert (
        np.isfinite(boxes).all()
        and np.isfinite(scores).all()
        and np.isfinite(raw_classes).all()
    )
    np.testing.assert_array_equal(raw_classes, np.rint(raw_classes))
    nonempty = masks.reshape(masks.shape[0], -1).any(axis=1)
    assert nonempty.any()
    if not allow_empty_masks:
        assert nonempty.all()
    assert (~masks).reshape(masks.shape[0], -1).any(axis=1).all()
    return boxes, scores, raw_classes.astype(np.int64), masks


def _assert_public_segmentation_path(
    native,
    deployed,
    repeated,
    *,
    minimum_box_iou: float,
    minimum_mask_iou: float,
    maximum_score_error: float,
    allow_empty_masks: bool = False,
):
    """Compare native and Core ML Results with one whole-instance assignment."""
    assert native.orig_shape == deployed.orig_shape == repeated.orig_shape
    assert native.names == deployed.names == repeated.names
    native_values = _segmentation_payload(
        native,
        allow_empty_masks=allow_empty_masks,
    )
    deployed_values = _segmentation_payload(
        deployed,
        allow_empty_masks=allow_empty_masks,
    )
    repeated_values = _segmentation_payload(
        repeated,
        allow_empty_masks=allow_empty_masks,
    )
    native_boxes, native_scores, native_classes, native_masks = native_values
    deployed_boxes, deployed_scores, deployed_classes, deployed_masks = (
        deployed_values
    )
    repeat_boxes, repeat_scores, repeat_classes, repeat_masks = repeated_values

    # A second public invocation must retain complete row order and payload.
    np.testing.assert_array_equal(deployed_classes, repeat_classes)
    np.testing.assert_array_equal(deployed_masks, repeat_masks)
    np.testing.assert_allclose(deployed_boxes, repeat_boxes, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        deployed_scores,
        repeat_scores,
        rtol=0.0,
        atol=1e-7,
    )

    assert native_boxes.shape[0] == deployed_boxes.shape[0]
    box_iou = _pairwise_detection_iou(native_boxes, deployed_boxes)
    mask_iou = _pairwise_mask_iou(native_masks, deployed_masks)
    class_mismatch = native_classes[:, None] != deployed_classes[None, :]
    score_delta = np.abs(native_scores[:, None] - deployed_scores[None, :])
    cost = (
        1.0
        - box_iou
        + 1.0
        - mask_iou
        + 2.0 * class_mismatch
        + 0.01 * score_delta
    )
    rows, columns = linear_sum_assignment(cost)
    order = columns[np.argsort(rows)]
    assert np.array_equal(np.sort(rows), np.arange(native_boxes.shape[0]))
    np.testing.assert_array_equal(native_classes, deployed_classes[order])
    matched_box_iou = box_iou[np.arange(native_boxes.shape[0]), order]
    if allow_empty_masks:
        native_empty = ~native_masks.reshape(native_masks.shape[0], -1).any(axis=1)
        deployed_empty = ~deployed_masks[order].reshape(
            deployed_masks.shape[0],
            -1,
        ).any(axis=1)
        np.testing.assert_array_equal(native_empty, deployed_empty)
        np.testing.assert_allclose(
            native_boxes[native_empty],
            deployed_boxes[order][native_empty],
            rtol=0.0,
            atol=1e-6,
        )
        # IoU is zero for two identical zero-area boxes. Exact coordinates are
        # the meaningful parity gate for those native degenerate rows.
        matched_box_iou[native_empty] = 1.0
    assert float(matched_box_iou.min()) >= minimum_box_iou
    assert (
        float(mask_iou[np.arange(native_boxes.shape[0]), order].min())
        >= minimum_mask_iou
    )
    assert (
        float(np.abs(native_scores - deployed_scores[order]).max())
        <= maximum_score_error
    )


def _assert_flagship_public_detection_path(
    model,
    artifact,
    *,
    family: str,
    imgsz: int,
    compute_units: str | None = None,
):
    """Gate native and deployed public detection with one shared assignment."""
    from libreyolo import LibreYOLO

    source_array = _canonical_byte_probes(317, 191)[0]
    source = Image.fromarray(source_array, mode="RGB")
    predict_kwargs = {
        "imgsz": imgsz,
        "conf": PUBLIC_DETECT_CONF,
        "iou": 0.45,
        "max_det": PUBLIC_DETECT_MAX_DET,
        "verbose": False,
    }
    native = model.predict(source, **predict_kwargs)

    # Baseline YOLO9 exercises the user-facing validated default. RF-DETR
    # independently exercises its explicit ALL route. Discovery campaigns can
    # override both with an explicit experimental planner.
    deployed_model = (
        LibreYOLO(
            artifact,
            compute_units=(
                compute_units
                if compute_units is not None
                else ("validated" if family == "yolo9" else "all")
            ),
        )
    )
    deployed = deployed_model.predict(source, **predict_kwargs)
    deployed_repeat = deployed_model.predict(source, **predict_kwargs)

    expected_shape = (source_array.shape[0], source_array.shape[1])
    assert native.orig_shape == expected_shape
    assert deployed.orig_shape == deployed_repeat.orig_shape == expected_shape
    _match_public_detections(
        native,
        deployed,
        deployed_repeat,
        minimum_iou=0.95,
        maximum_score_error=0.01,
    )


def _assert_rtmdet_segment_public_path(
    model,
    artifact,
    *,
    compute_units: str = "validated",
):
    """Gate raw RTMDet-Ins outputs through native host mask semantics."""
    from libreyolo import LibreYOLO

    source = _public_non_square_source()
    predict_kwargs = {
        "conf": 0.2,
        "iou": 0.6,
        "max_det": 12,
        "verbose": False,
    }
    native = model.predict(source, imgsz=640, **predict_kwargs)
    deployed_model = LibreYOLO(artifact, compute_units=compute_units)
    deployed = deployed_model.predict(source, **predict_kwargs)
    repeated = deployed_model.predict(source, **predict_kwargs)
    _assert_public_segmentation_path(
        native,
        deployed,
        repeated,
        minimum_box_iou=0.99,
        minimum_mask_iou=0.995,
        maximum_score_error=5e-4,
    )


def _assert_public_semantic_path(
    model,
    artifact,
    *,
    imgsz: int,
    minimum_agreement: float,
    require_multiple_classes: bool = False,
    compute_units: str = "validated",
):
    """Gate task-specific geometry and dense semantic argmax on a real image."""
    from libreyolo import LibreYOLO

    source = _public_non_square_source()
    native = model.predict(source, imgsz=imgsz, verbose=False)
    deployed_model = LibreYOLO(artifact, compute_units=compute_units)
    deployed = deployed_model.predict(source, verbose=False)
    repeated = deployed_model.predict(source, verbose=False)

    assert native.orig_shape == deployed.orig_shape == repeated.orig_shape == (
        173,
        257,
    )
    assert native.names == deployed.names == repeated.names
    assert native.semantic_mask is not None
    assert deployed.semantic_mask is not None
    assert repeated.semantic_mask is not None
    native_mask = native.semantic_mask.data.detach().cpu().numpy()
    deployed_mask = deployed.semantic_mask.data.detach().cpu().numpy()
    repeated_mask = repeated.semantic_mask.data.detach().cpu().numpy()
    assert native_mask.shape == deployed_mask.shape == repeated_mask.shape == (
        173,
        257,
    )
    assert np.issubdtype(native_mask.dtype, np.integer)
    assert np.issubdtype(deployed_mask.dtype, np.integer)
    np.testing.assert_array_equal(deployed_mask, repeated_mask)
    native_classes = np.unique(native_mask)
    deployed_classes = np.unique(deployed_mask)
    if require_multiple_classes:
        assert native_classes.size >= 2
    np.testing.assert_array_equal(native_classes, deployed_classes)
    agreement = float(np.mean(native_mask == deployed_mask))
    assert agreement >= minimum_agreement
    for class_id in native_classes:
        native_class = native_mask == class_id
        deployed_class = deployed_mask == class_id
        intersection = np.logical_and(native_class, deployed_class).sum()
        union = np.logical_or(native_class, deployed_class).sum()
        assert union > 0
        assert float(intersection) / float(union) >= 0.95


def _assert_public_classification_path(
    model,
    artifact,
    *,
    imgsz: int,
    compute_units: str = "validated",
):
    """Gate labels, native preprocessing, activation, and top-k."""
    from libreyolo import LibreYOLO

    source = _public_non_square_source()
    native = model.predict(source, imgsz=imgsz, verbose=False)
    deployed_model = LibreYOLO(artifact, compute_units=compute_units)
    deployed = deployed_model.predict(source, verbose=False)
    repeated = deployed_model.predict(source, verbose=False)

    assert native.orig_shape == deployed.orig_shape == repeated.orig_shape == (
        173,
        257,
    )
    assert native.names == deployed.names == repeated.names
    assert native.probs is not None
    assert deployed.probs is not None
    assert repeated.probs is not None
    native_probs = native.probs.data.detach().cpu().numpy()
    deployed_probs = deployed.probs.data.detach().cpu().numpy()
    repeated_probs = repeated.probs.data.detach().cpu().numpy()
    assert native_probs.shape == deployed_probs.shape == repeated_probs.shape
    assert np.isfinite(deployed_probs).all()
    assert np.logical_and(deployed_probs >= 0.0, deployed_probs <= 1.0).all()
    assert float(np.ptp(native_probs)) >= 1e-5
    np.testing.assert_allclose(
        deployed_probs,
        repeated_probs,
        rtol=0.0,
        atol=1e-7,
    )
    assert float(np.abs(native_probs - deployed_probs).max()) <= 5e-4
    assert native.probs.top1 == deployed.probs.top1 == repeated.probs.top1
    assert native.probs.top5 == deployed.probs.top5 == repeated.probs.top5


def _assert_public_detection_path(
    model,
    artifact,
    *,
    family: str,
    imgsz: int,
    compute_units: str = "validated",
):
    """Gate a generic detector's complete public Results contract."""
    from libreyolo import LibreYOLO

    source = _public_non_square_source()
    predict_kwargs = {
        "conf": PUBLIC_GENERIC_CONF,
        # Darknet-v2/v3 probes have unstable suppression boundaries below
        # 0.7. At 0.7 all 25 public detections retain exact class parity.
        "iou": 0.7 if family in {"yolo2", "yolo3"} else 0.45,
        "max_det": PUBLIC_GENERIC_MAX_DET,
        "verbose": False,
    }
    native = model.predict(source, imgsz=imgsz, **predict_kwargs)
    deployed_model = LibreYOLO(artifact, compute_units=compute_units)
    deployed = deployed_model.predict(source, **predict_kwargs)
    repeated = deployed_model.predict(source, **predict_kwargs)
    _match_public_detections(
        native,
        deployed,
        repeated,
        minimum_iou=0.95,
        # EC's public top-k path amplifies an otherwise passing raw-graph
        # conversion delta to 1.0402e-2 on the deterministic M4 probe.
        maximum_score_error=0.011 if family == "ec" else 0.01,
    )


def _pose_payload(result) -> np.ndarray:
    """Return validated row-aligned public keypoints."""
    from libreyolo.utils.results import Results

    assert isinstance(result, Results)
    assert result.boxes is not None
    assert result.keypoints is not None
    keypoints = result.keypoints.data.detach().cpu().numpy().astype(np.float64)
    assert keypoints.ndim == 3 and keypoints.shape[-1] in {2, 3}
    assert keypoints.shape[0] == len(result.boxes) > 0
    assert np.isfinite(keypoints).all()
    return keypoints


def _assert_public_pose_path(
    model,
    artifact,
    *,
    family: str,
    imgsz: int,
    compute_units: str | None = None,
):
    """Gate boxes and their row-aligned keypoints through public predict."""
    from libreyolo import LibreYOLO

    source = _public_non_square_source()
    predict_kwargs = {
        "conf": PUBLIC_GENERIC_CONF,
        "iou": 0.6,
        "max_det": PUBLIC_GENERIC_MAX_DET,
        "verbose": False,
    }
    native = model.predict(source, imgsz=imgsz, **predict_kwargs)
    # RF-DETR pose's proven deployment profile is CPU_ONLY. Its ALL/GPU
    # planner changes the graph numerically beyond the conversion parity gate.
    deployed_model = LibreYOLO(
        artifact,
        compute_units=(
            compute_units
            if compute_units is not None
            else ("cpu_only" if family == "rfdetr" else "validated")
        ),
    )
    deployed = deployed_model.predict(source, **predict_kwargs)
    repeated = deployed_model.predict(source, **predict_kwargs)
    order = _match_public_detections(
        native,
        deployed,
        repeated,
        # One low-area RF-DETR pose box amplifies a sub-pixel coordinate
        # difference into a lower IoU. Pair its narrow IoU gate with the
        # explicit absolute-coordinate gate below; other pose families retain
        # the original threshold.
        minimum_iou=0.99 if family == "rfdetr" else 0.995,
        maximum_score_error=0.005,
    )
    if family == "rfdetr":
        native_boxes = _result_detections(native)[0]
        deployed_boxes = _result_detections(deployed)[0][order]
        assert float(np.abs(native_boxes - deployed_boxes).max()) <= 0.25

    native_keypoints = _pose_payload(native)
    deployed_keypoints = _pose_payload(deployed)
    repeated_keypoints = _pose_payload(repeated)
    assert native_keypoints.shape == deployed_keypoints.shape
    np.testing.assert_allclose(
        deployed_keypoints,
        repeated_keypoints,
        rtol=0.0,
        atol=1e-6,
    )
    aligned = deployed_keypoints[order]
    assert float(np.abs(native_keypoints[..., :2] - aligned[..., :2]).max()) <= 0.5
    if native_keypoints.shape[-1] == 3:
        assert (
            float(np.abs(native_keypoints[..., 2] - aligned[..., 2]).max())
            <= 0.005
        )


def _obb_payload(result):
    """Return validated public rotated-box geometry, scores, and classes."""
    from libreyolo.utils.results import Results

    assert isinstance(result, Results)
    assert result.obb is not None
    geometry = result.obb.xywhr.detach().cpu().numpy().astype(np.float64)
    scores = result.obb.conf.detach().cpu().numpy().astype(np.float64)
    raw_classes = result.obb.cls.detach().cpu().numpy().astype(np.float64)
    assert geometry.ndim == 2 and geometry.shape[1] == 5
    assert scores.shape == raw_classes.shape == (geometry.shape[0],)
    assert geometry.shape[0] > 0, "public Core ML parity requires rotated boxes"
    assert np.isfinite(geometry).all()
    assert np.isfinite(scores).all()
    assert np.isfinite(raw_classes).all()
    assert np.logical_and(geometry[:, 2] > 0.0, geometry[:, 3] > 0.0).all()
    np.testing.assert_array_equal(raw_classes, np.rint(raw_classes))
    return geometry, scores, raw_classes.astype(np.int64)


def _assert_public_obb_path(
    model,
    artifact,
    *,
    imgsz: int,
    compute_units: str = "validated",
):
    """Gate rotated geometry with one semantic whole-object assignment."""
    from libreyolo import LibreYOLO

    source = _public_non_square_source()
    predict_kwargs = {
        "conf": PUBLIC_GENERIC_CONF,
        "iou": 0.45,
        "max_det": PUBLIC_GENERIC_MAX_DET,
        "verbose": False,
    }
    native = model.predict(source, imgsz=imgsz, **predict_kwargs)
    deployed_model = LibreYOLO(artifact, compute_units=compute_units)
    deployed = deployed_model.predict(source, **predict_kwargs)
    repeated = deployed_model.predict(source, **predict_kwargs)

    assert native.orig_shape == deployed.orig_shape == repeated.orig_shape
    assert native.names == deployed.names == repeated.names
    native_geometry, native_scores, native_classes = _obb_payload(native)
    deployed_geometry, deployed_scores, deployed_classes = _obb_payload(deployed)
    repeat_geometry, repeat_scores, repeat_classes = _obb_payload(repeated)
    np.testing.assert_array_equal(deployed_classes, repeat_classes)
    np.testing.assert_allclose(
        deployed_geometry,
        repeat_geometry,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        deployed_scores,
        repeat_scores,
        rtol=0.0,
        atol=1e-7,
    )

    assert native_geometry.shape[0] == deployed_geometry.shape[0]
    ious = _pairwise_obb_iou(native_geometry, deployed_geometry)
    score_delta = np.abs(native_scores[:, None] - deployed_scores[None, :])
    class_mismatch = native_classes[:, None] != deployed_classes[None, :]
    cost = 1.0 - ious + 2.0 * class_mismatch + 0.01 * score_delta
    rows, columns = linear_sum_assignment(cost)
    order = columns[np.argsort(rows)]
    assert np.array_equal(np.sort(rows), np.arange(native_geometry.shape[0]))
    np.testing.assert_array_equal(native_classes, deployed_classes[order])
    assert float(ious[np.arange(native_geometry.shape[0]), order].min()) >= 0.95
    assert float(np.abs(native_scores - deployed_scores[order]).max()) <= 0.01


def _assert_public_depth_path(
    model,
    artifact,
    *,
    family: str,
    imgsz: int,
    compute_units: str = "validated",
):
    """Gate depth's fixed-stretch public profile without claiming false parity."""
    from libreyolo import LibreYOLO

    deployed_model = LibreYOLO(artifact, compute_units=compute_units)
    odd_source = _public_non_square_source()
    odd_deployed = deployed_model.predict(odd_source, verbose=False)
    odd_repeated = deployed_model.predict(odd_source, verbose=False)
    assert odd_deployed.orig_shape == odd_repeated.orig_shape == (
        173,
        257,
    )
    assert odd_deployed.names == odd_repeated.names
    assert odd_deployed.depth_map is not None
    assert odd_repeated.depth_map is not None
    odd_depth = (
        odd_deployed.depth_map.data.detach().cpu().numpy().astype(np.float64)
    )
    odd_repeat_depth = (
        odd_repeated.depth_map.data.detach().cpu().numpy().astype(np.float64)
    )
    assert odd_depth.shape == odd_repeat_depth.shape == (173, 257)
    assert np.isfinite(odd_depth).all()
    assert float(np.ptp(odd_depth)) >= 1e-6
    np.testing.assert_allclose(
        odd_depth,
        odd_repeat_depth,
        rtol=0.0,
        atol=1e-7,
    )

    # Independently reproduce the artifact's declared OpenCV stretch, then run
    # that exact square through a pristine native model. Resizing the resulting
    # square depth back with the documented align_corners=True operation gives
    # an oracle for the odd public path without conflating it with native
    # keep-aspect preprocessing.
    import cv2

    interpolation = (
        cv2.INTER_CUBIC if family == "depth_anything" else cv2.INTER_LINEAR
    )
    stretched = cv2.resize(
        np.asarray(odd_source, dtype=np.uint8),
        (imgsz, imgsz),
        interpolation=interpolation,
    )
    source = Image.fromarray(stretched, mode="RGB")
    native = model.predict(source, imgsz=imgsz, verbose=False)
    deployed = deployed_model.predict(source, verbose=False)
    repeated = deployed_model.predict(source, verbose=False)
    assert native.orig_shape == deployed.orig_shape == repeated.orig_shape == (
        imgsz,
        imgsz,
    )
    assert native.names == deployed.names == repeated.names
    assert native.depth_map is not None
    assert deployed.depth_map is not None
    assert repeated.depth_map is not None
    native_depth = native.depth_map.data.detach().cpu().numpy().astype(np.float64)
    deployed_depth = (
        deployed.depth_map.data.detach().cpu().numpy().astype(np.float64)
    )
    repeated_depth = (
        repeated.depth_map.data.detach().cpu().numpy().astype(np.float64)
    )
    assert native_depth.shape == deployed_depth.shape == repeated_depth.shape == (
        imgsz,
        imgsz,
    )
    assert np.isfinite(native_depth).all()
    assert np.isfinite(deployed_depth).all()
    assert float(np.ptp(native_depth)) >= 1e-6
    np.testing.assert_allclose(
        deployed_depth,
        repeated_depth,
        rtol=0.0,
        atol=1e-7,
    )
    scale = max(float(np.abs(native_depth).max()), 1e-8)
    assert float(np.abs(native_depth - deployed_depth).max()) / scale <= 5e-4
    odd_expected = (
        F.interpolate(
            torch.from_numpy(native_depth)[None, None].float(),
            size=(173, 257),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        .numpy()
        .astype(np.float64)
    )
    odd_scale = max(float(np.abs(odd_expected).max()), 1e-8)
    assert float(np.abs(odd_expected - odd_depth).max()) / odd_scale <= 5e-4


def _assert_public_restore_path(
    model,
    artifact,
    *,
    imgsz: int,
    compute_units: str = "validated",
):
    """Gate the bounded native-canvas profile and its fail-closed geometry."""
    from libreyolo import LibreYOLO

    deployed_model = LibreYOLO(artifact, compute_units=compute_units)
    with pytest.raises(ValueError, match="fixed native geometry"):
        deployed_model.predict(_public_non_square_source(), verbose=False)

    # Fixed native geometry cannot honestly resize the odd source. Exercise
    # its supported public profile with an exact canvas derived from that same
    # SAMPLE_IMAGE, then compare the complete quantized restored RGB payload.
    source = _public_fixed_canvas_source(imgsz)
    native = model.predict(source, imgsz=imgsz, verbose=False)
    deployed = deployed_model.predict(source, verbose=False)
    repeated = deployed_model.predict(source, verbose=False)
    assert native.orig_shape == deployed.orig_shape == repeated.orig_shape == (
        imgsz,
        imgsz,
    )
    assert native.names == deployed.names == repeated.names
    assert native.restore_scale == deployed.restore_scale == repeated.restore_scale
    assert native.restored is not None
    assert deployed.restored is not None
    assert repeated.restored is not None
    native_rgb = native.restored.array
    deployed_rgb = deployed.restored.array
    repeated_rgb = repeated.restored.array
    output_side = imgsz * int(native.restore_scale)
    assert native_rgb.shape == deployed_rgb.shape == repeated_rgb.shape == (
        output_side,
        output_side,
        3,
    )
    assert native_rgb.dtype == deployed_rgb.dtype == repeated_rgb.dtype == np.uint8
    np.testing.assert_array_equal(deployed_rgb, repeated_rgb)
    absolute_error = np.abs(
        native_rgb.astype(np.int16) - deployed_rgb.astype(np.int16)
    )
    assert int(absolute_error.max()) <= 1
    assert float(absolute_error.mean()) <= 0.02


def _assert_public_segment_path(
    model,
    artifact,
    *,
    family: str,
    imgsz: int,
    compute_units: str = "validated",
    allow_empty_masks: bool = False,
):
    """Gate generic public instance segmentation and repeat stability."""
    from libreyolo import LibreYOLO

    source = _public_non_square_source()
    predict_kwargs = {
        "conf": 0.05,
        "iou": 0.6,
        "max_det": PUBLIC_SEGMENT_MAX_DET,
        "verbose": False,
    }
    native = model.predict(source, imgsz=imgsz, **predict_kwargs)
    deployed_model = LibreYOLO(artifact, compute_units=compute_units)
    deployed = deployed_model.predict(source, **predict_kwargs)
    repeated = deployed_model.predict(source, **predict_kwargs)
    _assert_public_segmentation_path(
        native,
        deployed,
        repeated,
        minimum_box_iou=0.95,
        # One of EC's 12 deterministic thresholded masks reaches 0.97619
        # after an otherwise passing raw-graph conversion; the other 11 are
        # exact. Keep the exception local to this public mask boundary.
        minimum_mask_iou=0.975 if family == "ec" else 0.98,
        maximum_score_error=0.01,
        allow_empty_masks=allow_empty_masks,
    )


def _panoptic_payload(result):
    """Return a validated panoptic map and row-aligned segment metadata."""
    from libreyolo.utils.results import Results

    assert isinstance(result, Results)
    assert result.panoptic is not None
    panoptic = result.panoptic.data.detach().cpu().numpy().astype(np.int64)
    assert panoptic.shape == result.orig_shape
    assert np.isfinite(panoptic).all()
    assert (panoptic >= 0).all()
    by_id = {}
    for raw_info in result.panoptic.segments_info:
        info = dict(raw_info)
        segment_id = int(info["id"])
        assert segment_id > 0 and segment_id not in by_id
        category_id = int(info["category_id"])
        assert category_id in result.names
        score = float(info["score"])
        assert np.isfinite(score)
        by_id[segment_id] = {
            "id": segment_id,
            "category_id": category_id,
            "isthing": bool(info["isthing"]),
            "score": score,
        }
    present_ids = set(np.unique(panoptic).tolist()) - {0}
    assert present_ids == set(by_id)
    assert present_ids, "public Core ML parity requires panoptic segments"
    info_rows = [by_id[segment_id] for segment_id in sorted(by_id)]
    masks = np.stack(
        [panoptic == info["id"] for info in info_rows],
        axis=0,
    )
    categories = np.asarray(
        [info["category_id"] for info in info_rows],
        dtype=np.int64,
    )
    isthing = np.asarray(
        [info["isthing"] for info in info_rows],
        dtype=bool,
    )
    scores = np.asarray(
        [info["score"] for info in info_rows],
        dtype=np.float64,
    )
    category_map = np.full(panoptic.shape, -1, dtype=np.int64)
    for info in info_rows:
        category_map[panoptic == info["id"]] = info["category_id"]
    return panoptic, info_rows, masks, categories, isthing, scores, category_map


def _assert_public_panoptic_path(
    model,
    artifact,
    *,
    imgsz: int,
    compute_units: str = "validated",
):
    """Gate whole panoptic segments independently of arbitrary segment IDs."""
    from libreyolo import LibreYOLO

    source = _public_non_square_source()
    predict_kwargs = {
        "conf": 0.05,
        "iou": 0.6,
        "max_det": PUBLIC_GENERIC_MAX_DET,
        "verbose": False,
    }
    native = model.predict(source, imgsz=imgsz, **predict_kwargs)
    deployed_model = LibreYOLO(artifact, compute_units=compute_units)
    deployed = deployed_model.predict(source, **predict_kwargs)
    repeated = deployed_model.predict(source, **predict_kwargs)
    assert native.orig_shape == deployed.orig_shape == repeated.orig_shape == (
        173,
        257,
    )
    assert native.names == deployed.names == repeated.names

    native_values = _panoptic_payload(native)
    deployed_values = _panoptic_payload(deployed)
    repeated_values = _panoptic_payload(repeated)
    (
        _native_map,
        _native_info,
        native_masks,
        native_categories,
        native_isthing,
        native_scores,
        native_category_map,
    ) = native_values
    (
        deployed_map,
        deployed_info,
        deployed_masks,
        deployed_categories,
        deployed_isthing,
        deployed_scores,
        deployed_category_map,
    ) = deployed_values
    repeated_map, repeated_info, *_ = repeated_values
    np.testing.assert_array_equal(deployed_map, repeated_map)
    assert deployed_info == repeated_info

    assert native_masks.shape[0] == deployed_masks.shape[0]
    mask_iou = _pairwise_mask_iou(native_masks, deployed_masks)
    category_mismatch = native_categories[:, None] != deployed_categories[None, :]
    thing_mismatch = native_isthing[:, None] != deployed_isthing[None, :]
    score_delta = np.abs(native_scores[:, None] - deployed_scores[None, :])
    cost = (
        1.0
        - mask_iou
        + 2.0 * category_mismatch
        + 2.0 * thing_mismatch
        + 0.01 * score_delta
    )
    rows, columns = linear_sum_assignment(cost)
    order = columns[np.argsort(rows)]
    assert np.array_equal(np.sort(rows), np.arange(native_masks.shape[0]))
    np.testing.assert_array_equal(native_categories, deployed_categories[order])
    np.testing.assert_array_equal(native_isthing, deployed_isthing[order])
    assert float(mask_iou[np.arange(native_masks.shape[0]), order].min()) >= 0.98
    assert float(np.abs(native_scores - deployed_scores[order]).max()) <= 0.01
    assert float(np.mean(native_category_map == deployed_category_map)) >= 0.99


def _assert_generic_public_path(
    model,
    artifact,
    *,
    family: str,
    task: str,
    imgsz: int,
    compute_units: str = "validated",
):
    """Dispatch every non-specialized trained row to a task-aware public gate."""
    if task == "classify":
        _assert_public_classification_path(
            model,
            artifact,
            imgsz=imgsz,
            compute_units=compute_units,
        )
    elif task == "detect":
        _assert_public_detection_path(
            model,
            artifact,
            family=family,
            imgsz=imgsz,
            compute_units=compute_units,
        )
    elif task == "semantic":
        _assert_public_semantic_path(
            model,
            artifact,
            imgsz=imgsz,
            minimum_agreement=0.99,
            compute_units=compute_units,
        )
    elif task == "depth":
        _assert_public_depth_path(
            model,
            artifact,
            family=family,
            imgsz=imgsz,
            compute_units=compute_units,
        )
    elif task == "restore":
        _assert_public_restore_path(
            model,
            artifact,
            imgsz=imgsz,
            compute_units=compute_units,
        )
    elif task == "segment":
        _assert_public_segment_path(
            model,
            artifact,
            family=family,
            imgsz=imgsz,
            compute_units=compute_units,
            # The released EoMT decoder can retain a low-confidence query whose
            # thresholded public mask is empty. Parity still requires the same
            # row, box, class, score, and empty mask from Core ML.
            allow_empty_masks=family == "eomt",
        )
    elif task == "pose":
        _assert_public_pose_path(
            model,
            artifact,
            family=family,
            imgsz=imgsz,
            compute_units=compute_units,
        )
    elif task == "obb":
        _assert_public_obb_path(
            model,
            artifact,
            imgsz=imgsz,
            compute_units=compute_units,
        )
    elif task == "panoptic":
        _assert_public_panoptic_path(
            model,
            artifact,
            imgsz=imgsz,
            compute_units=compute_units,
        )
    else:
        raise AssertionError(
            f"Trained Core ML case has no public-path gate: {family}/{task}."
        )


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
    compute_units="cpu_only",
):
    probes = _canonical_byte_probes(imgsz, imgsz)
    names, reference = _prepared_reference(model, family, task, imgsz, probes)
    output_path = tmp_path / f"{family}-{task}.mlpackage"
    artifact = model.export(
        format="coreml",
        imgsz=imgsz,
        output_path=str(output_path),
        compute_units=compute_units,
        half=half,
    )
    assert Path(artifact).is_dir()

    runtime, contract, artifact_names = _load_artifact(artifact)
    assert artifact_names == names
    actual = _artifact_outputs(runtime, contract, artifact_names, probes)
    # Set-prediction graphs may return an equally valid query permutation
    # after Core ML optimization. Keep one whole-query assignment shared by
    # every semantic output; never align scalar values independently.
    if family in UNORDERED_QUERY_FAMILIES:
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

    from libreyolo.export.support import get_support

    requested_compute_units = (
        "validated"
        if get_support(family, task, "coreml").tier == "validated"
        else "cpu_only"
    )
    model = LibreYOLO(weights, device="cpu")
    assert model.task == task
    artifact = _assert_model_artifact_parity(
        model,
        family,
        task,
        imgsz,
        tmp_path,
        compute_units=requested_compute_units,
    )
    del model
    gc.collect()
    model = LibreYOLO(weights, device="cpu")
    assert model.task == task
    if (weights, family, task, imgsz) in FLAGSHIP_PUBLIC_DETECT_CASES:
        _assert_flagship_public_detection_path(
            model,
            artifact,
            family=family,
            imgsz=imgsz,
            compute_units=requested_compute_units,
        )
    elif (family, task) == ("rtmdet", "segment"):
        _assert_rtmdet_segment_public_path(
            model,
            artifact,
            compute_units=requested_compute_units,
        )
    elif (family, task) == ("pidnet", "semantic"):
        _assert_public_semantic_path(
            model,
            artifact,
            imgsz=imgsz,
            minimum_agreement=0.995,
            require_multiple_classes=True,
            compute_units=requested_compute_units,
        )
    else:
        _assert_generic_public_path(
            model,
            artifact,
            family=family,
            task=task,
            imgsz=imgsz,
            compute_units=requested_compute_units,
        )


@pytest.mark.parametrize(("task", "imgsz"), [("semantic", 518), ("classify", 224)])
def test_coreml_dinov2_source_model_raw_and_public_parity(task, imgsz, tmp_path):
    """Prove the DINOv2 graph when the unpublished LibreYOLO head is unavailable."""
    from libreyolo import LibreDINOv2

    torch.manual_seed(0)
    model = LibreDINOv2(
        model_path=None,
        size="n",
        task=task,
        nb_classes=3,
        device="cpu",
    )
    model.model.eval()
    artifact = _assert_model_artifact_parity(
        model,
        "dinov2",
        task,
        imgsz,
        tmp_path,
        compute_units="cpu_only",
    )
    _assert_generic_public_path(
        model,
        artifact,
        family="dinov2",
        task=task,
        imgsz=imgsz,
        compute_units="cpu_only",
    )


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
        compute_units="validated",
    )
    staged = ct.models.MLModel(str(artifact), skip_model_load=True)
    staged_metadata = dict(staged.user_defined_metadata)
    assert staged_metadata["coreml_execution_profile_status"] == "validated"
    assert staged_metadata["coreml_profile_source_sha256"] == (
        "3ab3ef80216176a850b0c47877f310567c3749ec73e1becbfdacd9a2c13a7b39"
    )
    assert staged_metadata["coreml_profile_abi_sha256"] == (
        "07edd1cea201e3119f4100f678a218d046f1ecc6d4a7e925e3b3ba448334a891"
    )
    assert staged_metadata["coreml_validation_evidence_sha256"] == (
        "17e0912e9a3510f0484b76da41dd1f669b02b314966b59fd5cc992c4acb40c53"
    )
    del staged
    del model
    gc.collect()
    model = LibreYOLO("LibreBiRefNetl-matte.pt", device="cpu")

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
    _runtime, contract, artifact_names = _load_artifact(artifact)
    del _runtime
    gc.collect()
    all_runtime = ct.models.MLModel(
        str(artifact),
        compute_units=ct.ComputeUnit.ALL,
    )
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
    del all_runtime
    gc.collect()

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
    deployed_model = LibreYOLO(artifact)
    deployed = deployed_model.predict(source, verbose=False)[0].matte.array
    repeated = deployed_model.predict(source, verbose=False)[0].matte.array
    assert native.shape == deployed.shape == repeated.shape == (173, 257)
    assert np.isfinite(deployed).all()
    np.testing.assert_array_equal(deployed, repeated)
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
        half=False,
    )
    del model
    gc.collect()
    model = LibreYOLO("LibreDepthAnything3l-depth.pt", device="cpu")

    # Use an already-native square so this verifies host orchestration without
    # conflating the documented non-square fixed-stretch approximation with
    # DA3's native keep-aspect geometry.
    source = Image.fromarray(
        _canonical_byte_probes(504, 504)[0],
        mode="RGB",
    )
    gc.collect()
    torch.manual_seed(9182)
    native_result = model.predict(source, verbose=False)
    deployed_model = LibreYOLO(artifact)
    torch.manual_seed(9182)
    deployed_result = deployed_model.predict(source, verbose=False)
    torch.manual_seed(9182)
    repeated_result = deployed_model.predict(source, verbose=False)
    assert native_result.depth_map is not None
    assert deployed_result.depth_map is not None
    assert repeated_result.depth_map is not None
    native = native_result.depth_map.data.detach().cpu().numpy()
    deployed = deployed_result.depth_map.data.detach().cpu().numpy()
    repeated = repeated_result.depth_map.data.detach().cpu().numpy()
    assert native.shape == deployed.shape == repeated.shape == (504, 504)
    assert np.isfinite(deployed).all()
    np.testing.assert_array_equal(deployed, repeated)
    scale = max(float(np.abs(native).max()), 1e-8)
    assert float(np.abs(native - deployed).max()) / scale <= 3e-4


@pytest.mark.parametrize(
    ("alias", "image_size"),
    [("owlv2-b16", 960), ("owlv2-l14", 1008)],
)
def test_coreml_owlv2_frozen_vocabulary_raw_and_public_path(
    alias,
    image_size,
    tmp_path,
):
    """Gate the text-free graph, exact host geometry, and named raw outputs."""
    from libreyolo import LibreOpenVocab, LibreYOLO
    from libreyolo.export.coreml_owlv2 import (
        build_owlv2_frozen_coreml_adapter,
        postprocess_owlv2_coreml_outputs,
        preprocess_owlv2_coreml_image,
    )

    labels = ["person", "dog", "remote control"]
    model = LibreOpenVocab(alias, device="cpu")
    model.set_classes(labels)
    adapter = build_owlv2_frozen_coreml_adapter(
        model.model,
        model.processor,
        size=model.size,
        names=model.names,
    ).eval()

    probes = _canonical_byte_probes(image_size, image_size)
    tensors = tuple(_as_canonical_tensor(probe) for probe in probes)
    with torch.no_grad():
        reference = [
            [
                tensor.detach().cpu().numpy()
                for tensor in adapter(value)
            ]
            for value in tensors
        ]
    source = Image.fromarray(
        _canonical_byte_probes(317, 191)[0],
        mode="RGB",
    )
    prepared = preprocess_owlv2_coreml_image(source, image_size=image_size)
    with torch.no_grad():
        logits, boxes = adapter(prepared)
    expected = postprocess_owlv2_coreml_outputs(
        logits,
        boxes,
        original_size=source.size,
        conf=0.001,
        max_det=20,
    )

    artifact = model.export(
        format="coreml",
        half=False,
        output_path=str(tmp_path / f"{alias}.mlpackage"),
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
    del runtime, adapter, model
    gc.collect()

    # Exercise the default ALL compute-unit route and the non-square public
    # preprocessing/postprocessing contract independently of the raw CPU gate.
    native_model = LibreOpenVocab(alias, device="cpu")
    native_model.set_classes(labels)
    native = native_model.predict(
        source,
        conf=0.001,
        max_det=20,
        verbose=False,
    )
    deployed_model = LibreYOLO(artifact)
    deployed = deployed_model.predict(
        source,
        conf=0.001,
        max_det=20,
        verbose=False,
    )
    repeated = deployed_model.predict(
        source,
        conf=0.001,
        max_det=20,
        verbose=False,
    )
    assert expected["num_detections"] > 0
    assert deployed.names == repeated.names == dict(enumerate(labels))
    assert len(deployed.boxes) == expected["num_detections"]
    assert len(repeated.boxes) == expected["num_detections"]
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
    torch.testing.assert_close(
        repeated.boxes.xyxy.cpu(),
        deployed.boxes.xyxy.cpu(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        repeated.boxes.conf.cpu(),
        deployed.boxes.conf.cpu(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        repeated.boxes.cls.cpu(),
        deployed.boxes.cls.cpu(),
        rtol=0.0,
        atol=0.0,
    )
    _match_public_detections(
        native,
        deployed,
        repeated,
        minimum_iou=0.9999,
        maximum_score_error=1e-4,
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

    artifact = model.export(
        format="coreml",
        half=False,
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
    del runtime, adapter, source_adapter, model
    gc.collect()

    native_model = LibreOpenVocab("omdet-turbo", device="cpu")
    native_model.set_classes(labels)
    native = native_model.predict(
        source,
        conf=0.3,
        iou=0.5,
        max_det=300,
        verbose=False,
    )
    deployed_model = LibreYOLO(artifact)
    deployed = deployed_model.predict(
        source,
        conf=0.3,
        iou=0.5,
        max_det=300,
        verbose=False,
    )
    repeated = deployed_model.predict(
        source,
        conf=0.3,
        iou=0.5,
        max_det=300,
        verbose=False,
    )
    assert expected["num_detections"] > 0
    assert deployed.names == repeated.names == dict(enumerate(labels))
    assert len(deployed.boxes) == expected["num_detections"]
    assert len(repeated.boxes) == expected["num_detections"]
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
    torch.testing.assert_close(
        repeated.boxes.xyxy.cpu(),
        deployed.boxes.xyxy.cpu(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        repeated.boxes.conf.cpu(),
        deployed.boxes.conf.cpu(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        repeated.boxes.cls.cpu(),
        deployed.boxes.cls.cpu(),
        rtol=0.0,
        atol=0.0,
    )
    _match_public_detections(
        native,
        deployed,
        repeated,
        minimum_iou=0.999,
        maximum_score_error=1e-4,
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
    from libreyolo.export.support import get_support

    requested_compute_units = (
        "validated"
        if get_support(family, "classify", "coreml").tier == "validated"
        else "cpu_only"
    )

    model = LibreYOLO(weights, device="cpu")
    labels = ["person", "building", "vehicle"]
    class_kwargs = {
        "templates": ["a photo of a {}."],
    }
    if family == "siglip2":
        # SigLIP2's independent sigmoid activation is materially different
        # from the default softmax and must survive artifact metadata loading.
        class_kwargs["multi_label"] = True
    model.set_classes(labels, **class_kwargs)

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

    artifact = model.export(
        format="coreml",
        imgsz=imgsz,
        output_path=str(tmp_path / f"{family}-classify.mlpackage"),
        compute_units=requested_compute_units,
    )
    runtime, contract, artifact_names = _load_artifact(artifact)
    assert artifact_names == names
    actual = _artifact_outputs(runtime, contract, names, probes)
    _assert_parity(names, reference, actual)
    del runtime, prepared, frozen, model
    gc.collect()
    model = LibreYOLO(weights, device="cpu")
    model.set_classes(labels, **class_kwargs)
    _assert_public_classification_path(
        model,
        artifact,
        imgsz=imgsz,
        compute_units=requested_compute_units,
    )


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
    source = _public_non_square_source()
    native = model.predict(
        source,
        face_boxes=[[0.0, 0.0, 257.0, 173.0]],
    )
    artifact = _assert_model_artifact_parity(
        model,
        "l2cs",
        "gaze",
        448,
        tmp_path,
    )
    del model
    gc.collect()
    deployed_model = LibreYOLO(artifact)
    deployed = deployed_model.predict(source, verbose=False)
    repeated = deployed_model.predict(source, verbose=False)
    assert native.orig_shape == deployed.orig_shape == repeated.orig_shape == (
        173,
        257,
    )
    assert native.gaze is not None
    assert deployed.gaze is not None
    assert repeated.gaze is not None
    native_gaze = native.gaze.data.detach().cpu().numpy()
    deployed_gaze = deployed.gaze.data.detach().cpu().numpy()
    repeated_gaze = repeated.gaze.data.detach().cpu().numpy()
    assert native_gaze.shape == deployed_gaze.shape == repeated_gaze.shape == (
        1,
        2,
    )
    np.testing.assert_array_equal(deployed_gaze, repeated_gaze)
    np.testing.assert_allclose(
        deployed_gaze,
        native_gaze,
        rtol=0.0,
        atol=5e-4,
    )


@pytest.mark.parametrize("size", ["r18", "r34", "r50", "r101", "r152"])
def test_coreml_l2cs_generated_model_raw_and_public_parity(size, tmp_path):
    """Prove the gaze graph without using restricted Gaze360-derived weights."""
    from libreyolo import LibreL2CS, LibreYOLO

    torch.manual_seed(20260730)
    model = LibreL2CS(
        None,
        size=size,
        num_bins=90,
        device="cpu",
    )
    source = _public_non_square_source()
    face_boxes = [[0.0, 0.0, 257.0, 173.0]]
    native = model.predict(source, face_boxes=face_boxes)
    artifact = _assert_model_artifact_parity(
        model,
        "l2cs",
        "gaze",
        448,
        tmp_path,
    )
    del model
    gc.collect()

    deployed_model = LibreYOLO(artifact)
    deployed = deployed_model.predict(
        source,
        verbose=False,
    )
    repeated = deployed_model.predict(
        source,
        verbose=False,
    )
    assert native.orig_shape == deployed.orig_shape == repeated.orig_shape == (
        173,
        257,
    )
    assert native.gaze is not None
    assert deployed.gaze is not None
    assert repeated.gaze is not None
    native_gaze = native.gaze.data.detach().cpu().numpy()
    deployed_gaze = deployed.gaze.data.detach().cpu().numpy()
    repeated_gaze = repeated.gaze.data.detach().cpu().numpy()
    assert native_gaze.shape == deployed_gaze.shape == repeated_gaze.shape == (
        1,
        2,
    )
    assert np.isfinite(deployed_gaze).all()
    np.testing.assert_array_equal(deployed_gaze, repeated_gaze)
    np.testing.assert_allclose(
        deployed_gaze,
        native_gaze,
        rtol=0.0,
        atol=5e-4,
    )


def test_coreml_fomo_synthetic_trained_parity(tmp_path):
    """Use generated tensors so point export does not depend on hosted weights."""
    from libreyolo import LibreFOMO, LibreYOLO
    from libreyolo.export.support import get_support

    requested_compute_units = (
        "validated"
        if get_support("fomo", "point", "coreml").tier == "validated"
        else "cpu_only"
    )

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
    source = _public_non_square_source()
    predict_kwargs = {
        "conf": 0.1,
        "max_det": 300,
        "verbose": False,
    }
    native = model.predict(source, imgsz=96, **predict_kwargs)
    artifact = _assert_model_artifact_parity(
        model,
        "fomo",
        "point",
        96,
        tmp_path,
        compute_units=requested_compute_units,
    )
    del model
    gc.collect()
    deployed_model = LibreYOLO(
        artifact,
        compute_units=requested_compute_units,
    )
    deployed = deployed_model.predict(source, **predict_kwargs)
    repeated = deployed_model.predict(source, **predict_kwargs)
    assert native.orig_shape == deployed.orig_shape == repeated.orig_shape == (
        173,
        257,
    )
    assert native.names == deployed.names == repeated.names
    assert native.points is not None
    assert deployed.points is not None
    assert repeated.points is not None
    native_points = native.points.data.detach().cpu().numpy().astype(np.float64)
    deployed_points = (
        deployed.points.data.detach().cpu().numpy().astype(np.float64)
    )
    repeated_points = (
        repeated.points.data.detach().cpu().numpy().astype(np.float64)
    )
    assert native_points.shape == deployed_points.shape == repeated_points.shape
    assert native_points.ndim == 2 and native_points.shape[1] == 4
    assert native_points.shape[0] > 0
    assert np.isfinite(native_points).all()
    assert np.isfinite(deployed_points).all()
    np.testing.assert_array_equal(deployed_points, repeated_points)

    class_mismatch = (
        native_points[:, None, 2] != deployed_points[None, :, 2]
    )
    xy_distance = np.linalg.norm(
        native_points[:, None, :2] - deployed_points[None, :, :2],
        axis=2,
    )
    score_delta = np.abs(
        native_points[:, None, 3] - deployed_points[None, :, 3]
    )
    rows, columns = linear_sum_assignment(
        xy_distance + 1_000.0 * class_mismatch + score_delta
    )
    order = columns[np.argsort(rows)]
    np.testing.assert_array_equal(
        native_points[:, 2],
        deployed_points[order, 2],
    )
    assert float(
        np.abs(native_points[:, :2] - deployed_points[order, :2]).max()
    ) <= 1e-3
    assert float(
        np.abs(native_points[:, 3] - deployed_points[order, 3]).max()
    ) <= 5e-5


def test_coreml_picosam3_trained_roi_component_parity(tmp_path):
    """Validate the fixed one-ROI graph and sequential public placement."""
    from libreyolo import LibrePicoSAM3, LibreYOLO

    model = LibrePicoSAM3(size="pico", device="cpu")
    # Two visibly occupied prompts exercise sequential fixed-batch ROI calls,
    # exact host crop/resize placement, and one fresh public artifact load.
    source = _public_non_square_source()
    boxes = [
        [70, 8, 128, 62],
        [78, 85, 215, 172],
    ]
    native = model.predict(source, bboxes=boxes, conf=0.0, max_det=2)
    artifact = _assert_model_artifact_parity(
        model,
        "picosam3",
        "segment",
        96,
        tmp_path,
    )
    del model
    gc.collect()
    deployed_model = LibreYOLO(artifact)
    deployed = deployed_model.predict(
        source,
        bboxes=boxes,
        conf=0.0,
        max_det=2,
    )
    repeated = deployed_model.predict(
        source,
        bboxes=boxes,
        conf=0.0,
        max_det=2,
    )
    for result in (native, deployed, repeated):
        assert _segmentation_payload(result)[0].shape[0] == len(boxes)
    _assert_public_segmentation_path(
        native,
        deployed,
        repeated,
        minimum_box_iou=0.995,
        minimum_mask_iou=0.995,
        maximum_score_error=5e-4,
    )


@pytest.mark.parametrize(
    "weights",
    ["LibrePPOCRt-ocr.pt", "LibrePPOCRl-ocr.pt"],
)
def test_coreml_ppocr_trained_multifunction_parity(weights, tmp_path):
    """Gate both named functions and the deterministic public OCR pipeline."""
    from libreyolo import LibreYOLO
    from libreyolo.export.coreml_ppocr import (
        PPOCR_COREML_DETECTOR_INPUT,
        PPOCR_COREML_DETECTOR_OUTPUT,
        PPOCR_COREML_RECOGNIZER_INPUT,
        PPOCR_COREML_RECOGNIZER_OUTPUT,
        validate_ppocr_coreml_metadata,
    )

    model = LibreYOLO(weights, device="cpu")
    det_inputs = [
        torch.linspace(-2.0, 2.0, 3 * 64 * 96).reshape(1, 3, 64, 96),
        torch.linspace(2.0, -2.0, 3 * 96 * 64).reshape(1, 3, 96, 64),
    ]
    rec_inputs = [
        torch.linspace(-1.0, 1.0, 3 * 48 * 320).reshape(1, 3, 48, 320),
        torch.linspace(1.0, -1.0, 2 * 3 * 48 * 641).reshape(2, 3, 48, 641),
    ]
    inverse_det_input = -det_inputs[0]
    inverse_rec_input = -rec_inputs[0]
    with torch.inference_mode():
        expected_det = [model.model.det(value).cpu().numpy() for value in det_inputs]
        expected_rec = [model.model.rec(value).cpu().numpy() for value in rec_inputs]
        expected_inverse_det = model.model.det(inverse_det_input).cpu().numpy()
        expected_inverse_rec = model.model.rec(inverse_rec_input).cpu().numpy()

    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "ocr20"
        / "images"
        / "val"
        / "ocr_01.png"
    )
    native = model.predict(
        str(fixture),
        conf=0.3,
        imgsz=960,
        rec_batch=6,
    )

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

    actual_inverse_det = np.asarray(
        detector.predict(
            {
                PPOCR_COREML_DETECTOR_INPUT: np.ascontiguousarray(
                    inverse_det_input.numpy(),
                )
            }
        )[PPOCR_COREML_DETECTOR_OUTPUT]
    )
    actual_inverse_rec = np.asarray(
        recognizer.predict(
            {
                PPOCR_COREML_RECOGNIZER_INPUT: np.ascontiguousarray(
                    inverse_rec_input.numpy(),
                )
            }
        )[PPOCR_COREML_RECOGNIZER_OUTPUT]
    )
    # The same-shape inversions use the common relative sensitivity and 100x
    # conversion-error-margin gate, so constant or nearly constant functions
    # cannot receive a false pass.
    _assert_parity(
        [PPOCR_COREML_DETECTOR_OUTPUT],
        [[expected_det[0]], [expected_inverse_det]],
        [[actual_det[0]], [actual_inverse_det]],
    )
    _assert_parity(
        [PPOCR_COREML_RECOGNIZER_OUTPUT],
        [[expected_rec[0]], [expected_inverse_rec]],
        [[actual_rec[0]], [actual_inverse_rec]],
    )

    # Release the direct CPU-only function proxies before loading the same
    # package once through the user-facing default compute-unit planner.
    del detector, recognizer, model
    gc.collect()

    deployed_model = LibreYOLO(artifact)
    deployed = deployed_model.predict(
        str(fixture),
        conf=0.3,
        imgsz=960,
        rec_batch=6,
    )
    repeated = deployed_model.predict(
        str(fixture),
        conf=0.3,
        imgsz=960,
        rec_batch=6,
    )
    assert native.orig_shape == deployed.orig_shape == repeated.orig_shape
    assert native.names == deployed.names == repeated.names
    assert native.ocr is not None
    assert deployed.ocr is not None
    assert repeated.ocr is not None
    native_ocr = native.ocr.numpy()
    deployed_ocr = deployed.ocr.numpy()
    repeated_ocr = repeated.ocr.numpy()
    assert len(native_ocr) == len(deployed_ocr) == len(repeated_ocr) > 0
    assert native_ocr.texts == deployed_ocr.texts == repeated_ocr.texts
    assert np.isfinite(deployed_ocr.data).all()
    assert np.isfinite(deployed_ocr.conf).all()
    assert np.isfinite(deployed_ocr.det_conf).all()

    # Repeated decode is exact because both Core ML functions and every host
    # contour/crop/CTC operation receive identical inputs.
    np.testing.assert_array_equal(deployed_ocr.data, repeated_ocr.data)
    np.testing.assert_array_equal(deployed_ocr.conf, repeated_ocr.conf)
    np.testing.assert_array_equal(deployed_ocr.det_conf, repeated_ocr.det_conf)

    # DB contouring is discrete; a one-pixel probability-boundary change can
    # move an expanded quad by up to two source pixels. Recognition scores use
    # the repository's existing pinned OCR golden tolerance.
    assert float(np.abs(native_ocr.data - deployed_ocr.data).max()) <= 2.0
    assert float(np.abs(native_ocr.conf - deployed_ocr.conf).max()) <= 1e-3
    assert (
        float(np.abs(native_ocr.det_conf - deployed_ocr.det_conf).max())
        <= 1e-3
    )


def test_coreml_yolonas_synthetic_trained_parity(tmp_path):
    """Exercise the decoded detection contract without hosted weights."""
    from libreyolo import LibreYOLONAS
    from libreyolo.export.support import get_support
    from libreyolo.models.yolonas.loss import PPYoloELoss

    requested_compute_units = (
        "validated"
        if get_support("yolonas", "detect", "coreml").tier == "validated"
        else "cpu_only"
    )
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

    artifact = _assert_model_artifact_parity(
        model,
        "yolonas",
        "detect",
        96,
        tmp_path,
        compute_units=requested_compute_units,
    )
    _assert_public_detection_path(
        model,
        artifact,
        family="yolonas",
        imgsz=96,
        compute_units=requested_compute_units,
    )


def test_coreml_yolonas_pose_synthetic_parity(tmp_path):
    """Exercise YOLO-NAS pose tensor and public keypoint contracts."""
    from libreyolo import LibreYOLONAS
    from libreyolo.export.support import get_support

    requested_compute_units = (
        "validated"
        if get_support("yolonas", "pose", "coreml").tier == "validated"
        else "cpu_only"
    )
    torch.manual_seed(20260730)
    model = LibreYOLONAS(
        None,
        size="n",
        task="pose",
        nb_classes=1,
        device="cpu",
    )
    model.model.eval()
    # Keep all decoded branches measurably input-sensitive without relying on
    # the incompatible upstream pose checkpoint.
    with torch.no_grad():
        for head in (
            model.model.heads.head1,
            model.model.heads.head2,
            model.model.heads.head3,
        ):
            head.reg_pred.weight.mul_(20.0)
            head.cls_pred.weight.mul_(20.0)
            head.pose_pred.weight.mul_(20.0)
    artifact = _assert_model_artifact_parity(
        model,
        "yolonas",
        "pose",
        96,
        tmp_path,
        compute_units=requested_compute_units,
    )
    _assert_public_pose_path(
        model,
        artifact,
        family="yolonas",
        imgsz=96,
        compute_units=requested_compute_units,
    )


def test_coreml_yolo9_p2_permissive_transfer_parity(tmp_path):
    """Cover the extra stride-4 branch with a verified compatible checkpoint."""
    from libreyolo import LibreYOLO, LibreYOLO9P2
    from libreyolo.export.support import get_support
    from libreyolo.utils.download import download_weights

    requested_compute_units = (
        "validated"
        if get_support("yolo9_p2", "detect", "coreml").tier == "validated"
        else "cpu_only"
    )
    torch.manual_seed(20260728)
    model = LibreYOLO9P2(None, size="t", device="cpu")
    weights_path = Path(model._resolve_weights_path("LibreYOLO9t.pt"))
    if not weights_path.exists():
        download_weights(str(weights_path), model.size)
    with weights_path.open("rb") as weights_file:
        digest = hashlib.file_digest(weights_file, "sha256").hexdigest()
    assert digest == "b4d7e93f9e0393830fb42e6135c0e3464b2673b05e5ecf4b7f2374ec18e39eb2"
    model._load_transfer_weights(weights_path)
    assert not model.model.training

    source = _public_non_square_source()
    predict_kwargs = {
        # The added P2 tower is intentionally random in this transfer fixture.
        # A 1e-3 floor retains three stable, input-derived detections while
        # excluding near-zero class ties that have no production meaning.
        "conf": 1e-3,
        "iou": 0.45,
        "max_det": PUBLIC_GENERIC_MAX_DET,
        "verbose": False,
    }
    native = model.predict(source, imgsz=640, **predict_kwargs)
    artifact = _assert_model_artifact_parity(
        model,
        "yolo9_p2",
        "detect",
        640,
        tmp_path,
        compute_units=requested_compute_units,
    )
    del model
    gc.collect()
    deployed_model = LibreYOLO(
        artifact,
        compute_units=requested_compute_units,
    )
    deployed = deployed_model.predict(source, **predict_kwargs)
    repeated = deployed_model.predict(source, **predict_kwargs)
    _match_public_detections(
        native,
        deployed,
        repeated,
        minimum_iou=0.999,
        maximum_score_error=1e-4,
    )
