"""Canonical export support tiers for model, task, and format combinations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

from ..tasks import TASKS


Tier = Literal["validated", "experimental", "blocked"]
EXPORT_FORMATS = (
    "onnx",
    "torchscript",
    "tensorrt",
    "openvino",
    "ncnn",
    "tflite",
    "coreml",
)


@dataclass(frozen=True)
class SupportEntry:
    """Support status and user-facing context for one export combination."""

    tier: Tier
    reason: str = ""
    since: str | None = None
    constraint: str | None = None


SUPPORT: dict[tuple[str, str, str], SupportEntry] = {}


def _add(
    tier: Tier,
    families: tuple[str, ...],
    tasks: tuple[str, ...],
    formats: tuple[str, ...],
    *,
    reason: str = "",
    since: str | None = None,
    constraint: str | None = None,
) -> None:
    entry = SupportEntry(tier, reason, since, constraint)
    keys = [
        (family, task, fmt) for family in families for task in tasks for fmt in formats
    ]
    seen: set[tuple[str, str, str]] = set()
    duplicates = []
    for key in keys:
        if key in SUPPORT or key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        rendered = ", ".join(repr(key) for key in duplicates)
        raise ValueError(f"Duplicate export support entries: {rendered}")
    for key in keys:
        SUPPORT[key] = entry


# Existing parity-backed paths. New validated rows must land with a parity test.
_add(
    "validated",
    ("yolo9",),
    ("detect",),
    ("onnx", "torchscript", "ncnn", "tflite"),
    since="1.3",
)
_add(
    "validated",
    ("yolo9",),
    ("detect",),
    ("tensorrt", "openvino"),
    reason=(
        "Runtime parity coverage lives in tests/e2e/test_tensorrt.py and "
        "tests/e2e/test_openvino.py."
    ),
    since="1.3",
)
_add(
    "blocked",
    ("yolo9",),
    ("segment",),
    EXPORT_FORMATS,
    reason="YOLO9 segmentation export is not supported; YOLO9 is detection-only in LibreYOLO.",
)
_add("validated", ("yolo9_p2",), ("detect",), ("onnx",), since="1.3")
_add(
    "validated",
    ("rfdetr",),
    ("detect",),
    ("onnx", "torchscript"),
    since="1.3",
)
_add(
    "validated",
    ("rfdetr",),
    ("detect",),
    ("tensorrt", "openvino"),
    reason=(
        "Runtime parity coverage lives in tests/e2e/test_tensorrt.py and "
        "tests/e2e/test_openvino.py."
    ),
    since="1.3",
)
_add(
    "experimental",
    ("rfdetr",),
    ("detect",),
    ("tflite",),
    reason=(
        "The RF-DETR converter path is available, but the project does not "
        "yet have a runtime parity test for the generated LiteRT artifact."
    ),
)
_add(
    "validated",
    ("mobilenetv4", "convnext", "efficientnetv2", "resnet"),
    ("classify",),
    ("onnx", "torchscript"),
    since="1.3",
)
_add(
    "validated",
    ("clip", "siglip2"),
    ("classify",),
    ("onnx",),
    since="1.3",
    constraint="frozen-class labels and fixed input resolution",
)
_add(
    "blocked",
    ("clip", "siglip2"),
    ("classify",),
    tuple(fmt for fmt in EXPORT_FORMATS if fmt != "onnx"),
    reason=(
        "Frozen-class vision-language export is ONNX-only in v1; re-export "
        "the frozen ONNX graph for a different deployment runtime."
    ),
)
_add(
    "blocked",
    ("dinov2",),
    ("classify",),
    tuple(fmt for fmt in EXPORT_FORMATS if fmt != "onnx"),
    reason="LibreDINOv2 classify export currently supports ONNX only.",
)
_add(
    "blocked",
    ("birefnet",),
    ("matte",),
    ("ncnn",),
    reason=(
        "BiRefNet's decoder requires torchvision deformable convolution, "
        "which PNNX/NCNN cannot lower to a runnable graph."
    ),
)

# Explicitly permitted but not yet parity-validated combinations.
_add(
    "blocked",
    ("rfdetr",),
    ("segment",),
    ("tflite",),
    reason=(
        "onnx2tf 2.4.x assigns an invalid NHWC layout to the segmentation-head "
        "Einsum (78 channels versus the required 256), so conversion fails."
    ),
)
_add(
    "experimental",
    ("birefnet",),
    ("matte",),
    ("onnx",),
    reason=(
        "The opset-19 DeformConv graph exports, but ONNX Runtime's CPU "
        "provider has no DeformConv implementation for runtime parity."
    ),
)
_add(
    "validated",
    ("birefnet",),
    ("matte",),
    ("torchscript",),
    since="1.4",
    constraint="fixed 1024x1024 input",
)
_add(
    "blocked",
    ("rfdetr",),
    ("pose",),
    ("tflite",),
    reason=(
        "RF-DETR pose-x TFLite conversion exceeded the CPU timebox and 8 GB "
        "working memory without producing an artifact on this toolchain."
    ),
)
_add(
    "experimental",
    ("yolox", "yolo9", "rtdetr", "rfdetr"),
    ("detect",),
    ("coreml",),
    reason="Conversion is available, but runtime parity requires a macOS runner.",
)
_add(
    "experimental",
    ("dinov2", "eomt", "pidnet"),
    ("semantic",),
    ("tensorrt", "openvino"),
    reason=(
        "The dense-logits contract is wired, but the project has not yet "
        "recorded TensorRT or OpenVINO runtime parity for these families."
    ),
)
_add(
    "validated",
    ("pidnet",),
    ("semantic",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "validated",
    ("l2cs",),
    ("gaze",),
    ("onnx",),
    since="1.4",
    constraint="head-only contract: each input image is one face crop",
)
_add(
    "validated",
    ("nafnet",),
    ("restore",),
    ("onnx", "torchscript", "ncnn"),
    since="1.4",
    constraint="fixed-resolution export canvas",
)
_add(
    "blocked",
    ("nafnet",),
    ("restore",),
    ("tflite",),
    reason=(
        "onnx2tf 2.4.x converts the fixed-canvas graph, but LiteRT fails at "
        "invoke time because an internal input tensor lacks data."
    ),
)
_add(
    "validated",
    ("realesrgan",),
    ("restore",),
    ("onnx", "torchscript", "ncnn"),
    since="1.4",
    constraint="ONNX supports dynamic spatial input; TorchScript and NCNN are fixed-canvas",
)
_add(
    "validated",
    ("realesrgan",),
    ("restore",),
    ("tflite",),
    since="1.4",
    constraint="fixed-resolution export canvas",
)
_add(
    "blocked",
    ("depth_anything",),
    ("depth",),
    ("tflite",),
    reason=(
        "onnx2tf 2.4.x converts the DINOv2 depth graph, but LiteRT rejects "
        "a generated FILL node because its dimensions are invalid."
    ),
)
_add(
    "validated",
    ("yolox",),
    ("detect",),
    ("tflite",),
    since="1.4",
)
_add(
    "validated",
    ("pidnet",),
    ("semantic",),
    ("ncnn",),
    since="1.4",
)
_add(
    "validated",
    ("fomo",),
    ("point",),
    ("ncnn",),
    since="1.4",
)
_add(
    "validated",
    ("picosam3",),
    ("segment",),
    ("onnx",),
    since="1.4",
    constraint="raw fixed-96 ROI contract: roi_image -> mask_logits",
)
_add(
    "blocked",
    ("picosam3",),
    ("segment",),
    tuple(fmt for fmt in EXPORT_FORMATS if fmt != "onnx"),
    reason="PicoSAM3 currently exports its raw ROI CNN through ONNX only.",
)
_add(
    "experimental",
    ("fomo",),
    ("point",),
    ("tensorrt", "openvino"),
    reason=(
        "The raw-heatmap contract is wired, but the project has not yet "
        "recorded TensorRT or OpenVINO runtime parity for FOMO."
    ),
)
_add(
    "validated",
    ("zipdepth",),
    ("depth",),
    ("onnx", "torchscript", "ncnn"),
    since="1.4",
    constraint="fixed-resolution export canvas",
)
_add(
    "blocked",
    ("zipdepth",),
    ("depth",),
    ("tflite",),
    reason=(
        "onnx2tf 2.4.x flatbuffer-direct conversion does not support the "
        "edge-mode Pad operation in ZipDepth's convex upsampler."
    ),
)
_add(
    "validated",
    ("picodet",),
    ("detect",),
    ("onnx", "torchscript", "ncnn"),
    since="1.4",
)
_add(
    "validated",
    ("yolo2", "yolo3", "yolo4"),
    ("detect",),
    ("ncnn",),
    since="1.4",
)
_add(
    "blocked",
    ("yolo2", "yolo3"),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf 2.4.x leaves an unresolved ONNX_CONCAT custom operation; "
        "LiteRT cannot prepare the converted detector graph."
    ),
)
_add(
    "blocked",
    ("yolo4",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf 2.4.x produces an invalid CONV_2D channel layout for YOLO4; "
        "LiteRT fails while allocating tensors."
    ),
)
_add(
    "validated",
    ("yolo7",),
    ("detect",),
    ("ncnn",),
    since="1.4",
)
_add(
    "blocked",
    ("yolo7",),
    ("detect",),
    ("tflite",),
    reason=(
        "The converted LiteRT graph changes decoded box coordinates beyond "
        "the detector parity tolerance."
    ),
)
_add(
    "validated",
    ("yolo9_e2e", "yolo9_p2", "yolox"),
    ("detect",),
    ("ncnn",),
    since="1.4",
)
_add(
    "validated",
    ("yolo1",),
    ("detect",),
    ("ncnn",),
    since="1.4",
    constraint="fixed 448x448 input",
)
_add(
    "validated",
    ("yolonas",),
    ("detect", "pose"),
    ("ncnn",),
    since="1.4",
)
_add(
    "validated",
    ("yolo2", "yolo3", "yolo4"),
    ("detect",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "validated",
    ("yolo1", "yolo7", "yolo9_e2e", "yolox"),
    ("detect",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "validated",
    ("yolo9_p2",),
    ("detect",),
    ("torchscript",),
    since="1.4",
)
_add(
    "validated",
    ("yolonas",),
    ("detect", "pose"),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "blocked",
    ("rtmdet",),
    ("detect",),
    ("ncnn",),
    reason=(
        "PNNX 20260526 reports an unregistered nn.Conv2d layer and leaves the "
        "RTMDet NCNN graph without usable input blobs."
    ),
)
_add(
    "validated",
    ("rtmdet",),
    ("detect",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "blocked",
    ("rtmdet",),
    ("segment",),
    EXPORT_FORMATS,
    reason=(
        "RTMDet-Ins export is not supported yet; the dynamic-kernel mask "
        "decode has no exported-runtime contract. Use native PyTorch "
        "inference for task='segment'."
    ),
)
_add(
    "experimental",
    ("swinir",),
    ("restore",),
    ("onnx",),
    reason=(
        "The exported graph runs at a fixed canvas. Inputs smaller than the "
        "canvas are padded before the transformer, and the window attention "
        "and layer norms see that padding: measured drift against native "
        "inference reaches many grey levels. Match the exported canvas size "
        "for best fidelity."
    ),
)
_add(
    "validated",
    ("dfine", "deim", "deimv2", "ec", "rtdetr", "rtdetrv2", "rtdetrv4"),
    ("detect",),
    ("torchscript",),
    since="1.4",
)
_add(
    "validated",
    ("dfine", "ec", "rtdetr"),
    ("detect",),
    ("onnx",),
    since="1.4",
)
_add(
    "experimental",
    ("deim",),
    ("detect",),
    ("onnx",),
    reason="Runtime parity leaves 8.7% of selected boxes outside tolerance.",
)
_add(
    "experimental",
    ("deimv2",),
    ("detect",),
    ("onnx",),
    reason="ONNX top-k selection changes score and box queries beyond tolerance.",
)
_add(
    "experimental",
    ("rtdetrv2",),
    ("detect",),
    ("onnx",),
    reason="Runtime parity leaves 9% of selected boxes outside tolerance.",
)
_add(
    "experimental",
    ("rtdetrv4",),
    ("detect",),
    ("onnx",),
    reason="Runtime parity leaves 7.3% of selected boxes outside tolerance.",
)
_add(
    "validated",
    ("dfine",),
    ("segment",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "validated",
    ("ec",),
    ("pose", "segment"),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="fixed 640x640 input",
)
_add(
    "validated",
    ("rfdetr",),
    ("segment", "pose", "obb"),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="fixed task-native input resolution",
)
_add(
    "validated",
    ("dinov2",),
    ("semantic",),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="fixed 518x518 input",
)
_add(
    "validated",
    ("dinov2",),
    ("classify",),
    ("onnx",),
    since="1.4",
    constraint="fixed 224x224 input",
)
_add(
    "validated",
    ("eomt",),
    ("semantic",),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="fixed 512x512 input",
)
_add(
    "blocked",
    ("fomo",),
    ("point",),
    ("coreml",),
    reason="The CoreML wrapper does not implement the raw point-heatmap contract.",
)
_add(
    "validated",
    ("fomo",),
    ("point",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "validated",
    ("depth_anything",),
    ("depth",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "blocked",
    ("depth_anything",),
    ("depth",),
    ("ncnn",),
    reason=(
        "PNNX 20260526 reports unsupported batch-index reshapes in the DINOv2 "
        "transformer graph; the produced NCNN artifact fails numeric parity."
    ),
)
_add(
    "validated",
    ("mobilenetv4", "convnext", "efficientnetv2", "resnet"),
    ("classify",),
    ("ncnn", "tflite"),
    since="1.4",
)
_add(
    "blocked",
    ("fomo",),
    ("point",),
    ("tflite",),
    reason=(
        "onnx2tf 2.4.x produces an invalid depthwise-convolution graph for the "
        "static SAME-padded FOMO backbone on this toolchain."
    ),
)
_add(
    "validated",
    ("pidnet",),
    ("semantic",),
    ("tflite",),
    since="1.4",
)
_add(
    "blocked",
    ("dinov2", "eomt"),
    ("semantic",),
    ("ncnn", "tflite"),
    reason=(
        "The dense-logits runtime contract is implemented, but this transformer "
        "graph has not produced a parity-valid edge-runtime artifact."
    ),
)
_add(
    "blocked",
    ("dinov2", "eomt", "pidnet"),
    ("semantic",),
    ("coreml",),
    reason="The CoreML wrapper does not implement the dense semantic-logits contract.",
)


_TASK_BLOCKS = {
    "ocr": (
        "OCR uses two networks for detection and recognition with dynamic "
        "per-region cropping, so it does not fit the single-graph export contract."
    ),
    "point": (
        "This family is not wired to the shared point heatmap and backend "
        "peak-decoding export contract."
    ),
    "semantic": (
        "This family is not wired to the shared dense-logits and backend "
        "argmax semantic export contract."
    ),
    "panoptic": "Panoptic export does not yet have a backend runtime contract.",
    "gaze": (
        "This family is not wired to the shared two-head logits and backend "
        "expectation-decoding gaze export contract."
    ),
}

_FAMILY_BLOCKS = {
    "depth_anything3": (
        "Depth Anything 3 currently rejects export for every format; its "
        "depth graph has not been added to the exported-runtime contract."
    ),
    "eomt": "EoMT instance and panoptic export do not yet have runtime parsing.",
    "l2cs": "The v1 L2CS gaze export contract supports ONNX only.",
    "sam": "Promptable model export is out of scope for the v1 runtime contract.",
    "sam2": "Promptable model export is out of scope for the v1 runtime contract.",
    "edgetam": "Promptable model export is out of scope for the v1 runtime contract.",
    "sam3": "Promptable model export is out of scope for the v1 runtime contract.",
    "mobilesam": "Promptable model export is out of scope for the v1 runtime contract.",
    "grounding_dino": "Open-vocabulary runtime export is out of scope for v1.",
    "owlv2": "Open-vocabulary runtime export is out of scope for v1.",
    "florence2": "Generative VLM export is out of scope for v1.",
    "kosmos2": "Generative VLM export is out of scope for v1.",
    "lfm2vl": "Generative VLM export is out of scope for v1.",
    "internvl3": "Generative VLM export is out of scope for v1.",
    "qwen3vl": "Generative VLM export is out of scope for v1.",
    "smolvlm2": "Generative VLM export is out of scope for v1.",
    "locateanything": "Generative VLM export is out of scope for v1.",
}

_NCNN_BLOCKS = {
    "dfine": "D-FINE",
    "deim": "DEIM",
    "deimv2": "DEIMv2",
    "rtdetr": "RT-DETR",
    "rtdetrv2": "RT-DETRv2",
    "rtdetrv4": "RT-DETRv4",
    "rfdetr": "RF-DETR",
    "ec": "EC",
}


def get_support(family: str, task: str, fmt: str) -> SupportEntry:
    """Return the canonical support entry for an export combination."""
    family = str(family or "").lower()
    task = str(task or "detect").lower()
    fmt = str(fmt or "").lower()
    if task not in TASKS:
        return SupportEntry("blocked", f"{task!r} is not a canonical LibreYOLO task.")
    if fmt not in EXPORT_FORMATS:
        return SupportEntry("blocked", f"{fmt!r} is not a registered export format.")

    exact = SUPPORT.get((family, task, fmt))
    if exact is not None:
        return exact
    if family in _FAMILY_BLOCKS:
        return SupportEntry("blocked", _FAMILY_BLOCKS[family])
    if task in _TASK_BLOCKS:
        return SupportEntry("blocked", _TASK_BLOCKS[task])
    if fmt == "ncnn" and family in _NCNN_BLOCKS:
        label = _NCNN_BLOCKS[family]
        return SupportEntry(
            "blocked",
            f"NCNN export is not supported for {label}: the model requires decoder "
            "or sampling operations unavailable in NCNN. "
            "Use ONNX, OpenVINO, TorchScript, or TensorRT instead.",
        )
    if fmt in {"tensorrt", "openvino"}:
        runtime = "TensorRT" if fmt == "tensorrt" else "OpenVINO"
        return SupportEntry(
            "experimental",
            f"The converter path is available, but the project has not yet "
            f"recorded {runtime} runtime parity for this family and task.",
        )
    if fmt == "tflite":
        return SupportEntry(
            "blocked",
            "This family and task have not been validated through the ONNX-to-TFLite path.",
        )
    if fmt == "coreml":
        return SupportEntry(
            "blocked",
            "This family and task are not covered by the family-aware CoreML wrapper.",
        )
    return SupportEntry(
        "experimental",
        "This combination exports without a numeric parity guarantee.",
    )


def iter_entries(
    tier: Tier | None = None,
) -> Iterator[tuple[tuple[str, str, str], SupportEntry]]:
    """Iterate explicit matrix entries, optionally filtered by tier."""
    for key, entry in sorted(SUPPORT.items()):
        if tier is None or entry.tier == tier:
            yield key, entry


def iter_validated() -> Iterator[tuple[tuple[str, str, str], SupportEntry]]:
    """Iterate explicit parity-validated entries."""
    return iter_entries("validated")


def iter_blocked() -> Iterator[tuple[tuple[str, str, str], SupportEntry]]:
    """Iterate explicit blocked entries."""
    return iter_entries("blocked")


def validated_alternatives(family: str, task: str) -> tuple[str, ...]:
    """Return validated formats for a concrete family and task."""
    return tuple(
        fmt
        for fmt in EXPORT_FORMATS
        if get_support(family, task, fmt).tier == "validated"
    )


__all__ = [
    "EXPORT_FORMATS",
    "SUPPORT",
    "SupportEntry",
    "Tier",
    "get_support",
    "iter_blocked",
    "iter_entries",
    "iter_validated",
    "validated_alternatives",
]
