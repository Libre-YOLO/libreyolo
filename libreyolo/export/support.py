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
    "coreai",
)


@dataclass(frozen=True)
class SupportEntry:
    """Support status and user-facing context for one export combination."""

    tier: Tier
    reason: str = ""
    since: str | None = None
    constraint: str | None = None


SUPPORT: dict[tuple[str, str, str], SupportEntry] = {}

# Artifact eligibility is deliberately separate from technical export support.
# A permissively licensed architecture can accept user-trained weights even
# when a published checkpoint is non-commercial, research-only, or unclear.
# These notes are sourced from the repository's existing NOTICE surfaces.
CHECKPOINT_GATES: dict[tuple[str, str], str] = {
    (
        "birefnet",
        "matte",
    ): (
        "The `l` checkpoint is MIT. The `t` checkpoint is not rehosted because "
        "its upstream repository has no explicit license metadata or LICENSE file."
    ),
    (
        "deimv2",
        "detect",
    ): (
        "DINOv3-backed variants carry Meta's custom, non-OSI DINOv3 terms. "
        "Do not treat conversion of the permissive HGNet variants as evidence "
        "for those variants."
    ),
    (
        "depth_anything",
        "depth",
    ): (
        "The `s` checkpoint is Apache-2.0. Published `b`, `l`, and `g` "
        "checkpoints are CC-BY-NC-4.0 and are not redistributed by LibreYOLO."
    ),
    (
        "l2cs",
        "gaze",
    ): (
        "Published gaze checkpoints are bound by the research/non-commercial "
        "Gaze360 dataset terms and are not bundled, mirrored, or auto-downloaded."
    ),
    (
        "internvl3",
        "detect",
    ): (
        "The published `-hf` weights carry the Qwen License rather than a "
        "permissive MIT, Apache-2.0, or BSD license."
    ),
    (
        "lfm2vl",
        "detect",
    ): (
        "Published checkpoints carry the non-permissive LFM Open License v1.0 "
        "with a revenue threshold."
    ),
    (
        "locateanything",
        "detect",
    ): "The published LocateAnything checkpoint is NVIDIA non-commercial.",
    (
        "locateanything",
        "point",
    ): "The published LocateAnything checkpoint is NVIDIA non-commercial.",
    (
        "nafnet",
        "restore",
    ): (
        "Some published GoPro checkpoints have no explicit standalone weights "
        "license. Convert only checkpoints the user has the right to use."
    ),
    (
        "ov_deim",
        "detect",
    ): (
        "Published detector weights are CC BY-NC 4.0, and the MobileCLIP text "
        "tower carries research-only model terms. These are not MIT artifacts."
    ),
    (
        "sam3",
        "segment",
    ): (
        "SAM 3 access is gated by Meta's custom SAM License; LibreYOLO does not "
        "redistribute the checkpoint under its MIT license."
    ),
    (
        "segformer",
        "semantic",
    ): (
        "The architecture port is Apache-2.0, but published ADE20K checkpoints "
        "are restricted to research or evaluation by NVIDIA's license."
    ),
    (
        "sensenovavision",
        "detect",
    ): "The published SenseNova-Vision checkpoint is CC BY-NC 4.0.",
    (
        "sensenovavision",
        "segment",
    ): "The published SenseNova-Vision checkpoint is CC BY-NC 4.0.",
    (
        "sensenovavision",
        "panoptic",
    ): "The published SenseNova-Vision checkpoint is CC BY-NC 4.0.",
    (
        "sensenovavision",
        "pose",
    ): "The published SenseNova-Vision checkpoint is CC BY-NC 4.0.",
    (
        "sensenovavision",
        "point",
    ): "The published SenseNova-Vision checkpoint is CC BY-NC 4.0.",
    (
        "sensenovavision",
        "depth",
    ): "The published SenseNova-Vision checkpoint is CC BY-NC 4.0.",
    (
        "sensenovavision",
        "ocr",
    ): "The published SenseNova-Vision checkpoint is CC BY-NC 4.0.",
    (
        "yolo9_p2",
        "detect",
    ): (
        "The VisDrone research-preview variant is CC BY-NC-SA. Permissive "
        "YOLO9 transfer weights may be used for conversion-only tests."
    ),
    (
        "yolonas",
        "detect",
    ): (
        "Published pretrained weights may carry separate non-commercial terms "
        "and are not bundled. Synthetic or user-trained states remain separate."
    ),
    (
        "yolonas",
        "pose",
    ): (
        "Published pretrained weights may carry separate non-commercial terms "
        "and are not bundled. Synthetic or user-trained states remain separate."
    ),
}


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
    tuple(fmt for fmt in EXPORT_FORMATS if fmt not in {"onnx", "coreai", "coreml"}),
    reason=(
        "Frozen-class vision-language export is available only for ONNX and "
        "the Apple runtimes in v1."
    ),
)
_add(
    "blocked",
    ("dinov2",),
    ("classify",),
    tuple(fmt for fmt in EXPORT_FORMATS if fmt not in {"onnx", "coreai", "coreml"}),
    reason=(
        "LibreDINOv2 classify export is not wired for this runtime; use ONNX, "
        "Core AI, or experimental Core ML export."
    ),
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
    reason=(
        "Fixed-canvas, batch-one raw-output conversion is available, but "
        "numeric Core ML runtime parity has not yet been recorded on macOS."
    ),
)
_add(
    "experimental",
    ("dinov2", "eomt", "pidnet", "lingbotvision"),
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
    tuple(fmt for fmt in EXPORT_FORMATS if fmt not in {"onnx", "coreml"}),
    reason=(
        "PicoSAM3's raw ROI component is currently wired only for ONNX and "
        "Core ML."
    ),
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
    tuple(fmt for fmt in EXPORT_FORMATS if fmt != "coreml"),
    reason=(
        "RTMDet-Ins dynamic-kernel mask decoding has no contract for this "
        "exported runtime. Use native PyTorch inference or the Core ML "
        "raw-output profile."
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
    "validated",
    ("lingbotvision",),
    ("semantic",),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="fixed 512x512 input",
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
    ("dinov2", "eomt", "lingbotvision"),
    ("semantic",),
    ("ncnn", "tflite"),
    reason=(
        "The dense-logits runtime contract is implemented, but this transformer "
        "graph has not produced a parity-valid edge-runtime artifact."
    ),
)
_add(
    "validated",
    ("yolo1", "yolo2", "yolo3", "yolo4"),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed family-native canvases (YOLO1 448, YOLO2 608, YOLO3 416, "
        "YOLO4 608); representative published trained checkpoints are covered "
        "on Apple hardware by direct named-output parity with a 3e-04 "
        "tolerance and a 100x input-sensitivity margin; Core AI graph "
        "preparation exactly folds Darknet inference batch normalization into "
        "the preceding convolutions because Core AI 0.4.1 does not preserve "
        "Darknet's epsilon-after-square-root formula"
    ),
)
_add(
    "validated",
    ("yolonas",),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed 96x96 export canvas with pre-shaped canonical RGB tensors; a "
        "deterministic, license-clean synthetic "
        "YOLO-NAS-S state is covered on Apple hardware by direct named-output "
        "parity with a 3e-04 tolerance and a 100x input-sensitivity margin; "
        "the state receives 12 native training steps and a 20x regression-head "
        "scale to make both exported outputs non-degenerate; this validates "
        "conversion, not detection accuracy, raw-image preprocessing, or "
        "native-640 behavior, and does not convert restricted official weights"
    ),
)
_add(
    "experimental",
    ("dinov2",),
    ("classify",),
    ("coreai",),
    reason=(
        "Conversion has been measured, but the LibreDINOv2 classification "
        "checkpoint is not publicly downloadable for a reproducible trained-"
        "weight Core AI parity gate."
    ),
)
_add(
    "validated",
    (
        "deim",
        "deimv2",
        "ec",
        "picodet",
        "rtdetr",
        "rtdetrv2",
        "rtdetrv4",
        "rtmdet",
        "yolo9_e2e",
        "yolox",
    ),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; a representative published trained checkpoint "
        "for each family is covered on Apple hardware by direct named-output "
        "parity with a 3e-04 tolerance and a 100x input-sensitivity margin; "
        "RT-DETRv2 permits one shared whole-query permutation across its box "
        "and logit outputs because DETR query rows are an unordered set"
    ),
)
_add(
    "validated",
    ("convnext", "efficientnetv2", "mobilenetv4", "resnet"),
    ("classify",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; a representative published trained ImageNet "
        "checkpoint for each family is covered on Apple hardware by direct "
        "named-output parity with a 3e-04 tolerance and a 100x "
        "input-sensitivity margin"
    ),
)
_add(
    "validated",
    ("depth_anything", "zipdepth"),
    ("depth",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; permissively licensed trained checkpoints are "
        "covered on Apple hardware by direct named-output parity with a "
        "3e-04 tolerance and a 100x input-sensitivity margin"
    ),
)
_add(
    "validated",
    ("nafnet", "realesrgan"),
    ("restore",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; permissively licensed trained restoration "
        "checkpoints are covered on Apple hardware by direct named-output "
        "parity with a 3e-04 tolerance and a 100x input-sensitivity margin"
    ),
)
_add(
    "validated",
    ("clip", "siglip2"),
    ("classify",),
    ("coreai",),
    since="1.5",
    constraint=(
        "frozen class set and fixed export canvas; permissively licensed "
        "trained checkpoints are covered on Apple hardware by direct named-"
        "output parity with a 3e-04 tolerance and a 100x input-sensitivity "
        "margin"
    ),
)

# HOW THE Core AI NUMBERS BELOW WERE MEASURED, and why an earlier set of them
# was withdrawn.
#
# Every figure is the worst relative error against a reference graph, with each
# artifact fed the input ITS OWN contract expects and reported alongside the
# reference's own input-sensitivity. Published trained weights are used where a
# permissive checkpoint exists. The FOMO, YOLO-NAS, and YOLO9-P2 entries state
# their license-clean synthetic or transfer fixture explicitly and make no
# accuracy claim.
#
# All three qualifiers were learned the hard way.
#
# Non-degenerate weights: a randomly initialised detection head emits nearly the same
# tensor whatever it is shown, because the constant anchor grid dominates its
# output. Measured on the ONNX reference between two very different probes,
# random-init yolox moved by 1.5e-09 and rtmdet by 8.9e-12. Agreement at 1e-08
# against a reference that moves 1.5e-09 certifies nothing. picodet was caught
# because it hit exactly zero and was recorded as blocked; its neighbours
# failed the same way by degrees and were recorded as validated.
#
# Input contract: _wrap_for_family wraps some families in a preprocessing
# module for the Apple formats, so a Core AI graph takes canonical RGB[0,1] and
# converts internally (YOLOX scales by 255 and swaps to BGR, RF-DETR applies
# ImageNet normalization). The ONNX exporter applies no such wrapper. Handing
# both the same tensor compares two different functions and reads ~0.5 however
# correct the conversion is.
#
# Sensitivity margin: a result counts only if parity is at least 100x below
# how far the reference itself moves between probes. Otherwise the honest
# answer is that the measurement cannot support a verdict.
_add(
    "validated",
    ("dfine",),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; trained LibreDFINEn weights are covered on "
        "macOS 27 by direct named-output parity with a 3e-04 tolerance and "
        "a 100x input-sensitivity margin"
    ),
)


_add(
    "validated",
    ("yolo9",),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; trained LibreYOLO9t weights are covered on "
        "macOS 27 by direct named-output parity with a 3e-04 tolerance and "
        "a 100x input-sensitivity margin"
    ),
)
_add(
    "validated",
    ("yolo9_p2",),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed 640x640 export canvas; a deterministic YOLO9-P2-T model "
        "initialized from the SHA-256-pinned, permissively licensed trained "
        "LibreYOLO9t checkpoint is covered on Apple hardware by direct "
        "named-output parity with a 3e-04 tolerance and a 100x "
        "input-sensitivity margin; this validates conversion, not P2 task "
        "accuracy, and does not depend on the restricted VisDrone "
        "research-preview checkpoint"
    ),
)
_add(
    "validated",
    ("yolo7",),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed 640x640 export canvas; trained LibreYOLO7b weights are covered on "
        "Apple hardware by direct named-output parity with a 3e-04 tolerance "
        "and a 100x input-sensitivity margin; the export decoder uses direct "
        "arange grids because Core AI 0.4.1 mislowers the equivalent "
        "cumulative-sum expression"
    ),
)
_add(
    "blocked",
    ("birefnet",),
    ("matte",),
    ("coreai",),
    reason=(
        "The decoder needs torchvision deform_conv2d, which the Core AI "
        "converter cannot lower ('unable to handle call function op: "
        "deform_conv2d.default'). The same operator already blocks the NCNN "
        "path. An encoder-only contract is the realistic route, matching the "
        "seam the CUDA graph work used."
    ),
)
_add(
    "validated",
    ("rfdetr",),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; trained LibreRFDETRn weights are covered on "
        "macOS 27 against the graph the exporter itself prepares, using "
        "direct named-output parity with a 3e-04 tolerance and a 100x "
        "input-sensitivity margin. "
        "Conversion needed _rebake_rfdetr_pos_embed in export/coreai.py: the "
        "backbone bakes its position embedding for its configured 384 canvas, "
        "so exporting at any other size left an antialiased bicubic in the "
        "graph and the converter has no lowering for "
        "aten._upsample_bicubic2d_aa. The rebake re-runs the model's OWN "
        "baking path for the actual canvas, so the interpolation happens "
        "eagerly, outside the graph, computing exactly what it computed "
        "before. "
        "NOTE the reference. This family is verified against the exporter's "
        "prepared graph, not against ONNX, and the difference is not a "
        "detail: at a 640 canvas the rfdetr ONNX artifact disagrees with that "
        "same prepared graph by 9.3e-01. Core AI's rebake preserves the "
        "antialiased resize the eager "
        "model performs, whereas the ONNX path disables antialiasing (the "
        "model checks torch.onnx.is_in_onnx_export). Which artifact is right "
        "is an ONNX question and is not settled here, but ONNX cannot be used "
        "as the reference for this family at a non-native canvas."
    ),
)
_add(
    "blocked",
    ("swinir",),
    ("restore",),
    ("coreai",),
    reason=(
        "The export process DIES rather than hangs, and the kill point moves "
        "between runs, which is the signature of memory exhaustion rather "
        "than a stuck loop. One run reached 'Step 3/3: Optimizing and writing "
        "the asset' before stopping; a later run of the same graph at the "
        "same 128 canvas died inside to_coreai() before returning, in both "
        "cases with a leaked-semaphore warning and no traceback. Window "
        "attention unrolls into a very large number of small ops, so the "
        "converter's peak memory is the prime suspect on a 16 GB machine. "
        "Next steps: watch RSS during conversion, try the smallest available "
        "size at a 64 canvas, and check the system log for a memory kill. Do "
        "NOT assume optimize() is at fault; an earlier note said so on the "
        "strength of a single run and the second run contradicted it."
    ),
)

_add(
    "validated",
    ("pidnet", "lingbotvision"),
    ("semantic",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed family-native canvases (PIDNet 1024, LingBotVision 512); trained "
        "LibrePIDNets-sem and LibreLingBotVisions-sem checkpoints are covered "
        "on Apple hardware by direct named-output parity with a 3e-04 "
        "tolerance and a 100x input-sensitivity margin; exported backends "
        "already implement the shared dense-logit resize and argmax contract"
    ),
)
_add(
    "blocked",
    ("segformer",),
    ("semantic",),
    ("coreai",),
    reason=(
        "LibreSegformer implements no export path at all ('Export is not "
        "implemented for LibreSegformer yet'), so this is not a Core AI "
        "limitation. Note its weights are non-commercial regardless."
    ),
)
_add(
    "blocked",
    ("eomt",),
    ("semantic",),
    ("coreai",),
    reason=(
        "torch.export refuses the graph: GuardOnDataDependentSymNode, "
        "'Could not guard on data-dependent expression Eq(u0, 1)'. Something "
        "in the mask path reads a value off a tensor and branches on it, "
        "which becomes an unbacked symbol with no hint the tracer can "
        "resolve. This is a real capture failure, not a missing operator and "
        "not the task gate: it was measured with the gate open. Fixing it "
        "means finding the host read and making the shape static for a fixed "
        "export canvas, the same shape of fix as the rfdetr torch._assert."
    ),
)
_add(
    "validated",
    ("fomo",),
    ("point",),
    ("coreai",),
    since="1.5",
    constraint=(
        "native 96 canvas; a deterministic model state trained from scratch "
        "for eight steps on synthetic tensors is covered on Apple hardware "
        "by direct named-output parity with a 3e-04 tolerance and a 100x "
        "input-sensitivity margin; this validates conversion and the existing "
        "heatmap contract, not point-localization accuracy"
    ),
)
_add(
    "blocked",
    ("l2cs",),
    ("gaze",),
    ("coreai",),
    reason=(
        "The model itself refuses: 'LibreL2CS export to coreai is not "
        "implemented. The v1 gaze export contract supports ONNX only.' That "
        "is a model-side decision, unchanged by opening the support gate, so "
        "nothing about Core AI is being tested here. Wiring the gaze contract "
        "beyond ONNX comes first."
    ),
)
_add(
    "blocked",
    ("depth_anything3",),
    ("depth",),
    ("coreai",),
    reason=(
        "The model raises NotImplementedError for every format: depth export "
        "is out of scope per ADR 0006, the depth task contract. Depth Anything "
        "V2 exports and validates at 5.2e-06, so this is specific to the V3 "
        "family and not a Core AI limitation."
    ),
)

# Core ML uses a fixed-canvas, batch-one raw-output contract. These rows are
# conversion-enabled, but none is validated until the saved artifact has run
# through numeric parity on macOS. Keep this list aligned with the strict
# family/task preflight in export/coreml.py.
_COREML_EXPERIMENTAL_REASON = (
    "Fixed-canvas, batch-one raw-output conversion is available, but numeric "
    "Core ML runtime parity has not yet been recorded on macOS."
)
_add(
    "experimental",
    (
        "deim",
        "dfine",
        "ec",
        "picodet",
        "rtdetrv2",
        "rtdetrv4",
        "rtmdet",
        "yolo1",
        "yolo2",
        "yolo3",
        "yolo4",
        "yolo7",
        "yolo9_e2e",
        "yolo9_p2",
    ),
    ("detect",),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
)
_add(
    "experimental",
    ("deimv2",),
    ("detect",),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
    constraint=(
        "Only the permissive `atto`, `femto`, `pico`, and `n` variants are "
        "accepted. DINOv3-backed `s`, `m`, `l`, `x`, and unknown variants "
        "fail in preflight."
    ),
)
_add(
    "experimental",
    (
        "clip",
        "convnext",
        "dinov2",
        "efficientnetv2",
        "mobilenetv4",
        "resnet",
        "siglip2",
    ),
    ("classify",),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
    constraint=(
        "CLIP and SigLIP2 freeze the current class set and require their native "
        "input resolution. SigLIP2 preserves the exported softmax or sigmoid "
        "classification activation."
    ),
)
_add(
    "experimental",
    ("depth_anything", "zipdepth"),
    ("depth",),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
)
_add(
    "experimental",
    ("nafnet", "realesrgan"),
    ("restore",),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
    constraint=(
        "The source image must exactly match the fixed export canvas. Re-export, "
        "crop, or tile instead of padding an arbitrary smaller image to the canvas."
    ),
)
_add(
    "experimental",
    ("dinov2", "lingbotvision", "pidnet"),
    ("semantic",),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
)
_add(
    "experimental",
    ("fomo",),
    ("point",),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
)
_add(
    "experimental",
    ("l2cs",),
    ("gaze",),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
)
_add(
    "experimental",
    ("dfine",),
    ("segment",),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
)
_add(
    "experimental",
    ("ec",),
    ("segment", "pose"),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
)
_add(
    "experimental",
    ("rfdetr",),
    ("segment", "pose", "obb"),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
)
_add(
    "experimental",
    ("eomt",),
    ("semantic", "segment", "panoptic"),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
    constraint=(
        "Fixed square DINOv2 S/B/L component. The graph emits compact raw "
        "class-query logits and stride-4 mask logits. LibreYOLO preserves "
        "EoMT's exact shortest-edge split/stitch geometry for semantic and "
        "longest-edge top-left pad/query decoding for instance and panoptic "
        "tasks on the host. The functional attention-mask graph has real "
        "Core ML Tools 9 conversion evidence; macOS runtime parity is pending."
    ),
)
_add(
    "experimental",
    ("segformer",),
    ("semantic",),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
    constraint=(
        "All b0-b5 eval graphs pass fixed-canvas two-probe TorchScript trace "
        "parity. The architecture/source is permissive, but published ADE20K "
        "weights remain restricted to research or evaluation."
    ),
)
_add(
    "experimental",
    ("swinir",),
    ("restore",),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
    constraint=(
        "Sizes `s`, `m`, and `l` are enabled at their native 64x64 canvas. "
        "Every full graph has bit-exact two-probe TorchScript parity and Core "
        "ML Tools 9 FP16 ML Program conversion evidence. The exact source "
        "canvas is required; non-native canvases and Apple runtime parity "
        "remain pending."
    ),
)
_add(
    "experimental",
    ("yolonas",),
    ("detect", "pose"),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
    constraint=(
        "Square fixed canvas with the native longest-side cap: 636-centered RGB "
        "padding for detect; 640 top-left placement, bottom/right padding, and "
        "BGR graph input for pose. Current evidence uses license-clean synthetic "
        "graphs, not restricted published weights."
    ),
)
_add(
    "experimental",
    ("picosam3",),
    ("segment",),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
    constraint=(
        "Raw fixed batch-one 96x96 ROI component. The host expands each box "
        "by 10%, crops and resizes the ROI, then places mask logits back into "
        "the source image; point/text/mask prompts remain unsupported."
    ),
)
_add(
    "experimental",
    ("rtmdet",),
    ("segment",),
    ("coreml",),
    reason=_COREML_EXPERIMENTAL_REASON,
    constraint=(
        "Fixed batch-one canvas divisible by 32. The graph emits three class "
        "maps, three box-distance maps, three 169-parameter dynamic-kernel "
        "maps, and one stride-8 mask feature map; LibreYOLO performs per-level "
        "top-k, class-aware NMS, dynamic mask decoding, and placement on the "
        "host."
    ),
)
_add(
    "experimental",
    ("ppocr",),
    ("ocr",),
    ("coreml",),
    reason=(
        "A strict iOS18/macOS15 two-function ML Program package is available, "
        "but numeric Core ML runtime parity has not yet been recorded on macOS."
    ),
    constraint=(
        "FP32 only. The detector and recognizer use named bounded-flexible "
        "TensorType functions; DB contours, perspective crops, reading order, "
        "bucketing, and CTC decoding remain exact host operations. Export "
        "requires an explicit finite rec_max_width; overflow is an error, "
        "never silent truncation."
    ),
)
_add(
    "experimental",
    ("edgetam", "mobilesam", "sam", "sam2", "sam3"),
    ("segment",),
    ("coreml",),
    reason=(
        "A strict iOS18/macOS15 seven-function ML Program package is "
        "implemented, but numeric Core ML runtime parity has not yet been "
        "recorded on macOS."
    ),
    constraint=(
        "FP32 only. One fixed model-ready image encoder and six named prompt "
        "decoders cover points, boxes, points+boxes, and single/multimask "
        "modes. Point count is bounded by prompt_max_points (default 16); "
        "raw-image preprocessing, prompt transforms, query loops, and mask "
        "upscaling remain exact host operations. SAM3 is visual-prompt-only "
        "and converted artifacts are local-user-only under its custom license."
    ),
)

# Explicit Core ML walls. A row must not disappear into get_support()'s
# fallback because the generated compatibility table is also the work queue.
_add(
    "experimental",
    ("birefnet",),
    ("matte",),
    ("coreml",),
    reason=(
        "The exact `t` and `l` graphs convert with Apple's post-9.0 "
        "torchvision::deform_conv2d ML Program lowering, but numeric runtime "
        "and matte-quality parity have not yet been recorded on Apple Silicon."
    ),
    constraint=(
        "Fixed 1024x1024, batch one, raw matte logits. Stable Core ML Tools "
        "9.0 predates the required lowering; export feature-detects the Apple "
        "converter implementation instead of trusting its version string. "
        "Published `l` weights are MIT; `t` artifacts remain local-user-only "
        "until their upstream weight provenance is explicit."
    ),
)
_add(
    "experimental",
    ("owlv2",),
    ("detect",),
    ("coreml",),
    reason=(
        "The current finite class vocabulary is frozen into an image-only "
        "detector graph. Strict TorchScript and reference preprocessing parity "
        "are implemented; numeric Core ML runtime parity is pending."
    ),
    constraint=(
        "Fixed native 960x960 (`b16`) or 1008x1008 (`l14`) FP32 TensorType "
        "input with exact pad-before-Gaussian-resize preprocessing. The text "
        "tower and tokenizer are intentionally absent at runtime, so changing "
        "classes requires re-export. Published mirror provenance must be "
        "completed before converted weights are treated as release artifacts."
    ),
)
_add(
    "experimental",
    ("grounding_dino",),
    ("detect",),
    ("coreml",),
    reason=(
        "The current finite prompt and class vocabulary are frozen into an "
        "image-only detector while the complete cross-modal encoder/decoder "
        "remains in the graph. A compact representative graph completes Core "
        "ML Tools 9 conversion; full released-checkpoint Apple runtime parity "
        "is pending."
    ),
    constraint=(
        "Sizes `t` and `b` use a fixed 800x800 batch-one RGB stretch profile, "
        "900 queries, and at most 256 BERT tokens. The exact pre-fusion BERT "
        "boundary, prompt tokens, masks, positions, and WordPiece ABI are "
        "frozen; changing classes requires re-export. Fixed stretch differs "
        "from the native keep-aspect image policy for non-square sources."
    ),
)
_add(
    "experimental",
    ("omdet_turbo",),
    ("detect",),
    ("coreml",),
    reason=(
        "The 51.98M-parameter image-only deployment graph derived from the "
        "pinned 115.4M checkpoint is bit-exact with the native detector for "
        "the frozen vocabulary, traces exactly, and completes Core ML Tools "
        "9 FP16 ML Program conversion; macOS runtime parity is pending."
    ),
    constraint=(
        "The released `t` checkpoint uses a fixed 640x640 batch-one FP32 "
        "TensorType boundary. The current class vocabulary and task-language "
        "embeddings are frozen; changing classes requires re-export. Exact "
        "Torchvision-v2 uint8 bilinear-antialias stretch remains on the host; "
        "the graph emits 900 normalized boxes and per-class logits for exact "
        "top-900 and class-aware NMS decoding."
    ),
)
_add(
    "experimental",
    ("depth_anything3",),
    ("depth",),
    ("coreml",),
    reason=(
        "The full DA3MONO-LARGE raw depth/sky component converts with Core ML "
        "Tools 9, but numeric runtime parity has not yet been recorded on macOS."
    ),
    constraint=(
        "Fixed 504x504, batch one. The graph emits positive relative depth "
        "and non-negative sky scores. LibreYOLO preserves the native sky-region "
        "gate, random-with-replacement sampling, 0.99 quantile, reciprocal, "
        "and final align_corners=True resize on the host. Non-square input uses "
        "the documented fixed-stretch depth approximation."
    ),
)
_add(
    "blocked",
    ("ov_deim",),
    ("detect",),
    ("coreml",),
    reason=(
        "Open-vocabulary detection needs a frozen-vocabulary or bounded "
        "text/image component contract before Core ML export can be enabled."
    ),
)
_add(
    "blocked",
    ("florence2", "internvl3", "kosmos2", "lfm2vl", "qwen3vl", "smolvlm2"),
    ("detect",),
    ("coreml",),
    reason=(
        "Generative vision-language inference requires tokenizer, prefill, "
        "decode, and state/cache runtime components rather than one image graph."
    ),
)
_add(
    "blocked",
    ("locateanything",),
    ("detect", "point"),
    ("coreml",),
    reason=(
        "LocateAnything is a generative multi-input pipeline and has no "
        "tokenizer, state/cache, or output-decoding Core ML component contract."
    ),
)
_add(
    "blocked",
    ("sensenovavision",),
    ("detect", "segment", "panoptic", "pose", "point", "depth", "ocr"),
    ("coreml",),
    reason=(
        "SenseNova-Vision is a generative multimodal pipeline with text and "
        "diffusion/VAE outputs; it has no stateful multi-component Core ML contract."
    ),
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
    "omdet_turbo": "Open-vocabulary runtime export is out of scope for v1.",
    "ov_deim": "Open-vocabulary runtime export is out of scope for v1.",
    "florence2": "Generative VLM export is out of scope for v1.",
    "kosmos2": "Generative VLM export is out of scope for v1.",
    "lfm2vl": "Generative VLM export is out of scope for v1.",
    "internvl3": "Generative VLM export is out of scope for v1.",
    "qwen3vl": "Generative VLM export is out of scope for v1.",
    "smolvlm2": "Generative VLM export is out of scope for v1.",
    "locateanything": "Generative VLM export is out of scope for v1.",
    "sensenovavision": (
        "Generative multimodal export needs tokenizer, state/cache, and "
        "diffusion/VAE component contracts; it is out of scope for v1."
    ),
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
    if fmt == "coreai":
        return SupportEntry(
            "blocked",
            "This family and task have not been validated for Core AI export.",
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
    "CHECKPOINT_GATES",
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
