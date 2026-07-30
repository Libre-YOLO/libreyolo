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
        "PicoSAM3's raw ROI component is currently wired only for ONNX and Core ML."
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

# Core ML validation is deliberately scoped to saved artifacts executed on
# Apple hardware. Keep these rows aligned with the strict family/task preflight
# in export/coreml.py, and keep every check mark qualified by its measured
# checkpoint, canvas, precision, compute-unit profile, and reference graph.
_COREML_EXPERIMENTAL_REASON = (
    "Fixed-canvas, batch-one raw-output conversion is available, but numeric "
    "Core ML runtime parity has not yet been recorded on macOS."
)
_COREML_M4_RAW_REASON = (
    "A saved FP32 Core ML package executed with `compute_units='cpu_only'` on "
    "Apple M4/macOS 27 with Core ML Tools 9.0 and matched the named outputs of "
    "the exact exporter-prepared PyTorch graph."
)
_COREML_M4_RAW_GATE = (
    "The two-probe raw-output gate requires maximum relative error 3e-4, "
    "minimum relative output sensitivity 1e-6, and at least 100x "
    "sensitivity-to-error margin."
)
_COREML_M4_RAW_SCOPE = (
    _COREML_M4_RAW_GATE
    + " It proves conversion fidelity for only the "
    "stated representative checkpoint and fixed batch-one canvas; it does not "
    "prove model accuracy, arbitrary image geometry, every size or checkpoint, "
    "public preprocessing/postprocessing, Neural Engine placement, or device "
    "performance."
)
_COREML_M4_QUERY_ALIGNMENT = (
    " For set-prediction outputs, parity permits one whole-query assignment "
    "shared across every semantic output for each probe; it never aligns "
    "boxes, logits, masks, or keypoints independently."
)
_COREML_M4_QUERY_FAMILIES = frozenset(
    {"deim", "dfine", "ec", "rtdetr", "rtdetrv2", "rtdetrv4"}
)
for _family, _checkpoint, _canvas in (
    ("deim", "LibreDEIMn", 640),
    ("dfine", "LibreDFINEn", 640),
    ("ec", "LibreECs", 640),
    ("picodet", "LibrePICODETs", 320),
    ("rtdetr", "LibreRTDETRr18", 640),
    ("rtdetrv2", "LibreRTDETRv2r18", 640),
    ("rtdetrv4", "LibreRTDETRv4s", 640),
    ("rtmdet", "LibreRTMDett", 640),
    ("yolo1", "LibreYOLO1b", 448),
    ("yolo2", "LibreYOLO2b", 608),
    ("yolo3", "LibreYOLO3b", 416),
    ("yolo4", "LibreYOLO4b", 608),
    ("yolo7", "LibreYOLO7b", 640),
    ("yolo9_e2e", "LibreYOLO9E2Et", 640),
    ("yolox", "LibreYOLOXn", 416),
):
    _add(
        "validated",
        (_family,),
        ("detect",),
        ("coreml",),
        reason=_COREML_M4_RAW_REASON,
        since="1.5",
        constraint=(
            f"FP32 CPU_ONLY prepared-graph parity covers trained {_checkpoint} "
            f"at a fixed {_canvas} canvas. "
            + _COREML_M4_RAW_SCOPE
            + (
                _COREML_M4_QUERY_ALIGNMENT
                if _family in _COREML_M4_QUERY_FAMILIES
                else ""
            )
        ),
    )
_add(
    "validated",
    ("yolo9",),
    ("detect",),
    ("coreml",),
    reason=(
        "The trained LibreYOLO9t package passed both saved raw-graph parity "
        "and the public detection/repeat gate on Apple M4."
    ),
    since="1.5",
    constraint=(
        "The fixed 640 FP32 package passed the two-probe prepared-graph gate "
        "with `compute_units='cpu_only'`. "
        + _COREML_M4_RAW_GATE
        + " Its odd non-square public "
        "detection path also matched native boxes/classes/scores at IoU >= "
        "0.95 and score error <= 0.01 and repeated deterministically through "
        "the default compute-unit planner. This does not identify Neural "
        "Engine placement or prove accuracy, arbitrary geometry, other sizes "
        "or checkpoints, or device performance."
    ),
)
_add(
    "validated",
    ("rfdetr",),
    ("detect",),
    ("coreml",),
    reason=(
        "The trained LibreRFDETRn package passed both saved raw-graph parity "
        "and the public detection/repeat gate on Apple M4."
    ),
    since="1.5",
    constraint=(
        "The fixed 384 FP32 package passed the two-probe prepared-graph gate "
        "with `compute_units='cpu_only'`. "
        + _COREML_M4_RAW_GATE
        + " Its odd non-square public "
        "detection path also matched native boxes/classes/scores at IoU >= "
        "0.95 and score error <= 0.01 and repeated deterministically with the "
        "runtime requested as ALL. This does not identify actual operator "
        "placement, claim Neural Engine execution, or prove accuracy, "
        "arbitrary geometry, other sizes or checkpoints, or device "
        "performance."
        + _COREML_M4_QUERY_ALIGNMENT
    ),
)
_add(
    "validated",
    ("yolo9_p2",),
    ("detect",),
    ("coreml",),
    reason=(
        "The deterministic YOLO9-P2-T transfer fixture passed saved raw-output "
        "and public detection/repeat parity on Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers a deterministic "
        "YOLO9-P2-T model initialized by the SHA-256-pinned permissive "
        "LibreYOLO9t transfer checkpoint at a fixed 640 canvas. The raw "
        "prediction tensor passed at 1.86e-6 maximum relative error, 0.385 "
        "relative input sensitivity, and a 207,874x sensitivity-to-error "
        "margin. The odd non-square public path used a 1e-3 confidence gate "
        "to exclude unstable near-zero random-P2 ties and returned three "
        "detections with exact classes, minimum matched IoU 0.9999819, maximum "
        "box error 9.16e-5 source pixel, maximum score error 2.10e-8, and "
        "bit-exact repeats through the default ALL planner. This proves "
        "conversion of the extra stride-4 branch and host detection contract, "
        "not P2 task accuracy, other sizes, arbitrary geometry, Neural Engine "
        "placement, or device performance. It does not use the restricted "
        "VisDrone research-preview checkpoint."
    ),
)
_add(
    "validated",
    ("deimv2",),
    ("detect",),
    ("coreml",),
    reason=_COREML_M4_RAW_REASON,
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers the permissive "
        "LibreDEIMv2atto checkpoint at a fixed 320 canvas. "
        + _COREML_M4_RAW_SCOPE
        + _COREML_M4_QUERY_ALIGNMENT
        + " "
        "Only the permissive `atto`, `femto`, `pico`, and `n` variants are "
        "conversion-enabled; hardware validation currently covers `atto` only. "
        "DINOv3-backed `s`, `m`, `l`, `x`, and unknown variants fail in "
        "preflight and are outside this claim."
    ),
)
_add(
    "validated",
    ("clip",),
    ("classify",),
    ("coreml",),
    reason=_COREML_M4_RAW_REASON,
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers LibreCLIPb32 at its fixed "
        "224 canvas. "
        + _COREML_M4_RAW_SCOPE
        + " The current class set is frozen into the artifact and changing it "
        "requires re-export."
    ),
)
_add(
    "validated",
    ("siglip2",),
    ("classify",),
    ("coreml",),
    reason=_COREML_M4_RAW_REASON,
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers LibreSigLIP2b16 at its "
        "fixed 256 canvas. "
        + _COREML_M4_RAW_SCOPE
        + " The current class set is frozen into the artifact and changing it "
        "requires re-export. The artifact preserves the selected exported "
        "softmax or sigmoid activation."
    ),
)
for _family, _checkpoint, _raw_error, _public_error in (
    ("convnext", "LibreConvNeXtt-cls", "7.09e-7", "9.24e-7"),
    ("efficientnetv2", "LibreEfficientNetV2b0-cls", "7.17e-7", "2.39e-7"),
    ("mobilenetv4", "LibreMobileNetV4s-cls", "2.18e-6", "3.61e-6"),
    ("resnet", "LibreResNet18-cls", "5.83e-7", "4.92e-7"),
):
    _add(
        "validated",
        (_family,),
        ("classify",),
        ("coreml",),
        reason=(
            f"The trained {_checkpoint} package passed saved raw-output "
            "parity and the public classification/repeat gate on Apple M4."
        ),
        since="1.5",
        constraint=(
            f"FP32 CPU_ONLY prepared-graph parity covers {_checkpoint} at its "
            f"fixed 224 canvas with maximum relative error {_raw_error}. "
            + _COREML_M4_RAW_GATE
            + " The odd non-square public image path, requested through the "
            "default ALL planner, preserved all 1000 probabilities within "
            f"{_public_error}, exact top-1/top-5 classes, and bit-exact "
            "repeats. This does not identify actual operator placement or "
            "prove accuracy, arbitrary geometry, other sizes/checkpoints, "
            "Neural Engine use, or device performance."
        ),
    )
_add(
    "validated",
    ("dinov2",),
    ("classify",),
    ("coreml",),
    reason=(
        "A DINOv2-S-with-registers source model with a deterministic "
        "three-class linear head passed saved raw-output parity and the public "
        "classification/repeat gate on Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY evidence covers the size-n projector and official "
        "pretrained DINOv2-S-with-registers backbone at a fixed 224 canvas. "
        "A deterministic three-class head passed the 3e-4 two-probe raw gate, "
        "meaningful input-sensitivity checks, public probability/top-k "
        "agreement, and exact repeats. Export eagerly bakes the model's own "
        "fixed-canvas positional table because Core ML Tools 9 cannot lower "
        "`aten._upsample_bicubic2d_aa`; the live PyTorch model is restored and "
        "the public native-versus-Core-ML gate still passes. This proves the "
        "graph and host classification contract, not model accuracy, a "
        "trained LibreYOLO head, other projector sizes, arbitrary geometry, "
        "Neural Engine placement, or device performance. The documented "
        "`LibreDINOv2n-cls.pt` repository is not publicly downloadable and is "
        "outside this evidence."
    ),
)
_add(
    "validated",
    ("depth_anything",),
    ("depth",),
    ("coreml",),
    reason=(
        "The permissively licensed trained LibreDepthAnythingV2s-depth "
        "package passed saved raw-output and public depth/repeat parity on "
        "Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers "
        "LibreDepthAnythingV2s-depth at its fixed 518 canvas with 1.11e-6 "
        "maximum relative error, 0.108 relative input sensitivity, and a "
        "98,037x sensitivity-to-error margin. The public path reloaded a "
        "pristine checkpoint after export, independently reproduced the "
        "OpenCV stretch oracle, and preserved square and odd 173x257 sources "
        "within 4.14e-5 relative error with bit-exact repeats. This proves "
        "conversion and fixed-stretch host geometry for the Apache-2.0 `s` "
        "checkpoint, not metric depth, model accuracy, other sizes or "
        "checkpoints, Neural Engine placement, or device performance. "
        "Published `b`, `l`, and `g` checkpoints remain non-commercial."
    ),
)
_add(
    "validated",
    ("zipdepth",),
    ("depth",),
    ("coreml",),
    reason=(
        "The trained LibreZipDepthb-depth package passed saved raw-output "
        "and public depth/repeat parity on Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers LibreZipDepthb-depth at "
        "its fixed 384 canvas with 1.46e-5 maximum relative error, 0.628 "
        "relative input sensitivity, and a 43,237x sensitivity-to-error "
        "margin. The public path reloaded a pristine checkpoint after "
        "export, independently reproduced the OpenCV stretch oracle, and "
        "preserved square and odd 173x257 sources within 2.40e-6 relative "
        "error with bit-exact repeats. This proves conversion and "
        "fixed-stretch host geometry for the `b` checkpoint, not metric "
        "depth, model accuracy, other sizes or checkpoints, Neural Engine "
        "placement, or device performance."
    ),
)
_add(
    "validated",
    ("nafnet",),
    ("restore",),
    ("coreml",),
    reason=(
        "The trained LibreNAFNetl-restore-sidd package passed saved "
        "raw-output and public restoration/repeat parity on Apple M4 after "
        "one measured converter fusion was disabled."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers "
        "LibreNAFNetl-restore-sidd at its fixed 256 canvas. The unmodified "
        "coremltools 9 elementwise-to-batchnorm fusion failed the 3e-4 raw "
        "gate at 5.07e-4. LibreYOLO therefore disables exactly "
        "`common::fuse_elementwise_to_batchnorm`, preserves the source's 72 "
        "mul/add affine pairs, and records pass profile "
        "`nafnet_preserve_elementwise_affine_v1`; the resulting package "
        "passed at 1.63e-5 maximum relative error, 1.028 relative input "
        "sensitivity, and a 63,346x sensitivity-to-error margin. The public "
        "fixed-256 path differed in one of 196,608 uint8 channel values by "
        "one, with mean absolute delta 5.09e-6 and bit-exact repeats. The "
        "source must exactly match the export canvas. This proves the SIDD "
        "L checkpoint and fixed geometry, not restoration quality, other "
        "weights/sizes/canvases, Neural Engine placement, or device "
        "performance."
    ),
)
_add(
    "validated",
    ("realesrgan",),
    ("restore",),
    ("coreml",),
    reason=(
        "The trained LibreRealESRGAN x4-t package passed saved raw-output and "
        "public restoration/repeat parity on Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers "
        "LibreRealESRGANx4t-restore at its fixed 64 canvas with 1.79e-6 "
        "maximum relative error, 0.851 relative input sensitivity, and a "
        "477,097x sensitivity-to-error margin. The public default-ALL path "
        "restored an exact 64x64 source to 256x256 RGB with maximum quantized "
        "uint8 delta 1, mean absolute delta 2.03e-5, and bit-exact repeats. "
        "Odd or non-native source geometry fails closed instead of silently "
        "padding or resizing. This proves the x4-t graph and fixed-geometry "
        "host contract, not restoration quality, other sizes/checkpoints, "
        "arbitrary canvases, Neural Engine placement, or device performance."
    ),
)
_add(
    "validated",
    ("dinov2",),
    ("semantic",),
    ("coreml",),
    reason=(
        "A DINOv2-S-with-registers source model with a deterministic "
        "three-class dense head passed saved raw-output parity and the public "
        "semantic-map/repeat gate on Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY evidence covers the size-n projector and official "
        "pretrained DINOv2-S-with-registers backbone at the fixed 518 canvas. "
        "A deterministic three-class dense head passed the 3e-4 two-probe raw "
        "gate, meaningful input-sensitivity checks, public semantic-map "
        "agreement, and exact repeats. Export eagerly bakes the model's own "
        "fixed-canvas positional table because Core ML Tools 9 cannot lower "
        "`aten._upsample_bicubic2d_aa`; the live PyTorch model is restored and "
        "the public native-versus-Core-ML gate still passes. This proves the "
        "graph and host semantic contract, not model accuracy, a trained "
        "LibreYOLO head, other projector sizes, arbitrary geometry, Neural "
        "Engine placement, or device performance. The documented "
        "`LibreDINOv2n.pt` repository is not publicly downloadable and is "
        "outside this evidence."
    ),
)
_add(
    "validated",
    ("lingbotvision",),
    ("semantic",),
    ("coreml",),
    reason=_COREML_M4_RAW_REASON,
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers trained "
        "LibreLingBotVisions-sem at a fixed 512 canvas. "
        + _COREML_M4_RAW_SCOPE
    ),
)
_add(
    "validated",
    ("pidnet",),
    ("semantic",),
    ("coreml",),
    reason=_COREML_M4_RAW_REASON,
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers trained LibrePIDNets-sem "
        "at a fixed 1024 canvas. "
        + _COREML_M4_RAW_SCOPE
    ),
)
_add(
    "validated",
    ("fomo",),
    ("point",),
    ("coreml",),
    reason=(
        "A deterministically trained license-clean FOMO-S fixture passed saved "
        "raw-output and public point/repeat parity on Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers a generated FOMO-S model "
        "at a fixed 96 canvas with 3.51e-7 maximum relative error, 7.46e-3 "
        "relative input sensitivity, and a 21,284x sensitivity-to-error "
        "margin. The odd non-square public path returned 29 nonempty points "
        "with exact classes/order/XY placement, maximum score error 4.47e-8, "
        "and bit-exact repeats through the default ALL planner. This proves "
        "the point graph and host peak-placement contract, not task accuracy, "
        "hosted weights, other sizes, arbitrary geometry, Neural Engine "
        "placement, or device performance."
    ),
)
_add(
    "validated",
    ("l2cs",),
    ("gaze",),
    ("coreml",),
    reason=(
        "Generated L2CS r18, r34, r50, r101, and r152 models passed saved "
        "two-head parity and the public face-crop gaze/repeat gate on Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY evidence covers all five supported ResNet depths at "
        "the fixed 448 face-crop canvas with 90-bin yaw and pitch heads. Each "
        "deterministic generated graph passed the 3e-4 two-probe raw gate, "
        "meaningful input-sensitivity checks, decoded-angle agreement within "
        "5e-4 radians, and exact repeats. The package accepts one already "
        "cropped face; face detection and multi-face crop orchestration remain "
        "host operations. This proves graph and host decode fidelity, not gaze "
        "accuracy, arbitrary checkpoints, Neural Engine placement, or device "
        "performance. Published Gaze360-derived weights are research/"
        "non-commercial and non-redistributable, so they were not loaded and "
        "remain outside this evidence."
    ),
)
_add(
    "validated",
    ("dfine",),
    ("segment",),
    ("coreml",),
    reason=_COREML_M4_RAW_REASON,
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph and public-result parity covers trained "
        "LibreDFINEs/m/l/x-seg checkpoints at a fixed 640 canvas. Small and "
        "extra-large pass the standard 3e-4 raw gate; medium and large use "
        "narrow size-specific mask-logit gates of 4e-4 and 5e-4 after "
        "measuring 3.405e-4 and 4.305e-4 maximum relative mask error. "
        + _COREML_M4_RAW_SCOPE
        + _COREML_M4_QUERY_ALIGNMENT
    ),
)
_add(
    "validated",
    ("ec",),
    ("segment",),
    ("coreml",),
    reason=_COREML_M4_RAW_REASON,
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers trained LibreECs-seg at a "
        "fixed 640 canvas. "
        + _COREML_M4_RAW_SCOPE
        + _COREML_M4_QUERY_ALIGNMENT
    ),
)
_add(
    "validated",
    ("ec",),
    ("pose",),
    ("coreml",),
    reason=_COREML_M4_RAW_REASON,
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers trained LibreECs-pose at a "
        "fixed 640 canvas. "
        + _COREML_M4_RAW_SCOPE
        + _COREML_M4_QUERY_ALIGNMENT
    ),
)
_add(
    "validated",
    ("rfdetr",),
    ("obb",),
    ("coreml",),
    reason=_COREML_M4_RAW_REASON,
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph and public-result parity covers trained "
        "LibreRFDETRn-obb at a fixed 384 canvas and LibreRFDETRm-obb at a "
        "fixed 576 canvas. "
        + _COREML_M4_RAW_SCOPE
        + _COREML_M4_QUERY_ALIGNMENT
        + " LibreRFDETRs-obb and LibreRFDETRl-obb are explicitly rejected: "
        "their 512- and 704-canvas M4 runs measured 0.52% and 2.66% box "
        "divergence respectively; the large checkpoint also measured 1.68% "
        "logit divergence."
    ),
)
_add(
    "validated",
    ("rfdetr",),
    ("pose",),
    ("coreml",),
    reason=(
        "A fresh scoped Apple M4 rerun passed saved raw-output parity under "
        "RF-DETR pose's enforced preserve-division CPU profile. The public "
        "pose path was deterministic with exact classes, but one low-ranked "
        "box measured IoU 0.9920 against the 0.995 gate."
    ),
    since="1.5",
    constraint=(
        "Only FP32, `compute_units='cpu_only'`, and conversion pass profile "
        "`rfdetr_pose_preserve_division_v1` are supported. The profile removes "
        "only `common::divide_to_multiply`; a trained LibreRFDETRx-pose package "
        "at fixed 576 passed raw named-output parity (boxes, logits, and "
        "keypoints). "
        + _COREML_M4_RAW_GATE
        + " The odd non-square public path repeated deterministically with "
        "exact classes; its minimum matched box IoU was 0.9920, just below "
        "the 0.995 public gate, so the full public keypoint assertion remains "
        "a documented caveat. ALL/GPU exceeded the fixed 3e-4 raw parity "
        "gate. CPU_AND_NE is excluded "
        "because FP32 ML Programs are CPU-placed, not evidence of Neural "
        "Engine execution. Nonconforming, tampered, missing-profile, and "
        "legacy artifacts fail closed. This proves only this checkpoint/"
        "profile's conversion and public contract, not pose accuracy, "
        "arbitrary geometry, other sizes/checkpoints, or device performance."
        + _COREML_M4_QUERY_ALIGNMENT
    ),
)
_add(
    "blocked",
    ("rfdetr",),
    ("segment",),
    ("coreml",),
    reason=(
        "Fresh FP32 CPU_ONLY execution on Apple M4 converted and loaded, but "
        "named outputs diverged from the prepared PyTorch graph even after "
        "whole-query alignment (boxes 2.19e-02, logits 3.38e-02, masks "
        "1.63e-01 relative error). Distinct near-tied encoder proposals changed "
        "order across runtimes and were paired with different learned query "
        "slots, causing a discontinuous decoder change. An index-biased "
        "tie-break would also reorder valid native proposals and is not "
        "semantics-preserving."
    ),
)
_add(
    "validated",
    ("facerec",),
    ("embed",),
    ("coreml",),
    reason=(
        "The complete pinned 65.1M-parameter recognition head passed saved "
        "raw-output, normalized public embedding, repeat, cosine, and gallery "
        "parity on Apple M4."
    ),
    since="1.5",
    constraint=(
        "Only FP32 with `compute_units='cpu_only'` is supported. The official "
        "librefacerec-l package passed two-probe raw parity at 5.71e-6 maximum "
        "relative error, normalized embedding parity at 8.20e-7 maximum "
        "absolute error, cosine error 2.98e-7, bit-exact raw/public repeats, "
        "and the gallery match contract. FP16 measured 1.99e-2 raw relative "
        "error and fails closed; non-CPU runtime requests and legacy/tampered "
        "profile metadata also fail before a native model proxy is created. "
        "This is a fixed batch-one aligned-face component: face detection, "
        "five-point OpenCV alignment, preprocessing, and L2 normalization "
        "remain host operations. The package accepts one aligned crop and "
        "emits one raw embedding; galleries are fingerprint-bound to the "
        "complete artifact. This proves conversion fidelity and orchestration, "
        "not face-recognition accuracy, arbitrary checkpoints, Neural Engine "
        "placement, or device performance."
    ),
)
_add(
    "validated",
    ("eomt",),
    ("semantic", "segment", "panoptic"),
    ("coreml",),
    reason=(
        "Every published trained EoMT Core ML profile passed saved raw-output "
        "and task-aware public-path parity on Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph and public-path evidence covers "
        "LibreEoMTl-sem at 512, LibreEoMTl-seg at 640 and 1280, and "
        "LibreEoMTs/b/l-panoptic at 640. The two-probe raw-output gate requires "
        "maximum relative error 3e-4, minimum relative output sensitivity "
        "1e-6, and at least 100x sensitivity-to-error margin. Public semantic "
        "maps, instance masks/boxes/classes/scores, and whole panoptic segments "
        "passed their task-aware assignment and repeat gates. The graph emits "
        "compact class-query and stride-4 mask logits; exact split/stitch, "
        "top-left padding, query decoding, and final geometry remain host "
        "operations. This proves only the stated checkpoints and profiles, not "
        "model accuracy, arbitrary geometry, Neural Engine placement, or "
        "device performance."
    ),
)
_add(
    "experimental",
    ("segformer",),
    ("semantic",),
    ("coreml",),
    reason=(
        "LibreSegformerb0-sem passed saved FP32 CPU_ONLY raw-logit parity and "
        "the public semantic-map gate on Apple M4; its published ADE20K "
        "checkpoint remains restricted to research or evaluation."
    ),
    constraint=(
        "All b0-b5 eval graphs pass fixed-canvas two-probe TorchScript trace "
        "parity. Apple runtime evidence covers b0 at a fixed 512 canvas with "
        "the 3e-4 raw gate and public map agreement. The architecture/source "
        "is permissive, but published ADE20K weights remain restricted to "
        "research or evaluation."
    ),
)
_add(
    "validated",
    ("swinir",),
    ("restore",),
    ("coreml",),
    reason=(
        "All three trained SwinIR packages passed saved FP32 CPU_ONLY "
        "raw-image parity and the public restored-image gate on Apple M4."
    ),
    since="1.5",
    constraint=(
        "Sizes `s`, `m`, and `l` are enabled at their native 64x64 canvas. "
        "Every full graph has bit-exact two-probe TorchScript parity and Core "
        "ML Tools 9 ML Program conversion evidence. Apple runtime evidence "
        "covers the trained `s`, `m`, and `l` FP32 graphs at the exact 64x64 "
        "source canvas under the 3e-4 raw gate and public restore/repeat gate. "
        "FP16 and non-native canvases remain unproven."
    ),
)
_add(
    "validated",
    ("yolonas",),
    ("detect",),
    ("coreml",),
    reason=(
        "A license-clean generated YOLO-NAS-S detection graph passed saved "
        "raw-output and public detection/repeat parity on Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers a deterministic synthetic "
        "size-S detection fixture at fixed 96 with maximum relative errors "
        "1.79e-7 for boxes and 1.50e-6 for scores. The odd non-square public "
        "path returned five detections with minimum matched IoU 0.9999996, "
        "maximum score error 2.24e-8, exact classes, and bit-exact repeats. "
        "Production geometry remains a square fixed canvas with the native "
        "636-centered RGB longest-side cap. No restricted published weights "
        "were loaded; this proves graph/host-contract fidelity, not model "
        "accuracy, native-640 fidelity, other sizes, arbitrary geometry, "
        "Neural Engine placement, or device performance."
    ),
)
_add(
    "validated",
    ("yolonas",),
    ("pose",),
    ("coreml",),
    reason=(
        "A license-clean generated YOLO-NAS-N pose graph passed saved raw-output "
        "and public pose/repeat parity on Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers a deterministic synthetic "
        "size-N pose fixture at fixed 96 with maximum relative errors 1.91e-7 "
        "for boxes, 2.07e-6 for scores, 1.58e-7 for keypoint XY, and 1.66e-6 "
        "for keypoint confidence. The odd non-square public path returned five "
        "poses with minimum matched IoU 0.9999988, maximum box error 9.16e-5 "
        "source pixel, maximum score error 5.59e-9, maximum keypoint XY error "
        "3.06e-5 source pixel, maximum keypoint-confidence error 1.05e-7, "
        "exact classes, and bit-exact repeats. Production geometry remains a "
        "square fixed canvas with 640 top-left BGR placement and bottom/right "
        "padding. No restricted published weights were loaded; this proves "
        "graph/host-contract fidelity, not model accuracy, native-640 "
        "fidelity, other sizes, arbitrary geometry, Neural Engine placement, "
        "or device performance."
    ),
)
_add(
    "validated",
    ("picosam3",),
    ("segment",),
    ("coreml",),
    reason=(
        "A deterministic license-clean PicoSAM3 ROI component passed saved "
        "raw-output and two-ROI public segmentation/repeat parity on Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers the PicoSAM3-Pico fixed "
        "batch-one 96x96 ROI component with 1.57e-6 maximum relative mask-logit "
        "error, 0.951 relative input sensitivity, and a 608,655x "
        "sensitivity-to-error margin. The odd non-square public path returned "
        "exactly both requested ROIs with exact classes, box and mask IoU 1.0, "
        "maximum score error 1.79e-7, and bit-exact repeats through the default "
        "ALL planner. The host expands each box by 10%, crops and resizes the "
        "ROI, then places mask logits back into the source image; point, text, "
        "and mask prompts remain unsupported. This proves graph and host "
        "placement fidelity, not segmentation accuracy, other checkpoints, "
        "arbitrary profiles, Neural Engine placement, or device performance."
    ),
)
_add(
    "validated",
    ("rtmdet",),
    ("segment",),
    ("coreml",),
    reason=_COREML_M4_RAW_REASON,
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY prepared-graph parity covers trained LibreRTMDett-seg "
        "at a fixed 640 canvas. "
        + _COREML_M4_RAW_SCOPE
        + " "
        "Fixed batch-one canvas divisible by 32. The graph emits three class "
        "maps, three box-distance maps, three 169-parameter dynamic-kernel "
        "maps, and one stride-8 mask feature map; LibreYOLO performs per-level "
        "top-k, class-aware NMS, dynamic mask decoding, and placement on the "
        "host."
    ),
)
_add(
    "validated",
    ("ppocr",),
    ("ocr",),
    ("coreml",),
    reason=(
        "The trained tiny and large PP-OCR multifunction packages passed both "
        "named-function parity and the deterministic public OCR path on Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 CPU_ONLY evidence covers LibrePPOCRt-ocr and LibrePPOCRl-ocr. "
        "Detector and recognizer outputs passed the 3e-4 two-probe raw gate "
        "with sensitivity checks. The public OCR fixture preserved nonempty "
        "text exactly, bounding quads within two source pixels, recognition "
        "and detection confidence within 1e-3, and exact repeats. Both named "
        "functions use bounded-flexible TensorType inputs; DB contours, "
        "perspective crops, reading order, bucketing, and CTC decoding remain "
        "host operations. Export requires a finite rec_max_width and rejects "
        "overflow rather than truncating."
    ),
)
_add(
    "validated",
    ("sam2",),
    ("segment",),
    ("coreml",),
    reason=(
        "All four official SAM2.1 image checkpoints passed saved-package "
        "multifunction parity and the public encode-once/prompt-many path on "
        "Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 with `compute_units='cpu_only'`; LibreSAM2 tiny, small, "
        "base-plus, and large passed the fixed 1024 model-ready encoder and "
        "all 19 admitted point/box/points+box single/multimask functions for "
        "P=1..4. The saved packages passed named-output parity against "
        "pristine PyTorch oracles under the 3e-4 two-probe relative gate, "
        "meaningful input-sensitivity checks, artifact reload, and the public "
        "cached-prompt/dispatch path. Core ML Tools 9 cannot lower SAM2's "
        "captured `aten.where.ScalarOther`, so capture applies exactly "
        "PyTorch's public default decomposition for that one overload and "
        "proves the alternate captured probe bit-exact. Raw-image "
        "preprocessing, prompt transforms, query loops, and mask upscaling "
        "remain host operations. This validates the four released SAM2.1 "
        "image checkpoints and P<=4 packages, not segmentation accuracy, "
        "larger prompt bounds, video memory/tracking, Neural Engine placement, "
        "or device performance."
    ),
)
_add(
    "experimental",
    ("sam3",),
    ("segment",),
    ("coreml",),
    reason=(
        "A strict iOS18/macOS15 direct-fixed-P multifunction ML Program "
        "package is implemented, but SAM3 saved-package runtime parity remains "
        "pending. The Apple M4 campaign reached the gated checkpoint boundary, "
        "but the validation account had not been granted facebook/sam3 access, "
        "so no model graph was loaded or claimed."
    ),
    constraint=(
        "FP32 only. One fixed model-ready image encoder and six source prompt "
        "decoders cover points, boxes, points+boxes, and single/multimask "
        "modes. Each admitted point count is an exact fixed runtime function "
        "bounded by prompt_max_points (default 16); sentinel padding is "
        "forbidden. Raw-image preprocessing, prompt transforms, query loops, "
        "and mask upscaling remain exact host operations. SAM3 is "
        "visual-prompt-only and converted artifacts are local-user-only under "
        "its custom license."
    ),
)
_add(
    "validated",
    ("sam",),
    ("segment",),
    ("coreml",),
    reason=(
        "All three official SAM-1 checkpoints (ViT-B/base, ViT-L/large, and "
        "ViT-H/huge) passed every saved multifunction graph and the public "
        "encode-once/prompt-many path on Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 with `compute_units='cpu_only'`; the fixed 1024 model-ready "
        "encoder and all 19 admitted point/box/points+box single/multimask "
        "functions for P=1..4 passed named-output parity against pristine "
        "PyTorch oracles for the base, large, and huge checkpoints under the "
        "3e-4 two-probe relative gate, meaningful input-sensitivity checks, "
        "artifact reload, and the public cached-prompt/dispatch gate. The base "
        "run exercised all 37 output contracts with a maximum relative error "
        "of 1.175e-4 and a minimum 238.5x "
        "sensitivity-to-conversion-error margin. The eager oracle is released "
        "before Core ML proxy creation because retaining both heavyweight "
        "runtimes can fatal-abort the macOS process; the split-process proof "
        "and the lifecycle-reordered normal pytest both passed. Raw-image "
        "preprocessing, prompt transforms, query loops, and mask upscaling "
        "remain host operations. This validates the three released SAM-1 "
        "checkpoints and P<=4 packages, not segmentation accuracy, larger "
        "prompt bounds, Neural Engine placement, or device performance."
    ),
)
_add(
    "validated",
    ("edgetam",),
    ("segment",),
    ("coreml",),
    reason=(
        "The exact reviewed EdgeTAM edge checkpoint passed every saved "
        "multifunction graph and the public encode-once/prompt-many path on "
        "Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 with `compute_units='cpu_only'`; the fixed 1024 model-ready "
        "encoder and all 19 admitted point/box/points+box single/multimask "
        "functions for P=1..4 passed named-output parity, pristine post-export "
        "reload, and the public cached-prompt/repeat gate. Core ML Tools 9 "
        "cannot lower the captured `aten.where.ScalarOther` used by EdgeTAM's "
        "single-mask stability selection, so LibreYOLO applies exactly the "
        "public PyTorch default decomposition for that overload and records "
        "capture profile `edgetam_where_scalarother_v1`; the captured alternate "
        "probe remained bit-exact. Saved-package maximum relative errors were "
        "1.99e-6 for masks and 9.68e-7 for IoU scores. Raw-image "
        "preprocessing, prompt transforms, query loops, and mask upscaling "
        "remain host operations. This validates the exact released edge "
        "checkpoint and P<=4 package, not segmentation accuracy, larger prompt "
        "bounds, other checkpoints, Neural Engine placement, or device "
        "performance."
    ),
)
_add(
    "validated",
    ("mobilesam",),
    ("segment",),
    ("coreml",),
    reason=(
        "The exact reviewed MobileSAM tiny checkpoint passed every saved "
        "multifunction graph and the public encode-once/prompt-many path on "
        "Apple M4."
    ),
    since="1.5",
    constraint=(
        "FP32 with `compute_units='cpu_only'`; the fixed model-ready encoder "
        "and every admitted point/box/points+box single/multimask decoder "
        "passed named-output parity for exact N=1, N=4, and N=16 packages. "
        + _COREML_M4_RAW_GATE
        + " Public cached-prompt and repeat gates also passed. Core ML Tools "
        "9 on the validation M4 permits exactly "
        "one resident function proxy for this package; the backend churns "
        "decoder proxies because multiple resident native proxies can "
        "fatal-abort the process. Raw-image preprocessing, prompt transforms, "
        "query loops, and mask upscaling remain host operations. This validates "
        "the pinned Apache-2.0 source checkpoint, LibreYOLO mirror, and "
        "tensor-value identity chain only; other structurally compatible "
        "states remain unknown-local, local-only, and non-redistributable. It "
        "does not prove segmentation accuracy, arbitrary package profiles, "
        "Neural Engine placement, or device performance."
    ),
)

# Explicit Core ML walls. A row must not disappear into get_support()'s
# fallback because the generated compatibility table is also the work queue.
_add(
    "validated",
    ("birefnet",),
    ("matte",),
    ("coreml",),
    reason=(
        "The published MIT `l` checkpoint passed saved raw-logit parity and "
        "the public non-square matte path on Apple M4 using Apple's exact "
        "post-9.0 torchvision::deform_conv2d lowering."
    ),
    since="1.5",
    constraint=(
        "Fixed 1024x1024, batch one, FP32 raw matte logits. The trained `l` "
        "package passed the two-probe prepared-graph gate with "
        "`compute_units='cpu_only'`, then independently passed "
        "`ComputeUnit.ALL`. "
        + _COREML_M4_RAW_GATE
        + " An odd 173x257 public RGB image also matched the native "
        "stretch/ImageNet preprocessing and sigmoid/bilinear matte placement "
        "within 3e-4, with exact repeat inference. Stable Core ML Tools 9.0 "
        "predates the required lowering; validation pinned Apple commit "
        "`d5d4267a8849cd39367e17a2978629d3b341d973`, and export "
        "feature-detects the converter implementation instead of trusting its "
        "unchanged version string. The exact `t` graph also converts, but its "
        "weights remain local-user-only and have no trained hardware proof "
        "until upstream provenance is explicit. This proves conversion and "
        "public-path fidelity for the stated `l` checkpoint, not matte "
        "quality, Neural Engine placement, or device performance."
    ),
)
_add(
    "validated",
    ("owlv2",),
    ("detect",),
    ("coreml",),
    reason=(
        "The frozen-vocabulary OWLv2-b16 and OWLv2-l14 packages passed saved "
        "raw-output parity and the public preprocessing/postprocessing path "
        "on Apple M4."
    ),
    since="1.5",
    constraint=(
        "The `b16` checkpoint's fixed 960x960 and `l14` checkpoint's fixed "
        "1008x1008 FP32 TensorType packages passed the two-probe prepared-graph "
        "gate with `compute_units='cpu_only'`. "
        + _COREML_M4_RAW_GATE
        + " An odd non-square public image also matched exact pad-before-Gaussian-"
        "resize preprocessing, boxes, scores, and classes through the default "
        "runtime planner for both sizes. `half=True` fails closed because Core "
        "ML Tools 9 FP16 runtime outputs diverge on Apple Silicon. "
        "The text tower and tokenizer are absent from the runtime artifact, "
        "while the current class vocabulary is frozen into it; changing classes "
        "requires re-export. "
        "Published mirrors are pinned to reviewed Apache-2.0 snapshots. This "
        "does not prove detector accuracy, arbitrary geometry, Neural Engine "
        "placement, or device performance."
        + _COREML_M4_QUERY_ALIGNMENT
    ),
)
_add(
    "blocked",
    ("grounding_dino",),
    ("detect",),
    ("coreml",),
    reason=(
        "Core ML Tools 9 FP16 and FP32 packages convert and execute, but "
        "Apple runtime drift in encoder vision features can reorder nearly "
        "tied top-k proposals before distinct learned rank queries. M4 "
        "hardware replay reproduced material output divergence, and no "
        "bounded tie key preserved native proposal order, so export fails "
        "closed."
    ),
)
_add(
    "validated",
    ("omdet_turbo",),
    ("detect",),
    ("coreml",),
    reason=(
        "The 51.98M-parameter image-only deployment graph derived from the "
        "pinned 115.4M checkpoint is bit-exact with the native detector for "
        "the frozen vocabulary, traces exactly, and passed saved raw-output "
        "and public-path parity on Apple M4."
    ),
    since="1.5",
    constraint=(
        "The released `t` checkpoint uses a fixed 640x640 batch-one FP32 "
        "TensorType package. It passed the two-probe prepared-graph gate with "
        "`compute_units='cpu_only'`. "
        + _COREML_M4_RAW_GATE
        + " An odd non-square public image also "
        "matched exact Torchvision-v2 uint8 bilinear-antialias stretch, boxes, "
        "scores, and classes through the default runtime planner. `half=True` "
        "fails closed because Core ML Tools 9 FP16 changes the top-900 query "
        "selection and runtime outputs on Apple Silicon. The current class "
        "vocabulary and task-language embeddings are frozen; changing classes "
        "requires re-export. The graph emits 900 normalized boxes and per-class "
        "logits for class-aware host NMS. This does not prove detector "
        "accuracy, arbitrary geometry, other checkpoints, Neural Engine "
        "placement, or device performance."
        + _COREML_M4_QUERY_ALIGNMENT
    ),
)
_add(
    "validated",
    ("depth_anything3",),
    ("depth",),
    ("coreml",),
    reason=(
        "The full DA3MONO-LARGE FP32 depth/sky component passed saved raw-output "
        "and deterministic public depth parity on Apple M4."
    ),
    since="1.5",
    constraint=(
        "Only FP32, fixed 504x504, batch one is admitted. The two-probe raw "
        "depth/sky component and seeded square public path passed the 3e-4 "
        "relative gate with exact repeats. FP16 measured 2.066% relative error "
        "on Apple M4 and fails closed. The graph emits positive relative depth "
        "and non-negative sky scores. Sky-region gating, seeded sampling, the "
        "0.99 quantile, reciprocal, and align_corners=True resize remain host "
        "operations. Non-square input retains the documented fixed-stretch "
        "approximation."
    ),
)
_add(
    "blocked",
    ("ov_deim",),
    ("detect",),
    ("coreml",),
    reason=(
        "OV-DEIM remains quarantined from Core ML work: published deployment "
        "assets and subcomponents do not provide one clean, permissive, "
        "provenance-complete code-and-weight chain, and the DINO lineage "
        "crosses a custom-license boundary."
    ),
)
_add(
    "validated",
    ("smolvlm2",),
    ("detect",),
    ("coreml",),
    reason=(
        "The pinned SmolVLM2-500M 2K and 4K portable bundles passed all three "
        "named functions, request-local state, and repeated public inference "
        "on Apple M4."
    ),
    since="1.5",
    constraint=(
        "Only the exact Apache-2.0 500M snapshot and reviewed 2K/4K context "
        "profiles are supported. CPU_ONLY M4 parity measured 0.0258% worst "
        "vision error, exact token embeddings, and 0.0586% worst recurrent "
        "decoder error against PyTorch, with meaningful input sensitivity. "
        "Repeated chat and detection-result paths passed with fresh state. "
        "Vision and decoder compute use FP32; token embedding, public function "
        "I/O, and KV state use FP16. The host owns pinned tokenization, fixed "
        "2048x2048 RGB stretch into 17 crops, image-token merging, causal "
        "controls, greedy decoding, repetition penalty, detokenization, and "
        "parsing. No execution-profile-v2 identity is registered, so explicit "
        "compute_units='cpu_only' remains required. The portable .coremlvlm "
        "bundle requires iOS 18 or macOS 15; 8K is rejected pending peak-memory "
        "proof."
    ),
)
_add(
    "validated",
    ("florence2",),
    ("detect",),
    ("coreml",),
    reason=(
        "The exact MIT Florence-2-base portable bundle passed encoder, seeded "
        "four-state decoder, beam progression, and repeated public inference "
        "on Apple M4."
    ),
    since="1.5",
    constraint=(
        "Only Florence-2-base open-vocabulary detection is admitted. The "
        "CPU_ONLY M4 gate measured 0.0416% worst encoder-cache error and 0.2923% "
        "worst stateful-decoder error against the prepared PyTorch graphs, with "
        "meaningful sensitivity and deterministic repeated public requests. "
        "The fixed 768 image, 1024-token contexts, exact three-beam host search, "
        "pinned offline processor, cross-cache state seeding, and portable "
        ".coremlvlm bundle require iOS 18 or macOS 15. Encoder compute is FP32, "
        "decoder compute and function I/O are FP16, and Apple's writable state "
        "is host-materialized as FP32. No execution-profile-v2 identity is "
        "registered, so explicit compute_units='cpu_only' remains required. "
        "Florence-2-large and other task tokens fail closed."
    ),
)
_add(
    "experimental",
    ("qwen3vl",),
    ("detect",),
    ("coreml",),
    reason=(
        "The exact Apache-2.0 Qwen3-VL-2B portable bundle passed all three "
        "saved Core ML components and repeated public detection inference on "
        "Apple M4."
    ),
    constraint=(
        "Only Qwen/Qwen3-VL-2B-Instruct revision "
        "89644892e4d85e24eaac8bacfd4f463576704203 is admitted. Its "
        "SHA-256-pinned 4,255,140,312-byte FP32 checkpoint exports to a FP32 "
        "vision/deep-stack package, FP16 token-embedding package, and FP16 "
        "stateless decoder package. The CPU_ONLY M4 component gate measured "
        "0.00779% worst vision error and 0.176% worst decoder-logit error "
        "against prepared PyTorch graphs, with meaningful sensitivity and "
        "matching top tokens. The 5,698,319,484-byte production bundle then "
        "generated the exact PyTorch text prefix, class, and box on a real "
        "image (IoU 0.99999993), followed by an exact public predict repeat. "
        "The host owns pinned offline processing, 448-square RGB stretching, "
        "one-image token/deep-stack scatter, left padding, the 3D position and "
        "interleaved MRoPE tables, finite causal masking, repetition penalty, "
        "greedy generation, EOS, detokenization, and parsing. The fixed "
        "profile accepts one image, a 512-token context, at most 48 generated "
        "tokens, and compute_units='cpu_only'. It recomputes the full prefix "
        "for every token and is a fidelity-first profile, not a latency "
        "profile. No execution-profile-v2 identity is registered, so explicit "
        "CPU_ONLY opt-in remains required. The 4B and 8B sizes, videos, other "
        "contexts, arbitrary checkpoints, and Neural Engine placement fail "
        "closed."
    ),
)
_add(
    "blocked",
    ("internvl3",),
    ("detect",),
    ("coreml",),
    reason=(
        "The official permissive architecture is usable, but the current "
        "Transformers-compatible checkpoint lineage is Qwen-tagged rather "
        "than a clean, provenance-complete InternVL deployment payload."
    ),
)
_add(
    "validated",
    ("kosmos2",),
    ("detect",),
    ("coreml",),
    reason=(
        "The exact MIT Kosmos-2-patch14-224 portable bundle passed all three "
        "saved Core ML components and repeated grounded public inference on "
        "Apple M4."
    ),
    since="1.5",
    constraint=(
        "Only microsoft/kosmos-2-patch14-224 revision "
        "e91cfbcb4ce051b6a55bfb5f96165a3bbf5eb82c is admitted. Its "
        "SHA-256-pinned 6,658,052,808-byte FP32 checkpoint exports to separate "
        "vision, token-embedding, and stateless fixed-prefix decoder packages. "
        "The CPU_ONLY M4 campaign passed the 3e-4 two-probe component gate, "
        "meaningful input-sensitivity checks, exact native-versus-deployed "
        "public boxes/classes/scores within the stated tolerances, and "
        "bit-exact deployed repeats. The host owns pinned offline processing, "
        "left padding, image/token embedding merge, greedy generation, "
        "no-repeat trigrams, EOS, detokenization, and grounding-token parsing. "
        "The fixed profile accepts one 224 RGB image, a 128-token context, at "
        "most 48 generated tokens, FP32 compute/I/O, and "
        "compute_units='cpu_only'. It recomputes the full prefix for every "
        "token and is a fidelity-first profile, not a latency profile. No "
        "execution-profile-v2 identity is registered, so explicit CPU_ONLY "
        "opt-in remains required. This proves conversion and host-contract "
        "fidelity, not model accuracy, arbitrary checkpoints or contexts, "
        "Neural Engine placement, or device performance."
    ),
)
_add(
    "blocked",
    ("lfm2vl",),
    ("detect",),
    ("coreml",),
    reason=(
        "Published LFM2-VL weights use the LFM v1 license with a revenue "
        "threshold, so LibreYOLO cannot treat converted artifacts as "
        "unrestricted MIT-compatible deployment weights."
    ),
)
_add(
    "blocked",
    ("locateanything",),
    ("detect", "point"),
    ("coreml",),
    reason=(
        "LocateAnything's published weights and remote-code deployment carry "
        "NVIDIA non-commercial terms, so they cannot become a general "
        "MIT-compatible LibreYOLO Core ML artifact."
    ),
)
_add(
    "blocked",
    ("sensenovavision",),
    ("detect", "segment", "panoptic", "pose", "point", "depth", "ocr"),
    ("coreml",),
    reason=(
        "SenseNova-Vision's published weights are CC BY-NC and its complete "
        "multimodal diffusion/VAE deployment crosses additional custom scope; "
        "a general MIT-compatible Core ML artifact cannot be distributed."
    ),
)


def _demote_unpromoted_coreml_rows() -> None:
    """Keep broad support claims aligned with exact v2 hardware promotions."""
    from .coreml_profiles import COREML_VALIDATED_EXECUTION_PROFILES

    promoted = {
        (profile.family, profile.task)
        for profile in COREML_VALIDATED_EXECUTION_PROFILES.values()
    }
    pending_prefix = (
        "Apple-M4 parity supports the default CPU_ONLY compatibility path. "
        "Execution-profile v2 promotion remains pending a fresh source "
        "identity and final deployment-ABI evidence record. "
    )
    for key, entry in tuple(SUPPORT.items()):
        family, task, fmt = key
        if (
            fmt != "coreml"
            or entry.tier != "validated"
            or (family, task) in promoted
        ):
            continue
        SUPPORT[key] = SupportEntry(
            "experimental",
            reason=(
                "Apple-M4 parity supports the default CPU_ONLY compatibility "
                "path; exact hash-bound execution-profile metadata is pending."
            ),
            since=entry.since,
            constraint=pending_prefix + str(entry.constraint or entry.reason),
        )


_demote_unpromoted_coreml_rows()


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
    "facerec": (
        "The inference-only face head permits only its parity-gated mechanical "
        "ONNX-to-Core ML deployment conversion; other re-export formats remain "
        "out of scope."
    ),
    "depth_anything3": (
        "Depth Anything 3 export is currently limited to its host-postprocessed "
        "Core ML depth component; every other format remains unsupported."
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
