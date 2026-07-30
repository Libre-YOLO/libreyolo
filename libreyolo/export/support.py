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
    "blocked",
    ("rfdetr",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf emits a flatbuffer at the native 384x384 canvas, but LiteRT "
        "cannot allocate it because STRIDED_SLICE receives an input above its "
        "supported 5-D rank."
    ),
)
_add(
    "blocked",
    ("yolo1",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 emits an ONNX_EINSUM custom operation that LiteRT "
        "2.1.2 cannot prepare at the native 448x448 canvas."
    ),
)
_add(
    "blocked",
    ("yolo9_e2e", "yolo9_p2"),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 exports a runnable artifact, but public top-k class "
        "membership changes after LiteRT 2.1.2 conversion."
    ),
)
_add(
    "blocked",
    ("rtmdet",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 exports, reloads, and preserves raw output parity, but "
        "at the native 640x640 canvas public boxes fall to 0.911 IoU with "
        "29.9 px coordinate drift."
    ),
)
_add(
    "blocked",
    ("picodet",),
    ("detect",),
    ("tflite",),
    reason=(
        "LiteRT 2.1.2 cannot prepare the onnx2tf 2.6.7 artifact because a "
        "RESHAPE maps 19,200 input elements to 9,600 output elements."
    ),
)
_add(
    "blocked",
    ("dfine",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf flatbuffer-direct lowering crashes in GatherElements shape "
        "handling with an axis IndexError."
    ),
)
_add(
    "blocked",
    ("ec",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 emits an ONNX_LAYERNORMALIZATION custom operation that "
        "LiteRT 2.1.2 cannot prepare."
    ),
)
_add(
    "blocked",
    ("rtdetr",),
    ("detect",),
    ("tflite",),
    reason=(
        "LiteRT 2.1.2 rejects the onnx2tf 2.6.7 graph because a CONCATENATION "
        "receives incompatible 256 and 1 dimensions."
    ),
)
_add(
    "validated",
    ("yolonas",),
    ("detect",),
    ("tflite",),
    since="1.6",
    constraint="fixed export canvas",
)
_add(
    "blocked",
    ("yolonas",),
    ("pose",),
    ("tflite",),
    reason=(
        "LiteRT rejects the converted pose graph because a CONCATENATION "
        "input has an unsupported/invalid tensor type."
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
    ("mobilenetv4", "convnext", "efficientnetv2", "resnet"),
    ("classify",),
    ("openvino",),
    since="1.6",
    constraint="fixed family-native input resolution",
)
_add(
    "validated",
    ("mobilenetv4", "convnext", "efficientnetv2", "resnet"),
    ("classify",),
    ("tensorrt",),
    since="1.6",
    constraint="FP32 with fixed family-native input resolution",
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
    tuple(fmt for fmt in EXPORT_FORMATS if fmt not in {"onnx", "coreai"}),
    reason=(
        "Frozen-class vision-language export is ONNX-only in v1; re-export "
        "the frozen ONNX graph for a different deployment runtime."
    ),
)
_add(
    "blocked",
    ("dinov2",),
    ("classify",),
    tuple(
        fmt for fmt in EXPORT_FORMATS if fmt not in {"onnx", "torchscript", "coreai"}
    ),
    reason="LibreDINOv2 classify export currently supports ONNX and TorchScript only.",
)
_add(
    "blocked",
    ("clip", "siglip2", "dinov2"),
    ("embed",),
    EXPORT_FORMATS,
    reason=(
        "Embedding export is not implemented in v1; use the native "
        "predict()/embed() API."
    ),
)
_add(
    "blocked",
    ("birefnet", "feynobg"),
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
    ("birefnet", "feynobg"),
    ("matte",),
    ("onnx",),
    reason=(
        "The opset-19 DeformConv graph exports, but ONNX Runtime's CPU "
        "provider has no DeformConv implementation for runtime parity."
    ),
)
_add(
    "blocked",
    ("birefnet", "feynobg"),
    ("matte",),
    ("tensorrt",),
    reason=(
        "TensorRT 10.16 reaches the shared ONNX DeformConv node but cannot "
        "parse it because ModulatedDeformConv2d is absent from the plugin "
        "registry."
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
    "validated",
    ("feynobg",),
    ("matte",),
    ("torchscript",),
    since="1.5",
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
    "validated",
    ("dinov2", "eomt", "lingbotvision"),
    ("semantic",),
    ("openvino",),
    since="1.6",
    constraint="fixed family-native export canvas",
)
_add(
    "validated",
    ("dinov2", "eomt"),
    ("semantic",),
    ("tensorrt",),
    since="1.6",
    constraint="FP32 with a fixed family-native export canvas",
)
_add(
    "experimental",
    ("lingbotvision",),
    ("semantic",),
    ("tensorrt",),
    reason=(
        "TensorRT 10.16 FP32 exports, reloads, and predicts, but repeated "
        "builds produced raw-logit cosine as low as 0.9842, below the 0.999 "
        "promotion gate."
    ),
)
_add(
    "experimental",
    ("pidnet",),
    ("semantic",),
    ("tensorrt",),
    reason=(
        "TensorRT 10.16 FP32 exports and runs, but repeated builds produced "
        "raw-logit cosine as low as 0.9970, below the 0.999 promotion gate."
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
    ("pidnet",),
    ("semantic",),
    ("openvino",),
    since="1.6",
    constraint="fixed square input",
)
_add(
    "validated",
    ("l2cs",),
    ("gaze",),
    ("onnx", "torchscript"),
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
    "validated",
    ("nafnet",),
    ("restore",),
    ("openvino",),
    since="1.6",
    constraint="fixed-resolution export canvas",
)
_add(
    "validated",
    ("nafnet",),
    ("restore",),
    ("tensorrt",),
    since="1.6",
    constraint="FP32 with a fixed-resolution export canvas",
)
_add(
    "blocked",
    ("nafnet",),
    ("restore",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 converts the fixed-canvas graph, but LiteRT 2.1.2 fails "
        "at invoke time because input tensor 4539 lacks data."
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
    ("openvino",),
    since="1.6",
    constraint="fixed-resolution export canvas",
)
_add(
    "validated",
    ("realesrgan",),
    ("restore",),
    ("tensorrt",),
    since="1.6",
    constraint="FP32 with a fixed-resolution export canvas",
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
        "onnx2tf 2.6.7 converts the DINOv2 depth graph, but LiteRT 2.1.2 "
        "cannot broadcast [1,3,3,32] and [1,72,72,32] in a generated ADD."
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
    "validated",
    ("fomo",),
    ("point",),
    ("tensorrt",),
    since="1.6",
    constraint="FP32 with a fixed 96x96 input",
)
_add(
    "validated",
    ("fomo",),
    ("point",),
    ("openvino",),
    since="1.6",
    constraint="fixed square input",
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
    "validated",
    ("teed", "dexined"),
    ("edge",),
    ("onnx",),
    since="1.5",
    constraint="fixed-resolution batch-1 edge-probability canvas",
)
_add(
    "blocked",
    ("teed", "dexined"),
    ("edge",),
    tuple(fmt for fmt in EXPORT_FORMATS if fmt != "onnx"),
    reason=(
        "The edge exported-runtime contract is ONNX-only in v1; add runtime "
        "parity before enabling another format."
    ),
)
_add(
    "validated",
    ("zipdepth",),
    ("depth",),
    ("openvino",),
    since="1.6",
    constraint="fixed-resolution export canvas",
)
_add(
    "experimental",
    ("zipdepth",),
    ("depth",),
    ("tensorrt",),
    reason=(
        "TensorRT 10.16 FP32 exports, reloads, and predicts, but repeated "
        "builds produced raw depth PSNR as low as 30.27 dB, below the 40 dB "
        "promotion gate."
    ),
)
_add(
    "blocked",
    ("zipdepth",),
    ("depth",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 flatbuffer-direct conversion does not support the "
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
    (
        "picodet",
        "rtmdet",
        "yolo1",
        "yolo2",
        "yolo3",
        "yolo4",
        "yolo7",
        "yolo9_e2e",
        "yolo9_p2",
        "yolox",
    ),
    ("detect",),
    ("openvino",),
    since="1.6",
    constraint="fixed export canvas; YOLO1 requires 448x448",
)
_add(
    "validated",
    ("yolo2", "yolo3", "yolo4"),
    ("detect",),
    ("tensorrt",),
    since="1.6",
    constraint="FP32 with a fixed export canvas",
)
_add(
    "validated",
    ("yolo1", "picodet", "rtmdet"),
    ("detect",),
    ("tensorrt",),
    since="1.6",
    constraint="TensorRT 10.16 FP32 with a fixed canvas; YOLO1 requires 448x448",
)
_add(
    "experimental",
    ("yolo7",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "TensorRT 10.16 FP32 exports and reloads, but the permissively licensed "
        "trained checkpoint changes the public top-k class membership."
    ),
)
_add(
    "experimental",
    ("yolo9_e2e",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "Repeated TensorRT 10.16 FP32 engine builds with the permissively "
        "licensed trained checkpoint alternate between public top-k class "
        "drift and parity."
    ),
)
_add(
    "experimental",
    ("yolo9_p2",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "TensorRT 10.16 FP32 exports and reloads, but the pinned permissive "
        "YOLO9 transfer fixture changes the public top-k class membership."
    ),
)
_add(
    "experimental",
    ("yolox",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "The permissively licensed trained checkpoint exports, reloads, and "
        "passes public predict parity, but normalized raw error is 1.6% and "
        "image signal is only 2.1 times the conversion error."
    ),
)
_add(
    "experimental",
    ("rtdetr",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "The permissively licensed trained checkpoint exports, reloads, and "
        "passes public predict parity, but normalized raw outputs drift by "
        "17% to 38% after TensorRT 10.16 conversion."
    ),
)
_add(
    "experimental",
    ("yolonas",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "A deterministic synthetic trained fixture exports, reloads, and "
        "passes public predict parity, but image signal is only 4 to 5 times "
        "the TensorRT conversion error."
    ),
)
_add(
    "experimental",
    ("yolonas",),
    ("pose",),
    ("tensorrt",),
    reason=(
        "A deterministic synthetic trained fixture exports, reloads, and "
        "passes public predict parity, but image signal is only 2 to 6 times "
        "the TensorRT conversion error."
    ),
)
_add(
    "experimental",
    ("dfine",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained checkpoint exports and reloads, but "
        "public top-k class membership changes after TensorRT 10.16 FP32 "
        "conversion."
    ),
)
_add(
    "experimental",
    ("dfine",),
    ("segment",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained segmentation checkpoint exports and "
        "reloads, but public top-k class membership changes after TensorRT "
        "10.16 FP32 conversion."
    ),
)
_add(
    "experimental",
    ("deim",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained checkpoint exports, reloads, and "
        "passes public predict parity, but normalized raw output error is "
        "0.41%, above the 0.1% promotion gate."
    ),
)
_add(
    "experimental",
    ("rtdetrv2",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "A deterministic synthetic fixture exports and reloads, but matched "
        "public boxes drift by at least 8 pixels and fall to 0.231 IoU."
    ),
)
_add(
    "experimental",
    ("rtdetrv4",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "A deterministic synthetic fixture exports, reloads, and predicts, "
        "but repeated TensorRT 10.16 FP32 builds change public top-k class "
        "membership or box geometry; a measured reconstruction reached "
        "0 IoU with 50.4-pixel coordinate drift."
    ),
)
_add(
    "experimental",
    ("ec",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained checkpoint exports, reloads, and "
        "passes public predict parity, but normalized raw output error is "
        "1.2%, above the 0.1% promotion gate."
    ),
)
_add(
    "experimental",
    ("ec",),
    ("pose",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained pose checkpoint exports and reloads, "
        "but matched public boxes fall to 0.920 IoU with 1.43-pixel "
        "coordinate drift."
    ),
)
_add(
    "experimental",
    ("ec",),
    ("segment",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained segmentation checkpoint exports and "
        "reloads, but public top-k class membership changes."
    ),
)
_add(
    "experimental",
    ("rfdetr",),
    ("segment",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained segmentation checkpoint exports and "
        "reloads, but public top-k class membership changes."
    ),
)
_add(
    "experimental",
    ("rfdetr",),
    ("pose",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained pose checkpoint exports and reloads, "
        "but matched public boxes fall to 0.704 IoU with 41.4-pixel "
        "coordinate drift."
    ),
)
_add(
    "experimental",
    ("rfdetr",),
    ("obb",),
    ("tensorrt",),
    reason=(
        "A deterministic synthetic OBB fixture exports and reloads, but "
        "public top-k class membership changes."
    ),
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
    ("yolo2",),
    ("detect",),
    ("tflite",),
    reason=(
        "LiteRT 2.1.2 cannot prepare the onnx2tf 2.6.7 artifact because a "
        "RESHAPE maps 4,225 input elements to one output element."
    ),
)
_add(
    "blocked",
    ("yolo3",),
    ("detect",),
    ("tflite",),
    reason=(
        "A public-domain trained checkpoint exports, reloads, and preserves "
        "normalized raw parity, but public top-k class membership changes."
    ),
)
_add(
    "blocked",
    ("yolo4",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 exports and runs, but public boxes fall to 0 IoU with "
        "176 px coordinate drift on the deterministic full model."
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
    ("yolonas",),
    ("detect", "pose"),
    ("openvino",),
    since="1.6",
    constraint="fixed export canvas",
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
    "validated",
    ("swinir",),
    ("restore",),
    ("onnx", "torchscript", "openvino", "tflite"),
    since="1.6",
    constraint=(
        "fixed export canvas; raw-output and predict parity are validated when "
        "the source dimensions exactly match that canvas. Smaller sources are "
        "padded to the canvas before the exported transformer and can diverge "
        "from native variable-size inference."
    ),
)
_add(
    "validated",
    ("swinir",),
    ("restore",),
    ("tensorrt",),
    since="1.6",
    constraint=(
        "FP32 with a fixed export canvas; raw-output and predict parity are "
        "validated when the source dimensions exactly match that canvas."
    ),
)
_add(
    "blocked",
    ("swinir",),
    ("restore",),
    ("ncnn",),
    reason=(
        "PNNX writes NCNN artifacts after reporting unsupported 5-rank "
        "Permute operations, but the NCNN runtime process exits while loading "
        "or executing the resulting graph."
    ),
)
_add(
    "blocked",
    ("birefnet", "feynobg"),
    ("matte",),
    ("openvino",),
    reason=(
        "OpenVINO 2026.2 cannot lower the shared matte decoder's standard "
        "ONNX DeformConv-19 operation."
    ),
)
_add(
    "blocked",
    ("dfine",),
    ("segment",),
    ("tflite",),
    reason=(
        "onnx2tf flatbuffer-direct lowering crashes in GatherElements shape "
        "handling with an axis IndexError."
    ),
)
_add(
    "blocked",
    ("rtdetrv4",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf flatbuffer-direct lowering crashes in GatherElements shape "
        "handling with an axis IndexError at the native 640x640 canvas."
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
    "validated",
    ("deim",),
    ("detect",),
    ("onnx",),
    since="1.6",
    constraint="DETR query rows are aligned as an unordered set for parity",
)
_add(
    "validated",
    ("dfine", "ec", "rtdetr", "rtdetrv4"),
    ("detect",),
    ("openvino",),
    since="1.6",
    constraint="fixed export canvas",
)
_add(
    "experimental",
    ("deim",),
    ("detect",),
    ("openvino",),
    reason=(
        "After Hungarian query alignment, exactly 95% of boxes meet the "
        "converted-runtime tolerance; validation requires more than 95%."
    ),
)
_add(
    "experimental",
    ("deimv2",),
    ("detect",),
    ("openvino",),
    reason=(
        "After Hungarian query alignment, only 42.3% of scores meet the "
        "converted-runtime tolerance."
    ),
)
_add(
    "experimental",
    ("rtdetrv2",),
    ("detect",),
    ("openvino",),
    reason=(
        "After Hungarian query alignment, 92.3% of boxes meet the "
        "converted-runtime tolerance."
    ),
)
_add(
    "experimental",
    ("deimv2",),
    ("detect",),
    ("onnx",),
    reason=(
        "After Hungarian query alignment, only 43.7% of score values meet "
        "tolerance because ONNX top-k selects a different query set."
    ),
)
_add(
    "experimental",
    ("rtdetrv2",),
    ("detect",),
    ("onnx",),
    reason=(
        "Raw outputs pass after unordered-query alignment, but only 41% of "
        "public predict() boxes match after top-100 selection."
    ),
)
_add(
    "experimental",
    ("rtdetrv4",),
    ("detect",),
    ("onnx",),
    reason=(
        "Raw outputs pass after unordered-query alignment, but only 80% of "
        "public predict() boxes match after top-100 selection."
    ),
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
    ("dfine",),
    ("segment",),
    ("openvino",),
    since="1.6",
    constraint="fixed export canvas",
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
    ("ec",),
    ("segment",),
    ("openvino",),
    since="1.6",
    constraint="fixed 640x640 input",
)
_add(
    "experimental",
    ("ec",),
    ("pose",),
    ("openvino",),
    reason=(
        "After Hungarian query alignment, 93.92% of pose values meet the "
        "converted-runtime tolerance."
    ),
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
    "experimental",
    ("rfdetr",),
    ("segment", "pose", "obb"),
    ("openvino",),
    reason=(
        "After Hungarian query alignment, the measured converted-runtime "
        "match rates remain below validation: segment 87%, OBB 86.17%, "
        "and pose 79%."
    ),
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
    ("segformer",),
    ("semantic",),
    ("onnx", "torchscript"),
    since="1.6",
    constraint="fixed square input divisible by 32",
)
_add(
    "validated",
    ("segformer",),
    ("semantic",),
    ("openvino",),
    since="1.6",
    constraint="fixed square input divisible by 32",
)
_add(
    "blocked",
    ("segformer",),
    ("semantic",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 emits a flatbuffer, but LiteRT 2.1.2 cannot prepare its "
        "attention reshape (1024 input elements versus 256 output elements)."
    ),
)
_add(
    "blocked",
    ("segformer",),
    ("semantic",),
    ("ncnn",),
    reason=(
        "PNNX leaves unsupported pnnx.Expression nodes in the SegFormer graph; "
        "the generated NCNN network reports 'network graph not ready' and has "
        "no runnable input blob."
    ),
)
_add(
    "validated",
    ("dinov2",),
    ("classify",),
    ("onnx", "torchscript"),
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
    "validated",
    ("depth_anything",),
    ("depth",),
    ("openvino",),
    since="1.6",
    constraint="fixed input resolution divisible by 14",
)
_add(
    "validated",
    ("depth_anything",),
    ("depth",),
    ("tensorrt",),
    since="1.6",
    constraint="FP32 with a fixed input resolution divisible by 14",
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
        "LiteRT 2.1.2 cannot invoke the onnx2tf 2.6.7 graph because a "
        "DEPTHWISE_CONV_2D reports 16 filter channels versus zero input channels."
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
    "blocked",
    ("dinov2", "eomt", "pidnet", "lingbotvision"),
    ("semantic",),
    ("coreml",),
    reason="The CoreML wrapper does not implement the dense semantic-logits contract.",
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
    ("moge2",),
    ("normal",),
    ("onnx",),
    since="1.5",
    constraint=(
        "fixed square batch-1 export canvas divisible by 14; exported inference "
        "rejects non-square sources rather than stretching image-plane geometry; "
        "the official MIT ViT-S/B/L normal checkpoints are covered by FP32 "
        "same-canvas native-versus-ONNX angular parity below 0.1 degree"
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
        "The SegFormer Core AI capture path has not been assessed. Its published "
        "weights are non-commercial regardless of export format."
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
        "implemented. The gaze export contract supports ONNX and TorchScript "
        "only.' That is a model-side decision, unchanged by opening the support "
        "gate, so nothing about Core AI is being tested here."
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
    "mesh": (
        "Body-mesh export is blocked until its graph outputs, metadata, and "
        "backend runtime contract are defined."
    ),
    "normal": (
        "This family is not wired to the fixed-canvas dense unit-normal "
        "export and backend renormalization contract."
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
    "l2cs": "The L2CS gaze export contract supports ONNX and TorchScript only.",
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
