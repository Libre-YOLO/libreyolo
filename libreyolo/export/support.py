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
    for family in families:
        for task in tasks:
            for fmt in formats:
                SUPPORT[(family, task, fmt)] = entry


# Existing parity-backed paths. New validated rows must land with a parity test.
_add(
    "validated",
    ("yolo9",),
    ("detect",),
    tuple(fmt for fmt in EXPORT_FORMATS if fmt != "coreml"),
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
    ("onnx", "torchscript", "tensorrt", "openvino"),
    since="1.3",
)
_add("validated", ("rfdetr",), ("detect",), ("tflite",), since="1.3")
_add(
    "validated",
    ("yolo3",),
    ("detect",),
    ("tflite",),
    since="1.3",
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

# Explicitly permitted but not yet parity-validated combinations.
_add(
    "experimental",
    ("rfdetr",),
    ("segment", "pose"),
    ("tflite",),
    reason="The converter path is wired, but parity is not part of the export matrix yet.",
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
    ("yolo2", "yolo4", "yolo7"),
    ("detect",),
    ("tflite",),
    reason="The shared converter path is wired, but family-specific parity is pending.",
)
_add(
    "experimental",
    ("dinov2",),
    ("classify",),
    ("onnx",),
    reason="The classify graph is wired; numeric parity validation is pending.",
)


_TASK_BLOCKS = {
    "point": (
        "Export for point-task models is not implemented yet. Point export needs a "
        "raw heatmap output and backend peak-decoding contract."
    ),
    "semantic": (
        "Export for semantic-segmentation models is not implemented yet. Semantic "
        "export needs a dense-logits output and backend argmax contract."
    ),
    "panoptic": "Panoptic export does not yet have a backend runtime contract.",
    "gaze": (
        "Gaze export needs a two-head logits wrapper and backend expectation decode."
    ),
}

_FAMILY_BLOCKS = {
    "depth_anything": (
        "Depth Anything V2 has not been validated against the depth export contract."
    ),
    "eomt": "EoMT export does not yet have semantic, instance, or panoptic runtime parsing.",
    "pidnet": "PIDNet export awaits the semantic dense-logits runtime contract.",
    "l2cs": "L2CS export awaits the gaze two-head runtime contract.",
    "sam": "Promptable model export is out of scope for the v1 runtime contract.",
    "sam2": "Promptable model export is out of scope for the v1 runtime contract.",
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
