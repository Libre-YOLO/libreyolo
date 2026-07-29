"""Core ML (Apple ``.mlpackage``) export implementation.

Most exported graphs accept one fixed-size, batch-one RGB input. LibrePPOCR is
the deliberate exception: its host-orchestrated detector/recognizer pipeline is
packaged as two named, bounded-flexible TensorType functions. Fixed profiles
otherwise use a uint8 ImageType boundary, except where exact preprocessing
requires a float TensorType. A small in-graph adapter applies the model
family's photometric contract. Spatial geometry remains an
application/runtime responsibility and is declared in metadata rather than
being silently approximated in this file.

The graph preparation is deliberately transactional. Fixed-canvas anchor
grids, RT-DETR state, RF-DETR position embeddings, and converter-specific
module substitutions are restored whether capture, conversion, or saving
succeeds or fails.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ImageNet stats used by RF-DETR.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# CLIP does not use ImageNet normalization. Keep the constants local to the
# exporter so importing the generic Core ML path never pulls in the optional
# CLIP tokenizer dependencies.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Family/task pairs whose fixed-canvas raw tensor contracts are explicit in
# LibreYOLO. These are conversion-capable/experimental until a real macOS
# runtime parity test promotes an individual support-matrix row.
_SUPPORTED_TASKS_BY_FAMILY: dict[str, frozenset[str]] = {
    "birefnet": frozenset({"matte"}),
    "clip": frozenset({"classify"}),
    "convnext": frozenset({"classify"}),
    "deim": frozenset({"detect"}),
    "deimv2": frozenset({"detect"}),
    "depth_anything": frozenset({"depth"}),
    "depth_anything3": frozenset({"depth"}),
    "dfine": frozenset({"detect", "segment"}),
    "dinov2": frozenset({"classify", "semantic"}),
    "ec": frozenset({"detect", "pose", "segment"}),
    "efficientnetv2": frozenset({"classify"}),
    "eomt": frozenset({"panoptic", "segment", "semantic"}),
    "fomo": frozenset({"point"}),
    "grounding_dino": frozenset({"detect"}),
    "l2cs": frozenset({"gaze"}),
    "lingbotvision": frozenset({"semantic"}),
    "mobilenetv4": frozenset({"classify"}),
    "nafnet": frozenset({"restore"}),
    "omdet_turbo": frozenset({"detect"}),
    "owlv2": frozenset({"detect"}),
    "picodet": frozenset({"detect"}),
    "picosam3": frozenset({"segment"}),
    "pidnet": frozenset({"semantic"}),
    "ppocr": frozenset({"ocr"}),
    "realesrgan": frozenset({"restore"}),
    "resnet": frozenset({"classify"}),
    "rfdetr": frozenset({"detect", "obb", "pose", "segment"}),
    "rtdetr": frozenset({"detect"}),
    "rtdetrv2": frozenset({"detect"}),
    "rtdetrv4": frozenset({"detect"}),
    "rtmdet": frozenset({"detect", "segment"}),
    "segformer": frozenset({"semantic"}),
    "siglip2": frozenset({"classify"}),
    "sam": frozenset({"segment"}),
    "sam2": frozenset({"segment"}),
    "sam3": frozenset({"segment"}),
    "edgetam": frozenset({"segment"}),
    "mobilesam": frozenset({"segment"}),
    "swinir": frozenset({"restore"}),
    "yolo1": frozenset({"detect"}),
    "yolo2": frozenset({"detect"}),
    "yolo3": frozenset({"detect"}),
    "yolo4": frozenset({"detect"}),
    "yolo7": frozenset({"detect"}),
    "yolo9": frozenset({"detect"}),
    "yolo9_e2e": frozenset({"detect"}),
    "yolo9_p2": frozenset({"detect"}),
    "yolonas": frozenset({"detect", "pose"}),
    "yolox": frozenset({"detect"}),
    "zipdepth": frozenset({"depth"}),
}
_SUPPORTED_FAMILIES = frozenset(_SUPPORTED_TASKS_BY_FAMILY)


def supported_coreml_exports() -> frozenset[tuple[str, str]]:
    """Return the fixed-canvas family/task contracts implemented here."""
    return frozenset(
        (family, task)
        for family, tasks in _SUPPORTED_TASKS_BY_FAMILY.items()
        for task in tasks
    )


# DEIMv2's larger variants use a separately licensed DINOv3 implementation.
# Do not inspect or export that affected area as part of LibreYOLO's permissive
# Core ML path. Keeping this size gate here prevents a family-level allow-list
# from accidentally crossing the repository's licensing boundary.
_PERMISSIVE_DEIMV2_SIZES = frozenset({"atto", "femto", "pico", "n"})

_CLASSIFIER_IMAGENET_FAMILIES = {
    "convnext",
    "efficientnetv2",
    "mobilenetv4",
    "resnet",
}
_OPENCV_RESIZE_FAMILIES = {
    "depth_anything",
    "lingbotvision",
    "picodet",
    "pidnet",
    "rtdetr",
    "yolo9",
    "yolo9_e2e",
    "yolo9_p2",
    "zipdepth",
}
_DETR_TUPLE_FAMILIES = {
    "deim",
    "deimv2",
    "dfine",
    "ec",
    "rtdetr",
    "rtdetrv2",
    "rtdetrv4",
}

# Families that fundamentally cannot use Apple's NonMaximumSuppression
# layer (DETR set-prediction: top-k over queries × classes, no IoU step).
_NMS_FREE_FAMILIES = {
    "rfdetr": "RF-DETR",
    "rtdetr": "RT-DETR",
    "rtdetrv2": "RT-DETRv2",
    "rtdetrv4": "RT-DETRv4",
    "dfine": "D-FINE",
    "deim": "DEIM",
    "deimv2": "DEIMv2",
    "ec": "EC",
    "eomt": "EoMT",
    "grounding_dino": "Grounding DINO",
    "omdet_turbo": "OMDet-Turbo",
    "owlv2": "OWLv2",
    "ppocr": "LibrePPOCR",
    "yolo9_e2e": "YOLO9-E2E",
}


class _YoloxPreprocess(nn.Module):
    """Map canonical RGB[0,1] input → BGR[0,255] expected by YOLOX."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Any:
        x = x * 255.0
        x = x[:, [2, 1, 0], :, :]
        return self.model(x)


class _RfdetrPreprocess(nn.Module):
    """Map canonical RGB[0,1] input → ImageNet-normalized RGB expected by RF-DETR."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> Any:
        x = (x - self._mean) / self._std
        return self.model(x)


class _ImageNetPreprocess(_RfdetrPreprocess):
    """Map canonical RGB[0,1] input to standard ImageNet-normalized RGB."""


class _CLIPPreprocess(nn.Module):
    """Map canonical RGB[0,1] input to CLIP's normalized RGB tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> Any:
        return self.model((x - self._mean) / self._std)


class _SigLIP2Preprocess(nn.Module):
    """Map canonical RGB[0,1] input to SigLIP2's RGB[-1,1] tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Any:
        return self.model((x - 0.5) / 0.5)


class _FOMOPreprocess(nn.Module):
    """Map canonical RGB[0,1] input to FOMO's RGB[-1,1] tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Any:
        return self.model((x - 0.5) / 0.5)


class _PicoDetPreprocess(nn.Module):
    """Map canonical RGB[0,1] to PicoDet's RGB 0-255 mean/std tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        mean = torch.tensor((123.675, 116.28, 103.53)).view(1, 3, 1, 1)
        std = torch.tensor((58.395, 57.12, 57.375)).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> Any:
        return self.model((x * 255.0 - self._mean) / self._std)


class _RTMDetPreprocess(nn.Module):
    """Map canonical RGB[0,1] to RTMDet's normalized BGR 0-255 tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        mean = torch.tensor((103.53, 116.28, 123.675)).view(1, 3, 1, 1)
        std = torch.tensor((57.375, 57.12, 58.395)).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> Any:
        bgr = x[:, [2, 1, 0], :, :] * 255.0
        return self.model((bgr - self._mean) / self._std)


class _RtdetrOutputAdapter(nn.Module):
    """Flatten RT-DETR's dict output into traceable tensor outputs."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        if isinstance(out, dict):
            return out["pred_logits"], out["pred_boxes"]
        return out


def _wrap_for_family(nn_model: nn.Module, model_family: str | None) -> nn.Module:
    """Legacy Apple wrapper shared verbatim with the Core AI exporter."""
    family = (model_family or "").lower()
    if family == "yolox":
        return _YoloxPreprocess(nn_model)
    if family == "rfdetr":
        return _RfdetrPreprocess(nn_model)
    if family == "rtdetr":
        return _RtdetrOutputAdapter(nn_model)
    # yolo9 and any others use canonical input directly.
    return nn_model


def _wrap_coreml_contract(
    nn_model: nn.Module,
    model_family: str,
    model_task: str,
) -> nn.Module:
    """Apply Core ML's complete photometric contract for one family/task."""
    if model_family == "yolonas":
        from .coreml_yolonas import wrap_yolonas_coreml_contract

        return wrap_yolonas_coreml_contract(nn_model, model_task)
    if model_family == "swinir":
        from .coreml_swinir import wrap_swinir_coreml_contract

        return wrap_swinir_coreml_contract(nn_model)
    if model_family == "picosam3":
        from .coreml_picosam3 import wrap_picosam3_coreml_contract

        return wrap_picosam3_coreml_contract(nn_model)
    if model_family == "eomt":
        from .coreml_eomt import wrap_eomt_coreml_contract

        return wrap_eomt_coreml_contract(nn_model)
    if model_family == "depth_anything3":
        from .coreml_depth_anything3 import (
            wrap_depth_anything3_coreml_contract,
        )

        return wrap_depth_anything3_coreml_contract(nn_model)
    if model_family == "owlv2":
        from .coreml_owlv2 import Owlv2FrozenCoreMLAdapter

        if not isinstance(nn_model, Owlv2FrozenCoreMLAdapter):
            raise TypeError(
                "OWLv2 Core ML export requires the image-only frozen-vocabulary "
                "adapter built by LibreOWLv2.export(format='coreml')."
            )
        return nn_model.eval()
    if model_family == "grounding_dino":
        from .coreml_grounding_dino import (
            GroundingDinoFrozenCoreMLAdapter,
        )

        if not isinstance(nn_model, GroundingDinoFrozenCoreMLAdapter):
            raise TypeError(
                "Grounding DINO Core ML export requires the image-only "
                "frozen-vocabulary adapter built by "
                "LibreGroundingDINO.export(format='coreml')."
            )
        return nn_model.eval()
    if model_family == "omdet_turbo":
        from .coreml_omdet_turbo import (
            OmDetTurboFrozenCoreMLAdapter,
        )

        if not isinstance(nn_model, OmDetTurboFrozenCoreMLAdapter):
            raise TypeError(
                "OMDet-Turbo Core ML export requires the image-only "
                "frozen-vocabulary adapter built by "
                "LibreOMDetTurbo.export(format='coreml')."
            )
        return nn_model.eval()

    wrapped = _wrap_for_family(nn_model, model_family)
    if model_family == "clip":
        wrapped = _CLIPPreprocess(wrapped)
    elif model_family == "siglip2":
        wrapped = _SigLIP2Preprocess(wrapped)
    elif model_family in _CLASSIFIER_IMAGENET_FAMILIES:
        wrapped = _ImageNetPreprocess(wrapped)
    elif model_family == "dinov2" and model_task == "classify":
        # The dense DINOv2 model normalizes internally; its classifier does not.
        wrapped = _ImageNetPreprocess(wrapped)
    elif model_family == "ec":
        wrapped = _ImageNetPreprocess(wrapped)
    elif model_family == "fomo":
        wrapped = _FOMOPreprocess(wrapped)
    elif model_family == "l2cs":
        wrapped = _ImageNetPreprocess(wrapped)
    elif model_family == "birefnet":
        wrapped = _ImageNetPreprocess(wrapped)
    elif model_family == "picodet":
        wrapped = _PicoDetPreprocess(wrapped)
    elif model_family == "rtmdet":
        wrapped = _RTMDetPreprocess(wrapped)
    return wrapped.eval()


def _prepare_yolo9_static_eval(nn_model: nn.Module, dummy: torch.Tensor):
    """Bake YOLOv9 head anchors as constants for the fixed CoreML export size.

    The head's ``_anchor_grid`` rebuilds per-scale anchor grids from traced
    feature-map shapes on every forward. Tracing that produces length-1 int
    tensors (``h * w`` products) that coremltools 9+ rejects in its ``int``
    cast op (numpy 2.x stopped accepting ``int(array([n]))``). A warm-up
    forward populates ``head.anchors`` / ``head.strides``; we then swap
    ``_anchor_grid`` for a stub returning those frozen tensors, so the traced
    graph carries constants instead of shape arithmetic.

    Returns a callable that restores the original ``_anchor_grid``.
    """
    head = getattr(nn_model, "head", None)
    if head is None or not hasattr(head, "_anchor_grid"):
        return lambda: None

    # Warm-up forward: input values are irrelevant — anchors depend only on
    # the feature-map geometry, which is fixed by dummy's H/W.
    with torch.no_grad():
        nn_model(dummy)

    frozen_anchors = head.anchors.detach().clone()
    frozen_strides = head.strides.detach().clone()

    def _const_anchor_grid(feats):
        # The head transposes each returned tensor; pre-transpose so the
        # round-trip reproduces the frozen (post-transpose) values.
        return frozen_anchors.transpose(0, 1), frozen_strides.transpose(0, 1)

    head._anchor_grid = _const_anchor_grid

    def _restore():
        head.__dict__.pop("_anchor_grid", None)

    return _restore


def _prepare_rtdetr_static_eval(nn_model: nn.Module, height: int, width: int) -> None:
    """Precompute RT-DETR eval tensors for the fixed CoreML export image size."""
    # The export pipeline hands us the _RTDETRExportWrapper whose only submodule
    # is ``.model``; descend until we reach the module that actually owns
    # encoder/decoder so the precomputed pos_embed/anchors are not silently
    # dropped. Guarded so an already-unwrapped model still works.
    while (
        getattr(nn_model, "encoder", None) is None
        and getattr(nn_model, "decoder", None) is None
        and getattr(nn_model, "model", None) is not None
    ):
        nn_model = nn_model.model
    device = next(nn_model.parameters(), torch.empty(0)).device
    eval_spatial_size = (height, width)

    encoder = getattr(nn_model, "encoder", None)
    if encoder is not None and hasattr(encoder, "build_2d_sincos_position_embedding"):
        encoder.eval_spatial_size = eval_spatial_size
        for idx in getattr(encoder, "use_encoder_idx", []):
            stride = encoder.feat_strides[idx]
            pos_embed = encoder.build_2d_sincos_position_embedding(
                width // stride,
                height // stride,
                encoder.hidden_dim,
                encoder.pe_temperature,
            ).to(device)
            setattr(encoder, f"pos_embed{idx}", pos_embed)

    decoder = getattr(nn_model, "decoder", None)
    if decoder is not None and hasattr(decoder, "_generate_anchors"):
        decoder.eval_spatial_size = eval_spatial_size
        anchors, valid_mask = decoder._generate_anchors(device=device)
        if "anchors" in decoder._buffers:
            decoder._buffers["anchors"] = anchors
        else:
            decoder.register_buffer("anchors", anchors, persistent=False)
        if "valid_mask" in decoder._buffers:
            decoder._buffers["valid_mask"] = valid_mask
        else:
            decoder.register_buffer("valid_mask", valid_mask, persistent=False)


class _NMSOutputAdapter(nn.Module):
    """Map detector outputs to CoreML NMS inputs: confidence and cxcywh boxes."""

    def __init__(self, model: nn.Module, model_family: str | None):
        super().__init__()
        self.model = model
        self.model_family = (model_family or "").lower()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)

        if self.model_family == "yolox":
            # YOLOX export output: (B, N, 5 + C), cxcywh + objectness + class scores.
            confidence = out[..., 5:] * out[..., 4:5]
            coordinates = out[..., :4]
        elif self.model_family == "yolo9":
            # YOLO9 export output: (B, 4 + C, N), xyxy + class scores.
            pred = out.transpose(1, 2)
            xyxy = pred[..., :4]
            x1, y1, x2, y2 = xyxy.unbind(dim=-1)
            coordinates = torch.stack(
                ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
                dim=-1,
            )
            confidence = pred[..., 4:]
        else:
            raise NotImplementedError(
                f"nms=True is not supported for model family {self.model_family!r}"
            )

        # CoreML's feature-engineering NMS model expects 2D arrays.
        return confidence[0], coordinates[0]


class _CoreMLOutputAdapter(nn.Module):
    """Flatten outputs into the semantic order declared by ``coreml_io``."""

    def __init__(self, model: nn.Module, output_names: list[str]):
        super().__init__()
        self.model = model
        self.output_names = output_names

    def forward(self, x: torch.Tensor) -> Any:
        out = self.model(x)
        if isinstance(out, dict):
            values = tuple(out[name] for name in self.output_names)
        elif isinstance(out, (tuple, list)):
            values = tuple(out)
        else:
            return out
        return values[0] if len(values) == 1 else values


def _validate_export_profile(
    family: str,
    task: str,
    size: str | None,
) -> None:
    tasks = _SUPPORTED_TASKS_BY_FAMILY.get(family)
    if tasks is None:
        raise NotImplementedError(
            f"Core ML export is not supported for model family {family!r}. "
            f"Supported families: {sorted(_SUPPORTED_FAMILIES)}."
        )
    if task not in tasks:
        raise NotImplementedError(
            f"Core ML export is not supported for {family!r} task {task!r}. "
            f"Implemented tasks for this family: {sorted(tasks)}."
        )
    if family == "deimv2" and size not in _PERMISSIVE_DEIMV2_SIZES:
        raise NotImplementedError(
            "Core ML export for DEIMv2 is limited to the permissive HGNet "
            f"sizes {sorted(_PERMISSIVE_DEIMV2_SIZES)}; got size={size!r}. "
            "The larger variants cross the repository's DINOv3 licensing boundary."
        )
    if family == "swinir":
        from .coreml_swinir import SWINIR_COREML_SIZES

        if size not in SWINIR_COREML_SIZES:
            raise NotImplementedError(
                "SwinIR Core ML export supports sizes 's', 'm', and 'l' at "
                f"the fixed 64x64 profile; got size={size!r}."
            )
    if family == "picosam3" and size != "pico":
        raise NotImplementedError(
            "PicoSAM3 Core ML export currently supports only size='pico'."
        )
    if family == "depth_anything3" and size != "l":
        raise NotImplementedError(
            "Depth Anything 3 Core ML export supports only the permissively "
            f"licensed DA3MONO-LARGE size='l'; got size={size!r}."
        )
    if family == "owlv2":
        from .coreml_owlv2 import validate_owlv2_coreml_profile

        validate_owlv2_coreml_profile(size=size)
    if family == "grounding_dino":
        from .coreml_grounding_dino import (
            validate_grounding_dino_coreml_profile,
        )

        validate_grounding_dino_coreml_profile(size=size)
    if family == "omdet_turbo":
        from .coreml_omdet_turbo import validate_omdet_turbo_coreml_profile

        validate_omdet_turbo_coreml_profile(size=size)


def _output_contract(
    family: str,
    task: str,
    *,
    nms: bool,
) -> list[dict[str, Any]]:
    """Return semantic outputs in the exact order emitted by the trace."""

    def output(name: str, role: str, encoding: str | None = None) -> dict[str, Any]:
        item: dict[str, Any] = {"name": name, "role": role}
        if encoding is not None:
            item["encoding"] = encoding
        return item

    if nms:
        return [
            output("confidence", "scores"),
            output("coordinates", "boxes", "cxcywh_pixels"),
        ]
    if task == "classify":
        return [output("class_logits", "class_logits")]
    if family == "birefnet":
        from .coreml_birefnet import birefnet_coreml_output_contract

        return birefnet_coreml_output_contract()
    if family == "segformer":
        from .coreml_segformer import segformer_coreml_output_contract

        return segformer_coreml_output_contract()
    if family == "swinir":
        from .coreml_swinir import swinir_coreml_output_contract

        return swinir_coreml_output_contract()
    if family == "picosam3":
        from .coreml_picosam3 import picosam3_coreml_output_contract

        return picosam3_coreml_output_contract()
    if family == "depth_anything3":
        from .coreml_depth_anything3 import (
            depth_anything3_coreml_output_contract,
        )

        return depth_anything3_coreml_output_contract()
    if family == "owlv2":
        from .coreml_owlv2 import owlv2_coreml_output_contract

        return owlv2_coreml_output_contract()
    if family == "grounding_dino":
        from .coreml_grounding_dino import (
            grounding_dino_coreml_output_contract,
        )

        return grounding_dino_coreml_output_contract()
    if family == "omdet_turbo":
        from .coreml_omdet_turbo import omdet_turbo_coreml_output_contract

        return omdet_turbo_coreml_output_contract()
    if family == "eomt":
        from .coreml_eomt import eomt_coreml_output_contract

        return eomt_coreml_output_contract()
    if task == "semantic":
        return [output("semantic_logits", "semantic_logits")]
    if task == "depth":
        return [output("depth", "depth")]
    if task == "restore":
        return [output("restored", "restored")]
    if task == "point":
        return [output("point_logits", "point_logits")]
    if task == "gaze":
        return [
            output("yaw_logits", "yaw_logits"),
            output("pitch_logits", "pitch_logits"),
        ]
    if family == "yolonas":
        from .coreml_yolonas import yolonas_coreml_output_contract

        return yolonas_coreml_output_contract(task)
    if family == "rtmdet" and task == "segment":
        from .coreml_rtmdet_ins import rtmdet_ins_coreml_output_contract

        return rtmdet_ins_coreml_output_contract()

    if family == "rfdetr":
        values = [
            output("pred_boxes", "boxes", "cxcywh_normalized"),
            output("pred_logits", "class_logits"),
        ]
        if task == "segment":
            values.append(output("pred_masks", "mask_logits"))
        elif task == "pose":
            values.append(output("pred_keypoints", "keypoints"))
        elif task == "obb":
            values.append(output("pred_angles", "angles"))
        return values

    if family == "ec" and task == "pose":
        return [
            output("pred_logits", "class_logits"),
            output("pred_keypoints", "keypoints"),
        ]

    if family in _DETR_TUPLE_FAMILIES:
        values = [
            output("pred_logits", "class_logits"),
            output("pred_boxes", "boxes", "cxcywh_normalized"),
        ]
        if task == "segment":
            if family == "dfine":
                values.append(
                    output(
                        "pred_masks",
                        "mask_probabilities",
                        "sigmoid_probabilities",
                    )
                )
            else:
                values.append(output("pred_masks", "mask_logits"))
        return values

    dense_encoding = (
        "cxcywh_pixels_objectness_classes"
        if family == "yolox"
        else "xyxy_pixels_class_scores"
        if family
        in {
            "yolo1",
            "yolo2",
            "yolo3",
            "yolo4",
            "yolo7",
            "yolo9",
            "yolo9_e2e",
            "yolo9_p2",
        }
        else None
    )
    return [output("prediction", "prediction", dense_encoding)]


def _input_contract(family: str, task: str, size: str | None) -> dict[str, Any]:
    if family == "birefnet":
        from .coreml_birefnet import birefnet_coreml_input_contract

        return birefnet_coreml_input_contract()
    if family == "yolonas":
        from .coreml_yolonas import yolonas_coreml_input_contract

        return yolonas_coreml_input_contract(task)
    if family == "segformer":
        from .coreml_segformer import segformer_coreml_input_contract

        return segformer_coreml_input_contract()
    if family == "swinir":
        from .coreml_swinir import swinir_coreml_input_contract

        return swinir_coreml_input_contract()
    if family == "picosam3":
        from .coreml_picosam3 import picosam3_coreml_input_contract

        return picosam3_coreml_input_contract()
    if family == "depth_anything3":
        from .coreml_depth_anything3 import (
            depth_anything3_coreml_input_contract,
        )

        return depth_anything3_coreml_input_contract()
    if family == "owlv2":
        from .coreml_owlv2 import owlv2_coreml_input_contract

        return owlv2_coreml_input_contract()
    if family == "grounding_dino":
        from .coreml_grounding_dino import (
            grounding_dino_coreml_input_contract,
        )

        return grounding_dino_coreml_input_contract()
    if family == "omdet_turbo":
        from .coreml_omdet_turbo import omdet_turbo_coreml_input_contract

        return omdet_turbo_coreml_input_contract()
    if family == "eomt":
        from .coreml_eomt import eomt_coreml_input_contract

        return eomt_coreml_input_contract(task)
    if family == "rfdetr" and task == "pose":
        # GroupPose uses antialiased bilinear interpolation on a float tensor.
        # A uint8 ImageType would quantize those fractional resized pixels.
        return {
            "name": "image",
            "kind": "tensor",
            "layout": "NCHW",
            "color": "rgb",
            "range": "0_1",
            "geometry": "stretch",
            "interpolation": "bilinear",
            "resize_backend": "torchvision",
            "pad_value": 0,
        }

    geometry = "stretch"
    interpolation = "bicubic" if family == "depth_anything" else "bilinear"
    pad_value = 0

    if family in {
        "clip",
        "convnext",
        "efficientnetv2",
        "mobilenetv4",
        "resnet",
    } or (
        family == "dinov2" and task == "classify"
    ):
        geometry = "center_crop"
        interpolation = "bilinear" if family == "dinov2" else "bicubic"
    elif family in {"yolo2", "yolo3", "yolo4"}:
        geometry = "letterbox_top_left"
        pad_value = 128
    elif family in {
        "pidnet",
        "rtmdet",
        "yolo7",
        "yolo9",
        "yolo9_e2e",
        "yolo9_p2",
        "yolox",
    }:
        geometry = "letterbox_top_left"
        pad_value = 114
    elif family in {"nafnet", "realesrgan"}:
        # A fixed graph cannot reproduce the native "pad only to divisor"
        # behavior for arbitrary smaller images: padding all the way to the
        # exported canvas changes the model's receptive context. Require the
        # source to match the fixed canvas exactly.
        geometry = "native"

    value: dict[str, Any] = {
        "name": "image",
        "kind": "image",
        "layout": "NCHW",
        "color": "rgb",
        "range": "uint8",
        "geometry": geometry,
        "interpolation": interpolation,
        "resize_backend": ("opencv" if family in _OPENCV_RESIZE_FAMILIES else "pillow"),
        "pad_value": pad_value,
    }
    crop_pct = {
        "clip": 1.0,
        "convnext": 0.875,
        "resnet": 0.95,
        "dinov2": 0.875,
        "siglip2": 1.0,
    }.get(family)
    if family == "mobilenetv4":
        crop_pct = 0.875 if size == "s" else 0.95
    elif family == "efficientnetv2":
        crop_pct = {
            "b0": 0.875,
            "b1": 0.882,
            "b2": 0.890,
            "b3": 0.904,
        }.get(size)
    if crop_pct is not None:
        value["crop_pct"] = crop_pct
    return value


def _validation_contract(family: str, task: str) -> dict[str, Any]:
    # Detection-style validators hand exported backends canonical RGB bytes;
    # the family-specific transform lives inside the Core ML graph.
    if family == "depth_anything3":
        from .coreml_depth_anything3 import (
            depth_anything3_coreml_validation_contract,
        )

        return depth_anything3_coreml_validation_contract()
    if family == "birefnet":
        from .coreml_birefnet import birefnet_coreml_validation_contract

        return birefnet_coreml_validation_contract()
    if family == "yolonas":
        from .coreml_yolonas import yolonas_coreml_validation_contract

        return yolonas_coreml_validation_contract(task)
    if family == "segformer":
        from .coreml_segformer import segformer_coreml_validation_contract

        return segformer_coreml_validation_contract()
    if family == "swinir":
        from .coreml_swinir import swinir_coreml_validation_contract

        return swinir_coreml_validation_contract()
    if family == "picosam3":
        from .coreml_picosam3 import picosam3_coreml_validation_contract

        return picosam3_coreml_validation_contract()
    if family == "owlv2":
        from .coreml_owlv2 import owlv2_coreml_validation_contract

        return owlv2_coreml_validation_contract()
    if family == "grounding_dino":
        from .coreml_grounding_dino import (
            grounding_dino_coreml_validation_contract,
        )

        return grounding_dino_coreml_validation_contract()
    if family == "omdet_turbo":
        from .coreml_omdet_turbo import (
            omdet_turbo_coreml_validation_contract,
        )

        return omdet_turbo_coreml_validation_contract()
    if family == "eomt":
        from .coreml_eomt import eomt_coreml_validation_contract

        return eomt_coreml_validation_contract(task)
    if task in {"detect", "segment", "pose", "obb"}:
        return {"color": "rgb", "range": "0_255"}
    if family == "yolox":
        return {"color": "bgr", "range": "0_255"}
    if family == "rtmdet":
        return {"color": "bgr", "range": "imagenet"}
    if family == "fomo":
        return {"color": "rgb", "range": "minus_1_1"}
    if family == "clip":
        return {
            "color": "rgb",
            "range": "standardized",
            "mean": list(_CLIP_MEAN),
            "std": list(_CLIP_STD),
        }
    if family == "siglip2":
        return {"color": "rgb", "range": "minus_1_1"}
    if (
        family in _CLASSIFIER_IMAGENET_FAMILIES
        or family in {"ec", "l2cs", "picodet", "rfdetr"}
        or (family == "dinov2" and task == "classify")
    ):
        return {"color": "rgb", "range": "imagenet"}
    return {"color": "rgb", "range": "0_1"}


def _flatten_tensor_outputs(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        tensors = list(value)
        if all(torch.is_tensor(item) for item in tensors):
            return tensors
    raise RuntimeError(
        "Core ML raw export requires a tensor or a flat tuple/list of tensors; "
        f"the prepared graph returned {type(value).__name__}."
    )


def _expected_dense_candidate_count(
    family: str,
    input_hw: tuple[int, int],
) -> int | None:
    """Return fixed candidate count for dense heads with a known stride ABI."""
    strides = {
        "yolox": (8, 16, 32),
        "picodet": (8, 16, 32, 64),
        "rtmdet": (8, 16, 32),
        "yolo9": (8, 16, 32),
        "yolo9_e2e": (8, 16, 32),
        "yolo9_p2": (4, 8, 16, 32),
    }.get(family)
    if strides is None:
        return None
    height, width = (int(value) for value in input_hw)
    return sum(
        math.ceil(height / stride) * math.ceil(width / stride)
        for stride in strides
    )


def _enrich_output_contract(
    outputs: list[dict[str, Any]],
    tensors: list[torch.Tensor],
) -> list[dict[str, Any]]:
    if len(outputs) != len(tensors):
        names = [item["name"] for item in outputs]
        raise RuntimeError(
            "Core ML semantic output contract does not match the prepared graph: "
            f"declared {names}, received {len(tensors)} tensor(s)."
        )
    enriched = []
    for declared, tensor in zip(outputs, tensors):
        item = dict(declared)
        item["dtype"] = str(tensor.dtype).removeprefix("torch.")
        item["rank"] = int(tensor.ndim)
        item["shape"] = [int(dimension) for dimension in tensor.shape]
        enriched.append(item)
    return enriched


def _validate_output_semantics(
    outputs: list[dict[str, Any]],
    tensors: list[torch.Tensor],
    *,
    family: str,
    task: str,
    nc: int,
    input_hw: tuple[int, int],
    size: str | None,
    nms: bool,
    metadata: dict[str, Any],
) -> None:
    """Reject parser-incompatible graphs before invoking coremltools.

    Names alone are not an ABI.  In particular, a rank-3 detector tensor can
    still have the wrong class width, and two RT-DETR outputs are ambiguous at
    ``nc == 4``.  These checks pin the sample tensors to the semantic contract
    that the backend parser consumes.  Exact fixed shapes are then serialized
    into schema-v2 metadata and checked again at runtime.
    """

    by_name = {item["name"]: tensor for item, tensor in zip(outputs, tensors)}

    def require(condition: bool, message: str) -> None:
        if not condition:
            shapes = {name: tuple(tensor.shape) for name, tensor in by_name.items()}
            raise RuntimeError(
                f"Core ML output contract violation for {family}/{task}: "
                f"{message}; got {shapes}."
            )

    require(len(outputs) == len(tensors), "output count mismatch")
    for name, tensor in by_name.items():
        require(torch.is_tensor(tensor), f"{name!r} is not a tensor")
        require(tensor.is_floating_point(), f"{name!r} must be floating point")
        require(tensor.ndim > 0, f"{name!r} must have positive rank")
        require(all(int(dim) > 0 for dim in tensor.shape), f"{name!r} is empty")
        require(
            bool(torch.isfinite(tensor.detach()).all()),
            f"{name!r} contains NaN or infinity",
        )

    if nms:
        confidence = by_name["confidence"]
        coordinates = by_name["coordinates"]
        require(confidence.ndim == 2, "confidence must have shape [N, nc]")
        require(coordinates.ndim == 2, "coordinates must have shape [N, 4]")
        require(confidence.shape[1] == nc, "confidence class width must equal nc")
        require(coordinates.shape[1] == 4, "coordinates width must be four")
        require(
            confidence.shape[0] == coordinates.shape[0],
            "NMS outputs must share their candidate axis",
        )
        return

    if task == "classify":
        logits = by_name["class_logits"]
        require(tuple(logits.shape) == (1, nc), "class logits must be [1, nc]")
        return

    if family == "depth_anything3":
        from .coreml_depth_anything3 import (
            expected_depth_anything3_coreml_shapes,
            validate_depth_anything3_coreml_raw_outputs,
        )

        expected_shapes = expected_depth_anything3_coreml_shapes(
            batch=1,
            canvas_hw=input_hw,
        )
        require(
            set(by_name) == set(expected_shapes),
            "Depth Anything 3 outputs must match its raw depth/sky ABI",
        )
        for name, expected_shape in expected_shapes.items():
            require(
                tuple(by_name[name].shape) == expected_shape,
                f"{name} must have shape {expected_shape}",
            )
        try:
            validate_depth_anything3_coreml_raw_outputs(
                by_name["relative_depth"],
                by_name["sky_score"],
            )
        except (TypeError, ValueError) as exc:
            require(False, str(exc))
        return

    if family == "owlv2":
        from .coreml_owlv2 import expected_owlv2_coreml_shapes

        expected_shapes = expected_owlv2_coreml_shapes(
            size=str(size),
            nc=nc,
        )
        for name, expected_shape in expected_shapes.items():
            require(
                tuple(by_name[name].shape) == expected_shape,
                f"{name} must have shape {expected_shape}",
            )
        boxes = by_name["pred_boxes"]
        require(
            bool(((boxes >= 0.0) & (boxes <= 1.0)).all()),
            "pred_boxes must be normalized to [0, 1]",
        )
        return

    if family == "grounding_dino":
        from .coreml_grounding_dino import (
            expected_grounding_dino_coreml_shapes,
        )

        sequence_length = int(
            metadata.get("grounding_dino_sequence_length", 0) or 0
        )
        expected_shapes = expected_grounding_dino_coreml_shapes(
            size=str(size),
            sequence_length=sequence_length,
        )
        require(
            set(by_name) == set(expected_shapes),
            "Grounding DINO outputs must match its frozen text/box ABI",
        )
        for name, expected_shape in expected_shapes.items():
            require(
                tuple(by_name[name].shape) == expected_shape,
                f"{name} must have shape {expected_shape}",
            )
        boxes = by_name["pred_boxes"]
        require(
            bool(((boxes >= 0.0) & (boxes <= 1.0)).all()),
            "pred_boxes must be normalized to [0, 1]",
        )
        return

    if family == "omdet_turbo":
        from .coreml_omdet_turbo import (
            expected_omdet_turbo_coreml_shapes,
        )

        expected_shapes = expected_omdet_turbo_coreml_shapes(
            size=str(size),
            nc=nc,
        )
        require(
            set(by_name) == set(expected_shapes),
            "OMDet-Turbo outputs must match its frozen detector ABI",
        )
        for name, expected_shape in expected_shapes.items():
            require(
                tuple(by_name[name].shape) == expected_shape,
                f"{name} must have shape {expected_shape}",
            )
        boxes = by_name["pred_boxes"]
        require(
            bool(((boxes >= 0.0) & (boxes <= 1.0)).all()),
            "pred_boxes must be normalized to [0, 1]",
        )
        return

    if family == "eomt":
        from .coreml_eomt import expected_eomt_coreml_shapes

        num_queries = int(metadata.get("eomt_num_queries", 0) or 0)
        require(num_queries > 0, "EoMT metadata must declare eomt_num_queries")
        expected_shapes = expected_eomt_coreml_shapes(
            nc=nc,
            num_queries=num_queries,
            canvas_hw=input_hw,
        )
        require(
            set(by_name) == set(expected_shapes),
            "EoMT outputs must match its compact query ABI",
        )
        for name, expected_shape in expected_shapes.items():
            require(
                tuple(by_name[name].shape) == expected_shape,
                f"{name} must have shape {expected_shape}",
            )
        return

    if task == "semantic":
        logits = by_name["semantic_logits"]
        require(logits.ndim == 4, "semantic logits must be [1, nc, H, W]")
        require(logits.shape[0] == 1, "semantic batch must be one")
        require(logits.shape[1] == nc, "semantic channel width must equal nc")
        return

    if task == "depth":
        depth = by_name["depth"]
        require(depth.ndim == 4, "depth must be [1, 1, H, W]")
        require(depth.shape[:2] == (1, 1), "depth batch/channel must be [1, 1]")
        return

    if task == "matte":
        matte = by_name["matte"]
        require(matte.ndim == 4, "matte logits must be [1, 1, H, W]")
        require(
            matte.shape[:2] == (1, 1),
            "matte logit batch/channel must be [1, 1]",
        )
        require(
            tuple(matte.shape[-2:]) == tuple(input_hw),
            "matte logit spatial dimensions must equal the export canvas",
        )
        return

    if task == "restore":
        restored = by_name["restored"]
        require(restored.ndim == 4, "restored output must be [1, 3, H, W]")
        require(
            restored.shape[:2] == (1, 3),
            "restored batch/channel must be [1, 3]",
        )
        scale = 1
        if family == "realesrgan":
            scale = {"x2": 2, "x4": 4, "x4t": 4}.get(str(size), 0)
        elif family == "swinir":
            scale = 4
        require(scale > 0, f"unknown restore scale for size={size!r}")
        input_h, input_w = input_hw
        require(
            tuple(restored.shape[-2:]) == (input_h * scale, input_w * scale),
            f"restored spatial shape must equal input times scale={scale}",
        )
        return

    if task == "point":
        logits = by_name["point_logits"]
        require(logits.ndim == 4, "point logits must be [1, nc+1, H, W]")
        require(logits.shape[0] == 1, "point batch must be one")
        require(
            logits.shape[1] == nc + 1,
            "point logits must include one background channel",
        )
        return

    if task == "gaze":
        yaw = by_name["yaw_logits"]
        pitch = by_name["pitch_logits"]
        require(yaw.ndim == pitch.ndim == 2, "gaze logits must be rank two")
        require(yaw.shape[0] == pitch.shape[0] == 1, "gaze batch must be one")
        require(yaw.shape == pitch.shape, "yaw and pitch logits must match")
        num_bins = int(metadata.get("num_bins", 0) or 0)
        require(num_bins > 0, "gaze metadata must declare num_bins")
        require(yaw.shape[1] == num_bins, "gaze logit width must equal num_bins")
        return

    if family == "picosam3":
        mask_logits = by_name["mask_logits"]
        require(
            tuple(mask_logits.shape) == (1, 1, input_hw[0], input_hw[1]),
            "mask_logits must be [1, 1, input_h, input_w]",
        )
        return

    if family == "rtmdet" and task == "segment":
        from .coreml_rtmdet_ins import expected_rtmdet_ins_coreml_shapes

        expected_shapes = expected_rtmdet_ins_coreml_shapes(
            nc=nc,
            canvas_hw=input_hw,
        )
        require(
            set(by_name) == set(expected_shapes),
            "RTMDet-Ins output names must match its fixed raw-output ABI",
        )
        for name, expected_shape in expected_shapes.items():
            require(
                tuple(by_name[name].shape) == expected_shape,
                f"{name} must have shape {expected_shape}",
            )
        return

    if family == "yolonas":
        boxes = by_name["boxes"]
        scores = by_name["scores"]
        require(boxes.ndim == scores.ndim == 3, "boxes/scores must be rank three")
        require(boxes.shape[0] == scores.shape[0] == 1, "batch must be one")
        require(boxes.shape[-1] == 4, "boxes width must be four")
        require(scores.shape[-1] == nc, "score width must equal nc")
        require(boxes.shape[1] == scores.shape[1], "anchor axes must match")
        if task == "pose":
            xy = by_name["keypoints_xy"]
            confidence = by_name["keypoints_conf"]
            require(xy.ndim == 4, "keypoints_xy must be [1, A, K, 2]")
            require(confidence.ndim == 3, "keypoints_conf must be [1, A, K]")
            require(xy.shape[-1] == 2, "keypoint coordinate width must be two")
            require(
                xy.shape[:3] == confidence.shape,
                "keypoint xy/confidence axes must match",
            )
            require(xy.shape[:2] == boxes.shape[:2], "keypoint anchor axis must match")
        return

    if family == "ec" and task == "pose":
        logits = by_name["pred_logits"]
        keypoints = by_name["pred_keypoints"]
        require(logits.ndim == 3, "EC pose logits must be [1, Q, C]")
        require(logits.shape[0] == 1, "EC pose batch must be one")
        require(logits.shape[-1] == 2, "EC pose logit width must be two")
        require(keypoints.ndim in {3, 4}, "EC pose keypoints rank must be 3 or 4")
        require(keypoints.shape[:2] == logits.shape[:2], "EC pose Q axes must match")
        keypoint_count = int(metadata["num_keypoints"])
        if keypoints.ndim == 3:
            require(
                keypoints.shape[-1] == 2 * keypoint_count,
                "EC flattened keypoint width must be 2*num_keypoints",
            )
        else:
            require(
                tuple(keypoints.shape[-2:]) == (keypoint_count, 2),
                "EC keypoints must end in [num_keypoints, 2]",
            )
        return

    if family == "rfdetr" or family in _DETR_TUPLE_FAMILIES:
        if family == "rfdetr":
            boxes = by_name["pred_boxes"]
            logits = by_name["pred_logits"]
        else:
            logits = by_name["pred_logits"]
            boxes = by_name["pred_boxes"]
        require(boxes.ndim == logits.ndim == 3, "DETR boxes/logits must be rank three")
        require(boxes.shape[0] == logits.shape[0] == 1, "DETR batch must be one")
        require(boxes.shape[-1] == 4, "DETR box width must be four")
        require(boxes.shape[1] == logits.shape[1], "DETR query axes must match")
        if family == "rfdetr" and task == "pose":
            schema = metadata.get("num_keypoints_per_class")
            expected_classes = len(schema) if schema else nc
            require(
                logits.shape[-1] == expected_classes,
                "RF-DETR pose logit width must match its class schema",
            )
            keypoints = by_name["pred_keypoints"]
            require(
                keypoints.ndim in {3, 4},
                "RF-DETR keypoints rank must be three or four",
            )
            require(
                keypoints.shape[:2] == boxes.shape[:2],
                "RF-DETR keypoint query axis must match boxes",
            )
            keypoint_count = int(metadata["num_keypoints"])
            keypoint_dim = int(metadata["keypoint_dim"])
            if schema:
                require(keypoints.ndim == 4, "GroupPose keypoints must be rank four")
                require(
                    tuple(keypoints.shape[-2:])
                    == (len(schema) * max(schema), keypoint_dim),
                    "GroupPose keypoints must match padded class slots and dimension",
                )
            elif keypoints.ndim == 3:
                require(
                    keypoints.shape[-1] == keypoint_count * keypoint_dim,
                    "flattened RF-DETR keypoints width must match metadata",
                )
            else:
                require(
                    tuple(keypoints.shape[-2:]) == (keypoint_count, keypoint_dim),
                    "RF-DETR keypoints must match metadata",
                )
        elif family == "rfdetr":
            allowed_widths = {nc, nc + 1}
            if nc == 80:
                allowed_widths.add(91)
            require(
                logits.shape[-1] in allowed_widths,
                "RF-DETR internal logit width must be nc, nc+1, or COCO91",
            )
        else:
            require(logits.shape[-1] == nc, "DETR logit width must equal nc")
        if task == "segment":
            masks = by_name["pred_masks"]
            require(masks.ndim == 4, "DETR masks must be [1, Q, H, W]")
            require(masks.shape[:2] == boxes.shape[:2], "mask query axis must match")
        if family == "rfdetr" and task == "obb":
            angles = by_name["pred_angles"]
            require(angles.ndim == 3, "RF-DETR angles must be [1, Q, 1]")
            require(angles.shape[:2] == boxes.shape[:2], "angle query axis must match")
            require(angles.shape[-1] == 1, "angle width must be one")
        return

    prediction = by_name["prediction"]
    require(prediction.ndim == 3, "detector prediction must be rank three")
    require(prediction.shape[0] == 1, "detector batch must be one")
    if family == "yolox":
        require(
            prediction.shape[-1] == 5 + nc,
            "YOLOX width must be 5+nc",
        )
    elif family in {"picodet", "rtmdet"}:
        require(
            prediction.shape[-1] == 4 + nc,
            f"{family} width must be 4+nc",
        )
    else:
        require(
            prediction.shape[1] == 4 + nc,
            f"{family} channel width must be 4+nc",
        )
    expected_candidates = _expected_dense_candidate_count(family, input_hw)
    if expected_candidates is not None:
        candidate_axis = (
            int(prediction.shape[1])
            if family in {"yolox", "picodet", "rtmdet"}
            else int(prediction.shape[2])
        )
        require(
            candidate_axis == expected_candidates,
            f"{family} candidate count must be {expected_candidates}",
        )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _save_mlpackage_atomic(mlmodel: Any, output_path: str | Path) -> None:
    """Stage, then transactionally replace one package without losing the old one."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".staging",
            dir=destination.parent,
        )
    )
    candidate = staging_root / "candidate.mlpackage"
    previous = staging_root / "previous"
    moved_previous = False
    cleanup_staging = True
    try:
        mlmodel.save(str(candidate))
        if not candidate.exists():
            raise RuntimeError(
                "coremltools returned from save() without creating the staged package."
            )
        if destination.exists() or destination.is_symlink():
            os.replace(destination, previous)
            moved_previous = True
        try:
            os.replace(candidate, destination)
        except Exception as swap_error:
            if moved_previous and previous.exists() and not destination.exists():
                try:
                    os.replace(previous, destination)
                    moved_previous = False
                except Exception as restore_error:
                    # Never delete the only remaining copy. Leave the staging
                    # directory in place and report its exact recovery path.
                    cleanup_staging = False
                    raise RuntimeError(
                        "Core ML package replacement and rollback both failed. "
                        f"The previous artifact remains at {previous}."
                    ) from restore_error
            raise swap_error
    finally:
        # On success this removes the old package now parked at ``previous``.
        # On failure it removes only the incomplete staged candidate.
        if cleanup_staging:
            _remove_path(staging_root)


def _normalize_mlpackage_path(output_path: str | Path) -> str:
    """Return a factory-dispatchable ``.mlpackage`` destination."""
    path = Path(output_path)
    if path.suffix.lower() != ".mlpackage":
        path = path.with_suffix(".mlpackage")
    elif path.suffix != ".mlpackage":
        path = path.with_suffix(".mlpackage")
    return str(path)


def _spec_output_names(mlmodel: Any) -> list[str]:
    try:
        return [str(item.name) for item in mlmodel.get_spec().description.output]
    except (AttributeError, TypeError):
        return []


def _prepare_strict_metadata(
    metadata: dict[str, Any],
    *,
    family: str,
    task: str,
    size: str | None,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Validate and complete metadata needed by the strict backend loader."""
    values = dict(metadata)

    def strict_int(value: Any, *, key: str, minimum: int = 1) -> int:
        if isinstance(value, bool):
            raise ValueError(
                f"Core ML pose metadata {key!r} must be an integer."
            )
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str):
            token = value.strip()
            if not token or not token.lstrip("-").isdigit():
                raise ValueError(
                    f"Core ML pose metadata {key!r} must be an integer."
                )
            parsed = int(token)
        else:
            raise ValueError(
                f"Core ML pose metadata {key!r} must be an integer."
            )
        if parsed < minimum:
            raise ValueError(
                f"Core ML pose metadata {key!r} must be "
                f"{'positive' if minimum == 1 else f'>= {minimum}'}."
            )
        return parsed
    expected = {
        "model_family": family,
        "task": task,
    }
    if size is not None:
        expected["size"] = size
    for key, expected_value in expected.items():
        current = values.get(key)
        if current not in (None, "") and str(current).lower() != expected_value:
            raise ValueError(
                f"Core ML metadata {key!r}={current!r} conflicts with "
                f"the requested value {expected_value!r}."
            )
        values[key] = expected_value

    if not values.get("size"):
        raise ValueError(
            "Core ML export requires a non-empty model size. Pass model_size=... "
            "or metadata['size']."
        )
    if values.get("model_size") not in (None, "", values["size"]):
        raise ValueError(
            "Core ML metadata 'model_size' conflicts with canonical 'size': "
            f"{values['model_size']!r} != {values['size']!r}."
        )
    values["model_size"] = values["size"]
    values.setdefault("supported_tasks", [task])
    values.setdefault("default_task", task)
    expected_spatial = {
        "imgsz": max(height, width),
        "imgsz_h": height,
        "imgsz_w": width,
    }
    for key, expected_value in expected_spatial.items():
        current = values.get(key)
        if current not in (None, ""):
            try:
                current_value = int(current)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Core ML metadata {key!r} must be an integer."
                ) from exc
            if current_value != expected_value:
                raise ValueError(
                    f"Core ML metadata {key!r}={current_value} disagrees with "
                    f"the traced canvas value {expected_value}."
                )
        values[key] = expected_value

    required = {
        "schema_version",
        "libreyolo_version",
        "names",
        "nc",
    }
    missing = sorted(
        key for key in required if key not in values or values[key] in (None, "")
    )
    if missing:
        raise ValueError(
            "Core ML export requires complete backend-loadable metadata; "
            f"missing {missing}."
        )

    raw_names = values["names"]
    if isinstance(raw_names, str):
        try:
            raw_names = json.loads(raw_names)
        except json.JSONDecodeError as exc:
            raise ValueError("Core ML metadata 'names' must be valid JSON.") from exc
    if not isinstance(raw_names, dict) or not raw_names:
        raise ValueError("Core ML metadata 'names' must be a non-empty mapping.")
    try:
        names = {int(key): str(value) for key, value in raw_names.items()}
        nc = int(values["nc"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Core ML metadata 'names' keys and 'nc' must be integers."
        ) from exc
    if sorted(names) != list(range(len(names))) or nc != len(names):
        raise ValueError(
            "Core ML metadata names must be contiguous from zero and match nc; "
            f"got keys={sorted(names)}, nc={nc}."
        )
    values["names"] = {str(key): value for key, value in names.items()}
    values["nc"] = nc
    existing_nb_classes = values.get("nb_classes")
    if existing_nb_classes not in (None, ""):
        try:
            parsed_nb_classes = int(existing_nb_classes)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Core ML metadata 'nb_classes' must be an integer."
            ) from exc
        if parsed_nb_classes != nc:
            raise ValueError(
                "Core ML metadata 'nb_classes' must match canonical 'nc': "
                f"{parsed_nb_classes} != {nc}."
            )
    values["nb_classes"] = nc

    if task == "pose":
        for key in ("num_keypoints", "keypoint_dim"):
            values[key] = strict_int(values.get(key), key=key)

        raw_schema = values.get("num_keypoints_per_class")
        schema = None
        if raw_schema not in (None, ""):
            if isinstance(raw_schema, str):
                try:
                    raw_schema = json.loads(raw_schema)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Core ML pose metadata 'num_keypoints_per_class' "
                        "must be valid JSON."
                    ) from exc
            if not isinstance(raw_schema, (list, tuple)) or not raw_schema:
                raise ValueError(
                    "Core ML pose metadata 'num_keypoints_per_class' must be "
                    "a non-empty integer list."
                )
            try:
                schema = [
                    strict_int(
                        count,
                        key=f"num_keypoints_per_class[{index}]",
                        minimum=0,
                    )
                    for index, count in enumerate(raw_schema)
                ]
            except ValueError as exc:
                raise ValueError(
                    "Core ML pose metadata 'num_keypoints_per_class' must "
                    "contain nonnegative integers."
                ) from exc
            if not any(count > 0 for count in schema):
                raise ValueError(
                    "Core ML pose metadata 'num_keypoints_per_class' must be "
                    "nonnegative and contain at least one active class."
                )
            values["num_keypoints_per_class"] = schema

        if family == "rfdetr" and schema:
            if values["keypoint_dim"] != 8:
                raise ValueError(
                    "RF-DETR GroupPose Core ML metadata requires keypoint_dim=8."
                )
            if values["num_keypoints"] != max(schema):
                raise ValueError(
                    "RF-DETR GroupPose num_keypoints must equal the maximum "
                    "num_keypoints_per_class value."
                )
            if nc != sum(count > 0 for count in schema):
                raise ValueError(
                    "RF-DETR GroupPose public nc must equal the number of "
                    "keypoint-bearing classes."
                )

        pose_encoding = (
            "rfdetr_grouppose_padded_v1"
            if family == "rfdetr" and schema
            else "rfdetr_flat_keypoints_v1"
            if family == "rfdetr"
            else "yolonas_split_xy_conf_v1"
            if family == "yolonas"
            else "ec_normalized_xy_v1"
            if family == "ec"
            else "keypoints_v1"
        )
        existing_pose_encoding = values.get("pose_encoding")
        if existing_pose_encoding not in (None, "", pose_encoding):
            raise ValueError(
                "Core ML pose_encoding conflicts with the exported family "
                f"contract: {existing_pose_encoding!r} != {pose_encoding!r}."
            )
        values["pose_encoding"] = pose_encoding

    raw_tasks = values["supported_tasks"]
    if isinstance(raw_tasks, str):
        try:
            raw_tasks = json.loads(raw_tasks)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Core ML metadata 'supported_tasks' must be valid JSON."
            ) from exc
    if not isinstance(raw_tasks, (list, tuple)) or task not in raw_tasks:
        raise ValueError(
            "Core ML metadata supported_tasks must contain the exported task "
            f"{task!r}; got {raw_tasks!r}."
        )
    # A Core ML package is task-specific even when its source checkpoint can
    # build several heads. Advertising the checkpoint's broader task set would
    # let the loader select a parser whose arity disagrees with this graph.
    values["supported_tasks"] = [task]
    values["default_task"] = task
    if task == "classify":
        activation = str(
            values.get("classification_activation") or "softmax"
        ).strip().lower()
        if activation not in {"softmax", "sigmoid"}:
            raise ValueError(
                "Core ML classification_activation must be 'softmax' or "
                f"'sigmoid', got {values.get('classification_activation')!r}."
            )
        values["classification_activation"] = activation

    dynamic_value = values.get("dynamic", False)
    if (
        dynamic_value is True
        or isinstance(dynamic_value, str)
        and dynamic_value.strip().lower() in {"1", "true", "yes"}
    ):
        raise ValueError(
            "Core ML metadata cannot declare dynamic=True for a fixed input contract."
        )
    values["dynamic"] = False
    return values


def prepare_frozen_classifier_coreml_export(
    model: Any,
    kwargs: dict[str, Any],
    *,
    default_output: str,
) -> tuple[int, str, dict[str, Any], str, str]:
    """Validate a frozen CLIP-style Core ML request and build its metadata.

    CLIP and SigLIP2 do not expose a single image-only ``model.forward``:
    their public models contain both image and text towers. Their export
    methods therefore construct a frozen image classifier first, then call
    :func:`export_coreml`. This helper deliberately runs the same precision,
    support-policy, optional-dependency, and metadata checks as the shared
    :class:`~libreyolo.export.exporter.CoreMLExporter` before either tower is
    moved to CPU.
    """
    from .exporter import CoreMLExporter

    options = dict(kwargs)
    imgsz = options.pop("imgsz", None)
    output_path = options.pop("output_path", None)
    output_alias = options.pop("output", None)
    if (
        output_path not in (None, "")
        and output_alias not in (None, "")
        and str(output_path) != str(output_alias)
    ):
        raise ValueError(
            "Pass only one Core ML destination: output_path= or output=."
        )
    output_path = output_path or output_alias or default_output

    half = bool(options.pop("half", False))
    int8 = bool(options.pop("int8", False))
    data = options.pop("data", None)
    dynamic = bool(options.pop("dynamic", False))
    batch = int(options.pop("batch", 1))
    nms = bool(options.pop("nms", False))
    device = options.pop("device", None)
    compute_units = str(options.pop("compute_units", "all")).lower()
    conf = options.pop("conf", 0.25)
    iou = options.pop("iou", 0.45)
    max_det = options.pop("max_det", 300)

    # Accepted by the shared public export signature but irrelevant to this
    # direct frozen-class graph path.
    for name in (
        "opset",
        "simplify",
        "verbose",
        "fraction",
        "allow_download_scripts",
        "_pre_trace_hook",
    ):
        options.pop(name, None)
    if options:
        names = ", ".join(sorted(options))
        raise TypeError(f"Unsupported Core ML frozen-class export options: {names}")

    if dynamic:
        raise NotImplementedError(
            "Core ML frozen-class ImageType export uses a fixed input shape; "
            "dynamic=True is not supported."
        )
    if batch != 1:
        raise ValueError(
            "Core ML frozen-class ImageType export requires batch=1; "
            f"got batch={batch}."
        )
    if device not in (None, "auto", "cpu", torch.device("cpu")):
        raise NotImplementedError(
            "Core ML frozen-class conversion traces on CPU; pass device='cpu', "
            "device='auto', or omit device."
        )

    native_size = int(model.input_size)
    if imgsz is None:
        height = width = native_size
    elif isinstance(imgsz, (tuple, list)):
        if len(imgsz) != 2:
            raise ValueError(f"imgsz must be an int or (height, width), got {imgsz}")
        height, width = (int(imgsz[0]), int(imgsz[1]))
    else:
        height = width = int(imgsz)
    if height <= 0 or width <= 0:
        raise ValueError(f"imgsz values must be positive, got {(height, width)}")
    if height != width:
        raise NotImplementedError(
            "Frozen CLIP-style Core ML export requires a square input."
        )
    if height != native_size:
        raise NotImplementedError(
            "Frozen CLIP-style Core ML export requires the model's native "
            f"{native_size}x{native_size} input because its learned image "
            f"position table is resolution-specific; got {height}x{width}."
        )

    exporter = CoreMLExporter(model)
    half, int8 = exporter._validate(half, int8, data)
    exporter._preflight(
        half=half,
        int8=int8,
        data=data,
        nms=nms,
        compute_units=compute_units,
        conf=conf,
        iou=iou,
        max_det=max_det,
    )
    precision = "fp16" if half else "fp32"
    metadata = exporter._build_metadata(
        precision,
        False,
        None,
        imgsz=(height, width),
    )
    metadata.update(
        {
            "frozen_classes": True,
            "classification_activation": (
                "sigmoid" if bool(getattr(model, "multi_label", False)) else "softmax"
            ),
        }
    )

    destination = Path(output_path)
    if destination.suffix.lower() != ".mlpackage":
        destination = destination.with_suffix(".mlpackage")
    return height, str(destination), metadata, precision, compute_units


def _stringify_metadata(metadata: dict) -> dict:
    """Convert metadata values to strings (CoreML user_defined_metadata requires str).

    Dict-typed values (e.g. ``names``) are JSON-encoded so they round-trip cleanly.
    """
    out: dict[str, str] = {}
    for k, v in metadata.items():
        if isinstance(v, (dict, list, tuple)):
            out[str(k)] = json.dumps(v)
        else:
            out[str(k)] = str(v)
    return out


def _to_compute_unit(compute_units: str):
    """Map a string compute_units value to a coremltools.ComputeUnit enum.

    Accepted: 'all', 'cpu_and_gpu', 'cpu_and_ne', 'cpu_only' (case-insensitive).
    """
    import coremltools as ct

    key = compute_units.lower()
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    if key not in mapping:
        raise ValueError(
            f"Invalid compute_units {compute_units!r}. "
            f"Must be one of: {sorted(mapping)}"
        )
    return mapping[key]


def _canonical_trace_probe(dummy: torch.Tensor) -> torch.Tensor:
    """Build a deterministic, chromatic, non-constant RGB probe in [0,1]."""
    height, width = int(dummy.shape[-2]), int(dummy.shape[-1])
    ys = torch.linspace(
        0.05, 0.95, height, dtype=torch.float32, device=dummy.device
    ).view(1, 1, height, 1)
    xs = torch.linspace(0.1, 0.9, width, dtype=torch.float32, device=dummy.device).view(
        1, 1, 1, width
    )
    red = xs.expand(1, 1, height, width)
    green = ys.expand(1, 1, height, width)
    blue = (0.65 * red + 0.35 * green).clamp(0.0, 1.0)
    return torch.cat((red, green, blue), dim=1).contiguous()


def _ppocr_trace_probes(
    profile: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build bounded probes that exercise both PPOCR dynamic axes."""
    from .coreml_ppocr import PPOCR_COREML_RECOGNIZER_MIN_WIDTH

    det_first_h = min(profile.det_tensor_upper, 64)
    det_first_w = min(profile.det_tensor_upper, 96)
    det_second_h = det_first_w
    det_second_w = det_first_h
    det_first = torch.linspace(
        -2.0,
        2.0,
        3 * det_first_h * det_first_w,
        dtype=torch.float32,
    ).reshape(1, 3, det_first_h, det_first_w)
    det_second = torch.linspace(
        2.0,
        -2.0,
        3 * det_second_h * det_second_w,
        dtype=torch.float32,
    ).reshape(1, 3, det_second_h, det_second_w)

    rec_first_width = PPOCR_COREML_RECOGNIZER_MIN_WIDTH
    # Widths 324 and 325 straddle a real CTC timestep boundary. Prefer 325
    # whenever the caller's explicit profile admits it.
    rec_second_width = min(profile.rec_max_width, 325)
    rec_second_batch = min(profile.rec_batch_max, 2)
    rec_first = torch.linspace(
        -1.0,
        1.0,
        3 * 48 * rec_first_width,
        dtype=torch.float32,
    ).reshape(1, 3, 48, rec_first_width)
    rec_second = torch.linspace(
        1.0,
        -1.0,
        rec_second_batch * 3 * 48 * rec_second_width,
        dtype=torch.float32,
    ).reshape(rec_second_batch, 3, 48, rec_second_width)
    return det_first, det_second, rec_first, rec_second


def _validate_ppocr_mil_output(
    mlmodel: Any,
    *,
    function_name: str,
    output_name: str,
    expected_rank: int,
    fixed_axes: dict[int, int],
) -> None:
    """Check the converted MIL symbolic type before multifunction packaging.

    Core ML's supported ML Program API serializes flexible input ranges, but
    its output ``FeatureDescription`` intentionally has no dynamic shape
    range. The in-memory MIL program is therefore the only conversion-time
    surface that exposes fixed output axes such as the CTC class width.
    Runtime loading additionally validates every realized output shape.
    """
    program = getattr(mlmodel, "_mil_program", None)
    functions = getattr(program, "functions", None)
    function = functions.get("main") if isinstance(functions, dict) else None
    if function is None:
        raise RuntimeError(
            f"Core ML conversion for PPOCR {function_name!r} did not retain "
            "an inspectable MIL main function."
        )
    outputs = list(getattr(function, "outputs", ()))
    if len(outputs) != 1 or str(outputs[0].name) != output_name:
        raise RuntimeError(
            f"Core ML conversion for PPOCR {function_name!r} changed its "
            f"output ABI: expected [{output_name!r}], got "
            f"{[str(value.name) for value in outputs]!r}."
        )
    output = outputs[0]
    shape = tuple(getattr(output, "shape", ()))
    if len(shape) != expected_rank:
        raise RuntimeError(
            f"Core ML conversion for PPOCR {function_name!r} produced rank "
            f"{len(shape)}; expected {expected_rank}."
        )
    for axis, expected in fixed_axes.items():
        value = shape[axis]
        try:
            actual = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Core ML conversion for PPOCR {function_name!r} lost fixed "
                f"output axis {axis}={expected}; got {value!r}."
            ) from exc
        if actual != expected:
            raise RuntimeError(
                f"Core ML conversion for PPOCR {function_name!r} changed "
                f"output axis {axis}: expected {expected}, got {actual}."
            )

    from coremltools.converters.mil.mil import types

    if output.dtype != types.fp32:
        raise RuntimeError(
            f"Core ML conversion for PPOCR {function_name!r} changed its "
            "output dtype; FP32 is required."
        )


def _validate_ppocr_unifunction_spec(
    mlmodel: Any,
    *,
    function_name: str,
    input_name: str,
    output_name: str,
    expected_ranges: tuple[tuple[int, int], ...],
) -> None:
    """Pin a converted component's serialized FP32 flexible input ABI."""
    spec = mlmodel.get_spec()
    description = getattr(spec, "description", None)
    inputs = list(getattr(description, "input", ()) or ())
    outputs = list(getattr(description, "output", ()) or ())
    if [str(value.name) for value in inputs] != [input_name]:
        raise RuntimeError(
            f"Core ML conversion for PPOCR {function_name!r} changed its "
            f"input name: {[str(value.name) for value in inputs]!r}."
        )
    if [str(value.name) for value in outputs] != [output_name]:
        raise RuntimeError(
            f"Core ML conversion for PPOCR {function_name!r} changed its "
            f"output name: {[str(value.name) for value in outputs]!r}."
        )

    input_array = inputs[0].type.multiArrayType
    output_array = outputs[0].type.multiArrayType
    if int(input_array.dataType) != 65568 or int(output_array.dataType) != 65568:
        raise RuntimeError(
            f"Core ML conversion for PPOCR {function_name!r} must expose "
            "FLOAT32 TensorType input and output."
        )
    flexibility = input_array.WhichOneof("ShapeFlexibility")
    ranges = tuple(
        (int(value.lowerBound), int(value.upperBound))
        for value in input_array.shapeRange.sizeRanges
    )
    if flexibility != "shapeRange" or ranges != expected_ranges:
        raise RuntimeError(
            f"Core ML conversion for PPOCR {function_name!r} changed its "
            f"input ranges: expected {expected_ranges}, got {ranges}."
        )


def _validate_ppocr_multifunction_spec(
    spec: Any,
    *,
    profile: Any,
) -> None:
    """Validate the package-level function table after Core ML deduplication."""
    from .coreml_ppocr import (
        PPOCR_COREML_DEFAULT_FUNCTION,
        PPOCR_COREML_DETECTOR_FUNCTION,
        PPOCR_COREML_DETECTOR_INPUT,
        PPOCR_COREML_DETECTOR_OUTPUT,
        PPOCR_COREML_FUNCTION_NAMES,
        PPOCR_COREML_RECOGNIZER_FUNCTION,
        PPOCR_COREML_RECOGNIZER_INPUT,
        PPOCR_COREML_RECOGNIZER_OUTPUT,
    )

    description = getattr(spec, "description", None)
    if list(getattr(description, "input", ()) or ()) or list(
        getattr(description, "output", ()) or ()
    ):
        raise RuntimeError(
            "LibrePPOCR multifunction package must not expose a false "
            "top-level single-function interface."
        )
    if str(getattr(description, "defaultFunctionName", "")) != (
        PPOCR_COREML_DEFAULT_FUNCTION
    ):
        raise RuntimeError(
            "LibrePPOCR multifunction package has the wrong default function."
        )
    functions = list(getattr(description, "functions", ()) or ())
    if [str(value.name) for value in functions] != list(PPOCR_COREML_FUNCTION_NAMES):
        raise RuntimeError(
            "LibrePPOCR multifunction package function order/names changed: "
            f"{[str(value.name) for value in functions]!r}."
        )
    by_name = {str(value.name): value for value in functions}
    expected = {
        PPOCR_COREML_DETECTOR_FUNCTION: (
            PPOCR_COREML_DETECTOR_INPUT,
            PPOCR_COREML_DETECTOR_OUTPUT,
            (
                (1, 1),
                (3, 3),
                (32, profile.det_tensor_upper),
                (32, profile.det_tensor_upper),
            ),
        ),
        PPOCR_COREML_RECOGNIZER_FUNCTION: (
            PPOCR_COREML_RECOGNIZER_INPUT,
            PPOCR_COREML_RECOGNIZER_OUTPUT,
            (
                (1, profile.rec_batch_max),
                (3, 3),
                (48, 48),
                (320, profile.rec_max_width),
            ),
        ),
    }
    for name, (input_name, output_name, expected_ranges) in expected.items():
        function = by_name[name]
        inputs = list(function.input)
        outputs = list(function.output)
        if [str(value.name) for value in inputs] != [input_name]:
            raise RuntimeError(
                f"LibrePPOCR function {name!r} has an invalid input interface."
            )
        if [str(value.name) for value in outputs] != [output_name]:
            raise RuntimeError(
                f"LibrePPOCR function {name!r} has an invalid output interface."
            )
        input_array = inputs[0].type.multiArrayType
        output_array = outputs[0].type.multiArrayType
        ranges = tuple(
            (int(value.lowerBound), int(value.upperBound))
            for value in input_array.shapeRange.sizeRanges
        )
        if (
            int(input_array.dataType) != 65568
            or int(output_array.dataType) != 65568
            or input_array.WhichOneof("ShapeFlexibility") != "shapeRange"
            or ranges != expected_ranges
        ):
            raise RuntimeError(
                f"LibrePPOCR function {name!r} does not match its FP32 "
                f"bounded-flexible ABI: ranges={ranges!r}."
            )


def _sam_expected_ranges(feature: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    """Normalize one SAM manifest shape to Core ML lower/upper bounds."""
    ranges: list[tuple[int, int]] = []
    for axis in feature["shape"]:
        if axis["kind"] == "fixed":
            value = int(axis["value"])
            ranges.append((value, value))
        elif axis["kind"] == "range":
            ranges.append(
                (
                    int(axis["lower_bound"]),
                    int(axis["upper_bound"]),
                )
            )
        else:
            raise RuntimeError(
                f"Unsupported LibreSAM Core ML axis kind {axis['kind']!r}."
            )
    return tuple(ranges)


def _sam_feature_ranges(feature: Any) -> tuple[tuple[int, int], ...]:
    """Read fixed or bounded tensor dimensions from a Core ML feature."""
    array = feature.type.multiArrayType
    if array.WhichOneof("ShapeFlexibility") == "shapeRange":
        return tuple(
            (int(value.lowerBound), int(value.upperBound))
            for value in array.shapeRange.sizeRanges
        )
    return tuple((int(value), int(value)) for value in array.shape)


def _validate_sam_function_description(
    description: Any,
    *,
    function_name: str,
    profile: Any,
) -> None:
    """Pin one serialized SAM function's exact names, dtypes, and input bounds."""
    from .coreml_sam import sam_coreml_function_contracts

    contract = sam_coreml_function_contracts(profile)[function_name]
    inputs = list(getattr(description, "input", ()) or ())
    outputs = list(getattr(description, "output", ()) or ())
    expected_input_names = [item["name"] for item in contract["inputs"]]
    expected_output_names = [item["name"] for item in contract["outputs"]]
    if [str(value.name) for value in inputs] != expected_input_names:
        raise RuntimeError(
            f"LibreSAM function {function_name!r} changed its input ABI: "
            f"{[str(value.name) for value in inputs]!r}."
        )
    if [str(value.name) for value in outputs] != expected_output_names:
        raise RuntimeError(
            f"LibreSAM function {function_name!r} changed its output ABI: "
            f"{[str(value.name) for value in outputs]!r}."
        )

    dtype_codes = {"float32": 65568, "int32": 131104}
    for actual, expected in zip(inputs, contract["inputs"]):
        array = actual.type.multiArrayType
        expected_dtype = dtype_codes[expected["dtype"]]
        if int(array.dataType) != expected_dtype:
            raise RuntimeError(
                f"LibreSAM function {function_name!r} input "
                f"{expected['name']!r} changed dtype."
            )
        expected_ranges = _sam_expected_ranges(expected)
        actual_ranges = _sam_feature_ranges(actual)
        if actual_ranges != expected_ranges:
            raise RuntimeError(
                f"LibreSAM function {function_name!r} input "
                f"{expected['name']!r} changed shape bounds: expected "
                f"{expected_ranges!r}, got {actual_ranges!r}."
            )

    for actual, expected in zip(outputs, contract["outputs"]):
        array = actual.type.multiArrayType
        if int(array.dataType) != dtype_codes[expected["dtype"]]:
            raise RuntimeError(
                f"LibreSAM function {function_name!r} output "
                f"{expected['name']!r} changed dtype."
            )
        # Core ML may omit static output shapes from a flexible ML Program's
        # protobuf. When present, however, they must remain exact. The MIL
        # validator below always checks every output axis before packaging.
        actual_ranges = _sam_feature_ranges(actual)
        if actual_ranges:
            expected_ranges = _sam_expected_ranges(expected)
            if actual_ranges != expected_ranges:
                raise RuntimeError(
                    f"LibreSAM function {function_name!r} output "
                    f"{expected['name']!r} changed shape: expected "
                    f"{expected_ranges!r}, got {actual_ranges!r}."
                )


def _validate_sam_mil_outputs(
    mlmodel: Any,
    *,
    function_name: str,
    profile: Any,
) -> None:
    """Check fixed SAM output axes in the converted in-memory MIL program."""
    from .coreml_sam import sam_coreml_function_contracts

    program = getattr(mlmodel, "_mil_program", None)
    functions = getattr(program, "functions", None)
    function = functions.get("main") if isinstance(functions, dict) else None
    if function is None:
        raise RuntimeError(
            f"Core ML conversion for LibreSAM {function_name!r} did not retain "
            "an inspectable MIL main function."
        )
    actual_outputs = list(getattr(function, "outputs", ()) or ())
    expected_outputs = sam_coreml_function_contracts(profile)[function_name][
        "outputs"
    ]
    if [str(value.name) for value in actual_outputs] != [
        item["name"] for item in expected_outputs
    ]:
        raise RuntimeError(
            f"Core ML conversion for LibreSAM {function_name!r} changed its "
            "output names."
        )

    from coremltools.converters.mil.mil import types

    for actual, expected in zip(actual_outputs, expected_outputs):
        if actual.dtype != types.fp32:
            raise RuntimeError(
                f"LibreSAM {function_name!r} output {expected['name']!r} "
                "must remain FP32."
            )
        shape = tuple(getattr(actual, "shape", ()) or ())
        expected_ranges = _sam_expected_ranges(expected)
        if len(shape) != len(expected_ranges):
            raise RuntimeError(
                f"LibreSAM {function_name!r} output {expected['name']!r} "
                f"has rank {len(shape)}; expected {len(expected_ranges)}."
            )
        for axis, (value, bounds) in enumerate(zip(shape, expected_ranges)):
            if bounds[0] != bounds[1]:
                continue
            try:
                resolved = int(value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"LibreSAM {function_name!r} lost fixed output axis "
                    f"{axis}={bounds[0]}."
                ) from exc
            if resolved != bounds[0]:
                raise RuntimeError(
                    f"LibreSAM {function_name!r} changed output axis {axis}: "
                    f"expected {bounds[0]}, got {resolved}."
                )


def _validate_sam_multifunction_spec(
    spec: Any,
    *,
    profile: Any,
) -> None:
    """Validate the complete native seven-function SAM package table."""
    from .coreml_sam import (
        SAM_COREML_ENCODER_FUNCTION,
        SAM_COREML_FUNCTION_NAMES,
    )

    description = getattr(spec, "description", None)
    if list(getattr(description, "input", ()) or ()) or list(
        getattr(description, "output", ()) or ()
    ):
        raise RuntimeError(
            "LibreSAM multifunction package must not expose a false top-level "
            "single-function interface."
        )
    if str(getattr(description, "defaultFunctionName", "")) != (
        SAM_COREML_ENCODER_FUNCTION
    ):
        raise RuntimeError(
            "LibreSAM multifunction package has the wrong default function."
        )
    functions = list(getattr(description, "functions", ()) or ())
    names = [str(value.name) for value in functions]
    if names != list(SAM_COREML_FUNCTION_NAMES):
        raise RuntimeError(
            "LibreSAM multifunction package function order/names changed: "
            f"{names!r}."
        )
    for function in functions:
        _validate_sam_function_description(
            function,
            function_name=str(function.name),
            profile=profile,
        )


def _sam_component_inputs(
    profile: Any,
    function_name: str,
    embeddings: tuple[torch.Tensor, ...],
    *,
    point_count: int,
    alternate: bool,
) -> tuple[torch.Tensor, ...]:
    """Build deterministic valid inputs for one SAM decoder ABI."""
    from .coreml_sam import (
        SAM_COREML_BOXES_INPUT,
        SAM_COREML_POINT_COORDS_INPUT,
        SAM_COREML_POINT_LABELS_INPUT,
        sam_coreml_function_contracts,
    )

    contract = sam_coreml_function_contracts(profile)[function_name]
    embedding_by_name = dict(zip(profile.embedding_names, embeddings))
    values: list[torch.Tensor] = []
    size = float(profile.image_size)
    for feature in contract["inputs"]:
        name = feature["name"]
        if name in embedding_by_name:
            values.append(embedding_by_name[name])
        elif name == SAM_COREML_POINT_COORDS_INPUT:
            count = int(point_count)
            start, stop = ((0.2, 0.85) if alternate else (0.1, 0.75))
            coords = torch.linspace(
                start * size,
                stop * size,
                count * 2,
                dtype=torch.float32,
            ).reshape(1, 1, count, 2)
            values.append(coords)
        elif name == SAM_COREML_POINT_LABELS_INPUT:
            labels = (torch.arange(point_count, dtype=torch.int32) % 2).reshape(
                1,
                1,
                point_count,
            )
            if alternate:
                labels = 1 - labels
            values.append(labels)
        elif name == SAM_COREML_BOXES_INPUT:
            fractions = (
                (0.2, 0.1, 0.9, 0.8)
                if alternate
                else (0.1, 0.2, 0.8, 0.9)
            )
            values.append(
                torch.tensor(fractions, dtype=torch.float32).reshape(1, 1, 4)
                * size
            )
        else:
            raise RuntimeError(
                f"Cannot construct a probe for SAM input {name!r}."
            )
    return tuple(values)


def _sam_named_tensor_outputs(
    value: Any,
    names: tuple[str, ...] | list[str],
) -> dict[str, torch.Tensor]:
    outputs = _flatten_tensor_outputs(value)
    if len(outputs) != len(names):
        raise RuntimeError(
            f"LibreSAM component emitted {len(outputs)} outputs for "
            f"{len(names)} declared names."
        )
    return {
        name: tensor.detach().contiguous()
        for name, tensor in zip(names, outputs)
    }


def _sam_validate_and_build_probes(
    components: dict[str, nn.Module],
    *,
    profile: Any,
    dummy: torch.Tensor,
) -> dict[
    str,
    tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        dict[str, torch.Tensor],
    ],
]:
    """Run every eager component twice and retain capture/check probes."""
    from .coreml_sam import (
        SAM_COREML_ENCODER_FUNCTION,
        SAM_COREML_ENCODER_INPUT,
        SAM_COREML_FUNCTION_NAMES,
        sam_coreml_function_contracts,
        validate_sam_coreml_function_io,
    )

    contracts = sam_coreml_function_contracts(profile)
    first_image = _canonical_trace_probe(dummy).float()
    second_image = (1.0 - first_image).contiguous()
    encoder = components[SAM_COREML_ENCODER_FUNCTION]
    with torch.inference_mode():
        first_raw = encoder(first_image)
        second_raw = encoder(second_image)
    first_outputs = _sam_named_tensor_outputs(
        first_raw,
        tuple(profile.embedding_names),
    )
    second_outputs = _sam_named_tensor_outputs(
        second_raw,
        tuple(profile.embedding_names),
    )
    validate_sam_coreml_function_io(
        SAM_COREML_ENCODER_FUNCTION,
        {SAM_COREML_ENCODER_INPUT: first_image},
        first_outputs,
        profile=profile,
    )
    validate_sam_coreml_function_io(
        SAM_COREML_ENCODER_FUNCTION,
        {SAM_COREML_ENCODER_INPUT: second_image},
        second_outputs,
        profile=profile,
    )
    probes = {
        SAM_COREML_ENCODER_FUNCTION: (
            (first_image,),
            (second_image,),
            second_outputs,
        )
    }
    first_embeddings = tuple(first_outputs[name] for name in profile.embedding_names)
    second_embeddings = tuple(
        second_outputs[name] for name in profile.embedding_names
    )
    for function_name in SAM_COREML_FUNCTION_NAMES[1:]:
        input_names = tuple(
            item["name"] for item in contracts[function_name]["inputs"]
        )
        output_names = tuple(
            item["name"] for item in contracts[function_name]["outputs"]
        )
        uses_points = "point_coords" in input_names
        first_inputs = _sam_component_inputs(
            profile,
            function_name,
            first_embeddings,
            point_count=2 if uses_points else 0,
            alternate=False,
        )
        second_inputs = _sam_component_inputs(
            profile,
            function_name,
            second_embeddings,
            point_count=profile.prompt_max_points if uses_points else 0,
            alternate=True,
        )
        component = components[function_name]
        with torch.inference_mode():
            first_decoded = _sam_named_tensor_outputs(
                component(*first_inputs),
                output_names,
            )
            second_decoded = _sam_named_tensor_outputs(
                component(*second_inputs),
                output_names,
            )
        validate_sam_coreml_function_io(
            function_name,
            dict(zip(input_names, first_inputs)),
            first_decoded,
            profile=profile,
        )
        validate_sam_coreml_function_io(
            function_name,
            dict(zip(input_names, second_inputs)),
            second_decoded,
            profile=profile,
        )
        probes[function_name] = (
            first_inputs,
            second_inputs,
            second_decoded,
        )
    return probes


def _sam_capture_component(
    component: nn.Module,
    *,
    function_name: str,
    profile: Any,
    probes: tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        dict[str, torch.Tensor],
    ],
) -> Any:
    """Capture one SAM graph and prove the alternate eager result is exact."""
    from .coreml_sam import (
        SAM_COREML_ENCODER_FUNCTION,
        sam_coreml_decoder_dynamic_shapes,
        sam_coreml_function_contracts,
    )

    first_inputs, second_inputs, expected_second = probes
    output_names = tuple(
        item["name"]
        for item in sam_coreml_function_contracts(profile)[function_name][
            "outputs"
        ]
    )
    if function_name == SAM_COREML_ENCODER_FUNCTION:
        captured = torch.jit.trace(
            component,
            first_inputs,
            check_trace=True,
            check_inputs=[second_inputs],
        )
        with torch.inference_mode():
            actual = _sam_named_tensor_outputs(
                captured(*second_inputs),
                output_names,
            )
    else:
        dynamic_shapes = (
            sam_coreml_decoder_dynamic_shapes(profile, function_name)
            if "points" in function_name
            else None
        )
        captured = torch.export.export(
            component,
            first_inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        ).run_decompositions({})
        with torch.inference_mode():
            actual = _sam_named_tensor_outputs(
                captured.module()(*second_inputs),
                output_names,
            )
    for name in output_names:
        torch.testing.assert_close(
            actual[name],
            expected_second[name],
            rtol=0.0,
            atol=0.0,
        )
    return captured


def _sam_coreml_tensor_type(ct: Any, feature: dict[str, Any]) -> Any:
    dtype = np.float32 if feature["dtype"] == "float32" else np.int32
    shape: list[Any] = []
    for axis in feature["shape"]:
        if axis["kind"] == "fixed":
            shape.append(int(axis["value"]))
        else:
            shape.append(
                ct.RangeDim(
                    lower_bound=int(axis["lower_bound"]),
                    upper_bound=int(axis["upper_bound"]),
                    default=int(axis["default"]),
                )
            )
    return ct.TensorType(
        name=feature["name"],
        shape=tuple(shape),
        dtype=dtype,
    )


def _export_sam_coreml_impl(
    nn_model: nn.Module,
    dummy: torch.Tensor,
    *,
    output_path: str,
    precision: str,
    compute_units: str,
    nms: bool,
    metadata: dict | None,
    model_family: str | None,
    model_task: str | None,
    model_size: str | None,
    prompt_max_points: int,
) -> str:
    """Export a visual-prompt SAM family as one iOS18 multifunction package."""
    from .coreml_sam import (
        SAM_COREML_ENCODER_FUNCTION,
        SAM_COREML_FUNCTION_NAMES,
        sam_coreml_function_contracts,
        sam_coreml_metadata,
        validate_sam_coreml_metadata,
        validate_sam_coreml_profile,
        wrap_sam_coreml_components,
    )

    values = dict(metadata or {})
    family = str(model_family or values.get("model_family") or "").lower()
    task = str(model_task or values.get("task") or "segment").lower()
    size_value = model_size or values.get("size") or values.get("model_size")
    size = str(size_value).lower() if size_value not in (None, "") else None
    output_path = _normalize_mlpackage_path(output_path)
    if task != "segment":
        raise NotImplementedError(
            f"LibreSAM Core ML export requires task='segment', got {task!r}."
        )
    if precision != "fp32":
        raise NotImplementedError("LibreSAM Core ML conversion is FP32-only.")
    if nms:
        raise NotImplementedError(
            "LibreSAM returns prompt mask logits and predicted IoU; nms=True "
            "is not applicable."
        )
    if str(compute_units).lower() not in {
        "all",
        "cpu_and_gpu",
        "cpu_and_ne",
        "cpu_only",
    }:
        raise ValueError(
            f"Invalid compute_units {compute_units!r}. Must be one of: "
            "['all', 'cpu_and_gpu', 'cpu_and_ne', 'cpu_only']"
        )
    if not torch.is_tensor(dummy) or dummy.ndim != 4:
        raise ValueError("LibreSAM Core ML export expects a BCHW dummy tensor.")
    profile = validate_sam_coreml_profile(
        family=family,
        size=size,
        prompt_max_points=prompt_max_points,
        precision=precision,
    )
    expected_shape = (1, 3, profile.image_size, profile.image_size)
    if tuple(dummy.shape) != expected_shape:
        raise ValueError(
            "LibreSAM Core ML export requires the family's fixed native "
            f"encoder frame {expected_shape}, got {tuple(dummy.shape)}."
        )
    graph_devices = {
        tensor.device.type
        for tensor in (*tuple(nn_model.parameters()), *tuple(nn_model.buffers()))
    }
    graph_devices.add(dummy.device.type)
    if graph_devices != {"cpu"}:
        raise NotImplementedError(
            "LibreSAM Core ML conversion requires a CPU graph; found "
            f"devices={sorted(graph_devices)}."
        )

    common_input = dict(values)
    common_input["dynamic"] = False
    prepared = _prepare_strict_metadata(
        common_input,
        family=family,
        task="segment",
        size=size,
        height=profile.image_size,
        width=profile.image_size,
    )
    contract_metadata = sam_coreml_metadata(profile)
    for key, expected in contract_metadata.items():
        current = values.get(key)
        if current not in (None, "") and current != expected:
            raise ValueError(
                f"LibreSAM Core ML metadata {key!r} conflicts with its "
                f"derived contract: {current!r} != {expected!r}."
            )
    prepared.update(contract_metadata)
    prepared.update(
        {
            "libreyolo_producer": "libreyolo",
            "artifact_format": "coreml",
            "precision": "fp32",
            "dynamic": True,
        }
    )
    validate_sam_coreml_metadata(prepared)

    components = wrap_sam_coreml_components(nn_model.eval(), profile=profile)
    probes = _sam_validate_and_build_probes(
        components,
        profile=profile,
        dummy=dummy,
    )

    import coremltools as ct

    compute_unit = _to_compute_unit(compute_units)
    contracts = sam_coreml_function_contracts(profile)
    conversion_kwargs = {
        "convert_to": "mlprogram",
        "compute_precision": ct.precision.FLOAT32,
        "minimum_deployment_target": ct.target.iOS18,
        "compute_units": compute_unit,
    }
    string_metadata = _stringify_metadata(prepared)

    class _SAMMultifunctionSaver:
        def save(self, candidate_path: str) -> None:
            with tempfile.TemporaryDirectory(
                prefix="libreyolo-sam-coreml-"
            ) as root:
                workspace = Path(root)
                descriptor = ct.utils.MultiFunctionDescriptor()
                for index, function_name in enumerate(SAM_COREML_FUNCTION_NAMES):
                    captured = _sam_capture_component(
                        components[function_name],
                        function_name=function_name,
                        profile=profile,
                        probes=probes[function_name],
                    )
                    contract = contracts[function_name]
                    converted = ct.convert(
                        captured,
                        inputs=[
                            _sam_coreml_tensor_type(ct, item)
                            for item in contract["inputs"]
                        ],
                        outputs=[
                            ct.TensorType(
                                name=item["name"],
                                dtype=np.float32,
                            )
                            for item in contract["outputs"]
                        ],
                        **conversion_kwargs,
                    )
                    _validate_sam_mil_outputs(
                        converted,
                        function_name=function_name,
                        profile=profile,
                    )
                    _validate_sam_function_description(
                        converted.get_spec().description,
                        function_name=function_name,
                        profile=profile,
                    )
                    component_path = workspace / (
                        f"{index:02d}-{function_name}.mlpackage"
                    )
                    converted.save(str(component_path))
                    descriptor.add_function(
                        str(component_path),
                        "main",
                        function_name,
                    )
                    del converted, captured

                descriptor.default_function_name = SAM_COREML_ENCODER_FUNCTION
                combined_path = workspace / "combined.mlpackage"
                ct.utils.save_multifunction(descriptor, str(combined_path))
                combined = ct.models.MLModel(
                    str(combined_path),
                    skip_model_load=True,
                )
                combined.user_defined_metadata.update(string_metadata)
                combined.save(candidate_path)

            reloaded = ct.models.MLModel(candidate_path, skip_model_load=True)
            _validate_sam_multifunction_spec(
                reloaded.get_spec(),
                profile=profile,
            )
            validate_sam_coreml_metadata(
                dict(reloaded.user_defined_metadata)
            )

    _save_mlpackage_atomic(_SAMMultifunctionSaver(), output_path)
    return str(output_path)


def _export_ppocr_coreml_impl(
    nn_model: nn.Module,
    dummy: torch.Tensor,
    *,
    output_path: str,
    precision: str,
    compute_units: str,
    nms: bool,
    metadata: dict | None,
    model_task: str | None,
    model_size: str | None,
    rec_batch_max: int,
    rec_max_width: int | None,
) -> str:
    """Export LibrePPOCR as an iOS18 two-function ML Program package."""
    from .coreml_ppocr import (
        PPOCR_COREML_DETECTOR_FUNCTION,
        PPOCR_COREML_DETECTOR_INPUT,
        PPOCR_COREML_DETECTOR_OUTPUT,
        PPOCR_COREML_RECOGNIZER_FUNCTION,
        PPOCR_COREML_RECOGNIZER_INPUT,
        PPOCR_COREML_RECOGNIZER_OUTPUT,
        ppocr_coreml_metadata,
        validate_ppocr_coreml_metadata,
        validate_ppocr_coreml_profile,
        validate_ppocr_detector_coreml_io,
        validate_ppocr_recognizer_coreml_io,
        wrap_ppocr_coreml_components,
    )

    values = dict(metadata or {})
    task = str(model_task or values.get("task") or "ocr").lower()
    size_value = model_size or values.get("size") or values.get("model_size")
    size = str(size_value).lower() if size_value not in (None, "") else None
    output_path = _normalize_mlpackage_path(output_path)

    if task != "ocr":
        raise NotImplementedError(
            f"LibrePPOCR Core ML export requires task='ocr', got {task!r}."
        )
    if precision != "fp32":
        raise NotImplementedError(
            "LibrePPOCR Core ML conversion is FP32-only."
        )
    if nms:
        raise NotImplementedError(
            "LibrePPOCR keeps DB contour extraction and CTC decoding on the "
            "host; nms=True is not applicable."
        )
    if rec_max_width is None:
        raise ValueError(
            "LibrePPOCR Core ML export requires an explicit finite "
            "rec_max_width=... (at least 320)."
        )
    if not torch.is_tensor(dummy) or tuple(dummy.shape[:2]) != (1, 3):
        raise ValueError(
            "LibrePPOCR Core ML detector export expects a [1, 3, H, W] "
            f"dummy tensor, got {getattr(dummy, 'shape', None)!r}."
        )
    if dummy.ndim != 4 or int(dummy.shape[2]) != int(dummy.shape[3]):
        raise ValueError(
            "LibrePPOCR Core ML detector limit must be represented by a "
            "square dummy canvas."
        )
    graph_devices = {
        tensor.device.type
        for tensor in (*tuple(nn_model.parameters()), *tuple(nn_model.buffers()))
    }
    graph_devices.add(dummy.device.type)
    if graph_devices != {"cpu"}:
        raise NotImplementedError(
            "LibrePPOCR Core ML conversion requires a CPU graph; found "
            f"devices={sorted(graph_devices)}."
        )
    if str(compute_units).lower() not in {
        "all",
        "cpu_and_gpu",
        "cpu_and_ne",
        "cpu_only",
    }:
        raise ValueError(
            f"Invalid compute_units {compute_units!r}. Must be one of: "
            "['all', 'cpu_and_gpu', 'cpu_and_ne', 'cpu_only']"
        )

    det_limit = int(dummy.shape[-1])
    profile = validate_ppocr_coreml_profile(
        size=size,
        precision=precision,
        det_limit_side_len=det_limit,
        rec_batch_max=rec_batch_max,
        rec_max_width=rec_max_width,
    )
    if profile.det_tensor_upper != det_limit:
        raise ValueError(
            "LibrePPOCR Core ML detector dummy canvas must equal the resolved "
            f"stride-rounded upper bound {profile.det_tensor_upper}, got "
            f"{det_limit}."
        )
    try:
        rec_num_classes = int(values["rec_num_classes"])
        charset = values["charset"]
        pipeline = dict(values["pipeline"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "LibrePPOCR Core ML export requires charset, pipeline, and "
            "rec_num_classes metadata from the loaded composite checkpoint."
        ) from exc
    pipeline["det_limit_side_len"] = det_limit

    common_input = dict(values)
    common_input["dynamic"] = False
    prepared = _prepare_strict_metadata(
        common_input,
        family="ppocr",
        task="ocr",
        size=size,
        height=det_limit,
        width=det_limit,
    )
    contract_metadata = ppocr_coreml_metadata(
        profile=profile,
        charset=charset,
        pipeline=pipeline,
        rec_num_classes=rec_num_classes,
    )
    for key, expected in contract_metadata.items():
        current = values.get(key)
        if current not in (None, "") and current != expected:
            raise ValueError(
                f"LibrePPOCR Core ML metadata {key!r} conflicts with its "
                f"derived contract: {current!r} != {expected!r}."
            )
    prepared.update(contract_metadata)
    prepared.update(
        {
            "libreyolo_producer": "libreyolo",
            "artifact_format": "coreml",
            "precision": "fp32",
            "dynamic": True,
        }
    )
    validate_ppocr_coreml_metadata(prepared)

    wrapped = wrap_ppocr_coreml_components(
        nn_model.eval(),
        profile=profile,
        rec_num_classes=rec_num_classes,
    )
    det_first, det_second, rec_first, rec_second = _ppocr_trace_probes(profile)
    with torch.inference_mode():
        expected_det_first = wrapped[PPOCR_COREML_DETECTOR_FUNCTION](det_first)
        expected_det_second = wrapped[PPOCR_COREML_DETECTOR_FUNCTION](det_second)
        expected_rec_first = wrapped[PPOCR_COREML_RECOGNIZER_FUNCTION](rec_first)
        expected_rec_second = wrapped[PPOCR_COREML_RECOGNIZER_FUNCTION](rec_second)
    validate_ppocr_detector_coreml_io(
        det_first,
        expected_det_first,
        profile=profile,
    )
    validate_ppocr_detector_coreml_io(
        det_second,
        expected_det_second,
        profile=profile,
    )
    validate_ppocr_recognizer_coreml_io(
        rec_first,
        expected_rec_first,
        profile=profile,
        rec_num_classes=rec_num_classes,
    )
    validate_ppocr_recognizer_coreml_io(
        rec_second,
        expected_rec_second,
        profile=profile,
        rec_num_classes=rec_num_classes,
    )

    det_trace = torch.jit.trace(
        wrapped[PPOCR_COREML_DETECTOR_FUNCTION],
        det_first,
        check_trace=True,
        check_inputs=[(det_second,)],
    )
    rec_trace = torch.jit.trace(
        wrapped[PPOCR_COREML_RECOGNIZER_FUNCTION],
        rec_first,
        check_trace=True,
        check_inputs=[(rec_second,)],
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            det_trace(det_second),
            expected_det_second,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            rec_trace(rec_second),
            expected_rec_second,
            rtol=0.0,
            atol=0.0,
        )

    import coremltools as ct

    compute_unit = _to_compute_unit(compute_units)
    conversion_kwargs = {
        "convert_to": "mlprogram",
        "compute_precision": ct.precision.FLOAT32,
        "minimum_deployment_target": ct.target.iOS18,
        "compute_units": compute_unit,
    }
    detector_model = ct.convert(
        det_trace,
        inputs=[
            ct.TensorType(
                name=PPOCR_COREML_DETECTOR_INPUT,
                shape=(
                    1,
                    3,
                    ct.RangeDim(
                        lower_bound=32,
                        upper_bound=profile.det_tensor_upper,
                        default=profile.det_tensor_upper,
                    ),
                    ct.RangeDim(
                        lower_bound=32,
                        upper_bound=profile.det_tensor_upper,
                        default=profile.det_tensor_upper,
                    ),
                ),
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(
                name=PPOCR_COREML_DETECTOR_OUTPUT,
                dtype=np.float32,
            )
        ],
        **conversion_kwargs,
    )
    recognizer_model = ct.convert(
        rec_trace,
        inputs=[
            ct.TensorType(
                name=PPOCR_COREML_RECOGNIZER_INPUT,
                shape=(
                    ct.RangeDim(
                        lower_bound=1,
                        upper_bound=profile.rec_batch_max,
                        default=1,
                    ),
                    3,
                    48,
                    ct.RangeDim(
                        lower_bound=320,
                        upper_bound=profile.rec_max_width,
                        default=320,
                    ),
                ),
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(
                name=PPOCR_COREML_RECOGNIZER_OUTPUT,
                dtype=np.float32,
            )
        ],
        **conversion_kwargs,
    )

    _validate_ppocr_mil_output(
        detector_model,
        function_name=PPOCR_COREML_DETECTOR_FUNCTION,
        output_name=PPOCR_COREML_DETECTOR_OUTPUT,
        expected_rank=4,
        fixed_axes={0: 1, 1: 1},
    )
    _validate_ppocr_mil_output(
        recognizer_model,
        function_name=PPOCR_COREML_RECOGNIZER_FUNCTION,
        output_name=PPOCR_COREML_RECOGNIZER_OUTPUT,
        expected_rank=3,
        fixed_axes={2: rec_num_classes},
    )
    _validate_ppocr_unifunction_spec(
        detector_model,
        function_name=PPOCR_COREML_DETECTOR_FUNCTION,
        input_name=PPOCR_COREML_DETECTOR_INPUT,
        output_name=PPOCR_COREML_DETECTOR_OUTPUT,
        expected_ranges=(
            (1, 1),
            (3, 3),
            (32, profile.det_tensor_upper),
            (32, profile.det_tensor_upper),
        ),
    )
    _validate_ppocr_unifunction_spec(
        recognizer_model,
        function_name=PPOCR_COREML_RECOGNIZER_FUNCTION,
        input_name=PPOCR_COREML_RECOGNIZER_INPUT,
        output_name=PPOCR_COREML_RECOGNIZER_OUTPUT,
        expected_ranges=(
            (1, profile.rec_batch_max),
            (3, 3),
            (48, 48),
            (320, profile.rec_max_width),
        ),
    )

    string_metadata = _stringify_metadata(prepared)

    class _PPOCRMultifunctionSaver:
        def save(self, candidate_path: str) -> None:
            with tempfile.TemporaryDirectory(prefix="libreyolo-ppocr-coreml-") as root:
                workspace = Path(root)
                detector_path = workspace / "detector.mlpackage"
                recognizer_path = workspace / "recognizer.mlpackage"
                combined_path = workspace / "combined.mlpackage"
                detector_model.save(str(detector_path))
                recognizer_model.save(str(recognizer_path))
                descriptor = ct.utils.MultiFunctionDescriptor()
                descriptor.add_function(
                    str(detector_path),
                    "main",
                    PPOCR_COREML_DETECTOR_FUNCTION,
                )
                descriptor.add_function(
                    str(recognizer_path),
                    "main",
                    PPOCR_COREML_RECOGNIZER_FUNCTION,
                )
                descriptor.default_function_name = (
                    PPOCR_COREML_DETECTOR_FUNCTION
                )
                ct.utils.save_multifunction(descriptor, str(combined_path))
                combined = ct.models.MLModel(
                    str(combined_path),
                    skip_model_load=True,
                )
                combined.user_defined_metadata.update(string_metadata)
                combined.save(candidate_path)

            reloaded = ct.models.MLModel(candidate_path, skip_model_load=True)
            _validate_ppocr_multifunction_spec(
                reloaded.get_spec(),
                profile=profile,
            )
            validate_ppocr_coreml_metadata(
                dict(reloaded.user_defined_metadata)
            )

    _save_mlpackage_atomic(_PPOCRMultifunctionSaver(), output_path)
    return str(output_path)


def _export_coreml_impl(
    nn_model: nn.Module,
    dummy: torch.Tensor,
    *,
    output_path: str,
    precision: str,
    compute_units: str,
    nms: bool,
    iou: float,
    conf: float,
    metadata: dict | None,
    model_family: str | None,
    model_task: str | None,
    model_size: str | None,
    dynamic: bool,
) -> str:
    metadata = dict(metadata or {})
    family = str(model_family or metadata.get("model_family") or "").lower()
    task = str(model_task or metadata.get("task") or "detect").lower()
    size_value = model_size or metadata.get("size") or metadata.get("model_size")
    size = str(size_value).lower() if size_value not in (None, "") else None
    output_path = _normalize_mlpackage_path(output_path)

    # Reject the entire request before imports, graph preparation, tracing, or
    # touching the destination. BaseExporter performs these checks before its
    # LoRA mutation point; this duplicate protects direct function callers.
    if dynamic:
        raise NotImplementedError(
            "Core ML export uses a fixed input shape; "
            "dynamic=True is not supported."
        )
    if not torch.is_tensor(dummy) or dummy.ndim != 4:
        raise ValueError(
            "Core ML export expects a BCHW tensor as dummy input; "
            f"got {type(dummy).__name__} with shape "
            f"{getattr(dummy, 'shape', None)!r}."
        )
    if int(dummy.shape[0]) != 1:
        raise ValueError(
            "Core ML export currently requires batch=1; "
            f"got batch={int(dummy.shape[0])}."
        )
    if int(dummy.shape[1]) != 3:
        raise ValueError(
            "Core ML export requires a three-channel RGB input; "
            f"got channels={int(dummy.shape[1])}."
        )
    graph_devices = {
        tensor.device.type
        for tensor in (*tuple(nn_model.parameters()), *tuple(nn_model.buffers()))
    }
    graph_devices.add(dummy.device.type)
    if graph_devices != {"cpu"}:
        raise NotImplementedError(
            "Core ML conversion requires a CPU PyTorch graph and dummy input; "
            f"found devices={sorted(graph_devices)}."
        )
    if precision not in {"fp32", "fp16"}:
        raise ValueError(
            f"Invalid Core ML precision {precision!r}; expected 'fp32' or 'fp16'."
        )
    if str(compute_units).lower() not in {
        "all",
        "cpu_and_gpu",
        "cpu_and_ne",
        "cpu_only",
    }:
        raise ValueError(
            f"Invalid compute_units {compute_units!r}. Must be one of: "
            "['all', 'cpu_and_gpu', 'cpu_and_ne', 'cpu_only']"
        )
    _validate_export_profile(family, task, size)
    if family == "birefnet":
        from .coreml_birefnet import validate_birefnet_coreml_profile

        validate_birefnet_coreml_profile(
            size=size,
            precision=precision,
            canvas_hw=(int(dummy.shape[-2]), int(dummy.shape[-1])),
        )
    if family == "swinir":
        from .coreml_swinir import validate_swinir_coreml_profile

        validate_swinir_coreml_profile(
            size=size,
            canvas_hw=(int(dummy.shape[-2]), int(dummy.shape[-1])),
        )
    if family == "picosam3":
        from .coreml_picosam3 import validate_picosam3_coreml_profile

        validate_picosam3_coreml_profile(
            size=size,
            canvas_hw=(int(dummy.shape[-2]), int(dummy.shape[-1])),
        )
    if family == "depth_anything3":
        from .coreml_depth_anything3 import (
            validate_depth_anything3_coreml_profile,
        )

        validate_depth_anything3_coreml_profile(
            nn_model,
            size=size,
            canvas_hw=(int(dummy.shape[-2]), int(dummy.shape[-1])),
        )
    if family == "owlv2":
        from .coreml_owlv2 import validate_owlv2_coreml_profile

        validate_owlv2_coreml_profile(
            size=size,
            canvas_hw=(int(dummy.shape[-2]), int(dummy.shape[-1])),
        )
    if family == "grounding_dino":
        from .coreml_grounding_dino import (
            validate_grounding_dino_coreml_profile,
        )

        validate_grounding_dino_coreml_profile(
            size=size,
            canvas_hw=(int(dummy.shape[-2]), int(dummy.shape[-1])),
        )
    if family == "omdet_turbo":
        from .coreml_omdet_turbo import validate_omdet_turbo_coreml_profile

        validate_omdet_turbo_coreml_profile(
            size=size,
            canvas_hw=(int(dummy.shape[-2]), int(dummy.shape[-1])),
        )
    if family == "rtmdet" and task == "segment":
        from .coreml_rtmdet_ins import validate_rtmdet_ins_coreml_profile

        validate_rtmdet_ins_coreml_profile(
            size=size,
            canvas_hw=(int(dummy.shape[-2]), int(dummy.shape[-1])),
        )
    eomt_net = None
    if family == "eomt":
        from .coreml_eomt import validate_eomt_coreml_profile

        eomt_net = validate_eomt_coreml_profile(
            nn_model,
            task=task,
            size=size,
            canvas_hw=(int(dummy.shape[-2]), int(dummy.shape[-1])),
        )
    if nms and family in _NMS_FREE_FAMILIES:
        raise NotImplementedError(
            f"nms=True is not supported for {_NMS_FREE_FAMILIES[family]} "
            "(set prediction does not use IoU NMS). Export raw outputs instead."
        )
    if nms and (family not in {"yolox", "yolo9"} or task != "detect"):
        raise NotImplementedError(
            "Core ML embedded NMS is limited to YOLOX and YOLO9 detection. "
            "Use the raw-output profile for every other family/task."
        )
    if nms:
        for name, value in (("conf", conf), ("iou", iou)):
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Core ML NMS {name} must be a finite number in [0, 1]."
                ) from exc
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError(
                    f"Core ML NMS {name} must be in [0, 1], got {value!r}."
                )
    metadata = _prepare_strict_metadata(
        metadata,
        family=family,
        task=task,
        size=size,
        height=int(dummy.shape[2]),
        width=int(dummy.shape[3]),
    )
    if family == "picosam3":
        from .coreml_picosam3 import picosam3_coreml_component_metadata

        for key, expected_value in picosam3_coreml_component_metadata().items():
            current = metadata.get(key)
            if current not in (None, "", expected_value):
                raise ValueError(
                    f"PicoSAM3 Core ML metadata {key!r}={current!r} conflicts "
                    f"with the ROI component contract value {expected_value!r}."
                )
            metadata[key] = expected_value
    if family == "depth_anything3":
        from .coreml_depth_anything3 import (
            depth_anything3_coreml_metadata,
            validate_depth_anything3_coreml_metadata,
        )

        for key, expected_value in depth_anything3_coreml_metadata().items():
            current = metadata.get(key)
            if current not in (None, "", expected_value):
                raise ValueError(
                    f"Depth Anything 3 Core ML metadata {key!r}={current!r} "
                    f"conflicts with the raw component value "
                    f"{expected_value!r}."
                )
            metadata[key] = expected_value
        validate_depth_anything3_coreml_metadata(metadata)
    if family == "owlv2":
        from .coreml_owlv2 import validate_owlv2_coreml_metadata

        validate_owlv2_coreml_metadata(
            metadata,
            size=str(size),
            names=metadata["names"],
        )
    if family == "grounding_dino":
        from .coreml_grounding_dino import (
            validate_grounding_dino_coreml_metadata,
        )

        validate_grounding_dino_coreml_metadata(
            metadata,
            size=str(size),
            names=metadata["names"],
        )
    if family == "omdet_turbo":
        from .coreml_omdet_turbo import (
            validate_omdet_turbo_coreml_metadata,
        )

        validate_omdet_turbo_coreml_metadata(
            metadata,
            size=str(size),
            names=metadata["names"],
        )
    if family == "rtmdet" and task == "segment":
        from .coreml_rtmdet_ins import rtmdet_ins_coreml_metadata

        for key, expected_value in rtmdet_ins_coreml_metadata().items():
            current = metadata.get(key)
            if current not in (None, "", expected_value):
                raise ValueError(
                    f"RTMDet-Ins Core ML metadata {key!r}={current!r} "
                    f"conflicts with the raw-output contract value "
                    f"{expected_value!r}."
                )
            metadata[key] = expected_value
    if family == "eomt":
        from .coreml_eomt import eomt_coreml_metadata

        assert eomt_net is not None
        expected_eomt_metadata = eomt_coreml_metadata(
            task=task,
            num_queries=int(eomt_net.num_queries),
            image_size=int(eomt_net.image_size),
        )
        for key, expected_value in expected_eomt_metadata.items():
            current = metadata.get(key)
            if current not in (None, "", expected_value):
                raise ValueError(
                    f"EoMT Core ML metadata {key!r}={current!r} conflicts "
                    f"with the compact query contract value {expected_value!r}."
                )
            metadata[key] = expected_value
        if task == "panoptic":
            raw_thing_ids = metadata.get("thing_class_ids")
            if not isinstance(raw_thing_ids, (list, tuple, set)):
                raise ValueError(
                    "EoMT panoptic Core ML export requires thing_class_ids "
                    "from the converted checkpoint metadata."
                )
            try:
                thing_ids = [int(value) for value in raw_thing_ids]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "EoMT panoptic thing_class_ids must contain integers."
                ) from exc
            if (
                not thing_ids
                or len(set(thing_ids)) != len(thing_ids)
                or any(value < 0 or value >= int(metadata["nc"]) for value in thing_ids)
            ):
                raise ValueError(
                    "EoMT panoptic thing_class_ids must be non-empty, unique, "
                    "and within the exported class range."
                )
            metadata["thing_class_ids"] = sorted(thing_ids)
            metadata.update(
                {
                    "eomt_panoptic_score_threshold": 0.8,
                    "eomt_panoptic_mask_threshold": 0.5,
                    "eomt_panoptic_overlap_threshold": 0.8,
                }
            )
    metadata["precision"] = precision

    import coremltools as ct

    if family == "birefnet":
        from .coreml_birefnet import require_birefnet_coreml_lowering

        require_birefnet_coreml_lowering(ct)

    compute_unit = _to_compute_unit(compute_units)
    canonical_dummy = _canonical_trace_probe(dummy)
    check_probe = 1.0 - canonical_dummy
    declared_outputs = _output_contract(family, task, nms=nms)

    wrapped = _wrap_coreml_contract(nn_model.eval(), family, task)
    if nms:
        wrapped = _NMSOutputAdapter(wrapped, family)
    wrapped = (
        _CoreMLOutputAdapter(wrapped, [item["name"] for item in declared_outputs])
        .to(device=dummy.device)
        .eval()
    )

    # Core AI's preparation transaction contains only PyTorch graph surgery;
    # it has no Core AI package/runtime dependency and is shared here so the
    # two Apple exporters cannot diverge on anchor/state restoration.
    from .coreai import _prepare_coreai_graph

    with _prepare_coreai_graph(wrapped, canonical_dummy, family):
        with torch.no_grad():
            sample_outputs = _flatten_tensor_outputs(wrapped(canonical_dummy))
        _validate_output_semantics(
            declared_outputs,
            sample_outputs,
            family=family,
            task=task,
            nc=int(metadata["nc"]),
            input_hw=(int(dummy.shape[-2]), int(dummy.shape[-1])),
            size=size,
            nms=nms,
            metadata=metadata,
        )
        output_contract = _enrich_output_contract(declared_outputs, sample_outputs)
        traced = torch.jit.trace(
            wrapped,
            canonical_dummy,
            check_trace=True,
            check_inputs=[(check_probe,)],
        )

    input_contract = _input_contract(family, task, size)
    if input_contract["kind"] == "image":
        coreml_input = ct.ImageType(
            name=input_contract["name"],
            shape=tuple(canonical_dummy.shape),
            scale=1.0 / 255.0,
            bias=[0.0, 0.0, 0.0],
            color_layout=ct.colorlayout.RGB,
        )
    else:
        coreml_input = ct.TensorType(
            name=input_contract["name"],
            shape=tuple(canonical_dummy.shape),
        )
    compute_precision = (
        ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    )
    convert_kwargs = {
        "inputs": [coreml_input],
        "outputs": [ct.TensorType(name=item["name"]) for item in output_contract],
        "convert_to": "mlprogram",
        "compute_precision": compute_precision,
        "minimum_deployment_target": ct.target.iOS15,
        "compute_units": compute_unit,
    }
    mlmodel = ct.convert(traced, **convert_kwargs)

    expected_names = [item["name"] for item in output_contract]
    converted_names = _spec_output_names(mlmodel)
    if converted_names and converted_names != expected_names:
        raise RuntimeError(
            "Core ML converter changed the declared semantic output order: "
            f"expected {expected_names}, got {converted_names}."
        )

    if nms:
        mlmodel = _wrap_with_nms(
            mlmodel,
            model_family=family,
            iou=iou,
            conf=conf,
            compute_units=compute_unit,
        )
        metadata.update({"nms": True, "nms_iou": iou, "nms_conf": conf})

    coreml_io = {
        "input": input_contract,
        "outputs": output_contract,
        "validation": _validation_contract(family, task),
    }
    metadata.update(
        {
            "libreyolo_producer": "libreyolo",
            "artifact_format": "coreml",
            "coreml_io_schema_version": "2",
            "coreml_io": coreml_io,
            "coreml_output_names": expected_names,
            "dynamic": False,
        }
    )
    mlmodel.user_defined_metadata.update(_stringify_metadata(metadata))
    _save_mlpackage_atomic(mlmodel, output_path)
    return str(output_path)


def export_coreml(
    nn_model,
    dummy,
    *,
    output_path: str,
    precision: str = "fp32",
    compute_units: str = "all",
    nms: bool = False,
    iou: float = 0.45,
    conf: float = 0.25,
    metadata: dict | None = None,
    model_family: str | None = None,
    model_task: str | None = None,
    model_size: str | None = None,
    dynamic: bool = False,
    rec_batch_max: int = 6,
    rec_max_width: int | None = None,
    prompt_max_points: int = 16,
) -> str:
    """Export a strict Core ML ML Program.

    Most profiles accept one fixed canonical RGB image. LibrePPOCR instead
    emits a bounded-flexible, two-function FP32 package because its detector
    and recognizer have distinct tensor interfaces and host orchestration.

    Args:
        nn_model: The PyTorch nn.Module to export. Must already be in eval/export mode.
        dummy: Reference input tensor — only its (B, C, H, W) shape is used.
        output_path: Destination .mlpackage path (a directory bundle).
        precision: 'fp32' or 'fp16'.
        compute_units: 'all' | 'cpu_and_gpu' | 'cpu_and_ne' | 'cpu_only'.
        nms: If True, embed Apple's NonMaximumSuppression as a CoreML pipeline.
            Not supported for DETR-style families (RT-DETR, RF-DETR, D-FINE,
            DEIM, EC).
        iou: IoU threshold for embedded NMS (default 0.45). Ignored when nms=False.
        conf: Confidence threshold for embedded NMS (default 0.25). Ignored
            when nms=False.
        metadata: Dict of metadata to embed under user_defined_metadata.
        model_family: Family string (yolox | yolo9 | rtdetr | rfdetr) — selects
            the preprocess wrapper.

    Returns:
        The normalized ``.mlpackage`` path on success.
    """
    family = str(
        model_family or (metadata or {}).get("model_family") or ""
    ).lower()
    if family in {"edgetam", "mobilesam", "sam", "sam2", "sam3"}:
        return _export_sam_coreml_impl(
            nn_model,
            dummy,
            output_path=output_path,
            precision=precision,
            compute_units=compute_units,
            nms=nms,
            metadata=metadata,
            model_family=model_family,
            model_task=model_task,
            model_size=model_size,
            prompt_max_points=prompt_max_points,
        )
    if family == "ppocr":
        return _export_ppocr_coreml_impl(
            nn_model,
            dummy,
            output_path=output_path,
            precision=precision,
            compute_units=compute_units,
            nms=nms,
            metadata=metadata,
            model_task=model_task,
            model_size=model_size,
            rec_batch_max=rec_batch_max,
            rec_max_width=rec_max_width,
        )
    return _export_coreml_impl(
        nn_model,
        dummy,
        output_path=output_path,
        precision=precision,
        compute_units=compute_units,
        nms=nms,
        iou=iou,
        conf=conf,
        metadata=metadata,
        model_family=model_family,
        model_task=model_task,
        model_size=model_size,
        dynamic=dynamic,
    )


def _wrap_with_nms(
    mlmodel: Any,
    *,
    model_family: str | None,
    iou: float = 0.45,
    conf: float = 0.25,
    compute_units: Any = None,
) -> Any:
    """Wrap a detector mlmodel in a Pipeline that embeds Apple's NMS layer.

    Output names are ``confidence`` and pixel-space ``coordinates`` (cxcywh).
    """
    import coremltools as ct

    model_spec = mlmodel.get_spec()
    output_by_name = {out.name: out for out in model_spec.description.output}
    if {"confidence", "coordinates"} - output_by_name.keys():
        raise RuntimeError(
            "CoreML NMS wrapping requires converted outputs named "
            "'confidence' and 'coordinates'."
        )

    confidence_shape = _multiarray_shape(output_by_name["confidence"])
    coordinates_shape = _multiarray_shape(output_by_name["coordinates"])
    if len(confidence_shape) != 2 or coordinates_shape != [confidence_shape[0], 4]:
        raise RuntimeError(
            "CoreML NMS wrapping requires confidence shape (N, C) and "
            f"coordinates shape (N, 4); got {confidence_shape} and {coordinates_shape}."
        )

    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 5
    _add_multiarray_feature(nms_spec.description.input, "confidence", confidence_shape)
    _add_multiarray_feature(
        nms_spec.description.input, "coordinates", coordinates_shape
    )
    _add_multiarray_feature(nms_spec.description.output, "confidence", confidence_shape)
    _add_multiarray_feature(
        nms_spec.description.output, "coordinates", coordinates_shape
    )

    nms = nms_spec.nonMaximumSuppression
    nms.iouThreshold = iou
    nms.confidenceThreshold = conf
    nms.confidenceInputFeatureName = "confidence"
    nms.coordinatesInputFeatureName = "coordinates"
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    # Native YOLO postprocess is class-aware. This still remains an optional
    # deployment profile because Apple's NMS cannot reproduce multi-label
    # candidate expansion or return indices for masks/keypoints.
    nms.pickTop.perClass = True

    pipeline_spec = ct.proto.Model_pb2.Model()
    pipeline_spec.specificationVersion = max(
        model_spec.specificationVersion,
        nms_spec.specificationVersion,
    )
    pipeline_spec.pipeline
    pipeline_spec.description.input.extend(model_spec.description.input)
    pipeline_spec.description.output.extend(nms_spec.description.output)
    pipeline_spec.pipeline.models.add().CopyFrom(model_spec)
    pipeline_spec.pipeline.models.add().CopyFrom(nms_spec)

    kwargs = {"weights_dir": mlmodel.weights_dir}
    if compute_units is not None:
        kwargs["compute_units"] = compute_units
    return ct.models.MLModel(pipeline_spec, **kwargs)


def _multiarray_shape(feature: Any) -> list[int]:
    return [int(dim) for dim in feature.type.multiArrayType.shape]


def _add_multiarray_feature(features: Any, name: str, shape: list[int]) -> None:
    import coremltools as ct

    feature = features.add()
    feature.name = name
    multiarray = feature.type.multiArrayType
    multiarray.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
    multiarray.shape.extend(shape)
