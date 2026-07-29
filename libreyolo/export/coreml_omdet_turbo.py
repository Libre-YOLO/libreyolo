"""Frozen-vocabulary OMDet-Turbo contract for Core ML.

OMDet-Turbo is open-vocabulary only while its language backbone is present.
The Core ML deployment graph therefore freezes one explicit class vocabulary
and its task prompt at export time, then retains only the image backbone,
hybrid encoder, detector decoder, and the resulting language embeddings.

The preprocessing and decoder ABI in this module follow Hugging Face
Transformers v5.12.1, source commit
``ddb849abe009d1089e6c691bfc897f27211c663c``, Apache-2.0.  The annotated
tag object is ``a030302dcd4777bbf042ee46c30c5dbe6d2a2eb2``.  The relevant
upstream files are ``models/detr/image_processing_detr.py`` and
``models/omdet_turbo/{modeling,processing}_omdet_turbo.py``.  Attribution and
hash-pinned checkpoint provenance are recorded in
``libreyolo/models/omdet_turbo/NOTICE``.

This module is import-safe on systems without coremltools.  Conversion is
performed only by :func:`export_omdet_turbo_coreml`.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

OMDET_TURBO_COREML_CONTRACT = "omdet_turbo_frozen_vocabulary_v1"
OMDET_TURBO_COREML_PREPROCESS = "rgb_bilinear_stretch_imagenet_v1"
OMDET_TURBO_COREML_POSTPROCESS = "top900_sigmoid_class_nms_v1"
OMDET_TURBO_COREML_TASK_TEMPLATE = "Detect {}."
OMDET_TURBO_COREML_INPUT_NAME = "image"
OMDET_TURBO_COREML_OUTPUT_NAMES = ("pred_logits", "pred_boxes")
OMDET_TURBO_COREML_MEAN = (123.675, 116.28, 103.53)
OMDET_TURBO_COREML_STD = (58.395, 57.12, 57.375)
OMDET_TURBO_COREMLTOOLS_MAJOR = 9
OMDET_TURBO_TRANSFORMERS_VERSION = "5.12.1"


@dataclass(frozen=True)
class OmDetTurboCoreMLProfile:
    """One fixed OMDet-Turbo image/query profile."""

    size: str
    image_size: int
    num_queries: int


OMDET_TURBO_COREML_PROFILES = {
    "t": OmDetTurboCoreMLProfile("t", 640, 900),
}


def require_omdet_turbo_transformers_toolchain(
    transformers_module: Any | None = None,
) -> Any:
    """Require the exact Transformers semantics used by the frozen graph."""
    if transformers_module is None:
        try:
            import transformers as transformers_module
        except ImportError as exc:
            raise RuntimeError(
                "OMDet-Turbo Core ML export requires transformers=="
                f"{OMDET_TURBO_TRANSFORMERS_VERSION}."
            ) from exc
    actual = str(getattr(transformers_module, "__version__", ""))
    if actual != OMDET_TURBO_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "OMDet-Turbo Core ML export is validated only with transformers=="
            f"{OMDET_TURBO_TRANSFORMERS_VERSION}; found {actual or 'unknown'}."
        )
    return transformers_module


def require_omdet_turbo_coremltools_toolchain(
    coremltools_module: Any | None = None,
) -> Any:
    """Require the Core ML Tools major used for full-checkpoint conversion."""
    if coremltools_module is None:
        try:
            import coremltools as coremltools_module
        except ImportError as exc:
            raise RuntimeError(
                "OMDet-Turbo Core ML export requires coremltools 9.x."
            ) from exc
    actual = str(getattr(coremltools_module, "__version__", ""))
    try:
        major = int(actual.split(".", 1)[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Could not determine the installed coremltools version; "
            "OMDet-Turbo requires 9.x."
        ) from exc
    if major != OMDET_TURBO_COREMLTOOLS_MAJOR:
        raise RuntimeError(
            "OMDet-Turbo Core ML export is validated only with coremltools "
            f"9.x; found {actual}."
        )
    return coremltools_module


def _ordered_names(names: Mapping[int, str] | Sequence[str]) -> list[str]:
    """Return a strict finite class vocabulary in class-id order."""
    if isinstance(names, Mapping):
        try:
            keys = sorted(int(key) for key in names)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "OMDet-Turbo Core ML class ids must be integers."
            ) from exc
        if keys != list(range(len(keys))):
            raise ValueError(
                "OMDet-Turbo Core ML class ids must be contiguous from zero; "
                f"got {keys!r}."
            )
        values = [names[key] if key in names else names[str(key)] for key in keys]
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        values = list(names)
    else:
        raise TypeError(
            "OMDet-Turbo Core ML classes must be a mapping or finite sequence."
        )

    if not values:
        raise ValueError(
            "OMDet-Turbo Core ML export requires at least one frozen class. "
            "Call set_classes([...]) before export."
        )
    if not all(isinstance(value, str) for value in values):
        raise TypeError(
            "OMDet-Turbo Core ML class labels must all be strings."
        )
    labels = [value.strip() for value in values]
    if any(not label for label in labels):
        raise ValueError(
            "OMDet-Turbo Core ML class labels must not be blank."
        )
    normalized = [label.casefold() for label in labels]
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "OMDet-Turbo Core ML class labels must be unique "
            "case-insensitively."
        )
    return labels


def omdet_turbo_coreml_task(
    names: Mapping[int, str] | Sequence[str],
) -> str:
    """Render the exact task prompt frozen into the detector."""
    return OMDET_TURBO_COREML_TASK_TEMPLATE.format(
        ", ".join(_ordered_names(names))
    )


def omdet_turbo_coreml_vocabulary_hash(
    names: Mapping[int, str] | Sequence[str],
) -> str:
    """Hash ordered labels and the task-template contract."""
    labels = _ordered_names(names)
    payload = {
        "labels": labels,
        "task_template": OMDET_TURBO_COREML_TASK_TEMPLATE,
        "task": omdet_turbo_coreml_task(labels),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_omdet_turbo_coreml_profile(
    *,
    size: str | None,
    canvas_hw: tuple[int, int] | None = None,
) -> OmDetTurboCoreMLProfile:
    """Resolve the released fixed OMDet-Turbo profile."""
    key = str(size or "").strip().lower()
    profile = OMDET_TURBO_COREML_PROFILES.get(key)
    if profile is None:
        raise NotImplementedError(
            "OMDet-Turbo Core ML export supports only the released "
            f"Swin-T size='t'; got size={size!r}."
        )
    if canvas_hw is not None:
        height, width = (int(value) for value in canvas_hw)
        expected = (profile.image_size, profile.image_size)
        if (height, width) != expected:
            raise NotImplementedError(
                "OMDet-Turbo Core ML export requires its fixed "
                f"{expected[1]}x{expected[0]} canvas; got {width}x{height}."
            )
    return profile


def omdet_turbo_coreml_input_contract() -> dict[str, Any]:
    """Declare the exact tensor boundary after Torchvision v2 resizing."""
    return {
        "name": OMDET_TURBO_COREML_INPUT_NAME,
        "kind": "tensor",
        "layout": "NCHW",
        "color": "rgb",
        "range": "0_255",
        "geometry": "stretch",
        "interpolation": "bilinear",
        "resize_backend": "torchvision",
        "pad_value": 0,
    }


def omdet_turbo_coreml_validation_contract() -> dict[str, str]:
    """Declare canonical validator images before graph normalization."""
    return {"color": "rgb", "range": "0_255"}


def omdet_turbo_coreml_output_contract() -> list[dict[str, Any]]:
    """Return the fixed raw detector ABI in graph order."""
    return [
        {
            "name": "pred_logits",
            "role": "class_logits",
            "encoding": "raw_logits",
            "rank": 3,
        },
        {
            "name": "pred_boxes",
            "role": "boxes",
            "encoding": "cxcywh_normalized",
            "rank": 3,
        },
    ]


def expected_omdet_turbo_coreml_shapes(
    *,
    size: str,
    nc: int,
) -> dict[str, tuple[int, ...]]:
    """Return exact output shapes for a frozen vocabulary."""
    profile = validate_omdet_turbo_coreml_profile(size=size)
    if isinstance(nc, bool) or int(nc) <= 0:
        raise ValueError(
            "OMDet-Turbo Core ML nc must be a positive integer."
        )
    return {
        "pred_logits": (1, profile.num_queries, int(nc)),
        "pred_boxes": (1, profile.num_queries, 4),
    }


def omdet_turbo_coreml_metadata(
    *,
    size: str,
    names: Mapping[int, str] | Sequence[str],
) -> dict[str, Any]:
    """Return strict frozen-vocabulary metadata for backend loading."""
    labels = _ordered_names(names)
    profile = validate_omdet_turbo_coreml_profile(size=size)
    return {
        "frozen_classes": True,
        "omdet_turbo_contract": OMDET_TURBO_COREML_CONTRACT,
        "omdet_turbo_preprocess": OMDET_TURBO_COREML_PREPROCESS,
        "omdet_turbo_postprocess": OMDET_TURBO_COREML_POSTPROCESS,
        "omdet_turbo_task_template": OMDET_TURBO_COREML_TASK_TEMPLATE,
        "omdet_turbo_task": omdet_turbo_coreml_task(labels),
        "omdet_turbo_vocabulary_sha256": (
            omdet_turbo_coreml_vocabulary_hash(labels)
        ),
        "omdet_turbo_image_size": profile.image_size,
        "omdet_turbo_num_queries": profile.num_queries,
        "omdet_turbo_num_classes": len(labels),
    }


def _strict_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "1"}:
        return True
    if token in {"false", "0"}:
        return False
    raise ValueError(
        f"OMDet-Turbo Core ML metadata {key!r} must be true or false."
    )


def _strict_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"OMDet-Turbo Core ML metadata {key!r} must be an integer."
        )
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise ValueError(
            f"OMDet-Turbo Core ML metadata {key!r} must be an integer."
        )
    return parsed


def validate_omdet_turbo_coreml_metadata(
    metadata: Mapping[str, Any],
    *,
    size: str,
    names: Mapping[int, str] | Sequence[str],
) -> OmDetTurboCoreMLProfile:
    """Reject forged or stale frozen-vocabulary artifacts."""
    expected = omdet_turbo_coreml_metadata(size=size, names=names)
    missing = sorted(key for key in expected if key not in metadata)
    if missing:
        raise ValueError(
            "OMDet-Turbo Core ML metadata is missing strict keys: "
            f"{missing!r}."
        )
    if not _strict_bool(metadata["frozen_classes"], key="frozen_classes"):
        raise ValueError(
            "OMDet-Turbo Core ML artifacts must declare frozen_classes=true."
        )

    integer_keys = {
        "omdet_turbo_image_size",
        "omdet_turbo_num_queries",
        "omdet_turbo_num_classes",
    }
    for key, expected_value in expected.items():
        if key == "frozen_classes":
            continue
        actual = (
            _strict_int(metadata[key], key=key)
            if key in integer_keys
            else str(metadata[key])
        )
        if actual != expected_value:
            raise ValueError(
                f"OMDet-Turbo Core ML metadata {key!r}={actual!r} does "
                f"not match the frozen contract value {expected_value!r}."
            )
    return validate_omdet_turbo_coreml_profile(size=size)


def freeze_omdet_turbo_language_embeddings(
    model: nn.Module,
    processor: Any,
    names: Mapping[int, str] | Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute class/task embeddings once with the live language backbone."""
    labels = _ordered_names(names)
    task = omdet_turbo_coreml_task(labels)
    if not callable(processor):
        raise TypeError(
            "OMDet-Turbo Core ML export requires the loaded processor."
        )
    get_embedding = getattr(model, "get_language_embedding", None)
    if not callable(get_embedding):
        raise TypeError(
            "OMDet-Turbo Core ML export requires "
            "OmDetTurboForObjectDetection.get_language_embedding()."
        )

    # The processor owns tokenizer padding/truncation defaults.  A one-pixel
    # image is sufficient because only the emitted language fields are used.
    inputs = processor(
        images=Image.new("RGB", (8, 8)),
        text=labels,
        task=task,
        return_tensors="pt",
    )
    required = (
        "classes_input_ids",
        "classes_attention_mask",
        "tasks_input_ids",
        "tasks_attention_mask",
        "classes_structure",
    )
    missing = [key for key in required if key not in inputs]
    if missing:
        raise RuntimeError(
            "OMDet-Turbo processor omitted language inputs required for "
            f"freezing: {missing!r}."
        )

    try:
        parameter = next(model.parameters())
    except StopIteration as exc:
        raise TypeError(
            "OMDet-Turbo Core ML export requires a parameterized model."
        ) from exc
    prepared = {
        key: inputs[key].to(device=parameter.device)
        for key in required
    }
    with torch.inference_mode():
        class_features, task_features, task_mask = get_embedding(
            prepared["classes_input_ids"],
            prepared["classes_attention_mask"],
            prepared["tasks_input_ids"],
            prepared["tasks_attention_mask"],
            prepared["classes_structure"],
        )

    class_features = class_features.detach().to(
        device="cpu",
        dtype=torch.float32,
    ).clone()
    task_features = task_features.detach().to(
        device="cpu",
        dtype=torch.float32,
    ).clone()
    task_mask = task_mask.detach().to(device="cpu").clone()
    expected_classes = len(labels)
    if (
        class_features.ndim != 3
        or tuple(class_features.shape[:2]) != (expected_classes, 1)
    ):
        raise RuntimeError(
            "OMDet-Turbo frozen class embeddings must have shape "
            f"[C, 1, D]; got {tuple(class_features.shape)}."
        )
    if (
        task_features.ndim != 3
        or task_features.shape[1] != 1
        or task_features.shape[2] != class_features.shape[2]
    ):
        raise RuntimeError(
            "OMDet-Turbo frozen task embeddings must have shape [T, 1, D] "
            "and share D with class embeddings; got "
            f"{tuple(task_features.shape)}."
        )
    if tuple(task_mask.shape) != (1, int(task_features.shape[0])):
        raise RuntimeError(
            "OMDet-Turbo frozen task mask must have shape [1, T]; got "
            f"{tuple(task_mask.shape)}."
        )
    if not bool(torch.isfinite(class_features).all()) or not bool(
        torch.isfinite(task_features).all()
    ):
        raise RuntimeError(
            "OMDet-Turbo frozen language embeddings contain NaN or infinity."
        )
    return (
        class_features.contiguous(),
        task_features.contiguous(),
        task_mask.contiguous(),
    )


class OmDetTurboFrozenCoreMLAdapter(nn.Module):
    """Image-only OMDet-Turbo graph with immutable language features."""

    def __init__(
        self,
        model: nn.Module,
        class_features: torch.Tensor,
        task_features: torch.Tensor,
        task_mask: torch.Tensor,
        *,
        encoder_override: nn.Module | None = None,
        decoder_override: nn.Module | None = None,
    ):
        super().__init__()
        for name in ("vision_backbone", "encoder", "decoder"):
            if not isinstance(getattr(model, name, None), nn.Module):
                raise TypeError(
                    "OMDet-Turbo Core ML adapter requires model."
                    f"{name} to be an nn.Module."
                )
        if class_features.ndim != 3 or class_features.shape[1] != 1:
            raise ValueError(
                "OMDet-Turbo class_features must have shape [C, 1, D]."
            )
        if (
            task_features.ndim != 3
            or task_features.shape[1] != 1
            or task_features.shape[2] != class_features.shape[2]
        ):
            raise ValueError(
                "OMDet-Turbo task_features must have shape [T, 1, D]."
            )
        if tuple(task_mask.shape) != (1, int(task_features.shape[0])):
            raise ValueError(
                "OMDet-Turbo task_mask must have shape [1, T]."
            )

        self.vision_backbone = model.vision_backbone
        self.encoder = (
            model.encoder if encoder_override is None else encoder_override
        )
        self.decoder = (
            model.decoder if decoder_override is None else decoder_override
        )
        self.register_buffer(
            "class_features",
            class_features.detach().to(dtype=torch.float32).contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "task_features",
            task_features.detach().to(dtype=torch.float32).contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "task_mask",
            task_mask.detach().contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "image_mean",
            torch.tensor(OMDET_TURBO_COREML_MEAN).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(OMDET_TURBO_COREML_STD).view(1, 3, 1, 1),
            persistent=True,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final raw class logits and normalized ``cxcywh`` boxes."""
        pixel_values = (image - self.image_mean) / self.image_std
        image_features = self.vision_backbone(pixel_values)
        encoder_outputs = self.encoder(
            image_features,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        decoder_outputs = self.decoder(
            encoder_outputs[-1],
            self.class_features,
            self.task_features,
            self.task_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        return decoder_outputs[4][-1], decoder_outputs[3][-1]


class OmDetTurboFixedEncoderCoreMLAdapter(nn.Module):
    """Hybrid encoder with fixed, pre-baked 2D position embeddings.

    CoreMLTools 9 lowers traced ``torch.arange(dtype=float32)`` as an int32
    range in this graph, then rejects its matrix multiplication with the fp32
    sinusoid frequencies.  The deployment canvas is fixed, so generating the
    deterministic position table once is both exact and removes that
    converter ambiguity.
    """

    def __init__(
        self,
        encoder: nn.Module,
        vision_backbone: nn.Module,
        *,
        image_size: int,
    ):
        super().__init__()
        required = (
            "in_channels",
            "encoder_hidden_dim",
            "encoder_projection_indices",
            "positional_encoding_temperature",
            "channel_projection_layers",
            "encoder",
            "lateral_convs",
            "fpn_blocks",
            "downsample_convs",
            "pan_blocks",
            "build_2d_sincos_position_embedding",
        )
        missing = [name for name in required if not hasattr(encoder, name)]
        if missing:
            raise TypeError(
                "OMDet-Turbo fixed Core ML encoder is missing attributes "
                f"{missing!r}."
            )
        self.in_channels = tuple(int(value) for value in encoder.in_channels)
        self.encoder_hidden_dim = int(encoder.encoder_hidden_dim)
        self.encoder_projection_indices = tuple(
            int(value) for value in encoder.encoder_projection_indices
        )
        self.channel_projection_layers = encoder.channel_projection_layers
        self.transformer_encoders = encoder.encoder
        self.lateral_convs = encoder.lateral_convs
        self.fpn_blocks = encoder.fpn_blocks
        self.downsample_convs = encoder.downsample_convs
        self.pan_blocks = encoder.pan_blocks

        mean = torch.tensor(OMDET_TURBO_COREML_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(OMDET_TURBO_COREML_STD).view(1, 3, 1, 1)
        dummy = torch.zeros(1, 3, image_size, image_size)
        with torch.no_grad():
            features = vision_backbone((dummy - mean) / std)
            projected = [
                layer(feature)
                for layer, feature in zip(
                    self.channel_projection_layers,
                    features,
                )
            ]
        self._position_buffer_names: list[str] = []
        for ordinal, feature_index in enumerate(
            self.encoder_projection_indices
        ):
            height, width = (
                int(value) for value in projected[feature_index].shape[-2:]
            )
            position = encoder.build_2d_sincos_position_embedding(
                width,
                height,
                self.encoder_hidden_dim,
                float(encoder.positional_encoding_temperature),
                device="cpu",
                dtype=torch.float32,
            )
            name = f"position_embedding_{ordinal}"
            self.register_buffer(
                name,
                position.detach().clone().contiguous(),
                persistent=True,
            )
            self._position_buffer_names.append(name)

    def forward(
        self,
        hidden_states,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        """Run the pinned hybrid encoder with constant positional tables."""
        if output_attentions or output_hidden_states or return_dict:
            raise ValueError(
                "OMDet-Turbo fixed Core ML encoder exposes only its compact "
                "return_dict=False inference path."
            )
        projected_features = [
            self.channel_projection_layers[index](feature)
            for index, feature in enumerate(hidden_states)
        ]
        for encoder_index, feature_index in enumerate(
            self.encoder_projection_indices
        ):
            feature = projected_features[feature_index]
            height, width = feature.shape[2:]
            flattened = feature.flatten(2).permute(0, 2, 1)
            position = getattr(
                self,
                self._position_buffer_names[encoder_index],
            )
            encoded = self.transformer_encoders[encoder_index](
                flattened,
                pos_embed=position,
                output_attentions=False,
            )[0]
            projected_features[feature_index] = (
                encoded.permute(0, 2, 1)
                .reshape(-1, self.encoder_hidden_dim, height, width)
                .contiguous()
            )

        fpn_feature_maps = [projected_features[-1]]
        for index in range(len(self.in_channels) - 1, 0, -1):
            high = fpn_feature_maps[0]
            low = projected_features[index - 1]
            block_index = len(self.in_channels) - 1 - index
            high = self.lateral_convs[block_index](high)
            fpn_feature_maps[0] = high
            upsampled = F.interpolate(high, scale_factor=2.0, mode="nearest")
            fused = self.fpn_blocks[block_index](
                torch.concat([upsampled, low], dim=1)
            )
            fpn_feature_maps.insert(0, fused)

        fpn_states = [fpn_feature_maps[0]]
        for index in range(len(self.in_channels) - 1):
            low = fpn_states[-1]
            high = fpn_feature_maps[index + 1]
            downsampled = self.downsample_convs[index](low)
            fused = self.pan_blocks[index](
                torch.concat(
                    [downsampled, high.to(downsampled.device)],
                    dim=1,
                )
            )
            fpn_states.append(fused)
        return (fpn_states[-1], None, None, fpn_states)


class OmDetTurboFixedDecoderCoreMLAdapter(nn.Module):
    """Inference decoder without CoreMLTools-incompatible bool assignment.

    Query selection deliberately retains the loaded decoder's exact projection
    modules.  Deep-copying the whole decoder changes low-order CPU GEMM
    numerics enough to swap nearly tied top-k queries.  Only the transformer
    layers that need an operator rewrite are cloned, keeping export
    non-mutating while preserving the source model's query order.
    """

    def __init__(self, decoder: nn.Module, task_mask: torch.Tensor):
        super().__init__()
        required = (
            "config",
            "num_queries",
            "class_distance_type",
            "learn_initial_query",
            "channel_projection_layers",
            "task_encoder",
            "layers",
            "decoder_num_layers",
            "query_position_head",
            "encoder_vision_features",
            "encoder_class_head",
            "encoder_bbox_head",
            "decoder_class_head",
            "decoder_bbox_head",
        )
        missing = [name for name in required if not hasattr(decoder, name)]
        if missing:
            raise TypeError(
                "OMDet-Turbo fixed Core ML decoder is missing attributes "
                f"{missing!r}."
            )
        self.config = decoder.config
        self.num_queries = int(decoder.num_queries)
        self.class_distance_type = str(decoder.class_distance_type)
        self.learn_initial_query = bool(decoder.learn_initial_query)
        self.decoder_num_layers = int(decoder.decoder_num_layers)
        self.channel_projection_layers = decoder.channel_projection_layers
        self.task_encoder = decoder.task_encoder
        self.task_project = getattr(decoder, "task_project", None)
        self.query_position_head = decoder.query_position_head
        self.encoder_vision_features = decoder.encoder_vision_features
        self.encoder_class_head = decoder.encoder_class_head
        self.encoder_bbox_head = decoder.encoder_bbox_head
        self.decoder_class_head = decoder.decoder_class_head
        self.decoder_bbox_head = decoder.decoder_bbox_head
        if self.learn_initial_query:
            self.tgt_embed = decoder.tgt_embed

        # Only decoder layers contain the deformable-attention operator that
        # must be lowered. Cloning this suffix avoids mutating the live model
        # and leaves pre-top-k projections byte-for-byte on their source
        # allocations.
        self.layers = deepcopy(decoder.layers).eval()
        replaced = _replace_omdet_turbo_deformable_attention(self.layers)
        if replaced <= 0:
            raise RuntimeError(
                "OMDet-Turbo Core ML export found no deformable-attention "
                "modules to lower."
            )
        from transformers.masking_utils import create_bidirectional_mask

        src_key_mask = task_mask == 0
        query_mask = torch.zeros(
            (task_mask.shape[0], self.num_queries),
            dtype=torch.bool,
        )
        key_padding_mask = torch.concat(
            (query_mask, src_key_mask),
            dim=1,
        )
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=torch.ones_like(
                key_padding_mask,
                dtype=torch.float32,
            )[..., None],
            attention_mask=~key_padding_mask,
        )
        self.register_buffer(
            "attention_mask",
            (
                None
                if attention_mask is None
                else attention_mask.detach().clone().contiguous()
            ),
            persistent=True,
        )

    @staticmethod
    def _generate_anchors(
        spatial_shapes,
        *,
        grid_size: float = 0.05,
        device="cpu",
        dtype=torch.float32,
    ):
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=height, dtype=dtype, device=device),
                torch.arange(end=width, dtype=dtype, device=device),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            valid_wh = torch.tensor(
                [width, height],
                dtype=dtype,
                device=device,
            )
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh
            wh = (
                torch.ones_like(grid_xy, dtype=dtype)
                * grid_size
                * (2.0**level)
            )
            anchors.append(
                torch.concat([grid_xy, wh], dim=-1).reshape(
                    -1,
                    height * width,
                    4,
                )
            )
        anchors = torch.concat(anchors, dim=1)
        valid_mask = ((anchors > 1e-2) * (anchors < 1 - 1e-2)).all(
            dim=-1,
            keepdim=True,
        )
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_encoder_input(self, vision_features):
        projected = [
            self.channel_projection_layers[index](feature)
            for index, feature in enumerate(vision_features)
        ]
        flattened = []
        shape_list = []
        for feature in projected:
            height, width = feature.shape[2:]
            flattened.append(feature.flatten(2).permute(0, 2, 1))
            shape_list.append((height, width))
        flattened_tensor = torch.cat(flattened, dim=1)
        shapes = torch.tensor(
            shape_list,
            dtype=torch.int64,
            device=projected[0].device,
        )
        level_start_index = torch.cat(
            (
                shapes.new_zeros((1,)),
                shapes.prod(1).cumsum(0)[:-1],
            )
        )
        return (
            flattened_tensor,
            shapes,
            shape_list,
            level_start_index,
        )

    def _get_decoder_input(
        self,
        vision_features,
        vision_shapes,
        class_features,
    ):
        from transformers.models.omdet_turbo.modeling_omdet_turbo import (
            get_class_similarity,
        )

        batch_size = len(vision_features)
        anchors, valid_mask = self._generate_anchors(
            vision_shapes,
            device=vision_features.device,
            dtype=vision_features.dtype,
        )
        predicted_class_features = self.encoder_vision_features(
            torch.where(
                valid_mask,
                vision_features,
                torch.tensor(
                    0.0,
                    dtype=vision_features.dtype,
                    device=vision_features.device,
                ),
            )
        )
        projected_classes = self.encoder_class_head(
            class_features
        ).permute(1, 2, 0)
        encoder_similarity = get_class_similarity(
            self.class_distance_type,
            predicted_class_features,
            projected_classes,
        )
        encoder_boxes = self.encoder_bbox_head(
            predicted_class_features
        ) + anchors
        topk_indices = torch.topk(
            encoder_similarity.max(-1).values,
            self.num_queries,
            dim=1,
        ).indices.view(-1)
        batch_indices = (
            torch.arange(
                end=batch_size,
                dtype=topk_indices.dtype,
                device=topk_indices.device,
            )
            .unsqueeze(-1)
            .repeat(1, self.num_queries)
            .view(-1)
        )
        reference_points = encoder_boxes[
            batch_indices,
            topk_indices,
        ].view(batch_size, self.num_queries, -1)
        if self.learn_initial_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(
                batch_size,
                1,
                1,
            )
        else:
            embeddings = predicted_class_features[
                batch_indices,
                topk_indices,
            ].view(batch_size, self.num_queries, -1)
        return embeddings, reference_points

    def forward(
        self,
        vision_features,
        class_features,
        task_features,
        task_mask,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        """Return the same compact tuple as the pinned inference decoder."""
        if output_attentions or output_hidden_states or return_dict:
            raise ValueError(
                "OMDet-Turbo fixed Core ML decoder exposes only its compact "
                "return_dict=False inference path."
            )
        from transformers.models.omdet_turbo.modeling_omdet_turbo import (
            _inverse_sigmoid,
            get_class_similarity,
        )

        (
            vision_features,
            vision_shapes,
            vision_shapes_list,
            level_start_index,
        ) = self._get_encoder_input(vision_features)
        task_features = self.task_encoder(task_features)
        if self.task_project is not None:
            task_features = self.task_project(task_features)

        predicted_class_features, reference_points = self._get_decoder_input(
            vision_features,
            tuple(vision_shapes_list),
            class_features,
        )

        reference_points = reference_points.sigmoid()
        final_boxes = None
        final_logits = None
        for index, layer in enumerate(self.layers):
            (
                predicted_class_features,
                task_features,
                _self_attention,
                _cross_attention,
            ) = layer(
                predicted_class_features,
                task_features,
                reference_points,
                vision_features,
                vision_shapes,
                vision_shapes_list,
                level_start_index=level_start_index,
                attention_mask=self.attention_mask,
                query_position=self.query_position_head(
                    reference_points
                ),
                output_attentions=False,
                output_hidden_states=False,
            )
            refined_boxes = torch.sigmoid(
                self.decoder_bbox_head[index](
                    predicted_class_features
                )
                + _inverse_sigmoid(reference_points)
            )
            if index == self.decoder_num_layers - 1:
                projected_classes = self.decoder_class_head[index](
                    class_features
                ).permute(1, 2, 0)
                final_logits = get_class_similarity(
                    self.class_distance_type,
                    predicted_class_features,
                    projected_classes,
                )
                final_boxes = refined_boxes
                break
            reference_points = refined_boxes

        if final_logits is None or final_boxes is None:
            raise RuntimeError(
                "OMDet-Turbo Core ML decoder did not emit its final layer."
            )
        return (
            predicted_class_features,
            None,
            None,
            final_boxes.unsqueeze(0),
            final_logits.unsqueeze(0),
            None,
            None,
            None,
            reference_points,
        )


class OmDetTurboDeformableAttentionCoreMLAdapter(nn.Module):
    """Rank-five equivalent of OMDet-Turbo deformable cross-attention."""

    def __init__(self, attention: nn.Module):
        super().__init__()
        for name in (
            "sampling_offsets",
            "attention_weights",
            "value_proj",
            "output_proj",
            "n_heads",
            "n_levels",
            "n_points",
            "d_model",
        ):
            if not hasattr(attention, name):
                raise TypeError(
                    "OMDet-Turbo deformable attention is missing "
                    f"attribute {name!r}."
                )
        self.sampling_offsets = attention.sampling_offsets
        self.attention_weights = attention.attention_weights
        self.value_proj = attention.value_proj
        self.output_proj = attention.output_proj
        self.n_heads = int(attention.n_heads)
        self.n_levels = int(attention.n_levels)
        self.n_points = int(attention.n_points)
        self.d_model = int(attention.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: torch.Tensor | None = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the same points without ever creating a rank-six tensor."""
        del encoder_attention_mask, level_start_index, kwargs
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        sequence_length = encoder_hidden_states.shape[1]
        head_dim = self.d_model // self.n_heads
        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(
                ~attention_mask[..., None],
                float(0),
            )
        value = value.view(
            batch_size,
            sequence_length,
            self.n_heads,
            head_dim,
        )

        offsets = self.sampling_offsets(hidden_states).view(
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels * self.n_points * 2,
        )
        offsets = offsets.permute(0, 2, 1, 3).reshape(
            batch_size * self.n_heads,
            num_queries,
            self.n_levels,
            self.n_points,
            2,
        )
        flat_weights = self.attention_weights(hidden_states).view(
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels * self.n_points,
        )
        flat_weights = F.softmax(flat_weights, dim=-1)
        returned_weights = flat_weights.view(
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels,
            self.n_points,
        )
        head_weights = flat_weights.permute(0, 2, 1, 3).reshape(
            batch_size * self.n_heads,
            num_queries,
            self.n_levels,
            self.n_points,
        )

        coordinates = int(reference_points.shape[-1])
        references = (
            reference_points.unsqueeze(1)
            .expand(
                batch_size,
                self.n_heads,
                num_queries,
                self.n_levels,
                coordinates,
            )
            .reshape(
                batch_size * self.n_heads,
                num_queries,
                self.n_levels,
                coordinates,
            )
        )
        if coordinates == 2:
            normalizer = torch.stack(
                (spatial_shapes[..., 1], spatial_shapes[..., 0]),
                dim=-1,
            )
            locations = references.unsqueeze(-2) + offsets / normalizer.view(
                1,
                1,
                self.n_levels,
                1,
                2,
            )
        elif coordinates == 4:
            locations = (
                references[..., :2].unsqueeze(-2)
                + offsets
                / self.n_points
                * references[..., 2:].unsqueeze(-2)
                * 0.5
            )
        else:
            raise ValueError(
                "OMDet-Turbo reference points must end in 2 or 4 "
                f"coordinates; got {coordinates}."
            )

        sampling_grids = 2 * locations - 1
        split_sizes = [
            int(height) * int(width)
            for height, width in spatial_shapes_list
        ]
        value_levels = value.split(split_sizes, dim=1)
        sampled_levels = []
        for level, (height, width) in enumerate(spatial_shapes_list):
            level_value = (
                value_levels[level]
                .flatten(2)
                .transpose(1, 2)
                .reshape(
                    batch_size * self.n_heads,
                    head_dim,
                    int(height),
                    int(width),
                )
            )
            grid = sampling_grids[:, :, level]
            sampled_levels.append(
                F.grid_sample(
                    level_value,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            )
        sampled = torch.stack(sampled_levels, dim=-2).flatten(-2)
        weights = head_weights.reshape(
            batch_size * self.n_heads,
            1,
            num_queries,
            self.n_levels * self.n_points,
        )
        output = (sampled * weights).sum(-1).view(
            batch_size,
            self.n_heads * head_dim,
            num_queries,
        )
        output = output.transpose(1, 2).contiguous()
        return self.output_proj(output), returned_weights


def _replace_omdet_turbo_deformable_attention(module: nn.Module) -> int:
    """Replace every pinned HF deformable-attention child in place."""
    from transformers.models.omdet_turbo.modeling_omdet_turbo import (
        OmDetTurboMultiscaleDeformableAttention,
    )

    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, OmDetTurboMultiscaleDeformableAttention):
            setattr(
                module,
                name,
                OmDetTurboDeformableAttentionCoreMLAdapter(child),
            )
            count += 1
        else:
            count += _replace_omdet_turbo_deformable_attention(child)
    return count


def build_omdet_turbo_frozen_coreml_adapter(
    model: nn.Module,
    processor: Any,
    *,
    size: str,
    names: Mapping[int, str] | Sequence[str],
) -> OmDetTurboFrozenCoreMLAdapter:
    """Freeze the text condition and return a CPU image-only graph."""
    require_omdet_turbo_transformers_toolchain()
    validate_omdet_turbo_coreml_profile(size=size)
    class_features, task_features, task_mask = (
        freeze_omdet_turbo_language_embeddings(model, processor, names)
    )
    cpu_model = model.to(device="cpu", dtype=torch.float32).eval()
    profile = validate_omdet_turbo_coreml_profile(size=size)
    fixed_encoder = OmDetTurboFixedEncoderCoreMLAdapter(
        cpu_model.encoder,
        cpu_model.vision_backbone,
        image_size=profile.image_size,
    )
    fixed_decoder = OmDetTurboFixedDecoderCoreMLAdapter(
        cpu_model.decoder,
        task_mask,
    )
    adapter = OmDetTurboFrozenCoreMLAdapter(
        cpu_model,
        class_features,
        task_features,
        task_mask,
        encoder_override=fixed_encoder,
        decoder_override=fixed_decoder,
    )
    return adapter.eval()


def prepare_omdet_turbo_coreml_export(
    model: Any,
    kwargs: Mapping[str, Any],
    *,
    default_output: str = "omdet_turbo_coreml.mlpackage",
) -> tuple[int, str, dict[str, Any], str, str]:
    """Validate one direct frozen-vocabulary export request."""
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
    conf = options.pop("conf", 0.3)
    iou = options.pop("iou", 0.5)
    max_det = options.pop("max_det", 300)

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
        raise TypeError(
            f"Unsupported OMDet-Turbo Core ML export options: {names}"
        )
    if dynamic:
        raise NotImplementedError(
            "Frozen-vocabulary OMDet-Turbo Core ML export uses fixed image "
            "and class shapes; dynamic=True is not supported."
        )
    if batch != 1:
        raise ValueError(
            "Frozen-vocabulary OMDet-Turbo Core ML export requires batch=1; "
            f"got batch={batch}."
        )
    if nms:
        raise NotImplementedError(
            "OMDet-Turbo Core ML exports raw logits and boxes, with exact "
            "class-aware NMS on the host; nms=True is not supported."
        )
    if device not in (None, "auto", "cpu", torch.device("cpu")):
        raise NotImplementedError(
            "Core ML conversion traces on CPU; pass device='cpu', "
            "device='auto', or omit device."
        )

    size = str(getattr(model, "size", "")).strip().lower()
    profile = validate_omdet_turbo_coreml_profile(size=size)
    if imgsz is None:
        requested = (profile.image_size, profile.image_size)
    elif isinstance(imgsz, (tuple, list)):
        if len(imgsz) != 2:
            raise ValueError(
                f"imgsz must be an int or (height, width), got {imgsz}"
            )
        requested = (int(imgsz[0]), int(imgsz[1]))
    else:
        requested = (int(imgsz), int(imgsz))
    validate_omdet_turbo_coreml_profile(size=size, canvas_hw=requested)

    labels = _ordered_names(getattr(model, "names", {}))
    if int(getattr(model, "nb_classes", 0)) != len(labels):
        raise RuntimeError(
            "OMDet-Turbo class metadata is inconsistent: nb_classes must "
            "match names."
        )

    exporter = CoreMLExporter(model)
    half, int8 = exporter._validate(half, int8, data)
    exporter._preflight(
        half=half,
        int8=int8,
        data=data,
        nms=False,
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
        imgsz=requested,
    )
    metadata.update(
        omdet_turbo_coreml_metadata(
            size=size,
            names=getattr(model, "names", {}),
        )
    )

    destination = Path(output_path)
    if destination.suffix != ".mlpackage":
        destination = destination.with_suffix(".mlpackage")
    return (
        profile.image_size,
        str(destination),
        metadata,
        precision,
        compute_units,
    )


def export_omdet_turbo_coreml(
    model: Any,
    kwargs: Mapping[str, Any],
) -> str:
    """Export the current class vocabulary as an image-only package."""
    (
        image_size,
        output_path,
        metadata,
        precision,
        compute_units,
    ) = prepare_omdet_turbo_coreml_export(model, kwargs)
    live_model = model.model
    try:
        reference_parameter = next(live_model.parameters())
    except StopIteration as exc:
        raise TypeError(
            "OMDet-Turbo Core ML export requires a parameterized model."
        ) from exc
    original_device = reference_parameter.device
    original_dtype = reference_parameter.dtype
    original_training = tuple(
        (module, module.training) for module in live_model.modules()
    )

    try:
        adapter = build_omdet_turbo_frozen_coreml_adapter(
            live_model,
            model.processor,
            size=model.size,
            names=model.names,
        )
        dummy = torch.zeros(
            1,
            3,
            image_size,
            image_size,
            dtype=torch.float32,
        )

        require_omdet_turbo_coremltools_toolchain()
        from .coreml import export_coreml

        return export_coreml(
            adapter,
            dummy,
            output_path=output_path,
            precision=precision,
            compute_units=compute_units,
            nms=False,
            metadata=metadata,
            model_family="omdet_turbo",
            model_task="detect",
            model_size=model.size,
        )
    finally:
        live_model.to(device=original_device, dtype=original_dtype)
        for module, training in original_training:
            module.training = training


def preprocess_omdet_turbo_coreml_image(
    image: Image.Image | np.ndarray,
    *,
    image_size: int = 640,
) -> torch.Tensor:
    """Resize an RGB image exactly and return NCHW float pixels in ``[0, 255]``.

    ImageNet normalization remains inside
    :class:`OmDetTurboFrozenCoreMLAdapter`.  Transformers v5.12.1 uses
    Torchvision v2 bilinear antialiasing on the uint8 tensor before converting
    it to float.  A Pillow resize or an antialiased float resize differs by up
    to one integer pixel, which is material to the detector.
    """
    if isinstance(image_size, bool) or int(image_size) <= 0:
        raise ValueError(
            "OMDet-Turbo Core ML image_size must be positive."
        )
    image_size = int(image_size)
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] not in {3, 4}:
            raise ValueError(
                "OMDet-Turbo Core ML preprocessing requires an HWC RGB/RGBA "
                "image."
            )
        if not np.issubdtype(rgb.dtype, np.number):
            raise TypeError(
                "OMDet-Turbo Core ML image values must be numeric."
            )
        if not np.isfinite(rgb).all():
            raise ValueError(
                "OMDet-Turbo Core ML image contains NaN or infinity."
            )
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        if np.issubdtype(rgb.dtype, np.floating) and float(rgb.max()) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        pil = Image.fromarray(rgb, mode="RGB")

    rgb = np.asarray(pil, dtype=np.uint8)
    uint8_tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1)
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as tv_functional

    resized = tv_functional.resize(
        uint8_tensor,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    tensor = resized.unsqueeze(0).to(dtype=torch.float32)
    expected = (1, 3, image_size, image_size)
    if tuple(tensor.shape) != expected:
        raise RuntimeError(
            "OMDet-Turbo Core ML preprocessing produced invalid shape "
            f"{tuple(tensor.shape)}; expected {expected}."
        )
    return tensor.contiguous()


def postprocess_omdet_turbo_coreml_outputs(
    pred_logits: torch.Tensor | np.ndarray,
    pred_boxes: torch.Tensor | np.ndarray,
    *,
    original_size: tuple[int, int],
    conf: float,
    iou: float,
    max_det: int,
    classes: Sequence[int] | None = None,
) -> dict[str, torch.Tensor | int]:
    """Decode frozen OMDet-Turbo outputs with its top-900/NMS contract."""
    logits = torch.as_tensor(pred_logits, dtype=torch.float32)
    boxes = torch.as_tensor(pred_boxes, dtype=torch.float32)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] <= 0:
        raise ValueError(
            "OMDet-Turbo Core ML pred_logits must have shape [1, Q, C] "
            "with C > 0."
        )
    if boxes.shape != (1, logits.shape[1], 4):
        raise ValueError(
            "OMDet-Turbo Core ML pred_boxes must have shape [1, Q, 4] "
            "and share Q."
        )
    if not bool(torch.isfinite(logits).all()) or not bool(
        torch.isfinite(boxes).all()
    ):
        raise RuntimeError(
            "OMDet-Turbo Core ML outputs contain NaN or infinity."
        )
    try:
        width, height = (int(value) for value in original_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "OMDet-Turbo original_size must be a (width, height) pair."
        ) from exc
    if width <= 0 or height <= 0:
        raise ValueError(
            "OMDet-Turbo original image dimensions must be positive."
        )
    try:
        threshold = float(conf)
        nms_threshold = float(iou)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "OMDet-Turbo confidence and IoU thresholds must be numeric."
        ) from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "OMDet-Turbo confidence threshold must be finite in [0, 1]."
        )
    if not np.isfinite(nms_threshold) or not 0.0 <= nms_threshold <= 1.0:
        raise ValueError(
            "OMDet-Turbo IoU threshold must be finite in [0, 1]."
        )
    if isinstance(max_det, bool) or int(max_det) <= 0:
        raise ValueError(
            "OMDet-Turbo max_det must be a positive integer."
        )

    num_queries = int(logits.shape[1])
    num_classes = int(logits.shape[2])
    flat_scores = torch.sigmoid(logits[0]).flatten()
    top_count = min(num_queries, int(flat_scores.numel()))
    scores, indices = flat_scores.topk(top_count, sorted=False)
    class_ids = indices.remainder(num_classes).to(dtype=torch.int64)
    query_ids = torch.div(
        indices,
        num_classes,
        rounding_mode="floor",
    )
    selected_boxes = boxes[0][query_ids]

    center_x, center_y, box_width, box_height = selected_boxes.unbind(dim=-1)
    decoded = torch.stack(
        (
            center_x - box_width / 2,
            center_y - box_height / 2,
            center_x + box_width / 2,
            center_y + box_height / 2,
        ),
        dim=-1,
    )
    decoded = decoded * torch.tensor(
        [width, height, width, height],
        dtype=decoded.dtype,
    )

    keep = scores > threshold
    if classes is not None:
        if isinstance(classes, (str, bytes)):
            raise TypeError(
                "OMDet-Turbo classes filter must be a sequence of ids."
            )
        try:
            allowed = torch.as_tensor(
                [int(value) for value in classes],
                dtype=torch.int64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "OMDet-Turbo classes filter must contain integer ids."
            ) from exc
        if allowed.numel() == 0:
            keep &= False
        else:
            if bool((allowed < 0).any()) or bool(
                (allowed >= num_classes).any()
            ):
                raise ValueError(
                    "OMDet-Turbo classes filter contains an out-of-range id."
                )
            keep &= (class_ids[:, None] == allowed[None, :]).any(dim=1)

    decoded = decoded[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    if decoded.numel():
        from torchvision.ops import batched_nms

        nms_keep = batched_nms(
            decoded,
            scores,
            class_ids,
            nms_threshold,
        )
        decoded = decoded[nms_keep]
        scores = scores[nms_keep]
        class_ids = class_ids[nms_keep]
        decoded[:, 0::2].clamp_(0.0, float(width))
        decoded[:, 1::2].clamp_(0.0, float(height))
        valid = (decoded[:, 2] > decoded[:, 0]) & (
            decoded[:, 3] > decoded[:, 1]
        )
        decoded = decoded[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
    if scores.numel():
        decoded = decoded[: int(max_det)]
        scores = scores[: int(max_det)]
        class_ids = class_ids[: int(max_det)]
    return {
        "boxes": decoded.contiguous(),
        "scores": scores.contiguous(),
        "classes": class_ids.contiguous(),
        "num_detections": int(scores.numel()),
    }


__all__ = [
    "OMDET_TURBO_COREML_CONTRACT",
    "OMDET_TURBO_COREML_INPUT_NAME",
    "OMDET_TURBO_COREML_MEAN",
    "OMDET_TURBO_COREML_OUTPUT_NAMES",
    "OMDET_TURBO_COREML_POSTPROCESS",
    "OMDET_TURBO_COREML_PREPROCESS",
    "OMDET_TURBO_COREML_PROFILES",
    "OMDET_TURBO_COREML_STD",
    "OMDET_TURBO_COREML_TASK_TEMPLATE",
    "OMDET_TURBO_COREMLTOOLS_MAJOR",
    "OMDET_TURBO_TRANSFORMERS_VERSION",
    "OmDetTurboCoreMLProfile",
    "OmDetTurboDeformableAttentionCoreMLAdapter",
    "OmDetTurboFixedDecoderCoreMLAdapter",
    "OmDetTurboFixedEncoderCoreMLAdapter",
    "OmDetTurboFrozenCoreMLAdapter",
    "build_omdet_turbo_frozen_coreml_adapter",
    "expected_omdet_turbo_coreml_shapes",
    "export_omdet_turbo_coreml",
    "freeze_omdet_turbo_language_embeddings",
    "omdet_turbo_coreml_input_contract",
    "omdet_turbo_coreml_metadata",
    "omdet_turbo_coreml_output_contract",
    "omdet_turbo_coreml_task",
    "omdet_turbo_coreml_validation_contract",
    "omdet_turbo_coreml_vocabulary_hash",
    "postprocess_omdet_turbo_coreml_outputs",
    "prepare_omdet_turbo_coreml_export",
    "preprocess_omdet_turbo_coreml_image",
    "require_omdet_turbo_coremltools_toolchain",
    "require_omdet_turbo_transformers_toolchain",
    "validate_omdet_turbo_coreml_metadata",
    "validate_omdet_turbo_coreml_profile",
]
