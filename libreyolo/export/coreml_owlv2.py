"""Frozen-vocabulary OWLv2 contract fragments for Core ML.

OWLv2 is open-vocabulary only while its text tower remains available.  A
single-input Core ML detector must therefore freeze an explicit, finite class
vocabulary at export time.  This module:

* computes the prompt embeddings once with the loaded text tower;
* transfers the compatible Hugging Face state dict into LibreYOLO's existing
  native OWLv2 graph;
* retains only the vision detector modules and the frozen query embeddings;
* declares the exact float input and raw detector output ABI; and
* provides the host preprocessing and postprocessing needed by the generic
  Core ML backend.

The pad/resize preprocessing order is adapted from Hugging Face Transformers
v5.12.1, source commit
``ddb849abe009d1089e6c691bfc897f27211c663c`` (annotated tag object
``a030302dcd4777bbf042ee46c30c5dbe6d2a2eb2``),
``models/owlv2/image_processing_owlv2.py``, Apache-2.0.  Attribution and the
hash-pinned weight provenance are recorded in
``libreyolo/models/owlv2/NOTICE``.
No Core ML conversion or Apple-runtime validation is performed in this
import-safe module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ..models.owlv2.nn import Owlv2DetectionModel, Owlv2Dims

OWLV2_COREML_CONTRACT = "owlv2_frozen_vocabulary_v1"
OWLV2_COREML_PREPROCESS = "rescale_pad_square_gaussian_bilinear_v1"
OWLV2_COREML_POSTPROCESS = "max_class_sigmoid_square_scale_v1"
OWLV2_COREML_PROMPT_TEMPLATE = "a photo of a {}"
OWLV2_COREML_INPUT_NAME = "image"
OWLV2_COREML_OUTPUT_NAMES = ("pred_logits", "pred_boxes")
OWLV2_COREML_MEAN = (0.48145466, 0.4578275, 0.40821073)
OWLV2_COREML_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class Owlv2CoreMLProfile:
    """One fixed OWLv2 image/patch profile."""

    size: str
    image_size: int
    patch_size: int
    projection_dim: int

    @property
    def patch_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.patch_side**2


OWLV2_COREML_PROFILES = {
    "b16": Owlv2CoreMLProfile("b16", 960, 16, 512),
    "l14": Owlv2CoreMLProfile("l14", 1008, 14, 768),
}


def _ordered_names(names: Mapping[int, str] | Sequence[str]) -> list[str]:
    """Return a strict, finite class vocabulary in class-id order."""
    if isinstance(names, Mapping):
        try:
            keys = sorted(int(key) for key in names)
        except (TypeError, ValueError) as exc:
            raise ValueError("OWLv2 Core ML class ids must be integers.") from exc
        if keys != list(range(len(keys))):
            raise ValueError(
                "OWLv2 Core ML class ids must be contiguous from zero; "
                f"got {keys!r}."
            )
        values = [names[key] if key in names else names[str(key)] for key in keys]
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        values = list(names)
    else:
        raise TypeError(
            "OWLv2 Core ML classes must be a mapping or a finite label sequence."
        )

    if not values:
        raise ValueError(
            "OWLv2 Core ML export requires at least one frozen class. "
            "Call set_classes([...]) before export."
        )
    if not all(isinstance(value, str) for value in values):
        raise TypeError("OWLv2 Core ML class labels must all be strings.")
    labels = [value.strip() for value in values]
    if any(not label for label in labels):
        raise ValueError("OWLv2 Core ML class labels must not be blank.")
    normalized = [label.casefold() for label in labels]
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "OWLv2 Core ML class labels must be unique case-insensitively."
        )
    return labels


def owlv2_coreml_prompts(
    names: Mapping[int, str] | Sequence[str],
) -> list[str]:
    """Render the exact prompts frozen into the exported detector."""
    return [
        OWLV2_COREML_PROMPT_TEMPLATE.format(label.lower())
        for label in _ordered_names(names)
    ]


def owlv2_coreml_vocabulary_hash(
    names: Mapping[int, str] | Sequence[str],
) -> str:
    """Hash the ordered labels and prompt template for tamper detection."""
    payload = {
        "labels": _ordered_names(names),
        "prompt_template": OWLV2_COREML_PROMPT_TEMPLATE,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_owlv2_coreml_profile(
    *,
    size: str | None,
    canvas_hw: tuple[int, int] | None = None,
) -> Owlv2CoreMLProfile:
    """Resolve a supported fixed OWLv2 profile and reject interpolation."""
    key = str(size or "").strip().lower()
    profile = OWLV2_COREML_PROFILES.get(key)
    if profile is None:
        raise NotImplementedError(
            "OWLv2 Core ML export supports only the b16 and l14 checkpoints; "
            f"got size={size!r}."
        )
    if canvas_hw is not None:
        height, width = (int(value) for value in canvas_hw)
        expected = (profile.image_size, profile.image_size)
        if (height, width) != expected:
            raise NotImplementedError(
                "OWLv2 learned vision position embeddings require the native "
                f"{expected[1]}x{expected[0]} canvas for size={key!r}; got "
                f"{width}x{height}."
            )
    return profile


def owlv2_coreml_input_contract() -> dict[str, Any]:
    """Declare the dedicated, fractional-pixel OWLv2 host boundary.

    ``owlv2_pad_square`` is intentionally not described as an ordinary
    letterbox.  OWLv2 pads the rescaled image *before* applying its Gaussian
    antialias filter and fixed-size bilinear resize.  Resizing first and then
    padding changes pixels along the real-image/pad boundary.
    """
    return {
        "name": OWLV2_COREML_INPUT_NAME,
        "kind": "tensor",
        "layout": "NCHW",
        "color": "rgb",
        "range": "0_1",
        "geometry": "owlv2_pad_square",
        "interpolation": "bilinear",
        "resize_backend": "torchvision",
        "resize_rounding": "floor",
        "pad_value": 0,
    }


def owlv2_coreml_validation_contract() -> dict[str, str]:
    """Declare canonical source images before dedicated host preprocessing."""
    return {"color": "rgb", "range": "0_255"}


def owlv2_coreml_output_contract() -> list[dict[str, Any]]:
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
            "encoding": "cxcywh_normalized_square",
            "rank": 3,
        },
    ]


def expected_owlv2_coreml_shapes(
    *,
    size: str,
    nc: int,
) -> dict[str, tuple[int, ...]]:
    """Return the exact fixed output shapes for a frozen vocabulary."""
    profile = validate_owlv2_coreml_profile(size=size)
    if isinstance(nc, bool) or int(nc) <= 0:
        raise ValueError("OWLv2 Core ML nc must be a positive integer.")
    return {
        "pred_logits": (1, profile.num_patches, int(nc)),
        "pred_boxes": (1, profile.num_patches, 4),
    }


def owlv2_coreml_metadata(
    *,
    size: str,
    names: Mapping[int, str] | Sequence[str],
) -> dict[str, Any]:
    """Return strict frozen-vocabulary metadata for the backend loader."""
    labels = _ordered_names(names)
    profile = validate_owlv2_coreml_profile(size=size)
    return {
        "frozen_classes": True,
        "owlv2_contract": OWLV2_COREML_CONTRACT,
        "owlv2_preprocess": OWLV2_COREML_PREPROCESS,
        "owlv2_postprocess": OWLV2_COREML_POSTPROCESS,
        "owlv2_prompt_template": OWLV2_COREML_PROMPT_TEMPLATE,
        "owlv2_vocabulary_sha256": owlv2_coreml_vocabulary_hash(labels),
        "owlv2_image_size": profile.image_size,
        "owlv2_patch_size": profile.patch_size,
        "owlv2_num_patches": profile.num_patches,
        "owlv2_num_classes": len(labels),
    }


def _strict_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "1"}:
        return True
    if token in {"false", "0"}:
        return False
    raise ValueError(f"OWLv2 Core ML metadata {key!r} must be true or false.")


def _strict_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"OWLv2 Core ML metadata {key!r} must be an integer.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise ValueError(f"OWLv2 Core ML metadata {key!r} must be an integer.")
    if parsed <= 0:
        raise ValueError(f"OWLv2 Core ML metadata {key!r} must be positive.")
    return parsed


def validate_owlv2_coreml_metadata(
    metadata: Mapping[str, Any],
    *,
    size: str,
    names: Mapping[int, str] | Sequence[str],
) -> Owlv2CoreMLProfile:
    """Pin metadata aliases to the class list the generic loader accepted."""
    expected = owlv2_coreml_metadata(size=size, names=names)
    missing = sorted(set(expected) - set(metadata))
    if missing:
        raise ValueError(
            f"OWLv2 Core ML artifact is missing metadata {missing!r}."
        )
    if not _strict_bool(metadata["frozen_classes"], key="frozen_classes"):
        raise ValueError(
            "OWLv2 Core ML artifacts must declare frozen_classes=true."
        )

    integer_keys = {
        "owlv2_image_size",
        "owlv2_patch_size",
        "owlv2_num_patches",
        "owlv2_num_classes",
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
                f"OWLv2 Core ML metadata {key!r} was modified: expected "
                f"{expected_value!r}, got {actual!r}."
            )
    return validate_owlv2_coreml_profile(size=size)


def _owlv2_dims_from_config(config: Any) -> Owlv2Dims:
    """Translate an OWLv2 config into LibreYOLO's native dimensions."""
    vision = getattr(config, "vision_config", None)
    text = getattr(config, "text_config", None)
    if vision is None or text is None:
        raise TypeError("OWLv2 Core ML export requires an OWLv2 model config.")
    vision_eps = float(vision.layer_norm_eps)
    text_eps = float(text.layer_norm_eps)
    if vision_eps != text_eps:
        raise NotImplementedError(
            "LibreYOLO's native OWLv2 port requires matching vision/text "
            f"LayerNorm eps values; got {vision_eps} and {text_eps}."
        )
    return Owlv2Dims(
        vision_hidden=int(vision.hidden_size),
        vision_layers=int(vision.num_hidden_layers),
        vision_heads=int(vision.num_attention_heads),
        vision_intermediate=int(vision.intermediate_size),
        patch_size=int(vision.patch_size),
        image_size=int(vision.image_size),
        vision_act=str(vision.hidden_act),
        text_hidden=int(text.hidden_size),
        text_layers=int(text.num_hidden_layers),
        text_heads=int(text.num_attention_heads),
        text_intermediate=int(text.intermediate_size),
        vocab_size=int(text.vocab_size),
        max_position_embeddings=int(text.max_position_embeddings),
        text_act=str(text.hidden_act),
        projection_dim=int(config.projection_dim),
        layer_norm_eps=vision_eps,
    )


def freeze_owlv2_text_embeddings(
    model: nn.Module,
    processor: Any,
    names: Mapping[int, str] | Sequence[str],
) -> torch.Tensor:
    """Compute one finite ``[1, C, D]`` query matrix for export."""
    labels = _ordered_names(names)
    prompts = [
        OWLV2_COREML_PROMPT_TEMPLATE.format(label.lower()) for label in labels
    ]
    if processor is None or not callable(processor):
        raise TypeError(
            "OWLv2 Core ML export requires the loaded processor/tokenizer to "
            "freeze class prompts."
        )
    encoded = processor(text=[prompts], return_tensors="pt")
    if not isinstance(encoded, Mapping):
        try:
            encoded = dict(encoded)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "OWLv2 processor returned an invalid text payload."
            ) from exc
    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    if not torch.is_tensor(input_ids) or not torch.is_tensor(attention_mask):
        raise RuntimeError(
            "OWLv2 processor must return input_ids and attention_mask tensors."
        )
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise RuntimeError(
            "OWLv2 frozen text inputs must have matching rank-two "
            "input_ids/attention_mask tensors."
        )
    if input_ids.shape[0] != len(labels):
        raise RuntimeError(
            "OWLv2 processor changed the frozen query count: expected "
            f"{len(labels)}, got {int(input_ids.shape[0])}."
        )

    owlv2 = getattr(model, "owlv2", None)
    text_model = getattr(owlv2, "text_model", None)
    text_projection = getattr(owlv2, "text_projection", None)
    if not isinstance(text_model, nn.Module) or not isinstance(
        text_projection, nn.Module
    ):
        raise TypeError(
            "OWLv2 Core ML export requires Owlv2ForObjectDetection-compatible "
            "text_model and text_projection modules."
        )
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    input_ids = input_ids.to(device=device, dtype=torch.long)
    attention_mask = attention_mask.to(device=device, dtype=torch.long)
    with torch.inference_mode():
        text_output = text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = getattr(text_output, "pooler_output", None)
        if pooled is None and isinstance(text_output, (tuple, list)):
            pooled = text_output[1] if len(text_output) > 1 else None
        if not torch.is_tensor(pooled):
            raise RuntimeError(
                "OWLv2 text tower did not return a pooled embedding tensor."
            )
        embeddings = text_projection(pooled)
        norms = torch.linalg.vector_norm(embeddings, ord=2, dim=-1, keepdim=True)
        if not bool(torch.isfinite(embeddings).all()) or not bool(
            torch.isfinite(norms).all()
        ):
            raise RuntimeError(
                "OWLv2 frozen class embeddings contain NaN or infinity."
            )
        if bool((norms <= 0).any()):
            raise RuntimeError("OWLv2 frozen class embeddings contain a zero vector.")
        embeddings = embeddings / norms
    return embeddings.detach().to(device="cpu", dtype=torch.float32).unsqueeze(0)


class Owlv2FrozenCoreMLAdapter(nn.Module):
    """Image-only OWLv2 detector with finite class embeddings as a buffer."""

    def __init__(
        self,
        native_model: Owlv2DetectionModel,
        query_embeddings: torch.Tensor,
    ) -> None:
        super().__init__()
        if not isinstance(native_model, Owlv2DetectionModel):
            raise TypeError(
                "OWLv2 frozen Core ML adapter requires Owlv2DetectionModel."
            )
        query = torch.as_tensor(query_embeddings, dtype=torch.float32)
        if query.ndim == 2:
            query = query.unsqueeze(0)
        if query.ndim != 3 or query.shape[0] != 1 or query.shape[1] <= 0:
            raise ValueError(
                "OWLv2 frozen query embeddings must have shape [1, C, D] "
                "with C > 0."
            )
        if query.shape[2] != native_model.dims.projection_dim:
            raise ValueError(
                "OWLv2 frozen query embedding width disagrees with the model: "
                f"{int(query.shape[2])} != {native_model.dims.projection_dim}."
            )
        if not bool(torch.isfinite(query).all()):
            raise ValueError(
                "OWLv2 frozen query embeddings contain NaN or infinity."
            )

        # Keep only modules used by the image-only graph.  The tokenizer and
        # text tower are absent from both the adapter state and the trace.
        self.vision_model = native_model.owlv2.vision_model
        self.layer_norm = native_model.layer_norm
        self.class_head = native_model.class_head
        self.box_head = native_model.box_head
        self.register_buffer(
            "box_bias",
            native_model.box_bias.detach().clone(),
            persistent=True,
        )
        self.register_buffer(
            "query_embeddings",
            query.detach().clone(),
            persistent=True,
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor(OWLV2_COREML_MEAN, dtype=torch.float32).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(OWLV2_COREML_STD, dtype=torch.float32).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = (image - self.pixel_mean) / self.pixel_std
        hidden = self.vision_model(image)
        image_embeddings = self.vision_model.post_layernorm(hidden)
        # Singleton-axis multiplication is the same broadcast as the native
        # torch.broadcast_to call, while Core ML Tools 9's TorchScript
        # frontend does not implement aten::broadcast_to.
        class_token = image_embeddings[:, :1, :]
        image_embeddings = image_embeddings[:, 1:, :] * class_token
        image_features = self.layer_norm(image_embeddings)

        image_class_embeddings = self.class_head.dense0(image_features)
        image_class_embeddings = image_class_embeddings / (
            torch.linalg.vector_norm(
                image_class_embeddings,
                ord=2,
                dim=-1,
                keepdim=True,
            )
            + 1e-6
        )
        query_embeddings = self.query_embeddings / (
            torch.linalg.vector_norm(
                self.query_embeddings,
                ord=2,
                dim=-1,
                keepdim=True,
            )
            + 1e-6
        )
        # The native class head spells this cosine product as a generic
        # ellipsis einsum. Core ML Tools 9 mis-lowers that equation's rank;
        # an explicit batched matrix multiplication is equivalent.
        logits = torch.matmul(
            image_class_embeddings,
            query_embeddings.transpose(-1, -2),
        )
        logit_shift = self.class_head.logit_shift(image_features)
        logit_scale = self.class_head.elu(
            self.class_head.logit_scale(image_features)
        ) + 1.0
        logits = (logits + logit_shift) * logit_scale
        boxes = torch.sigmoid(self.box_head(image_features) + self.box_bias)
        return logits.float(), boxes.float()


def build_owlv2_frozen_coreml_adapter(
    model: nn.Module,
    processor: Any,
    *,
    size: str,
    names: Mapping[int, str] | Sequence[str],
) -> Owlv2FrozenCoreMLAdapter:
    """Build a CPU image-only graph from the loaded OWLv2 detector."""
    profile = validate_owlv2_coreml_profile(size=size)
    dims = _owlv2_dims_from_config(getattr(model, "config", None))
    actual = (dims.image_size, dims.patch_size, dims.projection_dim)
    expected = (profile.image_size, profile.patch_size, profile.projection_dim)
    if actual != expected:
        raise RuntimeError(
            "Loaded OWLv2 config disagrees with the requested Core ML profile: "
            f"expected image/patch/projection={expected}, got {actual}."
        )

    query_embeddings = freeze_owlv2_text_embeddings(model, processor, names)
    native = Owlv2DetectionModel(dims).to(device="cpu", dtype=torch.float32).eval()
    state = {
        key: value.detach().to(device="cpu", dtype=(
            torch.float32 if value.is_floating_point() else value.dtype
        ))
        for key, value in model.state_dict().items()
    }
    incompatible = native.load_state_dict(state, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.endswith(("position_ids", "box_bias"))
    ]
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.endswith(("position_ids", "box_bias"))
    ]
    if missing or unexpected:
        raise RuntimeError(
            "OWLv2 Hugging Face/native state trees are incompatible: "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )
    return Owlv2FrozenCoreMLAdapter(native, query_embeddings).eval()


def prepare_owlv2_coreml_export(
    model: Any,
    kwargs: Mapping[str, Any],
    *,
    default_output: str = "owlv2_coreml.mlpackage",
) -> tuple[int, str, dict[str, Any], str, str]:
    """Validate the direct frozen-vocabulary export request.

    OWLv2 cannot use :class:`BaseExporter`'s ordinary model context: its live
    Hugging Face forward requires token tensors, while the deployment graph is
    built only after those tokens have been embedded.  This helper still
    delegates policy, precision, dependency, and metadata checks to the shared
    Core ML exporter before constructing the special graph.
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
    compute_units = options.pop("compute_units", "cpu_only")
    conf = options.pop("conf", 0.1)
    iou = options.pop("iou", 0.45)
    max_det = options.pop("max_det", 300)

    # Accepted by the common public signature but unused by this direct graph.
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
        raise TypeError(f"Unsupported OWLv2 Core ML export options: {names}")

    if half:
        raise NotImplementedError(
            "OWLv2 Core ML export is FP32-only. Core ML Tools 9 FP16 "
            "conversion completes, but real Apple runtime outputs diverge "
            "from the prepared graph; pass half=False."
        )
    if dynamic:
        raise NotImplementedError(
            "Frozen-vocabulary OWLv2 Core ML export uses a fixed image and "
            "class shape; dynamic=True is not supported."
        )
    if batch != 1:
        raise ValueError(
            "Frozen-vocabulary OWLv2 Core ML export requires batch=1; "
            f"got batch={batch}."
        )
    if nms:
        raise NotImplementedError(
            "OWLv2 does not run NMS. Core ML export preserves raw logits and "
            "boxes; nms=True is not applicable."
        )
    if device not in (None, "auto", "cpu", torch.device("cpu")):
        raise NotImplementedError(
            "Core ML conversion traces on CPU; pass device='cpu', "
            "device='auto', or omit device."
        )

    size = str(getattr(model, "size", "")).strip().lower()
    profile = validate_owlv2_coreml_profile(size=size)
    if imgsz is None:
        requested = (profile.image_size, profile.image_size)
    elif isinstance(imgsz, (tuple, list)):
        if len(imgsz) != 2:
            raise ValueError(f"imgsz must be an int or (height, width), got {imgsz}")
        requested = (int(imgsz[0]), int(imgsz[1]))
    else:
        requested = (int(imgsz), int(imgsz))
    validate_owlv2_coreml_profile(size=size, canvas_hw=requested)
    labels = _ordered_names(getattr(model, "names", {}))
    if int(getattr(model, "nb_classes", 0)) != len(labels):
        raise RuntimeError(
            "OWLv2 class metadata is inconsistent: nb_classes must match names."
        )
    from .coreml_profiles import resolve_coreml_export_compute_units

    compute_units, _ = resolve_coreml_export_compute_units(
        compute_units,
        family="owlv2",
        task="detect",
        size=size,
        canvas=requested,
        precision="fp32",
        nms=False,
        class_count=len(labels),
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
        owlv2_coreml_metadata(
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


def export_owlv2_coreml(
    model: Any,
    kwargs: Mapping[str, Any],
) -> str:
    """Export the current OWLv2 class vocabulary as an image-only package."""
    (
        image_size,
        output_path,
        metadata,
        precision,
        compute_units,
    ) = prepare_owlv2_coreml_export(model, kwargs)
    adapter = build_owlv2_frozen_coreml_adapter(
        model.model,
        model.processor,
        size=model.size,
        names=model.names,
    )
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)

    from .coreml import export_coreml

    return export_coreml(
        adapter,
        dummy,
        output_path=output_path,
        precision=precision,
        compute_units=compute_units,
        nms=False,
        metadata=metadata,
        model_family="owlv2",
        model_task="detect",
        model_size=model.size,
    )


def preprocess_owlv2_coreml_image(
    image: Image.Image | np.ndarray,
    *,
    image_size: int,
) -> torch.Tensor:
    """Return OWLv2's exact host tensor in RGB ``[0, 1]``.

    The result has shape ``[1, 3, image_size, image_size]``.  Padding is
    bottom/right, then the padded square is Gaussian-filtered only while
    downsampling and resized bilinearly without a second antialias pass.
    Normalization stays inside :class:`Owlv2FrozenCoreMLAdapter`.
    """
    if isinstance(image_size, bool) or int(image_size) <= 0:
        raise ValueError("OWLv2 Core ML image_size must be positive.")
    image_size = int(image_size)
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    else:
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] not in {3, 4}:
            raise ValueError(
                "OWLv2 Core ML preprocessing requires an HWC RGB/RGBA image."
            )
        if not np.issubdtype(rgb.dtype, np.number):
            raise TypeError("OWLv2 Core ML image values must be numeric.")
        if not np.isfinite(rgb).all():
            raise ValueError("OWLv2 Core ML image contains NaN or infinity.")
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        if np.issubdtype(rgb.dtype, np.floating) and float(rgb.max()) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    tensor = (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .div(255.0)
    )
    height, width = (int(value) for value in tensor.shape[-2:])
    square = max(height, width)
    if height != square or width != square:
        tensor = torch.nn.functional.pad(
            tensor,
            (0, square - width, 0, square - height),
            mode="constant",
            value=0.0,
        )

    factor = torch.tensor(square, dtype=torch.float32) / torch.tensor(
        image_size,
        dtype=torch.float32,
    )
    sigma_tensor = ((factor - 1.0) / 2.0).clamp(min=0.0)
    sigma = float(sigma_tensor)
    if sigma > 0.0:
        from torchvision.transforms.v2 import functional as tv_functional

        kernel = int(2 * torch.ceil(3 * sigma_tensor) + 1)
        tensor = tv_functional.gaussian_blur(
            tensor,
            [kernel, kernel],
            sigma=[sigma, sigma],
        )

    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as tv_functional

    tensor = tv_functional.resize(
        tensor,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=False,
    )
    if tensor.shape != (1, 3, image_size, image_size):
        raise RuntimeError(
            "OWLv2 Core ML preprocessing produced an invalid tensor shape "
            f"{tuple(tensor.shape)}."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(
            "OWLv2 Core ML preprocessing produced NaN or infinity."
        )
    return tensor.contiguous()


def postprocess_owlv2_coreml_outputs(
    pred_logits: torch.Tensor | np.ndarray,
    pred_boxes: torch.Tensor | np.ndarray,
    *,
    original_size: tuple[int, int],
    conf: float,
    max_det: int,
    classes: Sequence[int] | None = None,
) -> dict[str, torch.Tensor | int]:
    """Decode frozen OWLv2 outputs with its square-padded coordinate contract."""
    logits = torch.as_tensor(pred_logits, dtype=torch.float32)
    boxes = torch.as_tensor(pred_boxes, dtype=torch.float32)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] <= 0:
        raise ValueError(
            "OWLv2 Core ML pred_logits must have shape [1, P, C] with C > 0."
        )
    if boxes.shape != (1, logits.shape[1], 4):
        raise ValueError(
            "OWLv2 Core ML pred_boxes must have shape [1, P, 4] and share P."
        )
    if not bool(torch.isfinite(logits).all()) or not bool(
        torch.isfinite(boxes).all()
    ):
        raise RuntimeError("OWLv2 Core ML outputs contain NaN or infinity.")
    try:
        width, height = (int(value) for value in original_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "OWLv2 original_size must be a (width, height) pair."
        ) from exc
    if width <= 0 or height <= 0:
        raise ValueError("OWLv2 original image dimensions must be positive.")
    try:
        threshold = float(conf)
    except (TypeError, ValueError) as exc:
        raise ValueError("OWLv2 confidence threshold must be numeric.") from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("OWLv2 confidence threshold must be finite in [0, 1].")
    if isinstance(max_det, bool) or int(max_det) <= 0:
        raise ValueError("OWLv2 max_det must be a positive integer.")

    scores, class_ids = torch.sigmoid(logits[0]).max(dim=-1)
    center_x, center_y, box_width, box_height = boxes[0].unbind(dim=-1)
    decoded = torch.stack(
        (
            center_x - box_width / 2,
            center_y - box_height / 2,
            center_x + box_width / 2,
            center_y + box_height / 2,
        ),
        dim=-1,
    )
    decoded = decoded * float(max(width, height))

    keep = scores > threshold
    if classes is not None:
        if isinstance(classes, (str, bytes)):
            raise TypeError("OWLv2 classes filter must be a sequence of ids.")
        try:
            allowed = torch.as_tensor(
                [int(value) for value in classes],
                dtype=torch.int64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "OWLv2 classes filter must contain integer ids."
            ) from exc
        if allowed.numel() == 0:
            keep &= False
        else:
            if bool((allowed < 0).any()) or bool(
                (allowed >= logits.shape[2]).any()
            ):
                raise ValueError(
                    "OWLv2 classes filter contains an out-of-range class id."
                )
            keep &= (class_ids[:, None] == allowed[None, :]).any(dim=1)

    decoded = decoded[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    if decoded.numel():
        decoded[:, 0::2].clamp_(0.0, float(width))
        decoded[:, 1::2].clamp_(0.0, float(height))
        valid = (decoded[:, 2] > decoded[:, 0]) & (
            decoded[:, 3] > decoded[:, 1]
        )
        decoded = decoded[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
    if scores.numel():
        order = scores.argsort(descending=True)[: int(max_det)]
        decoded = decoded[order]
        scores = scores[order]
        class_ids = class_ids[order]
    return {
        "boxes": decoded.contiguous(),
        "scores": scores.contiguous(),
        "classes": class_ids.to(dtype=torch.int64).contiguous(),
        "num_detections": int(scores.numel()),
    }


__all__ = [
    "OWLV2_COREML_CONTRACT",
    "OWLV2_COREML_INPUT_NAME",
    "OWLV2_COREML_MEAN",
    "OWLV2_COREML_OUTPUT_NAMES",
    "OWLV2_COREML_POSTPROCESS",
    "OWLV2_COREML_PREPROCESS",
    "OWLV2_COREML_PROFILES",
    "OWLV2_COREML_PROMPT_TEMPLATE",
    "OWLV2_COREML_STD",
    "Owlv2CoreMLProfile",
    "Owlv2FrozenCoreMLAdapter",
    "build_owlv2_frozen_coreml_adapter",
    "expected_owlv2_coreml_shapes",
    "export_owlv2_coreml",
    "freeze_owlv2_text_embeddings",
    "owlv2_coreml_input_contract",
    "owlv2_coreml_metadata",
    "owlv2_coreml_output_contract",
    "owlv2_coreml_prompts",
    "owlv2_coreml_validation_contract",
    "owlv2_coreml_vocabulary_hash",
    "postprocess_owlv2_coreml_outputs",
    "prepare_owlv2_coreml_export",
    "preprocess_owlv2_coreml_image",
    "validate_owlv2_coreml_metadata",
    "validate_owlv2_coreml_profile",
]
