# Copyright 2026 EPFL Visual Intelligence and Learning Lab (VILAB) and the
# MODUS authors.
# SPDX-License-Identifier: Apache-2.0
#
# Registry values follow Modus
# conf/modalities/instruction_16mod_stage2.yaml at
# c299ef0fbba1cfe7c93336c45d7085afd770c0fa.

"""LibreMODUS modality registry, task routing, and public matrix."""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable

from .modality import CodeTokenGroup, ModalityRegistry, ModalitySpec

_DET_GROUPS = (
    CodeTokenGroup("<|x1_{i:03d}|>", 0, 999),
    CodeTokenGroup("<|y1_{i:03d}|>", 0, 999),
    CodeTokenGroup("<|x2_{i:03d}|>", 0, 999),
    CodeTokenGroup("<|y2_{i:03d}|>", 0, 999),
)

MODALITY_SPECS = (
    ModalitySpec(
        "text",
        0,
        "text",
        "bos_token_id",
        "eos_token_id",
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "caption",
        1,
        "text",
        "bos_token_id",
        "eos_token_id",
        represent_vae=True,
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "rgb",
        2,
        "image",
        "start_of_image",
        "end_of_image",
        represent_vit=True,
        represent_vae=True,
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "depth",
        3,
        "image",
        "start_of_depth",
        "end_of_image",
        start_token="<|depth_start|>",
        represent_vit=True,
        represent_vae=True,
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "normal",
        4,
        "image",
        "start_of_normal",
        "end_of_image",
        start_token="<|normal_start|>",
        represent_vit=True,
        represent_vae=True,
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "det",
        5,
        "codebook",
        "start_of_det",
        "end_of_det",
        start_token="<|det_start|>",
        end_token="<|det_end|>",
        extra_tokens=("<|box_start|>", "<|box_end|>"),
        code_token_groups=_DET_GROUPS + (CodeTokenGroup("<|score_{i:02d}|>", 0, 99),),
        represent_vae=True,
        inference_decode_method="detection",
        inference_max_tokens=1000,
        inference_cfg_uncond="text",
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "seg",
        6,
        "image",
        "start_of_seg",
        "end_of_image",
        start_token="<|seg_start|>",
        represent_vit=True,
        represent_vae=True,
    ),
    ModalitySpec(
        "canny",
        7,
        "image",
        "start_of_canny",
        "end_of_image",
        start_token="<|canny_start|>",
        represent_vit=True,
        represent_vae=True,
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "dino",
        8,
        "codebook",
        "start_of_dino",
        "end_of_dino",
        start_token="<|dino_start|>",
        end_token="<|dino_end|>",
        code_vocab_size=8192,
        code_token_format="<|dino_{i:04d}|>",
        pos_embed_size=16,
        apply_pos_embed_in_forward=True,
        represent_vae=True,
        inference_decode_method="dino",
        inference_max_tokens=17,
        inference_cfg_uncond="img",
        inference_cfg_img_scale=1.0,
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "dinolocal",
        9,
        "codebook",
        "start_of_dinolocal",
        "end_of_dinolocal",
        start_token="<|dinolocal_start|>",
        end_token="<|dinolocal_end|>",
        code_vocab_size=8192,
        code_token_format="<|dinolocal_{i:04d}|>",
        pos_embed_size=1024,
        apply_pos_embed_in_forward=True,
        represent_vae=True,
        inference_decode_method="dinolocal",
        inference_max_tokens=1025,
        inference_cfg_uncond="img",
        inference_cfg_img_scale=1.0,
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "clip",
        10,
        "codebook",
        "start_of_clip",
        "end_of_clip",
        start_token="<|clip_start|>",
        end_token="<|clip_end|>",
        code_vocab_size=8192,
        code_token_format="<|clip_{i:04d}|>",
        pos_embed_size=784,
        apply_pos_embed_in_forward=True,
        represent_vae=True,
        inference_decode_method="clip",
        inference_max_tokens=785,
        inference_cfg_uncond="img",
        inference_cfg_img_scale=1.0,
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "imagebind",
        11,
        "codebook",
        "start_of_imagebind",
        "end_of_imagebind",
        start_token="<|imagebind_start|>",
        end_token="<|imagebind_end|>",
        code_vocab_size=8192,
        code_token_format="<|imagebind_{i:04d}|>",
        pos_embed_size=16,
        apply_pos_embed_in_forward=True,
        represent_vae=True,
        inference_decode_method="imagebind",
        inference_max_tokens=17,
        inference_cfg_uncond="img",
        inference_cfg_img_scale=1.0,
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "imagebindlocal",
        12,
        "codebook",
        "start_of_imagebindlocal",
        "end_of_imagebindlocal",
        start_token="<|imagebindlocal_start|>",
        end_token="<|imagebindlocal_end|>",
        code_vocab_size=8192,
        code_token_format="<|imagebindlocal_{i:04d}|>",
        pos_embed_size=1024,
        apply_pos_embed_in_forward=True,
        represent_vae=True,
        inference_decode_method="imagebindlocal",
        inference_max_tokens=1025,
        inference_cfg_uncond="img",
        inference_cfg_img_scale=1.0,
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "cocodet",
        13,
        "codebook",
        "start_of_cocodet",
        "end_of_cocodet",
        start_token="<|cocodet_start|>",
        end_token="<|cocodet_end|>",
        code_token_groups=_DET_GROUPS
        + (CodeTokenGroup("<|coco_cls_{i:02d}|>", 0, 90),),
        dispersed_code_tokens=True,
        represent_vae=True,
        inference_decode_method="cocodet",
        inference_max_tokens=1000,
        inference_cfg_uncond="text",
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "samseg",
        14,
        "image",
        "start_of_samseg",
        "end_of_image",
        start_token="<|samseg_start|>",
        represent_vit=True,
        represent_vae=True,
        inference_add_instruction=False,
    ),
    ModalitySpec(
        "samedge",
        15,
        "image",
        "start_of_samedge",
        "end_of_image",
        start_token="<|samedge_start|>",
        represent_vit=True,
        represent_vae=True,
        inference_add_instruction=False,
    ),
)

BASE_MODALITY_REGISTRY = ModalityRegistry(MODALITY_SPECS)

INPUT_ALIASES = MappingProxyType(
    {
        "rgb": "rgb",
        "image": "rgb",
        "depth": "depth",
        "normal": "normal",
        "normals": "normal",
        "canny": "canny",
        "edge": "canny",
        "edges": "canny",
        "text": "text",
        "prompt": "text",
    }
)
TARGET_ALIASES = MappingProxyType(
    {
        "depth": "depth",
        "normal": "normal",
        "normals": "normal",
        "edge": "canny",
        "edges": "canny",
        "canny": "canny",
        "samedge": "samedge",
        "detect": "cocodet",
        "cocodet": "cocodet",
        "grounding": "det",
        "det": "det",
    }
)

PUBLIC_IMAGE_INPUTS = frozenset({"rgb", "depth", "normal", "canny"})
PUBLIC_TARGETS = ("depth", "normal", "canny", "samedge", "cocodet", "det")

# Stage 3 trains unrestricted 1..3-condition some-to-some examples. LibreYOLO
# exposes the safe analysis subset only: image-derived inputs and no RGB output.
SUPPORTED_MATRIX = MappingProxyType(
    {source: frozenset(PUBLIC_TARGETS) for source in sorted(PUBLIC_IMAGE_INPUTS)}
)

TASK_TO_TARGET = MappingProxyType(
    {"depth": "depth", "normal": "normal", "edge": "canny", "detect": "cocodet"}
)

GROUNDING_PROMPT = "[start grounding the phrase] {phrase}"


def normalize_input_modality(name: str) -> str:
    try:
        return INPUT_ALIASES[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported input modality {name!r}. Supported keys: "
            f"{', '.join(sorted(INPUT_ALIASES))}."
        ) from exc


def normalize_target(name: str) -> str:
    try:
        return TARGET_ALIASES[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported target {name!r}. Supported targets: "
            f"{', '.join(sorted(TARGET_ALIASES))}."
        ) from exc


def validate_any2any_request(
    input_modalities: Iterable[str], target: str
) -> tuple[tuple[str, ...], str]:
    """Normalize and validate the public 1..3 image-plus-optional-text contract."""
    normalized = tuple(normalize_input_modality(name) for name in input_modalities)
    if not 1 <= len(normalized) <= 4:
        raise ValueError("any2any() accepts 1..3 image modalities plus optional text.")
    image_inputs = tuple(name for name in normalized if name in PUBLIC_IMAGE_INPUTS)
    if not 1 <= len(image_inputs) <= 3:
        raise ValueError(
            "any2any() requires 1..3 image-derived inputs "
            "(rgb, depth, normal, or canny); text cannot be the sole input."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "any2any() input modalities must be unique after alias normalization."
        )
    normalized_target = normalize_target(target)
    unsupported = [
        source
        for source in image_inputs
        if normalized_target not in SUPPORTED_MATRIX[source]
    ]
    if unsupported:
        raise ValueError(
            f"Unsupported MODUS route {unsupported} -> {normalized_target}. "
            f"Supported targets: {', '.join(PUBLIC_TARGETS)}."
        )
    return normalized, normalized_target
