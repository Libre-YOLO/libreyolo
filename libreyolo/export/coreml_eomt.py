"""EoMT's compact query-output Core ML component contract.

The native Hugging Face EoMT eval graph mutates a boolean attention-mask slice
in place.  Core ML Tools 9 cannot lower that operation for the iOS 15 ML
Program opset.  :class:`EoMTCoreMLAdapter` expresses the same deterministic
eval computation functionally, using concatenation to construct the mask.

This module's eval-forward structure is adapted from Hugging Face Transformers
v5.12.1 (Apache-2.0).  Exact provenance is recorded in
``libreyolo/models/eomt/NOTICE``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

EOMT_COREML_CONTRACT = "eomt_raw_queries_v1"
EOMT_COREML_PATCH_SIZE = 16
EOMT_COREML_MASK_STRIDE = 4
EOMT_COREML_NUM_UPSCALE_BLOCKS = 2
EOMT_COREML_ALIGN_CORNERS = False
EOMT_COREML_ANTIALIAS = True
EOMT_COREML_ATTENTION_MASK = "functional_concat_v1"
EOMT_COREML_PREPROCESS = {
    "semantic": "semantic_shortest_edge_split_v1",
    "segment": "coco_longest_edge_pad_top_left_v1",
    "panoptic": "coco_longest_edge_pad_top_left_v1",
}
EOMT_COREML_POSTPROCESS = {
    "semantic": "semantic_stitch_v1",
    "segment": "instance_queries_v1",
    "panoptic": "panoptic_merge_v1",
}
EOMT_COREML_ARTIFACT_SCOPE = {
    "semantic": "patch_component",
    "segment": "full_image_component",
    "panoptic": "full_image_component",
}
EOMT_COREML_SUPPORTED_SIZES = frozenset({"s", "b", "l"})
EOMT_COREML_SUPPORTED_TASKS = frozenset(EOMT_COREML_PREPROCESS)


def _unwrap_eomt_net(model: nn.Module) -> nn.Module:
    """Find the :class:`LibreEoMTNet` inside generic export wrappers."""
    current: Any = model
    visited: set[int] = set()
    for _ in range(12):
        if all(
            hasattr(current, attribute)
            for attribute in ("eomt", "pixel_mean", "pixel_std", "nb_classes")
        ):
            return current
        marker = id(current)
        if marker in visited:
            break
        visited.add(marker)
        nested = getattr(current, "model", None)
        if not isinstance(nested, nn.Module):
            break
        current = nested
    raise TypeError(
        "EoMT Core ML export could not find LibreEoMTNet inside the prepared "
        f"{type(model).__name__} graph."
    )


def _validate_attention_mask_invariants(net: nn.Module) -> None:
    """Require EoMT's deterministic eval-time attention-mask configuration."""
    eomt = net.eomt
    probabilities = getattr(eomt, "attn_mask_probs", None)
    if not torch.is_tensor(probabilities):
        raise RuntimeError(
            "EoMT Core ML export requires the attn_mask_probs buffer."
        )
    expected_blocks = int(eomt.config.num_blocks)
    if int(probabilities.numel()) != expected_blocks:
        raise RuntimeError(
            "EoMT Core ML attention-mask probability count disagrees with "
            f"num_blocks: {int(probabilities.numel())} != {expected_blocks}."
        )
    if not bool(torch.isfinite(probabilities).all()) or not bool(
        torch.equal(probabilities.detach().cpu(), torch.ones(expected_blocks))
    ):
        raise RuntimeError(
            "EoMT Core ML export requires every attn_mask_probs value to equal "
            "1. Values below one randomly disable query masks and cannot form "
            "a deterministic deployment graph."
        )


class EoMTCoreMLAdapter(nn.Module):
    """Return compact raw EoMT query tensors without an in-place bool update."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = _unwrap_eomt_net(model)
        _validate_attention_mask_invariants(self.model)

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = (image - self.model.pixel_mean) / self.model.pixel_std
        eomt = self.model.eomt
        hidden_states = eomt.embeddings(image)
        attention_mask = None

        first_query_layer = eomt.num_hidden_layers - eomt.config.num_blocks
        for index, layer_module in enumerate(eomt.layers):
            if index == first_query_layer:
                query = eomt.query.weight[None, :, :].expand(
                    hidden_states.shape[0],
                    -1,
                    -1,
                )
                hidden_states = torch.cat((query, hidden_states), dim=1)

            if index >= first_query_layer:
                normalized = eomt.layernorm(hidden_states)
                intermediate_masks, _ = eomt.predict(normalized)
                interpolated = F.interpolate(
                    intermediate_masks,
                    size=eomt.grid_size,
                    mode="bilinear",
                ).flatten(2)

                num_queries = eomt.config.num_queries
                encoder_start = num_queries + eomt.embeddings.num_prefix_tokens
                batch, sequence, _ = hidden_states.shape

                # This is exactly the native all-ones attention mask followed
                # by:
                #   mask[:, :Q, encoder_start:] = interpolated > 0
                # expressed without tensor assignment so Core ML Tools can
                # lower the graph at the iOS 15 opset.
                query_rows = torch.cat(
                    (
                        torch.ones(
                            batch,
                            num_queries,
                            encoder_start,
                            dtype=torch.bool,
                            device=hidden_states.device,
                        ),
                        interpolated > 0,
                    ),
                    dim=2,
                )
                encoder_rows = torch.ones(
                    batch,
                    sequence - num_queries,
                    sequence,
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                attention_mask = torch.cat((query_rows, encoder_rows), dim=1)
                attention_mask = attention_mask[:, None].expand(
                    -1,
                    eomt.config.num_attention_heads,
                    -1,
                    -1,
                )
                attention_mask = attention_mask.float().masked_fill(
                    ~attention_mask,
                    -1e9,
                )

            hidden_states = layer_module(hidden_states, attention_mask)

        sequence_output = eomt.layernorm(hidden_states)
        masks_queries_logits, class_queries_logits = eomt.predict(sequence_output)
        return class_queries_logits.float(), masks_queries_logits.float()


def wrap_eomt_coreml_contract(model: nn.Module) -> nn.Module:
    """Return EoMT's deterministic compact-output export graph."""
    return EoMTCoreMLAdapter(model).eval()


def validate_eomt_coreml_profile(
    model: nn.Module,
    *,
    task: str,
    size: str | None,
    canvas_hw: tuple[int, int],
) -> nn.Module:
    """Validate a fixed EoMT graph and return its unwrapped network."""
    if task not in EOMT_COREML_SUPPORTED_TASKS:
        raise NotImplementedError(
            f"EoMT Core ML task must be one of "
            f"{sorted(EOMT_COREML_SUPPORTED_TASKS)}; got {task!r}."
        )
    if size not in EOMT_COREML_SUPPORTED_SIZES:
        raise NotImplementedError(
            "EoMT Core ML export supports DINOv2 sizes s/b/l; "
            f"got size={size!r}."
        )
    height, width = (int(value) for value in canvas_hw)
    if height <= 0 or width <= 0 or height != width:
        raise NotImplementedError(
            "EoMT Core ML export requires one positive square component "
            f"canvas; got {height}x{width}."
        )
    net = _unwrap_eomt_net(model)
    expected_size = int(getattr(net, "image_size", 0) or 0)
    if expected_size != height:
        raise NotImplementedError(
            "EoMT position embeddings are fixed to the checkpoint image size: "
            f"graph={expected_size}, requested={height}."
        )
    patch_size = int(getattr(net, "patch_size", 0) or 0)
    if patch_size != EOMT_COREML_PATCH_SIZE or height % patch_size:
        raise NotImplementedError(
            "EoMT Core ML requires patch_size=16 and an exactly divisible "
            f"canvas; got patch_size={patch_size}, canvas={height}."
        )
    _validate_attention_mask_invariants(net)
    return net


def eomt_coreml_input_contract(task: str) -> dict[str, Any]:
    """Describe one host-prepared EoMT patch or padded canvas."""
    if task not in EOMT_COREML_SUPPORTED_TASKS:
        raise ValueError(f"Unsupported EoMT Core ML task {task!r}.")
    return {
        "name": "image",
        "kind": "tensor",
        "layout": "NCHW",
        "color": "rgb",
        "range": "0_1",
        "geometry": (
            "eomt_split" if task == "semantic" else "eomt_pad_top_left"
        ),
        "interpolation": "bilinear",
        "resize_backend": "torchvision",
        "resize_rounding": "floor",
        "pad_value": 0,
    }


def eomt_coreml_validation_contract(task: str) -> dict[str, str]:
    """Declare the source domain before EoMT's dedicated host geometry."""
    if task not in EOMT_COREML_SUPPORTED_TASKS:
        raise ValueError(f"Unsupported EoMT Core ML task {task!r}.")
    return {
        "color": "rgb",
        "range": "0_1" if task == "semantic" else "0_255",
    }


def eomt_coreml_output_contract() -> list[dict[str, Any]]:
    """Return the exact compact query ABI in graph order."""
    return [
        {
            "name": "class_queries_logits",
            "role": "class_queries_logits",
            "encoding": "raw_logits_with_no_object",
            "rank": 3,
        },
        {
            "name": "masks_queries_logits",
            "role": "masks_queries_logits",
            "encoding": "stride4_raw_logits",
            "rank": 4,
        },
    ]


def expected_eomt_coreml_shapes(
    *,
    nc: int,
    num_queries: int,
    canvas_hw: tuple[int, int],
) -> dict[str, tuple[int, ...]]:
    """Return exact schema-v2 shapes for the compact query outputs."""
    height, width = (int(value) for value in canvas_hw)
    return {
        "class_queries_logits": (1, int(num_queries), int(nc) + 1),
        "masks_queries_logits": (
            1,
            int(num_queries),
            height // EOMT_COREML_MASK_STRIDE,
            width // EOMT_COREML_MASK_STRIDE,
        ),
    }


def eomt_coreml_metadata(
    *,
    task: str,
    num_queries: int,
    image_size: int,
) -> dict[str, Any]:
    """Return task-specific orchestration metadata for the strict loader."""
    if task not in EOMT_COREML_SUPPORTED_TASKS:
        raise ValueError(f"Unsupported EoMT Core ML task {task!r}.")
    return {
        "artifact_scope": EOMT_COREML_ARTIFACT_SCOPE[task],
        "eomt_contract": EOMT_COREML_CONTRACT,
        "eomt_preprocess": EOMT_COREML_PREPROCESS[task],
        "eomt_postprocess": EOMT_COREML_POSTPROCESS[task],
        "eomt_num_queries": int(num_queries),
        "eomt_image_size": int(image_size),
        "eomt_patch_size": EOMT_COREML_PATCH_SIZE,
        "eomt_mask_stride": EOMT_COREML_MASK_STRIDE,
        "eomt_num_upscale_blocks": EOMT_COREML_NUM_UPSCALE_BLOCKS,
        "eomt_mask_align_corners": EOMT_COREML_ALIGN_CORNERS,
        "eomt_antialias": EOMT_COREML_ANTIALIAS,
        "eomt_attention_mask": EOMT_COREML_ATTENTION_MASK,
    }


def reconstruct_eomt_full_outputs(
    class_queries_logits: torch.Tensor,
    masks_queries_logits: torch.Tensor,
    *,
    nc: int,
    canvas_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Rebuild the native full-canvas masks and semantic logits exactly."""
    height, width = (int(value) for value in canvas_hw)
    masks_full = F.interpolate(
        masks_queries_logits.float(),
        size=(height, width),
        mode="bilinear",
        align_corners=EOMT_COREML_ALIGN_CORNERS,
    )
    classes = class_queries_logits.float()
    semantic = torch.einsum(
        "bqc,bqhw->bchw",
        classes.softmax(dim=-1)[..., : int(nc)],
        masks_full.sigmoid(),
    )
    return {
        "semantic_logits": semantic,
        "class_queries_logits": classes,
        "masks_queries_logits": masks_full,
    }


__all__ = [
    "EOMT_COREML_ALIGN_CORNERS",
    "EOMT_COREML_ANTIALIAS",
    "EOMT_COREML_ARTIFACT_SCOPE",
    "EOMT_COREML_ATTENTION_MASK",
    "EOMT_COREML_CONTRACT",
    "EOMT_COREML_MASK_STRIDE",
    "EOMT_COREML_NUM_UPSCALE_BLOCKS",
    "EOMT_COREML_PATCH_SIZE",
    "EOMT_COREML_POSTPROCESS",
    "EOMT_COREML_PREPROCESS",
    "EoMTCoreMLAdapter",
    "eomt_coreml_input_contract",
    "eomt_coreml_metadata",
    "eomt_coreml_output_contract",
    "eomt_coreml_validation_contract",
    "expected_eomt_coreml_shapes",
    "reconstruct_eomt_full_outputs",
    "validate_eomt_coreml_profile",
    "wrap_eomt_coreml_contract",
]
