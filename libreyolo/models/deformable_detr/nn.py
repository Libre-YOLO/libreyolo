# SPDX-License-Identifier: Apache-2.0
# Ported from https://github.com/fundamentalvision/Deformable-DETR
# commit 11169a60c33333af00a4849f1808023eba96a931.
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Modified from DETR (https://github.com/facebookresearch/detr).
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Native Deformable DETR inference architecture.

Module and attribute names intentionally mirror the pinned upstream source so
official checkpoints load strictly. Training-only matcher, criterion, and
segmentation code are outside this inference port. Multi-scale deformable
attention is provided by the pure-PyTorch implementation in
``ms_deform_attn.py``; no custom extension is loaded.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .common import NestedTensor, build_backbone, inverse_sigmoid
from .common import nested_tensor_from_tensor_list
from .transformer import DeformableTransformer


DEFORMABLE_DETR_CONFIGS: dict[str, dict[str, bool | int]] = {
    "r50ss": {
        "num_feature_levels": 1,
        "dilation": False,
        "with_box_refine": False,
        "two_stage": False,
    },
    "r50ssdc5": {
        "num_feature_levels": 1,
        "dilation": True,
        "with_box_refine": False,
        "two_stage": False,
    },
    "r50": {
        "num_feature_levels": 4,
        "dilation": False,
        "with_box_refine": False,
        "two_stage": False,
    },
    "r50refine": {
        "num_feature_levels": 4,
        "dilation": False,
        "with_box_refine": True,
        "two_stage": False,
    },
    "r50twostage": {
        "num_feature_levels": 4,
        "dilation": False,
        "with_box_refine": True,
        "two_stage": True,
    },
}


def _get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


class MLP(nn.Module):
    """Multi-layer perceptron used by each box-regression head."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip([input_dim, *hidden], [*hidden, output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class LibreDeformableDETRModel(nn.Module):
    """Original ResNet-50 Deformable DETR architecture."""

    def __init__(self, size: str, nc: int = 91):
        super().__init__()
        if size not in DEFORMABLE_DETR_CONFIGS:
            raise ValueError(
                f"Unknown Deformable DETR size {size!r}; expected one of "
                f"{', '.join(DEFORMABLE_DETR_CONFIGS)}"
            )
        config = DEFORMABLE_DETR_CONFIGS[size]
        num_feature_levels = int(config["num_feature_levels"])
        with_box_refine = bool(config["with_box_refine"])
        two_stage = bool(config["two_stage"])

        backbone = build_backbone(
            num_feature_levels=num_feature_levels,
            dilation=bool(config["dilation"]),
        )
        transformer = DeformableTransformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=num_feature_levels,
            dec_n_points=4,
            enc_n_points=4,
            two_stage=two_stage,
            two_stage_num_proposals=300,
        )

        self.num_queries = 300
        self.num_select = 300
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, nc)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(self.num_queries, hidden_dim * 2)

        if num_feature_levels > 1:
            num_backbone_outputs = len(backbone.strides)
            input_projections = []
            in_channels = 0
            for index in range(num_backbone_outputs):
                in_channels = backbone.num_channels[index]
                input_projections.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outputs):
                input_projections.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            hidden_dim,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_projections)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = True
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_probability = 0.01
        bias_value = -math.log((1 - prior_probability) / prior_probability)
        self.class_embed.bias.data = torch.ones(nc) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for projection in self.input_proj:
            nn.init.xavier_uniform_(projection[0].weight, gain=1)
            nn.init.constant_(projection[0].bias, 0)

        prediction_layers = (
            transformer.decoder.num_layers + 1
            if two_stage
            else transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, prediction_layers)
            self.bbox_embed = _get_clones(self.bbox_embed, prediction_layers)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(prediction_layers)]
            )
            self.bbox_embed = nn.ModuleList(
                [self.bbox_embed for _ in range(prediction_layers)]
            )
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(
        self, samples: Tensor | NestedTensor | list[Tensor] | tuple[Tensor, ...]
    ) -> dict[str, Tensor | list[dict[str, Tensor]] | dict[str, Tensor]]:
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, positions = self.backbone(samples)

        srcs = []
        masks = []
        for level, feature in enumerate(features):
            src, mask = feature.decompose()
            if mask is None:
                raise ValueError("Deformable DETR feature is missing its padding mask")
            srcs.append(self.input_proj[level](src))
            masks.append(mask)

        if self.num_feature_levels > len(srcs):
            backbone_levels = len(srcs)
            for level in range(backbone_levels, self.num_feature_levels):
                if level == backbone_levels:
                    src = self.input_proj[level](features[-1].tensors)
                else:
                    src = self.input_proj[level](srcs[-1])
                if samples.mask is None:
                    raise ValueError(
                        "Deformable DETR input is missing its padding mask"
                    )
                mask = F.interpolate(
                    samples.mask[None].float(), size=src.shape[-2:]
                ).to(torch.bool)[0]
                position = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                positions.append(position)

        query_embeds = None if self.two_stage else self.query_embed.weight
        (
            hidden_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(srcs, masks, positions, query_embeds)

        output_classes = []
        output_coords = []
        for level in range(hidden_states.shape[0]):
            reference = init_reference if level == 0 else inter_references[level - 1]
            reference = inverse_sigmoid(reference)
            output_class = self.class_embed[level](hidden_states[level])
            box_delta = self.bbox_embed[level](hidden_states[level])
            if reference.shape[-1] == 4:
                box_delta += reference
            else:
                if reference.shape[-1] != 2:
                    raise ValueError("Reference points must have width 2 or 4")
                box_delta[..., :2] += reference
            output_classes.append(output_class)
            output_coords.append(box_delta.sigmoid())

        stacked_classes = torch.stack(output_classes)
        stacked_coords = torch.stack(output_coords)
        output: dict = {
            "pred_logits": stacked_classes[-1],
            "pred_boxes": stacked_coords[-1],
        }
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(stacked_classes, stacked_coords)
        if self.two_stage:
            if enc_outputs_class is None or enc_outputs_coord_unact is None:
                raise RuntimeError("Two-stage transformer omitted encoder outputs")
            output["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord_unact.sigmoid(),
            }
        return output

    @torch.jit.unused
    def _set_aux_loss(
        self, output_classes: Tensor, output_coords: Tensor
    ) -> list[dict[str, Tensor]]:
        return [
            {"pred_logits": logits, "pred_boxes": boxes}
            for logits, boxes in zip(output_classes[:-1], output_coords[:-1])
        ]


class DeformableDETRExportWrapper(nn.Module):
    """Expose the two final detection tensors used by exported runtimes."""

    def __init__(self, model: LibreDeformableDETRModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        output = self.model(x)
        return output["pred_logits"], output["pred_boxes"]


__all__ = [
    "DEFORMABLE_DETR_CONFIGS",
    "DeformableDETRExportWrapper",
    "LibreDeformableDETRModel",
    "MLP",
]
