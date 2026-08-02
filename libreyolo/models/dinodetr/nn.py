# SPDX-License-Identifier: Apache-2.0
# Ported from https://github.com/IDEA-Research/DINO at
# d84a491d41898b3befd8294d1cf2614661fc0953.
# Copyright 2022 IDEA.
# Includes work derived from Conditional DETR, DETR, and Deformable DETR.
"""Native inference architecture for the released DINO-DETR checkpoints."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .common import NestedTensor, build_backbone, inverse_sigmoid
from .common import nested_tensor_from_tensor_list
from .transformer import DeformableTransformer, MLP


DINODETR_CONFIGS: dict[str, int] = {
    "r50": 4,
    "r50s5": 5,
    "swinl": 5,
}


class LibreDINODETRModel(nn.Module):
    """DINO with the exact shared heads and two-stage query initialization."""

    def __init__(self, size: str, nc: int = 91):
        super().__init__()
        if size not in DINODETR_CONFIGS:
            raise ValueError(
                f"Unknown DINO-DETR size {size!r}; expected one of "
                f"{', '.join(DINODETR_CONFIGS)}"
            )
        num_feature_levels = DINODETR_CONFIGS[size]
        backbone = build_backbone(size)
        transformer = DeformableTransformer(num_feature_levels)

        self.num_queries = 900
        self.num_select = 300
        self.transformer = transformer
        self.num_classes = nc
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = 8
        self.label_enc = nn.Embedding(nc + 1, self.hidden_dim)
        self.query_dim = 4
        self.random_refpoints_xy = False
        self.fix_refpoints_hw = -1
        self.num_patterns = 0
        self.dn_number = 100
        self.dn_box_noise_scale = 1.0
        self.dn_label_noise_ratio = 0.5
        self.dn_labelbook_size = nc

        input_projections = []
        in_channels = 0
        for in_channels in backbone.num_channels:
            input_projections.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )
            )
        for _ in range(num_feature_levels - len(backbone.num_channels)):
            input_projections.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(32, self.hidden_dim),
                )
            )
            in_channels = self.hidden_dim
        self.input_proj = nn.ModuleList(input_projections)
        self.backbone = backbone
        self.aux_loss = True
        self.box_pred_damping = None
        self.iter_update = True
        self.dec_pred_class_embed_share = True
        self.dec_pred_bbox_embed_share = True

        class_head = nn.Linear(self.hidden_dim, nc)
        box_head = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        prior_probability = 0.01
        class_head.bias.data = torch.ones(nc) * -math.log(
            (1 - prior_probability) / prior_probability
        )
        nn.init.constant_(box_head.layers[-1].weight.data, 0)
        nn.init.constant_(box_head.layers[-1].bias.data, 0)
        self.bbox_embed = nn.ModuleList(
            [box_head for _ in range(transformer.num_decoder_layers)]
        )
        self.class_embed = nn.ModuleList(
            [class_head for _ in range(transformer.num_decoder_layers)]
        )
        transformer.decoder.bbox_embed = self.bbox_embed
        transformer.decoder.class_embed = self.class_embed
        # Released configs train independent encoder proposal heads while the
        # six decoder layers share one class head and one box head.
        transformer.enc_out_bbox_embed = copy.deepcopy(box_head)
        transformer.enc_out_class_embed = copy.deepcopy(class_head)

        self.two_stage_type = "standard"
        self.two_stage_add_query_num = 0
        self.refpoint_embed = None
        self.decoder_sa_type = "sa"
        self.label_embedding = None
        for layer in transformer.decoder.layers:
            layer.label_embedding = None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for projection in self.input_proj:
            nn.init.xavier_uniform_(projection[0].weight, gain=1)
            nn.init.constant_(projection[0].bias, 0)

    def forward(
        self, samples: Tensor | NestedTensor | list[Tensor] | tuple[Tensor, ...]
    ) -> dict[str, Tensor | list[dict[str, Tensor]] | dict[str, Tensor] | None]:
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, positions = self.backbone(samples)
        srcs = []
        masks = []
        for level, feature in enumerate(features):
            src, mask = feature.decompose()
            if mask is None:
                raise ValueError("DINO-DETR feature is missing its padding mask")
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
                    raise ValueError("DINO-DETR input is missing its padding mask")
                mask = F.interpolate(
                    samples.mask[None].float(), size=src.shape[-2:]
                ).to(torch.bool)[0]
                position = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                positions.append(position)

        hidden_states, references, encoder_states, encoder_references, initial = (
            self.transformer(srcs, masks, None, positions, None, None)
        )
        hidden_states[0] += self.label_enc.weight[0, 0] * 0.0
        output_boxes = []
        for reference, box_head, hidden_state in zip(
            references[:-1], self.bbox_embed, hidden_states
        ):
            output_boxes.append(
                (box_head(hidden_state) + inverse_sigmoid(reference)).sigmoid()
            )
        stacked_boxes = torch.stack(output_boxes)
        stacked_classes = torch.stack(
            [
                class_head(hidden_state)
                for class_head, hidden_state in zip(self.class_embed, hidden_states)
            ]
        )
        output: dict = {
            "pred_logits": stacked_classes[-1],
            "pred_boxes": stacked_boxes[-1],
        }
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(stacked_classes, stacked_boxes)
        intermediate_classes = self.transformer.enc_out_class_embed(encoder_states[-1])
        output["interm_outputs"] = {
            "pred_logits": intermediate_classes,
            "pred_boxes": encoder_references[-1],
        }
        output["interm_outputs_for_matching_pre"] = {
            "pred_logits": intermediate_classes,
            "pred_boxes": initial,
        }
        output["dn_meta"] = None
        return output

    @torch.jit.unused
    def _set_aux_loss(self, classes: Tensor, boxes: Tensor) -> list[dict[str, Tensor]]:
        return [
            {"pred_logits": logits, "pred_boxes": coordinates}
            for logits, coordinates in zip(classes[:-1], boxes[:-1])
        ]


__all__ = ["DINODETR_CONFIGS", "LibreDINODETRModel"]
