"""Native Faster R-CNN inference architecture.

This file is derived from torchvision v0.26.0 at commit
``336d36e8db990a905498c73933e35231876e28bc`` under the BSD-3-Clause
license. See ``docs/provenance/faster_rcnn.md`` and the repository notice
files. Copyright (c) Soumith Chintala 2016 and the torchvision contributors.

Faster R-CNN established the two-stage RPN + RoI-head detector pattern. The
four models here are modernized torchvision variants, not the original
paper's VGG16 architecture. Training-only matching, sampling, and loss code is
intentionally excluded; the complete inference graph is implemented locally.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Optional

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large, resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import Conv2dNormActivation, MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.ops.feature_pyramid_network import (
    ExtraFPNBlock,
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)
from torchvision.ops.misc import FrozenBatchNorm2d

__all__ = ["FASTER_RCNN_CONFIGS", "LibreFasterRCNNModel"]


FASTER_RCNN_CONFIGS = {
    "n": {
        "backbone": "mobilenet_v3_large",
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 150,
        "rpn_post_nms_top_n_test": 150,
        "rpn_score_thresh": 0.05,
    },
    "s": {
        "backbone": "mobilenet_v3_large",
        "min_size": 800,
        "max_size": 1333,
        "rpn_pre_nms_top_n_test": 1000,
        "rpn_post_nms_top_n_test": 1000,
        "rpn_score_thresh": 0.05,
    },
    "m": {
        "backbone": "resnet50_fpn",
        "min_size": 800,
        "max_size": 1333,
        "rpn_pre_nms_top_n_test": 1000,
        "rpn_post_nms_top_n_test": 1000,
        "rpn_score_thresh": 0.0,
    },
    "l": {
        "backbone": "resnet50_fpn_v2",
        "min_size": 800,
        "max_size": 1333,
        "rpn_pre_nms_top_n_test": 1000,
        "rpn_post_nms_top_n_test": 1000,
        "rpn_score_thresh": 0.0,
    },
}


class BackboneWithFPN(nn.Module):
    """Expose selected backbone stages through a Feature Pyramid Network."""

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: dict[str, str],
        in_channels_list: list[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list,
            out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return self.fpn(self.body(x))


def _resnet_fpn_backbone(*, v2: bool) -> BackboneWithFPN:
    norm_layer = nn.BatchNorm2d if v2 else FrozenBatchNorm2d
    backbone = resnet50(weights=None, norm_layer=norm_layer)
    return_layers = {f"layer{index}": str(index - 1) for index in range(1, 5)}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2**index for index in range(4)]
    return BackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        256,
        norm_layer=nn.BatchNorm2d if v2 else None,
    )


def _mobilenet_fpn_backbone() -> BackboneWithFPN:
    backbone = mobilenet_v3_large(weights=None, norm_layer=FrozenBatchNorm2d).features
    stage_indices = [0]
    stage_indices.extend(
        index for index, block in enumerate(backbone) if getattr(block, "_is_cn", False)
    )
    stage_indices.append(len(backbone) - 1)
    returned_layers = [len(stage_indices) - 2, len(stage_indices) - 1]
    return_layers = {
        str(stage_indices[stage]): str(output_index)
        for output_index, stage in enumerate(returned_layers)
    }
    in_channels_list = [
        backbone[stage_indices[stage]].out_channels for stage in returned_layers
    ]
    return BackboneWithFPN(backbone, return_layers, in_channels_list, 256)


class BoxCoder:
    """Decode the box-delta representation used by the RPN and RoI head."""

    def __init__(
        self,
        weights: tuple[float, float, float, float],
        bbox_xform_clip: float = math.log(1000.0 / 16),
    ) -> None:
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def decode(self, rel_codes: Tensor, boxes: list[Tensor]) -> Tensor:
        boxes_per_image = [box.size(0) for box in boxes]
        concatenated = torch.cat(boxes, dim=0)
        box_sum = sum(boxes_per_image)
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        decoded = self.decode_single(rel_codes, concatenated)
        if box_sum > 0:
            decoded = decoded.reshape(box_sum, -1, 4)
        return decoded

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        boxes = boxes.to(rel_codes.dtype)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        center_x = boxes[:, 0] + 0.5 * widths
        center_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = torch.clamp(rel_codes[:, 2::4] / ww, max=self.bbox_xform_clip)
        dh = torch.clamp(rel_codes[:, 3::4] / wh, max=self.bbox_xform_clip)

        pred_center_x = dx * widths[:, None] + center_x[:, None]
        pred_center_y = dy * heights[:, None] + center_y[:, None]
        pred_width = torch.exp(dw) * widths[:, None]
        pred_height = torch.exp(dh) * heights[:, None]
        half_width = (
            torch.tensor(0.5, dtype=pred_center_x.dtype, device=pred_width.device)
            * pred_width
        )
        half_height = (
            torch.tensor(0.5, dtype=pred_center_y.dtype, device=pred_height.device)
            * pred_height
        )
        return torch.stack(
            (
                pred_center_x - half_width,
                pred_center_y - half_height,
                pred_center_x + half_width,
                pred_center_y + half_height,
            ),
            dim=2,
        ).flatten(1)


class RPNHead(nn.Module):
    """Convolutional objectness and proposal-regression head."""

    _version = 2

    def __init__(self, in_channels: int, num_anchors: int, conv_depth: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            *[
                Conv2dNormActivation(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    norm_layer=None,
                )
                for _ in range(conv_depth)
            ]
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        version = local_metadata.get("version")
        if version is None or version < 2:
            for parameter in ("weight", "bias"):
                old_key = f"{prefix}conv.{parameter}"
                new_key = f"{prefix}conv.0.0.{parameter}"
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, features: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        logits: list[Tensor] = []
        bbox_regression: list[Tensor] = []
        for feature in features:
            hidden = self.conv(feature)
            logits.append(self.cls_logits(hidden))
            bbox_regression.append(self.bbox_pred(hidden))
        return logits, bbox_regression


def _permute_and_flatten(
    layer: Tensor, n: int, anchors: int, channels: int, height: int, width: int
) -> Tensor:
    layer = layer.view(n, -1, channels, height, width)
    layer = layer.permute(0, 3, 4, 1, 2)
    return layer.reshape(n, -1, channels)


def _concat_box_prediction_layers(
    box_cls: list[Tensor], box_regression: list[Tensor]
) -> tuple[Tensor, Tensor]:
    cls_flattened: list[Tensor] = []
    box_flattened: list[Tensor] = []
    for cls_level, box_level in zip(box_cls, box_regression):
        n, axc, height, width = cls_level.shape
        anchors = box_level.shape[1] // 4
        channels = axc // anchors
        cls_flattened.append(
            _permute_and_flatten(
                cls_level, n, anchors, channels, height, width
            )
        )
        box_flattened.append(
            _permute_and_flatten(box_level, n, anchors, 4, height, width)
        )
    flattened_cls = torch.cat(cls_flattened, dim=1).flatten(0, -2)
    flattened_box = torch.cat(box_flattened, dim=1).reshape(-1, 4)
    return flattened_cls, flattened_box


def _topk_min(input_tensor: Tensor, requested: int, axis: int):
    if not torch.jit.is_tracing():
        return min(requested, input_tensor.size(axis))
    axis_size = torch._shape_as_tensor(input_tensor)[axis].unsqueeze(0)
    return torch.min(
        torch.cat((torch.tensor([requested], dtype=axis_size.dtype), axis_size), 0)
    )


class RegionProposalNetwork(nn.Module):
    """Inference half of Faster R-CNN's Region Proposal Network."""

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: RPNHead,
        pre_nms_top_n: dict[str, int],
        post_nms_top_n: dict[str, int],
        nms_thresh: float,
        score_thresh: float,
    ) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder((1.0, 1.0, 1.0, 1.0))
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self) -> int:
        mode = "training" if self.training else "testing"
        return self._pre_nms_top_n[mode]

    def post_nms_top_n(self) -> int:
        mode = "training" if self.training else "testing"
        return self._post_nms_top_n[mode]

    def _get_top_n_idx(
        self, objectness: Tensor, num_anchors_per_level: list[int]
    ) -> Tensor:
        selected: list[Tensor] = []
        offset = 0
        for level_scores in objectness.split(num_anchors_per_level, 1):
            count = level_scores.shape[1]
            top_n = _topk_min(level_scores, self.pre_nms_top_n(), 1)
            _, indices = level_scores.topk(top_n, dim=1)
            selected.append(indices + offset)
            offset += count
        return torch.cat(selected, dim=1)

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: list[tuple[int, int]],
        num_anchors_per_level: list[int],
    ) -> tuple[list[Tensor], list[Tensor]]:
        num_images = proposals.shape[0]
        objectness = objectness.detach().reshape(num_images, -1)
        levels = torch.cat(
            [
                torch.full(
                    (count,), index, dtype=torch.int64, device=proposals.device
                )
                for index, count in enumerate(num_anchors_per_level)
            ],
            dim=0,
        )
        levels = levels.reshape(1, -1).expand_as(objectness)
        top_n_indices = self._get_top_n_idx(objectness, num_anchors_per_level)
        batch_indices = torch.arange(num_images, device=proposals.device)[:, None]
        objectness = objectness[batch_indices, top_n_indices]
        levels = levels[batch_indices, top_n_indices]
        proposals = proposals[batch_indices, top_n_indices]
        objectness = torch.sigmoid(objectness)

        final_boxes: list[Tensor] = []
        final_scores: list[Tensor] = []
        for boxes, scores, level, image_shape in zip(
            proposals, objectness, levels, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, level = boxes[keep], scores[keep], level[keep]
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, level = boxes[keep], scores[keep], level[keep]
            keep = box_ops.batched_nms(boxes, scores, level, self.nms_thresh)
            keep = keep[: self.post_nms_top_n()]
            final_boxes.append(boxes[keep])
            final_scores.append(scores[keep])
        return final_boxes, final_scores

    def forward(
        self, images: ImageList, features: dict[str, Tensor]
    ) -> tuple[list[Tensor], tuple[list[Tensor], list[Tensor]]]:
        feature_list = list(features.values())
        objectness_levels, bbox_levels = self.head(feature_list)
        anchors = self.anchor_generator(images, feature_list)
        num_images = len(anchors)
        per_level = [
            level[0].shape[0] * level[0].shape[1] * level[0].shape[2]
            for level in objectness_levels
        ]
        objectness, bbox_deltas = _concat_box_prediction_layers(
            objectness_levels, bbox_levels
        )
        proposals = self.box_coder.decode(bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, _ = self.filter_proposals(
            proposals, objectness, images.image_sizes, per_level
        )
        return boxes, (objectness_levels, bbox_levels)


class TwoMLPHead(nn.Module):
    """The v1 FPN variant's pair of fully connected RoI layers."""

    def __init__(self, in_channels: int, representation_size: int) -> None:
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        return F.relu(self.fc7(x))


class FastRCNNConvFCHead(nn.Sequential):
    """The v2 recipe's four convolutional and one fully connected RoI layers."""

    def __init__(
        self,
        input_size: tuple[int, int, int],
        conv_layers: list[int],
        fc_layers: list[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        in_channels, in_height, in_width = input_size
        blocks: list[nn.Module] = []
        previous_channels = in_channels
        for current_channels in conv_layers:
            blocks.append(
                Conv2dNormActivation(
                    previous_channels, current_channels, norm_layer=norm_layer
                )
            )
            previous_channels = current_channels
        blocks.append(nn.Flatten())
        previous_channels *= in_height * in_width
        for current_channels in fc_layers:
            blocks.extend(
                [
                    nn.Linear(previous_channels, current_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            previous_channels = current_channels
        super().__init__(*blocks)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class FastRCNNPredictor(nn.Module):
    """Per-RoI classification and class-specific box regression."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                "Fast R-CNN predictor expects spatial dimensions [1, 1]",
            )
        x = x.flatten(start_dim=1)
        return self.cls_score(x), self.bbox_pred(x)


class RoIHeads(nn.Module):
    """Inference-only RoI pooling, classification, regression, and NMS."""

    def __init__(
        self,
        box_roi_pool: MultiScaleRoIAlign,
        box_head: nn.Module,
        box_predictor: FastRCNNPredictor,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 100,
        bbox_reg_weights: Optional[tuple[float, float, float, float]] = None,
    ) -> None:
        super().__init__()
        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = BoxCoder(bbox_reg_weights)
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def postprocess_detections(
        self,
        class_logits: Tensor,
        box_regression: Tensor,
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        num_classes = class_logits.shape[-1]
        boxes_per_image = [proposal.shape[0] for proposal in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, dim=-1)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes: list[Tensor] = []
        all_scores: list[Tensor] = []
        all_labels: list[Tensor] = []
        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=class_logits.device)
            labels = labels.view(1, -1).expand_as(scores)
            boxes = boxes[:, 1:].reshape(-1, 4)
            scores = scores[:, 1:].reshape(-1)
            labels = labels[:, 1:].reshape(-1)
            keep = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]
            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])
            all_labels.append(labels[keep])
        return all_boxes, all_scores, all_labels

    def forward(
        self,
        features: dict[str, Tensor],
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> tuple[list[dict[str, Tensor]], tuple[Tensor, Tensor]]:
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        boxes, scores, labels = self.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )
        detections = [
            {"boxes": box, "labels": label, "scores": score}
            for box, label, score in zip(boxes, labels, scores)
        ]
        return detections, (class_logits, box_regression)


@torch.jit.unused
def _get_shape_onnx(image: Tensor) -> Tensor:
    from torch.onnx import operators

    return operators.shape_as_tensor(image)[-2:]


def _resize_image(image: Tensor, min_size: int, max_size: int) -> Tensor:
    if torchvision._is_tracing():
        image_shape = _get_shape_onnx(image)
        shortest = torch.min(image_shape).to(dtype=torch.float32)
        longest = torch.max(image_shape).to(dtype=torch.float32)
        scale = torch.min(
            torch.tensor(float(min_size), device=image.device) / shortest,
            torch.tensor(float(max_size), device=image.device) / longest,
        )
        scale_factor = scale
    else:
        image_shape = image.shape[-2:]
        scale_factor = min(min_size / min(image_shape), max_size / max(image_shape))
    return F.interpolate(
        image[None],
        scale_factor=scale_factor,
        mode="bilinear",
        recompute_scale_factor=True,
        align_corners=False,
    )[0]


def _resize_boxes(
    boxes: Tensor,
    original_size: tuple[int, int],
    new_size: tuple[int, int],
) -> Tensor:
    ratios = [
        torch.tensor(new, dtype=torch.float32, device=boxes.device)
        / torch.tensor(original, dtype=torch.float32, device=boxes.device)
        for new, original in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack(
        (
            xmin * ratio_width,
            ymin * ratio_height,
            xmax * ratio_width,
            ymax * ratio_height,
        ),
        dim=1,
    )


class GeneralizedRCNNTransform(nn.Module):
    """ImageNet-normalize, aspect-resize, pad, then restore box coordinates."""

    def __init__(
        self,
        min_size: int,
        max_size: int,
        image_mean: list[float],
        image_std: list[float],
        size_divisible: int = 32,
    ) -> None:
        super().__init__()
        self.min_size = (min_size,)
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisible = size_divisible

    def normalize(self, image: Tensor) -> Tensor:
        if not image.is_floating_point():
            raise TypeError(
                "Faster R-CNN expects floating point images in the [0, 1] range"
            )
        mean = torch.as_tensor(
            self.image_mean, dtype=image.dtype, device=image.device
        )
        std = torch.as_tensor(self.image_std, dtype=image.dtype, device=image.device)
        return (image - mean[:, None, None]) / std[:, None, None]

    @staticmethod
    def max_by_axis(sizes: list[list[int]]) -> list[int]:
        maximum = sizes[0]
        for size in sizes[1:]:
            for index, value in enumerate(size):
                maximum[index] = max(maximum[index], value)
        return maximum

    @torch.jit.unused
    def _onnx_batch_images(self, images: list[Tensor]) -> Tensor:
        maximum: list[Tensor] = []
        for dimension in range(images[0].dim()):
            maximum.append(
                torch.max(
                    torch.stack([image.shape[dimension] for image in images]).to(
                        torch.float32
                    )
                ).to(torch.int64)
            )
        stride = self.size_divisible
        maximum[1] = (
            torch.ceil(maximum[1].to(torch.float32) / stride) * stride
        ).to(torch.int64)
        maximum[2] = (
            torch.ceil(maximum[2].to(torch.float32) / stride) * stride
        ).to(torch.int64)
        padded: list[Tensor] = []
        for image in images:
            padding = [
                target - actual for target, actual in zip(maximum, tuple(image.shape))
            ]
            padded.append(
                F.pad(image, (0, padding[2], 0, padding[1], 0, padding[0]))
            )
        return torch.stack(padded)

    def batch_images(self, images: list[Tensor]) -> Tensor:
        if torchvision._is_tracing():
            return self._onnx_batch_images(images)
        maximum = self.max_by_axis([list(image.shape) for image in images])
        stride = float(self.size_divisible)
        maximum[1] = int(math.ceil(float(maximum[1]) / stride) * stride)
        maximum[2] = int(math.ceil(float(maximum[2]) / stride) * stride)
        batched = images[0].new_full([len(images)] + maximum, 0)
        for index, image in enumerate(images):
            batched[
                index, : image.shape[0], : image.shape[1], : image.shape[2]
            ].copy_(image)
        return batched

    def forward(self, images: list[Tensor]) -> ImageList:
        processed: list[Tensor] = []
        for image in images:
            if image.dim() != 3:
                raise ValueError(
                    "Faster R-CNN expects a list of [C, H, W] image tensors"
                )
            processed.append(
                _resize_image(
                    self.normalize(image), self.min_size[-1], self.max_size
                )
            )
        image_sizes = [image.shape[-2:] for image in processed]
        batched = self.batch_images(processed)
        return ImageList(
            batched, [(size[0], size[1]) for size in image_sizes]
        )

    def postprocess(
        self,
        detections: list[dict[str, Tensor]],
        image_shapes: list[tuple[int, int]],
        original_image_sizes: list[tuple[int, int]],
    ) -> list[dict[str, Tensor]]:
        for index, (prediction, image_shape, original_shape) in enumerate(
            zip(detections, image_shapes, original_image_sizes)
        ):
            detections[index]["boxes"] = _resize_boxes(
                prediction["boxes"], image_shape, original_shape
            )
        return detections


def _default_anchor_generator() -> AnchorGenerator:
    return AnchorGenerator(
        ((32,), (64,), (128,), (256,), (512,)),
        ((0.5, 1.0, 2.0),) * 5,
    )


class LibreFasterRCNNModel(nn.Module):
    """Native, checkpoint-compatible Faster R-CNN inference graph."""

    def __init__(self, size: str, num_classes: int = 91) -> None:
        super().__init__()
        if size not in FASTER_RCNN_CONFIGS:
            raise ValueError(
                f"Unknown Faster R-CNN size '{size}'. "
                f"Valid sizes: {', '.join(FASTER_RCNN_CONFIGS)}"
            )
        config = FASTER_RCNN_CONFIGS[size]
        self.size = size
        self.num_classes = num_classes

        if config["backbone"] == "mobilenet_v3_large":
            backbone = _mobilenet_fpn_backbone()
            anchor_sizes = ((32, 64, 128, 256, 512),) * 3
            anchor_generator = AnchorGenerator(
                anchor_sizes, ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            )
        else:
            backbone = _resnet_fpn_backbone(
                v2=config["backbone"] == "resnet50_fpn_v2"
            )
            anchor_generator = _default_anchor_generator()

        rpn_depth = 2 if size == "l" else 1
        rpn_head = RPNHead(
            backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0],
            conv_depth=rpn_depth,
        )
        self.backbone = backbone
        self.rpn = RegionProposalNetwork(
            anchor_generator,
            rpn_head,
            pre_nms_top_n={
                "training": 2000,
                "testing": config["rpn_pre_nms_top_n_test"],
            },
            post_nms_top_n={
                "training": 2000,
                "testing": config["rpn_post_nms_top_n_test"],
            },
            nms_thresh=0.7,
            score_thresh=config["rpn_score_thresh"],
        )

        roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )
        if size == "l":
            box_head: nn.Module = FastRCNNConvFCHead(
                (backbone.out_channels, 7, 7),
                [256, 256, 256, 256],
                [1024],
                norm_layer=nn.BatchNorm2d,
            )
        else:
            box_head = TwoMLPHead(backbone.out_channels * 7**2, 1024)
        self.roi_heads = RoIHeads(
            roi_pool,
            box_head,
            FastRCNNPredictor(1024, num_classes),
        )
        self.transform = GeneralizedRCNNTransform(
            config["min_size"],
            config["max_size"],
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

        # The released v1 ResNet-50 checkpoint is evaluated with FrozenBN eps=0.
        if size == "m":
            for module in self.modules():
                if isinstance(module, FrozenBatchNorm2d):
                    module.eps = 0.0

        self._last_head_tensors: Optional[
            dict[str, tuple[list[Tensor], list[Tensor]] | tuple[Tensor, Tensor]]
        ] = None

    def forward(
        self, images: Tensor | list[Tensor]
    ) -> list[dict[str, Tensor]]:
        if self.training:
            raise NotImplementedError(
                "The native Faster R-CNN port is inference-only"
            )
        if isinstance(images, Tensor):
            images = list(images)
        original_image_sizes = [
            (image.shape[-2], image.shape[-1]) for image in images
        ]
        image_list = self.transform(images)
        features = self.backbone(image_list.tensors)
        if isinstance(features, Tensor):
            features = OrderedDict([("0", features)])
        proposals, rpn_tensors = self.rpn(image_list, features)
        detections, roi_tensors = self.roi_heads(
            features, proposals, image_list.image_sizes
        )
        self._last_head_tensors = {"rpn": rpn_tensors, "roi": roi_tensors}
        return self.transform.postprocess(
            detections, image_list.image_sizes, original_image_sizes
        )
