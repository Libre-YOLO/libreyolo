"""Native Mask R-CNN inference architecture.

This mask-specific graph is derived from torchvision v0.26.0 at commit
``336d36e8db990a905498c73933e35231876e28bc`` under the BSD-3-Clause
license. It extends LibreYOLO's native Faster R-CNN graph with the RoIAlign
mask branch introduced by Mask R-CNN. See ``docs/provenance/mask_rcnn.md``
and the family notice for full attribution.

Mask R-CNN defined the modern two-stage instance-segmentation paradigm by
adding an aligned per-RoI mask branch to Faster R-CNN. This first release is
inference-only and ships the enhanced ResNet-50-FPN v2 COCO model.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import Conv2dNormActivation, MultiScaleRoIAlign

from ..faster_rcnn.nn import (
    GeneralizedRCNNTransform,
    LibreFasterRCNNModel,
    RoIHeads,
)

__all__ = [
    "LibreMaskRCNNModel",
    "MaskRCNNHeads",
    "MaskRCNNPredictor",
]


def maskrcnn_inference(
    mask_logits: Tensor,
    labels: list[Tensor],
) -> list[Tensor]:
    """Select each detection's class-specific sigmoid mask."""
    mask_probabilities = mask_logits.sigmoid()
    boxes_per_image = [label.shape[0] for label in labels]
    concatenated_labels = torch.cat(labels)
    indices = torch.arange(mask_logits.shape[0], device=concatenated_labels.device)
    selected = mask_probabilities[indices, concatenated_labels][:, None]
    return list(selected.split(boxes_per_image, dim=0))


def _onnx_expand_boxes(boxes: Tensor, scale) -> Tensor:
    half_width = (boxes[:, 2] - boxes[:, 0]) * 0.5
    half_height = (boxes[:, 3] - boxes[:, 1]) * 0.5
    center_x = (boxes[:, 2] + boxes[:, 0]) * 0.5
    center_y = (boxes[:, 3] + boxes[:, 1]) * 0.5
    half_width = half_width.to(dtype=torch.float32) * scale
    half_height = half_height.to(dtype=torch.float32) * scale
    return torch.stack(
        (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ),
        dim=1,
    )


def _expand_boxes(boxes: Tensor, scale) -> Tensor:
    if torchvision._is_tracing():
        return _onnx_expand_boxes(boxes, scale)
    half_width = (boxes[:, 2] - boxes[:, 0]) * 0.5
    half_height = (boxes[:, 3] - boxes[:, 1]) * 0.5
    center_x = (boxes[:, 2] + boxes[:, 0]) * 0.5
    center_y = (boxes[:, 3] + boxes[:, 1]) * 0.5
    half_width *= scale
    half_height *= scale
    expanded = torch.zeros_like(boxes)
    expanded[:, 0] = center_x - half_width
    expanded[:, 1] = center_y - half_height
    expanded[:, 2] = center_x + half_width
    expanded[:, 3] = center_y + half_height
    return expanded


@torch.jit.unused
def _expand_masks_tracing_scale(size: int, padding: int):
    return torch.tensor(size + 2 * padding).to(torch.float32) / torch.tensor(
        size
    ).to(torch.float32)


def _expand_masks(masks: Tensor, padding: int):
    size = masks.shape[-1]
    if torch._C._get_tracing_state():
        scale = _expand_masks_tracing_scale(size, padding)
    else:
        scale = float(size + 2 * padding) / size
    return F.pad(masks, (padding,) * 4), scale


def _paste_mask_in_image(
    mask: Tensor,
    box: Tensor,
    image_height: int,
    image_width: int,
) -> Tensor:
    width = max(int(box[2] - box[0] + 1), 1)
    height = max(int(box[3] - box[1] + 1), 1)
    mask = F.interpolate(
        mask.expand((1, 1, -1, -1)),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    image_mask = torch.zeros(
        (image_height, image_width),
        dtype=mask.dtype,
        device=mask.device,
    )
    x0 = max(box[0], 0)
    x1 = min(box[2] + 1, image_width)
    y0 = max(box[1], 0)
    y1 = min(box[3] + 1, image_height)
    image_mask[y0:y1, x0:x1] = mask[
        (y0 - box[1]) : (y1 - box[1]),
        (x0 - box[0]) : (x1 - box[0]),
    ]
    return image_mask


def _onnx_paste_mask_in_image(
    mask: Tensor,
    box: Tensor,
    image_height: Tensor,
    image_width: Tensor,
) -> Tensor:
    one = torch.ones(1, dtype=torch.int64)
    zero = torch.zeros(1, dtype=torch.int64)
    width = torch.max(torch.cat((box[2] - box[0] + one, one)))
    height = torch.max(torch.cat((box[3] - box[1] + one, one)))
    mask = F.interpolate(
        mask.expand((1, 1, mask.size(0), mask.size(1))),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    x0 = torch.max(torch.cat((box[0].unsqueeze(0), zero)))
    x1 = torch.min(torch.cat((box[2].unsqueeze(0) + one, image_width.unsqueeze(0))))
    y0 = torch.max(torch.cat((box[1].unsqueeze(0), zero)))
    y1 = torch.min(torch.cat((box[3].unsqueeze(0) + one, image_height.unsqueeze(0))))
    unpadded = mask[
        (y0 - box[1]) : (y1 - box[1]),
        (x0 - box[0]) : (x1 - box[0]),
    ]
    zeros_y0 = torch.zeros(y0, unpadded.size(1))
    zeros_y1 = torch.zeros(image_height - y1, unpadded.size(1))
    rows = torch.cat((zeros_y0, unpadded.to(torch.float32), zeros_y1), dim=0)[
        :image_height, :
    ]
    zeros_x0 = torch.zeros(rows.size(0), x0)
    zeros_x1 = torch.zeros(rows.size(0), image_width - x1)
    return torch.cat((zeros_x0, rows, zeros_x1), dim=1)[:, :image_width]


@torch.jit._script_if_tracing
def _onnx_paste_masks_loop(
    masks: Tensor,
    boxes: Tensor,
    image_height: Tensor,
    image_width: Tensor,
) -> Tensor:
    pasted = torch.zeros(0, image_height, image_width)
    for index in range(masks.size(0)):
        current = _onnx_paste_mask_in_image(
            masks[index][0],
            boxes[index],
            image_height,
            image_width,
        )
        pasted = torch.cat((pasted, current.unsqueeze(0)))
    return pasted


def paste_masks_in_image(
    masks: Tensor,
    boxes: Tensor,
    image_shape: tuple[int, int],
    padding: int = 1,
) -> Tensor:
    """Paste fixed-resolution soft RoI masks into an image canvas."""
    masks, scale = _expand_masks(masks, padding)
    boxes = _expand_boxes(boxes, scale).to(dtype=torch.int64)
    image_height, image_width = image_shape
    if torchvision._is_tracing():
        return _onnx_paste_masks_loop(
            masks,
            boxes,
            torch.scalar_tensor(image_height, dtype=torch.int64),
            torch.scalar_tensor(image_width, dtype=torch.int64),
        )[:, None]
    pasted = [
        _paste_mask_in_image(mask[0], box, image_height, image_width)
        for mask, box in zip(masks, boxes)
    ]
    if pasted:
        return torch.stack(pasted, dim=0)[:, None]
    return masks.new_empty((0, 1, image_height, image_width))


class MaskRCNNTransform(GeneralizedRCNNTransform):
    """Restore boxes and paste soft masks onto each original image canvas."""

    def postprocess(
        self,
        detections: list[dict[str, Tensor]],
        image_shapes: list[tuple[int, int]],
        original_image_sizes: list[tuple[int, int]],
    ) -> list[dict[str, Tensor]]:
        detections = super().postprocess(
            detections,
            image_shapes,
            original_image_sizes,
        )
        for detection, original_shape in zip(detections, original_image_sizes):
            if "masks" in detection:
                detection["masks"] = paste_masks_in_image(
                    detection["masks"],
                    detection["boxes"],
                    original_shape,
                )
        return detections


class MaskRCNNHeads(nn.Sequential):
    """Four aligned convolutional layers over 14 x 14 RoI features."""

    _version = 2

    def __init__(
        self,
        in_channels: int,
        layers: list[int],
        dilation: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        blocks: list[nn.Module] = []
        next_channels = in_channels
        for layer_channels in layers:
            blocks.append(
                Conv2dNormActivation(
                    next_channels,
                    layer_channels,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    norm_layer=norm_layer,
                )
            )
            next_channels = layer_channels
        super().__init__(*blocks)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
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
            for index in range(len(self)):
                for parameter in ("weight", "bias"):
                    old_key = f"{prefix}mask_fcn{index + 1}.{parameter}"
                    new_key = f"{prefix}{index}.0.{parameter}"
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


class MaskRCNNPredictor(nn.Sequential):
    """Upsample each RoI and emit one 28 x 28 logit map per class."""

    def __init__(self, in_channels: int, dim_reduced: int, num_classes: int) -> None:
        super().__init__(
            OrderedDict(
                [
                    (
                        "conv5_mask",
                        nn.ConvTranspose2d(
                            in_channels,
                            dim_reduced,
                            kernel_size=2,
                            stride=2,
                        ),
                    ),
                    ("relu", nn.ReLU(inplace=True)),
                    (
                        "mask_fcn_logits",
                        nn.Conv2d(dim_reduced, num_classes, kernel_size=1),
                    ),
                ]
            )
        )
        for name, parameter in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(
                    parameter,
                    mode="fan_out",
                    nonlinearity="relu",
                )


class MaskRoIHeads(RoIHeads):
    """Faster R-CNN box heads followed by the class-specific mask branch."""

    def __init__(
        self,
        box_roi_pool: MultiScaleRoIAlign,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_roi_pool: MultiScaleRoIAlign,
        mask_head: nn.Module,
        mask_predictor: nn.Module,
        *,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 100,
        return_masks: bool = True,
    ) -> None:
        super().__init__(
            box_roi_pool,
            box_head,
            box_predictor,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
        )
        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor
        self.return_masks = return_masks

    def forward(
        self,
        features: dict[str, Tensor],
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> tuple[list[dict[str, Tensor]], dict[str, Tensor]]:
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        boxes, scores, labels = self.postprocess_detections(
            class_logits,
            box_regression,
            proposals,
            image_shapes,
        )
        detections = [
            {"boxes": box, "labels": label, "scores": score}
            for box, label, score in zip(boxes, labels, scores)
        ]

        if self.return_masks:
            mask_features = self.mask_roi_pool(features, boxes, image_shapes)
            mask_logits = self.mask_predictor(self.mask_head(mask_features))
            mask_probabilities = maskrcnn_inference(mask_logits, labels)
            for masks, detection in zip(mask_probabilities, detections):
                detection["masks"] = masks
        return detections, {}


class LibreMaskRCNNModel(LibreFasterRCNNModel):
    """Checkpoint-compatible ResNet-50-FPN v2 Mask R-CNN inference graph."""

    def __init__(
        self,
        size: str = "r50",
        num_classes: int = 91,
        *,
        return_masks: bool = True,
    ) -> None:
        if size != "r50":
            raise ValueError("Mask R-CNN currently ships only size 'r50'.")

        # The released Mask R-CNN v2 shares the complete ResNet-50-FPN v2,
        # two-layer RPN, and deep box head with LibreFasterRCNN size l.
        super().__init__(size="l", num_classes=num_classes)
        box_heads = self.roi_heads
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=14,
            sampling_ratio=2,
        )
        mask_head = MaskRCNNHeads(
            self.backbone.out_channels,
            [256, 256, 256, 256],
            dilation=1,
            norm_layer=nn.BatchNorm2d,
        )
        self.roi_heads = MaskRoIHeads(
            box_heads.box_roi_pool,
            box_heads.box_head,
            box_heads.box_predictor,
            mask_roi_pool,
            mask_head,
            MaskRCNNPredictor(256, 256, num_classes),
            score_thresh=box_heads.score_thresh,
            nms_thresh=box_heads.nms_thresh,
            detections_per_img=box_heads.detections_per_img,
            return_masks=return_masks,
        )
        self.transform = MaskRCNNTransform(
            800,
            1333,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
        self.size = size
