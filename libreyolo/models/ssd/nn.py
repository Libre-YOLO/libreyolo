"""Native SSD300-VGG16 inference graph.

The module layout is derived from ``pytorch/vision`` v0.26.0 at commit
``336d36e8db990a905498c73933e35231876e28bc`` (BSD-3-Clause).  Training-only
matching and loss code is intentionally excluded.  Attribute names mirror the
upstream graph so official state dictionaries load without tensor remapping.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from ...postprocess.ssd import _default_boxes
from ...utils.coco import COCO91_CATEGORY_IDS


def _xavier_init(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def _make_vgg16_features() -> nn.Sequential:
    """Build the VGG16-D convolutional tower with upstream module indices."""
    configuration: tuple[int | str, ...] = (
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    )
    layers: list[nn.Module] = []
    in_channels = 3
    for value in configuration:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        out_channels = int(value)
        layers.extend(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
        )
        in_channels = out_channels
    return nn.Sequential(*layers)


class SSDFeatureExtractorVGG(nn.Module):
    """VGG16 through conv4_3 plus SSD300's five later feature blocks."""

    out_channels = (512, 1024, 512, 256, 256, 256)

    def __init__(self) -> None:
        super().__init__()
        backbone = _make_vgg16_features()
        maxpool_positions = [
            index
            for index, layer in enumerate(backbone)
            if isinstance(layer, nn.MaxPool2d)
        ]
        maxpool3_pos = maxpool_positions[2]
        maxpool4_pos = maxpool_positions[3]
        backbone[maxpool3_pos].ceil_mode = True

        self.scale_weight = nn.Parameter(torch.ones(512) * 20)
        self.features = nn.Sequential(*backbone[:maxpool4_pos])

        fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        _xavier_init(fc)

        extra = nn.ModuleList(
            [
                nn.Sequential(*backbone[maxpool4_pos:-1], fc),
                nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        _xavier_init(extra[1:])
        self.extra = extra

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.features(images)
        rescaled = self.scale_weight.view(1, -1, 1, 1) * F.normalize(features)
        outputs = [rescaled]
        for block in self.extra:
            features = block(features)
            outputs.append(features)
        return OrderedDict((str(index), value) for index, value in enumerate(outputs))


class SSDScoringHead(nn.Module):
    """Apply one convolution per feature level and concatenate anchor rows."""

    def __init__(self, module_list: nn.ModuleList, num_columns: int) -> None:
        super().__init__()
        self.module_list = module_list
        self.num_columns = int(num_columns)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        results = []
        for level, feature in enumerate(features):
            prediction = self.module_list[level](feature)
            batch, _, height, width = prediction.shape
            prediction = prediction.view(
                batch,
                -1,
                self.num_columns,
                height,
                width,
            )
            prediction = prediction.permute(0, 3, 4, 1, 2)
            results.append(prediction.reshape(batch, -1, self.num_columns))
        return torch.cat(results, dim=1)


class SSDClassificationHead(SSDScoringHead):
    def __init__(
        self,
        in_channels: tuple[int, ...],
        num_anchors: tuple[int, ...],
        num_classes: int,
    ) -> None:
        layers = nn.ModuleList(
            nn.Conv2d(channels, anchors * num_classes, kernel_size=3, padding=1)
            for channels, anchors in zip(in_channels, num_anchors)
        )
        _xavier_init(layers)
        super().__init__(layers, num_classes)


class SSDRegressionHead(SSDScoringHead):
    def __init__(
        self,
        in_channels: tuple[int, ...],
        num_anchors: tuple[int, ...],
    ) -> None:
        layers = nn.ModuleList(
            nn.Conv2d(channels, anchors * 4, kernel_size=3, padding=1)
            for channels, anchors in zip(in_channels, num_anchors)
        )
        _xavier_init(layers)
        super().__init__(layers, 4)


class SSDHead(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, ...],
        num_anchors: tuple[int, ...],
        num_classes: int,
    ) -> None:
        super().__init__()
        self.classification_head = SSDClassificationHead(
            in_channels,
            num_anchors,
            num_classes,
        )
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "bbox_regression": self.regression_head(features),
            "cls_logits": self.classification_head(features),
        }


class LibreSSDModel(nn.Module):
    """Fixed-resolution SSD300-VGG16 raw-head graph."""

    NUM_ANCHORS = (4, 6, 6, 6, 4, 4)

    def __init__(self, num_classes: int = 91) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone = SSDFeatureExtractorVGG()
        self.head = SSDHead(
            SSDFeatureExtractorVGG.out_channels,
            self.NUM_ANCHORS,
            self.num_classes,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = list(self.backbone(images).values())
        return self.head(features)


class SSDExportWrapper(nn.Module):
    """Export decoded boxes and class probabilities as a YOLO-grid tensor."""

    def __init__(self, model: LibreSSDModel) -> None:
        super().__init__()
        self.model = model
        anchors = _default_boxes(device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("anchors", anchors, persistent=False)
        if model.num_classes == 91:
            class_indices = torch.as_tensor(COCO91_CATEGORY_IDS, dtype=torch.long)
        else:
            class_indices = torch.arange(1, model.num_classes, dtype=torch.long)
        self.register_buffer("class_indices", class_indices, persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(images)
        regression = outputs["bbox_regression"]
        anchors = self.anchors.to(dtype=regression.dtype)
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        center_x = anchors[:, 0] + 0.5 * widths
        center_y = anchors[:, 1] + 0.5 * heights

        dx = regression[..., 0] / 10.0
        dy = regression[..., 1] / 10.0
        dw = (regression[..., 2] / 5.0).clamp(max=4.135166556742356)
        dh = (regression[..., 3] / 5.0).clamp(max=4.135166556742356)
        predicted_center_x = dx * widths + center_x
        predicted_center_y = dy * heights + center_y
        predicted_width = dw.exp() * widths
        predicted_height = dh.exp() * heights
        boxes = torch.stack(
            (
                predicted_center_x - 0.5 * predicted_width,
                predicted_center_y - 0.5 * predicted_height,
                predicted_center_x + 0.5 * predicted_width,
                predicted_center_y + 0.5 * predicted_height,
            ),
            dim=-1,
        )
        boxes = boxes.clamp(min=0.0, max=300.0)
        probabilities = outputs["cls_logits"].softmax(dim=-1)
        probabilities = probabilities.index_select(-1, self.class_indices)
        packed = torch.cat((boxes, probabilities), dim=-1)
        return packed.transpose(1, 2)


__all__ = [
    "LibreSSDModel",
    "SSDClassificationHead",
    "SSDExportWrapper",
    "SSDFeatureExtractorVGG",
    "SSDHead",
    "SSDRegressionHead",
    "SSDScoringHead",
]
