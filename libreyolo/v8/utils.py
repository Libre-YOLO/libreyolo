"""
Utility functions for Libre YOLO8.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Union
from PIL import Image

# Import shared utilities
from ..common.utils import preprocess_image, draw_boxes, COCO_CLASSES, get_class_color


def make_anchors(feats: List[torch.Tensor], strides: List[int], grid_cell_offset: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate anchor points from feature map sizes.
    
    Args:
        feats: List of feature tensors from different scales
        strides: List of stride values corresponding to each feature map
        grid_cell_offset: Offset for grid cell centers (default: 0.5)
        
    Returns:
        Tuple of (anchor_points, stride_tensor)
    """
    anchor_points = []
    stride_tensor = []
    
    for i, (feat, stride) in enumerate(zip(feats, strides)):
        _, _, h, w = feat.shape
        dtype, device = feat.dtype, feat.device
        
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def decode_boxes(box_preds: torch.Tensor, anchors: torch.Tensor, stride_tensor: torch.Tensor) -> torch.Tensor:
    """
    Decode box predictions to xyxy coordinates.
    
    Args:
        box_preds: Box predictions [l, t, r, b] distances from anchors (B, N, 4)
        anchors: Anchor points (N, 2)
        stride_tensor: Stride values (N, 1)
        
    Returns:
        Decoded boxes in xyxy format (B, N, 4)
    """
    # box_preds shape: (B, N, 4) where 4 = [l, t, r, b]
    # anchors shape: (N, 2) where 2 = [x, y]
    # stride_tensor shape: (N, 1)
    
    # Decode: xyxy = [x - l, y - t, x + r, y + b] * stride
    x1 = (anchors[:, 0:1] - box_preds[0, :, 0:1]) * stride_tensor[:, 0:1]
    y1 = (anchors[:, 1:2] - box_preds[0, :, 1:2]) * stride_tensor[:, 0:1]
    x2 = (anchors[:, 0:1] + box_preds[0, :, 2:3]) * stride_tensor[:, 0:1]
    y2 = (anchors[:, 1:2] + box_preds[0, :, 3:4]) * stride_tensor[:, 0:1]
    
    decoded_boxes = torch.cat([x1, y1, x2, y2], dim=1)
    return decoded_boxes


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> torch.Tensor:
    """
    Non-Maximum Suppression using torch operations.
    
    Args:
        boxes: Boxes in xyxy format (N, 4)
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    # Sort by scores (descending)
    _, order = scores.sort(0, descending=True)
    keep = []
    
    while len(order) > 0:
        # Keep the box with highest score
        i = order[0]
        keep.append(i.item())
        
        if len(order) == 1:
            break
        
        # Calculate IoU with remaining boxes
        box_i = boxes[i]
        boxes_remaining = boxes[order[1:]]
        
        # Calculate intersection
        x1_i, y1_i, x2_i, y2_i = box_i
        x1_r, y1_r, x2_r, y2_r = boxes_remaining[:, 0], boxes_remaining[:, 1], boxes_remaining[:, 2], boxes_remaining[:, 3]
        
        x1_inter = torch.max(x1_i, x1_r)
        y1_inter = torch.max(y1_i, y1_r)
        x2_inter = torch.min(x2_i, x2_r)
        y2_inter = torch.min(y2_i, y2_r)
        
        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
        
        # Calculate union
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        area_r = (x2_r - x1_r) * (y2_r - y1_r)
        union_area = area_i + area_r - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-7)
        
        # Keep boxes with IoU < threshold
        order = order[1:][iou < iou_threshold]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def postprocess(output: Dict, conf_thres: float = 0.25, iou_thres: float = 0.45, input_size: int = 640, original_size: Tuple[int, int] = None) -> Dict:
    """
    Postprocess model outputs to get final detections.
    
    Args:
        output: Model output dictionary with 'x8', 'x16', 'x32' keys
        conf_thres: Confidence threshold (default: 0.25)
        iou_thres: IoU threshold for NMS (default: 0.45)
        input_size: Input image size (default: 640)
        original_size: Original image size (width, height) for scaling
        
    Returns:
        Dictionary with boxes, scores, classes, num_detections
    """
    # Collect outputs from the 3 heads
    box_layers = [output['x8']['box'], output['x16']['box'], output['x32']['box']]
    cls_layers = [output['x8']['cls'], output['x16']['cls'], output['x32']['cls']]
    strides = [8, 16, 32]
    
    # Generate anchors
    anchors, stride_tensor = make_anchors(box_layers, strides)
    
    # Flatten and concatenate predictions
    # Box: (Batch, 4, H, W) -> (Batch, H*W, 4)
    box_preds = torch.cat([x.flatten(2).permute(0, 2, 1) for x in box_layers], dim=1)
    # Cls: (Batch, 80, H, W) -> (Batch, H*W, 80)
    cls_preds = torch.cat([x.flatten(2).permute(0, 2, 1) for x in cls_layers], dim=1)
    
    # Decode boxes
    decoded_boxes = decode_boxes(box_preds, anchors, stride_tensor)
    
    # Apply confidence threshold
    scores = cls_preds[0].sigmoid()  # (N, 80)
    max_scores, class_ids = torch.max(scores, dim=1)
    
    mask = max_scores > conf_thres
    if not mask.any():
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0
        }
    
    valid_boxes = decoded_boxes[mask]
    valid_scores = max_scores[mask]
    valid_classes = class_ids[mask]
    
    # Scale boxes to original image size if provided
    if original_size is not None:
        scale_x = original_size[0] / input_size
        scale_y = original_size[1] / input_size
        valid_boxes[:, [0, 2]] *= scale_x
        valid_boxes[:, [1, 3]] *= scale_y
    
    # Apply NMS
    keep_indices = nms(valid_boxes, valid_scores, iou_thres)
    
    final_boxes = valid_boxes[keep_indices].cpu().numpy()
    final_scores = valid_scores[keep_indices].cpu().numpy()
    final_classes = valid_classes[keep_indices].cpu().numpy()
    
    return {
        "boxes": final_boxes.tolist(),
        "scores": final_scores.tolist(),
        "classes": final_classes.tolist(),
        "num_detections": len(final_boxes)
    }

