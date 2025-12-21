"""
Utility functions for YOLO11 preprocessing, postprocessing, and visualization.
"""

import torch
import numpy as np
from typing import Union, Tuple, Dict, List, Optional
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


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = False, dim: int = -1) -> torch.Tensor:
    """
    Transform distance(ltrb) to box(xywh or xyxy).
    Matches YOLO11's dist2bbox implementation.
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def decode_boxes(box_preds: torch.Tensor, anchors: torch.Tensor, stride_tensor: torch.Tensor) -> torch.Tensor:
    """
    Decode box predictions to xyxy coordinates using YOLO11's method.
    
    Args:
        box_preds: Box predictions [l, t, r, b] distances from anchors (B, N, 4) in grid space
        anchors: Anchor points (N, 2) in grid coordinates
        stride_tensor: Stride values (N, 1)
        
    Returns:
        Decoded boxes in xyxy format (B, N, 4) in pixel coordinates
    """
    # box_preds shape: (B, N, 4) where 4 = [l, t, r, b] (already DFL-decoded, in grid space)
    # anchors shape: (N, 2) where 2 = [x, y] in grid coordinates
    # stride_tensor shape: (N, 1)
    
    # Reshape anchors and strides for broadcasting: (1, N, 2) and (1, N, 1)
    anchors = anchors.unsqueeze(0)
    stride_tensor = stride_tensor.unsqueeze(0)
    
    # Decode in grid space first
    # dist2bbox expects (B, N, 4) and (B, N, 2) or similar
    decoded_boxes_grid = dist2bbox(box_preds, anchors, xywh=True, dim=-1)  # (B, N, 4) in xywh format, grid space
    
    # Convert to pixel coordinates: multiply center and size by stride
    decoded_boxes_px = decoded_boxes_grid * stride_tensor  # (B, N, 4) in xywh format, pixel space
    
    # Convert xywh to xyxy
    cx, cy, w, h = decoded_boxes_px[..., 0], decoded_boxes_px[..., 1], decoded_boxes_px[..., 2], decoded_boxes_px[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    decoded_boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, N, 4) in xyxy format
    
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


def postprocess(output: Dict, conf_thres: float = 0.25, iou_thres: float = 0.45, input_size: int = 640, original_size: Tuple[int, int] = None, max_det: int = 300) -> Dict:
    """
    Postprocess model outputs to get final detections.
    
    Args:
        output: Model output dictionary with 'x8', 'x16', 'x32' keys
        conf_thres: Confidence threshold (default: 0.25)
        iou_thres: IoU threshold for NMS (default: 0.45)
        input_size: Input image size (default: 640)
        original_size: Original image size (width, height) for scaling
        max_det: Maximum number of detections to return (default: 300)
        
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
    decoded_boxes = decode_boxes(box_preds, anchors, stride_tensor)  # (1, N, 4)
    decoded_boxes = decoded_boxes[0]  # Remove batch dimension: (N, 4)
    
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
    
    # decoded_boxes is (N, 4) after removing batch dim above
    valid_boxes = decoded_boxes[mask]
    valid_scores = max_scores[mask]
    valid_classes = class_ids[mask]
    
    # Scale boxes to original image size if provided
    if original_size is not None:
        scale_x = original_size[0] / input_size
        scale_y = original_size[1] / input_size
        valid_boxes[:, [0, 2]] *= scale_x
        valid_boxes[:, [1, 3]] *= scale_y
        
        # Clip boxes to image boundaries
        valid_boxes[:, [0, 2]] = torch.clamp(valid_boxes[:, [0, 2]], 0, original_size[0])
        valid_boxes[:, [1, 3]] = torch.clamp(valid_boxes[:, [1, 3]], 0, original_size[1])
        
        # Remove boxes with invalid dimensions (width or height <= 0)
        box_widths = valid_boxes[:, 2] - valid_boxes[:, 0]
        box_heights = valid_boxes[:, 3] - valid_boxes[:, 1]
        valid_mask = (box_widths > 0) & (box_heights > 0)
        
        if not valid_mask.all():
            valid_boxes = valid_boxes[valid_mask]
            valid_scores = valid_scores[valid_mask]
            valid_classes = valid_classes[valid_mask]
    
    # Apply NMS - YOLO11 uses per-class NMS (agnostic=False)
    # Use torchvision.ops.nms if available, otherwise fall back to custom implementation
    try:
        import torchvision.ops
        use_torchvision_nms = True
    except ImportError:
        use_torchvision_nms = False
    
    # Apply NMS per-class (like YOLO11 with agnostic=False)
    unique_classes = torch.unique(valid_classes)
    keep_indices_list = []
    
    for cls in unique_classes:
        cls_mask = valid_classes == cls
        cls_boxes = valid_boxes[cls_mask]
        cls_scores = valid_scores[cls_mask]
        
        if len(cls_boxes) == 0:
            continue
        
        # Apply NMS for this class
        if use_torchvision_nms:
            # For per-class NMS, offset boxes by class to prevent cross-class suppression
            # This is what YOLO11 does when agnostic=False
            max_wh = 7680.0
            boxes_for_nms = cls_boxes + cls.float() * max_wh
            cls_keep = torchvision.ops.nms(boxes_for_nms, cls_scores, iou_thres)
        else:
            cls_keep = nms(cls_boxes, cls_scores, iou_thres)
        
        # Get original indices in valid_boxes
        cls_indices = torch.where(cls_mask)[0]
        keep_indices_list.append(cls_indices[cls_keep])
    
    if len(keep_indices_list) == 0:
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0
        }
    
    # Concatenate all kept indices
    keep_indices = torch.cat(keep_indices_list)
    
    # Sort by score (descending) and limit to max_det
    if len(keep_indices) > max_det:
        final_scores_temp = valid_scores[keep_indices]
        _, top_indices = torch.topk(final_scores_temp, max_det)
        keep_indices = keep_indices[top_indices]
    
    final_boxes = valid_boxes[keep_indices].cpu().numpy()
    final_scores = valid_scores[keep_indices].cpu().numpy()
    final_classes = valid_classes[keep_indices].cpu().numpy()
    
    return {
        "boxes": final_boxes.tolist(),
        "scores": final_scores.tolist(),
        "classes": final_classes.tolist(),
        "num_detections": len(final_boxes)
    }

