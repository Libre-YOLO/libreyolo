"""
Libre YOLO11 implementation.
"""

import os
from typing import Union
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from .model11 import LibreYOLO11Model
from .utils11 import preprocess_image, postprocess, draw_boxes


class LIBREYOLO11:
    """
    Libre YOLO11 model for object detection.
    
    Args:
        model_path: Path to model weights file (required)
        size: Model size variant (required). Must be one of: "n", "s", "m", "l", "x"
        reg_max: Regression max value for DFL (default: 16)
        nb_classes: Number of classes (default: 80 for COCO)
    
    Example:
        >>> model = LIBREYOLO11(model_path="path/to/weights.pt", size="n")
        >>> detections = model(image=image_path, save=True)
    """
    
    def __init__(self, model_path: Union[str, dict], size: str, reg_max: int = 16, nb_classes: int = 80):
        """
        Initialize the Libre YOLO11 model.
        
        Args:
            model_path: Path to user-provided model weights file or loaded state dict
            size: Model size variant. Must be "n", "s", "m", "l", or "x"
            reg_max: Regression max value for DFL (default: 16)
            nb_classes: Number of classes (default: 80)
        """
        if size not in ['n', 's', 'm', 'l', 'x']:
            raise ValueError(f"Invalid size: {size}. Must be one of: 'n', 's', 'm', 'l', 'x'")
        
        self.size = size
        self.reg_max = reg_max
        self.nb_classes = nb_classes
        
        # Initialize model
        self.model = LibreYOLO11Model(config=size, reg_max=reg_max, nb_classes=nb_classes)
        
        # Load weights
        if isinstance(model_path, dict):
            self.model_path = None
            self.model.load_state_dict(model_path, strict=True)
        else:
            self.model_path = model_path
            self._load_weights(model_path)
        
        # Set to evaluation mode
        self.model.eval()
    
    def _load_weights(self, model_path: str):
        """Load model weights from file."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")
        
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {model_path}: {e}") from e
    
    def __call__(self, image: Union[str, Image.Image, np.ndarray], save: bool = False, conf_thres: float = 0.5, iou_thres: float = 0.7) -> dict:
        """
        Run inference on an image.
        
        Args:
            image: Input image. Can be a file path (str), PIL Image, or numpy array.
            save: If True, saves the image with detections drawn. Defaults to False.
            conf_thres: Confidence threshold (default: 0.25)
            iou_thres: IoU threshold for NMS (default: 0.45)
        
        Returns:
            Dictionary containing detection results with keys:
            - boxes: List of bounding boxes in xyxy format
            - scores: List of confidence scores
            - classes: List of class IDs
            - num_detections: Number of detections
            - saved_path: Path to saved image (if save=True)
        """
        # Store original image path for saving
        image_path = image if isinstance(image, str) else None
        
        # Preprocess image
        input_tensor, original_img, original_size = preprocess_image(image, input_size=640)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        detections = postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=640,
            original_size=original_size,
            max_det=100
        )
        
        # Draw and save if requested
        if save:
            if detections["num_detections"] > 0:
                annotated_img = draw_boxes(
                    original_img,
                    detections["boxes"],
                    detections["scores"],
                    detections["classes"]
                )
            else:
                annotated_img = original_img
            
            if image_path:
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_detections{ext}"
            else:
                output_path = "detections_output.jpg"
            
            annotated_img.save(output_path)
            detections["saved_path"] = output_path
        
        return detections
    
    def predict(self, image: Union[str, Image.Image, np.ndarray], save: bool = False, conf_thres: float = 0.25, iou_thres: float = 0.45) -> dict:
        """
        Alias for __call__ method.
        
        Args:
            image: Input image. Can be a file path (str), PIL Image, or numpy array.
            save: If True, saves the image with detections drawn. Defaults to False.
            conf_thres: Confidence threshold (default: 0.25)
            iou_thres: IoU threshold for NMS (default: 0.45)
        
        Returns:
            Dictionary containing detection results.
        """
        return self(image=image, save=save, conf_thres=conf_thres, iou_thres=iou_thres)

