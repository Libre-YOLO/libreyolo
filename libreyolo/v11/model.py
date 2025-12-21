"""
Libre YOLO11 implementation.
"""

import os
import json
from datetime import datetime
from typing import Union, List
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from .nn import LibreYOLO11Model
from .utils import preprocess_image, postprocess, draw_boxes


class LIBREYOLO11:
    """
    Libre YOLO11 model for object detection.
    
    Args:
        model_path: Path to model weights file (required)
        size: Model size variant (required). Must be one of: "n", "s", "m", "l", "x"
        reg_max: Regression max value for DFL (default: 16)
        nb_classes: Number of classes (default: 80 for COCO)
        save_feature_maps: Feature map saving mode. Options:
            - False: Disabled (default)
            - True: Save all layers
            - List of layer names: Save only specified layers (e.g., ["backbone_p1", "neck_c2f21"])
    
    Example:
        >>> model = LIBREYOLO11(model_path="path/to/weights.pt", size="x", save_feature_maps=True)
        >>> detections = model(image=image_path, save=True)
    """
    
    def __init__(self, model_path: Union[str, dict], size: str, reg_max: int = 16, nb_classes: int = 80, save_feature_maps: Union[bool, List[str]] = False):
        """
        Initialize the Libre YOLO11 model.
        
        Args:
            model_path: Path to user-provided model weights file or loaded state dict
            size: Model size variant. Must be "n", "s", "m", "l", or "x"
            reg_max: Regression max value for DFL (default: 16)
            nb_classes: Number of classes (default: 80)
            save_feature_maps: Feature map saving mode. Options:
                - False: Disabled
                - True: Save all layers
                - List[str]: Save only specified layer names
        """
        if size not in ['n', 's', 'm', 'l', 'x']:
            raise ValueError(f"Invalid size: {size}. Must be one of: 'n', 's', 'm', 'l', 'x'")
        
        self.size = size
        self.reg_max = reg_max
        self.nb_classes = nb_classes
        self.save_feature_maps = save_feature_maps
        self.feature_maps = {}
        self.hooks = []
        
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
        
        # Register hooks for feature map extraction
        if self.save_feature_maps:
            self._register_hooks()
    
    def _load_weights(self, model_path: str):
        """Load model weights from file."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")
        
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {model_path}: {e}") from e
    
    def get_available_layer_names(self) -> List[str]:
        """
        Get list of available layer names for feature map saving.
        
        Returns:
            List of layer names that can be used with save_feature_maps parameter.
        """
        return sorted(self._get_available_layers().keys())
    
    def _get_available_layers(self) -> dict:
        """Get mapping of layer names to module objects."""
        layers = {
            # Backbone layers
            "backbone_p1": self.model.backbone.p1,
            "backbone_p2": self.model.backbone.p2,
            "backbone_c2f1": self.model.backbone.c2f1,
            "backbone_p3": self.model.backbone.p3,
            "backbone_c2f2_P3": self.model.backbone.c2f2,
            "backbone_p4": self.model.backbone.p4,
            "backbone_c2f3_P4": self.model.backbone.c2f3,
            "backbone_p5": self.model.backbone.p5,
            "backbone_c2f4": self.model.backbone.c2f4,
            "backbone_sppf_P5": self.model.backbone.sppf,
            "backbone_c2psa_P5": self.model.backbone.c2psa,
            # Neck layers
            "neck_c2f21": self.model.neck.c2f21,
            "neck_c2f11": self.model.neck.c2f11,
            "neck_c2f12": self.model.neck.c2f12,
            "neck_c2f22": self.model.neck.c2f22,
            # Head layers
            "head8_conv11": self.model.head8.conv11,
            "head8_conv21": self.model.head8.conv21,
            "head16_conv11": self.model.head16.conv11,
            "head16_conv21": self.model.head16.conv21,
            "head32_conv11": self.model.head32.conv11,
            "head32_conv21": self.model.head32.conv21,
        }
        return layers
    
    def _register_hooks(self):
        """Register forward hooks to capture feature maps from model layers."""
        def get_hook(name):
            def hook(module, input, output):
                # Detach and move to CPU to prevent memory leaks
                self.feature_maps[name] = output.detach().cpu()
            return hook
        
        available_layers = self._get_available_layers()
        
        if self.save_feature_maps is True:
            # Hook into all available layers
            for layer_name, module in available_layers.items():
                self.hooks.append(module.register_forward_hook(get_hook(layer_name)))
        
        elif isinstance(self.save_feature_maps, list):
            # Hook into specified layers only
            invalid_layers = []
            for layer_name in self.save_feature_maps:
                if layer_name not in available_layers:
                    invalid_layers.append(layer_name)
                else:
                    self.hooks.append(available_layers[layer_name].register_forward_hook(get_hook(layer_name)))
            
            if invalid_layers:
                available = ", ".join(sorted(available_layers.keys()))
                raise ValueError(
                    f"Invalid layer names: {invalid_layers}. "
                    f"Available layers: {available}"
                )
    
    def _save_feature_maps(self, image_path):
        """Save feature map visualizations to disk."""
        # Determine the base name for the output directory
        if isinstance(image_path, str):
            stem = Path(image_path).stem
        else:
            stem = "inference"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("runs/feature_maps") / f"{stem}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "model": "LIBREYOLO11",
            "size": self.size,
            "input_size": [640, 640],
            "image_source": str(image_path) if isinstance(image_path, str) else "PIL/numpy input",
            "layers_captured": list(self.feature_maps.keys())
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature map visualizations
        for layer_name, fmap in self.feature_maps.items():
            # fmap shape: (batch, channels, H, W) - take first batch item
            fmap = fmap[0] if fmap.dim() == 4 else fmap
            
            # Create a 4x4 grid of the first 16 channels
            channels = min(fmap.shape[0], 16)
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            
            for i in range(16):
                ax = axes[i // 4, i % 4]
                if i < channels:
                    # Normalize the feature map for better visualization
                    channel_data = fmap[i].numpy()
                    ax.imshow(channel_data, cmap='viridis')
                ax.axis('off')
            
            plt.suptitle(f"Feature Maps: {layer_name}\nShape: {list(fmap.shape)}", fontsize=14)
            plt.tight_layout()
            plt.savefig(save_dir / f"{layer_name}.png", bbox_inches='tight', dpi=100)
            plt.close()
        
        # Clear feature maps after saving
        self.feature_maps.clear()
        
        return str(save_dir)
    
    def __call__(self, image: str | Image.Image | np.ndarray, save: bool = False, output_path: str = None, conf_thres: float = 0.25, iou_thres: float = 0.45) -> dict:
        """
        Run inference on an image.
        
        Args:
            image: Input image. Can be a file path (str), PIL Image, or numpy array.
            save: If True, saves the image with detections drawn. Defaults to False.
            output_path: Optional path to save the annotated image. If not provided, 
                         saves to 'runs/detections/' with a timestamped name.
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
            original_size=original_size
        )
        
        # Save feature maps if enabled
        if self.save_feature_maps:
            feature_maps_path = self._save_feature_maps(image_path)
            detections["feature_maps_path"] = feature_maps_path
        
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
            
            if output_path:
                final_output_path = Path(output_path)
                if final_output_path.suffix == "":
                    # If directory, create it and use default naming
                    final_output_path.mkdir(parents=True, exist_ok=True)
                    if isinstance(image_path, str):
                        stem = Path(image_path).stem
                        ext = Path(image_path).suffix
                    else:
                        stem = "inference"
                        ext = ".jpg"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    final_output_path = final_output_path / f"{stem}_{timestamp}{ext}"
                else:
                    # If file path, ensure parent directory exists
                    final_output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Determine save directory (matching feature map style)
                if isinstance(image_path, str):
                    stem = Path(image_path).stem
                    ext = Path(image_path).suffix
                else:
                    stem = "inference"
                    ext = ".jpg"
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = Path("runs/detections")
                save_dir.mkdir(parents=True, exist_ok=True)
                final_output_path = save_dir / f"{stem}_{timestamp}{ext}"
            
            annotated_img.save(final_output_path)
            detections["saved_path"] = str(final_output_path)
        
        return detections
    
        
    def export(self, output_path: str = None, input_size: int = 640, opset: int = 12) -> str:
        """
        Export the model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX file. If None, uses the model's weights path with .onnx extension.
            input_size: The image size to export for (default: 640).
            opset: ONNX opset version (default: 12).
            
        Returns:
            Path to the exported ONNX file.
        """
        import torch.onnx
        
        if output_path is None:
            if self.model_path and isinstance(self.model_path, str):
                output_path = str(Path(self.model_path).with_suffix('.onnx'))
            else:
                output_path = f"libreyolo11{self.size}.onnx"
        
        print(f"Exporting LibreYOLO11 {self.size} to {output_path}...")
        
        # 1. Create a dummy input (Batch, Channels, Height, Width)
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
        
        # 2. Define a wrapper that decodes boxes for end-to-end inference
        class ONNXWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                output = self.model(x)
                
                # Collect outputs from the 3 heads
                box_layers = [output['x8']['box'], output['x16']['box'], output['x32']['box']]
                cls_layers = [output['x8']['cls'], output['x16']['cls'], output['x32']['cls']]
                strides = [8, 16, 32]
                
                # Generate anchors (Traceable)
                anchors, stride_tensor = make_anchors(box_layers, strides)
                
                # Flatten and concatenate predictions
                # Box: (Batch, 4, H, W) -> (Batch, N, 4)
                box_preds = torch.cat([x.flatten(2).permute(0, 2, 1) for x in box_layers], dim=1)
                # Cls: (Batch, 80, H, W) -> (Batch, N, 80)
                cls_preds = torch.cat([x.flatten(2).permute(0, 2, 1) for x in cls_layers], dim=1)
                
                # Decode boxes to xyxy (Batch, N, 4)
                decoded_boxes = decode_boxes(box_preds, anchors, stride_tensor)
                
                # Apply sigmoid to class scores
                cls_scores = cls_preds.sigmoid()
                
                # Return concatenated [boxes, scores]: (Batch, N, 84)
                return torch.cat([decoded_boxes, cls_scores], dim=-1)

        wrapper = ONNXWrapper(self.model)
        wrapper.eval()
        
        # 3. Perform the export
        try:
            import torch.onnx
            torch.onnx.export(
                wrapper,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'output': {0: 'batch'}
                }
            )
            print(f"Export complete: {output_path}")
            return output_path
        except Exception as e:
            print(f"Export failed: {e}")
            raise e

    def predict(self, image: Union[str, Image.Image, np.ndarray], save: bool = False, output_path: str = None, conf_thres: float = 0.25, iou_thres: float = 0.45) -> dict:
        """
        Alias for __call__ method.
        
        Args:
            image: Input image. Can be a file path (str), PIL Image, or numpy array.
            save: If True, saves the image with detections drawn. Defaults to False.
            output_path: Optional path to save the annotated image.
            conf_thres: Confidence threshold (default: 0.25)
            iou_thres: IoU threshold for NMS (default: 0.45)
        
        Returns:
            Dictionary containing detection results.
        """
        return self(image=image, save=save, output_path=output_path, conf_thres=conf_thres, iou_thres=iou_thres)
