"""Validation configuration for LibreYOLO."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import yaml


@dataclass
class ValidationConfig:
    """
    Configuration for model validation.

    Attributes:
        data: Path to data.yaml file containing dataset configuration.
        data_dir: Direct path to dataset directory (alternative to data).
        split: Dataset split to validate on ("val" or "test").
        batch_size: Batch size for validation.
        imgsz: Image size for validation. Accepts an int (square) or (height, width) tuple.
        conf_thres: Confidence threshold. Use 0.0 or a low value for mAP calculation.
        iou_thres: IoU threshold for NMS.
        max_det: Maximum detections per image.
        iou_thresholds: IoU thresholds for mAP calculation (default: 0.50 to 0.95).
        device: Device to use ("auto", "cuda", "mps", "cpu").
        save_dir: Directory to save results.
        save_json: Whether to save predictions in COCO JSON format.
        save_plots: Whether to save validation plots (metrics bar, per-class AP,
            confusion matrix, sample images). Default False.
        verbose: Whether to print detailed metrics.
        num_workers: Number of dataloader workers.
        half: Whether to use FP16 inference.
    """

    # Data
    data: Optional[str] = None
    data_dir: Optional[str] = None
    split: str = "val"

    # Inference
    batch_size: int = 16
    imgsz: Union[int, Tuple[int, int]] = 640
    conf_thres: float = 0.001
    iou_thres: float = 0.6
    max_det: int = 300

    # Metrics
    # NOTE: iou_thresholds is only honored on the OBB validation path. The COCO
    # (detect/segment) path evaluates through pycocotools, which is locked to
    # its own default IoU sweep (0.50:0.05:0.95) via coco_eval.params.iouThrs
    # and ignores this value.
    iou_thresholds: Tuple[float, ...] = (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    )

    # Device
    device: str = "auto"

    # Output
    save_dir: Optional[str] = None
    save_json: bool = False
    verbose: bool = True
    save_plots: bool = field(default=False, kw_only=True)

    # Workers
    num_workers: int = 4

    # Precision
    half: bool = False
    allow_download_scripts: bool = False

    # TTA
    augment: bool = False

    # Pose validation
    keypoints_json: Optional[str] = None
    images_dir: Optional[str] = None
    oks_sigmas: Optional[List[float]] = None

    # Point validation
    dist_thresholds: Optional[List[float]] = None

    # Edge validation (BSDS-style normalized correspondence tolerance)
    edge_max_dist: float = 0.0075
    edge_thresholds: Tuple[float, ...] = field(
        default_factory=lambda: tuple(index / 100.0 for index in range(1, 100))
    )

    def __post_init__(self) -> None:
        if self.data is None and self.data_dir is None and self.keypoints_json is None:
            raise ValueError(
                "Specify one of: 'data' (yaml), 'data_dir' (detection/segmentation), "
                "or 'data' / 'keypoints_json' + 'images_dir' (pose)"
            )

        if self.split not in ("val", "test", "train"):
            raise ValueError(
                f"Invalid split: {self.split}. Must be 'val', 'test', or 'train'"
            )

        if not 0 <= self.conf_thres < 1:
            raise ValueError(f"conf_thres must be in [0, 1), got {self.conf_thres}")

        if not 0 < self.iou_thres < 1:
            raise ValueError(f"iou_thres must be in (0, 1), got {self.iou_thres}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if not 0.0 <= self.edge_max_dist <= 1.0:
            raise ValueError(
                f"edge_max_dist must be in [0, 1], got {self.edge_max_dist}"
            )
        self.edge_thresholds = tuple(float(value) for value in self.edge_thresholds)
        if not self.edge_thresholds or any(
            not 0.0 <= value <= 1.0 for value in self.edge_thresholds
        ):
            raise ValueError(
                "edge_thresholds must contain one or more values in [0, 1]"
            )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ValidationConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        if "iou_thresholds" in config_dict and isinstance(
            config_dict["iou_thresholds"], list
        ):
            config_dict["iou_thresholds"] = tuple(config_dict["iou_thresholds"])
        if "edge_thresholds" in config_dict and isinstance(
            config_dict["edge_thresholds"], list
        ):
            config_dict["edge_thresholds"] = tuple(config_dict["edge_thresholds"])

        return cls(**config_dict)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()
        config_dict["iou_thresholds"] = list(config_dict["iou_thresholds"])
        config_dict["edge_thresholds"] = list(config_dict["edge_thresholds"])

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    def update(self, **kwargs) -> "ValidationConfig":
        """Create new configuration with updated values."""
        current = self.to_dict()
        current.update(kwargs)

        if isinstance(current.get("iou_thresholds"), list):
            current["iou_thresholds"] = tuple(current["iou_thresholds"])
        if isinstance(current.get("edge_thresholds"), list):
            current["edge_thresholds"] = tuple(current["edge_thresholds"])

        return ValidationConfig(**current)
