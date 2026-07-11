"""LibrePatchCore training-free visual anomaly detector."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.patchcore import postprocess as patchcore_postprocess
from ...utils.general import increment_path
from ...utils.image_loader import ImageInput
from ...utils.serialization import load_untrusted_torch_file, wrap_libreyolo_checkpoint
from ..base import BaseModel
from .config import PatchCoreConfig
from .nn import PatchCoreNet, greedy_coreset
from .utils import (
    iter_preprocessed_batches,
    preprocess_image,
    preprocess_numpy,
    resolve_anomaly_test_samples,
    resolve_good_training_images,
)

logger = logging.getLogger(__name__)


class LibrePatchCore(BaseModel):
    """PatchCore anomaly detection with a frozen WideResNet-50-2 backbone."""

    FAMILY = "patchcore"
    FILENAME_PREFIX = "LibrePatchCore"
    INPUT_SIZES: ClassVar[dict[str, int]] = {"b": 224}
    SUPPORTED_TASKS = ("anomaly",)
    DEFAULT_TASK = "anomaly"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = PatchCoreConfig
    TTA_ENABLED = False
    SUPPORTS_BATCHED_PREDICT = True

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        bank = weights_dict.get("memory_bank")
        return (
            isinstance(bank, torch.Tensor)
            and bank.ndim == 2
            and bank.shape[1] == 1536
            and "anomaly_threshold" in weights_dict
            and "fitted" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return "b" if cls.can_load(weights_dict) else None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "anomaly" if cls.can_load(state_dict) else None

    @classmethod
    def format_weight_filename(cls, size_code: str) -> str:
        return f"{cls.FILENAME_PREFIX}{size_code}-anomaly{cls.WEIGHT_EXT}"

    def __init__(
        self,
        model_path=None,
        size: str = "b",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        pretrained: bool = True,
        reweight_neighbors: int = 9,
        query_chunk_size: int = 2048,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            pretrained=pretrained,
            reweight_neighbors=reweight_neighbors,
            query_chunk_size=query_chunk_size,
            **kwargs,
        )
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
            self._load_patchcore_metadata(str(model_path))
        self.nb_classes = 1
        self.names = {0: "anomaly"}

    def _init_model(self) -> nn.Module:
        use_pretrained = bool(getattr(self, "pretrained", True)) and not bool(
            getattr(self, "_in_rebuild", False)
        )
        return PatchCoreNet(
            pretrained=use_pretrained,
            reweight_neighbors=int(getattr(self, "reweight_neighbors", 9)),
            query_chunk_size=int(getattr(self, "query_chunk_size", 2048)),
        )

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        bank = state_dict.get("memory_bank")
        if isinstance(bank, torch.Tensor) and bank.ndim == 2:
            self.model.memory_bank = torch.empty_like(bank)
        return state_dict

    def _strict_loading(self) -> bool:
        # Category checkpoints intentionally omit the frozen torchvision
        # backbone. It is restored from torchvision when the wrapper is built.
        return False

    def _load_patchcore_metadata(self, model_path: str) -> None:
        checkpoint = load_untrusted_torch_file(
            model_path, map_location="cpu", context="PatchCore metadata"
        )
        if not isinstance(checkpoint, dict):
            return
        self.backbone_id = str(checkpoint.get("backbone", "wide_resnet50_2"))
        self.feature_layers = tuple(checkpoint.get("feature_layers", ("layer2", "layer3")))
        self.model.reweight_neighbors = int(checkpoint.get("reweight_neighbors", 9))
        self.model.query_chunk_size = int(checkpoint.get("query_chunk_size", 2048))

    @property
    def threshold(self) -> float | None:
        value = float(self.model.anomaly_threshold.detach().cpu())
        return value if torch.isfinite(torch.tensor(value)) else None

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"backbone": self.model.backbone, "layer2": self.model.backbone.layer2, "layer3": self.model.backbone.layer3}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        return preprocess_image(
            image, input_size=input_size or self.input_size, color_format=color_format
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor.float())

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        del iou_thres, max_det, ratio, kwargs
        threshold = self.threshold
        if conf_thres is not None and conf_thres != 0.25:
            threshold = float(conf_thres)
        return patchcore_postprocess(
            output, original_size=original_size, threshold=threshold, sigma=4.0
        )

    def _extract_dataset_features(self, images: list[Path], batch: int) -> torch.Tensor:
        features: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for tensors in iter_preprocessed_batches(images, batch, self.input_size):
                grid = self.model.extract_features(tensors.to(self.device).float())
                features.append(grid.reshape(-1, grid.shape[-1]).cpu())
        return torch.cat(features, dim=0)

    def _calibrate_threshold(self, images: list[Path], batch: int) -> float:
        scores: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for tensors in iter_preprocessed_batches(images, batch, self.input_size):
                output = self.model(tensors.to(self.device).float())
                scores.append(output["image_scores"].detach().cpu())
        return float(torch.cat(scores).max())

    def train(
        self,
        data: str,
        *,
        batch: int = 8,
        imgsz: int | None = None,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/anomaly/train",
        name: str = "patchcore",
        exist_ok: bool = False,
        coreset: float = 10.0,
        projection_dim: int = 128,
        reweight_neighbors: int = 9,
        query_chunk_size: int = 2048,
        epochs: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fit the memory bank from good images; no optimizer or epochs are used."""
        del workers, kwargs
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("LibrePatchCore fit is single-process only; DDP is not supported.")
        if epochs not in (None, 0, 1):
            logger.warning("LibrePatchCore is training-free; epochs=%s is ignored.", epochs)
        if imgsz is not None and int(imgsz) != self.input_size:
            raise ValueError(f"LibrePatchCoreb uses imgsz={self.input_size}; got {imgsz}.")
        if device and str(device).lower() not in {"auto", ""}:
            self.device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
            self.model.to(self.device)
        if batch < 1:
            raise ValueError("batch must be at least 1.")
        self.model.reweight_neighbors = int(reweight_neighbors)
        self.model.query_chunk_size = int(query_chunk_size)

        train_images = resolve_good_training_images(data)
        features = self._extract_dataset_features(train_images, batch)
        bank = greedy_coreset(
            features, percent=float(coreset), projection_dim=int(projection_dim), seed=int(seed)
        )
        self.model.set_memory_bank(bank.to(self.device), float(coreset))

        try:
            good_test = [
                path for path, label, _ in resolve_anomaly_test_samples(data) if label == 0
            ]
        except ValueError:
            good_test = []
        calibration_images = good_test or train_images
        threshold = self._calibrate_threshold(calibration_images, batch)
        self.model.anomaly_threshold.fill_(threshold)

        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = wrap_libreyolo_checkpoint(
            {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
                if not key.startswith("backbone.")
            },
            model_family=self.FAMILY,
            size=self.size,
            task=self.task,
            nc=1,
            names={0: "anomaly"},
            imgsz=self.input_size,
            backbone="wide_resnet50_2",
            backbone_weights="torchvision:Wide_ResNet50_2_Weights.IMAGENET1K_V2",
            feature_layers=["layer2", "layer3"],
            coreset_percent=float(coreset),
            projection_dim=int(projection_dim),
            reweight_neighbors=int(reweight_neighbors),
            query_chunk_size=int(query_chunk_size),
            calibrated_threshold=threshold,
            calibration_split="test/good" if good_test else "train/good",
            training_images=len(train_images),
        )
        best_path = weights_dir / "best.pt"
        last_path = weights_dir / "last.pt"
        torch.save(checkpoint, best_path)
        torch.save(checkpoint, last_path)
        self.model_path = str(best_path)
        self.model.eval()
        return {
            "save_dir": str(save_dir),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "memory_bank_size": int(bank.shape[0]),
            "training_patches": int(features.shape[0]),
            "threshold": threshold,
        }

    def export(self, format: str = "onnx", **kwargs) -> str:
        del format, kwargs
        raise NotImplementedError(
            "LibrePatchCore export is not supported in v1 because its dynamic memory bank and kNN search have no runtime contract."
        )


__all__ = ["LibrePatchCore"]
