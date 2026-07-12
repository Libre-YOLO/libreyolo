"""LibreRTDETRv4 — RT-DETRv4 student detectors."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from libreyolo.training.ddp_spawn import ddp_aware

from ...training.callbacks import TrainCallbacks
from ...training.config import RTDETRv4Config
from ..dfine.model import LibreDFINE
from ..dfine.nn import LibreDFINEModel


_TRAIN_DEFAULTS = RTDETRv4Config()


class LibreRTDETRv4(LibreDFINE):
    FAMILY = "rtdetrv4"
    FILENAME_PREFIX = "LibreRTDETRv4"
    INPUT_SIZES = {"s": 640, "m": 640, "l": 640, "x": 640}
    # D-FINE grew a segment task (D-FINE-seg mask head); RT-DETRv4 has no mask
    # head, so pin the inherited task surface back to detect-only.
    SUPPORTED_TASKS = ("detect",)
    TASK_INPUT_SIZES = {}
    TRAIN_CONFIG = RTDETRv4Config

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # Disambiguates from D-FINE via the factory's ``model_family`` metadata
        # gate; raw upstream ckpts still carry ``feature_projector`` keys.
        return LibreDFINE.can_load(weights_dict) or any(
            "feature_projector" in k for k in weights_dict
        )

    @classmethod
    def convert_upstream_state_dict(cls, weights_dict: dict) -> Optional[dict]:
        """Strip training-only tensors from raw upstream RT-DETRv4 checkpoints.

        Claims only checkpoints carrying ``feature_projector`` tensors — the
        one marker that separates raw v4 files from their D-FINE/DEIM
        siblings. Bare native-keyed dicts are indistinguishable from D-FINE
        and stay on the factory's existing dispatch rules.
        """
        from .convert import drop_training_only_keys, has_training_only_keys

        if not has_training_only_keys(weights_dict):
            return None
        cleaned, _dropped = drop_training_only_keys(weights_dict)
        return cleaned if cls.can_load(cleaned) else None

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        detected = super().detect_size_from_filename(filename)
        if detected is not None:
            return detected
        m = re.search(r"rtv4(?:_hgnetv2)?_([smlx])(?:_|\.|$)", filename.lower())
        if m:
            return m.group(1)
        return None

    def _init_model(self) -> nn.Module:
        return LibreDFINEModel(
            config=self.size,
            nb_classes=self.nb_classes,
            eval_spatial_size=(self.input_size, self.input_size),
            activation="silu",
        )

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = _TRAIN_DEFAULTS.epochs,
        batch: int = 16,
        imgsz: int = 640,
        lr0: float = _TRAIN_DEFAULTS.lr0,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = _TRAIN_DEFAULTS.name,
        exist_ok: bool = False,
        resume: str | Path | bool = False,
        amp: bool = False,
        patience: int = 50,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Fine-tune or train RT-DETRv4 on a YOLO-format dataset config.

        Args:
            data: Path to the dataset YAML file.
            epochs: Number of epochs to train.
            batch: Batch size.
            imgsz: Input image size.
            lr0: Initial learning rate.
            device: Device to train on ('' = auto-detect).
            workers: Number of dataloader workers.
            seed: Random seed for reproducibility.
            project: Root directory for training runs.
            name: Experiment name.
            exist_ok: If True, overwrite existing experiment directory.
            resume: Checkpoint path, or True to resume the loaded checkpoint.
            amp: Enable automatic mixed precision training.
            patience: Early stopping patience.
            callbacks: Optional training callback or iterable of callbacks.
            loggers: Optional built-in experiment loggers: a name
                ('tensorboard', 'mlflow', 'wandb'), a configured logger
                instance, or an iterable mixing both.
        """
        from libreyolo.data import load_data_config

        from .trainer import RTDETRv4Trainer

        try:
            data_config = load_data_config(data, autodownload=True)
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")
        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)
        if yaml_names is not None:
            if isinstance(yaml_names, list):
                yaml_names = {i: n for i, n in enumerate(yaml_names)}
            self.names = self._sanitize_names(yaml_names, self.nb_classes)

        if seed >= 0:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if str(device).lower() not in ("cpu", "mps") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        resume_path = None
        if resume:
            checkpoint = self.model_path if resume is True else resume
            if checkpoint is None:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreRTDETRv4('path/to/last.pt'); "
                    "model.train(data=..., resume=True)"
                )
            resume_path = Path(checkpoint).expanduser()
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        trainer = RTDETRv4Trainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=bool(resume),
            amp=amp,
            patience=patience,
            callbacks=callbacks,
            loggers=loggers,
            **kwargs,
        )

        if resume_path is not None:
            trainer.resume(str(resume_path))

        results = trainer.train()
        self._restore_after_training(results)

        return results
