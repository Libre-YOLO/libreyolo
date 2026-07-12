"""LibreYOLO YOLO-NAS wrapper (detect + pose)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from libreyolo.training.ddp_spawn import ddp_aware

from ...training.callbacks import TrainCallbacks
from ..base import BaseModel
from ...tasks import normalize_task
from ...utils.image_loader import ImageInput
from ...utils.serialization import load_untrusted_torch_file
from ...validation.preprocessors import YOLONASValPreprocessor
from .nn import LibreYOLONASModel, LibreYOLONASPoseModel
from ...postprocess.yolonas import postprocess, postprocess_pose
from .utils import (
    preprocess_image,
    preprocess_pose_image,
    unwrap_yolonas_checkpoint,
)

logger = logging.getLogger(__name__)

_POSE_HEAD_KEY = "heads.head1.pose_pred.weight"


class LibreYOLONAS(BaseModel):
    FAMILY = "yolonas"
    FILENAME_PREFIX = "LibreYOLONAS"
    INPUT_SIZES = {"s": 640, "m": 640, "l": 640}
    POSE_INPUT_SIZES = {"n": 640, "s": 640, "m": 640, "l": 640}
    SUPPORTED_TASKS = ("detect", "pose")
    DEFAULT_TASK = "detect"
    TASK_INPUT_SIZES = {
        "detect": INPUT_SIZES,
        "pose": POSE_INPUT_SIZES,
    }
    POSE_NUM_KEYPOINTS = 17
    KEYPOINT_DIM = 3
    val_preprocessor_class = YOLONASValPreprocessor

    _REQUIRED_SIGNATURE_KEYS = (
        "backbone.stem.conv.branch_3x3.conv.weight",
        "backbone.stem.conv.branch_1x1.weight",
        "backbone.stem.conv.rbr_reparam.weight",
        "heads.head1.cls_pred.weight",
        "heads.head1.reg_pred.weight",
    )
    _SIZE_FROM_HEAD_WIDTH = {64: "s", 96: "m", 128: "l"}
    _SIZE_FROM_HEAD_WIDTH_POSE = {48: "n", 64: "s", 96: "m", 128: "l"}
    _NUM_CLASSES_KEY = "heads.head1.cls_pred.weight"

    _DECI_CDN_BASE = "https://d2gjn4b69gu75n.cloudfront.net/models"

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return all(key in weights_dict for key in cls._REQUIRED_SIGNATURE_KEYS)

    @classmethod
    def is_pose_state_dict(cls, weights_dict: dict) -> bool:
        return _POSE_HEAD_KEY in weights_dict

    @classmethod
    def detect_checkpoint_task(cls, weights_dict: dict) -> Optional[str]:
        return "pose" if cls.is_pose_state_dict(weights_dict) else None

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        # Accept the LibreYOLO convention (LibreYOLONAS<size>.pt) handled by the
        # base regex, and also the native Deci filenames the CDN serves
        # (yolo_nas_<size>_coco.pth / yolo_nas_pose_<size>_coco_pose.pth) so
        # locally staged native checkpoints resolve to the right model size.
        size = super().detect_size_from_filename(filename)
        if size is not None:
            return size
        match = re.search(r"yolo_nas(?:_pose)?_([nsml])_coco", filename.lower())
        return match.group(1) if match else None

    @classmethod
    def detect_task_from_filename(cls, filename: str) -> Optional[str]:
        # Native Deci pose checkpoints are named yolo_nas_pose_<size>_coco_pose.pth.
        # Detect that here so local checkpoints route to the pose architecture.
        if re.search(r"yolo_nas_pose_[nsml]_coco", filename.lower()):
            return "pose"
        return super().detect_task_from_filename(filename)

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        # YOLO-NAS weights are under Deci's proprietary license — LibreYOLO
        # links to Deci's public CDN instead of mirroring on its own HF org.
        size = cls.detect_size_from_filename(filename)
        if size is None:
            return None
        task = cls.detect_task_from_filename(filename)
        if task == "pose":
            if size not in cls.POSE_INPUT_SIZES:
                return None
            return f"{cls._DECI_CDN_BASE}/yolo_nas_pose_{size}_coco_pose.pth"
        if size not in cls.INPUT_SIZES:
            return None
        return f"{cls._DECI_CDN_BASE}/yolo_nas_{size}_coco.pth"

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        tensor = weights_dict.get(cls._NUM_CLASSES_KEY)
        if tensor is None or tensor.ndim < 2:
            return None
        size_map = (
            cls._SIZE_FROM_HEAD_WIDTH_POSE
            if cls.is_pose_state_dict(weights_dict)
            else cls._SIZE_FROM_HEAD_WIDTH
        )
        return size_map.get(tensor.shape[1])

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        tensor = weights_dict.get(cls._NUM_CLASSES_KEY)
        if tensor is None or tensor.ndim == 0:
            return None
        if cls.is_pose_state_dict(weights_dict):
            # The pose cls head fuses num_classes class scores with K
            # per-keypoint visibility logits: out_channels == num_classes + K.
            # Recover the class count by subtracting K (from pose_pred). A
            # single-class COCO checkpoint yields (1 + 17) - 17 = 1.
            num_keypoints = cls.detect_num_keypoints(weights_dict)
            if num_keypoints is None:
                return 1
            return max(1, int(tensor.shape[0]) - int(num_keypoints))
        return int(tensor.shape[0])

    @classmethod
    def detect_num_keypoints(cls, weights_dict: dict) -> Optional[int]:
        """Infer the keypoint count from a pose state dict.

        ``pose_pred`` emits ``2 * K`` channels (x, y per keypoint).
        """
        tensor = weights_dict.get(_POSE_HEAD_KEY)
        if tensor is None or tensor.ndim == 0:
            return None
        return int(tensor.shape[0]) // 2

    @staticmethod
    def _detect_pose(model_path) -> bool:
        if not isinstance(model_path, str):
            return False
        try:
            ckpt = load_untrusted_torch_file(
                model_path, map_location="cpu", context="YOLO-NAS task probe"
            )
            if isinstance(ckpt, dict) and isinstance(ckpt.get("task"), str):
                return normalize_task(ckpt["task"]) == "pose"
            state = unwrap_yolonas_checkpoint(ckpt)
            return _POSE_HEAD_KEY in state
        except Exception:
            return False

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        reg_max: int = 16,
        task: str | None = None,
        **kwargs,
    ):
        resolved_task = normalize_task(task) if task is not None else None
        has_checkpoint_names = bool(
            isinstance(model_path, dict) and model_path.get("names") is not None
        )
        self.reg_max = reg_max
        # Default keypoint count; overridden from checkpoint metadata/state
        # before model construction or from dataset kpt_shape in train().
        self.num_keypoints = self.POSE_NUM_KEYPOINTS
        self.keypoint_dim = self.KEYPOINT_DIM
        if isinstance(model_path, dict):
            state = unwrap_yolonas_checkpoint(model_path)
            if not (
                {"schema_version", "model_family", "task", "nc", "names"}
                & set(model_path)
            ):
                model_path = state
            if resolved_task == "pose":
                ckpt_k = self.detect_num_keypoints(state)
                if ckpt_k is not None:
                    self.num_keypoints = ckpt_k
                # Recover the class count from the state dict so a multi-class
                # pose checkpoint builds the right head width.
                ckpt_nc = self.detect_nb_classes(state)
                nb_classes = ckpt_nc if ckpt_nc is not None else 1
        elif resolved_task == "pose":
            # Fresh model or file path: default to single-class (person) pose.
            # A file path's real class count is resolved in _load_weights;
            # training resolves it from the dataset yaml.
            nb_classes = 1
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )
        if self.task == "pose" and not has_checkpoint_names:
            # Placeholder names; overridden by checkpoint metadata in
            # _load_weights or by the dataset yaml at train() time.
            if self.nb_classes == 1:
                self.names = {0: "person"}
            else:
                self.names = {i: str(i) for i in range(self.nb_classes)}
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        if self.task == "pose":
            return LibreYOLONASPoseModel(
                config=self.size,
                num_keypoints=self.num_keypoints,
                num_classes=self.nb_classes,
                reg_max=self.reg_max,
            )
        return LibreYOLONASModel(
            config=self.size,
            nb_classes=self.nb_classes,
            reg_max=self.reg_max,
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone_stem": self.model.backbone.stem,
            "backbone_stage1": self.model.backbone.stage1,
            "backbone_stage2": self.model.backbone.stage2,
            "backbone_stage3": self.model.backbone.stage3,
            "backbone_stage4": self.model.backbone.stage4,
            "backbone_context_module": self.model.backbone.context_module,
            "neck1": self.model.neck.neck1,
            "neck2": self.model.neck.neck2,
            "neck3": self.model.neck.neck3,
            "neck4": self.model.neck.neck4,
            "heads": self.model.heads,
        }

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        self.nb_classes = new_nb_classes
        if self.task == "pose":
            # Rebuild the pose head's class channels; the per-keypoint
            # visibility channels and the rest of the model are preserved.
            self.model.replace_num_classes(new_nb_classes)
            self.model.to(self.device)
            return
        self.model.nc = new_nb_classes
        self.model.heads.replace_num_classes(new_nb_classes)
        self.model.to(self.device)

    def _rebuild_for_new_keypoints(self, new_num_keypoints: int):
        """Rebuild the pose head for a different keypoint count.

        Used to fine-tune a COCO (17-keypoint) checkpoint on a dataset with a
        different number of keypoints: the backbone, neck and box layers keep
        their pretrained weights; only the keypoint-dependent head layers are
        reinitialised.
        """
        if self.task != "pose":
            return
        if new_num_keypoints == self.num_keypoints:
            return
        self.model.replace_num_keypoints(new_num_keypoints)
        self.num_keypoints = new_num_keypoints
        self.model.to(self.device)

    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy

        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        effective_size = input_size if input_size is not None else self.input_size
        if self.task == "pose":
            return preprocess_pose_image(
                image,
                input_size=effective_size,
                color_format=color_format,
            )
        return preprocess_image(
            image,
            input_size=effective_size,
            color_format=color_format,
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        output = self.model(input_tensor)
        if self.task == "pose":
            if isinstance(output, tuple) and len(output) == 2 and isinstance(output[0], tuple):
                output = output[0]
            # Heads return the inference 4-tuple
            # (bboxes, scores, pose_xy, pose_scores).
            if isinstance(output, tuple) and len(output) == 4:
                bboxes, scores, pose_xy, pose_scores = output
                return {
                    "boxes": bboxes,
                    "scores": scores,
                    "keypoints_xy": pose_xy,
                    "keypoints_conf": pose_scores,
                }
            return output
        if isinstance(output, tuple):
            if len(output) == 2 and isinstance(output[0], tuple):
                boxes, scores = output[0]
                return {
                    "boxes": boxes,
                    "scores": scores,
                    "raw_predictions": output[1],
                }
            if len(output) == 2 and all(isinstance(x, torch.Tensor) for x in output):
                boxes, scores = output
                return {"boxes": boxes, "scores": scores}
        return output

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        actual_input_size = kwargs.get("input_size", self.input_size)
        if self.task == "pose":
            return postprocess_pose(
                output,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                input_size=actual_input_size,
                original_size=original_size,
                post_nms_max_predictions=max_det,
                letterbox=kwargs.get("letterbox", True),
            )
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            max_det=max_det,
            letterbox=kwargs.get("letterbox", True),
        )

    def _strict_loading(self) -> bool:
        return False

    # SHA-256 of the official Deci CDN checkpoints (detection + pose). Auto-downloaded
    # YOLO-NAS weights are third-party pickles that must be loaded with
    # weights_only=False, so they are verified against these pins before being
    # unpickled; a compromised/tampered CDN object then fails closed instead of
    # executing code during model construction. Locally staged files are the
    # user's own trust decision and are not re-verified.
    _DECI_CHECKPOINT_SHA256 = {
        "yolo_nas_s_coco.pth": "c1b1d9148ab8ae5d5984699547e850955ff9efccaf568c67b3d605acb4bfe1cb",
        "yolo_nas_m_coco.pth": "b194fc7fa196f76161c6356558bedf04fb99a62325a74a36a4bec3ca8ba48250",
        "yolo_nas_l_coco.pth": "91a06beaa1ce1a651d6691e3198061da996eafc8890503238dedacbd4c392a32",
        "yolo_nas_pose_n_coco_pose.pth": "3544cd4bef7a4930e79c2d9a9ec50167be6fa366be834d52d462393edfc3a64f",
        "yolo_nas_pose_s_coco_pose.pth": "54f0933cb3760c5f9ba47e901c58d6d114cd206718667a586031bea0ab9ea849",
        "yolo_nas_pose_m_coco_pose.pth": "6d0f92a589fd2f39a9fb92c42894cca76e81eaf2fcb3a00f1cac2e7089fb91ec",
        "yolo_nas_pose_l_coco_pose.pth": "d05c55157b3eb917e43d3669cc1e99fbe35a8a93c7883b95b93036a81216c5ab",
    }

    @classmethod
    def verify_downloaded_file(cls, local_path: str, source_url: str) -> None:
        """Verify a freshly auto-downloaded YOLO-NAS checkpoint before it loads.

        Called by ``download_weights`` for every download (the factory and the
        direct loader both flow through it), so the gate cannot be bypassed and
        keys on the *source* filename — the destination may use the LibreYOLO
        convention (e.g. ``weights/LibreYOLONASs.pt``) while the CDN object is
        ``yolo_nas_s_coco.pth``.
        """
        import hashlib
        from urllib.parse import urlparse

        name = Path(urlparse(source_url).path).name
        expected = cls._DECI_CHECKPOINT_SHA256.get(name)
        if expected is None:
            raise RuntimeError(
                f"Refusing to auto-load YOLO-NAS checkpoint '{name}': no pinned "
                "checksum is known for it, so this freshly downloaded third-party "
                "pickle cannot be verified before loading. Download it manually "
                "from a source you trust and pass its path instead."
            )
        digest = hashlib.sha256()
        with open(local_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for downloaded YOLO-NAS checkpoint '{name}': "
                f"expected {expected}, got {actual}. Refusing to load a possibly "
                "tampered file."
            )

    def _load_weights(self, model_path: str):
        # Newly auto-downloaded YOLO-NAS weights are checksum-pinned because
        # loading these third-party pickles requires full torch
        # deserialization. Locally staged paths are the user's trust decision
        # and download_weights deliberately leaves them untouched.
        from ...utils.download import download_weights

        download_weights(model_path, self.size)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        try:
            loaded = torch.load(model_path, map_location="cpu", weights_only=False)
            if not isinstance(loaded, dict):
                raise TypeError("YOLO-NAS checkpoints must be dictionaries")
            loaded, is_native_v1 = self._parse_checkpoint_metadata(
                loaded,
                context=f"YOLO-NAS weights from {model_path}",
            )
            state_dict = dict(unwrap_yolonas_checkpoint(loaded))
            if not is_native_v1:
                state_dict = self._prepare_state_dict(
                    self._strip_ddp_prefix(state_dict)
                )

            ckpt_is_pose = self.is_pose_state_dict(state_dict)
            tensor_task = "pose" if ckpt_is_pose else "detect"
            declared_task = loaded.get("task")
            if declared_task is not None and normalize_task(declared_task) != tensor_task:
                raise RuntimeError(
                    f"Checkpoint metadata declares task={normalize_task(declared_task)!r}, "
                    f"but its tensors identify a YOLO-NAS {tensor_task} model."
                )
            if ckpt_is_pose and self.task != "pose":
                raise RuntimeError(
                    "Checkpoint is a YOLO-NAS pose model but this instance was "
                    "initialized for detection. Pass task='pose' or use a "
                    "detection checkpoint."
                )
            if not ckpt_is_pose and self.task == "pose":
                raise RuntimeError(
                    "Checkpoint is a YOLO-NAS detection model but this instance "
                    "was initialized for pose. Pass task='detect' or use a pose "
                    "checkpoint."
                )

            # Match the pose head to the checkpoint's keypoint count before
            # loading (e.g. a 4-keypoint fine-tune of a COCO-17 model).
            if ckpt_is_pose:
                ckpt_k = self.detect_num_keypoints(state_dict)
                if ckpt_k is not None and ckpt_k != self.num_keypoints:
                    self._rebuild_for_new_keypoints(ckpt_k)
                # Match the class count from the state-dict shapes, covering
                # bare pose checkpoints that carry no ``nc`` metadata (the
                # metadata path below is a no-op when they already agree).
                ckpt_nc = self.detect_nb_classes(state_dict)
                if ckpt_nc is not None and ckpt_nc != self.nb_classes:
                    self._rebuild_for_new_classes(ckpt_nc)

            if isinstance(loaded, dict):
                ckpt_family = loaded.get("model_family", "")
                own_family = self._get_model_name()
                if ckpt_family and ckpt_family != own_family:
                    raise RuntimeError(
                        f"Checkpoint was trained with model_family='{ckpt_family}' "
                        f"but is being loaded into '{own_family}'. "
                        f"Use the correct model class for this checkpoint."
                    )

                ckpt_names = loaded.get("names")
                ckpt_nc = self._normalize_checkpoint_nc(loaded.get("nc"))
                if ckpt_nc is None and ckpt_names is not None:
                    if not isinstance(ckpt_names, (dict, list)):
                        raise ValueError(
                            "checkpoint names must be a dict[int, str] or list[str]"
                        )
                    ckpt_nc = len(ckpt_names)
                if ckpt_nc is None:
                    ckpt_nc = self._normalize_checkpoint_nc(
                        self.detect_nb_classes(state_dict)
                    )
                if ckpt_nc is not None and ckpt_nc != self.nb_classes:
                    self._rebuild_for_new_classes(int(ckpt_nc))

                effective_nc = int(ckpt_nc) if ckpt_nc is not None else self.nb_classes
                if ckpt_names is not None:
                    self.names = self._sanitize_names(ckpt_names, effective_nc)

            self._load_state_dict_checked(
                state_dict,
                checkpoint=loaded if isinstance(loaded, dict) else None,
                checkpoint_task=tensor_task,
                context=f"YOLO-NAS weights from {model_path}",
            )
            self.model.to(self.device).eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO-NAS weights from {model_path}: {e}"
            ) from e

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: Optional[int] = None,
        batch: int = 16,
        imgsz: int = 640,
        lr0: Optional[float] = None,
        optimizer: str = "AdamW",
        device: str = "",
        workers: int = 8,
        seed: int = 0,
        project: str = "runs/train",
        name: Optional[str] = None,
        exist_ok: bool = False,
        resume: bool = False,
        amp: Optional[bool] = None,
        patience: int = 50,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Train the YOLO-NAS model on a YOLO-format dataset.

        Args:
            data: Path to the dataset YAML file.
            epochs: Number of epochs to train (None uses the task default).
            batch: Batch size.
            imgsz: Input image size.
            lr0: Initial learning rate (None uses the task default).
            optimizer: Optimizer name ('SGD', 'Adam', 'AdamW').
            device: Device to train on ('' = auto-detect).
            workers: Number of dataloader workers.
            seed: Random seed for reproducibility.
            project: Root directory for training runs.
            name: Experiment name (None uses the task default).
            exist_ok: If True, overwrite existing experiment directory.
            resume: If True, resume training from the loaded checkpoint.
            amp: Enable automatic mixed precision training (None uses the
                task default).
            patience: Early stopping patience.
            callbacks: Optional training callback or iterable of callbacks.
            loggers: Optional built-in experiment loggers: a name
                ('tensorboard', 'mlflow', 'wandb'), a configured logger
                instance, or an iterable mixing both.
        """
        # Task-specific defaults for arguments left unset by the caller.
        if lr0 is None:
            lr0 = 2e-3 if self.task == "pose" else 5e-4
        if epochs is None:
            epochs = 1000 if self.task == "pose" else 300
        if amp is None:
            amp = self.task == "pose"

        if self.task == "pose":
            return self._train_pose(
                data,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                lr0=lr0,
                optimizer=optimizer,
                device=device,
                workers=workers,
                seed=seed,
                project=project,
                name=name or "yolonas_pose_exp",
                exist_ok=exist_ok,
                resume=resume,
                amp=amp,
                patience=patience,
                callbacks=callbacks,
                loggers=loggers,
                **kwargs,
            )

        name = name or "yolonas_exp"
        from libreyolo.data import load_data_config

        from .trainer import YOLONASTrainer

        try:
            data_config = load_data_config(data, autodownload=True)
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")
        # If no nc in data.yaml, infer it by counting.
        if yaml_nc is None and yaml_names is not None:
            yaml_nc = len(yaml_names)
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

        trainer = YOLONASTrainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            optimizer=optimizer.lower(),
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            patience=patience,
            callbacks=callbacks,
            loggers=loggers,
            **kwargs,
        )

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreYOLONAS('path/to/last.pt'); model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))
            return trainer.train()

        results = trainer.train()

        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self.model_path = best_ckpt
            self._load_weights(best_ckpt)
            self.model.eval()

        return results

    def _train_pose(
        self,
        data: str,
        *,
        epochs: int,
        batch: int,
        imgsz: int,
        lr0: float,
        optimizer: str,
        device: str,
        workers: int,
        seed: int,
        project: str,
        name: str,
        exist_ok: bool,
        resume: bool,
        amp: bool,
        patience: int,
        callbacks=None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Train the YOLO-NAS pose head on a YOLO-format keypoint dataset.

        The dataset ``data.yaml`` must declare ``kpt_shape: [num_keypoints, 2|3]``.
        If the keypoint count differs from the loaded checkpoint, the pose head
        is rebuilt for the new count while the backbone/neck keep their
        pretrained weights.
        """
        from libreyolo.data import load_data_config

        from .pose_trainer import YOLONASPoseTrainer

        try:
            data_config = load_data_config(data, autodownload=True)
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        kpt_shape = data_config.get("kpt_shape")
        if not kpt_shape or len(kpt_shape) < 1:
            raise ValueError(
                "Pose training requires 'kpt_shape: [num_keypoints, 2|3]' in the "
                "dataset data.yaml."
            )
        num_keypoints = int(kpt_shape[0])
        keypoint_dim = int(kpt_shape[1]) if len(kpt_shape) > 1 else 3
        if keypoint_dim not in (2, 3):
            raise ValueError(
                "Pose training requires kpt_shape second value to be 2 or 3 "
                f"(got {keypoint_dim})."
            )

        # Resolve the class count from the dataset yaml. Multi-class pose uses a
        # single shared keypoint skeleton (one kpt_shape for every class), so nc
        # only affects the class/box branch, not the keypoints.
        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")
        if yaml_nc is None and yaml_names is not None:
            yaml_nc = len(yaml_names)

        # Rebuild the pose head for the dataset's keypoint count first, then its
        # class count (both touch cls_pred; this order carries the freshly sized
        # visibility channels through the class rebuild).
        if num_keypoints != self.num_keypoints:
            logger.info(
                "Rebuilding YOLO-NAS pose head for %d keypoints (was %d)",
                num_keypoints,
                self.num_keypoints,
            )
            self._rebuild_for_new_keypoints(num_keypoints)
        if yaml_nc is not None and int(yaml_nc) != self.nb_classes:
            logger.info(
                "Rebuilding YOLO-NAS pose head for %d classes (was %d)",
                int(yaml_nc),
                self.nb_classes,
            )
            self._rebuild_for_new_classes(int(yaml_nc))

        # Carry the dataset's class names into checkpoints.
        if yaml_names is not None:
            if isinstance(yaml_names, list):
                yaml_names = {i: n for i, n in enumerate(yaml_names)}
            self.names = self._sanitize_names(yaml_names, self.nb_classes)

        if self.size in {"m", "l"}:
            kwargs.setdefault("dfl_loss_weight", 0.5)
            kwargs.setdefault("pose_reg_loss_weight", 10.0)

        if seed >= 0:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if str(device).lower() not in ("cpu", "mps") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = YOLONASPoseTrainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            num_keypoints=num_keypoints,
            keypoint_dim=keypoint_dim,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            optimizer=optimizer.lower(),
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            patience=patience,
            callbacks=callbacks,
            loggers=loggers,
            **kwargs,
        )

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreYOLONAS('path/to/last.pt', task='pose'); "
                    "model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))
            return trainer.train()

        results = trainer.train()

        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self.model_path = best_ckpt
            self._load_weights(best_ckpt)
            self.model.eval()

        return results
