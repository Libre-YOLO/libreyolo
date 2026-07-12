"""Base trainer for LibreYOLO models.

Model-specific trainers subclass BaseTrainer and override hooks.
"""

import contextlib
import inspect
import logging
import math
import os
import random
import shutil
import sys
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .artifacts import TrainingArtifactsCallback, TrainingStatusCallback
from .callbacks import (
    TrainCallbackList,
    TrainCallbacks,
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from .config import TrainConfig
from .loggers import resolve_loggers
from .distributed import (
    all_reduce_avg_scalar,
    get_local_rank,
    get_rank,
    get_world_size,
    has_torchrun_env,
    init_distributed,
    is_distributed,
    is_main_process,
    parse_device_arg,
    run_rank_zero_phase,
    scale_loss_for_ddp,
    seed_for_rank,
    unwrap_model,
    wants_distributed,
)
from .ema import ModelEMA
from .freezing import FreezeGroup, apply_freeze, default_freeze_groups
from ..data.dataset import YOLODataset, COCODataset, create_dataloader
from ..data import (
    dataloader_seed_kwargs,
    distributed_sampler_seed,
    get_coco_annotation_file,
    get_coco_image_dir,
    get_img_files,
    img2label_paths,
    load_data_config,
    resolve_default_coco_image_dir,
)
from ..tasks import normalize_task
from ..utils.serialization import (
    REQUIRED_CHECKPOINT_METADATA_KEYS,
    SCHEMA_VERSION,
    build_class_names,
    load_trusted_torch_file,
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)


logger = logging.getLogger(__name__)

_TRAINING_CHECKPOINT_CORE_KEYS = set(REQUIRED_CHECKPOINT_METADATA_KEYS) | {
    "epoch",
    "optimizer",
    "config",
    "loss",
    "best_mAP50_95",
    "best_mAP50",
    "best_metric_key",
    "best_metric_value",
    "best_epoch",
    "is_ema_weights",
    "best_metric",
    "best_metric_name",
    "train_model",
    "ema",
    "ema_updates",
    "distiller",
    "scaler",
    "rng_state",
}


def _atomic_save_checkpoint(checkpoint: Dict[str, Any], targets: List[Path]) -> None:
    """Validate once and atomically replace each target with error rollback.

    A filesystem cannot atomically replace multiple directory entries as one
    operation. If a promotion raises while this process is running, previously
    promoted targets are restored; a process or machine crash between replaces
    can still expose different generations briefly.
    """
    if not targets:
        return

    targets = [Path(target) for target in targets]
    if len(set(targets)) != len(targets):
        raise ValueError("Checkpoint targets must be unique.")
    directory = targets[0].parent
    if any(target.parent != directory for target in targets):
        raise ValueError("Atomic checkpoint targets must share one directory.")
    directory.mkdir(parents=True, exist_ok=True)

    serialized = directory / f".checkpoint.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    staged_targets: Dict[Path, Path] = {}
    backups: Dict[Path, Path | None] = {}

    def stage_file(source: Path, destination: Path) -> None:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
            with open(destination, "rb+") as file:
                file.flush()
                os.fsync(file.fileno())

    try:
        with open(serialized, "xb") as file:
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())

        reloaded = load_trusted_torch_file(
            serialized,
            map_location="cpu",
            context="newly serialized training checkpoint",
        )
        validate_checkpoint_metadata(reloaded, strict=True)
        del reloaded

        for target in targets:
            staged = directory / f".{target.name}.{uuid.uuid4().hex}.tmp"
            staged_targets[target] = staged
            stage_file(serialized, staged)

        for target in targets:
            backup = None
            if target.exists():
                backup = directory / f".{target.name}.{uuid.uuid4().hex}.bak"
                backups[target] = backup
                stage_file(target, backup)
            else:
                backups[target] = None

        promoted = []
        try:
            for target in targets:
                os.replace(staged_targets[target], target)
                promoted.append(target)
        except Exception as promotion_error:
            rollback_errors = []
            for target in reversed(promoted):
                backup = backups[target]
                try:
                    if backup is None:
                        target.unlink()
                    else:
                        os.replace(backup, target)
                        backups[target] = None
                except OSError as rollback_error:
                    if backup is not None:
                        backups[target] = None
                        rollback_errors.append(
                            f"{target}: {rollback_error}; previous file retained "
                            f"at {backup}"
                        )
                    else:
                        rollback_errors.append(f"{target}: {rollback_error}")
            if rollback_errors:
                raise RuntimeError(
                    "Checkpoint publication failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from promotion_error
            raise
    finally:
        for temporary in [*staged_targets.values(), *backups.values(), serialized]:
            if temporary is None:
                continue
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


class BaseTrainer(ABC):
    """Base trainer for all LibreYOLO model families.

    Subclasses override hook methods to customise transforms, schedulers,
    loss extraction, and family-specific behaviour.
    """

    best_metric_key: str = "metrics/mAP50-95"
    artifact_model_families: Tuple[str, ...] = ()
    # Whether this family supports ``lora=True`` fine-tuning. Overridden to True
    # by trainers with LoRA-amenable (transformer/nn.Linear) backbones.
    supports_lora: bool = False

    def __init__(
        self,
        model: nn.Module,
        wrapper_model: Optional[Any] = None,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ):
        self.config = self._config_class().from_kwargs(**kwargs)
        self.model = model
        self.wrapper_model = wrapper_model
        wrapper_explicit_keys = (
            getattr(wrapper_model, "_active_train_explicit_keys", None)
            if wrapper_model is not None
            else None
        )
        self._explicit_train_config_keys = frozenset(
            kwargs.keys()
            if wrapper_explicit_keys is None
            else wrapper_explicit_keys
        )
        self.callbacks = TrainCallbackList(callbacks)
        for logger_callback in resolve_loggers(loggers):
            self.callbacks.append(logger_callback)
        # TrainingArtifactsCallback is family-gated (results.csv / summary.json
        # only for opted-in families). TrainingStatusCallback is universal: every
        # run gets a live status.json + train.log so agents and the `libreyolo
        # monitor` web UI can watch any training without extra configuration.
        self.artifact_callbacks = TrainCallbackList(
            [
                TrainingArtifactsCallback(enabled_families=self.artifact_model_families),
                TrainingStatusCallback(),
            ]
        )

        # Distributed state. We init the process group eagerly when launched
        # under torchrun (LOCAL_RANK set in env) — this also covers the case
        # where the user passed device=[0,1] and ran with torchrun. If the
        # user passed a list-form device but did NOT launch with torchrun,
        # we raise a clear error in _setup_device pointing them at it.
        if has_torchrun_env() and not is_distributed():
            init_distributed()
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.world_size = get_world_size()
        self.is_distributed = is_distributed()

        # Seed every trainer, with a rank offset under DDP so Python, NumPy,
        # Torch, CUDA, samplers, and workers can derive one coherent stream.
        configured_seed = getattr(self.config, "seed", None)
        if configured_seed is not None and int(configured_seed) >= 0:
            seed = (
                seed_for_rank(int(configured_seed))
                if self.is_distributed
                else int(configured_seed)
            )
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Device
        self.device = self._setup_device()

        # Training state
        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.optimizer_step_count = 0
        self._optimizer_step_count_restored = False

        # Metric tracking
        self.best_mAP50_95 = 0.0
        self.best_mAP50 = 0.0
        self.best_epoch = 0
        self.final_loss = 0.0
        self.epoch_losses: List[float] = []
        self.epoch_events: List[TrainEpochEvent] = []
        self.patience_counter = 0

        # Initialised in setup()
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.ema_model = None
        self.freeze_summary = None
        self._frozen_bn_modules: Tuple[nn.Module, ...] = ()
        self.train_loader = None
        self._is_setup = False
        self.distiller = None
        self._distill_loss_val = 0.0

        # Profiling (opt-in via config.profile). None = disabled, zero overhead.
        self._profiler = None
        self._stop_training = False

    # =========================================================================
    # Config
    # =========================================================================

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        """Return the config dataclass for this trainer. Subclasses override."""
        return TrainConfig

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def effective_lr(self) -> float:
        """Optimizer base learning rate."""
        return self.config.lr0

    @property
    def _accum_steps(self) -> int:
        """Micro-batches accumulated per optimizer step (1 disables accumulation).

        Derived from ``config.nbs`` (nominal batch size) as
        ``round(nbs / batch)``. When ``nbs`` is unset the
        trainer runs the standard one-optimizer-step-per-batch loop, unchanged
        from a build without this feature.
        """
        nbs = getattr(self.config, "nbs", None)
        if nbs is None:
            return 1
        return max(1, round(nbs / self.config.batch))

    def _scheduler_steps_per_epoch(self) -> int:
        """Optimizer steps per epoch — the unit the LR schedule advances in.

        Equals ``len(train_loader)`` without accumulation; with accumulation it
        is ``ceil(len / accum)`` so the schedule still advances exactly once per
        optimizer step. Requires ``train_loader`` to be set.
        """
        steps = len(self.train_loader)
        if self._accum_steps > 1:
            steps = max(1, math.ceil(steps / self._accum_steps))
        return steps

    def _per_rank_batch_size(self) -> int:
        """Resolve the exact local batch represented by the global config."""
        batch = int(self.config.batch)
        world_size = max(int(getattr(self, "world_size", 1)), 1)
        if batch <= 0:
            raise ValueError(f"batch must be positive after setup, got {batch}")
        if world_size > 1 and batch % world_size != 0:
            raise ValueError(
                f"Global batch={batch} must be divisible by world_size={world_size}; "
                "choose a divisible batch so DDP does not silently change it"
            )
        per_rank_batch = batch // world_size
        if per_rank_batch <= 0:
            raise ValueError(
                f"Global batch={batch} is smaller than world_size={world_size}"
            )
        return per_rank_batch

    @property
    def input_size(self) -> Tuple[int, int]:
        return (self.config.imgsz, self.config.imgsz)

    # =========================================================================
    # Hook methods — subclasses override these
    # =========================================================================

    @abstractmethod
    def get_model_family(self) -> str:
        """Return canonical model family string for checkpoint metadata."""

    @abstractmethod
    def get_model_tag(self) -> str:
        """Return human-readable model tag for log messages (e.g. 'YOLOX-s')."""

    @abstractmethod
    def create_transforms(self):
        """Return (preproc_transform, mosaic_dataset_class)."""

    @abstractmethod
    def create_scheduler(self, iters_per_epoch: int):
        """Return a scheduler with an ``update_lr(iters)`` method."""

    @abstractmethod
    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        """Extract per-component losses for progress bar and epoch metrics.

        Returns:
            Dict mapping loss name → scalar value.
        """

    def on_setup(self):
        """Called after model is on device, before data setup (e.g. bias init)."""

    def get_freeze_groups(self) -> List[FreezeGroup]:
        """Return integer-addressable freeze groups for this family."""
        return default_freeze_groups(self.model)

    def preserve_freeze_param(self, name: str, param: nn.Parameter) -> bool:
        """Return True for trainable params that freezing must not disable."""
        return False

    def on_num_classes_resolved(self):
        """Called before on_setup() for trainers that pre-sync class counts."""

    def on_mosaic_disable(self):
        """Called when mosaic is disabled for final no-aug epochs."""
        dataset = getattr(self.train_loader, "dataset", None)
        if hasattr(dataset, "close_mosaic"):
            dataset.close_mosaic()

    def on_forward(
        self,
        imgs: torch.Tensor,
        targets: torch.Tensor,
        polygons: Optional[List] = None,
    ) -> Dict:
        """Run the model forward pass. Override if call signature differs.

        When ``load_segments=True`` is enabled, ``polygons`` follows the shared
        preservation contract:

        - list length equals batch size
        - each image entry is a list of instances matching that image's target rows
        - each instance is a list of polygon rings
        - each ring is an ``Nx2`` array in original image pixel coordinates

        Detection rows without polygon labels use an empty ring list for that
        instance. Detection-only trainers may ignore ``polygons``.
        """
        return self.model(imgs, targets)

    # =========================================================================
    # Shared infrastructure
    # =========================================================================

    def _setup_device(self) -> torch.device:
        # Distributed mode: device is dictated by LOCAL_RANK + intent.
        # The user can force CPU/MPS even with CUDA available (useful for
        # CPU-DDP smoke tests with gloo). Otherwise default to cuda:LOCAL_RANK.
        if self.is_distributed:
            cfg_device = str(self.config.device).strip().lower() if not isinstance(self.config.device, (list, tuple, int)) else None
            forced_cpu = cfg_device == "cpu"
            forced_mps = cfg_device == "mps"
            if forced_cpu:
                device = torch.device("cpu")
            elif forced_mps and torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                device = torch.device(f"cuda:{self.local_rank}")
            else:
                device = torch.device("cpu")
            if is_main_process():
                logger.info(
                    f"DDP active: rank={self.rank}/{self.world_size} device={device}"
                )
            return device

        # Single-process mode. Accept list/comma device only as an intent signal
        # — fail loudly with a torchrun pointer rather than silently degrading.
        raw_device = self.config.device

        # Normalise single-element list/tuple to its int. Multi-element forms
        # fall through to the wants_distributed check below.
        if isinstance(raw_device, (list, tuple)) and len(raw_device) == 1:
            raw_device = raw_device[0]

        if wants_distributed(raw_device):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Multi-GPU requested (device={raw_device!r}) but CUDA is not "
                    "available."
                )
            n = len(parse_device_arg(raw_device))
            raise RuntimeError(
                f"Multi-GPU device {raw_device!r} was passed directly to the trainer "
                "without an active process group. Use the model API instead — it "
                f"spawns DDP workers automatically: model.train(data=..., device={raw_device!r}). "
                f"Alternatively launch with torchrun: "
                f"`torchrun --nproc_per_node={n} your_script.py`."
            )

        device_str = str(raw_device).strip().lower() if not isinstance(raw_device, int) else str(raw_device)
        if isinstance(raw_device, int):
            device_str = f"cuda:{raw_device}"
        elif device_str in ("", "auto"):
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            logger.info(f"Using device: {device}")
            return device
        # YOLO-style "0" -> "cuda:0"
        if device_str.isdigit():
            device_str = f"cuda:{device_str}"
        device = torch.device(device_str)
        logger.info(f"Using device: {device}")
        return device

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        pg0, pg1, pg2 = [], [], []
        captured_ids: set = set()

        def add_if_trainable(group: list, param: Optional[nn.Parameter]) -> None:
            if isinstance(param, nn.Parameter) and param.requires_grad:
                group.append(param)
                captured_ids.add(id(param))

        # Catch every batch-norm flavour, including SyncBN: BatchNorm{1,2,3}d
        # and SyncBatchNorm are all siblings under ``_BatchNorm``. The naive
        # ``isinstance(v, nn.BatchNorm2d)`` check would silently put SyncBN
        # weights into the weight-decay group post sync_bn conversion.
        bn_types = nn.modules.batchnorm._BatchNorm
        for _k, v in self.model.named_modules():
            if hasattr(v, "bias"):
                add_if_trainable(pg2, v.bias)
            if isinstance(v, bn_types):
                add_if_trainable(pg0, v.weight)
            elif hasattr(v, "weight"):
                add_if_trainable(pg1, v.weight)

        # Bare nn.Parameters (LayerScale gamma, NAFNet beta/gamma, YOLO-NAS
        # alpha, ...) are not exposed as a module ``.weight``/``.bias``
        # attribute, so the named_modules() sweep above never captures them and
        # they would silently never join a param group (no gradient updates).
        # Add any still-uncaptured trainable parameter to the no-weight-decay
        # group (pg2), matching how biases/norms are handled.
        for _pk, p in self.model.named_parameters():
            if p.requires_grad and id(p) not in captured_ids:
                pg2.append(p)
                captured_ids.add(id(p))

        lr = self.effective_lr
        opt_name = self.config.optimizer
        # BN and bias groups carry an explicit weight_decay=0.0: groups without
        # the key inherit the optimizer's default, which is 0.01 for AdamW --
        # silently decaying norm gammas and biases that every upstream recipe
        # (paramwise norm/bias_decay_mult=0) exempts. No-op for SGD/Adam.
        param_groups = []
        if pg0:
            param_groups.append({"params": pg0, "lr": lr, "weight_decay": 0.0})
        if pg1:
            param_groups.append(
                {"params": pg1, "lr": lr, "weight_decay": self.config.weight_decay}
            )
        if pg2:
            param_groups.append({"params": pg2, "lr": lr, "weight_decay": 0.0})
        if not param_groups:
            raise ValueError(
                "No trainable parameters remain after layer freezing; "
                "reduce the freeze value or choose a narrower selector."
            )

        if opt_name == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=self.config.momentum,
                nesterov=self.config.nesterov,
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=lr,
                betas=(self.config.momentum, 0.999),
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                betas=(self.config.momentum, 0.999),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        if is_main_process():
            logger.info(f"Optimizer: {opt_name}")
            logger.info(f"  - pg0 (BN): {len(pg0)} params")
            logger.info(f"  - pg1 (Conv, wd={self.config.weight_decay}): {len(pg1)} params")
            logger.info(f"  - pg2 (Bias): {len(pg2)} params")
        return optimizer

    def _apply_freeze_config(self) -> None:
        summary = apply_freeze(
            self.model,
            getattr(self.config, "freeze", None),
            freeze_groups=self.get_freeze_groups(),
            preserve_trainable_param=self.preserve_freeze_param,
        )
        self.freeze_summary = summary
        self._frozen_bn_modules = summary.frozen_bn_modules if summary else ()
        if summary is not None and is_main_process():
            logger.info(
                "Layer freezing: selectors=%s, tensors=%d, params=%d, trainable=%d/%d",
                list(summary.selectors),
                summary.frozen_tensor_count,
                summary.frozen_param_count,
                summary.trainable_param_count,
                summary.total_param_count,
            )

    def _enforce_frozen_bn_eval(self) -> None:
        for module in getattr(self, "_frozen_bn_modules", ()):
            module.eval()

    def _setup_distillation(self):
        """Set up knowledge distillation when ``config.distill_model`` is set."""
        if not getattr(self.config, "distill_model", None):
            if getattr(self, "_resume_distiller_state", None) is not None:
                logger.warning(
                    "Resume checkpoint contains distiller state, but distillation "
                    "is disabled"
                )
                self._resume_distiller_state = None
            return

        from ..distillation import Distiller
        from ..distillation.teachers import is_foundation_teacher

        if self.wrapper_model is None:
            raise ValueError(
                "distill_model requires the trainer to be constructed with "
                "wrapper_model set (the student wrapper provides tap points)."
            )

        if is_foundation_teacher(self.config.distill_model):
            # Foundation teacher (e.g. DINOv2): a frozen semantic ViT supervises
            # a single student backbone stage via feature-MSE. Features come
            # through the teacher's extract_features(), not forward hooks, and
            # the loss handles the teacher/student spatial-grid mismatch.
            from ..distillation.teachers import DINOv2Teacher

            logger.info(f"Loading foundation teacher: {self.config.distill_model}")
            teacher = DINOv2Teacher(self.config.distill_model).to(self.device)

            if not hasattr(self.wrapper_model, "get_backbone_distill_config"):
                family = getattr(self.wrapper_model, "FAMILY", type(self.wrapper_model).__name__)
                raise NotImplementedError(
                    f"Foundation-model distillation into the '{family}' family is "
                    f"not supported yet (no get_backbone_distill_config())."
                )

            self.distiller = Distiller(
                teacher_model=teacher,
                student_model=self.model,
                teacher_config=teacher.get_distill_config(),
                student_config=self.wrapper_model.get_backbone_distill_config(),
                loss_type="feat_mse",
                loss_weight=self.config.dis,
                teacher_feature_fn=teacher.extract_features,
                normalize=getattr(self.config, "distill_normalize", False),
            )
        else:
            from ..models import LibreYOLO

            # Load teacher via the factory (handles family detection, weight loading)
            logger.info(f"Loading teacher model: {self.config.distill_model}")
            teacher_wrapper = LibreYOLO(self.config.distill_model)
            teacher_nn = teacher_wrapper.model.to(self.device)

            # Get distillation configs from the models themselves. Families that
            # don't support distillation raise NotImplementedError here with a
            # message naming the family.
            teacher_cfg = teacher_wrapper.get_distill_config()
            student_cfg = self.wrapper_model.get_distill_config()

            self.distiller = Distiller(
                teacher_model=teacher_nn,
                student_model=self.model,
                teacher_config=teacher_cfg,
                student_config=student_cfg,
                loss_type=self.config.distill_loss_type,
                loss_weight=self.config.dis,
                mask_ratio=self.config.distill_mask_ratio,
                tau=self.config.distill_tau,
            )
        self.distiller.to(self.device)

        # resume() may run before setup() — apply deferred adapter state now.
        deferred_state = getattr(self, "_resume_distiller_state", None)
        if deferred_state is not None:
            try:
                self.distiller.loss_modules.load_state_dict(deferred_state)
                logger.info("Distiller adapter state restored from resume checkpoint")
            except Exception as e:
                raise RuntimeError(f"Cannot resume distiller state: {e}") from e
            finally:
                self._resume_distiller_state = None

        # Add distiller's learnable params (align/generation convs) to optimizer
        distill_decay = []
        distill_no_decay = []
        for name, param in self.distiller.loss_modules.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                distill_no_decay.append(param)
            else:
                distill_decay.append(param)
        if distill_decay:
            self.optimizer.add_param_group(
                {
                    "params": distill_decay,
                    "lr": self.effective_lr,
                    "weight_decay": self.config.weight_decay,
                }
            )
        if distill_no_decay:
            self.optimizer.add_param_group(
                {
                    "params": distill_no_decay,
                    "lr": self.effective_lr,
                    "weight_decay": 0.0,
                }
            )
        distill_param_count = len(distill_decay) + len(distill_no_decay)
        if distill_param_count:
            logger.info(
                "Added %d distillation params to optimizer (%d decay, %d no-decay)",
                distill_param_count,
                len(distill_decay),
                len(distill_no_decay),
            )

    def _sync_distiller_grads(self):
        """All-reduce distiller adapter/generator gradients across DDP ranks.

        The distiller's loss modules live outside the DDP-wrapped student, so
        DDP's reducer never sees their gradients; without an explicit sync each
        rank would train its own diverging adapters. No-op outside DDP.
        """
        distiller = getattr(self, "distiller", None)
        if distiller is None or not is_distributed():
            return
        import torch.distributed as dist

        world_size = float(get_world_size())
        for param in distiller.loss_modules.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size

    def _get_save_dir(self) -> Path:
        resume_save_dir = getattr(self, "_resume_save_dir", None)
        if resume_save_dir is not None:
            save_dir = Path(resume_save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            return save_dir

        project = Path(self.config.project)
        name = self.config.name

        save_dir = project / name
        if not self.config.exist_ok and save_dir.exists():
            i = 2
            while (project / f"{name}{i}").exists():
                i += 1
            save_dir = project / f"{name}{i}"

        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _setup_data(self):
        wrapper_task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        if wrapper_task == "classify":
            return self._setup_classify_data()
        if wrapper_task == "semantic":
            return self._setup_semantic_data()
        if wrapper_task == "depth":
            return self._setup_depth_data()
        if wrapper_task == "restore":
            return self._setup_restore_data()

        img_size = self.input_size
        preproc, MosaicDatasetClass = self.create_transforms()
        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        load_segments = task == "segment"
        load_obb = task == "obb"

        if self.config.data:
            data_cfg = load_data_config(
                self.config.data,
                allow_scripts=self.config.allow_download_scripts,
            )
            data_dir = data_cfg["root"]
            data_nc = data_cfg.get("nc")
            if data_nc is None and data_cfg.get("names") is not None:
                data_nc = len(data_cfg["names"])
            self.num_classes = (
                int(data_nc) if data_nc is not None else self.config.num_classes
            )

            ann_file = Path(data_dir) / "annotations" / "instances_train2017.json"
            coco_ann_file = get_coco_annotation_file(data_cfg, "train")

            # Prefer pre-resolved file lists from load_data_config (.txt format)
            img_files = data_cfg.get("train_img_files")
            label_files = data_cfg.get("train_label_files")

            if coco_ann_file:
                default_image_dir = resolve_default_coco_image_dir(
                    data_dir,
                    "train",
                    coco_ann_file,
                )
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file=coco_ann_file,
                    name=get_coco_image_dir(data_cfg, "train", default_image_dir),
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                    load_obb=load_obb,
                    num_classes=self.num_classes,
                    names=data_cfg.get("names"),
                )
            elif img_files:
                train_dataset = YOLODataset(
                    img_files=img_files,
                    label_files=label_files,
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                    load_obb=load_obb,
                    num_classes=self.num_classes,
                )
            elif ann_file.exists():
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file="instances_train2017.json",
                    name=resolve_default_coco_image_dir(
                        data_dir,
                        "train",
                        "instances_train2017.json",
                    ),
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                    load_obb=load_obb,
                    num_classes=self.num_classes,
                    names=data_cfg.get("names"),
                )
            else:
                train_path = data_cfg.get("train", "images/train")
                train_img_dir = Path(train_path)
                if not train_img_dir.is_absolute():
                    train_img_dir = Path(data_dir) / train_img_dir

                try:
                    img_files = get_img_files(train_path, prefix=data_dir)
                except (FileNotFoundError, ValueError):
                    img_files = []

                if len(img_files) == 0:
                    raise FileNotFoundError(f"No images found in {train_img_dir}")

                label_files = img2label_paths(img_files)

                train_dataset = YOLODataset(
                    img_files=img_files,
                    label_files=label_files,
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                    load_obb=load_obb,
                    num_classes=self.num_classes,
                )
        elif self.config.data_dir:
            data_dir = self.config.data_dir
            self.num_classes = self.config.num_classes

            if (Path(data_dir) / "annotations").exists():
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file="instances_train2017.json",
                    name=resolve_default_coco_image_dir(
                        data_dir,
                        "train",
                        "instances_train2017.json",
                    ),
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                    load_obb=load_obb,
                    num_classes=self.num_classes,
                )
            else:
                train_dataset = YOLODataset(
                    data_dir=data_dir,
                    split="train",
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                    load_obb=load_obb,
                    num_classes=self.num_classes,
                )
        else:
            raise ValueError("Either 'data' or 'data_dir' must be specified")

        self.config.num_classes = int(self.num_classes)

        mosaic_enabled = not load_obb
        if load_obb and is_main_process():
            logger.info(
                "Disabling mosaic/mixup for OBB training until corner-aware "
                "OBB augmentation is implemented."
            )

        train_dataset.enable_image_cache(getattr(self.config, "cache", False))

        dataset_kwargs = dict(
            dataset=train_dataset,
            img_size=img_size,
            mosaic=mosaic_enabled,
            preproc=preproc,
            degrees=self.config.degrees,
            translate=self.config.translate,
            mosaic_scale=self.config.mosaic_scale,
            mixup_scale=self.config.mixup_scale,
            shear=self.config.shear,
            perspective=getattr(self.config, "perspective", 0.0),
            enable_mixup=mosaic_enabled and self.config.mixup_prob > 0,
            mosaic_prob=self.config.mosaic_prob if mosaic_enabled else 0.0,
            mixup_prob=self.config.mixup_prob if mosaic_enabled else 0.0,
        )
        # Copy-paste is only wired for the mosaic datasets whose constructor
        # accepts it (segmentation-capable pipelines); pass it through only
        # there so the shared instantiation stays valid for every family. OBB
        # has no segments, so leave it off in that case.
        cp_prob = float(getattr(self.config, "copy_paste", 0.0) or 0.0)
        if "copy_paste" in inspect.signature(MosaicDatasetClass).parameters:
            dataset_kwargs["copy_paste"] = 0.0 if load_obb else cp_prob
            dataset_kwargs["copy_paste_mode"] = getattr(
                self.config, "copy_paste_mode", "flip"
            )
        train_dataset = MosaicDatasetClass(**dataset_kwargs)

        # ``batch`` is the global batch under DDP. Each rank's loader is built
        # with ``batch // world_size``.
        per_rank_batch = self._per_rank_batch_size()
        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(train_dataset) >= self.world_size,
                seed=distributed_sampler_seed(self.config.seed),
            )

        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=per_rank_batch,
            num_workers=self.config.workers,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
            sampler=sampler,
            seed=self.config.seed,
            rank=self.rank,
            distributed=self.is_distributed,
        )

        if is_main_process():
            logger.info(f"Training dataset: {len(train_dataset)} images")
            _hint_task = getattr(
                getattr(self, "wrapper_model", None), "task", "detect"
            )
            if _hint_task == "detect" and self.config.data:
                logger.info(
                    f"Tip: sanity-check your dataset for common issues first with "
                    f"`libreyolo doctor {self.config.data}`"
                )
            logger.info(
                f"Iterations per epoch: {len(self.train_loader)} "
                f"(batch_per_rank={per_rank_batch}, world_size={self.world_size})"
            )
        return train_dataset

    def _setup_classify_data(self):
        """Build the classification train dataloader from an ImageFolder root.

        Classification bypasses the detection mosaic/letterbox pipeline: ``data``
        is a dataset root (or known name) with ``train``/``val`` splits, each a
        folder-per-class. The class count is the source of truth here, so the
        wrapper head is (re)built to match before the optimizer is created.
        """
        from torch.utils.data import DataLoader

        from ..data.classify_dataset import (
            ClassifyDataset,
            build_classify_collate,
            get_class_names,
            resolve_classify_data,
        )

        dataset_root = resolve_classify_data(self.config.data)
        classes = get_class_names(dataset_root, split="train")
        num_classes = len(classes)
        self.num_classes = num_classes
        self.config.num_classes = num_classes
        class_to_idx = {name: i for i, name in enumerate(classes)}

        wrapper = self.wrapper_model
        if wrapper is not None:
            if (
                getattr(wrapper, "nb_classes", None) != num_classes
                and hasattr(wrapper, "_rebuild_for_new_classes")
            ):
                wrapper._rebuild_for_new_classes(num_classes)
                self.model = wrapper.model.to(self.device)
            wrapper.nb_classes = num_classes
            wrapper.names = {i: name for i, name in enumerate(classes)}

        imgsz = self.config.imgsz
        train_dataset = ClassifyDataset(
            dataset_root=dataset_root,
            split="train",
            imgsz=imgsz,
            augment=True,
            class_to_idx=class_to_idx,
            transform_kwargs={
                "crop_pct": getattr(wrapper, "crop_pct", 0.875),
                "interpolation": getattr(wrapper, "interpolation", "bilinear"),
                "auto_augment": getattr(self.config, "auto_augment", None),
                "erasing": getattr(self.config, "erasing", 0.0),
            },
        )

        # Batch-level MixUp / CutMix (soft labels) when requested; otherwise this
        # returns the plain classify collate so default training is unchanged.
        collate_fn = build_classify_collate(
            num_classes,
            mixup=getattr(self.config, "mixup", 0.0),
            cutmix=getattr(self.config, "cutmix", 0.0),
        )

        per_rank_batch = self._per_rank_batch_size()
        if per_rank_batch < 2:
            raise ValueError(
                "Classification training needs an effective per-rank batch size >= 2 "
                f"(got {per_rank_batch} from batch={self.config.batch}, "
                f"world_size={self.world_size}). A batch of 1 breaks the BatchNorm in "
                "the pooled classifier head (e.g. MobileNetV4/EfficientNetV2 norm_head). "
                "Increase batch (or reduce world_size)."
            )
        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(train_dataset) >= self.world_size,
                seed=distributed_sampler_seed(self.config.seed),
            )

        # Under DDP each rank only sees ``len(sampler)`` samples, so base the
        # drop_last decision on the per-rank visible count. Otherwise a small
        # dataset split across ranks could drop every rank's only partial
        # batch and leave zero iterations.
        try:
            visible_samples = len(sampler) if sampler is not None else len(train_dataset)
        except TypeError:
            visible_samples = len(train_dataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_batch,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.config.workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_fn,
            drop_last=visible_samples >= per_rank_batch,
            **dataloader_seed_kwargs(
                self.config.seed,
                rank=self.rank,
                distributed=self.is_distributed,
            ),
        )

        if is_main_process():
            logger.info(
                "Classification dataset: %d images, %d classes",
                len(train_dataset),
                num_classes,
            )
            logger.info(
                "Iterations per epoch: %d (batch_per_rank=%d, world_size=%d)",
                len(self.train_loader),
                per_rank_batch,
                self.world_size,
            )
        return train_dataset

    def _setup_semantic_data(self):
        """Build the semantic-segmentation train dataloader from a dataset YAML.

        Dense masks bypass the detection mosaic/letterbox pipeline. The
        dataset's class space (including the background class appended for
        polygon-derived masks) is the source of truth, so the wrapper head is
        (re)built to match before the optimizer is created.
        """
        from torch.utils.data import DataLoader

        from ..data.semantic_dataset import (
            SemanticDataset,
            resolve_semantic_data,
            semantic_collate_fn,
        )

        if not self.config.data:
            raise ValueError("Semantic training requires data= (a dataset YAML).")
        data_config = resolve_semantic_data(
            self.config.data,
            allow_scripts=self.config.allow_download_scripts,
        )
        resize_mode = getattr(self.wrapper_model, "semantic_resize_mode", "letterbox")
        divisor = getattr(self.wrapper_model, "semantic_imgsz_divisor", None)
        if divisor and self.config.imgsz % int(divisor):
            raise ValueError(
                f"Semantic training imgsz={self.config.imgsz} must be divisible "
                f"by {int(divisor)} for this model family."
            )
        # Family-scoped scale-jitter range. Families that do not define this
        # attribute (default None) keep the SemanticDataset default jitter,
        # unchanged. Input standardization is family-internal (applied in the
        # model's forward on the raw [0, 1] tensor), so the dataset stays
        # /255-only for every family.
        scale_jitter = getattr(self.wrapper_model, "semantic_scale_jitter", None)
        semantic_kwargs = {}
        if scale_jitter is not None:
            semantic_kwargs["scale_jitter"] = tuple(scale_jitter)
        # Same deal for photometric jitter: opt in per family, or keep the
        # SemanticDataset default. Note this deliberately does NOT read
        # config.hsv_prob -- SemanticDataset has always used its own default for
        # every semantic family, so honoring the config here would silently
        # change RF-DETR's and DINOv2's training too.
        hsv_prob = getattr(self.wrapper_model, "semantic_hsv_prob", None)
        if hsv_prob is not None:
            semantic_kwargs["hsv_prob"] = float(hsv_prob)
        train_dataset = SemanticDataset(
            data_config,
            split="train",
            imgsz=self.config.imgsz,
            augment=True,
            resize_mode=resize_mode,
            **semantic_kwargs,
        )

        num_classes = train_dataset.nc
        self.num_classes = num_classes
        self.config.num_classes = num_classes

        wrapper = self.wrapper_model
        if wrapper is not None:
            if (
                getattr(wrapper, "nb_classes", None) != num_classes
                and hasattr(wrapper, "_rebuild_for_new_classes")
            ):
                wrapper._rebuild_for_new_classes(num_classes)
                self.model = wrapper.model.to(self.device)
            wrapper.nb_classes = num_classes
            wrapper.names = dict(train_dataset.names)

        per_rank_batch = self._per_rank_batch_size()
        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(train_dataset) >= self.world_size,
                seed=distributed_sampler_seed(self.config.seed),
            )

        try:
            visible_samples = len(sampler) if sampler is not None else len(train_dataset)
        except TypeError:
            visible_samples = len(train_dataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_batch,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.config.workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=semantic_collate_fn,
            drop_last=visible_samples >= per_rank_batch,
            **dataloader_seed_kwargs(
                self.config.seed,
                rank=self.rank,
                distributed=self.is_distributed,
            ),
        )

        if is_main_process():
            logger.info(
                "Semantic dataset: %d images, %d classes",
                len(train_dataset),
                num_classes,
            )
            logger.info(
                "Iterations per epoch: %d (batch_per_rank=%d, world_size=%d)",
                len(self.train_loader),
                per_rank_batch,
                self.world_size,
            )
        return train_dataset

    def _setup_depth_data(self):
        """Build the depth train dataloader from a dataset YAML."""
        from torch.utils.data import DataLoader

        from ..data.depth_dataset import (
            DepthDataset,
            depth_collate_fn,
            resolve_depth_data,
        )

        if not self.config.data:
            raise ValueError("Depth training requires data= (a dataset YAML).")
        data_config = resolve_depth_data(
            self.config.data,
            allow_scripts=self.config.allow_download_scripts,
        )
        resize_mode = getattr(self.wrapper_model, "depth_resize_mode", "letterbox")
        divisor = getattr(self.wrapper_model, "depth_imgsz_divisor", None)
        if divisor and self.config.imgsz % int(divisor):
            raise ValueError(
                f"Depth training imgsz={self.config.imgsz} must be divisible "
                f"by {int(divisor)} for this model family."
            )
        train_dataset = DepthDataset(
            data_config,
            split="train",
            imgsz=self.config.imgsz,
            augment=True,
            resize_mode=resize_mode,
        )

        self.num_classes = 1
        self.config.num_classes = 1
        if self.wrapper_model is not None:
            self.wrapper_model.nb_classes = 1
            self.wrapper_model.names = {0: "depth"}

        per_rank_batch = self._per_rank_batch_size()
        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(train_dataset) >= self.world_size,
                seed=distributed_sampler_seed(self.config.seed),
            )

        try:
            visible_samples = len(sampler) if sampler is not None else len(train_dataset)
        except TypeError:
            visible_samples = len(train_dataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_batch,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.config.workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=depth_collate_fn,
            drop_last=visible_samples >= per_rank_batch,
            **dataloader_seed_kwargs(
                self.config.seed,
                rank=self.rank,
                distributed=self.is_distributed,
            ),
        )

        if is_main_process():
            logger.info("Depth dataset: %d images", len(train_dataset))
            logger.info(
                "Iterations per epoch: %d (batch_per_rank=%d, world_size=%d)",
                len(self.train_loader),
                per_rank_batch,
                self.world_size,
            )
        return train_dataset

    def _setup_restore_data(self):
        """Build the restoration train dataloader from a paired dataset YAML."""
        from torch.utils.data import DataLoader

        from ..data.restore_dataset import (
            RestoreDataset,
            resolve_restore_data,
            restore_collate_fn,
        )

        if not self.config.data:
            raise ValueError("Restore training requires data= (a dataset YAML).")
        data_config = resolve_restore_data(
            self.config.data,
            allow_scripts=self.config.allow_download_scripts,
        )
        train_dataset = RestoreDataset(
            data_config,
            split="train",
            imgsz=self.config.imgsz,
            augment=True,
        )

        self.num_classes = 1
        self.config.num_classes = 1
        if self.wrapper_model is not None:
            self.wrapper_model.nb_classes = 1
            self.wrapper_model.names = {0: "image"}

        per_rank_batch = self._per_rank_batch_size()
        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(train_dataset) >= self.world_size,
                seed=distributed_sampler_seed(self.config.seed),
            )

        try:
            visible_samples = len(sampler) if sampler is not None else len(train_dataset)
        except TypeError:
            visible_samples = len(train_dataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_batch,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.config.workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=restore_collate_fn,
            drop_last=visible_samples >= per_rank_batch,
            **dataloader_seed_kwargs(
                self.config.seed,
                rank=self.rank,
                distributed=self.is_distributed,
            ),
        )

        if is_main_process():
            logger.info("Restore dataset: %d image pairs", len(train_dataset))
            logger.info(
                "Iterations per epoch: %d (batch_per_rank=%d, world_size=%d)",
                len(self.train_loader),
                per_rank_batch,
                self.world_size,
            )
        return train_dataset

    def _resolve_num_classes_from_data_config(self) -> int:
        """Resolve dataset class count before criterion construction."""
        resolved = int(self.config.num_classes)
        if self.config.data:
            # Only the YAML's class count is needed here; the dataset itself is
            # downloaded later in _setup_data.
            data_cfg = load_data_config(
                self.config.data,
                autodownload=False,
                allow_scripts=self.config.allow_download_scripts,
            )
            resolved = int(data_cfg.get("nc", resolved))

        self.num_classes = resolved
        self.config.num_classes = resolved
        return resolved

    def _infer_model_num_classes(self) -> Optional[int]:
        """Best-effort class-count introspection for detector heads."""
        model = unwrap_model(self.model)
        for obj in (getattr(model, "decoder", None), getattr(model, "head", None), model):
            value = getattr(obj, "num_classes", None)
            if value is not None:
                return int(value)
        return None

    def _sync_wrapped_model_num_classes(self, num_classes: int) -> None:
        """Rebuild wrapper-owned heads or fail before criterion/head desync."""
        num_classes = int(num_classes)
        self.num_classes = num_classes
        self.config.num_classes = num_classes

        wrapper = self.wrapper_model
        model_nc = self._infer_model_num_classes()
        wrapper_nc = getattr(wrapper, "nb_classes", None) if wrapper is not None else None
        needs_rebuild = (
            (model_nc is not None and model_nc != num_classes)
            or (wrapper_nc is not None and int(wrapper_nc) != num_classes)
        )

        if not needs_rebuild:
            return

        if wrapper is None or not hasattr(wrapper, "_rebuild_for_new_classes"):
            raise RuntimeError(
                f"{self.get_model_family()} trainer resolved num_classes={num_classes}, "
                f"but the model head exposes num_classes={model_nc}. Pass a "
                "wrapper_model with _rebuild_for_new_classes() or construct the "
                "raw model with the resolved class count."
            )

        wrapper._rebuild_for_new_classes(num_classes)
        self.model = wrapper.model.to(self.device)
        wrapper.device = self.device

        rebuilt_nc = self._infer_model_num_classes()
        if rebuilt_nc is not None and rebuilt_nc != num_classes:
            raise RuntimeError(
                f"{self.get_model_family()} wrapper rebuild did not sync the model "
                f"head to num_classes={num_classes}; got {rebuilt_nc}."
            )

    # =========================================================================
    # Setup / train / epoch
    # =========================================================================

    def setup(self):
        if self._is_setup:
            return

        if getattr(self.config, "lora", False) and not self.supports_lora:
            family = self.get_model_family() if hasattr(self, "get_model_family") else "this model"
            raise ValueError(
                f"LoRA fine-tuning (lora=True) is not supported for {family}. "
                "LoRA targets transformer backbones with nn.Linear layers (e.g. RF-DETR)."
            )

        if is_main_process():
            logger.info("Setting up training...")
        self.model.to(self.device)
        if self.wrapper_model is not None:
            self.wrapper_model.device = self.device

        self.on_num_classes_resolved()

        # SyncBatchNorm conversion: only meaningful under DDP. Single-GPU
        # runs skip this regardless of the flag so single-GPU is unchanged.
        if self.is_distributed and getattr(self.config, "sync_bn", False):
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if is_main_process():
                logger.info("Converted BatchNorm to SyncBatchNorm")

        self.on_setup()

        if getattr(self.config, "batch", 16) == -1:
            from libreyolo.training.autobatch import resolve_auto_batch, _DEFAULT_FRACTION

            self.config.batch = resolve_auto_batch(
                self.model,
                imgsz=self.config.imgsz,
                amp=self.config.amp,
                world_size=self.world_size,
                nbs=getattr(self.config, "nbs", None),
                fraction=getattr(self.wrapper_model, "autobatch_fraction", _DEFAULT_FRACTION),
            )
            if is_main_process():
                logger.info("AutoBatch: resolved global batch size = %d", self.config.batch)

        # BN statistics quality under DDP: with SyncBatchNorm off, each rank's
        # BatchNorm tracks only its own per-rank shard (batch // world_size).
        # A small per-rank batch produces noisy running stats and degrades the
        # converged model (issue #484). Warn (do not silently change behavior)
        # so users of BatchNorm families can enable sync_bn.
        if self.is_distributed and not getattr(self.config, "sync_bn", False):
            per_rank_batch = self._per_rank_batch_size()
            has_batchnorm = any(
                isinstance(m, nn.modules.batchnorm._BatchNorm)
                for m in self.model.modules()
            )
            if has_batchnorm and per_rank_batch < 16 and is_main_process():
                logger.warning(
                    "DDP per-rank batch is %d (global batch %d / world_size %d) "
                    "and sync_bn is disabled; BatchNorm running statistics are "
                    "computed per rank on this small shard, which can reduce "
                    "accuracy versus single-GPU. Consider setting sync_bn=True.",
                    per_rank_batch,
                    self.config.batch,
                    self.world_size,
                )

        self._setup_data()
        if self.train_loader is None or len(self.train_loader) == 0:
            raise ValueError(
                "Training dataloader has zero batches; reduce batch size, add "
                "training samples, or reduce world_size"
            )
        deferred_model_state = getattr(self, "_resume_model_state", None)
        if deferred_model_state is not None:
            self._load_resume_model_state(deferred_model_state)
            self._resume_model_state = None
        self._apply_freeze_config()
        self.optimizer = self._setup_optimizer()
        self._setup_distillation()
        self.lr_scheduler = self.create_scheduler(self._scheduler_steps_per_epoch())

        deferred_scheduler_state = getattr(self, "_resume_scheduler_state", None)
        if deferred_scheduler_state is not None:
            load_scheduler_state = getattr(self.lr_scheduler, "load_state_dict", None)
            if callable(load_scheduler_state):
                try:
                    load_scheduler_state(deferred_scheduler_state)
                    logger.info("Scheduler state restored from resume checkpoint")
                except Exception as exc:
                    raise RuntimeError(
                        f"Cannot resume scheduler state: {exc}"
                    ) from exc
            else:
                raise RuntimeError(
                    "Cannot resume scheduler state: "
                    f"{type(self.lr_scheduler).__name__} does not support "
                    "load_state_dict()"
                )
            self._resume_scheduler_state = None

        # resume() may be called before setup() when the optimizer doesn't exist
        # yet. Apply the deferred state now so momentum buffers are restored before
        # _initialize_scheduler_lr() sets the correct LR on top.
        if getattr(self, "_resume_optimizer_state", None) is not None:
            try:
                self.optimizer.load_state_dict(self._resume_optimizer_state)
                logger.info("Optimizer state restored from resume checkpoint")
            except Exception as e:
                raise RuntimeError(f"Cannot resume optimizer state: {e}") from e
            finally:
                self._resume_optimizer_state = None

        self._initialize_scheduler_lr()

        # DDP wrap AFTER optimizer setup so _setup_optimizer's
        # named_parameters() sees the raw model. EMA below also reads the
        # raw model — ModelEMA already unwraps via is_parallel() check.
        if self.is_distributed:
            ddp_kwargs = self._ddp_kwargs()
            if self.device.type == "cuda":
                ddp_kwargs["device_ids"] = [self.local_rank]
                ddp_kwargs["output_device"] = self.local_rank
            self.model = nn.parallel.DistributedDataParallel(self.model, **ddp_kwargs)
            if is_main_process():
                logger.info(
                    "Wrapped model in DDP ("
                    + ", ".join(f"{k}={v}" for k, v in ddp_kwargs.items() if k not in ("device_ids", "output_device"))
                    + ")"
                )

        if self.config.amp and self.device.type == "cuda":
            self.scaler = GradScaler("cuda")
            if is_main_process():
                logger.info("Using mixed precision training (AMP)")
        else:
            self.scaler = None

        if self.config.ema:
            ema_tau = getattr(self.config, "ema_tau", 2000)
            self.ema_model = ModelEMA(
                self.model, decay=self.config.ema_decay, tau=ema_tau
            )
            if is_main_process():
                logger.info(
                    "Using EMA with decay=%s, tau=%s",
                    self.config.ema_decay,
                    ema_tau,
                )

        deferred_ema_state = getattr(self, "_resume_ema_state", None)
        if deferred_ema_state is not None:
            if self.ema_model is None:
                logger.warning(
                    "Resume checkpoint contains EMA state, but EMA is disabled"
                )
            else:
                try:
                    self.ema_model.ema.load_state_dict(deferred_ema_state)
                    self.ema_model.updates = int(
                        getattr(self, "_resume_ema_updates", 0)
                    )
                    logger.info("EMA state restored from resume checkpoint")
                except Exception as exc:
                    raise RuntimeError(f"Cannot resume EMA state: {exc}") from exc
            self._resume_ema_state = None
            self._resume_ema_updates = None

        deferred_scaler_state = getattr(self, "_resume_scaler_state", None)
        if deferred_scaler_state is not None:
            if self.scaler is None:
                logger.warning(
                    "Resume checkpoint contains GradScaler state, but AMP is disabled"
                )
            else:
                try:
                    self.scaler.load_state_dict(deferred_scaler_state)
                    logger.info("GradScaler state restored from resume checkpoint")
                except Exception as exc:
                    raise RuntimeError(
                        f"Cannot resume GradScaler state: {exc}"
                    ) from exc
            self._resume_scaler_state = None

        # Create and persist the output location once. All ranks receive either
        # the resolved path or the rank-zero failure, avoiding a peer hang when
        # filesystem setup fails.
        def setup_output_dir() -> str:
            save_dir = self._get_save_dir()
            self.config.to_yaml(save_dir / "train_config.yaml")
            logger.info(f"Saving to: {save_dir}")
            logger.info(
                f"Tip: watch this run live in your browser with "
                f"`libreyolo monitor {save_dir}`"
            )
            return str(save_dir)

        self.save_dir = Path(
            run_rank_zero_phase("training output setup", setup_output_dir)
        )

        # Optional training-step profiler (opt-in via config.profile). Built on
        # the main process; emits the breakdown + Chrome trace into save_dir.
        # Disabled under DDP (rank-0-only syncs, and the profile_then_stop
        # early stop would desync ranks).
        if getattr(self.config, "profile", False):
            if self.is_distributed:
                if is_main_process():
                    logger.warning("profile=True is ignored under distributed training.")
            elif is_main_process():
                from libreyolo.training.profiler import TrainStepProfiler

                profile_warmup = getattr(self.config, "profile_warmup", 5)
                profile_steps = getattr(self.config, "profile_steps", 20)
                accum_steps = self._accum_steps
                if accum_steps > 1:
                    profile_warmup = math.ceil(profile_warmup / accum_steps) * accum_steps
                    profile_steps = math.ceil(profile_steps / accum_steps) * accum_steps
                    logger.info(
                        "profile window rounded to accumulation boundaries "
                        "(warmup=%d, steps=%d, accum=%d)",
                        profile_warmup,
                        profile_steps,
                        accum_steps,
                    )
                self._profiler = TrainStepProfiler(
                    device=self.device,
                    warmup=profile_warmup,
                    active=profile_steps,
                    trace=getattr(self.config, "profile_trace", True),
                    open_report=getattr(self.config, "profile_open", True),
                    save_dir=self.save_dir,
                    logger=logger,
                    meta={
                        "model": self.get_model_tag(),
                        "device": str(self.device),
                        "batch": self.config.batch,
                        "imgsz": self.config.imgsz,
                        "amp": bool(self.config.amp),
                        "workers": self.config.workers,
                    },
                )

        deferred_rng_state = getattr(self, "_resume_rng_state", None)
        if deferred_rng_state is not None:
            try:
                self._restore_rng_state(deferred_rng_state)
                logger.info("RNG state restored from resume checkpoint")
            except Exception as exc:
                raise RuntimeError(f"Cannot resume RNG state: {exc}") from exc
            self._resume_rng_state = None

        self._is_setup = True

    def _ddp_find_unused_parameters(self) -> bool:
        """Subclasses override to flip when their forward graph is conditional.

        Default False matches PyTorch's default. rf-detr flips True when a
        segmentation head is present because the sparse branch leaves some
        params un-grad'd on some batches.
        """
        return False

    def _ddp_static_graph(self) -> bool:
        """Whether to pass ``static_graph=True`` to DDP.

        ``static_graph=True`` defers DDP's reducer analysis until after
        the first iteration, which correctly handles models whose
        gradients land with non-contiguous strides (e.g. multi-head
        attention QKV projections). It can only be combined with
        ``find_unused_parameters=False`` — when the forward graph has
        conditional branches, static_graph is unsound.

        Default: enabled when find_unused is False. Subclasses can
        override for finer control.
        """
        return not self._ddp_find_unused_parameters()

    def _ddp_kwargs(self) -> Dict[str, Any]:
        """Assemble DDP constructor kwargs. Subclasses can override.

        gradient_as_bucket_view defaults False because some flagship
        models (RF-DETR's transformer) produce gradient tensors whose
        strides don't match DDP's bucket view, causing silent sync
        misses. The memory cost is small for the models in scope.
        """
        return {
            "find_unused_parameters": self._ddp_find_unused_parameters(),
            "static_graph": self._ddp_static_graph(),
            "gradient_as_bucket_view": False,
        }

    def train(self) -> Dict:
        start_time = time.time()
        # May be stale from a previous profile_then_stop run on this instance;
        # a leftover True would silently truncate this run's first epoch.
        self._stop_training = False
        try:
            self.setup()

            if is_main_process():
                logger.info(f"Starting training for {self.config.epochs} epochs")
                logger.info(f"Model: {self.get_model_tag()}")
                logger.info(f"Batch size: {self.config.batch}")
                logger.info(f"Learning rate: {self.effective_lr}")

            start_event = self._build_train_start_event()
            # Callbacks execute on rank 0 because they may write shared files.
            # Every rank joins the phase so user callback failures propagate.
            def on_train_start() -> None:
                self._dispatch_artifact_callbacks("on_train_start", start_event)
                self.callbacks.on_train_start(start_event)

            run_rank_zero_phase("on_train_start callback", on_train_start)

            no_aug_start = self.config.epochs - self.config.no_aug_epochs
            if self.config.no_aug_epochs > 0 and self.start_epoch > no_aug_start:
                if is_main_process():
                    logger.info(
                        f"Resumed past no-aug threshold (epoch {self.start_epoch} > {no_aug_start}), "
                        f"disabling mosaic/mixup immediately"
                    )
                self.on_mosaic_disable()

            for epoch in range(self.start_epoch, self.config.epochs):
                self.current_epoch = epoch

                if epoch == no_aug_start:
                    if is_main_process():
                        logger.info(
                            f"Disabling mosaic/mixup for final {self.config.no_aug_epochs} epochs"
                        )
                    self.on_mosaic_disable()

                epoch_start_time = time.time()
                epoch_result = self._train_epoch(epoch)
                epoch_seconds = time.time() - epoch_start_time
                epoch_loss, val_metrics, loss_items, lr = self._normalize_epoch_result(
                    epoch_result
                )
                self.final_loss = epoch_loss
                self.epoch_losses.append(epoch_loss)

                profile_truncated = bool(getattr(self, "_stop_training", False))
                is_best = (
                    False
                    if profile_truncated
                    else self._update_best_state(epoch, val_metrics)
                )
                # Write ``last.pt`` every epoch so a crash never costs more than
                # a single epoch. ``best.pt`` (is_best) and periodic
                # ``epoch_N.pt`` (save_period) stay gated inside
                # _save_checkpoint, so those are unaffected. A profile-only run
                # (profile_then_stop) truncated the epoch after the profile
                # window, so no checkpoint is written for it: stamping the
                # partial epoch as complete would make a later resume skip the
                # rest of it.
                if not profile_truncated:
                    self._save_checkpoint(
                        epoch, epoch_loss, val_metrics, is_best=is_best
                    )

                event = self._build_train_epoch_event(
                    epoch=epoch,
                    train_loss=epoch_loss,
                    train_loss_items=loss_items,
                    lr=lr,
                    val_metrics=val_metrics,
                    is_best=is_best,
                    epoch_seconds=epoch_seconds,
                )
                self.epoch_events.append(event)
                def on_train_epoch_end() -> None:
                    self._dispatch_artifact_callbacks("on_train_epoch_end", event)
                    self.callbacks.on_train_epoch_end(event)

                run_rank_zero_phase("on_train_epoch_end callback", on_train_epoch_end)

                # Patience counts completed validation opportunities. With an
                # eval interval, unevaluated epochs must not consume patience.
                should_stop = (
                    self.config.patience > 0
                    and self.patience_counter >= self.config.patience
                )
                if self.is_distributed:
                    import torch.distributed as _dist

                    flag = torch.tensor(int(should_stop), dtype=torch.int, device=self.device)
                    _dist.broadcast(flag, src=0)
                    should_stop = bool(flag.item())
                if should_stop:
                    if (
                        bool(getattr(self.config, "save_plots", False))
                        and not self._is_final_epoch(epoch)
                    ):
                        self._validate_epoch(epoch, save_plots=True)
                    if is_main_process():
                        logger.info(
                            f"Early stopping triggered after {epoch + 1} epochs "
                            f"(patience={self.config.patience}, no improvement for "
                            f"{self.patience_counter} validation opportunities)"
                        )
                    break

                if getattr(self, "_stop_training", False):
                    break

            total_time = time.time() - start_time
            if is_main_process():
                if getattr(self, "_stop_training", False):
                    logger.info(
                        "Profile-only run (profile_then_stop=True): training "
                        f"stopped after the profile window in {total_time:.1f}s; "
                        "the partial epoch was not validated or checkpointed"
                    )
                else:
                    logger.info(
                        f"Training complete in {total_time / 3600:.2f} hours"
                    )

            results = self._build_train_results()
            end_event = self._build_train_end_event(total_time, results)
            def on_train_end() -> None:
                self._dispatch_artifact_callbacks("on_train_end", end_event)
                self.callbacks.on_train_end(end_event)

            run_rank_zero_phase("on_train_end callback", on_train_end)
            return results

        except BaseException as exc:
            elapsed_seconds = time.time() - start_time
            exception_event = self._build_train_exception_event(exc, elapsed_seconds)
            def on_train_exception() -> None:
                self._dispatch_artifact_callbacks("on_train_exception", exception_event)
                try:
                    self.callbacks.on_train_exception(exception_event)
                except Exception:
                    logger.exception("Training exception callback failed")

            run_rank_zero_phase("on_train_exception callback", on_train_exception)
            raise
        finally:
            distiller = getattr(self, "distiller", None)
            if distiller is not None:
                try:
                    distiller.cleanup()
                except Exception:
                    logger.exception("Distillation cleanup failed")

    def _dispatch_artifact_callbacks(self, method_name: str, event) -> None:
        try:
            getattr(self.artifact_callbacks, method_name)(event)
        except Exception:
            logger.exception("Training artifact callback failed")

    def _build_train_results(self) -> Dict[str, Any]:
        weights_dir = self.save_dir / "weights"
        best_checkpoint = weights_dir / "best.pt"
        last_checkpoint = weights_dir / "last.pt"
        epoch_metrics = [self._event_to_dict(event) for event in self.epoch_events]
        return {
            "final_loss": self.final_loss,
            "epoch_losses": list(self.epoch_losses),
            "epoch_lrs": [dict(event.lr) for event in self.epoch_events],
            "epoch_loss_items": [
                dict(event.train_loss_items) for event in self.epoch_events
            ],
            "val_metrics": [dict(event.val_metrics) for event in self.epoch_events],
            "epoch_metrics": epoch_metrics,
            "best_mAP50": self.best_mAP50,
            "best_mAP50_95": self.best_mAP50_95,
            "best_epoch": self.best_epoch,
            "save_dir": str(self.save_dir),
            "best_checkpoint": (
                str(best_checkpoint) if best_checkpoint.exists() else None
            ),
            "last_checkpoint": (
                str(last_checkpoint) if last_checkpoint.exists() else None
            ),
        }

    def _event_context(self) -> Dict[str, Any]:
        return {
            "total_epochs": self.config.epochs,
            "model_family": self.get_model_family(),
            "model_size": getattr(self.config, "size", None),
            "task": getattr(getattr(self, "wrapper_model", None), "task", "detect"),
            "save_dir": str(getattr(self, "save_dir", "")),
        }

    def _build_train_start_event(self) -> TrainStartEvent:
        return TrainStartEvent(
            start_epoch=self.start_epoch + 1,
            config=self.config.to_dict(),
            **self._event_context(),
        )

    def _build_train_end_event(
        self, total_seconds: float, results: Mapping[str, Any]
    ) -> TrainEndEvent:
        return TrainEndEvent(
            completed_epochs=len(self.epoch_events),
            final_loss=self.final_loss,
            best_metric=self.best_mAP50_95 if self.best_epoch else None,
            best_epoch=self.best_epoch if self.best_epoch else None,
            total_seconds=total_seconds,
            results=results,
            **self._event_context(),
        )

    def _build_train_exception_event(
        self, exc: BaseException, elapsed_seconds: float
    ) -> TrainExceptionEvent:
        return TrainExceptionEvent(
            epoch=self.current_epoch + 1 if self._is_setup else None,
            exception=exc,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            elapsed_seconds=elapsed_seconds,
            **self._event_context(),
        )

    def _scale_lr(self, base_lr: float, param_group: dict) -> float:
        """Hook for per-group LR scaling. Override in subclasses."""
        return base_lr

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            scalar = float(value.detach().item())
            return scalar if math.isfinite(scalar) else None
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return None
        return scalar if math.isfinite(scalar) else None

    def _scalar_mapping(self, values: Optional[Mapping]) -> Dict[str, float]:
        if not isinstance(values, Mapping):
            return {}

        scalars = {}
        for name, value in values.items():
            scalar = self._as_float(value)
            if scalar is not None:
                scalars[str(name)] = scalar
        return scalars

    def _require_validation_metric(
        self,
        metrics: Mapping[str, Any],
        keys: Tuple[str, ...],
        *,
        context: str,
    ) -> float:
        """Return the first finite required metric or fail the validation phase."""
        for key in keys:
            if key not in metrics:
                continue
            value = self._as_float(metrics[key])
            if value is None:
                raise ValueError(
                    f"{context} metric {key!r} must be a finite scalar, "
                    f"got {metrics[key]!r}."
                )
            return value
        raise ValueError(
            f"{context} did not return required metric "
            + " or ".join(repr(key) for key in keys)
            + "."
        )

    def _require_finite_training_loss(
        self, value: Any, *, context: str
    ) -> torch.Tensor:
        """Fail every rank before backward if any rank produced an invalid loss."""
        is_tensor = isinstance(value, torch.Tensor)
        local_valid = bool(
            is_tensor
            and value.numel() == 1
            and not value.is_complex()
            and bool(torch.isfinite(value.detach()).all())
        )
        all_valid = local_valid
        if is_distributed():
            import torch.distributed as dist

            device = value.device if is_tensor else self.device
            valid_flag = torch.tensor(
                int(local_valid),
                dtype=torch.int32,
                device=device,
            )
            dist.all_reduce(valid_flag, op=dist.ReduceOp.MIN)
            all_valid = bool(valid_flag.item())

        if not all_valid:
            if local_valid:
                detail = "a different distributed rank produced an invalid loss"
            elif is_tensor and value.numel() == 1:
                detail = f"got {value.detach().item()!r}"
            elif is_tensor:
                detail = f"got tensor shape {tuple(value.shape)}"
            else:
                detail = f"got {type(value).__name__}"
            raise FloatingPointError(
                f"{context} must produce one finite scalar loss; {detail}."
            )
        return value

    def _current_lrs(self) -> Dict[str, float]:
        if self.optimizer is None:
            return {}
        return {
            f"group{i}": float(param_group.get("lr", 0.0))
            for i, param_group in enumerate(self.optimizer.param_groups)
        }

    @staticmethod
    def _event_to_dict(event: TrainEpochEvent) -> Dict[str, Any]:
        return {
            "epoch": event.epoch,
            "total_epochs": event.total_epochs,
            "model_family": event.model_family,
            "model_size": event.model_size,
            "task": event.task,
            "save_dir": event.save_dir,
            "train_loss": event.train_loss,
            "train_loss_items": dict(event.train_loss_items),
            "lr": dict(event.lr),
            "val_metrics": dict(event.val_metrics),
            "validated": event.validated,
            "is_best": event.is_best,
            "current_metric": event.current_metric,
            "current_metric_name": event.current_metric_name,
            "best_metric": event.best_metric,
            "best_metric_name": event.best_metric_name,
            "best_epoch": event.best_epoch,
            "epoch_seconds": event.epoch_seconds,
        }

    def _normalize_epoch_result(
        self, epoch_result: Tuple
    ) -> Tuple[float, Optional[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
        if not isinstance(epoch_result, tuple):
            raise TypeError("_train_epoch must return a tuple")

        if len(epoch_result) == 2:
            epoch_loss, val_metrics = epoch_result
            loss_items = {}
            lr = self._current_lrs()
        elif len(epoch_result) == 4:
            epoch_loss, val_metrics, loss_items, lr = epoch_result
            loss_items = self._scalar_mapping(loss_items)
            lr = self._scalar_mapping(lr) or self._current_lrs()
        else:
            raise ValueError(
                "_train_epoch must return (loss, val_metrics) or "
                "(loss, val_metrics, loss_items, lr)"
            )

        normalized_loss = self._as_float(epoch_loss)
        if normalized_loss is None:
            raise ValueError(
                f"Training epoch loss must be a finite scalar, got {epoch_loss!r}."
            )

        return normalized_loss, val_metrics, dict(loss_items), dict(lr)

    def _best_metric_value(
        self, val_metrics: Optional[Dict[str, Any]]
    ) -> Optional[float]:
        if not val_metrics:
            return None

        if "best_metric" in val_metrics:
            value = val_metrics["best_metric"]
        elif "mAP50_95" in val_metrics:
            value = val_metrics["mAP50_95"]
        else:
            return None
        return self._as_float(value)

    def _best_metric_name(self, val_metrics: Optional[Dict[str, Any]]) -> str:
        if val_metrics:
            return str(
                val_metrics.get(
                    "best_metric_key",
                    getattr(self, "best_metric_key", "metrics/mAP50-95"),
                )
            )
        return str(getattr(self, "best_metric_key", "metrics/mAP50-95"))

    def _validation_metrics_for_event(
        self, val_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        if not val_metrics:
            return {}

        raw_metrics = val_metrics.get("metrics")
        if isinstance(raw_metrics, Mapping):
            return self._scalar_mapping(raw_metrics)
        return self._scalar_mapping(val_metrics)

    def _build_train_epoch_event(
        self,
        *,
        epoch: int,
        train_loss: float,
        train_loss_items: Mapping[str, float],
        lr: Mapping[str, float],
        val_metrics: Optional[Dict[str, Any]],
        is_best: bool,
        epoch_seconds: float,
    ) -> TrainEpochEvent:
        current_metric = self._best_metric_value(val_metrics) if val_metrics else None
        current_metric_name = (
            self._best_metric_name(val_metrics) if val_metrics else None
        )
        best_metric = self.best_mAP50_95 if self.best_epoch else None
        best_metric_name = (
            self._best_metric_name(val_metrics) if self.best_epoch else None
        )

        return TrainEpochEvent(
            epoch=epoch + 1,
            total_epochs=self.config.epochs,
            model_family=self.get_model_family(),
            model_size=getattr(self.config, "size", None),
            task=getattr(getattr(self, "wrapper_model", None), "task", "detect"),
            save_dir=str(self.save_dir),
            train_loss=float(train_loss),
            train_loss_items=self._scalar_mapping(train_loss_items),
            lr=self._scalar_mapping(lr),
            val_metrics=self._validation_metrics_for_event(val_metrics),
            validated=bool(val_metrics),
            is_best=is_best,
            current_metric=current_metric,
            current_metric_name=current_metric_name,
            best_metric=best_metric,
            best_metric_name=best_metric_name,
            best_epoch=self.best_epoch if self.best_epoch else None,
            epoch_seconds=float(epoch_seconds),
        )

    def _update_best_state(
        self, epoch: int, val_metrics: Optional[Dict[str, Any]]
    ) -> bool:
        if not val_metrics:
            return False

        best_metric = self._best_metric_value(val_metrics)
        mAP50 = self._as_float(val_metrics.get("mAP50", 0.0))
        if best_metric is None or mAP50 is None:
            logger.warning(
                "Ignoring missing or non-finite validation metric at epoch %d: %r",
                epoch + 1,
                val_metrics.get(
                    "best_metric", val_metrics.get("mAP50_95")
                ),
            )
            self.patience_counter += 1
            return False

        is_best = self.best_epoch == 0 or best_metric > self.best_mAP50_95
        if is_best:
            self.best_mAP50_95 = best_metric
            self.best_mAP50 = mAP50
            self.best_epoch = epoch + 1
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return is_best

    def _get_clip_max_norm(self) -> float:
        value = getattr(self.config, "clip_max_norm", 0.0)
        if value is None:
            return 0.0
        try:
            max_norm = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"clip_max_norm must be a finite non-negative number, got {value!r}"
            ) from exc
        if max_norm < 0.0 or not math.isfinite(max_norm):
            raise ValueError(
                f"clip_max_norm must be a finite non-negative number, got {value!r}"
            )
        return max_norm

    def _should_clip_gradients(self) -> bool:
        return self._get_clip_max_norm() > 0.0

    def _set_optimizer_lr(self, base_lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self._scale_lr(base_lr, param_group)

    def _initialize_scheduler_lr(self) -> None:
        if self.optimizer is None or self.lr_scheduler is None:
            return
        init_iter = int(getattr(self, "optimizer_step_count", 0))
        if not getattr(self, "_optimizer_step_count_restored", False):
            init_iter = (
                getattr(self, "start_epoch", 0)
                * self._scheduler_steps_per_epoch()
            )
            self.optimizer_step_count = init_iter
        self._set_optimizer_lr(self.lr_scheduler.update_lr(init_iter))

    def _run_optimizer_step(self) -> bool:
        """Step the optimizer and report whether AMP actually applied it."""
        if self.scaler is None:
            self.optimizer.step()
            return True

        get_scale = getattr(self.scaler, "get_scale", None)
        scale_before = float(get_scale()) if callable(get_scale) else None
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if scale_before is None:
            # Lightweight test/custom scalers may not expose scale state. Their
            # step contract has no observable skip signal, so preserve legacy
            # behavior and treat the call as successful.
            return True
        scale_after = float(get_scale())
        return scale_after >= scale_before

    def _advance_optimizer_dependent_state(self, step_succeeded: bool) -> float:
        """Advance counters, EMA, and LR only after a real optimizer update."""
        if not step_succeeded:
            return float(self.optimizer.param_groups[0].get("lr", 0.0))
        self.optimizer_step_count = int(
            getattr(self, "optimizer_step_count", 0)
        ) + 1
        if getattr(self, "ema_model", None) is not None:
            self.ema_model.update(self.model)
        if getattr(self, "lr_scheduler", None) is not None:
            lr = self.lr_scheduler.update_lr(self.optimizer_step_count)
            self._set_optimizer_lr(lr)
            return float(lr)
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    def _normalize_accumulated_gradients(self, divisor: float) -> None:
        """Normalize a sample-summed accumulation window exactly once."""
        if not math.isfinite(divisor) or divisor <= 0:
            raise ValueError(
                f"gradient accumulation divisor must be finite and positive, got {divisor!r}"
            )
        seen: set[int] = set()
        for group in self.optimizer.param_groups:
            for param in group.get("params", ()):
                if param.grad is None or id(param) in seen:
                    continue
                seen.add(id(param))
                param.grad.div_(divisor)

    def _gradient_clip_parameters(self) -> List[torch.nn.Parameter]:
        if self.optimizer is None:
            return []
        params = []
        seen = set()
        for group in self.optimizer.param_groups:
            for param in group.get("params", ()):
                if param.grad is None:
                    continue
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                params.append(param)
        return params

    def _clip_gradients(self) -> Optional[torch.Tensor]:
        max_norm = self._get_clip_max_norm()
        if max_norm <= 0.0:
            return None
        return torch.nn.utils.clip_grad_norm_(
            self._gradient_clip_parameters(),
            max_norm,
        )

    def _prof_phase(self, name: str):
        """Profiler phase context manager (no-op when profiling is disabled)."""
        prof = getattr(self, "_profiler", None)
        return prof.phase(name) if prof is not None else contextlib.nullcontext()

    def _train_epoch(
        self, epoch: int
    ) -> Tuple[float, Optional[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
        self.model.train()
        self._enforce_frozen_bn_eval()

        # Gradient accumulation is opt-in. When enabled, delegate to the
        # accumulation loop; otherwise fall through to the standard
        # one-optimizer-step-per-batch loop below, unchanged.
        if self._accum_steps > 1:
            return self._train_epoch_accum(epoch)

        # DistributedSampler needs its epoch set so shuffling differs per
        # epoch while staying deterministic for resume.
        if is_distributed() and hasattr(self.train_loader, "sampler"):
            sampler = self.train_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            total=len(self.train_loader),
            disable=not sys.stderr.isatty() or not is_main_process(),
            file=sys.stderr,
        )

        total_loss = 0.0
        num_batches = 0
        loss_component_sums: Dict[str, float] = {}
        lr = float(self.optimizer.param_groups[0].get("lr", 0.0))

        # getattr: test doubles may bypass BaseTrainer.__init__, same as _profiler.
        distiller = getattr(self, "distiller", None)
        prof = getattr(self, "_profiler", None)
        loader = prof.wrap_loader(pbar) if prof is not None else pbar
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 5:
                imgs, targets, img_infos, img_ids, polygons = batch
            else:
                imgs, targets, img_infos, img_ids = batch
                polygons = None
            self.current_iter = epoch * len(self.train_loader) + batch_idx

            with self._prof_phase("to_device"):
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            if hasattr(self, "_apply_multi_scale_batch"):
                imgs, targets, polygons = self._apply_multi_scale_batch(
                    imgs,
                    targets,
                    polygons,
                    step=self.current_iter,
                )

            # Teacher forward (no-grad). Under AMP it runs in autocast too, so
            # the frozen teacher doesn't pay full-precision compute each step.
            if distiller is not None:
                if self.scaler is not None:
                    with autocast("cuda"):
                        distiller.teacher_forward(imgs)
                else:
                    distiller.teacher_forward(imgs)

            # Forward + backward. Under DDP the loss needs no rescaling:
            # every family's loss is mean/ratio-normalized, so DDP's gradient
            # averaging already composes the per-rank gradients into the
            # single-GPU-equivalent gradient (see scale_loss_for_ddp, #484).
            # The distiller's adapter grads are averaged the same way in
            # _sync_distiller_grads, keeping student and distiller consistent.
            if self.scaler is not None:
                with self._prof_phase("forward"):
                    with autocast("cuda"):
                        outputs = self.on_forward(imgs, targets, polygons=polygons)
                        total_loss_raw = outputs["total_loss"]
                        if distiller is not None:
                            distill_loss = distiller.compute_loss()
                            total_loss_raw = total_loss_raw + distill_loss
                            self._distill_loss_val = distill_loss.item()
                total_loss_raw = self._require_finite_training_loss(
                    total_loss_raw,
                    context=f"Epoch {epoch + 1} batch {batch_idx + 1}",
                )
                loss = scale_loss_for_ddp(total_loss_raw)
                self.optimizer.zero_grad()
                with self._prof_phase("backward"):
                    self.scaler.scale(loss).backward()
                    self._sync_distiller_grads()
                    if self._should_clip_gradients():
                        self.scaler.unscale_(self.optimizer)
                        self._clip_gradients()
                with self._prof_phase("optimizer"):
                    step_succeeded = self._run_optimizer_step()
            else:
                with self._prof_phase("forward"):
                    outputs = self.on_forward(imgs, targets, polygons=polygons)
                    total_loss_raw = outputs["total_loss"]
                    if distiller is not None:
                        distill_loss = distiller.compute_loss()
                        total_loss_raw = total_loss_raw + distill_loss
                        self._distill_loss_val = distill_loss.item()
                total_loss_raw = self._require_finite_training_loss(
                    total_loss_raw,
                    context=f"Epoch {epoch + 1} batch {batch_idx + 1}",
                )
                loss = scale_loss_for_ddp(total_loss_raw)
                self.optimizer.zero_grad()
                with self._prof_phase("backward"):
                    loss.backward()
                    self._sync_distiller_grads()
                    self._clip_gradients()
                with self._prof_phase("optimizer"):
                    step_succeeded = self._run_optimizer_step()

            if distiller is not None:
                distiller.step()

            lr = self._advance_optimizer_dependent_state(step_succeeded)

            # Logging captures the pre-scale value so single-GPU and DDP
            # report identical magnitudes (single-GPU semantics). ``.item()``
            # already returns a Python float and detaches from autograd.
            loss_val = float(total_loss_raw.item())
            loss_components = self._scalar_mapping(self.get_loss_components(outputs))
            if distiller is not None:
                loss_components["distill"] = self._distill_loss_val
            total_loss += loss_val
            for name, value in loss_components.items():
                loss_component_sums[name] = loss_component_sums.get(name, 0.0) + value

            del outputs, loss

            num_batches += 1

            # Progress bar
            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{lr:.6f}"}
            postfix.update({k: f"{v:.4f}" for k, v in loss_components.items()})
            pbar.set_postfix(postfix)

            if prof is not None:
                prof.step()
                if prof.finished:
                    if getattr(self.config, "profile_then_stop", False):
                        self._stop_training = True
                        break
                    # Window closed: drop the hooks so the rest of the run
                    # pays nothing, and keep training.
                    logger.info(
                        "Profile window complete; training continues "
                        "(profile_then_stop=True stops here instead)"
                    )
                    self._profiler = None
                    prof = None

        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_components = {
            name: value / max(num_batches, 1)
            for name, value in loss_component_sums.items()
        }
        if is_main_process():
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

        # Validation. A profile-only run (profile_then_stop) truncated the
        # epoch, so validating the barely-trained weights would waste time and
        # could poison the best-metric state.
        val_metrics = None
        if not getattr(self, "_stop_training", False) and self._should_validate_epoch(
            epoch
        ):
            val_metrics = self._validate_epoch(epoch)

        return avg_loss, val_metrics, avg_loss_components, self._current_lrs()

    def _train_epoch_accum(
        self, epoch: int
    ) -> Tuple[float, Optional[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
        """``_train_epoch`` variant with gradient accumulation enabled.

        Weights each already-reduced micro-batch loss by its image count, then
        divides once by the local/global window sample count. This makes short
        or variable-size windows exact for per-image-mean losses and prevents a
        one-image tail batch from weighing as much as a full batch. Losses that
        normalize separate terms by positives, boxes, or pixels cannot in
        general be recomposed exactly without unreduced numerator metadata;
        equal-size production windows retain their established semantics.
        Optimizer, clipping, EMA, and LR advance once per successful update.
        """
        self.model.train()
        self._enforce_frozen_bn_eval()

        if is_distributed() and hasattr(self.train_loader, "sampler"):
            sampler = self.train_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            total=len(self.train_loader),
            disable=not sys.stderr.isatty() or not is_main_process(),
            file=sys.stderr,
        )

        accum = self._accum_steps
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / accum))
        total_loss = 0.0
        num_batches = 0
        loss_component_sums: Dict[str, float] = {}
        window_local_samples = 0
        lr = self.optimizer.param_groups[0]["lr"]

        # getattr: test doubles may bypass BaseTrainer.__init__, same as _profiler.
        distiller = getattr(self, "distiller", None)

        prof = getattr(self, "_profiler", None)
        loader = prof.wrap_loader(pbar) if prof is not None else pbar
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 5:
                imgs, targets, img_infos, img_ids, polygons = batch
            else:
                imgs, targets, img_infos, img_ids = batch
                polygons = None

            is_opt_step = (batch_idx + 1) % accum == 0 or batch_idx == len(self.train_loader) - 1
            opt_step = epoch * steps_per_epoch + batch_idx // accum
            self.current_iter = opt_step

            with self._prof_phase("to_device"):
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            if hasattr(self, "_apply_multi_scale_batch"):
                imgs, targets, polygons = self._apply_multi_scale_batch(
                    imgs,
                    targets,
                    polygons,
                    step=opt_step,
                )

            if batch_idx % accum == 0:
                self.optimizer.zero_grad(set_to_none=True)
                window_local_samples = 0
            batch_samples = int(imgs.shape[0])
            if batch_samples <= 0:
                raise ValueError("gradient accumulation received an empty micro-batch")
            window_local_samples += batch_samples

            # Teacher forward (no-grad). Under AMP it runs in autocast too, so
            # the frozen teacher doesn't pay full-precision compute each step.
            if distiller is not None:
                if self.scaler is not None:
                    with autocast("cuda"):
                        distiller.teacher_forward(imgs)
                else:
                    distiller.teacher_forward(imgs)

            # Forward + backward. Gradients accumulate across the window; the
            # optimizer step, clipping, EMA and LR update fire only on the
            # window boundary (``is_opt_step``). Image-count weighting corrects
            # per-image-mean losses and partial batches; DDP's gradient average
            # is paired with the global average sample count below.
            if self.scaler is not None:
                with self._prof_phase("forward"):
                    with autocast("cuda"):
                        outputs = self.on_forward(imgs, targets, polygons=polygons)
                        total_loss_raw = outputs["total_loss"]
                        if distiller is not None:
                            distill_loss = distiller.compute_loss()
                            total_loss_raw = total_loss_raw + distill_loss
                            self._distill_loss_val = distill_loss.item()
                        total_loss_raw = self._require_finite_training_loss(
                            total_loss_raw,
                            context=f"Epoch {epoch + 1} batch {batch_idx + 1}",
                        )
                        loss = total_loss_raw * batch_samples
                loss = scale_loss_for_ddp(loss)
                with self._prof_phase("backward"):
                    self.scaler.scale(loss).backward()
                if is_opt_step:
                    with self._prof_phase("optimizer"):
                        self._sync_distiller_grads()
                        self.scaler.unscale_(self.optimizer)
                        sample_divisor = all_reduce_avg_scalar(
                            window_local_samples,
                            device=self.device,
                            min_value=1.0,
                        )
                        self._normalize_accumulated_gradients(sample_divisor)
                        if self._should_clip_gradients():
                            self._clip_gradients()
                        step_succeeded = self._run_optimizer_step()
            else:
                with self._prof_phase("forward"):
                    outputs = self.on_forward(imgs, targets, polygons=polygons)
                    total_loss_raw = outputs["total_loss"]
                    if distiller is not None:
                        distill_loss = distiller.compute_loss()
                        total_loss_raw = total_loss_raw + distill_loss
                        self._distill_loss_val = distill_loss.item()
                    total_loss_raw = self._require_finite_training_loss(
                        total_loss_raw,
                        context=f"Epoch {epoch + 1} batch {batch_idx + 1}",
                    )
                    loss = total_loss_raw * batch_samples
                loss = scale_loss_for_ddp(loss)
                with self._prof_phase("backward"):
                    loss.backward()
                if is_opt_step:
                    with self._prof_phase("optimizer"):
                        self._sync_distiller_grads()
                        sample_divisor = all_reduce_avg_scalar(
                            window_local_samples,
                            device=self.device,
                            min_value=1.0,
                        )
                        self._normalize_accumulated_gradients(sample_divisor)
                        self._clip_gradients()
                        step_succeeded = self._run_optimizer_step()

            if distiller is not None:
                distiller.step()

            if is_opt_step:
                lr = self._advance_optimizer_dependent_state(step_succeeded)

            # Logging uses the raw pre-scale value (single-GPU semantics).
            loss_val = float(total_loss_raw.detach().item())
            loss_components = self._scalar_mapping(self.get_loss_components(outputs))
            if distiller is not None:
                loss_components["distill"] = self._distill_loss_val
            total_loss += loss_val
            num_batches += 1
            for name, value in loss_components.items():
                loss_component_sums[name] = loss_component_sums.get(name, 0.0) + value

            del outputs, loss

            # Progress bar
            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{lr:.6f}"}
            postfix.update({k: f"{v:.4f}" for k, v in loss_components.items()})
            pbar.set_postfix(postfix)

            if prof is not None:
                prof.step()
                if prof.finished:
                    if getattr(self.config, "profile_then_stop", False):
                        self._stop_training = True
                        break
                    # Window closed: drop the hooks so the rest of the run
                    # pays nothing, and keep training.
                    logger.info(
                        "Profile window complete; training continues "
                        "(profile_then_stop=True stops here instead)"
                    )
                    self._profiler = None
                    prof = None

        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_components = {
            name: value / max(num_batches, 1)
            for name, value in loss_component_sums.items()
        }
        if is_main_process():
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

        # Validation. A profile-only run (profile_then_stop) truncated the
        # epoch, so validating the barely-trained weights would waste time and
        # could poison the best-metric state.
        val_metrics = None
        if not getattr(self, "_stop_training", False) and self._should_validate_epoch(
            epoch
        ):
            val_metrics = self._validate_epoch(epoch)

        return avg_loss, val_metrics, avg_loss_components, self._current_lrs()

    # =========================================================================
    # Validation
    # =========================================================================

    def _should_validate_epoch(self, epoch: int) -> bool:
        scheduled = (
            self.config.eval_interval > 0
            and (epoch + 1) % self.config.eval_interval == 0
        )
        final_plot = (
            bool(getattr(self.config, "save_plots", False))
            and self._is_final_epoch(epoch)
        )
        return scheduled or final_plot

    def _is_final_epoch(self, epoch: int) -> bool:
        return (epoch + 1) >= self.config.epochs

    def _validate_epoch(
        self, epoch: int, *, save_plots: bool | None = None
    ) -> Optional[Dict[str, Any]]:
        return run_rank_zero_phase(
            f"validation epoch {epoch + 1}",
            lambda: self._run_validation(epoch, save_plots=save_plots),
        )

    def _run_validation(
        self, epoch: int, *, save_plots: bool | None = None
    ) -> Optional[Dict[str, Any]]:
        validation_task = getattr(
            getattr(self, "wrapper_model", None), "task", "detect"
        )
        if validation_task == "classify":
            return self._run_classify_validation(epoch)
        if validation_task == "semantic":
            return self._run_semantic_validation(epoch)
        if validation_task == "depth":
            return self._run_depth_validation(epoch)
        if validation_task == "restore":
            return self._run_restore_validation(epoch)
        try:
            from libreyolo.validation import (
                DetectionValidator,
                OBBValidator,
                PointValidator,
                SegmentationValidator,
                ValidationConfig,
            )

            logger.info(f"Running validation for epoch {epoch + 1}")

            # Only save plots on the final epoch when explicitly requested.
            is_final_epoch = self._is_final_epoch(epoch)
            val_save_plots = (
                bool(save_plots)
                if save_plots is not None
                else bool(getattr(self.config, "save_plots", False)) and is_final_epoch
            )
            val_save_dir = (
                str(self.save_dir / "val") if val_save_plots else None
            )

            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=self._per_rank_batch_size(),
                imgsz=self.config.imgsz,
                conf_thres=0.001,
                iou_thres=0.65,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                verbose=False,
                num_workers=self.config.workers,
                save_plots=val_save_plots,
                save_dir=val_save_dir,
            )

            if self.wrapper_model is None:
                raise RuntimeError(
                    "Validation requires wrapper_model to be provided to trainer"
                )
            # Validator wants the un-DDP-wrapped module.
            eval_pytorch_model = (
                self.ema_model.ema if self.ema_model else unwrap_model(self.model)
            )
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model

            try:
                task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
                if task == "segment":
                    validator_cls = SegmentationValidator
                elif task == "obb":
                    validator_cls = OBBValidator
                elif task == "point":
                    validator_cls = PointValidator
                else:
                    validator_cls = DetectionValidator
                validator = validator_cls(model=self.wrapper_model, config=val_config)
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            raw_metrics = self._scalar_mapping(results)
            best_key = getattr(self, "best_metric_key", "metrics/mAP50-95")
            if task == "point" and best_key == "metrics/mAP50-95":
                best_key = "fitness"
            best_metric = self._require_validation_metric(
                raw_metrics,
                (best_key, "metrics/mAP50-95"),
                context=f"{task} validation",
            )
            if task == "point":
                primary_th = getattr(validator, "_primary_threshold", 0.01)
                mAP50 = raw_metrics.get(f"metrics/mAP@{primary_th:.2f}", 0.0)
            else:
                mAP50 = raw_metrics.get(
                    "metrics/mAP50", raw_metrics.get("metrics/mAP50(B)", 0.0)
                )
            metrics = {
                "mAP50": mAP50,
                "mAP50_95": best_metric,
                "best_metric": best_metric,
                "best_metric_key": best_key,
                "metrics": raw_metrics,
            }

            logger.debug(
                f"Extracted metrics: mAP50={metrics['mAP50']:.4f}, mAP50_95={metrics['mAP50_95']:.4f}"
            )
            logger.info(
                "Validation - mAP50: %.4f, mAP50-95: %.4f",
                metrics["mAP50"],
                metrics["mAP50_95"],
            )
            return metrics

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import traceback

            logger.debug(f"Validation traceback:\n{traceback.format_exc()}")
            raise RuntimeError("Training validation failed.") from e

    def _run_classify_validation(
        self, epoch: int
    ) -> Optional[Dict[str, Any]]:
        """Validate the classification head (top-1/top-5) on the val split."""
        try:
            from libreyolo.validation import ClassifyValidator, ValidationConfig

            if self.wrapper_model is None:
                raise RuntimeError(
                    "Validation requires wrapper_model to be provided to trainer"
                )

            logger.info(f"Running classification validation for epoch {epoch + 1}")
            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=self._per_rank_batch_size(),
                imgsz=self.config.imgsz,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                verbose=False,
                num_workers=self.config.workers,
                split="val",
            )

            eval_pytorch_model = (
                self.ema_model.ema if self.ema_model else unwrap_model(self.model)
            )
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model
            try:
                validator = ClassifyValidator(model=self.wrapper_model, config=val_config)
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            raw_metrics = self._scalar_mapping(results)
            top1 = self._require_validation_metric(
                raw_metrics,
                ("metrics/accuracy_top1",),
                context="classification validation",
            )
            top5 = raw_metrics.get("metrics/accuracy_top5", 0.0)
            logger.info("Validation - top1: %.4f, top5: %.4f", top1, top5)
            return {
                "mAP50": top1,
                "mAP50_95": top1,
                "best_metric": top1,
                "best_metric_key": "metrics/accuracy_top1",
                "metrics": raw_metrics,
            }
        except Exception as e:
            logger.error(f"Classification validation failed: {e}")
            import traceback

            logger.debug(f"Validation traceback:\n{traceback.format_exc()}")
            raise RuntimeError("Classification training validation failed.") from e

    def _run_semantic_validation(
        self, epoch: int
    ) -> Optional[Dict[str, Any]]:
        """Validate the semantic head (mIoU / pixel accuracy) on the val split."""
        try:
            from libreyolo.validation import SemanticValidator, ValidationConfig

            if self.wrapper_model is None:
                raise RuntimeError(
                    "Validation requires wrapper_model to be provided to trainer"
                )

            logger.info(f"Running semantic validation for epoch {epoch + 1}")
            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=self._per_rank_batch_size(),
                imgsz=self.config.imgsz,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                verbose=False,
                num_workers=self.config.workers,
                split="val",
            )

            eval_pytorch_model = (
                self.ema_model.ema if self.ema_model else unwrap_model(self.model)
            )
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model
            try:
                validator = SemanticValidator(model=self.wrapper_model, config=val_config)
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            raw_metrics = self._scalar_mapping(results)
            miou = self._require_validation_metric(
                raw_metrics,
                ("metrics/mIoU",),
                context="semantic validation",
            )
            accuracy = raw_metrics.get("metrics/pixel_accuracy", 0.0)
            logger.info("Validation - mIoU: %.4f, pixel accuracy: %.4f", miou, accuracy)
            return {
                "mAP50": miou,
                "mAP50_95": miou,
                "best_metric": miou,
                "best_metric_key": "metrics/mIoU",
                "metrics": raw_metrics,
            }
        except Exception as e:
            logger.error(f"Semantic validation failed: {e}")
            import traceback

            logger.debug(f"Validation traceback:\n{traceback.format_exc()}")
            raise RuntimeError("Semantic training validation failed.") from e

    def _run_depth_validation(
        self, epoch: int
    ) -> Optional[Dict[str, Any]]:
        """Validate the depth head (AbsRel / delta1) on the val split."""
        try:
            from libreyolo.validation import DepthValidator, ValidationConfig

            if self.wrapper_model is None:
                raise RuntimeError(
                    "Validation requires wrapper_model to be provided to trainer"
                )

            logger.info(f"Running depth validation for epoch {epoch + 1}")
            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=self._per_rank_batch_size(),
                imgsz=self.config.imgsz,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                verbose=False,
                num_workers=self.config.workers,
                split="val",
                allow_download_scripts=self.config.allow_download_scripts,
            )

            eval_pytorch_model = (
                self.ema_model.ema if self.ema_model else unwrap_model(self.model)
            )
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model
            try:
                validator = DepthValidator(model=self.wrapper_model, config=val_config)
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            raw_metrics = self._scalar_mapping(results)
            delta1 = self._require_validation_metric(
                raw_metrics,
                ("metrics/delta1",),
                context="depth validation",
            )
            abs_rel = raw_metrics.get("metrics/abs_rel", 0.0)
            logger.info("Validation - delta1: %.4f, AbsRel: %.4f", delta1, abs_rel)
            return {
                "mAP50": delta1,
                "mAP50_95": delta1,
                "best_metric": delta1,
                "best_metric_key": "metrics/delta1",
                "metrics": raw_metrics,
            }
        except Exception as e:
            logger.error(f"Depth validation failed: {e}")
            import traceback

            logger.debug(f"Validation traceback:\n{traceback.format_exc()}")
            raise RuntimeError("Depth training validation failed.") from e

    def _run_restore_validation(
        self, epoch: int
    ) -> Optional[Dict[str, Any]]:
        """Validate the restoration model (PSNR / SSIM) on the val split."""
        try:
            from libreyolo.validation import RestoreValidator, ValidationConfig

            if self.wrapper_model is None:
                raise RuntimeError(
                    "Validation requires wrapper_model to be provided to trainer"
                )

            logger.info(f"Running restore validation for epoch {epoch + 1}")
            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=self._per_rank_batch_size(),
                imgsz=self.config.imgsz,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                verbose=False,
                num_workers=self.config.workers,
                split="val",
                allow_download_scripts=self.config.allow_download_scripts,
            )

            eval_pytorch_model = (
                self.ema_model.ema if self.ema_model else unwrap_model(self.model)
            )
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model
            try:
                validator = RestoreValidator(model=self.wrapper_model, config=val_config)
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            raw_metrics = self._scalar_mapping(results)
            psnr = self._require_validation_metric(
                raw_metrics,
                ("metrics/PSNR",),
                context="restore validation",
            )
            ssim = raw_metrics.get("metrics/SSIM", 0.0)
            logger.info("Validation - PSNR: %.4f, SSIM: %.4f", psnr, ssim)
            return {
                "mAP50": psnr,
                "mAP50_95": psnr,
                "best_metric": psnr,
                "best_metric_key": "metrics/PSNR",
                "metrics": raw_metrics,
            }
        except Exception as e:
            logger.error(f"Restore validation failed: {e}")
            import traceback

            logger.debug(f"Validation traceback:\n{traceback.format_exc()}")
            raise RuntimeError("Restore training validation failed.") from e

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def _capture_rng_state(self) -> Dict[str, Any]:
        """Capture weights-only-safe RNG state for the current rank."""
        python_version, python_keys, python_gauss = random.getstate()
        numpy_state = np.random.get_state()
        state: Dict[str, Any] = {
            "python": {
                "version": int(python_version),
                "keys": torch.tensor(python_keys, dtype=torch.int64),
                "gauss": python_gauss,
            },
            "numpy": {
                "keys": torch.from_numpy(
                    numpy_state[1].astype("int64", copy=True)
                ),
                "pos": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch": torch.get_rng_state(),
        }
        loader_generator = getattr(
            getattr(self, "train_loader", None), "generator", None
        )
        if loader_generator is not None:
            state["train_loader_generator"] = loader_generator.get_state()
        if torch.cuda.is_available() and self.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(self.device)
        return state

    def _gather_rng_states(self) -> List[Dict[str, Any]]:
        """Gather each rank's independent RNG stream before a checkpoint."""
        local_state = self._capture_rng_state()
        if not is_distributed():
            return [local_state]

        import torch.distributed as _dist

        gathered: List[Optional[Dict[str, Any]]] = [None] * get_world_size()
        _dist.all_gather_object(gathered, local_state)
        if any(state is None for state in gathered):
            raise RuntimeError("distributed RNG-state gather returned an empty rank")
        return [state for state in gathered if state is not None]

    def _restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        """Restore one rank's Python, NumPy, Torch, and CUDA RNG streams."""
        python_state = rng_state.get("python")
        if python_state is not None:
            random.setstate(
                (
                    int(python_state["version"]),
                    tuple(int(value) for value in python_state["keys"].tolist()),
                    python_state.get("gauss"),
                )
            )

        numpy_state = rng_state.get("numpy")
        if numpy_state is not None:
            np.random.set_state(
                (
                    "MT19937",
                    numpy_state["keys"].cpu().numpy().astype("uint32"),
                    int(numpy_state["pos"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )

        torch_state = rng_state.get("torch")
        if torch_state is not None:
            torch.set_rng_state(torch_state.cpu())

        loader_state = rng_state.get("train_loader_generator")
        loader_generator = getattr(
            getattr(self, "train_loader", None), "generator", None
        )
        if loader_state is not None and loader_generator is not None:
            loader_generator.set_state(loader_state.cpu())

        cuda_state = rng_state.get("cuda")
        if (
            cuda_state is not None
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        ):
            if isinstance(cuda_state, (list, tuple)):
                # Backward compatibility with checkpoints that captured every
                # visible CUDA device from a single process.
                torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
            else:
                torch.cuda.set_rng_state(cuda_state.cpu(), self.device)

    def _save_checkpoint(
        self,
        epoch: int,
        loss: float,
        val_metrics: Optional[Dict[str, Any]] = None,
        is_best: Optional[bool] = None,
    ):
        loss_value = self._as_float(loss)
        if loss_value is None:
            raise ValueError(f"Checkpoint loss must be finite, got {loss!r}.")
        loss = loss_value

        rng_states_by_rank = self._gather_rng_states()
        return run_rank_zero_phase(
            f"checkpoint epoch {epoch + 1}",
            lambda: self._write_checkpoint(
                epoch,
                loss,
                val_metrics=val_metrics,
                is_best=is_best,
                rng_states_by_rank=rng_states_by_rank,
            ),
        )

    def _write_checkpoint(
        self,
        epoch: int,
        loss: float,
        val_metrics: Optional[Dict[str, Any]] = None,
        is_best: Optional[bool] = None,
        rng_states_by_rank: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if is_best is None:
            is_best = self._update_best_state(epoch, val_metrics)

        # Always unwrap DDP/compile wrappers before reading state_dict so the
        # checkpoint is interchangeable with single-GPU runs.
        raw_model = unwrap_model(self.model)
        model_to_save = self.ema_model.ema if self.ema_model else raw_model

        best_metric_key = (
            val_metrics.get(
                "best_metric_key",
                getattr(self, "best_metric_key", "metrics/mAP50-95"),
            )
            if val_metrics
            else getattr(self, "best_metric_key", "metrics/mAP50-95")
        )
        names = (
            self.wrapper_model.names
            if self.wrapper_model is not None and hasattr(self.wrapper_model, "names")
            else build_class_names(int(getattr(self, "num_classes", self.config.num_classes)))
        )
        checkpoint_nc = int(getattr(self, "num_classes", self.config.num_classes))
        checkpoint_imgsz = getattr(self.config, "imgsz", None)
        if checkpoint_imgsz is None and self.wrapper_model is not None:
            get_input_size = getattr(self.wrapper_model, "_get_input_size", None)
            if callable(get_input_size):
                checkpoint_imgsz = get_input_size()
        if checkpoint_imgsz is None:
            checkpoint_imgsz = 640
            logger.warning(
                "Training config has no imgsz. Writing checkpoint metadata "
                "imgsz=640; set config.imgsz to avoid this compatibility fallback."
            )

        extra_metadata = self._checkpoint_extra_metadata()
        if not isinstance(extra_metadata, Mapping):
            raise TypeError("_checkpoint_extra_metadata() must return a mapping.")
        extra_metadata = dict(extra_metadata)
        reserved_extras = sorted(
            set(extra_metadata) & _TRAINING_CHECKPOINT_CORE_KEYS
        )
        if reserved_extras:
            raise ValueError(
                "Checkpoint extension metadata cannot override core fields: "
                + ", ".join(reserved_extras)
            )

        checkpoint = wrap_libreyolo_checkpoint(
            model_to_save.state_dict(),
            model_family=self.get_model_family(),
            size=self.config.size,
            task=getattr(getattr(self, "wrapper_model", None), "task", "detect"),
            nc=checkpoint_nc,
            names=names,
            imgsz=int(checkpoint_imgsz),
            epoch=epoch,
            optimizer=self.optimizer.state_dict(),
            config=self.config.to_dict(),
            loss=loss,
            best_mAP50_95=self.best_mAP50_95,
            best_mAP50=self.best_mAP50,
            best_metric_key=best_metric_key,
            best_metric_value=self.best_mAP50_95,
            best_epoch=self.best_epoch,
            is_ema_weights=self.ema_model is not None,
            **extra_metadata,
        )
        checkpoint["best_metric"] = self.best_mAP50_95
        checkpoint["best_metric_name"] = checkpoint["best_metric_key"]
        checkpoint["optimizer_step_count"] = int(
            getattr(self, "optimizer_step_count", 0)
        )
        checkpoint["patience_counter"] = int(getattr(self, "patience_counter", 0))
        scheduler_state_dict = getattr(
            getattr(self, "lr_scheduler", None), "state_dict", None
        )
        if callable(scheduler_state_dict):
            checkpoint["scheduler"] = scheduler_state_dict()
        if self.ema_model is not None:
            checkpoint["train_model"] = raw_model.state_dict()
            checkpoint["ema"] = self.ema_model.ema.state_dict()
            checkpoint["ema_updates"] = self.ema_model.updates
        if getattr(self, "distiller", None) is not None:
            # Adapter/generator weights live outside the student model; persist
            # them so resume doesn't restart the distillation projectors cold.
            checkpoint["distiller"] = self.distiller.loss_modules.state_dict()

        # AMP + RNG state so ``resume()`` continues bit-for-bit instead of
        # re-warming the GradScaler from 65536 and reseeding the RNGs. All keys
        # are optional; older checkpoints without them still load fine.
        if self.scaler is not None:
            try:
                checkpoint["scaler"] = self.scaler.state_dict()
            except Exception:
                logger.warning("Could not capture GradScaler state for checkpoint")
        if rng_states_by_rank is None:
            rng_states_by_rank = [self._capture_rng_state()]
        checkpoint["rng_states_by_rank"] = rng_states_by_rank
        checkpoint["rng_state"] = rng_states_by_rank[0]

        validate_checkpoint_metadata(checkpoint, strict=True)

        weights_dir = self.save_dir / "weights"
        weights_dir.mkdir(exist_ok=True)

        latest_path = weights_dir / "last.pt"
        targets = [latest_path]

        if is_best:
            best_path = weights_dir / "best.pt"
            targets.append(best_path)

        if self.config.save_period > 0 and (epoch + 1) % self.config.save_period == 0:
            epoch_path = weights_dir / f"epoch_{epoch + 1}.pt"
            targets.append(epoch_path)

        _atomic_save_checkpoint(checkpoint, targets)

        if is_best:
            metric_key = checkpoint["best_metric_key"]
            metric_value = self.best_mAP50_95
            logger.info(
                f"New best model saved - Epoch {epoch + 1}: "
                f"{metric_key}={metric_value:.4f}"
            )

        logger.info(f"Checkpoint saved: {latest_path}")

    def _checkpoint_extra_metadata(self) -> Dict[str, Any]:
        return {}

    def _prepare_resume_model_architecture(
        self,
        checkpoint: Mapping[str, Any],
        model_state: Mapping[str, Any],
    ) -> None:
        """Rebuild checkpoint-dependent wrapper structure before setup/load."""
        wrapper = self.wrapper_model
        if wrapper is None:
            return
        checkpoint_nc = checkpoint.get("nc")
        checkpoint_names = checkpoint.get("names")
        if checkpoint_nc is None and checkpoint_names is not None:
            checkpoint_nc = len(checkpoint_names)
        if checkpoint_nc is None:
            return
        checkpoint_nc = int(checkpoint_nc)
        current_nc = getattr(wrapper, "nb_classes", None)
        if current_nc is None or int(current_nc) != checkpoint_nc:
            if getattr(self, "_is_setup", False):
                raise RuntimeError(
                    "Cannot resume checkpoint with nc="
                    f"{checkpoint_nc} after setup() built a {current_nc}-class "
                    "training graph; create a new trainer and resume before setup()"
                )
            rebuild = getattr(wrapper, "_rebuild_for_checkpoint_classes", None)
            if not callable(rebuild):
                raise RuntimeError(
                    "Cannot resume checkpoint with nc="
                    f"{checkpoint_nc}: wrapper cannot rebuild its class head"
                )
            rebuild(checkpoint_nc, model_state)
            self.model = wrapper.model.to(self.device)
            wrapper.device = self.device
        wrapper.nb_classes = checkpoint_nc
        self.config.num_classes = checkpoint_nc
        self.num_classes = checkpoint_nc
        if checkpoint_names is not None:
            sanitize_names = getattr(wrapper, "_sanitize_names", None)
            wrapper.names = (
                sanitize_names(checkpoint_names, checkpoint_nc)
                if callable(sanitize_names)
                else build_class_names(checkpoint_nc)
            )

    def _load_resume_model_state(self, model_state: Mapping[str, Any]) -> None:
        """Strictly restore training weights after architecture preparation."""
        try:
            unwrap_model(self.model).load_state_dict(model_state)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot resume: model architecture mismatch - {exc}"
            ) from exc

    def _restore_checkpoint_config(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore saved run settings while preserving explicit runtime choices.

        Arguments explicitly supplied by the current invocation override saved
        values, including when the supplied value equals the family default.
        Omitted values inherit the saved value. Dataset/model identity, device,
        and security/runtime-only controls stay authoritative from the current
        invocation. The checkpoint path defines the resumed run directory.
        """
        saved_config = checkpoint.get("config")
        if not isinstance(saved_config, Mapping) or getattr(
            self, "_is_setup", False
        ):
            return

        config_class = self._config_class()
        current = self.config.to_dict()
        explicit_keys = getattr(self, "_explicit_train_config_keys", ())
        optimizer_state_keys = {"momentum", "nesterov", "optimizer", "weight_decay"}
        if "optimizer" in checkpoint:
            for key in optimizer_state_keys & set(explicit_keys):
                if key in saved_config and current.get(key) != saved_config[key]:
                    raise RuntimeError(
                        f"Cannot resume with {key}={current.get(key)!r}: checkpoint "
                        f"optimizer state requires {key}={saved_config[key]!r}"
                    )
        protected = {
            "allow_download_scripts",
            "data",
            "data_dir",
            "device",
            "exist_ok",
            "name",
            "num_classes",
            "profile",
            "profile_open",
            "profile_then_stop",
            "project",
            "resume",
            "size",
        }
        restored = []
        for key, saved_value in saved_config.items():
            if key in protected or key not in current:
                continue
            if key in explicit_keys:
                continue
            if current[key] != saved_value:
                current[key] = saved_value
                restored.append(key)

        if restored:
            self.config = config_class.from_kwargs(**current)
            logger.info(
                "Restored checkpoint training config values: %s",
                ", ".join(sorted(restored)),
            )

    def _validate_resume_identity(self, checkpoint: Mapping[str, Any]) -> None:
        """Reject a resume checkpoint for a different model identity."""
        expected = {
            "model_family": str(self.get_model_family()).lower(),
            "task": normalize_task(
                getattr(getattr(self, "wrapper_model", None), "task", "detect")
            ),
        }
        current_size = getattr(self.config, "size", None)
        if current_size is not None:
            expected["size"] = str(current_size).lower()
        for key, expected_value in expected.items():
            actual = checkpoint.get(key)
            if actual is None:
                continue
            actual_value = (
                normalize_task(actual)
                if key == "task"
                else str(actual).strip().lower()
            )
            if actual_value != expected_value:
                raise RuntimeError(
                    f"Cannot resume: checkpoint {key}={actual!r} does not match "
                    f"current {key}={expected_value!r}"
                )

    @staticmethod
    def _validate_resume_runtime_states(checkpoint: Mapping[str, Any]) -> None:
        """Reject malformed component states before they can look absent."""
        for key in ("optimizer", "scheduler", "distiller", "ema", "scaler"):
            if key in checkpoint and not isinstance(checkpoint[key], Mapping):
                raise RuntimeError(
                    f"Cannot resume {key} state: expected a mapping, got "
                    f"{type(checkpoint[key]).__name__}"
                )

        if "rng_state" in checkpoint and not isinstance(
            checkpoint["rng_state"], Mapping
        ):
            raise RuntimeError(
                "Cannot resume RNG state: expected a mapping, got "
                f"{type(checkpoint['rng_state']).__name__}"
            )

        if "rng_states_by_rank" in checkpoint:
            rank_states = checkpoint["rng_states_by_rank"]
            if (
                not isinstance(rank_states, (list, tuple))
                or not rank_states
                or not all(isinstance(state, Mapping) for state in rank_states)
            ):
                raise RuntimeError(
                    "Cannot resume per-rank RNG state: expected a non-empty "
                    "sequence of mappings"
                )

    def resume(self, checkpoint_path: str):
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        if getattr(self, "_is_setup", False):
            raise RuntimeError(
                "resume() must be called before setup(); create a new trainer "
                "for this checkpoint so model, optimizer, scheduler, EMA, AMP, "
                "and distillation state are rebuilt coherently"
            )

        # Continue writing into the interrupted run instead of auto-incrementing
        # a sibling directory (for example exp -> exp2). Standard checkpoints
        # live under <run>/weights; custom checkpoint locations resume beside
        # the checkpoint itself.
        self._resume_save_dir = (
            checkpoint_path.parent.parent
            if checkpoint_path.parent.name == "weights"
            else checkpoint_path.parent
        )
        self.config.project = str(self._resume_save_dir.parent)
        self.config.name = self._resume_save_dir.name
        self.config.exist_ok = True
        if getattr(self, "_is_setup", False):
            self._resume_save_dir.mkdir(parents=True, exist_ok=True)
            self.save_dir = self._resume_save_dir

        logger.info(f"Resuming from {checkpoint_path}")
        checkpoint = load_trusted_torch_file(
            checkpoint_path,
            map_location=self.device,
            context="training resume checkpoint",
        )
        metadata_errors = validate_checkpoint_metadata(checkpoint, strict=False)
        if metadata_errors:
            logger.warning(
                "Resume checkpoint %s predates LibreYOLO checkpoint metadata v%s "
                "or is incomplete: %s. Training will resume through compatibility "
                "mode; the next saved checkpoint will be written with v%s metadata.",
                checkpoint_path,
                SCHEMA_VERSION,
                "; ".join(metadata_errors),
                SCHEMA_VERSION,
            )

        if "epoch" not in checkpoint:
            raise RuntimeError(
                "Cannot resume: checkpoint has no training epoch/resume state. "
                "Load it as pretrained weights and use resume=False instead."
            )

        self._validate_resume_runtime_states(checkpoint)
        self._validate_resume_identity(checkpoint)
        self._restore_checkpoint_config(checkpoint)

        model_state = checkpoint.get("train_model", checkpoint["model"])
        self._prepare_resume_model_architecture(checkpoint, model_state)
        if getattr(self, "_is_setup", False):
            self._load_resume_model_state(model_state)
        else:
            # setup() may still rebuild dataset-dependent heads or inject LoRA.
            # Load only after those structural phases and before optimizer setup.
            self._resume_model_state = model_state

        self.start_epoch = checkpoint["epoch"] + 1
        default_step_count = 0
        if self.train_loader is not None:
            default_step_count = self.start_epoch * self._scheduler_steps_per_epoch()
        has_step_count = "optimizer_step_count" in checkpoint
        raw_step_count = checkpoint.get("optimizer_step_count", default_step_count)
        try:
            self.optimizer_step_count = int(raw_step_count)
            if self.optimizer_step_count < 0:
                raise ValueError
            self._optimizer_step_count_restored = has_step_count
        except (TypeError, ValueError):
            if has_step_count:
                raise RuntimeError(
                    "Cannot resume: invalid optimizer_step_count="
                    f"{raw_step_count!r}"
                )
            self.optimizer_step_count = default_step_count
            self._optimizer_step_count_restored = False

        if "optimizer" in checkpoint:
            if self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                    logger.info("Optimizer state restored")
                except Exception as e:
                    raise RuntimeError(f"Cannot resume optimizer state: {e}") from e
            else:
                # setup() hasn't run yet — defer until the optimizer exists.
                self._resume_optimizer_state = checkpoint["optimizer"]
                logger.info("Optimizer state deferred until after setup()")

        if "scheduler" in checkpoint:
            load_scheduler_state = getattr(
                getattr(self, "lr_scheduler", None), "load_state_dict", None
            )
            if callable(load_scheduler_state):
                try:
                    load_scheduler_state(checkpoint["scheduler"])
                    logger.info("Scheduler state restored")
                except Exception as exc:
                    raise RuntimeError(
                        f"Cannot resume scheduler state: {exc}"
                    ) from exc
            else:
                self._resume_scheduler_state = checkpoint["scheduler"]

        if "distiller" in checkpoint:
            if self.distiller is not None:
                try:
                    self.distiller.loss_modules.load_state_dict(checkpoint["distiller"])
                    logger.info("Distiller adapter state restored")
                except Exception as e:
                    raise RuntimeError(f"Cannot resume distiller state: {e}") from e
            else:
                # setup() hasn't run yet — defer until the distiller exists.
                self._resume_distiller_state = checkpoint["distiller"]
                logger.info("Distiller state deferred until after setup()")

        metric_tracking_compatible = True
        if "best_metric_value" in checkpoint or "best_mAP50_95" in checkpoint:
            checkpoint_metric_key = checkpoint.get("best_metric_key", "metrics/mAP50-95")
            current_metric_key = getattr(self, "best_metric_key", "metrics/mAP50-95")
            if checkpoint_metric_key != current_metric_key:
                metric_tracking_compatible = False
                logger.warning(
                    "Checkpoint best metric key %s differs from current key %s. "
                    "Resetting best metric tracking for this run.",
                    checkpoint_metric_key,
                    current_metric_key,
                )
                self.best_mAP50_95 = 0.0
                self.best_mAP50 = 0.0
                self.best_epoch = 0
            else:
                restored_best = self._as_float(
                    checkpoint.get(
                        "best_metric_value",
                        checkpoint.get("best_mAP50_95", 0.0),
                    )
                )
                restored_map50 = self._as_float(checkpoint.get("best_mAP50", 0.0))
                if restored_best is None or restored_map50 is None:
                    logger.warning(
                        "Checkpoint contains non-finite best metrics. Resetting "
                        "best metric tracking so later finite metrics can improve."
                    )
                    self.best_mAP50_95 = 0.0
                    self.best_mAP50 = 0.0
                    self.best_epoch = 0
                else:
                    self.best_mAP50_95 = restored_best
                    self.best_mAP50 = restored_map50
                    self.best_epoch = int(checkpoint.get("best_epoch", 0))
                    logger.info(
                        f"Restored best metrics: mAP50={self.best_mAP50:.4f}, "
                        f"mAP50-95={self.best_mAP50_95:.4f} "
                        f"(epoch {self.best_epoch})"
                    )
        elif "loss" in checkpoint:
            logger.warning(
                "Old checkpoint format detected (loss-based). Converting to mAP tracking."
            )
            self.best_mAP50_95 = 0.0
            self.best_mAP50 = 0.0
            self.best_epoch = 0

        if "ema" in checkpoint:
            if self.ema_model is not None:
                try:
                    self.ema_model.ema.load_state_dict(checkpoint["ema"])
                    self.ema_model.updates = int(checkpoint.get("ema_updates", 0))
                    logger.info("EMA weights restored")
                except Exception as e:
                    raise RuntimeError(f"Cannot resume EMA state: {e}") from e
            else:
                self._resume_ema_state = checkpoint["ema"]
                self._resume_ema_updates = int(checkpoint.get("ema_updates", 0))
                logger.info("EMA state deferred until after setup()")

        if "scaler" in checkpoint:
            if self.scaler is not None:
                try:
                    self.scaler.load_state_dict(checkpoint["scaler"])
                    logger.info("GradScaler state restored")
                except Exception as e:
                    raise RuntimeError(f"Cannot resume GradScaler state: {e}") from e
            else:
                self._resume_scaler_state = checkpoint["scaler"]
                logger.info("GradScaler state deferred until after setup()")

        rank_rng_states = checkpoint.get("rng_states_by_rank")
        if rank_rng_states:
            current_world_size = get_world_size()
            if len(rank_rng_states) != current_world_size:
                logger.warning(
                    "Checkpoint has RNG state for %d ranks, but current world_size "
                    "is %d; keeping newly seeded per-rank RNG streams",
                    len(rank_rng_states),
                    current_world_size,
                )
                rng_state = None
            else:
                rng_state = rank_rng_states[get_rank()]
        else:
            rng_state = checkpoint.get("rng_state")
            if rng_state and is_distributed():
                logger.warning(
                    "Legacy checkpoint has only rank-zero RNG state; exact "
                    "per-rank RNG replay is unavailable"
                )
                if get_rank() != 0:
                    rng_state = None
        if rng_state:
            if getattr(self, "_is_setup", False):
                try:
                    self._restore_rng_state(rng_state)
                    logger.info("RNG state restored")
                except Exception as e:
                    raise RuntimeError(f"Cannot resume RNG state: {e}") from e
            else:
                # setup() constructs loaders, schedulers, EMA, and AMP objects;
                # restore afterward so setup-time random draws do not perturb
                # the resumed stream.
                self._resume_rng_state = rng_state
                logger.info("RNG state deferred until after setup()")

        raw_patience = checkpoint.get("patience_counter", 0)
        try:
            self.patience_counter = int(raw_patience)
            if self.patience_counter < 0:
                raise ValueError
        except (TypeError, ValueError):
            if "patience_counter" in checkpoint:
                raise RuntimeError(
                    f"Cannot resume: invalid patience_counter={raw_patience!r}"
                )
            self.patience_counter = 0
        if not metric_tracking_compatible:
            self.patience_counter = 0
        logger.info(
            f"Resumed from epoch {self.start_epoch} "
            f"(will train to epoch {self.config.epochs})"
        )
