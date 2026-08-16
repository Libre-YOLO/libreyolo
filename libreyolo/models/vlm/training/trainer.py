"""LoRA SFT trainer for the LibreVLM tier.

Deliberately NOT a ``BaseTrainer`` subclass: that chassis assumes stacked image
tensors, detection losses, EMA, and mAP validation, none of which apply to an
autoregressive fine-tune. This trainer keeps the repo's user-facing surface
(run directories, callbacks, loggers, results dict) and owns a compact loop:
cross-entropy on assistant tokens, gradient accumulation, cosine schedule with
warmup, best/last adapter checkpoints.

Checkpoints follow the directory contract in :mod:`.checkpoint`; best/last
selection uses validation loss (validation mAP lands with the tier's ``val()``
in a later iteration, as documented in the design docs).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ....data import load_data_config
from ....training.callbacks import (
    TrainCallbackList,
    TrainCallbacks,
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from ....training.loggers import resolve_loggers
from .checkpoint import (
    is_vlm_checkpoint,
    read_contract,
    save_vlm_checkpoint,
    validate_lora_artifact,
)
from .collate import VLMChatCollator
from .data import VLMDetectDataset, resolve_split_source
from .recipes import VLMTrainRecipe, get_recipe
from .targets import FamilyFormat

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "VLM fine-tuning requires the 'vlm-train' extra. Install with:\n"
    "    pip install 'libreyolo[vlm-train]'"
)

__all__ = [
    "VLMDetectionTrainer",
    "VLMTrainConfig",
    "require_vlm_lora_dependencies",
    "resolve_vlm_training_device",
    "validate_vlm_resume_checkpoint",
]


@dataclass
class VLMTrainConfig:
    """Resolved VLM training configuration (user kwargs over recipe defaults)."""

    data: str = ""
    epochs: int = 10
    batch: int = 1
    accumulate: int = 8
    lr0: Optional[float] = None
    lora: bool = True
    output_dir: str = "runs/vlm/train"
    project: Optional[str] = None
    name: Optional[str] = None
    exist_ok: bool = True
    workers: int = 0
    seed: int = 0
    device: Optional[str] = None
    gradient_checkpointing: bool = True
    hflip: float = 0.5
    vram_check: bool = True
    allow_download_scripts: bool = False
    resume: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _normalize_names(raw) -> Dict[int, str]:
    if isinstance(raw, dict):
        names = {int(k): str(v) for k, v in raw.items()}
    elif isinstance(raw, (list, tuple)):
        names = {i: str(v) for i, v in enumerate(raw)}
    else:
        raise ValueError("Dataset YAML must define names as a list or an id mapping.")
    if not names or any(not name.strip() for name in names.values()):
        raise ValueError("Dataset YAML names must contain non-empty labels.")
    if set(names) != set(range(len(names))):
        raise ValueError("Dataset YAML names must use contiguous ids starting at 0.")
    normalized = [name.strip().lower() for name in names.values()]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Dataset YAML names must be unique case-insensitively.")
    return names


def _finite_real(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _accumulation_group_size(step: int, total_steps: int, accumulate: int) -> int:
    """Return the true size of the accumulation group containing ``step``."""
    group_start = ((step - 1) // accumulate) * accumulate
    return min(accumulate, total_steps - group_start)


def resolve_vlm_training_device(requested, fallback) -> torch.device:
    """Normalize the standard LibreYOLO device forms for VLM training."""
    if requested is None:
        return torch.device(fallback)
    if isinstance(requested, bool):
        raise ValueError(
            f"device must be a device string or integer, got {requested!r}."
        )
    if isinstance(requested, int):
        requested = f"cuda:{requested}"
    if isinstance(requested, torch.device):
        return requested
    if not isinstance(requested, str):
        raise ValueError(
            f"device must be a device string or integer, got {requested!r}."
        )
    normalized = requested.strip().lower()
    if normalized in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized.isdigit():
        normalized = f"cuda:{normalized}"
    try:
        return torch.device(normalized)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid VLM training device {requested!r}.") from exc


def require_vlm_lora_dependencies() -> None:
    """Require the PEFT version used by the VLM adapter contract."""
    try:
        import peft  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    try:
        raw_version = version("peft")
    except PackageNotFoundError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    try:
        from packaging.version import InvalidVersion, Version  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    try:
        supported = Version(raw_version) >= Version("0.17.0")
    except InvalidVersion:
        supported = False
    if not supported:
        raise ImportError(
            f"VLM fine-tuning requires peft>=0.17.0, found {raw_version!r}.\n"
            + _INSTALL_HINT
        )


def validate_vlm_resume_checkpoint(directory, wrapper) -> Path:
    """Validate a LoRA resume checkpoint against a pristine base wrapper."""
    directory = Path(directory)
    if not is_vlm_checkpoint(directory):
        raise FileNotFoundError(
            f"resume={str(directory)!r} is not a VLM checkpoint directory "
            "(missing libreyolo_vlm.json)."
        )
    if getattr(wrapper, "_checkpoint_dir", None) is not None:
        raise ValueError(
            "resume= requires a pristine base wrapper, not an inference-loaded "
            "checkpoint wrapper."
        )

    contract = read_contract(directory)
    expected = {
        "family": wrapper.FAMILY,
        "size": wrapper.size,
        "base_repo": wrapper.HF_REPOS[wrapper.size],
        "base_revision": wrapper.HF_REVISIONS.get(wrapper.size),
        "bbox_key": wrapper.BBOX_KEY,
        "coord_divisor": float(wrapper.COORD_DIVISOR),
        "box_format": wrapper.BOX_FORMAT,
    }
    mismatches = [key for key, value in expected.items() if contract[key] != value]
    if mismatches:
        raise ValueError(
            f"resume={str(directory)!r} does not match the loaded base model "
            f"contract fields: {', '.join(mismatches)}."
        )

    custom_prompt = getattr(wrapper, "_custom_prompt", None)
    expected_prompt = (
        custom_prompt
        if custom_prompt is not None
        else wrapper._format_detection_prompt(", ".join(contract["names"]))
    )
    if contract["prompt"] != expected_prompt:
        raise ValueError(
            f"resume={str(directory)!r} uses a prompt that the loaded base "
            "wrapper cannot reconstruct. Load the base with the checkpoint's "
            "exact prompt= before resuming."
        )
    try:
        validate_lora_artifact(directory)
    except ValueError as exc:
        raise ValueError(f"resume={str(directory)!r} is not loadable: {exc}") from exc
    return directory


class VLMDetectionTrainer:
    """Drive one LoRA (or full) detection fine-tune of a LibreVLM wrapper."""

    def __init__(
        self,
        wrapper,
        data: str,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> None:
        if not data:
            raise ValueError("train() requires data=<dataset yaml>.")
        if getattr(wrapper, "_checkpoint_dir", None) is not None:
            raise ValueError(
                "VLMDetectionTrainer requires a pristine base wrapper. A wrapper "
                "loaded from a VLM checkpoint already has its adapter merged; "
                "construct the corresponding base model and pass resume= instead."
            )
        known = set(VLMTrainConfig.__dataclass_fields__) - {"data", "extra"}
        config_kwargs = {k: v for k, v in kwargs.items() if k in known}
        extra = {k: v for k, v in kwargs.items() if k not in known}
        if extra:
            raise ValueError(
                "Unsupported VLM train() kwargs: " + ", ".join(sorted(extra)) + "."
            )
        self.config = VLMTrainConfig(data=data, extra=extra, **config_kwargs)
        for name in (
            "lora",
            "exist_ok",
            "gradient_checkpointing",
            "vram_check",
            "allow_download_scripts",
        ):
            value = getattr(self.config, name)
            if type(value) is not bool:
                raise ValueError(f"{name} must be a bool, got {value!r}.")
        for name in ("epochs", "batch", "accumulate"):
            value = getattr(self.config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
        if (
            isinstance(self.config.workers, bool)
            or not isinstance(self.config.workers, int)
            or self.config.workers < 0
        ):
            raise ValueError(
                f"workers must be an integer >= 0, got {self.config.workers!r}."
            )
        hflip = _finite_real(self.config.hflip)
        if hflip is None or not 0.0 <= hflip <= 1.0:
            raise ValueError(
                f"hflip must be finite and within [0, 1], got {self.config.hflip!r}."
            )
        lr0 = _finite_real(self.config.lr0) if self.config.lr0 is not None else None
        if self.config.lr0 is not None and (lr0 is None or lr0 <= 0.0):
            raise ValueError(
                f"lr0 must be a finite positive number, got {self.config.lr0!r}."
            )
        if isinstance(self.config.seed, bool) or not isinstance(self.config.seed, int):
            raise ValueError(f"seed must be an integer, got {self.config.seed!r}.")
        self.wrapper = wrapper
        self.recipe: VLMTrainRecipe = get_recipe(wrapper.FAMILY)
        self.callbacks = TrainCallbackList(callbacks)
        for logger_cb in resolve_loggers(loggers):
            self.callbacks.append(logger_cb)
        self.save_dir = self._resolve_save_dir()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _resolve_save_dir(self) -> Path:
        cfg = self.config
        output = Path(cfg.output_dir)
        project = Path(cfg.project) if cfg.project else output.parent
        name = cfg.name if cfg.name else output.name
        run_dir = Path(project) / str(name)
        if run_dir.exists() and not cfg.exist_ok:
            base = run_dir
            counter = 2
            while run_dir.exists():
                run_dir = base.with_name(f"{base.name}{counter}")
                counter += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _resolve_device(self) -> torch.device:
        return resolve_vlm_training_device(self.config.device, self.wrapper.device)

    @staticmethod
    def _training_dtype(device: torch.device) -> torch.dtype:
        if device.type != "cuda":
            return torch.float32
        with torch.cuda.device(device):
            bf16_supported = torch.cuda.is_bf16_supported()
        if not bf16_supported:
            raise RuntimeError(
                "VLM CUDA fine-tuning requires a BF16-capable GPU. Unscaled FP16 "
                "training is intentionally unsupported; use a newer GPU or CPU."
            )
        return torch.bfloat16

    def _check_vram_for_full_ft(self, device: torch.device) -> None:
        if self.config.lora:
            return
        if device.type != "cuda":
            logger.warning(
                "Full fine-tuning on %s will be extremely slow; lora=True is "
                "the supported path on this hardware.",
                device.type,
            )
            return
        if not self.config.vram_check:
            return
        params = sum(p.numel() for p in self.wrapper.model.parameters())
        # Full FT with AdamW: weights + grads + two fp32 moments, before
        # activations. ~16 bytes/param is the optimistic floor in bf16.
        needed = params * 16
        total = torch.cuda.get_device_properties(device).total_memory
        if needed > total:
            raise RuntimeError(
                f"Full fine-tuning needs roughly {needed / 1e9:.0f} GB for "
                f"optimizer state alone; this GPU has {total / 1e9:.0f} GB. "
                "Use lora=True (the default), or pass vram_check=False to "
                "proceed anyway."
            )

    def _resolve_resume_dir(self) -> Optional[Path]:
        resume = self.config.resume
        if not resume:
            return None
        candidate = (
            self.save_dir / "weights" / "last" if resume is True else Path(resume)
        )
        if not self.config.lora:
            raise NotImplementedError(
                "resume= is only supported for LoRA training (lora=True)."
            )
        self._validate_resume_contract(candidate)
        return candidate

    def _validate_resume_contract(self, directory: Path) -> None:
        validate_vlm_resume_checkpoint(directory, self.wrapper)

    def _build_train_model(self, resume_dir: Optional[Path]):
        """Return the trainable module: a PeftModel (LoRA) or the base model."""
        if self.config.lora:
            require_vlm_lora_dependencies()
            try:
                from peft import LoraConfig, PeftModel, get_peft_model
            except ImportError as exc:
                raise ImportError(_INSTALL_HINT) from exc
            if resume_dir is not None:
                model = PeftModel.from_pretrained(
                    self.wrapper.model, str(resume_dir), is_trainable=True
                )
                logger.warning(
                    "Resumed adapter weights from %s (optimizer state starts fresh).",
                    resume_dir,
                )
            else:
                lora_config = LoraConfig(
                    r=self.recipe.lora_r,
                    lora_alpha=self.recipe.lora_alpha,
                    lora_dropout=self.recipe.lora_dropout,
                    target_modules=self.recipe.target_modules,
                    bias="none",
                )
                model = get_peft_model(self.wrapper.model, lora_config)
            try:
                injected = [
                    name
                    for name, module in model.named_modules()
                    if getattr(module, "lora_A", None) is not None
                ]
                if not injected:
                    raise RuntimeError(
                        f"LoRA recipe for {self.wrapper.FAMILY!r} matched no "
                        "modules; the base model layout changed. This is a "
                        "LibreYOLO bug."
                    )
                leaks = [
                    name
                    for name in injected
                    if any(prefix in name for prefix in self.recipe.frozen_prefixes)
                ]
                if leaks:
                    raise RuntimeError(
                        f"LoRA recipe for {self.wrapper.FAMILY!r} leaked adapters "
                        f"into frozen scope: {leaks[:3]}. This is a LibreYOLO bug."
                    )
            except BaseException:
                unload = getattr(model, "unload", None)
                if not callable(unload):
                    raise RuntimeError(
                        "LoRA setup failed and PEFT cannot unload the injected "
                        "adapter safely. Reconstruct the base wrapper before retrying."
                    )
                self.wrapper.model = unload()
                self.wrapper.model.eval()
                raise
            logger.info("Injected LoRA adapters into %d modules.", len(injected))
            return model

        # Full fine-tune: freeze the recipe's frozen prefixes, train the rest.
        frozen = 0
        for name, param in self.wrapper.model.named_parameters():
            if any(name.startswith(prefix) for prefix in self.recipe.frozen_prefixes):
                param.requires_grad_(False)
                frozen += 1
        logger.info("Full fine-tune: froze %d frozen-scope parameters.", frozen)
        return self.wrapper.model

    def _build_dataloaders(
        self, data_cfg: Dict, names: Dict[int, str], fmt: FamilyFormat
    ):
        cfg = self.config
        train_source = resolve_split_source(data_cfg, "train")
        if not train_source:
            raise ValueError(f"Dataset {cfg.data!r} has no train split.")
        val_source = resolve_split_source(data_cfg, "val")

        collator = VLMChatCollator(
            self.wrapper.processor, max_length_warn=self.recipe.max_length_warn
        )
        pin = self._resolve_device().type == "cuda"
        train_set = VLMDetectDataset(
            train_source,
            names,
            fmt,
            augment=cfg.hflip > 0,
            hflip_p=cfg.hflip,
            seed=cfg.seed,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch,
            shuffle=True,
            num_workers=cfg.workers,
            collate_fn=collator,
            generator=torch.Generator().manual_seed(cfg.seed),
            pin_memory=pin,
            persistent_workers=cfg.workers > 0,
        )
        val_loader = None
        if val_source:
            val_set = VLMDetectDataset(val_source, names, fmt, augment=False)
            val_loader = DataLoader(
                val_set,
                batch_size=cfg.batch,
                shuffle=False,
                num_workers=cfg.workers,
                collate_fn=collator,
                pin_memory=pin,
                persistent_workers=cfg.workers > 0,
            )
        return train_loader, val_loader

    def _prepare_optimization(
        self,
        train_model,
        train_loader,
        device: torch.device,
        training_dtype: torch.dtype,
    ):
        """Move the live model and build optimizer state before callbacks run."""
        cfg = self.config
        train_model.to(device=device, dtype=training_dtype)
        self.wrapper.device = device
        self.wrapper._model_dtype = training_dtype
        train_model.train()
        if cfg.gradient_checkpointing and hasattr(
            train_model, "gradient_checkpointing_enable"
        ):
            train_model.gradient_checkpointing_enable()
            if hasattr(train_model, "enable_input_require_grads"):
                train_model.enable_input_require_grads()

        lr0 = (
            cfg.lr0
            if cfg.lr0 is not None
            else (self.recipe.lr0 if cfg.lora else self.recipe.full_ft_lr0)
        )
        trainable = [p for p in train_model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("VLM training produced no trainable parameters.")
        optimizer = torch.optim.AdamW(
            trainable, lr=lr0, weight_decay=self.recipe.weight_decay
        )
        steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.accumulate))
        total_steps = steps_per_epoch * cfg.epochs
        warmup_steps = max(1, int(total_steps * self.recipe.warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=training_dtype)
            if device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        return lr0, trainable, optimizer, scheduler, autocast_ctx

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        start_time = time.time()
        torch.manual_seed(cfg.seed)

        device = self._resolve_device()
        training_dtype = self._training_dtype(device)
        self._check_vram_for_full_ft(device)

        resume_dir = self._resolve_resume_dir()

        data_cfg = load_data_config(cfg.data, allow_scripts=cfg.allow_download_scripts)
        names = _normalize_names(data_cfg.get("names"))
        # Vocabulary comes from the dataset; sticky on the wrapper from here on.
        self.wrapper.set_classes([names[i] for i in range(len(names))])
        fmt = FamilyFormat.from_model(self.wrapper)
        train_loader, val_loader = self._build_dataloaders(data_cfg, names, fmt)

        train_model = self._build_train_model(resume_dir)
        try:
            lr0, trainable, optimizer, scheduler, autocast_ctx = (
                self._prepare_optimization(
                    train_model, train_loader, device, training_dtype
                )
            )
        except BaseException:
            self._restore_inference_model(train_model)
            raise

        weights_dir = self.save_dir / "weights"
        best_metric: Optional[float] = None
        best_epoch: Optional[int] = None
        final_loss = float("nan")
        completed_epochs = 0
        config_dump = {
            **asdict(cfg),
            "lr0": lr0,
            "resume": str(cfg.resume) if cfg.resume else None,
            "family": self.wrapper.FAMILY,
        }
        config_dump.pop("extra", None)

        try:
            self.callbacks.on_train_start(
                TrainStartEvent(
                    start_epoch=1,
                    total_epochs=cfg.epochs,
                    model_family=self.wrapper.FAMILY,
                    model_size=self.wrapper.size,
                    task="detect",
                    save_dir=str(self.save_dir),
                    config=config_dump,
                )
            )
        except BaseException:
            self._restore_inference_model(train_model)
            raise

        try:
            for epoch in range(1, cfg.epochs + 1):
                epoch_start = time.time()
                running, seen = 0.0, 0
                optimizer.zero_grad(set_to_none=True)
                progress = tqdm(
                    train_loader,
                    desc=f"vlm train {epoch}/{cfg.epochs}",
                    unit="batch",
                    leave=False,
                )
                for step, batch in enumerate(progress, 1):
                    batch = self._to_device(batch, device)
                    with autocast_ctx:
                        loss = train_model(**batch).loss
                    if loss.numel() != 1 or not bool(
                        torch.isfinite(loss.detach()).item()
                    ):
                        raise FloatingPointError(
                            f"Non-finite scalar training loss at epoch {epoch}, batch {step}."
                        )
                    group_size = _accumulation_group_size(
                        step, len(train_loader), cfg.accumulate
                    )
                    (loss / group_size).backward()
                    running += float(loss.detach())
                    seen += 1
                    if step % cfg.accumulate == 0 or step == len(train_loader):
                        self._clip_gradients(trainable, self.recipe.clip_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    progress.set_postfix(loss=f"{running / max(seen, 1):.4f}")

                train_loss = running / max(seen, 1)
                final_loss = train_loss
                val_metrics: Dict[str, float] = {}
                if val_loader is not None:
                    val_metrics["val/loss"] = self._eval_loss(
                        train_model, val_loader, device, autocast_ctx
                    )
                metric_name = "val/loss" if val_loader is not None else "train/loss"
                current = val_metrics.get("val/loss", train_loss)
                is_best = best_metric is None or current < best_metric
                if is_best:
                    best_metric = current
                    best_epoch = epoch

                metrics = {"train/loss": train_loss, "epoch": epoch, **val_metrics}
                save_vlm_checkpoint(
                    weights_dir / "last",
                    peft_model=train_model,
                    processor=self.wrapper.processor,
                    wrapper=self.wrapper,
                    metrics=metrics,
                )
                if is_best:
                    save_vlm_checkpoint(
                        weights_dir / "best",
                        peft_model=train_model,
                        processor=self.wrapper.processor,
                        wrapper=self.wrapper,
                        metrics=metrics,
                    )
                completed_epochs = epoch

                self.callbacks.on_train_epoch_end(
                    TrainEpochEvent(
                        epoch=epoch,
                        total_epochs=cfg.epochs,
                        model_family=self.wrapper.FAMILY,
                        model_size=self.wrapper.size,
                        task="detect",
                        save_dir=str(self.save_dir),
                        train_loss=train_loss,
                        train_loss_items={"loss": train_loss},
                        lr={"lr0": optimizer.param_groups[0]["lr"]},
                        val_metrics=val_metrics,
                        validated=val_loader is not None,
                        is_best=is_best,
                        current_metric=current,
                        current_metric_name=metric_name,
                        best_metric=best_metric,
                        best_metric_name=metric_name,
                        best_epoch=best_epoch,
                        epoch_seconds=time.time() - epoch_start,
                    )
                )
        except BaseException as exc:
            self.callbacks.on_train_exception(
                TrainExceptionEvent(
                    epoch=completed_epochs or None,
                    total_epochs=cfg.epochs,
                    model_family=self.wrapper.FAMILY,
                    model_size=self.wrapper.size,
                    task="detect",
                    save_dir=str(self.save_dir),
                    exception=exc,
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                    elapsed_seconds=time.time() - start_time,
                )
            )
            raise
        finally:
            self._restore_inference_model(train_model)

        results: Dict[str, Any] = {
            "save_dir": str(self.save_dir),
            "best": str(weights_dir / "best"),
            "last": str(weights_dir / "last"),
            "epochs": completed_epochs,
            "final_loss": final_loss,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "metric_name": "val/loss" if val_loader is not None else "train/loss",
        }
        self.callbacks.on_train_end(
            TrainEndEvent(
                total_epochs=cfg.epochs,
                completed_epochs=completed_epochs,
                model_family=self.wrapper.FAMILY,
                model_size=self.wrapper.size,
                task="detect",
                save_dir=str(self.save_dir),
                final_loss=final_loss,
                best_metric=best_metric,
                best_epoch=best_epoch,
                total_seconds=time.time() - start_time,
                results=results,
            )
        )
        return results

    # ------------------------------------------------------------------
    # Pieces
    # ------------------------------------------------------------------

    @staticmethod
    def _to_device(batch, device: torch.device):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @staticmethod
    def _clip_gradients(parameters, max_norm: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, error_if_nonfinite=True
        )

    def _eval_loss(self, model, loader, device, autocast_ctx) -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch, device)
                with autocast_ctx:
                    loss = model(**batch).loss
                if loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).item()):
                    raise FloatingPointError("Non-finite scalar validation loss.")
                total += float(loss.detach())
                count += 1
        model.train()
        return total / max(count, 1)

    def _restore_inference_model(self, train_model) -> None:
        """Leave the wrapper usable for predict() after training.

        The LoRA path trained a PeftModel around ``self.wrapper.model``;
        merging the trained adapter into dense weights hands the wrapper back a
        plain model whose ``predict()`` reflects the fine-tune. The full-FT
        path trained the wrapper's model in place, so only eval() is needed.
        """
        try:
            from peft import PeftModel
        except ImportError:
            self.wrapper.model.eval()
        else:
            if isinstance(train_model, PeftModel):
                self.wrapper.model = train_model.merge_and_unload()
                logger.info("Merged trained LoRA adapters into the loaded model.")
        self.wrapper.model.eval()
        try:
            parameter = next(self.wrapper.model.parameters())
        except StopIteration:
            return
        self.wrapper.device = parameter.device
        self.wrapper._model_dtype = parameter.dtype
