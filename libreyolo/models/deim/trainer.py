"""DEIMTrainer — BaseTrainer subclass for native DEIM training.

The integration tricks in this file:

1. ``on_forward`` translates LibreYOLO's padded ``(B, max_labels, 5)`` target
   tensor to DEIM's ``list[dict{labels, boxes_cxcywh_normalized}]`` per-image
   format expected by the criterion.

2. ``_setup_optimizer`` builds 4 param groups (backbone wd / no-wd, head
   wd / no-wd) and stamps each with an ``lr_mult`` (backbone groups at
   ``config.backbone_lr_mult`` or the DEIM size-specific default).

3. ``_setup_data`` swaps the parent's standard collate for
   ``DEIMMultiScaleCollate`` (random per-batch resize until stop_epoch) when
   ``config.multi_scale=True``.

4. ``_train_epoch`` is a copy of ``BaseTrainer._train_epoch`` with three
   tweaks: per-epoch ``set_epoch`` propagation to the dataset and collate,
   gradient clipping at ``config.clip_max_norm``, and per-group LR (the
   scheduler's single output is multiplied by each group's ``lr_mult`` instead
   of being applied uniformly).

   Why a wholesale override: ``BaseTrainer._train_epoch`` doesn't expose
   pre-step / post-step hooks. The copy is intentionally kept structurally
   close to the parent so drift is easy to audit; if a third family ends up
   needing the same hooks, promote them into ``BaseTrainer``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import torch
from torch.amp import autocast
from tqdm import tqdm

from ...data import (
    dataloader_seed_kwargs,
    distributed_sampler_seed,
    get_coco_annotation_file,
    get_coco_image_dir,
    get_img_files,
    img2label_paths,
    load_data_config,
)
from ...data.dataset import COCODataset, YOLODataset
from ...training.config import DEIMConfig, TrainConfig, require_training_choice
from ...training.distributed import all_reduce_avg_scalar
from ...training.scheduler import FlatCosineScheduler
from ...training.trainer import BaseTrainer
from .loss import DEIMCriterion
from .matcher import HungarianMatcher
from .transforms import (
    DEIMMultiScaleCollate,
    DEIMPassThroughDataset,
    DEIMTrainTransform,
)


class DEIMTrainer(BaseTrainer):
    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return DEIMConfig

    def get_model_family(self) -> str:
        return "deim"

    def get_model_tag(self) -> str:
        return f"DEIM-{self.config.size}"

    def create_transforms(self):
        preproc = DEIMTrainTransform(
            max_labels=120,
            flip_prob=self.config.flip_prob,
            imgsz=self.config.imgsz,
        )
        return preproc, DEIMPassThroughDataset

    def create_scheduler(self, iters_per_epoch: int):
        require_training_choice(
            self.config.scheduler,
            field="scheduler",
            supported=("flat_cosine",),
            family=self.get_model_family(),
        )
        return FlatCosineScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            no_aug_epochs=self.config.no_aug_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def _scale_lr(self, base_lr: float, param_group: dict) -> float:
        """Apply DEIM's per-group backbone learning-rate multiplier."""
        return base_lr * param_group.get("lr_mult", 1.0)

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        # FGL/DDF are emitted only by the aux/dn paths (no main-loss key);
        # bare ``outputs.get("loss_ddf")`` was always 0. Aggregate over every
        # variant key so the tqdm display reflects the actual loss magnitude.
        def _sum_with_prefix(prefix: str) -> float:
            total = 0.0
            for k, v in outputs.items():
                if k == prefix or k.startswith(prefix + "_"):
                    total += v.item() if isinstance(v, torch.Tensor) else float(v)
            return total

        return {
            "mal": _sum_with_prefix("loss_mal"),
            "bbox": _sum_with_prefix("loss_bbox"),
            "giou": _sum_with_prefix("loss_giou"),
            "fgl": _sum_with_prefix("loss_fgl"),
            "ddf": _sum_with_prefix("loss_ddf"),
        }

    def _setup_device(self) -> torch.device:
        """Override the parent's device autodetect to avoid MPS.

        DEIM's training backward pass crashes on Apple's MPS backend in the
        ``linear_backward`` op (the Integral's 33-bin softmax × W matmul hits
        a known MPS / MetalPerformanceShadersGraph compilation failure). Eval
        mode is fine — this only applies to training. Force CPU when the
        parent would have picked MPS.
        """
        device = super()._setup_device()
        if device.type == "mps":
            import logging

            logging.getLogger(__name__).warning(
                "DEIM training on Apple MPS triggers a torch backward bug "
                "(mps_linear_backward in Metal). Falling back to CPU. "
                "Pass device='cuda' or device='cpu' explicitly to override."
            )
            return torch.device("cpu")
        return device

    def on_num_classes_resolved(self):
        num_classes = self._resolve_num_classes_from_data_config()
        self._sync_wrapped_model_num_classes(num_classes)

    def on_setup(self):
        matcher = HungarianMatcher(
            weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        )
        self.criterion = DEIMCriterion(
            matcher=matcher,
            weight_dict={
                "loss_mal": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            },
            losses=["mal", "boxes", "local"],
            alpha=0.75,
            gamma=1.5,
            num_classes=self.config.num_classes,
            reg_max=32,
        ).to(self.device)

    def on_mosaic_disable(self):
        super().on_mosaic_disable()
        # DEIM's "EMA restart": switch to a constant decay for the final phase.
        if self.ema_model is not None:
            decay = getattr(self.config, "ema_restart_decay", self.config.ema_decay)
            self.ema_model.set_decay(decay)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """AdamW with 4 param groups: {backbone, head} × {wd, no-wd}.

        Each group gets an ``lr_mult`` that ``_train_epoch`` reads back when
        applying the scheduler-returned base LR. The default
        ``backbone_lr_mult=0.5`` matches DEIM's published fine-tune recipe;
        head groups stay at 1.0×.
        """
        require_training_choice(
            self.config.optimizer,
            field="optimizer",
            supported=("adamw",),
            family=self.get_model_family(),
        )
        backbone_wd, backbone_no_wd, head_wd, head_no_wd = [], [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # Match upstream's regex semantics ``(?:norm|bn|bias)`` — substring,
            # not suffix. The previous ``endswith('.bias')`` missed
            # ``self_attn.in_proj_bias`` (PyTorch MHA's fused QKV bias) on five
            # parameters per model, which silently received weight decay.
            is_norm_or_bias = (
                "norm" in name
                or ".bn." in name
                or "bias" in name
                or "lab.scale" in name
            )
            is_backbone = name.startswith("backbone.")
            if is_backbone and is_norm_or_bias:
                backbone_no_wd.append(p)
            elif is_backbone:
                backbone_wd.append(p)
            elif is_norm_or_bias:
                head_no_wd.append(p)
            else:
                head_wd.append(p)

        lr = self.effective_lr
        wd = self.config.weight_decay
        bb_mult_cfg = getattr(self.config, "backbone_lr_mult", None)
        if bb_mult_cfg is None:
            bb_mult_cfg = {
                "n": 0.5,
                "s": 0.5,
                "m": 0.1,
                "l": 0.05,
                "x": 0.01,
            }.get(self.config.size, 0.5)
        bb_mult = float(bb_mult_cfg)

        param_groups = []
        if head_wd:
            param_groups.append(
                {"params": head_wd, "lr": lr, "weight_decay": wd, "lr_mult": 1.0}
            )
        if head_no_wd:
            param_groups.append(
                {"params": head_no_wd, "lr": lr, "weight_decay": 0.0, "lr_mult": 1.0}
            )
        if backbone_wd:
            param_groups.append(
                {
                    "params": backbone_wd,
                    "lr": lr * bb_mult,
                    "weight_decay": wd,
                    "lr_mult": bb_mult,
                }
            )
        if backbone_no_wd:
            param_groups.append(
                {
                    "params": backbone_no_wd,
                    "lr": lr * bb_mult,
                    "weight_decay": 0.0,
                    "lr_mult": bb_mult,
                }
            )

        return torch.optim.AdamW(
            param_groups, betas=(float(self.config.momentum), 0.999)
        )

    def _targets_to_detr(self, imgs: torch.Tensor, targets: torch.Tensor):
        """Translate padded LibreYOLO labels to DETR target dictionaries."""
        B = targets.shape[0]
        # Read actual image size from the batch — multi-scale collate may have
        # resized to a non-default value (576..704), so we cannot trust
        # ``config.imgsz`` here.
        H, W = imgs.shape[-2], imgs.shape[-1]
        scale = torch.tensor([W, H, W, H], device=targets.device, dtype=targets.dtype)

        target_list = []
        for b in range(B):
            t = targets[b]
            # Padding rows are zero in all 5 columns; valid boxes have w>0 and h>0.
            valid = (t[:, 3] > 0) & (t[:, 4] > 0)
            t_valid = t[valid]
            if t_valid.numel() == 0:
                target_list.append(
                    {
                        "labels": torch.zeros(0, dtype=torch.int64, device=self.device),
                        "boxes": torch.zeros(
                            0, 4, dtype=torch.float32, device=self.device
                        ),
                    }
                )
            else:
                target_list.append(
                    {
                        "labels": t_valid[:, 0].long(),
                        "boxes": (t_valid[:, 1:] / scale).clamp(0.0, 1.0),
                    }
                )

        return target_list

    def _compute_criterion_losses(self, outputs: Dict, target_list) -> Dict:
        return self.criterion(outputs, target_list)

    def _format_loss_outputs(self, losses: Dict) -> Dict:
        total = sum(losses.values())
        # Expose every named loss (including aux_/dn_/pre/enc variants) so
        # ``get_loss_components`` can aggregate by prefix. FGL/DDF appear only
        # in the aux/dn paths — bare ``loss_ddf`` would always be 0 otherwise.
        result = {"total_loss": total}
        result.update(losses)
        return result

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        """Forward + loss in one go.

        Translates the ``(B, max_labels, 5)`` ``[class, cx, cy, w, h]`` pixel
        target tensor into DEIM's per-image dict list with cxcywh-normalized
        boxes, then runs model + criterion.
        """
        target_list = self._targets_to_detr(imgs, targets)
        outputs = self.model(imgs, targets=target_list)
        losses = self._compute_criterion_losses(outputs, target_list)
        return self._format_loss_outputs(losses)

    # =========================================================================
    # _setup_data override — wire DEIMMultiScaleCollate (when enabled)
    # =========================================================================

    def _setup_data(self):
        """Mirror of ``BaseTrainer._setup_data`` but uses ``DEIMMultiScaleCollate``.

        Built by hand instead of inheriting because ``create_dataloader`` doesn't
        expose ``collate_fn`` and we need our epoch-aware collate. Dataset-build
        logic is duplicated from the parent for clarity.
        """
        from torch.utils.data import DataLoader

        img_size = self.input_size
        preproc, MosaicDatasetClass = self.create_transforms()

        if self.config.data:
            data_cfg = load_data_config(self.config.data)
            data_dir = data_cfg["root"]
            self.num_classes = data_cfg.get("nc", self.config.num_classes)

            ann_file = Path(data_dir) / "annotations" / "instances_train2017.json"
            coco_ann_file = get_coco_annotation_file(data_cfg, "train")
            img_files = data_cfg.get("train_img_files")
            label_files = data_cfg.get("train_label_files")

            if coco_ann_file:
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file=coco_ann_file,
                    name=get_coco_image_dir(data_cfg, "train", "train2017"),
                    img_size=img_size,
                    preproc=preproc,
                    num_classes=int(self.num_classes),
                    names=data_cfg.get("names"),
                )
            elif img_files:
                train_dataset = YOLODataset(
                    img_files=img_files,
                    label_files=label_files,
                    img_size=img_size,
                    preproc=preproc,
                    num_classes=int(self.num_classes),
                )
            elif ann_file.exists():
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file="instances_train2017.json",
                    name="train2017",
                    img_size=img_size,
                    preproc=preproc,
                    num_classes=int(self.num_classes),
                    names=data_cfg.get("names"),
                )
            else:
                train_path = data_cfg.get("train", "images/train")
                try:
                    img_files = get_img_files(train_path, prefix=data_dir)
                except (FileNotFoundError, ValueError):
                    img_files = []
                if not img_files:
                    raise FileNotFoundError(f"No images found in {train_path}")
                label_files = img2label_paths(img_files)
                train_dataset = YOLODataset(
                    img_files=img_files,
                    label_files=label_files,
                    img_size=img_size,
                    preproc=preproc,
                    num_classes=int(self.num_classes),
                )
        elif self.config.data_dir:
            data_dir = self.config.data_dir
            self.num_classes = self.config.num_classes
            if (Path(data_dir) / "annotations").exists():
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file="instances_train2017.json",
                    name="train2017",
                    img_size=img_size,
                    preproc=preproc,
                    num_classes=int(self.num_classes),
                )
            else:
                train_dataset = YOLODataset(
                    data_dir=data_dir,
                    split="train",
                    img_size=img_size,
                    preproc=preproc,
                    num_classes=int(self.num_classes),
                )
        else:
            raise ValueError("Either 'data' or 'data_dir' must be specified")

        train_dataset.enable_image_cache(getattr(self.config, "cache", False))

        train_dataset = MosaicDatasetClass(
            dataset=train_dataset,
            img_size=img_size,
            mosaic=True,
            preproc=preproc,
            degrees=self.config.degrees,
            translate=self.config.translate,
            mosaic_scale=self.config.mosaic_scale,
            mixup_scale=self.config.mixup_scale,
            shear=self.config.shear,
            enable_mixup=self.config.mixup_prob > 0,
            mosaic_prob=self.config.mosaic_prob,
            mixup_prob=self.config.mixup_prob,
        )

        # Wire stop_epoch on the dataset wrapper so set_epoch can disable
        # strong augs at the right moment.
        stop_epoch = int(
            self.config.epochs
            * float(getattr(self.config, "aug_stop_epoch_ratio", 1.0))
        )
        if hasattr(train_dataset, "set_stop_epoch"):
            train_dataset.set_stop_epoch(stop_epoch)

        # Multi-scale collate (or default yolox_collate_fn).
        if getattr(self.config, "multi_scale", False):
            collate_fn = DEIMMultiScaleCollate(
                base_size=self.config.imgsz,
                base_size_repeat=getattr(self.config, "base_size_repeat", 3),
                stop_epoch=stop_epoch,
            )
        else:
            from ...data.dataset import yolox_collate_fn

            collate_fn = yolox_collate_fn

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
        visible_samples = len(sampler) if sampler is not None else len(train_dataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_batch,
            num_workers=self.config.workers,
            shuffle=sampler is None,
            sampler=sampler,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=visible_samples >= per_rank_batch,
            **dataloader_seed_kwargs(
                getattr(self.config, "seed", None),
                rank=self.rank,
                distributed=self.is_distributed,
            ),
        )

        return train_dataset

    # =========================================================================
    # _train_epoch override — set_epoch propagation, grad clip, per-group LR
    # =========================================================================

    def _train_epoch(
        self, epoch: int
    ) -> Tuple[float, Optional[Dict[str, float]], Dict[str, float], Dict[str, float]]:
        """Copy of ``BaseTrainer._train_epoch`` with three DEIM-specific tweaks:

        1. Propagate the current epoch to dataset + collate (drives stop_epoch
           augmentation/multi-scale gating).
        2. Apply gradient clipping at ``config.clip_max_norm`` before the
           optimizer step.
        3. Apply per-group LR multipliers (the scheduler returns one base LR;
           each param group's ``lr_mult`` scales it).
        """
        # Gradient accumulation is opt-in; delegate when enabled.
        if self._accum_steps > 1:
            return self._train_epoch_accum(epoch)

        # 1. Epoch propagation.
        ds = self.train_loader.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        cf = getattr(self.train_loader, "collate_fn", None)
        if cf is not None and hasattr(cf, "set_epoch"):
            cf.set_epoch(epoch)

        should_clip = self._should_clip_gradients()

        self.model.train()
        self._enforce_frozen_bn_eval()
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            total=len(self.train_loader),
        )

        total_loss = 0.0
        num_batches = 0
        loss_component_sums: Dict[str, float] = {}

        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 5:
                imgs, targets, img_infos, img_ids, polygons = batch
            else:
                imgs, targets, img_infos, img_ids = batch
                polygons = None
            self.current_iter = epoch * len(self.train_loader) + batch_idx

            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.scaler is not None:
                with autocast("cuda"):
                    outputs = self.on_forward(imgs, targets, polygons=polygons)
                    loss = outputs["total_loss"]
                loss = self._require_finite_training_loss(
                    loss,
                    context=f"Epoch {epoch + 1} batch {batch_idx + 1}",
                )
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                if should_clip:
                    self.scaler.unscale_(self.optimizer)
                    self._clip_gradients()
                step_succeeded = self._run_optimizer_step()
            else:
                outputs = self.on_forward(imgs, targets, polygons=polygons)
                loss = self._require_finite_training_loss(
                    outputs["total_loss"],
                    context=f"Epoch {epoch + 1} batch {batch_idx + 1}",
                )
                self.optimizer.zero_grad()
                loss.backward()
                if should_clip:
                    self._clip_gradients()
                step_succeeded = self._run_optimizer_step()

            loss_val = loss.item()
            loss_components = self._scalar_mapping(self.get_loss_components(outputs))
            total_loss += loss_val
            for name, value in loss_components.items():
                loss_component_sums[name] = loss_component_sums.get(name, 0.0) + value
            del outputs, loss

            # 3. EMA, optimizer-step counter, and per-group LR advance only
            # when GradScaler actually applied the optimizer update.
            base_lr = self._advance_optimizer_dependent_state(step_succeeded)
            num_batches += 1

            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{base_lr:.6f}"}
            postfix.update({k: f"{v:.4f}" for k, v in loss_components.items()})
            pbar.set_postfix(postfix)

        num_batches = max(num_batches, 1)
        avg_loss = total_loss / num_batches
        avg_loss_components = {
            name: value / num_batches for name, value in loss_component_sums.items()
        }

        val_metrics = None
        if self._should_validate_epoch(epoch):
            val_metrics = self._validate_epoch(epoch)

        return avg_loss, val_metrics, avg_loss_components, self._current_lrs()

    def _train_epoch_accum(
        self, epoch: int
    ) -> Tuple[float, Optional[Dict[str, float]], Dict[str, float], Dict[str, float]]:
        """``_train_epoch`` variant with gradient accumulation (``_accum_steps`` > 1).

        Same DEIM tweaks as ``_train_epoch`` (epoch propagation, clipping,
        per-group LR), but gradients accumulate over ``accum`` micro-batches and
        the optimizer step, clipping, EMA and LR update fire once per window.
        """
        # 1. Epoch propagation.
        ds = self.train_loader.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        cf = getattr(self.train_loader, "collate_fn", None)
        if cf is not None and hasattr(cf, "set_epoch"):
            cf.set_epoch(epoch)

        should_clip = self._should_clip_gradients()

        self.model.train()
        self._enforce_frozen_bn_eval()
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            total=len(self.train_loader),
            disable=not sys.stderr.isatty(),
        )

        accum = self._accum_steps
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / accum))
        total_loss = 0.0
        num_batches = 0
        loss_component_sums: Dict[str, float] = {}
        window_local_samples = 0
        base_lr = self.optimizer.param_groups[0]["lr"]

        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 5:
                imgs, targets, img_infos, img_ids, polygons = batch
            else:
                imgs, targets, img_infos, img_ids = batch
                polygons = None

            is_opt_step = (batch_idx + 1) % accum == 0 or batch_idx == len(self.train_loader) - 1
            opt_step = epoch * steps_per_epoch + batch_idx // accum
            self.current_iter = opt_step

            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if batch_idx % accum == 0:
                self.optimizer.zero_grad(set_to_none=True)
                window_local_samples = 0
            batch_samples = int(imgs.shape[0])
            if batch_samples <= 0:
                raise ValueError("gradient accumulation received an empty micro-batch")
            window_local_samples += batch_samples

            if self.scaler is not None:
                with autocast("cuda"):
                    outputs = self.on_forward(imgs, targets, polygons=polygons)
                    total_loss_raw = self._require_finite_training_loss(
                        outputs["total_loss"],
                        context=f"Epoch {epoch + 1} batch {batch_idx + 1}",
                    )
                    loss = total_loss_raw * batch_samples
                self.scaler.scale(loss).backward()
                if is_opt_step:
                    self.scaler.unscale_(self.optimizer)
                    sample_divisor = all_reduce_avg_scalar(
                        window_local_samples,
                        device=self.device,
                        min_value=1.0,
                    )
                    self._normalize_accumulated_gradients(sample_divisor)
                    if should_clip:
                        self._clip_gradients()
                    step_succeeded = self._run_optimizer_step()
            else:
                outputs = self.on_forward(imgs, targets, polygons=polygons)
                total_loss_raw = self._require_finite_training_loss(
                    outputs["total_loss"],
                    context=f"Epoch {epoch + 1} batch {batch_idx + 1}",
                )
                loss = total_loss_raw * batch_samples
                loss.backward()
                if is_opt_step:
                    sample_divisor = all_reduce_avg_scalar(
                        window_local_samples,
                        device=self.device,
                        min_value=1.0,
                    )
                    self._normalize_accumulated_gradients(sample_divisor)
                    if should_clip:
                        self._clip_gradients()
                    step_succeeded = self._run_optimizer_step()

            if is_opt_step:
                # 3. EMA, optimizer-step counter, and per-group LR advance
                # only after an applied optimizer update.
                base_lr = self._advance_optimizer_dependent_state(step_succeeded)

            loss_val = total_loss_raw.item()
            loss_components = self._scalar_mapping(self.get_loss_components(outputs))
            total_loss += loss_val
            for name, value in loss_components.items():
                loss_component_sums[name] = loss_component_sums.get(name, 0.0) + value
            num_batches += 1
            del outputs, loss

            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{base_lr:.6f}"}
            postfix.update({k: f"{v:.4f}" for k, v in loss_components.items()})
            pbar.set_postfix(postfix)

        num_batches = max(num_batches, 1)
        avg_loss = total_loss / num_batches
        avg_loss_components = {
            name: value / num_batches for name, value in loss_component_sums.items()
        }

        val_metrics = None
        if self._should_validate_epoch(epoch):
            val_metrics = self._validate_epoch(epoch)

        return avg_loss, val_metrics, avg_loss_components, self._current_lrs()
