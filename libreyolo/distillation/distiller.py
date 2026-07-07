"""
Model-agnostic distillation orchestrator.

Wires together:
    1. A frozen teacher model
    2. FeatureHookManagers on both teacher and student
    3. Per-scale loss modules (MGD or CWD)

The Distiller is architecture-agnostic — it receives distillation configs
(tap points + channel dims) from the model wrappers and handles the rest.

Usage::

    from libreyolo.distillation import Distiller

    distiller = Distiller(
        teacher_model=teacher.model,      # nn.Module
        student_model=student.model,      # nn.Module
        teacher_config=teacher.get_distill_config(),
        student_config=student.get_distill_config(),
        loss_type="mgd",
    )

    # In training loop:
    teacher_out = distiller.teacher_forward(images)  # no_grad internally
    student_out = model(images, targets)              # normal forward, hooks capture features
    distill_loss = distiller.compute_loss()
    total_loss = task_loss + distill_loss
    distiller.step()  # clear features for next iteration
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .hooks import FeatureHookManager
from .losses import MGDLoss, CWDLoss, FeatureMSELoss, DISTILL_LOSSES

logger = logging.getLogger(__name__)


class Distiller(nn.Module):
    """Model-agnostic knowledge distillation orchestrator.

    Manages the teacher model, feature extraction hooks, channel adaptation,
    and distillation loss computation. Works with any architecture that
    provides a ``get_distill_config()`` method.

    Args:
        teacher_model: The teacher's ``nn.Module`` (will be frozen).
        student_model: The student's ``nn.Module`` (hooks are read-only).
        teacher_config: Dict from ``teacher.get_distill_config()`` with keys:
            - tap_points: list of module path strings
            - channels: list of int channel dimensions
            - strides: list of int spatial strides
        student_config: Dict from ``student.get_distill_config()`` (same format).
        loss_type: ``"mgd"`` or ``"cwd"`` (default: ``"mgd"``).
        loss_weight: Global distillation loss weight (alpha). Default: 2e-5 for MGD, 1.0 for CWD.
        mask_ratio: MGD mask ratio (default: 0.65). Ignored for CWD.
        tau: CWD temperature (default: 1.0). Ignored for MGD.
        per_scale_weight: Optional list of per-scale weights. If None, uniform.

    Example::

        distiller = Distiller(
            teacher_model=teacher_nn,
            student_model=student_nn,
            teacher_config={"tap_points": ["neck.elan_down2"], "channels": [512], "strides": [32]},
            student_config={"tap_points": ["neck.elan_down2"], "channels": [128], "strides": [32]},
            loss_type="mgd",
        )
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        teacher_config: Dict,
        student_config: Dict,
        loss_type: str = "mgd",
        loss_weight: Optional[float] = None,
        mask_ratio: float = 0.65,
        tau: float = 1.0,
        per_scale_weight: Optional[List[float]] = None,
        teacher_feature_fn: Optional[Callable[[torch.Tensor], List[torch.Tensor]]] = None,
        normalize: bool = False,
    ):
        super().__init__()

        self.loss_type = loss_type.lower()
        self.loss_weight = loss_weight if loss_weight is not None else self._default_weight()
        self.normalize = normalize

        # A foundation-model teacher (e.g. DINOv2) emits token features that are
        # not a hookable BCHW module output and lives on a different spatial
        # grid/stride than the student. Callers pass ``teacher_feature_fn`` to
        # extract its per-scale feature maps directly; the feature-MSE loss then
        # resizes them to the student. In that mode we skip teacher hooks and the
        # stride-equality check (the loss handles spatial mismatch).
        self.teacher_feature_fn = teacher_feature_fn
        self._teacher_feats: Optional[List[torch.Tensor]] = None

        # Validate configs
        t_strides = teacher_config["strides"]
        s_strides = student_config["strides"]
        if teacher_feature_fn is None and t_strides != s_strides:
            raise ValueError(
                f"Teacher and student must have matching strides. "
                f"Teacher: {t_strides}, Student: {s_strides}"
            )

        self.num_scales = len(t_strides)
        t_channels = teacher_config["channels"]
        s_channels = student_config["channels"]

        # Freeze teacher
        self.teacher = teacher_model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Register hooks. The student is always hooked. The teacher is hooked
        # only when it exposes hookable BCHW features (detector teachers);
        # foundation teachers deliver features via ``teacher_feature_fn``.
        self.t_hooks = (
            FeatureHookManager(self.teacher, teacher_config["tap_points"])
            if teacher_feature_fn is None
            else None
        )
        self.s_hooks = FeatureHookManager(student_model, student_config["tap_points"])

        # Per-scale weights
        if per_scale_weight is not None:
            if len(per_scale_weight) != self.num_scales:
                raise ValueError(
                    f"per_scale_weight has {len(per_scale_weight)} entries, "
                    f"expected {self.num_scales} (one per feature scale)"
                )
            self._scale_weights = per_scale_weight
        else:
            self._scale_weights = [1.0] * self.num_scales

        # Build loss modules (one per feature scale)
        self.loss_modules = nn.ModuleList()
        for i in range(self.num_scales):
            loss_fn = self._build_loss(
                s_channels[i], t_channels[i],
                mask_ratio=mask_ratio, tau=tau,
                scale_weight=self._scale_weights[i],
            )
            self.loss_modules.append(loss_fn)

        # Log configuration
        logger.info("Distiller initialized:")
        logger.info(f"  Loss type: {self.loss_type}")
        logger.info(f"  Global weight (alpha): {self.loss_weight}")
        logger.info(f"  Num scales: {self.num_scales}")
        for i, (sc, tc) in enumerate(zip(s_channels, t_channels)):
            logger.info(
                f"  Scale {i} (stride {t_strides[i]}): "
                f"student={sc}ch -> teacher={tc}ch"
            )

    # =========================================================================
    # Configuration
    # =========================================================================

    def _default_weight(self) -> float:
        """Return sensible default loss weight for the chosen loss type."""
        defaults = {"mgd": 2e-5, "cwd": 1.0, "feat_mse": 1.0}
        if self.loss_type not in defaults:
            raise ValueError(
                f"No default weight for loss type '{self.loss_type}'. "
                f"Available: {list(defaults.keys())}. "
                f"Pass loss_weight explicitly."
            )
        return defaults[self.loss_type]

    def _build_loss(
        self,
        student_ch: int,
        teacher_ch: int,
        mask_ratio: float,
        tau: float,
        scale_weight: float,
    ) -> nn.Module:
        """Construct a loss module for one feature scale."""
        if self.loss_type == "mgd":
            return MGDLoss(
                student_channels=student_ch,
                teacher_channels=teacher_ch,
                mask_ratio=mask_ratio,
                loss_weight=scale_weight,
            )
        elif self.loss_type == "cwd":
            return CWDLoss(
                student_channels=student_ch,
                teacher_channels=teacher_ch,
                tau=tau,
                loss_weight=scale_weight,
            )
        elif self.loss_type == "feat_mse":
            return FeatureMSELoss(
                student_channels=student_ch,
                teacher_channels=teacher_ch,
                loss_weight=scale_weight,
                normalize=self.normalize,
            )
        else:
            raise ValueError(
                f"Unknown loss type: '{self.loss_type}'. "
                f"Available: {list(DISTILL_LOSSES.keys())}"
            )

    # =========================================================================
    # Forward pass
    # =========================================================================

    @torch.no_grad()
    def teacher_forward(self, images: torch.Tensor) -> Any:
        """Run the frozen teacher model with no gradients.

        The forward hooks automatically capture the teacher's features.
        Call this BEFORE the student forward pass.

        Args:
            images: Input batch of shape (N, 3, H, W).

        Returns:
            Teacher model output (usually ignored — we only need the hooks).
        """
        if self.teacher_feature_fn is not None:
            # Foundation teacher: extract features directly and stash them.
            self._teacher_feats = [f.detach() for f in self.teacher_feature_fn(images)]
            return None
        return self.teacher(images)

    def compute_loss(self) -> torch.Tensor:
        """Compute total distillation loss across all feature scales.

        Must be called AFTER both teacher_forward() and the student forward
        pass have been executed (so that hooks have captured features).

        Returns:
            Scalar distillation loss, scaled by ``self.loss_weight``.

        Raises:
            RuntimeError: If features haven't been captured yet.
        """
        if self.teacher_feature_fn is not None:
            if self._teacher_feats is None:
                raise RuntimeError(
                    "Teacher features not extracted. "
                    "Did you call teacher_forward() before compute_loss()?"
                )
            t_feats = self._teacher_feats
        else:
            t_feats = self.t_hooks.get_feature_list()
        s_feats = self.s_hooks.get_feature_list()

        if len(t_feats) != self.num_scales:
            missing = (
                [
                    p
                    for p in self.t_hooks.tap_points
                    if p not in self.t_hooks.get_features()
                ]
                if self.t_hooks is not None
                else []
            )
            raise RuntimeError(
                f"Expected {self.num_scales} teacher features, got {len(t_feats)} "
                f"(missing tap points: {missing}). "
                f"Did you call teacher_forward() before compute_loss()?"
            )
        if len(s_feats) != self.num_scales:
            missing = [
                p for p in self.s_hooks.tap_points if p not in self.s_hooks.get_features()
            ]
            raise RuntimeError(
                f"Expected {self.num_scales} student features, got {len(s_feats)} "
                f"(missing tap points: {missing}). "
                f"Did the student forward pass run before compute_loss()?"
            )

        total = torch.tensor(0.0, device=s_feats[0].device)
        for i, (loss_fn, s_feat, t_feat) in enumerate(
            zip(self.loss_modules, s_feats, t_feats)
        ):
            scale_loss = loss_fn(s_feat, t_feat.detach())
            total = total + scale_loss

        return self.loss_weight * total

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def step(self):
        """Clear captured features. Call at the end of each training step."""
        if self.t_hooks is not None:
            self.t_hooks.clear()
        self._teacher_feats = None
        self.s_hooks.clear()

    def cleanup(self):
        """Remove all hooks and free resources. Call when training ends."""
        if self.t_hooks is not None:
            self.t_hooks.remove()
        self.s_hooks.remove()
        logger.info("Distiller cleaned up")

    def __repr__(self) -> str:
        return (
            f"Distiller(loss_type='{self.loss_type}', "
            f"loss_weight={self.loss_weight}, "
            f"num_scales={self.num_scales})"
        )
