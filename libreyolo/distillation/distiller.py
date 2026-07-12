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
    distiller.step()  # clear features for the next microbatch
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .hooks import FeatureHookManager, _resolve_module
from .losses import MGDLoss, CWDLoss, FeatureMSELoss, DISTILL_LOSSES

logger = logging.getLogger(__name__)


def _as_sequence(value: Any, name: str) -> list[Any]:
    """Return a public config sequence as a list, rejecting scalar strings."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return list(value)


def _positive_ints(values: list[Any], name: str) -> list[int]:
    """Validate positive integer channel or stride declarations."""
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name}[{index}] must be a positive integer")
        if value <= 0:
            raise ValueError(f"{name}[{index}] must be a positive integer")
        result.append(int(value))
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    """Validate a scalar loss weight and return its floating-point value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validate_config(
    config: Dict,
    name: str,
    *,
    allow_empty_tap_points: bool,
) -> dict[str, list[Any]]:
    """Validate and normalize one model's feature-distillation config."""
    if not isinstance(config, Mapping):
        raise TypeError(f"{name} must be a mapping")

    required = ("tap_points", "channels", "strides")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{name} is missing required keys: {missing}")

    tap_points = _as_sequence(config["tap_points"], f"{name}['tap_points']")
    channels = _positive_ints(
        _as_sequence(config["channels"], f"{name}['channels']"),
        f"{name}['channels']",
    )
    strides = _positive_ints(
        _as_sequence(config["strides"], f"{name}['strides']"),
        f"{name}['strides']",
    )

    if not strides:
        raise ValueError(f"{name} must declare at least one feature scale")
    if len(channels) != len(strides):
        raise ValueError(
            f"{name} channels and strides must have matching lengths; "
            f"got {len(channels)} and {len(strides)}"
        )
    valid_tap_counts = {len(strides)}
    if allow_empty_tap_points:
        valid_tap_counts.add(0)
    if len(tap_points) not in valid_tap_counts:
        qualifier = "zero or " if allow_empty_tap_points else ""
        raise ValueError(
            f"{name} tap_points must contain {qualifier}{len(strides)} entries; "
            f"got {len(tap_points)}"
        )
    for index, path in enumerate(tap_points):
        if not isinstance(path, str) or not path:
            raise TypeError(f"{name}['tap_points'][{index}] must be a non-empty string")

    return {
        "tap_points": tap_points,
        "channels": channels,
        "strides": strides,
    }


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
        loss_type: ``"mgd"``, ``"cwd"``, or ``"feat_mse"`` (default: ``"mgd"``).
        loss_weight: Finite nonnegative global loss weight. Defaults are 2e-5
            for MGD and 1.0 for CWD/feature MSE.
        mask_ratio: Finite MGD mask ratio in ``[0, 1)`` (default: 0.65).
            Ignored for other loss types.
        tau: Finite positive CWD temperature (default: 1.0). Ignored for other
            loss types.
        per_scale_weight: Optional finite nonnegative weight per feature scale.
            If None, every scale has weight 1.0.
        teacher_feature_fn: Optional direct teacher feature extractor. This
            skips teacher hooks and permits teacher/student stride mismatch.
        normalize: L2-normalize feature-MSE inputs when true.

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

        if not isinstance(teacher_model, nn.Module):
            raise TypeError("teacher_model must be an nn.Module")
        if not isinstance(student_model, nn.Module):
            raise TypeError("student_model must be an nn.Module")
        if not isinstance(loss_type, str):
            raise TypeError("loss_type must be a string")
        if teacher_feature_fn is not None and not callable(teacher_feature_fn):
            raise TypeError("teacher_feature_fn must be callable or None")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool")

        self.loss_type = loss_type.lower()
        if self.loss_type not in DISTILL_LOSSES:
            raise ValueError(
                f"Unknown loss type: '{self.loss_type}'. "
                f"Available: {list(DISTILL_LOSSES.keys())}"
            )

        t_config = _validate_config(
            teacher_config,
            "teacher_config",
            allow_empty_tap_points=teacher_feature_fn is not None,
        )
        s_config = _validate_config(
            student_config,
            "student_config",
            allow_empty_tap_points=False,
        )
        t_strides = t_config["strides"]
        s_strides = s_config["strides"]
        if len(t_strides) != len(s_strides):
            raise ValueError(
                "Teacher and student must declare the same number of feature scales. "
                f"Teacher: {len(t_strides)}, Student: {len(s_strides)}"
            )
        if teacher_feature_fn is None and t_strides != s_strides:
            raise ValueError(
                "Teacher and student must have matching strides. "
                f"Teacher: {t_strides}, Student: {s_strides}"
            )

        if self.loss_type == "mgd":
            if isinstance(mask_ratio, bool) or not isinstance(mask_ratio, Real):
                raise TypeError("mask_ratio must be a real number")
            mask_ratio = float(mask_ratio)
            if not math.isfinite(mask_ratio) or not 0.0 <= mask_ratio < 1.0:
                raise ValueError(
                    "mask_ratio must be finite and satisfy 0 <= mask_ratio < 1"
                )
        elif self.loss_type == "cwd":
            if isinstance(tau, bool) or not isinstance(tau, Real):
                raise TypeError("tau must be a real number")
            tau = float(tau)
            if not math.isfinite(tau) or tau <= 0.0:
                raise ValueError("tau must be finite and greater than zero")

        default_weight = self._default_weight() if loss_weight is None else loss_weight
        self.loss_weight = _finite_nonnegative(default_weight, "loss_weight")
        self.normalize = normalize

        self.num_scales = len(t_strides)
        if per_scale_weight is None:
            self._scale_weights = [1.0] * self.num_scales
        else:
            scale_weights = _as_sequence(per_scale_weight, "per_scale_weight")
            if len(scale_weights) != self.num_scales:
                raise ValueError(
                    f"per_scale_weight has {len(scale_weights)} entries, "
                    f"expected {self.num_scales} (one per feature scale)"
                )
            self._scale_weights = [
                _finite_nonnegative(weight, f"per_scale_weight[{index}]")
                for index, weight in enumerate(scale_weights)
            ]

        # Resolve every configured module before freezing the teacher or
        # registering any hook. A malformed later tap point must not leave
        # earlier hooks installed on either model.
        if teacher_feature_fn is None:
            for path in t_config["tap_points"]:
                module = _resolve_module(teacher_model, path)
                if not isinstance(module, nn.Module):
                    raise TypeError(
                        f"teacher_config tap point '{path}' does not resolve to an nn.Module"
                    )
        for path in s_config["tap_points"]:
            module = _resolve_module(student_model, path)
            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"student_config tap point '{path}' does not resolve to an nn.Module"
                )

        # A foundation-model teacher (e.g. DINOv2) emits token features that are
        # not a hookable BCHW module output and lives on a different spatial
        # grid/stride than the student. Callers pass ``teacher_feature_fn`` to
        # extract its per-scale feature maps directly; the feature-MSE loss then
        # resizes them to the student. In that mode we skip teacher hooks and the
        # stride-equality check (the loss handles spatial mismatch).
        self.teacher_feature_fn = teacher_feature_fn
        self._teacher_feats: Optional[List[torch.Tensor]] = None

        t_channels = t_config["channels"]
        s_channels = s_config["channels"]

        # Build every loss module before hooks are installed. Channel and
        # weight errors therefore cannot leave live forward hooks behind.
        self.loss_modules = nn.ModuleList()
        for i in range(self.num_scales):
            loss_fn = self._build_loss(
                s_channels[i],
                t_channels[i],
                mask_ratio=mask_ratio,
                tau=tau,
                scale_weight=self._scale_weights[i],
            )
            self.loss_modules.append(loss_fn)

        # Freeze teacher
        self.teacher = teacher_model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Register hooks. The student is always hooked. The teacher is hooked
        # only when it exposes hookable BCHW features (detector teachers);
        # foundation teachers deliver features via ``teacher_feature_fn``.
        self.t_hooks = (
            FeatureHookManager(self.teacher, t_config["tap_points"])
            if teacher_feature_fn is None
            else None
        )
        self.s_hooks = FeatureHookManager(student_model, s_config["tap_points"])

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

    def step(self) -> None:
        """Clear features after each microbatch.

        This is feature-lifecycle cleanup, not an optimizer-step transition.
        Call it for every attempted microbatch, including accumulation
        microbatches and iterations where AMP skips the optimizer update.
        """
        if self.t_hooks is not None:
            self.t_hooks.clear()
        self._teacher_feats = None
        self.s_hooks.clear()

    def cleanup(self) -> None:
        """Idempotently clear captured graphs and remove every feature hook."""
        self.step()
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
