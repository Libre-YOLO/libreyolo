"""LibreNAFNet restoration trainer."""

from __future__ import annotations

from typing import Any, Dict, Type

import torch

from ...training.config import TrainConfig
from ...training.scheduler import WarmupCosineScheduler
from ...training.trainer import BaseTrainer
from .config import NAFNetConfig


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Robust L1 loss commonly used for image restoration."""

    return torch.sqrt((pred - target).pow(2) + eps * eps).mean()


class NAFNetTrainer(BaseTrainer):
    """Paired RGB restoration trainer for NAFNet."""

    best_metric_key = "metrics/PSNR"

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return NAFNetConfig

    def get_model_family(self) -> str:
        return "nafnet"

    def get_model_tag(self) -> str:
        return f"NAFNet-{self.config.size}"

    def create_transforms(self):
        return None, None

    def on_setup(self) -> None:
        # TLC (NAFNetLocal) local pooling is an inference-time technique with a
        # window fixed by a warm-up forward. Upstream trains the plain
        # global-average-pool NAFNet and only applies TLC local pooling at test
        # time, so forwarding training through the fixed-window local pool is a
        # train/inference mismatch. Switch the model to global pooling for
        # training (weight-preserving — pooling ops carry no parameters) and
        # remember the removed TLC modules so inference-time local pooling is
        # restored in :meth:`train`. The optimizer/EMA are built after this hook,
        # so they operate on the plain-pooling model.
        from .nn import use_global_pooling

        self._tlc_restore = use_global_pooling(self.model)

    def _restore_inference_pooling(self) -> None:
        from .nn import restore_local_pooling

        restore = getattr(self, "_tlc_restore", None)
        if restore:
            restore_local_pooling(restore)
        self._tlc_restore = None

    def train(self) -> Dict[str, Any]:
        # Train through the plain global-pool model, then re-attach the TLC
        # local pooling so the (in-place trained) model infers with TLC again.
        try:
            return super().train()
        finally:
            self._restore_inference_pooling()

    def create_scheduler(self, iters_per_epoch: int):
        return WarmupCosineScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            plateau_epochs=self.config.no_aug_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def on_forward(
        self,
        imgs: torch.Tensor,
        targets: torch.Tensor,
        polygons=None,
    ) -> Dict[str, torch.Tensor]:
        del polygons
        pred = self.model(imgs)
        pred = pred[:, :, : targets.shape[-2], : targets.shape[-1]]
        loss = charbonnier_loss(pred, targets)
        mse = torch.mean((pred.detach() - targets.detach()).pow(2)).clamp_min(1e-12)
        psnr = -10.0 * torch.log10(mse)
        return {"total_loss": loss, "loss_restore": loss.detach(), "psnr": psnr}

    def get_loss_components(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        return {
            "restore": float(outputs.get("loss_restore", outputs["total_loss"])),
            "psnr": float(outputs.get("psnr", 0.0)),
        }

    def _checkpoint_extra_metadata(self) -> Dict[str, Any]:
        return {
            "degradation": getattr(self.config, "degradation", None),
            "dataset": getattr(self.config, "dataset", None),
        }


__all__ = ["NAFNetTrainer", "charbonnier_loss"]

