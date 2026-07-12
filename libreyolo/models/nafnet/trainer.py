"""LibreNAFNet restoration trainer."""

from __future__ import annotations

from typing import Any, Dict, Type

import torch

from ...training.config import TrainConfig, require_training_choice
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
        # remember that inference-time local pooling must be rebuilt in
        # :meth:`train`. The optimizer/EMA are built after this hook, so they
        # operate on the plain-pooling model.
        from .nn import use_global_pooling

        use_global_pooling(self.model)
        self._pooling_prepared = True

    def _inference_pooling_geometry(self) -> tuple[int, int]:
        """Return the training-crop geometry that defines TLC windows."""
        imgsz = self.config.imgsz
        if isinstance(imgsz, (tuple, list)):
            return int(imgsz[0]), int(imgsz[1])
        side = int(imgsz)
        return side, side

    def _enable_inference_pooling(self, model: torch.nn.Module) -> None:
        """Rebuild and materialize TLC pooling for the configured crop size."""
        from .nn import replace_adaptive_avg_pool2d, use_global_pooling

        # Always start from the plain graph. This also makes repeated
        # validation calls safe after a partially completed conversion.
        use_global_pooling(model)
        train_h, train_w = self._inference_pooling_geometry()
        replace_adaptive_avg_pool2d(
            model,
            base_size=(int(train_h * 1.5), int(train_w * 1.5)),
            train_size=(1, 3, train_h, train_w),
        )
        was_training = model.training
        try:
            model.eval()
            parameter = next(model.parameters())
            with torch.no_grad():
                model(
                    torch.zeros(
                        (1, 3, train_h, train_w),
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
        finally:
            model.train(was_training)

    def _restore_inference_pooling(self) -> None:
        from ...training.distributed import unwrap_model

        if getattr(self, "_pooling_prepared", False):
            self._enable_inference_pooling(unwrap_model(self.model))
        self._pooling_prepared = False

    def train(self) -> Dict[str, Any]:
        # Train through the plain global-pool model, then re-attach the TLC
        # local pooling so the (in-place trained) model infers with TLC again.
        try:
            return super().train()
        finally:
            self._restore_inference_pooling()

    def _run_restore_validation(self, epoch: int):
        """Evaluate checkpoints with the TLC graph used by public inference."""
        from ...training.distributed import unwrap_model
        from .nn import use_global_pooling

        eval_model = self.ema_model.ema if self.ema_model else unwrap_model(self.model)
        was_training = eval_model.training
        try:
            self._enable_inference_pooling(eval_model)
            return super()._run_restore_validation(epoch)
        finally:
            use_global_pooling(eval_model)
            eval_model.train(was_training)

    def create_scheduler(self, iters_per_epoch: int):
        require_training_choice(
            self.config.scheduler,
            field="scheduler",
            supported=("yoloxwarmcos",),
            family=self.get_model_family(),
        )
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

