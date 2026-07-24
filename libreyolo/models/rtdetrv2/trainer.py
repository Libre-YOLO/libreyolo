"""RTDETRv2Trainer — falls back to CPU on MPS for backward pass."""

from __future__ import annotations

import logging

import torch

from ..rtdetr.trainer import RTDETRTrainer


class RTDETRv2Trainer(RTDETRTrainer):
    def on_setup(self):
        """Initialize the v2 loss criterion.

        v1's ``RTDETRLoss`` ignores the ``enc_aux_outputs`` key that the v2
        decoder emits, leaving ``enc_score_head``/``enc_bbox_head`` unsupervised
        (zero gradient). ``RTDETRv2Loss`` adds that encoder query-selection
        supervision, matching upstream RT-DETRv2.
        """
        from .loss import RTDETRv2Loss

        self._maybe_apply_lora()
        self.criterion = RTDETRv2Loss(num_classes=self.config.num_classes)
        self.criterion.to(self.device)

    def _setup_device(self) -> torch.device:
        device = super()._setup_device()
        if device.type == "mps":
            logging.getLogger(__name__).warning(
                "RT-DETRv2 training on Apple MPS triggers a torch backward bug "
                "(aten::grid_sampler_2d_backward not implemented for MPS). "
                "Falling back to CPU. Pass device='cuda' or device='cpu' "
                "explicitly to override."
            )
            return torch.device("cpu")
        return device

    def get_model_family(self) -> str:
        return "rtdetrv2"
