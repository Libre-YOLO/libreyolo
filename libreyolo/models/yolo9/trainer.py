"""
YOLOv9 Trainer for LibreYOLO.

Thin subclass of BaseTrainer with yolo9-specific transforms, scheduler,
and loss extraction.
"""

import torch
from typing import Dict, List, Type

from libreyolo.training.trainer import BaseTrainer
from libreyolo.training.config import TrainConfig, YOLO9Config
from libreyolo.training.freezing import FreezeGroup
from ...training.scheduler import LinearLRScheduler, CosineAnnealingScheduler
from .transforms import YOLO9TrainTransform, YOLO9MosaicMixupDataset


class YOLO9Trainer(BaseTrainer):
    """YOLOv9-specific trainer."""

    artifact_model_families = ("yolo9", "yolo9_e2e")

    # Module names inspected by get_freeze_groups, in freeze order.
    # Subclasses with extra modules (e.g. yolo9_p2) extend these.
    _BACKBONE_FREEZE_MODULES = (
        "conv0",
        "conv1",
        "elan1",
        "down2",
        "elan2",
        "down3",
        "elan3",
        "down4",
        "elan4",
        "spp",
    )
    _NECK_FREEZE_MODULES = (
        "elan_up1",
        "elan_up2",
        "down1",
        "elan_down1",
        "down2",
        "elan_down2",
    )

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return YOLO9Config

    def get_model_family(self) -> str:
        return "yolo9"

    def get_model_tag(self) -> str:
        return f"YOLOv9-{self.config.size}"

    def validate_validation_loss_config(self) -> None:
        if not getattr(self.config, "val_loss", False):
            return

        from .nn import DDetect, LibreYOLO9Model

        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        # ``isinstance`` covers yolo9_p2, which is the same dense head over a
        # fourth stride. YOLO9-E2E subclasses this model too but swaps in a
        # dual-branch head, so the exact head check routes it to its own
        # trainer override.
        standard_model = (
            isinstance(self.model, LibreYOLO9Model)
            and type(self.model.head) is DDetect
        )
        if task != "detect" or not standard_model:
            raise ValueError(
                "val_loss=True currently supports YOLO9 detection only; "
                "non-detect tasks are not supported"
            )

    def build_validation_loss_adapter(self, model: torch.nn.Module):
        from .validation_loss import YOLO9ValidationLoss

        return YOLO9ValidationLoss(
            model,
            max_labels=int(getattr(self.config, "max_labels", 100)),
        )

    def get_freeze_groups(self) -> List[FreezeGroup]:
        model = self.model
        backbone = getattr(model, "backbone", None)
        neck = getattr(model, "neck", None)
        head = getattr(model, "head", None)
        groups: List[FreezeGroup] = []
        if backbone is not None:
            for name in self._BACKBONE_FREEZE_MODULES:
                module = getattr(backbone, name, None)
                if module is not None:
                    groups.append((f"backbone.{name}", module))
        if neck is not None:
            for name in self._NECK_FREEZE_MODULES:
                module = getattr(neck, name, None)
                if module is not None:
                    groups.append((f"neck.{name}", module))
        if head is not None:
            groups.append(("head", head))
        return groups or super().get_freeze_groups()

    def create_transforms(self):
        preproc = YOLO9TrainTransform(
            max_labels=getattr(self.config, "max_labels", 100),
            flip_prob=self.config.flip_prob,
            vertical_flip_prob=getattr(self.config, "flipud", 0.0),
            hsv_prob=self.config.hsv_prob,
            rot90_prob=getattr(self.config, "rot90", 0.0),
        )
        return preproc, YOLO9MosaicMixupDataset

    def create_scheduler(self, iters_per_epoch: int):
        scheduler_name = self.config.scheduler
        if scheduler_name == "linear":
            return LinearLRScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        elif scheduler_name in ("cos", "warmcos"):
            return CosineAnnealingScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        # Transfer every logging scalar with ONE .cpu() call (the rfdetr
        # pattern from PR #761): a per-key ``.item()`` here would be one GPU
        # pipeline drain each per logged step (issue #763).
        values = {name: outputs.get(name, 0) for name in ("box", "cls", "dfl")}
        tensor_names = [n for n, v in values.items() if isinstance(v, torch.Tensor)]
        if tensor_names:
            stacked = torch.stack(
                [values[n].detach().reshape(()).float() for n in tensor_names]
            ).cpu()
            for i, name in enumerate(tensor_names):
                values[name] = stacked[i]
        return {name: float(v) for name, v in values.items()}

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        return self.model(imgs, targets=targets)

    def cuda_graph_train_spec(self):
        """Capture spec: graph the network, keep the DFL/TAL loss eager.

        The split reuses the model's own boundary: a train-mode forward
        without targets returns the concatenated head maps, and
        ``assemble`` replays exactly the loss path ``LibreYOLO9Model.
        forward`` takes with targets (anchors tracking the input size,
        then the head's loss over the raw maps). Restricted to the plain
        detect head: subclasses with derived heads (e2e dual assignment)
        or other tasks compute loss at a different boundary and run eager.
        """
        from libreyolo.training.cuda_graph import (
            CudaGraphTrainSpec,
            GraphableNetwork,
        )
        from .nn import DDetect, LibreYOLO9Model

        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        if task != "detect":
            return None
        if not isinstance(self.model, LibreYOLO9Model):
            return None
        if type(self.model.head) is not DDetect:
            return None

        network = GraphableNetwork(self.model)

        def assemble(flat, imgs, targets, polygons=None):
            loss_fn = self.model.head._get_loss_fn(imgs.device)
            loss_fn.update_anchors([imgs.shape[3], imgs.shape[2]])
            return loss_fn(network.rebuild(flat), targets)

        return CudaGraphTrainSpec(network=network, assemble=assemble)
