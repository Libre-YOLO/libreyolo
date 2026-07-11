"""Training-free fit configuration for LibrePatchCore."""

from dataclasses import dataclass

from ...training.config import TrainConfig


@dataclass(kw_only=True)
class PatchCoreConfig(TrainConfig):
    size: str = "b"
    num_classes: int = 1
    imgsz: int = 224
    epochs: int = 0
    batch: int = 8
    device: str = "auto"
    workers: int = 4
    project: str = "runs/anomaly/train"
    name: str = "patchcore"
    coreset: float = 10.0
    projection_dim: int = 128
    reweight_neighbors: int = 9
    query_chunk_size: int = 2048


__all__ = ["PatchCoreConfig"]
