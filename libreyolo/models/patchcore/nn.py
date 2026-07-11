"""Frozen feature extractor, coreset sampling, and nearest-neighbor scoring."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCoreNet(nn.Module):
    """WideResNet feature memory used by PatchCore."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        reweight_neighbors: int = 9,
        query_chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = wide_resnet50_2(weights=weights)
        self.backbone.fc = nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

        self.register_buffer("memory_bank", torch.empty(0, 1536), persistent=True)
        self.register_buffer("anomaly_threshold", torch.tensor(float("nan")), persistent=True)
        self.register_buffer("coreset_percent", torch.tensor(10.0), persistent=True)
        self.register_buffer("fitted", torch.tensor(False), persistent=True)
        self.reweight_neighbors = int(reweight_neighbors)
        self.query_chunk_size = int(query_chunk_size)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return locally aggregated patch features as ``(B, H, W, C)``."""
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(x)
        layer3 = self.backbone.layer3(layer2)
        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat((layer2, layer3), dim=1)
        return features.permute(0, 2, 3, 1).contiguous()

    def set_memory_bank(self, bank: torch.Tensor, coreset_percent: float) -> None:
        if bank.ndim != 2 or bank.shape[1] != 1536 or bank.shape[0] == 0:
            raise ValueError(f"Expected a non-empty (N, 1536) memory bank, got {tuple(bank.shape)}.")
        self.memory_bank = bank.detach().float().to(self.memory_bank.device)
        self.coreset_percent.fill_(float(coreset_percent))
        self.fitted.fill_(True)

    def nearest_distances(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.fitted) or self.memory_bank.numel() == 0:
            raise RuntimeError("LibrePatchCore is not fitted. Run model.train(data=...) first.")
        queries = queries.float()
        bank = self.memory_bank.float()
        distances: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        for start in range(0, len(queries), self.query_chunk_size):
            chunk = queries[start : start + self.query_chunk_size]
            values, nearest = torch.cdist(chunk, bank).min(dim=1)
            distances.append(values)
            indices.append(nearest)
        return torch.cat(distances), torch.cat(indices)

    def _image_score(
        self, queries: torch.Tensor, patch_distances: torch.Tensor, nearest: torch.Tensor
    ) -> torch.Tensor:
        max_index = int(patch_distances.argmax())
        raw_score = patch_distances[max_index]
        count = min(self.reweight_neighbors, int(self.memory_bank.shape[0]))
        if count <= 1:
            return raw_score
        support_index = int(nearest[max_index])
        support = self.memory_bank[support_index : support_index + 1].float()
        _, neighbor_indices = torch.cdist(support, self.memory_bank.float()).topk(
            count, largest=False, dim=1
        )
        neighbor_features = self.memory_bank[neighbor_indices[0]].float()
        neighbor_distances = torch.linalg.vector_norm(
            neighbor_features - queries[max_index].float(), dim=1
        )
        weights = torch.softmax(neighbor_distances, dim=0)
        return raw_score * (1.0 - weights[0])

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_grid = self.extract_features(images)
        batch, height, width, channels = feature_grid.shape
        maps: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        for features in feature_grid.reshape(batch, height * width, channels):
            distances, nearest = self.nearest_distances(features)
            maps.append(distances.reshape(height, width))
            scores.append(self._image_score(features, distances, nearest))
        return {"patch_scores": torch.stack(maps), "image_scores": torch.stack(scores)}


def greedy_coreset(
    features: torch.Tensor,
    *,
    percent: float = 10.0,
    projection_dim: int = 128,
    seed: int = 0,
) -> torch.Tensor:
    """Deterministic projected farthest-first coreset selection."""
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("Coreset features must be a non-empty 2D tensor.")
    if not 1.0 <= percent <= 25.0:
        raise ValueError(f"coreset must be in [1, 25] percent, got {percent}.")
    count = max(1, int(math.ceil(len(features) * percent / 100.0)))
    if count >= len(features):
        return features.detach().float()

    source = features.detach().float().cpu()
    dim = min(int(projection_dim), int(source.shape[1]))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(source.shape[1], dim, generator=generator) / math.sqrt(dim)
    projected = source @ projection

    centroid = projected.mean(dim=0, keepdim=True)
    first = int(torch.linalg.vector_norm(projected - centroid, dim=1).argmax())
    selected = torch.empty(count, dtype=torch.long)
    selected[0] = first
    min_distances = torch.linalg.vector_norm(projected - projected[first], dim=1)
    min_distances[first] = -1
    for index in range(1, count):
        chosen = int(min_distances.argmax())
        selected[index] = chosen
        distance = torch.linalg.vector_norm(projected - projected[chosen], dim=1)
        min_distances = torch.minimum(min_distances, distance)
        min_distances[selected[: index + 1]] = -1
    return source[selected]


__all__ = ["PatchCoreNet", "greedy_coreset"]
