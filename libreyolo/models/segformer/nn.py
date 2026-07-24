"""SegFormer network definition for LibreYOLO.

PyTorch port of the MiT (Mix Transformer) encoder and all-MLP decode head from
"SegFormer: Simple and Efficient Design for Semantic Segmentation with
Transformers" (Xie et al., NeurIPS 2021).

Derived from (adapted, with parts copied) HuggingFace Transformers'
**Apache-2.0** ``modeling_segformer.py`` (https://github.com/huggingface/
transformers), Copyright 2021 NVIDIA and The HuggingFace Inc. team, with the
config plumbing stripped. This is a derivative work of that Apache-2.0 source,
not clean-room. It is NOT derived from NVIDIA's original NVlabs/SegFormer
repository (NVIDIA Source Code License, non-commercial), which was never read
or used. There is no runtime dependency on the ``transformers`` package. See
``NOTICE`` in this directory for full attribution.

Inputs are expected as RGB float tensors in ``[0, 1]``; ImageNet normalization
is applied inside ``forward`` so training and inference tensors share one
contract, matching the LibreYOLO house convention for semantic families.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255
# Every LayerNorm in the Apache-2.0 reference implementation is built as
# ``nn.LayerNorm(dim)``, i.e. with PyTorch's default eps. The ``layer_norm_eps``
# field shipped in the upstream configs (1e-6) is read by nothing in that
# implementation, so 1e-5 is the value the published checkpoints are actually
# consumed with, and the one that reproduces them exactly. Setting 1e-6 here
# instead costs ~3e-3 of logit and buys nothing.
LAYER_NORM_EPS = 1e-5
DROP_PATH_RATE = 0.1
# Dropout inside the Mix-FFN (reference: hidden_dropout_prob, default 0.0).
HIDDEN_DROPOUT_PROB = 0.0
# Dropout before the decode head classifier only (reference: classifier_dropout_prob).
CLASSIFIER_DROPOUT_PROB = 0.1
# Std of the normal init for Linear/Conv weights (reference: initializer_range).
INITIALIZER_RANGE = 0.02


@dataclass(frozen=True)
class SegformerSizeConfig:
    depths: tuple[int, int, int, int]
    embed_dims: tuple[int, int, int, int]
    decoder_hidden_size: int
    num_heads: tuple[int, int, int, int] = (1, 2, 5, 8)
    sr_ratios: tuple[int, int, int, int] = (8, 4, 2, 1)
    mlp_ratios: tuple[int, int, int, int] = (4, 4, 4, 4)
    patch_sizes: tuple[int, int, int, int] = (7, 3, 3, 3)
    strides: tuple[int, int, int, int] = (4, 2, 2, 2)


SIZE_CONFIGS: dict[str, SegformerSizeConfig] = {
    "b0": SegformerSizeConfig(depths=(2, 2, 2, 2), embed_dims=(32, 64, 160, 256), decoder_hidden_size=256),
    "b1": SegformerSizeConfig(depths=(2, 2, 2, 2), embed_dims=(64, 128, 320, 512), decoder_hidden_size=256),
    "b2": SegformerSizeConfig(depths=(3, 4, 6, 3), embed_dims=(64, 128, 320, 512), decoder_hidden_size=768),
    "b3": SegformerSizeConfig(depths=(3, 4, 18, 3), embed_dims=(64, 128, 320, 512), decoder_hidden_size=768),
    "b4": SegformerSizeConfig(depths=(3, 8, 27, 3), embed_dims=(64, 128, 320, 512), decoder_hidden_size=768),
    "b5": SegformerSizeConfig(depths=(3, 6, 40, 3), embed_dims=(64, 128, 320, 512), decoder_hidden_size=768),
}


class OverlapPatchEmbeddings(nn.Module):
    """Overlapping patch embeddings via strided convolution with symmetric padding."""

    def __init__(self, patch_size: int, stride: int, num_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=patch_size // 2
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS)

    def forward(self, pixel_values: torch.Tensor):
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


class SequenceReduction(nn.Module):
    """Spatially reduces key/value tokens via a strided convolution before attention."""

    def __init__(self, hidden_size: int, sequence_reduction_ratio: int) -> None:
        super().__init__()
        self.sequence_reduction = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, num_channels, height, width)
        hidden_states = self.sequence_reduction(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, num_channels, -1).transpose(1, 2)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class EfficientSelfAttention(nn.Module):
    """Self-attention with spatially-reduced keys/values (PVT-style efficient attention)."""

    def __init__(self, hidden_size: int, num_attention_heads: int, sequence_reduction_ratio: int) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size)
        self.sequence_reduction_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sequence_reduction = SequenceReduction(hidden_size, sequence_reduction_ratio)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        kv_hidden_states = hidden_states
        if self.sequence_reduction_ratio > 1:
            kv_hidden_states = self.sequence_reduction(hidden_states, height, width)

        kv_hidden_shape = (*kv_hidden_states.shape[:-1], -1, self.head_dim)
        key_states = self.k_proj(kv_hidden_states).view(kv_hidden_shape).transpose(1, 2)
        value_states = self.v_proj(kv_hidden_states).view(kv_hidden_shape).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class DepthWiseConv(nn.Module):
    """Depthwise 3x3 convolution used inside Mix-FFN to encode positional information."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states


class MixMLP(nn.Module):
    """Mix-FFN: fc1 -> depthwise conv -> GELU -> fc2, replacing the plain ViT MLP."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DepthWiseConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class DropPath(nn.Module):
    """Stochastic depth per sample. Identity when ``drop_prob`` is 0 or not training."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states
        keep_prob = 1 - self.drop_prob
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
        random_tensor = torch.rand(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return hidden_states.div(keep_prob) * random_tensor


class SegformerLayer(nn.Module):
    """Pre-norm transformer block with efficient self-attention and Mix-FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        drop_path: float,
        sequence_reduction_ratio: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS)
        self.attention = EfficientSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS)
        self.mlp = MixMLP(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio))

    def forward(self, hidden_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = self.attention(hidden_states, height, width)
        hidden_states = self.drop_path(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states, height, width)
        hidden_states = self.drop_path(hidden_states) + residual
        return hidden_states


class SegformerStage(nn.Module):
    """One encoder stage: overlap patch embedding -> transformer blocks -> LayerNorm."""

    def __init__(
        self,
        stage_idx: int,
        in_channels: int,
        cfg: SegformerSizeConfig,
        drop_path_decays: list[float],
    ) -> None:
        super().__init__()
        depth_start = sum(cfg.depths[:stage_idx])
        hidden_size = cfg.embed_dims[stage_idx]
        self.patch_embeddings = OverlapPatchEmbeddings(
            patch_size=cfg.patch_sizes[stage_idx],
            stride=cfg.strides[stage_idx],
            num_channels=in_channels,
            hidden_size=hidden_size,
        )
        self.blocks = nn.ModuleList(
            [
                SegformerLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=cfg.num_heads[stage_idx],
                    drop_path=drop_path_decays[depth_start + layer_idx],
                    sequence_reduction_ratio=cfg.sr_ratios[stage_idx],
                    mlp_ratio=cfg.mlp_ratios[stage_idx],
                )
                for layer_idx in range(cfg.depths[stage_idx])
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, height, width = self.patch_embeddings(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, height, width)
        hidden_states = self.layer_norm(hidden_states)
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        return hidden_states


class SegformerEncoder(nn.Module):
    """MiT (Mix Transformer) encoder: 4 stages producing progressively coarser feature maps."""

    def __init__(self, cfg: SegformerSizeConfig) -> None:
        super().__init__()
        total_depth = sum(cfg.depths)
        drop_path_decays = [DROP_PATH_RATE * i / max(total_depth - 1, 1) for i in range(total_depth)]
        stages = []
        in_channels = 3
        for stage_idx in range(4):
            stages.append(SegformerStage(stage_idx, in_channels, cfg, drop_path_decays))
            in_channels = cfg.embed_dims[stage_idx]
        self.stages = nn.ModuleList(stages)

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        hidden_states = pixel_values
        outputs = []
        for stage in self.stages:
            hidden_states = stage(hidden_states)
            outputs.append(hidden_states)
        return outputs


class SegformerMLP(nn.Module):
    """Projects one encoder stage's feature map to the common decoder hidden size."""

    def __init__(self, input_dim: int, decoder_hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return self.proj(hidden_states)


class SegformerDecodeHead(nn.Module):
    """All-MLP decode head: per-stage linear projection, upsample, fuse, classify."""

    def __init__(self, cfg: SegformerSizeConfig, num_classes: int) -> None:
        super().__init__()
        self.linear_projections = nn.ModuleList(
            SegformerMLP(cfg.embed_dims[i], cfg.decoder_hidden_size) for i in range(4)
        )
        self.linear_fuse = nn.Conv2d(
            in_channels=cfg.decoder_hidden_size * 4,
            out_channels=cfg.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(cfg.decoder_hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(CLASSIFIER_DROPOUT_PROB)
        self.classifier = nn.Conv2d(cfg.decoder_hidden_size, num_classes, kernel_size=1)

    def forward(self, encoder_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]
        target_size = encoder_hidden_states[0].shape[2:]

        projected_states = []
        for hidden_state, linear_proj in zip(encoder_hidden_states, self.linear_projections):
            height, width = hidden_state.shape[2], hidden_state.shape[3]
            hidden_state = linear_proj(hidden_state)
            hidden_state = hidden_state.transpose(1, 2).reshape(batch_size, -1, height, width)
            hidden_state = F.interpolate(hidden_state, size=target_size, mode="bilinear", align_corners=False)
            projected_states.append(hidden_state)

        hidden_states = self.linear_fuse(torch.cat(projected_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.classifier(hidden_states)


class LibreSegformerNet(nn.Module):
    """SegFormer semantic-segmentation network: MiT encoder + all-MLP decode head.

    Callers pass RGB float tensors in ``[0, 1]``; ImageNet normalization is
    applied internally. When ``training`` and ``targets`` are given, the loss
    is computed internally (upsampled to the target resolution) and returned
    as ``{"total_loss": ..., "sem": ...}``, matching the LibreYOLO house
    convention for semantic families (``RFDETRSemanticSegmenter``). In eval
    mode, logits are upsampled to the *input* resolution (not HF's raw stage-1
    resolution) so callers get a ready-to-use dense prediction.
    """

    IGNORE_INDEX = IGNORE_INDEX

    def __init__(self, size: str = "b0", num_classes: int = 150) -> None:
        super().__init__()
        if size not in SIZE_CONFIGS:
            raise ValueError(f"Unknown SegFormer size {size!r}. Must be one of {sorted(SIZE_CONFIGS)}")
        cfg = SIZE_CONFIGS[size]
        self.size = size
        self.num_classes = int(num_classes)
        self.encoder = SegformerEncoder(cfg)
        self.decode_head = SegformerDecodeHead(cfg, self.num_classes)
        self.apply(self._init_weights)
        self.register_buffer(
            "pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    @staticmethod
    @torch.no_grad()
    def _init_weights(module: nn.Module) -> None:
        """Reference init (the Apache-2.0 one: normal(0, 0.02), zeroed biases).
        PyTorch's default (kaiming-uniform, non-zero bias) is far wider than 0.02
        at the narrow stages, and this family can train from scratch, so the init
        is load-bearing rather than cosmetic.
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=INITIALIZER_RANGE)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        h, w = x.shape[-2], x.shape[-1]
        x = (x - self.pixel_mean.to(dtype=x.dtype)) / self.pixel_std.to(dtype=x.dtype)
        encoder_hidden_states = self.encoder(x)
        logits = self.decode_head(encoder_hidden_states)

        if self.training and targets is not None:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            targets = targets.long()
            if bool((targets != self.IGNORE_INDEX).any()):
                loss = F.cross_entropy(logits, targets, ignore_index=self.IGNORE_INDEX)
            else:
                # cross_entropy returns NaN when every pixel is ignored; emit a
                # graph-connected zero so the optimizer step stays sane.
                loss = logits.sum() * 0.0
            return {"total_loss": loss, "sem": loss}

        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


__all__ = [
    "SegformerSizeConfig",
    "SIZE_CONFIGS",
    "LibreSegformerNet",
    "SegformerEncoder",
    "SegformerDecodeHead",
    "IGNORE_INDEX",
]
