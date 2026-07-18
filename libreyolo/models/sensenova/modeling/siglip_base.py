# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Copyright 2024 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
#
# Ported for LibreYOLO from OpenSenseNova/SenseNova-Vision
# (commit 12ccd96e32b32967a11cacb6c5bd5fe3a555fc0c, Apache-2.0), files
# modeling/siglip/modeling_siglip.py and configuration_siglip.py, themselves
# derived from Hugging Face transformers (Apache-2.0). Trimmed to what the
# packed NaViT vision tower instantiates: SiglipAttention keeps only its
# constructor (the NaViT subclass overrides forward), and the text/pooling
# branches of the upstream weight init are dropped because this port only
# ever loads released checkpoints on top of a meta-initialized skeleton.

from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


class SiglipVisionConfigBase(PretrainedConfig):
    """Vision-tower configuration (upstream ``SiglipVisionConfig``)."""

    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act


class SiglipAttention(nn.Module):
    """Projection layout of SigLIP multi-headed attention.

    Only the constructor is used: the NaViT flash/SDPA subclass overrides
    forward with the packed-sequence implementation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)


class SiglipPreTrainedModel(PreTrainedModel):
    """Abstract class wiring config/meta-init for the vendored vision tower."""

    config_class = SiglipVisionConfigBase
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SiglipVisionEmbeddings", "SiglipEncoderLayer"]

    def _init_weights(self, module):
        """Generic init; real checkpoints always overwrite these values."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
