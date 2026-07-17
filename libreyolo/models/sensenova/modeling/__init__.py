# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Inference-only port of the Bagel-MoT architecture used by SenseNova-Vision.
# See the per-file headers for upstream provenance and the LibreYOLO
# adaptations (SDPA fallback, training-path removal, clean-source layers).

from .bagel import Bagel, BagelConfig
from .qwen2_navit import Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from .siglip_navit import SiglipVisionConfig, SiglipVisionModel

__all__ = [
    "Bagel",
    "BagelConfig",
    "Qwen2Config",
    "Qwen2ForCausalLM",
    "Qwen2Model",
    "SiglipVisionConfig",
    "SiglipVisionModel",
]
