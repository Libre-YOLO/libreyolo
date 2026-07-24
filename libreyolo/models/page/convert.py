"""Upstream PaGE checkpoint key normalization.

Shared by the runtime auto-converter and ``weights/convert_page_weights.py``.

The released PaGE safetensors (Octopus1/page-*) were produced under
transformers 5.6.x, where the DINOv3 layer stack nests under an inner
``.model`` (``<branch>.model.model.layer.N``). LibreYOLO's canonical
checkpoint stores the flat 4.56-style naming (``<branch>.model.layer.N``);
``PageBackbone``'s load hook re-nests at load time when the installed
transformers requires it. Conversion is purely syntactic — numerics are
untouched.
"""

from __future__ import annotations

from typing import Dict

import torch

_DECODER_SIGNATURE = "scene_head_interaction_layers.0.cross_attn_scene.attn.q.weight"
_BRANCHES = ("scene_branch_backbone.", "head_branch_backbone.")


def is_upstream_state_dict(state_dict: dict) -> bool:
    """True only for the raw PaGE HF layout (nested DINOv3 layer stack)."""
    if _DECODER_SIGNATURE not in state_dict:
        return False
    return any(
        k.startswith(_BRANCHES) and ".model.model.layer." in k for k in state_dict
    )


def convert_upstream(state_dict: dict) -> Dict[str, torch.Tensor]:
    """Flatten the nested DINOv3 naming into LibreYOLO's canonical form."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(_BRANCHES):
            k = k.replace(".model.model.layer.", ".model.layer.", 1)
        out[k] = v
    return out
