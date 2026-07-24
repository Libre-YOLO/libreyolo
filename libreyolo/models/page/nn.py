"""LibrePAGE network modules.

Ported from PaGE ("PAGE: Towards Practical Human-level Gaze Target
Estimation", arXiv:2607.04860), upstream repository
https://github.com/OctopusWen/PaGE (MIT, commit 4352bead) and its
self-contained HF modeling file https://huggingface.co/Octopus1/PaGE (MIT).

Architecture: two DINOv3 ViT towers (scene image 512x512, head crop
256x256) project to a shared 256-dim token space; register + in/out
tokens are prepended; one self-attention block per stream and five
bidirectional scene<->head interaction layers (axial-RoPE cross
attention, where head-crop patch coordinates are mapped into the scene
grid through the head rect) feed a deconv heatmap head (64x64 gaze
target probability grid) and an MLP in/out-of-frame head.

Scope: the released checkpoints all use ``pos_encoding="rope"``,
``mlp_layer="geglu"`` and ``inout=True``; this port implements exactly
that path and rejects other configurations at build time.

The DINOv3 towers use transformers' built-in ``DINOv3ViTModel``
(Apache-2.0), constructed from config only — tower weights ship inside
the LibrePAGE checkpoint. transformers changed the DINOv3 parameter
naming between 4.56.x (``model.layer.N``) and 5.x
(``model.model.layer.N``); a load-time remap hook normalizes whichever
convention the checkpoint carries into the one the installed
transformers expects.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, LayerNorm, LayerScale, use_fused_attn
from timm.layers.mlp import Mlp, SwiGLU

# Per-size DINOv3 tower hyper-parameters, from the upstream HF config.json
# of each released checkpoint (Octopus1/page-vits / -vitsplus / -vitb /
# -vithplus). The decoder config is identical across sizes.
PAGE_CONFIGS: Dict[str, Dict[str, object]] = {
    "s": {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "use_gated_mlp": False,
    },
    "sp": {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "use_gated_mlp": True,
    },
    "b": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "use_gated_mlp": False,
    },
    "hp": {
        "hidden_size": 1280,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "intermediate_size": 5120,
        "use_gated_mlp": True,
    },
}

SCENE_SIZE = (512, 512)
HEAD_SIZE = (256, 256)
HEATMAP_SIZE = (64, 64)
PATCH = 16
DIM = 256
NUM_HEADS = 8
MLP_RATIO = 4.0
N_REG_TOKENS = 4
N_SCENE_SELF_LAYERS = 1
N_HEAD_SELF_LAYERS = 1
N_INTERACTION_LAYERS = 5
ROPE_BASE = 100.0
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


# =========================================================================
# Axial 2D RoPE self-attention
# =========================================================================
class Axial2dRotaryEmbedding(nn.Module):
    """Axial 2D RoPE over ViT patch tokens; front tokens stay unrotated."""

    def __init__(self, dim: int, base: float = 100.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 4 != 0:
            raise ValueError(f"`dim` must be a positive multiple of 4, got {dim}.")
        self.dim = dim
        self.axis_dim = dim // 2
        self.base = base
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim)
        )

    def _axis_cos_sin(self, coords, *, device, dtype):
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = coords.to(device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
        return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)

    @staticmethod
    def _rotate_axis(x, cos, sin):
        x = x.reshape(*x.shape[:-1], -1, 2)
        x_even, x_odd = x.unbind(dim=-1)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        x_rot = torch.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
        )
        return x_rot.flatten(-2)

    def forward(self, q, k, grid_size, num_front_tokens=0):
        _, _, n, head_dim = q.shape
        gh, gw = grid_size
        expected = num_front_tokens + gh * gw
        if n != expected:
            raise ValueError(
                f"Token count mismatch: got N={n}, expected "
                f"{num_front_tokens} front + {gh}*{gw} = {expected}."
            )
        if self.dim > head_dim:
            raise ValueError(f"RoPE dim {self.dim} exceeds head_dim {head_dim}.")
        q_front, q_patch = q[:, :, :num_front_tokens], q[:, :, num_front_tokens:]
        k_front, k_patch = k[:, :, :num_front_tokens], k[:, :, num_front_tokens:]
        yy, xx = torch.meshgrid(
            torch.arange(gh, device=q.device),
            torch.arange(gw, device=q.device),
            indexing="ij",
        )
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)
        cos_y, sin_y = self._axis_cos_sin(yy, device=q.device, dtype=q.dtype)
        cos_x, sin_x = self._axis_cos_sin(xx, device=q.device, dtype=q.dtype)

        def apply_rope(t):
            t_rope, t_pass = t[..., : self.dim], t[..., self.dim :]
            t_y, t_x = t_rope.split(self.axis_dim, dim=-1)
            t_y = self._rotate_axis(t_y, cos_y, sin_y)
            t_x = self._rotate_axis(t_x, cos_x, sin_x)
            return torch.cat((t_y, t_x, t_pass), dim=-1)

        q_patch = apply_rope(q_patch)
        k_patch = apply_rope(k_patch)
        q = torch.cat((q_front, q_patch), dim=2)
        k = torch.cat((k_front, k_patch), dim=2)
        return q, k


class AxialRoPEAttention(nn.Module):
    fused_attn: bool

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_front_tokens: int = 0,
        rope_base: float = 100.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dim = num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()
        self.num_front_tokens = num_front_tokens
        self.qkv = nn.Linear(dim, self.attn_dim * 3, bias=False)
        self.rope = Axial2dRotaryEmbedding(self.head_dim, base=rope_base)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attn_dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def _infer_grid_size(self, num_patch_tokens: int) -> Tuple[int, int]:
        side = math.isqrt(num_patch_tokens)
        if side * side != num_patch_tokens:
            raise ValueError("Cannot infer a non-square patch grid from the tokens.")
        return side, side

    def forward(self, x):
        b, n, _ = x.shape
        num_patch_tokens = n - self.num_front_tokens
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        grid_size = self._infer_grid_size(num_patch_tokens)
        q, k = self.rope(q, k, grid_size=grid_size, num_front_tokens=self.num_front_tokens)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(b, n, self.attn_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AxialRoPEBlock(nn.Module):
    """Pre-norm ViT block with axial-RoPE self-attention (timm Block layout)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        mlp_layer=SwiGLU,
        act_layer=nn.GELU,
        drop_path: float = 0.0,
        num_front_tokens: int = 0,
        rope_base: float = 100.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = AxialRoPEAttention(
            dim=dim,
            num_heads=num_heads,
            num_front_tokens=num_front_tokens,
            rope_base=rope_base,
        )
        self.ls1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = LayerNorm(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )
        self.ls2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# =========================================================================
# Axial 2D RoPE cross-attention (head rect maps head patches into the
# scene grid so both streams share one coordinate frame)
# =========================================================================
def _rect_grid_coords(grid_size, rect):
    """Coordinates of ``grid_size`` cell centers spread over ``rect``.

    ``rect`` is [B, 4] (ymin, xmin, ymax, xmax) in scene-grid units; the
    upstream ``align_corners=False`` sampling is used.
    """
    b = rect.shape[0]
    gh, gw = grid_size
    device, dtype = rect.device, rect.dtype
    y0, x0, y1, x1 = rect.unbind(dim=-1)
    iy = torch.arange(gh, device=device, dtype=dtype) + 0.5
    ix = torch.arange(gw, device=device, dtype=dtype) + 0.5
    ys = y0[:, None] + iy[None, :] * ((y1 - y0) / gh)[:, None] - 0.5
    xs = x0[:, None] + ix[None, :] * ((x1 - x0) / gw)[:, None] - 0.5
    yy = ys[:, :, None].expand(b, gh, gw)
    xx = xs[:, None, :].expand(b, gh, gw)
    return torch.stack((yy.reshape(b, -1), xx.reshape(b, -1)), dim=-1)


def _native_grid_coords(grid_size, *, batch_size, device, dtype):
    gh, gw = grid_size
    yy, xx = torch.meshgrid(
        torch.arange(gh, device=device, dtype=dtype),
        torch.arange(gw, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=-1)
    return coords[None, :, :].expand(batch_size, -1, -1)


class Axial2dCrossRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 100.0):
        super().__init__()
        if dim <= 0 or dim % 4 != 0:
            raise ValueError(f"dim must be a positive multiple of 4, got {dim}.")
        self.dim = dim
        self.axis_dim = dim // 2
        self.base = base
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim)
        )

    def _axis_cos_sin(self, coords, *, out_dtype):
        coords = coords.to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=coords.device, dtype=torch.float32)
        freqs = coords[..., None] * inv_freq[None, None, :]
        return freqs.cos().to(dtype=out_dtype), freqs.sin().to(dtype=out_dtype)

    @staticmethod
    def _rotate_axis(x, cos, sin):
        x = x.reshape(*x.shape[:-1], -1, 2)
        x_even, x_odd = x.unbind(dim=-1)
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        x_rot = torch.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
        )
        return x_rot.flatten(-2)

    def rotate_one(self, x, coords_yx, *, num_front_tokens):
        x_front = x[:, :, :num_front_tokens, :]
        x_patch = x[:, :, num_front_tokens:, :]
        y = coords_yx[..., 0]
        x_coord = coords_yx[..., 1]
        cos_y, sin_y = self._axis_cos_sin(y, out_dtype=x.dtype)
        cos_x, sin_x = self._axis_cos_sin(x_coord, out_dtype=x.dtype)
        x_rope = x_patch[..., : self.dim]
        x_pass = x_patch[..., self.dim :]
        x_y, x_x = x_rope.split(self.axis_dim, dim=-1)
        x_y = self._rotate_axis(x_y, cos_y, sin_y)
        x_x = self._rotate_axis(x_x, cos_x, sin_x)
        x_patch = torch.cat((x_y, x_x, x_pass), dim=-1)
        if num_front_tokens == 0:
            return x_patch
        return torch.cat((x_front, x_patch), dim=2)

    def forward(self, q, k, *, q_coords_yx, kv_coords_yx, q_num_front_tokens, kv_num_front_tokens):
        q = self.rotate_one(q, q_coords_yx, num_front_tokens=q_num_front_tokens)
        k = self.rotate_one(k, kv_coords_yx, num_front_tokens=kv_num_front_tokens)
        return q, k


class AxialRoPECrossAttention(nn.Module):
    fused_attn: bool

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        q_num_front_tokens: int = 0,
        kv_num_front_tokens: int = 0,
        rope_base: float = 100.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dim = num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()
        self.q_num_front_tokens = q_num_front_tokens
        self.kv_num_front_tokens = kv_num_front_tokens
        self.q = nn.Linear(dim, self.attn_dim, bias=False)
        self.k = nn.Linear(dim, self.attn_dim, bias=False)
        self.v = nn.Linear(dim, self.attn_dim, bias=False)
        self.rope = Axial2dCrossRotaryEmbedding(dim=self.head_dim, base=rope_base)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attn_dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _square_grid(num_patch_tokens: int) -> Tuple[int, int]:
        side = math.isqrt(num_patch_tokens)
        if side * side != num_patch_tokens:
            raise ValueError("Cannot infer square grid for cross-attention stream.")
        return side, side

    def forward(self, x_q, x_kv, *, q_rect=None, kv_rect=None):
        b, nq, _ = x_q.shape
        _, nk, _ = x_kv.shape
        q_patches = nq - self.q_num_front_tokens
        kv_patches = nk - self.kv_num_front_tokens
        coord_dtype = torch.float32
        if q_rect is not None:
            q_coords = _rect_grid_coords(self._square_grid(q_patches), q_rect.to(coord_dtype))
        else:
            q_coords = _native_grid_coords(
                self._square_grid(q_patches), batch_size=b, device=x_q.device, dtype=coord_dtype
            )
        if kv_rect is not None:
            kv_coords = _rect_grid_coords(self._square_grid(kv_patches), kv_rect.to(coord_dtype))
        else:
            kv_coords = _native_grid_coords(
                self._square_grid(kv_patches), batch_size=b, device=x_kv.device, dtype=coord_dtype
            )
        q = self.q(x_q).reshape(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x_kv).reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x_kv).reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(
            q,
            k,
            q_coords_yx=q_coords,
            kv_coords_yx=kv_coords,
            q_num_front_tokens=self.q_num_front_tokens,
            kv_num_front_tokens=self.kv_num_front_tokens,
        )
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(b, nq, self.attn_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AxialRoPECrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        q_num_front_tokens: int = 0,
        kv_num_front_tokens: int = 0,
        rope_base: float = 100.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm_q = LayerNorm(dim)
        self.norm_kv = LayerNorm(dim)
        self.attn = AxialRoPECrossAttention(
            dim=dim,
            num_heads=num_heads,
            q_num_front_tokens=q_num_front_tokens,
            kv_num_front_tokens=kv_num_front_tokens,
            rope_base=rope_base,
        )
        self.ls1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_q, x_kv, *, q_rect=None, kv_rect=None):
        x_q = x_q + self.drop_path1(
            self.ls1(self.attn(self.norm_q(x_q), self.norm_kv(x_kv), q_rect=q_rect, kv_rect=kv_rect))
        )
        return x_q


class SceneHeadInteraction(nn.Module):
    """Symmetric scene<->head interaction: cross-attn + ViT block per stream."""

    def __init__(self, dim, num_heads, mlp_ratio, mlp_layer, act_layer, num_front_tokens):
        super().__init__()
        self.cross_attn_scene = AxialRoPECrossAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            q_num_front_tokens=num_front_tokens,
            kv_num_front_tokens=num_front_tokens,
        )
        self.vit_block_scene = AxialRoPEBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            mlp_layer=mlp_layer,
            act_layer=act_layer,
            num_front_tokens=num_front_tokens,
        )
        self.cross_attn_head = AxialRoPECrossAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            q_num_front_tokens=num_front_tokens,
            kv_num_front_tokens=num_front_tokens,
        )
        self.vit_block_head = AxialRoPEBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            mlp_layer=mlp_layer,
            act_layer=act_layer,
            num_front_tokens=num_front_tokens,
        )

    def forward(self, scene_tokens, head_tokens, head_rects):
        out_scene = self.cross_attn_scene(scene_tokens, head_tokens, kv_rect=head_rects)
        out_head = self.cross_attn_head(head_tokens, scene_tokens, q_rect=head_rects)
        out_scene = self.vit_block_scene(out_scene)
        out_head = self.vit_block_head(out_head)
        return out_scene, out_head


# =========================================================================
# DINOv3 tower (transformers built-in implementation, config-only build)
# =========================================================================
class PageBackbone(nn.Module):
    """DINOv3 ViT tower emitting a [B, C, H/16, W/16] patch feature map."""

    def __init__(self, size: str, in_size: Tuple[int, int]):
        super().__init__()
        try:
            from transformers.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel
        except ImportError as e:  # pragma: no cover - guarded by lazy registration
            raise ModuleNotFoundError(
                "LibrePAGE requires transformers>=4.56 with built-in DINOv3 "
                "support. Install with: pip install libreyolo[page]"
            ) from e

        cfg = PAGE_CONFIGS[size]
        dinov3_config = DINOv3ViTConfig(
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            intermediate_size=cfg["intermediate_size"],
            num_register_tokens=N_REG_TOKENS,
            patch_size=PATCH,
            use_gated_mlp=cfg["use_gated_mlp"],
            layerscale_value=1.0,
            drop_path_rate=0.0,
            layer_norm_eps=1e-5,
            image_size=SCENE_SIZE[0],
        )
        self.in_size = in_size
        self.model = DINOv3ViTModel(dinov3_config)
        self.patch_size = PATCH
        self.embed_dim = int(cfg["hidden_size"])
        # CLS(1) + register tokens
        self._num_front = 1 + N_REG_TOKENS
        self._register_load_state_dict_pre_hook(self._remap_dinov3_keys)

    # The transformers DINOv3 layer stack is flattened onto the model in
    # 4.56.x ("model.layer.N") but nested under an inner ``.model`` in 5.x
    # ("model.model.layer.N"). Checkpoints may carry either convention.
    @staticmethod
    def _dinov3_has_nested_layer(dinov3_module: nn.Module) -> bool:
        inner = getattr(dinov3_module, "model", None)
        if not isinstance(inner, nn.Module):
            return False
        return any(k.startswith("layer.") for k in inner.state_dict().keys())

    def _remap_dinov3_keys(self, state_dict, prefix, *args, **kwargs):
        nested = self._dinov3_has_nested_layer(self.model)
        model_pref = prefix + "model."
        new = {}
        for k in list(state_dict.keys()):
            if not k.startswith(model_pref):
                continue
            rest = k[len(model_pref):]
            if rest.startswith("model.layer."):
                core = rest[len("model."):]
            else:
                core = rest
            if nested and core.startswith("layer."):
                target = model_pref + "model." + core
            else:
                target = model_pref + core
            if target != k:
                new[target] = state_dict.pop(k)
        state_dict.update(new)

    def _get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x, return_dict=True)
        tokens = getattr(out, "last_hidden_state", None)
        if tokens is None:
            tokens = out[0]
        return tokens[:, self._num_front :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        patch_tokens = self._get_patch_tokens(x)
        if patch_tokens.shape[1] != out_h * out_w:
            raise RuntimeError(
                f"[PageBackbone] token count mismatch: {patch_tokens.shape[1]} "
                f"vs {out_h * out_w} (input {(h, w)})."
            )
        return patch_tokens.view(b, out_h, out_w, -1).permute(0, 3, 1, 2).contiguous()

    def get_dimension(self) -> int:
        return self.embed_dim

    def get_out_size(self, in_size: Tuple[int, int]) -> Tuple[int, int]:
        h, w = in_size
        return (h // self.patch_size, w // self.patch_size)


# =========================================================================
# LibrePAGE model
# =========================================================================
class LibrePAGEModel(nn.Module):
    """PaGE gaze-target estimator (rope + geglu + inout configuration).

    Forward contract (single scene, Np people):
        scene:      [1, 3, 512, 512] normalized scene image
        heads:      [Np, 3, 256, 256] normalized head crops
        head_rects: [Np, 4] (ymin, xmin, ymax, xmax) in scene-grid units
                    (normalized bbox * 32, rounded, as upstream)
    Returns:
        heatmap_logits: [Np, 64, 64]
        inout_logits:   [Np]
    """

    def __init__(self, size: str):
        super().__init__()
        if size not in PAGE_CONFIGS:
            raise ValueError(f"Unknown LibrePAGE size {size!r}; expected one of {list(PAGE_CONFIGS)}.")
        self.size = size
        self.scene_branch_backbone = PageBackbone(size, SCENE_SIZE)
        self.head_branch_backbone = PageBackbone(size, HEAD_SIZE)

        self.dim = DIM
        self.scene_featmap_h, self.scene_featmap_w = self.scene_branch_backbone.get_out_size(SCENE_SIZE)
        self.head_featmap_h, self.head_featmap_w = self.head_branch_backbone.get_out_size(HEAD_SIZE)
        self.n_reg_tokens = N_REG_TOKENS
        # +1: the in/out token is prepended in front of the register tokens.
        self.n_front_tokens = N_REG_TOKENS + 1
        self.heatmap_out_size = HEATMAP_SIZE

        self.scene_proj = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(self.scene_branch_backbone.get_dimension(), self.dim, 1),
        )
        self.head_proj = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(self.head_branch_backbone.get_dimension(), self.dim, 1),
        )

        self.scene_inout_token = nn.Parameter(torch.zeros((1, 1, self.dim)))
        self.head_inout_token = nn.Parameter(torch.zeros((1, 1, self.dim)))
        self.scene_register_tokens = nn.Parameter(torch.zeros((1, self.n_reg_tokens, self.dim)))
        self.head_register_tokens = nn.Parameter(torch.zeros((1, self.n_reg_tokens, self.dim)))

        # geglu: timm's SwiGLU with GELU activation is equivalent to GEGLU.
        mlp_layer = SwiGLU
        act_layer = nn.GELU

        self.scene_self_attn_layers = nn.Sequential(
            *[
                AxialRoPEBlock(
                    dim=self.dim,
                    num_heads=NUM_HEADS,
                    mlp_ratio=MLP_RATIO,
                    mlp_layer=mlp_layer,
                    act_layer=act_layer,
                    num_front_tokens=self.n_front_tokens,
                )
                for _ in range(N_SCENE_SELF_LAYERS)
            ]
        )
        self.head_self_attn_layers = nn.Sequential(
            *[
                AxialRoPEBlock(
                    dim=self.dim,
                    num_heads=NUM_HEADS,
                    mlp_ratio=MLP_RATIO,
                    mlp_layer=mlp_layer,
                    act_layer=act_layer,
                    num_front_tokens=self.n_front_tokens,
                )
                for _ in range(N_HEAD_SELF_LAYERS)
            ]
        )
        self.scene_head_interaction_layers = nn.ModuleList(
            [
                SceneHeadInteraction(
                    dim=self.dim,
                    num_heads=NUM_HEADS,
                    mlp_ratio=MLP_RATIO,
                    mlp_layer=mlp_layer,
                    act_layer=act_layer,
                    num_front_tokens=self.n_front_tokens,
                )
                for _ in range(N_INTERACTION_LAYERS)
            ]
        )

        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(self.dim, self.dim, kernel_size=2, stride=2),
            nn.Conv2d(self.dim, 1, kernel_size=1, bias=False),
        )
        self.inout_head = nn.Sequential(
            nn.Linear(self.dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        scene: torch.Tensor,
        heads: torch.Tensor,
        head_rects: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_people = heads.shape[0]

        scene_featmap = self.scene_branch_backbone(scene)
        scene_featmap = self.scene_proj(scene_featmap)
        scene_featmap = scene_featmap.expand(num_people, -1, -1, -1)

        head_featmap = self.head_branch_backbone(heads)
        head_featmap = self.head_proj(head_featmap)

        scene_tokens = scene_featmap.flatten(start_dim=2).permute(0, 2, 1)
        head_tokens = head_featmap.flatten(start_dim=2).permute(0, 2, 1)

        scene_tokens = torch.cat(
            [self.scene_register_tokens.expand(num_people, -1, -1), scene_tokens], dim=1
        )
        head_tokens = torch.cat(
            [self.head_register_tokens.expand(num_people, -1, -1), head_tokens], dim=1
        )
        scene_tokens = torch.cat(
            [self.scene_inout_token.expand(num_people, -1, -1), scene_tokens], dim=1
        )
        head_tokens = torch.cat(
            [self.head_inout_token.expand(num_people, -1, -1), head_tokens], dim=1
        )

        scene_tokens = self.scene_self_attn_layers(scene_tokens)
        head_tokens = self.head_self_attn_layers(head_tokens)
        for layer in self.scene_head_interaction_layers:
            scene_tokens, head_tokens = layer(scene_tokens, head_tokens, head_rects)

        scene_patch_tokens = scene_tokens[:, self.n_front_tokens :, :]
        scene_inout_token = scene_tokens[:, 0, :]
        head_inout_token = head_tokens[:, 0, :]

        inout_logits = self.inout_head(
            torch.cat((scene_inout_token, head_inout_token), dim=1)
        ).squeeze(dim=-1)

        scene_featmap = scene_patch_tokens.reshape(
            num_people, self.scene_featmap_h, self.scene_featmap_w, self.dim
        ).permute(0, 3, 1, 2)
        # The deconv head emits exactly heatmap_out_size (32x32 grid, 2x
        # upsample -> 64x64); upstream's trailing resize is an identity no-op.
        heatmap_logits = self.heatmap_head(scene_featmap).squeeze(dim=1)
        return heatmap_logits, inout_logits


def detect_size_from_state_dict(state_dict: dict) -> Optional[str]:
    """Infer the LibrePAGE size code from tower weight shapes."""
    key = None
    for candidate in state_dict:
        if candidate.endswith("scene_branch_backbone.model.embeddings.patch_embeddings.weight"):
            key = candidate
            break
    if key is None:
        return None
    hidden = int(state_dict[key].shape[0])
    gated = any("gate_proj" in k for k in state_dict)
    if hidden == 384:
        return "sp" if gated else "s"
    if hidden == 768:
        return "b"
    if hidden == 1280:
        return "hp"
    return None
