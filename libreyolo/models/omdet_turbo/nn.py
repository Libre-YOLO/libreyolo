"""Native OMDet-Turbo open-vocabulary detector.

Clean-room port of the Apache-2.0 ``transformers`` OMDet-Turbo reference,
mirroring its module tree so the HF ``state_dict`` loads and the forward is
bit-identical. Composes native components:

- vision backbone: native Swin-T (``libreyolo.models.swin``) + per-level LayerNorm
- language backbone: native CLIP text tower + text projection
- hybrid encoder: RT-DETR AIFI + CSP-Rep CCFM (native)
- decoder: multi-scale deformable transformer + Efficient Fusion Head

Modules take the same config object as transformers (a plain scalar namespace)
to maximize fidelity. Eager attention throughout for deterministic parity.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..swin.nn import SwinBackbone, SwinDims

ACT = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}  # conv/csp activations; text uses _act()


def _act(name):
    if name == "quick_gelu":
        return lambda x: x * torch.sigmoid(1.702 * x)
    return {"silu": F.silu, "relu": F.relu, "gelu": F.gelu}[name]


# --------------------------------------------------------------------------- #
# CLIP text tower (language backbone)
# --------------------------------------------------------------------------- #
class _TextAttention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.num_heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, x, attn_mask):
        b, n, _ = x.shape
        shp = (b, n, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shp).transpose(1, 2)
        k = self.k_proj(x).view(*shp).transpose(1, 2)
        v = self.v_proj(x).view(*shp).transpose(1, 2)
        attn = (q @ k.transpose(2, 3)) * self.scale + attn_mask
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.out_proj(out)


class _TextLayer(nn.Module):
    def __init__(self, hidden, heads, interm, act, eps):
        super().__init__()
        self.self_attn = _TextAttention(hidden, heads)
        self.layer_norm1 = nn.LayerNorm(hidden, eps=eps)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(hidden, interm)
        self.mlp.fc2 = nn.Linear(interm, hidden)
        self.layer_norm2 = nn.LayerNorm(hidden, eps=eps)
        self._act = _act(act)

    def forward(self, x, attn_mask):
        x = x + self.self_attn(self.layer_norm1(x), attn_mask)
        h = self.layer_norm2(x)
        h = self.mlp.fc2(self._act(self.mlp.fc1(h)))
        return x + h


class _TextTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, eps = cfg.hidden_size, cfg.layer_norm_eps
        self.embeddings = nn.Module()
        self.embeddings.token_embedding = nn.Embedding(cfg.vocab_size, h)
        self.embeddings.position_embedding = nn.Embedding(cfg.max_position_embeddings, h)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [_TextLayer(h, cfg.num_attention_heads, cfg.intermediate_size, cfg.hidden_act, eps)
             for _ in range(cfg.num_hidden_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(h, eps=eps)

    def forward(self, input_ids):
        n = input_ids.shape[-1]
        pos = torch.arange(n, device=input_ids.device)
        x = self.embeddings.token_embedding(input_ids) + self.embeddings.position_embedding(pos)
        b = x.shape[0]
        minv = torch.finfo(x.dtype).min
        causal = torch.full((n, n), minv, dtype=x.dtype, device=x.device).triu(1)[None, None]
        for layer in self.encoder.layers:
            x = layer(x, causal)
        return self.final_layer_norm(x)


class OmDetLanguageBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = _TextTransformer(config.text_config)
        self.text_projection = nn.Parameter(
            torch.zeros(config.text_projection_in_dim, config.text_projection_out_dim)
        )

    def forward(self, input_ids, mask=None, encode_type="task"):
        last_hidden = self.model(input_ids)
        if encode_type == "task":
            max_len = (mask != 0).sum(1).max().item()
            return last_hidden[:, :max_len, :].transpose(0, 1), mask[:, :max_len]
        # class encoding: EOS token features -> projection
        pooled = last_hidden[torch.arange(last_hidden.shape[0]), input_ids.argmax(dim=-1)]
        return pooled @ self.text_projection


# --------------------------------------------------------------------------- #
# Vision backbone wrapper
# --------------------------------------------------------------------------- #
class OmDetVisionBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.apply_ln = config.apply_layernorm_after_vision_backbone
        self.vision_backbone = nn.Module()
        self.vision_backbone._backbone = SwinBackbone(
            SwinDims(embed_dim=config.backbone_embed_dim, depths=tuple(config.backbone_depths),
                     num_heads=tuple(config.backbone_num_heads), window_size=config.backbone_window_size,
                     out_indices=tuple(config.backbone_out_indices))
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(c, eps=config.layer_norm_eps) for c in config.encoder_in_channels]
        )

    def forward(self, pixel_values):
        feats = self.vision_backbone._backbone(pixel_values)  # list of NHWC
        if self.apply_ln:
            feats = [ln(f).permute(0, 3, 1, 2).contiguous() for ln, f in zip(self.layer_norms, feats)]
        return feats


# --------------------------------------------------------------------------- #
# Deformable attention
# --------------------------------------------------------------------------- #
def _msda(value, shapes_list, sampling_locations, attention_weights):
    b, _, nh, hd = value.shape
    _, nq, _, nl, npnt, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in shapes_list], dim=1)
    grids = 2 * sampling_locations - 1
    out_list = []
    for lid, (h, w) in enumerate(shapes_list):
        v = value_list[lid].flatten(2).transpose(1, 2).reshape(b * nh, hd, h, w)
        g = grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        out_list.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    aw = attention_weights.transpose(1, 2).reshape(b * nh, 1, nq, nl * npnt)
    out = (torch.stack(out_list, dim=-2).flatten(-2) * aw).sum(-1).view(b, nh * hd, nq)
    return out.transpose(1, 2).contiguous()


class OmDetMSDeformAttn(nn.Module):
    def __init__(self, config, num_heads, n_points):
        super().__init__()
        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, hidden_states, encoder_hidden_states, reference_points, spatial_shapes, spatial_shapes_list):
        b, nq, _ = hidden_states.shape
        _, seq, _ = encoder_hidden_states.shape
        value = self.value_proj(encoder_hidden_states).view(b, seq, self.n_heads, self.d_model // self.n_heads)
        off = self.sampling_offsets(hidden_states).view(b, nq, self.n_heads, self.n_levels, self.n_points, 2)
        aw = self.attention_weights(hidden_states).view(b, nq, self.n_heads, self.n_levels * self.n_points)
        aw = F.softmax(aw, -1).view(b, nq, self.n_heads, self.n_levels, self.n_points)
        nc = reference_points.shape[-1]
        if nc == 2:
            normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            loc = reference_points[:, :, None, :, None, :] + off / normalizer[None, None, None, :, None, :]
        else:
            loc = reference_points[:, :, None, :, None, :2] + off / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        out = _msda(value, spatial_shapes_list, loc, aw)
        return self.output_proj(out)


# --------------------------------------------------------------------------- #
# RT-DETR-style conv blocks + hybrid encoder
# --------------------------------------------------------------------------- #
class ConvNorm(nn.Module):
    def __init__(self, config, cin, cout, k, s, padding=None, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, padding=(k - 1) // 2 if padding is None else padding, bias=False)
        self.norm = nn.BatchNorm2d(cout, config.batch_norm_eps)
        self.activation = nn.Identity() if activation is None else ACT[activation]()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class RepVgg(nn.Module):
    def __init__(self, config):
        super().__init__()
        hc = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = ConvNorm(config, hc, hc, 3, 1, padding=1)
        self.conv2 = ConvNorm(config, hc, hc, 1, 1, padding=0)
        self.activation = ACT[config.csp_activation]()

    def forward(self, x):
        return self.activation(self.conv1(x) + self.conv2(x))


class CSPRep(nn.Module):
    def __init__(self, config):
        super().__init__()
        cin = config.encoder_hidden_dim * 2
        cout = config.encoder_hidden_dim
        hc = int(cout * config.hidden_expansion)
        act = config.csp_activation
        self.conv1 = ConvNorm(config, cin, hc, 1, 1, activation=act)
        self.conv2 = ConvNorm(config, cin, hc, 1, 1, activation=act)
        self.bottlenecks = nn.Sequential(*[RepVgg(config) for _ in range(3)])
        self.conv3 = ConvNorm(config, hc, cout, 1, 1, activation=act) if hc != cout else nn.Identity()

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class MHA(nn.Module):
    def __init__(self, config, hidden, heads, dropout):
        super().__init__()
        self.num_attention_heads = heads
        self.attention_head_size = hidden // heads
        self.all_head_size = hidden
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, q, k, v, attention_mask=None):
        b = q.shape[0]
        def shape(t, lin):
            return lin(t).view(b, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        ql, kl, vl = shape(q, self.query), shape(k, self.key), shape(v, self.value)
        scores = (ql @ kl.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = scores.softmax(-1)
        ctx = (probs @ vl).permute(0, 2, 1, 3).contiguous().view(b, -1, self.all_head_size)
        return self.out_proj(ctx)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MHA(config, config.encoder_hidden_dim, config.num_attention_heads, config.encoder_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.encoder_hidden_dim, config.encoder_dim_feedforward)
        self.fc2 = nn.Linear(config.encoder_dim_feedforward, config.encoder_hidden_dim)
        self.final_layer_norm = nn.LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)
        self._act = _act(config.encoder_feedforward_activation)

    def forward(self, hidden_states, pos_embed=None):
        q = k = hidden_states if pos_embed is None else hidden_states + pos_embed
        h = self.self_attn(q, k, hidden_states)
        hidden_states = self.self_attn_layer_norm(hidden_states + h)
        h = self.fc2(self._act(self.fc1(hidden_states)))
        return self.final_layer_norm(hidden_states + h)


class _EncoderStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])


class HybridEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encoder_projection_indices = config.encoder_projection_indices
        self.positional_encoding_temperature = config.positional_encoding_temperature
        self.eval_size = config.eval_size
        self.channel_projection_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, self.encoder_hidden_dim, 1, bias=False),
                           nn.BatchNorm2d(self.encoder_hidden_dim)) for c in self.in_channels]
        )
        self.encoder = nn.ModuleList([_EncoderStack(config) for _ in self.encoder_projection_indices])
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNorm(config, self.encoder_hidden_dim, self.encoder_hidden_dim, 1, 1,
                                               activation=config.conv_norm_activation))
            self.fpn_blocks.append(CSPRep(config))
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(ConvNorm(config, self.encoder_hidden_dim, self.encoder_hidden_dim, 3, 2,
                                                  activation=config.conv_norm_activation))
            self.pan_blocks.append(CSPRep(config))

    @staticmethod
    def _pos_embed(w, h, dim, temp, device, dtype):
        gw = torch.arange(int(w), dtype=dtype, device=device)
        gh = torch.arange(int(h), dtype=dtype, device=device)
        gw, gh = torch.meshgrid(gw, gh, indexing="ij")
        pd = dim // 4
        omega = torch.arange(pd, dtype=dtype, device=device) / pd
        omega = 1.0 / (temp**omega)
        ow = gw.flatten()[..., None] @ omega[None]
        oh = gh.flatten()[..., None] @ omega[None]
        return torch.concat([ow.sin(), ow.cos(), oh.sin(), oh.cos()], dim=1)[None]

    def forward(self, feats):
        proj = [self.channel_projection_layers[i](f) for i, f in enumerate(feats)]
        for eidx, fidx in enumerate(self.encoder_projection_indices):
            h, w = proj[fidx].shape[2:]
            src = proj[fidx].flatten(2).permute(0, 2, 1)
            # upstream recomputes the sincos pos-embed in training even when eval_size is set
            if self.training or self.eval_size is None:
                pos = self._pos_embed(w, h, self.encoder_hidden_dim, self.positional_encoding_temperature,
                                      src.device, src.dtype)
            else:
                pos = None
            x = src
            for layer in self.encoder[eidx].layers:
                x = layer(x, pos_embed=pos)
            proj[fidx] = x.permute(0, 2, 1).reshape(-1, self.encoder_hidden_dim, h, w).contiguous()

        fpn = [proj[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](fpn[0])
            fpn[0] = feat_high
            up = F.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            fpn.insert(0, self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([up, proj[idx - 1]], dim=1)))

        pan = [fpn[0]]
        for idx in range(len(self.in_channels) - 1):
            down = self.downsample_convs[idx](pan[-1])
            pan.append(self.pan_blocks[idx](torch.concat([down, fpn[idx + 1]], dim=1)))
        return pan


# --------------------------------------------------------------------------- #
# Decoder + Efficient Fusion Head
# --------------------------------------------------------------------------- #
def _cosine_scaled(a, b, logit_scale):
    a = a / a.norm(dim=2, keepdim=True).clamp_min(1e-12)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return logit_scale.exp() * torch.bmm(a, b)


def _class_similarity(kind, cls_feature, class_proj):
    logit_scale = torch.tensor(1 / 0.07).log()  # fixed CLIP temperature (1/0.07), as in upstream
    if kind == "cosine":
        return _cosine_scaled(cls_feature, class_proj, logit_scale)
    return torch.bmm(cls_feature, class_proj)


def _inv_sigmoid(x, eps=1e-5):
    x = x.clamp(0, 1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TaskEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.linear1 = nn.Linear(config.class_embed_dim, config.task_encoder_hidden_dim)
        self.mlp.linear2 = nn.Linear(config.task_encoder_hidden_dim, config.class_embed_dim)
        self._act = _act(config.decoder_activation)
        self.res1 = nn.Module()
        self.res1.norm1 = nn.LayerNorm(config.class_embed_dim, eps=config.layer_norm_eps)

    def forward(self, x):
        y = self.mlp.linear2(self._act(self.mlp.linear1(x)))
        return self.res1.norm1(x + y)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MHA(config, config.decoder_hidden_dim, config.decoder_num_heads, config.decoder_dropout)
        self.norm1 = nn.LayerNorm(config.decoder_hidden_dim, eps=config.layer_norm_eps)
        self.cross_attn = OmDetMSDeformAttn(config, config.decoder_num_heads, config.decoder_num_points)
        self.norm2 = nn.LayerNorm(config.decoder_hidden_dim, eps=config.layer_norm_eps)
        self.linear1 = nn.Linear(config.decoder_hidden_dim, config.decoder_dim_feedforward)
        self.linear2 = nn.Linear(config.decoder_dim_feedforward, config.decoder_hidden_dim)
        self.norm3 = nn.LayerNorm(config.decoder_hidden_dim, eps=config.layer_norm_eps)
        self._act = _act(config.decoder_activation)

    def forward(self, emb, task_features, reference_points, vision_features, vision_shapes,
                vision_shapes_list, attention_mask, query_position):
        olen = emb.shape[1]
        q = k = emb + query_position
        tf = task_features.transpose(0, 1)
        q = torch.cat((q, tf), dim=1)
        k = torch.cat((k, tf), dim=1)
        emb = torch.cat((emb, tf), dim=1)
        emb = self.norm1(emb + self.self_attn(q, k, emb, attention_mask=attention_mask))
        task_features = emb[:, olen:, :].transpose(0, 1)
        emb = emb[:, :olen, :]
        h = emb + query_position
        out = self.cross_attn(h, vision_features, reference_points.unsqueeze(2), vision_shapes, vision_shapes_list)
        emb = self.norm2(emb + out)
        h = self.linear2(self._act(self.linear1(emb)))
        emb = self.norm3(emb + h)
        return emb, task_features


class OmDetDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden = config.decoder_hidden_dim
        self.num_queries = config.num_queries
        self.class_distance_type = config.class_distance_type
        self.learn_initial_query = config.learn_initial_query
        self.decoder_num_layers = config.decoder_num_layers
        self.channel_projection_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, hidden, 1, bias=False), nn.BatchNorm2d(hidden))
             for c in config.vision_features_channels]
        )
        self.task_encoder = TaskEncoder(config)
        self.task_project = nn.Linear(config.class_embed_dim, hidden) if config.class_embed_dim != hidden else None
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_num_layers)])
        if self.learn_initial_query:
            self.tgt_embed = nn.Embedding(self.num_queries, hidden)
        self.query_position_head = MLP(4, 2 * hidden, hidden, 2)
        self.encoder_vision_features = nn.Sequential(nn.Linear(hidden, hidden),
                                                     nn.LayerNorm(hidden, eps=config.layer_norm_eps))
        self.encoder_class_head = nn.Linear(config.class_embed_dim, hidden)
        self.encoder_bbox_head = MLP(hidden, hidden, 4, 3)
        self.decoder_class_head = nn.ModuleList([nn.Linear(config.class_embed_dim, hidden)
                                                 for _ in range(config.decoder_num_layers)])
        self.decoder_bbox_head = nn.ModuleList([MLP(hidden, hidden, 4, 3)
                                                for _ in range(config.decoder_num_layers)])

    @lru_cache(maxsize=32)
    def generate_anchors(self, spatial_shapes, grid_size=0.05, device="cpu", dtype=torch.float32):
        anchors = []
        for level, (h, w) in enumerate(spatial_shapes):
            gy, gx = torch.meshgrid(torch.arange(h, dtype=dtype, device=device),
                                    torch.arange(w, dtype=dtype, device=device), indexing="ij")
            grid_xy = torch.stack([gx, gy], -1)
            valid_wh = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))
        anchors = torch.concat(anchors, 1)
        eps = 1e-2
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_encoder_input(self, vision_features):
        vf = [self.channel_projection_layers[i](f) for i, f in enumerate(vision_features)]
        feats, shapes_list = [], []
        for f in vf:
            h, w = f.shape[2:]
            feats.append(f.flatten(2).permute(0, 2, 1))
            shapes_list.append((h, w))
        feats = torch.cat(feats, 1)
        shapes = torch.tensor(shapes_list, dtype=torch.int64, device=feats.device)
        level_start = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
        return feats, shapes, shapes_list, level_start

    def _get_decoder_input(self, vision_features, vision_shapes, class_features):
        b = len(vision_features)
        anchors, valid_mask = self.generate_anchors(vision_shapes, device=vision_features.device,
                                                    dtype=vision_features.dtype)
        pcf = self.encoder_vision_features(torch.where(valid_mask, vision_features,
                                                       torch.tensor(0.0, dtype=vision_features.dtype,
                                                                    device=vision_features.device)))
        ocp = self.encoder_class_head(class_features).permute(1, 2, 0)
        enc_sim = _class_similarity(self.class_distance_type, pcf, ocp)
        enc_bboxes = self.encoder_bbox_head(pcf) + anchors
        topk = torch.topk(enc_sim.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(b, dtype=topk.dtype, device=topk.device).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        reference_points = enc_bboxes[batch_ind, topk].view(b, self.num_queries, -1)
        encoder_bboxes = reference_points.sigmoid()
        enc_sim = enc_sim[batch_ind, topk].view(b, self.num_queries, -1)
        if self.learn_initial_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        else:
            embeddings = pcf[batch_ind, topk].view(b, self.num_queries, -1)
        return embeddings, reference_points, encoder_bboxes, enc_sim, anchors

    def forward(self, vision_features, class_features, task_features, task_mask):
        vf, vshapes, vshapes_list, level_start = self._get_encoder_input(vision_features)
        b = task_mask.shape[0]
        task_features = self.task_encoder(task_features)
        if self.task_project is not None:
            task_features = self.task_project(task_features)
        src_key_mask = (task_mask == 0)
        attn_len = self.num_queries
        fusion = attn_len + task_features.shape[0]
        key_padding = torch.zeros([b, fusion], dtype=torch.bool, device=task_features.device)
        key_padding[:, attn_len:] = src_key_mask
        emb, reference_points, encoder_bboxes, enc_sim, init_ref = self._get_decoder_input(
            vf, tuple(vshapes_list), class_features)
        # bidirectional additive mask: padded keys -> min
        minv = torch.finfo(emb.dtype).min
        attention_mask = torch.where(key_padding[:, None, None, :], minv, 0.0).to(emb.dtype)
        pcf = emb
        reference_points = reference_points.sigmoid()
        dec_bboxes, dec_classes, last_refined = [], [], None
        for i, layer in enumerate(self.layers):
            pcf, task_features = layer(pcf, task_features, reference_points, vf, vshapes, vshapes_list,
                                       attention_mask, self.query_position_head(reference_points))
            refined = torch.sigmoid(self.decoder_bbox_head[i](pcf) + _inv_sigmoid(reference_points))
            ocp = self.decoder_class_head[i](class_features).permute(1, 2, 0)
            if i == self.decoder_num_layers - 1:
                dec_classes.append(_class_similarity(self.class_distance_type, pcf, ocp))
                dec_bboxes.append(refined)
                break
            # detach between layers in training (Deformable-DETR look-forward-once); no-op in eval
            reference_points = refined.detach() if self.training else refined
        return dec_bboxes[-1], dec_classes[-1], encoder_bboxes, enc_sim


class OmDetTurboDetectionModel(nn.Module):
    """Native equivalent of transformers ``OmDetTurboForObjectDetection``."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_backbone = OmDetVisionBackbone(config)
        self.language_backbone = OmDetLanguageBackbone(config)
        self.encoder = HybridEncoder(config)
        self.decoder = OmDetDecoder(config)

    def _language(self, classes_input_ids, tasks_input_ids, tasks_attention_mask, classes_structure):
        class_embs = self.language_backbone(classes_input_ids, encode_type="class")  # [total_classes, embed]
        # group flat classes back per task using classes_structure
        task_feats, task_masks = self.language_backbone(tasks_input_ids, mask=tasks_attention_mask, encode_type="task")
        # class_features: pad to max classes per image -> [max_classes, batch, embed]
        splits = torch.split(class_embs, classes_structure.tolist())
        max_c = max(s.shape[0] for s in splits)
        padded = [F.pad(s, (0, 0, 0, max_c - s.shape[0])) for s in splits]
        class_features = torch.stack(padded, dim=1)  # [max_c, batch, embed]
        return class_features, task_feats, task_masks

    def forward(self, pixel_values, classes_input_ids, classes_attention_mask,
                tasks_input_ids, tasks_attention_mask, classes_structure):
        feats = self.vision_backbone(pixel_values)
        extracted = self.encoder(feats)
        class_features, task_features, task_mask = self._language(
            classes_input_ids, tasks_input_ids, tasks_attention_mask, classes_structure)
        coords, classes, enc_coords, enc_classes = self.decoder(
            extracted, class_features, task_features, task_mask)
        return {
            "decoder_coord_logits": coords,
            "decoder_class_logits": classes,
            "encoder_coord_logits": enc_coords,
            "encoder_class_logits": enc_classes,
        }


__all__ = ["OmDetTurboDetectionModel"]
