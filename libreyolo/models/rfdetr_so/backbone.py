"""RF-DETR-SO backbone: stock DINOv2 + projector with SSA and PBM on top.

``BackboneSO`` adopts the encoder and multi-scale projector of an
already-built stock :class:`~libreyolo.models.rfdetr.backbone.Backbone`
(constructed by ``build_model`` with ``projector_scale=(P3, P4, P5)``) and
adds the small-object modules:

- an SDE branch on the raw image (stride-4 / stride-8 spatial detail),
- a 1x1 fusion of the SDE stride-8 map into the projector's P3 level,
- two parallel bi-fusion (PBM) blocks refining P3 and P4.

Adopting the stock modules by attribute transplant (instead of rebuilding
them) keeps state-dict keys identical to the base family
(``backbone.0.encoder.*`` / ``backbone.0.projector.*``), which is what makes
base RF-DETR checkpoints transfer with a pure key remap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ..rfdetr.backbone import Backbone, ConvX
from ..rfdetr.tensors import NestedTensor
from .ssa import BiFusionBlock, SpatialDetailExtractor


class BackboneSO(Backbone):
    """Small-object RF-DETR backbone (3 pyramid levels + SSA + PBM)."""

    def __init__(self, stock: Backbone, hidden_dim: int, ssa_channels: int = 32):
        # Bypass Backbone.__init__ (it would rebuild the encoder); adopt the
        # already-constructed modules instead so parameters, freezing state,
        # and state-dict keys carry over untouched.
        nn.Module.__init__(self)
        if stock.cross_attn_projector is not None:
            raise ValueError("RF-DETR-SO does not support the dual projector")
        if len(stock.projector_scale) != 3:
            raise ValueError(
                "BackboneSO expects a 3-level projector (P3, P4, P5); got "
                f"{stock.projector_scale}"
            )
        self.encoder = stock.encoder
        self.projector = stock.projector
        self.projector_scale = stock.projector_scale
        self.dual_projector = stock.dual_projector
        self.cross_attn_projector = None
        self._export = False

        self.ssa_sde = SpatialDetailExtractor(ssa_channels)
        self.ssa_fuse = ConvX(
            hidden_dim + self.ssa_sde.out_channels_s8,
            hidden_dim,
            kernel=1,
            layer_norm=True,
        )
        self.pbm3 = BiFusionBlock(
            hidden_dim, shallow_channels=self.ssa_sde.out_channels_s4
        )
        self.pbm4 = BiFusionBlock(hidden_dim, shallow_channels=hidden_dim)

    def _forward_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        v3, v4, v5 = self.projector(self.encoder(images))

        f2, sde3 = self.ssa_sde(images)
        f3 = self.ssa_fuse(torch.cat([v3, sde3], dim=1))

        # PBM is parallel: both bi-fusion blocks consume the pre-fusion
        # pyramid (f2/f3/v4/v5), not each other's outputs.
        p3 = self.pbm3(deep=v4, cur=f3, shallow=f2)
        p4 = self.pbm4(deep=v5, cur=v4, shallow=f3)
        return [p3, p4, v5]

    def forward(self, tensor_list: NestedTensor):
        feats = self._forward_features(tensor_list.tensors)
        out = []
        for feat in feats:
            mask = tensor_list.mask
            assert mask is not None
            mask = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(
                torch.bool
            )[0]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        feats = self._forward_features(tensors)
        out_feats = []
        out_masks = []
        for feat in feats:
            b, _, h, w = feat.shape
            out_masks.append(
                torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
            )
            out_feats.append(feat)
        return out_feats, out_masks


__all__ = ["BackboneSO"]
