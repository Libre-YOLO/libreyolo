"""Native MiDaS v2.1 Small and DPT-Large inference graphs.

The architecture and module names follow ``isl-org/MiDaS`` at commit
``454597711a62eabcbf7d1e89f3fb9f569051ac9b`` (MIT). Encoder construction uses
the optional Apache-2.0 ``timm`` dependency. timm 1.0.28 was checked against
both official checkpoints: every encoder key and shape matches, and its
EfficientNet-Lite3 forward is bit-identical to MiDaS's historical Torch Hub
dependency.
"""

from __future__ import annotations

import importlib.util
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DPT_MEAN = (0.5, 0.5, 0.5)
_DPT_STD = (0.5, 0.5, 0.5)


def _require_timm():
    if importlib.util.find_spec("timm") is None:
        raise ModuleNotFoundError(
            "MiDaS support requires timm. Install with: pip install 'libreyolo[midas]'"
        )
    import timm

    return timm


class Interpolate(nn.Module):
    """Module form of ``torch.nn.functional.interpolate``."""

    def __init__(self, scale_factor: float, mode: str, align_corners: bool = False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )


class ResidualConvUnitCustom(nn.Module):
    """Two-convolution residual unit used by the MiDaS refinement decoder."""

    def __init__(self, features: int, activation: nn.Module, bn: bool = False):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        if bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlockCustom(nn.Module):
    """Residual feature fusion followed by 2x bilinear upsampling."""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        *,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
    ):
        super().__init__()
        self.align_corners = align_corners
        self.expand = expand
        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_features, 1, bias=True)
        self.resConfUnit1 = ResidualConvUnitCustom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnitCustom(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(
        self,
        *xs: torch.Tensor,
        size: Sequence[int] | None = None,
    ) -> torch.Tensor:
        output = xs[0]
        if len(xs) == 2:
            output = self.skip_add.add(output, self.resConfUnit1(xs[1]))
        output = self.resConfUnit2(output)
        if size is None:
            output = F.interpolate(
                output,
                scale_factor=2,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        else:
            output = F.interpolate(
                output,
                size=list(size),
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return self.out_conv(output)


def _make_scratch(
    in_channels: Sequence[int],
    features: int,
    *,
    expand: bool,
) -> nn.Module:
    scratch = nn.Module()
    out_channels = [features] * 4
    if expand:
        out_channels = [features, features * 2, features * 4, features * 8]
    for index, (in_ch, out_ch) in enumerate(
        zip(in_channels, out_channels, strict=True), start=1
    ):
        setattr(
            scratch,
            f"layer{index}_rn",
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        )
    return scratch


class _NormalizedMiDaS(nn.Module):
    """Keep normalization in-graph for native/exported input parity."""

    def _set_normalization(
        self,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.register_buffer(
            "pixel_mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.pixel_mean) / self.pixel_std
        return self.forward_normalized(normalized).unsqueeze(1)


def _make_efficientnet_lite3() -> nn.Module:
    timm = _require_timm()
    efficientnet = timm.create_model(
        "tf_efficientnet_lite3",
        pretrained=False,
        exportable=True,
    )
    pretrained = nn.Module()
    # timm >= 1 folds act1 into BatchNormAct2d. The Identity preserves the
    # historical sequential indexes (and therefore official checkpoint keys)
    # while producing the same arithmetic as BatchNorm2d + ReLU6.
    pretrained.layer1 = nn.Sequential(
        efficientnet.conv_stem,
        efficientnet.bn1,
        nn.Identity(),
        *efficientnet.blocks[0:2],
    )
    pretrained.layer2 = nn.Sequential(*efficientnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*efficientnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*efficientnet.blocks[5:9])
    return pretrained


class MiDaSSmall(_NormalizedMiDaS):
    """MiDaS v2.1 Small with an EfficientNet-Lite3 encoder."""

    def __init__(self):
        super().__init__()
        features = 64
        self.pretrained = _make_efficientnet_lite3()
        self.scratch = _make_scratch(
            [32, 48, 136, 384],
            features,
            expand=True,
        )
        self.scratch.activation = nn.ReLU(False)
        self.scratch.refinenet4 = FeatureFusionBlockCustom(
            features * 8,
            self.scratch.activation,
            expand=True,
        )
        self.scratch.refinenet3 = FeatureFusionBlockCustom(
            features * 4,
            self.scratch.activation,
            expand=True,
        )
        self.scratch.refinenet2 = FeatureFusionBlockCustom(
            features * 2,
            self.scratch.activation,
            expand=True,
        )
        self.scratch.refinenet1 = FeatureFusionBlockCustom(
            features,
            self.scratch.activation,
        )
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, 3, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, 1),
            nn.ReLU(True),
            nn.Identity(),
        )
        self._set_normalization(_IMAGENET_MEAN, _IMAGENET_STD)

    def forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        return self.scratch.output_conv(path_1).squeeze(1)


class Slice(nn.Module):
    def __init__(self, start_index: int = 1):
        super().__init__()
        self.start_index = start_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.start_index :]


class ProjectReadout(nn.Module):
    def __init__(self, in_features: int, start_index: int = 1):
        super().__init__()
        self.start_index = start_index
        self.project = nn.Sequential(
            nn.Linear(2 * in_features, in_features),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x[:, self.start_index :]
        readout = x[:, 0].unsqueeze(1).expand_as(tokens)
        return self.project(torch.cat((tokens, readout), dim=-1))


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


def _make_dpt_backbone() -> nn.Module:
    timm = _require_timm()
    model = timm.create_model("vit_large_patch16_384", pretrained=False)
    pretrained = nn.Module()
    pretrained.model = model
    features = [256, 512, 1024, 1024]
    postprocess = []
    scales = (4, 2, 1, 0.5)
    for out_channels, scale in zip(features, scales, strict=True):
        layers: list[nn.Module] = [
            ProjectReadout(1024),
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([24, 24])),
            nn.Conv2d(1024, out_channels, 1),
        ]
        if scale == 4:
            layers.append(nn.ConvTranspose2d(out_channels, out_channels, 4, stride=4))
        elif scale == 2:
            layers.append(nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2))
        elif scale == 0.5:
            layers.append(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1))
        postprocess.append(nn.Sequential(*layers))

    for index, module in enumerate(postprocess, start=1):
        setattr(pretrained, f"act_postprocess{index}", module)
    pretrained.model.start_index = 1
    pretrained.model.patch_size = [16, 16]
    return pretrained


def _resize_pos_embed(
    model: nn.Module,
    posemb: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    posemb_tok = posemb[:, : model.start_index]
    posemb_grid = posemb[0, model.start_index :]
    old_grid = int(model.patch_embed.grid_size[0])
    posemb_grid = posemb_grid.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(grid_h, grid_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
    return torch.cat([posemb_tok, posemb_grid], dim=1)


class DPTLarge(_NormalizedMiDaS):
    """MiDaS DPT-Large with a ViT-L/16 encoder and refinement decoder."""

    _HOOKS = (5, 11, 17, 23)

    def __init__(self):
        super().__init__()
        features = 256
        self.pretrained = _make_dpt_backbone()
        self.scratch = _make_scratch(
            [256, 512, 1024, 1024],
            features,
            expand=False,
        )
        activation = nn.ReLU(False)
        self.scratch.refinenet1 = FeatureFusionBlockCustom(features, activation)
        self.scratch.refinenet2 = FeatureFusionBlockCustom(features, activation)
        self.scratch.refinenet3 = FeatureFusionBlockCustom(features, activation)
        self.scratch.refinenet4 = FeatureFusionBlockCustom(features, activation)
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(True),
            nn.Identity(),
        )
        self._set_normalization(_DPT_MEAN, _DPT_STD)

    def _forward_encoder(self, x: torch.Tensor) -> list[torch.Tensor]:
        model = self.pretrained.model
        _, _, height, width = x.shape
        pos_embed = _resize_pos_embed(
            model,
            model.pos_embed,
            height // model.patch_size[1],
            width // model.patch_size[0],
        )
        tokens = model.patch_embed.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = model.cls_token.expand(tokens.shape[0], -1, -1)
        if model.no_embed_class:
            tokens = tokens + pos_embed
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        if not model.no_embed_class:
            tokens = tokens + pos_embed
        tokens = model.pos_drop(tokens)

        activations: list[torch.Tensor] = []
        for index, block in enumerate(model.blocks):
            tokens = block(tokens)
            if index in self._HOOKS:
                activations.append(tokens)
        # MiDaS's hooks capture the selected block outputs before ``model.norm``;
        # the normalized final token sequence is intentionally unused.
        model.norm(tokens)
        return activations

    def _postprocess_tokens(
        self,
        tokens: torch.Tensor,
        stage: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        postprocess = getattr(self.pretrained, f"act_postprocess{stage}")
        features = postprocess[0:2](tokens)
        features = features.unflatten(2, (grid_h, grid_w))
        return postprocess[3:](features)

    def forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        grid_h, grid_w = height // 16, width // 16
        tokens = self._forward_encoder(x)
        layers = [
            self._postprocess_tokens(value, index, grid_h, grid_w)
            for index, value in enumerate(tokens, start=1)
        ]

        layer_1_rn = self.scratch.layer1_rn(layers[0])
        layer_2_rn = self.scratch.layer2_rn(layers[1])
        layer_3_rn = self.scratch.layer3_rn(layers[2])
        layer_4_rn = self.scratch.layer4_rn(layers[3])

        path_4 = self.scratch.refinenet4(
            layer_4_rn,
            size=layer_3_rn.shape[2:],
        )
        path_3 = self.scratch.refinenet3(
            path_4,
            layer_3_rn,
            size=layer_2_rn.shape[2:],
        )
        path_2 = self.scratch.refinenet2(
            path_3,
            layer_2_rn,
            size=layer_1_rn.shape[2:],
        )
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        return self.scratch.output_conv(path_1).squeeze(1)


def build_midas_model(size: str) -> nn.Module:
    if size == "s":
        return MiDaSSmall()
    if size == "l":
        return DPTLarge()
    raise ValueError(f"Unknown MiDaS size {size!r}; expected 's' or 'l'.")


__all__ = ["DPTLarge", "MiDaSSmall", "build_midas_model"]
