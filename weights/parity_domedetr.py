"""Cross-load upstream Dome-DETR weights into LibreDOMEDETR and assert max_abs_diff == 0.

Not part of CI: it needs the upstream checkout and the published checkpoints.

    export DOMEDETR_UPSTREAM_DIR=/path/to/Dome-DETR         # git clone, master
    export DOMEDETR_OFFICIAL_CKPT_DIR=/path/to/best_ckpts_dome_2026
    python weights/parity_domedetr.py

The MWAS window mask is compared *before* the logits: window selection is data
dependent, so a mask that drifts would make a matching set of logits an
accident rather than a proof.

Upstream pulls tensorboard / scikit-image / a prebuilt MSDeformAttn extension
through package __init__ chains its eval path never touches, so those are
stubbed here rather than installed.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch

UPSTREAM_DIR = os.environ.get("DOMEDETR_UPSTREAM_DIR")
CKPT_DIR = os.environ.get("DOMEDETR_OFFICIAL_CKPT_DIR")
if not UPSTREAM_DIR or not CKPT_DIR:
    raise SystemExit("Set DOMEDETR_UPSTREAM_DIR and DOMEDETR_OFFICIAL_CKPT_DIR")

UPSTREAM_ROOT = Path(UPSTREAM_DIR)

# Upstream config, transcribed from configs/dome/Dome-{S,M,L}-{AITOD,VisDrone}.yml
SIZE_CONFIGS = {
    "s": dict(backbone="B0", use_lab=True, in_channels=[64, 256, 512, 1024],
              depth_mult=0.34, expansion=0.5),
    "m": dict(backbone="B2", use_lab=True, in_channels=[96, 384, 768, 1536],
              depth_mult=0.67, expansion=1.0),
    "l": dict(backbone="B4", use_lab=False, in_channels=[128, 512, 1024, 2048],
              depth_mult=1.0, expansion=1.0),
}
# L is 4 decoder layers on AI-TOD-V2 but 6 on VisDrone.
DEC_NUM_LAYERS = {("s", "aitod"): 3, ("m", "aitod"): 4, ("l", "aitod"): 4,
                  ("s", "visdrone"): 3, ("m", "visdrone"): 4, ("l", "visdrone"): 6}
VARIANTS = {
    "aitod": dict(min_num_select=300, max_num_select=1500, nc=9),
    "visdrone": dict(min_num_select=250, max_num_select=500, nc=12),
}
CHECKPOINTS = {
    ("s", "aitod"): "aitod-s-best.pth",
    ("m", "aitod"): "aitod-m-best.pth",
    ("l", "aitod"): "aitod-l-best.pth",
    ("s", "visdrone"): "dome-s-visdrone_converted.pth",
    ("m", "visdrone"): "dome-m-visdrone_converted.pth",
    ("l", "visdrone"): "dome-l-visdrone_converted.pth",
}


def _stub(name: str, path: Path | None = None) -> types.ModuleType:
    mod = types.ModuleType(name)
    if path is not None:
        mod.__path__ = [str(path)]
    sys.modules[name] = mod
    return mod


def install_upstream_shim() -> None:
    if "src.zoo.dome" in sys.modules:
        return
    _stub("tools")
    _stub("tools.visualize_src_flatten").visualize_src_flatten = lambda *a, **k: None
    _stub("tools.visualize_image_annotation").visualize_detection = lambda *a, **k: None
    _stub("skimage")
    measure = _stub("skimage.measure")
    measure.label = measure.regionprops = None

    core = _stub("src.core", UPSTREAM_ROOT / "src" / "core")
    core.register = lambda *a, **k: (lambda cls: cls)

    _stub("src", UPSTREAM_ROOT / "src")
    _stub("src.zoo", UPSTREAM_ROOT / "src" / "zoo")
    _stub("src.zoo.dome", UPSTREAM_ROOT / "src" / "zoo" / "dome")
    _stub("src.nn", UPSTREAM_ROOT / "src" / "nn")
    _stub("src.nn.backbone", UPSTREAM_ROOT / "src" / "nn" / "backbone")
    _stub("src.misc")
    _dist = _stub("src.misc.dist_utils")
    _dist.get_world_size = lambda: 1
    _dist.is_dist_available_and_initialized = lambda: False
    _stub("src.zoo.dome.ops")
    _stub("src.zoo.dome.ops.modules").MSDeformAttn = object
    sys.path.insert(0, str(UPSTREAM_ROOT))


def build_upstream(size: str, variant: str):
    install_upstream_shim()
    from src.nn.backbone.hgnetv2 import HGNetv2
    from src.zoo.dome.dome import DOME
    from src.zoo.dome.dome_decoder import DomeTransformer
    from src.zoo.dome.hybrid_encoder import HybridEncoder

    cfg, var = SIZE_CONFIGS[size], VARIANTS[variant]
    eval_size = [800, 800]
    backbone = HGNetv2(
        name=cfg["backbone"], return_idx=[0, 1, 2, 3], freeze_at=-1, freeze_norm=False,
        use_lab=cfg["use_lab"], freeze_stem_only=True, pretrained=False,
    )
    encoder = HybridEncoder(
        in_channels=cfg["in_channels"], feat_strides=[4, 8, 16, 32], hidden_dim=256,
        nhead=8, dim_feedforward=1024, dropout=0.0, enc_act="gelu", use_encoder_idx=[3],
        num_encoder_layers=1, expansion=cfg["expansion"], depth_mult=cfg["depth_mult"],
        act="silu", eval_spatial_size=eval_size, use_hybrid=True, use_deformable=False,
        enc_n_points=6, num_feature_levels=4, use_defe=True, defe_type="light",
        use_mwas=True, mwas_window_size=10,
    )
    decoder = DomeTransformer(
        num_classes=var["nc"], hidden_dim=256, feat_channels=[256, 256, 256, 256],
        feat_strides=[4, 8, 16, 32], num_levels=4, num_points=[4, 4, 4, 4],
        num_layers=DEC_NUM_LAYERS[(size, variant)], eval_idx=-1, num_denoising=100, reg_max=32,
        reg_scale=4, layer_scale=1, eval_spatial_size=eval_size, aux_loss=True,
        min_num_select=var["min_num_select"], max_num_select=var["max_num_select"],
    )
    return DOME(backbone=backbone, encoder=encoder, decoder=decoder)


def loss_parity(size: str = "s", variant: str = "visdrone") -> bool:
    """Compare every training loss term against upstream's criterion.

    Inference parity says nothing about the objective: the losses could all be
    subtly wrong and the eval forward still match. This runs both criteria over
    identical model outputs and targets and diffs every term, including the
    auxiliary, denoising and encoder variants.
    """
    from src.zoo.dome.dome_criterion import DomeCriterion as UpstreamCriterion
    from src.zoo.dome.matcher import HungarianMatcher as UpstreamMatcher

    from libreyolo.models.dfine.matcher import HungarianMatcher as OurMatcher
    from libreyolo.models.domedetr.loss import DomeCriterion as OurCriterion
    from libreyolo.models.domedetr.nn import LibreDOMEDETRModel

    nc = VARIANTS[variant]["nc"]
    torch.manual_seed(0)
    model = LibreDOMEDETRModel(config=size, nb_classes=nc, variant=variant).train()
    targets = [
        {"labels": torch.randint(0, nc, (6,)), "boxes": torch.rand(6, 4) * 0.2 + 0.4},
        {"labels": torch.randint(0, nc, (11,)), "boxes": torch.rand(11, 4) * 0.2 + 0.4},
    ]
    torch.manual_seed(0)
    outputs = model(torch.randn(2, 3, 800, 800), targets=targets)

    weights = {"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0,
               "loss_fgl": 0.15, "loss_ddf": 1.5}
    match_w = {"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0}
    kwargs = dict(weight_dict=weights, losses=["vfl", "boxes", "local"],
                  alpha=0.75, gamma=2.0, num_classes=nc, reg_max=32)

    up = UpstreamCriterion(
        matcher=UpstreamMatcher(weight_dict=match_w, use_focal_loss=True, alpha=0.25, gamma=2.0),
        # Upstream's code default is 4; every shipped config sets 1, which is
        # what the released checkpoints were trained with and what we default to.
        defe_density_map_weight=1,
        **kwargs,
    )
    ours = OurCriterion(
        matcher=OurMatcher(weight_dict=match_w, use_focal_loss=True, alpha=0.25, gamma=2.0),
        **kwargs,
    )

    torch.manual_seed(0)
    up_losses = {k: float(v.detach()) for k, v in up(outputs, targets).items()}
    torch.manual_seed(0)
    our_losses = {k: float(v.detach()) for k, v in ours(outputs, targets).items()}
    rename = {"defe_reg_loss": "loss_defe_reg", "defe_density_loss": "loss_defe_density"}
    up_losses = {rename.get(k, k): v for k, v in up_losses.items()}

    worst, ok = 0.0, True
    for key in sorted(set(up_losses) | set(our_losses)):
        if key not in up_losses or key not in our_losses:
            print(f"  {key}: MISSING from one side")
            ok = False
            continue
        diff = abs(up_losses[key] - our_losses[key])
        worst = max(worst, diff)
        if diff > 0.0:
            print(f"  {key}: upstream={up_losses[key]:.6f} ours={our_losses[key]:.6f} diff={diff:.3e}")
            ok = False
    print(f"loss parity {size}/{variant}: {len(our_losses)} terms, worst diff {worst:.3e}")
    return ok


def main() -> int:
    # Running as a script puts weights/ on sys.path, not the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from libreyolo.models.domedetr.nn import LibreDOMEDETRModel

    torch.manual_seed(0)
    x = torch.randn(1, 3, 800, 800)
    failures = []

    for (size, variant), filename in CHECKPOINTS.items():
        path = Path(CKPT_DIR) / filename
        if not path.exists():
            print(f"{size}/{variant}: SKIP (missing {filename})")
            continue

        state = torch.load(path, map_location="cpu", weights_only=False)["model"]

        upstream = build_upstream(size, variant)
        upstream.load_state_dict(state, strict=True)
        upstream.eval()

        ours = LibreDOMEDETRModel(
            config=size, nb_classes=VARIANTS[variant]["nc"], variant=variant
        )
        ours.load_state_dict(state, strict=True)
        ours.eval()

        with torch.no_grad():
            up_out = upstream(x)
            # Stepped through by hand so the intermediate DeFE mask is visible.
            encoder_out = ours.encoder(ours.backbone(x), img_inputs=x)
            our_out = ours.decoder(encoder_out)

        # Gate 1: the MWAS window mask must match bit for bit.
        up_mask = up_out["defe"]["defe_window_mask"]
        our_mask = encoder_out["defe"]["defe_window_mask"]
        mask_equal = bool(torch.equal(up_mask, our_mask))

        diffs = {}
        for key in ("pred_logits", "pred_boxes"):
            diffs[key] = (up_out[key] - our_out[key]).abs().max().item()

        ok = mask_equal and all(v == 0.0 for v in diffs.values())
        status = "OK" if ok else "FAIL"
        print(
            f"{size}/{variant:8s}: {status}  mask_identical={mask_equal} "
            f"queries={up_out['pred_logits'].shape[1]}/{our_out['pred_logits'].shape[1]} "
            + " ".join(f"{k}_max_abs_diff={v:g}" for k, v in diffs.items())
        )
        if not ok:
            failures.append(f"{size}/{variant}")

    print("\n== training-objective parity ==")
    if not loss_parity():
        failures.append("loss")

    if failures:
        print(f"\nPARITY FAILED for: {', '.join(failures)}")
        return 1
    print("\nAll sizes/variants: max_abs_diff == 0.0; every loss term matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
