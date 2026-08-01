"""Inference-parity proof for LibreLWDETR vs upstream Atten4Vis/LW-DETR.

Per the libreyolo-port-model skill (section 12). Two checks per size, gated on
local paths:

1. **Structural (always):** the upstream ``LWDETR_<size>_60e_coco.pth`` model
   state dict strict-loads into the native port with 0 missing / 0 unexpected
   keys.
2. **Numerical (when the upstream repo is available):** build the official
   ``LWDETR`` from the ``args`` namespace stored in the checkpoint, load the
   same weights into both, and compare ``pred_logits`` / ``pred_boxes`` on
   identical input in eval mode.

Confirmed 2026-08-01 on all five released sizes:
``max_abs_diff == 0.0`` for both outputs — bitwise identical.

    export LWDETR_OFFICIAL_CKPT_DIR=/path/to/checkpoints   # required
    export LWDETR_OFFICIAL_REPO=/path/to/Atten4Vis/LW-DETR # optional; enables (2)
    python weights/parity_lwdetr.py

Two upstream imports are stubbed for check (2), neither of which executes in
this configuration:

* ``fairscale.nn.checkpoint`` — only used when ``use_act_checkpoint=True``,
  which the released configs never set.
* ``MultiScaleDeformableAttention`` — the CUDA extension. Every upstream
  ``MSDeformAttn`` is switched to its own pure-PyTorch reference core (the path
  upstream itself takes when exporting or running fp16), which is the same core
  the port uses. Both stubs raise if reached, so they cannot mask a difference.
"""

from __future__ import annotations

import copy
import os
import sys
import types
from pathlib import Path

import torch

from _conversion_utils import add_repo_root_to_path

add_repo_root_to_path()

SIZES = {"t": "tiny", "s": "small", "m": "medium", "l": "large", "x": "xlarge"}


def _install_upstream_stubs() -> None:
    fairscale = types.ModuleType("fairscale")
    fs_nn = types.ModuleType("fairscale.nn")
    fs_ckpt = types.ModuleType("fairscale.nn.checkpoint")

    def _unreachable(*args, **kwargs):
        raise AssertionError("stubbed upstream helper must not be reached")

    fs_ckpt.checkpoint_wrapper = _unreachable
    fs_nn.checkpoint = fs_ckpt
    fairscale.nn = fs_nn
    sys.modules.setdefault("fairscale", fairscale)
    sys.modules.setdefault("fairscale.nn", fs_nn)
    sys.modules.setdefault("fairscale.nn.checkpoint", fs_ckpt)

    msda = types.ModuleType("MultiScaleDeformableAttention")
    msda.ms_deform_attn_forward = _unreachable
    msda.ms_deform_attn_backward = _unreachable
    sys.modules.setdefault("MultiScaleDeformableAttention", msda)


def _structural_check(size: str, state_dict: dict) -> tuple[int, int]:
    from libreyolo.models.lwdetr.nn import LibreLWDETRModel

    nc = int(state_dict["class_embed.weight"].shape[0])
    model = LibreLWDETRModel(size=size, nc=nc)
    result = model.load_state_dict(state_dict, strict=False)
    return len(result.missing_keys), len(result.unexpected_keys)


def _numerical_check(size: str, checkpoint: dict, repo: str) -> dict[str, float] | None:
    """Return per-output max_abs_diff vs the official model, or None if absent."""
    if not repo or not Path(repo).is_dir():
        return None

    sys.path.insert(0, repo)
    _install_upstream_stubs()
    try:
        import models as upstream_models
        from models.ops.modules import MSDeformAttn as UpstreamMSDeformAttn
        from util.misc import NestedTensor
    except Exception as exc:  # pragma: no cover - depends on local checkout
        print(f"  numerical check skipped: cannot import upstream ({exc})")
        return None

    from libreyolo.models.lwdetr.nn import LibreLWDETRModel

    args = copy.deepcopy(checkpoint["args"])
    args.device = "cpu"
    args.pretrained_encoder = None
    official, _, _ = upstream_models.build_model(args)
    official.load_state_dict(checkpoint["model"], strict=True)
    official.eval()
    for module in official.modules():
        if isinstance(module, UpstreamMSDeformAttn):
            module._export = True  # take the pure-PyTorch reference core

    nc = int(checkpoint["model"]["class_embed.weight"].shape[0])
    ours = LibreLWDETRModel(size=size, nc=nc)
    ours.load_state_dict(checkpoint["model"], strict=True)
    ours.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)
    mask = torch.zeros((1, 640, 640), dtype=torch.bool)
    with torch.no_grad():
        official_out = official(NestedTensor(x, mask))
        our_out = ours(x)

    return {
        key: (official_out[key] - our_out[key]).abs().max().item()
        for key in ("pred_logits", "pred_boxes")
    }


def main() -> int:
    ckpt_dir = os.environ.get("LWDETR_OFFICIAL_CKPT_DIR")
    if not ckpt_dir:
        raise SystemExit(
            "Set LWDETR_OFFICIAL_CKPT_DIR to the directory holding "
            "LWDETR_<size>_60e_coco.pth"
        )
    repo = os.environ.get("LWDETR_OFFICIAL_REPO", "")

    failures = []
    for size, name in SIZES.items():
        path = Path(ckpt_dir) / f"LWDETR_{name}_60e_coco.pth"
        if not path.exists():
            print(f"{size}/{name}: SKIP (missing {path})")
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"]

        missing, unexpected = _structural_check(size, state_dict)
        ok = missing == 0 and unexpected == 0
        print(f"{size}/{name}: missing={missing} unexpected={unexpected}")

        diffs = _numerical_check(size, checkpoint, repo)
        if diffs is not None:
            for key, diff in diffs.items():
                print(f"    {key}: max_abs_diff={diff!r}")
                ok = ok and diff == 0.0

        print(f"    => {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(size)

    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
