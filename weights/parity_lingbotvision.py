"""Cross-load upstream LingBot-Vision weights and assert backbone parity.

Compares the reference implementation (https://github.com/robbyant/lingbot-vision)
against the LibreYOLO native port on identical inputs, fp32, eval mode. The
port must produce max_abs_diff == 0 on both the CLS token and the patch tokens.

Requires the reference repo on PYTHONPATH (plus its omegaconf dependency) and
the upstream checkpoints (auto-downloaded from Hugging Face by the reference
loader). Not a CI test — run manually:

    LINGBOTVISION_REFERENCE_REPO=/path/to/lingbot-vision \
        python weights/parity_lingbotvision.py [--sizes s b l] [--res 224]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REFERENCE_REPO = os.environ.get("LINGBOTVISION_REFERENCE_REPO")
if not REFERENCE_REPO:
    raise SystemExit("Set LINGBOTVISION_REFERENCE_REPO to a clone of robbyant/lingbot-vision")
sys.path.insert(0, REFERENCE_REPO)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", default=["s", "b", "l"])
    p.add_argument("--res", type=int, default=224)
    args = p.parse_args()

    from lingbot_vision import load_pretrained_backbone

    from libreyolo.models.lingbotvision.nn import LingBotVisionBackbone

    for size in args.sizes:
        reference, embed_dim = load_pretrained_backbone(
            variant=size, device="cpu", dtype=torch.float32
        )
        reference.eval()

        ours = LingBotVisionBackbone(size=size)
        ours.load_state_dict(reference.state_dict(), strict=True)
        ours.eval()

        torch.manual_seed(0)
        x = torch.rand(1, 3, args.res, args.res)
        with torch.no_grad():
            ref_out = reference(x, is_training=True)
            our_cls, our_patch = ours(x)

        diff_patch = (ref_out["x_norm_patchtokens"] - our_patch).abs().max().item()
        diff_cls = (ref_out["x_norm_clstoken"] - our_cls).abs().max().item()
        assert diff_patch == 0.0, f"size={size} patch tokens max_abs_diff={diff_patch}"
        assert diff_cls == 0.0, f"size={size} cls token max_abs_diff={diff_cls}"
        print(f"size={size}: OK (embed_dim={embed_dim}, {args.res}px, max_abs_diff=0.0)")


if __name__ == "__main__":
    main()
