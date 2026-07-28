"""Cross-load parity: LibreFeyNobg vs the upstream nobg reference.

FeyNobg (https://huggingface.co/feyninc/FeyNobg, Apache-2.0) is BiRefNet with
stage 3 of the Swin-L backbone deepened from 18 to 24 blocks. The reference
implementation is the ``nobg`` library (https://github.com/feyninc/nobg).
This is a torch-to-torch port, so the gate is exact zero at fp32 CPU eval.

Prereqs (throwaway env; do NOT install nobg into the main .venv):
    pip install torch nobg safetensors
Set:
    FEYNOBG_CKPT = path to the FeyNobg model.safetensors (or leave unset to
                   let nobg pull it from the Hugging Face cache)

Run:
    python weights/parity_feynobg.py
"""

from __future__ import annotations

import os
import sys

import torch


def _extract_logit(out):
    """Last-scale single-channel logit from either API's output shape."""
    if hasattr(out, "logits"):
        out = out.logits
    if isinstance(out, (list, tuple)):
        out = out[-1]
    return out


def main() -> int:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from libreyolo.models.feynobg.model import LibreFeyNobg

    from nobg import BiRefNet

    torch.manual_seed(0)
    x = torch.randn(2, 3, 1024, 1024)

    ref = BiRefNet.from_pretrained(os.environ.get("FEYNOBG_CKPT", "feyninc/FeyNobg"))
    ref.eval()

    ours = LibreFeyNobg(model_path=None, size="l", device="cpu")
    result = ours.model.load_state_dict(ref.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys

    with torch.no_grad():
        ref_out = _extract_logit(ref(pixel_values=x))
        our_out = _extract_logit(ours.model(x))

    diff = (ref_out - our_out).abs().max().item()
    print(f"max_abs_diff = {diff}")
    if diff != 0.0:
        print("FAIL: expected exact fp32 parity for a torch-to-torch port")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
