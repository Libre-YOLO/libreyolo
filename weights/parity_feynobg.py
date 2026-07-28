"""Cross-load parity: LibreFeyNobg vs the upstream nobg reference.

FeyNobg (https://huggingface.co/feyninc/FeyNobg, Apache-2.0) is BiRefNet with
stage 3 of the Swin-L backbone deepened from 18 to 24 blocks. The reference
implementation is the ``nobg`` library (https://github.com/feyninc/nobg).
This is a torch-to-torch port, so the gate is exact zero at fp32 CPU eval.

The script runs inside a throwaway env (torch + torchvision + nobg +
safetensors; do NOT install nobg into the main .venv) and imports the shared
BiRefNet nn module standalone, so no other LibreYOLO dependencies are needed.

Set:
    FEYNOBG_REF_DIR = dir with the upstream config.json + model.safetensors
                      (or leave unset to let nobg pull feyninc/FeyNobg from HF)
    LIBRE_FEYNOBG_CKPT = path to the converted LibreFeyNobgl-matte.pt

Run:
    python weights/parity_feynobg.py
"""

from __future__ import annotations

import importlib.util
import os

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_birefnet_nn():
    path = os.path.join(_REPO, "libreyolo", "models", "birefnet", "nn.py")
    spec = importlib.util.spec_from_file_location("birefnet_nn_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract_logit(out):
    """Last-scale single-channel logit from either API's output shape."""
    if isinstance(out, dict):
        for key in ("logits", "alphas", "predictions", "last_hidden_state"):
            if key in out:
                out = out[key]
                break
        else:
            out = list(out.values())[-1]
    elif hasattr(out, "logits"):
        out = out.logits
    if isinstance(out, (list, tuple)):
        out = out[-1]
    return out


def main() -> int:
    from nobg import BiRefNet

    nn_mod = _load_birefnet_nn()
    dims = nn_mod.BiRefNetDims(192, (2, 2, 24, 2), (6, 12, 24, 48), 12, (1536, 768, 384, 192))
    ours = nn_mod.LibreBiRefNetModel(size="l", dims=dims)

    ckpt_path = os.environ.get(
        "LIBRE_FEYNOBG_CKPT", os.path.join(_REPO, "weights", "LibreFeyNobgl-matte.pt")
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    result = ours.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    ours.eval()

    ref = BiRefNet.from_pretrained(os.environ.get("FEYNOBG_REF_DIR", "feyninc/FeyNobg"))
    ref.eval()
    # The transformers SwinBackbone defaults to SDPA; force the eager (manual
    # matmul + softmax) path so the arithmetic order matches our deterministic
    # port and the fp32 comparison is exact rather than kernel-dependent.
    for mod in ref.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"

    # transformers scales attention scores AFTER the QK matmul; original Swin
    # (and our port) scales Q before it. Mathematically identical, numerically
    # not. Patch the reference to pre-scale so the comparison isolates real
    # porting errors; exact 0 here proves op-for-op equivalence.
    import transformers.models.swin.modeling_swin as hf_swin

    _orig_eager = hf_swin.eager_attention_forward

    def _prescaled_eager(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        if scaling is None:
            scaling = query.size(-1) ** -0.5
        return _orig_eager(
            module, query * scaling, key, value, attention_mask, scaling=1.0, dropout=dropout, **kw
        )

    hf_swin.eager_attention_forward = _prescaled_eager

    torch.manual_seed(0)
    x = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        ref_out = _extract_logit(ref(pixel_values=x))
        our_out = _extract_logit(ours(x))

    diff = (ref_out.float() - our_out.float()).abs().max().item()
    print(f"max_abs_diff = {diff}")
    if diff != 0.0:
        print("FAIL: expected exact fp32 parity for a torch-to-torch port")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
