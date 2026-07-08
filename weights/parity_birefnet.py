"""Cross-load parity: LibreBiRefNet vs upstream BiRefNet, max_abs_diff == 0.

This is a torch-to-torch port, so the gate is exact zero at fp32 CPU eval.

Prereqs (throwaway env; do NOT install into the main .venv):
    pip install torch torchvision timm einops kornia safetensors
    git clone https://github.com/ZhengPeng7/BiRefNet
Set:
    BIREFNET_UPSTREAM_DIR = path to the cloned BiRefNet repo
    BIREFNET_CKPT_DIR     = dir with BiRefNet.safetensors + BiRefNet_lite.safetensors

Run:
    python weights/parity_birefnet.py            # both sizes

The upstream config is patched in-memory to disable SDPA (so the manual
attention path is used, matching our deterministic port) and to select the
backbone per size; batch_size stays > 1 so BatchNorm layers are present, as in
the released weights.
"""

from __future__ import annotations

import os
import sys
import types

import torch

_SIZES = {"t": ("swin_v1_t", "BiRefNet_lite"), "l": ("swin_v1_l", "BiRefNet")}


def _patch_upstream_config(repo: str) -> None:
    cfg_path = os.path.join(repo, "config.py")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = f.read()
    if "BB_OVERRIDE" in cfg:
        return
    marker = "self.freeze_bb = 'dino_v3' in self.bb"
    cfg = cfg.replace(
        marker,
        "import os as _os\n        self.bb = _os.environ.get('BB_OVERRIDE', self.bb)\n"
        "        self.SDPA_enabled = False\n        " + marker,
        1,
    )
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(cfg)


def main() -> int:
    repo = os.environ.get("BIREFNET_UPSTREAM_DIR")
    ckpt_dir = os.environ.get("BIREFNET_CKPT_DIR")
    if not repo or not ckpt_dir:
        raise SystemExit("Set BIREFNET_UPSTREAM_DIR and BIREFNET_CKPT_DIR (see module docstring).")

    from safetensors.torch import load_file

    _patch_upstream_config(repo)
    sys.modules.setdefault("dataset", types.ModuleType("dataset"))
    sys.modules["dataset"].class_labels_TR_sorted = tuple()
    os.chdir(repo)
    sys.path.insert(0, repo)

    # Import our port standalone (torch + torchvision only).
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    nn_path = os.path.join(here, "..", "libreyolo", "models", "birefnet", "nn.py")
    spec = importlib.util.spec_from_file_location("_libre_birefnet_nn", nn_path)
    bnn = importlib.util.module_from_spec(spec)
    sys.modules["_libre_birefnet_nn"] = bnn
    spec.loader.exec_module(bnn)

    from models.birefnet import BiRefNet  # upstream

    ok = True
    for size, (bb, ckpt) in _SIZES.items():
        os.environ["BB_OVERRIDE"] = bb
        sd = load_file(os.path.join(ckpt_dir, ckpt + ".safetensors"))

        up = BiRefNet(bb_pretrained=False).eval()
        up.load_state_dict(sd, strict=False)
        ours = bnn.LibreBiRefNetModel(size=size).eval()
        ours.load_state_dict(sd, strict=True)

        torch.manual_seed(0)
        x = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            up_out = up(x)
            up_last = up_out[-1] if isinstance(up_out, (list, tuple)) else up_out
            our_out = ours(x)
        diff = (up_last - our_out).abs().max().item()
        print(f"size={size}: max_abs_diff={diff:.3e} -> {'OK' if diff == 0.0 else 'MISMATCH'}")
        ok = ok and diff == 0.0

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
