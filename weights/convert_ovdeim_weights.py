"""Convert upstream OV-DEIM checkpoints into LibreOVDEIM HF snapshot repos.

This family lives in the open-vocabulary tier (Hugging Face snapshot repos,
not ``.pt`` checkpoint files). Each output directory holds ``config.json``
plus ``model.safetensors`` containing:

* the OV-DEIM detector (``backbone``/``encoder``/``decoder``/``text_adapter``),
  key-stripped from the released EMA state dict (``module.`` prefix removed,
  training-only ``decoder.denoising_class_embed`` dropped);
* the MobileCLIP-B(LT) text tower under ``text_encoder.*``, taken unchanged
  from Apple's open_clip release (``apple/MobileCLIP-B-LT-OpenCLIP``).

Learned parameters are unchanged; the conversion is a key remap plus merge.

Usage:
    python weights/convert_ovdeim_weights.py <upstream_dir> <output_dir> --size s

where ``<upstream_dir>`` contains ``ovdeim_{s,m,l}.pth`` and ``<output_dir>``
receives ``LibreOVDEIM<size>/{config.json,model.safetensors}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _conversion_utils import add_repo_root_to_path

MOBILECLIP_HF_REPO = "apple/MobileCLIP-B-LT-OpenCLIP"
UPSTREAM_REPO = "https://github.com/wleilei/OV-DEIM"
UPSTREAM_COMMIT = "dfbf394672407b7f837ec08e7d68e8127548b254"


def load_text_tower_state_dict() -> dict[str, torch.Tensor]:
    """Text-tower weights from Apple's open_clip checkpoint (key strip only)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(MOBILECLIP_HF_REPO, "open_clip_model.safetensors")
    full = load_file(path)
    return {
        f"text_encoder.{k[len('text.'):]}": v
        for k, v in full.items()
        if k.startswith("text.")
    }


def convert(input_dir: str, output_dir: str, size: str) -> Path:
    add_repo_root_to_path()
    from libreyolo.models.openvocab.ov_deim import LibreOVDEIM
    from libreyolo.models.openvocab.ovdeim import LibreOVDEIMModel

    from safetensors.torch import save_file

    ckpt_path = Path(input_dir) / f"ovdeim_{size}.pth"
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "ema_state_dict" in raw:
        raw = raw["ema_state_dict"]
        if "module" in raw:
            raw = raw["module"]
    state_dict = {}
    for key, value in raw.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("module."):
            key = key[len("module."):]
        if "decoder.denoising_class_embed" in key:
            continue  # training-only, skipped by upstream's own eval script
        state_dict[key] = value
    print(f"{ckpt_path.name}: {len(state_dict)} detector tensors")

    text_sd = load_text_tower_state_dict()
    print(f"{MOBILECLIP_HF_REPO}: {len(text_sd)} text-tower tensors")
    state_dict.update(text_sd)

    model = LibreOVDEIMModel(size)
    model.load_state_dict(state_dict, strict=True)
    print("strict load into LibreOVDEIMModel: OK")

    repo_name = f"{LibreOVDEIM.FILENAME_PREFIX}{size}"
    out = Path(output_dir) / repo_name
    out.mkdir(parents=True, exist_ok=True)

    tmp = out / "model.safetensors.tmp"
    save_file(state_dict, str(tmp))
    tmp.rename(out / "model.safetensors")

    config = {
        "model_type": "ov_deim",
        "family": "ov_deim",
        "size": size,
        "input_size": 640,
        "text_encoder": "MobileCLIP-B(LT) text tower",
        "source": {
            "detector": {"repo": UPSTREAM_REPO, "commit": UPSTREAM_COMMIT},
            "text_encoder": {"repo": f"https://huggingface.co/{MOBILECLIP_HF_REPO}"},
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing ovdeim_{s,m,l}.pth")
    parser.add_argument("output_dir", help="Directory receiving LibreOVDEIM<size>/")
    parser.add_argument("--size", required=True, choices=["s", "m", "l"])
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir, args.size)
