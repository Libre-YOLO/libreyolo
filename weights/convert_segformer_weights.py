"""Convert upstream SegFormer ADE20K weights (b0-b5) into LibreYOLO format.

The upstream checkpoints are published by NVIDIA under the NVIDIA Source Code
License. That license permits redistribution of the work and of derivative
works, provided the license travels with them and attribution notices are kept
intact, but it limits *use* to non-commercial research or evaluation. The
converted checkpoints therefore inherit the non-commercial restriction: they
are NOT covered by LibreYOLO's permissive license, and LibreSegformer prints a
notice to that effect before every auto-download.

Conversion is a state-dict key remapping only (the upstream ``segformer.``
encoder prefix becomes ``encoder.``); learned parameters are unchanged, and the
result reproduces upstream logits bit-exactly.

Usage:
    python weights/convert_segformer_weights.py --size b0
    python weights/convert_segformer_weights.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

# b0-b4 were fine-tuned on ADE20K at 512; b5 is the one trained at 640.
HF_REPOS: dict[str, str] = {
    "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    "b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    "b5": "nvidia/segformer-b5-finetuned-ade-640-640",
}
IMGSZ: dict[str, int] = {**{s: 512 for s in HF_REPOS}, "b5": 640}
NC = 150  # ADE20K scene-parsing classes


def convert(size: str, output_dir: Path) -> Path:
    add_repo_root_to_path()
    from libreyolo.models.segformer.model import LibreSegformer
    from libreyolo.models.segformer.nn import LibreSegformerNet

    from transformers import SegformerForSemanticSegmentation

    repo = HF_REPOS[size]
    print(f"[{size}] loading {repo}")
    hf = SegformerForSemanticSegmentation.from_pretrained(repo)
    names = {int(i): str(n) for i, n in hf.config.id2label.items()}
    if len(names) != NC:
        raise RuntimeError(f"{repo} has {len(names)} classes, expected {NC}")

    state = LibreSegformer.convert_upstream_state_dict(extract_state_dict(hf.state_dict()))
    if state is None:
        raise RuntimeError(f"{repo} is not a recognizable upstream SegFormer checkpoint")

    # A key remap that silently drops or renames a tensor is the classic way to
    # ship a subtly broken checkpoint, so load it strictly and check parity.
    net = LibreSegformerNet(size=size, num_classes=NC).eval()
    net.load_state_dict(state, strict=True)
    _assert_parity(hf.eval(), net, size)

    checkpoint = wrap_libreyolo_checkpoint(
        state,
        model_family="segformer",
        size=size,
        task="semantic",
        nc=NC,
        names=names,
        imgsz=IMGSZ[size],
    )
    out = save_checkpoint(checkpoint, output_dir / f"LibreSegformer{size}-sem.pt")
    print(f"[{size}] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def _assert_parity(hf, net, size: str) -> None:
    """Upstream and converted model must agree on the same input."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    torch.manual_seed(0)
    x = torch.rand(1, 3, IMGSZ[size], IMGSZ[size])
    with torch.no_grad():
        upstream = hf(pixel_values=(x - mean) / std).logits
        ours = net.decode_head(net.encoder((x - mean) / std))
    diff = (upstream - ours).abs().max().item()
    if diff > 1e-4:
        raise RuntimeError(f"[{size}] parity check FAILED: max|diff|={diff:.3e}")
    print(f"[{size}] parity OK: max|logit diff| = {diff:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=sorted(HF_REPOS), help="single size to convert")
    parser.add_argument("--all", action="store_true", help="convert every size (b0-b5)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    if not args.all and not args.size:
        parser.error("pass --size <b0..b5> or --all")
    sizes = sorted(HF_REPOS) if args.all else [args.size]
    for size in sizes:
        convert(size, args.output_dir)


if __name__ == "__main__":
    main()
