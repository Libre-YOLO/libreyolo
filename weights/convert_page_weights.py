"""Convert upstream PaGE weights to LibreYOLO format.

The released PaGE checkpoints (https://huggingface.co/Octopus1/page-*)
are self-contained safetensors: the DINOv3 tower weights ship alongside
the gaze-target decoder. Conversion is a metadata wrap plus one syntactic
normalization — the transformers-5.x nested DINOv3 layer naming
(``<branch>.model.model.layer.N``) is flattened to the canonical
``<branch>.model.layer.N`` (see ``libreyolo/models/page/convert.py``).

Weight licensing: decoder weights are MIT; the DINOv3 towers are
derivatives of Meta's DINOv3 and remain governed by the DINOv3 License
(redistribution must include the license text).

Usage::

    python weights/convert_page_weights.py s weights/LibrePAGEs-gazetarget.pt
    python weights/convert_page_weights.py path/to/model.safetensors weights/LibrePAGEb-gazetarget.pt --size b

The first argument is a size code (downloads the matching Octopus1 repo
from Hugging Face) or a local ``.safetensors`` path. Add ``--verify`` to
load the converted file through the normal LibreYOLO API and run a smoke
forward pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _conversion_utils import save_checkpoint, wrap_libreyolo_checkpoint

SIZE_TO_REPO = {
    "s": "Octopus1/page-vits",
    "sp": "Octopus1/page-vitsplus",
    "b": "Octopus1/page-vitb",
    "hp": "Octopus1/page-vithplus",
}


def _load_upstream_state(source: str, size: str | None):
    """Return (state_dict, size) from a local safetensors file or a size code."""
    from safetensors.torch import load_file

    if source in SIZE_TO_REPO:
        from huggingface_hub import hf_hub_download

        size = source
        repo = SIZE_TO_REPO[source]
        print(f"Downloading {repo} ...")
        # The released repos ship a single-file model.safetensors; the largest
        # (hp) is sharded, so consult the index when present.
        try:
            import json

            idx_path = hf_hub_download(repo, "model.safetensors.index.json")
            with open(idx_path) as fh:
                files = sorted(set(json.load(fh)["weight_map"].values()))
        except Exception:
            files = ["model.safetensors"]
        state = {}
        for f in files:
            state.update(load_file(hf_hub_download(repo, f)))
        return state, size
    if size is None:
        raise SystemExit("--size is required when converting a local file")
    return load_file(source), size


def convert(source: str, output_path: str, size: str | None = None, verify: bool = False) -> None:
    state_dict, size = _load_upstream_state(source, size)
    print(f"Loaded {len(state_dict)} parameter entries (size={size})")

    import sys

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from libreyolo.models.page.convert import convert_upstream, is_upstream_state_dict

    if is_upstream_state_dict(state_dict):
        state_dict = convert_upstream(state_dict)
        print("Normalized nested DINOv3 layer naming to canonical flat form")

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="page",
        size=size,
        nc=1,
        names={0: "person"},
        task="gazetarget",
        supported_tasks=("gazetarget",),
        default_task="gazetarget",
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(out)
    print(f"Wrote {out}")

    if verify:
        from libreyolo import LibreYOLO

        model = LibreYOLO(str(out))
        assert model.family == "page", model.family
        assert model.size == size, model.size
        import torch

        device = model.device
        with torch.no_grad():
            heatmap, inout = model.model(
                torch.zeros(1, 3, 512, 512, device=device),
                torch.zeros(1, 3, 256, 256, device=device),
                torch.tensor([[8.0, 8.0, 16.0, 16.0]], device=device),
            )
        assert tuple(heatmap.shape) == (1, 64, 64), heatmap.shape
        assert tuple(inout.shape) == (1,), inout.shape
        print("Verified: LibreYOLO loads the checkpoint and the forward pass runs")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", help="size code (s/sp/b/hp, downloads from HF) or local .safetensors path")
    p.add_argument("output")
    p.add_argument("--size", choices=list(SIZE_TO_REPO), default=None)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    convert(args.source, args.output, args.size, args.verify)
