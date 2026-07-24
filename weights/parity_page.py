"""Cross-check LibrePAGE against the upstream PaGE HF checkpoints.

For each requested size, loads the upstream model via transformers
``trust_remote_code`` (modeling code from https://huggingface.co/Octopus1/PaGE,
MIT) and the converted LibreYOLO checkpoint, feeds identical inputs
through both on CPU fp32, and asserts the sigmoid heatmap and in/out
outputs match exactly (``max_abs_diff == 0``).

Usage::

    python weights/parity_page.py s [sp b hp]

Requires the converted ``weights/LibrePAGE<size>-gazetarget.pt`` files
(see ``weights/convert_page_weights.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Weight repos pinned to the revisions this port was parity-verified against
# (2026-07-24). trust_remote_code executes the MIT modeling file referenced by
# these snapshots (Octopus1/PaGE, reviewed at 7147c67c); pinning the weights
# revision keeps later upstream pushes from silently changing what runs here.
SIZE_TO_REPO = {
    "s": ("Octopus1/page-vits", "aa8ddb4e5251b27e63a5365dc479b765c11c7296"),
    "sp": ("Octopus1/page-vitsplus", "6f4a84a99d7be270bb1c72282e3c87d79cb744b1"),
    "b": ("Octopus1/page-vitb", "b991d8bcb2890a97db611a2080ca250ecd943952"),
    "hp": ("Octopus1/page-vithplus", None),  # not yet converted (disk); pin on first run
}

BBOXES = [(0.30, 0.12, 0.48, 0.40), (0.60, 0.20, 0.75, 0.45)]


def check_size(size: str) -> None:
    from transformers import AutoModel

    from libreyolo import LibreYOLO
    from libreyolo.models.page.utils import head_rects_grid

    torch.manual_seed(0)
    scene = torch.randn(1, 3, 512, 512)
    heads = torch.randn(len(BBOXES), 3, 256, 256)

    repo, revision = SIZE_TO_REPO[size]
    upstream = (
        AutoModel.from_pretrained(repo, revision=revision, trust_remote_code=True)
        .eval()
        .cpu()
    )
    with torch.inference_mode():
        up = upstream({"images": scene, "head_images": [heads], "bboxes": [list(BBOXES)]})
    up_heatmap = up["heatmap"][0]
    up_inout = up["inout"][0]

    ours = LibreYOLO(f"weights/LibrePAGE{size}-gazetarget.pt", device="cpu")
    rects = head_rects_grid(BBOXES)
    with torch.inference_mode():
        heatmap_logits, inout_logits = ours.model(scene, heads, rects)
    our_heatmap = torch.sigmoid(heatmap_logits)
    our_inout = torch.sigmoid(inout_logits)

    hm_diff = (up_heatmap - our_heatmap).abs().max().item()
    io_diff = (up_inout - our_inout).abs().max().item()
    print(f"size={size}: heatmap max_abs_diff={hm_diff}, inout max_abs_diff={io_diff}")
    assert hm_diff == 0.0, f"size={size} heatmap mismatch: {hm_diff}"
    assert io_diff == 0.0, f"size={size} inout mismatch: {io_diff}"
    print(f"size={size}: OK")


if __name__ == "__main__":
    sizes = sys.argv[1:] or ["s"]
    for size in sizes:
        check_size(size)
