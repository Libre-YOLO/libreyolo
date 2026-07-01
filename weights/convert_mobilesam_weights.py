"""Convert the upstream MobileSAM checkpoint to LibreMobileSAM.pt.

Usage:
    python weights/convert_mobilesam_weights.py \
        --input /path/to/mobile_sam.pt \
        --output weights/LibreMobileSAM.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def convert(input_path: str | Path, output_path: str | Path) -> Path:
    add_repo_root_to_path()
    from libreyolo.models.mobilesam.model import MobileSAMNetwork

    checkpoint = load_checkpoint(input_path)
    state_dict = _unwrap_state_dict(checkpoint)
    model = MobileSAMNetwork()
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "MobileSAM strict load failed: "
            f"missing={result.missing_keys[:10]}, "
            f"unexpected={result.unexpected_keys[:10]}"
        )
    model.eval()
    wrapped = wrap_libreyolo_checkpoint(
        model.state_dict(),
        model_family="mobilesam",
        size="tiny",
        task="segment",
        nc=1,
        names={0: "object"},
        imgsz=1024,
    )

    output_path = Path(output_path)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(output_path)
    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to mobile_sam.pt")
    parser.add_argument(
        "--output",
        default="weights/LibreMobileSAM.pt",
        help="Output checkpoint path",
    )
    args = parser.parse_args()
    convert(args.input, args.output)
