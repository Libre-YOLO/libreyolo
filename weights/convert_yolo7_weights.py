"""Convert MultimediaTechLab/YOLO v7 weights (MIT) to LibreYOLO format.

The upstream ``v7.pt`` (MIT, from the MMT release) is a flat state dict with
numbered layer keys (``0.conv.weight`` ... ``105.heads.0.head_conv.weight``).
The native LibreYOLO7 port mirrors those module names under a ``layers.`` prefix,
so conversion is a metadata-wrap with a one-token prefix add (no structural
remap). Verified: v7.pt strict-loads with 0 missing / 0 unexpected keys.

Usage::

    # download v7.pt from the MMT release, then:
    python weights/convert_yolo7_weights.py v7.pt weights/LibreYOLO7b.pt --size b
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    build_class_names,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

add_repo_root_to_path()


def convert(input_path: str, output_path: str, size: str = "b") -> dict:
    raw = load_checkpoint(input_path)
    upstream = extract_state_dict(raw, state_dict_keys=("model", "state_dict"),
                                  prefer_ema=True)
    # prefix upstream flat numbered keys to match the native port's ModuleList
    state_dict = {f"layers.{k}": v for k, v in upstream.items()}

    # nc from the head conv (anchors * (5 + nc))
    nc = 80
    for k, v in state_dict.items():
        if k.endswith("heads.0.head_conv.weight"):
            nc = int(v.shape[0]) // 3 - 5
            break

    # dry-load into a fresh model to surface any key diff
    from libreyolo.models.yolo7.net import YOLOv7Model

    model = YOLOv7Model(num_classes=nc)
    res = model.load_state_dict(state_dict, strict=False)
    print(f"Extracted {len(state_dict)} tensors; nc={nc}")
    print(f"  missing: {len(res.missing_keys)} unexpected: {len(res.unexpected_keys)}")

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="yolo7",
        size=size,
        nc=nc,
        names=build_class_names(nc),
        task="detect",
        imgsz=640,
    )
    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.rename(out)  # atomic
    print(f"Wrote {out}")
    return wrapped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Upstream v7.pt")
    p.add_argument("output", help="Output LibreYOLO7b.pt")
    p.add_argument("--size", default="b", choices=["b"])
    args = p.parse_args()
    convert(args.input, args.output, args.size)


if __name__ == "__main__":
    main()
