"""Convert official MoGe-2 normal weights into LibreYOLO format.

The official checkpoint contains point, normal, mask, and metric-scale heads.
LibreYOLO's ``normal`` family keeps the encoder, shared neck, and native normal
head; this converter removes only unused-head tensors and wraps the remaining
tensors in checkpoint schema v1.0.

Official source:
  repository: https://github.com/microsoft/MoGe
  code commit: 925b8ed835a7a9cdb7578ba15c658a0afc969030 (MIT)
  ViT-S normal weights:
    https://huggingface.co/Ruicheng/moge-2-vits-normal
    revision 679230677b4d282c6f304189a93e98e14f085902 (MIT)
  ViT-B normal weights:
    https://huggingface.co/Ruicheng/moge-2-vitb-normal
    revision 54ad3a693e61907ea4633d13dec6ee682fa09419 (MIT)
  ViT-L normal weights:
    https://huggingface.co/Ruicheng/moge-2-vitl-normal
    revision b135031bae30b5ac2ae141a0e68717795ce38340 (MIT)

Usage::

    python weights/convert_moge2_weights.py model.pt \
        weights/LibreMoGe2l-normal.pt --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

_EMBED_DIM_TO_SIZE = {384: "s", 768: "b", 1024: "l"}
_KEEP_PREFIXES = ("encoder.", "neck.", "normal_head.")
_IMGSZ = 518


def detect_size(state_dict: dict) -> str | None:
    cls_token = state_dict.get("encoder.backbone.cls_token")
    if cls_token is None or getattr(cls_token, "ndim", 0) < 1:
        return None
    return _EMBED_DIM_TO_SIZE.get(int(cls_token.shape[-1]))


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    size: str | None = None,
) -> dict:
    print(f"Loading official MoGe-2 weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw)
    if not isinstance(state_dict, dict):
        raise ValueError("MoGe-2 checkpoint did not contain a tensor state dict.")
    if not (
        "encoder.backbone.cls_token" in state_dict
        and "neck.input_blocks.0.weight" in state_dict
        and "normal_head.output_blocks.4.weight" in state_dict
    ):
        raise ValueError(
            "This does not look like an official MoGe-2 normal checkpoint."
        )

    detected = detect_size(state_dict)
    if size is None:
        size = detected
    elif detected is not None and detected != size:
        raise ValueError(
            f"--size {size!r} contradicts checkpoint width (detected {detected!r})."
        )
    if size is None:
        raise ValueError("Could not detect MoGe-2 size; pass --size explicitly.")

    normal_state = {
        key: value
        for key, value in state_dict.items()
        if key.startswith(_KEEP_PREFIXES)
    }
    removed = len(state_dict) - len(normal_state)
    print(
        f"Keeping {len(normal_state)} encoder/neck/normal tensors; "
        f"dropping {removed} unused-head tensors"
    )

    wrapped = wrap_libreyolo_checkpoint(
        normal_state,
        model_family="moge2",
        size=size,
        nc=1,
        names={0: "normal"},
        task="normal",
        imgsz=_IMGSZ,
    )
    output = Path(output_path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(wrapped, temporary)
    temporary.replace(output)
    print(f"Wrote {output}")
    return wrapped


def verify(output_path: str) -> None:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    model = LibreYOLO(output_path, device="cpu")
    image = np.full((196, 280, 3), 127, dtype=np.uint8)
    result = model.predict(image, verbose=False)
    normal_map = result.normal_map
    assert normal_map is not None
    assert normal_map.data.shape == (196, 280, 3)
    normal_map.assert_normalized(atol=1e-5)
    print(
        f"Verified {output_path}: family={model.family}, size={model.size}, "
        f"task={model.task}, shape={tuple(normal_map.data.shape)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="Official MoGe-2 model.pt")
    parser.add_argument("output", help="Output LibreYOLO .pt checkpoint")
    parser.add_argument(
        "--size",
        choices=["s", "b", "l"],
        default=None,
        help="Size override (normally detected from encoder width).",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output, size=args.size)
    if args.verify:
        verify(args.output)
