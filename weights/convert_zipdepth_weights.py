"""Convert official ZipDepth weights into LibreYOLO format.

The released checkpoints (https://github.com/fabiotosi92/ZipDepth, MIT) ship as
bare unfused state dicts under ``checkpoints/`` with keys ``encoder.*`` /
``decoder.*`` plus the ImageNet ``mean``/``std`` buffers. The architecture is
ported at ``libreyolo/models/zipdepth/nn.py`` with identical key names, so this
script only strips DDP/torch.compile prefixes and wraps the state dict in the
LibreYOLO v1.0 checkpoint schema; tensors are unchanged and the load is
0-missing / 0-unexpected.

The GPU checkpoint (``zipdepth_base.pth``, ``decoder.convex_up.mask_pred.*``)
and the NPU checkpoint (``zipdepth_base_npu.pth``,
``decoder.convex_up.where_conv.*``) are separately trained weight sets; the
size code (``b`` vs ``bnpu``) is auto-detected from those keys.

Usage::

    python weights/convert_zipdepth_weights.py \
        ZipDepth/checkpoints/zipdepth_base.pth weights/LibreZipDepthb-depth.pt

Add ``--verify`` to load the converted file through the normal LibreYOLO API
and run a smoke forward pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

_IMGSZ = 384

# Upstream stem width (dims[0] // 2) -> capacity tier. Only 'base' has released
# checkpoints today.
_STEM_CH_TO_SIZE = {24: "b"}


def _strip_wrapper_prefixes(state_dict: dict) -> dict:
    """Strip DDP ('module.') and torch.compile ('_orig_mod.') key prefixes."""
    prefixes = ("module.", "_orig_mod.")

    def _clean(key: str) -> str:
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if key.startswith(p):
                    key = key[len(p) :]
                    changed = True
        return key

    return {_clean(k): v for k, v in state_dict.items()}


def detect_size(state_dict: dict) -> str | None:
    """Infer the size code from the stem width and the upsampling-head flavor."""
    stem = state_dict.get("encoder.stem_half.conv.weight")
    if stem is None or getattr(stem, "ndim", 0) != 4:
        return None
    size = _STEM_CH_TO_SIZE.get(int(stem.shape[0]))
    if size is None:
        return None
    if any(k.startswith("decoder.convex_up.where_conv.") for k in state_dict):
        return f"{size}npu"
    return size


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    size: str | None = None,
) -> dict:
    print(f"Loading upstream ZipDepth weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = _strip_wrapper_prefixes(extract_state_dict(raw))
    print(f"Found {len(state_dict)} parameter entries")

    if any(".fused_conv." in k for k in state_dict):
        raise ValueError(
            "This checkpoint is already fused (reparameterized). LibreYOLO "
            "stores the unfused training-form weights and fuses at load time; "
            "convert the unfused upstream checkpoint instead."
        )
    if not any(k.startswith("encoder.stem_half.") for k in state_dict) or not any(
        k.startswith("decoder.convex_up.") for k in state_dict
    ):
        raise ValueError(
            "This does not look like a ZipDepth checkpoint (expected "
            "'encoder.stem_half.*' stem and 'decoder.convex_up.*' head keys)."
        )

    detected = detect_size(state_dict)
    if size is None:
        size = detected
        if size is None:
            raise ValueError(
                "Could not auto-detect the ZipDepth size; pass --size explicitly."
            )
        print(f"Auto-detected size: {size}")
    elif detected is not None and detected != size:
        raise ValueError(
            f"--size {size} contradicts the checkpoint (detected {detected!r}: "
            "the GPU/NPU upsampling-head flavor is part of the size code)."
        )

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="zipdepth",
        size=size,
        nc=1,
        names={0: "depth"},
        task="depth",
        imgsz=_IMGSZ,
    )

    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.rename(out)
    print(f"Wrote {out}")
    return wrapped


def verify(output_path: str) -> None:
    """Load the converted checkpoint via the public API and smoke-forward."""
    add_repo_root_to_path()
    import numpy as np

    from libreyolo import LibreYOLO

    model = LibreYOLO(output_path)
    dummy = np.full((240, 320, 3), 127, dtype=np.uint8)
    results = model.predict(dummy, verbose=False)
    depth = results[0].depth_map
    assert depth is not None, "predict() returned no depth_map"
    assert depth.data.shape == (240, 320), f"bad depth shape {tuple(depth.data.shape)}"
    print(
        f"Verified {output_path}: size={model.size}, "
        f"depth range [{float(depth.data.min()):.4f}, {float(depth.data.max()):.4f}]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input", help="Upstream .pth checkpoint")
    parser.add_argument("output", help="Output LibreYOLO .pt path")
    parser.add_argument(
        "--size",
        default=None,
        choices=["b", "bnpu"],
        help="Size code override (auto-detected from the state dict by default)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Load the converted checkpoint through LibreYOLO and smoke-forward",
    )
    args = parser.parse_args()
    convert_weights(args.input, args.output, size=args.size)
    if args.verify:
        verify(args.output)
