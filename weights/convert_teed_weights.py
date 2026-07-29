"""Convert a local TEED checkpoint into LibreYOLO format.

Architecture source: https://github.com/xavysp/TEED at commit
40fa4b1391dc6424f88989d0ca75d5b592c8681d (MIT).

The official released checkpoint was trained on BIPED. BIPED's published
dataset terms restrict use to non-commercial purposes, so LibreYOLO does not
host or download that checkpoint. This converter only operates on a local
file supplied by the user and does not change the checkpoint's applicable
terms.

Usage::

    python weights/convert_teed_weights.py upstream.pth \
        weights/LibreTEEDt-edge.pt --verify
"""

from __future__ import annotations

import argparse

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


def _strip_prefixes(state_dict: dict) -> dict:
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        normalized[key] = value
    return normalized


def convert_weights(input_path: str, output_path: str) -> dict:
    """Validate, prefix, and metadata-wrap a TEED tensor dictionary."""
    add_repo_root_to_path()
    from libreyolo.models.edge_common import prefix_upstream_state_dict
    from libreyolo.models.teed.model import LibreTEED

    raw = load_checkpoint(input_path)
    state_dict = _strip_prefixes(extract_state_dict(raw))
    if not LibreTEED.can_load(state_dict):
        raise ValueError(
            "This does not look like a TEED checkpoint (expected the tiny "
            "dense block and DoubleFusion parameter keys)."
        )
    runtime_state = prefix_upstream_state_dict(state_dict)

    target = LibreTEED(None, device="cpu").model.state_dict()
    missing = sorted(set(target) - set(runtime_state))
    unexpected = sorted(set(runtime_state) - set(target))
    if missing or unexpected:
        raise ValueError(
            "TEED checkpoint does not match the runtime architecture: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for key, tensor in target.items():
        if runtime_state[key].shape != tensor.shape:
            raise ValueError(
                f"TEED tensor {key} has shape {tuple(runtime_state[key].shape)}, "
                f"expected {tuple(tensor.shape)}."
            )

    checkpoint = wrap_libreyolo_checkpoint(
        runtime_state,
        model_family="teed",
        size="t",
        task="edge",
        nc=1,
        names={0: "edge"},
        imgsz=352,
    )
    save_checkpoint(checkpoint, output_path)
    print(f"Saved LibreYOLO TEED checkpoint to {output_path}")
    return checkpoint


def verify_conversion(output_path: str) -> None:
    """Load through the public factory and run a dense-output smoke test."""
    add_repo_root_to_path()
    import numpy as np

    from libreyolo import LibreYOLO

    model = LibreYOLO(output_path, device="cpu")
    result = model.predict(
        np.full((48, 64, 3), 127, dtype=np.uint8),
        verbose=False,
    )[0]
    assert result.edges is not None
    assert result.edges.data.shape == (48, 64)
    assert bool(torch.isfinite(result.edges.data).all())
    print("Verified TEED factory load and 48x64 edge output.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input", help="Local upstream TEED checkpoint")
    parser.add_argument("output", help="Output LibreYOLO checkpoint")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output)
    if args.verify:
        verify_conversion(args.output)
