"""Convert the official RF-DETR keypoint-preview weights into LibreYOLO format.

The released ``rf-detr-keypoint-preview`` checkpoint (ships as
``{"model": state_dict}`` with 740 tensors) is an xlarge-encoder GroupPose
model. Architecture, postprocess, and the GroupPose head are ported/adapted
from RF-DETR v1.8.0 and live under ``libreyolo/models/rfdetr``; this script only
wraps the upstream state dict in the LibreYOLO checkpoint schema so the unified
loader (``LibreRFDETR``) can rebuild the architecture and load the weights.

Two upstream quirks are handled here:

* The checkpoint carries four vestigial ``keypoint_head.keypoint_proj.*``
  tensors that the LibreYOLO GroupPose graph does not build (its keypoint
  parameters live under ``transformer.*keypoint*``). They are dropped so the
  load is 0-missing / 0-unexpected.
* The detection head has ``out_features == 2`` (background + person), so the
  contiguous-class interface exposes ``nc=1`` with a single ``person`` class.

Usage::

    python weights/convert_rfdetr_keypoint_weights.py \
        downloads/rf-detr-keypoint-preview-xlarge.pth \
        downloads/LibreRFDETRx-pose.pt

Add ``--verify`` to load the converted weights through the normal LibreYOLO
API and run a smoke inference, confirming round-trip integrity.

Ported from / adapted from RF-DETR v1.8.0 (Apache-2.0).
"""

from __future__ import annotations

import argparse

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
)

# Vestigial GroupPose projection that the LibreYOLO graph does not build.
# Dropping these four tensors makes the converted checkpoint load cleanly
# (0-missing / 0-unexpected) into the ported architecture.
_VESTIGIAL_PREFIX = "keypoint_head.keypoint_proj."

def convert_weights(
    input_path: str,
    output_path: str,
    *,
    size: str = "x",
    num_keypoints: int = 17,
) -> dict:
    print(f"Loading upstream RF-DETR keypoint weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw)
    print(f"Found {len(state_dict)} parameter entries")

    dropped = [k for k in state_dict if k.startswith(_VESTIGIAL_PREFIX)]
    state_dict = {
        k: v for k, v in state_dict.items() if not k.startswith(_VESTIGIAL_PREFIX)
    }
    print(f"Dropped {len(dropped)} vestigial {_VESTIGIAL_PREFIX}* tensors")

    # GroupPose schema: class 0 carries no keypoints, the person class carries
    # ``num_keypoints``. The detection head therefore has out_features == 2.
    num_keypoints_per_class = [0, int(num_keypoints)]

    # Call the library helper directly so the keypoint extra-metadata fields
    # (num_keypoints, keypoint_dim, oks_sigmas, num_keypoints_per_class) flow
    # through ``**extra_metadata`` into the checkpoint.
    add_repo_root_to_path()
    from libreyolo.data import default_oks_sigmas
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="rfdetr",
        size=size,
        task="pose",
        nc=1,
        names={0: "person"},
        imgsz=576,
        num_keypoints=int(num_keypoints),
        keypoint_dim=3,
        oks_sigmas=list(default_oks_sigmas(int(num_keypoints))),
        num_keypoints_per_class=num_keypoints_per_class,
    )

    save_checkpoint(libreyolo_ckpt, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}")
    return libreyolo_ckpt


def verify_conversion(converted_path: str) -> bool:
    """Load through the normal LibreYOLO API and run a smoke forward pass."""
    add_repo_root_to_path()
    from libreyolo import LibreRFDETR

    print(f"\nLoading converted weights via LibreRFDETR({converted_path})...")
    m = LibreRFDETR(converted_path, device="cpu")
    print(f"  family={m.FAMILY} size={m.size} task={m.task} nc={m.nb_classes} "
          f"num_keypoints={m.num_keypoints} names={m.names}")

    m.model.model.eval()
    with torch.no_grad():
        out = m.model(torch.zeros(1, 3, 576, 576))
    if isinstance(out, dict):
        keys = sorted(out.keys())
    else:
        keys = [type(out).__name__]
    print(f"  forward pass OK - output keys/type: {keys}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RF-DETR keypoint-preview weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream RF-DETR keypoint checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument(
        "--size",
        default="x",
        help="RF-DETR pose size config key (default: x)",
    )
    parser.add_argument(
        "--num-keypoints",
        type=int,
        default=17,
        help="Keypoints per person (default: 17, COCO)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify round-trip after conversion"
    )
    args = parser.parse_args()

    convert_weights(
        args.input,
        args.output,
        size=args.size,
        num_keypoints=args.num_keypoints,
    )
    if args.verify:
        verify_conversion(args.output)
