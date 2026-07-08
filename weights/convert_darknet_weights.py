"""Convert Darknet ``.weights`` files (YOLOv2/v3/v4) to LibreYOLO checkpoints.

The YOLOv2/v3/v4 architectures and their pretrained weights are public domain
(pjreddie/darknet, AlexeyAB/darknet — the "YOLO LICENSE"). This script builds
the matching LibreYOLO network from the bundled cfg, loads the ``.weights``
binary into it, and writes a metadata-wrapped LibreYOLO v1.0 checkpoint.

Because LibreYOLO bundles the cfgs, you normally only pass ``--family`` and
``--size``; the cfg is resolved automatically.

Usage::

    # Download the public-domain weights first, e.g.:
    #   yolov3:  https://pjreddie.com/media/files/yolov3.weights
    #   yolov4:  https://github.com/AlexeyAB/darknet/releases/download/\
    #            darknet_yolo_v3_optimal/yolov4.weights
    python weights/convert_darknet_weights.py \
        --weights yolov3.weights --family yolo3 --size b \
        --out weights/LibreYOLO3b.pt

    # tiny / spp variants:
    python weights/convert_darknet_weights.py \
        --weights yolov3-tiny.weights --family yolo3 --size t \
        --out weights/LibreYOLO3t.pt

A hard correctness check runs during conversion: the ``.weights`` file must be
consumed exactly (byte-for-byte). A mismatch means the cfg/size does not match
the weights file and the conversion is aborted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    build_class_names,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

add_repo_root_to_path()

from libreyolo.models.darknet import (  # noqa: E402
    build_darknet,
    load_darknet_weights_file,
    parse_cfg_file,
)
from libreyolo.models.darknet.net import DarknetNet  # noqa: E402
from libreyolo.models.yolo1.model import VOC_NAMES  # noqa: E402


def _build_net(cfg_name_or_path: str, num_classes: int | None) -> DarknetNet:
    """Build a network from a bundled cfg name or a cfg file path."""
    if Path(cfg_name_or_path).exists():
        net_options, layers = parse_cfg_file(cfg_name_or_path)
        return DarknetNet(net_options, layers, num_classes=num_classes)
    return build_darknet(cfg_name_or_path, num_classes=num_classes)

# family -> {size -> bundled cfg name}, kept in sync with the family classes.
_FAMILY_CFGS = {
    "yolo1": {"t": "yolov1-tiny", "b": "yolov1"},
    "yolo2": {"t": "yolov2-tiny", "b": "yolov2"},
    "yolo3": {"t": "yolov3-tiny", "b": "yolov3", "spp": "yolov3-spp"},
    "yolo4": {"t": "yolov4-tiny", "b": "yolov4"},
}

# YOLOv1 is trained on Pascal VOC (20 classes), not COCO, so its checkpoints
# carry the VOC label map rather than the generic ``build_class_names`` output.
_FAMILY_NAMES = {
    "yolo1": {index: name for index, name in enumerate(VOC_NAMES)},
}


def convert_weights(
    weights_path: str,
    family: str,
    size: str,
    output_path: str,
    cfg_override: str | None = None,
    num_classes: int | None = None,
) -> dict:
    """Convert one Darknet ``.weights`` file into a LibreYOLO checkpoint on disk."""
    if family not in _FAMILY_CFGS:
        raise ValueError(f"Unknown family {family!r}. Choose from {list(_FAMILY_CFGS)}.")
    sizes = _FAMILY_CFGS[family]
    if size not in sizes and cfg_override is None:
        raise ValueError(f"Unknown size {size!r} for {family}. Choose from {list(sizes)}.")

    cfg_name = cfg_override if cfg_override is not None else sizes[size]
    net = _build_net(cfg_name, num_classes=num_classes).eval()

    load_darknet_weights_file(net, weights_path)  # asserts exact byte consumption

    nc = net.detection_specs()[0].num_classes
    imgsz = net.input_width

    # Prefix keys so the checkpoint carries a stable per-family namespace,
    # matching the DarknetFamily holder module.
    state_dict = {f"{family}.{k}": v for k, v in net.state_dict().items()}

    family_names = _FAMILY_NAMES.get(family)
    if family_names is not None and len(family_names) == nc:
        names = family_names
    else:
        names = build_class_names(nc)
    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family=family,
        size=size,
        nc=nc,
        names=names,
        task="detect",
        imgsz=imgsz,
    )
    save_checkpoint(checkpoint, output_path)
    print(
        f"Converted {Path(weights_path).name} -> {output_path}\n"
        f"  family={family} size={size} cfg={cfg_name} nc={nc} imgsz={imgsz} "
        f"params={len(state_dict)} tensors"
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to Darknet .weights")
    parser.add_argument(
        "--family", required=True, choices=list(_FAMILY_CFGS), help="LibreYOLO family"
    )
    parser.add_argument("--size", required=True, help="Size code (t/b/spp)")
    parser.add_argument("--out", required=True, help="Output .pt path")
    parser.add_argument(
        "--cfg",
        default=None,
        help="Override bundled cfg (bundled name or path). Rarely needed.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=None,
        help="Override class count (defaults to the value in the cfg).",
    )
    args = parser.parse_args()
    convert_weights(
        weights_path=args.weights,
        family=args.family,
        size=args.size,
        output_path=args.out,
        cfg_override=args.cfg,
        num_classes=args.classes,
    )


if __name__ == "__main__":
    main()
