"""Inference-parity proof for the Darknet families (YOLOv2/v3/v4).

Per the libreyolo-port-model skill (§12), this proves the native port produces
the same outputs as an upstream reference on identical input, BEFORE trusting
the decode/postprocess. Darknet's reference runtime is C, so we use OpenCV's
Darknet importer (`cv2.dnn.readNetFromDarknet`, Apache-2.0) as the oracle:
same architecture, same `.weights`, same input blob → compare the raw head
feature maps.

YOLOv1 (yolo1) is deliberately NOT covered here: OpenCV's Darknet importer does
not support the `[connected]` / `[local]` / `[detection]` layers YOLOv1 uses, so
it is not a valid oracle for it. YOLOv1 faithfulness rests instead on (a) the
byte-exact `.weights` reader assertion (the built net must consume yolov1.weights
to the last byte) and (b) the classic dog/bicycle/car golden fixture on the real
converted weights (tests/unit/test_yolo1.py::test_yolo1b_golden_dog_bicycle_car,
gated on LIBREYOLO1B_CKPT).

This is a gated one-off, NOT a PR-gate unit test: it needs real (public-domain)
`.weights` files and OpenCV. Point it at a directory of Darknet weights:

    export DARKNET_WEIGHTS_DIR=/path/with/yolov3-tiny.weights,yolov3.weights,...
    python weights/parity_darknet.py

Confirmed result (2026-07-04, OpenCV 4.13): yolov3-tiny raw-head
max_abs_diff = 1.79e-04 vs the oracle on an identical blob — cross-framework
FP noise, not a structural difference. A wrong BatchNorm epsilon convention
(darknet uses `sqrt(var)+1e-6`, not torch's `sqrt(var+eps)`) would blow this
up and grow with depth, so this is the load-bearing check.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from _conversion_utils import add_repo_root_to_path

add_repo_root_to_path()

# (family, size, bundled cfg name, expected .weights filename)
_CASES = [
    ("yolo3", "t", "yolov3-tiny", "yolov3-tiny.weights"),
    ("yolo3", "b", "yolov3", "yolov3.weights"),
    ("yolo3", "spp", "yolov3-spp", "yolov3-spp.weights"),
    ("yolo4", "t", "yolov4-tiny", "yolov4-tiny.weights"),
    ("yolo4", "b", "yolov4", "yolov4.weights"),
    ("yolo2", "t", "yolov2-tiny", "yolov2-tiny.weights"),
    ("yolo2", "b", "yolov2", "yolov2.weights"),
]

# Cross-framework (torch vs OpenCV conv/BN) FP tolerance on raw head maps.
# Deeper nets accumulate more torch-vs-OpenCV fp drift (yolov3/spp land ~3-4e-3),
# so the bar is 1e-2 — still ~100x tighter than any real structural mismatch.
_TOL = 1e-2

# OpenCV's Darknet ``Reorg`` layer does not reproduce Darknet's reorg channel
# ordering (a long-standing OpenCV-specific quirk), so it is not a valid oracle
# for the YOLOv2 passthrough. yolov2 (full) is reported but not gated here;
# verify it against compiled Darknet instead. (yolov2-tiny has no reorg.)
_ORACLE_CAVEAT = {"yolov2"}


def _run() -> int:
    weights_dir = os.environ.get("DARKNET_WEIGHTS_DIR")
    if not weights_dir:
        raise SystemExit(
            "Set DARKNET_WEIGHTS_DIR to a directory containing public-domain "
            "Darknet .weights files (e.g. yolov3-tiny.weights)."
        )
    try:
        import cv2
    except ImportError:
        raise SystemExit("OpenCV (cv2) is required for the parity oracle.")

    import torch

    from libreyolo.models.darknet import build_darknet, load_darknet_weights_file
    from libreyolo.models.darknet.cfg import bundled_cfg_path
    from libreyolo.models.darknet.preprocess import preprocess_numpy

    weights_dir = Path(weights_dir)
    # Fixed pseudo-random image so the run is deterministic without a dataset.
    rng = np.random.default_rng(0)

    worst_overall = 0.0
    ran = 0
    for family, size, cfg_name, wname in _CASES:
        wpath = weights_dir / wname
        if not wpath.exists():
            print(f"skip {cfg_name}: {wname} not found")
            continue
        ran += 1

        net = build_darknet(cfg_name, num_classes=80).eval()
        load_darknet_weights_file(net, wpath)  # asserts exact byte consumption

        insize = net.input_width
        img = rng.integers(0, 256, (insize + 37, insize + 91, 3), dtype=np.uint8)
        chw, _ = preprocess_numpy(img, insize)
        blob = chw[None].astype(np.float32)

        with torch.no_grad():
            my_outs = net(torch.from_numpy(blob))
        mine = {tuple(o.shape[1:]): o[0].numpy() for o in my_outs}  # (C,H,W)->arr

        ocv = cv2.dnn.readNetFromDarknet(str(bundled_cfg_path(cfg_name)), str(wpath))
        ocv.setInput(blob)
        names = ocv.getLayerNames()
        outs = ocv.forward(names)
        head_channels = {c for (c, _, _) in mine}
        ocv_heads = {}
        for name, out in zip(names, outs):
            a = np.asarray(out)
            if a.ndim == 4 and a.shape[1] in head_channels:
                ocv_heads[tuple(a.shape[1:])] = a[0]

        worst = 0.0
        for chw_key, my_arr in mine.items():
            if chw_key in ocv_heads:
                d = float(np.abs(my_arr - ocv_heads[chw_key]).max())
                worst = max(worst, d)
        if cfg_name in _ORACLE_CAVEAT:
            print(f"[CAVEAT] {cfg_name:14s} raw-head max_abs_diff = {worst:.3e} "
                  f"(OpenCV reorg differs from Darknet — verify vs compiled darknet)")
            continue
        worst_overall = max(worst_overall, worst)
        status = "OK" if worst < _TOL else "FAIL"
        print(f"[{status}] {cfg_name:14s} raw-head max_abs_diff = {worst:.3e}")

    if ran == 0:
        raise SystemExit("No .weights files found in DARKNET_WEIGHTS_DIR.")
    print(f"\nworst overall = {worst_overall:.3e} (tolerance {_TOL:.0e})")
    return 0 if worst_overall < _TOL else 1


if __name__ == "__main__":
    sys.exit(_run())
