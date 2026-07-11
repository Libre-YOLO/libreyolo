"""Dump PP-OCRv5 Paddle training checkpoints to plain .npz files.

Run this INSIDE a scratch virtualenv that has the ``paddlepaddle`` pip
package installed (paddlepaddle is never a LibreYOLO dependency)::

    python weights/_dump_ppocr_paddle.py PP-OCRv5_mobile_det_pretrained.pdparams out.npz

The output is a name -> float32 array archive consumed by
``weights/convert_ppocr_weights.py`` from the normal LibreYOLO environment.
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python _dump_ppocr_paddle.py <in.pdparams> <out.npz>")
    src, dst = sys.argv[1], sys.argv[2]

    import paddle

    state = paddle.load(src)
    arrays = {}
    for name, value in state.items():
        if name == "StructuredToParameterName@@":
            continue
        arrays[name] = np.asarray(value)
    np.savez(dst, **arrays)
    print(f"dumped {len(arrays)} tensors from {src} -> {dst}")


if __name__ == "__main__":
    main()
