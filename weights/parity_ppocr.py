"""Stage-parity check: LibrePPOCR vs the official PP-OCRv5 inference graphs.

Runs the LibreYOLO det and rec networks and the official Paddle inference
graphs on IDENTICAL input tensors and compares the raw outputs, bypassing
preprocessing and batching entirely. This is the decisive fidelity gate for
the port: end-to-end transcript comparisons additionally depend on batch
composition (PaddleX zero-pads per-image-resized crops to the batch maximum
and the SVTR global attention sees that padding), so two official upstream
runtimes (tools/infer vs PaddleX) already disagree with each other on
borderline crops.

Reference result (2026-07, both tiers): det maps match to <= 1e-4 and rec
probabilities to <= 6e-5 with identical argmax; the residual noise is the
unfused multi-branch forward vs Paddle's rep-fused inference graphs.

This script needs BOTH environments, so it runs in two steps::

    # Step 1 (LibreYOLO env): dump inputs + our outputs
    python weights/parity_ppocr.py torch --checkpoint weights/LibrePPOCRt-ocr.pt \
        --image some_text_photo.jpg --out stage_parity.npz

    # Step 2 (scratch venv with paddlepaddle + paddleocr, models cached by
    # a prior PaddleOCR() run): compare against the official graphs
    python weights/parity_ppocr.py paddle --tier t --npz stage_parity.npz
"""

from __future__ import annotations

import argparse

import numpy as np

_PADDLE_MODEL_NAMES = {
    "t": ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"),
    "l": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
}


def run_torch(checkpoint: str, image: str, out: str) -> None:
    import torch
    from PIL import Image

    from _conversion_utils import add_repo_root_to_path

    add_repo_root_to_path()
    from libreyolo import LibreYOLO
    from libreyolo.models.ppocr.preprocess import (
        det_normalize,
        det_resize,
        rec_resize_norm,
    )
    from libreyolo.postprocess.ppocr import db_postprocess

    model = LibreYOLO(checkpoint, device="cpu")
    img = np.asarray(Image.open(image).convert("RGB"))[:, :, ::-1].copy()

    resized, _, _ = det_resize(img)
    det_in = det_normalize(resized).numpy()
    with torch.no_grad():
        det_out = model.model.det(torch.from_numpy(det_in)).numpy()

    # Feed the top-scoring detected region back through rec.
    quads, scores = db_postprocess(det_out[0, 0], img.shape[1], img.shape[0])
    payload = {"det_in": det_in, "det_out": det_out, "size": np.array([model.size == "l"])}
    if len(scores):
        from libreyolo.models.ppocr.preprocess import get_rotate_crop_image

        best = int(np.argmax(scores))
        crop = get_rotate_crop_image(img, quads[best].copy())
        ratio = max(320 / 48, crop.shape[1] / crop.shape[0])
        rec_in = rec_resize_norm(crop, ratio)[None]
        with torch.no_grad():
            rec_out = model.model.rec(torch.from_numpy(rec_in)).numpy()
        payload["rec_in"] = rec_in
        payload["rec_out"] = rec_out
    np.savez(out, **payload)
    print(f"saved {out} (det map {det_out.shape}, rec: {'yes' if 'rec_in' in payload else 'no text found'})")


def run_paddle(tier: str, npz: str) -> None:
    from pathlib import Path

    import paddle

    det_name, rec_name = _PADDLE_MODEL_NAMES[tier]
    home = Path.home() / ".paddlex" / "official_models"
    data = np.load(npz)

    det = paddle.jit.load(str(home / det_name / "inference"))
    det.eval()
    ref = det(paddle.to_tensor(data["det_in"])).numpy()
    diff = np.abs(ref - data["det_out"]).max()
    print(f"det max|diff| = {diff:.3e}")
    assert diff < 1e-3, "det stage diverges beyond fp32 kernel noise"

    if "rec_in" in data:
        rec = paddle.jit.load(str(home / rec_name / "inference"))
        rec.eval()
        ref = rec(paddle.to_tensor(data["rec_in"])).numpy()
        ours = data["rec_out"]
        diff = np.abs(ref - ours).max()
        same_argmax = bool(np.array_equal(ref.argmax(-1), ours.argmax(-1)))
        print(f"rec max|prob diff| = {diff:.3e}, argmax identical = {same_argmax}")
        assert diff < 1e-3 and same_argmax, "rec stage diverges beyond fp32 kernel noise"
    print("stage parity OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    t = sub.add_parser("torch", help="dump inputs + LibreYOLO outputs")
    t.add_argument("--checkpoint", required=True)
    t.add_argument("--image", required=True)
    t.add_argument("--out", default="stage_parity.npz")
    p = sub.add_parser("paddle", help="compare against official inference graphs")
    p.add_argument("--tier", required=True, choices=("t", "l"))
    p.add_argument("--npz", default="stage_parity.npz")
    args = parser.parse_args()
    if args.step == "torch":
        run_torch(args.checkpoint, args.image, args.out)
    else:
        run_paddle(args.tier, args.npz)


if __name__ == "__main__":
    main()
