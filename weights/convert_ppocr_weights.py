"""Convert PP-OCRv5 Paddle weights into a composite LibrePPOCR checkpoint.

Source checkpoints are the official PP-OCRv5 *training* checkpoints
(``.pdparams``, Apache-2.0) released by the PaddleOCR project:

- PP-OCRv5_mobile_det_pretrained.pdparams + PP-OCRv5_mobile_rec_pretrained.pdparams -> size ``t``
- PP-OCRv5_server_det_pretrained.pdparams + PP-OCRv5_server_rec_pretrained.pdparams -> size ``l``

Loading ``.pdparams`` requires the ``paddlepaddle`` pip package, which is never
a LibreYOLO dependency. Dump each file to ``.npz`` inside a scratch virtualenv
first::

    python weights/_dump_ppocr_paddle.py PP-OCRv5_mobile_det_pretrained.pdparams mobile_det.npz

then convert from the normal LibreYOLO environment::

    python weights/convert_ppocr_weights.py \
        --det mobile_det.npz --rec mobile_rec.npz \
        --dict ppocrv5_dict.txt --size t \
        --output weights/LibrePPOCRt-ocr.pt

Paddle -> PyTorch mapping performed here:
- ``._mean`` / ``._variance`` batch-norm buffers -> ``running_mean`` / ``running_var``
- ``nn.Linear`` weights are stored ``[in, out]`` in Paddle and transposed to
  torch's ``[out, in]`` (selected by module type, not by shape, so square
  matrices are handled correctly)
- det tensors gain the ``det.`` namespace, rec tensors the ``rec.`` namespace
- the train-only NRTR guidance branch (``head.before_gtc.*``/``head.gtc_head.*``)
  and the unused HGNetV2 classification tail (``backbone.last_conv``/``fc``)
  are dropped

The recognition dictionary (``ppocr/utils/dict/ppocrv5_dict.txt`` from the
Apache-2.0 upstream) is embedded in the checkpoint metadata as the ``charset``
list (blank + dictionary + space), making the ``.pt`` self-contained. Raw
Paddle files are out of scope for runtime autoconversion: conversion happens
via this script only.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Set

import numpy as np
import torch

from _conversion_utils import add_repo_root_to_path, save_checkpoint

add_repo_root_to_path()

from libreyolo.models.ppocr.model import LibrePPOCRModel  # noqa: E402
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint  # noqa: E402

_IMGSZ = 960

# Train-only or unused-at-inference upstream parameters, dropped with a report.
_DROP_PREFIXES = (
    "head.before_gtc.",
    "head.gtc_head.",
    "backbone.last_conv.",
    "backbone.fc.",
    "backbone.lab.",
    "backbone.dropout.",
)

_PIPELINE = {
    "det_limit_side_len": 960,
    "det_db_thresh": 0.3,
    "det_db_box_thresh": 0.6,
    "det_db_unclip_ratio": 1.5,
    "rec_image_shape": [3, 48, 320],
}


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def _linear_weight_keys(model: torch.nn.Module, prefix: str) -> Set[str]:
    keys = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            keys.add(f"{prefix}{name}.weight")
    return keys


def _map_namespace(
    arrays: Dict[str, np.ndarray],
    prefix: str,
    target_keys: Iterable[str],
    linear_keys: Set[str],
) -> tuple[Dict[str, torch.Tensor], list[str]]:
    """Map one submodel's Paddle arrays into the ``det.``/``rec.`` namespace."""
    target_keys = set(target_keys)
    mapped: Dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for name, value in arrays.items():
        if any(name.startswith(p) for p in _DROP_PREFIXES):
            dropped.append(name)
            continue
        torch_name = prefix + name.replace("._mean", ".running_mean").replace(
            "._variance", ".running_var"
        )
        if torch_name not in target_keys:
            raise KeyError(
                f"Unmapped Paddle tensor {name!r} (torch name {torch_name!r} is "
                "not in the LibrePPOCR module tree). The upstream architecture "
                "may have changed; refusing to convert."
            )
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        if torch_name in linear_keys:
            tensor = tensor.t().contiguous()
        mapped[torch_name] = tensor
    return mapped, dropped


def _read_charset(dict_path: str) -> list[str]:
    chars = []
    with open(dict_path, "rb") as f:
        for line in f.readlines():
            chars.append(line.decode("utf-8").rstrip("\n").rstrip("\r"))
    # CTC alphabet: blank + dictionary + space (use_space_char=True upstream).
    return ["blank"] + chars + [" "]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(det_npz: str, rec_npz: str, dict_path: str, size: str, output: str) -> Path:
    charset = _read_charset(dict_path)
    print(f"charset: {len(charset)} entries (blank + dict + space)")

    model = LibrePPOCRModel(size=size, num_classes=len(charset))
    model.eval()
    target_state = model.state_dict()
    linear_keys = _linear_weight_keys(model.det, "det.") | _linear_weight_keys(
        model.rec, "rec."
    )

    det_mapped, det_dropped = _map_namespace(
        _load_npz(det_npz), "det.", target_state.keys(), linear_keys
    )
    rec_mapped, rec_dropped = _map_namespace(
        _load_npz(rec_npz), "rec.", target_state.keys(), linear_keys
    )
    state_dict = {**det_mapped, **rec_mapped}

    missing = [
        k
        for k in target_state
        if k not in state_dict and "num_batches_tracked" not in k
    ]
    if missing:
        raise KeyError(
            f"{len(missing)} LibrePPOCR tensors missing from the Paddle "
            f"sources, e.g. {missing[:8]}"
        )
    for key, tensor in state_dict.items():
        if tuple(tensor.shape) != tuple(target_state[key].shape):
            raise ValueError(
                f"shape mismatch for {key}: paddle {tuple(tensor.shape)} vs "
                f"torch {tuple(target_state[key].shape)}"
            )

    incompatible = model.load_state_dict(state_dict, strict=False)
    leftover = [k for k in incompatible.missing_keys if "num_batches_tracked" not in k]
    assert not leftover and not incompatible.unexpected_keys, (
        leftover,
        incompatible.unexpected_keys,
    )
    print(
        f"mapped {len(det_mapped)} det + {len(rec_mapped)} rec tensors; "
        f"dropped {len(det_dropped)} det / {len(rec_dropped)} rec train-only tensors"
    )

    checkpoint = wrap_libreyolo_checkpoint(
        model.state_dict(),
        model_family="ppocr",
        size=size,
        task="ocr",
        nc=1,
        names={0: "text"},
        imgsz=_IMGSZ,
        charset=charset,
        pipeline=dict(_PIPELINE),
        # Reserved for optional pipeline stages (document orientation,
        # unwarping, textline rotation) so adding them later is not a
        # schema break.
        components={},
        source={
            "det": Path(det_npz).name,
            "rec": Path(rec_npz).name,
            "upstream": "https://github.com/PaddlePaddle/PaddleOCR (Apache-2.0)",
        },
    )
    out = save_checkpoint(checkpoint, output)
    print(f"saved {out}")
    print(f"sha256 {_sha256(out)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--det", required=True, help="det .npz from _dump_ppocr_paddle.py")
    parser.add_argument("--rec", required=True, help="rec .npz from _dump_ppocr_paddle.py")
    parser.add_argument("--dict", required=True, dest="dict_path", help="ppocrv5_dict.txt")
    parser.add_argument("--size", required=True, choices=("t", "l"))
    parser.add_argument("--output", required=True, help="e.g. weights/LibrePPOCRt-ocr.pt")
    args = parser.parse_args()
    convert(args.det, args.rec, args.dict_path, args.size, args.output)


if __name__ == "__main__":
    main()
