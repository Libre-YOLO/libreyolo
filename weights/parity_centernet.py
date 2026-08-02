"""Exact raw-output parity for LibreYOLO CenterNet against pinned upstream.

Set ``CENTERNET_UPSTREAM_DIR`` to xingyizhou/CenterNet at the pinned commit and
``CENTERNET_OFFICIAL_CKPT_DIR`` to a directory containing the two official
checkpoints. The upstream network uses a local torchvision substitution for
the obsolete DCNv2 extension; no upstream source files are modified.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d

from _conversion_utils import add_repo_root_to_path

UPSTREAM_COMMIT = "4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c"
UPSTREAM_LICENSE_SHA256 = (
    "6bda22a8e97bc877ec446f1943a5120debc56bc5bb0c544511121d7b2aa34085"
)
OFFICIAL_CASES = {
    "resdcn18": (
        "ctdet_coco_resdcn18.pth",
        "f9e413f91cdb235adbcb41c5c4052b8f7ff53999374048949789c29d6df18eaa",
    ),
    "dla34": (
        "ctdet_coco_dla_2x.pth",
        "43bf4cc2efe00e02c1ae8484035b062a35543872d276c7dcfeb4db3e64203e4f",
    ),
}


class _ReferenceDCN(nn.Module):
    """Drop-in torchvision replacement for upstream's legacy DCNv2 module."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=(stride, stride),
            padding=(padding, padding),
            bias=True,
        )
        bound = 1.0 / math.sqrt(
            in_channels * self.kernel_size[0] * self.kernel_size[1]
        )
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)

    def forward(self, inputs):
        offset_mask = self.conv_offset_mask(inputs)
        first, second, mask = torch.chunk(offset_mask, 3, dim=1)
        return deform_conv2d(
            inputs,
            torch.cat((first, second), dim=1),
            self.weight,
            self.bias,
            (self.stride, self.stride),
            (self.padding, self.padding),
            (self.dilation, self.dilation),
            torch.sigmoid(mask),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_upstream(path: Path) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"Upstream checkout is {commit}, expected {UPSTREAM_COMMIT}")
    if _sha256(path / "LICENSE") != UPSTREAM_LICENSE_SHA256:
        raise RuntimeError("Pinned upstream LICENSE hash does not match")


def _install_reference_dcn() -> None:
    module = types.ModuleType("models.networks.DCNv2.dcn_v2")
    module.DCN = _ReferenceDCN
    module.DCNv2 = _ReferenceDCN
    sys.modules[module.__name__] = module


def _build_upstream(size: str):
    from models.networks.pose_dla_dcn import DLASeg
    from models.networks.resnet_dcn import BasicBlock, PoseResNet

    heads = {"hm": 80, "wh": 2, "reg": 2}
    if size == "resdcn18":
        return PoseResNet(BasicBlock, [2, 2, 2, 2], heads, head_conv=64)
    model = DLASeg("dla34", heads, False, 4, 1, 5, 256)
    # Upstream creates this unused classifier only while loading ImageNet weights.
    model.base.fc = nn.Conv2d(512, 1000, 1, bias=True)
    return model


def run() -> dict[str, dict[str, float]]:
    upstream = Path(os.environ["CENTERNET_UPSTREAM_DIR"]).resolve()
    checkpoints = Path(os.environ["CENTERNET_OFFICIAL_CKPT_DIR"]).resolve()
    _verify_upstream(upstream)
    sys.path.insert(0, str(upstream / "src" / "lib"))
    _install_reference_dcn()
    add_repo_root_to_path()
    from libreyolo.models.centernet.nn import build_centernet
    from libreyolo.models.centernet.utils import (
        CENTERNET_MEAN,
        CENTERNET_STD,
        preprocess_bgr,
    )
    from libreyolo.postprocess.centernet import postprocess
    from models.decode import ctdet_decode
    from utils.image import affine_transform as upstream_transform_point
    from utils.image import get_affine_transform as upstream_affine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, dict[str, float]] = {}
    rng = np.random.default_rng(637)
    image = rng.integers(0, 256, size=(317, 509, 3), dtype=np.uint8)
    center = np.array([image.shape[1] / 2.0, image.shape[0] / 2.0], dtype=np.float32)
    scale = float(max(image.shape[:2]))
    transform = upstream_affine(center, scale, 0, [512, 512])
    reference_image = cv2.warpAffine(
        image, transform, (512, 512), flags=cv2.INTER_LINEAR
    )
    reference_image = (
        (reference_image / 255.0 - CENTERNET_MEAN) / CENTERNET_STD
    ).astype(np.float32)
    reference_image = np.ascontiguousarray(reference_image.transpose(2, 0, 1))
    candidate_image, _ = preprocess_bgr(image, input_size=512)
    preprocess_difference = float(np.max(np.abs(reference_image - candidate_image)))
    if preprocess_difference != 0.0:
        raise AssertionError(f"Preprocess max_abs_diff={preprocess_difference}")
    for size, (filename, expected_hash) in OFFICIAL_CASES.items():
        path = checkpoints / filename
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"Official checkpoint hash mismatch: {path}")
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        state = {
            key.removeprefix("module."): value
            for key, value in loaded["state_dict"].items()
        }
        reference = _build_upstream(size).eval()
        candidate = build_centernet(size, num_classes=80).eval()
        reference.load_state_dict(state, strict=True)
        candidate.load_state_dict(state, strict=True)
        reference.to(device)
        candidate.to(device)

        inputs = torch.from_numpy(candidate_image).unsqueeze(0).to(device)
        with torch.inference_mode():
            expected = reference(inputs)[-1]
            actual = candidate(inputs)
        results[size] = {
            key: (expected[key] - actual[key]).abs().max().item()
            for key in ("hm", "wh", "reg")
        }

        reference_decoded = ctdet_decode(
            expected["hm"].sigmoid(), expected["wh"], reg=expected["reg"], K=100
        )
        reference_array = reference_decoded.detach().cpu().numpy()[0]
        inverse = upstream_affine(center, scale, 0, (128, 128), inv=1)
        for row in reference_array:
            row[0:2] = upstream_transform_point(row[0:2], inverse)
            row[2:4] = upstream_transform_point(row[2:4], inverse)
        reference_classes = {
            class_id + 1: reference_array[
                reference_array[:, 5] == class_id, :5
            ].astype(np.float32)
            for class_id in range(80)
        }
        reference_rows = []
        for class_id, class_rows in reference_classes.items():
            for row in class_rows:
                reference_rows.append([*row[:5], class_id - 1])
        reference_rows = np.asarray(reference_rows, dtype=np.float32)
        reference_rows = reference_rows[
            np.argsort(-reference_rows[:, 4], kind="stable")
        ]
        candidate_result = postprocess(
            actual,
            conf_thres=0.0,
            original_size=(image.shape[1], image.shape[0]),
            input_size=512,
            max_det=100,
        )
        candidate_rows = np.concatenate(
            (
                candidate_result["boxes"],
                candidate_result["scores"][:, None],
                candidate_result["classes"][:, None].astype(np.float32),
            ),
            axis=1,
        )
        results[size]["e2e_boxes"] = float(
            np.max(np.abs(reference_rows[:, :4] - candidate_rows[:, :4]))
        )
        results[size]["e2e_scores"] = float(
            np.max(np.abs(reference_rows[:, 4] - candidate_rows[:, 4]))
        )
        results[size]["e2e_classes"] = float(
            np.max(np.abs(reference_rows[:, 5] - candidate_rows[:, 5]))
        )
        del reference, candidate, inputs, expected, actual
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def main() -> int:
    missing = [
        name
        for name in ("CENTERNET_UPSTREAM_DIR", "CENTERNET_OFFICIAL_CKPT_DIR")
        if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)}")
    results = run()
    failed = False
    for size, outputs in results.items():
        for name, difference in outputs.items():
            print(f"{size} {name}: max_abs_diff={difference}")
            tolerance = 1e-4 if name == "e2e_boxes" else 0.0
            failed = failed or difference > tolerance
    print("PASS" if not failed else "FAIL")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
