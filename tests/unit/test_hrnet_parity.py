"""Opt-in exact parity gate against the pinned official HRNet repository.

Set ``HRNET_UPSTREAM_DIR`` to the checkout at commit
``6f69e4676ad8d43d0d61b64b1b9726f0c369e7b1`` and
``HRNET_OFFICIAL_DIR`` to a directory containing the two official source
checkpoints selected for LibreYOLO.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torchvision.transforms import functional as transform_functional

from libreyolo.models.hrnet.nn import HRNetPoseModel
from libreyolo.models.hrnet.utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    box_to_center_scale,
    get_affine_transform,
    preprocess_box_numpy,
)
from libreyolo.postprocess.hrnet import decode_heatmaps, flip_back, flip_back_tensor

pytestmark = [pytest.mark.unit, pytest.mark.external_data]

CASES = {
    "w32": (32, (256, 192), "pose_hrnet_w32_256x192.pth"),
    "w48": (48, (384, 288), "pose_hrnet_w48_384x288.pth"),
}
FLIP_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))


def _required_directory(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable} to run the official HRNet parity gate")
    path = Path(value)
    if not path.is_dir():
        pytest.fail(f"{variable} is not a directory: {path}")
    return path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import upstream module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upstream_modules(upstream_root: Path):
    library = upstream_root / "lib"
    transforms_path = library / "utils" / "transforms.py"
    inference_path = library / "core" / "inference.py"
    model_path = library / "models" / "pose_hrnet.py"
    for path in (transforms_path, inference_path, model_path):
        if not path.is_file():
            pytest.fail(f"pinned upstream file is missing: {path}")

    transforms = _load_module("_hrnet_upstream_transforms", transforms_path)
    old_utils = sys.modules.get("utils")
    old_transforms = sys.modules.get("utils.transforms")
    package = types.ModuleType("utils")
    package.__path__ = [str(library / "utils")]
    sys.modules["utils"] = package
    sys.modules["utils.transforms"] = transforms
    try:
        inference = _load_module("_hrnet_upstream_inference", inference_path)
    finally:
        if old_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = old_utils
        if old_transforms is None:
            sys.modules.pop("utils.transforms", None)
        else:
            sys.modules["utils.transforms"] = old_transforms
    model = _load_module("_hrnet_upstream_pose_model", model_path)
    return transforms, inference, model


def _upstream_config(width: int) -> dict:
    def stage(modules: int, branches: int, channels: list[int]) -> dict:
        return {
            "NUM_MODULES": modules,
            "NUM_BRANCHES": branches,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4] * branches,
            "NUM_CHANNELS": channels,
            "FUSE_METHOD": "SUM",
        }

    return {
        "MODEL": {
            "NUM_JOINTS": 17,
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "PRETRAINED_LAYERS": ["*"],
                "STAGE2": stage(1, 2, [width, width * 2]),
                "STAGE3": stage(4, 3, [width, width * 2, width * 4]),
                "STAGE4": stage(
                    3,
                    4,
                    [width, width * 2, width * 4, width * 8],
                ),
            },
        }
    }


def _reference_input(
    transforms,
    image: np.ndarray,
    box: np.ndarray,
    input_size: tuple[int, int],
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = box
    width = float(x2 - x1)
    height = float(y2 - y1)
    center = np.asarray([x1 + width * 0.5, y1 + height * 0.5], dtype=np.float32)
    aspect_ratio = input_size[1] / input_size[0]
    if width > aspect_ratio * height:
        height = width / aspect_ratio
    elif width < aspect_ratio * height:
        width = height * aspect_ratio
    scale = np.asarray([width / 200.0, height / 200.0], dtype=np.float32) * 1.25

    transform = transforms.get_affine_transform(
        center,
        scale,
        0,
        (input_size[1], input_size[0]),
    )
    crop = cv2.warpAffine(
        image,
        transform,
        (input_size[1], input_size[0]),
        flags=cv2.INTER_LINEAR,
    )
    tensor = transform_functional.to_tensor(crop)
    tensor = transform_functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor, center, scale


@pytest.mark.parametrize("size", CASES)
def test_official_hrnet_end_to_end_parity(size):
    upstream_root = _required_directory("HRNET_UPSTREAM_DIR")
    weights_root = _required_directory("HRNET_OFFICIAL_DIR")
    transforms, inference, upstream_model_module = _upstream_modules(upstream_root)
    width, input_size, filename = CASES[size]
    weight_path = weights_root / filename
    if not weight_path.is_file():
        pytest.fail(f"official checkpoint is missing: {weight_path}")

    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    parity_device = torch.device(
        os.environ.get(
            "HRNET_PARITY_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    upstream_model = upstream_model_module.PoseHighResolutionNet(
        _upstream_config(width)
    ).eval().to(parity_device)
    port_model = HRNetPoseModel(width=width, num_keypoints=17).eval().to(parity_device)
    upstream_model.load_state_dict(state_dict, strict=True)
    port_model.load_state_dict(state_dict, strict=True)
    assert upstream_model.state_dict().keys() == port_model.state_dict().keys()

    generator = np.random.default_rng(637)
    image = generator.integers(0, 256, size=(411, 307, 3), dtype=np.uint8)
    box = np.asarray([31.25, 18.5, 271.75, 389.25], dtype=np.float32)
    reference_input, reference_center, reference_scale = _reference_input(
        transforms,
        image,
        box,
        input_size,
    )
    port_input, center, scale = preprocess_box_numpy(image, box, input_size)

    assert np.array_equal(center, reference_center)
    assert np.array_equal(scale, reference_scale)
    assert np.array_equal(
        get_affine_transform(
            center,
            scale,
            0,
            (input_size[1], input_size[0]),
        ),
        transforms.get_affine_transform(
            reference_center,
            reference_scale,
            0,
            (input_size[1], input_size[0]),
        ),
    )
    assert torch.equal(torch.from_numpy(port_input), reference_input)

    batch = reference_input.unsqueeze(0).to(parity_device)
    with torch.inference_mode():
        upstream_heatmaps = upstream_model(batch)
        port_heatmaps = port_model(batch)
    max_abs_diff = float((upstream_heatmaps - port_heatmaps).abs().max())
    assert max_abs_diff == 0.0
    assert torch.equal(upstream_heatmaps, port_heatmaps)

    centers = center[None, :]
    scales = scale[None, :]
    config = SimpleNamespace(TEST=SimpleNamespace(POST_PROCESS=True))
    upstream_heatmaps_numpy = upstream_heatmaps.cpu().numpy()
    port_heatmaps_numpy = port_heatmaps.cpu().numpy()
    reference_points, reference_scores = inference.get_final_preds(
        config,
        upstream_heatmaps_numpy,
        centers,
        scales,
    )
    points, scores = decode_heatmaps(
        port_heatmaps_numpy,
        centers,
        scales,
        post_process=True,
    )
    assert np.array_equal(points, reference_points)
    assert np.array_equal(scores, reference_scores)


def test_flip_and_shift_match_upstream():
    upstream_root = _required_directory("HRNET_UPSTREAM_DIR")
    transforms, _inference, _model = _upstream_modules(upstream_root)
    heatmaps = np.random.default_rng(637).standard_normal((2, 17, 9, 7)).astype(
        np.float32
    )

    reference = transforms.flip_back(heatmaps.copy(), FLIP_PAIRS)
    assert np.array_equal(flip_back(heatmaps), reference)

    reference_tensor = torch.from_numpy(reference.copy())
    reference_tensor[:, :, :, 1:] = reference_tensor.clone()[:, :, :, :-1]
    restored = flip_back_tensor(torch.from_numpy(heatmaps.copy()), shift=True)
    assert torch.equal(restored, reference_tensor)


def test_box_to_center_scale_rejects_degenerate_boxes():
    with pytest.raises(ValueError, match="positive area"):
        box_to_center_scale((10, 20, 10, 40), (256, 192))
