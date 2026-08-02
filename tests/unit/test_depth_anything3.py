"""Hermetic unit coverage for the Depth Anything 3 family."""

from __future__ import annotations

import importlib.util
import json
import numpy as np
from pathlib import Path
import pytest
import sys
import torch
import torch.nn as nn
import yaml
from PIL import Image

pytestmark = pytest.mark.unit


def _load_converter_module():
    root = Path(__file__).resolve().parents[2]
    weights_dir = root / "weights"
    sys.path.insert(0, str(weights_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "convert_depth_anything3_weights",
            weights_dir / "convert_depth_anything3_weights.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_metadata_registration_and_download_url():
    from libreyolo.models import LibreDepthAnything3
    from libreyolo.models.base import BaseModel

    assert LibreDepthAnything3 in BaseModel._registry
    assert LibreDepthAnything3.FAMILY == "depth_anything3"
    assert LibreDepthAnything3.INPUT_SIZES == {"l": 504}
    assert LibreDepthAnything3.SUPPORTED_TASKS == ("depth",)
    assert LibreDepthAnything3.DEFAULT_TASK == "depth"
    assert LibreDepthAnything3.depth_imgsz_divisor == 14
    assert LibreDepthAnything3.get_download_url("LibreDepthAnything3l-depth.pt") == (
        "https://huggingface.co/LibreYOLO/LibreDepthAnything3l-depth/resolve/"
        "main/LibreDepthAnything3l-depth.pt"
    )


def test_checkpoint_detection_is_specific():
    from libreyolo.models.depth_anything3.model import LibreDepthAnything3

    state = {
        "backbone.pretrained.cls_token": torch.zeros(1, 1, 1024),
        "head.scratch.output_conv2.0.weight": torch.zeros(1),
    }
    assert LibreDepthAnything3.can_load(state)
    assert LibreDepthAnything3.detect_size(state) == "l"
    assert LibreDepthAnything3.detect_nb_classes(state) == 1
    assert not LibreDepthAnything3.can_load(
        {
            "pretrained.cls_token": torch.zeros(1, 1, 1024),
            "depth_head.weight": torch.zeros(1),
        }
    )


def test_preprocess_matches_native_upper_bound_geometry():
    from libreyolo.models.depth_anything3.utils import preprocess_numpy

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    chw, ratio = preprocess_numpy(image, 504)
    assert chw.shape == (3, 378, 504)
    assert chw.dtype == np.float32
    assert np.isfinite(chw).all()
    assert ratio == 1.0


def test_converter_requires_explicit_mono_config(tmp_path):
    converter = _load_converter_module()
    weights = tmp_path / "model.safetensors"
    weights.touch()
    with pytest.raises(ValueError, match="mono-versus-metric"):
        converter._validate_config(str(weights), None)

    config = {
        "model_name": "da3metric-large",
        "config": {
            "net": {
                "name": "vitl",
                "out_layers": [4, 11, 17, 23],
                "alt_start": -1,
                "qknorm_start": -1,
                "rope_start": -1,
                "cat_token": False,
            },
            "head": {"dim_in": 1024, "output_dim": 1},
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="not the supported"):
        converter._validate_config(str(weights), str(config_path))


class _FakeBackbone(nn.Module):
    def forward(self, x, **kwargs):
        return [x], []


class _FakeHead(nn.Module):
    def forward(self, features, height, width, patch_start_idx):
        batch = features[0].shape[0]
        depth = torch.full(
            (batch, 1, height, width),
            2.0,
            dtype=features[0].dtype,
            device=features[0].device,
        )
        depth[..., width // 2 :] = 0.5
        return {"depth": depth}


def _lightweight_da3_net():
    from libreyolo.models.depth_anything3.nn import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        LibreDepthAnything3Net,
    )

    net = LibreDepthAnything3Net.__new__(LibreDepthAnything3Net)
    nn.Module.__init__(net)
    net.backbone = _FakeBackbone()
    net.head = _FakeHead()
    net.register_buffer(
        "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
    )
    net.register_buffer(
        "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
    )
    return net


def test_forward_shape_and_depth_inversion_direction():
    net = _lightweight_da3_net().eval()
    output = net(torch.rand(2, 3, 28, 42))
    assert output.shape == (2, 1, 28, 42)
    assert torch.isfinite(output).all()
    # Raw depth is 0.5 on the right (near) and 2.0 on the left (far).
    assert torch.allclose(output[..., 30], torch.tensor(2.0))
    assert torch.allclose(output[..., 5], torch.tensor(0.5))


def test_official_architecture_has_strict_406_tensor_layout_on_meta():
    from libreyolo.models.depth_anything3.nn import LibreDepthAnything3Net

    with torch.device("meta"):
        source = LibreDepthAnything3Net()
        target = LibreDepthAnything3Net()
    state = source.state_dict()
    assert len(state) == 406
    assert not any("pixel_mean" in key or "pixel_std" in key for key in state)
    result = target.load_state_dict(state, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys


class _TinyDepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.head = nn.Identity()

    def forward(self, x):
        return x.mean(dim=1, keepdim=True).clamp_min(1e-6)


@pytest.fixture
def lightweight_model(monkeypatch):
    from libreyolo.models.depth_anything3.model import LibreDepthAnything3

    monkeypatch.setattr(
        LibreDepthAnything3, "_init_model", lambda self: _TinyDepthNet()
    )
    return LibreDepthAnything3(None, size="l", task="depth", device="cpu")


def test_predict_returns_original_canvas_depth(lightweight_model, tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (90, 45), color=(30, 80, 140)).save(path)
    result = lightweight_model.predict(str(path), imgsz=70)
    assert result.depth_map is not None
    assert result.boxes is None
    assert tuple(result.depth_map.data.shape) == (45, 90)
    assert torch.isfinite(result.depth_map.data).all()


def _make_depth_dataset(root):
    for split in ("train", "val"):
        image_dir = root / "images" / split
        depth_dir = root / "depths" / split
        image_dir.mkdir(parents=True)
        depth_dir.mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (70, 70), color=(50 + index, 90, 130)).save(
                image_dir / f"{index}.png"
            )
            depth = np.full((70, 70), 4 * 256, dtype=np.uint16)
            Image.fromarray(depth).save(depth_dir / f"{index}.png")
    config = root / "data.yaml"
    config.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val"}
        )
    )
    return config


def test_zero_shot_val_plumbing_is_finite(lightweight_model, tmp_path):
    metrics = lightweight_model.val(
        data=str(_make_depth_dataset(tmp_path)),
        imgsz=70,
        batch=1,
        workers=0,
        verbose=False,
    )
    assert np.isfinite(metrics["metrics/abs_rel"])
    assert np.isfinite(metrics["metrics/rmse"])
    assert np.isfinite(metrics["metrics/delta1"])


def test_out_of_scope_surfaces(lightweight_model, monkeypatch):
    from libreyolo.models.base.model import BaseModel

    with pytest.raises(NotImplementedError, match="Training"):
        lightweight_model.train(data="unused.yaml")

    captured = {}

    def fake_export(self, format="onnx", **kwargs):
        captured.update(format=format, **kwargs)
        return f"depth-anything3.{format}"

    monkeypatch.setattr(BaseModel, "export", fake_export)
    assert lightweight_model.export(format="onnx", dynamic=False) == (
        "depth-anything3.onnx"
    )
    assert captured == {"format": "onnx", "opset": 17, "dynamic": False}


def test_apply_mono_sky_uses_per_image_statistics():
    """Sky far-depth must come from each image's own quantile; batched
    validation must not mix depth statistics across independent images."""
    from libreyolo.models.depth_anything3.nn import LibreDepthAnything3Net

    depth = torch.ones(2, 1, 20, 20)
    depth[0] *= 2.0
    depth[1] *= 5.0
    sky = torch.zeros(2, 1, 20, 20)
    sky[:, :, :5] = 1.0  # top quarter is sky in both images

    out = LibreDepthAnything3Net._apply_mono_sky(depth, sky)

    assert torch.allclose(out[0, :, :5], torch.full((1, 5, 20), 2.0))
    assert torch.allclose(out[1, :, :5], torch.full((1, 5, 20), 5.0))
    assert torch.equal(out[0, :, 5:], depth[0, :, 5:])
    assert torch.equal(out[1, :, 5:], depth[1, :, 5:])
