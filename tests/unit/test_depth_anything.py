"""Unit tests for the Depth Anything V2 family.

The DINOv2 + DPT architecture builds from random init with no network access,
so a real ViT-S is exercised at a small multiple-of-14 input to keep the tests
hermetic and fast. Metadata/detection tests need no model build at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml as _yaml
from PIL import Image

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- metadata


class TestDepthAnythingMetadata:
    def test_task_and_size_metadata(self):
        from libreyolo.models.depth_anything.model import LibreDepthAnythingV2

        assert LibreDepthAnythingV2.FAMILY == "depth_anything"
        assert LibreDepthAnythingV2.SUPPORTED_TASKS == ("depth",)
        assert LibreDepthAnythingV2.DEFAULT_TASK == "depth"
        assert LibreDepthAnythingV2.depth_imgsz_divisor == 14
        assert set(LibreDepthAnythingV2.INPUT_SIZES) == {"s", "b", "l", "g"}
        assert set(LibreDepthAnythingV2.INPUT_SIZES.values()) == {518}

    def test_registered_in_factory(self):
        from libreyolo.models import LibreDepthAnythingV2
        from libreyolo.models.base import BaseModel

        assert LibreDepthAnythingV2 in BaseModel._registry

    def test_can_load_recognizes_da_signature(self):
        from libreyolo.models.depth_anything.model import LibreDepthAnythingV2

        state = {
            "pretrained.cls_token": torch.zeros(1, 1, 384),
            "pretrained.pos_embed": torch.zeros(1, 1370, 384),
            "depth_head.scratch.output_conv2.0.weight": torch.zeros(1),
        }
        assert LibreDepthAnythingV2.can_load(state)

    def test_can_load_rejects_rfdetr_depth_signature(self):
        from libreyolo.models.depth_anything.model import LibreDepthAnythingV2

        # RF-DETR depth uses backbone.* (not pretrained.*) for the encoder.
        rfdetr_depth = {
            "backbone.encoder.proj.weight": torch.zeros(1),
            "depth_head.weight": torch.zeros(1, 8, 1, 1),
        }
        assert not LibreDepthAnythingV2.can_load(rfdetr_depth)

    @pytest.mark.parametrize(
        "embed_dim,expected",
        [(384, "s"), (768, "b"), (1024, "l"), (1536, "g")],
    )
    def test_detect_size_from_encoder_width(self, embed_dim, expected):
        from libreyolo.models.depth_anything.model import LibreDepthAnythingV2

        state = {"pretrained.cls_token": torch.zeros(1, 1, embed_dim)}
        assert LibreDepthAnythingV2.detect_size(state) == expected

    def test_detect_nb_classes_is_one(self):
        from libreyolo.models.depth_anything.model import LibreDepthAnythingV2

        assert LibreDepthAnythingV2.detect_nb_classes({"pretrained.cls_token": 1}) == 1

    def test_filename_size_and_task_detection(self):
        from libreyolo.models.depth_anything.model import LibreDepthAnythingV2

        fname = "LibreDepthAnythingV2l-depth.pt"
        assert LibreDepthAnythingV2.detect_size_from_filename(fname) == "l"
        assert LibreDepthAnythingV2.detect_task_from_filename(fname) == "depth"


# ---------------------------------------------------------------- preprocess


class TestPreprocess:
    def test_keep_aspect_multiple_of_14_and_range(self):
        from libreyolo.models.depth_anything.utils import preprocess_numpy

        # A smooth gradient (real-image-like); cubic resize stays well-behaved.
        yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
        gradient = ((xx / 640.0 + yy / 480.0) * 0.5 * 255.0).astype(np.uint8)
        img = np.repeat(gradient[:, :, None], 3, axis=2)

        chw, ratio = preprocess_numpy(img, input_size=518)

        assert chw.dtype == np.float32
        assert chw.shape[0] == 3
        h, w = chw.shape[1], chw.shape[2]
        assert h % 14 == 0
        assert w % 14 == 0
        assert h >= 518  # lower-bound resize
        assert w >= 518
        assert ratio == 1.0
        # Normalized to ~[0, 1] (divided by 255). cv2.INTER_CUBIC can overshoot
        # the [0, 1] range slightly, so allow a small margin rather than a hard
        # clamp; the model applies ImageNet normalization next regardless.
        assert np.isfinite(chw).all()
        assert chw.min() >= -0.5
        assert chw.max() <= 1.5


# ---------------------------------------------------------------- forward


@pytest.fixture(scope="module")
def da_small():
    """A random-init ViT-S depth model wrapper on CPU (no weights download)."""
    from libreyolo.models.depth_anything.model import LibreDepthAnythingV2

    return LibreDepthAnythingV2(model_path=None, size="s", task="depth", device="cpu")


class TestForwardAndPredict:
    def test_net_forward_shape_and_nonneg(self):
        from libreyolo.models.depth_anything.nn import LibreDepthAnythingV2Net

        net = LibreDepthAnythingV2Net("vits").eval()
        x = torch.rand(2, 3, 70, 70)  # 70 = 5 * 14
        with torch.no_grad():
            out = net(x)
        assert out.shape == (2, 1, 70, 70)
        assert float(out.min()) >= 0.0  # relative inverse depth is non-negative

    def test_normalization_buffers_not_in_state_dict(self):
        from libreyolo.models.depth_anything.nn import LibreDepthAnythingV2Net

        keys = LibreDepthAnythingV2Net("vits").state_dict().keys()
        assert not any("pixel_mean" in k or "pixel_std" in k for k in keys)
        assert all(
            k.startswith("pretrained.") or k.startswith("depth_head.") for k in keys
        )

    def test_wrapper_metadata(self, da_small):
        assert da_small.task == "depth"
        assert da_small.input_size == 518
        assert da_small.nb_classes == 1
        assert da_small.names == {0: "depth"}

    def test_predict_returns_depth_map(self, da_small, tmp_path):
        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (90, 45), color=(50, 90, 130)).save(img_path)

        result = da_small.predict(str(img_path), imgsz=70)

        assert result.boxes is None
        assert result.depth_map is not None
        assert tuple(result.depth_map.data.shape) == (45, 90)

    def test_predict_rejects_non_patch_imgsz(self, da_small, tmp_path):
        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (64, 64), color=(10, 20, 30)).save(img_path)
        with pytest.raises(ValueError, match="divisible by 14"):
            da_small.predict(str(img_path), imgsz=100)

    def test_predict_rejects_augment(self, da_small, tmp_path):
        # augment must raise, not silently fall through to plain inference.
        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (56, 56), color=(10, 20, 30)).save(img_path)
        with pytest.raises(ValueError, match="augment"):
            da_small.predict(str(img_path), imgsz=70, augment=True)

    def test_train_and_export_out_of_scope(self, da_small):
        with pytest.raises(NotImplementedError):
            da_small.train(data="x.yaml")
        with pytest.raises(NotImplementedError):
            da_small.export(format="onnx")

    def test_inherited_infer_image_is_blocked(self, da_small):
        # The vendored infer_image would double-normalize; it must redirect.
        with pytest.raises(NotImplementedError, match="predict"):
            da_small.model.infer_image(np.zeros((10, 10, 3), dtype=np.uint8))


# ---------------------------------------------------------------- val + ckpt


def _make_depth_yaml(root, n_images=4, size=70):
    for split in ("train", "val"):
        img_dir = root / "images" / split
        depth_dir = root / "depths" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_images):
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            arr[:, : size // 2] = (200, 40, 40)
            arr[:, size // 2 :] = (40, 40, 200)
            Image.fromarray(arr).save(img_dir / f"img{i}.jpg")
            depth = np.full((size, size), 8.0)
            depth[:, : size // 2] = 2.0
            encoded = (depth * 256.0).round().astype(np.uint16)
            Image.fromarray(encoded).save(depth_dir / f"img{i}.png")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        _yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val"}
        )
    )
    return yaml_path


def test_zero_shot_val_runs(da_small, tmp_path):
    """Zero-shot eval through the shared DepthValidator yields finite metrics."""
    yaml_path = _make_depth_yaml(tmp_path)
    metrics = da_small.val(
        data=str(yaml_path), imgsz=70, batch=2, workers=0, verbose=False
    )
    assert np.isfinite(metrics["metrics/abs_rel"])
    assert "metrics/delta1" in metrics
    assert metrics["fitness"] == metrics["metrics/delta1"]


def test_checkpoint_round_trip(da_small, tmp_path):
    """A converted-style checkpoint loads back through the factory as depth."""
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    ckpt = wrap_libreyolo_checkpoint(
        da_small.model.state_dict(),
        model_family="depth_anything",
        size="s",
        task="depth",
        nc=1,
        names={0: "depth"},
        imgsz=518,
    )
    ckpt_path = tmp_path / "LibreDepthAnythingV2s-depth.pt"
    torch.save(ckpt, str(ckpt_path))

    loaded = LibreYOLO(str(ckpt_path), device="cpu")
    assert loaded.FAMILY == "depth_anything"
    assert loaded.task == "depth"
    assert loaded.size == "s"
    assert loaded.names == {0: "depth"}


def test_names_forced_to_depth(da_small, tmp_path):
    """A checkpoint carrying a generic class_0 name is normalized to depth."""
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    ckpt = wrap_libreyolo_checkpoint(
        da_small.model.state_dict(),
        model_family="depth_anything",
        size="s",
        task="depth",
        nc=1,
        names={0: "class_0"},  # what the generic autoconverter would write
        imgsz=518,
    )
    ckpt_path = tmp_path / "LibreDepthAnythingV2s-depth.pt"
    torch.save(ckpt, str(ckpt_path))

    loaded = LibreYOLO(str(ckpt_path), device="cpu")
    assert loaded.names == {0: "depth"}
