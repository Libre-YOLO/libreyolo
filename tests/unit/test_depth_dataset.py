"""Unit tests for the depth dataset loader."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo.data.depth_dataset import (
    DepthDataset,
    depth_collate_fn,
    img2depth_mask_paths,
    img2depth_paths,
    resolve_depth_data,
)

pytestmark = pytest.mark.unit


def _write_image(path, size=(40, 30), color=(50, 90, 130)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _write_depth_png(path, shape=(30, 40), scale=256.0, near=2.0, far=8.0):
    """Left half ``near`` units, right half ``far`` units, 16-bit PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = np.full(shape, far, dtype=np.float64)
    depth[:, : shape[1] // 2] = near
    encoded = (depth * scale).round().astype(np.uint16)
    Image.fromarray(encoded).save(path)
    return depth


def _make_dataset_root(tmp_path, n_images=3, depth_format="png", extra_yaml=None):
    for split in ("train", "val"):
        for i in range(n_images):
            _write_image(tmp_path / "images" / split / f"img{i}.jpg")
            depth_path = tmp_path / "depths" / split / f"img{i}.{depth_format}"
            if depth_format == "png":
                _write_depth_png(depth_path)
            else:
                depth_path.parent.mkdir(parents=True, exist_ok=True)
                depth = np.full((30, 40), 8.0, dtype=np.float32)
                depth[:, :20] = 2.0
                np.save(depth_path, depth)
    config = {
        "path": str(tmp_path),
        "train": "images/train",
        "val": "images/val",
    }
    if extra_yaml:
        config.update(extra_yaml)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(config))
    return yaml_path


class TestDepthPaths:
    def test_img2depth_paths_substitutes_images_dir(self, tmp_path):
        img = tmp_path / "images" / "train" / "a.jpg"
        _write_image(img)
        _write_depth_png(tmp_path / "depths" / "train" / "a.png")

        paths = img2depth_paths([img])
        assert paths[0] == tmp_path / "depths" / "train" / "a.png"

    def test_img2depth_paths_custom_dir_and_npy(self, tmp_path):
        img = tmp_path / "images" / "train" / "a.jpg"
        _write_image(img)
        target = tmp_path / "gt" / "train" / "a.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, np.ones((30, 40), dtype=np.float32))

        paths = img2depth_paths([img], depths_dir="gt")
        assert paths[0] == target

    def test_img2depth_paths_finds_depth_suffix_in_same_dir(self, tmp_path):
        img = tmp_path / "val" / "indoors" / "a.png"
        _write_image(img)
        target = tmp_path / "val" / "indoors" / "a_depth.npy"
        np.save(target, np.ones((30, 40), dtype=np.float32))

        paths = img2depth_paths([img])
        assert paths[0] == target

    def test_img2depth_mask_paths_finds_depth_mask_suffix(self, tmp_path):
        depth = tmp_path / "val" / "indoors" / "a_depth.npy"
        depth.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth, np.ones((30, 40), dtype=np.float32))
        mask = tmp_path / "val" / "indoors" / "a_depth_mask.npy"
        np.save(mask, np.ones((30, 40), dtype=bool))

        paths = img2depth_mask_paths([depth])
        assert paths[0] == mask


class TestDepthDataset:
    def test_opencv_bicubic_resize_matches_declared_coreml_pixels(self, tmp_path):
        import cv2

        yaml_path = _make_dataset_root(tmp_path, n_images=1)
        config = resolve_depth_data(yaml_path)
        dataset = DepthDataset(
            config,
            split="train",
            imgsz=17,
            resize_mode="stretch",
            resize_backend="opencv",
            interpolation="bicubic",
        )
        yy, xx = np.mgrid[:7, :11]
        image = np.stack(
            (
                (xx * 17 + yy * 3) % 256,
                (xx * 5 + yy * 29) % 256,
                (xx * 31 + yy * 7) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        depth = (xx + 10 * yy).astype(np.float32)

        resized_image, resized_depth, _ratio, _pad = dataset._resize(image, depth)

        np.testing.assert_array_equal(
            resized_image,
            cv2.resize(image, (17, 17), interpolation=cv2.INTER_CUBIC),
        )
        np.testing.assert_array_equal(
            resized_depth,
            cv2.resize(depth, (17, 17), interpolation=cv2.INTER_NEAREST),
        )

    def test_loads_16bit_png_with_scale(self, tmp_path):
        yaml_path = _make_dataset_root(tmp_path)
        config = resolve_depth_data(yaml_path)
        ds = DepthDataset(config, split="train", imgsz=40, resize_mode="stretch")

        img, depth, info, idx = ds[0]
        assert img.shape == (3, 40, 40)
        assert img.dtype == torch.float32
        assert depth.shape == (40, 40)
        # 16-bit PNG values divided by depth_scale=256 recover the units.
        assert float(depth.min()) == pytest.approx(2.0, abs=1e-2)
        assert float(depth.max()) == pytest.approx(8.0, abs=1e-2)

    def test_loads_npy_float_depth_as_is(self, tmp_path):
        yaml_path = _make_dataset_root(tmp_path, depth_format="npy")
        config = resolve_depth_data(yaml_path)
        ds = DepthDataset(config, split="train", imgsz=40, resize_mode="stretch")

        _, depth, _, _ = ds[0]
        assert float(depth.max()) == pytest.approx(8.0, abs=1e-5)

    def test_custom_depth_scale(self, tmp_path):
        yaml_path = _make_dataset_root(tmp_path, extra_yaml={"depth_scale": 512.0})
        config = resolve_depth_data(yaml_path)
        ds = DepthDataset(config, split="train", imgsz=40, resize_mode="stretch")

        _, depth, _, _ = ds[0]
        assert float(depth.max()) == pytest.approx(4.0, abs=1e-2)

    def test_zero_depth_scale_is_rejected(self, tmp_path):
        yaml_path = _make_dataset_root(tmp_path, extra_yaml={"depth_scale": 0})
        config = resolve_depth_data(yaml_path)

        with pytest.raises(ValueError, match="depth_scale"):
            DepthDataset(config, split="train", imgsz=40, resize_mode="stretch")

    def test_letterbox_pads_with_invalid_zero(self, tmp_path):
        yaml_path = _make_dataset_root(tmp_path)
        config = resolve_depth_data(yaml_path)
        ds = DepthDataset(config, split="train", imgsz=40, resize_mode="letterbox")

        _, depth, _, _ = ds[0]
        assert depth.shape == (40, 40)
        # 40x30 image letterboxed into 40x40: bottom 10 rows are padding,
        # which must be invalid (0) so it self-excludes from loss/metrics.
        assert float(depth[31:, :].abs().sum()) == 0.0
        assert float(depth[:30, :].min()) > 0.0

    def test_missing_depth_file_raises(self, tmp_path):
        yaml_path = _make_dataset_root(tmp_path)
        (tmp_path / "depths" / "train" / "img0.png").unlink()
        config = resolve_depth_data(yaml_path)

        with pytest.raises(FileNotFoundError, match="depth file"):
            DepthDataset(config, split="train", imgsz=40)

    def test_shape_mismatch_raises(self, tmp_path):
        yaml_path = _make_dataset_root(tmp_path)
        _write_depth_png(tmp_path / "depths" / "train" / "img0.png", shape=(10, 10))
        config = resolve_depth_data(yaml_path)
        ds = DepthDataset(config, split="train", imgsz=40)

        with pytest.raises(ValueError, match="does not match image shape"):
            _ = ds[0]

    def test_negative_and_nonfinite_become_invalid(self, tmp_path):
        yaml_path = _make_dataset_root(tmp_path, depth_format="npy")
        bad = np.full((30, 40), 5.0, dtype=np.float32)
        bad[0, 0] = -3.0
        bad[0, 1] = np.nan
        bad[0, 2] = np.inf
        np.save(tmp_path / "depths" / "train" / "img0.npy", bad)
        config = resolve_depth_data(yaml_path)
        ds = DepthDataset(config, split="train", imgsz=40, resize_mode="stretch")

        _, depth, _, _ = ds[0]
        assert torch.isfinite(depth).all()
        assert float(depth.min()) >= 0.0

    def test_applies_optional_depth_mask(self, tmp_path):
        yaml_path = _make_dataset_root(tmp_path, depth_format="npy")
        mask = np.ones((30, 40), dtype=np.uint8)
        mask[:, :10] = 0
        np.save(tmp_path / "depths" / "train" / "img0_mask.npy", mask)
        config = resolve_depth_data(yaml_path)
        ds = DepthDataset(config, split="train", imgsz=40, resize_mode="stretch")

        _, depth, _, _ = ds[0]
        assert float(depth[:, :10].sum()) == 0.0
        assert float(depth[:, 10:].min()) > 0.0

    def test_loads_diode_style_depth_and_mask(self, tmp_path):
        root = tmp_path / "diode"
        image = root / "val" / "indoors" / "scene_00001" / "scan_00001" / "00001.png"
        _write_image(image)
        depth = np.full((30, 40), 3.0, dtype=np.float32)
        depth[0, 0] = 9.0
        np.save(image.with_name("00001_depth.npy"), depth)
        mask = np.ones((30, 40), dtype=np.uint8)
        mask[0, 0] = 0
        np.save(image.with_name("00001_depth_mask.npy"), mask)
        yaml_path = root / "data.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "path": str(root),
                    "train": "val/indoors",
                    "val": "val/indoors",
                    "depth_stem_suffix": "_depth",
                    "depth_mask_suffix": "_mask",
                }
            )
        )

        config = resolve_depth_data(yaml_path)
        ds = DepthDataset(config, split="train", imgsz=40, resize_mode="stretch")

        _, target, _, _ = ds[0]
        assert float(target[0, 0]) == 0.0
        assert float(target.max()) == pytest.approx(3.0)

    def test_invalid_resize_mode_rejected(self, tmp_path):
        yaml_path = _make_dataset_root(tmp_path)
        config = resolve_depth_data(yaml_path)

        with pytest.raises(ValueError, match="resize_mode"):
            DepthDataset(config, split="train", imgsz=40, resize_mode="crop")

    def test_missing_split_raises(self, tmp_path):
        yaml_path = _make_dataset_root(tmp_path)
        config = resolve_depth_data(yaml_path)
        config.pop("test", None)

        with pytest.raises(ValueError, match="no 'test' split"):
            DepthDataset(config, split="test", imgsz=40)

    def test_augment_preserves_shapes_and_values(self, tmp_path):
        yaml_path = _make_dataset_root(tmp_path)
        config = resolve_depth_data(yaml_path)
        ds = DepthDataset(
            config, split="train", imgsz=40, augment=True, resize_mode="stretch"
        )

        for _ in range(4):
            img, depth, _, _ = ds[0]
            assert img.shape == (3, 40, 40)
            assert depth.shape == (40, 40)
            valid = depth[depth > 0]
            assert set(np.unique(valid.numpy()).round(2)) <= {2.0, 8.0}

    def test_collate_stacks_batch(self, tmp_path):
        yaml_path = _make_dataset_root(tmp_path)
        config = resolve_depth_data(yaml_path)
        ds = DepthDataset(config, split="train", imgsz=40, resize_mode="stretch")

        imgs, depths, infos, ids = depth_collate_fn([ds[0], ds[1]])
        assert imgs.shape == (2, 3, 40, 40)
        assert depths.shape == (2, 40, 40)
        assert len(infos) == 2 and len(ids) == 2
