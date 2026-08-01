"""Tests for optional RAM/disk image caching (libreyolo.data.cache)."""

import time

import numpy as np
import pytest
from PIL import Image

from libreyolo.data.cache import normalize_cache
from libreyolo.data.dataset import YOLODataset

pytestmark = pytest.mark.unit


def _write_files(tmp_path, n=5):
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n):
        arr = rng.randint(0, 255, (40 + i, 50 + i, 3), dtype=np.uint8)
        Image.fromarray(arr).save(image_dir / f"img{i}.png")
        (label_dir / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def _build(tmp_path):
    return YOLODataset(data_dir=str(tmp_path), split="train", img_size=(64, 64))


def _make_dataset(tmp_path, n=5):
    _write_files(tmp_path, n)
    return _build(tmp_path)


def test_normalize_cache_values():
    assert normalize_cache(False) is None
    assert normalize_cache(None) is None
    assert normalize_cache(True) == "ram"
    assert normalize_cache("ram") == "ram"
    assert normalize_cache("DISK") == "disk"
    assert normalize_cache("true") == "ram"
    assert normalize_cache("false") is None
    with pytest.raises(ValueError):
        normalize_cache("bogus")


def test_cache_disabled_by_default(tmp_path):
    ds = _make_dataset(tmp_path)
    assert ds.cache is None  # disabled until enabled
    ds.enable_image_cache(False)
    assert ds.cache is None


def test_ram_cache_matches_uncached_and_returns_copies(tmp_path):
    _write_files(tmp_path)
    ref = _build(tmp_path)
    ref.enable_image_cache(False)
    expected = [ref.load_image(i).copy() for i in range(len(ref))]

    ram = _build(tmp_path)
    ram.enable_image_cache("ram")
    for i in range(len(ram)):
        first = ram.load_image(i)
        second = ram.load_image(i)
        assert np.array_equal(first, expected[i])
        assert np.array_equal(second, expected[i])
        # Copy-on-read so in-place augmentation cannot corrupt the cache.
        assert first is not second


def test_disk_cache_creates_npy_and_reloads(tmp_path):
    _write_files(tmp_path)
    ref = _build(tmp_path)
    ref.enable_image_cache(False)
    expected = [ref.load_image(i).copy() for i in range(len(ref))]

    disk = _build(tmp_path)
    disk.enable_image_cache("disk")
    for i in range(len(disk)):
        assert np.array_equal(disk.load_image(i), expected[i])

    # .npy sidecars created with appended suffix (collision-free).
    for i in range(len(disk)):
        assert (tmp_path / "images" / "train" / f"img{i}.png.npy").exists()

    # Fresh dataset reads from the .npy cache and matches.
    reload = _build(tmp_path)
    reload.enable_image_cache("disk")
    for i in range(len(reload)):
        assert np.array_equal(reload.load_image(i), expected[i])


def test_disk_cache_invalidates_on_source_change(tmp_path):
    _write_files(tmp_path)
    disk = _build(tmp_path)
    disk.enable_image_cache("disk")
    _ = disk.load_image(0)  # populate cache

    time.sleep(1.1)  # ensure a newer mtime than the .npy
    new = np.full((40, 50, 3), 123, dtype=np.uint8)
    Image.fromarray(new).save(tmp_path / "images" / "train" / "img0.png")

    fresh = _build(tmp_path)
    fresh.enable_image_cache("disk")
    import cv2

    expected = cv2.imread(str(tmp_path / "images" / "train" / "img0.png"))
    assert np.array_equal(fresh.load_image(0), expected)


# ---------------------------------------------------------------------------
# Post-resize cache point (load_resized_img)
# ---------------------------------------------------------------------------


class _WantsUnresized:
    """Minimal stand-in for transforms that own all resizing (e.g. RT-DETR)."""

    wants_unresized_image = True

    def __call__(self, img, targets, input_dim):
        return img, targets


def test_resized_cache_bit_identical_across_modes(tmp_path):
    """The cached resized image must be byte-equal to a fresh decode+resize."""
    _write_files(tmp_path)
    ref = _build(tmp_path)
    ref.enable_image_cache(False)
    expected = [ref.load_resized_img(i).copy() for i in range(len(ref))]

    for mode in ("ram", "disk"):
        ds = _build(tmp_path)
        ds.enable_image_cache(mode)
        for i in range(len(ds)):
            first = ds.load_resized_img(i)   # fill
            second = ds.load_resized_img(i)  # cached read
            assert np.array_equal(first, expected[i]), mode
            assert np.array_equal(second, expected[i]), mode


def test_resized_ram_cache_returns_copies(tmp_path):
    """In-place augmentation downstream must not corrupt the cache."""
    ds = _make_dataset(tmp_path)
    ds.enable_image_cache("ram")
    clean = ds.load_resized_img(0).copy()
    ds.load_resized_img(0)[:] = 0  # simulate an in-place augmentation
    assert np.array_equal(ds.load_resized_img(0), clean)


def test_resized_disk_sidecar_is_keyed_by_target_size(tmp_path):
    """Two runs at different imgsz must not read each other's pixels."""
    _write_files(tmp_path)

    small = YOLODataset(data_dir=str(tmp_path), split="train", img_size=(64, 64))
    small.enable_image_cache("disk")
    small_img = small.load_resized_img(0)

    big = YOLODataset(data_dir=str(tmp_path), split="train", img_size=(128, 128))
    big.enable_image_cache("disk")
    big_img = big.load_resized_img(0)

    image_dir = tmp_path / "images" / "train"
    assert (image_dir / "img0.png.r64x64.npy").exists()
    assert (image_dir / "img0.png.r128x128.npy").exists()
    assert small_img.shape != big_img.shape

    # Cached re-reads still resolve to their own size.
    assert np.array_equal(small.load_resized_img(0), small_img)
    assert np.array_equal(big.load_resized_img(0), big_img)


def test_resized_disk_cache_invalidates_on_source_change(tmp_path):
    _write_files(tmp_path)
    disk = _build(tmp_path)
    disk.enable_image_cache("disk")
    _ = disk.load_resized_img(0)  # populate cache

    time.sleep(1.1)  # ensure a newer mtime than the .npy
    new = np.full((40, 50, 3), 123, dtype=np.uint8)
    Image.fromarray(new).save(tmp_path / "images" / "train" / "img0.png")

    fresh_ref = _build(tmp_path)
    fresh_ref.enable_image_cache(False)
    expected = fresh_ref.load_resized_img(0)

    fresh = _build(tmp_path)
    fresh.enable_image_cache("disk")
    assert np.array_equal(fresh.load_resized_img(0), expected)


def test_pull_item_identical_across_cache_modes(tmp_path):
    """End to end: the tensors entering augmentation are identical with and
    without the cache, for both the resized path and the unresized opt-in."""
    _write_files(tmp_path)

    for preproc in (None, _WantsUnresized()):
        ref = YOLODataset(
            data_dir=str(tmp_path), split="train", img_size=(64, 64), preproc=preproc
        )
        ref.enable_image_cache(False)
        expected = [ref.pull_item(i)[0].copy() for i in range(len(ref))]

        for mode in ("ram", "disk"):
            ds = YOLODataset(
                data_dir=str(tmp_path), split="train", img_size=(64, 64), preproc=preproc
            )
            ds.enable_image_cache(mode)
            for i in range(len(ds)):
                _ = ds.pull_item(i)  # fill
                assert np.array_equal(ds.pull_item(i)[0], expected[i]), (
                    mode,
                    type(preproc).__name__,
                )


def test_resized_path_does_not_fill_decode_cache(tmp_path):
    """Caching both the decode and its resized product would double the
    footprint for no reuse; the resized path must write only its own entry."""
    ds = _make_dataset(tmp_path)
    ds.enable_image_cache("ram")
    _ = ds.load_resized_img(0)
    assert ds._resized_ram_cache[0] is not None
    assert ds._ram_cache[0] is None

    disk = _build(tmp_path)
    disk.enable_image_cache("disk")
    _ = disk.load_resized_img(1)
    image_dir = tmp_path / "images" / "train"
    assert (image_dir / "img1.png.r64x64.npy").exists()
    assert not (image_dir / "img1.png.npy").exists()
