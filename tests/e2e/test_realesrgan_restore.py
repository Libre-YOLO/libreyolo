"""E2E smoke for Real-ESRGAN super-resolution restore (real weights, autodownload)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from libreyolo import LibreYOLO

from .conftest import require_test_weights

pytestmark = pytest.mark.e2e


def _image(h, w, seed=0):
    rng = np.random.default_rng(seed)
    low = rng.random((h // 8 + 1, w // 8 + 1, 3)).astype(np.float32)
    img = Image.fromarray((low * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return img.convert("RGB")


def test_realesrgan_x4t_predict_upscales_4x():
    weights = require_test_weights(
        "weights/LibreRealESRGANx4t-restore.pt", expected_family="realesrgan"
    )
    model = LibreYOLO(weights, device="cpu")
    img = _image(40, 56)
    result = model.predict(img)
    assert result.restore_scale == 4
    assert result.restored.array.shape == (160, 224, 3)
    assert result.restored.array.dtype == np.uint8


def test_realesrgan_x4t_tiled_matches_untiled_with_adequate_pad():
    # SRVGG's effective receptive field is small; tile_pad=48 makes the tiled
    # output bit-identical to the untiled result (see the family NOTICE and
    # docs). The default tile_pad=10 reproduces upstream Real-ESRGAN exactly.
    weights = require_test_weights(
        "weights/LibreRealESRGANx4t-restore.pt", expected_family="realesrgan"
    )
    model = LibreYOLO(weights, device="cpu")
    img = _image(96, 96, seed=3)
    untiled = model.predict(img).restored.array.astype(int)
    tiled = model.predict(img, tile=48, tile_pad=48).restored.array.astype(int)
    assert int(np.abs(untiled - tiled).max()) <= 1


def test_nafnet_sidd_denoise_weights_load_and_run():
    weights = require_test_weights(
        "weights/LibreNAFNetl-restore-sidd.pt", expected_family="nafnet"
    )
    model = LibreYOLO(weights, device="cpu")
    img = _image(64, 64, seed=5)
    result = model.predict(img)
    assert result.restore_scale == 1
    assert result.restored.array.shape == (64, 64, 3)
