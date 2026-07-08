"""Gated end-to-end tests for the LibreBiRefNet matte family.

These need real weights. ``LibreBiRefNetl-matte.pt`` auto-downloads from the
LibreYOLO Hugging Face org (``LibreYOLO/LibreBiRefNetl-matte``); the lite ``t``
tests use a locally converted ``weights/LibreBiRefNett-matte.pt`` when present.
Each test skips cleanly when its weights cannot be obtained (offline CI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
FIXT = REPO / "tests" / "fixtures" / "matte8"
_SAMPLE = REPO / "libreyolo" / "assets" / "parkour.jpg"
_LOCAL_T = REPO / "weights" / "LibreBiRefNett-matte.pt"


def _load(name: str, device: str = "cpu"):
    from libreyolo import LibreYOLO

    try:
        return LibreYOLO(name, device=device)
    except Exception as exc:  # network/weights unavailable
        pytest.skip(f"Could not obtain {name}: {exc}")


def test_matte_predict_cutout_save_autodownload(tmp_path):
    """l (general) auto-downloads from HF and the full matte UX works."""
    model = _load("LibreBiRefNetl-matte.pt")
    assert model.FAMILY == "birefnet" and model.size == "l" and model.task == "matte"

    res = model.predict(str(_SAMPLE), verbose=False)[0]
    assert res.matte is not None
    w, h = Image.open(_SAMPLE).size
    matte = res.matte.array
    assert matte.shape == (h, w)  # original canvas
    assert 0.0 <= float(matte.min()) and float(matte.max()) <= 1.0

    cut = res.cutout()
    assert cut.shape == (h, w, 4) and cut.dtype == np.uint8

    out = tmp_path / "cutout.png"
    res.save(out)
    saved = Image.open(out)
    assert saved.mode == "RGBA" and saved.size == (w, h)


@pytest.mark.skipif(not _LOCAL_T.exists(), reason="lite weights not staged locally")
def test_matte_golden_stable_cpu():
    """OUR CPU t-model output reproduces the pinned goldens (pipeline is stable)."""
    model = _load(str(_LOCAL_T))
    for name in ("product_circle", "fine_hair_star", "portrait_silhouette"):
        img = FIXT / "images" / f"{name}.png"
        golden = np.asarray(Image.open(FIXT / "goldens_t" / f"{name}.png").convert("L"), np.int16)
        pred = model.predict(str(img), verbose=False)[0].matte.array
        pred_u8 = np.rint(np.clip(pred, 0, 1) * 255).astype(np.int16)
        max_abs = int(np.abs(pred_u8 - golden).max())
        assert max_abs <= 2, f"{name}: golden drift max_abs={max_abs} (>2 LSB)"


@pytest.mark.skipif(not _LOCAL_T.exists(), reason="lite weights not staged locally")
def test_matte_matches_official_reference():
    """OUR matte agrees with the official reference mattes on matte8 (MAE small)."""
    from libreyolo.validation.matte_validator import matte_mae

    model = _load(str(_LOCAL_T))
    for name in ("product_circle", "portrait_silhouette", "two_shapes"):
        img = FIXT / "images" / f"{name}.png"
        ref = np.asarray(Image.open(FIXT / "mattes" / f"{name}.png").convert("L"), np.float32) / 255.0
        pred = model.predict(str(img), verbose=False)[0].matte.array
        assert matte_mae(pred, ref) <= 0.02
