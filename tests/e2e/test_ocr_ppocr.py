"""Gated end-to-end tests for the LibrePPOCR ocr family.

These need real weights. ``LibrePPOCRt-ocr.pt`` auto-downloads from the
LibreYOLO Hugging Face org (``LibreYOLO/LibrePPOCRt-ocr``); the golden tests
use a locally staged ``weights/LibrePPOCRt-ocr.pt`` when present. Each test
skips cleanly when its weights cannot be obtained (offline CI).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
FIXT = REPO / "tests" / "fixtures" / "ocr20"
_LOCAL_T = REPO / "weights" / "LibrePPOCRt-ocr.pt"


def _load(name: str, device: str = "cpu"):
    from libreyolo import LibreYOLO

    try:
        return LibreYOLO(name, device=device)
    except Exception as exc:  # network/weights unavailable
        pytest.skip(f"Could not obtain {name}: {exc}")


def test_ocr_predict_autodownload(tmp_path):
    """t (mobile) auto-downloads from HF and the full ocr UX works."""
    model = _load("LibrePPOCRt-ocr.pt")
    assert model.FAMILY == "ppocr" and model.size == "t" and model.task == "ocr"
    assert model.charset is not None and model.charset[0] == "blank"

    img = FIXT / "images" / "val" / "ocr_01.png"
    res = model(str(img), conf=0.3, save=True, output_path=str(tmp_path))
    assert res.ocr is not None and len(res.ocr) == 2
    assert res.ocr.texts == ["OPEN 24 HOURS", "MAIN STREET 7"]
    assert res.ocr.data.shape == (2, 4, 2)
    assert float(res.ocr.conf.min()) > 0.5
    saved = list(tmp_path.glob("*.jpg"))
    assert saved, "annotated image was not written"


def test_ocr_empty_and_degenerate_inputs():
    model = _load("LibrePPOCRt-ocr.pt")
    blank = Image.new("RGB", (640, 480), (255, 255, 255))
    res = model(blank)
    assert res.ocr is not None and len(res) == 0

    tiny = Image.new("RGB", (6, 5), (127, 127, 127))
    assert len(model(tiny)) == 0


@pytest.mark.skipif(not _LOCAL_T.exists(), reason="ocr weights not staged locally")
def test_ocr_golden_stable_cpu():
    """OUR CPU t-model output reproduces the pinned goldens (pipeline is stable)."""
    model = _load(str(_LOCAL_T))
    golden = json.loads((FIXT / "goldens_t" / "expected.json").read_text(encoding="utf-8"))
    for name, expected in golden.items():
        res = model(str(FIXT / "images" / "val" / name), conf=0.3)
        assert res.ocr.texts == [e["text"] for e in expected], name
        for e, poly, rec in zip(expected, res.ocr.data.tolist(), res.ocr.conf.tolist()):
            drift = np.abs(np.asarray(poly) - np.asarray(e["poly"])).max()
            assert drift <= 2.0, f"{name}: polygon drift {drift:.2f} px"
            assert abs(rec - e["rec"]) <= 1e-3, f"{name}: rec-confidence drift"


@pytest.mark.skipif(not _LOCAL_T.exists(), reason="ocr weights not staged locally")
def test_ocr_validator_on_fixture():
    """val() on the synthetic fixture: the mobile tier reads it perfectly."""
    model = _load(str(_LOCAL_T))
    metrics = model.val(data=str(FIXT), conf=0.3, verbose=False)
    assert metrics["metrics/det_hmean"] >= 0.99
    assert metrics["metrics/e2e_f1"] >= 0.99
    assert metrics["metrics/rec_1-NED"] >= 0.99
