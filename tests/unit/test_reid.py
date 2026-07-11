"""Tests for the OSNet ReID embedder (libreyolo.tracking.reid).

Architecture parity is proven against Torchreid's ``osnet_ain`` without
shipping weights: both networks are filled with the same *procedurally
generated* state dict (deterministic values derived from each parameter's
name), fed a fixed input, and must produce the same features. The committed
golden fixture records the Torchreid output
(``scratchpad make_osnet_arch_fixture.py`` regenerates it); the live test
re-derives it from a Torchreid checkout when ``OSNET_REF_PATH`` is set.
"""

from __future__ import annotations

import importlib.util
import json
import os
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

from libreyolo.tracking.reid import (
    _OSNET_CHANNELS,
    OSNet,
    OSNetEmbedder,
    _load_osnet_state_dict,
)

pytestmark = pytest.mark.unit

GOLDEN_PATH = Path(__file__).parent / "data" / "osnet_arch_golden.json"
VARIANT = "osnet_ain_x0_25"


# --------------------------------------------------------------------------
# Procedural weights (shared with the fixture maker)
# --------------------------------------------------------------------------


def procedural_state_dict(reference_sd: dict) -> dict:
    """Deterministic parameter values derived from each key's name.

    Same-named parameters get identical values in any architecture, so two
    structurally-identical networks loaded from this dict must agree.
    """
    out = {}
    for key, tensor in reference_sd.items():
        if key.endswith("num_batches_tracked"):
            out[key] = torch.zeros_like(tensor)
            continue
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        values = rng.normal(0.0, 0.1, size=tuple(tensor.shape))
        if key.endswith("running_var"):
            values = np.abs(values) + 0.5  # variances must be positive
        out[key] = torch.as_tensor(values, dtype=tensor.dtype)
    return out


def fixed_input() -> torch.Tensor:
    rng = np.random.default_rng(42)
    return torch.as_tensor(
        rng.normal(0.0, 1.0, size=(2, 3, 256, 128)), dtype=torch.float32
    )


def our_features() -> np.ndarray:
    model = OSNet(_OSNET_CHANNELS[VARIANT]).eval()
    model.load_state_dict(procedural_state_dict(model.state_dict()))
    with torch.no_grad():
        return model(fixed_input()).numpy()


# --------------------------------------------------------------------------
# Architecture parity
# --------------------------------------------------------------------------


def test_osnet_arch_golden():
    """Our OSNet reproduces Torchreid's recorded output on procedural weights."""
    if not GOLDEN_PATH.exists():
        pytest.skip(f"golden fixture missing at {GOLDEN_PATH}")
    golden = json.loads(GOLDEN_PATH.read_text())
    assert golden["variant"] == VARIANT
    expected = np.array(golden["features"], dtype=np.float32)
    got = our_features()
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def _import_torchreid_osnet_ain():
    """Load Torchreid's osnet_ain module (self-contained file), or skip."""
    ref_path = os.environ.get("OSNET_REF_PATH")
    candidates = [ref_path] if ref_path else []
    candidates.append(
        str(Path(__file__).resolve().parents[2] / "parity_ref" / "deep-person-reid")
    )
    for cand in candidates:
        if not cand:
            continue
        src = Path(cand) / "torchreid" / "models" / "osnet_ain.py"
        if not src.exists():
            continue
        spec = importlib.util.spec_from_file_location("osnet_ain_ref", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    pytest.skip("Torchreid osnet_ain reference not found (set OSNET_REF_PATH)")


def test_live_torchreid_parity():
    """Same procedural weights + input through Torchreid produce same features."""
    ref = _import_torchreid_osnet_ain()
    ref_model = getattr(ref, VARIANT)(num_classes=7, pretrained=False).eval()
    ref_sd = {
        k: v for k, v in ref_model.state_dict().items()
        if not k.startswith("classifier.")
    }
    ours = OSNet(_OSNET_CHANNELS[VARIANT])
    assert set(ours.state_dict().keys()) == set(ref_sd.keys()), (
        "state-dict key drift vs Torchreid"
    )
    filled = procedural_state_dict(ref_model.state_dict())
    ref_model.load_state_dict(filled)
    with torch.no_grad():
        expected = ref_model(fixed_input()).numpy()
    np.testing.assert_allclose(our_features(), expected, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------


def test_load_state_dict_handles_torchreid_checkpoint(tmp_path):
    """Raw Torchreid checkpoints (wrapper, module. prefix, classifier) load."""
    model = OSNet(_OSNET_CHANNELS[VARIANT])
    sd = {f"module.{k}": v for k, v in model.state_dict().items()}
    sd["module.classifier.weight"] = torch.zeros(10, 512)
    sd["module.classifier.bias"] = torch.zeros(10)
    path = tmp_path / "ckpt.pth.tar"
    torch.save({"state_dict": sd, "epoch": 50}, path)

    loaded = _load_osnet_state_dict(path)
    assert set(loaded.keys()) == set(model.state_dict().keys())
    model.load_state_dict(loaded)


# --------------------------------------------------------------------------
# Embedder behaviour
# --------------------------------------------------------------------------


@pytest.fixture()
def embedder(tmp_path):
    """An OSNetEmbedder over procedural weights (no download)."""
    model = OSNet(_OSNET_CHANNELS[VARIANT])
    path = tmp_path / f"{VARIANT}.pt"
    torch.save(procedural_state_dict(model.state_dict()), path)
    return OSNetEmbedder(variant=VARIANT, weights=path, device="cpu")


def test_embedder_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown OSNet variant"):
        OSNetEmbedder(variant="osnet_x1_0_typo", weights="unused.pt")


def test_embedder_output_shape_and_normalization(embedder):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    boxes = np.array(
        [
            [10, 20, 90, 220],
            [300, 100, 380, 300],
            [-15, -10, 50, 120],  # partially outside: clipped
            [600, 400, 900, 700],  # extends past the frame: clipped
        ],
        dtype=np.float64,
    )
    embs = embedder(image, boxes)
    assert embs.shape == (4, 512)
    assert embs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-5)


def test_embedder_degenerate_box_is_safe(embedder):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = np.array([[700, 500, 720, 520]], dtype=np.float64)  # fully outside
    embs = embedder(image, boxes)
    assert embs.shape == (1, 512)
    assert np.all(np.isfinite(embs))


def test_embedder_empty_boxes(embedder):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    embs = embedder(image, np.empty((0, 4)))
    assert embs.shape == (0, 512)


def test_embedder_deterministic(embedder):
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    boxes = np.array([[50, 50, 150, 250]], dtype=np.float64)
    a = embedder(image, boxes)
    b = embedder(image, boxes)
    np.testing.assert_array_equal(a, b)
