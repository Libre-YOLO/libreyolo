"""Per-family inference profiler support (``libreyolo profile infer``).

``InferenceProfiler`` duck-types on the shared predict-stage contract
(``_preprocess`` / ``_forward`` / ``_postprocess``), so in principle any
detection wrapper works. This file proves it per family instead of assuming
it: every g0/g1 family plus the g2 detect/point families runs a short
profiled inference window and must emit a valid ``profile.json`` with the
stage split and latency percentiles.

Classification / semantic / restoration predict paths are documented as
best-effort for ``profile infer`` (their postprocess semantics differ from
the detect contract this tool measures), so they are not enrolled here.
Training-side profiler coverage for those lives in
``test_profile_training_families.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

# The g2 detect/point families (the rest of g2 is classify/semantic/restore).
G2_DETECT = {
    "yolox": ("LibreYOLOX", "t", None, {}),
    "yolo7": ("LibreYOLO7", "b", None, {}),
    "rtmdet": ("LibreRTMDet", "t", None, {}),
    "picodet": ("LibrePICODET", "s", None, {}),
    "fomo": ("LibreFOMO", "s", None, {"task": "point"}),
}

# (family id, class name, size, ctor model_path, ctor kwargs)
CASES = [
    # g0
    ("yolo9", "LibreYOLO9", "t", None, {}),
    ("rfdetr", "LibreRFDETR", "n", {}, {}),
    # g1
    ("yolo9_e2e", "LibreYOLO9E2E", "t", None, {}),
    ("yolo9_p2", "LibreYOLO9P2", "t", None, {}),
    ("ec", "LibreEC", "s", None, {}),
    ("rtdetr", "LibreRTDETR", "r18", None, {}),
    ("rtdetrv2", "LibreRTDETRv2", "r18", None, {}),
    ("rtdetrv4", "LibreRTDETRv4", "s", None, {}),
    ("dfine", "LibreDFINE", "n", None, {}),
    ("deim", "LibreDEIM", "n", None, {}),
    ("deimv2", "LibreDEIMv2", "atto", None, {}),
    ("yolonas", "LibreYOLONAS", "s", None, {}),
    # g2 detect/point
    *[(fam, *spec) for fam, spec in G2_DETECT.items()],
]


@pytest.fixture(scope="module")
def sample_image(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("profile_infer_data")
    rng = np.random.default_rng(0)
    path = root / "img.jpg"
    Image.fromarray(
        rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)
    ).save(path)
    return str(path)


@pytest.mark.external_data  # several families fetch a pretrained backbone
@pytest.mark.parametrize(
    "family,class_name,size,ctor,ctor_kwargs",
    CASES,
    ids=[case[0] for case in CASES],
)
def test_infer_profile_emits_report(
    family, class_name, size, ctor, ctor_kwargs, sample_image, tmp_path
):
    import libreyolo
    from libreyolo.profiling import InferenceProfiler

    torch.manual_seed(0)
    model = getattr(libreyolo, class_name)(ctor, size, **ctor_kwargs)
    prof = InferenceProfiler(
        model,
        warmup=2,
        runs=6,
        batch=1,
        trace=True,
        save_dir=tmp_path,
        meta={"model": family},
    )
    analysis = prof.run([sample_image])

    profile_json = tmp_path / "profile.json"
    assert profile_json.exists(), f"{family}: profile.json was not emitted"
    out = json.loads(profile_json.read_text(encoding="utf-8"))
    assert out.get("schema") == "libreyolo.profile.analysis/v1", f"{family}"
    assert out.get("mode") == "inference", f"{family}"

    stages = analysis.get("stages_ms") or {}
    assert stages.get("forward", 0) > 0, f"{family}: no forward time: {stages}"
    lat = analysis.get("latency") or {}
    assert lat.get("p50_ms") and lat["p50_ms"] > 0, f"{family}: no latency"
    assert analysis.get("img_per_s") and analysis["img_per_s"] > 0, f"{family}"
    assert analysis.get("bound"), f"{family}: no verdict"


def test_enrollment_tracks_the_registry():
    """g0/g1 entirely, plus the g2 detect/point subset, must stay enrolled."""
    from libreyolo.models.registry import families_in

    enrolled = {case[0] for case in CASES}
    expected = (
        set(families_in("g0")) | set(families_in("g1")) | set(G2_DETECT)
    )
    assert enrolled == expected, (
        f"inference-profiler e2e coverage drifted: "
        f"missing={sorted(expected - enrolled)} extra={sorted(enrolled - expected)}"
    )
    # If a G2_DETECT family is renamed/regrouped, fail here rather than
    # silently testing a ghost.
    assert set(G2_DETECT) <= set(families_in("g2"))
