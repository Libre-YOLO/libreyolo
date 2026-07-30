"""Face-embedding Core ML conversion/runtime parity on Apple hardware."""

from __future__ import annotations

import gc
import sys

import numpy as np
import pytest

pytestmark = [
    pytest.mark.coreml,
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
    pytest.mark.general_nightly,
]

if sys.platform != "darwin":  # pragma: no cover - platform gate
    pytest.skip("Core ML artifacts only run on macOS", allow_module_level=True)

pytest.importorskip(
    "coremltools",
    reason="Core ML parity requires the coremltools runtime",
)


def _aligned_probes() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:112, :112]
    first = np.stack(
        (
            (3 * xx + 5 * yy + 17) % 256,
            (11 * xx + 7 * yy + 53) % 256,
            (13 * xx + 19 * yy + 101) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    second = np.stack(
        (
            (23 * xx + 2 * yy + 211) % 256,
            (5 * xx + 29 * yy + 37) % 256,
            (17 * xx + 31 * yy + 149) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return first, second


def test_facerec_official_coreml_raw_public_and_gallery_parity(tmp_path):
    from libreyolo import FaceGallery, LibreFaceEmbedder, LibreYOLO
    from libreyolo.models.facerec.preprocess import preprocess_aligned

    source = LibreYOLO("librefacerec-l", device="cpu")
    probes = _aligned_probes()
    blobs = [preprocess_aligned(probe, source.cfg) for probe in probes]
    expected_raw = [
        np.asarray(
            source.session.run(None, {source.input_name: blob})[0],
            dtype=np.float32,
        )
        for blob in blobs
    ]
    expected = source.embed_aligned(list(probes))

    artifact = source.export(
        format="coreml",
        half=False,
        output_path=tmp_path / "librefacerec-l.mlpackage",
        compute_units="cpu_only",
    )
    del source
    gc.collect()
    deployed = LibreYOLO(artifact, compute_units="cpu_only")
    assert isinstance(deployed, LibreFaceEmbedder)
    actual_raw = [
        np.asarray(
            deployed.session.run(None, {deployed.input_name: blob})[0],
            dtype=np.float32,
        )
        for blob in blobs
    ]
    repeated_raw = [
        np.asarray(
            deployed.session.run(None, {deployed.input_name: blob})[0],
            dtype=np.float32,
        )
        for blob in blobs
    ]
    for actual_value, repeated_value in zip(actual_raw, repeated_raw):
        np.testing.assert_array_equal(actual_value, repeated_value)

    worst = 0.0
    for expected_value, actual_value in zip(expected_raw, actual_raw):
        scale = max(float(np.abs(expected_value).max()), 1e-12)
        worst = max(
            worst,
            float(np.abs(actual_value - expected_value).max()) / scale,
        )
    assert worst <= 3e-4
    sensitivity_scale = max(
        float(np.abs(expected_raw[0]).max()),
        float(np.abs(expected_raw[1]).max()),
        1e-12,
    )
    relative_sensitivity = (
        float(np.abs(expected_raw[1] - expected_raw[0]).max())
        / sensitivity_scale
    )
    assert relative_sensitivity > max(worst * 100.0, 1e-6)

    actual = deployed.embed_aligned(list(probes))
    repeated = deployed.embed_aligned(list(probes))
    np.testing.assert_array_equal(actual, repeated)
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(
        np.linalg.norm(actual, axis=1),
        np.ones(2, dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )
    assert float(actual[0] @ actual[1]) == pytest.approx(
        float(expected[0] @ expected[1]),
        abs=3e-4,
    )

    gallery = FaceGallery(embedder=deployed)
    gallery.enroll_embedding("probe", actual[0])
    assert gallery.match(actual[0], threshold=0.99)[0][0][0] == "probe"
