"""Gated parity between native and Transformers OMDet-Turbo."""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.openvocab,
    pytest.mark.external_data,
    pytest.mark.network,
]


@pytest.mark.skipif(
    not os.getenv("LIBREYOLO_OMDET_PARITY"),
    reason="set LIBREYOLO_OMDET_PARITY=1 to run OMDet-Turbo parity (downloads weights)",
)
def test_omdet_native_matches_transformers():
    from weights.parity_omdet_turbo import main

    main()
