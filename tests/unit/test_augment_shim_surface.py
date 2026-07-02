"""Shim-surface superset test for the augmentations refactor.

``tests/unit/fixtures/augment_shim_surface.json`` records the full runtime
attribute surface (``dir()`` minus dunders) of every module the refactor
turned into a re-export shim, captured from the pre-refactor tree
(dev @ c1bdf813). Each shim must expose a superset: a dropped attribute means
some import or monkeypatch dotted path that used to work no longer does.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SURFACE_FILE = Path(__file__).parent / "fixtures" / "augment_shim_surface.json"

with open(SURFACE_FILE) as f:
    RECORDED = json.load(f)


@pytest.mark.parametrize("module_name", sorted(RECORDED))
def test_shim_exposes_historical_surface(module_name):
    mod = importlib.import_module(module_name)
    current = {a for a in dir(mod) if not a.startswith("__")}
    missing = set(RECORDED[module_name]) - current
    assert not missing, (
        f"{module_name} lost historical attributes {sorted(missing)}; "
        "re-export them from the shim so existing imports and monkeypatch "
        "dotted paths keep working."
    )
