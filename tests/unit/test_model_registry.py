"""The model-registry coverage enrollment contract.

Every family the factory registers must be enrolled in a coverage group in
``libreyolo/models/registry.py``. This is what keeps the registry maintained:
a port that forgets to enroll fails here, not in review.
"""

from __future__ import annotations

import pytest

from libreyolo.models.inventory import collect_model_inventory
from libreyolo.models.registry import GROUPS, MODEL_GROUPS, families_in, group_of

pytestmark = pytest.mark.unit


def test_every_registered_family_is_enrolled():
    inventory = collect_model_inventory()
    missing = sorted(set(inventory) - set(MODEL_GROUPS))
    assert not missing, (
        f"Families registered but not enrolled in a model group: {missing}. "
        "Add each to MODEL_GROUPS in libreyolo/models/registry.py "
        "(see docs/nomenclature.md, 'Model groups')."
    )


def test_every_enrollment_uses_a_defined_group():
    invalid = {f: g for f, g in MODEL_GROUPS.items() if g not in GROUPS}
    assert not invalid, f"Enrollments with undefined groups: {invalid}"


def test_flagship_group_is_exactly_yolo9_and_rfdetr():
    assert set(families_in("g0")) == {"yolo9", "rfdetr"}


def test_inventory_exposes_the_group():
    inventory = collect_model_inventory()
    assert inventory["yolo9"]["group"] == "g0"
    for family, entry in inventory.items():
        assert entry["group"] == group_of(family)


def test_families_in_rejects_unknown_group():
    with pytest.raises(KeyError, match="Unknown model group"):
        families_in("g9")
