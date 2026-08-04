"""Default-nightly scope contract tests."""

import pytest

from tests.e2e.nightly_contract import nightly_advanced_marker_conflicts

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "advanced",
    [
        "cuda_graph",
        "executorch",
        "export_backend",
        "extended_training",
        "training_nightly",
    ],
)
def test_advanced_features_cannot_overlap_default_nightly(advanced):
    assert nightly_advanced_marker_conflicts({"e2e", "general_nightly", advanced}) == (
        advanced,
    )


def test_non_nightly_advanced_features_remain_opt_in():
    assert nightly_advanced_marker_conflicts({"e2e", "cuda_graph"}) == ()


def test_core_nightly_markers_remain_allowed():
    assert (
        nightly_advanced_marker_conflicts({"e2e", "flagship_nightly", "rfdetr"}) == ()
    )
