"""Tests for the panoptic-segmentation task scaffolding (issue #555).

These cover the API surface only: task registration, the PanopticSegmentation
result payload, and that PanopticValidator is importable/dispatchable but its
metric hooks are still unimplemented. The PQ metric and COCO-panoptic loader
are intentionally not tested here because they are not implemented yet.
"""

import numpy as np
import pytest

from libreyolo.tasks import (
    TASKS,
    normalize_task,
    suffix_to_task,
    task_suffix_pattern,
    task_to_suffix,
)

pytestmark = pytest.mark.unit


def test_panoptic_task_registered():
    assert "panoptic" in TASKS
    assert normalize_task("panoptic") == "panoptic"


@pytest.mark.parametrize(
    "alias",
    ["panoptic", "panoptic-segmentation", "panoptic_segmentation", "panseg", "pano"],
)
def test_panoptic_aliases(alias):
    assert normalize_task(alias) == "panoptic"


def test_panoptic_suffix_roundtrip():
    assert task_to_suffix("panoptic") == "panoptic"
    assert suffix_to_task("panoptic") == "panoptic"
    assert suffix_to_task("-panoptic") == "panoptic"


def test_panoptic_suffix_in_pattern():
    pattern = task_suffix_pattern(["panoptic"])
    assert pattern == "-panoptic"
    # Longest-first ordering keeps -panoptic from being masked by shorter suffixes.
    multi = task_suffix_pattern(["panoptic", "point", "pose"])
    assert multi.split("|")[0] == "-panoptic"


def test_panoptic_segmentation_payload_basics():
    from libreyolo.utils.results import PanopticSegmentation

    data = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
    segments_info = [
        {"id": 1, "category_id": 0, "isthing": True},
        {"id": 2, "category_id": 15, "isthing": False},
    ]
    pan = PanopticSegmentation(data, segments_info)

    assert pan.orig_shape == (2, 3)
    assert pan.segment_ids == [1, 2]  # 0 is void, excluded
    assert pan.segment_mask(2).tolist() == [[False, False, False], [True, True, False]]


def test_panoptic_payload_rejects_non_2d():
    from libreyolo.utils.results import PanopticSegmentation

    with pytest.raises(ValueError):
        PanopticSegmentation(np.zeros((2, 3, 4), dtype=np.int32))


def test_panoptic_payload_preserves_segments_info_across_moves():
    from libreyolo.utils.results import PanopticSegmentation

    data = np.zeros((4, 4), dtype=np.int32)
    data[1, 1] = 7
    info = [{"id": 7, "category_id": 3, "isthing": True}]
    pan = PanopticSegmentation(data, info)

    # segments_info is plain-Python metadata and must survive device/array moves
    # and whole-image slicing.
    assert pan.cpu().segments_info == info
    assert pan.numpy().segments_info == info
    assert pan[0].segments_info == info
    # Slicing must not collapse the dense (H, W) map.
    assert pan[0].data.shape == (4, 4)


def test_results_panoptic_slot_roundtrips():
    from libreyolo.utils.results import PanopticSegmentation, Results

    data = np.zeros((3, 3), dtype=np.int32)
    data[0, 0] = 5
    info = [{"id": 5, "category_id": 1, "isthing": True}]
    result = Results(
        boxes=None,
        orig_shape=(3, 3),
        panoptic=PanopticSegmentation(data, info),
    )
    assert result.panoptic is not None
    assert "panoptic" in repr(result)
    # cpu()/numpy() rebuild Results via _new; the slot and its metadata survive.
    assert result.cpu().panoptic.segments_info == info
    assert result[0].panoptic.data.shape == (3, 3)


def test_panoptic_validator_importable_but_unimplemented():
    from libreyolo.validation import PanopticValidator, ValidationConfig

    assert PanopticValidator.task == "panoptic"

    class _StubModel:
        task = "panoptic"
        nb_classes = 133

    config = ValidationConfig(data="dummy.yaml", device="cpu")
    validator = PanopticValidator(model=_StubModel(), config=config)
    # Every metric hook is scaffolding and must fail loudly until implemented.
    with pytest.raises(NotImplementedError):
        validator._init_metrics()
    with pytest.raises(NotImplementedError):
        validator._compute_metrics()


def test_panoptic_validator_exported_from_package():
    import libreyolo

    assert hasattr(libreyolo, "PanopticValidator")
    assert hasattr(libreyolo, "PanopticSegmentation")
