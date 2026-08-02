"""SSD family shell, naming, and maturity tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_family_metadata_and_filename_detection():
    from libreyolo import LibreSSD

    assert LibreSSD.FAMILY == "ssd"
    assert LibreSSD.FILENAME_PREFIX == "LibreSSD"
    assert LibreSSD.INPUT_SIZES == {"300": 300}
    assert LibreSSD.SUPPORTED_TASKS == ("detect",)
    assert LibreSSD.DEFAULT_TASK == "detect"
    assert LibreSSD.TRAIN_CONFIG is None
    assert LibreSSD.detect_size_from_filename("LibreSSD300.pt") == "300"
    assert LibreSSD.detect_size_from_filename("LibreSSD30.pt") is None


def test_family_constructs_and_training_is_explicitly_unavailable():
    from libreyolo import LibreSSD

    model = LibreSSD(None, size="300", device="cpu")
    assert model.family == "ssd"
    assert model.model.num_classes == 91
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="coco128.yaml")
