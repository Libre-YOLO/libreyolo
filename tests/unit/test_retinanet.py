"""RetinaNet family skeleton and registry tests."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.retinanet]


def test_family_metadata_and_filename_detection():
    from libreyolo import LibreRetinaNet

    assert LibreRetinaNet.FAMILY == "retinanet"
    assert LibreRetinaNet.FILENAME_PREFIX == "LibreRetinaNet"
    assert LibreRetinaNet.INPUT_SIZES == {"r50": 800, "r50v2": 800}
    assert LibreRetinaNet.SUPPORTED_TASKS == ("detect",)
    assert LibreRetinaNet.DEFAULT_TASK == "detect"
    assert LibreRetinaNet.TRAIN_CONFIG is None
    assert LibreRetinaNet.detect_size_from_filename("LibreRetinaNetr50.pt") == "r50"
    assert (
        LibreRetinaNet.detect_size_from_filename("LibreRetinaNetr50v2.pt")
        == "r50v2"
    )
    assert (
        LibreRetinaNet.detect_size_from_filename(
            "retinanet_resnet50_fpn_v2_coco-5905b1c5.pth"
        )
        == "r50v2"
    )


def test_skeleton_constructs_and_training_is_explicitly_unavailable():
    from libreyolo import LibreRetinaNet

    model = LibreRetinaNet(None, size="r50", device="cpu")
    assert model.family == "retinanet"
    assert model.task == "detect"
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="coco128.yaml")
