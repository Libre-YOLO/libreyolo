"""Gated real-weight smoke tests for LibreOpenVocab."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.openvocab,
]


def _sample_image():
    from libreyolo import SAMPLE_IMAGE

    return SAMPLE_IMAGE


def _assert_detection_smoke(result, expected_names):
    assert result.orig_shape[0] > 0
    assert result.orig_shape[1] > 0
    assert result.names == expected_names
    assert len(result) > 0
    cls = result.boxes.cls.int().tolist()
    assert cls
    assert all(0 <= class_id < len(expected_names) for class_id in cls)


def test_grounding_dino_tiny_predict_smoke():
    pytest.importorskip("transformers")
    from libreyolo import LibreOpenVocab

    model = LibreOpenVocab("grounding-dino-tiny", device="cpu")
    model.set_classes(["person", "dog", "skateboard"])
    result = model.predict(_sample_image(), conf=0.2, text_threshold=0.2)
    _assert_detection_smoke(result, {0: "person", 1: "dog", 2: "skateboard"})


def test_owlv2_base_predict_smoke():
    pytest.importorskip("transformers")
    from libreyolo import LibreOpenVocab

    model = LibreOpenVocab("owlv2", device="cpu")
    model.set_classes(["person", "dog", "skateboard"])
    result = model.predict(_sample_image(), conf=0.1)
    _assert_detection_smoke(result, {0: "person", 1: "dog", 2: "skateboard"})


def test_openvocab_val_raises_in_v1():
    pytest.importorskip("transformers")
    from libreyolo import LibreOpenVocab

    model = LibreOpenVocab("grounding-dino-tiny", device="cpu")
    with pytest.raises(NotImplementedError, match="dedicated validator"):
        model.val(data="coco128.yaml")
