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


@pytest.fixture(scope="module")
def omdet_model():
    pytest.importorskip("transformers")
    from libreyolo import LibreOpenVocab

    return LibreOpenVocab("omdet-turbo", device="cpu")


@pytest.mark.general_nightly
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


@pytest.mark.general_nightly
def test_omdet_turbo_matches_transformers_postprocessing(omdet_model):
    from PIL import Image
    import torch
    from transformers import AutoProcessor, OmDetTurboForObjectDetection

    classes = ["person", "dog", "skateboard"]
    image = Image.open(_sample_image()).convert("RGB")
    omdet_model.set_classes(classes)

    captured = {}

    def capture_forward(_module, _args, kwargs, output):
        captured["inputs"] = kwargs
        captured["output"] = output

    handle = omdet_model.model.register_forward_hook(capture_forward, with_kwargs=True)
    try:
        result = omdet_model.predict(image, conf=0.3)
    finally:
        handle.remove()

    _assert_detection_smoke(result, {0: "person", 1: "dog", 2: "skateboard"})

    assert isinstance(omdet_model.model, OmDetTurboForObjectDetection)
    snapshot_dir = omdet_model._ensure_weights()
    raw_processor = AutoProcessor.from_pretrained(snapshot_dir, local_files_only=True)
    raw_inputs = raw_processor(
        images=image,
        text=classes,
        task="Detect person, dog, skateboard.",
        return_tensors="pt",
    )
    raw_inputs = omdet_model._prepare_inputs(raw_inputs)
    assert set(captured["inputs"]) == set(raw_inputs)
    for key, expected in raw_inputs.items():
        torch.testing.assert_close(captured["inputs"][key], expected)

    width, height = image.size
    raw = raw_processor.post_process_grounded_object_detection(
        captured["output"],
        text_labels=classes,
        threshold=0.3,
        nms_threshold=0.5,
        target_sizes=[(height, width)],
    )[0]
    order = raw["scores"].argsort(descending=True)[:300]
    expected_ids = torch.tensor(
        [classes.index(label) for label in raw["text_labels"]],
        dtype=torch.int64,
    )[order]

    assert len(result) == len(order)
    torch.testing.assert_close(
        result.boxes.xyxy.cpu(), raw["boxes"][order].cpu(), rtol=0, atol=1e-5
    )
    torch.testing.assert_close(
        result.boxes.conf.cpu(), raw["scores"][order].cpu(), rtol=0, atol=1e-5
    )
    assert torch.equal(result.boxes.cls.long().cpu(), expected_ids)
    assert torch.equal(raw["labels"][order].long().cpu(), expected_ids)


def test_omdet_turbo_replaces_vocabulary_and_can_return_empty(omdet_model):
    omdet_model.set_classes(["traffic light", "bicycle"])
    first = omdet_model.predict(_sample_image(), conf=0.3)
    assert first.names == {0: "traffic light", 1: "bicycle"}

    omdet_model.set_classes(["giraffe"])
    second = omdet_model.predict(_sample_image(), conf=0.5)
    assert second.names == {0: "giraffe"}
    assert len(second) == 0


def test_openvocab_val_raises_in_v1():
    pytest.importorskip("transformers")
    from libreyolo import LibreOpenVocab

    model = LibreOpenVocab("grounding-dino-tiny", device="cpu")
    with pytest.raises(NotImplementedError, match="dedicated validator"):
        model.val(data="coco128.yaml")
