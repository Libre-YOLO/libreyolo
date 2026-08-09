"""Unit contract for the standalone DETR family skeleton."""

from __future__ import annotations

import pytest
import torch
import numpy as np

pytestmark = pytest.mark.unit


def _official_signature() -> dict[str, torch.Tensor]:
    return {
        "query_embed.weight": torch.zeros(100, 256),
        "transformer.decoder.layers.0.multihead_attn.in_proj_weight": torch.zeros(
            768, 256
        ),
        "backbone.0.body.conv1.weight": torch.zeros(64, 3, 7, 7),
        "class_embed.weight": torch.zeros(92, 256),
    }


def test_detr_registration_and_filename_contract():
    from libreyolo import LibreDETR
    from libreyolo.models.base.model import BaseModel

    assert any(cls is LibreDETR for cls in BaseModel._registry)
    assert LibreDETR.FAMILY == "detr"
    assert LibreDETR.FILENAME_PREFIX == "LibreDETR"
    assert LibreDETR.SUPPORTED_TASKS == ("detect",)
    assert LibreDETR.DEFAULT_TASK == "detect"
    assert LibreDETR.TRAIN_CONFIG is None

    for size in ("r50", "r50dc5", "r101", "r101dc5"):
        assert LibreDETR.detect_size_from_filename(f"LibreDETR{size}.pt") == size

    assert LibreDETR.detect_size_from_filename("detr-r50-e632da11.pth") == "r50"
    assert LibreDETR.detect_size_from_filename("detr-r50-dc5-f0fb7ef5.pth") == "r50dc5"
    assert LibreDETR.detect_size_from_filename("detr-r101-2c7b67e5.pth") == "r101"
    assert (
        LibreDETR.detect_size_from_filename("detr-r101-dc5-a2e86def.pth") == "r101dc5"
    )
    assert LibreDETR.detect_size_from_filename("LibreRTDETRr50.pt") is None


def test_detr_official_signature_and_class_count():
    from libreyolo import LibreDETR

    state = _official_signature()
    assert LibreDETR.can_load(state) is True
    assert LibreDETR.detect_nb_classes(state) == 80
    # Dilation is not serialized, so raw state alone cannot honestly choose
    # between r50/r50dc5 or r101/r101dc5.
    assert LibreDETR.detect_size(state) is None


def test_detr_raw_checkpoint_requires_an_unambiguous_filename(tmp_path):
    from libreyolo.models.autoconvert import autoconvert_upstream_checkpoint

    state = _official_signature()
    ambiguous = tmp_path / "renamed.pth"
    torch.save({"model": state}, ambiguous)
    assert autoconvert_upstream_checkpoint(str(ambiguous)) is None

    official = tmp_path / "detr-r50-e632da11.pth"
    torch.save({"model": state}, official)
    converted_path = autoconvert_upstream_checkpoint(str(official))
    assert converted_path is not None
    converted = torch.load(converted_path, map_location="cpu", weights_only=True)
    assert converted["model_family"] == "detr"
    assert converted["size"] == "r50"
    assert converted["task"] == "detect"
    assert converted["nc"] == 80


@pytest.mark.parametrize("size", ("r50", "r50dc5", "r101", "r101dc5"))
def test_detr_native_model_builds_and_forwards(size):
    from libreyolo import LibreDETR

    model = LibreDETR(None, size=size, nb_classes=3, device="cpu")
    assert model.family == "detr"
    assert model.task == "detect"
    model.model.eval()
    with torch.no_grad():
        output = model.model(torch.zeros(1, 3, 64, 64))
    assert output["pred_logits"].shape == (1, 100, 4)
    assert output["pred_boxes"].shape == (1, 100, 4)


def test_detr_native_model_is_inference_only():
    from libreyolo import LibreDETR

    model = LibreDETR(None, size="r50", nb_classes=3, device="cpu")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="coco128.yaml")


def test_detr_postprocess_uses_softmax_no_object_and_coco_mapping():
    from libreyolo.postprocess.detr import postprocess
    from libreyolo.utils.coco import COCO91_TO_COCO80

    logits = torch.zeros(1, 3, 92)
    logits[0, 0, 1] = 4.0  # COCO category 1 -> public class 0
    logits[0, 1, 90] = 3.0  # COCO category 90 -> public class 79
    logits[0, 2, -1] = 5.0  # no-object wins this query
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1], [0.5] * 4]])

    result = postprocess(
        {"pred_logits": logits, "pred_boxes": boxes},
        conf_thres=0.1,
        original_size=(200, 100),
        max_det=3,
        class_map=COCO91_TO_COCO80,
    )

    assert result["num_detections"] == 2
    assert result["classes"].tolist() == [0, 79]
    np.testing.assert_allclose(
        result["boxes"],
        [[80.0, 30.0, 120.0, 70.0], [40.0, 20.0, 60.0, 30.0]],
        rtol=0,
        atol=1e-5,
    )
    expected_score = torch.softmax(logits, dim=-1)[0, 0, 1].item()
    assert result["scores"][0] == pytest.approx(expected_score)


def test_detr_unmapped_coco_ids_do_not_consume_max_det():
    from libreyolo.postprocess.detr import postprocess
    from libreyolo.utils.coco import COCO91_TO_COCO80

    queries = 8
    logits = torch.full((1, queries, 92), -10.0)
    unmapped = [index for index in range(91) if index not in COCO91_TO_COCO80]
    for query in range(queries):
        logits[0, query, unmapped[query]] = 20.0
        logits[0, query, query + 1] = float(8 - query)

    result = postprocess(
        {
            "pred_logits": logits,
            "pred_boxes": torch.full((1, queries, 4), 0.5),
        },
        conf_thres=0.0,
        max_det=5,
        class_map=COCO91_TO_COCO80,
    )

    assert result["num_detections"] == 5
    assert set(result["classes"].tolist()) <= set(COCO91_TO_COCO80.values())


def test_detr_val_preprocessor_matches_inference_preprocess():
    from libreyolo.models.detr.utils import preprocess_numpy
    from libreyolo.validation.preprocessors import DETRValPreprocessor

    rng = np.random.default_rng(0)
    image_bgr = rng.integers(0, 256, (7, 5, 3), dtype=np.uint8)
    preprocessor = DETRValPreprocessor(img_size=(64, 64))

    actual, _ = preprocessor(image_bgr, np.zeros((0, 5), dtype=np.float32), (64, 64))
    expected, _ = preprocess_numpy(image_bgr[:, :, ::-1], 64)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert preprocessor.custom_normalization is True
    assert preprocessor.uses_letterbox is False
