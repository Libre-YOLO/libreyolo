"""End-to-end smoke tests for the LibreEdgeTAM promptable segmenter."""

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.sam,
    pytest.mark.network,
    pytest.mark.external_data,
]

pytest.importorskip("transformers", reason="LibreEdgeTAM requires the 'sam' extra")


@pytest.fixture(scope="module")
def edgetam_model():
    from libreyolo import LibreEdgeTAM

    return LibreEdgeTAM("edge", device="cpu")


def _assert_mask_contract(result, model):
    from libreyolo import Results

    assert isinstance(result, Results)
    assert result.masks is not None
    assert len(result.masks) >= 1
    assert result.boxes.xyxy.shape[1] == 4
    assert len(result.boxes.conf) == len(result.masks)
    assert result.names[0] == model.names[0] == "object"


def test_edgetam_real_point_prompt_returns_mask(edgetam_model):
    from libreyolo import SAMPLE_IMAGE

    result = edgetam_model.predict(SAMPLE_IMAGE, points=[640, 360], labels=[1])

    _assert_mask_contract(result, edgetam_model)


def test_edgetam_real_box_prompt_returns_mask(edgetam_model):
    from libreyolo import SAMPLE_IMAGE

    result = edgetam_model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 500, 500])

    _assert_mask_contract(result, edgetam_model)
    height, width = result.orig_shape
    assert result.masks.data.shape[-2:] == (height, width)


def test_edgetam_real_session_reuses_cache_and_reset_clears_it(edgetam_model):
    from libreyolo import SAMPLE_IMAGE

    assert edgetam_model.set_image(SAMPLE_IMAGE) is edgetam_model
    cached = edgetam_model._image_embeddings
    assert isinstance(cached, list)
    cached_ids = [id(embedding) for embedding in cached]

    point_result = edgetam_model.predict(points=[640, 360], labels=[1])
    box_result = edgetam_model.predict(bboxes=[100, 100, 500, 500])

    _assert_mask_contract(point_result, edgetam_model)
    _assert_mask_contract(box_result, edgetam_model)
    assert edgetam_model._image_embeddings is cached
    assert [id(embedding) for embedding in cached] == cached_ids

    assert edgetam_model.reset_image() is edgetam_model
    assert edgetam_model._image is None
    assert edgetam_model._image_path is None
    assert edgetam_model._image_embeddings is None
    with pytest.raises(RuntimeError, match="No image set"):
        edgetam_model.predict(points=[640, 360], labels=[1])
