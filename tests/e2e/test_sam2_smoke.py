"""End-to-end smoke test for the LibreSAM2 promptable segmenter."""

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.sam,
    pytest.mark.network,
    pytest.mark.external_data,
]

pytest.importorskip("transformers", reason="LibreSAM2 requires the 'sam' extra")


@pytest.fixture(scope="module")
def sam2_model():
    from libreyolo import LibreSAM2

    return LibreSAM2("tiny", device="cpu")


def test_sam2_point_prompt_returns_mask(sam2_model):
    from libreyolo import Results, SAMPLE_IMAGE

    result = sam2_model.predict(SAMPLE_IMAGE, points=[640, 360], labels=[1])

    assert isinstance(result, Results)
    assert result.masks is not None
    assert len(result.masks) >= 1
    assert result.boxes.xyxy.shape[1] == 4
    assert len(result.boxes.conf) == len(result.masks)
    assert sam2_model.names[0] == "object"


def test_sam2_box_prompt_returns_mask(sam2_model):
    from libreyolo import SAMPLE_IMAGE

    result = sam2_model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 500, 500])
    assert result.masks is not None
    h, w = result.orig_shape
    assert result.masks.data.shape[-2:] == (h, w)


def test_sam2_multimask_returns_all_ambiguity_masks(sam2_model):
    from libreyolo import SAMPLE_IMAGE

    single = sam2_model.predict(SAMPLE_IMAGE, points=[640, 360], labels=[1])
    triple = sam2_model.predict(
        SAMPLE_IMAGE, points=[640, 360], labels=[1], multimask=True
    )
    assert len(single.masks) == 1
    assert len(triple.masks) == 3


def test_sam2_segment_everything_runs(sam2_model):
    from libreyolo import Results, SAMPLE_IMAGE

    result = sam2_model.predict(SAMPLE_IMAGE, points_per_side=6)
    assert isinstance(result, Results)
    assert result.masks is not None
    assert len(result.masks) == len(result.boxes.xyxy)


def test_sam2_interactive_session_reuses_embeddings(sam2_model):
    from libreyolo import SAMPLE_IMAGE

    sam2_model.set_image(SAMPLE_IMAGE)
    try:
        assert isinstance(sam2_model._image_embeddings, list)
        a = sam2_model.predict(points=[640, 360], labels=[1])
        b = sam2_model.predict(bboxes=[100, 100, 500, 500])
        assert a.masks is not None and b.masks is not None
        sam2_model._set_device("cpu")
        c = sam2_model.predict(points=[640, 360], labels=[1])
        assert c.masks is not None
    finally:
        sam2_model.reset_image()


def test_sam2_interactive_surface_and_scope(sam2_model):
    assert callable(sam2_model)
    assert hasattr(sam2_model, "set_image")
    assert hasattr(sam2_model, "reset_image")
    assert sam2_model.task == "segment"
    with pytest.raises(RuntimeError):
        sam2_model.reset_image().predict(points=[10, 10], labels=[1])
    with pytest.raises(NotImplementedError):
        sam2_model.train()
    with pytest.raises(NotImplementedError, match="video"):
        sam2_model.track("video.mp4")
