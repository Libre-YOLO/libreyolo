"""End-to-end smoke test for the native LibreMobileSAM segmenter."""

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.sam,
    pytest.mark.network,
    pytest.mark.external_data,
]


@pytest.fixture(scope="module")
def mobilesam_model():
    from libreyolo import LibreMobileSAM

    return LibreMobileSAM(device="cpu")


def test_mobilesam_point_prompt_returns_mask(mobilesam_model):
    from libreyolo import Results, SAMPLE_IMAGE

    result = mobilesam_model.predict(SAMPLE_IMAGE, points=[640, 360], labels=[1])

    assert isinstance(result, Results)
    assert result.masks is not None
    assert len(result.masks) >= 1
    assert result.boxes.xyxy.shape[1] == 4
    assert len(result.boxes.conf) == len(result.masks)
    assert mobilesam_model.names[0] == "object"


def test_mobilesam_box_prompt_returns_mask(mobilesam_model):
    from libreyolo import SAMPLE_IMAGE

    result = mobilesam_model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 500, 500])
    assert result.masks is not None
    h, w = result.orig_shape
    assert result.masks.data.shape[-2:] == (h, w)


def test_mobilesam_multimask_returns_all_ambiguity_masks(mobilesam_model):
    from libreyolo import SAMPLE_IMAGE

    single = mobilesam_model.predict(SAMPLE_IMAGE, points=[640, 360], labels=[1])
    triple = mobilesam_model.predict(
        SAMPLE_IMAGE, points=[640, 360], labels=[1], multimask=True
    )
    assert len(single.masks) == 1
    assert len(triple.masks) == 3


def test_mobilesam_segment_everything_runs(mobilesam_model):
    from libreyolo import Results, SAMPLE_IMAGE

    result = mobilesam_model.predict(SAMPLE_IMAGE, points_per_side=6)
    assert isinstance(result, Results)
    assert result.masks is not None
    assert len(result.masks) == len(result.boxes.xyxy)


def test_mobilesam_interactive_session_reuses_embeddings(mobilesam_model):
    from libreyolo import SAMPLE_IMAGE

    mobilesam_model.set_image(SAMPLE_IMAGE)
    try:
        a = mobilesam_model.predict(points=[640, 360], labels=[1])
        b = mobilesam_model.predict(bboxes=[100, 100, 500, 500])
        assert a.masks is not None and b.masks is not None
    finally:
        mobilesam_model.reset_image()


def test_mobilesam_interactive_surface_and_scope(mobilesam_model):
    assert callable(mobilesam_model)
    assert hasattr(mobilesam_model, "set_image")
    assert hasattr(mobilesam_model, "reset_image")
    assert mobilesam_model.task == "segment"
    with pytest.raises(RuntimeError):
        mobilesam_model.reset_image().predict(points=[10, 10], labels=[1])
    with pytest.raises(NotImplementedError):
        mobilesam_model.train()
    with pytest.raises(NotImplementedError):
        mobilesam_model.track("video.mp4")
