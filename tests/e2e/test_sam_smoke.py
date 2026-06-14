"""End-to-end smoke test for the LibreSAM promptable segmenter.

Heavy: downloads the SAM-1 base weights (Apache-2.0) and loads the model, so it
is gated behind the sam/network/external_data markers and skipped if
transformers is unavailable. The model is loaded once via a module-scoped
fixture.
"""

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.sam,
    pytest.mark.network,
    pytest.mark.external_data,
]

pytest.importorskip("transformers", reason="LibreSAM requires the 'sam' extra")


@pytest.fixture(scope="module")
def sam_model():
    from libreyolo import LibreSAM

    return LibreSAM("base", device="cpu")


def test_point_prompt_returns_mask(sam_model):
    from libreyolo import Results, SAMPLE_IMAGE

    result = sam_model.predict(SAMPLE_IMAGE, points=[640, 360], labels=[1])

    assert isinstance(result, Results)
    assert result.masks is not None
    assert len(result.masks) >= 1
    # Tight boxes are derived from the masks; same Results contract.
    assert result.boxes.xyxy.shape[1] == 4
    assert len(result.boxes.conf) == len(result.masks)
    assert sam_model.names[0] == "object"


def test_box_prompt_returns_mask(sam_model):
    from libreyolo import SAMPLE_IMAGE

    result = sam_model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 500, 500])
    assert result.masks is not None
    h, w = result.orig_shape
    # Mask is upscaled to the original image resolution.
    assert result.masks.data.shape[-2:] == (h, w)


def test_multimask_returns_all_ambiguity_masks(sam_model):
    from libreyolo import SAMPLE_IMAGE

    single = sam_model.predict(SAMPLE_IMAGE, points=[640, 360], labels=[1])
    triple = sam_model.predict(
        SAMPLE_IMAGE, points=[640, 360], labels=[1], multimask=True
    )
    # multimask must actually surface SAM's 3 whole-vs-part masks, not 1.
    assert len(single.masks) == 1
    assert len(triple.masks) == 3


def test_segment_everything_runs(sam_model):
    from libreyolo import Results, SAMPLE_IMAGE

    # Small grid keeps the CPU pass fast; just exercise the AMG path.
    result = sam_model.predict(SAMPLE_IMAGE, points_per_side=6)
    assert isinstance(result, Results)
    assert result.masks is not None
    assert len(result.masks) == len(result.boxes.xyxy)


def test_interactive_session_reuses_embeddings(sam_model):
    from libreyolo import SAMPLE_IMAGE

    sam_model.set_image(SAMPLE_IMAGE)
    try:
        a = sam_model.predict(points=[640, 360], labels=[1])
        b = sam_model.predict(bboxes=[100, 100, 500, 500])
        assert a.masks is not None and b.masks is not None
    finally:
        sam_model.reset_image()


def test_interactive_surface_and_scope(sam_model):
    """Mirrors the documented promptable surface; detector-only ops are out."""
    assert callable(sam_model)
    assert hasattr(sam_model, "set_image")
    assert hasattr(sam_model, "reset_image")
    assert sam_model.task == "segment"
    with pytest.raises(RuntimeError):
        # No image set and no source given.
        sam_model.reset_image().predict(points=[10, 10], labels=[1])
    with pytest.raises(NotImplementedError):
        sam_model.train()
    with pytest.raises(NotImplementedError):
        sam_model.track("video.mp4")
