"""Authenticated end-to-end smoke coverage for SAM 3 image inference."""

import pytest
from huggingface_hub import get_token

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.sam,
    pytest.mark.network,
    pytest.mark.external_data,
]

pytest.importorskip("transformers", reason="LibreSAM3 requires the 'sam' extra")

if get_token() is None:
    pytest.skip(
        "SAM 3 is gated; accept its terms and authenticate with Hugging Face",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def sam3_model():
    from libreyolo import LibreSAM3

    return LibreSAM3(device="cpu")


@pytest.mark.parametrize(
    "prompt",
    [
        {"points": [640, 360], "labels": [1]},
        {"bboxes": [100, 100, 500, 500]},
    ],
)
def test_sam3_visual_prompt_returns_mask(sam3_model, prompt):
    from libreyolo import Results, SAMPLE_IMAGE

    result = sam3_model.predict(SAMPLE_IMAGE, **prompt)
    assert isinstance(result, Results)
    assert result.masks is not None
    assert len(result.masks) >= 1
    assert result.masks.data.shape[-2:] == result.orig_shape


def test_sam3_segment_everything_runs(sam3_model):
    from libreyolo import SAMPLE_IMAGE

    result = sam3_model.predict(SAMPLE_IMAGE, points_per_side=4)
    assert result.masks is not None
    assert len(result.masks) == len(result.boxes)


def test_sam3_text_prompt_returns_scored_instances(sam3_model):
    from libreyolo import SAMPLE_IMAGE

    result = sam3_model.predict(SAMPLE_IMAGE, text="person", conf=0.2, max_det=10)
    assert result.masks is not None
    assert 1 <= len(result.masks) <= 10
    assert result.names == {0: "person"}
    assert result.masks.data.shape[-2:] == result.orig_shape
