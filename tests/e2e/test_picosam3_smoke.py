"""End-to-end smoke test for LibrePicoSAM3 and its upstream download route."""

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.sam,
    pytest.mark.network,
    pytest.mark.external_data,
]


def test_picosam3_box_prompt_autodownloads_and_returns_mask():
    from libreyolo import LibreSAM, Results, SAMPLE_IMAGE

    model = LibreSAM("picosam3", device="cpu")
    result = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 500, 500], conf=0.0)

    assert isinstance(result, Results)
    assert result.masks is not None
    assert len(result.masks) == 1
    assert result.masks.data.shape[-2:] == result.orig_shape
    assert result.boxes.xyxy.shape == (1, 4)
