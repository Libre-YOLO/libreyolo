from types import SimpleNamespace

import pytest

from libreyolo.ui.server import _summarize_result

pytestmark = pytest.mark.unit


def result(**attrs):
    return SimpleNamespace(**attrs)


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (result(boxes=[object()]), ("detect", "1 object")),
        (result(boxes=[object(), object()], masks=object()), ("segment", "2 instances")),
        (result(boxes=[object()], keypoints=object()), ("pose", "1 pose")),
        (result(obb=[object(), object()]), ("obb", "2 objects")),
        (result(points=[object(), object(), object()]), ("point", "3 points")),
        (
            result(
                probs=SimpleNamespace(top1=1, top1conf=0.876),
                names=["tabby", "mug"],
            ),
            ("classify", "mug 88%"),
        ),
        (
            result(semantic_mask=SimpleNamespace(classes=[0, 2])),
            ("semantic", "2 regions"),
        ),
        (result(depth_map=object()), ("depth", "depth map")),
        (result(restored=object(), restore_scale=4), ("restore", "upscaled x4")),
        (result(restored=object(), restore_scale=1), ("restore", "restored")),
        (result(matte=object()), ("matte", "alpha matte")),
        (result(gaze=[object(), object()]), ("gaze", "2 gaze")),
        (result(), ("detect", "0 objects")),
    ],
)
def test_summarize_result(item, expected):
    assert _summarize_result(item) == expected
