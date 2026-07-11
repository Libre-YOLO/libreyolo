from types import SimpleNamespace

import pytest

from libreyolo.ui.server import (
    _openvocab_model_names,
    _result_count,
    _summarize_result,
)

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
        # Real gaze results also carry the face boxes; gaze must win.
        (
            result(gaze=[object(), object()], boxes=[object(), object()]),
            ("gaze", "2 faces"),
        ),
        (result(gaze=[object()]), ("gaze", "1 face")),
        (
            result(panoptic=SimpleNamespace(segments_info=[{}, {}, {}])),
            ("panoptic", "3 segments"),
        ),
        (result(), ("detect", "0 objects")),
    ],
)
def test_summarize_result(item, expected):
    assert _summarize_result(item) == expected


def test_result_count_panoptic():
    item = result(panoptic=SimpleNamespace(segments_info=[{}, {}]))
    assert _result_count(item) == 2


def test_openvocab_model_names_flags_set_classes_families():
    """Open-vocab detectors and zero-shot classifiers get the classes box;
    fixed-vocabulary detectors do not."""
    from libreyolo.cli.config import get_all_cli_names
    from libreyolo.ui.server import _factory_openvocab_names

    names = sorted(get_all_cli_names() + _factory_openvocab_names())
    flagged = set(_openvocab_model_names(names))
    assert "grounding-dino-t" in flagged
    assert "owlv2-b16" in flagged
    assert any(n.startswith("clip-") for n in flagged)
    assert not any(n.startswith("yolo9-") for n in flagged)


def test_factory_openvocab_names_resolve_in_the_factory():
    """Every UI-listed factory name must stay a valid LibreOpenVocab alias."""
    from libreyolo.models.openvocab import _ALIASES
    from libreyolo.ui.server import _factory_openvocab_names

    for name in _factory_openvocab_names():
        assert name in _ALIASES


def test_apply_vocabulary_sets_and_restores():
    from libreyolo.ui.server import _UIState

    state = _UIState(device="cpu")
    calls = []

    class FakeOV:
        def __init__(self):
            self.names = {0: "person", 1: "car"}

        def set_classes(self, classes):
            calls.append(list(classes))
            self.names = {i: c for i, c in enumerate(classes)}

    model = FakeOV()
    state._apply_vocabulary("fake-ov", model, ["boat", "dock"])
    assert calls == [["boat", "dock"]]
    # No classes on the next request: the original vocabulary is restored.
    state._apply_vocabulary("fake-ov", model, None)
    assert calls == [["boat", "dock"], ["person", "car"]]
    # Already at the default: no redundant set_classes call.
    state._apply_vocabulary("fake-ov", model, None)
    assert len(calls) == 2
    # Models without set_classes are left alone.
    state._apply_vocabulary("plain", object(), ["boat"])


def test_self_managed_download_l2cs_only_r50():
    """l2cs auto-downloads only the published r50 checkpoint; the other size
    names stay greyed out. Snapshot-based families count as available."""
    from libreyolo.ui.server import _self_managed_download

    assert _self_managed_download("LibreL2CSr50.pt") is True
    assert _self_managed_download("LibreL2CSr18.pt") is False
    assert _self_managed_download("definitely-not-a-weight.pt") is False
