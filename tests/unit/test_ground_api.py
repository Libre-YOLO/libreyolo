"""Offline tests for the LibreGround API (no model download)."""

import torch
import pytest

from libreyolo.models.ground import (
    _ALIASES,
    _DEFAULT_MODEL,
    LibreGround,
    LibreGroundFlorence2,
    LibreGroundLocateAnything,
    LibreGroundMoondream,
    LibreGroundQwen3VL,
    LibreHolo,
    LibreShowUI,
    LibreTinyClick,
    LibreUITARS,
)
from libreyolo.models.ground.base import GroundAPIMixin, LibreGroundModel
from libreyolo.utils.results import Points, Results

pytestmark = pytest.mark.unit


class _Runner:
    def __init__(self):
        self.names = {}
        self.seen = []

    def set_classes(self, classes):
        self.names = {i: str(item) for i, item in enumerate(classes)}
        return self

    def __call__(self, source=None, **kwargs):
        self.seen.append(self.names[0])
        data = torch.tensor([[10.0, 20.0, 0.0, 1.0]])
        return Results(
            boxes=None,
            orig_shape=(80, 100),
            path=str(source) if source is not None else None,
            names=dict(self.names),
            points=Points(data, (80, 100)),
        )


class _Host(GroundAPIMixin, _Runner):
    def __init__(self):
        _Runner.__init__(self)
        self._queries = []


class TestSetQuery:
    def test_builds_names(self):
        host = _Host()
        assert host.set_query("Bluetooth") is host
        assert host.names == {0: "Bluetooth"}
        assert host._queries == ["Bluetooth"]

    def test_list_of_queries(self):
        host = _Host()
        host.set_query(["Wi-Fi", "Bluetooth"])
        assert host.names == {0: "Wi-Fi", 1: "Bluetooth"}

    def test_set_classes_is_synonym(self):
        host = _Host()
        host.set_classes(["Save"])
        assert host._queries == ["Save"]


class TestPredictQuery:
    def test_missing_query_raises(self):
        host = _Host()
        with pytest.raises(ValueError, match="needs a query"):
            host("screen.png")

    def test_prompt_kwarg(self):
        host = _Host()
        result = host("screen.png", prompt="Bluetooth")
        assert host.seen == ["Bluetooth"]
        assert result.points.xy.tolist() == [[10.0, 20.0]]
        assert result.names[0] == "Bluetooth"

    def test_query_alias(self):
        host = _Host()
        host("screen.png", query="Wi-Fi")
        assert host.seen == ["Wi-Fi"]

    def test_sticky_then_predict(self):
        host = _Host()
        host.set_query("Bluetooth")
        host("screen.png")
        assert host.seen == ["Bluetooth"]

    def test_multi_query_merges_class_ids(self):
        host = _Host()
        result = host("screen.png", prompt=["Wi-Fi", "Bluetooth"])
        assert host.seen == ["Wi-Fi", "Bluetooth"]
        assert result.points.data[:, 2].tolist() == [0.0, 1.0]
        assert result.names == {0: "Wi-Fi", 1: "Bluetooth"}

    def test_multi_query_rejects_directory(self):
        host = _Host()
        with pytest.raises(ValueError, match="single image"):
            host(".", prompt=["A", "B"])


class TestFactoryResolution:
    def test_default_is_showui_2b(self):
        assert _DEFAULT_MODEL == "showui-2b"
        assert _ALIASES[_DEFAULT_MODEL] == (LibreShowUI, "2b")

    def test_known_aliases(self):
        assert _ALIASES["showui"] == (LibreShowUI, "2b")
        assert _ALIASES["holo"] == (LibreHolo, "7b")
        assert _ALIASES["holo1-7b"] == (LibreHolo, "1-7b")
        assert _ALIASES["ui-tars"] == (LibreUITARS, "7b")
        assert _ALIASES["tinyclick"] == (LibreTinyClick, "b")
        assert _ALIASES["florence-2"] == (LibreGroundFlorence2, "base")
        assert _ALIASES["locate-anything"] == (LibreGroundLocateAnything, "3b")
        assert _ALIASES["moondream"] == (LibreGroundMoondream, "2")
        assert _ALIASES["qwen3-vl-4b"] == (LibreGroundQwen3VL, "4b")

    def test_unknown_alias_raises_before_loading(self):
        with pytest.raises(ValueError, match="Unknown grounding model"):
            LibreGround("definitely-not-a-real-grounder")

    def test_underscore_alias_folds(self):
        assert _ALIASES["show-ui-2b"] == (LibreShowUI, "2b")
        with pytest.raises(ValueError):
            LibreGround("not_a_model_either")


class TestContract:
    def test_point_task(self):
        assert LibreGroundModel.DEFAULT_TASK == "point"
        assert LibreGroundModel.SUPPORTED_TASKS == ("point",)

    def test_family_coord_spaces(self):
        assert LibreShowUI.COORD_SPACE == "unit"
        assert LibreHolo.COORD_SPACE == "pixel_view"
        assert LibreUITARS.COORD_SPACE == "milli"
        assert LibreTinyClick.COORD_SPACE == "milli"
        assert LibreGroundFlorence2.COORD_SPACE == "pixel"
        assert LibreGroundQwen3VL.COORD_SPACE == "milli"

    def test_tinyclick_uses_native_florence(self):
        assert LibreTinyClick.TRUST_REMOTE_CODE is False
        assert LibreTinyClick.HF_REPOS["b"] == "Krystianz/TinyClick"
