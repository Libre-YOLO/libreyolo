"""Offline tests for the LibreOpenVocab tier."""

from __future__ import annotations

from types import SimpleNamespace

import warnings

import pytest
import torch

from libreyolo.models.base.model import BaseModel
from libreyolo.models.openvocab import LibreOpenVocab
from libreyolo.models.openvocab.grounding_dino import LibreGroundingDINO
from libreyolo.models.openvocab.omdet_turbo import LibreOMDetTurbo
from libreyolo.models.openvocab.ov_deim import LibreOVDEIM
from libreyolo.models.openvocab.owlv2 import LibreOWLv2

pytestmark = [pytest.mark.unit, pytest.mark.openvocab]


def _bare(cls=LibreOWLv2):
    m = object.__new__(cls)
    m.names = {0: "person", 1: "dog"}
    m.nb_classes = 2
    m._refresh_name_index()
    return m


class TestSetClasses:
    def test_builds_names_and_reverse_map(self):
        m = _bare()
        m.set_classes(["Pink Car", "Remote Control"])
        assert m.names == {0: "Pink Car", 1: "Remote Control"}
        assert m.nb_classes == 2
        assert m._name_to_id == {"pink car": 0, "remote control": 1}

    def test_rejects_string_scalar_empty_and_blank(self):
        m = _bare()
        with pytest.raises(TypeError):
            m.set_classes("person")
        with pytest.raises(TypeError):
            m.set_classes(123)
        with pytest.raises(ValueError):
            m.set_classes([])
        with pytest.raises(ValueError):
            m.set_classes(["person", "   "])

    def test_duplicate_casefolded_names_raise(self):
        m = _bare()
        with pytest.raises(ValueError):
            m.set_classes(["Boat", "boat"])

    def test_second_call_replaces_the_vocabulary(self):
        m = _bare()
        m.set_classes(["person", "remote control"])
        m.set_classes(["Traffic Light"])

        assert m.names == {0: "Traffic Light"}
        assert m.nb_classes == 1
        assert m._name_to_id == {"traffic light": 0}


class TestFactoryAliases:
    def test_default_resolves_to_grounding_dino_tiny(self):
        from libreyolo.models.openvocab import _ALIASES, _DEFAULT_MODEL

        assert _DEFAULT_MODEL == "grounding-dino-tiny"
        assert _ALIASES[_DEFAULT_MODEL] == (LibreGroundingDINO, "t")

    def test_known_aliases_map_to_family_and_size(self):
        from libreyolo.models.openvocab import _ALIASES

        assert _ALIASES["grounding-dino-base"] == (LibreGroundingDINO, "b")
        assert _ALIASES["owlv2"] == (LibreOWLv2, "b16")
        assert _ALIASES["owl-v2-large"] == (LibreOWLv2, "l14")
        assert _ALIASES["omdet-turbo"] == (LibreOMDetTurbo, "t")
        assert _ALIASES["omdet"] == (LibreOMDetTurbo, "t")
        assert _ALIASES["omdet-turbo-swin-tiny"] == (LibreOMDetTurbo, "t")
        assert _ALIASES["ov-deim"] == (LibreOVDEIM, "s")
        assert _ALIASES["ovdeim-l"] == (LibreOVDEIM, "l")

    def test_unknown_alias_raises_before_loading(self):
        with pytest.raises(ValueError, match="Unknown open-vocabulary detector"):
            LibreOpenVocab("definitely-not-real")

    def test_every_name_the_inventory_advertises_is_loadable(self):
        """`libreyolo models` prints <family>-<size>, with the family's
        underscore. Those names must resolve, or the CLI advertises models the
        factory rejects."""
        from libreyolo.models.inventory import collect_model_inventory
        from libreyolo.models.openvocab import _ALIASES

        inventory = collect_model_inventory()
        advertised = [
            f"{family}-{size}"
            for family, entry in inventory.items()
            if entry["optional_extra"] == "openvocab"
            for size in entry["sizes"]
        ]
        assert advertised  # guard against the filter silently matching nothing

        unloadable = [
            name for name in advertised if name.replace("_", "-") not in _ALIASES
        ]
        assert not unloadable


class TestSnapshotComplete:
    def _mark_complete(self, path, marker=None):
        import json

        (path / ".libreyolo_snapshot_complete").write_text(json.dumps(marker or {}))

    def test_single_file_complete(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("x")
        self._mark_complete(tmp_path, {"repo": "example/model"})
        assert LibreOWLv2._snapshot_complete(tmp_path, repo="example/model") is True

    def test_missing_marker_or_config_is_incomplete(self, tmp_path):
        (tmp_path / "model.safetensors").write_text("x")
        assert LibreOWLv2._snapshot_complete(tmp_path, repo="example/model") is False
        self._mark_complete(tmp_path, {"repo": "example/model"})
        assert LibreOWLv2._snapshot_complete(tmp_path, repo="example/model") is False

    def test_sharded_complete_only_when_all_shards_exist(self, tmp_path):
        import json

        (tmp_path / "config.json").write_text("{}")
        self._mark_complete(tmp_path, {"repo": "example/model"})
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}})
        )
        (tmp_path / "s1.safetensors").write_text("x")
        assert LibreOWLv2._snapshot_complete(tmp_path, repo="example/model") is False
        (tmp_path / "s2.safetensors").write_text("x")
        assert LibreOWLv2._snapshot_complete(tmp_path, repo="example/model") is True

    def test_repo_marker_must_match(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("x")
        self._mark_complete(tmp_path, {"repo": "other/model"})
        assert LibreOWLv2._snapshot_complete(tmp_path, repo="example/model") is False

    def test_pinned_revision_marker_must_match(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("x")
        self._mark_complete(
            tmp_path,
            {"repo": "example/model", "revision": "abc123"},
        )
        assert (
            LibreOWLv2._snapshot_complete(
                tmp_path,
                repo="example/model",
                revision="abc123",
            )
            is True
        )
        assert (
            LibreOWLv2._snapshot_complete(
                tmp_path,
                repo="example/model",
                revision="def456",
            )
            is False
        )


class TestCallDefaults:
    def test_owlv2_default_conf_is_injected(self, monkeypatch):
        captured = {}

        def fake_call(self, source=None, **kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(BaseModel, "__call__", fake_call)
        m = _bare(LibreOWLv2)
        assert m("image.jpg") == "ok"
        assert captured["conf"] == pytest.approx(0.1)

    def test_omdet_turbo_default_conf_is_injected(self, monkeypatch):
        captured = {}

        def fake_call(self, source=None, **kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(BaseModel, "__call__", fake_call)
        m = _bare(LibreOMDetTurbo)
        assert m("image.jpg") == "ok"
        assert captured["conf"] == pytest.approx(0.3)

    def test_explicit_conf_is_preserved(self, monkeypatch):
        captured = {}

        def fake_call(self, source=None, **kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(BaseModel, "__call__", fake_call)
        m = _bare(LibreOWLv2)
        assert m("image.jpg", conf=0.42) == "ok"
        assert captured["conf"] == pytest.approx(0.42)

    def test_text_threshold_is_temporary_for_grounding_dino(self, monkeypatch):
        seen = {}

        def fake_call(self, source=None, **kwargs):
            seen["during"] = self._text_threshold
            return "ok"

        monkeypatch.setattr(BaseModel, "__call__", fake_call)
        m = _bare(LibreGroundingDINO)
        m._text_threshold = 0.25

        assert m("image.jpg", text_threshold=0.4) == "ok"

        assert seen["during"] == pytest.approx(0.4)
        assert m._text_threshold == pytest.approx(0.25)

    def test_text_threshold_rejected_for_owlv2(self):
        m = _bare(LibreOWLv2)
        with pytest.raises(TypeError, match="does not support text_threshold"):
            m("image.jpg", text_threshold=0.4)

    def test_text_threshold_rejected_for_omdet_turbo(self):
        m = _bare(LibreOMDetTurbo)
        with pytest.raises(TypeError, match="does not support text_threshold"):
            m("image.jpg", text_threshold=0.4)

    def test_text_threshold_rejected_for_streaming_call(self):
        m = _bare(LibreGroundingDINO)
        m._text_threshold = 0.25
        with pytest.raises(ValueError, match="stream=True"):
            m("video.mp4", stream=True, text_threshold=0.4)
        assert m._text_threshold == pytest.approx(0.25)

    def test_imgsz_is_rejected_instead_of_silently_ignored(self):
        m = _bare(LibreOWLv2)
        with pytest.raises(ValueError, match="does not support imgsz"):
            m("image.jpg", imgsz=320)

    def test_augment_true_is_rejected_instead_of_silently_ignored(self):
        m = _bare(LibreOWLv2)
        with pytest.raises(ValueError, match="augment=True"):
            m("image.jpg", augment=True)

    def test_iou_warns_for_families_that_suppress_nothing(self, monkeypatch):
        def fake_call(self, source=None, **kwargs):
            return "ok"

        monkeypatch.setattr(BaseModel, "__call__", fake_call)
        m = _bare(LibreOWLv2)
        with pytest.warns(UserWarning, match="iou=.+ignored"):
            assert m("image.jpg", iou=0.7) == "ok"

    def test_omdet_turbo_honours_an_explicit_iou(self, monkeypatch):
        captured = {}

        def fake_call(self, source=None, **kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(BaseModel, "__call__", fake_call)
        m = _bare(LibreOMDetTurbo)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # honouring it must not also warn
            assert m("image.jpg", iou=0.7) == "ok"
        assert captured["iou"] == pytest.approx(0.7)

    def test_omdet_turbo_injects_its_own_nms_default_when_iou_is_unset(
        self, monkeypatch
    ):
        """predict() defaults iou=0.45, which is not OMDet-Turbo's default."""
        captured = {}

        def fake_call(self, source=None, **kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(BaseModel, "__call__", fake_call)
        m = _bare(LibreOMDetTurbo)
        assert m("image.jpg") == "ok"
        assert captured["iou"] == pytest.approx(0.5)

    def test_track_raises_in_v1(self):
        m = _bare(LibreOWLv2)
        with pytest.raises(NotImplementedError, match="tracking is out of scope"):
            m.track("video.mp4")


class TestGroundingDinoPhraseMapping:
    def test_exact_match(self):
        m = _bare(LibreGroundingDINO)
        m.set_classes(["remote control", "cat"])
        assert m._phrase_to_class_id("a remote control") == 0
        assert m._phrase_to_class_id("Cat") == 1

    def test_word_boundary_matching_does_not_match_carpet(self):
        m = _bare(LibreGroundingDINO)
        m.set_classes(["car"])
        assert m._phrase_to_class_id("red car") == 0
        assert m._phrase_to_class_id("carpet") is None

    def test_ambiguous_non_exact_match_is_dropped(self):
        m = _bare(LibreGroundingDINO)
        m.set_classes(["bus", "school"])
        assert m._phrase_to_class_id("school bus") is None

    def test_exact_match_wins_for_overlapping_label(self):
        m = _bare(LibreGroundingDINO)
        m.set_classes(["bus", "school bus"])
        assert m._phrase_to_class_id("school bus") == 1

    def test_postprocess_uses_text_labels_not_integer_labels(self):
        m = _bare(LibreGroundingDINO)
        m.set_classes(["cat"])
        m._text_threshold = 0.25

        class Processor:
            def post_process_grounded_object_detection(self, *args, **kwargs):
                return [
                    {
                        "boxes": torch.tensor([[1.0, 2.0, 10.0, 12.0]]),
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([7]),
                        "text_labels": ["cat"],
                    }
                ]

        m.processor = Processor()
        det = m._postprocess(
            {"outputs": object(), "input_ids": torch.tensor([[1]])},
            0.25,
            0.45,
            (20, 20),
        )
        assert det["num_detections"] == 1
        assert det["classes"].tolist() == [0]


class TestGroundingDinoPromptChunking:
    class Processor:
        class Tokenizer:
            def __call__(self, text, add_special_tokens=True, truncation=False):
                import re

                tokens = re.findall(r"[a-z0-9]+", text.lower())
                if add_special_tokens:
                    tokens = ["[CLS]", *tokens, "[SEP]"]
                return {"input_ids": tokens}

        tokenizer = Tokenizer()

        def __call__(self, *, images, text, return_tensors):
            return {"text": text, "input_ids": torch.tensor([[1]])}

    def _model(self, names):
        m = _bare(LibreGroundingDINO)
        m.set_classes(names)
        m.processor = self.Processor()
        m.model = SimpleNamespace(config=SimpleNamespace(max_text_len=6))
        m._text_threshold = 0.25
        return m

    def test_chunks_prompt_to_model_max_text_len(self):
        m = self._model(["cat", "dog", "bus", "boat", "kite"])

        chunks = m._prompt_chunks()

        assert chunks == ["a cat. a dog.", "a bus. a boat.", "a kite."]
        assert all(m._token_count(prompt) <= 6 for prompt in chunks)

    def test_rejects_single_prompt_that_exceeds_model_max_text_len(self):
        m = self._model(["very long class name"])

        with pytest.raises(ValueError, match="exceeds max_text_len=6"):
            m._prompt_chunks()

    def test_build_inputs_returns_runner_compatible_chunk_payload(self):
        m = self._model(["cat", "dog", "bus"])

        payload = m._build_inputs(object())

        assert len(payload) == 2
        assert hasattr(payload, "to")
        assert payload.to("cpu") is payload
        assert [item["text"] for item in payload] == ["a cat. a dog.", "a bus."]

    def test_postprocess_merges_chunk_outputs_before_global_sort(self):
        m = self._model(["cat", "dog"])

        class Processor(self.Processor):
            def post_process_grounded_object_detection(self, outputs, *args, **kwargs):
                if outputs == "cat-output":
                    return [
                        {
                            "boxes": torch.tensor([[1.0, 2.0, 10.0, 12.0]]),
                            "scores": torch.tensor([0.8]),
                            "text_labels": ["cat"],
                        }
                    ]
                return [
                    {
                        "boxes": torch.tensor([[3.0, 4.0, 9.0, 11.0]]),
                        "scores": torch.tensor([0.9]),
                        "text_labels": ["dog"],
                    }
                ]

        m.processor = Processor()
        det = m._postprocess(
            [
                {"outputs": "cat-output", "input_ids": torch.tensor([[1]])},
                {"outputs": "dog-output", "input_ids": torch.tensor([[2]])},
            ],
            0.1,
            0.45,
            (20, 20),
        )

        assert det["num_detections"] == 2
        assert det["classes"].tolist() == [1, 0]


class TestOWLv2Mapping:
    def test_integer_labels_map_to_query_class_ids(self):
        m = _bare(LibreOWLv2)
        m.set_classes(["cat", "dog", "bus"])
        labels = m._labels_to_class_ids(torch.tensor([2, 0, 99]))
        assert labels.tolist() == [2, 0, -1]

    def test_prompt_template(self):
        m = _bare(LibreOWLv2)
        m.set_classes(["Remote Control", "Cat"])
        assert m._text_labels() == [["a photo of a remote control", "a photo of a cat"]]

    def test_postprocess_filters_invalid_query_ids(self):
        m = _bare(LibreOWLv2)
        m.set_classes(["cat", "dog"])

        class Processor:
            def post_process_grounded_object_detection(self, *args, **kwargs):
                return [
                    {
                        "boxes": torch.tensor(
                            [[1.0, 2.0, 10.0, 12.0], [3.0, 4.0, 9.0, 11.0]]
                        ),
                        "scores": torch.tensor([0.9, 0.8]),
                        "labels": torch.tensor([1, 5]),
                    }
                ]

        m.processor = Processor()
        det = m._postprocess(object(), 0.1, 0.45, (20, 20))
        assert det["num_detections"] == 1
        assert det["classes"].tolist() == [1]


class TestOMDetTurboMapping:
    def test_task_prompt_lists_all_classes(self):
        m = _bare(LibreOMDetTurbo)
        m.set_classes(["person", "remote control"])
        names = m._class_names()
        assert names == ["person", "remote control"]
        assert m._task_prompt(names) == "Detect person, remote control."

    def test_build_inputs_passes_classes_and_task_to_processor(self):
        m = _bare(LibreOMDetTurbo)
        m.set_classes(["cat", "dog"])

        captured = {}

        class Processor:
            def __call__(self, *, images, text, task, return_tensors):
                captured.update(images=images, text=text, task=task, rt=return_tensors)
                return {"ok": True}

        m.processor = Processor()
        sentinel = object()
        out = m._build_inputs(sentinel)
        assert out == {"ok": True}
        assert captured["images"] is sentinel
        assert captured["text"] == ["cat", "dog"]
        assert captured["task"] == "Detect cat, dog."
        assert captured["rt"] == "pt"

    def test_postprocess_maps_text_labels_directly(self):
        m = _bare(LibreOMDetTurbo)
        m.set_classes(["cat", "dog"])

        captured = {}

        class Processor:
            def post_process_grounded_object_detection(self, *args, **kwargs):
                captured.update(kwargs)
                return [
                    {
                        "boxes": torch.tensor(
                            [
                                [1.0, 2.0, 10.0, 12.0],
                                [3.0, 4.0, 9.0, 11.0],
                                [0.0, 0.0, 5.0, 5.0],
                            ]
                        ),
                        "scores": torch.tensor([0.9, 0.8, 0.7]),
                        # "elephant" is not in the vocabulary -> dropped.
                        "text_labels": ["dog", "cat", "elephant"],
                    }
                ]

        m.processor = Processor()
        det = m._postprocess(object(), 0.3, 0.45, (20, 20))
        assert det["num_detections"] == 2
        assert det["classes"].tolist() == [1, 0]
        assert captured == {
            "text_labels": ["cat", "dog"],
            "threshold": pytest.approx(0.3),
            # Reaches the processor as given: __call__ owns the default.
            "nms_threshold": pytest.approx(0.45),
            "target_sizes": [(20, 20)],
        }

    def test_multiword_labels_map_case_insensitively(self):
        m = _bare(LibreOMDetTurbo)
        m.set_classes(["Traffic Light", "Remote Control"])

        labels = m._labels_to_class_ids(["traffic light", "REMOTE CONTROL"])

        assert labels.tolist() == [0, 1]

    def test_postprocess_handles_no_detections(self):
        m = _bare(LibreOMDetTurbo)
        m.set_classes(["giraffe"])

        class Processor:
            def post_process_grounded_object_detection(self, *args, **kwargs):
                return [
                    {
                        "boxes": torch.zeros((0, 4)),
                        "scores": torch.zeros((0,)),
                        "labels": torch.zeros((0,), dtype=torch.int64),
                        "text_labels": [],
                    }
                ]

        m.processor = Processor()
        det = m._postprocess(object(), 0.3, 0.45, (20, 20))

        assert det["num_detections"] == 0
        assert det["boxes"].shape == (0, 4)
        assert det["scores"].shape == (0,)
        assert det["classes"].shape == (0,)

    def test_postprocess_falls_back_to_classes_key(self):
        m = _bare(LibreOMDetTurbo)
        m.set_classes(["cat", "dog"])

        class Processor:
            def post_process_grounded_object_detection(self, *args, **kwargs):
                # Keep compatibility with processors that use the "classes" key.
                return [
                    {
                        "boxes": torch.tensor([[1.0, 2.0, 10.0, 12.0]]),
                        "scores": torch.tensor([0.9]),
                        "classes": torch.tensor([1]),
                    }
                ]

        m.processor = Processor()
        det = m._postprocess(object(), 0.3, 0.45, (20, 20))
        assert det["num_detections"] == 1
        assert det["classes"].tolist() == [1]


class TestOVDEIM:
    def _bare_ovdeim(self, names=("person", "dog")):
        m = object.__new__(LibreOVDEIM)
        m.names = {i: n for i, n in enumerate(names)}
        m.nb_classes = len(names)
        m._refresh_name_index()
        m._text_feats_cache = None
        m._tokenizer = None
        m.size = "s"
        return m

    def test_letterbox_params_match_upstream_rounding(self):
        # 480x640 image into 640: scale 1.0 wide side, rint rounding, centered pad.
        new_w, new_h, left, top = LibreOVDEIM._letterbox_params(640, 480, 640)
        assert (new_w, new_h) == (640, 480)
        assert (left, top) == (0, 80)

        # Non-integer scale: rint, and odd padding splits with rint on top/left.
        new_w, new_h, left, top = LibreOVDEIM._letterbox_params(500, 375, 640)
        assert (new_w, new_h) == (640, 480)
        assert (left, top) == (0, 80)

    def test_postprocess_maps_flat_topk_to_labels_and_queries(self):
        m = self._bare_ovdeim()
        logits = torch.full((1, 300, 2), -10.0)
        boxes = torch.full((1, 300, 4), 0.5)
        # Query 7 scores high on class 1; centered box covering half the image.
        logits[0, 7, 1] = 3.0
        boxes[0, 7] = torch.tensor([0.5, 0.5, 0.5, 0.5])
        det = m._postprocess(
            {"pred_logits": logits, "pred_boxes": boxes},
            conf_thres=0.5,
            iou_thres=0.45,
            original_size=(640, 640),
        )
        assert det["num_detections"] == 1
        assert det["classes"].tolist() == [1]
        assert torch.allclose(
            det["boxes"][0], torch.tensor([160.0, 160.0, 480.0, 480.0])
        )

    def test_postprocess_undoes_letterbox_padding(self):
        m = self._bare_ovdeim()
        logits = torch.full((1, 300, 2), -10.0)
        boxes = torch.zeros((1, 300, 4))
        logits[0, 0, 0] = 3.0
        # Full-width box in letterbox space for a 640x480 original (80px top pad).
        boxes[0, 0] = torch.tensor([0.5, 0.5, 1.0, 480.0 / 640.0])
        det = m._postprocess(
            {"pred_logits": logits, "pred_boxes": boxes},
            conf_thres=0.5,
            iou_thres=0.45,
            original_size=(640, 480),
        )
        assert det["num_detections"] == 1
        assert torch.allclose(
            det["boxes"][0], torch.tensor([0.0, 0.0, 640.0, 480.0]), atol=1e-4
        )

    def test_cached_text_feats_follow_the_model_across_devices(self):
        """predict(device=...) moves the model between calls; a cache hit must
        not hand back features stranded on the previous device."""

        class _FakeTower:
            def encode_texts(self, tokens):
                return torch.zeros(tokens.shape[0], 512, device=tokens.device)

        m = self._bare_ovdeim()
        m.model = _FakeTower()
        m.device = torch.device("meta")  # stands in for a second, distinct device
        m._tokenizer = lambda names: torch.zeros(len(names), 77, dtype=torch.long)

        cached = torch.zeros(2, 512)  # cached on CPU by an earlier call
        m._text_feats_cache = (("person", "dog"), cached)

        feats = m._class_text_feats()
        assert feats.device == m.device
        # The refreshed tensor is written back, so the next hit stays correct.
        assert m._text_feats_cache[1].device == m.device

    def test_set_classes_invalidates_text_cache(self):
        m = self._bare_ovdeim()
        m._text_feats_cache = (("person", "dog"), torch.zeros(2, 512))
        m.set_classes(["a red hat"])
        assert m._text_feats_cache is None
        assert m.names == {0: "a red hat"}

    def test_no_nms_iou_warns(self):
        assert LibreOVDEIM.NMS_THRESHOLD is None

    def test_build_inputs_produces_normalized_letterboxed_tensor(self):
        from PIL import Image

        m = self._bare_ovdeim()
        img = Image.new("RGB", (320, 240), (114, 114, 114))
        tensor = m._build_inputs(img)
        assert tensor.shape == (1, 3, 640, 640)
        # A uniform 114 image letterboxed with 114 padding stays constant,
        # so every channel equals (114/255 - mean) / std everywhere.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        expected = (torch.full((3, 640, 640), 114.0 / 255.0) - mean) / std
        assert torch.allclose(tensor[0], expected, atol=1e-6)


class TestDetectionDict:
    def test_outputs_are_normalized_to_cpu_tensors(self):
        m = _bare(LibreOWLv2)
        det = m._detections_to_dict(
            torch.tensor([[1.0, 2.0, 10.0, 12.0]]),
            torch.tensor([0.9]),
            [0],
            conf_thres=0.1,
            original_size=(20, 20),
            max_det=10,
        )
        assert det["boxes"].device.type == "cpu"
        assert det["scores"].device.type == "cpu"
        assert det["classes"].device.type == "cpu"
