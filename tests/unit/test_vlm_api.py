"""Unit tests for the LibreVLM open-vocabulary API (offline, no model load).

``set_classes`` only manipulates the vocabulary maps, so it can be exercised on
a bare instance without downloading or loading any model.
"""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from libreyolo.models.vlm.base import LibreVLMModel

pytestmark = pytest.mark.unit


def _bare_model():
    # Bypass __init__ (which would load an 8GB model); we only test the
    # vocabulary-map logic of set_classes.
    return object.__new__(LibreVLMModel)


class TestSetClasses:
    def test_builds_names_and_reverse_map(self):
        m = _bare_model()
        m.set_classes(["Pink Car", "Wheel"])
        assert m.names == {0: "Pink Car", 1: "Wheel"}
        assert m.nb_classes == 2
        # reverse map is lowercased for case-insensitive label resolution
        assert m._name_to_id == {"pink car": 0, "wheel": 1}

    def test_is_sticky_and_replaces(self):
        m = _bare_model()
        m.set_classes(["boat"])
        m.set_classes(["person", "dog"])
        assert m.names == {0: "person", 1: "dog"}
        assert m.nb_classes == 2
        assert m._name_to_id == {"person": 0, "dog": 1}

    def test_returns_self_for_chaining(self):
        m = _bare_model()
        assert m.set_classes(["boat"]) is m

    def test_empty_raises(self):
        m = _bare_model()
        with pytest.raises(ValueError):
            m.set_classes([])
        with pytest.raises(ValueError, match="non-empty"):
            m.set_classes(["boat", "   "])

    def test_string_or_scalar_raises(self):
        # A bare string would enumerate into one-character classes; reject it.
        m = _bare_model()
        with pytest.raises(TypeError):
            m.set_classes("person")
        with pytest.raises(TypeError):
            m.set_classes(123)

    def test_coerces_to_str(self):
        m = _bare_model()
        m.set_classes(["boat", 7])
        assert m.names == {0: "boat", 1: "7"}
        assert m._name_to_id["7"] == 1

    def test_duplicate_casefolded_names_raise(self):
        m = _bare_model()
        with pytest.raises(ValueError):
            m.set_classes(["Boat", "boat"])


class TestFactoryResolution:
    """The LibreVLM(...) name resolution (offline; no model is loaded)."""

    def test_default_resolves_to_qwen3vl_4b(self):
        from libreyolo.models.vlm import _ALIASES, _DEFAULT_MODEL
        from libreyolo.models.vlm.qwen3vl import LibreQwen3VL

        assert _ALIASES[_DEFAULT_MODEL] == (LibreQwen3VL, "4b")

    def test_known_aliases_map_to_family_and_size(self):
        from libreyolo.models.vlm import _ALIASES
        from libreyolo.models.vlm.lfm2 import LibreLFM2VL
        from libreyolo.models.vlm.locateanything import LibreLocateAnything
        from libreyolo.models.vlm.qwen3vl import LibreQwen3VL
        from libreyolo.models.vlm.smolvlm import LibreSmolVLM2

        assert _ALIASES["qwen3-vl-8b"] == (LibreQwen3VL, "8b")
        assert _ALIASES["lfm2-vl-450m"] == (LibreLFM2VL, "450m")
        assert _ALIASES["lfm2-vl-3b"] == (LibreLFM2VL, "3b")
        assert _ALIASES["smolvlm2"] == (LibreSmolVLM2, "2.2b")
        assert _ALIASES["locate-anything"] == (LibreLocateAnything, "3b")

        from libreyolo.models.vlm.northmicro import LibreNorthMicroVision

        assert _ALIASES["north-micro-vision"] == (LibreNorthMicroVision, "2.4b")
        assert _ALIASES["north-micro-vision-2.4b"] == (LibreNorthMicroVision, "2.4b")

        from libreyolo.models.vlm.internvl3 import LibreInternVL3

        assert _ALIASES["internvl3"] == (LibreInternVL3, "2b")

        from libreyolo.models.vlm.florence2 import LibreFlorence2
        from libreyolo.models.vlm.kosmos2 import LibreKosmos2

        assert _ALIASES["florence-2"] == (LibreFlorence2, "base")
        assert _ALIASES["kosmos-2"] == (LibreKosmos2, "224")

        from libreyolo.models.vlm.gemma4 import LibreGemma4
        from libreyolo.models.vlm.moondream import LibreMoondream

        assert _ALIASES["gemma-4"] == (LibreGemma4, "e4b")
        assert _ALIASES["gemma-4-e2b"] == (LibreGemma4, "e2b")
        assert _ALIASES["moondream"] == (LibreMoondream, "2")
        assert _ALIASES["moondream-3"] == (LibreMoondream, "3")

    def test_unknown_alias_raises_before_loading(self):
        from libreyolo.models.vlm import LibreVLM

        # Raises during resolution, before any model download/load.
        with pytest.raises(ValueError):
            LibreVLM("definitely-not-a-real-model")


class TestReferenceInspection:
    """VLM references can be classified without loading model weights."""

    @staticmethod
    def _write_contract(directory: Path, **updates) -> Path:
        directory.mkdir()
        contract = {
            "schema": 1,
            "family": "qwen3vl",
            "size": "2b",
            "base_repo": "Qwen/Qwen3-VL-2B-Instruct",
            "base_revision": None,
            "names": ["person"],
            "bbox_key": "bbox_2d",
            "coord_divisor": 1000.0,
            "box_format": "xyxy",
            "prompt": "Detect person.",
            "task": "detect",
        }
        contract.update(updates)
        (directory / "libreyolo_vlm.json").write_text(json.dumps(contract))
        (directory / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA"})
        )
        (directory / "adapter_model.safetensors").write_bytes(b"adapter")
        return directory

    def test_all_aliases_are_exposed_and_inspectable(self):
        from libreyolo.models import vlm as vlm_module
        from libreyolo.models.vlm import (
            _ALIASES,
            _LAZY_ALIASES,
            _MODUS_ALIASES,
            get_vlm_aliases,
            inspect_vlm_reference,
        )

        expected = tuple(
            sorted(set(_ALIASES) | set(_LAZY_ALIASES) | set(_MODUS_ALIASES))
        )
        aliases = get_vlm_aliases()

        assert aliases == expected
        assert isinstance(aliases, tuple)
        assert all(inspect_vlm_reference(alias) is not None for alias in aliases)
        assert {
            "VLMReference",
            "get_vlm_aliases",
            "inspect_vlm_reference",
        } <= set(vlm_module.__all__)

    def test_alias_metadata_is_immutable_and_does_not_construct_model(
        self, monkeypatch
    ):
        from libreyolo.models.vlm import inspect_vlm_reference
        from libreyolo.models.vlm.qwen3vl import LibreQwen3VL

        def fail_if_constructed(*_args, **_kwargs):
            raise AssertionError("reference inspection constructed a model")

        monkeypatch.setattr(LibreQwen3VL, "__init__", fail_if_constructed)
        reference = inspect_vlm_reference("  QWEN3-VL-4B  ")

        assert reference is not None
        assert reference.family == "qwen3vl"
        assert reference.size == "4b"
        assert reference.trainable is True
        assert reference.trainable_sizes == ("2b", "4b")
        assert reference.checkpoint is False
        with pytest.raises(FrozenInstanceError):
            reference.size = "2b"

    def test_lazy_alias_reports_nontrainable_family(self):
        from libreyolo.models.vlm import inspect_vlm_reference

        reference = inspect_vlm_reference("sensenova-vision")

        assert reference is not None
        assert reference.family == "sensenovavision"
        assert reference.size == "7b"
        assert reference.trainable is False
        assert reference.trainable_sizes == ()
        assert reference.checkpoint is False

    def test_schema_one_checkpoint_is_inspected_without_loading(self, tmp_path):
        from libreyolo.models.vlm import inspect_vlm_reference

        checkpoint = self._write_contract(tmp_path / "checkpoint")
        reference = inspect_vlm_reference(checkpoint)

        assert reference is not None
        assert reference.family == "qwen3vl"
        assert reference.size == "2b"
        assert reference.trainable is True
        assert reference.trainable_sizes == ("2b", "4b")
        assert reference.checkpoint is True

    def test_valid_unknown_family_contract_remains_a_vlm_reference(self, tmp_path):
        from libreyolo.models.vlm import inspect_vlm_reference

        checkpoint = self._write_contract(
            tmp_path / "future-checkpoint", family="future-vlm", size="tiny"
        )
        reference = inspect_vlm_reference(checkpoint)

        assert reference is not None
        assert reference.family == "future-vlm"
        assert reference.size == "tiny"
        assert reference.trainable is False
        assert reference.trainable_sizes == ()
        assert reference.checkpoint is True

    def test_incomplete_checkpoint_artifact_fails_reference_inspection(self, tmp_path):
        from libreyolo.models.vlm import inspect_vlm_reference

        checkpoint = self._write_contract(tmp_path / "incomplete")
        (checkpoint / "adapter_model.safetensors").unlink()

        with pytest.raises(ValueError, match="no adapter tensor payload"):
            inspect_vlm_reference(checkpoint)

    def test_malformed_contract_raises_its_validation_error(self, tmp_path):
        from libreyolo.models.vlm import inspect_vlm_reference

        checkpoint = tmp_path / "malformed"
        checkpoint.mkdir()
        (checkpoint / "libreyolo_vlm.json").write_text('{"schema": 1}')

        with pytest.raises(ValueError, match="missing 'family'"):
            inspect_vlm_reference(checkpoint)

    def test_non_file_contract_marker_also_fails_closed(self, tmp_path):
        from libreyolo.models.vlm import inspect_vlm_reference

        checkpoint = tmp_path / "wrong-contract-type"
        checkpoint.mkdir()
        (checkpoint / "libreyolo_vlm.json").mkdir()

        with pytest.raises(ValueError, match="Unreadable VLM checkpoint contract"):
            inspect_vlm_reference(checkpoint)

    def test_non_vlm_references_return_none(self, tmp_path):
        from libreyolo.models.vlm import inspect_vlm_reference

        assert inspect_vlm_reference("LibreYOLO9s.pt") is None
        assert inspect_vlm_reference(tmp_path) is None
        assert inspect_vlm_reference(object()) is None


class TestLFM2CoordinateConvention:
    """The 3B emits 0-1000 boxes; smaller sizes emit [0, 1] (offline)."""

    def test_3b_shadows_divisor_and_prompt(self):
        from libreyolo.models.vlm.lfm2 import LibreLFM2VL

        assert LibreLFM2VL._COORD_DIVISORS["3b"] == 1000.0
        assert LibreLFM2VL.COORD_DIVISOR == 1.0  # class default untouched

        m = object.__new__(LibreLFM2VL)
        m.COORD_DIVISOR = 1000.0
        assert "0-1000 scale" in m._format_detection_prompt("boat")

        small = object.__new__(LibreLFM2VL)  # falls back to the class attr
        assert "[0,1]" in small._format_detection_prompt("boat")


class TestNorthMicroTransformersGuard:
    """North Micro Vision fails fast on transformers < 5.16 (offline)."""

    def test_old_transformers_raises_with_hint(self, monkeypatch):
        transformers = pytest.importorskip("transformers")
        from libreyolo.models.vlm import northmicro

        monkeypatch.setattr(transformers, "__version__", "5.15.0")
        with pytest.raises(ImportError, match="transformers>=5.16.0"):
            northmicro._require_transformers()

    def test_new_transformers_passes(self, monkeypatch):
        transformers = pytest.importorskip("transformers")
        from libreyolo.models.vlm import northmicro

        monkeypatch.setattr(transformers, "__version__", "5.16.0")
        northmicro._require_transformers()
        # dev builds of the next minor also pass
        monkeypatch.setattr(transformers, "__version__", "5.16.0.dev0")
        northmicro._require_transformers()


class TestSnapshotComplete:
    """The weights-completeness sentinel (offline; no download)."""

    def _base(self):
        from libreyolo.models.vlm.base import LibreVLMModel

        return LibreVLMModel

    def _mark_complete(self, path, marker=None):
        import json

        (path / ".libreyolo_snapshot_complete").write_text(json.dumps(marker or {}))

    def test_single_file_complete(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("x")
        self._mark_complete(tmp_path)
        assert self._base()._snapshot_complete(tmp_path) is True

    def test_missing_completion_marker_is_incomplete(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("x")
        assert self._base()._snapshot_complete(tmp_path) is False

    def test_missing_config_is_incomplete(self, tmp_path):
        (tmp_path / "model.safetensors").write_text("x")
        self._mark_complete(tmp_path)
        assert self._base()._snapshot_complete(tmp_path) is False

    def test_sharded_incomplete_when_a_shard_missing(self, tmp_path):
        import json

        (tmp_path / "config.json").write_text("{}")
        self._mark_complete(tmp_path)
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}})
        )
        (tmp_path / "s1.safetensors").write_text("x")  # only shard 1 of 2
        assert self._base()._snapshot_complete(tmp_path) is False

    def test_sharded_complete_when_all_shards_present(self, tmp_path):
        import json

        (tmp_path / "config.json").write_text("{}")
        self._mark_complete(tmp_path)
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}})
        )
        (tmp_path / "s1.safetensors").write_text("x")
        (tmp_path / "s2.safetensors").write_text("x")
        assert self._base()._snapshot_complete(tmp_path) is True

    def test_pinned_revision_marker_must_match(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("x")
        self._mark_complete(tmp_path, {"repo": "example/model", "revision": "abc123"})

        assert (
            self._base()._snapshot_complete(
                tmp_path, repo="example/model", revision="abc123"
            )
            is True
        )
        assert (
            self._base()._snapshot_complete(
                tmp_path, repo="example/model", revision="def456"
            )
            is False
        )

    def test_unpinned_request_does_not_reuse_pinned_snapshot(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("x")
        self._mark_complete(tmp_path, {"repo": "example/model", "revision": "abc123"})

        assert (
            self._base()._snapshot_complete(
                tmp_path, repo="example/model", revision=None
            )
            is False
        )

        self._mark_complete(tmp_path, {"repo": "example/model", "revision": None})
        assert (
            self._base()._snapshot_complete(
                tmp_path, repo="example/model", revision=None
            )
            is True
        )

    def test_pinned_repo_marker_must_match_and_be_present(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("x")

        self._mark_complete(tmp_path, {"revision": "abc123"})
        assert (
            self._base()._snapshot_complete(
                tmp_path, repo="example/model", revision="abc123"
            )
            is False
        )

        self._mark_complete(tmp_path, {"repo": "other/model", "revision": "abc123"})
        assert (
            self._base()._snapshot_complete(
                tmp_path, repo="example/model", revision="abc123"
            )
            is False
        )

    def test_remote_code_revision_must_be_commit_sha(self):
        from libreyolo.models.vlm.base import LibreVLMModel

        class MutableRevisionVLM(LibreVLMModel):
            FAMILY = "mutable-revision-vlm"
            FILENAME_PREFIX = "MutableRevisionVLM"
            HF_REPOS = {"x": "example/mutable-revision-vlm"}
            HF_REVISIONS = {"x": "main"}
            INPUT_SIZES = {"x": 1}
            TRUST_REMOTE_CODE = True

        m = object.__new__(MutableRevisionVLM)
        m.size = "x"

        with pytest.raises(ValueError, match="40-char commit SHA"):
            m._ensure_weights()

    def test_qwen_download_uses_the_audited_immutable_revision(
        self, tmp_path, monkeypatch
    ):
        import huggingface_hub

        from libreyolo.models.vlm.qwen3vl import LibreQwen3VL

        observed = {}

        def fake_snapshot_download(repo, *, local_dir, ignore_patterns, revision):
            observed.update(
                repo=repo,
                local_dir=local_dir,
                ignore_patterns=ignore_patterns,
                revision=revision,
            )
            target = Path(local_dir)
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text("{}")
            (target / "model.safetensors").write_text("weights")

        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", fake_snapshot_download
        )
        monkeypatch.chdir(tmp_path)
        model = object.__new__(LibreQwen3VL)
        model.size = "2b"

        resolved = Path(model._ensure_weights())

        assert observed["repo"] == LibreQwen3VL.HF_REPOS["2b"]
        assert observed["revision"] == LibreQwen3VL.HF_REVISIONS["2b"]
        assert resolved == Path("weights/LibreQwen3VL2b")

    def test_license_notice_is_logged_for_cached_snapshot(
        self, tmp_path, monkeypatch, caplog
    ):
        import logging
        from pathlib import Path

        from libreyolo.models.vlm.base import LibreVLMModel

        class NoticeVLM(LibreVLMModel):
            FAMILY = "notice-vlm"
            FILENAME_PREFIX = "NoticeVLM"
            HF_REPOS = {"x": "example/notice-vlm"}
            INPUT_SIZES = {"x": 1}
            _LICENSE_NOTICE = "cached snapshot notice"
            _LICENSE_NOTICE_SHOWN = False

            def _init_model(self):
                raise NotImplementedError

            def _get_available_layers(self):
                raise NotImplementedError

            @staticmethod
            def _get_preprocess_numpy():
                raise NotImplementedError

            def _preprocess(self, *args, **kwargs):
                raise NotImplementedError

            def _forward(self, *args, **kwargs):
                raise NotImplementedError

            def _postprocess(self, *args, **kwargs):
                raise NotImplementedError

        cached = tmp_path / "weights" / "NoticeVLMx"
        cached.mkdir(parents=True)
        (cached / "config.json").write_text("{}")
        (cached / "model.safetensors").write_text("x")
        self._mark_complete(cached, {"repo": "example/notice-vlm", "revision": None})

        monkeypatch.chdir(tmp_path)
        caplog.set_level(logging.WARNING, logger="libreyolo.models.vlm.base")
        m = object.__new__(NoticeVLM)
        m.size = "x"

        assert Path(m._ensure_weights()).resolve() == cached.resolve()
        assert "cached snapshot notice" in caplog.text


class TestCastInputs:
    def test_casts_float_tensors_inside_mutable_payload(self):
        import torch

        m = _bare_model()
        m._model_dtype = torch.float16
        payload = {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
            "nested": [{"image_grid_thw": torch.ones((1, 3), dtype=torch.float32)}],
        }

        out = m._cast_inputs(payload)

        assert out["input_ids"].dtype == torch.int64
        assert out["pixel_values"].dtype == torch.float16
        assert out["nested"][0]["image_grid_thw"].dtype == torch.float16

    def test_prepare_generation_inputs_drops_token_type_ids(self):
        import torch

        m = _bare_model()
        m._model_dtype = torch.float16
        payload = {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "token_type_ids": torch.tensor([[0, 0]], dtype=torch.int64),
            "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
        }

        out = m._prepare_generation_inputs(payload)

        assert "token_type_ids" not in out
        assert out["pixel_values"].dtype == torch.float16


class _StubProc:
    def __init__(self, payload):
        self._payload = payload

    def batch_decode(self, *args, **kwargs):
        return ["<decoded>"]

    def post_process_generation(self, *args, **kwargs):
        return self._payload


class TestOverrideConfThreshold:
    """Florence-2 and Kosmos-2 honor conf= even though they build the dict directly."""

    def _florence(self):
        from libreyolo.models.vlm.florence2 import LibreFlorence2

        m = object.__new__(LibreFlorence2)
        m._name_to_id = {"boat": 0}
        m.processor = _StubProc(
            {
                LibreFlorence2.TASK: {
                    "bboxes": [[0, 0, 10, 10]],
                    "bboxes_labels": ["boat"],
                }
            }
        )
        return m

    def _kosmos(self):
        from libreyolo.models.vlm.kosmos2 import LibreKosmos2

        m = object.__new__(LibreKosmos2)
        m._name_to_id = {"boat": 0}
        m.processor = _StubProc(("a boat", [("boat", (0, 4), [[0.0, 0.0, 0.5, 0.5]])]))
        return m

    def test_florence_conf_below_score_keeps(self):
        det = self._florence()._postprocess(None, 0.5, 0.5, (100, 100))
        assert det["num_detections"] == 1

    def test_florence_conf_above_score_drops_all(self):
        det = self._florence()._postprocess(None, 1.5, 0.5, (100, 100))
        assert det["num_detections"] == 0

    def test_kosmos_conf_below_score_keeps(self):
        det = self._kosmos()._postprocess(None, 0.5, 0.5, (100, 100))
        assert det["num_detections"] == 1

    def test_kosmos_conf_above_score_drops_all(self):
        det = self._kosmos()._postprocess(None, 1.5, 0.5, (100, 100))
        assert det["num_detections"] == 0


class TestKosmosMatchLabel:
    """Kosmos-2's lenient noun-phrase to vocabulary matching (pure, offline)."""

    def _kosmos(self, names):
        from libreyolo.models.vlm.kosmos2 import LibreKosmos2

        m = object.__new__(LibreKosmos2)
        m._name_to_id = {n.lower(): i for i, n in enumerate(names)}
        return m

    def test_exact_match(self):
        assert self._kosmos(["boat", "person"])._match_label("boat") == 0

    def test_lenient_plural_phrase(self):
        # Kosmos grounds noun phrases ("the boats"); lenient substring still maps.
        assert self._kosmos(["boat"])._match_label("the boats") == 0

    def test_out_of_vocab_returns_none(self):
        assert self._kosmos(["boat"])._match_label("airplane") is None


class TestInternVL3Flatten:
    """InternVL3's nested-box flatten override (pure, offline)."""

    def _flat(self, items):
        from libreyolo.models.vlm.internvl3 import LibreInternVL3

        return LibreInternVL3._flatten_nested(items)

    def test_nested_boxes_expand_to_one_item_each(self):
        items = [
            {"label": "boat", "bbox": [[120, 400, 250, 550], [600, 100, 700, 200]]}
        ]
        assert self._flat(items) == [
            {"label": "boat", "bbox": [120, 400, 250, 550]},
            {"label": "boat", "bbox": [600, 100, 700, 200]},
        ]

    def test_flat_box_passes_through(self):
        items = [{"label": "boat", "bbox": [120, 400, 250, 550]}]
        assert self._flat(items) == items

    def test_mixed_nested_and_flat(self):
        items = [
            {"label": "boat", "bbox": [[1, 2, 3, 4], [5, 6, 7, 8]]},
            {"label": "ship", "bbox": [9, 10, 11, 12]},
        ]
        assert self._flat(items) == [
            {"label": "boat", "bbox": [1, 2, 3, 4]},
            {"label": "boat", "bbox": [5, 6, 7, 8]},
            {"label": "ship", "bbox": [9, 10, 11, 12]},
        ]
