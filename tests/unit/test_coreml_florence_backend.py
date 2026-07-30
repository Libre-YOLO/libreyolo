"""Portable-bundle and fake-runtime tests for Florence-2 Core ML."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from libreyolo.backends import coreml_florence as backend
from libreyolo.export.coreml_florence import (
    FLORENCE2_BASE_REQUIRED_ASSETS,
    FLORENCE2_TASK,
    FLORENCE_BEAM_PARENT_INDICES_INPUT,
    FLORENCE_CAUSAL_MASK_INPUT,
    FLORENCE_CROSS_ATTENTION_MASK_INPUT,
    FLORENCE_CROSS_KEY_CACHE_STATE,
    FLORENCE_CROSS_KEY_OUTPUT,
    FLORENCE_CROSS_VALUE_CACHE_STATE,
    FLORENCE_CROSS_VALUE_OUTPUT,
    FLORENCE_DECODE_FUNCTION,
    FLORENCE_DECODER_INPUT_IDS_INPUT,
    FLORENCE_ENCODE_FUNCTION,
    FLORENCE_ENCODER_ATTENTION_MASK_INPUT,
    FLORENCE_ENCODER_INPUT_IDS_INPUT,
    FLORENCE_LAST_LOGITS_OUTPUT,
    FLORENCE_PIXEL_VALUES_INPUT,
    FlorenceCoreMLProfile,
    florence2_base_coreml_metadata,
    stringify_florence_coreml_metadata,
)

pytestmark = [pytest.mark.unit, pytest.mark.experimental_backend]


def _fake_ct(load_model=None):
    if load_model is None:

        def load_model(*_args, **_kwargs):
            return None

    return SimpleNamespace(
        __version__="9.0",
        ComputeUnit=SimpleNamespace(
            ALL="ALL",
            CPU_AND_GPU="CPU_AND_GPU",
            CPU_AND_NE="CPU_AND_NE",
            CPU_ONLY="CPU_ONLY",
        ),
        models=SimpleNamespace(MLModel=load_model),
        utils=SimpleNamespace(load_spec=lambda _path: object()),
    )


def _source_tree(tmp_path: Path):
    package = tmp_path / "source.mlpackage"
    weights = package / "Data" / "com.apple.CoreML" / "weights"
    weights.mkdir(parents=True)
    (package / "Manifest.json").write_text(
        '{"fileFormatVersion":"1.0.0"}',
        encoding="utf-8",
    )
    (weights / "weight.bin").write_bytes(b"converted-florence-weights")
    processor = tmp_path / "processor"
    processor.mkdir()
    values = {}
    for index, name in enumerate(FLORENCE2_BASE_REQUIRED_ASSETS):
        value = f"florence-asset-{index}-{name}".encode()
        (processor / name).write_bytes(value)
        values[name] = hashlib.sha256(value).hexdigest()
    return package, processor, values


def _patch_bundle_contract(monkeypatch, asset_hashes):
    profile = backend.florence2_base_coreml_profile()
    metadata = florence2_base_coreml_metadata(profile)
    monkeypatch.setattr(
        backend,
        "FLORENCE2_BASE_REQUIRED_ASSETS",
        dict(asset_hashes),
    )
    monkeypatch.setattr(
        backend,
        "_load_package_contract",
        lambda *_args, **_kwargs: (profile, metadata, {}),
    )
    monkeypatch.setattr(
        backend,
        "validate_florence2_base_processor_assets",
        lambda *_args, **_kwargs: metadata["processor"],
    )
    return profile, metadata


def test_bundle_is_exact_self_contained_and_preserves_source(
    monkeypatch,
    tmp_path,
):
    package, processor, hashes = _source_tree(tmp_path)
    profile, metadata = _patch_bundle_contract(monkeypatch, hashes)
    destination = tmp_path / "florence.coremlvlm"

    result = backend.build_coreml_florence_bundle(
        package,
        processor_dir=processor,
        output_path=destination,
        coremltools_module=_fake_ct(),
    )

    assert result == str(destination)
    assert package.is_dir()
    assert (destination / "Model.mlpackage" / "Manifest.json").is_file()
    assert sorted(
        path.name for path in (destination / "Processor").iterdir()
    ) == sorted(hashes)
    assert not list(destination.rglob("model.safetensors"))
    assert "Copyright (c) Microsoft Corporation" in (
        destination / "LICENSES" / "MIT-Florence.txt"
    ).read_text(encoding="utf-8")
    assert (
        (destination / "LICENSES" / "Apache-2.0.txt")
        .read_text(encoding="utf-8")
        .lstrip()
        .startswith("Apache License")
    )
    notice = (destination / "NOTICE.txt").read_text(encoding="utf-8")
    assert "00921df66db728a9ceb750f5eca43e5c203a2051" in notice
    assert "ddb849abe009d1089e6c691bfc897f27211c663c" in notice

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == profile.as_dict()
    assert (
        manifest["coreml_contract_sha256"]
        == metadata["coreml_florence_contract_sha256"]
    )
    assert manifest["source_weights_included"] is False
    assert manifest["provenance"]["model"]["license"] == "MIT"
    assert manifest["provenance"]["transformers"]["license"] == "Apache-2.0"
    assert set(manifest["payload_files"]) == {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }

    info = backend.validate_coreml_florence_bundle(
        destination,
        coremltools_module=_fake_ct(),
    )
    assert info.profile == profile
    assert info.model_path == (destination / "Model.mlpackage").resolve()
    assert info.processor_path == (destination / "Processor").resolve()


def test_bundle_rejects_overwrite_and_tamper_without_mutating_source(
    monkeypatch,
    tmp_path,
):
    package, processor, hashes = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch, hashes)
    destination = tmp_path / "existing.coremlvlm"
    destination.mkdir()
    sentinel = destination / "mine.txt"
    sentinel.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        backend.build_coreml_florence_bundle(
            package,
            processor_dir=processor,
            output_path=destination,
            coremltools_module=_fake_ct(),
        )
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert package.is_dir()

    destination = tmp_path / "built.coremlvlm"
    backend.build_coreml_florence_bundle(
        package,
        processor_dir=processor,
        output_path=destination,
        coremltools_module=_fake_ct(),
    )
    (destination / "Processor" / next(iter(hashes))).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte length|SHA-256"):
        backend.validate_coreml_florence_bundle(
            destination,
            coremltools_module=_fake_ct(),
        )


def test_bundle_publication_fails_closed_without_atomic_rename(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "staged.coremlvlm"
    destination = tmp_path / "published.coremlvlm"
    source.mkdir()
    monkeypatch.setattr(backend.os, "name", "posix")
    monkeypatch.setattr(backend.sys, "platform", "unsupported")
    monkeypatch.setattr(
        backend.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="atomic no-replace"):
        backend._publish_directory_no_replace(source, destination)

    assert source.is_dir()
    assert not destination.exists()


def test_bundle_rejects_destination_inside_source_and_renamed_source_weight(
    monkeypatch,
    tmp_path,
):
    package, processor, hashes = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch, hashes)
    nested = package / "nested" / "bad.coremlvlm"
    with pytest.raises(ValueError, match="inside the source package"):
        backend.build_coreml_florence_bundle(
            package,
            processor_dir=processor,
            output_path=nested,
            coremltools_module=_fake_ct(),
        )
    assert not nested.exists()

    disguised = package / "Data" / "com.apple.CoreML" / "weights" / "innocent.bin"
    disguised.write_bytes(b"source-weight")
    monkeypatch.setattr(
        backend,
        "FLORENCE2_BASE_WEIGHTS_SIZE",
        len(b"source-weight"),
    )
    monkeypatch.setattr(
        backend,
        "FLORENCE2_BASE_WEIGHTS_SHA256",
        hashlib.sha256(b"source-weight").hexdigest(),
    )
    with pytest.raises(ValueError, match="renamed"):
        backend.build_coreml_florence_bundle(
            package,
            processor_dir=processor,
            output_path=tmp_path / "blocked.coremlvlm",
            coremltools_module=_fake_ct(),
        )


def _tiny_profile():
    return FlorenceCoreMLProfile(
        image_size=4,
        image_token_count=2,
        encoder_context_length=4,
        decoder_context_length=4,
        hidden_size=2,
        vocab_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        head_dim=2,
        max_new_tokens=4,
    )


class _TinyCursor:
    def __init__(self, profile):
        self.profile = profile
        self.position = 0

    def controls(self):
        return (
            np.zeros(
                (self.profile.num_beams, 1, 1, self.position + 1),
                dtype=np.float16,
            ),
            np.full(
                (self.profile.num_beams, 1),
                self.position,
                dtype=np.int32,
            ),
        )

    def commit(self, *, causal_mask, position_ids):
        assert causal_mask.shape[-1] == self.position + 1
        assert np.all(position_ids == self.position)
        self.position += 1


class _Beam:
    def __init__(self, *, max_new_tokens, vocab_size):
        assert max_new_tokens == 2
        assert vocab_size == 8
        self.done = False
        self.calls = 0
        self.output_sequence = (2, 0, 2)
        self.output_score = -0.5

    def advance(self, logits):
        assert np.asarray(logits).shape == (3, 8)
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(
                done=False,
                next_token_ids=(4, 5, 6),
                parent_indices=(2, 0, 0),
            )
        self.done = True
        return SimpleNamespace(
            done=True,
            next_token_ids=None,
            parent_indices=None,
        )


class _FakeProcessor:
    def __init__(self):
        self.decode_calls = []
        self.post_calls = []

    def batch_decode(self, token_ids, **kwargs):
        self.decode_calls.append((np.asarray(token_ids), kwargs))
        return ["<s>cat<loc_0><loc_0><loc_500><loc_500></s>"]

    def post_process_generation(self, text, **kwargs):
        self.post_calls.append((text, kwargs))
        return {
            FLORENCE2_TASK: {
                "bboxes": [
                    [1, 2, 20, 30],
                    [3, 4, 10, 12],
                    [0, 0, float("nan"), 5],
                ],
                "bboxes_labels": ["Cat", "unknown", "dog"],
            }
        }


class _FakeState:
    def __init__(self, *, fail_second_write=False):
        self.writes = []
        self.fail_second_write = fail_second_write

    def write_state(self, *, name, value):
        self.writes.append((name, np.asarray(value)))
        if self.fail_second_write and len(self.writes) == 2:
            self.fail_second_write = False
            raise RuntimeError("simulated state seed failure")


class _FakeModel:
    def __init__(
        self,
        function_name,
        metadata,
        calls,
        *,
        fail_seed_once=False,
        fail_decode_once=False,
    ):
        self.function_name = function_name
        self.user_defined_metadata = metadata
        self.calls = calls
        self.fail_seed_once = fail_seed_once
        self.fail_decode_once = fail_decode_once
        self.states = []

    def make_state(self):
        state = _FakeState(fail_second_write=self.fail_seed_once)
        self.fail_seed_once = False
        self.states.append(state)
        return state

    def predict(self, inputs, **kwargs):
        self.calls.append((self.function_name, inputs, kwargs))
        if self.function_name == FLORENCE_ENCODE_FUNCTION:
            key = np.arange(8, dtype=np.float16).reshape(1, 1, 1, 4, 2)
            return {
                FLORENCE_CROSS_KEY_OUTPUT: key,
                FLORENCE_CROSS_VALUE_OUTPUT: key + np.float16(20),
            }
        state = kwargs["state"]
        assert state is self.states[-1]
        assert [name for name, _value in state.writes] == [
            FLORENCE_CROSS_KEY_CACHE_STATE,
            FLORENCE_CROSS_VALUE_CACHE_STATE,
        ]
        if self.fail_decode_once:
            self.fail_decode_once = False
            raise RuntimeError("simulated decode failure")
        return {FLORENCE_LAST_LOGITS_OUTPUT: np.zeros((3, 8), dtype=np.float16)}


def test_runtime_fp16_output_normalizes_apple_float32_materialization():
    values = np.asarray([1.5, -2.25], dtype=np.float32)

    actual = backend._fp16_output(
        {"output": values},
        expected_names={"output"},
        name="output",
        shape=(2,),
    )

    assert actual.dtype == np.float16
    assert actual.flags.c_contiguous
    assert np.array_equal(actual, values.astype(np.float16))


def test_runtime_fp16_output_rejects_non_floating_materialization():
    with pytest.raises(RuntimeError, match="float16 or float32"):
        backend._fp16_output(
            {"output": np.asarray([1, 2], dtype=np.int32)},
            expected_names={"output"},
            name="output",
            shape=(2,),
        )


def test_runtime_default_rejects_before_processor_or_native_proxy(
    monkeypatch,
    tmp_path,
):
    profile = FlorenceCoreMLProfile()
    metadata = florence2_base_coreml_metadata(profile)
    bundle = tmp_path / "experimental.coremlvlm"
    model_path = bundle / "Model.mlpackage"
    processor_path = bundle / "Processor"
    model_path.mkdir(parents=True)
    processor_path.mkdir()
    info = backend.CoreMLFlorenceBundleInfo(
        path=bundle.resolve(),
        model_path=model_path.resolve(),
        processor_path=processor_path.resolve(),
        profile=profile,
        metadata=metadata,
        manifest={},
    )
    monkeypatch.setattr(
        backend,
        "validate_coreml_florence_bundle",
        lambda *_args, **_kwargs: info,
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    processor_calls = []
    proxy_calls = []
    monkeypatch.setattr(
        backend,
        "_load_florence_processor",
        lambda path: processor_calls.append(path),
    )

    def load_model(*args, **kwargs):
        proxy_calls.append((args, kwargs))
        return object()

    with pytest.raises(NotImplementedError, match="explicitly experimental"):
        backend.CoreMLFlorenceRuntime(
            bundle,
            names=["cat"],
            coremltools_module=_fake_ct(load_model),
        )

    assert processor_calls == []
    assert proxy_calls == []


def test_runtime_rejects_invalid_names_before_bundle_or_toolchain(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(sys, "platform", "darwin")
    toolchain_calls = []
    monkeypatch.setattr(
        backend,
        "_package_ct",
        lambda *_args, **_kwargs: toolchain_calls.append(True),
    )

    with pytest.raises(ValueError, match="unique"):
        backend.CoreMLFlorenceRuntime(
            tmp_path / "missing.coremlvlm",
            names=["cat", "CAT"],
            compute_units="cpu_only",
        )

    assert toolchain_calls == []


def _runtime(
    monkeypatch,
    tmp_path,
    *,
    fail_seed_once=False,
    fail_decode_once=False,
):
    profile = _tiny_profile()
    metadata = florence2_base_coreml_metadata()
    bundle = tmp_path / "runtime.coremlvlm"
    model_path = bundle / "Model.mlpackage"
    processor_path = bundle / "Processor"
    model_path.mkdir(parents=True)
    processor_path.mkdir()
    info = backend.CoreMLFlorenceBundleInfo(
        path=bundle.resolve(),
        model_path=model_path.resolve(),
        processor_path=processor_path.resolve(),
        profile=profile,
        metadata=metadata,
        manifest={},
    )
    monkeypatch.setattr(
        backend,
        "validate_coreml_florence_bundle",
        lambda *_args, **_kwargs: info,
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(backend, "FlorenceDecodeCursor", _TinyCursor)
    monkeypatch.setattr(backend, "Florence2BeamSearch", _Beam)
    cross_mask = np.zeros((3, 1, 1, 4), dtype=np.float16)

    def prepare(processor, image, class_names, *, profile):
        assert processor is fake_processor
        assert image.size == (32, 24)
        assert class_names == ["cat", "dog"]
        return {
            FLORENCE_PIXEL_VALUES_INPUT: np.zeros((1, 3, 4, 4), dtype=np.float16),
            FLORENCE_ENCODER_INPUT_IDS_INPUT: np.asarray(
                [[7, 7, 0, 1]], dtype=np.int32
            ),
            FLORENCE_ENCODER_ATTENTION_MASK_INPUT: np.zeros(
                (1, 1, 1, 4), dtype=np.float16
            ),
            FLORENCE_CROSS_ATTENTION_MASK_INPUT: cross_mask,
        }

    monkeypatch.setattr(
        backend,
        "prepare_florence2_base_processor_batch",
        prepare,
    )
    calls = []
    models = {}
    runtime_metadata = stringify_florence_coreml_metadata(metadata)

    def load_model(path, **kwargs):
        assert path == str(model_path.resolve())
        function_name = kwargs["function_name"]
        model = _FakeModel(
            function_name,
            runtime_metadata,
            calls,
            fail_seed_once=(
                fail_seed_once and function_name == FLORENCE_DECODE_FUNCTION
            ),
            fail_decode_once=(
                fail_decode_once and function_name == FLORENCE_DECODE_FUNCTION
            ),
        )
        models[function_name] = model
        return model

    fake_processor = _FakeProcessor()
    monkeypatch.setattr(
        backend,
        "_load_florence_processor",
        lambda path: fake_processor,
    )
    runtime = backend.CoreMLFlorenceRuntime(
        bundle,
        names=["cat", "dog"],
        compute_units="cpu_only",
        coremltools_module=_fake_ct(load_model),
    )
    return runtime, fake_processor, calls, models


def test_runtime_seeds_cross_state_before_decode_and_propagates_beam_parents(
    monkeypatch,
    tmp_path,
):
    runtime, processor, calls, models = _runtime(monkeypatch, tmp_path)
    generated = runtime.generate(
        Image.new("RGB", (32, 24)),
        max_new_tokens=2,
    )

    assert generated["token_ids"] == [2, 0, 2]
    assert generated["beam_score"] == -0.5
    decode = models[FLORENCE_DECODE_FUNCTION]
    assert len(decode.states) == 1
    state = decode.states[0]
    assert [name for name, _value in state.writes] == [
        FLORENCE_CROSS_KEY_CACHE_STATE,
        FLORENCE_CROSS_VALUE_CACHE_STATE,
    ]
    key = state.writes[0][1]
    value = state.writes[1][1]
    assert key.dtype == value.dtype == np.float32
    assert key.shape == value.shape == (1, 3, 1, 4, 2)
    for beam in range(3):
        assert np.array_equal(key[:, beam], key[:, 0])
        assert np.array_equal(value[:, beam], value[:, 0])
    decode_calls = [
        inputs for name, inputs, _kwargs in calls if name == FLORENCE_DECODE_FUNCTION
    ]
    assert len(decode_calls) == 2
    assert np.array_equal(
        decode_calls[0][FLORENCE_DECODER_INPUT_IDS_INPUT],
        np.asarray([[2], [2], [2]], dtype=np.int32),
    )
    assert np.array_equal(
        decode_calls[0][FLORENCE_BEAM_PARENT_INDICES_INPUT],
        np.asarray([0, 1, 2], dtype=np.int32),
    )
    assert np.array_equal(
        decode_calls[1][FLORENCE_DECODER_INPUT_IDS_INPUT],
        np.asarray([[4], [5], [6]], dtype=np.int32),
    )
    assert np.array_equal(
        decode_calls[1][FLORENCE_BEAM_PARENT_INDICES_INPUT],
        np.asarray([2, 0, 0], dtype=np.int32),
    )
    assert decode_calls[0][FLORENCE_CAUSAL_MASK_INPUT].shape[-1] == 1
    assert decode_calls[1][FLORENCE_CAUSAL_MASK_INPUT].shape[-1] == 2
    assert processor.post_calls[0][1] == {
        "task": FLORENCE2_TASK,
        "image_size": (32, 24),
    }


def test_runtime_uses_fresh_state_for_every_request(
    monkeypatch,
    tmp_path,
):
    runtime, _processor, _calls, models = _runtime(monkeypatch, tmp_path)
    image = Image.new("RGB", (32, 24))

    first = runtime.predict(image, max_new_tokens=2)
    second = runtime.predict(image, max_new_tokens=2, classes=[1])

    assert first == {
        "boxes": [[1.0, 2.0, 20.0, 30.0]],
        "scores": [1.0],
        "classes": [0],
        "num_detections": 1,
    }
    assert second["num_detections"] == 0
    decode = models[FLORENCE_DECODE_FUNCTION]
    assert len(decode.states) == 2
    assert decode.states[0] is not decode.states[1]
    assert all(len(state.writes) == 2 for state in decode.states)
    assert runtime._active_decode is None


@pytest.mark.parametrize("failure", ["seed", "decode"])
def test_failed_state_request_is_discarded_and_next_request_is_clean(
    monkeypatch,
    tmp_path,
    failure,
):
    runtime, _processor, _calls, models = _runtime(
        monkeypatch,
        tmp_path,
        fail_seed_once=failure == "seed",
        fail_decode_once=failure == "decode",
    )
    image = Image.new("RGB", (32, 24))

    with pytest.raises(RuntimeError, match=failure):
        runtime.generate(image, max_new_tokens=2)
    assert runtime._active_decode is None

    generated = runtime.generate(image, max_new_tokens=2)
    assert generated["token_ids"] == [2, 0, 2]
    decode = models[FLORENCE_DECODE_FUNCTION]
    assert len(decode.states) == 2
    assert decode.states[0] is not decode.states[1]


def test_runtime_fail_closes_names_limits_and_closed_state(
    monkeypatch,
    tmp_path,
):
    runtime, _processor, _calls, _models = _runtime(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="unique"):
        runtime.set_classes(["cat", "CAT"])
    with pytest.raises(ValueError, match="contiguous"):
        runtime.set_classes({1: "cat"})
    with pytest.raises(ValueError, match="exceeds"):
        runtime.generate(
            Image.new("RGB", (32, 24)),
            max_new_tokens=5,
        )
    with pytest.raises(TypeError, match="integer"):
        runtime.generate(
            Image.new("RGB", (32, 24)),
            max_new_tokens=True,
        )
    with pytest.raises(ValueError, match="finite"):
        runtime.predict(
            Image.new("RGB", (32, 24)),
            conf=float("nan"),
        )

    runtime.close()
    assert runtime.closed
    with pytest.raises(RuntimeError, match="closed"):
        runtime.generate(Image.new("RGB", (32, 24)), max_new_tokens=2)
