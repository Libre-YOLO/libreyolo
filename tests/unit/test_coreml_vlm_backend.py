"""Portable-bundle and fake-runtime tests for the dedicated Core ML VLM host."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from libreyolo.backends import coreml_vlm as backend
from libreyolo.export.coreml_vlm import (
    COREML_VLM_DECODE_FUNCTION,
    COREML_VLM_EMBED_TOKENS_FUNCTION,
    COREML_VLM_ENCODE_IMAGE_FUNCTION,
    COREML_VLM_FUNCTION_NAMES,
    COREML_VLM_INPUT_IDS_INPUT,
    COREML_VLM_LAST_LOGITS_OUTPUT,
    COREML_VLM_PIXEL_VALUES_INPUT,
    COREML_VLM_TOKEN_EMBEDDINGS_OUTPUT,
    COREML_VLM_TRANSFORMERS_VERSION,
    SMOLVLM2_500M_EOS_TOKEN_ID,
    SMOLVLM2_500M_IMAGE_TOKEN_ID,
    SMOLVLM2_500M_REQUIRED_ASSETS,
    prepare_smolvlm2_500m_coreml_processor_batch,
    preprocess_smolvlm2_500m_coreml_image,
    smolvlm2_500m_coreml_metadata,
    smolvlm2_500m_coreml_profile,
    stringify_coreml_vlm_metadata,
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
    (package / "Data" / "com.apple.CoreML" / "weights").mkdir(parents=True)
    (package / "Manifest.json").write_text('{"fileFormatVersion":"1.0.0"}')
    (package / "Data" / "com.apple.CoreML" / "weights" / "weight.bin").write_bytes(
        b"converted-coreml-weights"
    )
    processor = tmp_path / "snapshot"
    processor.mkdir()
    for index, name in enumerate(SMOLVLM2_500M_REQUIRED_ASSETS):
        (processor / name).write_bytes(f"asset-{index}-{name}".encode())
    return package, processor


def _patch_bundle_contract(monkeypatch, *, context=2048):
    profile = smolvlm2_500m_coreml_profile(context)
    metadata = smolvlm2_500m_coreml_metadata(profile)
    monkeypatch.setattr(
        backend,
        "_load_package_contract",
        lambda *_args, **_kwargs: (profile, metadata, {}),
    )
    monkeypatch.setattr(
        backend,
        "validate_smolvlm2_500m_processor_assets",
        lambda *_args, **_kwargs: metadata["processor"],
    )
    return profile, metadata


def test_bundle_is_self_contained_exact_and_preserves_existing_package(
    monkeypatch,
    tmp_path,
):
    package, processor = _source_tree(tmp_path)
    profile, metadata = _patch_bundle_contract(monkeypatch)
    destination = tmp_path / "smol.coremlvlm"

    result = backend.build_coreml_vlm_bundle(
        package,
        processor_dir=processor,
        output_path=destination,
        coremltools_module=_fake_ct(),
    )

    assert result == str(destination)
    assert package.is_dir()
    assert (destination / "Model.mlpackage" / "Manifest.json").is_file()
    processor_files = sorted(
        path.name for path in (destination / "Processor").iterdir()
    )
    assert processor_files == sorted(SMOLVLM2_500M_REQUIRED_ASSETS)
    assert not list(destination.rglob("model.safetensors"))
    assert (
        destination / "LICENSES" / "Apache-2.0.txt"
    ).read_text().lstrip().startswith("Apache License")
    assert "revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467" in (
        destination / "NOTICE.txt"
    ).read_text()

    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["source_weights_included"] is False
    assert manifest["profile"] == profile.as_dict()
    assert (
        manifest["coreml_contract_sha256"]
        == metadata["coreml_vlm_contract_sha256"]
    )
    assert manifest["provenance"]["model"]["license"] == "Apache-2.0"
    assert manifest["provenance"]["bundle_runtime"]["license"] == "MIT"
    assert manifest["licenses"]["Apache-2.0"]["spdx"] == "Apache-2.0"
    assert (
        manifest["licenses"]["Apache-2.0"]["sha256"]
        == manifest["payload_files"]["LICENSES/Apache-2.0.txt"]["sha256"]
    )
    assert manifest["notice"]["sha256"] == manifest["payload_files"][
        "NOTICE.txt"
    ]["sha256"]
    assert set(manifest["payload_files"]) == {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }

    info = backend.validate_coreml_vlm_bundle(
        destination,
        coremltools_module=_fake_ct(),
    )
    assert info.profile == profile
    assert info.model_path == (destination / "Model.mlpackage").resolve()
    assert info.processor_path == (destination / "Processor").resolve()


def test_bundle_default_is_no_overwrite(monkeypatch, tmp_path):
    package, processor = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch)
    destination = tmp_path / "existing.coremlvlm"
    destination.mkdir()
    sentinel = destination / "mine.txt"
    sentinel.write_text("unchanged")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        backend.build_coreml_vlm_bundle(
            package,
            processor_dir=processor,
            output_path=destination,
            coremltools_module=_fake_ct(),
        )
    assert sentinel.read_text() == "unchanged"
    assert package.is_dir()


def test_bundle_rejects_destination_inside_source_without_mutation(
    monkeypatch,
    tmp_path,
):
    package, processor = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch)
    nested_parent = package / "must-not-be-created"

    with pytest.raises(ValueError, match="inside the source package"):
        backend.build_coreml_vlm_bundle(
            package,
            processor_dir=processor,
            output_path=nested_parent / "bad.coremlvlm",
            coremltools_module=_fake_ct(),
        )

    assert not nested_parent.exists()


def test_orchestrated_move_publishes_package_without_copy(
    monkeypatch,
    tmp_path,
):
    package, processor = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch)
    destination = tmp_path / "moved.coremlvlm"

    backend.build_coreml_vlm_bundle(
        package,
        processor_dir=processor,
        output_path=destination,
        move_model=True,
        coremltools_module=_fake_ct(),
    )

    assert not package.exists()
    assert (destination / "Model.mlpackage" / "Manifest.json").is_file()


def test_orchestrated_move_rolls_back_when_publication_loses_race(
    monkeypatch,
    tmp_path,
):
    package, processor = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch)
    destination = tmp_path / "raced.coremlvlm"
    original_publish = backend._publish_directory_no_replace

    def race(source, target):
        destination.mkdir()
        return original_publish(source, target)

    monkeypatch.setattr(backend, "_publish_directory_no_replace", race)
    with pytest.raises(FileExistsError):
        backend.build_coreml_vlm_bundle(
            package,
            processor_dir=processor,
            output_path=destination,
            move_model=True,
            coremltools_module=_fake_ct(),
        )

    assert package.is_dir()
    assert (package / "Manifest.json").is_file()
    assert destination.is_dir()


def test_orchestrated_move_preserves_model_when_source_path_is_reoccupied(
    monkeypatch,
    tmp_path,
):
    package, processor = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch)

    def collide(_root, _payload):
        package.mkdir()
        raise RuntimeError("simulated validation failure")

    monkeypatch.setattr(backend, "_verify_staged_payload", collide)
    with pytest.raises(RuntimeError, match="original is preserved"):
        backend.build_coreml_vlm_bundle(
            package,
            processor_dir=processor,
            output_path=tmp_path / "collision.coremlvlm",
            move_model=True,
            coremltools_module=_fake_ct(),
        )

    preserved = list(
        tmp_path.glob(
            ".libreyolo-coreml-vlm-bundle-*/Model.mlpackage/Manifest.json"
        )
    )
    assert len(preserved) == 1
    assert preserved[0].is_file()


def test_bundle_hash_inventory_rejects_mutation_and_additions(
    monkeypatch,
    tmp_path,
):
    package, processor = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch)
    destination = tmp_path / "strict.coremlvlm"
    backend.build_coreml_vlm_bundle(
        package,
        processor_dir=processor,
        output_path=destination,
        coremltools_module=_fake_ct(),
    )
    changed = destination / "Processor" / "vocab.json"
    changed.write_bytes(changed.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed byte length"):
        backend.validate_coreml_vlm_bundle(
            destination,
            coremltools_module=_fake_ct(),
        )

    changed.write_bytes(processor.joinpath("vocab.json").read_bytes())
    (destination / "Processor" / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="payload inventory changed"):
        backend.validate_coreml_vlm_bundle(
            destination,
            coremltools_module=_fake_ct(),
        )


def test_bundle_rejects_boolean_schema_version(monkeypatch, tmp_path):
    package, processor = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch)
    destination = tmp_path / "boolean-version.coremlvlm"
    backend.build_coreml_vlm_bundle(
        package,
        processor_dir=processor,
        output_path=destination,
        coremltools_module=_fake_ct(),
    )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bundle_schema_version"] = True
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="identity contract changed"):
        backend.validate_coreml_vlm_bundle(
            destination,
            coremltools_module=_fake_ct(),
        )


def _declare_bundle_file(bundle: Path, relative_name: str, content: bytes):
    path = bundle / Path(relative_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["payload_files"][relative_name] = {
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest))


def test_bundle_rejects_manifest_declared_payload_outside_approved_roots(
    monkeypatch,
    tmp_path,
):
    package, processor = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch)
    destination = tmp_path / "extra.coremlvlm"
    backend.build_coreml_vlm_bundle(
        package,
        processor_dir=processor,
        output_path=destination,
        coremltools_module=_fake_ct(),
    )
    _declare_bundle_file(destination, "Extra/declared.bin", b"declared")

    with pytest.raises(ValueError, match="outside its approved roots"):
        backend.validate_coreml_vlm_bundle(
            destination,
            coremltools_module=_fake_ct(),
        )


@pytest.mark.parametrize("mode", ["missing", "tampered", "extra"])
def test_bundle_license_payload_is_exact(monkeypatch, tmp_path, mode):
    package, processor = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch)
    destination = tmp_path / f"license-{mode}.coremlvlm"
    backend.build_coreml_vlm_bundle(
        package,
        processor_dir=processor,
        output_path=destination,
        coremltools_module=_fake_ct(),
    )
    license_path = destination / "LICENSES" / "Apache-2.0.txt"
    if mode == "missing":
        license_path.unlink()
        match = "payload inventory changed"
    elif mode == "tampered":
        content = license_path.read_bytes()
        license_path.write_bytes(b"X" + content[1:])
        match = "failed SHA-256"
    else:
        _declare_bundle_file(
            destination,
            "LICENSES/Unexpected.txt",
            b"unexpected",
        )
        match = "outside its approved roots"

    with pytest.raises(ValueError, match=match):
        backend.validate_coreml_vlm_bundle(
            destination,
            coremltools_module=_fake_ct(),
        )


@pytest.mark.parametrize("mode", ["missing", "tampered"])
def test_bundle_notice_payload_is_exact(monkeypatch, tmp_path, mode):
    package, processor = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch)
    destination = tmp_path / f"notice-{mode}.coremlvlm"
    backend.build_coreml_vlm_bundle(
        package,
        processor_dir=processor,
        output_path=destination,
        coremltools_module=_fake_ct(),
    )
    notice_path = destination / "NOTICE.txt"
    if mode == "missing":
        notice_path.unlink()
        match = "payload inventory changed"
    else:
        content = notice_path.read_bytes()
        notice_path.write_bytes(b"X" + content[1:])
        match = "failed SHA-256"

    with pytest.raises(ValueError, match=match):
        backend.validate_coreml_vlm_bundle(
            destination,
            coremltools_module=_fake_ct(),
        )


def test_apple_package_manifest_rejects_unmanifested_file(tmp_path):
    package = tmp_path / "strict.mlpackage"
    model = package / "Data" / "com.apple.CoreML" / "model.mlmodel"
    weight = package / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
    model.parent.mkdir(parents=True)
    weight.parent.mkdir(parents=True)
    model.write_bytes(b"spec")
    weight.write_bytes(b"weights")
    manifest = {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            "model-id": {"path": "com.apple.CoreML/model.mlmodel"},
            "weight-id": {"path": "com.apple.CoreML/weights"},
        },
        "rootModelIdentifier": "model-id",
    }
    (package / "Manifest.json").write_text(json.dumps(manifest))
    backend._validate_apple_package_manifest(
        package,
        backend._safe_tree_files(package, label="test package"),
    )

    (package / "ignored.bin").write_bytes(b"ignored")
    with pytest.raises(ValueError, match="unmanifested payload"):
        backend._validate_apple_package_manifest(
            package,
            backend._safe_tree_files(package, label="test package"),
        )


def test_bundle_rejects_source_safetensors_payload(monkeypatch, tmp_path):
    package, processor = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch)
    (package / "model.safetensors").write_bytes(b"must-never-be-copied")

    with pytest.raises(ValueError, match="2 GB source safetensors"):
        backend.build_coreml_vlm_bundle(
            package,
            processor_dir=processor,
            output_path=tmp_path / "bad.coremlvlm",
            coremltools_module=_fake_ct(),
        )


def test_bundle_rejects_symlinked_source_when_supported(monkeypatch, tmp_path):
    package, processor = _source_tree(tmp_path)
    _patch_bundle_contract(monkeypatch)
    linked = processor / "linked.json"
    try:
        os.symlink(processor / "config.json", linked)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable in this Windows environment")

    with pytest.raises(ValueError, match="symbolic link"):
        backend.build_coreml_vlm_bundle(
            package,
            processor_dir=processor,
            output_path=tmp_path / "linked.coremlvlm",
            coremltools_module=_fake_ct(),
        )


def test_runtime_rejects_8k_profile_before_loading_vendor_model():
    with pytest.raises(ValueError, match="only the reviewed 2K/4K"):
        backend._runtime_profile(8192)


def test_repetition_penalty_matches_sign_rule_and_is_non_mutating():
    logits = np.asarray([[6.0, -4.0, 3.0, 2.0]], dtype=np.float16)
    original = logits.copy()
    adjusted = backend.apply_coreml_vlm_repetition_penalty(
        logits,
        np.asarray([0, 1, 0], dtype=np.int32),
        penalty=2.0,
    )
    assert adjusted.dtype == np.float16
    assert adjusted.tolist() == [[3.0, -8.0, 3.0, 2.0]]
    assert np.array_equal(logits, original)


def test_repetition_penalty_preserves_fp16_tie_like_transformers():
    logits = np.asarray([[1.0, 0.9091796875]], dtype=np.float16)
    adjusted = backend.apply_coreml_vlm_repetition_penalty(
        logits,
        np.asarray([0], dtype=np.int32),
        penalty=1.1,
    )
    assert adjusted.dtype == np.float16
    assert adjusted.view(np.uint16).tolist() == [[15174, 15174]]
    assert int(np.argmax(adjusted[0])) == 0


def _local_pinned_processor_snapshot() -> Path:
    relative = (
        "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
        "/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    )
    candidates = []
    explicit = os.environ.get("LIBREYOLO_SMOLVLM2_500M_SNAPSHOT")
    if explicit:
        candidates.append(Path(explicit))
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        candidates.append(Path(HF_HUB_CACHE) / relative)
    except ImportError:
        pass
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub" / relative)
    for candidate in candidates:
        if all(
            (candidate / name).is_file()
            for name in SMOLVLM2_500M_REQUIRED_ASSETS
        ):
            return candidate
    pytest.skip(
        "exact local SmolVLM2-500M processor snapshot is not available"
    )


@pytest.mark.vlm
def test_exact_local_auto_processor_produces_runtime_crop_and_prompt_abi():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != COREML_VLM_TRANSFORMERS_VERSION:
        pytest.skip(
            "processor integration requires Transformers "
            f"{COREML_VLM_TRANSFORMERS_VERSION}"
        )
    pytest.importorskip("num2words")
    snapshot = _local_pinned_processor_snapshot()
    processor = backend._load_smolvlm2_processor(snapshot)
    image = preprocess_smolvlm2_500m_coreml_image(
        Image.new("RGB", (37, 29), color=(17, 89, 143))
    )
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is visible?"},
            ],
        }
    ]

    batch = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    )
    input_ids = np.asarray(batch["input_ids"])
    pixel_values = np.asarray(batch["pixel_values"])
    pixel_mask = np.asarray(batch["pixel_attention_mask"])
    assert image.size == (2048, 2048)
    assert pixel_values.shape == (1, 17, 3, 512, 512)
    assert pixel_mask.shape == (1, 17, 512, 512)
    assert bool(pixel_mask.all())
    assert (
        np.count_nonzero(input_ids == SMOLVLM2_500M_IMAGE_TOKEN_ID)
        == 1088
    )
    prepared = prepare_smolvlm2_500m_coreml_processor_batch(
        smolvlm2_500m_coreml_profile(2048),
        batch,
        max_new_tokens=512,
    )
    assert prepared[COREML_VLM_INPUT_IDS_INPUT].dtype == np.int32
    assert prepared[COREML_VLM_PIXEL_VALUES_INPUT].dtype == np.float16


class _FakeProcessor:
    def __init__(self):
        self.template_calls = []
        self.decode_calls = []
        self.decoded = "generated answer"

    def apply_chat_template(self, conversation, **kwargs):
        self.template_calls.append((conversation, kwargs))
        return {"fake": "batch"}

    def batch_decode(self, token_ids, **kwargs):
        self.decode_calls.append((np.asarray(token_ids), kwargs))
        return [self.decoded]


class _FakeModel:
    def __init__(
        self,
        function_name,
        metadata,
        calls,
        *,
        fail_decode=False,
    ):
        self.function_name = function_name
        self.user_defined_metadata = metadata
        self.calls = calls
        self.fail_decode = fail_decode
        self.states = []

    def make_state(self):
        state = object()
        self.states.append(state)
        return state

    def predict(self, inputs, **kwargs):
        self.calls.append((self.function_name, inputs, kwargs))
        if self.function_name == COREML_VLM_ENCODE_IMAGE_FUNCTION:
            return {
                "image_embeddings": np.zeros(
                    (1, 1088, 960),
                    dtype=np.float16,
                )
            }
        if self.function_name == COREML_VLM_EMBED_TOKENS_FUNCTION:
            ids = np.asarray(inputs[COREML_VLM_INPUT_IDS_INPUT])
            return {
                COREML_VLM_TOKEN_EMBEDDINGS_OUTPUT: np.zeros(
                    (1, ids.shape[1], 960),
                    dtype=np.float16,
                )
            }
        if self.fail_decode:
            self.fail_decode = False
            raise RuntimeError("simulated Core ML state failure")
        decode_index = sum(
            name == COREML_VLM_DECODE_FUNCTION
            for name, _inputs, _kwargs in self.calls
        )
        logits = np.full((1, 49280), -20.0, dtype=np.float16)
        logits[0, 37 if decode_index == 1 else SMOLVLM2_500M_EOS_TOKEN_ID] = 20.0
        return {COREML_VLM_LAST_LOGITS_OUTPUT: logits}


def test_fp16_output_normalizes_apple_float32_materialization():
    values = np.asarray([1.5, -2.25], dtype=np.float32)

    actual = backend._only_fp16_output(
        {"output": values},
        name="output",
        shape=(2,),
    )

    assert actual.dtype == np.float16
    assert actual.flags.c_contiguous
    assert np.array_equal(actual, values.astype(np.float16))


def test_fp16_output_rejects_non_floating_materialization():
    with pytest.raises(RuntimeError, match="float16 or float32"):
        backend._only_fp16_output(
            {"output": np.asarray([1, 2], dtype=np.int32)},
            name="output",
            shape=(2,),
        )


def _runtime(
    monkeypatch,
    tmp_path,
    *,
    fail_decode=False,
):
    profile = smolvlm2_500m_coreml_profile(2048)
    metadata = smolvlm2_500m_coreml_metadata(profile)
    bundle = tmp_path / "runtime.coremlvlm"
    model_path = bundle / "Model.mlpackage"
    processor_path = bundle / "Processor"
    model_path.mkdir(parents=True)
    processor_path.mkdir()
    info = backend.CoreMLVLMBundleInfo(
        path=bundle.resolve(),
        model_path=model_path.resolve(),
        processor_path=processor_path.resolve(),
        profile=profile,
        metadata=metadata,
        manifest={},
    )
    monkeypatch.setattr(
        backend,
        "validate_coreml_vlm_bundle",
        lambda *_args, **_kwargs: info,
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        backend,
        "preprocess_smolvlm2_500m_coreml_image",
        lambda image: image,
    )
    prompt_ids = np.full((1, 1089), SMOLVLM2_500M_IMAGE_TOKEN_ID, dtype=np.int32)
    prompt_ids[0, -1] = 3

    def prepare(_profile, batch, *, max_new_tokens):
        assert batch == {"fake": "batch"}
        assert max_new_tokens == 3
        return {
            COREML_VLM_INPUT_IDS_INPUT: prompt_ids,
            COREML_VLM_PIXEL_VALUES_INPUT: np.zeros(
                (1, 17, 3, 512, 512),
                dtype=np.float16,
            ),
        }

    monkeypatch.setattr(
        backend,
        "prepare_smolvlm2_500m_coreml_processor_batch",
        prepare,
    )
    calls = []
    loads = []
    models = {}
    runtime_metadata = stringify_coreml_vlm_metadata(metadata)

    def load_model(path, **kwargs):
        function_name = kwargs["function_name"]
        loads.append((path, kwargs))
        model = _FakeModel(
            function_name,
            runtime_metadata,
            calls,
            fail_decode=fail_decode and function_name == COREML_VLM_DECODE_FUNCTION,
        )
        models[function_name] = model
        return model

    processor = _FakeProcessor()
    monkeypatch.setattr(
        backend,
        "_load_smolvlm2_processor",
        lambda path: processor,
    )
    runtime = backend.CoreMLVLMRuntime(
        bundle,
        compute_units="cpu_only",
        coremltools_module=_fake_ct(load_model),
    )
    return runtime, processor, calls, loads, models


def test_fake_runtime_loads_named_functions_and_runs_fresh_stateful_generation(
    monkeypatch,
    tmp_path,
):
    runtime, processor, calls, loads, models = _runtime(monkeypatch, tmp_path)

    text = runtime.chat(
        Image.new("RGB", (9, 7), color=(20, 30, 40)),
        "what is visible?",
        max_new_tokens=3,
    )

    assert text == "generated answer"
    assert [kwargs["function_name"] for _path, kwargs in loads] == list(
        COREML_VLM_FUNCTION_NAMES
    )
    assert all(kwargs["compute_units"] == "CPU_ONLY" for _path, kwargs in loads)
    assert len(models[COREML_VLM_DECODE_FUNCTION].states) == 1
    decode_calls = [
        (inputs, kwargs)
        for name, inputs, kwargs in calls
        if name == COREML_VLM_DECODE_FUNCTION
    ]
    assert len(decode_calls) == 2
    first_inputs, first_kwargs = decode_calls[0]
    second_inputs, second_kwargs = decode_calls[1]
    assert first_kwargs["state"] is second_kwargs["state"]
    assert first_inputs["position_ids"][0, [0, -1]].tolist() == [0, 1088]
    assert first_inputs["causal_mask"].shape == (1, 1, 1089, 1089)
    assert second_inputs["position_ids"].tolist() == [[1089]]
    assert second_inputs["causal_mask"].shape == (1, 1, 1, 1090)
    embedded_ids = [
        inputs[COREML_VLM_INPUT_IDS_INPUT].tolist()
        for name, inputs, _kwargs in calls
        if name == COREML_VLM_EMBED_TOKENS_FUNCTION
    ]
    assert embedded_ids[-1] == [[37]]
    assert runtime._active_decode is None
    assert processor.template_calls[0][1] == {
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
        "tokenize": True,
    }
    assert processor.decode_calls[0][0].tolist() == [
        [37, SMOLVLM2_500M_EOS_TOKEN_ID]
    ]
    assert processor.decode_calls[0][1] == {"skip_special_tokens": True}


def test_decode_failure_discards_pair_and_next_request_gets_fresh_state(
    monkeypatch,
    tmp_path,
):
    runtime, _processor, _calls, _loads, models = _runtime(
        monkeypatch,
        tmp_path,
        fail_decode=True,
    )
    image = Image.new("RGB", (4, 4))

    with pytest.raises(RuntimeError, match="simulated Core ML state failure"):
        runtime.chat(image, "first", max_new_tokens=3)
    assert runtime._active_decode is None
    assert len(models[COREML_VLM_DECODE_FUNCTION].states) == 1

    assert runtime.chat(image, "retry", max_new_tokens=3) == "generated answer"
    assert len(models[COREML_VLM_DECODE_FUNCTION].states) == 2
    assert runtime._active_decode is None


def test_detection_host_parses_normalized_bbox_contract(
    monkeypatch,
    tmp_path,
):
    runtime, processor, _calls, _loads, _models = _runtime(
        monkeypatch,
        tmp_path,
    )
    processor.decoded = (
        '[{"label":"Boat","bbox":[0.1,0.2,0.8,0.9]},'
        '{"label":"ignored","bbox":[0,0,1,1]}]'
    )

    result = runtime.detect(
        Image.new("RGB", (20, 10)),
        "detect boats",
        name_to_id={"boat": 4},
        max_new_tokens=3,
    )

    assert result == {
        "boxes": [[2.0, 2.0, 16.0, 9.0]],
        "scores": [1.0],
        "classes": [4],
        "num_detections": 1,
    }


def test_runtime_default_rejects_before_processor_or_native_proxy(
    monkeypatch,
    tmp_path,
):
    profile = smolvlm2_500m_coreml_profile(2048)
    metadata = smolvlm2_500m_coreml_metadata(profile)
    bundle = tmp_path / "experimental.coremlvlm"
    model_path = bundle / "Model.mlpackage"
    processor_path = bundle / "Processor"
    model_path.mkdir(parents=True)
    processor_path.mkdir()
    info = backend.CoreMLVLMBundleInfo(
        path=bundle.resolve(),
        model_path=model_path.resolve(),
        processor_path=processor_path.resolve(),
        profile=profile,
        metadata=metadata,
        manifest={},
    )
    monkeypatch.setattr(
        backend,
        "validate_coreml_vlm_bundle",
        lambda *_args, **_kwargs: info,
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    processor_calls = []
    proxy_calls = []
    monkeypatch.setattr(
        backend,
        "_load_smolvlm2_processor",
        lambda path: processor_calls.append(path),
    )

    def load_model(*args, **kwargs):
        proxy_calls.append((args, kwargs))
        return object()

    with pytest.raises(NotImplementedError, match="explicitly experimental"):
        backend.CoreMLVLMRuntime(
            bundle,
            coremltools_module=_fake_ct(load_model),
        )

    assert processor_calls == []
    assert proxy_calls == []


def test_runtime_rejects_modified_runtime_metadata(monkeypatch, tmp_path):
    profile = smolvlm2_500m_coreml_profile(2048)
    metadata = smolvlm2_500m_coreml_metadata(profile)
    bundle = tmp_path / "metadata.coremlvlm"
    model_path = bundle / "Model.mlpackage"
    processor_path = bundle / "Processor"
    model_path.mkdir(parents=True)
    processor_path.mkdir()
    info = backend.CoreMLVLMBundleInfo(
        path=bundle.resolve(),
        model_path=model_path.resolve(),
        processor_path=processor_path.resolve(),
        profile=profile,
        metadata=metadata,
        manifest={},
    )
    monkeypatch.setattr(
        backend,
        "validate_coreml_vlm_bundle",
        lambda *_args, **_kwargs: info,
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    bad_metadata = stringify_coreml_vlm_metadata(metadata)
    bad_metadata["coreml_vlm_contract_sha256"] = "0" * 64

    def load_model(_path, **_kwargs):
        return SimpleNamespace(user_defined_metadata=bad_metadata)

    monkeypatch.setattr(
        backend,
        "_load_smolvlm2_processor",
        lambda _path: _FakeProcessor(),
    )
    with pytest.raises(ValueError, match="strict Core ML VLM contract"):
        backend.CoreMLVLMRuntime(
            bundle,
            compute_units="cpu_only",
            coremltools_module=_fake_ct(load_model),
        )


def test_runtime_close_rejects_reuse(monkeypatch, tmp_path):
    runtime, _processor, _calls, _loads, _models = _runtime(
        monkeypatch,
        tmp_path,
    )
    runtime.close()
    assert runtime.closed
    with pytest.raises(RuntimeError, match="closed"):
        runtime.chat(Image.new("RGB", (3, 3)), "closed", max_new_tokens=3)
