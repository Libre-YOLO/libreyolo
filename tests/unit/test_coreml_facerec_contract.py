"""Hermetic trust-boundary tests for face-embedding Core ML export."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from libreyolo.backends.coreml_facerec import (
    validate_facerec_coreml_spec,
)
from libreyolo.export.coreml import _stringify_metadata
from libreyolo.export.coreml_facerec import (
    FACEREC_COREML_ARTIFACT_SCOPE,
    FACEREC_COREML_CONTRACT,
    FACEREC_COREML_GEOMETRY,
    FACEREC_COREML_INPUT_NAME,
    FACEREC_COREML_OUTPUT_NAME,
    FACEREC_COREML_PREPROCESS_HASH_KEY,
    FACEREC_COREML_PREPROCESS_KEY,
    FACEREC_COREML_REQUIRED_COMPUTE_UNITS,
    FACEREC_COREML_SOURCE_HASH_KEY,
    FACEREC_COREML_SOURCE_MANIFEST_KEY,
    _apply_coreml_execution_profile,
    _inspect_onnx_source,
    _official_embedder_spec,
    _official_provenance_metadata,
    _resolve_options,
    _require_exact_onnx2torch,
    _source_manifest_hash,
    _validate_onnx_io,
    _validate_reserved_official_source,
    export_facerec_coreml,
    facerec_coreml_preprocess_hash,
    facerec_onnx_source_manifest,
    validate_facerec_coreml_metadata,
)
from libreyolo.export.coreml_profiles import (
    COREML_EXECUTION_PROFILES,
    COREML_EXECUTION_PROFILES_BY_ID,
    match_coreml_execution_profile,
    merge_coreml_execution_profile_metadata,
)
from libreyolo.export.coreml_identity import COREML_DEPLOYMENT_ABI_SCHEMA

pytestmark = pytest.mark.unit


def _coreml_multiarray_feature(
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: int = 65568,
):
    array = SimpleNamespace(
        shape=list(shape),
        dataType=dtype,
        WhichOneof=lambda _name: None,
    )
    feature_type = SimpleNamespace(
        isOptional=False,
        multiArrayType=array,
        WhichOneof=lambda _name: "multiArrayType",
    )
    return SimpleNamespace(name=name, type=feature_type)


def _facerec_coreml_spec(
    *,
    output_shape: tuple[int, ...] = (1, 512),
    output_dtype: int = 65568,
):
    return SimpleNamespace(
        description=SimpleNamespace(
            input=[
                _coreml_multiarray_feature(
                    FACEREC_COREML_INPUT_NAME,
                    (1, 3, 112, 112),
                )
            ],
            output=[
                _coreml_multiarray_feature(
                    FACEREC_COREML_OUTPUT_NAME,
                    output_shape,
                    dtype=output_dtype,
                )
            ],
        )
    )


def _external_tensor(onnx, name: str, location: str):
    tensor = onnx.TensorProto()
    tensor.name = name
    tensor.data_type = onnx.TensorProto.FLOAT
    tensor.dims.extend([1])
    tensor.data_location = onnx.TensorProto.EXTERNAL
    external = tensor.external_data.add()
    external.key = "location"
    external.value = location
    return tensor


def _write_model(path: Path, model) -> None:
    path.write_bytes(model.SerializeToString())


def _plain_model(onnx):
    from onnx import TensorProto, helper

    source = helper.make_tensor_value_info(
        "aligned", TensorProto.FLOAT, [1, 3, 112, 112]
    )
    output = helper.make_tensor_value_info(
        "embedding", TensorProto.FLOAT, [1, 3, 112, 112]
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["aligned"], ["embedding"])],
        "face",
        [source],
        [output],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    return model


def _model_with_external_initializer(onnx, location: str):
    from onnx import TensorProto, helper

    source = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    weight = _external_tensor(onnx, "weight", location)
    graph = helper.make_graph(
        [helper.make_node("Add", ["input", "weight"], ["output"])],
        "external",
        [source],
        [output],
        initializer=[weight],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    return model


def _manifest_metadata() -> dict:
    preprocess = {
        "size": 112,
        "color_order": "RGB",
        "mean": 127.5,
        "scale": 1.0 / 127.5,
        "layout": "NCHW",
    }
    entries = [
        {
            "path": "custom.onnx",
            "kind": "onnx",
            "bytes": 123,
            "sha256": "a" * 64,
        }
    ]
    embedding_dim = 512
    mean = [preprocess["mean"] / 255.0] * 3
    std = [1.0 / (preprocess["scale"] * 255.0)] * 3
    return {
        "schema_version": "1.0",
        "libreyolo_producer": "libreyolo",
        "artifact_format": "coreml",
        "coreml_io_schema_version": "2",
        "model_family": "facerec",
        "artifact_scope": FACEREC_COREML_ARTIFACT_SCOPE,
        "size": "custom",
        "model_size": "custom",
        "task": "embed",
        "supported_tasks": ["embed"],
        "default_task": "embed",
        "names": {"0": "face"},
        "nc": 1,
        "nb_classes": 1,
        "imgsz": 112,
        "imgsz_h": 112,
        "imgsz_w": 112,
        "precision": "fp32",
        "coreml_required_compute_units": (
            FACEREC_COREML_REQUIRED_COMPUTE_UNITS
        ),
        "dynamic": False,
        "facerec_contract": FACEREC_COREML_CONTRACT,
        FACEREC_COREML_PREPROCESS_KEY: json.dumps(
            preprocess, sort_keys=True, separators=(",", ":")
        ),
        FACEREC_COREML_PREPROCESS_HASH_KEY: facerec_coreml_preprocess_hash(
            preprocess
        ),
        FACEREC_COREML_SOURCE_HASH_KEY: _source_manifest_hash(entries),
        FACEREC_COREML_SOURCE_MANIFEST_KEY: json.dumps(
            entries, sort_keys=True, separators=(",", ":")
        ),
        "facerec_embedding_dim": embedding_dim,
        "facerec_onnx_to_torch_max_abs_error": 1e-6,
        "coreml_output_names": [FACEREC_COREML_OUTPUT_NAME],
        "coreml_io": {
            "input": {
                "name": FACEREC_COREML_INPUT_NAME,
                "kind": "tensor",
                "layout": "NCHW",
                "color": "rgb",
                "range": "standardized",
                "mean": mean,
                "std": std,
                "geometry": FACEREC_COREML_GEOMETRY,
                "interpolation": "bilinear",
                "resize_backend": "opencv",
                "pad_value": 0,
                "shape_mode": "fixed",
            },
            "validation": {
                "color": "rgb",
                "range": "standardized",
                "mean": mean,
                "std": std,
            },
            "outputs": [
                {
                    "name": FACEREC_COREML_OUTPUT_NAME,
                    "role": "embedding",
                    "encoding": "raw_identity_embedding",
                    "rank": 2,
                    "dtype": "float32",
                    "shape": [1, embedding_dim],
                }
            ],
        },
    }


def _official_manifest_metadata() -> dict:
    metadata = _manifest_metadata()
    spec = _official_embedder_spec()
    entries = [
        {
            "path": str(spec["filename"]),
            "kind": "onnx",
            "bytes": int(spec["size_bytes"]),
            "sha256": str(spec["sha256"]),
        }
    ]
    metadata.update(
        {
            "size": "l",
            "model_size": "l",
            FACEREC_COREML_SOURCE_HASH_KEY: _source_manifest_hash(entries),
            FACEREC_COREML_SOURCE_MANIFEST_KEY: json.dumps(
                entries,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    )
    metadata.update(_official_provenance_metadata())
    return metadata


def _promoted_official_metadata(monkeypatch):
    candidate = match_coreml_execution_profile(
        "facerec",
        "embed",
        "l",
        112,
        class_count=1,
        embedding_dim=512,
    )
    assert candidate is not None and not candidate.evidence_complete
    source = _official_manifest_metadata()
    profile = replace(
        candidate,
        source_kind="facerec-onnx-source-manifest-v1",
        source_sha256=source[FACEREC_COREML_SOURCE_HASH_KEY],
        deployment_abi_sha256="3" * 64,
        evidence_sha256="4" * 64,
    )
    key = next(
        key
        for key, value in COREML_EXECUTION_PROFILES.items()
        if value is candidate
    )
    monkeypatch.setitem(COREML_EXECUTION_PROFILES, key, profile)
    monkeypatch.setitem(
        COREML_EXECUTION_PROFILES_BY_ID,
        profile.profile_id,
        profile,
    )
    source.update(
        {
            "coreml_profile_source_kind": profile.source_kind,
            "coreml_profile_source_sha256": profile.source_sha256,
            "coreml_profile_abi_schema": COREML_DEPLOYMENT_ABI_SCHEMA,
            "coreml_profile_abi_sha256": profile.deployment_abi_sha256,
        }
    )
    metadata = merge_coreml_execution_profile_metadata(
        source,
        profile,
        conversion_compute_units="cpu_only",
    )
    return _stringify_metadata(metadata), profile


def test_converted_facerec_spec_binds_fixed_fp32_embedding_abi():
    parsed = validate_facerec_coreml_spec(
        _facerec_coreml_spec(),
        _stringify_metadata(_manifest_metadata()),
    )
    assert parsed["input"].shape == (1, 3, 112, 112)
    assert parsed["output"].shape == (1, 512)


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (
            _facerec_coreml_spec(output_shape=(1, 511)),
            "output disagrees",
        ),
        (
            _facerec_coreml_spec(output_dtype=65552),
            "output must expose FP32",
        ),
    ],
)
def test_converted_facerec_spec_rejects_name_only_abi_false_pass(
    spec,
    match,
):
    with pytest.raises(ValueError, match=match):
        validate_facerec_coreml_spec(
            spec,
            _stringify_metadata(_manifest_metadata()),
        )


def test_source_manifest_loads_without_external_data(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    path = tmp_path / "plain.onnx"
    _write_model(path, _plain_model(onnx))
    calls: list[bool] = []
    original = onnx.load

    def wrapped(*args, **kwargs):
        calls.append(kwargs.get("load_external_data"))
        return original(*args, **kwargs)

    monkeypatch.setattr(onnx, "load", wrapped)
    digest, entries = facerec_onnx_source_manifest(path)
    assert calls == [False]
    assert len(digest) == 64
    assert entries == [
        {
            "path": "plain.onnx",
            "kind": "onnx",
            "bytes": path.stat().st_size,
            "sha256": entries[0]["sha256"],
        }
    ]


def test_source_manifest_recurses_nested_graph_attributes_and_functions(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    for filename, value in (
        ("main.bin", 1.0),
        ("nested.bin", 2.0),
        ("attribute.bin", 3.0),
        ("function.bin", 4.0),
    ):
        (tmp_path / filename).write_bytes(
            __import__("struct").pack("<f", value)
        )

    main_weight = _external_tensor(onnx, "main_weight", "main.bin")
    nested_weight = _external_tensor(onnx, "nested_weight", "nested.bin")
    nested_output = helper.make_tensor_value_info(
        "nested_output", TensorProto.FLOAT, [1]
    )
    then_graph = helper.make_graph(
        [helper.make_node("Identity", ["nested_weight"], ["nested_output"])],
        "then",
        [],
        [nested_output],
        initializer=[nested_weight],
    )
    else_graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["nested_output"],
                value=_external_tensor(
                    onnx, "attribute_weight", "attribute.bin"
                ),
            )
        ],
        "else",
        [],
        [nested_output],
    )
    condition = helper.make_tensor_value_info(
        "condition", TensorProto.BOOL, []
    )
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [
            helper.make_node(
                "If",
                ["condition"],
                ["output"],
                then_branch=then_graph,
                else_branch=else_graph,
            )
        ],
        "root",
        [condition],
        [output],
        initializer=[main_weight],
    )
    function_node = helper.make_node(
        "Constant",
        [],
        ["function_output"],
        value=_external_tensor(onnx, "function_weight", "function.bin"),
    )
    function = helper.make_function(
        "libreyolo.test",
        "ExternalConstant",
        [],
        ["function_output"],
        [function_node],
        [helper.make_opsetid("", 13)],
    )
    model = helper.make_model(
        graph,
        functions=[function],
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    path = tmp_path / "nested.onnx"
    _write_model(path, model)

    _, entries = facerec_onnx_source_manifest(path)
    assert {entry["path"] for entry in entries} == {
        "attribute.bin",
        "function.bin",
        "main.bin",
        "nested.bin",
        "nested.onnx",
    }


@pytest.mark.parametrize(
    "location",
    [
        "../outside.bin",
        "/absolute.bin",
        r"C:\absolute.bin",
        "sub//weights.bin",
        "./weights.bin",
        ".",
    ],
)
def test_external_data_location_must_be_contained_and_canonical(
    tmp_path, location
):
    onnx = pytest.importorskip("onnx")
    path = tmp_path / "unsafe.onnx"
    _write_model(path, _model_with_external_initializer(onnx, location))
    with pytest.raises(ValueError, match="relative|canonical|portable"):
        facerec_onnx_source_manifest(path)


def test_external_data_symlink_is_rejected(tmp_path):
    onnx = pytest.importorskip("onnx")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"external")
    link = tmp_path / "linked.bin"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable on this platform.")
    path = tmp_path / "symlink.onnx"
    _write_model(
        path,
        _model_with_external_initializer(onnx, "linked.bin"),
    )
    with pytest.raises(ValueError, match="symlink"):
        facerec_onnx_source_manifest(path)


def test_external_data_mutation_changes_manifest(tmp_path):
    onnx = pytest.importorskip("onnx")
    external = tmp_path / "weights.bin"
    external.write_bytes(b"first")
    path = tmp_path / "external.onnx"
    _write_model(
        path,
        _model_with_external_initializer(onnx, "weights.bin"),
    )
    first_digest, first_entries = facerec_onnx_source_manifest(path)
    external.write_bytes(b"second")
    second_digest, second_entries = facerec_onnx_source_manifest(path)
    assert first_digest != second_digest
    assert first_entries != second_entries


def test_export_rejects_external_mutation_during_hydration(
    tmp_path, monkeypatch
):
    onnx = pytest.importorskip("onnx")
    external = tmp_path / "weights.bin"
    external.write_bytes(b"\x00\x00\x80?")
    path = tmp_path / "external.onnx"
    _write_model(
        path,
        _model_with_external_initializer(onnx, "weights.bin"),
    )
    original = onnx.load_external_data_for_model

    def mutate_after_load(*args, **kwargs):
        result = original(*args, **kwargs)
        external.write_bytes(b"\x00\x00\x00@")
        return result

    monkeypatch.setattr(
        onnx,
        "load_external_data_for_model",
        mutate_after_load,
    )
    model = SimpleNamespace(
        model_path=str(path),
        cfg=SimpleNamespace(size=112),
    )
    with pytest.raises(RuntimeError, match="changed between"):
        export_facerec_coreml(model, {})


def test_reserved_official_name_cannot_bypass_single_file_manifest(
    tmp_path, monkeypatch
):
    onnx = pytest.importorskip("onnx")
    from libreyolo.models.facerec.weights import FACEREC_OFFICIAL_EMBEDDER

    external = tmp_path / "weights.bin"
    external.write_bytes(b"external")
    filename = str(FACEREC_OFFICIAL_EMBEDDER["filename"])
    path = tmp_path / filename
    _write_model(
        path,
        _model_with_external_initializer(onnx, "weights.bin"),
    )
    _, _, entries, _ = _inspect_onnx_source(onnx, path)
    source_entry = next(
        entry for entry in entries if entry["kind"] == "onnx"
    )
    monkeypatch.setitem(
        FACEREC_OFFICIAL_EMBEDDER, "size_bytes", source_entry["bytes"]
    )
    monkeypatch.setitem(
        FACEREC_OFFICIAL_EMBEDDER, "sha256", source_entry["sha256"]
    )
    with pytest.raises(ValueError, match="single-file"):
        _validate_reserved_official_source(path, entries)


def test_reserved_official_name_requires_exact_case(tmp_path):
    onnx = pytest.importorskip("onnx")
    from libreyolo.models.facerec.weights import FACEREC_OFFICIAL_EMBEDDER

    path = tmp_path / str(FACEREC_OFFICIAL_EMBEDDER["filename"]).upper()
    _write_model(path, _plain_model(onnx))
    _, _, entries, _ = _inspect_onnx_source(onnx, path)
    with pytest.raises(ValueError, match="case-sensitive"):
        _validate_reserved_official_source(path, entries)


def test_official_size_metadata_pins_every_provenance_field(monkeypatch):
    from libreyolo.models.facerec.weights import FACEREC_OFFICIAL_EMBEDDER

    metadata = _manifest_metadata()
    monkeypatch.setitem(
        FACEREC_OFFICIAL_EMBEDDER, "filename", "custom.onnx"
    )
    monkeypatch.setitem(FACEREC_OFFICIAL_EMBEDDER, "size_bytes", 123)
    monkeypatch.setitem(FACEREC_OFFICIAL_EMBEDDER, "sha256", "a" * 64)
    metadata["size"] = "l"
    metadata["model_size"] = "l"
    metadata.update(_official_provenance_metadata())
    validate_facerec_coreml_metadata(metadata)
    metadata["facerec_source_upstream_revision"] = "0" * 40
    with pytest.raises(ValueError, match="provenance field"):
        validate_facerec_coreml_metadata(metadata)


@pytest.mark.parametrize(
    ("input_type", "output_type", "label"),
    [
        ("FLOAT16", "FLOAT", "input"),
        ("FLOAT", "DOUBLE", "output"),
    ],
)
def test_onnx_io_requires_float_element_types(
    input_type, output_type, label
):
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    source = helper.make_tensor_value_info(
        "aligned",
        getattr(TensorProto, input_type),
        [1, 3, 112, 112],
    )
    output = helper.make_tensor_value_info(
        "embedding",
        getattr(TensorProto, output_type),
        [1, 512],
    )
    model = helper.make_model(
        helper.make_graph([], "types", [source], [output])
    )
    with pytest.raises(ValueError, match=rf"{label} element type"):
        _validate_onnx_io(
            model,
            preprocess={
                "size": 112,
                "color_order": "RGB",
                "mean": 127.5,
                "scale": 1.0 / 127.5,
                "layout": "NCHW",
            },
        )


def test_onnx_io_binds_only_symbolic_batch_to_one():
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    source = helper.make_tensor_value_info(
        "aligned",
        TensorProto.FLOAT,
        ["batch", 3, 112, 112],
    )
    output = helper.make_tensor_value_info(
        "embedding",
        TensorProto.FLOAT,
        ["batch", 512],
    )
    model = helper.make_model(
        helper.make_graph([], "dynamic_batch", [source], [output])
    )

    input_shape, embedding_dim = _validate_onnx_io(
        model,
        preprocess={
            "size": 112,
            "color_order": "RGB",
            "mean": 127.5,
            "scale": 1.0 / 127.5,
            "layout": "NCHW",
        },
    )

    assert input_shape == (1, 3, 112, 112)
    assert embedding_dim == 512


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "label"),
    [
        (["batch", 3, "height", 112], ["batch", 512], "input axis 2"),
        (["batch", 3, 112, 112], ["batch", "embedding"], "output axis 1"),
    ],
)
def test_onnx_io_rejects_non_batch_dynamic_axes(
    input_shape, output_shape, label
):
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    source = helper.make_tensor_value_info(
        "aligned",
        TensorProto.FLOAT,
        input_shape,
    )
    output = helper.make_tensor_value_info(
        "embedding",
        TensorProto.FLOAT,
        output_shape,
    )
    model = helper.make_model(
        helper.make_graph([], "dynamic_non_batch", [source], [output])
    )

    with pytest.raises(NotImplementedError, match=label):
        _validate_onnx_io(
            model,
            preprocess={
                "size": 112,
                "color_order": "RGB",
                "mean": 127.5,
                "scale": 1.0 / 127.5,
                "layout": "NCHW",
            },
        )


def test_metadata_contract_round_trips_stringified_values():
    metadata = _manifest_metadata()
    parsed = validate_facerec_coreml_metadata(metadata)
    stringified = validate_facerec_coreml_metadata(
        _stringify_metadata(metadata)
    )
    assert parsed == stringified
    assert parsed["preprocess"]["size"] == 112
    assert parsed["embedding_dim"] == 512


def test_official_metadata_awaits_fresh_deployment_v2_evidence():
    with pytest.raises(NotImplementedError, match="not yet been promoted"):
        _apply_coreml_execution_profile(
            _official_manifest_metadata(),
            size="l",
            canvas=112,
            precision="fp32",
            compute_units="validated",
            embedding_dim=512,
        )

    with pytest.warns(RuntimeWarning, match="awaiting"):
        metadata, compute_units, profile = (
            _apply_coreml_execution_profile(
                _official_manifest_metadata(),
                size="l",
                canvas=112,
                precision="fp32",
                compute_units="cpu_only",
                embedding_dim=512,
            )
        )
    validate_facerec_coreml_metadata(metadata)
    assert compute_units == "cpu_only"
    assert profile is None
    assert metadata["coreml_profile_source_kind"] == (
        "facerec-onnx-source-manifest-v1"
    )


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("task", lambda value: value.__setitem__("task", "detect")),
        (
            "tasks",
            lambda value: value.__setitem__(
                "supported_tasks", ["embed", "detect"]
            ),
        ),
        ("imgsz", lambda value: value.__setitem__("imgsz_h", 224)),
        ("precision", lambda value: value.__setitem__("precision", "fp16")),
        (
            "compute_units",
            lambda value: value.__setitem__(
                "coreml_required_compute_units",
                "all",
            ),
        ),
        ("dynamic", lambda value: value.__setitem__("dynamic", True)),
        (
            "manifest",
            lambda value: value.__setitem__(
                FACEREC_COREML_SOURCE_HASH_KEY, "b" * 64
            ),
        ),
        (
            "output_names",
            lambda value: value.__setitem__("coreml_output_names", ["other"]),
        ),
        (
            "resize_backend",
            lambda value: value["coreml_io"]["input"].__setitem__(
                "resize_backend", "pillow"
            ),
        ),
        (
            "output_shape",
            lambda value: value["coreml_io"]["outputs"][0].__setitem__(
                "shape", [1, 256]
            ),
        ),
        (
            "artifact_scope",
            lambda value: value.__setitem__("artifact_scope", "full_pipeline"),
        ),
        ("official_claim", lambda value: value.__setitem__("size", "l")),
    ],
)
def test_metadata_rejects_one_field_tampering(name, mutate):
    metadata = copy.deepcopy(_manifest_metadata())
    mutate(metadata)
    with pytest.raises(ValueError, match="Face Core ML"):
        validate_facerec_coreml_metadata(metadata)


def test_metadata_rejects_manifest_structure_even_with_recomputed_hash():
    metadata = _manifest_metadata()
    entries = json.loads(metadata[FACEREC_COREML_SOURCE_MANIFEST_KEY])
    entries[0]["unexpected"] = True
    metadata[FACEREC_COREML_SOURCE_MANIFEST_KEY] = json.dumps(entries)
    with pytest.raises(ValueError, match="manifest fields"):
        validate_facerec_coreml_metadata(metadata)


def test_options_use_shared_path_normalization_and_reject_irrelevant_values():
    model = SimpleNamespace(
        model_path="weights/source.onnx",
        cfg=SimpleNamespace(size=112),
    )
    destination, precision, compute_units = _resolve_options(
        model,
        {"output_path": "artifact", "output": "artifact.mlpackage"},
    )
    assert destination == "artifact.mlpackage"
    assert precision == "fp32"
    assert compute_units == "cpu_only"
    with pytest.raises(TypeError, match="irrelevant"):
        _resolve_options(model, {"data": None})
    with pytest.raises(TypeError, match="boolean"):
        _resolve_options(model, {"half": "false"})
    with pytest.raises(NotImplementedError, match="FP32-only"):
        _resolve_options(model, {"half": True})
    with pytest.raises(NotImplementedError, match="cpu_only"):
        _resolve_options(model, {"compute_units": "all"})


def test_direct_export_rejects_fp16_before_source_or_dependencies():
    model = SimpleNamespace(
        model_path="missing.onnx",
        cfg=SimpleNamespace(size=112),
    )
    with pytest.raises(NotImplementedError, match="FP32-only"):
        export_facerec_coreml(
            model,
            {
                "half": True,
                "compute_units": "cpu_only",
            },
        )


def test_runtime_rejects_fp16_metadata_before_proxy(tmp_path, monkeypatch):
    from libreyolo.backends import coreml_facerec as backend

    package = tmp_path / "face.mlpackage"
    package.mkdir()
    metadata = _stringify_metadata(_manifest_metadata())
    metadata["precision"] = "fp16"
    spec = SimpleNamespace()
    proxy_calls = []
    fake_coremltools = SimpleNamespace(
        utils=SimpleNamespace(load_spec=lambda _path: spec),
        models=SimpleNamespace(
            MLModel=lambda *args, **kwargs: proxy_calls.append((args, kwargs))
        ),
    )
    monkeypatch.setattr(backend.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "coremltools", fake_coremltools)
    monkeypatch.setattr(
        backend,
        "_metadata_from_spec",
        lambda _spec: metadata,
    )
    monkeypatch.setattr(
        backend,
        "validate_facerec_coreml_spec",
        lambda _spec, values: validate_facerec_coreml_metadata(values),
    )

    with pytest.raises(ValueError, match="precision.*fp32"):
        backend.CoreMLFaceSession(
            str(package),
            compute_units="cpu_only",
        )
    assert proxy_calls == []


def test_runtime_rejects_unvalidated_compute_units_before_proxy(
    tmp_path,
    monkeypatch,
):
    from libreyolo.backends import coreml_facerec as backend

    package = tmp_path / "face.mlpackage"
    package.mkdir()
    spec = SimpleNamespace()
    proxy_calls = []
    fake_coremltools = SimpleNamespace(
        utils=SimpleNamespace(load_spec=lambda _path: spec),
        models=SimpleNamespace(
            MLModel=lambda *args, **kwargs: proxy_calls.append((args, kwargs))
        ),
    )
    monkeypatch.setattr(backend.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "coremltools", fake_coremltools)
    monkeypatch.setattr(backend, "_metadata_from_spec", lambda _spec: {})
    monkeypatch.setattr(
        backend,
        "validate_facerec_coreml_spec",
        lambda _spec, _metadata: {},
    )

    with pytest.raises(NotImplementedError, match="cpu_only"):
        backend.CoreMLFaceSession(str(package), compute_units="all")
    assert proxy_calls == []


def test_runtime_rejects_tampered_execution_profile_before_proxy(
    tmp_path,
    monkeypatch,
):
    from libreyolo.backends import coreml_facerec as backend

    package = tmp_path / "face.mlpackage"
    package.mkdir()
    metadata, _ = _promoted_official_metadata(monkeypatch)
    metadata["coreml_default_compute_units"] = "all"
    spec = SimpleNamespace()
    proxy_calls = []
    fake_coremltools = SimpleNamespace(
        utils=SimpleNamespace(load_spec=lambda _path: spec),
        models=SimpleNamespace(
            MLModel=lambda *args, **kwargs: proxy_calls.append((args, kwargs))
        ),
    )
    monkeypatch.setattr(backend.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "coremltools", fake_coremltools)
    monkeypatch.setattr(backend, "_metadata_from_spec", lambda _spec: metadata)
    monkeypatch.setattr(
        backend,
        "validate_facerec_coreml_spec",
        lambda _spec, values: validate_facerec_coreml_metadata(values),
    )
    from libreyolo.export import coreml_identity

    monkeypatch.setattr(
        coreml_identity,
        "validate_coreml_deployment_abi",
        lambda _spec, _metadata: "3" * 64,
    )

    with pytest.raises(ValueError, match="coreml_default_compute_units"):
        backend.CoreMLFaceSession(str(package))
    assert proxy_calls == []


def test_runtime_rejects_accelerator_for_exact_profile_before_proxy(
    tmp_path,
    monkeypatch,
):
    from libreyolo.backends import coreml_facerec as backend

    package = tmp_path / "face.mlpackage"
    package.mkdir()
    metadata, _ = _promoted_official_metadata(monkeypatch)
    spec = SimpleNamespace()
    proxy_calls = []
    fake_coremltools = SimpleNamespace(
        utils=SimpleNamespace(load_spec=lambda _path: spec),
        models=SimpleNamespace(
            MLModel=lambda *args, **kwargs: proxy_calls.append((args, kwargs))
        ),
    )
    monkeypatch.setattr(backend.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "coremltools", fake_coremltools)
    monkeypatch.setattr(backend, "_metadata_from_spec", lambda _spec: metadata)
    monkeypatch.setattr(
        backend,
        "validate_facerec_coreml_spec",
        lambda _spec, values: validate_facerec_coreml_metadata(values),
    )
    from libreyolo.export import coreml_identity

    monkeypatch.setattr(
        coreml_identity,
        "validate_coreml_deployment_abi",
        lambda _spec, _metadata: "3" * 64,
    )

    with pytest.raises(NotImplementedError, match="runtime is validated only"):
        backend.CoreMLFaceSession(str(package), compute_units="all")
    assert proxy_calls == []


def test_onnx2torch_version_gate_is_exact(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: "1.5.16",
    )
    with pytest.raises(RuntimeError, match="exactly onnx2torch==1.5.15"):
        _require_exact_onnx2torch()

    sentinel = object()
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: "1.5.15",
    )
    monkeypatch.setitem(
        sys.modules,
        "onnx2torch",
        SimpleNamespace(convert=sentinel),
    )
    assert _require_exact_onnx2torch() is sentinel


def test_optional_import_failure_is_actionable(
    tmp_path, monkeypatch
):
    onnx = pytest.importorskip("onnx")
    path = tmp_path / "plain.onnx"
    _write_model(path, _plain_model(onnx))
    model = SimpleNamespace(
        model_path=str(path),
        cfg=SimpleNamespace(size=112),
    )
    original_import = __import__

    def reject_onnx(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("deliberate")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_onnx)
    with pytest.raises(ImportError, match="requires ONNX"):
        export_facerec_coreml(model, {})
