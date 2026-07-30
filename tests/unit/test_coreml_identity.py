import json
import subprocess
import sys
import textwrap

import pytest
import torch

from libreyolo.export.coreml_identity import (
    COREML_CONTRACT_ABI_SCHEMA,
    COREML_DEPLOYMENT_ABI_SCHEMA,
    COREML_PROFILE_ABI_SHA256_KEY,
    COREML_PROFILE_SOURCE_BYTE_COUNT_KEY,
    COREML_PROFILE_SOURCE_GRAPH_SHA256_KEY,
    COREML_PROFILE_SOURCE_SHA256_KEY,
    bind_coreml_deployment_abi,
    canonical_json_sha256,
    coreml_contract_abi_manifest,
    coreml_contract_abi_sha256,
    coreml_deployment_abi_sha256,
    pytorch_captured_bundle_source_identity,
    pytorch_captured_graph_sha256,
    pytorch_module_source_identity,
    pytorch_traced_source_identity,
    validate_coreml_deployment_abi,
)

pytestmark = pytest.mark.unit


class _TinyState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
        self.register_buffer("offset", torch.tensor([0.25]))


class _TinyGraph(_TinyState):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, value):
        return value * self.scale + self.weight.sum() + self.offset


class _ScalarState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("counter", torch.tensor(7, dtype=torch.int64))


class _FakeMetadata:
    def __init__(self):
        self.userDefined = {}

    def copy_from(self, other):
        self.userDefined = dict(other.userDefined)

    def Clear(self):
        self.userDefined.clear()


class _FakeDescription:
    def __init__(self):
        self.metadata = _FakeMetadata()
        self.functions = []
        self.defaultFunctionName = ""

    def copy_from(self, other):
        self.metadata.copy_from(other.metadata)
        self.functions = list(other.functions)
        self.defaultFunctionName = other.defaultFunctionName

    def ClearField(self, name):
        if name != "metadata":
            raise ValueError(name)
        self.metadata.Clear()


class _FakeProgram:
    def __init__(self):
        self.functions = {"main": object()}
        self.defaultFunctionName = "main"
        self.graph = "add"

    def copy_from(self, other):
        self.functions = dict(other.functions)
        self.defaultFunctionName = other.defaultFunctionName
        self.graph = other.graph


class _FakeSpec:
    def __init__(self):
        self.specificationVersion = 9
        self.description = _FakeDescription()
        self.mlProgram = _FakeProgram()

    def CopyFrom(self, other):
        self.specificationVersion = other.specificationVersion
        self.description.copy_from(other.description)
        self.mlProgram.copy_from(other.mlProgram)

    def SerializeToString(self, deterministic=False):
        del deterministic
        return json.dumps(
            {
                "specification_version": self.specificationVersion,
                "metadata": self.description.metadata.userDefined,
                "functions": sorted(self.mlProgram.functions),
                "default": self.mlProgram.defaultFunctionName,
                "graph": self.mlProgram.graph,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def WhichOneof(self, name):
        assert name == "Type"
        return "mlProgram"


def _single_function_metadata():
    coreml_io = {
        "input": {
            "name": "image",
            "kind": "image",
            "layout": "nchw",
            "shape_mode": "fixed",
        },
        "validation": {"color": "rgb", "range": "0_255"},
        "outputs": [
            {
                "name": "prediction",
                "role": "prediction",
                "dtype": "float32",
                "rank": 3,
                "shape": [1, 4, 5],
            }
        ],
    }
    return {
        "coreml_io_schema_version": "2",
        "coreml_io": coreml_io,
        "coreml_output_names": ["prediction"],
    }


def test_pytorch_source_identity_is_repeatable_and_weight_sensitive():
    first = _TinyState()
    second = _TinyState()
    identity = pytorch_module_source_identity(first)
    assert identity == pytorch_module_source_identity(second)
    assert identity[COREML_PROFILE_SOURCE_BYTE_COUNT_KEY] == 28

    with torch.no_grad():
        second.weight[0, 0] += 1
    changed = pytorch_module_source_identity(second)
    assert (
        changed[COREML_PROFILE_SOURCE_SHA256_KEY]
        != identity[COREML_PROFILE_SOURCE_SHA256_KEY]
    )


def test_pytorch_source_identity_binds_buffer_and_topology():
    baseline = _TinyState()
    changed_buffer = _TinyState()
    changed_buffer.offset.add_(1)
    assert (
        pytorch_module_source_identity(changed_buffer)[
            COREML_PROFILE_SOURCE_SHA256_KEY
        ]
        != pytorch_module_source_identity(baseline)[
            COREML_PROFILE_SOURCE_SHA256_KEY
        ]
    )

    wrapped = torch.nn.Sequential(_TinyState())
    assert (
        pytorch_module_source_identity(wrapped)[
            COREML_PROFILE_SOURCE_SHA256_KEY
        ]
        != pytorch_module_source_identity(baseline)[
            COREML_PROFILE_SOURCE_SHA256_KEY
        ]
    )


def test_pytorch_source_identity_hashes_scalar_integer_buffers():
    baseline = _ScalarState()
    repeated = _ScalarState()
    identity = pytorch_module_source_identity(baseline)
    assert identity == pytorch_module_source_identity(repeated)
    assert identity[COREML_PROFILE_SOURCE_BYTE_COUNT_KEY] == 8

    repeated.counter.fill_(8)
    assert (
        pytorch_module_source_identity(repeated)[
            COREML_PROFILE_SOURCE_SHA256_KEY
        ]
        != identity[COREML_PROFILE_SOURCE_SHA256_KEY]
    )


def test_traced_source_identity_binds_graph_affecting_python_state():
    first = _TinyGraph(1.0).eval()
    second = _TinyGraph(2.0).eval()
    probe = torch.ones(1, 2, 3)
    first_trace = torch.jit.trace(first, probe)
    second_trace = torch.jit.trace(second, probe)
    assert (
        pytorch_module_source_identity(first)[
            COREML_PROFILE_SOURCE_SHA256_KEY
        ]
        == pytorch_module_source_identity(second)[
            COREML_PROFILE_SOURCE_SHA256_KEY
        ]
    )
    assert (
        pytorch_traced_source_identity(first, first_trace)[
            COREML_PROFILE_SOURCE_SHA256_KEY
        ]
        != pytorch_traced_source_identity(second, second_trace)[
            COREML_PROFILE_SOURCE_SHA256_KEY
        ]
    )


def test_traced_source_identity_is_stable_across_fresh_processes():
    script = textwrap.dedent(
        """
        import torch
        from libreyolo.export.coreml_identity import (
            COREML_PROFILE_SOURCE_SHA256_KEY,
            pytorch_traced_source_identity,
        )

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = 1.25
                self.weight = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))

            def forward(self, value):
                return value * self.scale + self.weight.sum()

        module = Tiny().eval()
        traced = torch.jit.trace(module, torch.ones(1, 2, 3))
        print(
            pytorch_traced_source_identity(module, traced)[
                COREML_PROFILE_SOURCE_SHA256_KEY
            ]
        )
        """
    )
    first = subprocess.check_output(
        [sys.executable, "-c", script],
        text=True,
    ).strip()
    second = subprocess.check_output(
        [sys.executable, "-c", script],
        text=True,
    ).strip()
    assert first == second


def test_captured_bundle_identity_binds_named_torchscript_and_export_graphs():
    module = _TinyGraph(1.25).eval()
    probe = torch.ones(1, 2, 3)
    traced = torch.jit.trace(module, probe)
    exported = torch.export.export(module, (probe,), strict=False)
    traced_sha256 = pytorch_captured_graph_sha256(traced)
    exported_sha256 = pytorch_captured_graph_sha256(exported)

    first = pytorch_captured_bundle_source_identity(
        module,
        {
            "trace": traced_sha256,
            "export": exported_sha256,
        },
    )
    reordered = pytorch_captured_bundle_source_identity(
        module,
        {
            "export": exported_sha256,
            "trace": traced_sha256,
        },
    )
    assert first == reordered
    assert len(first[COREML_PROFILE_SOURCE_GRAPH_SHA256_KEY]) == 64

    changed = pytorch_captured_bundle_source_identity(
        module,
        {
            "trace": pytorch_captured_graph_sha256(
                torch.jit.trace(_TinyGraph(2.0).eval(), probe)
            ),
            "export": exported_sha256,
        },
    )
    assert (
        changed[COREML_PROFILE_SOURCE_SHA256_KEY]
        != first[COREML_PROFILE_SOURCE_SHA256_KEY]
    )


def test_single_function_abi_hash_accepts_native_and_serialized_metadata():
    metadata = _single_function_metadata()
    native_hash = coreml_contract_abi_sha256(metadata)
    serialized = {
        key: json.dumps(value) if isinstance(value, (dict, list)) else value
        for key, value in metadata.items()
    }
    assert coreml_contract_abi_sha256(serialized) == native_hash
    manifest = coreml_contract_abi_manifest(metadata)
    assert manifest["schema"] == COREML_CONTRACT_ABI_SCHEMA
    assert manifest["output_names"] == ["prediction"]


def test_abi_hash_binds_output_order_shape_dtype_and_input_contract():
    baseline = _single_function_metadata()
    baseline_hash = coreml_contract_abi_sha256(baseline)
    mutations = []
    for key, value in (
        ("shape", [1, 5, 4]),
        ("dtype", "float16"),
        ("name", "other"),
    ):
        changed = json.loads(json.dumps(baseline))
        changed["coreml_io"]["outputs"][0][key] = value
        if key == "name":
            changed["coreml_output_names"] = ["other"]
        mutations.append(changed)
    changed_input = json.loads(json.dumps(baseline))
    changed_input["coreml_io"]["input"]["layout"] = "nhwc"
    mutations.append(changed_input)
    assert all(
        coreml_contract_abi_sha256(value) != baseline_hash
        for value in mutations
    )


def test_multifunction_abi_hash_binds_function_order_and_default():
    metadata = {
        "coreml_default_function": "encode",
        "coreml_function_names": ["encode", "decode_p1"],
        "sam_coreml_functions": {
            "encode": {"inputs": [{"name": "image"}], "outputs": [{"name": "e"}]},
            "decode_p1": {
                "inputs": [{"name": "e"}, {"name": "point"}],
                "outputs": [{"name": "mask"}],
            },
        },
    }
    baseline = coreml_contract_abi_sha256(metadata)
    changed = dict(metadata)
    changed["coreml_function_names"] = ["decode_p1", "encode"]
    assert coreml_contract_abi_sha256(changed) != baseline


def test_abi_contract_rejects_output_alias_disagreement():
    metadata = _single_function_metadata()
    metadata["coreml_output_names"] = ["other"]
    with pytest.raises(ValueError, match="output-name metadata disagrees"):
        coreml_contract_abi_sha256(metadata)


def test_canonical_json_hash_is_key_order_independent():
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": 2}
    )


def test_deployment_abi_binds_full_spec_and_ignores_user_metadata():
    metadata = _single_function_metadata()
    spec = _FakeSpec()
    baseline = coreml_deployment_abi_sha256(spec, metadata)
    spec.description.metadata.userDefined["volatile"] = "value"
    assert coreml_deployment_abi_sha256(spec, metadata) == baseline

    changed = _FakeSpec()
    changed.mlProgram.graph = "multiply"
    assert coreml_deployment_abi_sha256(changed, metadata) != baseline


def test_bind_and_validate_deployment_abi_reject_spec_tampering():
    metadata = _single_function_metadata()
    spec = _FakeSpec()
    bound = bind_coreml_deployment_abi(metadata, spec)
    assert bound["coreml_profile_abi_schema"] == (
        COREML_DEPLOYMENT_ABI_SCHEMA
    )
    assert len(bound[COREML_PROFILE_ABI_SHA256_KEY]) == 64
    assert validate_coreml_deployment_abi(spec, bound) == bound[
        COREML_PROFILE_ABI_SHA256_KEY
    ]

    spec.mlProgram.graph = "tampered"
    with pytest.raises(ValueError, match="protobuf ABI"):
        validate_coreml_deployment_abi(spec, bound)
