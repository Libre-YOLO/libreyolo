"""Deterministic source and ABI identities for Core ML execution evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

COREML_PROFILE_SOURCE_KIND_KEY = "coreml_profile_source_kind"
COREML_PROFILE_SOURCE_SHA256_KEY = "coreml_profile_source_sha256"
COREML_PROFILE_SOURCE_TENSOR_COUNT_KEY = "coreml_profile_source_tensor_count"
COREML_PROFILE_SOURCE_BYTE_COUNT_KEY = "coreml_profile_source_bytes"
COREML_PROFILE_SOURCE_GRAPH_SHA256_KEY = (
    "coreml_profile_source_graph_sha256"
)
COREML_PROFILE_ABI_SCHEMA_KEY = "coreml_profile_abi_schema"
COREML_PROFILE_ABI_SHA256_KEY = "coreml_profile_abi_sha256"

COREML_PYTORCH_SOURCE_KIND = "pytorch-module-state-v1"
COREML_PYTORCH_TRACED_SOURCE_KIND = "pytorch-traced-graph-state-v2"
COREML_PYTORCH_CAPTURED_BUNDLE_SOURCE_KIND = (
    "pytorch-captured-bundle-state-v1"
)
COREML_CONTRACT_ABI_SCHEMA = "coreml-contract-abi-v1"
COREML_DEPLOYMENT_ABI_SCHEMA = "coreml-deployment-abi-v2"

_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one bounded contract without whitespace or key-order drift."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 of a canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_lower_sha256(value: Any, *, key: str) -> str:
    """Return one normalized digest or reject a non-canonical value."""
    digest = str(value).strip()
    if _LOWER_SHA256.fullmatch(digest) is None:
        raise ValueError(
            f"Core ML metadata {key!r} must be 64 lowercase hexadecimal "
            "characters."
        )
    return digest


def _metadata_json(metadata: Mapping[str, Any], key: str) -> Any:
    value = metadata.get(key)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Core ML metadata {key!r} must be valid JSON."
            ) from exc
    return value


def coreml_contract_abi_manifest(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical host-visible ABI manifest for one artifact."""
    function_contract_key = (
        "sam_coreml_functions"
        if metadata.get("sam_coreml_functions") not in (None, "")
        else "coreml_functions"
        if metadata.get("coreml_functions") not in (None, "")
        else None
    )
    if function_contract_key is not None:
        functions = _metadata_json(metadata, function_contract_key)
        function_names = _metadata_json(metadata, "coreml_function_names")
        if not isinstance(functions, dict) or not functions:
            raise ValueError(
                "Core ML multifunction ABI metadata must contain a non-empty "
                "function "
                "contract object."
            )
        if not isinstance(function_names, list) or not function_names:
            raise ValueError(
                "Core ML multifunction ABI metadata must contain an ordered function "
                "name list."
            )
        return {
            "schema": COREML_CONTRACT_ABI_SCHEMA,
            "kind": "multifunction",
            "default_function": str(
                metadata.get("coreml_default_function", "")
            ),
            "function_names": function_names,
            "functions": functions,
        }

    io_contract = _metadata_json(metadata, "coreml_io")
    if not isinstance(io_contract, dict) or not io_contract:
        raise ValueError(
            "Core ML profile ABI identity requires a non-empty coreml_io "
            "contract."
        )
    output_names = _metadata_json(metadata, "coreml_output_names")
    if not isinstance(output_names, list) or not output_names:
        raise ValueError(
            "Core ML profile ABI identity requires an ordered "
            "coreml_output_names list."
        )
    declared_outputs = io_contract.get("outputs")
    if not isinstance(declared_outputs, list) or [
        item.get("name") if isinstance(item, dict) else None
        for item in declared_outputs
    ] != output_names:
        raise ValueError(
            "Core ML output-name metadata disagrees with the ordered "
            "coreml_io ABI."
        )
    return {
        "schema": COREML_CONTRACT_ABI_SCHEMA,
        "kind": "single_function",
        "default_function": str(
            metadata.get("coreml_default_function", "main")
        ),
        "io_schema_version": str(
            metadata.get("coreml_io_schema_version", "")
        ),
        "io": io_contract,
        "output_names": output_names,
    }


def coreml_contract_abi_sha256(metadata: Mapping[str, Any]) -> str:
    """Hash the exact ordered IO/function contract embedded in an artifact."""
    return canonical_json_sha256(coreml_contract_abi_manifest(metadata))


def _coreml_spec_sha256_without_metadata(spec: Any) -> str:
    """Hash the full converted protobuf after clearing circular metadata."""
    clone = type(spec)()
    copy_from = getattr(clone, "CopyFrom", None)
    serialize = getattr(clone, "SerializeToString", None)
    if not callable(copy_from) or not callable(serialize):
        raise TypeError(
            "Core ML deployment ABI hashing requires a protobuf Model spec."
        )
    copy_from(spec)
    description = getattr(clone, "description", None)
    clear_field = getattr(description, "ClearField", None)
    if callable(clear_field):
        try:
            clear_field("metadata")
        except (ValueError, TypeError):
            metadata = getattr(description, "metadata", None)
            clear = getattr(metadata, "Clear", None)
            if callable(clear):
                clear()
    try:
        payload = serialize(deterministic=True)
    except TypeError:  # pragma: no cover - old protobuf compatibility
        payload = serialize()
    return hashlib.sha256(payload).hexdigest()


def _coreml_model_kind(spec: Any) -> str:
    which_oneof = getattr(spec, "WhichOneof", None)
    if not callable(which_oneof):
        raise TypeError(
            "Core ML deployment ABI hashing requires a protobuf Model spec."
        )
    kind = which_oneof("Type")
    if not isinstance(kind, str) or not kind:
        raise ValueError("Core ML spec does not declare a model type.")
    return kind


def _function_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    for key in ("name", "functionName"):
        name = getattr(value, key, None)
        if isinstance(name, str) and name:
            return name
    raise ValueError("Core ML multifunction description has an unnamed function.")


def coreml_deployment_abi_manifest(
    spec: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the final converter-produced deployment boundary.

    The protobuf description is hashed after clearing its metadata field so
    the digest can itself be stored in user-defined metadata without becoming
    recursive. The declared LibreYOLO contract is included separately, making
    the identity sensitive to both converter-visible features and host-side
    semantic roles.
    """
    description = getattr(spec, "description", None)
    if description is None:
        raise ValueError("Core ML spec is missing its model description.")
    specification_version = getattr(spec, "specificationVersion", None)
    if (
        isinstance(specification_version, bool)
        or not isinstance(specification_version, int)
        or specification_version <= 0
    ):
        raise ValueError(
            "Core ML spec must declare a positive integer specificationVersion."
        )

    described_functions = [
        _function_name(value)
        for value in list(getattr(description, "functions", ()) or ())
    ]
    program = getattr(spec, "mlProgram", None)
    program_functions = getattr(program, "functions", None)
    program_function_names = (
        sorted(str(key) for key in program_functions.keys())
        if hasattr(program_functions, "keys")
        else []
    )
    default_function = str(
        getattr(description, "defaultFunctionName", "")
        or getattr(program, "defaultFunctionName", "")
        or metadata.get("coreml_default_function", "")
    )
    return {
        "schema": COREML_DEPLOYMENT_ABI_SCHEMA,
        "specification_version": specification_version,
        "model_kind": _coreml_model_kind(spec),
        "spec_sha256": _coreml_spec_sha256_without_metadata(spec),
        "described_functions": described_functions,
        "program_functions": program_function_names,
        "default_function": default_function,
        "declared_contract": coreml_contract_abi_manifest(metadata),
    }


def coreml_deployment_abi_sha256(
    spec: Any,
    metadata: Mapping[str, Any],
) -> str:
    """Hash the final Core ML protobuf boundary plus its semantic contract."""
    return canonical_json_sha256(
        coreml_deployment_abi_manifest(spec, metadata)
    )


def bind_coreml_deployment_abi(
    metadata: Mapping[str, Any],
    spec: Any,
) -> dict[str, Any]:
    """Bind final converted-spec identity without accepting caller overrides."""
    bound = dict(metadata)
    actual = coreml_deployment_abi_sha256(spec, bound)
    expected_values = {
        COREML_PROFILE_ABI_SCHEMA_KEY: COREML_DEPLOYMENT_ABI_SCHEMA,
        COREML_PROFILE_ABI_SHA256_KEY: actual,
    }
    for key, expected in expected_values.items():
        current = bound.get(key)
        if current not in (None, "") and str(current).strip() != expected:
            raise ValueError(
                f"Core ML metadata {key!r} conflicts with the final converted "
                "deployment ABI."
            )
        bound[key] = expected
    return bound


def validate_coreml_deployment_abi(
    spec: Any,
    metadata: Mapping[str, Any],
) -> str:
    """Verify final package-spec identity before any native model compilation."""
    schema = str(metadata.get(COREML_PROFILE_ABI_SCHEMA_KEY, "")).strip()
    if schema != COREML_DEPLOYMENT_ABI_SCHEMA:
        raise ValueError(
            "Core ML metadata must declare deployment ABI schema "
            f"{COREML_DEPLOYMENT_ABI_SCHEMA!r}."
        )
    declared = require_lower_sha256(
        metadata.get(COREML_PROFILE_ABI_SHA256_KEY),
        key=COREML_PROFILE_ABI_SHA256_KEY,
    )
    actual = coreml_deployment_abi_sha256(spec, metadata)
    if declared != actual:
        raise ValueError(
            "Core ML package protobuf ABI does not match its declared "
            "deployment identity."
        )
    return actual


def pytorch_module_source_identity(module: Any) -> dict[str, Any]:
    """Hash current parameters, buffers, and module topology deterministically.

    This deliberately hashes the live module after checkpoint loading and any
    in-scope export preparation. It therefore distinguishes custom/fine-tuned
    weights even when family, size, and tensor ABI are otherwise identical.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Core ML export requires torch
        raise ImportError(
            "PyTorch is required to fingerprint a Core ML export source."
        ) from exc
    if not isinstance(module, torch.nn.Module):
        raise TypeError(
            "Core ML source fingerprinting requires a torch.nn.Module, got "
            f"{type(module).__name__}."
        )

    digest = hashlib.sha256()
    digest.update(b"libreyolo-coreml-pytorch-module-state-v1\0")
    modules = sorted(
        (
            str(name),
            f"{type(child).__module__}.{type(child).__qualname__}",
        )
        for name, child in module.named_modules()
    )
    topology = canonical_json_bytes(modules)
    digest.update(len(topology).to_bytes(8, "big"))
    digest.update(topology)

    tensors: list[tuple[str, str, Any]] = [
        ("parameter", str(name), tensor)
        for name, tensor in module.named_parameters()
    ]
    tensors.extend(
        ("buffer", str(name), tensor)
        for name, tensor in module.named_buffers()
    )
    tensors.sort(key=lambda item: (item[0], item[1]))

    byte_count = 0
    for kind, name, tensor in tensors:
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"Core ML source state {kind} {name!r} is not a tensor."
            )
        if tensor.layout != torch.strided:
            raise ValueError(
                f"Core ML source state {kind} {name!r} uses unsupported "
                f"layout {tensor.layout}."
            )
        value = tensor.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        header = canonical_json_bytes(
            {
                "kind": kind,
                "name": name,
                "dtype": str(value.dtype),
                "shape": [int(axis) for axis in value.shape],
            }
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        byte_count += len(raw)

    return {
        COREML_PROFILE_SOURCE_KIND_KEY: COREML_PYTORCH_SOURCE_KIND,
        COREML_PROFILE_SOURCE_SHA256_KEY: digest.hexdigest(),
        COREML_PROFILE_SOURCE_TENSOR_COUNT_KEY: len(tensors),
        COREML_PROFILE_SOURCE_BYTE_COUNT_KEY: byte_count,
        "coreml_profile_source_module_count": len(modules),
    }


def _normalized_torchscript_graph(traced: Any) -> bytes:
    graph = getattr(traced, "inlined_graph", None)
    if graph is None:
        raise TypeError(
            "Core ML traced-source identity requires a TorchScript module "
            "with an inlined_graph."
        )
    graph = graph.copy()
    try:
        import torch

        graph = torch._C._jit_pass_canonicalize(graph)
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "This PyTorch build cannot canonicalize the traced Core ML graph."
        ) from exc
    text = str(graph)
    text = re.sub(r"\.___torch_mangle_\d+", "", text)
    text = re.sub(r"[ \t]+# [^\r\n]*", "", text)
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not text.startswith("graph(") or "return (" not in text:
        raise RuntimeError(
            "TorchScript canonicalization returned an invalid graph."
        )
    return text.encode("utf-8")


def pytorch_traced_source_identity(
    module: Any,
    traced: Any,
) -> dict[str, Any]:
    """Bind live tensor state and the exact prepared TorchScript graph."""
    state = pytorch_module_source_identity(module)
    graph = _normalized_torchscript_graph(traced)
    graph_sha256 = hashlib.sha256(graph).hexdigest()
    digest = hashlib.sha256()
    digest.update(b"libreyolo-coreml-pytorch-traced-graph-state-v2\0")
    digest.update(
        canonical_json_bytes(
            {
                "state_kind": state[COREML_PROFILE_SOURCE_KIND_KEY],
                "state_sha256": state[COREML_PROFILE_SOURCE_SHA256_KEY],
                "tensor_count": state[
                    COREML_PROFILE_SOURCE_TENSOR_COUNT_KEY
                ],
                "tensor_bytes": state[COREML_PROFILE_SOURCE_BYTE_COUNT_KEY],
                "module_count": state[
                    "coreml_profile_source_module_count"
                ],
                "graph_sha256": graph_sha256,
            }
        )
    )
    state.update(
        {
            COREML_PROFILE_SOURCE_KIND_KEY: (
                COREML_PYTORCH_TRACED_SOURCE_KIND
            ),
            COREML_PROFILE_SOURCE_SHA256_KEY: digest.hexdigest(),
            COREML_PROFILE_SOURCE_GRAPH_SHA256_KEY: graph_sha256,
        }
    )
    return state


def pytorch_captured_graph_sha256(captured: Any) -> str:
    """Hash one TorchScript or ``torch.export`` graph deterministically."""
    if getattr(captured, "inlined_graph", None) is not None:
        payload = _normalized_torchscript_graph(captured)
    else:
        graph_module = getattr(captured, "graph_module", None)
        graph = getattr(graph_module, "graph", None)
        if graph_module is None or graph is None:
            raise TypeError(
                "Core ML captured-graph identity requires a TorchScript "
                "module or torch.export.ExportedProgram."
            )
        graph_text = "\n".join(
            line.rstrip() for line in str(graph).splitlines()
        ).strip()
        graph_code = "\n".join(
            line.rstrip() for line in str(graph_module.code).splitlines()
        ).strip()
        if not graph_text.startswith("graph(") or "return" not in graph_text:
            raise RuntimeError(
                "torch.export returned an invalid graph for Core ML source "
                "identity."
            )
        range_constraints = getattr(captured, "range_constraints", {})
        payload = canonical_json_bytes(
            {
                "kind": "torch-exported-program-v1",
                "graph": graph_text,
                "code": graph_code,
                "graph_signature": str(
                    getattr(captured, "graph_signature", "")
                ),
                "range_constraints": sorted(
                    (str(key), str(value))
                    for key, value in range_constraints.items()
                ),
            }
        )
    return hashlib.sha256(payload).hexdigest()


def pytorch_captured_bundle_source_identity(
    module: Any,
    graph_sha256_by_name: Mapping[str, str],
) -> dict[str, Any]:
    """Bind live tensor state to every named graph in a component bundle."""
    if not isinstance(graph_sha256_by_name, Mapping) or not (
        graph_sha256_by_name
    ):
        raise ValueError(
            "Core ML component-bundle identity requires at least one graph."
        )
    graphs = []
    for raw_name, raw_sha256 in graph_sha256_by_name.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(
                "Core ML component-bundle graph names must be non-empty."
            )
        graphs.append(
            (
                name,
                require_lower_sha256(
                    raw_sha256,
                    key=f"graph_sha256_by_name[{name!r}]",
                ),
            )
        )
    graphs.sort()
    if len({name for name, _ in graphs}) != len(graphs):
        raise ValueError(
            "Core ML component-bundle graph names must be unique."
        )

    state = pytorch_module_source_identity(module)
    graph_manifest_sha256 = canonical_json_sha256(graphs)
    digest = hashlib.sha256()
    digest.update(b"libreyolo-coreml-pytorch-captured-bundle-state-v1\0")
    digest.update(
        canonical_json_bytes(
            {
                "state_kind": state[COREML_PROFILE_SOURCE_KIND_KEY],
                "state_sha256": state[COREML_PROFILE_SOURCE_SHA256_KEY],
                "tensor_count": state[
                    COREML_PROFILE_SOURCE_TENSOR_COUNT_KEY
                ],
                "tensor_bytes": state[COREML_PROFILE_SOURCE_BYTE_COUNT_KEY],
                "module_count": state[
                    "coreml_profile_source_module_count"
                ],
                "graph_manifest_sha256": graph_manifest_sha256,
                "graphs": graphs,
            }
        )
    )
    state.update(
        {
            COREML_PROFILE_SOURCE_KIND_KEY: (
                COREML_PYTORCH_CAPTURED_BUNDLE_SOURCE_KIND
            ),
            COREML_PROFILE_SOURCE_SHA256_KEY: digest.hexdigest(),
            COREML_PROFILE_SOURCE_GRAPH_SHA256_KEY: (
                graph_manifest_sha256
            ),
        }
    )
    return state


__all__ = [
    "COREML_CONTRACT_ABI_SCHEMA",
    "COREML_DEPLOYMENT_ABI_SCHEMA",
    "COREML_PROFILE_ABI_SCHEMA_KEY",
    "COREML_PROFILE_ABI_SHA256_KEY",
    "COREML_PROFILE_SOURCE_BYTE_COUNT_KEY",
    "COREML_PROFILE_SOURCE_GRAPH_SHA256_KEY",
    "COREML_PROFILE_SOURCE_KIND_KEY",
    "COREML_PROFILE_SOURCE_SHA256_KEY",
    "COREML_PROFILE_SOURCE_TENSOR_COUNT_KEY",
    "COREML_PYTORCH_CAPTURED_BUNDLE_SOURCE_KIND",
    "COREML_PYTORCH_SOURCE_KIND",
    "COREML_PYTORCH_TRACED_SOURCE_KIND",
    "bind_coreml_deployment_abi",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "coreml_contract_abi_manifest",
    "coreml_contract_abi_sha256",
    "coreml_deployment_abi_manifest",
    "coreml_deployment_abi_sha256",
    "pytorch_captured_bundle_source_identity",
    "pytorch_captured_graph_sha256",
    "pytorch_module_source_identity",
    "pytorch_traced_source_identity",
    "require_lower_sha256",
    "validate_coreml_deployment_abi",
]
