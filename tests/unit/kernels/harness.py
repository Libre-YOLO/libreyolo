"""Mechanical parity and benchmark gates for quantization kernels."""

from __future__ import annotations

import gc
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from libreyolo.quant import fake_quant, packing


SHAPES_PATH = Path(__file__).resolve().parents[3] / "tools" / "shapes.json"

_REFERENCE_NAMES = {
    "unpack_int_grouped": "unpack_int_grouped_weight",
    "unpack_nvfp4": "unpack_nvfp4_weight",
}

_SHAPE_GROUPS = {
    "fake_quant_nvfp4_weight": ("linear_weights",),
    "fake_quant_nvfp4_dynamic": ("linear_activations",),
    "fake_quant_int8_affine": ("linear_activations", "conv_activations"),
    "fake_quant_int8_per_channel": ("linear_weights", "conv_weights"),
    "fake_quant_int_grouped": ("linear_weights",),
    "fake_quant_fp8": (
        "linear_weights",
        "conv_weights",
        "linear_activations",
        "conv_activations",
    ),
    "fake_quant_mxfp4_weight": ("linear_weights",),
    "fake_quant_mxfp4_dynamic": ("linear_activations",),
    "unpack_int_grouped": ("linear_weights",),
    "unpack_nvfp4": ("linear_weights",),
}


@dataclass
class OpCase:
    name: str
    shape: tuple[int, ...]
    primary: torch.Tensor
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None
    check_gradient: bool = True

    def invoke(self, function: Callable, primary: torch.Tensor):
        return function(primary, *self.args, **(self.kwargs or {}))


def load_shapes(path: str | Path = SHAPES_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _reference(op: str) -> Callable:
    name = _REFERENCE_NAMES.get(op, op)
    owner = packing if op.startswith("unpack_") else fake_quant
    return getattr(owner, name)


def _entries(op: str, shapes: dict[str, Any]) -> Iterable[dict[str, Any]]:
    seen: set[tuple[int, ...]] = set()
    for group in _SHAPE_GROUPS[op]:
        for entry in shapes["canonical"][group]:
            shape = tuple(int(dim) for dim in entry["shape"])
            if shape in seen:
                continue
            seen.add(shape)
            yield {"group": group, "shape": shape}


def _generator(device: torch.device, label: str) -> torch.Generator:
    seed = zlib.crc32(label.encode("utf-8"))
    return torch.Generator(device=device).manual_seed(seed)


def _random_tensor(
    shape: tuple[int, ...], device: torch.device, label: str, scale: float = 1.0
) -> torch.Tensor:
    return torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
        generator=_generator(device, label),
    ).mul_(scale)


def _build_cases(
    op: str, shapes: dict[str, Any], device: torch.device
) -> Iterable[OpCase]:
    for index, entry in enumerate(_entries(op, shapes)):
        shape = entry["shape"]
        label = f"{op}:{entry['group']}:{shape}:{index}"
        tensor = _random_tensor(shape, device, label)

        if op == "fake_quant_nvfp4_weight":
            tensor.mul_(0.05)
            amax = tensor.abs().amax().reshape(1)
            yield OpCase(label, shape, tensor, (amax,))
        elif op == "fake_quant_nvfp4_dynamic":
            yield OpCase(label, shape, tensor)
        elif op == "fake_quant_int8_affine":
            flat = tensor.view(-1)
            flat[0] = -5.0
            if flat.numel() > 1:
                flat[1] = 6.0
            lo = torch.tensor([-2.5], device=device)
            hi = torch.tensor([3.25], device=device)
            yield OpCase(label, shape, tensor, (lo, hi))
        elif op == "fake_quant_int8_per_channel":
            yield OpCase(f"{label}:dynamic-scale", shape, tensor)
            scale = fake_quant.int8_weight_qparams(tensor, 0).mul(0.75)
            yield OpCase(
                f"{label}:frozen-scale",
                shape,
                tensor,
                (),
                {"scale": scale},
            )
        elif op == "fake_quant_int_grouped":
            for bits, group_size in ((4, 128), (2, 64)):
                suffix = f"bits{bits}-group{group_size}"
                yield OpCase(
                    f"{label}:{suffix}:dynamic-scale",
                    shape,
                    tensor,
                    (bits, group_size),
                )
                scale = fake_quant.int_group_qparams(tensor, bits, group_size).mul(0.75)
                yield OpCase(
                    f"{label}:{suffix}:frozen-scale",
                    shape,
                    tensor,
                    (bits, group_size),
                    {"scale": scale},
                )
        elif op == "fake_quant_fp8":
            if entry["group"].endswith("weights"):
                dims = tuple(range(1, tensor.dim()))
                amax = tensor.abs()
                if dims:
                    amax = amax.amax(dim=dims)
                scale = (amax / fake_quant.E4M3_MAX).clamp(min=1e-12)
                scale = scale.reshape(-1, *([1] * (tensor.dim() - 1)))
                variant = "per-channel"
            else:
                scale = tensor.abs().amax().reshape(1) / fake_quant.E4M3_MAX
                variant = "per-tensor"
            yield OpCase(f"{label}:{variant}", shape, tensor, (scale,))
        elif op in ("fake_quant_mxfp4_weight", "fake_quant_mxfp4_dynamic"):
            yield OpCase(label, shape, tensor)
        elif op == "unpack_int_grouped":
            for bits, group_size in ((4, 128), (2, 64)):
                payload, scale = packing.pack_int_grouped_weight(
                    tensor, bits, group_size
                )
                yield OpCase(
                    f"{label}:bits{bits}-group{group_size}",
                    tuple(payload.shape),
                    payload,
                    (scale, bits, group_size, shape[1]),
                    check_gradient=False,
                )
        elif op == "unpack_nvfp4":
            amax = tensor.abs().amax().reshape(1)
            payload, block_scale = packing.pack_nvfp4_weight(tensor, amax)
            yield OpCase(
                label,
                tuple(payload.shape),
                payload,
                (block_scale, amax, shape[1]),
                check_gradient=False,
            )
        else:
            raise KeyError(f"Unsupported harness operation: {op}")


def _clone_primary(primary: torch.Tensor, gradient: bool) -> torch.Tensor:
    clone = primary.detach().clone()
    if gradient:
        clone.requires_grad_(True)
    return clone


def check_parity(
    op: str,
    impl: Callable,
    shapes: dict[str, Any] | str | Path,
    *,
    device: str | torch.device = "cuda",
) -> list[dict[str, Any]]:
    """Check exact forward and reference-gradient parity on every case."""
    if not isinstance(shapes, dict):
        shapes = load_shapes(shapes)
    target = torch.device(device)
    reference = _reference(op)
    results = []

    for case in _build_cases(op, shapes, target):
        ref_input = _clone_primary(case.primary, case.check_gradient)
        impl_input = _clone_primary(case.primary, case.check_gradient)
        ref_output = case.invoke(reference, ref_input)
        impl_output = case.invoke(impl, impl_input)

        if ref_output.dtype != impl_output.dtype:
            raise AssertionError(
                f"{case.name}: dtype mismatch {ref_output.dtype} != {impl_output.dtype}"
            )
        if not torch.equal(ref_output, impl_output):
            mismatch = torch.count_nonzero(ref_output != impl_output).item()
            max_abs = (ref_output.float() - impl_output.float()).abs().max().item()
            raise AssertionError(
                f"{case.name}: forward mismatch in {mismatch}/"
                f"{ref_output.numel()} values (max_abs={max_abs})"
            )

        grad_max_abs = None
        if case.check_gradient:
            ref_output.sum().backward()
            impl_output.sum().backward()
            torch.testing.assert_close(
                impl_input.grad,
                ref_input.grad,
                rtol=0.0,
                atol=1e-6,
                msg=lambda message: f"{case.name}: gradient mismatch: {message}",
            )
            grad_max_abs = (impl_input.grad - ref_input.grad).abs().max().item()

        results.append(
            {
                "case": case.name,
                "shape": list(case.shape),
                "forward_exact": True,
                "gradient": "pass" if case.check_gradient else "not-applicable",
                "gradient_max_abs": grad_max_abs,
            }
        )
        del ref_input, impl_input, ref_output, impl_output
        gc.collect()

    return results


def bench(
    op: str,
    impl: Callable,
    shapes: dict[str, Any] | str | Path,
    *,
    device: str | torch.device = "cuda",
    warmup: int = 25,
    rep: int = 100,
    gate: float | None = None,
) -> list[dict[str, Any]]:
    """Benchmark implementation and eager oracle with Triton's ``do_bench``."""
    import triton

    if not isinstance(shapes, dict):
        shapes = load_shapes(shapes)
    target = torch.device(device)
    reference = _reference(op)
    results = []

    for case in _build_cases(op, shapes, target):
        primary = case.primary.detach()
        with torch.no_grad():
            case.invoke(impl, primary)
            case.invoke(reference, primary)
            torch.cuda.synchronize(target)
            impl_ms = triton.testing.do_bench(
                lambda: case.invoke(impl, primary), warmup=warmup, rep=rep
            )
            reference_ms = triton.testing.do_bench(
                lambda: case.invoke(reference, primary), warmup=warmup, rep=rep
            )
        speedup = reference_ms / impl_ms
        row = {
            "case": case.name,
            "shape": list(case.shape),
            "reference_ms": reference_ms,
            "implementation_ms": impl_ms,
            "speedup": speedup,
        }
        results.append(row)
        print(
            f"{case.name:72} ref={reference_ms:9.5f} ms "
            f"impl={impl_ms:9.5f} ms speedup={speedup:7.3f}x"
        )
        gc.collect()

    if not results:
        raise AssertionError(f"No benchmark cases found for {op}")
    mean = sum(row["speedup"] for row in results) / len(results)
    minimum = min(row["speedup"] for row in results)
    print(f"{op}: mean={mean:.3f}x min={minimum:.3f}x cases={len(results)}")
    if gate is not None and minimum < gate:
        raise AssertionError(
            f"{op}: minimum speedup {minimum:.3f}x does not meet {gate:.3f}x"
        )
    if not math.isfinite(mean):
        raise AssertionError(f"{op}: non-finite benchmark result")
    return results
