# Kernel registry

`libreyolo/kernels/` hosts the library-wide registry of pluggable accelerated
implementations. Every op has a portable default; accelerated variants
register on top and are selected per-op by predicate. A missing optional
dependency is never an error, only a fallback.

## Layout (purpose first, backend second)

- `kernels/quant/simulate/`: fake-quantization Triton kernels. Numerics-true
  simulation with STE backward, any device. They serve QAT/QAD **and**
  simulated PTQ/`val()` inference; the enforced boundary is
  `is_finalized`, not train-vs-deploy.
- `kernels/quant/execute/`: finalized-only real-precision paths. No backward,
  real hardware: the `fp8_gemm` tensor-core GEMM (`torch._scaled_mm`), its
  fused Triton prologue/epilogue, and the packed-weight unpack kernels.
- `kernels/attention/`: attention ops shared across model families. Currently
  the `ms_deform_attn` slot (multi-scale deformable attention) consumed by
  the Deformable-DETR-lineage families.
- The reference implementations stay in `libreyolo/quant/fake_quant.py` and
  `libreyolo/quant/packing.py`: `quant/` defines what the numbers mean,
  `kernels/` makes them fast. `packing.py` never has variants because it is
  the checkpoint contract.

## Selection

Implementations are tried newest-first; the first one whose predicate passes
wins, falling back to the reference. `libreyolo.kernels.active()` reports the
current selection.

- `LIBREYOLO_KERNELS=off|reference` forces the reference implementations;
  any other value selects only implementations registered under that name.
  `LIBREYOLO_QUANT_KERNELS` is honored as a legacy alias.
- GEMM and attention slots (`fp8_gemm`, `ms_deform_attn`, `nvfp4_gemm`, ...)
  have no reference implementation. Callers must check `resolve()` returns
  non-None and keep their portable path as the fallback; exported graphs
  (ONNX, TensorRT, torch.export) always use the portable path.

## Hub kernels

Compiled kernels published on the Hugging Face Hub load at runtime through
the optional `kernels` package. Installing the extra is the opt-in:
`pip install libreyolo[hub-kernels]` enables them, and without the package
nothing changes (no network access, portable paths everywhere). Set
`LIBREYOLO_HUB_KERNELS=0` to disable them without uninstalling. Nothing is
vendored; artifacts are fetched and cached by the `kernels` package, and a
kernel that fails to load or run disables itself for the process and falls
back to the portable path with one warning. Every hub kernel is pinned to an
audited commit revision in its provider module — a moved branch on the Hub
can never change the binary that runs in-process. Bumping a pin requires a
GPU parity run of the provider's `*_matches_portable_on_cuda` test.

Current hub-backed slot:

- `ms_deform_attn` <- [`kernels-community/deformable-detr`](https://huggingface.co/kernels-community/deformable-detr)
  (Apache-2.0): the compiled CUDA multi-scale deformable attention
  forward/backward from Deformable DETR. Wired into the RF-DETR,
  LibreDeformableDETR, and LibreDINO-DETR attention cores; eligible inputs
  are CUDA fp32 in eager mode. Training is accelerated too (the compiled
  backward registers through an autograd bridge). Active whenever the
  `kernels` package is installed, unless `LIBREYOLO_HUB_KERNELS=0`.

Out-of-tree compiled kernels can also ship as a `libreyolo_kernels` package,
which self-registers on import (e.g. a future CUTLASS NVFP4 GEMM for the
documented `nvfp4_gemm` slot).

## Hardware behavior

Accelerated implementations engage only where their predicate passes; every
op always has a portable path, so no platform needs configuration:

- CPU-only and Apple Silicon (MPS): all CUDA/Triton predicates fail;
  reference implementations run as plain torch ops.
- NVIDIA CUDA: Triton kernels and eligible hub/GEMM kernels engage.
- AMD ROCm: torch reports CUDA and ROCm wheels ship Triton's AMD backend,
  so Triton kernels can engage; parity is currently only exercised on
  NVIDIA in CI. Hub kernels resolve per-variant; a missing variant simply
  falls back.

## Adding an implementation

Register a callable with the slot's reference signature and a cheap
predicate:

```python
from libreyolo import kernels

kernels.register("fake_quant_fp8", my_impl, name="mybackend", predicate=my_check)
```

Parity is gated by `tests/unit/kernels/` against shapes harvested from real
models into `tools/shapes.json`; any accelerated implementation must match
the reference exactly (forward) and to 1e-6 (STE gradients).
