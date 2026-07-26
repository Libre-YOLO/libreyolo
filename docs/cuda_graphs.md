# CUDA graphs

`predict(..., cuda_graph=True)` replays the forward pass from a captured CUDA
graph instead of launching each kernel individually. It helps most where launch
overhead dominates, so small models at small batch sizes; it does nothing for
work that is already compute bound.

Support is declared per family through the `SUPPORTS_CUDA_GRAPH` class
attribute, which defaults to `False`. Anything that has not opted in raises
`NotImplementedError` rather than falling back silently.

## Why capture is opt-in

A graph records a fixed sequence of kernels along with the memory addresses
they read and write. It does not record values, shapes, or control flow. If a
forward does something the graph cannot encode, replay does not raise: it
returns **wrong numbers silently**. A wrong-but-plausible mAP is worse than
"not supported yet", so families are enabled only after verification.

Three things make a forward uncapturable:

1. **Host-to-device copies.** Building a tensor from Python values inside the
   forward copies from host memory, which capture rejects, even when the
   destination is CUDA. `torch.tensor([w, h], device="cuda")` is a copy.
2. **Stream syncs.** `.item()`, `.cpu()`, `bool(tensor)`, or an `assert` on a
   device tensor forces a sync.
3. **Data-dependent shapes or branches.** A head that emits a variable number
   of elements cannot be a fixed kernel sequence.

Note that NMS and postprocessing are *not* a barrier: capture wraps
`model._forward(x)` only, which ends before postprocessing runs.

## Verifying a new family

Parity is weight-independent, so no checkpoint is needed:

```python
model = LibreSomething(model_path=None, size="s", device="cuda")
model.model.eval()          # see the trap below, this is mandatory
```

Capture at a fixed shape, then replay against **two different inputs** and
require every output tensor to match eager exactly. Add the family to
`tests/unit/test_cuda_graph_families.py` and set `SUPPORTS_CUDA_GRAPH = True`.

Traps that have each produced a wrong answer in practice:

- `model_path=None` leaves the network in **train mode**, and several families
  take a CPU-building branch while training. `predict()` runs in eval, so
  probing without `.eval()` measures a path users never hit.
- The first output tensor is an **anchor grid** for several families and does
  not depend on the input, so a replay that ignored its input entirely would
  still match on it. Check input dependence across all outputs.
- A **failed capture can poison the CUDA context** for the rest of the process
  ("Offset increment outside graph capture encountered unexpectedly"). Probe
  each family in its own process, or one failure cascades into false negatives
  for everything after it.

## Not supported, and why

| Family | Reason |
| --- | --- |
| `birefnet` | Captures, but replay drifts from eager by ~1.6e-2 across nearly every element. Not stale and it tracks its input, so the signature points at kernel selection under capture. Cause unidentified. |
| `eomt` | The blocking operation is inside the vendored transformers module rather than LibreYOLO code. |
| `depth_anything3` | `_apply_mono_sky` branches on tensor values and produces data-dependent shapes. Structural, not a placement issue. |
| `ppocr` | Two-stage pipeline; does not use the single-tensor `_forward` hook. |
| SAM family (`sam`, `mobilesam`, `picosam3`, EdgeTAM) | Promptable; entry point is `set_image()` / `predict(points=, bboxes=)`, not a single-tensor forward. |
| `siglip2` | Untested here; its text tokenizer needs the optional `sentencepiece` dependency. |
| `l2cs` (gaze) | Out of scope. |

`rfdetr` is verified on `detect`, `segment` and `pose`. Its `obb` task is not,
because constructing it requires real checkpoint weights rather than random
init; the class-level flag covers it regardless.

For the record on `birefnet`, the divergence is not TF32, not cuDNN
autotuning, not model state mutated by capture, and not uninitialized memory:
eager is bit-stable before and after capture and no buffer changes. The cause
is still unidentified, which is exactly why the family stays off.
