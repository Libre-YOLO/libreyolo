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
| `l2cs` (gaze) | Out of scope. |
| `sensenova` | 7B vision-language model; random init needs far more memory than a single consumer card. |

The SAM family is supported through a family-specific path too. Its entry
point is `set_image()` / `predict(points=)`, and the image encoder is both the
dominant cost and a single fixed-size tensor in, so that is the unit captured:
encode once, prompt many times against the cached result. Prompt encoding and
mask decoding stay eager, being cheap and varying per click.

This one needs a shim. Upstream `SamVisionAttention.get_rel_pos` builds its
relative-position index with `torch.arange` on the host and then indexes a GPU
tensor with it, which capture rejects. `models/sam/transformers_compat.py`
replaces the method with one that builds the same index on the embedding's
device and memoises it per `(q_size, k_size, device)`. Values are identical,
verified against the upstream computation. The patch installs on import of
`libreyolo.models.sam` and declines quietly if a future transformers release
restructures the method, in which case SAM keeps working and stays eager.

`depth_anything3` splits its forward instead. The sky-to-far-depth step
branches on tensor values and selects a data-dependent number of pixels, so it
cannot be recorded; the network in front of it can. Capture stops at the raw
head outputs and the sky step runs eagerly on the replayed result, which leaves
the numbers identical to the fully eager path. Because that tail lives outside
the graph, the class sets `GRAPH_DISPATCH_IN_FORWARD`, which tells
`forward_maybe_graphed` to route through `_forward` rather than calling the
runner directly. Without that flag the shared helper returns the partial
network output and silently skips the tail.

`eomt` needed only that its attention-mask schedule stay on the host. Upstream
tests `attn_mask_probs[i] > 0` and `prob < 1`; both compare a tensor against a
Python scalar, which syncs the stream on a device tensor whether or not the
guarded branch runs. `LibreEoMTNet._apply` keeps that buffer on the CPU across
device moves. It is read only as a scalar, so nothing else changes, and it
stays a registered buffer so checkpoints are unaffected.

`ppocr` is supported through a family-specific path rather than the shared
one. Its `_forward` hook stays unimplemented by design, so the class overrides
`_get_graph_runner` to wrap the detection stage and exposes `forward_det`,
which the pipeline in `models/ppocr/inference.py` calls. Recognition stays
eager on purpose: it runs on text crops whose width varies per line, so a
shape-keyed cache would evict constantly and cost more in capture than replay
returns. Detection input size follows the source aspect ratio, so mixed images
produce several graphs; the runner's cache cap bounds that and falls back to
eager past it.

`rfdetr` is verified on `detect`, `segment` and `pose`. Its `obb` task is not,
because constructing it requires real checkpoint weights rather than random
init; the class-level flag covers it regardless.

For the record on `birefnet`, the divergence is not TF32, not cuDNN
autotuning, not model state mutated by capture, and not uninitialized memory:
eager is bit-stable before and after capture and no buffer changes. The cause
is still unidentified, which is exactly why the family stays off.
