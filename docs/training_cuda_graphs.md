# CUDA Graphs for Training

Opt-in capture of the training network's forward and backward passes into
CUDA graphs, cutting per-step kernel-launch overhead on launch-bound runs.

```python
from libreyolo import LIBREYOLO

model = LIBREYOLO("libreyolo9t.pt")
model.train(data="data.yaml", epochs=100, cuda_graph=True)
```

```bash
libreyolo train --model libreyolo9t.pt --data data.yaml --cuda-graph
```

The flag is always safe to pass: a family, task or configuration that
cannot be captured logs one line and trains eager, unchanged.

## What is captured, and why that bounds the win

Only the network's forward and backward. The loss stays eager by design:
detection losses select with boolean masks, run Hungarian matching, and
branch on assignment results, none of which a graph can record. The
optimizer step, gradient clipping, EMA update and LR schedule stay eager
too.

So the ceiling is set by how much of the step is network, and that varies
enormously by family. Measured on an RTX 5070 Ti at 640 px, batch 8:

| Family | Network share of the step | Ceiling if the network were free |
| --- | --- | --- |
| yolo9-t | 84% | 6.1x |
| yolo7-b | 44% | 1.8x |
| yolox-t | 31% | 1.5x |
| rtmdet-t | 26% | 1.3x |

YOLOX and RTMDet spend most of a step inside SimOTA and the dynamic
soft-label assigner, and D-FINE spends most of one inside its aux and
denoising losses. Capturing the network still helps those families, but
their bottleneck is the loss, not launches.

## Measured speedups

RTX 5070 Ti, Windows, AMP, generated data (no dataloader bottleneck),
fastest step of 24 after warm-up. Detection runs at 640 px, classification
at 224 px.

| Family | Size | Batch | Eager | Graphed | Speedup | Numerics |
| --- | --- | --- | --- | --- | --- | --- |
| fomo | s | 16 | 7.0 ms | 1.9 ms | **3.63x** | 1 ULP |
| mobilenetv4 | s | 16 | 14.5 ms | 5.3 ms | **2.74x** | exact |
| efficientnetv2 | b0 | 16 | 29.0 ms | 11.9 ms | **2.44x** | exact |
| yolo9 | t | 8 | 93.6 ms | 47.0 ms | **1.99x** | exact |
| yolo9_e2e | t | 8 | 114.7 ms | 65.3 ms | 1.76x | exact |
| yolo9_p2 | t | 8 | 150.0 ms | 101.0 ms | 1.49x | exact |
| nafnet | s | 8 | 132.5 ms | 105.5 ms | 1.26x | exact |
| resnet | 18 | 16 | 9.5 ms | 7.6 ms | 1.25x | exact |
| picodet | s | 8 | 145.0 ms | 118.7 ms | 1.22x | exact |
| deim | n | 4 | 187.5 ms | 158.3 ms | 1.18x | eager noise |
| rtdetrv4 | s | 4 | 187.0 ms | 158.5 ms | 1.18x | eager noise |
| yolonas | s | 8 | 87.3 ms | 74.4 ms | 1.17x | exact |
| dfine | n | 4 | 185.3 ms | 159.2 ms | 1.16x | eager noise |
| deimv2 | atto | 4 | 124.8 ms | 107.5 ms | 1.16x | eager noise |
| rfdetr | n | 4 | 276.3 ms | 239.8 ms | 1.15x | eager noise |
| rtdetrv2 | r18 | 4 | 95.1 ms | 83.0 ms | 1.15x | eager noise |
| ec | s | 4 | 216.3 ms | 188.2 ms | 1.15x | eager noise |
| convnext | t | 16 | 23.1 ms | 20.1 ms | 1.15x | exact |
| yolox | t | 8 | 102.2 ms | 90.5 ms | 1.13x | exact |
| segformer | b0 | 8 | 60.1 ms | 53.7 ms | 1.12x | own RNG stream |
| rtdetr | r18 | 4 | 97.6 ms | 87.0 ms | 1.12x | eager noise |
| rtmdet | t | 8 | 149.7 ms | 136.2 ms | 1.10x | float rounding |
| yolo7 | b | 4 | 102.5 ms | 98.0 ms | 1.05x | exact |
| lingbotvision | s | 8 | 34.4 ms | 33.1 ms | 1.04x | 1 ULP |

Three things move these numbers:

- **Batch size.** Small batches are launch-bound, large ones are
  compute-bound. RT-DETR-r18 gains 1.19x at batch 2 and 1.04x at batch 8.
- **Platform.** Launch overhead is highest on Windows; Linux gains are
  roughly a third to half of the above.
- **The dataloader.** Graphs only speed up the GPU step. A dataloader-bound
  run sees no wall-clock change, so check `libreyolo profile` first if you
  are not sure where the time goes.

## Numerics

The four "Numerics" values above mean:

- **exact**: the graphed run reproduces the eager loss trajectory bit for
  bit. Most families.
- **1 ULP**: differs in the last bit of float32 (about 1e-7 relative) from
  a different summation order.
- **eager noise**: the family does not reproduce its *own* eager run
  either: deformable attention accumulates its backward with atomics, and
  TF32 convolutions pick a reduction order per launch. The graphed run stays
  inside that spread, and at step 0 its gradients differ from eager by the
  same magnitude two eager runs differ from each other.
- **float rounding** (RTMDet only): RTMDet shares its head convolutions
  across all three pyramid levels, so those two weights' gradient is a sum
  of three contributions. Eager autograd and the graphed backward sum them
  in a different order, which under fp16 differs in the last bits: 137 of
  139 gradients stay bit-identical and the other two differ by about 3e-4
  relative. The dynamic assigner's discrete choices then amplify that over
  a run.
- **own RNG stream** (SegFormer only): its MiT encoder has stochastic depth
  inside the captured region. Capture does not disable it, but a replayed
  graph consumes the generator on its own schedule, so it does not reproduce
  the sequence an eager step would draw. The run is statistically equivalent
  to eager, in the same way a different seed is. The manager logs this once
  at capture time for any family it applies to.

`tests/unit/test_cuda_graph_training.py` gates the dispatch and gating
rules; `tests/e2e/test_cuda_graph_training_families.py` runs real two-epoch
trainings per family and holds each one to the tolerance documented here.

## Shapes

A graph is valid for exactly the input shape it was captured with. The
trainer counts batch shapes and captures once a shape has repeated three
times. Batches at any other shape (multi-scale batches, the last partial
batch of an epoch) run eager, unchanged.

This matters for the DETR families, which resize every batch by default:
with `multi_scale=True` a short run may never see one shape often enough to
capture at all. Pass `multi_scale=False` when you want the full speedup.

Some families change what the captured region computes partway through a
run. YOLOX turns on its L1 regression branch when mosaic closes
(`no_aug_epochs`), which adds tensors to the network's output; the trainer
invalidates the capture at that point and re-captures once the new shape has
settled. A family hook that does something similar should call
`self.invalidate_cuda_graph(reason)`.

## Supported families

| Task | Families |
| --- | --- |
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Everything else, meaning other tasks on the families above, families not
listed, distributed (DDP) runs and distillation runs, downgrades to plain eager
training with a single log message. Any capture failure at runtime also
falls back to eager for the rest of the run.

For the encoder-decoder detectors (D-FINE, DEIM, DEIMv2, RT-DETR v1/v2/v4,
EC) only the backbone and encoder are captured. Their decoder reads the
ground truth to build contrastive-denoising queries, and the number of those
queries comes from the largest ground-truth count in the batch, so the
decoder's token count changes from batch to batch.

## Adding a family

Implement `cuda_graph_train_spec` on the family trainer, returning a
`CudaGraphTrainSpec` from `libreyolo.training.cuda_graph`:

- `network`: a `GraphableNetwork` wrapping the capturable half. Its forward
  must take images only and be static-shaped for a fixed input shape: no
  host syncs, no data-dependent shapes, and no host-to-device copies of
  unpinned CPU tensors (a `torch.arange(...).to(device)` or
  `torch.zeros(...).type_as(x)` inside the forward is illegal during capture
  so build those on the target device, see `YOLOXHead.forward_train_maps`).
- `assemble(flat, imgs, targets, polygons)`: rebuild the network output with
  `network.rebuild(flat)` and run the family's loss exactly as `on_forward`
  would, returning the same outputs dict.

Prefer factoring the eager remainder into a method both `on_forward` and
`assemble` call, so the two paths cannot drift. Three shared mixins already
cover the common shapes:

| Mixin | Covers |
| --- | --- |
| `models/base/classify_cuda_graph.py` | `logits = model(imgs)` + cross-entropy |
| `models/base/semantic_cuda_graph.py` | networks exposing `forward_logits` / `loss_from_logits` |
| `models/base/detr_cuda_graph.py` | `backbone -> encoder -> decoder(feats, targets)` |

Gate the spec on task and model type so derived heads with different loss
boundaries are excluded, return `None` for anything unverified, and add the
family to `tests/e2e/test_cuda_graph_training_families.py` with a measured
tolerance before enabling it.

One trap worth knowing: a family trainer that overrides `_train_epoch` must
call `self._forward_train(imgs, targets, polygons)`, not `self.on_forward`.
`on_forward` is the eager path; `_forward_train` is the one that routes
through the graph. D-FINE and DEIM copied the base loop and called
`on_forward`, which made `cuda_graph=True` a silent no-op for five families.

## Memory

A captured graph pins static input, output and workspace buffers for the
forward and backward pass, so peak VRAM rises by roughly one extra set of
activations for the captured shape. Measured across the families above, peak
allocation moved between −5% and +19%; the relative cost is largest for the
small classification models, whose activations are small to begin with
(ResNet-18 at 224 px, batch 16: 0.48 GB eager, 0.57 GB graphed), and is
negligible for the detectors. If it pushes a run over the limit, reduce the
batch size or leave the flag off.
