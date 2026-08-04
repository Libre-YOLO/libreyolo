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

## When it helps

Small and mid-size models are launch-bound during training: the host
submits thousands of short kernels per step and the GPU idles between
launches. Capturing the network into a graph replays the whole step as one
launch. Measured on an RTX 5070 Ti (Windows, AMP, 640 px, synthetic data,
no dataloader bottleneck):

| Model | Batch | Eager | Graphed | Speedup |
| --- | --- | --- | --- | --- |
| yolo9-t | 16 | 105.1 ms/step | 67.5 ms/step | 1.56x |
| yolo9-t | 8 | 92.6 ms/step | 41.2 ms/step | 2.25x |
| yolo9-s | 16 | 115.5 ms/step | 94.0 ms/step | 1.23x |
| yolo9-m | 8 | 93.0 ms/step | 84.2 ms/step | 1.10x |

Two caveats. Launch overhead is highest on Windows; Linux gains are
smaller (roughly a third to half of the numbers above). And graphs only
speed up the GPU step: a dataloader-bound run sees no wall-clock change,
so check `libreyolo profile` first if unsure where the time goes.

## What is captured

Only the network forward and backward. The loss (data-dependent mask
selects, Hungarian matching), optimizer step, gradient clipping, EMA
update and LR schedule stay eager. A graph is valid for exactly one input
shape: the trainer counts batch shapes and captures once a shape has
repeated a few times. Batches at any other shape, such as multi-scale
batches or the last partial batch of an epoch, run eager unchanged.

Enabling the flag never changes training numerics. YOLO9 trains
bit-identically to eager; RF-DETR matches within its own eager
run-to-run noise (its deformable-attention backward uses atomic
accumulation, so even two identical seeded eager runs differ slightly).
`tests/unit/test_cuda_graph_training.py` gates both.

Multi-scale training composes with capture (one recurring shape gets the
graph, other scales run eager), but most of the benefit comes from
static-shape runs; pass `multi_scale=False` for RF-DETR when you want the
full speedup.

## Supported families

| Family | Task | Support |
| --- | --- | --- |
| yolo9 (and yolo9_p2) | detect | yes |
| rfdetr | detect | yes |
| all others / other tasks | | eager fallback with a warning |

Unsupported configurations downgrade to plain eager training with a
single log message; `cuda_graph=True` is always safe to pass. Not
supported in this version: distributed (DDP) runs, distillation runs, and
non-detect tasks. Any capture failure at runtime also falls back to eager
for the rest of the run.

## Adding a family

Implement `cuda_graph_train_spec` on the family trainer, returning a
`CudaGraphTrainSpec` from `libreyolo.training.cuda_graph`:

- `network`: a `GraphableNetwork` wrapping the model. Its forward must be
  target-free and static-shaped for a fixed input shape (no host syncs,
  no data-dependent shapes; constants created per call must be cached,
  see `_cached_spatial_shapes` in the RF-DETR transformer for the
  pattern).
- `assemble(flat, imgs, targets, polygons)`: rebuild the network output
  (`network.rebuild(flat)`) and run the family's loss exactly as
  `on_forward` would, returning the same outputs dict.

Gate the spec on task and model type so derived heads with different loss
boundaries are excluded, and add a parity test comparing eager and
graphed loss trajectories before enabling.

## Memory

A captured graph pins static input, output and workspace buffers for the
forward and backward pass, so peak VRAM rises roughly by one extra set of
activations for the captured shape. If that pushes a run over the limit,
reduce batch size or leave the flag off.
