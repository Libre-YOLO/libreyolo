# PyTorch-Native Quantization

LibreYOLO quantizes models directly in PyTorch. Quantized models keep the
normal `predict` / `val` / `train` / `save` contract, so accuracy is measured
with the same validators as float models and accuracy recovery reuses the
existing training and distillation notation.

## Grammar

Two steps. Step 1 always happens; step 2 is optional accuracy recovery.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")

# Step 1: quantize (structure + calibration). calib is a small UNLABELED
# image set used forward-only to derive activation ranges and scales.
qmodel = model.quantize(recipe="int8", calib="coco128.yaml", samples=128)

qmodel.val(data="coco8.yaml")            # honest accuracy, same validators
qmodel.predict("bus.jpg")
qmodel.save("LibreYOLO9s-int8.pt")       # manifest-carrying checkpoint

# Step 2 (optional): QAT is plain train() on the quantized model.
qmodel.train(data="coco.yaml", epochs=5)

# QAD: same step plus the existing distillation kwargs.
qmodel.train(data="coco.yaml", epochs=5, distill_model="LibreYOLO9m.pt")
```

CLI:

```bash
libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib coco8.yaml
libreyolo train --model LibreYOLO9s-int8.pt --data coco.yaml --epochs 5
```

`LibreYOLO("LibreYOLO9s-int8.pt")` restores the quantized structure and
scales automatically (checkpoints carry a `quant` manifest; see
`checkpoint_schema.md`). Trainer checkpoints written during QAT/QAD carry the
manifest too, so `best.pt` from a QAT run is itself a quantized checkpoint.

## Recipes

| Recipe | What it does | Families (v1) | Calibration |
|---|---|---|---|
| `fp16` | Cast to half precision with a float32 I/O contract. Inference-only. | yolo9, rfdetr | none |
| `bf16` | Cast to bfloat16 (fp32's exponent range at half storage; the fix when fp16 overflows on DETR-style models). Inference-only. | yolo9, rfdetr | none |
| `fp8` | E4M3 W+A simulation: per-channel weight scales, calibrated per-tensor activation scales, on `Conv2d` and `Linear`. | yolo9, rfdetr | required for activations |
| `int8` | W8A8 simulation: per-channel symmetric INT8 weights, per-tensor affine INT8 activations, on `Conv2d` and `Linear`. | yolo9, rfdetr | required for activations (skipped with `calib=None`, weights-only) |
| `w4a16` | Grouped symmetric INT4 weights (group 128 along in_features), float activations, on `Linear`. | rfdetr | not needed (weight-only) |
| `w4a8` | Grouped INT4 weights plus calibrated INT8 activations, on `Linear`. Maps to NPU W4A8 deployments (Hexagon, Hailo `a8_w4`). | rfdetr | required for activations |
| `nvfp4` | W4A4 NVFP4 simulation on `Linear`: E2M1 elements, 16-element blocks, FP8 E4M3 block scales, FP32 tensor scale. Dynamic activation scaling. | rfdetr | not needed (dynamic) |
| `mxfp4` | OCP MXFP4 on `Linear`: E2M1 elements, 32-element blocks, power-of-two (E8M0) block scales. Dynamic activation scaling. | rfdetr | not needed (dynamic) |
| `int2` | Research preview: grouped 2-bit weights (group 64) plus INT8 activations, on `Linear`. PTQ alone is unusable; QAT/QAD required. | rfdetr | required for activations |

Linear-only recipes are rejected for conv-heavy families such as yolo9 on
purpose: sub-8-bit acceleration is GEMM-only on current hardware, so
convolutions stay in higher precision. Transformer families (RF-DETR) are
the target; yolo9 uses `int8` or `fp8`.

Per-family `keep_high_precision` defaults protect the first layer and the
heads (and always the YOLO9 DFL conv). Override with
`quantize(..., keep_high_precision=("head.",))` if you know what you are
doing.

## Calibration data is not training data

- `calib=` (quantize): a few hundred images, no labels read, forward-only.
  Purpose: activation ranges and scale generation. Default: `coco128.yaml`
  (auto-downloaded); multiple batches matter because ranges are estimated
  across them.
- `data=` (train/val): the labeled dataset. Purpose: gradients and metrics.

Activation range estimation (`algorithm=`): the default `minmax` keeps the
absolute extremes seen across calibration batches; `percentile`
(experimental) uses the mean of per-batch 0.1/99.9 percentiles. Measured on
coco128, minmax with a multi-batch calibration set wins for every tested
model, and percentile clipping collapses DETR-family accuracy because
transformer activation outliers are functionally load-bearing. What
actually fixes small-model int8 sensitivity is calibrating on enough
batches (hence the coco128 default: with it, YOLO9-t lands within about one
mAP point of fp32). The chosen algorithm is recorded in the checkpoint
manifest.

## Execution tiers

v1 executes quantized arithmetic in **simulation** (fake-quantization with
straight-through-estimator gradients, computed in fp32 islands even under
AMP). Simulation is numerics-true: a `val()` score on any device is a real
claim about the quantized arithmetic. It is not a speed claim; packed
low-bit kernels are a separate deployment concern. `fp16` is the exception:
it executes natively.

`model.quant_info()` reports the recipe, module counts, calibration state,
and execution tier.

## Export

### Finalized PyTorch checkpoints (`format="pt"`)

A prepared checkpoint keeps fp32 masters because training needs them. When
you are done, crystallize:

```python
qmodel.export(format="pt")   # -> <name>-final.pt, packed low-bit weights
```

Finalized checkpoints store real packed weights (int8 tensors + per-channel
scales; nvfp4 as two-codes-per-byte E2M1 payload + E4M3 block scales),
strip the masters, and cast the non-quantized remainder to fp16
(`remainder="fp32"` keeps it exact). Measured: YOLO9-s int8 29.5 to 9.6 MB,
RF-DETR-n nvfp4 122 to 26 MB. The packing invariant: unpacking reproduces
the simulation bit for bit on the device you finalized on, so the finalized
file scores exactly what you validated. Loading one gives an
inference-ready model; `train()` on it re-prepares masters from the packed
weights automatically (QAT-from-PTQ); ONNX export from it re-prepares
internally and emits the same QDQ graph. The packed layout is documented in
`checkpoint_schema.md` as the connection contract for external exporters
and runtimes.

### ONNX (`format="onnx"`)

int8-quantized models export directly to ONNX with in-graph
QuantizeLinear/DequantizeLinear pairs carrying the model's own calibrated
(or QAT-trained) scales:

```python
qmodel = LibreYOLO("LibreYOLO9s-int8.pt")   # PTQ or QAT/QAD checkpoint
qmodel.export(format="onnx")                # scale-exact QDQ INT8 ONNX
```

ONNX Runtime and TensorRT consume the QDQ graph with real INT8 kernels; on
coco8 the exported artifact tracks the PyTorch simulation within sub-point
noise. The CLI equivalent is
`libreyolo export --model model-int8.pt --format onnx`. Notes:

- Cast recipes (`fp16`/`bf16`): call `dequantize()` and use the float
  exporters (`half=True` gives fp16 ONNX).
- Sub-8-bit linear recipes (`w4a16`, `w4a8`, `nvfp4`, `mxfp4`, `int2`) and
  `fp8` have no deployable ONNX form here yet; they execute in PyTorch and
  crystallize via `format="pt"`.
- Other deployment formats for int8 are built downstream from the QDQ ONNX;
  direct engine export is planned.
- `dequantize()` remains available to restore float masters (QAT-trained
  weights are kept) and use any float exporter.

## QAT and QAD mechanics

Quantized modules keep fp32 master weights; fake-quantization applies STE so
gradients flow to the masters. The existing trainers work unchanged: EMA,
AMP, checkpoint resume, and the `distill_*` kwargs (MGD/CWD) all compose.
`fp16`-quantized models are inference-only; the trainer rejects them with a
pointer to `amp=True`.

QAT is a finetune of an already-trained model: use finetune learning rates
(for example `lr0=1e-4` for yolo9), not the from-scratch defaults, or the
short run will destroy the pretrained weights regardless of quantization.

QAD availability follows family distillation support: it works wherever the
family implements `get_distill_config()` (yolo9 and rfdetr today; the
RF-DETR tap point is the stride-16 backbone projector output, probed from
the live model so future sizes stay correct).

Family notes: RF-DETR calibration exercises the inference path, so modules
that only run during training (denoising branches) keep their activation
observers open and stay unquantized on activations until QAT runs;
`quant_info()["calibrated"]` reports this honestly. The RF-DETR trainer also
reinitializes the detection head when the dataset class count differs from
the checkpoint head width (COCO checkpoints have a 91-wide head), which
applies to quantized finetunes exactly as it does to float ones.
