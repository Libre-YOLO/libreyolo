# LibreMODUS

LibreMODUS integrates the MODUS 14B-A7B checkpoint as an inference-only,
image-conditioned analysis model. It can solve one task through the normal
LibreYOLO result surface or compose several modalities with `any2any()`.

The permanent safety and product boundary is simple: every request needs at
least one image-derived input, and RGB is never an output. LibreMODUS does not
expose text-to-image generation, captioning, VQA, feature-token generation,
or training.

## Install and checkpoint setup

Install the optional runtime:

```bash
pip install "libreyolo[modus]"
```

LibreYOLO does not redistribute MODUS weights. By default, the loader fetches
the required files directly from `EPFL-VILAB/MODUS` at pinned Hugging Face
revision `8428a81602c19141e422b1e1795dddcb5d2bc14b`. A fresh download always
requires the user's own authenticated Hugging Face account, even if the
upstream hosting gate is temporarily open. Review and accept the upstream
terms, then log in or pass a token:

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

To avoid any network request, point at an existing upstream snapshot:

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

The directory must contain:

```text
model.safetensors
ae.safetensors
llm_config.json
vit_config.json
tokenizer_config.json
vocab.json
merges.txt
```

The upstream source repository is Apache-2.0. The checkpoint is a separate
artifact: its model card currently declares `license: other`, describes it as
`bagel-derived`, and requests research-only use. Loading prints that distinction
and links to the upstream terms. Users are responsible for reviewing the
current terms. See `libreyolo/models/modus/NOTICE` for the exact code and
checkpoint provenance.

## Standard task API

The canonical size is `14b-a7b`. Both the class and the `LibreVLM` factory are
supported:

```python
from libreyolo import LibreMODUS, LibreVLM

model = LibreMODUS(size="14b-a7b", task="normal")
# Equivalent:
model = LibreVLM("libremodus-14b-a7b", task="normal")
```

The regular `predict()` result depends on the active task:

| Task | MODUS target | Result payload |
| --- | --- | --- |
| `depth` | `depth` | `result.depth_map` |
| `normal` | `normal` | `result.normal_map` / `result.normals` |
| `edge` | `canny` | `result.edges` |
| `detect` | `cocodet` | COCO-80 `result.boxes` |

```python
model = LibreMODUS(task="normal")
result = model.predict("room.jpg")
normals = result.normal_map.data

model.set_task("edge")
result = model.predict("room.jpg")
edge_probability = result.edges.data
```

The released standard inference recipe uses text guidance `4.0`, image
guidance `2.0`, and ten flow updates. They can be changed explicitly with
`inference_cfg=`, `inference_image_cfg=`, and `inference_steps=` at
construction.

Detection uses the released constrained coordinate and label grammar. With no
custom vocabulary it decodes the checkpoint's COCO label tokens into contiguous
COCO-80 class ids:

```python
model.set_task("detect")
result = model.predict("street.jpg")
```

Calling `set_classes()` switches detection to phrase grounding. Each phrase is
run independently and returned through the same `Boxes` contract:

```python
model.set_classes(["red bus", "cyclist"])
result = model.predict("street.jpg", conf=0.2)
```

`prompt="red bus"` at construction is a one-phrase convenience. If both
`names=[...]` and `prompt=...` are supplied, the explicit class list wins.

## `any2any()`

`any2any()` accepts one to three image-derived inputs plus optional auxiliary
text. It returns the same `Results` type as `predict()`.

```python
result = model.any2any(
    inputs={"rgb": "room.jpg", "depth": depth_array},
    target="normal",
    steps=10,
    cfg=2.0,
    seed=0,
)
```

Unlike the standard single-image recipe, `any2any(cfg=...)` applies one
guidance scale to both the text and image channels. All image-derived inputs
must describe the same aligned canvas and have the same width and height;
LibreMODUS rejects mismatched inputs instead of resizing them independently.

Public input aliases and targets are:

| Inputs | Aliases |
| --- | --- |
| RGB image | `rgb`, `image` |
| Relative depth | `depth` |
| Surface normals | `normal`, `normals` |
| Edge image | `canny`, `edge`, `edges` |
| Auxiliary text | `text`, `prompt` |

| Target | Aliases / result |
| --- | --- |
| Relative depth | `depth` → `depth_map` |
| Surface normals | `normal`, `normals` → `normal_map` |
| Canny-style edges | `edge`, `edges`, `canny` → `edges` |
| SAM-derived edges | `samedge` → `edges` |
| COCO detection | `detect`, `cocodet` → `boxes` |
| Phrase grounding | `grounding`, `det` → `boxes` |

Every image-derived input can condition every public analysis target:

| Input \ target | depth | normal | edge | samedge | detect | grounding |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| rgb | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| depth | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| normal | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| canny / edge | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Grounding needs auxiliary text:

```python
result = model.any2any(
    {"rgb": "street.jpg", "text": "red bus"},
    target="grounding",
)
```

Text cannot be the only input. RGB cannot be a target. Duplicate aliases such
as `{"edge": ..., "canny": ...}` are rejected because they name the same
modality.

Floating-point relative-depth arrays may use arbitrary finite values and are
min-max normalized into the checkpoint's grayscale representation. Canny
arrays use `[0, 1]`. Floating-point normal arrays use LibreYOLO's documented
camera frame (`+x` right, `+y` down, `+z` into the scene) and unit vectors face
the camera, so a fronto-parallel surface is `(0, 0, -1)`. LibreMODUS reverses
the complete vector at the checkpoint boundary because the released MODUS
normal raster uses the opposite orientation. Integer/PIL normal rasters are
interpreted as the public `(normal + 1) / 2` visualization and reoriented the
same way.

### Chaining

Intermediate targets can be generated and fed back into the same context:

```python
result = model.any2any(
    {"rgb": "room.jpg"},
    target="normal",
    chain=("edge", "depth"),
    steps=10,
)
```

Input modalities plus chained intermediates may not exceed the checkpoint's
three-condition training budget. Chaining is sequential and deterministic for
a fixed seed; each stage advances the seed by one.

### Self-verification

`verify=N` generates `N` candidates and asks the same model a constrained
yes/no consistency question for each candidate. The highest yes-probability
candidate is returned:

```python
result = model.any2any(
    {"rgb": "street.jpg", "text": "red bus"},
    target="grounding",
    verify=3,
)
print(result.verification_score)
print(result.verification_candidates)
```

`verify=0` disables this extra work. Enabled values start at 2. The score ranks
candidates within one request; it is not a calibrated task confidence.

## Precision tiers

BF16 is the default:

```python
model = LibreMODUS(dtype="bf16")
```

The local FP8 tier stores eligible decoder-trunk linear weights as E4M3 with a
per-output-channel scale:

```python
model = LibreMODUS(dtype="fp8")
```

The first and last decoder blocks, embeddings, language head, norms,
timestep/AdaLN modulation, projectors, SigLIP tower, and FLUX VAE remain BF16.
Weights are dequantized to the input dtype for each GEMM; this is a
memory-focused weight-only tier, not the activation-quantized native FP8 tier
described for ordinary LibreYOLO checkpoints in `quantization.md`.

Conversion streams the original safetensor and writes a sharded cache under
`~/.cache/libreyolo/modus/fp8`, keyed by the pinned source revision (or a local
source SHA-256) and the complete recipe. The cache is never uploaded and does
not change the source checkpoint's terms.

## Unsupported surfaces

- `train()` and fine-tuning
- `val()` through the generic dataset validator
- ONNX, TensorRT, TFLite, or other export
- batched predict and test-time augmentation
- RGB generation, captioning, VQA, feature tokenizers, `samseg`, and instance
  segmentation decoding

The edge specialists remain the deployment choice when one compact,
exportable edge model is sufficient. LibreMODUS is for cross-modal composition
and one-model operations.

## Validation contract

CPU CI uses a two-layer toy configuration to check the 196,840-row tokenizer
contract, exact learned checkpoint keys, prompt boundaries, constrained
detection grammars, dense decoding, dispatch, chaining, verification, and
local FP8 conversion. It never downloads the external checkpoint.

Full-weight BF16 parity, task metrics, FP8 quality deltas, and the under-12-GB
VRAM gate require the pinned external checkpoint and suitable GPU hardware.
Their exact manual commands and thresholds are recorded in `testing.md`; they
must be reported before a release claims those gates as passed.
