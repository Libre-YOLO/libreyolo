# Upgrading

What changes when you move between LibreYOLO versions, and what you have to do
about it. Only versions with user-visible migration work are listed.

The full list of changes for any release is in
[CHANGELOG.md](../CHANGELOG.md); this page carries only the parts that require
you to edit code or re-check numbers.

## v1.4.0 to v1.5.0

Nothing was removed from the public API surface: every class and function that
worked in v1.4.0 still imports and still works (`__all__` grew from 101 names
to 142 with zero removals). Four things need a code change, and three change
numbers you may be comparing against.

### Code changes you must make

#### `allow_experimental=True` no longer exists

The acknowledgement gate is gone, along with the
`ddp_aware(experimental_key=...)` mechanism behind it. EC, RTMDet, PicoDet and
FOMO training and export previously required the argument, so any script that
trains one of those families is affected.

```python
# v1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# v1.5.0: delete the argument
model.train(data="data.yaml", epochs=100)
```

There is no deprecation shim. A call that still passes it raises `TypeError`.

`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` was removed with it. The
`get_download_notice()` hook survives and is still overridden by midas,
segformer and yolo9_p2; only the base implementation returns `None`.

#### The export support tier `"experimental"` no longer exists

```python
from libreyolo.export.support import Tier
# v1.4.0: Literal["validated", "experimental", "blocked"]
# v1.5.0: Literal["validated", "available", "blocked"]
```

If you branch on the tier string, replace `"experimental"` with `"available"`.
`BaseExporter` no longer emits a `RuntimeWarning` for those formats.

#### `pretrained=False` with `resume` is now rejected

The combination previously proceeded incoherently. It now raises:

```
ValueError: pretrained=False cannot be combined with resume.
```

Pick one. `pretrained=False` starts from a fresh seeded initialization;
`resume` continues an interrupted run from its checkpoint.

#### CLI `--imgsz` is a string, not an int

This one is narrower than it sounds. Both of these are unaffected:

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # still fine
```

```python
model.predict("img.jpg", imgsz=640)   # still fine
```

Only code that calls the CLI *command functions* directly from Python needs to
change, because `predict`, `train` and `val` widened `--imgsz` from `int` to
`str` so it can accept rectangular sizes:

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)     # v1.4.0
predict_cmd(..., imgsz="640")   # v1.5.0, and "480x640" now works too
```

`train`'s default is now the string `"640"`. `export --imgsz` was already a
string, and `profile` is unchanged.

### Numbers that change

If you track metrics across versions, three changes move them at default
settings.

#### faster-coco-eval is the default COCO metrics backend

`val()` and per-epoch training validation now compute COCO metrics with the
faster-coco-eval C++ backend instead of pycocotools.

The switch was measured across all 100 RF100-VL test splits: 1381 of 1400
metric values bit-identical, maximum deviation 2.22e-16, headline deltas
exactly 0, at 15.6x faster overall and 56x on detection-dense datasets. In
practice your numbers should not move, but they are produced by a different
implementation, so this is worth knowing before you compare a v1.5.0 run
against a v1.4.0 one.

pycocotools remains the automatic fallback when faster-coco-eval is not
installed. To force it:

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

or set `LIBREYOLO_FASTER_COCO_EVAL=0`. The backend actually used is logged at
INFO, exposed as `model.last_eval_backend` after `val()`, and included as
`eval_backend` in the CLI JSON payload. Install the fast path with
`pip install libreyolo[fast-eval]`.

#### YOLOX checkpoints trained before v1.5.0 need an eps override to score faithfully

This one is a real trap, so read it if you have fine-tuned YOLOX.

YOLOX specifies BatchNorm `eps=1e-3` and `momentum=0.03`. Until v1.5.0 those
values were applied as a post-hoc fixup that did **not** survive the
class-count rebuild `train()` performs when your dataset's `nc` differs from
the checkpoint's. So such a fine-tune trained and reported in-training
validation at torch's default `eps=1e-5`, then reloaded for inference at
`1e-3`: the same tensors under different normalization.

Regular-conv sizes barely move. Depthwise `n` moves a lot, because its
per-channel `running_var` is small enough for eps to dominate. On RF100-VL
`ball`, the same nano checkpoint scores **0.566** mAP50-95 evaluated at its
trained eps and **0.151** after a stock reload.

A checkpoint trained before v1.5.0 carries eps=1e-5 semantics. To report
faithful numbers for it, either evaluate with BN eps overridden to 1e-5, or
fold `sqrt((var + 1e-3) / (var + 1e-5))` into the BN weights. Checkpoints
trained on v1.5.0 and later need neither.

#### D-FINE multi-scale training uses the upstream per-size recipe

`base_size_repeat` was hardcoded to 3 for every size; it now resolves per size
as upstream specifies: **n** trains at fixed size (multi-scale off), **s** 20,
**m** 6, **l** 4, **x** 3. Only x matched before, so n/s/m/l now see a
different scale distribution and converge to different metrics.

To restore the old behavior, set it explicitly:

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM still uses the hardcoded 3.

### Worth knowing, but no action needed

- **Rectangular `imgsz` results changed because they were wrong before.** Box
  coordinates, RTMDet mask resizing, YOLO-NAS rescaling and validator
  ground-truth scaling now use per-axis height and width instead of one
  scalar. Square `imgsz` is bit-unchanged. If you ran rectangular inference or
  validation on v1.4.0, those numbers were mis-scaled. YOLO-NAS now rejects
  rectangular `imgsz` outright rather than silently producing wrong output.
- **Metrics dictionaries gained keys.** `max_det`, `ar_max_det` and
  `AR_max_det` from the COCO evaluator, and `metrics/loss` plus
  `metrics/loss/ce` from FOMO. Values at defaults are unchanged, but anything
  iterating metric keys (custom loggers, CSV headers) sees new columns.
- **Seeded YOLO9 runs that trigger a head rebuild** start from a different
  initialization, because the seed is now applied before the rebuild rather
  than after. A seeded v1.4.0 fine-tune onto a different class count is not
  reproducible bit-for-bit on v1.5.0.
- **`libreyolo[hub-kernels]` on CUDA now actually engages the native
  MS-deform-attn kernel.** v1.4.0 gated it behind a condition RF-DETR never
  took, so the kernel never ran. Predictions can now shift at float tolerance
  for RF-DETR and the other deformable-attention families. Stock installs are
  unaffected; `LIBREYOLO_HUB_KERNELS=0` disables it.
- **`libreyolo predict` drops unsupported options instead of raising.** The
  CLI filters kwargs against the model's `__call__` signature, so an option a
  family does not accept is now ignored rather than raising `TypeError`. A
  typo in a flag name will be silently ignored.
- **Live sources change the JSON output shape.** Webcams, RTSP streams and
  screen capture implicitly enable streaming, which emits one record per frame
  rather than one for the call. These sources are new in v1.5.0, so no v1.4.0
  script is affected.
- **Re-exporting `rfdetr-pose` or `yolonas-pose` to ONNX yields different
  output names.** v1.4.0 misread their multi-tensor pose heads as segmentation
  via an output-count heuristic. Existing `.onnx` files on disk are untouched.
- **On a torch-free install**, results hold numpy arrays rather than
  `torch.Tensor`, so `.boxes.data` returns a different type and NMS
  tie-breaking may differ. With torch installed, behavior is byte-for-byte
  unchanged.
- **Config objects validate more at construction.** `TrainConfig` gained a
  `__post_init__` where it had none, so a config that was already invalid now
  raises immediately instead of failing deep into a run.
- **Weight filenames for task-suffixed families resolve differently.**
  `segformer-b0` now resolves to `LibreSegformerb0-sem.pt`. This fixes
  auto-download 404s, but breaks any script that hardcoded the old
  unsuffixed filename.
- **The pytest marker `experimental_backend` is now `extended_backend`.**
  Only relevant if you run the test suite with `-m`.

### Checkpoints and datasets

Checkpoints written by v1.4.0 load unchanged. The schema gained
`imgsz_h`/`imgsz_w` for rectangular models, and still writes the scalar
`imgsz = max(h, w)` for older readers. ExecuTorch and MNN exports now require
a sidecar (`<program>.pte.json`, `<model>.mnn.json`), and HRNet exports carry
`pose_input: "person_crop"`. Dataset formats are unchanged.
