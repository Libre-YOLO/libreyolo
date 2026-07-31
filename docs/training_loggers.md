# Training hooks and experiment loggers

## Training hooks

Every trainable LibreYOLO model family accepts `callbacks=` and emits four
events through its trainer. This includes YOLOv9, YOLOv9-E2E, YOLOX,
YOLOv7, YOLO-NAS, RT-DETR, RT-DETRv2, RT-DETRv4, RF-DETR, D-FINE, DEIM,
DEIMv2, PicoDet, RTMDet, EC, and FOMO. Inference-only families still raise
`NotImplementedError` from `train()`; this includes SAM, L2CS, Depth
Anything V2, and the VLM tier.

Pass handlers via `callbacks=` on `model.train(...)`:

| Event | When | Key fields |
|---|---|---|
| `TrainStartEvent` | After setup, before the first epoch | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | After each epoch (train + val) | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | After training completes | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | If training raises | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

`TrainStartEvent.config` is the fully resolved training configuration
(user kwargs merged with model-family defaults) as a read-only mapping.

A plain callable receives `TrainEpochEvent` only. An object may implement
any subset of `on_train_start`, `on_train_epoch_end`, `on_train_end`,
`on_train_exception`:

```python
from libreyolo import LibreYOLO9
from libreyolo.training import TrainEpochEvent

def on_epoch(e: TrainEpochEvent):
    print(f"epoch {e.epoch}/{e.total_epochs} loss={e.train_loss:.4f}")

model = LibreYOLO9("yolo9-s.pt")
model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
```

Callbacks fire on rank 0 only under DDP. For multi-GPU spawn
(`device="0,1"`), callbacks must be picklable: define them as a
module-level class, not a closure or lambda.

## Built-in loggers

Built-in loggers are callback objects layered on the same universal hooks.
Enable TensorBoard, MLflow, or Weights & Biases by name, or pass configured
instances:

```python
model.train(data="coco8.yaml", loggers="tensorboard")
model.train(data="coco8.yaml", loggers="mlflow")

from libreyolo.training import MLflowLogger
model.train(
    data="coco8.yaml",
    loggers=[MLflowLogger(experiment_name="my-exp"), "tensorboard"],
)
```

All three log the same canonical metric names per epoch: `train/loss`,
`train/loss/<component>`, `lr/<group>`, `val/<metric>`,
`time/epoch_seconds`. They also log the resolved training config at
start. A backend failure mid-run (server down, auth expired) disables
the logger with a warning; training is never interrupted. A missing
backend package raises at construction with the install command.

### Validation loss for detection

Standard YOLO9 and RF-DETR detection training can opt in to validation loss:

```python
model.train(data="coco8.yaml", val_loss=True)
```

The validator reuses the model output already produced for mAP; it does not
run a second network forward. YOLO9 reports `val/loss`, `val/loss/box`,
`val/loss/cls`, and `val/loss/dfl`. RF-DETR reports `val/loss`,
`val/loss/ce`, `val/loss/bbox`, and `val/loss/giou`, with the total covering
the same main, auxiliary-decoder, and encoder terms as training. The always-on
artifact names are the corresponding `metrics/loss...` keys, and
`libreyolo monitor` overlays `metrics/loss` with `train/loss`.

This option is off by default because target assignment adds work and memory
to validation. It runs under `torch.no_grad()` with the evaluation/EMA model,
and distributed training computes it locally on rank 0 without collectives.
Best-checkpoint selection remains based on the configured accuracy metric.
YOLO9-E2E, YOLO9-P2, augmented validation, and non-detection tasks are not
supported by this first implementation and raise a clear configuration error.

### TensorBoard

```
pip install libreyolo[tensorboard]
```

`TensorBoardLogger(log_dir=None)` — event files default to
`<save_dir>/tensorboard`. View with `tensorboard --logdir runs/train`.

### MLflow

```
pip install libreyolo[mlflow]
```

`MLflowLogger(tracking_uri=None, experiment_name=None, run_name=None,
log_artifacts=True, log_checkpoints=False)` — the tracking URI falls
back to `MLFLOW_TRACKING_URI`, then MLflow's default local store. At
train end it uploads `results.csv`, `train_config.yaml` and
`summary.json` (plus `weights/best.pt` with `log_checkpoints=True`) and
closes the run as FINISHED, or FAILED if training raised.

Note: MLflow 3.x deprecated the local `./mlruns` file store and raises
unless `MLFLOW_ALLOW_FILE_STORE=true`. For server-less local tracking
pass a database URI instead, e.g.
`MLflowLogger(tracking_uri="sqlite:///mlflow.db")`, and view it with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

### Weights & Biases

```
pip install libreyolo[wandb]
```

`WandbLogger(project=None, name=None, entity=None,
log_checkpoints=False)` — project falls back to `WANDB_PROJECT`, then
`"libreyolo"`. The resolved config becomes the run config;
`log_checkpoints=True` uploads `weights/best.pt` as a model artifact.

Run names default to `<family><size>-<task>` (e.g. `yolo9s-detect`).

## Always-on run status (`status.json`, `metrics.jsonl`, `train.log`)

Separate from the opt-in loggers above, every training run (all
families, no configuration) writes a small set of monitoring artifacts into
its `save_dir`. They exist so an agent-launched run can be watched cheaply,
without a third-party account or tailing the full log.

| File | Written | Contents |
|---|---|---|
| `status.json` | rewritten atomically every epoch (+ on start/end/failure) | live snapshot: `state` (`running`/`completed`/`failed`), `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, latest `metrics`, `best_metric`/`best_epoch`, and on failure an `error` `{type, message}` |
| `metrics.jsonl` | appended once per epoch | one JSON row per epoch (same schema as the family `results.csv`), the full history for charts |
| `train.log` | tee'd live | the run's `libreyolo` console output |

These are produced by `TrainingStatusCallback`, attached automatically
alongside the family artifact writer. `status.json` is the cheap read for a
polling agent (a few tokens vs. re-parsing a log); the atomic write means a
reader never observes a half-written file.

### Live web dashboard

```bash
libreyolo monitor                     # watch the most recent run under runs/
libreyolo monitor runs/train/exp      # watch a specific run
```

`libreyolo monitor` serves a zero-dependency (stdlib HTTP server) browser
dashboard over the files above: live metric charts, the log tail, and any
validation/plot images, auto-refreshing while the run is active. It is
read-only and never touches the training process, so it attaches to a live
run, re-opens a finished one, or inspects a crashed one, and keeps working
even if the trainer dies.
