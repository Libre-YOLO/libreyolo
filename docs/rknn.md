# Rockchip RKNN export

Status: experimental F0 integration.

LibreYOLO can compile a fixed-shape ONNX export to Rockchip's `.rknn` format
and run tensor-level parity checks in Toolkit2's x86 host simulator. A board is
not required for export or numerical parity. An RK3588 board is required only
for NPU latency, power, and thermal measurements.

## Licensing boundary

LibreYOLO's RKNN integration is MIT. It was implemented using LibreYOLO's own
export framework and the official Rockchip `rknn_model_zoo` at commit
`bad6c7334531becaf90a561988519b7bec34d0ab` (Apache-2.0) as the public API
reference. No Ultralytics source code was inspected or used.

`rknn-toolkit2` is a separate vendor SDK under a custom license. LibreYOLO does not bundle
or install it. Review Rockchip's SDK license and install the matching x86_64
Linux wheel yourself. On Windows, use WSL2 or a Linux Docker container.

Toolkit2 2.3.2 also requires `setuptools<81` and currently fails with ONNX 1.19
or newer because its compiler imports the removed `onnx.mapping` symbol. The
validated host environment uses `setuptools==80.9.0` and `onnx==1.18.0`.

## Validated scope

High-level `format="rknn"` export is intentionally limited to the exact
floating-build combinations that passed RK3588 compilation, host simulation,
raw-output checks, and real-image detection comparison:

| Model variant | Task | Status |
| --- | --- | --- |
| YOLO9-t | detect | Experimental, simulator validated |
| YOLO9-E2E-t | detect | Experimental, simulator validated |
| PicoDet-s | detect | Experimental, simulator validated |
| YOLO-NAS-s | detect | Experimental, simulator validated |

Other sizes, tasks, and Rockchip targets are rejected before compilation.
The lower-level ONNX-to-RKNN helpers remain available for development probes,
but compile-only results are not presented as model support.

## Export

The Python API follows the common YOLO export shape; `name` selects the target
SoC and defaults to `rk3588`:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t.pt")
path = model.export(
    format="rknn",
    name="rk3588",
    imgsz=640,
    batch=1,
    dynamic=False,
    verify=True,
)
```

The CLI equivalent is:

```bash
libreyolo export model=LibreYOLO9t.pt format=rknn name=rk3588 imgsz=640 verify=true
```

RKNN exports are static and currently require `batch=1`. LibreYOLO exports an
opset-19 ONNX intermediary, compiles it, optionally verifies the compiled model
in the PC simulator, and then removes the intermediary. Metadata is written to
`<model>.rknn.metadata.json` because RKNN has no portable LibreYOLO metadata
field. A successful `verify=True` run also writes
`<model>.rknn.parity.json` with per-output error metrics.
An unsuccessful parity gate writes `<model>.rknn.failed.parity.json`; the
candidate artifact is discarded and any earlier successful export at the
requested path remains untouched.

High-level INT8 export is blocked until representative calibration and
task-accuracy results exist.

## Board-free parity

Keep an ONNX artifact with the same fixed input shape, then compare its raw
outputs with the RKNN host simulator:

```python
import numpy as np
from libreyolo.export import verify_rknn_simulator_parity

input_tensor = np.random.default_rng(0).standard_normal(
    (1, 3, 640, 640), dtype=np.float32
)
metrics = verify_rknn_simulator_parity(
    "LibreYOLO9t.onnx",
    input_tensor,
    target_platform="rk3588",
    rtol=1e-3,
    atol=1e-4,
    raise_on_failure=False,
)
```

Toolkit2's PC simulator runs the in-memory graph produced by `load_onnx()` and
`build()`. It cannot reload a target-specific `.rknn` artifact without a board.
`verify=True` therefore performs compilation, artifact export, and simulation
within one Toolkit2 session.

`verify=True` retains strict elementwise `allclose` in every output record. The
floating-build acceptance result additionally requires cosine similarity of at
least `0.9999` and normalized RMSE of at most `0.02` for each output that is not
strictly allclose. Those scale-independent limits separated every retained
detector from the failed candidates below. They are used only because the
retained model variants also passed the real-image box/class/score gate.

## Validation record (2026-08-04)

The first real run used WSL2 Ubuntu 22.04 x86_64, Python 3.10.12,
`rknn-toolkit2==2.3.2` (wheel repository commit
`59a913d172e7f5ff03c9076e2ec7b1b1288ffd08`), PyTorch 2.4.0,
ONNX 1.18.0, and ONNX Runtime 1.23.2. The SDK and all generated artifacts were
kept outside the LibreYOLO repository.

The canonical repository groups were tested one family at a time. G0 is
YOLO9 and RF-DETR. G1 is YOLO9-E2E, YOLO9-P2, EC, RT-DETR, RT-DETRv2,
RT-DETRv4, D-FINE, DEIM, DEIMv2, and YOLO-NAS. PicoDet and YOLOX were also
tested as additional CNN candidates.

The task-level gate used the CC BY-SA 4.0 Wikimedia Commons image
`People_busy_on_the_road_and_cars.jpg` at 1080x720, SHA-256
`42f8f7f2de8d6719ae379b7bf57e4d6c2fda989431487169fab0393ac75f9fcc`.
The image and generated artifacts were not added to the repository.

### Retained variants, strongest first

| Rank | Model | Raw real-image result | Detection result versus ONNX Runtime |
| --- | --- | --- | --- |
| 1 | YOLO9-t detect, 640 | cosine `0.999999873`, normalized RMSE `0.000505` | 11/11 detections and classes matched; minimum IoU `0.9936`; maximum score error `0.00222` |
| 2 | YOLO9-E2E-t detect, 640 | cosine `0.999999788`, normalized RMSE `0.000652` | 14/14 matched; minimum IoU `0.9946`; maximum score error `0.00355` |
| 3 | PicoDet-s detect, 320 | cosine `0.999999916`, normalized RMSE `0.000411` | 29/29 matched; minimum IoU `0.9906`; maximum score error `0.00383` |
| 4 | YOLO-NAS-s detect, 640 | boxes: cosine `0.999996594`, normalized RMSE `0.00261`; scores: cosine `0.999972251`, normalized RMSE `0.00758` | 20/20 matched; minimum IoU `0.9925`; maximum score error `0.00638` |

All four compiled and ran in the host simulator. Strict elementwise `allclose`
failed because the vendor floating build uses reduced-precision internal
tensors; the task-aware results above are the reason these exact variants are
retained. Toolkit2 also prints its nonspecific `Unknown op target: 0`
diagnostic while building PicoDet-s and YOLO-NAS-s. Their artifacts still run
and pass the recorded gates, but that diagnostic is another reason they rank
below the two cleanly compiling YOLO9 variants.

After the integration was applied to a clean current-`dev` worktree, the
high-level `model.export(format="rknn", verify=True)` path was rerun for all
four variants. Every export passed. On its deterministic random verification
input, normalized RMSE was `0.000588` for YOLO9-t, `0.000424` for YOLO9-E2E-t,
`0.000356` for PicoDet-s, and at most `0.009903` across YOLO-NAS-s's two
outputs; every output exceeded the `0.9999` cosine gate.

### Excluded results

| Model | Compiler / simulator outcome | Reason excluded |
| --- | --- | --- |
| RF-DETR-n, 384 | Requires disabling Toolkit2's failing SDPA fusion; two decoder `GridSample` nodes are not lowered | boxes MAE `0.186`, logits MAE `0.298` |
| YOLO9-P2-t, 640 | Compiles and simulates with a seeded, permissive YOLO9-t transfer fixture | Toolkit2 emits RK3588 register-width errors and the fixture produces no detections; no permissively redistributable trained P2 fixture is available for the task gate |
| EC-s, 640 | Compiles and simulates | boxes cosine `0.99790`, logits cosine `0.99972`; decoded outputs are materially wrong |
| RT-DETR-r18, 640 | Compiles with nine unlowered `GridSample` nodes | boxes cosine `0.81413`, logits cosine `0.99093` |
| RT-DETRv2-r18, 640 | Compiles with nine unlowered `GridSample` nodes | boxes cosine `0.87892`, logits cosine `0.99502` |
| RT-DETRv4-s, 640 | Compiles and simulates | boxes cosine `0.88055`, logits cosine `0.99737` |
| D-FINE-n, 640 | Compiles and simulates | boxes cosine `0.72150`, logits cosine `0.99232` |
| DEIM-n, 640 | Compiles and simulates | boxes cosine `0.91234`, logits cosine `0.99754` |
| DEIMv2-atto, 320 | Compiles and simulates | boxes cosine `0.80983`, logits cosine `0.99308` |
| YOLOX-n, 416 | Compiles and simulates | cosine `0.999727`, normalized RMSE `0.02336`, maximum decoded error `95.45` |

Only the listed variants were tested. Larger variants may share operator
topology, but they remain unverified and are not enabled. YOLO-NAS weights were
used under Deci's separate `LICENSE.YOLONAS.md`; neither weights nor the vendor
SDK are redistributed by LibreYOLO.

No INT8 accuracy or RK3588 board latency result has been recorded. Toolkit2
also constrains PyTorch to 2.4; RF-DETR testing used `transformers==5.1.0`.

## F0 acceptance gate

The integration becomes a generally supported backend only after both flagship
families pass in a pinned Toolkit2 environment:

1. YOLO9 detection: expand the retained real-image gate into repeatable CI.
2. RF-DETR detection: replace or lower `GridSample`, then pass separate logits
   and box parity checks.
3. INT8 YOLO9: representative calibration plus task-level accuracy comparison.
4. RK3588 hardware: latency and memory measurements. This is the only gate that
   requires a board.

Until those results are recorded, the four enabled variants remain an
experimental simulator-validated surface rather than a generally supported
backend. RF-DETR's failure also means this integration does not yet satisfy
LibreYOLO's two-flagship coverage requirement for promotion.
