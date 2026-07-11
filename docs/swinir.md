# SwinIR super-resolution

LibreSwinIR provides inference and paired PSNR/SSIM validation for the official
SwinIR 4x super-resolution checkpoints. It uses the canonical `restore` task
and returns the upscaled RGB image in `Results.restored` with
`Results.restore_scale == 4`.

| Size | Official tier | Intended use |
|---|---|---|
| `s` | SwinIR-S lightweight | Lower memory and latency |
| `m` | SwinIR-M real-world | Default real-photo upscaling |
| `l` | SwinIR-L real-world | Highest-capacity tier |

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreSwinIRm-restore.pt")
result = model.predict("input.jpg", tile=256, tile_pad=16)
result.save("upscaled.png")
```

Inputs run at native resolution and are reflect-padded to an 8-pixel window
multiple. Tiling is optional and bounds peak memory for large images. Training
and dynamic spatial export are outside the first release. Static ONNX export is
supported; backend prediction pads inputs that fit within the exported canvas
and crops the 4x result back to the expected output shape.

The architecture is adapted from the official Apache-2.0 SwinIR repository at
commit `6545850fbf8df298df73d81f3e8cba638787c8bd`. See
`libreyolo/models/swinir/NOTICE` and `THIRD_PARTY_NOTICES.txt` for provenance.
