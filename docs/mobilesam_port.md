# MobileSAM Native Port

LibreMobileSAM is a native promptable-segmentation family under the LibreSAM
tier. It is not registered in the `LibreYOLO()` detector factory.

## Architecture

- `MobileSAMNetwork.image_encoder`: TinyViT image encoder, output
  `(B, 256, 64, 64)` for a 1024 input frame.
- `MobileSAMNetwork.prompt_encoder`: point, box, and dense-mask prompt encoder.
- `MobileSAMNetwork.mask_decoder`: two-way transformer decoder plus IoU head.
- `preprocess.py`: resize-longest-side image geometry, prompt-coordinate
  transforms, normalization/padding, and mask upscaling.

The native module names intentionally match the upstream MobileSAM v1 checkpoint
layout. `weights/convert_mobilesam_weights.py` loads `mobile_sam.pt` directly
with `strict=True` and writes a schema-compliant LibreYOLO checkpoint wrapper
with `model_family="mobilesam"`, `size="tiny"`, `task="segment"`, `nc=1`, and
`imgsz=1024`.

## Parity Gate

Run the gated parity test with an upstream checkout and checkpoint:

```bash
LIBREYOLO_MOBILESAM_UPSTREAM=/path/to/MobileSAM \
LIBREYOLO_MOBILESAM_CHECKPOINT=/path/to/mobile_sam.pt \
pytest tests/unit/test_mobilesam_parity.py
```

The gate asserts `max_abs_diff == 0` for:

- TinyViT image embeddings.
- Prompt encoder sparse and dense embeddings.
- Mask decoder logits and IoU scores for point and box prompts.

The eval-time TinyViT attention-bias cache must be refreshed after loading
weights; the parity helper calls `eval()` after `load_state_dict()` for that
reason.
