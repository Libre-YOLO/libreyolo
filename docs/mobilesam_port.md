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

## Pinned Provenance

The clean Apache-2.0 source is MobileSAM revision
`f706ad9c4eb7f219c00d9050e46328518ffb65d2` (LICENSE SHA-256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`).
Its 40,728,226-byte `weights/mobile_sam.pt` checkpoint has SHA-256
`6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`.

The LibreYOLO mirror is pinned to revision
`c80f272421d38fc26ef4bd0c02111b6c1f1c8cb9`. Its 40,730,739-byte
`LibreMobileSAM.pt` has SHA-256
`79f09a3671f38696d45da0aed49ef382fde2efd1bc966d172ac9822b952e35fe`.
After unwrapping the two containers, all 439 named tensors and 10,140,231
parameter values are identical. The canonical sorted state digest is
`92dc21da1d9d0ca2721ac08745d4e77c8f02b4af96b2e8de0aced98c5b4622ea`.

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
