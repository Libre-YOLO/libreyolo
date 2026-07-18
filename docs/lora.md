# LoRA fine-tuning

Parameter-efficient fine-tuning for transformer-based detectors on low-VRAM
GPUs. Freezes the pretrained heavy parts and trains small low-rank adapters
plus the layers that must stay dense (heads, projections).

## Use

```python
model = LibreYOLO("LibreRFDETRn.pt")
model.train(data="data.yaml", lora=True)

model = LibreYOLO("LibreDFINEs.pt")
model.train(data="data.yaml", lora=True)
```

`lora=True` is the whole API. Needs the optional extra:

```
pip install "libreyolo[lora]"
```

## Recipes

Fixed per family, not user-facing knobs.

### RF-DETR

DoRA, rank 16, alpha 16, on the DINOv2 backbone attention
`query`/`key`/`value`. Matches the RF-DETR reference. The ViT backbone is
frozen; the projector, decoder, and detection head keep training normally.

### D-FINE / DEIM / RT-DETR (v1, v2, v4)

These pair a CNN (HGNetv2 or PResNet) backbone with a transformer hybrid
encoder + deformable decoder, so the recipe differs:

- The CNN backbone is frozen entirely. It is the first stage, so no gradient
  flows through it at all and its backward pass is skipped.
- The transformer blocks (AIFI encoder layers, decoder layers) freeze their
  base weights and train plain LoRA adapters (rank 16, alpha 16) on their
  `nn.Linear` layers: FFN `linear1`/`linear2`, the gate, and the deformable
  attention projections.
- Decoder self-attention (`nn.MultiheadAttention`) stays frozen without
  adapters: PyTorch's MHA reads `out_proj.weight` directly, which would
  silently bypass an injected adapter.
- Everything else trains normally: encoder conv fusion, input projections,
  prediction heads, query embeddings.

Plain LoRA instead of DoRA because several decoder Linears are zero-init by
design and DoRA's magnitude normalization divides by the weight norm.

RT-DETRv4 reuses the D-FINE architecture and trainer, so it takes this
recipe by inheritance.

### DEIMv2

Same DETR recipe as above (its SwiGLU FFN `w12`/`w3` Linears are targeted
instead of `linear1`/`linear2`), with one addition for the S/M/L/X sizes,
which carry a DINOv3 ViT backbone: the ViT base weights freeze and its fused
attention `qkv` Linears get adapters (the same `qkv` target the RF-DETR
reference lists for fused layouts), while the Spatial Tuning Adapter conv
pyramid keeps training as the projector analog. The `qkv` adapters are
injected even when the config shipped the ViT frozen: adapting a frozen
backbone is the point of LoRA. Sub-S sizes use HGNetv2 and take the plain
DETR recipe.

## Checkpoints and export

Training checkpoints (`best.pt` and `last.pt`) keep the adapter tensors so
they can be resumed or inspected. Loading those checkpoints requires the
`lora` extra; the loader replays the adapter injection so the peft keys line
up. RF-DETR merges adapters into dense weights on `export()`;
`libreyolo.training.lora.merge_lora_adapters` does the same for injected
D-FINE/DEIM models.

## Scope

- RF-DETR, D-FINE, DEIM, DEIMv2, and RT-DETR v1/v2/v4. Other families raise
  instead of silently ignoring `lora=True`.
- Detection tasks only for D-FINE (segment raises); RF-DETR semantic raises.
- The detection heads always stay trainable (custom class counts need them).
- Saves optimizer/gradient memory and skips the frozen backbone's backward;
  activation memory is unchanged. For the tightest VRAM, lower `batch` or
  `imgsz`.
