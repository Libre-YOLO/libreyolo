# LibreViT provenance

LibreViT is a native PyTorch port of the fixed 224px, patch-16 classic Vision
Transformer graph. No incompatible source was consulted or used.

## Code sources

- `huggingface/pytorch-image-models`, tag `v1.0.28`, commit
  `8ef73809f622e0031bd7f4940265734aef8b9978`, Apache-2.0:
  <https://github.com/huggingface/pytorch-image-models/tree/8ef73809f622e0031bd7f4940265734aef8b9978>
- `google-research/vision_transformer`, commit
  `64801f1b3b367b3611cc27a3d45cc22870a36fb3`, Apache-2.0:
  <https://github.com/google-research/vision_transformer/tree/64801f1b3b367b3611cc27a3d45cc22870a36fb3>

The implementation retains only the learned class token, absolute position
embedding, pre-normalized attention/MLP blocks, final normalization, and linear
classifier needed by the four shipped checkpoints. State-dict names remain
aligned with timm; conversion adds LibreYOLO metadata without changing learned
tensors.

## Weight sources

All source model cards declare `apache-2.0`. The source artifact in each row is
`model.safetensors`; revisions and SHA-256 hashes pin the exact downloaded
bytes. Converted hashes pin the corresponding published LibreYOLO checkpoint.

| Size | Upstream model and revision | Source SHA-256 | LibreYOLO file | Converted SHA-256 |
|---|---|---|---|---|
| `ti` | [`timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/tree/7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106) at `7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106` | `fecf81b492bd13ee7a5297cb74d1d417aac8bf7e1b7d96aed89c4691984587ed` | `LibreViTti-cls.pt` | `faa3cbe82da7c5a94677b3235c743b47562c4da26ede5d563ebdb89f4fec3394` |
| `s` | [`timm/vit_small_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k/tree/7e2c55630205e1266030f18370f4c6ed1a514b52) at `7e2c55630205e1266030f18370f4c6ed1a514b52` | `79c03c635cdfd798a364a9d8c4e5c0b7255b975ea2c9616046d4f77ab01435aa` | `LibreViTs-cls.pt` | `070717a28e61f8759ae0017c191bf4c80c59228cc863001f128467c08a0d95e0` |
| `b` | [`timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/tree/063c6c38a5d8510b2e57df480445e94b231dad2c) at `063c6c38a5d8510b2e57df480445e94b231dad2c` | `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b` | `LibreViTb-cls.pt` | `f084ee6c8a94dff5a59d34907da82c51a9c71d2cbc6b8ba540820a784bd54ca6` |
| `l` | [`timm/vit_large_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k/tree/0930ab3308b84cb2ae091a4a80703c459412a4c7) at `0930ab3308b84cb2ae091a4a80703c459412a4c7` | `109390825a5bada2864f1b74445a63d29c51ef079c349644bc98996c452f6cf1` | `LibreViTl-cls.pt` | `8881263620f2e19362ccbcce929ca6cf07725f3ce4c24bd81479e16492d262be` |

Conversion is reproducible with `weights/convert_vit_weights.py`. Each output
passes the strict checkpoint metadata validator and strict native state-dict
loading. Pretrained logits match timm exactly (`max_abs_diff == 0`) for all
four sizes.

## Published artifacts

Each public repository contains exactly `.gitattributes`, `README.md`,
`LICENSE`, `NOTICE`, and its one canonical checkpoint:

| Size | LibreYOLO repository | Initial verified revision |
|---|---|---|
| `ti` | [`LibreYOLO/LibreViTti-cls`](https://huggingface.co/LibreYOLO/LibreViTti-cls/tree/b5275cde8067f04681f8b1536538544cc311d95f) | `b5275cde8067f04681f8b1536538544cc311d95f` |
| `s` | [`LibreYOLO/LibreViTs-cls`](https://huggingface.co/LibreYOLO/LibreViTs-cls/tree/c68ab6f37c533bf32580f400fb10b00d0c2aed3a) | `c68ab6f37c533bf32580f400fb10b00d0c2aed3a` |
| `b` | [`LibreYOLO/LibreViTb-cls`](https://huggingface.co/LibreYOLO/LibreViTb-cls/tree/003b0fede7e890fc994478c8dac3dfc48bcbab86) | `003b0fede7e890fc994478c8dac3dfc48bcbab86` |
| `l` | [`LibreYOLO/LibreViTl-cls`](https://huggingface.co/LibreYOLO/LibreViTl-cls/tree/4aa51c2ee7149c97b277154ec8b09e160b26fce8) | `4aa51c2ee7149c97b277154ec8b09e160b26fce8` |
