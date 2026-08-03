# AlexNet

- **LibreYOLO module:** `libreyolo/models/alexnet/`, postprocessing in
  `libreyolo/postprocess/alexnet.py`, and converter at
  `weights/convert_alexnet_weights.py`.
- **Upstream code:** `pytorch/vision` at commit
  `336d36e8db990a905498c73933e35231876e28bc` (torchvision v0.26.0). The
  upstream AlexNet source file's latest commit at that revision is
  `683baf8ee762cceeeae01b3ff04ae4ad606abe70`.
- **Code license:** BSD-3-Clause, Copyright (c) Soumith Chintala 2016. The full
  notice is reproduced in `libreyolo/models/alexnet/NOTICE`.
- **Scope:** ImageNet-1K classification inference, top-1/top-5 validation, and
  export. The original training recipe is intentionally excluded and
  `train()` raises `NotImplementedError`.

## Source checkpoint

The official checkpoint was downloaded from PyTorch's model host, checked by
SHA-256, and converted without changing learned tensors. The source and
converted files are not included in this repository.

| File | Bytes | SHA-256 |
|---|---:|---|
| `alexnet-owt-7be5be79.pth` | 244,408,911 | `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` |
| `LibreAlexNetb-cls.pt` | 244,431,825 | `95f6996b7b4c5526e7e47ad99cf78b2a3643baa3ba1d4107ab840a05e73d1f5e` |

The official URL is
`https://download.pytorch.org/models/alexnet-owt-7be5be79.pth`. Conversion
preserves every state-dict tensor and adds LibreYOLO checkpoint metadata.

### Weight-license status

The torchvision source is BSD-3-Clause. The publisher does not attach a
separate license file to the checkpoint object, so redistribution uses
**BSD-3-Clause implied by the releasing project**, not an explicit
checkpoint-specific grant. Torchvision separately warns that pretrained-model
terms may derive from training data and leaves use-case permission to the
user. The model was trained on ImageNet-1K, whose image rights remain with the
individual sources.

The maintainer-approved public mirror therefore carries the verbatim
torchvision BSD-3-Clause license, an attribution notice, and that caveat. Its
card must not describe the checkpoint license as publisher-confirmed.
`weights/upload_alexnet_hf.py` builds and validates the exact five-file
repository at
[`LibreYOLO/LibreAlexNetb-cls`](https://huggingface.co/LibreYOLO/LibreAlexNetb-cls).
The repository was verified public with exactly five files, added to the
LibreYOLO classification collection, and strict-loaded through the bare
filename auto-download route from a fresh cache.

## Graph contract

LibreYOLO ports torchvision's single-tower AlexNet graph, sometimes called the
"one weird trick" variant. It has a 64-channel first convolution and omits the
original paper's two-GPU groups and local response normalization. The
`features`, `avgpool`, and `classifier` names and indices match the pinned
torchvision state dictionary.

## Measured evidence

- Preprocessing on the bundled parkour image: `max_abs_diff == 0.0` against
  `AlexNet_Weights.IMAGENET1K_V1.transforms()`.
- Eager logits on the same tensor: `max_abs_diff == 0.0` against torchvision.
- Unified prediction: probability `max_abs_diff == 0.0`; ordered top-5
  `[795, 908, 667, 701, 442]`, with top-1 class 795 (`ski`).
- ONNX Runtime: raw-logit maximum absolute difference
  `5.245208740234375e-06`; probability maximum absolute difference
  `2.384185791015625e-07`; identical top-1.
- TorchScript: probability `max_abs_diff == 0.0`; identical top-1.
- OpenVINO: probability maximum absolute difference
  `0.0005454719066619873`, cosine similarity `0.9999995827674866`, and
  identical ordered top-5.
- TensorRT: probability maximum absolute difference
  `0.0004989802837371826`, cosine similarity `0.9999999403953552`, and
  identical ordered top-5.
