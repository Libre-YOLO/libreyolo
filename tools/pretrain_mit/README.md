# MiT encoder pretraining (throwaway tooling)

`LibreSegformer` (`libreyolo/models/segformer/`) trains fully from scratch —
there is no permissively-licensed pretrained MiT/SegFormer encoder anywhere
(the upstream `nvidia/mit-b0..b5` and `nvidia/segformer-*` checkpoints are all
under NVIDIA's non-commercial license; see `libreyolo/models/segformer/
NOTICE`). This directory pretrains the encoder ourselves on ImageNet-1K
classification — the same recipe the original SegFormer paper used before its
ADE20K fine-tune — using only our own native, Apache-2.0-derived encoder code.

**This is not part of the LibreYOLO library.** Nothing under `libreyolo/`
imports from here, and no permanent `classify` task was added to
`LibreSegformer` to support it — that would mean carrying ImageNet
classification training machinery in the library forever for a pipeline that
runs once per size. Everything in this directory can be deleted once its
encoder checkpoints are on disk; the only trace it leaves in the library is
one generic constructor kwarg, `LibreSegformer(..., pretrained_encoder=...)`,
documented below.

## Why classification, not self-supervised pretraining

Self-supervised methods (MAE, SimMIM, DINO) can match or beat supervised
ImageNet pretraining for dense-prediction transfer, but that evidence is for
architectures with an established SSL recipe — plain single-scale ViT (MAE)
or Swin-style windowed attention (SimMIM). MiT is neither: a 4-stage
hierarchical encoder with spatial-reduction attention. There's no published
SSL recipe for this shape, so building one means inventing the masking
strategy, loss placement, and schedule from scratch with nothing to validate
against. Classification pretraining is what actually produced the paper's own
numbers for this encoder — the one approach where success is expected, not a
research bet.

## Usage

Validate the whole pipeline cheaply first, before committing to a full
ImageNet-1K run — `imagenette160` is a small, already-integrated 10-class
proxy dataset (see `libreyolo/data/classify_dataset.py`):

```bash
python tools/pretrain_mit/train_classify.py --size b0 --data imagenette160 --epochs 5
```

One size, full ImageNet-1K (point `--data` at a local `ImageFolder` root with
`train/` and `val/` sub-directories — ImageNet-1K itself is not
auto-downloadable, you must obtain it separately):

```bash
python tools/pretrain_mit/train_classify.py --size b0 --data /data/imagenet
```

All six sizes, smallest to largest:

```bash
python tools/pretrain_mit/pretrain_all_sizes.py --data /data/imagenet
```

Cost note: the default recipe (300 epochs, DeiT-style AdamW + cosine +
RandAugment/MixUp/CutMix) is a multi-day job per size on one GPU. Shorten
`--epochs` if you want a partial-pretraining starting point instead — even an
imperfectly-converged encoder should beat random init.

## Feeding the result back into LibreSegformer

Each size writes `runs/pretrain_mit/mit_<size>_imagenet1k_encoder.pt`
(overwritten on every new best-val-top1 epoch) in this format:

```python
{"encoder": <SegformerEncoder.state_dict()>, "size": "b0", "source": "imagenet1k-classify"}
```

Load it when constructing a fresh `LibreSegformer` for fine-tuning (only the
encoder is populated; the decode head stays at random init):

```python
from libreyolo import LibreSegformer

model = LibreSegformer(
    size="b0",
    task="semantic",
    pretrained_encoder="runs/pretrain_mit/mit_b0_imagenet1k_encoder.pt",
)
model.train(data="ade20k", epochs=160, ...)
```

Or via `scripts/train_segformer_all_sizes.py --pretrained-encoders-dir
runs/pretrain_mit`, which looks up the matching `mit_<size>_imagenet1k_
encoder.pt` per size automatically.
