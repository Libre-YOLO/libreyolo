"""CUDA-graph training capture for the encoder-decoder detector families.

D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4 and EC all share one
network shape::

    feats = backbone(images)
    feats = encoder(feats)
    out   = decoder(feats, targets=...)

Only the first two stages can be captured. The decoder reads the ground
truth to build contrastive-denoising queries, and the number of those
queries is derived from the largest ground-truth count in the batch, so the
decoder's token count changes from batch to batch — the one thing a CUDA
graph cannot tolerate. The Hungarian criterion after it is host-side by
nature.

So the split is backbone + encoder inside the graph, decoder + criterion
eager. That is roughly a fifth to a quarter of the step for these families:
the win is real at small batch sizes, where the backbone is launch-bound,
and fades at large ones, where it is compute-bound.

Neither half is reimplemented here, on purpose. The capturable half runs the
family's *own* ``forward`` with the decoder stubbed out, so per-family
details (D-FINE splitting a low-level feature map off the backbone before
the encoder, EC handing an encoder map to a mask head) are honoured without
this module knowing about them. The eager half swaps ``forward`` for one
that resumes at the decoder and then calls the family's own ``on_forward``,
so target conversion and loss aggregation stay in one place per family and
cannot drift from the eager path.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple

import torch
from torch import nn

__all__ = ["DETREncoderCudaGraphMixin"]


@contextmanager
def _patched_forward(module: nn.Module, replacement) -> Iterator[None]:
    """Swap ``module.forward`` for ``replacement`` inside the block.

    Patched on the instance, so the class is untouched and other models of
    the same family in this process are unaffected. Removed rather than
    reassigned on exit, or the instance would keep a shadow of the class
    forward forever. ``nn.Module.__call__`` reads ``self.forward``, so hooks
    and autograd behave exactly as they do for the real method.
    """
    had_own = "forward" in module.__dict__
    original = module.__dict__.get("forward")
    module.forward = replacement
    try:
        yield
    finally:
        if had_own and original is not None:
            module.forward = original
        else:
            module.__dict__.pop("forward", None)


class _BackboneEncoder(nn.Module):
    """The capturable half: everything the family's forward does before the
    decoder.

    Rather than re-deriving that prefix, this runs the real forward with the
    decoder stubbed out and keeps whatever the model handed it. The decoder
    call's positional ``feats`` become the graph's outputs; so do any tensor
    keyword arguments, unless the model passed one of the feats itself (EC
    hands ``feats[0]`` to its mask head), in which case the position is
    recorded and the tensor is not duplicated in the output tuple.

    The recorded layout is fixed after the first call. That is safe because
    capture warm-up runs this eagerly before recording, and the model's
    forward takes the same branch for every batch at a fixed input shape.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # (kwarg name, ("feat", index) | ("extra", index)), set on first call.
        self.kwarg_layout: List[Tuple[str, Tuple[str, int]]] = []
        self.num_feats: int = 0

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        grabbed: Dict[str, Any] = {}

        def stub(feats, targets=None, **kwargs):
            grabbed["feats"] = list(feats)
            grabbed["kwargs"] = dict(kwargs)
            # The real return value is discarded: the caller only wants what
            # reached the decoder. An empty tuple keeps every family's
            # ``return self.decoder(...)`` valid.
            return ()

        with _patched_forward(self.model.decoder, stub):
            self.model(imgs, targets=None)

        feats = grabbed["feats"]
        kwargs = grabbed["kwargs"]
        extras: List[torch.Tensor] = []
        layout: List[Tuple[str, Tuple[str, int]]] = []
        for name, value in kwargs.items():
            if not torch.is_tensor(value):
                continue
            shared = next(
                (i for i, f in enumerate(feats) if f is value),
                None,
            )
            if shared is not None:
                layout.append((name, ("feat", shared)))
            else:
                layout.append((name, ("extra", len(extras))))
                extras.append(value)
        self.kwarg_layout = layout
        self.num_feats = len(feats)
        return (*feats, *extras)

    def split(self, flat: List[torch.Tensor]):
        """Undo :meth:`forward`'s packing: (feats, decoder kwargs)."""
        feats = flat[: self.num_feats]
        extras = flat[self.num_feats :]
        kwargs = {
            name: (feats[index] if kind == "feat" else extras[index])
            for name, (kind, index) in self.kwarg_layout
        }
        return feats, kwargs


class DETREncoderCudaGraphMixin:
    """Provide ``cuda_graph_train_spec`` for the encoder-decoder detectors.

    Mixed in ahead of ``BaseTrainer`` so it overrides the base hook. Families
    whose model lacks the ``backbone``/``encoder``/``decoder`` trio, or which
    run a non-detect task, fall back to eager.
    """

    # Tasks whose loss reaches past this boundary in ways the split does not
    # cover (segmentation feeds an encoder map to a mask head and carries
    # mask targets, pose adds keypoint targets). Detect only, for now.
    _CUDA_GRAPH_TASKS: Tuple[str, ...] = ("detect",)

    def cuda_graph_train_spec(self):
        from libreyolo.training.cuda_graph import (
            CudaGraphTrainSpec,
            GraphableNetwork,
        )

        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        # getattr default keeps partially-constructed trainers (test doubles,
        # subclasses skipping the mixin's own __init__ path) on the eager
        # route rather than raising out of the spec resolution.
        if task not in getattr(self, "_CUDA_GRAPH_TASKS", ("detect",)):
            return None
        model = getattr(self, "model", None)
        raw = getattr(model, "module", model)
        if not isinstance(raw, nn.Module):
            return None
        if not all(hasattr(raw, name) for name in ("backbone", "encoder", "decoder")):
            return None
        if getattr(self, "criterion", None) is None:
            return None

        adapter = _BackboneEncoder(raw)
        network = GraphableNetwork(adapter)

        def assemble(flat, imgs, targets, polygons=None):
            feats, decoder_kwargs = adapter.split(list(network.rebuild(flat)))

            def resume(_imgs, targets=None, **kwargs):
                kwargs.update(decoder_kwargs)
                return raw.decoder(feats, targets=targets, **kwargs)

            with _patched_forward(raw, resume):
                return self.on_forward(imgs, targets, polygons=polygons)

        return CudaGraphTrainSpec(network=network, assemble=assemble)
