"""Dome-DETR postprocessing.

Dome-DETR emits the same set-prediction contract as D-FINE
(``pred_logits`` + ``pred_boxes`` in normalised ``cxcywh``), so the decode is
D-FINE's and is re-exported rather than duplicated. PAQI's density-adaptive
NMS runs *inside* the decoder on encoder proposals; the decoder output itself
is a set prediction and takes no further IoU suppression, which is why this
family is in the NMS-free backend allowlist.

The only Dome-specific detail is the query count: it varies per image, and for
batched inference the short rows are padded with a large negative logit by
``DomeTransformer._get_decoder_input``, so the shared top-k decode drops them
without any extra bookkeeping here.
"""

from .dfine import postprocess  # noqa: F401

__all__ = ["postprocess"]
