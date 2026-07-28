"""FeyNobg network: BiRefNet's architecture with stage 3 deepened to 24 blocks.

FeyNobg (https://github.com/feyninc/nobg, Apache-2.0) keeps BiRefNet's forward
path bit-for-bit and only widens capacity: the third Swin-L stage grows from 18
to 24 blocks (222M -> 263M parameters). The module graph and state-dict key
schema are therefore identical to ``libreyolo/models/birefnet/nn.py``, which
this file reuses with a family-local dimension table instead of duplicating the
Swin tower and bilateral-reference decoder.
"""

from __future__ import annotations

from ..birefnet.nn import BiRefNetDims, LibreBiRefNetModel

FEYNOBG_DIMS = {
    # Single released variant: Swin-L tier, depths (2, 2, 24, 2).
    "l": BiRefNetDims(192, (2, 2, 24, 2), (6, 12, 24, 48), 12, (1536, 768, 384, 192)),
}


class LibreFeyNobgModel(LibreBiRefNetModel):
    """FeyNobg: image -> single-channel logit map (sigmoid -> alpha matte)."""

    def __init__(self, size: str = "l"):
        if size not in FEYNOBG_DIMS:
            raise ValueError(
                f"Unknown FeyNobg size {size!r}; expected one of {sorted(FEYNOBG_DIMS)}"
            )
        super().__init__(size=size, dims=FEYNOBG_DIMS[size])


__all__ = ["FEYNOBG_DIMS", "LibreFeyNobgModel"]
