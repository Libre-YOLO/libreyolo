"""FeyNobg postprocess (matte task).

FeyNobg shares BiRefNet's output contract exactly: a single-channel logit map
that sigmoids to a soft alpha matte and is resized back to the original canvas.
One module per family (ADR 0005); the implementation is delegated.
"""

from __future__ import annotations

from .birefnet import postprocess

__all__ = ["postprocess"]
