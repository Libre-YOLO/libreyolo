"""Native BERT text encoder (parity with transformers BertModel).

Grounding DINO's text backbone.
"""

from .nn import BertModel

__all__ = ["BertModel"]
