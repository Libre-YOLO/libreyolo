"""Native Swin Transformer backbone (timm-parity, max_abs_diff == 0).

Shared vision backbone for the OMDet-Turbo and Grounding DINO native ports.
"""

from .nn import SwinBackbone, SwinDims

__all__ = ["SwinBackbone", "SwinDims"]
