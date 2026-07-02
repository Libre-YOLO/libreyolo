"""Native Grounding DINO open-vocabulary detector (parity-verified).

Composes the native Swin-T backbone (transformers-convention variant), native
BERT text encoder, a cross-modal fusion encoder, a deformable decoder with text
cross-attention, and a contrastive-embedding class head.
"""

from .nn import GroundingDinoDetectionModel

__all__ = ["GroundingDinoDetectionModel"]
