"""Native OMDet-Turbo real-time open-vocabulary detector (parity-verified).

Consumed by the ``LibreOpenVocab`` tier. Composes the native Swin-T backbone,
a native CLIP text tower, an RT-DETR hybrid encoder, and a deformable decoder
with the Efficient Fusion Head.
"""

from .nn import OmDetTurboDetectionModel

__all__ = ["OmDetTurboDetectionModel"]
