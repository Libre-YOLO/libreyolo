"""Native OMDet-Turbo real-time open-vocabulary detector (parity-verified).

This native graph is export groundwork and is not used by the public
``LibreOpenVocab.predict()`` path, which delegates to ``transformers``. It
composes the native Swin-T backbone, a native CLIP text tower, an RT-DETR
hybrid encoder, and a deformable decoder with the Efficient Fusion Head.
"""

from .nn import OmDetTurboDetectionModel

__all__ = ["OmDetTurboDetectionModel"]
