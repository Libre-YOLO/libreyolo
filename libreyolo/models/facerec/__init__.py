"""Face-embedding / facial-recognition task (inference only).

Two-stage, ONNX-backed: a face detector locates faces (+5 landmarks), each is
aligned to a canonical 112x112 crop, and a recognition head emits an
L2-normalized identity embedding. Verification is cosine similarity.
"""

from .align import ARCFACE_DST_112, align_face, estimate_norm
from .gallery import FaceGallery, Gallery
from .model import LibreFaceEmbedder, OpenCVFaceDetector
from .preprocess import PreprocCfg
from .weights import FACEREC_WEIGHT_URLS, resolve_facerec_weight

__all__ = [
    "LibreFaceEmbedder",
    "OpenCVFaceDetector",
    "FaceGallery",
    "Gallery",
    "PreprocCfg",
    "align_face",
    "estimate_norm",
    "ARCFACE_DST_112",
    "FACEREC_WEIGHT_URLS",
    "resolve_facerec_weight",
]
