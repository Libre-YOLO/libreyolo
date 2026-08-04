"""Compatibility import for the original face-specific gallery path."""

from ...utils.gallery import Gallery, model_file_fingerprint

FaceGallery = Gallery

__all__ = ["Gallery", "FaceGallery", "model_file_fingerprint"]
