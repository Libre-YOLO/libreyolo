"""ViT preprocessing placeholders, implemented after the parity gate."""

from __future__ import annotations


def preprocess_image(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError("ViT preprocessing lands after the upstream parity gate.")


def preprocess_numpy(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError("ViT preprocessing lands after the upstream parity gate.")
