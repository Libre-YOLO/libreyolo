"""Torch-free input preprocessing.

Mirrors ``libreyolo/postprocess/``: the family-specific numpy preprocessing
recipes live here rather than under ``libreyolo/models/``, because importing
anything under ``models/`` runs ``models/__init__.py``, which eagerly
constructs every model class to populate the ``can_load()`` registry. Those
classes are ``nn.Module`` subclasses and genuinely require torch, so a backend
that only needs a letterbox recipe would drag the whole torch wheel in with it.

Keeping these functions here is what lets ``backends/base.py`` (and therefore
``backends/onnx.py``) import without torch, so a lightweight deployment can run
ONNX inference with just numpy + onnxruntime.

``libreyolo/models/<family>/utils.py`` re-exports everything here, so existing
import paths keep working.

See https://github.com/LibreYOLO/libreyolo/discussions/711.
"""

from __future__ import annotations

import numpy as np

from ..utils.lazy import lazy_module, module_available

torch = lazy_module("torch")


def as_batched_input(img_chw: np.ndarray):
    """Add a batch dim, as a ``torch.Tensor`` if torch is installed.

    With torch present this is exactly ``torch.from_numpy(chw).unsqueeze(0)``,
    which is what every family preprocessor did before this module existed.
    Without torch it returns the equivalent ``(1, C, H, W)`` ndarray, which the
    ONNX backend feeds to onnxruntime directly (it converted the tensor back to
    numpy anyway).
    """
    if module_available("torch"):
        return torch.from_numpy(img_chw).unsqueeze(0)
    return img_chw[None, ...]


__all__ = ["as_batched_input"]
