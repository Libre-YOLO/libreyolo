"""
Utility functions for YOLOX.

YOLOX uses different preprocessing and postprocessing:
- Preprocessing: Letterbox with gray padding (114,114,114), NO normalization (0-255 range)
- Postprocessing: Box decoding with exp() for width/height, objectness score

Postprocessing lives in ``libreyolo.postprocess.yolox`` and is re-exported
here for backward compatibility.
"""

from __future__ import annotations

from ...utils.lazy import lazy_module


from ...preprocess.yolox import (  # noqa: F401  (moved; re-exported for backward compatibility)
    preprocess_image,
    preprocess_numpy,
)
from ...postprocess.yolox import (  # noqa: F401  (backward-compatible re-exports)
    decode_outputs,
    make_grids,
    postprocess,
)


# torch is resolved on first use so this module stays importable in a
# torch-free ONNX deployment (discussions/711).
torch = lazy_module("torch")
