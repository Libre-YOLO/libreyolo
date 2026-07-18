"""PyTorch-native quantization for LibreYOLO models.

Public grammar::

    model = LibreYOLO("LibreYOLO9s.pt")
    qmodel = model.quantize(recipe="int8", calib="coco8.yaml", samples=128)
    qmodel.val(data="coco8.yaml")          # honest accuracy, same validators
    qmodel.train(data="coco8.yaml", ...)   # QAT (add distill_model=... for QAD)
    qmodel.save("LibreYOLO9s-int8.pt")     # manifest-carrying checkpoint
"""

from .api import (
    DEFAULT_CALIB_DATA,
    QUANT_SCHEMA_VERSION,
    QuantizationError,
    RECIPES,
    SUPPORTED_FAMILIES,
    apply_quant_structure,
    default_keep_high_precision,
    dequantize_model,
    export_finalized_pt,
    quant_info,
    quantize_model,
    reprepare_model,
)
from .modules import NVFP4Linear, QuantConv2d, QuantLinear

__all__ = [
    "DEFAULT_CALIB_DATA",
    "QUANT_SCHEMA_VERSION",
    "QuantizationError",
    "RECIPES",
    "SUPPORTED_FAMILIES",
    "apply_quant_structure",
    "default_keep_high_precision",
    "dequantize_model",
    "export_finalized_pt",
    "quant_info",
    "quantize_model",
    "reprepare_model",
    "NVFP4Linear",
    "QuantConv2d",
    "QuantLinear",
]
