"""Native OV-DEIM architecture modules (vendored, inference-only).

Ported from OV-DEIM (https://github.com/wleilei/OV-DEIM), Apache-2.0, at
commit dfbf3946. The encoder/decoder lineage is RT-DETR (Apache-2.0,
Copyright (c) 2023 lyuwenyu) via DEIMv2 (Apache-2.0); the backbone adapter is
reused from ``libreyolo.models.deimv2``. Training-only code paths (denoising,
criterion, GridSynthetic augmentation) are intentionally not ported.
"""

from .nn import OVDEIM_SIZE_CONFIGS, LibreOVDEIMModel

__all__ = ["LibreOVDEIMModel", "OVDEIM_SIZE_CONFIGS"]
