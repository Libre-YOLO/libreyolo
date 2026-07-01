"""Training configuration for native LibreRFDETRSO."""

from dataclasses import dataclass

from ..rfdetr.config import RFDETRConfig


@dataclass(kw_only=True)
class RFDETRSOConfig(RFDETRConfig):
    """RF-DETR-SO fine-tuning defaults.

    Same recipe as stock RF-DETR, with a one-epoch linear warmup: the
    freshly initialized SSA/PBM modules and the replicated deformable
    attention weights benefit from a gentle start.
    """

    warmup_epochs: int = 1
    name: str = "rfdetr_so_exp"
