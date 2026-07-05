"""Native OWLv2 open-vocabulary detector model.

Clean-room port of the Apache-2.0 OWLv2 architecture (parity-verified against
``transformers``). Consumed by the ``LibreOpenVocab`` tier, which supplies the
open-vocabulary ``set_classes``/``predict`` surface.
"""

from .nn import OWLV2_DIMS, Owlv2DetectionModel, Owlv2Dims

__all__ = ["Owlv2DetectionModel", "Owlv2Dims", "OWLV2_DIMS"]
