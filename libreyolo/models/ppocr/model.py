"""LibrePPOCR: two-stage text detection + recognition (ocr task).

PP-OCRv5 (PaddleOCR 3.0, Apache-2.0) ported to PyTorch. One composite
checkpoint per tier bundles both submodels under ``det.*`` and ``rec.*``
state-dict namespaces plus the recognition charset and pipeline defaults in
the checkpoint metadata (see ``docs/checkpoint_schema.md``):

- ``t`` = PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec (CPU tier)
- ``l`` = PP-OCRv5_server_det + PP-OCRv5_server_rec (quality tier)

``LibreYOLO("LibrePPOCRl-ocr.pt")(image)`` returns ``Results.ocr``: located
text quads with transcripts covering Simplified/Traditional Chinese, English,
Japanese, and pinyin with one dictionary and one model.

Inference and validation only in v1: training/fine-tuning and export are out
of scope (export raises a clear error; the pipeline is two networks with
dynamic shapes). Document-orientation classification, image unwarping, and
the 0/180 textline rotator are optional upstream pipeline components that are
also out of scope; the checkpoint metadata reserves a ``components`` mapping
so they can be added without a schema break.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import torch
import torch.nn as nn

from ..base import BaseModel
from .det import PPOCRDetModel
from .rec import PPOCRRecModel

# CTC alphabet size of the PP-OCRv5 dictionary: 1 blank + 18383 dictionary
# entries + 1 space. The exact charset ships in the checkpoint metadata; this
# constant only sizes the head before weights load.
PPOCR_V5_NUM_CLASSES = 18385

_DEFAULT_PIPELINE: Dict[str, Any] = {
    "det_limit_side_len": 960,
    "det_db_thresh": 0.3,
    "det_db_box_thresh": 0.6,
    "det_db_unclip_ratio": 1.5,
}


class LibrePPOCRModel(nn.Module):
    """Composite module: ``det`` and ``rec`` submodels moving together."""

    def __init__(self, size: str, num_classes: int = PPOCR_V5_NUM_CLASSES):
        super().__init__()
        self.det = PPOCRDetModel(size)
        self.rec = PPOCRRecModel(size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "LibrePPOCRModel is a two-network pipeline; call .det(x) or .rec(x), "
            "or run the full pipeline through LibrePPOCR.predict()."
        )


class LibrePPOCR(BaseModel):
    """PP-OCRv5 text detection + recognition pipeline."""

    FAMILY = "ppocr"
    FILENAME_PREFIX = "LibrePPOCR"
    WEIGHT_EXT = ".pt"
    # The size value is the detection long-side limit (DetResizeForTest).
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"t": 960, "l": 960}
    SUPPORTED_TASKS = ("ocr",)
    DEFAULT_TASK = "ocr"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None  # inference + val only in v1
    # Image-level batching would multiply the two-stage pipeline complexity;
    # list inputs are handled sequentially by the family runner.
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False

    _UPSTREAM_URL = "https://github.com/PaddlePaddle/PaddleOCR"

    # ====================================================================
    # Checkpoint detection
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # The composite layout is unique to this family: a det.* DB tower and
        # a rec.* CTC head in one flat state dict.
        has_det = any(k.startswith("det.head.binarize.") for k in weights_dict)
        has_rec = "rec.head.ctc_head.fc.weight" in weights_dict
        return has_det and has_rec and cls.detect_size(weights_dict) is not None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        if "det.backbone.conv1.conv.weight" in weights_dict:
            return "t"  # PP-LCNetV3 stem
        if "det.backbone.stem.stem1.conv.weight" in weights_dict:
            return "l"  # PP-HGNetV2 stem
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # OCR has no detection classes; the single slot is a schema placeholder.
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "ocr" if cls.can_load(state_dict) else None

    @classmethod
    def format_weight_filename(cls, size_code: str) -> str:
        # REQUIRE_TASK_SUFFIX: the canonical filename always carries -ocr, so
        # the CLI name "ppocr-t" must resolve to LibrePPOCRt-ocr.pt.
        return f"{cls.FILENAME_PREFIX}{size_code}-ocr{cls.WEIGHT_EXT}"

    # ====================================================================
    # Construction
    # ====================================================================

    def __init__(
        self,
        model_path=None,
        size: str = "l",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        self.charset: Optional[List[str]] = None
        self.pipeline_config: Dict[str, Any] = dict(_DEFAULT_PIPELINE)
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,  # always 1 ("text"); ignore any caller-provided value
            device=device,
            task=task,
            **kwargs,
        )
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.nb_classes = 1
        self.names = {0: "text"}
        self.model.eval()

    def _init_model(self) -> nn.Module:
        return LibrePPOCRModel(size=self.size)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"det": self.model.det, "rec": self.model.rec}

    def _validate_loaded_state_dict_for_task(
        self,
        state_dict: dict,
        checkpoint: dict | None = None,
    ) -> None:
        # The composite checkpoint carries the recognition charset and the
        # pipeline defaults; capture them while the full wrapper is in hand.
        if isinstance(checkpoint, dict):
            charset = checkpoint.get("charset")
            if isinstance(charset, (list, tuple)) and charset:
                self.charset = [str(c) for c in charset]
            pipeline = checkpoint.get("pipeline")
            if isinstance(pipeline, dict):
                self.pipeline_config = {**_DEFAULT_PIPELINE, **pipeline}
        fc = state_dict.get("rec.head.ctc_head.fc.weight")
        if fc is not None and self.charset is not None:
            expected = len(self.charset)
            if int(fc.shape[0]) != expected:
                raise RuntimeError(
                    f"Checkpoint CTC head emits {int(fc.shape[0])} classes but its "
                    f"charset metadata lists {expected} entries; the checkpoint "
                    "is inconsistent."
                )
        return None

    def _rebuild_for_checkpoint_classes(self, new_nb_classes: int, state_dict: dict):
        # nc is always 1 for ocr; never rebuild the composite from class count.
        self.nb_classes = 1

    # ====================================================================
    # Inference surface (custom two-stage runner)
    # ====================================================================

    @property
    def _runner(self):
        if not hasattr(self, "_runner_instance") or self._runner_instance is None:
            from .inference import OCRInferenceRunner

            self._runner_instance = OCRInferenceRunner(self)
        return self._runner_instance

    @staticmethod
    def _get_preprocess_numpy():
        raise NotImplementedError(
            "LibrePPOCR does not use the detection-shaped preprocess hook; "
            "the two-stage pipeline lives in models/ppocr/inference.py."
        )

    def _preprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "LibrePPOCR does not use the detection-shaped _preprocess hook; "
            "the two-stage pipeline lives in models/ppocr/inference.py."
        )

    def _forward(self, *args, **kwargs):
        raise NotImplementedError(
            "LibrePPOCR does not use the detection-shaped _forward hook; "
            "the two-stage pipeline lives in models/ppocr/inference.py."
        )

    def _postprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "LibrePPOCR does not use the detection-shaped _postprocess hook; "
            "the two-stage pipeline lives in models/ppocr/inference.py."
        )

    # ====================================================================
    # Out of scope (v1)
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training/fine-tuning LibrePPOCR is not wired in this release "
            "(ocr v1 is inference + val only). The upstream repo ships "
            "Apache-2.0 training code; fine-tune there and convert the result "
            f"with weights/convert_ppocr_weights.py. Upstream: {self._UPSTREAM_URL}"
        )


__all__ = ["LibrePPOCR", "LibrePPOCRModel", "PPOCR_V5_NUM_CLASSES"]
