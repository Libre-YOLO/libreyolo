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

Inference and validation are available, and Core ML export packages the
detector and recognizer as two bounded-flexible named functions while keeping
cropping and decoding on the host. Other single-graph export formats remain
out of scope. Document-orientation classification, image unwarping, and the
0/180 textline rotator are optional upstream pipeline components that are also
out of scope; the checkpoint metadata reserves a ``components`` mapping so
they can be added without a schema break.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import torch
import torch.nn as nn

from ..base import BaseModel
from .det import PPOCRDetModel
from .rec import PPOCRRecModel

logger = logging.getLogger(__name__)

# CTC alphabet size of the PP-OCRv5 dictionary: 1 blank + 18383 dictionary
# entries + 1 space. The exact charset ships in the checkpoint metadata; this
# constant only sizes the head before weights load.
PPOCR_V5_NUM_CLASSES = 18385

_DEFAULT_PIPELINE: Dict[str, Any] = {
    "det_limit_side_len": 960,
    "det_db_thresh": 0.3,
    "det_db_box_thresh": 0.6,
    "det_db_unclip_ratio": 1.5,
    "rec_image_shape": [3, 48, 320],
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
        self.components_config: Dict[str, Any] = {}
        self._ocr_checkpoint_metadata_errors: tuple[str, ...] = (
            "no LibrePPOCR checkpoint OCR metadata has been loaded",
        )
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
        """Capture OCR extras without making legacy checkpoints unloadable.

        The base loader deliberately accepts legacy metadata. PP-OCR keeps that
        compatibility for native inference, but records every missing or
        malformed OCR-specific field so an export cannot turn defaults or
        coerced values into apparently authoritative artifact metadata.
        """
        from ...export.coreml_ppocr import (
            validate_ppocr_charset,
            validate_ppocr_pipeline_config,
        )

        self.charset = None
        self.pipeline_config = dict(_DEFAULT_PIPELINE)
        self.components_config = {}
        metadata_errors: list[str] = []

        fc = state_dict.get("rec.head.ctc_head.fc.weight")
        rec_num_classes = int(fc.shape[0]) if fc is not None else None

        if not isinstance(checkpoint, dict):
            metadata_errors.append("checkpoint wrapper is not a mapping")
        else:
            if "charset" not in checkpoint:
                metadata_errors.append("missing required OCR metadata key 'charset'")
            else:
                try:
                    self.charset = validate_ppocr_charset(checkpoint["charset"])
                except ValueError as exc:
                    metadata_errors.append(f"invalid charset: {exc}")

            if "pipeline" not in checkpoint:
                metadata_errors.append("missing required OCR metadata key 'pipeline'")
            else:
                raw_pipeline = checkpoint["pipeline"]
                # Preserve the old native-inference compatibility behavior for
                # dict-shaped partial metadata, but do not certify it for export.
                if isinstance(raw_pipeline, dict):
                    self.pipeline_config = {**_DEFAULT_PIPELINE, **raw_pipeline}
                try:
                    self.pipeline_config = validate_ppocr_pipeline_config(raw_pipeline)
                except ValueError as exc:
                    metadata_errors.append(f"invalid pipeline: {exc}")

            if "components" not in checkpoint:
                metadata_errors.append("missing required OCR metadata key 'components'")
            else:
                raw_components = checkpoint["components"]
                if not isinstance(raw_components, dict):
                    metadata_errors.append(
                        "invalid components: LibrePPOCR components metadata "
                        "must be a dict"
                    )
                else:
                    self.components_config = dict(raw_components)
                    if self.components_config:
                        metadata_errors.append(
                            "unsupported components: Core ML export does not "
                            "package optional PP-OCR pipeline stages "
                            f"{sorted(map(str, self.components_config))}"
                        )

        if rec_num_classes is not None and self.charset is not None:
            expected = len(self.charset)
            if rec_num_classes != expected:
                raise RuntimeError(
                    f"Checkpoint CTC head emits {rec_num_classes} classes but its "
                    f"charset metadata lists {expected} entries; the checkpoint "
                    "is inconsistent."
                )

        self._ocr_checkpoint_metadata_errors = tuple(metadata_errors)
        if metadata_errors:
            logger.warning(
                "LibrePPOCR checkpoint OCR metadata is incomplete or invalid: %s. "
                "Native load compatibility is retained, but Core ML export is "
                "disabled until a schema-compliant checkpoint is loaded.",
                "; ".join(metadata_errors),
            )
        return None

    def _require_complete_ocr_metadata_for_export(self) -> None:
        """Fail closed unless OCR metadata came intact from the checkpoint."""
        if self._ocr_checkpoint_metadata_errors:
            raise RuntimeError(
                "LibrePPOCR Core ML export requires complete, schema-compliant "
                "checkpoint OCR metadata: "
                + "; ".join(self._ocr_checkpoint_metadata_errors)
            )

        from ...export.coreml_ppocr import (
            validate_ppocr_charset,
            validate_ppocr_pipeline_config,
        )

        rec_head = getattr(
            getattr(getattr(self.model, "rec", None), "head", None),
            "ctc_head",
            None,
        )
        fc = getattr(rec_head, "fc", None)
        rec_num_classes = getattr(fc, "out_features", None)
        if rec_num_classes is None:
            weight = getattr(fc, "weight", None)
            if torch.is_tensor(weight):
                rec_num_classes = int(weight.shape[0])
        if rec_num_classes is None:
            raise RuntimeError(
                "LibrePPOCR Core ML export could not derive the CTC class count "
                "from rec.head.ctc_head.fc."
            )

        try:
            validate_ppocr_charset(
                self.charset,
                rec_num_classes=int(rec_num_classes),
            )
            validate_ppocr_pipeline_config(self.pipeline_config)
        except ValueError as exc:
            raise RuntimeError(
                "LibrePPOCR Core ML export metadata was modified or is invalid: "
                f"{exc}"
            ) from exc
        if not isinstance(self.components_config, dict):
            raise RuntimeError(
                "LibrePPOCR Core ML export requires components metadata to be a dict."
            )
        if self.components_config:
            raise RuntimeError(
                "LibrePPOCR Core ML export does not package optional PP-OCR "
                f"pipeline stages {sorted(map(str, self.components_config))}."
            )

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
    # Training remains out of scope
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training/fine-tuning LibrePPOCR is not wired in this release "
            "(ocr v1 is inference + val only). The upstream repo ships "
            "Apache-2.0 training code; fine-tune there and convert the result "
            f"with weights/convert_ppocr_weights.py. Upstream: {self._UPSTREAM_URL}"
        )


__all__ = ["LibrePPOCR", "LibrePPOCRModel", "PPOCR_V5_NUM_CLASSES"]
