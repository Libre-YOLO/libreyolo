"""OWLv2 open-vocabulary detector adapter."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Tuple

import torch

from .base import _INSTALL_HINT
from .base import LibreOpenVocabDetector


class LibreOWLv2(LibreOpenVocabDetector):
    """OWLv2 zero-shot object detector loaded through ``transformers``."""

    FAMILY = "owlv2"
    FILENAME_PREFIX = "LibreOWLv2"
    HF_REPOS: ClassVar[Dict[str, str]] = {
        "b16": "LibreYOLO/LibreOWLv2b16",
        "l14": "LibreYOLO/LibreOWLv2l14",
    }
    # Informational only: the HF processor owns resizing and predict(imgsz=...)
    # is rejected by the open-vocab base. Values mirror the published configs.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"b16": 960, "l14": 1008}
    DEFAULT_CONF: ClassVar[float] = 0.1
    PROMPT_TEMPLATE: ClassVar[str] = "a photo of a {}"

    def _load_pretrained(self, snapshot_dir: str):
        try:
            from transformers import AutoProcessor, Owlv2ForObjectDetection
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        model = Owlv2ForObjectDetection.from_pretrained(
            snapshot_dir, dtype=self._resolve_dtype()
        )
        processor = AutoProcessor.from_pretrained(snapshot_dir)
        return model, processor

    def _text_labels(self) -> list[list[str]]:
        labels = []
        for class_id in range(len(self.names)):
            name = str(self.names[class_id]).strip().lower()
            labels.append(self.PROMPT_TEMPLATE.format(name))
        return [labels]

    def _build_inputs(self, img: Any) -> Any:
        return self.processor(
            text=self._text_labels(),
            images=img,
            return_tensors="pt",
        )

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        width, height = original_size
        results = self.processor.post_process_grounded_object_detection(
            output,
            threshold=float(conf_thres),
            target_sizes=[(height, width)],
            text_labels=self._text_labels(),
        )
        result = results[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        raw_labels = result.get("labels")
        if raw_labels is None:
            raw_labels = result.get("text_labels", [])

        class_ids = self._labels_to_class_ids(raw_labels)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        scores_t = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
        keep = class_ids >= 0
        n = min(boxes_t.shape[0], scores_t.shape[0], class_ids.shape[0])
        boxes_t, scores_t, class_ids, keep = (
            boxes_t[:n],
            scores_t[:n],
            class_ids[:n],
            keep[:n],
        )
        return self._detections_to_dict(
            boxes_t[keep],
            scores_t[keep],
            class_ids[keep],
            conf_thres=conf_thres,
            original_size=original_size,
            max_det=max_det,
            classes=kwargs.get("classes"),
        )


__all__ = ["LibreOWLv2"]
