"""OWLv2 open-vocabulary detector adapter."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import torch

from .base import _INSTALL_HINT, _contains_subsequence, _label_tokens
from .base import LibreOpenVocabDetector


class LibreOWLv2(LibreOpenVocabDetector):
    """OWLv2 zero-shot object detector loaded through ``transformers``."""

    FAMILY = "owlv2"
    FILENAME_PREFIX = "LibreOWLv2"
    HF_REPOS: ClassVar[Dict[str, str]] = {
        "b16": "google/owlv2-base-patch16-ensemble",
        "l14": "google/owlv2-large-patch14-ensemble",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"b16": 960, "l14": 960}
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

    def _labels_to_class_ids(self, labels: Any) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            class_ids = labels.detach().cpu().long().reshape(-1)
        else:
            values = list(labels)
            if not values:
                return torch.zeros((0,), dtype=torch.int64)
            if all(isinstance(value, (int, float)) for value in values):
                class_ids = torch.as_tensor(values, dtype=torch.int64).reshape(-1)
            else:
                mapped = [self._text_label_to_class_id(str(value)) for value in values]
                return torch.as_tensor(
                    [-1 if class_id is None else class_id for class_id in mapped],
                    dtype=torch.int64,
                )
        valid = (class_ids >= 0) & (class_ids < len(self.names))
        return torch.where(valid, class_ids, torch.full_like(class_ids, -1))

    def _text_label_to_class_id(self, text: str) -> Optional[int]:
        phrase = _label_tokens(text)
        matches = []
        for class_id, name in self.names.items():
            label = _label_tokens(name)
            if phrase == label:
                return class_id
            if _contains_subsequence(phrase, label) or _contains_subsequence(
                label, phrase
            ):
                matches.append(class_id)
        return matches[0] if len(matches) == 1 else None


__all__ = ["LibreOWLv2"]
