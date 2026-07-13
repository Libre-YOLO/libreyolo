"""OWLv2 open-vocabulary detector adapter."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch

from .base import (
    _INSTALL_HINT,
    _contains_subsequence,
    _has_leading_article,
    _label_tokens,
    _prompt_label,
)
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
            name = _prompt_label(self.names[class_id])
            if _has_leading_article(name):
                labels.append(f"a photo of {name}")
            else:
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
        return self._detections_to_dict(
            boxes,
            scores,
            class_ids,
            conf_thres=conf_thres,
            original_size=original_size,
            max_det=max_det,
            classes=kwargs.get("classes"),
            iou_thres=iou_thres,
        )

    def _labels_to_class_ids(self, labels: Any) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            raw = labels.detach().cpu().reshape(-1)
            if raw.dtype == torch.bool:
                return torch.full((raw.numel(),), -1, dtype=torch.int64)
            try:
                numeric = raw.to(torch.float64)
            except (TypeError, RuntimeError):
                return torch.full((raw.numel(),), -1, dtype=torch.int64)
        else:
            if isinstance(labels, (str, bytes)):
                values = [labels]
            else:
                try:
                    values = list(labels)
                except TypeError:
                    values = [labels]
            if not values:
                return torch.zeros((0,), dtype=torch.int64)
            if all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
                numeric = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
            else:
                mapped = [self._text_label_to_class_id(str(value)) for value in values]
                return torch.as_tensor(
                    [-1 if class_id is None else class_id for class_id in mapped],
                    dtype=torch.int64,
                )
        integral = torch.isfinite(numeric) & (numeric == numeric.round())
        in_range = (numeric >= 0) & (numeric < len(self.names))
        valid = integral & in_range
        class_ids = torch.full(numeric.shape, -1, dtype=torch.int64)
        class_ids[valid] = numeric[valid].to(torch.int64)
        return class_ids

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
