"""Grounding DINO open-vocabulary detector adapter."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import torch

from .base import (
    _INSTALL_HINT,
    _contains_subsequence,
    _label_tokens,
    _normalize_label,
    LibreOpenVocabDetector,
)


class LibreGroundingDINO(LibreOpenVocabDetector):
    """Grounding DINO detector loaded through ``transformers``."""

    FAMILY = "grounding_dino"
    FILENAME_PREFIX = "LibreGroundingDINO"
    HF_REPOS: ClassVar[Dict[str, str]] = {
        "t": "LibreYOLO/LibreGroundingDINOt",
        "b": "LibreYOLO/LibreGroundingDINOb",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"t": 800, "b": 800}
    DEFAULT_CONF: ClassVar[float] = 0.25
    DEFAULT_TEXT_THRESHOLD: ClassVar[float] = 0.25
    SUPPORTS_TEXT_THRESHOLD: ClassVar[bool] = True

    def _load_pretrained(self, snapshot_dir: str):
        try:
            from transformers import AutoProcessor, GroundingDinoForObjectDetection
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        model = GroundingDinoForObjectDetection.from_pretrained(
            snapshot_dir, dtype=self._resolve_dtype()
        )
        processor = AutoProcessor.from_pretrained(snapshot_dir)
        return model, processor

    def _prompt(self) -> str:
        phrases = []
        for class_id in range(len(self.names)):
            name = str(self.names[class_id]).strip().lower()
            phrases.append(f"a {name}")
        return ". ".join(phrases) + "."

    def _build_inputs(self, img: Any) -> Any:
        return self.processor(
            images=img,
            text=self._prompt(),
            return_tensors="pt",
        )

    def _forward(self, inputs: Any) -> Dict[str, Any]:
        inputs = self._prepare_inputs(inputs)
        input_ids = inputs.get("input_ids")
        return {"outputs": self.model(**inputs), "input_ids": input_ids}

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
            output["outputs"],
            output["input_ids"],
            threshold=float(conf_thres),
            text_threshold=float(self._text_threshold),
            target_sizes=[(height, width)],
        )
        result = results[0]
        boxes = torch.as_tensor(result.get("boxes", []), dtype=torch.float32).reshape(
            -1, 4
        )
        scores = torch.as_tensor(result.get("scores", []), dtype=torch.float32).reshape(
            -1
        )
        phrases = self._read_text_labels(result)
        class_ids = []
        keep_indices = []
        for index, phrase in enumerate(phrases[: min(len(phrases), boxes.shape[0])]):
            class_id = self._phrase_to_class_id(phrase)
            if class_id is not None:
                keep_indices.append(index)
                class_ids.append(class_id)
        if not keep_indices:
            return self._empty_detections()
        keep = torch.as_tensor(keep_indices, dtype=torch.long)
        return self._detections_to_dict(
            boxes[keep],
            scores[keep],
            torch.as_tensor(class_ids, dtype=torch.int64),
            conf_thres=conf_thres,
            original_size=original_size,
            max_det=max_det,
            classes=kwargs.get("classes"),
        )

    @staticmethod
    def _read_text_labels(result: Dict[str, Any]) -> list[str]:
        labels = result.get("text_labels")
        if labels is None:
            labels = result.get("labels", [])
        if isinstance(labels, torch.Tensor):
            return []
        return [str(label) for label in labels]

    def _phrase_to_class_id(self, phrase: str) -> Optional[int]:
        norm = _normalize_label(phrase)
        if not norm:
            return None
        exact = self._name_to_id.get(norm)
        if exact is not None:
            return exact

        phrase_tokens = _label_tokens(phrase)
        matches = []
        for class_id, name in self.names.items():
            label_tokens = _label_tokens(name)
            if _contains_subsequence(phrase_tokens, label_tokens) or (
                _contains_subsequence(label_tokens, phrase_tokens)
            ):
                matches.append(class_id)
        return matches[0] if len(matches) == 1 else None


__all__ = ["LibreGroundingDINO"]
