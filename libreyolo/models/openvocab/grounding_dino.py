"""Grounding DINO open-vocabulary detector adapter."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import torch

from .base import (
    _INSTALL_HINT,
    LibreOpenVocabDetector,
    _contains_subsequence,
    _label_tokens,
    _normalize_label,
)


class _GroundingDinoInputChunks(list):
    """List-like payload that satisfies InferenceRunner's ``.to(device)`` call."""

    def to(self, device):
        for item in self:
            if hasattr(item, "to"):
                item.to(device)
        return self


class LibreGroundingDINO(LibreOpenVocabDetector):
    """Grounding DINO detector loaded through ``transformers``."""

    FAMILY = "grounding_dino"
    FILENAME_PREFIX = "LibreGroundingDINO"
    HF_REPOS: ClassVar[Dict[str, str]] = {
        "t": "LibreYOLO/LibreGroundingDINOt",
        "b": "LibreYOLO/LibreGroundingDINOb",
    }
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "t": "842ba2640b927b39c4c06a62e4d2238a9df965f1",
        "b": "2d2f9b44aa8740a3c6a5b84c5d739441f22b7494",
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

    def _class_phrase(self, class_id: int) -> str:
        name = str(self.names[class_id]).strip().lower()
        return f"a {name}"

    def _prompt(self, class_ids: list[int] | None = None) -> str:
        class_ids = list(range(len(self.names))) if class_ids is None else class_ids
        phrases = []
        for class_id in class_ids:
            phrases.append(self._class_phrase(class_id))
        return ". ".join(phrases) + "."

    def _max_text_len(self) -> int | None:
        config = getattr(self.model, "config", None)
        value = getattr(config, "max_text_len", None)
        return int(value) if value else None

    def _token_count(self, prompt: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return 0
        encoded = tokenizer(prompt, add_special_tokens=True, truncation=False)
        return len(encoded["input_ids"])

    def _prompt_chunks(self) -> list[str]:
        max_text_len = self._max_text_len()
        all_class_ids = list(range(len(self.names)))
        if (
            max_text_len is None
            or self._token_count(self._prompt(all_class_ids)) <= max_text_len
        ):
            return [self._prompt(all_class_ids)]

        chunks: list[str] = []
        current: list[int] = []
        for class_id in all_class_ids:
            candidate = current + [class_id]
            prompt = self._prompt(candidate)
            if self._token_count(prompt) <= max_text_len:
                current = candidate
                continue
            if not current:
                raise ValueError(
                    f"Grounding DINO prompt for class {self.names[class_id]!r} "
                    f"exceeds max_text_len={max_text_len}."
                )
            chunks.append(self._prompt(current))
            current = [class_id]
            if self._token_count(self._prompt(current)) > max_text_len:
                raise ValueError(
                    f"Grounding DINO prompt for class {self.names[class_id]!r} "
                    f"exceeds max_text_len={max_text_len}."
                )
        if current:
            chunks.append(self._prompt(current))
        return chunks

    def _build_inputs(self, img: Any) -> Any:
        chunks = self._prompt_chunks()
        inputs = _GroundingDinoInputChunks(
            self.processor(images=img, text=prompt, return_tensors="pt")
            for prompt in chunks
        )
        return inputs[0] if len(inputs) == 1 else inputs

    def _forward(self, inputs: Any) -> Dict[str, Any] | list[Dict[str, Any]]:
        if isinstance(inputs, list):
            return [self._forward(chunk) for chunk in inputs]
        inputs = self._prepare_inputs(inputs)
        input_ids = inputs.get("input_ids")
        return {"outputs": self.model(**inputs), "input_ids": input_ids}

    def _postprocess_chunk(
        self,
        output: Dict[str, Any],
        conf_thres: float,
        original_size: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
            )
        keep = torch.as_tensor(keep_indices, dtype=torch.long)
        return (
            boxes[keep],
            scores[keep],
            torch.as_tensor(class_ids, dtype=torch.int64),
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
        outputs = output if isinstance(output, list) else [output]
        chunks = [
            self._postprocess_chunk(chunk, conf_thres, original_size)
            for chunk in outputs
        ]
        boxes = torch.cat([chunk[0] for chunk in chunks], dim=0)
        scores = torch.cat([chunk[1] for chunk in chunks], dim=0)
        class_ids = torch.cat([chunk[2] for chunk in chunks], dim=0)
        if boxes.shape[0] == 0:
            return self._empty_detections()
        return self._detections_to_dict(
            boxes,
            scores,
            class_ids,
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

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export the current finite class vocabulary as an image-only graph."""
        if str(format).strip().lower() != "coreml":
            return super().export(format=format, **kwargs)
        raise NotImplementedError(
            "Grounding DINO Core ML export is disabled. Core ML Tools 9 "
            "FP16 and FP32 runtime numerics can reorder nearly tied encoder "
            "proposals before distinct learned rank queries, producing "
            "materially different detections on Apple Silicon."
        )


__all__ = ["LibreGroundingDINO"]
