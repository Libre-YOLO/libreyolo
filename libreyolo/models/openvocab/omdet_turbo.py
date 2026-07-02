"""OMDet-Turbo open-vocabulary detector adapter."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Tuple

import torch

from .base import _INSTALL_HINT
from .base import LibreOpenVocabDetector


class LibreOMDetTurbo(LibreOpenVocabDetector):
    """OMDet-Turbo real-time zero-shot detector loaded through ``transformers``.

    Unlike Grounding DINO, OMDet-Turbo decouples the class embeddings from the
    task prompt, so its post-processing returns labels that map directly back to
    the queried class list. There is no phrase-to-class disambiguation. It also
    runs its own NMS inside ``post_process_grounded_object_detection``.
    """

    FAMILY = "omdet_turbo"
    FILENAME_PREFIX = "LibreOMDetTurbo"
    HF_REPOS: ClassVar[Dict[str, str]] = {
        "t": "LibreYOLO/LibreOMDetTurbot",
    }
    # Informational only: the HF processor owns resizing and predict(imgsz=...)
    # is rejected by the open-vocab base. Mirrors the published config image_size.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"t": 640}
    DEFAULT_CONF: ClassVar[float] = 0.3
    # OMDet-Turbo applies its own NMS in post-processing (independent of the
    # LibreYOLO NMS path, which this tier does not run). ``iou=`` is ignored by
    # the base __call__, so this fixed default owns box suppression.
    NMS_THRESHOLD: ClassVar[float] = 0.5
    TASK_TEMPLATE: ClassVar[str] = "Detect {}."

    def _load_pretrained(self, snapshot_dir: str):
        try:
            from transformers import AutoProcessor, OmDetTurboForObjectDetection
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        model = OmDetTurboForObjectDetection.from_pretrained(
            snapshot_dir, dtype=self._resolve_dtype()
        )
        processor = AutoProcessor.from_pretrained(snapshot_dir)
        return model, processor

    def _class_names(self) -> list[str]:
        return [str(self.names[class_id]) for class_id in range(len(self.names))]

    def _task_prompt(self, names: list[str]) -> str:
        return self.TASK_TEMPLATE.format(", ".join(names))

    def _build_inputs(self, img: Any) -> Any:
        names = self._class_names()
        return self.processor(
            images=img,
            text=names,
            task=self._task_prompt(names),
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
        names = self._class_names()
        results = self.processor.post_process_grounded_object_detection(
            output,
            text_labels=names,
            threshold=float(conf_thres),
            nms_threshold=float(self.NMS_THRESHOLD),
            target_sizes=[(height, width)],
        )
        result = results[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        raw_labels = result.get("text_labels")
        if raw_labels is None:
            raw_labels = result.get("classes")
        if raw_labels is None:
            raw_labels = result.get("labels", [])

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


__all__ = ["LibreOMDetTurbo"]
