"""LibreYOLO wrapper for Microsoft's Florence-2 vision foundation model.

Florence-2 (MIT) is a small, purpose-built detection/grounding model. It does not
use a chat template: it is driven by task tokens (here ``<OPEN_VOCABULARY_DETECTION>``
plus the class list) through a plain ``processor(text=..., images=...)`` call, and
its boxes are decoded by the processor's ``post_process_generation`` into PIXEL
xyxy coordinates. So this family overrides the three inference hooks rather than
using the JSON path, and builds the detection dict directly (boxes are already in
pixels, no scaling needed).

Use the ``florence-community/*`` checkpoints (native ``Florence2ForConditionalGeneration``
in current transformers). The original ``microsoft/*`` remote-code checkpoints do
not load on recent transformers.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Tuple

from ...utils.image_loader import ImageInput, ImageLoader
from .base import LibreVLMModel
from .parsing import finalize_detection_dict


class LibreFlorence2(LibreVLMModel):
    """Florence-2 used as an open-vocabulary detector (task tokens, pixel boxes)."""

    FAMILY = "florence2"
    FILENAME_PREFIX = "LibreFlorence2"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "base": "florence-community/Florence-2-base",
        "large": "florence-community/Florence-2-large",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "base": 768,
        "large": 768,
    }

    # Task token that drives open-vocabulary detection.
    TASK = "<OPEN_VOCABULARY_DETECTION>"
    NUM_BEAMS = 3

    # MIT weights: no restrictive-license notice needed.
    _LICENSE_NOTICE = ""

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size=None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        query = ", ".join(self.names[i] for i in range(len(self.names)))
        inputs = self.processor(text=self.TASK + query, images=img, return_tensors="pt")
        return inputs, img, img.size, 1.0

    def _forward(self, inputs: Any) -> Any:
        inputs = self._prepare_generation_inputs(inputs)
        return self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=self.MAX_NEW_TOKENS,
            num_beams=self.NUM_BEAMS,
            do_sample=False,
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
        text = self.processor.batch_decode(output, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text, task=self.TASK, image_size=original_size
        )
        od = parsed.get(self.TASK, {})
        raw_boxes = od.get("bboxes", [])
        raw_labels = od.get("bboxes_labels", od.get("labels", []))
        try:
            detections = zip(list(raw_boxes), list(raw_labels))
        except (TypeError, ValueError):
            detections = []
        boxes, scores, classes = [], [], []
        for box, label in detections:
            class_id = self._name_to_id.get(str(label).strip().lower())
            if class_id is None:
                continue
            boxes.append(box)
            scores.append(self.DEFAULT_SCORE)
            classes.append(class_id)
        # Florence returns pixel xyxy already, so the shared finalizer only
        # validates/clips/filter/NMSes; it does not rescale geometry.
        return finalize_detection_dict(
            boxes,
            scores,
            classes,
            original_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            classes=kwargs.get("classes"),
        )

    def chat(self, *args, **kwargs):
        raise NotImplementedError(
            "Florence-2 is driven by task tokens, not free-form chat; use predict()."
        )
