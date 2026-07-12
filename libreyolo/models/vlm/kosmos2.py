"""LibreYOLO wrapper for Microsoft's Kosmos-2 grounding model.

Kosmos-2 (MIT, native ``Kosmos2ForConditionalGeneration``) is a grounded
multimodal model: given ``<grounding>`` text it generates a caption and grounds
the noun phrases, and its processor's ``post_process_generation`` returns the
entities with NORMALIZED [0,1] xyxy boxes. So this family overrides the inference
hooks (non-chat processor + entity post-processing) and scales the normalized
boxes to pixels.

Kosmos-2 is a 2023-era model: it loads cleanly and is a useful different
mechanism, but its boxes are coarse compared with newer grounders.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

from ...utils.image_loader import ImageInput, ImageLoader
from .base import LibreVLMModel
from .parsing import finalize_detection_dict


class LibreKosmos2(LibreVLMModel):
    """Kosmos-2 used as an open-vocabulary detector (grounded entities)."""

    FAMILY = "kosmos2"
    FILENAME_PREFIX = "LibreKosmos2"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "224": "microsoft/kosmos-2-patch14-224",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "224": 224,
    }

    # MIT weights: no restrictive-license notice needed.
    _LICENSE_NOTICE = ""

    def _match_label(self, name: str) -> Optional[int]:
        # Kosmos grounds noun phrases ("the boats"), so match leniently against
        # the vocabulary in addition to exact lookup. Prefer the longest unique
        # match so overlapping labels never depend on vocabulary insertion order.
        key = str(name).strip().lower()
        if key in self._name_to_id:
            return self._name_to_id[key]
        matches = []
        for cname, cid in self._name_to_id.items():
            if cname in key or key in cname:
                matches.append((len(cname.split()), len(cname), cname, cid))
        if not matches:
            return None
        matches.sort(reverse=True)
        best_specificity = matches[0][:2]
        best = [match for match in matches if match[:2] == best_specificity]
        return best[0][3] if len(best) == 1 else None

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size=None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        query = ", ".join(self.names[i] for i in range(len(self.names)))
        prompt = f"<grounding> Detect: {query}."
        inputs = self.processor(text=prompt, images=img, return_tensors="pt")
        return inputs, img, img.size, 1.0

    def _forward(self, inputs: Any) -> Any:
        inputs = self._prepare_generation_inputs(inputs)
        return self.model.generate(
            **inputs, max_new_tokens=self.MAX_NEW_TOKENS, do_sample=False
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
        text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        _caption, entities = self.processor.post_process_generation(text)
        width, height = original_size
        boxes, scores, classes = [], [], []
        try:
            entity_rows = list(entities)
        except (TypeError, ValueError):
            entity_rows = []
        for entity in entity_rows:
            try:
                name, _span, entity_boxes = entity
                entity_boxes = list(entity_boxes)
            except (TypeError, ValueError):
                continue
            class_id = self._match_label(name)
            if class_id is None:
                continue
            for box in entity_boxes:  # normalized [0,1] xyxy
                try:
                    values = list(box)
                except (TypeError, ValueError):
                    continue
                if len(values) != 4:
                    continue
                try:
                    boxes.append(
                        [
                            float(values[0]) * width,
                            float(values[1]) * height,
                            float(values[2]) * width,
                            float(values[3]) * height,
                        ]
                    )
                except (TypeError, ValueError):
                    continue
                scores.append(self.DEFAULT_SCORE)
                classes.append(class_id)
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
            "Kosmos-2 is driven by grounding prompts, not free-form chat; use predict()."
        )
