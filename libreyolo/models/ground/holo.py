"""Holo1.5-7B: Apache-2.0 Qwen2.5-VL GUI localizer from H Company.

Holo is prompted to emit ``Click(x, y)`` in *resized-view pixels*. The
adapter records the Qwen smart-resize view in ``_preprocess`` and maps
back to the original canvas via ``COORD_SPACE="pixel_view"``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

from ...utils.image_loader import ImageInput, ImageLoader
from .base import LibreGroundModel


def _qwen_view_size(processor, width: int, height: int) -> Tuple[int, int]:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return width, height
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    except ImportError:
        try:
            from transformers.models.qwen2_5_vl.image_processing_qwen2_5_vl import (
                smart_resize,
            )
        except ImportError:
            return width, height
    factor = getattr(image_processor, "patch_size", 14) * getattr(
        image_processor, "merge_size", 2
    )
    resized_h, resized_w = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=getattr(image_processor, "min_pixels", 4 * 28 * 28),
        max_pixels=getattr(image_processor, "max_pixels", 1280 * 28 * 28),
    )
    return int(resized_w), int(resized_h)


class LibreHolo(LibreGroundModel):
    FAMILY = "holo"
    FILENAME_PREFIX = "LibreHolo"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "7b": "Hcompany/Holo1.5-7B",
        "1-7b": "Hcompany/Holo1-7B",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "7b": 1280,
        "1-7b": 1280,
    }
    COORD_SPACE = "pixel_view"
    MAX_NEW_TOKENS = 64

    def _format_grounding_prompt(self, query: str) -> str:
        return (
            "Localize an element on the GUI image according to my instructions "
            "and output a click position as Click(x, y) with x num pixels from "
            "the left edge and y num pixels from the top edge.\n"
            f"{query}"
        )

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        self._view_size = _qwen_view_size(self.processor, img.size[0], img.size[1])
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": self._grounding_prompt()},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        return inputs, img, img.size, 1.0
