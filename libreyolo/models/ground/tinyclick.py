"""TinyClick: 0.27B Florence-2-base GUI clicker (MIT).

Samsung Labs fine-tune. Loads with ``trust_remote_code`` pinned to a commit
SHA. Prompt and ``<loc_N>`` decode follow the model card; coordinates are
Florence's 0–1000 grid.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Tuple

from ...utils.image_loader import ImageInput, ImageLoader
from ..vlm.base import _INSTALL_HINT
from .base import LibreGroundModel


class LibreTinyClick(LibreGroundModel):
    FAMILY = "tinyclick"
    FILENAME_PREFIX = "LibreTinyClick"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "b": "kzawistowsk/TinyClick",
    }
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "b": "a9c0e2cfb9d3660217254838198d227501629085",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "b": 768,
    }
    COORD_SPACE = "milli"
    TRUST_REMOTE_CODE = True
    MAX_NEW_TOKENS = 64

    _LICENSE_NOTICE = (
        "\n"
        "----------------------------------------------------------------\n"
        "TinyClick is a Florence-2 fine-tune (MIT). Loading it downloads\n"
        "and executes Hugging Face remote code from the model repository.\n"
        "LibreYOLO pins the snapshot revision and does not redistribute it.\n"
        "  https://huggingface.co/kzawistowsk/TinyClick\n"
        "----------------------------------------------------------------\n"
    )

    def _load_pretrained(self, snapshot_dir: str):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_dir,
            dtype=self._resolve_dtype(),
            trust_remote_code=self.TRUST_REMOTE_CODE,
        )
        processor = AutoProcessor.from_pretrained(
            snapshot_dir, trust_remote_code=self.TRUST_REMOTE_CODE
        )
        return model, processor

    def _format_grounding_prompt(self, query: str) -> str:
        return ("What to do to execute the command? " + query.strip()).lower()

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size=None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        self._view_size = img.size
        inputs = self.processor(
            images=img,
            text=self._grounding_prompt(),
            return_tensors="pt",
            do_resize=True,
        )
        return inputs, img, img.size, 1.0

    def _forward(self, inputs: Any) -> Any:
        inputs = self._prepare_generation_inputs(inputs)
        return self.model.generate(
            **inputs,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=False,
        )

    def _decode_generated(self, output: Any) -> str:
        if isinstance(output, str):
            return output
        return self.processor.batch_decode(output, skip_special_tokens=False)[0]

    def chat(self, *args, **kwargs):
        raise NotImplementedError(
            "TinyClick is a single-turn clicker, not a chat model; use predict()."
        )
