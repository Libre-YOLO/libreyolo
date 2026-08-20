"""LibreYOLO wrapper for Alibaba's Qwen3-VL vision-language models.

Qwen3-VL is a strong open-weight general VLM with native 2D grounding. For
detection it returns JSON objects with a ``bbox_2d`` key whose coordinates are
on a **0-1000** scale relative to the image (verified empirically: a box at
pixels [240,180,480,420] on an 800x600 image comes back as ~[300,300,600,700]).
That differs from LFM2-VL's ``bbox`` on a [0,1] scale, so this family sets
``BBOX_KEY``/``COORD_DIVISOR`` accordingly; the shared base handles the rest.

Qwen3-VL (Apache-2.0 on the supported 2B, 4B, and 8B sizes) loads through the same
``AutoModelForImageTextToText`` path as the rest of the LibreVLM tier.
"""

from __future__ import annotations

from typing import ClassVar, Dict

from .base import LibreVLMModel


class LibreQwen3VL(LibreVLMModel):
    """Qwen3-VL repurposed as a closed-set object detector."""

    FAMILY = "qwen3vl"
    FILENAME_PREFIX = "LibreQwen3VL"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "2b": "Qwen/Qwen3-VL-2B-Instruct",
        "4b": "Qwen/Qwen3-VL-4B-Instruct",
        "8b": "Qwen/Qwen3-VL-8B-Instruct",
    }
    # Immutable upstream snapshots. All three model cards declare Apache-2.0
    # and contain weights/configuration only; model code comes from the native
    # Transformers dependency with ``trust_remote_code=False``.
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "2b": "89644892e4d85e24eaac8bacfd4f463576704203",
        "4b": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "8b": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    }
    # Nominal only; the Qwen processor owns the real smart-resize.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "2b": 1024,
        "4b": 1024,
        "8b": 1024,
    }

    # Qwen emits {"bbox_2d": [x1,y1,x2,y2], "label": ...} on a 0-1000 scale.
    BBOX_KEY = "bbox_2d"
    COORD_DIVISOR = 1000.0
    # The bounded-memory scoring candidate remains off until a real-data gate
    # establishes useful ordering and a safe interaction with ``conf=``.
    TOKEN_LOGPROB_CONFIDENCE = False

    # First trainable family: grounding-pretrained, Apache-2.0, and native
    # transformers classes. The recipe uses the official Apache-2.0 Qwen3-VL
    # fine-tuning code as a reference for language-model LoRA with a frozen
    # vision tower; LibreYOLO's scope and hyperparameters differ. See
    # ``training/recipes.py`` and ``THIRD_PARTY_NOTICES.txt``.
    TRAINABLE = True
    TRAINABLE_SIZES = ("2b", "4b")

    # Apache-2.0 weights for every supported size: no restrictive notice needed.
    _LICENSE_NOTICE = ""

    def _format_detection_prompt(self, labels: str) -> str:
        return (
            f"Detect all instances of: {labels}. "
            "Output the result as a JSON array, one object per instance: "
            '[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
            "Only include objects that are actually visible; if there are none, "
            "respond with an empty array []."
        )
