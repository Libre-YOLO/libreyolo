"""Interactive segmentation for LibreLabel (SAM click-to-mask).

Click a point on an object -> the Segment Anything model returns a mask ->
simplified to a polygon the human accepts/edits. Runs **fully local / offline**
on the cached ``facebook/sam-vit-base`` weights, reusing the ``torch`` /
``transformers`` already present for LibreYOLO. Optional and gated: if SAM isn't
installed or cached, the tool stays a clean box+polygon manual labeller.

Clean-room note: SAM checkpoints are Apache-2.0; this module wraps the public
``transformers`` API and contains no third-party annotation-tool code.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SAM_MODEL = "facebook/sam-vit-base"


class SamEngine:
    """Lazy, thread-safe SAM point-prompt -> polygon (no writes, no cloud)."""

    def __init__(self, enabled: bool = True, device: str = "auto"):
        self.enabled = enabled
        self._device_pref = device
        self._device = None
        self._model = None
        self._proc = None
        self._lock = threading.Lock()
        self._checked: Optional[bool] = None

    def available(self) -> bool:
        if not self.enabled:
            return False
        if self._checked is None:
            try:
                import importlib.util

                has_tf = importlib.util.find_spec("transformers") is not None
                hub = Path.home() / ".cache" / "huggingface" / "hub"
                cached = bool(
                    has_tf and hub.exists() and any(hub.glob("models--facebook--sam-vit-base"))
                )
                self._checked = cached
            except Exception:  # noqa: BLE001
                self._checked = False
        return self._checked

    def _resolve_device(self) -> str:
        if self._device_pref and self._device_pref != "auto":
            return self._device_pref
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure(self):
        if self._model is None:
            from transformers import SamModel, SamProcessor

            self._device = self._resolve_device()
            logger.info("LibreLabel loading SAM (%s) on %s", SAM_MODEL, self._device)
            self._proc = SamProcessor.from_pretrained(SAM_MODEL)
            self._model = SamModel.from_pretrained(SAM_MODEL).to(self._device).eval()
        return self._model, self._proc

    def segment(
        self,
        image_path: Path,
        point: Optional[tuple] = None,
        box: Optional[List[float]] = None,
        simplify: float = 0.005,
    ) -> Optional[List[float]]:
        """Segment with a normalised point ``(x, y)`` **or** box ``[x1,y1,x2,y2]``.

        Returns a normalised polygon ``[x1, y1, x2, y2, ...]`` (the simplified
        largest contour of the best mask), or ``None`` if nothing was found.
        """
        import cv2
        import torch
        from PIL import Image

        with self._lock:
            model, proc = self._ensure()
            img = Image.open(image_path).convert("RGB")
            w_img, h_img = img.size
            if box is not None:
                x1, y1, x2, y2 = box
                prompt = {"input_boxes": [[[
                    float(x1) * w_img, float(y1) * h_img, float(x2) * w_img, float(y2) * h_img,
                ]]]}
            else:
                px, py = float(point[0]) * w_img, float(point[1]) * h_img
                prompt = {"input_points": [[[px, py]]]}
            inputs = proc(img, return_tensors="pt", **prompt).to(self._device)
            with torch.no_grad():
                out = model(**inputs)
            masks = proc.image_processor.post_process_masks(
                out.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            scores = out.iou_scores[0, 0]
            best = int(scores.argmax())
            mask = masks[0][0, best].numpy().astype("uint8")

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        eps = simplify * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            return None
        poly: List[float] = []
        for vx, vy in approx:
            poly.append(min(max(float(vx) / w_img, 0.0), 1.0))
            poly.append(min(max(float(vy) / h_img, 0.0), 1.0))
        return poly
