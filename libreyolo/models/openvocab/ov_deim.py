"""OV-DEIM real-time open-vocabulary detector adapter.

Unlike the other families in this tier, OV-DEIM is a native port (no
``transformers`` model class exists for it). The detector comes from
https://github.com/wleilei/OV-DEIM (code Apache-2.0, weights CC BY-NC 4.0)
and classifies queries by cosine similarity against MobileCLIP-B(LT) text
embeddings, produced online by the bundled text tower for arbitrary prompts.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch

from .base import LibreOpenVocabDetector

_TEXT_TOWER_LICENSE_NOTICE = (
    "LibreOVDEIM weights are CC BY-NC 4.0 (non-commercial, upstream: "
    "wleilei/OV-DEIM); the bundled MobileCLIP-B(LT) text tower is covered by "
    "the Apple Machine Learning Research Model license (research use only). "
    "See the LICENSE files in the weight repository."
)


class LibreOVDEIM(LibreOpenVocabDetector):
    """OV-DEIM: real-time DETR-style open-vocabulary detector (no NMS).

    The classification branch scores each query against the text embedding of
    every prompt, so ``set_classes([...])`` / ``predict(classes=[...])`` work
    with arbitrary phrases. Text embeddings are cached per vocabulary.
    """

    FAMILY = "ov_deim"
    FILENAME_PREFIX = "LibreOVDEIM"
    HF_REPOS: ClassVar[Dict[str, str]] = {
        "s": "LibreYOLO/LibreOVDEIMs",
        "m": "LibreYOLO/LibreOVDEIMm",
        "l": "LibreYOLO/LibreOVDEIMl",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": 640, "m": 640, "l": 640}
    DEFAULT_CONF: ClassVar[float] = 0.25
    # One-to-one matching, top-K selection: no NMS anywhere.
    NMS_THRESHOLD: ClassVar[float | None] = None

    _LICENSE_NOTICE = _TEXT_TOWER_LICENSE_NOTICE

    # ImageNet statistics (upstream ToTensor defaults).
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _PAD_VALUE = 114

    def __init__(self, size: str, **kwargs):
        self._text_feats_cache: tuple[tuple[str, ...], torch.Tensor] | None = None
        self._tokenizer = None
        super().__init__(size, **kwargs)

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def _ensure_weights(self) -> str:
        # Same flow as the base class but without requiring ``transformers``:
        # this family is a native port and only needs huggingface_hub.
        import json
        from pathlib import Path

        from .base import _SNAPSHOT_COMPLETE_MARKER

        repo = self.HF_REPOS[self.size]
        local_dir = Path("weights") / f"{self.FILENAME_PREFIX}{self.size}"
        self._notify_license_once()
        if self._snapshot_complete(local_dir, repo=repo):
            return str(local_dir)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo,
            local_dir=str(local_dir),
            ignore_patterns=list(self.SNAPSHOT_IGNORE_PATTERNS),
        )
        (local_dir / _SNAPSHOT_COMPLETE_MARKER).write_text(
            json.dumps({"repo": repo}) + "\n", encoding="utf-8"
        )
        if not self._snapshot_complete(local_dir, repo=repo):
            (local_dir / _SNAPSHOT_COMPLETE_MARKER).unlink(missing_ok=True)
            raise FileNotFoundError(
                f"Downloaded snapshot for {repo} is missing config or weight files in {local_dir}."
            )
        return str(local_dir)

    def _load_pretrained(self, snapshot_dir: str):
        from pathlib import Path

        from safetensors.torch import load_file

        from .ovdeim import LibreOVDEIMModel

        model = LibreOVDEIMModel(self.size)
        state_dict = load_file(str(Path(snapshot_dir) / "model.safetensors"))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model, None

    # ------------------------------------------------------------------
    # Text side
    # ------------------------------------------------------------------

    def set_classes(self, classes: list[str]) -> "LibreOVDEIM":
        super().set_classes(classes)
        self._text_feats_cache = None
        return self

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from ..clip.tokenizer import SimpleTokenizer

            self._tokenizer = SimpleTokenizer()
        return self._tokenizer

    def _class_text_feats(self) -> torch.Tensor:
        """L2-normalized text embeddings for the current vocabulary, cached."""
        names = tuple(str(self.names[i]) for i in range(len(self.names)))
        if self._text_feats_cache is not None and self._text_feats_cache[0] == names:
            return self._text_feats_cache[1]
        tokens = self._get_tokenizer()(list(names)).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_texts(tokens)
        self._text_feats_cache = (names, feats)
        return feats

    # ------------------------------------------------------------------
    # Image side (mirrors upstream KeepRatioResize + LetterResize + ToTensor)
    # ------------------------------------------------------------------

    @classmethod
    def _letterbox_params(
        cls, width: int, height: int, target: int
    ) -> tuple[int, int, int, int]:
        """Return (new_w, new_h, left_pad, top_pad) for an original size."""
        scale = min(target / max(height, width), target / min(height, width))
        new_w = int(np.rint(width * scale))
        new_h = int(np.rint(height * scale))
        top = int(np.rint((target - new_h) / 2))
        left = int(np.rint((target - new_w) / 2))
        return new_w, new_h, left, top

    def _build_inputs(self, img: Any) -> Dict[str, torch.Tensor]:
        import cv2

        array = np.asarray(img.convert("RGB") if hasattr(img, "convert") else img)
        height, width = array.shape[:2]
        target = self.INPUT_SIZES[self.size]
        new_w, new_h, left, top = self._letterbox_params(width, height, target)

        if (new_w, new_h) != (width, height):
            interpolation = cv2.INTER_AREA if new_w < width else cv2.INTER_LINEAR
            array = cv2.resize(array, (new_w, new_h), interpolation=interpolation)
        canvas = np.full((target, target, 3), self._PAD_VALUE, dtype=array.dtype)
        canvas[top : top + new_h, left : left + new_w] = array

        tensor = torch.from_numpy(
            np.ascontiguousarray(canvas.transpose(2, 0, 1))
        ).to(torch.float32) / 255.0
        tensor = (tensor - torch.from_numpy(self._MEAN).view(-1, 1, 1)) / torch.from_numpy(
            self._STD
        ).view(-1, 1, 1)
        return tensor.unsqueeze(0)

    def _forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        images = inputs.to(self.device, dtype=torch.float32)
        text_feats = self._class_text_feats().unsqueeze(0)
        return self.model(images, text_feats)

    # ------------------------------------------------------------------
    # Postprocess (mirrors upstream OVDEIMPostProcessor)
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        output: Dict[str, torch.Tensor],
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        logits = output["pred_logits"][0]  # (Q, C)
        boxes = output["pred_boxes"][0]  # (Q, 4) cxcywh in [0, 1]
        num_classes = logits.shape[-1]
        num_queries = logits.shape[0]
        target = self.INPUT_SIZES[self.size]

        scores = torch.sigmoid(logits).flatten()
        topk = min(num_queries, scores.shape[0])
        scores, index = torch.topk(scores, topk, dim=-1)
        labels = index % num_classes
        query_index = index // num_classes
        boxes = boxes[query_index]

        # cxcywh (normalized) -> xyxy in letterboxed pixels
        cxcy, wh = boxes[:, :2] * target, boxes[:, 2:] * target
        boxes_xyxy = torch.cat([cxcy - wh / 2, cxcy + wh / 2], dim=-1)

        # Undo the letterbox for the original image size.
        width, height = original_size
        new_w, new_h, left, top = self._letterbox_params(width, height, target)
        pad = boxes_xyxy.new_tensor([left, top, left, top])
        scale = boxes_xyxy.new_tensor(
            [new_w / width, new_h / height, new_w / width, new_h / height]
        )
        boxes_xyxy = (boxes_xyxy - pad) / scale

        return self._detections_to_dict(
            boxes_xyxy,
            scores,
            labels,
            conf_thres=conf_thres,
            original_size=original_size,
            max_det=max_det,
            classes=kwargs.get("classes"),
        )


__all__ = ["LibreOVDEIM"]
