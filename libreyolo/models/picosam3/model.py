"""LibrePicoSAM3: native PicoSAM3 on the LibreSAM promptable tier."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar, Dict, Optional

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import (
    load_untrusted_torch_file,
    unwrap_libreyolo_checkpoint,
)
from ..sam.base import _INSTALL_HINT, _SNAPSHOT_COMPLETE_MARKER, LibreSAMModel
from ..sam.prompts import normalize_boxes
from .nn import PicoSAM3Network
from .preprocess import padded_square_roi, place_roi_logits, preprocess_roi

logger = logging.getLogger(__name__)


class LibrePicoSAM3(LibreSAMModel):
    """PicoSAM3 edge segmenter conditioned by one or more box-defined ROIs.

    ``bboxes=`` is the only supported prompt. The box is expanded by 10%, made
    square, clipped to the image, and resized to 96x96 before the CNN runs.
    Point, text, mask, multimask, and segment-everything prompts are not part of
    the upstream model contract and fail clearly; use LibreSAM2 or LibreSAM3 for
    those modes.

    Unlike encoder/decoder SAM models, PicoSAM3 performs one complete CNN
    forward per ROI. ``set_image()`` therefore caches the source image rather
    than an embedding, preserving the common interactive API without implying
    nonexistent encoder reuse.
    """

    FAMILY = "picosam3"
    FILENAME_PREFIX = "LibrePicoSAM3"
    HF_REPOS: ClassVar[Dict[str, str]] = {"pico": "pietrobonazzi/picosam3"}
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"pico": 96}
    WEIGHT_FILE: ClassVar[str] = "PicoSAM3_SAM3_student_best.pt"
    UPSTREAM_REVISION: ClassVar[str] = "af49e4322b6b7cf448499fee5c073d4576f59444"

    def __init__(self, size: str = "pico", **kwargs) -> None:
        super().__init__(size=size, **kwargs)

    @staticmethod
    def _snapshot_complete(local_dir: Path) -> bool:
        return (local_dir / _SNAPSHOT_COMPLETE_MARKER).exists() and (
            local_dir / LibrePicoSAM3.WEIGHT_FILE
        ).exists()

    def _ensure_weights(self) -> str:
        repo = self.HF_REPOS[self.size]
        local_dir = Path("weights") / f"{self.FILENAME_PREFIX}{self.size}"
        if self._snapshot_complete(local_dir):
            return str(local_dir)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        local_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading %s weights from %s@%s -> %s ...",
            self.FAMILY,
            repo,
            self.UPSTREAM_REVISION,
            local_dir,
        )
        hf_hub_download(
            repo_id=repo,
            filename=self.WEIGHT_FILE,
            revision=self.UPSTREAM_REVISION,
            local_dir=local_dir,
        )
        (local_dir / _SNAPSHOT_COMPLETE_MARKER).write_text(
            json.dumps({"repo": repo, "revision": self.UPSTREAM_REVISION}) + "\n",
            encoding="utf-8",
        )
        if not self._snapshot_complete(local_dir):
            (local_dir / _SNAPSHOT_COMPLETE_MARKER).unlink(missing_ok=True)
            raise FileNotFoundError(
                f"Downloaded snapshot for {repo} is missing {self.WEIGHT_FILE} "
                f"in {local_dir}."
            )
        return str(local_dir)

    def _init_model(self) -> nn.Module:
        checkpoint_path = Path(self._ensure_weights()) / self.WEIGHT_FILE
        loaded = load_untrusted_torch_file(
            checkpoint_path,
            map_location="cpu",
            context="PicoSAM3 Apache-2.0 checkpoint",
        )
        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded, strict=False)
        if metadata:
            expected = {
                "model_family": self.FAMILY,
                "size": self.size,
                "task": self.DEFAULT_TASK,
            }
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatches:
                details = ", ".join(
                    f"{key}={got!r} (expected {want!r})"
                    for key, (got, want) in mismatches.items()
                )
                raise ValueError(
                    f"Invalid LibrePicoSAM3 checkpoint metadata: {details}"
                )

        model = PicoSAM3Network()
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            if any(key.startswith("output_head.") for key in state_dict):
                raise ValueError(
                    "This checkpoint contains the older PicoSAM2 architecture, not "
                    "PicoSAM3. Use PicoSAM3_SAM3_student_best.pt."
                ) from exc
            raise
        self.processor = None
        self._model_dtype = next(model.parameters()).dtype
        return model

    def set_image(
        self, source: ImageInput, color_format: str = "auto"
    ) -> "LibrePicoSAM3":
        """Cache an image for repeated ROI prompts."""

        self._image = ImageLoader.load(source, color_format=color_format)
        self._image_path = source if isinstance(source, (str, Path)) else None
        self._image_embeddings = None
        return self

    def predict(
        self,
        source: Optional[ImageInput] = None,
        *,
        points=None,
        bboxes=None,
        labels=None,
        masks=None,
        text: str | None = None,
        conf: Optional[float] = None,
        multimask: Optional[bool] = None,
        max_det: int = 300,
        device: Optional[str] = None,
        color_format: str = "auto",
        points_per_side: Optional[int] = None,
    ):
        """Segment the ROI inside each xyxy box prompt."""

        unsupported = []
        if points is not None:
            unsupported.append("points")
        if labels is not None:
            unsupported.append("labels")
        if masks is not None:
            unsupported.append("masks")
        if text is not None:
            unsupported.append("text")
        if points_per_side is not None:
            unsupported.append("points_per_side")
        if unsupported:
            raise ValueError(
                "PicoSAM3 supports only bboxes= ROI prompts; unsupported: "
                f"{', '.join(unsupported)}. Use LibreSAM2 or LibreSAM3 for "
                "point, text, mask, or segment-everything prompts."
            )
        if multimask not in (None, False):
            raise ValueError(
                "PicoSAM3 produces one mask per ROI and does not support "
                "multimask=True; use LibreSAM2 or LibreSAM3."
            )
        if bboxes is None or (hasattr(bboxes, "__len__") and len(bboxes) == 0):
            raise ValueError(
                "PicoSAM3 requires bboxes=; segment-everything mode is not "
                "supported. Use LibreSAM2 or LibreSAM3 for that mode."
            )
        if max_det < 1:
            raise ValueError("max_det must be >= 1.")
        conf_thres = 0.0 if conf is None else float(conf)
        if not 0.0 <= conf_thres <= 1.0:
            raise ValueError("conf must be between 0.0 and 1.0.")
        if device is not None:
            self._set_device(device)

        img, image_path, _ = self._resolve_image(source, color_format)
        boxes = normalize_boxes(bboxes)
        width, height = img.size
        rois = [padded_square_roi(box, width, height) for box in boxes]
        batch = torch.stack([preprocess_roi(img, roi) for roi in rois]).to(
            self.device, dtype=self._model_dtype
        )
        with torch.no_grad():
            logits = self.model(batch)
        full_masks, scores = place_roi_logits(logits, rois, height, width)
        return self._assemble_results(
            full_masks,
            scores,
            img,
            image_path,
            conf_thres=conf_thres,
            max_det=max_det,
        )

    def export(
        self,
        format: str = "onnx",
        *,
        output: str | Path | None = None,
        output_path: str | Path | None = None,
        opset: int = 13,
        dynamic: bool = True,
        **kwargs,
    ) -> str:
        """Export the raw 96x96 ROI CNN as ``roi_image -> mask_logits``."""

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported PicoSAM3 export arguments: {unknown}")
        if str(format).lower() != "onnx":
            raise NotImplementedError(
                "LibrePicoSAM3 export currently supports ONNX only."
            )
        if output is not None and output_path is not None:
            raise ValueError("Pass only one of output= or output_path=.")
        destination = Path(output_path or output or "LibrePicoSAM3pico.onnx")
        destination.parent.mkdir(parents=True, exist_ok=True)
        axes = None
        if dynamic:
            axes = {"roi_image": {0: "batch"}, "mask_logits": {0: "batch"}}
        self.model.eval()
        torch.onnx.export(
            self.model,
            torch.zeros((1, 3, 96, 96), device=self.device),
            str(destination),
            input_names=["roi_image"],
            output_names=["mask_logits"],
            dynamic_axes=axes,
            opset_version=int(opset),
            dynamo=False,
        )
        return str(destination)


__all__ = ["LibrePicoSAM3", "PicoSAM3Network"]
