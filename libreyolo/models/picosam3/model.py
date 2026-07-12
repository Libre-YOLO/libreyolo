"""LibrePicoSAM3: native PicoSAM3 on the LibreSAM promptable tier."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar, Dict, Optional

import torch
import torch.nn as nn

from ...utils.download import WeightPublicationError
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import (
    SCHEMA_VERSION,
    load_untrusted_torch_file,
    unwrap_libreyolo_checkpoint,
)
from ..manifest import get_artifact_spec
from ..sam.base import (
    _INSTALL_HINT,
    _SNAPSHOT_COMPLETE_MARKER,
    LibreSAMModel,
    _synchronized_image_state,
)
from ..sam.prompts import normalize_boxes, validate_max_det
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
    HF_REPOS: ClassVar[Dict[str, str]] = {"pico": "LibreYOLO/LibrePicoSAM3"}
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"pico": 96}
    INPUT_SIZE_FIXED: ClassVar[bool] = True
    WEIGHT_FILE: ClassVar[str] = "LibrePicoSAM3pico.pt"

    def __init__(self, size: str = "pico", **kwargs) -> None:
        super().__init__(size=size, **kwargs)

    @classmethod
    def _snapshot_complete(cls, local_dir: Path) -> bool:
        return (local_dir / _SNAPSHOT_COMPLETE_MARKER).exists() and (
            local_dir / cls.WEIGHT_FILE
        ).exists()

    def _ensure_weights(self) -> str:
        local_dir = Path("weights") / f"{self.FILENAME_PREFIX}{self.size}"
        if self._snapshot_complete(local_dir):
            return str(local_dir)

        artifact = get_artifact_spec(self.FAMILY, self.size, self.DEFAULT_TASK)
        if (
            artifact is None
            or not artifact.downloadable
            or artifact.download_kind != "snapshot"
            or not artifact.repository
        ):
            publication = (
                artifact.publication.value if artifact is not None else "undeclared"
            )
            raise WeightPublicationError(
                f"{self.FAMILY!r} size {self.size!r} is a valid architecture "
                f"artifact, but its publication state is {publication!r} and "
                "LibreYOLO does not declare a public snapshot route. Provide "
                "a complete local snapshot."
            )

        repo = artifact.repository
        class_repo = self.HF_REPOS.get(self.size)
        if class_repo != repo:
            raise RuntimeError(
                f"PicoSAM3 repository metadata drift: class declares "
                f"{class_repo!r}, manifest declares {repo!r}."
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        local_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading %s weights from %s -> %s ...",
            self.FAMILY,
            repo,
            local_dir,
        )
        hf_hub_download(
            repo_id=repo,
            filename=self.WEIGHT_FILE,
            local_dir=local_dir,
        )
        (local_dir / _SNAPSHOT_COMPLETE_MARKER).write_text(
            json.dumps({"repo": repo}) + "\n",
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

    @_synchronized_image_state
    def set_image(
        self, source: ImageInput, color_format: str = "auto"
    ) -> "LibrePicoSAM3":
        """Cache an image for repeated ROI prompts."""

        self._image = ImageLoader.load(source, color_format=color_format)
        self._image_path = source if isinstance(source, (str, Path)) else None
        self._image_embeddings = None
        return self

    @_synchronized_image_state
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

        max_det = validate_max_det(max_det)
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
        imgsz: int | tuple[int, int] | list[int] | None = None,
        opset: int | None = None,
        simplify: bool = True,
        dynamic: bool = True,
        half: bool = False,
        int8: bool = False,
        batch: int = 1,
        device: str | torch.device | int | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> str:
        """Export the raw 96x96 ROI CNN as ``roi_image -> mask_logits``."""

        if str(format).lower() != "onnx":
            raise NotImplementedError(
                "LibrePicoSAM3 export currently supports ONNX only."
            )
        if half:
            raise NotImplementedError(
                "LibrePicoSAM3 ONNX export does not support half=True."
            )
        if int8:
            raise NotImplementedError(
                "LibrePicoSAM3 ONNX export does not support int8=True."
            )
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported PicoSAM3 export arguments: {unknown}")
        if output is not None and output_path is not None:
            raise ValueError("Pass only one of output= or output_path=.")
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError(
                f"LibrePicoSAM3 export batch must be a positive integer, got {batch!r}."
            )
        native_imgsz = self.INPUT_SIZES[self.size]
        export_imgsz = native_imgsz if imgsz is None else imgsz
        if isinstance(export_imgsz, (list, tuple)):
            if len(export_imgsz) == 2 and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value == native_imgsz
                for value in export_imgsz
            ):
                export_imgsz = native_imgsz
        if (
            isinstance(export_imgsz, bool)
            or not isinstance(export_imgsz, int)
            or export_imgsz != native_imgsz
        ):
            raise ValueError(
                f"LibrePicoSAM3 export requires imgsz={native_imgsz}, "
                f"got {export_imgsz!r}."
            )
        opset = 13 if opset is None else int(opset)

        destination_value = output_path if output_path is not None else output
        destination = Path(destination_value or "LibrePicoSAM3pico.onnx")
        destination.parent.mkdir(parents=True, exist_ok=True)
        axes = None
        if dynamic:
            axes = {"roi_image": {0: "batch"}, "mask_logits": {0: "batch"}}

        first_parameter = next(self.model.parameters())
        original_device = first_parameter.device
        trace_device = _resolve_export_device(device, original_device)
        trace_dtype = first_parameter.dtype
        training_states = {module: module.training for module in self.model.modules()}

        try:
            self.model.to(trace_device).eval()
            torch.onnx.export(
                self.model,
                torch.zeros(
                    (batch, 3, export_imgsz, export_imgsz),
                    device=trace_device,
                    dtype=trace_dtype,
                ),
                str(destination),
                input_names=["roi_image"],
                output_names=["mask_logits"],
                dynamic_axes=axes,
                opset_version=opset,
                verbose=bool(verbose),
                dynamo=False,
            )

            from ...export.onnx import _get_version, finalize_onnx_artifact

            nb_classes = int(getattr(self, "nb_classes", 1))
            names = getattr(self, "names", {0: "object"})
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "libreyolo_version": _get_version(),
                "model_family": self.FAMILY,
                "size": self.size,
                "model_size": self.size,
                "task": "segment",
                "supported_tasks": json.dumps(["segment"]),
                "default_task": "segment",
                "nc": str(nb_classes),
                "nb_classes": str(nb_classes),
                "names": json.dumps({str(key): value for key, value in names.items()}),
                "imgsz": str(export_imgsz),
                "imgsz_h": str(export_imgsz),
                "imgsz_w": str(export_imgsz),
                "precision": "fp16" if trace_dtype == torch.float16 else "fp32",
                "dynamic": str(bool(dynamic)),
                "half": str(trace_dtype == torch.float16),
            }
            finalize_onnx_artifact(
                str(destination),
                simplify=bool(simplify),
                dynamic=bool(dynamic),
                half=trace_dtype == torch.float16,
                metadata=metadata,
            )
        finally:
            self.model.to(original_device)
            for module, training in training_states.items():
                module.training = training
        return str(destination)


def _resolve_export_device(
    requested: str | torch.device | int | None,
    current: torch.device,
) -> torch.device:
    """Resolve the tracing device without changing the owning model's device."""
    if requested is None or str(requested).lower() == "auto":
        return current
    if isinstance(requested, int) or (
        isinstance(requested, str) and requested.isdigit()
    ):
        requested = f"cuda:{requested}"
    return torch.device(requested)


__all__ = ["LibrePicoSAM3", "PicoSAM3Network"]
