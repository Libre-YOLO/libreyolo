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

import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.general import COCO_CLASSES
from ...utils.image_loader import ImageInput, ImageLoader
from .base import LibreVLMModel


def _florence_detection_dict(
    parsed: Any,
    *,
    task_token: str,
    name_to_id: Dict[str, int],
    conf_thres: float,
    max_det: int,
    classes: Any,
    default_score: float,
) -> Dict:
    """Convert one processor-parsed Florence response to the shared contract."""

    if not isinstance(parsed, Mapping):
        raise TypeError("Florence processor returned an invalid parsed response.")
    od = parsed.get(task_token, {})
    if not isinstance(od, Mapping):
        raise TypeError("Florence processor returned an invalid task response.")
    labels = od.get("bboxes_labels", od.get("labels", []))
    boxes_value = od.get("bboxes", [])
    if not isinstance(labels, (list, tuple)) or not isinstance(
        boxes_value, (list, tuple)
    ):
        raise TypeError("Florence parsed boxes and labels must be sequences.")

    boxes, scores, class_ids = [], [], []
    allowed_classes = set(classes) if classes is not None else None
    if max_det <= 0:
        return {
            "boxes": boxes,
            "scores": scores,
            "classes": class_ids,
            "num_detections": 0,
        }
    detections = zip(boxes_value, labels) if default_score >= conf_thres else []
    for box, label in detections:
        class_id = name_to_id.get(str(label).strip().lower())
        if class_id is None:
            continue
        if allowed_classes is not None and class_id not in allowed_classes:
            continue
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        scores.append(default_score)
        class_ids.append(class_id)
        if len(boxes) >= max_det:
            break
    return {
        "boxes": boxes,
        "scores": scores,
        "classes": class_ids,
        "num_detections": len(boxes),
    }


class LibreFlorence2(LibreVLMModel):
    """Florence-2 used as an open-vocabulary detector (task tokens, pixel boxes)."""

    FAMILY = "florence2"
    FILENAME_PREFIX = "LibreFlorence2"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "base": "florence-community/Florence-2-base",
        "large": "florence-community/Florence-2-large",
    }
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        # Exact MIT snapshot covered by the Core ML graph, processor, weight,
        # portable-bundle, and host-generation contracts.
        "base": "00921df66db728a9ceb750f5eca43e5c203a2051",
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
        return _florence_detection_dict(
            parsed,
            task_token=self.TASK,
            name_to_id=self._name_to_id,
            conf_thres=conf_thres,
            max_det=max_det,
            classes=kwargs.get("classes"),
            default_score=self.DEFAULT_SCORE,
        )

    def chat(self, *args, **kwargs):
        raise NotImplementedError(
            "Florence-2 is driven by task tokens, not free-form chat; use predict()."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export the exact base snapshot as an experimental Core ML bundle.

        Until an Apple execution profile is registered, callers must opt in
        with an explicit native planner; ``compute_units="cpu_only"`` is the
        recommended discovery setting.
        """

        normalized = str(format).strip().lower()
        if normalized not in {"coreml", "coremlvlm"}:
            return super().export(format=format, **kwargs)
        if self.size != "base":
            raise NotImplementedError(
                "Florence-2 Core ML currently supports only the reviewed base "
                "profile; the large architecture has no completed conversion "
                "contract."
            )
        if self.device.type != "cpu":
            raise NotImplementedError(
                "Florence-2 Core ML conversion requires a CPU-loaded FP32 "
                "model. Construct it with "
                "LibreVLM('florence-2-base', device='cpu')."
            )
        floating_dtypes = {
            value.dtype
            for value in (
                *tuple(self.model.parameters()),
                *tuple(self.model.buffers()),
            )
            if value.is_floating_point()
        }
        if floating_dtypes != {torch.float32}:
            raise NotImplementedError(
                "Florence-2 Core ML conversion requires an FP32-loaded model; "
                f"found {sorted(str(value) for value in floating_dtypes)}."
            )

        output_path = kwargs.pop("output_path", None)
        output_alias = kwargs.pop("output", None)
        if output_path not in (None, "") and output_alias not in (None, ""):
            if Path(output_path) != Path(output_alias):
                raise ValueError(
                    "Pass only one Florence Core ML destination: output_path= "
                    "or output=."
                )
        compute_units = kwargs.pop("compute_units", "validated")
        if kwargs:
            raise TypeError(
                "Unsupported or irrelevant Florence Core ML export options: "
                + ", ".join(sorted(kwargs))
            )
        from ...backends.coreml_florence import (
            COREML_FLORENCE_BUNDLE_SUFFIX,
            build_coreml_florence_bundle,
        )
        from ...export.coreml_florence import (
            FLORENCE2_BASE_REVISION,
            export_florence2_base_coreml_package,
            resolve_florence2_base_coreml_export_compute_units,
        )

        compute_units = resolve_florence2_base_coreml_export_compute_units(
            compute_units
        )
        destination_value = output_path or output_alias
        destination = (
            Path(destination_value)
            if destination_value not in (None, "")
            else Path("weights") / "LibreFlorence2-base.coremlvlm"
        )
        if destination.suffix.lower() != COREML_FLORENCE_BUNDLE_SUFFIX:
            raise ValueError(
                "Florence Core ML export produces a portable "
                f"{COREML_FLORENCE_BUNDLE_SUFFIX} directory, got "
                f"{destination}."
            )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite Florence Core ML bundle: {destination}."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        snapshot_dir = Path(self._ensure_weights())
        with tempfile.TemporaryDirectory(
            prefix=".libreyolo-florence-coreml-",
            dir=str(destination.parent),
        ) as workspace:
            package = Path(workspace) / "Model.mlpackage"
            export_florence2_base_coreml_package(
                self.model,
                checkpoint_dir=snapshot_dir,
                processor_revision=FLORENCE2_BASE_REVISION,
                output_path=package,
                compute_units=compute_units,
            )
            return build_coreml_florence_bundle(
                package,
                processor_dir=snapshot_dir,
                output_path=destination,
                move_model=True,
            )


class _CoreMLFlorenceInputs:
    """One host-owned Florence request accepted by the inference runner."""

    def __init__(self, image: Any) -> None:
        self.image = image

    def to(self, _device: Any) -> "_CoreMLFlorenceInputs":
        return self


class CoreMLFlorence2(LibreFlorence2):
    """Public LibreVLM adapter over a portable Florence Core ML bundle."""

    def __init__(
        self,
        bundle_path: str,
        *,
        nb_classes: int = 80,
        names: Optional[list] = None,
        device: str = "auto",
        task: str | None = None,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        compute_units: str = "validated",
    ) -> None:
        if str(device).strip().lower() not in {"auto", "cpu"}:
            raise ValueError(
                "A Core ML Florence bundle owns Apple accelerator routing. "
                "Pass device='auto' or 'cpu' and select compute_units "
                "separately."
            )
        if isinstance(nb_classes, bool) or not isinstance(nb_classes, int):
            raise TypeError("nb_classes must be an integer.")
        if nb_classes <= 0:
            raise ValueError("nb_classes must be positive.")
        if prompt is not None:
            raise ValueError(
                "Florence-2 uses its fixed detection task token; prompt= is "
                "not supported by the Core ML adapter."
            )
        resolved_task = self._resolve_task(task)
        initial_names = list(
            names
            if names is not None
            else (
                COCO_CLASSES[:nb_classes]
                if nb_classes <= len(COCO_CLASSES)
                else [
                    *COCO_CLASSES,
                    *(
                        f"class_{index}"
                        for index in range(len(COCO_CLASSES), nb_classes)
                    ),
                ]
            )
        )

        from ...backends.coreml_florence import CoreMLFlorenceRuntime

        runtime = CoreMLFlorenceRuntime(
            bundle_path,
            names=initial_names,
            compute_units=compute_units,
        )
        self.family = self.FAMILY
        self.task = resolved_task
        self.size = "base"
        self.nb_classes = len(initial_names)
        self.input_size = runtime.profile.image_size
        self.device = torch.device("cpu")
        self.model_path = str(bundle_path)
        self.model = nn.Identity()
        self.processor = runtime.processor
        self._coreml_runtime = runtime
        self._custom_prompt = None
        self._graph_runner = None
        self._cuda_graph_mode = None
        self._runner_instance = None
        self.names = dict(enumerate(initial_names))
        self._name_to_id = {
            value.strip().lower(): key for key, value in self.names.items()
        }
        try:
            # Reuse the public validation for empty, duplicate, and non-string
            # class names, then synchronize the already-created runtime.
            self.set_classes(initial_names)
            if max_new_tokens is None:
                self.MAX_NEW_TOKENS = runtime.profile.max_new_tokens
            elif isinstance(max_new_tokens, bool) or not isinstance(
                max_new_tokens, int
            ):
                raise TypeError("max_new_tokens must be an integer.")
            elif max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be positive.")
            elif max_new_tokens > runtime.profile.max_new_tokens:
                raise ValueError(
                    "max_new_tokens exceeds this Florence Core ML profile's "
                    f"reviewed limit of {runtime.profile.max_new_tokens}."
                )
            else:
                self.MAX_NEW_TOKENS = max_new_tokens
        except Exception:
            runtime.close()
            raise

    def set_classes(self, classes: list) -> "CoreMLFlorence2":
        if isinstance(classes, str) or not isinstance(classes, (list, tuple)):
            raise TypeError(
                "set_classes() expects a list/tuple of label strings, "
                f"not {type(classes).__name__}."
            )
        runtime = getattr(self, "_coreml_runtime", None)
        if runtime is not None:
            # Validate and publish to the runtime first. If it is closed or
            # rejects the vocabulary, the public adapter must retain its
            # previous class mapping instead of becoming half-updated.
            runtime.set_classes(classes)
        super().set_classes(classes)
        return self

    def close(self) -> None:
        self._coreml_runtime.close()
        self.processor = None

    def __enter__(self) -> "CoreMLFlorence2":
        if self._coreml_runtime.closed:
            raise RuntimeError("Core ML Florence runtime is closed.")
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {}

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size=None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        if input_size not in (None, self.input_size):
            raise ValueError(
                "Core ML Florence uses its fixed 768x768 processor contract; "
                f"got imgsz={input_size}."
            )
        loaded = ImageLoader.load(image, color_format=color_format)
        return _CoreMLFlorenceInputs(loaded), loaded, loaded.size, 1.0

    def _forward(self, inputs: Any) -> Dict[str, Any]:
        if not isinstance(inputs, _CoreMLFlorenceInputs):
            raise TypeError("Core ML Florence received an invalid request.")
        return self._coreml_runtime.generate(
            inputs.image,
            max_new_tokens=self.MAX_NEW_TOKENS,
            color_format="rgb",
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
        if not isinstance(output, Mapping) or "parsed" not in output:
            raise RuntimeError("Core ML Florence runtime returned an invalid response.")
        return _florence_detection_dict(
            output["parsed"],
            task_token=self.TASK,
            name_to_id=self._name_to_id,
            conf_thres=conf_thres,
            max_det=max_det,
            classes=kwargs.get("classes"),
            default_score=self.DEFAULT_SCORE,
        )

    def export(self, format: str = "coreml", **kwargs) -> str:
        raise NotImplementedError(
            "A .coremlvlm Florence bundle is already a deployed Core ML "
            "artifact and cannot be re-exported."
        )
