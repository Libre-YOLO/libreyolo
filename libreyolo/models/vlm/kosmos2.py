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

import tempfile
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ...utils.general import COCO_CLASSES
from ...utils.image_loader import ImageInput, ImageLoader
from .base import LibreVLMModel


class LibreKosmos2(LibreVLMModel):
    """Kosmos-2 used as an open-vocabulary detector (grounded entities)."""

    FAMILY = "kosmos2"
    FILENAME_PREFIX = "LibreKosmos2"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "224": "microsoft/kosmos-2-patch14-224",
    }
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "224": "e91cfbcb4ce051b6a55bfb5f96165a3bbf5eb82c",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "224": 224,
    }

    # MIT weights: no restrictive-license notice needed.
    _LICENSE_NOTICE = ""

    def _match_label(self, name: str) -> Optional[int]:
        # Kosmos grounds noun phrases ("the boats"), so match leniently against
        # the vocabulary in addition to exact lookup.
        key = str(name).strip().lower()
        if key in self._name_to_id:
            return self._name_to_id[key]
        for cname, cid in self._name_to_id.items():
            if cname in key or key in cname:
                return cid
        return None

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
        allowed_classes = (
            set(kwargs["classes"]) if kwargs.get("classes") is not None else None
        )
        if max_det <= 0:
            return {
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
                "num_detections": 0,
            }
        # Every box carries the placeholder score, so conf filtering is all-or-nothing.
        scored = entities if self.DEFAULT_SCORE >= conf_thres else []
        for name, _span, entity_boxes in scored:
            if len(boxes) >= max_det:
                break
            class_id = self._match_label(name)
            if class_id is None:
                continue
            if allowed_classes is not None and class_id not in allowed_classes:
                continue
            for box in entity_boxes:  # normalized [0,1] xyxy
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
                scores.append(self.DEFAULT_SCORE)
                classes.append(class_id)
                if len(boxes) >= max_det:
                    break
        return {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "num_detections": len(boxes),
        }

    def chat(self, *args, **kwargs):
        raise NotImplementedError(
            "Kosmos-2 is driven by grounding prompts, not free-form chat; use predict()."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export the pinned 224 checkpoint as a bounded Core ML bundle."""

        normalized = str(format).strip().lower()
        if normalized not in {"coreml", "coremlvlm"}:
            return super().export(format=format, **kwargs)
        if self.size != "224":
            raise NotImplementedError(
                "Kosmos-2 Core ML supports only the pinned 224 checkpoint."
            )
        if self.device.type != "cpu":
            raise NotImplementedError(
                "Kosmos-2 Core ML conversion requires a CPU-loaded FP32 model."
            )
        output_path = kwargs.pop("output_path", None)
        output_alias = kwargs.pop("output", None)
        if output_path not in (None, "") and output_alias not in (None, ""):
            if Path(output_path) != Path(output_alias):
                raise ValueError(
                    "Pass only one Kosmos-2 Core ML destination: output_path= "
                    "or output=."
                )
        context_length = kwargs.pop("context_length", 128)
        compute_units = kwargs.pop("compute_units", "validated")
        if kwargs:
            raise TypeError(
                "Unsupported or irrelevant Kosmos-2 Core ML export options: "
                + ", ".join(sorted(kwargs))
            )

        from ...backends.coreml_kosmos import COREML_KOSMOS2_BUNDLE_SUFFIX
        from ...export.coreml_kosmos import (
            KOSMOS2_COREML_CONTEXT_LENGTH,
            build_kosmos2_coreml_bundle,
            export_kosmos2_coreml_components,
            resolve_kosmos2_coreml_compute_units,
            validate_kosmos2_source_model,
        )

        if context_length != KOSMOS2_COREML_CONTEXT_LENGTH:
            raise ValueError(
                "Kosmos-2 Core ML currently supports only context_length="
                f"{KOSMOS2_COREML_CONTEXT_LENGTH}."
            )
        compute_units = resolve_kosmos2_coreml_compute_units(compute_units)
        validate_kosmos2_source_model(self.model)
        destination_value = output_path or output_alias
        destination = (
            Path(destination_value)
            if destination_value not in (None, "")
            else Path("weights") / "LibreKosmos2-224-128.coremlvlm"
        )
        if destination.suffix.lower() != COREML_KOSMOS2_BUNDLE_SUFFIX:
            raise ValueError(
                "Kosmos-2 Core ML export produces a portable "
                f"{COREML_KOSMOS2_BUNDLE_SUFFIX} directory."
            )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite Kosmos-2 Core ML bundle: {destination}."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(self._ensure_weights())
        with tempfile.TemporaryDirectory(
            prefix=".libreyolo-kosmos2-coreml-",
            dir=str(destination.parent),
        ) as workspace:
            component_dir = Path(workspace) / "Components"
            export_kosmos2_coreml_components(
                self.model,
                checkpoint_dir=checkpoint_dir,
                output_dir=component_dir,
                compute_units=compute_units,
            )
            return build_kosmos2_coreml_bundle(
                component_dir,
                processor_dir=checkpoint_dir,
                output_path=destination,
            )


class _CoreMLKosmos2Inputs:
    def __init__(self, image: Any, prompt: str) -> None:
        self.image = image
        self.prompt = prompt

    def to(self, _device: Any) -> "_CoreMLKosmos2Inputs":
        return self


class CoreMLKosmos2(LibreKosmos2):
    """Public LibreVLM adapter over a portable Kosmos-2 Core ML bundle."""

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
        compute_units: str = "cpu_only",
    ) -> None:
        if str(device).strip().lower() not in {"auto", "cpu"}:
            raise ValueError(
                "A Kosmos-2 Core ML bundle owns Apple routing; use device='auto' "
                "or 'cpu'."
            )
        if prompt is not None:
            raise ValueError(
                "Kosmos-2 uses its fixed grounding prompt; prompt= is unsupported."
            )
        if isinstance(nb_classes, bool) or not isinstance(nb_classes, int):
            raise TypeError("nb_classes must be an integer.")
        if nb_classes <= 0:
            raise ValueError("nb_classes must be positive.")
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

        from ...backends.coreml_kosmos import CoreMLKosmos2Runtime

        runtime = CoreMLKosmos2Runtime(
            bundle_path,
            compute_units=compute_units,
        )
        self.family = self.FAMILY
        self.task = resolved_task
        self.size = "224"
        self.nb_classes = len(initial_names)
        self.input_size = 224
        self.device = torch.device("cpu")
        self.model_path = str(bundle_path)
        self.model = nn.Identity()
        self.processor = runtime.processor
        self._model_dtype = torch.float32
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
            self.set_classes(initial_names)
            if max_new_tokens is None:
                self.MAX_NEW_TOKENS = runtime.max_new_tokens
            elif isinstance(max_new_tokens, bool) or not isinstance(
                max_new_tokens, int
            ):
                raise TypeError("max_new_tokens must be an integer.")
            elif max_new_tokens <= 0 or max_new_tokens > runtime.max_new_tokens:
                raise ValueError(
                    "max_new_tokens must be between 1 and "
                    f"{runtime.max_new_tokens} for this Kosmos-2 Core ML bundle."
                )
            else:
                self.MAX_NEW_TOKENS = max_new_tokens
        except Exception:
            runtime.close()
            raise

    def close(self) -> None:
        self._coreml_runtime.close()
        self.processor = None

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {}

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size=None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        if input_size not in (None, 224):
            raise ValueError("Kosmos-2 Core ML owns its fixed 224px processor.")
        loaded = ImageLoader.load(image, color_format=color_format)
        query = ", ".join(self.names[index] for index in range(len(self.names)))
        prompt = f"<grounding> Detect: {query}."
        return (
            _CoreMLKosmos2Inputs(loaded, prompt),
            loaded,
            loaded.size,
            1.0,
        )

    def _forward(self, inputs: Any) -> np.ndarray:
        if not isinstance(inputs, _CoreMLKosmos2Inputs):
            raise TypeError("Kosmos-2 Core ML received an invalid request.")
        return self._coreml_runtime.generate(
            inputs.image,
            inputs.prompt,
            max_new_tokens=self.MAX_NEW_TOKENS,
        )

    def export(self, format: str = "coreml", **kwargs) -> str:
        raise NotImplementedError(
            "A Kosmos-2 .coremlvlm bundle is already a deployed artifact."
        )
