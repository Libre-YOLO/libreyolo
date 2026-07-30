"""LibreYOLO wrapper for Alibaba's Qwen3-VL vision-language models.

Qwen3-VL is a strong open-weight general VLM with native 2D grounding. For
detection it returns JSON objects with a ``bbox_2d`` key whose coordinates are
on a **0-1000** scale relative to the image (verified empirically: a box at
pixels [240,180,480,420] on an 800x600 image comes back as ~[300,300,600,700]).
That differs from LFM2-VL's ``bbox`` on a [0,1] scale, so this family sets
``BBOX_KEY``/``COORD_DIVISOR`` accordingly; the shared base handles the rest.

Qwen3-VL (Apache-2.0 on the small sizes) loads through the same
``AutoModelForImageTextToText`` path as the rest of the LibreVLM tier.
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


class LibreQwen3VL(LibreVLMModel):
    """Qwen3-VL repurposed as a closed-set object detector."""

    FAMILY = "qwen3vl"
    FILENAME_PREFIX = "LibreQwen3VL"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "2b": "Qwen/Qwen3-VL-2B-Instruct",
        "4b": "Qwen/Qwen3-VL-4B-Instruct",
        "8b": "Qwen/Qwen3-VL-8B-Instruct",
    }
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        # Exact Apache-2.0 snapshot under Core ML feasibility review.
        "2b": "89644892e4d85e24eaac8bacfd4f463576704203",
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

    # Apache-2.0 weights: no restrictive-license notice needed.
    _LICENSE_NOTICE = ""

    def _format_detection_prompt(self, labels: str) -> str:
        return (
            f"Detect all instances of: {labels}. "
            "Output the result as a JSON array, one object per instance: "
            '[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
            "Only include objects that are actually visible; if there are none, "
            "respond with an empty array []."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export the pinned 2B checkpoint as a bounded Core ML bundle."""

        normalized = str(format).strip().lower()
        if normalized not in {"coreml", "coremlvlm"}:
            return super().export(format=format, **kwargs)
        if self.size != "2b":
            raise NotImplementedError(
                "Qwen3-VL Core ML supports only the pinned 2B checkpoint."
            )
        if self.device.type != "cpu":
            raise NotImplementedError(
                "Qwen3-VL Core ML conversion requires a CPU-loaded FP32 model."
            )
        output_path = kwargs.pop("output_path", None)
        output_alias = kwargs.pop("output", None)
        if output_path not in (None, "") and output_alias not in (None, ""):
            if Path(output_path) != Path(output_alias):
                raise ValueError(
                    "Pass only one Qwen3-VL Core ML destination: output_path= "
                    "or output=."
                )
        context_length = kwargs.pop("context_length", 512)
        compute_units = kwargs.pop("compute_units", "validated")
        if kwargs:
            raise TypeError(
                "Unsupported or irrelevant Qwen3-VL Core ML export options: "
                + ", ".join(sorted(kwargs))
            )

        from ...backends.coreml_qwen3vl import (
            COREML_QWEN3VL_BUNDLE_SUFFIX,
        )
        from ...export.coreml_qwen3vl import (
            QWEN3VL_COREML_CONTEXT_LENGTH,
            build_qwen3vl_coreml_bundle,
            export_qwen3vl_coreml_components,
            resolve_qwen3vl_coreml_compute_units,
            validate_qwen3vl_processor_assets,
            validate_qwen3vl_source_model,
            validate_qwen3vl_weight_asset,
        )

        if isinstance(context_length, bool) or not isinstance(
            context_length,
            int,
        ):
            raise TypeError("Qwen3-VL Core ML context_length must be an integer.")
        if context_length != QWEN3VL_COREML_CONTEXT_LENGTH:
            raise ValueError(
                "Qwen3-VL Core ML currently supports only context_length="
                f"{QWEN3VL_COREML_CONTEXT_LENGTH}."
            )
        compute_units = resolve_qwen3vl_coreml_compute_units(compute_units)
        snapshot_dir = Path(self._ensure_weights())
        validate_qwen3vl_processor_assets(snapshot_dir)
        validate_qwen3vl_weight_asset(snapshot_dir)
        validate_qwen3vl_source_model(self.model)

        destination_value = output_path or output_alias
        destination = (
            Path(destination_value)
            if destination_value not in (None, "")
            else Path("weights") / "LibreQwen3VL-2b-448-512.coremlvlm"
        )
        if destination.suffix.lower() != COREML_QWEN3VL_BUNDLE_SUFFIX:
            raise ValueError(
                "Qwen3-VL Core ML export produces a portable "
                f"{COREML_QWEN3VL_BUNDLE_SUFFIX} directory."
            )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite Qwen3-VL Core ML bundle: {destination}."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".libreyolo-qwen3vl-coreml-",
            dir=str(destination.parent),
        ) as workspace:
            component_dir = Path(workspace) / "Components"
            export_qwen3vl_coreml_components(
                self.model,
                checkpoint_dir=snapshot_dir,
                output_dir=component_dir,
                compute_units=compute_units,
            )
            return build_qwen3vl_coreml_bundle(
                component_dir,
                processor_dir=snapshot_dir,
                output_path=destination,
            )


class _CoreMLQwen3VLInputs:
    def __init__(self, image: Any, prompt: str) -> None:
        self.image = image
        self.prompt = prompt

    def to(self, _device: Any) -> "_CoreMLQwen3VLInputs":
        return self


class CoreMLQwen3VL(LibreQwen3VL):
    """Public LibreVLM adapter over a portable Qwen3-VL Core ML bundle."""

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
                "A Qwen3-VL Core ML bundle owns Apple routing; use "
                "device='auto' or 'cpu'."
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

        from ...backends.coreml_qwen3vl import CoreMLQwen3VLRuntime

        runtime = CoreMLQwen3VLRuntime(
            bundle_path,
            compute_units=compute_units,
        )
        self.family = self.FAMILY
        self.task = resolved_task
        self.size = "2b"
        self.nb_classes = len(initial_names)
        self.input_size = 448
        self.device = torch.device("cpu")
        self.model_path = str(bundle_path)
        self.model = nn.Identity()
        self.processor = runtime.processor
        self._model_dtype = torch.float32
        self._coreml_runtime = runtime
        self._custom_prompt = prompt
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
                max_new_tokens,
                int,
            ):
                raise TypeError("max_new_tokens must be an integer.")
            elif max_new_tokens <= 0 or max_new_tokens > runtime.max_new_tokens:
                raise ValueError(
                    "max_new_tokens must be between 1 and "
                    f"{runtime.max_new_tokens} for this Qwen3-VL bundle."
                )
            else:
                self.MAX_NEW_TOKENS = max_new_tokens
        except Exception:
            runtime.close()
            raise

    def close(self) -> None:
        self._coreml_runtime.close()
        self.processor = None

    def __enter__(self) -> "CoreMLQwen3VL":
        if self._coreml_runtime.closed:
            raise RuntimeError("Qwen3-VL Core ML runtime is closed.")
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
                "Qwen3-VL Core ML owns its fixed 448px image contract."
            )
        loaded = ImageLoader.load(image, color_format=color_format)
        return (
            _CoreMLQwen3VLInputs(loaded, self._detection_prompt()),
            loaded,
            loaded.size,
            1.0,
        )

    def _forward(self, inputs: Any) -> np.ndarray:
        if not isinstance(inputs, _CoreMLQwen3VLInputs):
            raise TypeError("Qwen3-VL Core ML received an invalid request.")
        return self._coreml_runtime.generate(
            inputs.image,
            inputs.prompt,
            max_new_tokens=self.MAX_NEW_TOKENS,
        )

    def chat(
        self,
        image: ImageInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        color_format: str = "auto",
    ) -> str:
        loaded = ImageLoader.load(image, color_format=color_format)
        budget = (
            self.MAX_NEW_TOKENS
            if max_new_tokens is None
            else max_new_tokens
        )
        generated = self._coreml_runtime.generate(
            loaded,
            str(prompt),
            max_new_tokens=budget,
        )
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]

    def export(self, format: str = "coreml", **kwargs) -> str:
        raise NotImplementedError(
            "A Qwen3-VL .coremlvlm bundle is already a deployed artifact."
        )
