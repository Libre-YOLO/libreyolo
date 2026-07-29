"""LibreYOLO wrapper for HuggingFace's SmolVLM2 vision-language models.

SmolVLM2 (HuggingFaceTB, Apache-2.0) is a small general VLM. It follows the same
chat-template plus JSON-bbox output style as the base default (a ``bbox`` key on
a [0, 1] scale), so this family needs no parser override: it works through the
shared base with only the repo table declared. SmolVLM2 is a weak detector
compared with purpose-built grounding models, but it demonstrates that a new
model with no special handling drops straight into the tier.

Its processor depends on ``num2words`` (declared in the ``vlm`` extra).

The exact 500M snapshot also has an experimental stateful Core ML profile.
Unlike native Transformers inference, that route is a portable ``.coremlvlm``
bundle with pinned processor assets and a dedicated host decode runtime.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.general import COCO_CLASSES
from ...utils.image_loader import ImageInput, ImageLoader
from .parsing import build_detection_dict, extract_detections

from .base import LibreVLMModel


class LibreSmolVLM2(LibreVLMModel):
    """SmolVLM2 used as an open-vocabulary detector (base default format)."""

    FAMILY = "smolvlm2"
    FILENAME_PREFIX = "LibreSmolVLM2"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "500m": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    }
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        # Exact Apache-2.0 snapshot covered by the Core ML component,
        # processor, source-value, and portable-bundle contracts.
        "500m": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "2.2b": 512,
        "500m": 512,
    }

    # Apache-2.0 weights: no restrictive-license notice needed.
    _LICENSE_NOTICE = ""

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export the exact 500M snapshot as a portable Core ML VLM bundle."""

        normalized = str(format).strip().lower()
        if normalized not in {"coreml", "coremlvlm"}:
            return super().export(format=format, **kwargs)
        if self.size != "500m":
            raise NotImplementedError(
                "SmolVLM2 Core ML currently supports only the reviewed 500M "
                "profile; the 2.2B architecture has no conversion contract."
            )
        if self.device.type != "cpu":
            raise NotImplementedError(
                "SmolVLM2 Core ML conversion requires a CPU-loaded FP32 model. "
                "Construct it with LibreVLM('smolvlm2-500m', device='cpu')."
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
                "SmolVLM2 Core ML conversion requires an FP32-loaded model; "
                f"found {sorted(str(value) for value in floating_dtypes)}."
            )

        output_path = kwargs.pop("output_path", None)
        output_alias = kwargs.pop("output", None)
        if output_path not in (None, "") and output_alias not in (None, ""):
            if Path(output_path) != Path(output_alias):
                raise ValueError(
                    "Pass only one Core ML VLM destination: output_path= or "
                    "output=."
                )
        context_length = kwargs.pop("context_length", 2048)
        compute_units = str(
            kwargs.pop("compute_units", "cpu_and_gpu")
        ).strip().lower()
        if kwargs:
            raise TypeError(
                "Unsupported or irrelevant SmolVLM2 Core ML export options: "
                + ", ".join(sorted(kwargs))
            )

        from ...backends.coreml_vlm import (
            COREML_VLM_BUNDLE_SUFFIX,
            COREML_VLM_RUNTIME_CONTEXTS,
            build_coreml_vlm_bundle,
        )
        from ...export.coreml_vlm import (
            SMOLVLM2_500M_REVISION,
            export_smolvlm2_500m_coreml_package,
            smolvlm2_500m_coreml_profile,
        )

        if isinstance(context_length, bool) or not isinstance(
            context_length,
            int,
        ):
            raise TypeError("Core ML VLM context_length must be an integer.")
        if context_length not in COREML_VLM_RUNTIME_CONTEXTS:
            raise ValueError(
                "Public SmolVLM2 Core ML export permits only the reviewed "
                f"contexts {list(COREML_VLM_RUNTIME_CONTEXTS)}; got "
                f"{context_length}."
            )
        # Resolve the full profile before any download or conversion side
        # effect, including its finite image/text/cache budget invariants.
        smolvlm2_500m_coreml_profile(context_length)
        valid_compute_units = {
            "all",
            "cpu_and_gpu",
            "cpu_and_ne",
            "cpu_only",
        }
        if compute_units not in valid_compute_units:
            raise ValueError(
                f"Invalid Core ML compute_units {compute_units!r}; expected "
                f"one of {sorted(valid_compute_units)}."
            )

        selected_output = output_path or output_alias
        destination = (
            Path(selected_output)
            if selected_output not in (None, "")
            else Path("weights")
            / f"LibreSmolVLM2-500m-{context_length // 1024}k.coremlvlm"
        )
        if destination.suffix.lower() != COREML_VLM_BUNDLE_SUFFIX:
            raise ValueError(
                "SmolVLM2 Core ML export produces a portable "
                f"{COREML_VLM_BUNDLE_SUFFIX} directory, got {destination}."
            )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite Core ML VLM bundle: {destination}."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        processor_dir = Path(self._ensure_weights())

        # Keep the temporary package beside the destination so the bundle
        # builder can publish it through a same-filesystem no-replace rename.
        with tempfile.TemporaryDirectory(
            prefix=".libreyolo-smolvlm2-coreml-",
            dir=str(destination.parent),
        ) as workspace:
            package = Path(workspace) / "Model.mlpackage"
            export_smolvlm2_500m_coreml_package(
                self.model,
                processor_dir=processor_dir,
                processor_revision=SMOLVLM2_500M_REVISION,
                output_path=package,
                context_length=context_length,
                compute_units=compute_units,
            )
            return build_coreml_vlm_bundle(
                package,
                processor_dir=processor_dir,
                output_path=destination,
                move_model=True,
            )


class _CoreMLSmolInputs:
    """One host-owned image/prompt request accepted by InferenceRunner."""

    def __init__(self, image: Any, prompt: str) -> None:
        self.image = image
        self.prompt = prompt

    def to(self, _device: Any) -> "_CoreMLSmolInputs":
        return self


class CoreMLSmolVLM2(LibreSmolVLM2):
    """Public LibreVLM adapter over a portable SmolVLM2 Core ML bundle."""

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
        compute_units: str = "all",
    ) -> None:
        if str(device).strip().lower() not in {"auto", "cpu"}:
            raise ValueError(
                "A Core ML VLM bundle owns Apple accelerator routing. Pass "
                "device='auto' or 'cpu' and select compute_units separately."
            )
        if isinstance(nb_classes, bool) or not isinstance(nb_classes, int):
            raise TypeError("nb_classes must be an integer.")
        if nb_classes <= 0:
            raise ValueError("nb_classes must be positive.")
        resolved_task = self._resolve_task(task)

        from ...backends.coreml_vlm import CoreMLVLMRuntime

        runtime = CoreMLVLMRuntime(
            bundle_path,
            compute_units=compute_units,
        )
        self.family = self.FAMILY
        self.task = resolved_task
        self.size = "500m"
        self.nb_classes = nb_classes
        self.input_size = 2048
        self.device = torch.device("cpu")
        self.model_path = str(bundle_path)
        self.model = nn.Identity()
        self.processor = runtime.processor
        self._coreml_runtime = runtime
        self._custom_prompt = prompt
        self._graph_runner = None
        self._cuda_graph_mode = None
        self._runner_instance = None
        self.names = {
            index: name
            for index, name in enumerate(
                COCO_CLASSES[:nb_classes]
                if nb_classes <= len(COCO_CLASSES)
                else [
                    *COCO_CLASSES,
                    *(
                        f"class_{index}"
                        for index in range(
                            len(COCO_CLASSES),
                            nb_classes,
                        )
                    ),
                ]
            )
        }
        self._name_to_id = {
            value.strip().lower(): key for key, value in self.names.items()
        }
        if names is not None:
            try:
                self.set_classes(names)
            except Exception:
                runtime.close()
                raise
        if max_new_tokens is None:
            self.MAX_NEW_TOKENS = runtime.profile.max_new_tokens
        elif isinstance(max_new_tokens, bool) or not isinstance(
            max_new_tokens,
            int,
        ):
            runtime.close()
            raise TypeError("max_new_tokens must be an integer.")
        elif max_new_tokens <= 0:
            runtime.close()
            raise ValueError("max_new_tokens must be positive.")
        elif max_new_tokens > runtime.profile.max_new_tokens:
            runtime.close()
            raise ValueError(
                "max_new_tokens exceeds this Core ML VLM profile's reviewed "
                f"limit of {runtime.profile.max_new_tokens}."
            )
        else:
            self.MAX_NEW_TOKENS = max_new_tokens

    def close(self) -> None:
        self._coreml_runtime.close()
        self.processor = None

    def __enter__(self) -> "CoreMLSmolVLM2":
        if self._coreml_runtime.closed:
            raise RuntimeError("Core ML SmolVLM2 runtime is closed.")
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {}

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        if input_size not in (None, self.input_size):
            raise ValueError(
                "Core ML SmolVLM2 uses its fixed 2048x2048 source-image "
                f"contract; got imgsz={input_size}."
            )
        loaded = ImageLoader.load(image, color_format=color_format)
        return (
            _CoreMLSmolInputs(loaded, self._detection_prompt()),
            loaded,
            loaded.size,
            1.0,
        )

    def _forward(self, inputs: _CoreMLSmolInputs) -> str:
        if not isinstance(inputs, _CoreMLSmolInputs):
            raise TypeError("Core ML SmolVLM2 received an invalid request.")
        return self._coreml_runtime.chat(
            inputs.image,
            inputs.prompt,
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
        if not isinstance(output, str):
            raise RuntimeError(
                "Core ML SmolVLM2 runtime must return decoded text."
            )
        items = extract_detections(output)
        return build_detection_dict(
            items,
            self._name_to_id,
            original_size,
            conf_thres=conf_thres,
            max_det=max_det,
            classes=kwargs.get("classes"),
            default_score=self._score_detections(items),
            bbox_key=self.BBOX_KEY,
            coord_divisor=self.COORD_DIVISOR,
            box_format=self.BOX_FORMAT,
            iou_thres=iou_thres,
        )

    def chat(
        self,
        image: ImageInput,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        color_format: str = "auto",
    ) -> str:
        return self._coreml_runtime.chat(
            image,
            prompt,
            max_new_tokens=(
                self.MAX_NEW_TOKENS
                if max_new_tokens is None
                else max_new_tokens
            ),
            color_format=color_format,
        )

    def export(self, format: str = "coreml", **kwargs) -> str:
        raise NotImplementedError(
            "A .coremlvlm bundle is already a deployed Core ML artifact and "
            "cannot be re-exported."
        )
