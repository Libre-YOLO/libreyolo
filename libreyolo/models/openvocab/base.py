"""Base class for open-vocabulary detector wrappers.

These discriminative Hugging Face detectors take an image plus a text
vocabulary and return real detection scores. They are not LibreYOLO checkpoint
families, so this tier intentionally does not define ``can_load`` and stays out
of the ``LibreYOLO(...)`` state-dict factory.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from types import GeneratorType
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.results import Results
from ..base.model import BaseModel

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "LibreOpenVocab models require the 'openvocab' extra. Install with:\n"
    "    pip install 'libreyolo[openvocab]'"
)
_SNAPSHOT_COMPLETE_MARKER = ".libreyolo_snapshot_complete"
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ARTICLES = {"a", "an", "the"}


def _normalize_label(text: str) -> str:
    """Lowercase, strip punctuation-like whitespace, and remove leading articles."""
    tokens = _label_tokens(text)
    return " ".join(tokens)


def _label_tokens(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(str(text).lower())
    while tokens and tokens[0] in _ARTICLES:
        tokens.pop(0)
    return tokens


def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


class LibreOpenVocabDetector(BaseModel):
    """Shared plumbing for discriminative open-vocabulary detectors."""

    FAMILY: ClassVar[str] = ""
    FILENAME_PREFIX: ClassVar[str] = ""
    HF_REPOS: ClassVar[Dict[str, str]] = {}
    HF_REVISIONS: ClassVar[Dict[str, str]] = {}
    INPUT_SIZES: ClassVar[Dict[str, int]] = {}
    SUPPORTED_TASKS: ClassVar[tuple[str, ...]] = ("detect",)
    DEFAULT_TASK: ClassVar[str] = "detect"

    DEFAULT_CONF: ClassVar[float] = 0.25
    DEFAULT_TEXT_THRESHOLD: ClassVar[float | None] = None
    SUPPORTS_TEXT_THRESHOLD: ClassVar[bool] = False
    # Families whose processor runs its own NMS set their default threshold
    # here, and honour iou=. ``None`` means the family suppresses nothing, so
    # iou= has no meaning and is warned about instead.
    NMS_THRESHOLD: ClassVar[float | None] = None

    TTA_ENABLED: ClassVar[bool] = False
    SUPPORTS_BATCHED_PREDICT: ClassVar[bool] = False

    _LICENSE_NOTICE: ClassVar[str] = ""
    _LICENSE_NOTICE_SHOWN: ClassVar[bool] = False
    SNAPSHOT_IGNORE_PATTERNS: ClassVar[tuple[str, ...]] = (
        "*.bin",
        "*.bin.index.json",
        "*.h5",
        "*.msgpack",
        "*.onnx",
        "*.ot",
        "*.tflite",
        "onnx/*",
    )

    def __init__(
        self,
        size: str,
        *,
        nb_classes: int = 80,
        names: Optional[list[str]] = None,
        device: str = "auto",
        task: str | None = None,
        text_threshold: float | None = None,
        **kwargs,
    ):
        if size not in self.HF_REPOS:
            raise ValueError(
                f"Invalid size {size!r} for {type(self).__name__}. "
                f"Must be one of: {', '.join(self.HF_REPOS)}"
            )
        if text_threshold is not None and not self.SUPPORTS_TEXT_THRESHOLD:
            raise TypeError(f"{type(self).__name__} does not support text_threshold.")
        self._text_threshold = (
            self.DEFAULT_TEXT_THRESHOLD
            if text_threshold is None
            else float(text_threshold)
        )
        super().__init__(
            model_path=self.HF_REPOS[size],
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task,
            **kwargs,
        )
        if names is not None:
            self.set_classes(names)
        else:
            self._refresh_name_index()
        self.model.eval()

    def __call__(
        self,
        source=None,
        *,
        conf: float | None = None,
        text_threshold: float | None = None,
        **kwargs,
    ) -> Union[Results, List[Results], GeneratorType]:
        """Run prediction with family-specific defaults.

        ``InferenceRunner`` owns predict kwargs and defaults, so this tier
        injects its default confidence here and keeps Grounding DINO's
        ``text_threshold`` off the runner kwarg path.
        """
        if conf is None:
            conf = self.DEFAULT_CONF
        if kwargs.get("imgsz") is not None:
            raise ValueError(
                f"{type(self).__name__} does not support imgsz=: the transformers "
                "processor owns resizing for this open-vocabulary detector."
            )
        if kwargs.get("augment", False):
            raise ValueError(
                f"{type(self).__name__} does not support augment=True; "
                "test-time augmentation is out of scope for this tier."
            )
        if "iou" in kwargs:
            if self.NMS_THRESHOLD is None:
                warnings.warn(
                    f"{type(self).__name__} does not run NMS; iou= is "
                    "accepted for API compatibility but ignored.",
                    stacklevel=2,
                )
        elif self.NMS_THRESHOLD is not None:
            # predict() defaults iou=0.45, which is not this family's NMS
            # default. Inject the family's own, exactly as conf= is injected
            # above, so an unset iou= keeps the upstream suppression behaviour.
            kwargs["iou"] = self.NMS_THRESHOLD
        if text_threshold is None:
            return super().__call__(source, conf=conf, **kwargs)
        if not self.SUPPORTS_TEXT_THRESHOLD:
            raise TypeError(f"{type(self).__name__} does not support text_threshold.")
        if kwargs.get("stream", False):
            raise ValueError(
                "Per-call text_threshold= is not supported with stream=True because "
                "streamed results are generated lazily. Pass text_threshold= to the "
                "model constructor for a persistent streaming threshold."
            )
        previous = self._text_threshold
        self._text_threshold = float(text_threshold)
        try:
            result = super().__call__(source, conf=conf, **kwargs)
        except Exception:
            self._text_threshold = previous
            raise
        if isinstance(result, GeneratorType):
            return self._restore_text_threshold_after_stream(result, previous)
        self._text_threshold = previous
        return result

    def _restore_text_threshold_after_stream(self, stream, previous):
        try:
            yield from stream
        finally:
            self._text_threshold = previous

    # ---------------------------------------------------------------------
    # Open-vocabulary API
    # ---------------------------------------------------------------------

    def set_classes(self, classes: list[str]) -> "LibreOpenVocabDetector":
        """Set the sticky class vocabulary for later ``predict()`` calls."""
        if isinstance(classes, str) or not isinstance(classes, (list, tuple)):
            raise TypeError(
                "set_classes() expects a list/tuple of label strings, "
                f'e.g. ["boat"], not {type(classes).__name__}.'
            )
        if not classes:
            raise ValueError("set_classes() requires a non-empty list of labels.")
        names = [str(c) for c in classes]
        keys = [_normalize_label(name) for name in names]
        if any(not key for key in keys):
            raise ValueError("set_classes() labels must not be blank.")
        if len(keys) != len(set(keys)):
            raise ValueError("set_classes() labels must be unique case-insensitively.")
        self.names = {i: name for i, name in enumerate(names)}
        self.nb_classes = len(self.names)
        self._refresh_name_index()
        return self

    def _refresh_name_index(self) -> None:
        self._name_to_id = {
            _normalize_label(name): class_id for class_id, name in self.names.items()
        }

    # ---------------------------------------------------------------------
    # Weight acquisition
    # ---------------------------------------------------------------------

    @staticmethod
    def _snapshot_complete(
        local_dir: Path,
        *,
        repo: str | None = None,
        revision: str | None = None,
    ) -> bool:
        marker_path = local_dir / _SNAPSHOT_COMPLETE_MARKER
        if not marker_path.exists():
            return False
        if repo is not None or revision is not None:
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return False
            if repo is not None and marker.get("repo") != repo:
                return False
            if revision is not None and marker.get("revision") != revision:
                return False
        if not (local_dir / "config.json").exists():
            return False
        for index_name in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ):
            index = local_dir / index_name
            if index.exists():
                try:
                    weight_map = json.loads(index.read_text(encoding="utf-8")).get(
                        "weight_map", {}
                    )
                except (ValueError, OSError):
                    return False
                shards = set(weight_map.values())
                return bool(shards) and all(
                    (local_dir / shard).exists() for shard in shards
                )
        return any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))

    @classmethod
    def _notify_license_once(cls) -> None:
        if cls._LICENSE_NOTICE_SHOWN or not cls._LICENSE_NOTICE:
            return
        cls._LICENSE_NOTICE_SHOWN = True
        logger.warning(cls._LICENSE_NOTICE)

    def _ensure_weights(self) -> str:
        repo = self.HF_REPOS[self.size]
        revision = self.HF_REVISIONS.get(self.size)
        local_dir = Path("weights") / f"{self.FILENAME_PREFIX}{self.size}"
        self._notify_license_once()
        if self._snapshot_complete(
            local_dir,
            repo=repo,
            revision=revision,
        ):
            return str(local_dir)
        try:
            from huggingface_hub import snapshot_download
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        source = f"{repo}@{revision}" if revision else repo
        logger.info(
            "Downloading %s weights from %s -> %s ...",
            self.FAMILY,
            source,
            local_dir,
        )
        download_kwargs = {}
        if revision is not None:
            download_kwargs["revision"] = revision
        snapshot_download(
            repo,
            local_dir=str(local_dir),
            ignore_patterns=list(self.SNAPSHOT_IGNORE_PATTERNS),
            **download_kwargs,
        )
        (local_dir / _SNAPSHOT_COMPLETE_MARKER).write_text(
            json.dumps({"repo": repo, "revision": revision}) + "\n",
            encoding="utf-8",
        )
        if not self._snapshot_complete(
            local_dir,
            repo=repo,
            revision=revision,
        ):
            (local_dir / _SNAPSHOT_COMPLETE_MARKER).unlink(missing_ok=True)
            raise FileNotFoundError(
                f"Downloaded snapshot for {repo} is missing config or weight files "
                f"in {local_dir}."
            )
        return str(local_dir)

    def _resolve_dtype(self) -> torch.dtype:
        return torch.float32

    def _load_pretrained(self, snapshot_dir: str):
        raise NotImplementedError

    def _init_model(self) -> nn.Module:
        snapshot_dir = self._ensure_weights()
        model, processor = self._load_pretrained(snapshot_dir)
        self.processor = processor
        self._model_dtype = next(model.parameters()).dtype
        return model

    # ---------------------------------------------------------------------
    # InferenceRunner hooks
    # ---------------------------------------------------------------------

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {name: module for name, module in self.model.named_modules() if name}

    @staticmethod
    def _get_preprocess_numpy():
        raise NotImplementedError(
            "Open-vocabulary detector families preprocess through transformers."
        )

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[Any, Any, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        inputs = self._build_inputs(img)
        return inputs, img, img.size, 1.0

    def _build_inputs(self, img: Any) -> Any:
        raise NotImplementedError

    def _forward(self, inputs: Any) -> Any:
        inputs = self._prepare_inputs(inputs)
        return self.model(**inputs)

    def _prepare_inputs(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        return self._cast_inputs(inputs)

    def _cast_inputs(self, inputs: Any) -> Any:
        dtype = getattr(self, "_model_dtype", None)
        if dtype is None:
            return inputs

        def cast(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
                return value.to(dtype=dtype) if value.is_floating_point() else value
            if isinstance(value, MutableMapping):
                for key, nested in value.items():
                    value[key] = cast(nested)
                return value
            if isinstance(value, tuple):
                return tuple(cast(nested) for nested in value)
            if isinstance(value, list):
                return [cast(nested) for nested in value]
            return value

        return cast(inputs)

    def _detections_to_dict(
        self,
        boxes: Any,
        scores: Any,
        class_ids: Any,
        *,
        conf_thres: float,
        original_size: Tuple[int, int],
        max_det: int,
        classes: Optional[List[int]] = None,
    ) -> Dict:
        boxes_t = self._as_cpu_tensor(boxes, torch.float32).reshape(-1, 4)
        scores_t = self._as_cpu_tensor(scores, torch.float32).reshape(-1)
        class_t = self._as_cpu_tensor(class_ids, torch.int64).reshape(-1)
        n = min(boxes_t.shape[0], scores_t.shape[0], class_t.shape[0])
        boxes_t, scores_t, class_t = boxes_t[:n], scores_t[:n], class_t[:n]
        if n == 0:
            return self._empty_detections()

        width, height = original_size
        finite = torch.isfinite(boxes_t).all(dim=1) & torch.isfinite(scores_t)
        keep = finite & (scores_t >= float(conf_thres))
        if classes is not None:
            allowed = torch.as_tensor(classes, dtype=torch.int64)
            keep &= (class_t[:, None] == allowed[None, :]).any(dim=1)
        boxes_t, scores_t, class_t = boxes_t[keep], scores_t[keep], class_t[keep]
        if boxes_t.numel() == 0:
            return self._empty_detections()

        boxes_t[:, 0::2] = boxes_t[:, 0::2].clamp(0, float(width))
        boxes_t[:, 1::2] = boxes_t[:, 1::2].clamp(0, float(height))
        valid = (boxes_t[:, 2] > boxes_t[:, 0]) & (boxes_t[:, 3] > boxes_t[:, 1])
        boxes_t, scores_t, class_t = boxes_t[valid], scores_t[valid], class_t[valid]
        if boxes_t.numel() == 0:
            return self._empty_detections()

        order = scores_t.argsort(descending=True)[:max_det]
        boxes_t, scores_t, class_t = boxes_t[order], scores_t[order], class_t[order]
        return {
            "boxes": boxes_t,
            "scores": scores_t,
            "classes": class_t,
            "num_detections": int(boxes_t.shape[0]),
        }

    @staticmethod
    def _as_cpu_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", dtype=dtype)
        return torch.as_tensor(value, dtype=dtype)

    @staticmethod
    def _empty_detections() -> Dict:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "classes": torch.zeros((0,), dtype=torch.int64),
            "num_detections": 0,
        }

    # ---------------------------------------------------------------------
    # Label -> class-id mapping (shared by direct-label families)
    # ---------------------------------------------------------------------

    def _labels_to_class_ids(self, labels: Any) -> torch.Tensor:
        """Map a processor's per-detection labels to vocabulary class ids.

        Handles both integer query indices and decoded label strings. Unknown
        or out-of-range labels map to ``-1`` so the caller can drop them.
        """
        if isinstance(labels, torch.Tensor):
            class_ids = labels.detach().cpu().long().reshape(-1)
        else:
            values = list(labels)
            if not values:
                return torch.zeros((0,), dtype=torch.int64)
            if all(isinstance(value, (int, float)) for value in values):
                class_ids = torch.as_tensor(values, dtype=torch.int64).reshape(-1)
            else:
                mapped = [self._text_label_to_class_id(str(value)) for value in values]
                return torch.as_tensor(
                    [-1 if class_id is None else class_id for class_id in mapped],
                    dtype=torch.int64,
                )
        valid = (class_ids >= 0) & (class_ids < len(self.names))
        return torch.where(valid, class_ids, torch.full_like(class_ids, -1))

    def _text_label_to_class_id(self, text: str) -> Optional[int]:
        phrase = _label_tokens(text)
        matches = []
        for class_id, name in self.names.items():
            label = _label_tokens(name)
            if phrase == label:
                return class_id
            if _contains_subsequence(phrase, label) or _contains_subsequence(
                label, phrase
            ):
                matches.append(class_id)
        return matches[0] if len(matches) == 1 else None

    # ---------------------------------------------------------------------
    # Out of scope for v1
    # ---------------------------------------------------------------------

    def track(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} tracking is out of scope for the v1 "
            "open-vocabulary detector tier. Use predict() per frame."
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            f"Training is out of scope for {type(self).__name__} in LibreYOLO. "
            "Fine-tune the model upstream and load the resulting weights."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            f"Dataset validation is not supported for {type(self).__name__} in v1. "
            "Open-vocabulary validation needs a dedicated validator because the "
            "standard detection validator calls _forward() with image tensors, "
            "while this tier requires text-conditioned transformers inputs."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} export is out of scope for the inference-only "
            "open-vocabulary detector tier."
        )


__all__ = [
    "LibreOpenVocabDetector",
    "_INSTALL_HINT",
    "_label_tokens",
    "_normalize_label",
    "_contains_subsequence",
]
