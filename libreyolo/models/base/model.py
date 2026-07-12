"""
Base model class for LibreYOLO model wrappers.

Provides shared functionality for all YOLO model variants.
"""

from __future__ import annotations

import functools
import inspect
import logging
import random
import re
from abc import ABC, abstractmethod
from numbers import Integral
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import batched_nms

from ...tasks import (
    detect_task_suffix,
    normalize_task,
    resolve_task,
    task_suffix_pattern,
    task_to_suffix,
)
from ...training.config import TrainConfig, load_train_cfg
from ...utils.general import COCO_CLASSES
from ...utils.image_loader import ImageInput
from ...utils.logging import ensure_default_logging
from ...utils.model_info import build_model_info, format_model_info
from ...utils.results import Results
from ...utils.serialization import (
    LEGACY_CHECKPOINT_LOAD_POLICY,
    NATIVE_CHECKPOINT_LOAD_POLICY,
    CheckpointLoadPolicy,
    CheckpointLoadReport,
    load_state_dict_checked,
    load_untrusted_torch_file,
    normalize_checkpoint_names,
    parse_checkpoint_metadata_for_load,
    validate_checkpoint_metadata,
)
from ...validation.preprocessors import StandardValPreprocessor

logger = logging.getLogger(__name__)


# Keys that come from the model wrapper instance (``self.size``,
# ``self.nb_classes``) and are passed explicitly to the family trainer. If a
# cfg yaml carries them too, they would collide with the explicit kwargs and
# raise ``TypeError: got multiple values``. ``TrainConfig.to_yaml()`` writes
# both, so a user-generated starter yaml hits this naturally.
_WRAPPER_OWNED_CFG_KEYS = frozenset({"size", "num_classes"})


def _wrap_train_with_cfg(train_fn: Callable) -> Callable:
    """Decorate a family ``train()`` method to accept ``cfg='path/to/yaml'``.

    Loads the yaml as a dict and merges it into kwargs with user-provided
    kwargs winning. Keys consumed by positional args (and a small set of
    wrapper-owned keys like ``size``/``num_classes``) are dropped from the
    cfg dict to avoid ``TypeError: got multiple values``.
    """
    sig = inspect.signature(train_fn)
    pos_names = [
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if pos_names and pos_names[0] == "self":
        pos_names = pos_names[1:]

    @functools.wraps(train_fn)
    def wrapper(self, *args, cfg=None, **user_kwargs):
        explicit_override = user_kwargs.pop(
            "_libreyolo_explicit_train_keys",
            None,
        )

        def call_train(
            call_kwargs: Dict[str, Any],
            explicit_keys: set[str],
        ):
            bound = sig.bind_partial(self, *args, **call_kwargs)
            bound.apply_defaults()
            imgsz = bound.arguments.get("imgsz")
            variadic_kwargs = bound.arguments.get("kwargs")
            if imgsz is None and isinstance(variadic_kwargs, dict):
                imgsz = variadic_kwargs.get("imgsz")
            if imgsz is not None:
                validate_input_size = getattr(
                    type(self),
                    "_validate_input_size",
                    None,
                )
                validated = (
                    validate_input_size(self, imgsz, context="train")
                    if callable(validate_input_size)
                    else imgsz
                )
                if "imgsz" in call_kwargs:
                    call_kwargs = dict(call_kwargs)
                    call_kwargs["imgsz"] = validated

            seed = bound.arguments.get("seed")
            if seed is None and isinstance(variadic_kwargs, dict):
                seed = variadic_kwargs.get("seed")
            if seed is None:
                config_class = getattr(self, "TRAIN_CONFIG", None)
                if callable(config_class):
                    seed = getattr(config_class(), "seed", None)
            if seed is not None:
                seed = int(seed)
                if seed >= 0:
                    random.seed(seed)
                    np.random.seed(seed % 2**32)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

            aliases = {
                "batch_size": "batch",
                "checkpoint_interval": "save_period",
                "early_stopping_patience": "patience",
                "img_size": "imgsz",
                "lr": "lr0",
                "num_workers": "workers",
                "use_ema": "ema",
            }
            normalized_explicit = {aliases.get(key, key) for key in explicit_keys}
            missing = object()
            previous = getattr(self, "_active_train_explicit_keys", missing)
            self._active_train_explicit_keys = normalized_explicit
            try:
                return train_fn(self, *args, **call_kwargs)
            finally:
                if previous is missing:
                    delattr(self, "_active_train_explicit_keys")
                else:
                    self._active_train_explicit_keys = previous

        positional_explicit = set(pos_names[: len(args)])
        if cfg is None:
            explicit_keys = (
                set(explicit_override)
                if explicit_override is not None
                else positional_explicit | set(user_kwargs)
            )
            return call_train(user_kwargs, explicit_keys)
        cfg_kwargs = load_train_cfg(cfg)
        consumed = set(pos_names[: len(args)]) | _WRAPPER_OWNED_CFG_KEYS
        merged = {k: v for k, v in cfg_kwargs.items() if k not in consumed}
        merged.update(user_kwargs)
        return call_train(
            merged,
            positional_explicit | set(cfg_kwargs) | set(user_kwargs),
        )

    wrapper._libreyolo_cfg_wrapped = True  # type: ignore[attr-defined]
    return wrapper


class BaseModel(ABC):
    """Abstract base class for LibreYOLO model wrappers.

    Subclasses must implement the abstract methods to provide model-specific
    behavior for initialization, forward pass, and postprocessing.

    Class constants subclasses should set:
        FAMILY: Model family identifier (e.g. "yolox").
        FILENAME_PREFIX: Prefix for weight filenames (e.g. "LibreYOLOX").
        INPUT_SIZES: Mapping of size code to input resolution.
        TRAIN_CONFIG: TrainConfig subclass with family-specific defaults.
        val_preprocessor_class: Preprocessor class for validation.
        validator_class: Override the validator used by val(); defaults to task-based dispatch.
    """

    # Class-level model metadata — subclasses override these
    FAMILY: ClassVar[str] = ""
    FILENAME_PREFIX: ClassVar[str] = ""
    WEIGHT_EXT: ClassVar[str] = ".pt"
    INPUT_SIZES: ClassVar[dict[str, int]] = {}
    SUPPORTED_TASKS: ClassVar[tuple[str, ...]] = ("detect",)
    DEFAULT_TASK: ClassVar[str] = "detect"
    # When True, the task suffix is mandatory in weight filenames (e.g. a
    # classify-only family requires ``-cls``); detect families leave it optional.
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = False
    TASK_INPUT_SIZES: ClassVar[dict[str, dict[str, int]]] = {}
    # User-facing input-size constraints. Fixed families accept only the
    # instance-native canvas; dynamic families may declare a stride/divisor and
    # minimum safe canvas. Family-local semantic/depth divisor attributes are
    # also honored for backward compatibility.
    INPUT_SIZE_FIXED: ClassVar[bool] = False
    INPUT_SIZE_DIVISOR: ClassVar[int | None] = None
    INPUT_SIZE_MIN: ClassVar[int] = 1
    SUPPORTS_RECTANGULAR_INPUT: ClassVar[bool] = False
    # Rare fixed-canvas families with separately published native resolutions
    # may let validated checkpoint metadata replace the construction default.
    CHECKPOINT_INPUT_SIZE_OVERRIDE: ClassVar[bool] = False
    TRAIN_CONFIG: ClassVar[Optional[type[TrainConfig]]] = None
    val_preprocessor_class = StandardValPreprocessor
    validator_class: ClassVar[Optional[type]] = None
    EXPERIMENTAL_WEIGHT_FILENAMES: ClassVar[frozenset[str]] = frozenset()
    # Dataset-variant weight suffixes (e.g. "visdrone" accepts
    # ``LibreYOLO9P2s-visdrone.pt``). Families that publish checkpoints
    # trained on a non-default dataset opt in; the variant stays part of the
    # Hugging Face repo name in ``get_download_url``.
    WEIGHT_VARIANTS: ClassVar[tuple[str, ...]] = ()

    # Batched-predict policy: True when ``_preprocess`` yields stackable
    # (1, C, H, W) tensors and every tensor in the ``_forward`` output keeps
    # a leading batch dim (the contract batched validation already relies
    # on). Set False where that does not hold (e.g. generative VLMs).
    SUPPORTS_BATCHED_PREDICT: ClassVar[bool] = True

    # TTA policy — subclasses may override
    TTA_ENABLED: ClassVar[bool] = True
    # True for families that resize to a fixed square regardless of input size
    # (DETR-style). Multi-scale TTA is a no-op for them; only flip adds value.
    TTA_FIXED_SIZE: ClassVar[bool] = False
    # Scale factors applied to the PIL image before each TTA pass.
    # Each scale × 2 flips = N passes. Default (1.0,) is flip-only.
    # Override with e.g. (0.83, 1.0, 1.33) for 6-pass multi-scale TTA.
    # Ignored when TTA_FIXED_SIZE is True.
    TTA_SCALES: ClassVar[Tuple[float, ...]] = (1.0,)

    # Model registry — auto-populated by __init_subclass__
    _registry: ClassVar[List[Type["BaseModel"]]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if (
            hasattr(cls, "can_load")
            and not getattr(cls.can_load, "__isabstractmethod__", False)
            and cls not in BaseModel._registry
        ):
            BaseModel._registry.append(cls)

        if "train" in cls.__dict__ and not getattr(
            cls.train, "_libreyolo_cfg_wrapped", False
        ):
            cls.train = _wrap_train_with_cfg(cls.train)

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        model_path: Union[str, dict, None],
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ):
        ensure_default_logging()
        self.family = self.FAMILY
        self.task = self._resolve_task(task)
        valid_sizes = self._get_valid_sizes()
        if size not in valid_sizes:
            raise ValueError(
                f"Invalid size: {size}. Must be one of: {', '.join(valid_sizes)}"
            )

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            if isinstance(device, int):
                device = f"cuda:{device}"
            if isinstance(device, str) and device.isdigit():
                device = f"cuda:{device}"
            self.device = torch.device(device)

        self.size = size
        self.nb_classes = nb_classes
        self.input_size = self._get_task_input_sizes()[size]

        if nb_classes == 80:
            self.names: Dict[int, str] = {i: n for i, n in enumerate(COCO_CLASSES)}
        else:
            self.names: Dict[int, str] = {i: f"class_{i}" for i in range(nb_classes)}

        for key, value in kwargs.items():
            setattr(self, key, value)

        # Resolve bare filenames (e.g. "LibreYOLOXn.pt") to weights/ directory
        # so direct instantiation works the same as the factory.
        if isinstance(model_path, str):
            model_path = self._resolve_weights_path(model_path)

        # Signal _init_model that weights will be loaded immediately after, so
        # subclasses can skip pretrained backbone downloads that would be wasted.
        self._loading_from_weights = isinstance(model_path, (str, Path, dict))
        try:
            self.model = self._init_model()
        finally:
            self._loading_from_weights = False

        if model_path is None:
            self.model_path = None
        elif isinstance(model_path, dict):
            self.model_path = None
            self._apply_loaded_checkpoint(
                model_path,
                context=f"{self.FAMILY or type(self).__name__} in-memory checkpoint",
            )
        else:
            self.model_path = model_path

        if model_path is None:
            self.model.train()
        else:
            self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _resolve_weights_path(model_path: str) -> str:
        """Resolve bare filenames (e.g. ``LibreYOLOXn.pt``) to ``weights/`` dir."""
        path = Path(model_path)
        if path.parent == Path(".") and not model_path.startswith(("./", "../")):
            weights_path = Path("weights") / path.name
            if weights_path.exists():
                return str(weights_path)
            if path.exists():
                return str(path)
            return str(weights_path)
        return model_path

    # =========================================================================
    # Abstract interface — subclasses must implement
    # =========================================================================

    @abstractmethod
    def _init_model(self) -> nn.Module:
        """Initialize and return the neural network model."""
        pass

    @abstractmethod
    def _get_available_layers(self) -> Dict[str, nn.Module]:
        """Return mapping of layer names to module objects."""
        pass

    @staticmethod
    @abstractmethod
    def _get_preprocess_numpy():
        """Return the ``preprocess_numpy(img_rgb_hwc, input_size)`` callable for this model family."""
        pass

    @abstractmethod
    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        """Preprocess image for inference.

        Returns:
            Tuple of (input_tensor, original_image, original_size, ratio).
        """
        pass

    @abstractmethod
    def _forward(self, input_tensor: torch.Tensor) -> Any:
        """Run model forward pass."""
        pass

    @abstractmethod
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
        """Postprocess model output to detections."""
        pass

    # =========================================================================
    # Concrete defaults — subclasses may override
    # =========================================================================

    def get_distill_config(self) -> Dict:
        """Return distillation config for this model instance.

        Returns:
            Dict with keys:
                - tap_points: List[str] — module paths for forward hooks
                - channels: List[int] — channel dimensions per tap point
                - strides: List[int] — spatial strides per tap point

        Subclasses that support distillation must override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_distill_config(). "
            f"Distillation is not yet supported for the '{self.FAMILY}' family."
        )

    def _get_valid_sizes(self) -> List[str]:
        return list(self._get_task_input_sizes().keys())

    @classmethod
    def _supported_tasks(cls) -> tuple[str, ...]:
        return tuple(normalize_task(task) for task in cls.SUPPORTED_TASKS)

    def _resolve_task(self, task: str | None) -> str:
        return resolve_task(
            explicit_task=task,
            default_task=self.DEFAULT_TASK,
            supported_tasks=self.SUPPORTED_TASKS,
        )

    def _get_task_input_sizes(self) -> dict[str, int]:
        if self.TASK_INPUT_SIZES:
            return self.TASK_INPUT_SIZES.get(self.task, self.INPUT_SIZES)
        return self.INPUT_SIZES

    def _get_model_name(self) -> str:
        return self.FAMILY

    def _get_input_size(self) -> int:
        return self.input_size

    def _validate_input_size(
        self,
        imgsz: Any,
        *,
        context: str,
        allow_fixed_override: bool = False,
    ) -> int:
        """Validate one square runtime canvas before any expensive work."""
        family = self.FAMILY or type(self).__name__
        if isinstance(imgsz, bool) or not isinstance(imgsz, Integral):
            raise ValueError(
                f"{family} {context} imgsz must be a positive integer, got "
                f"{imgsz!r}"
            )
        value = int(imgsz)
        if value <= 0:
            raise ValueError(
                f"{family} {context} imgsz must be a positive integer, got "
                f"{value}"
            )

        native = int(self._get_input_size())
        if self.INPUT_SIZE_FIXED and value != native and not allow_fixed_override:
            raise ValueError(
                f"{family} only supports imgsz={native}; {context} requires "
                f"imgsz={native}, got {value}"
            )

        minimum = int(getattr(self, "INPUT_SIZE_MIN", 1) or 1)
        if value < minimum:
            raise ValueError(
                f"{family} {context} imgsz must be at least {minimum}; got {value}"
            )

        divisor = getattr(self, "INPUT_SIZE_DIVISOR", None)
        if divisor is None:
            if getattr(self, "task", None) == "semantic":
                divisor = getattr(self, "semantic_imgsz_divisor", None)
            elif getattr(self, "task", None) == "depth":
                divisor = getattr(self, "depth_imgsz_divisor", None)
        divisor = int(divisor or 1)
        if divisor <= 0:
            raise RuntimeError(
                f"{family} declares invalid INPUT_SIZE_DIVISOR={divisor}"
            )
        if value % divisor:
            raise ValueError(
                f"{family} {context} imgsz={value} must be divisible by "
                f"{divisor}"
            )
        return value

    def _validate_predict_input_size(self, imgsz: Any) -> int | tuple[int, int]:
        """Validate scalar or rectangular prediction canvas dimensions."""
        if isinstance(imgsz, (list, tuple)):
            if len(imgsz) != 2:
                raise ValueError(
                    f"{self.FAMILY or type(self).__name__} inference imgsz must "
                    f"be an integer or (height, width), got {imgsz!r}"
                )
            height = self._validate_input_size(imgsz[0], context="inference")
            width = self._validate_input_size(imgsz[1], context="inference")
            if height == width:
                return height
            if not self.SUPPORTS_RECTANGULAR_INPUT:
                raise ValueError(
                    f"{self.FAMILY or type(self).__name__} inference does not "
                    f"support rectangular imgsz={(height, width)}"
                )
            return height, width
        return self._validate_input_size(imgsz, context="inference")

    def _validate_export_input_size(self, imgsz: Any) -> int | tuple[int, int]:
        """Validate export dimensions before exporter construction."""
        if isinstance(imgsz, (list, tuple)):
            if len(imgsz) != 2:
                raise ValueError(
                    f"{self.FAMILY or type(self).__name__} export imgsz must be "
                    f"an integer or (height, width), got {imgsz!r}"
                )
            return (
                self._validate_input_size(imgsz[0], context="export"),
                self._validate_input_size(imgsz[1], context="export"),
            )
        return self._validate_input_size(imgsz, context="export")

    def _apply_checkpoint_input_size(
        self,
        checkpoint: dict,
        *,
        is_native_v1: bool,
    ) -> None:
        """Adopt validated native checkpoint canvas metadata when supported."""
        if not is_native_v1 or "imgsz" not in checkpoint:
            return
        raw_imgsz = checkpoint["imgsz"]
        allow_fixed_override = bool(self.CHECKPOINT_INPUT_SIZE_OVERRIDE)
        try:
            validated = self._validate_input_size(
                raw_imgsz,
                context="checkpoint",
                allow_fixed_override=allow_fixed_override,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Cannot use checkpoint imgsz={raw_imgsz!r} for "
                f"{self.FAMILY or type(self).__name__}: {exc}"
            ) from exc
        self.input_size = validated

    def _strict_loading(self) -> bool:
        """Return whether legacy raw checkpoints require exact loading."""
        return True

    def _checkpoint_load_policy(
        self,
        checkpoint: dict | None,
        checkpoint_task: str | None = None,
    ) -> CheckpointLoadPolicy:
        """Return the explicit tensor-coverage policy for one checkpoint.

        Complete native v1 checkpoints are exact by default. Families that
        historically accepted raw upstream/legacy weights can retain bounded
        missing-key compatibility through ``_strict_loading() == False``.
        Deliberate transfer modes override this method to name their head keys.
        """
        is_native = bool(
            isinstance(checkpoint, dict)
            and not validate_checkpoint_metadata(checkpoint, strict=False)
        )
        if is_native or self._strict_loading():
            return NATIVE_CHECKPOINT_LOAD_POLICY
        return LEGACY_CHECKPOINT_LOAD_POLICY

    def _load_state_dict_checked(
        self,
        state_dict: dict,
        *,
        checkpoint: dict | None = None,
        checkpoint_task: str | None = None,
        context: str | None = None,
        policy: CheckpointLoadPolicy | None = None,
    ) -> CheckpointLoadReport:
        """Load model state only after enforcing its declared coverage policy."""
        effective_policy = policy or self._checkpoint_load_policy(
            checkpoint,
            checkpoint_task,
        )
        return load_state_dict_checked(
            self.model,
            state_dict,
            policy=effective_policy,
            context=context or f"{self.FAMILY or type(self).__name__} checkpoint",
        )

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        """Transform state dict keys before loading.

        Override in subclasses that need to remap legacy key names.
        """
        return state_dict

    def _adapt_checkpoint_num_classes(
        self,
        ckpt_nc: int | None,
        checkpoint_task: str | None = None,
    ) -> int | None:
        """Return the class count to use when adapting checkpoint weights."""
        return ckpt_nc

    def _filter_incoming_state_dict(
        self,
        state_dict: dict,
        *,
        loaded: dict | None = None,
        checkpoint_task: str | None = None,
    ) -> dict:
        """Filter checkpoint tensors before loading.

        Families can override this when a permitted cross-task load keeps the
        reusable backbone/neck tensors but drops incompatible task-specific
        heads.
        """
        return state_dict

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        """Rebuild model with a new class count, preserving weights where shapes match."""
        old_state = self.model.state_dict()
        self.nb_classes = new_nb_classes
        self.names = {i: f"class_{i}" for i in range(new_nb_classes)}
        # Signal _init_model to skip pretrained backbone downloads — old_state
        # already holds all backbone weights which are restored below, so
        # downloading pretrained weights here is pure waste.
        self._in_rebuild = True
        try:
            self.model = self._init_model()
        finally:
            self._in_rebuild = False

        new_state = self.model.state_dict()
        for key in old_state:
            if key in new_state and old_state[key].shape == new_state[key].shape:
                new_state[key] = old_state[key]

        self.model.load_state_dict(new_state)
        self.model.to(self.device)

    def _rebuild_for_checkpoint_classes(self, new_nb_classes: int, state_dict: dict):
        """Rebuild for checkpoint class count before loading its state dict."""
        self._rebuild_for_new_classes(new_nb_classes)

    def _validate_loaded_state_dict_for_task(
        self,
        state_dict: dict,
        checkpoint: dict | None = None,
    ) -> None:
        """Validate task-specific state-dict shape before non-strict loading."""
        return None

    @classmethod
    def _filename_regex(cls) -> Optional[re.Pattern]:
        """Compile regex for matching weight filenames with optional task suffix."""
        if not cls.INPUT_SIZES or not cls.FILENAME_PREFIX:
            return None
        all_sizes = set(cls.INPUT_SIZES)
        for task_sizes in cls.TASK_INPUT_SIZES.values():
            all_sizes.update(task_sizes)
        sizes = sorted(all_sizes, key=len, reverse=True)
        sizes_pattern = "|".join(re.escape(size) for size in sizes)
        prefix = cls.FILENAME_PREFIX.lower()
        ext = re.escape(cls.WEIGHT_EXT)
        suffixes = task_suffix_pattern(cls.SUPPORTED_TASKS)
        if suffixes:
            # Families with no suffixless (detect) task can require the task
            # suffix so that e.g. ``LibreResNet50.pt`` is not accepted as a
            # classify checkpoint -- only ``LibreResNet50-cls.pt`` is canonical.
            optional = "" if getattr(cls, "REQUIRE_TASK_SUFFIX", False) else "?"
            suffix_group = rf"(?P<task>{suffixes}){optional}"
        else:
            suffix_group = ""
        variant_group = ""
        if cls.WEIGHT_VARIANTS:
            variants = "|".join(
                re.escape(variant.lower()) for variant in cls.WEIGHT_VARIANTS
            )
            variant_group = rf"(?P<variant>-(?:{variants}))?"
        return re.compile(
            rf"{prefix}(?P<size>{sizes_pattern}){suffix_group}{variant_group}{ext}"
        )

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        """Extract model size from a weight filename."""
        pattern = cls._filename_regex()
        if pattern is None:
            return None
        m = pattern.search(filename.lower())
        return m.group("size") if m else None

    @classmethod
    def detect_task_from_filename(cls, filename: str) -> Optional[str]:
        """Extract canonical task from a weight filename (e.g. '-seg' -> 'segment')."""
        pattern = cls._filename_regex()
        if pattern is None:
            return detect_task_suffix(filename)
        m = pattern.search(filename.lower())
        task_suffix = m.groupdict().get("task") if m else None
        if task_suffix:
            return normalize_task(task_suffix.lstrip("-"))
        return None

    @classmethod
    def detect_variant_from_filename(cls, filename: str) -> Optional[str]:
        """Extract the dataset-variant suffix from a weight filename, if any."""
        pattern = cls._filename_regex()
        if pattern is None:
            return None
        m = pattern.search(filename.lower())
        variant = m.groupdict().get("variant") if m else None
        return variant.lstrip("-") if variant else None

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        """Return this family's native tensor dict for a recognized upstream layout.

        Called by :mod:`libreyolo.models.autoconvert` on metadata-less
        checkpoints. The default claims layouts whose keys already match the
        native port (``can_load``). Families whose upstream key naming differs
        from the native port override this with a remap, and return ``None``
        for layouts they do not recognize.
        """
        return dict(state_dict) if cls.can_load(state_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        """Infer the task from task-specific head keys, or ``None`` if unknown."""
        return None

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        """Return the Hugging Face download URL for the given weight filename."""
        # Exact public identities are owned by the static manifest.  In
        # particular, do not synthesize a plausible URL for a known
        # config-only or otherwise unpublished checkpoint.
        from ..manifest import get_artifact_spec, match_weight_filename

        artifact = match_weight_filename(filename)
        if artifact is not None:
            if artifact.family != cls.FAMILY or artifact.download_kind != "hf":
                return None
            return artifact.download_url

        # Preserve the historical convention for compatible third-party or
        # locally published filenames that are not part of the public catalog.
        size = cls.detect_size_from_filename(filename)
        if size is None:
            return None
        task = cls.detect_task_from_filename(filename) or cls.DEFAULT_TASK
        variant = cls.detect_variant_from_filename(filename)

        # A permissive legacy filename regex must not bypass a fail-closed
        # publication declaration.  For example, adding an arbitrary prefix to
        # a config-only canonical filename makes ``match_weight_filename`` miss,
        # while ``detect_size_from_filename`` still recognizes the embedded
        # model name.  Resolve the parsed identity through the manifest before
        # retaining the compatibility URL fallback.
        declared = get_artifact_spec(
            cls.FAMILY,
            size,
            task,
            variant=variant,
        )
        if declared is not None:
            if declared.download_kind != "hf":
                return None
            return declared.download_url

        task_suffix = task_to_suffix(task)
        suffix = f"-{task_suffix}" if task_suffix else ""
        variant_suffix = f"-{variant}" if variant else ""
        name = f"{cls.FILENAME_PREFIX}{size}{suffix}{variant_suffix}"
        return f"https://huggingface.co/LibreYOLO/{name}/resolve/main/{name}{cls.WEIGHT_EXT}"

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> Optional[str]:
        """Return an optional warning shown before auto-downloading weights."""
        if Path(filename).name.lower() not in cls.EXPERIMENTAL_WEIGHT_FILENAMES:
            return None
        return (
            f"{Path(filename).name} is an EXTREMELY experimental preview checkpoint. "
            "It is provided for early pose-estimation testing and may change without "
            "compatibility guarantees."
        )

    @classmethod
    def verify_downloaded_file(cls, local_path: str, source_url: str) -> None:
        """Verify a freshly auto-downloaded weight file before it is loaded.

        Hook called by ``download_weights`` after transfer and before the file
        is promoted to its public cache path. The default trusts LibreYOLO's
        own Hugging Face mirror and does nothing;
        families that fetch third-party objects (e.g. YOLO-NAS from Deci's CDN)
        override this to checksum-pin the download and fail closed on mismatch.
        """
        return None

    def _get_val_preprocessor(self, img_size: int | None = None):
        """Return the validation preprocessor for this model."""
        if img_size is None:
            img_size = self._get_input_size()
        return self.val_preprocessor_class(img_size=(img_size, img_size))

    # =========================================================================
    # Weight loading internals
    # =========================================================================

    @staticmethod
    def _strip_ddp_prefix(state_dict: dict) -> dict:
        """Strip 'module.' prefix from DDP-wrapped state_dict keys."""
        if any(k.startswith("module.") for k in state_dict):
            normalized = {}
            source_keys = {}
            for raw_key, value in state_dict.items():
                key = raw_key.removeprefix("module.")
                if key in normalized:
                    raise ValueError(
                        "Checkpoint key normalization collision: "
                        f"{source_keys[key]!r} and {raw_key!r} both map to {key!r}."
                    )
                normalized[key] = value
                source_keys[key] = raw_key
            return normalized
        return state_dict

    @staticmethod
    def _sanitize_names(names: dict | list, nc: int) -> Dict[int, str]:
        """Normalize reader-compatible dict/list names without dropping bad keys."""
        return normalize_checkpoint_names(names, nc)

    @staticmethod
    def _parse_checkpoint_metadata(
        checkpoint: dict,
        *,
        context: str,
    ) -> tuple[dict, bool]:
        """Apply the shared strict-v1/legacy metadata boundary."""
        return parse_checkpoint_metadata_for_load(checkpoint, context=context)

    @staticmethod
    def _normalize_checkpoint_nc(value: Any) -> int | None:
        """Normalize a legacy checkpoint class count and reject unsafe values."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("checkpoint nc must be a positive integer, not bool")
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint nc must be a positive integer, got {value!r}"
            ) from exc
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(
                f"checkpoint nc must be a positive integer, got {value!r}"
            )
        if normalized <= 0:
            raise ValueError(
                f"checkpoint nc must be a positive integer, got {normalized}"
            )
        return normalized

    def _apply_loaded_checkpoint(
        self,
        loaded: dict,
        *,
        context: str,
    ) -> CheckpointLoadReport:
        """Normalize metadata/raw wrappers and apply one checked state load."""
        if not isinstance(loaded, dict):
            raise TypeError(f"{context} must be a dictionary")
        loaded, is_native_v1 = parse_checkpoint_metadata_for_load(
            loaded,
            context=context,
        )

        if isinstance(loaded.get("model"), dict):
            state_dict = loaded["model"]
        elif isinstance(loaded.get("state_dict"), dict):
            state_dict = loaded["state_dict"]
        else:
            state_dict = loaded
        state_dict = dict(state_dict)
        if not is_native_v1:
            state_dict = self._prepare_state_dict(
                self._strip_ddp_prefix(state_dict)
            )

        own_family = self._get_model_name()
        ckpt_family = loaded.get("model_family", "")
        if ckpt_family and ckpt_family != own_family:
            raise RuntimeError(
                f"Checkpoint was trained with model_family='{ckpt_family}' "
                f"but is being loaded into '{own_family}'. "
                "Use the correct model class for this checkpoint."
            )

        normalized_ckpt_task = None
        ckpt_task = loaded.get("task")
        if ckpt_task is not None:
            normalized_ckpt_task = normalize_task(ckpt_task)
            if (
                normalized_ckpt_task != self.task
                and not self._allow_checkpoint_task_mismatch(normalized_ckpt_task)
            ):
                raise RuntimeError(
                    f"Checkpoint was trained for task='{normalized_ckpt_task}' "
                    f"but this model was initialized for task='{self.task}'. "
                    "Pass the matching task or use the correct checkpoint."
                )

        self._apply_checkpoint_input_size(
            loaded,
            is_native_v1=is_native_v1,
        )

        ckpt_nc = self._normalize_checkpoint_nc(loaded.get("nc"))
        ckpt_names = loaded.get("names")
        if ckpt_nc is None and ckpt_names is not None:
            if not isinstance(ckpt_names, (dict, list)):
                raise ValueError(
                    "checkpoint names must be a dict[int, str] or list[str]"
                )
            ckpt_nc = len(ckpt_names)
        if ckpt_nc is None and hasattr(self, "detect_nb_classes"):
            ckpt_nc = self._normalize_checkpoint_nc(
                self.detect_nb_classes(state_dict)
            )

        ckpt_nc = self._adapt_checkpoint_num_classes(
            ckpt_nc,
            normalized_ckpt_task,
        )
        normalized_names = (
            self._sanitize_names(ckpt_names, ckpt_nc)
            if ckpt_names is not None and ckpt_nc is not None
            else None
        )
        state_dict = self._filter_incoming_state_dict(
            state_dict,
            loaded=loaded,
            checkpoint_task=normalized_ckpt_task,
        )

        if ckpt_nc is not None and ckpt_nc != self.nb_classes:
            self._rebuild_for_checkpoint_classes(ckpt_nc, state_dict)

        if normalized_names is not None:
            self.names = normalized_names
        self._validate_loaded_state_dict_for_task(state_dict, loaded)

        report = self._load_state_dict_checked(
            state_dict,
            checkpoint=loaded,
            checkpoint_task=normalized_ckpt_task,
            context=context,
        )
        self.model.to(self.device).eval()
        return report

    def _load_weights(self, model_path: str):
        """Load model weights from file.

        Handles raw state_dicts and training checkpoint dicts.
        Auto-rebuilds model architecture if checkpoint has different nc.
        Also handles DDP prefix stripping and cross-family rejection.
        """
        path = Path(model_path)
        if not path.exists() and path.parent == Path("."):
            weights_path = Path("weights") / path.name
            if weights_path.exists():
                model_path = str(weights_path)
                path = weights_path

        if not path.exists():
            from ...utils.download import download_weights

            download_weights(model_path, self.size)
            path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        try:
            loaded = load_untrusted_torch_file(
                model_path,
                map_location="cpu",
                context="model weights",
            )

            self._apply_loaded_checkpoint(
                loaded,
                context=f"model weights from {model_path}",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from {model_path}: {e}"
            ) from e

    def _restore_after_training(self, results: Dict[str, Any]) -> Optional[str]:
        """Load the best available training checkpoint into the live wrapper.

        Validation may not run in a short job, in which case ``best.pt`` does
        not exist but ``last.pt`` still contains the current EMA weights. This
        helper makes every wrapper converge on the same best-then-last policy,
        restores inference mode/device, and records the checkpoint used for a
        subsequent ``resume=True`` call.
        """
        candidates = (
            results.get("best_checkpoint"),
            results.get("last_checkpoint"),
        )
        checkpoint_path = next(
            (
                Path(candidate).expanduser().resolve()
                for candidate in candidates
                if candidate and Path(candidate).expanduser().is_file()
            ),
            None,
        )
        if checkpoint_path is None:
            move = getattr(self.model, "to", None)
            if callable(move):
                move(self.device)
            evaluate = getattr(self.model, "eval", None)
            if callable(evaluate):
                evaluate()
            self.model_path = None
            return None

        self._load_weights(str(checkpoint_path))
        move = getattr(self.model, "to", None)
        if callable(move):
            move(self.device)
        evaluate = getattr(self.model, "eval", None)
        if callable(evaluate):
            evaluate()
        self.model_path = str(checkpoint_path)
        return self.model_path

    def _allow_checkpoint_task_mismatch(self, checkpoint_task: str) -> bool:
        """Return whether a family permits loading a checkpoint from another task."""
        return False

    # =========================================================================
    # Public API
    # =========================================================================

    def get_available_layer_names(self) -> List[str]:
        """Get list of available layer names."""
        return sorted(self._get_available_layers().keys())

    def info(self, detailed: bool = False, verbose: bool = True) -> Dict[str, Any]:
        """Return model metadata and lightweight architecture counts.

        Args:
            detailed: Include per-parameter rows.
            verbose: Log a human-readable summary.

        Returns:
            JSON-friendly model information dictionary.
        """
        data = build_model_info(self, detailed=detailed)
        if verbose:
            logger.info(format_model_info(data))
        return data

    @property
    def _runner(self):
        if not hasattr(self, "_runner_instance") or self._runner_instance is None:
            from .inference import InferenceRunner

            self._runner_instance = InferenceRunner(self)
        return self._runner_instance

    def __call__(
        self, source=None, **kwargs
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        return self._runner(source, **kwargs)

    def predict(
        self, *args, **kwargs
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Alias for __call__ method."""
        return self(*args, **kwargs)

    def _predict_augment(
        self,
        image,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        **kwargs,
    ) -> Results:
        """Run TTA inference and merge via per-class NMS.

        Scales are read from TTA_SCALES (class variable); each scale x 2 flips
        = one batch of passes. TTA_FIXED_SIZE models always use flip-only.
        """
        if getattr(self, "task", "detect") == "obb":
            raise ValueError(
                "Test-time augmentation does not support oriented boxes yet. "
                "Use augment=False for OBB models."
            )
        if getattr(self, "task", "detect") == "pose":
            raise ValueError(
                "Test-time augmentation does not support pose keypoints yet. "
                "Use augment=False for pose models."
            )
        if getattr(self, "task", "detect") == "point":
            raise ValueError(
                "Test-time augmentation does not support point-task models yet. "
                "Use augment=False for point models."
            )
        if getattr(self, "task", "detect") == "semantic":
            raise ValueError(
                "Test-time augmentation does not support semantic segmentation yet. "
                "Use augment=False for semantic models."
            )
        if getattr(self, "task", "detect") == "panoptic":
            raise ValueError(
                "Test-time augmentation does not support panoptic segmentation yet. "
                "Use augment=False for panoptic models."
            )
        if getattr(self, "task", "detect") == "depth":
            raise ValueError(
                "Test-time augmentation does not support depth estimation yet. "
                "Use augment=False for depth models."
            )
        if getattr(self, "task", "detect") == "restore":
            raise ValueError(
                "Test-time augmentation does not support restoration models yet. "
                "Use augment=False for restore models."
            )
        if getattr(self, "task", "detect") == "ocr":
            raise ValueError(
                "Test-time augmentation does not support OCR models yet. "
                "Use augment=False for OCR models."
            )

        from PIL import Image as PILImage
        from ...utils.image_loader import ImageLoader

        effective_imgsz = imgsz if imgsz is not None else self._get_input_size()
        img_pil = ImageLoader.load(image, color_format=color_format)
        image_path = image if isinstance(image, (str, Path)) else None
        orig_w, orig_h = img_pil.size

        scales = (1.0,) if self.TTA_FIXED_SIZE else self.TTA_SCALES
        postprocess_kwargs = dict(kwargs)
        postprocess_kwargs["input_size"] = effective_imgsz

        aug_dets = []
        for scale in scales:
            if scale == 1.0:
                scaled = img_pil
            else:
                scaled = img_pil.resize(
                    (int(orig_w * scale), int(orig_h * scale)),
                    PILImage.Resampling.BILINEAR,
                )
            for is_flipped in (False, True):
                src = (
                    scaled.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT)
                    if is_flipped
                    else scaled
                )
                tensor, _, orig_size, ratio = self._preprocess(
                    src, color_format, input_size=effective_imgsz
                )
                with torch.no_grad():
                    raw = self._forward(tensor.to(self.device))
                det = self._postprocess(
                    raw,
                    conf,
                    iou,
                    orig_size,
                    max_det=max_det,
                    ratio=ratio,
                    **postprocess_kwargs,
                )
                aug_dets.append((det, orig_size, is_flipped, scale))

        if getattr(self, "task", "detect") == "classify":
            return self._merge_classify_tta(aug_dets, image_path, (orig_w, orig_h))

        return self._merge_tta(aug_dets, iou, image_path, (orig_w, orig_h), classes)

    def _merge_classify_tta(
        self,
        aug_dets: list,
        image_path,
        original_size: Tuple[int, int],
    ) -> Results:
        """Merge classification TTA by averaging probability vectors."""
        from ...utils.results import Probs, Results

        probs = [
            torch.as_tensor(det["probs"], dtype=torch.float32)
            for det, _, _, _ in aug_dets
            if "probs" in det
        ]
        avg_probs = (
            torch.stack(probs, dim=0).mean(dim=0)
            if probs
            else torch.zeros(0, dtype=torch.float32)
        )
        orig_w, orig_h = original_size
        return Results(
            boxes=None,
            orig_shape=(orig_h, orig_w),
            path=str(image_path) if image_path else None,
            names=self.names,
            probs=Probs(avg_probs),
        )

    def _merge_tta(
        self,
        aug_dets: list,
        iou_thres: float,
        image_path,
        original_size: Tuple[int, int],
        classes: Optional[List[int]] = None,
    ) -> Results:
        """Merge TTA detections from multiple augmented views via per-class NMS."""
        from ...utils.results import Boxes, Masks, Results

        orig_w, orig_h = original_size
        orig_shape = (orig_h, orig_w)

        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_classes: List[torch.Tensor] = []
        all_masks: List[Optional[torch.Tensor]] = []
        has_masks = False

        for det, orig_size, is_flipped, scale in aug_dets:
            if det["num_detections"] == 0:
                continue

            w = orig_size[0]  # width of the (possibly scaled) augmented image
            boxes = torch.as_tensor(det["boxes"], dtype=torch.float32)
            scores = torch.as_tensor(det["scores"], dtype=torch.float32)
            cls = torch.as_tensor(det["classes"], dtype=torch.float32)

            if is_flipped:
                boxes = torch.stack(
                    [w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]],
                    dim=1,
                )

            if scale != 1.0:
                boxes = boxes / scale
                orig_w_val, orig_h_val = original_size
                boxes[:, 0::2].clamp_(0, orig_w_val)
                boxes[:, 1::2].clamp_(0, orig_h_val)

            raw_m = det.get("masks")
            m = None
            # Masks in scaled views are in the wrong pixel space; skip them
            if raw_m is not None and scale == 1.0:
                has_masks = True
                m = raw_m if isinstance(raw_m, torch.Tensor) else torch.as_tensor(raw_m)
                if is_flipped:
                    m = m.flip(-1)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(cls)
            all_masks.append(m)

        def _empty_results():
            return Results(
                boxes=Boxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    torch.zeros(0, dtype=torch.float32),
                    torch.zeros(0, dtype=torch.float32),
                ),
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=self.names,
            )

        if not all_boxes:
            return _empty_results()

        masks_cat: Optional[torch.Tensor] = None
        if has_masks:
            # Drop aug views that returned boxes but no masks to keep rows aligned
            paired = [
                (b, s, c, m)
                for b, s, c, m in zip(all_boxes, all_scores, all_classes, all_masks)
                if m is not None
            ]
            if paired:
                all_boxes, all_scores, all_classes, mask_list = map(list, zip(*paired))
                masks_cat = torch.cat(mask_list, dim=0)

        boxes_cat = torch.cat(all_boxes, dim=0)
        scores_cat = torch.cat(all_scores, dim=0)
        classes_cat = torch.cat(all_classes, dim=0)

        # Drop non-finite rows — batched_nms is undefined on NaN/Inf inputs.
        finite_mask = torch.isfinite(boxes_cat).all(dim=1) & torch.isfinite(scores_cat)
        if not finite_mask.all():
            boxes_cat = boxes_cat[finite_mask]
            scores_cat = scores_cat[finite_mask]
            classes_cat = classes_cat[finite_mask]
            if masks_cat is not None:
                masks_cat = masks_cat[finite_mask]
            if boxes_cat.numel() == 0:
                return _empty_results()

        # Shift to non-negative coords — batched_nms's class-offset trick
        # uses (boxes.max() + 1), which only separates classes when all
        # coords are non-negative. Translation-invariant for IoU.
        nms_boxes = boxes_cat - boxes_cat.min().clamp(max=0)
        # Per-class NMS in a single batched dispatch (class-offset trick).
        keep = batched_nms(nms_boxes, scores_cat, classes_cat.long(), iou_thres)
        if len(keep) == 0:
            return _empty_results()
        final_boxes = boxes_cat[keep]
        final_scores = scores_cat[keep]
        final_classes = classes_cat[keep]

        if classes is not None:
            cls_mask = torch.zeros(len(final_classes), dtype=torch.bool)
            for cid in classes:
                cls_mask |= final_classes == cid
            final_boxes = final_boxes[cls_mask]
            final_scores = final_scores[cls_mask]
            final_classes = final_classes[cls_mask]
            keep = keep[cls_mask]

        masks_obj = None
        if masks_cat is not None:
            masks_obj = Masks(masks_cat[keep], orig_shape)

        return Results(
            boxes=Boxes(final_boxes, final_scores, final_classes),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
            masks=masks_obj,
        )

    def track(
        self,
        source: str | Path,
        *,
        track_conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        show: bool = False,
        vid_stride: int = 1,
        output_path: Optional[str] = None,
        tracker: str = "bytetrack",
        tracker_config=None,
        augment: bool = False,
        **tracker_kwargs,
    ) -> Generator[Results, None, None]:
        """Track objects across video frames.

        Runs detection on each frame and associates detections across time.
        Three trackers are available via ``tracker``: ByteTrack (default) and
        OC-SORT are motion-only; Deep OC-SORT adds appearance (ReID)
        embeddings so identities survive long occlusions and crossing
        targets, at the cost of a small embedding network run per frame (its
        weights are downloaded on first use). Yields one Results per frame
        with ``track_id`` set.

        Args:
            source: Path to a video file.
            track_conf: Confidence threshold for the tracker's first
                association stage — ``track_high_thresh`` for ByteTrack,
                ``det_thresh`` for OC-SORT and Deep OC-SORT. For the motion
                trackers the detector runs at a lower threshold internally so
                low-confidence detections remain available for recovery. For ByteTrack it must be >=
                ``track_low_thresh`` (default 0.1). Ignored when *tracker_config*
                is given, or when the matching key is passed explicitly in
                ``tracker_kwargs``.
            iou: IoU threshold for NMS during detection.
            imgsz: Override input image size.
            classes: Filter to specific class IDs.
            max_det: Maximum detections per frame.
            save: If True, save annotated video to *output_path*.
            show: Display tracked frames in a window.
            vid_stride: Process every N-th frame.
            output_path: Path for saved video. Defaults to
                ``runs/track/<video_stem>.mp4``.
            tracker: Which tracker to use: ``"bytetrack"``, ``"ocsort"`` or
                ``"deepocsort"``. Ignored when *tracker_config* is given (the
                config type selects the tracker).
            tracker_config: A ``TrackConfig`` (ByteTrack), ``OCSortConfig``
                (OC-SORT) or ``DeepOCSortConfig`` (Deep OC-SORT) instance, or
                None to build one from **tracker_kwargs.
            **tracker_kwargs: Forwarded to the selected tracker's
                ``from_kwargs`` (``TrackConfig``, ``OCSortConfig`` or
                ``DeepOCSortConfig``).

        Yields:
            Results with ``track_id`` attribute set as an (N,) int tensor.
        """
        task = getattr(self, "task", "detect")
        if task == "classify":
            raise NotImplementedError(
                "Tracking does not support classification models. Use predict()."
            )
        if task == "obb":
            raise NotImplementedError(
                "Tracking does not support oriented boxes yet. "
                "Use predict() for OBB models."
            )
        if task == "point":
            raise NotImplementedError(
                "Tracking does not support point results yet. "
                "Use predict() for point models."
            )
        if task == "depth":
            raise NotImplementedError(
                "Tracking does not support depth maps yet. "
                "Use predict() for depth models."
            )
        if task == "semantic":
            raise NotImplementedError(
                "Tracking does not support semantic segmentation yet. "
                "Use predict() for semantic models."
            )
        if task == "panoptic":
            raise NotImplementedError(
                "Tracking does not support panoptic segmentation yet. "
                "Use predict() for panoptic models."
            )
        if task == "restore":
            raise NotImplementedError(
                "Tracking does not support restoration models. Use predict()."
            )
        if task == "ocr":
            raise NotImplementedError(
                "Tracking does not support OCR models yet. Use predict()."
            )

        from ...tracking import (
            ByteTracker,
            DeepOCSortConfig,
            DeepOCSortTracker,
            OCSortConfig,
            OCSortTracker,
            TrackConfig,
        )
        from ...utils.drawing import draw_boxes, draw_masks
        from ...utils.video import run_video_inference

        # A provided config picks the tracker; otherwise honour the selector.
        if isinstance(tracker_config, DeepOCSortConfig):
            tracker = "deepocsort"
        elif isinstance(tracker_config, OCSortConfig):
            tracker = "ocsort"
        elif isinstance(tracker_config, TrackConfig):
            tracker = "bytetrack"
        tracker = (tracker or "bytetrack").lower()

        if tracker == "deepocsort":
            if tracker_config is None:
                tracker_kwargs.setdefault("det_thresh", track_conf)
                tracker_config = DeepOCSortConfig.from_kwargs(**tracker_kwargs)
            # Deep OC-SORT has no low-score recovery band; the detector only
            # needs to produce boxes down to det_thresh.
            effective_conf = tracker_config.det_thresh
            tracker_obj = DeepOCSortTracker(
                config=tracker_config, device=str(self.device)
            )
        elif tracker == "ocsort":
            if tracker_config is None:
                tracker_kwargs.setdefault("det_thresh", track_conf)
                tracker_config = OCSortConfig.from_kwargs(**tracker_kwargs)
            # OC-SORT consumes low-score detections (>0.1) for recovery.
            effective_conf = min(0.1, tracker_config.det_thresh)
            tracker_obj = OCSortTracker(config=tracker_config)
        elif tracker == "bytetrack":
            if tracker_config is None:
                tracker_kwargs.setdefault("track_high_thresh", track_conf)
                tracker_config = TrackConfig.from_kwargs(**tracker_kwargs)
            # ByteTrack needs to see low-confidence detections.
            effective_conf = tracker_config.track_low_thresh
            tracker_obj = ByteTracker(config=tracker_config)
        else:
            raise ValueError(
                f"Unknown tracker {tracker!r}; "
                "choose 'bytetrack', 'ocsort' or 'deepocsort'."
            )

        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Video file not found: {source}")

        model_names = self.names

        def predict_and_track(pil_img):
            result = self._runner(
                pil_img,
                conf=effective_conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format="rgb",
            )
            if isinstance(tracker_obj, DeepOCSortTracker):
                # Appearance tracking needs the frame pixels for ReID crops.
                return tracker_obj.update(result, pil_img)
            return tracker_obj.update(result)

        def annotate_tracked(pil_img, result):
            if len(result) == 0:
                return pil_img
            img = pil_img
            if result.masks is not None:
                masks_np = result.masks.data
                if isinstance(masks_np, torch.Tensor):
                    masks_np = masks_np.cpu().numpy()
                img = draw_masks(img, masks_np, result.boxes.cls.tolist())
            tid_list = result.track_id.tolist() if result.track_id is not None else None
            return draw_boxes(
                img,
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
                class_names=model_names,
                track_ids=tid_list,
            )

        # Use runs/track/ prefix instead of runs/detect/
        track_output = output_path
        if save and output_path is None:
            from ...utils.general import increment_path

            track_output = str(
                increment_path(
                    Path("runs") / "track" / f"{source.stem}.mp4",
                    exist_ok=False,
                )
            )

        yield from run_video_inference(
            source,
            predict_and_track,
            vid_stride=vid_stride,
            save=save,
            show=show,
            output_path=track_output,
            annotate_fn=annotate_tracked,
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export model to deployment format.

        Args:
            format: Target format ("onnx", "torchscript", "tensorrt",
                "openvino", "ncnn", "tflite"). "litert" is accepted as an
                alias for "tflite" (LiteRT is TensorFlow Lite's new name).
            **kwargs: Format-specific parameters forwarded to the exporter.

        Returns:
            Path to the exported model file.
        """
        from libreyolo.export import BaseExporter

        if kwargs.get("imgsz") is not None:
            kwargs["imgsz"] = self._validate_export_input_size(kwargs["imgsz"])

        return BaseExporter.create(format, self)(**kwargs)

    def val(
        self,
        data: str | None = None,
        batch: int = 16,
        imgsz: int | None = None,
        conf: float = 0.001,
        iou: float = 0.6,
        workers: int = 4,
        allow_download_scripts: bool = False,
        device: str | None = None,
        split: str = "val",
        augment: bool = False,
        save_json: bool = False,
        verbose: bool = True,
        *,
        plots: bool | None = None,
        **kwargs,
    ) -> Dict:
        """Run validation on a dataset.

        Args:
            data: Path to data.yaml file.
            batch: Batch size.
            imgsz: Image size (defaults to model's native input size).
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            workers: Number of dataloader workers.
            allow_download_scripts: Allow embedded Python in dataset YAML downloads.
            device: Device to use (default: same as model).
            split: Dataset split ("val", "test").
            save_json: Save predictions in COCO JSON format.
            plots: Alias for save_plots.
            verbose: Print detailed metrics.

        Returns:
            Dictionary with metrics/precision, metrics/recall,
            metrics/mAP50, metrics/mAP50-95.
        """
        from libreyolo.validation import (
            ClassifyValidator,
            DepthValidator,
            DetectionValidator,
            MatteValidator,
            OBBValidator,
            OCRValidator,
            PanopticValidator,
            PointValidator,
            PoseValidator,
            RestoreValidator,
            SegmentationValidator,
            SemanticValidator,
            ValidationConfig,
        )

        if imgsz is None:
            imgsz = self._get_input_size()
        validate_input_size = getattr(type(self), "_validate_input_size", None)
        if callable(validate_input_size):
            imgsz = validate_input_size(self, imgsz, context="validation")
        if plots is not None and "save_plots" not in kwargs:
            kwargs["save_plots"] = plots
        if augment and self.task == "obb":
            raise ValueError(
                "Augmented validation does not support oriented boxes yet. "
                "Use augment=False for OBB models."
            )
        if augment and self.task == "pose":
            raise ValueError(
                "Augmented validation does not support pose keypoints yet. "
                "Use augment=False for pose models."
            )
        if augment and self.task == "point":
            raise ValueError(
                "Augmented validation does not support point-task models yet. "
                "Use augment=False for point models."
            )
        if augment and self.task == "semantic":
            raise ValueError(
                "Augmented validation does not support semantic segmentation "
                "yet. Use augment=False for semantic models."
            )
        if augment and self.task == "panoptic":
            raise ValueError(
                "Augmented validation does not support panoptic segmentation "
                "yet. Use augment=False for panoptic models."
            )
        if augment and self.task == "depth":
            raise ValueError(
                "Augmented validation does not support depth estimation yet. "
                "Use augment=False for depth models."
            )
        if augment and self.task == "restore":
            raise ValueError(
                "Augmented validation does not support restoration models yet. "
                "Use augment=False for restore models."
            )
        if augment and self.task == "matte":
            raise ValueError(
                "Augmented validation does not support matte models yet. "
                "Use augment=False for matte models."
            )
        if augment and self.task == "ocr":
            raise ValueError(
                "Augmented validation does not support OCR models yet. "
                "Use augment=False for OCR models."
            )

        config = ValidationConfig(
            data=data,
            batch_size=batch,
            imgsz=imgsz,
            conf_thres=conf,
            iou_thres=iou,
            num_workers=workers,
            allow_download_scripts=allow_download_scripts,
            device=device or str(self.device),
            split=split,
            augment=augment,
            save_json=save_json,
            verbose=verbose,
            **kwargs,
        )

        if self.task == "gaze":
            raise NotImplementedError(
                "Validation against gaze ground-truth datasets (MPIIGaze, Gaze360) "
                "is out of scope for LibreYOLO. Evaluate upstream at "
                "https://github.com/Ahmednull/L2CS-Net."
            )
        if self.validator_class is not None:
            validator_cls = self.validator_class
        elif self.task == "pose":
            validator_cls = PoseValidator
        elif self.task == "point":
            validator_cls = PointValidator
        elif self.task == "segment":
            validator_cls = SegmentationValidator
        elif self.task == "semantic":
            validator_cls = SemanticValidator
        elif self.task == "panoptic":
            validator_cls = PanopticValidator
        elif self.task == "depth":
            validator_cls = DepthValidator
        elif self.task == "restore":
            validator_cls = RestoreValidator
        elif self.task == "matte":
            validator_cls = MatteValidator
        elif self.task == "ocr":
            validator_cls = OCRValidator
        elif self.task == "classify":
            validator_cls = ClassifyValidator
        elif self.task == "obb":
            validator_cls = OBBValidator
        else:
            validator_cls = DetectionValidator
        validator = validator_cls(model=self, config=config)
        return validator()
