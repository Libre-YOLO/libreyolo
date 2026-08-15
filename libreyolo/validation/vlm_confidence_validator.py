"""Internal real-data validator for candidate LibreVLM confidence scores.

This module deliberately has no package export and is not wired to
``LibreVLMModel.val()``. It is an experiment harness: one generated response is
decoded into candidate-score and constant-score views, which are evaluated by
independent COCO evaluators while confidence quality is measured over a single,
score-independent prediction geometry set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import re
import sys
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from libreyolo import __version__ as libreyolo_version
from libreyolo.utils.coco_geometry import clipped_coco_bbox_xyxy

from .coco_evaluator import COCOEvaluator
from .detection_validator import DetectionValidator
from .vlm_confidence import (
    VLMDetection,
    benchmark_manifest_hash,
    build_confidence_run,
)

if TYPE_CHECKING:
    from .config import ValidationConfig

logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_CONFIDENCE_METHOD = "qwen_generation_policy_label_bbox_geomean_v1"
_CALIBRATION_BINS = 10
_MODEL_WEIGHT_SUFFIXES = {
    ".bin",
    ".h5",
    ".msgpack",
    ".onnx",
    ".ot",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
}
_PROCESSOR_ARTIFACT_SUFFIXES = {
    ".jinja",
    ".json",
    ".model",
    ".tiktoken",
    ".txt",
    ".vocab",
    ".yaml",
    ".yml",
}


def _probability(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError(f"{name} must be a real number in [0, 1].")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number in [0, 1].") from exc
    lower_ok = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not lower_ok or result > 1.0:
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must be finite and lie in {interval}.")
    return result


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class VLMConfidenceValidator(DetectionValidator):
    """Internal confidence gate over a fixed VLM generation per source image."""

    def __init__(
        self,
        model,
        config: Optional["ValidationConfig"] = None,
        *,
        generation_model: Optional[torch.nn.Module] = None,
        seed: int = 0,
        default_conf: float = 0.25,
        confidence_iou: float = 0.5,
        **kwargs,
    ) -> None:
        self.generation_model = generation_model
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        self.seed = seed
        self.benchmark_config: Optional[dict[str, Any]] = None
        self.dataset_manifest: Optional[dict[str, Any]] = None
        self.generation_manifest: list[dict[str, Any]] = []
        self._ordering_ground_truth_manifest: Optional[dict[str, Any]] = None
        self._generation_device: Optional[torch.device] = None
        self.default_conf = _probability(default_conf, "default_conf")
        self.confidence_iou = _probability(
            confidence_iou, "confidence_iou", positive=True
        )
        super().__init__(model, config, **kwargs)
        self._validate_gate_contract()

    def _validate_gate_contract(self) -> None:
        family = getattr(self.model, "FAMILY", None)
        if family != "qwen3vl":
            raise NotImplementedError(
                "VLM confidence validation is currently implemented only for "
                f"FAMILY='qwen3vl'; got {family!r}."
            )
        if self.config.augment:
            raise NotImplementedError(
                "VLM confidence validation supports original-image generation only; "
                "augment=True is not supported."
            )
        if getattr(self.config, "cuda_graph", False):
            raise NotImplementedError(
                "VLM confidence validation is serial autoregressive generation; "
                "cuda_graph is not supported."
            )
        for hook in (
            "_preprocess",
            "_forward_for_confidence_gate",
            "_postprocess_score_variants",
            "_detection_prompt",
            "set_classes",
        ):
            if not callable(getattr(self.model, hook, None)):
                raise TypeError(f"VLM confidence validation requires model.{hook}().")

    def run(self, **kwargs) -> Dict[str, float]:
        """Run the internal gate without changing the public validation surface."""

        self._validate_gate_contract()
        original_names = self._ordered_model_names()
        try:
            return super().run(**kwargs)
        finally:
            if (
                original_names is not None
                and self._ordered_model_names() != original_names
            ):
                self.model.set_classes(list(original_names))

    def _ordered_model_names(self) -> Optional[tuple[str, ...]]:
        names = getattr(self.model, "names", None)
        if isinstance(names, Mapping):
            try:
                return tuple(str(names[index]) for index in range(len(names)))
            except (KeyError, TypeError):
                return None
        if isinstance(names, (list, tuple)):
            return tuple(str(name) for name in names)
        return None

    def _setup_dataloader(self):
        data_dir = self.config.data_dir
        if data_dir is not None and not Path(data_dir).expanduser().is_dir():
            raise FileNotFoundError(
                f"VLM confidence dataset directory does not exist: {data_dir}"
            )
        # DetectionValidator/load_data_config owns registry and built-in aliases
        # such as ``coco8.yaml``. Prechecking a bare name as a local path would
        # reject a supported dataset before that resolver gets a chance to run.
        dataloader = super()._setup_dataloader()
        if len(dataloader.dataset) == 0:
            raise RuntimeError("VLM confidence validation resolved an empty dataset.")

        names = self.class_names
        if not names:
            names = getattr(dataloader.dataset, "_classes", None)
        if not names:
            model_names = getattr(self.model, "names", None)
            if isinstance(model_names, dict):
                names = [model_names[i] for i in range(len(model_names))]
        if not names:
            raise RuntimeError(
                "VLM confidence validation requires an ordered dataset vocabulary."
            )
        self.class_names = [str(name) for name in names]
        self.nc = len(self.class_names)
        self.model.set_classes(self.class_names)
        return dataloader

    def _new_coco_evaluator(self) -> COCOEvaluator:
        if self._gt_coco_api is None:
            raise RuntimeError("Ground-truth COCO API was not initialized.")
        return COCOEvaluator(
            self._gt_coco_api,
            iou_type="bbox",
            label_to_category_id=self._coco_label_to_category_id,
            max_det=self._coco_max_det(),
            faster_coco_eval=getattr(self.config, "faster_coco_eval", False),
        )

    def _init_metrics(self) -> None:
        super()._init_metrics()
        self.candidate_evaluator = self.coco_evaluator
        self.constant_evaluator = self._new_coco_evaluator()
        self._reset_confidence_records()

    def _reset_confidence_records(self) -> None:
        """Reset score-independent records and coverage counters for one run."""

        self._predictions: list[VLMDetection] = []
        self._ground_truth: list[VLMDetection] = []
        self._manifest_images: list[dict[str, Any]] = []
        self.generation_manifest = []
        self.dataset_manifest = None
        self.benchmark_config = None
        self._ordering_ground_truth_manifest = None
        self._generation_device = None
        self._responses = 0
        self._scored_responses = 0
        self._parsed_detections = 0
        self._scored_parsed_detections = 0
        self.fallback_reasons: Counter[str] = Counter()
        self.confidence_run = None

    def _warmup_model(self, n_warmup: int = 3) -> None:
        """Skip tensor warmup; the first real-path generation is the only safe one."""

        del n_warmup

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _directory_sha256(
        cls, root: Path, *, processor_artifacts: bool = False
    ) -> tuple[str, int]:
        files = sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file()
                and (
                    not processor_artifacts
                    or (
                        ".cache" not in path.relative_to(root).parts
                        and path.suffix.lower() in _PROCESSOR_ARTIFACT_SUFFIXES
                        and path.suffix.lower() not in _MODEL_WEIGHT_SUFFIXES
                    )
                )
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        if not files:
            kind = "processor" if processor_artifacts else "checkpoint"
            raise RuntimeError(
                f"Could not fingerprint an empty {kind} directory: {root}"
            )

        digest = hashlib.sha256()
        for path in files:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(path.stat().st_size.to_bytes(16, "big"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest(), len(files)

    def _generation_target(self) -> torch.nn.Module:
        target = self.generation_model
        if target is None:
            target = getattr(self.model, "model", None)
        if not isinstance(target, torch.nn.Module):
            raise TypeError("The selected generation model must be a torch.nn.Module.")
        return target

    @staticmethod
    def _target_device_and_dtype(
        target: torch.nn.Module,
    ) -> tuple[torch.device, str]:
        tensors = [*target.parameters(), *target.buffers()]
        if not tensors:
            raise RuntimeError(
                "Cannot derive the generation device from a model without "
                "parameters or buffers."
            )
        if any(tensor.device.type == "meta" for tensor in tensors):
            raise RuntimeError(
                "VLM confidence validation cannot benchmark a model with meta "
                "parameters or buffers."
            )
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            rendered = ", ".join(sorted(str(device) for device in devices))
            raise RuntimeError(
                "VLM confidence validation requires one generation device; "
                f"found {rendered}."
            )
        device = next(iter(devices))
        floating_dtypes = sorted(
            {
                str(tensor.dtype).removeprefix("torch.")
                for tensor in tensors
                if tensor.is_floating_point()
            }
        )
        if not floating_dtypes:
            raise RuntimeError(
                "Cannot derive a floating generation dtype from the model."
            )
        dtype = (
            floating_dtypes[0]
            if len(floating_dtypes) == 1
            else "mixed[" + ",".join(floating_dtypes) + "]"
        )
        return device, dtype

    @staticmethod
    def _update_digest_field(digest, value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    @classmethod
    def _canonical_config_value(cls, value: Any, path: str) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"{path} contains a non-finite float.")
            return value
        if isinstance(value, Enum):
            return cls._canonical_config_value(value.value, path)
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError(f"{path} contains a non-string mapping key.")
            return {
                key: cls._canonical_config_value(value[key], f"{path}.{key}")
                for key in sorted(value)
            }
        if isinstance(value, (list, tuple)):
            return [
                cls._canonical_config_value(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        if isinstance(value, (set, frozenset)):
            normalized = [
                cls._canonical_config_value(item, f"{path}[]") for item in value
            ]
            return sorted(
                normalized,
                key=lambda item: json.dumps(
                    item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            )
        raise TypeError(
            f"{path} contains unsupported {type(value).__name__} state; "
            "refusing an incomplete PEFT fingerprint."
        )

    @classmethod
    def _tensor_collection_identity(
        cls,
        tensors: list[tuple[str, torch.Tensor]],
        *,
        tensor_count_key: str,
        value_count_key: str,
    ) -> dict[str, Any]:
        """Hash tensor values in bounded host-memory chunks."""

        digest = hashlib.sha256()
        numel = 0
        chunk_bytes = 8 * 1024 * 1024
        for name, tensor in tensors:
            if tensor.layout != torch.strided:
                raise RuntimeError(f"Cannot fingerprint non-strided tensor {name!r}.")
            value = tensor.detach()
            if value.device.type == "meta":
                raise RuntimeError(f"Cannot fingerprint meta tensor {name!r}.")
            if getattr(value, "is_quantized", False):
                raise RuntimeError(f"Cannot fingerprint quantized tensor {name!r}.")
            if not value.is_contiguous():
                raise RuntimeError(
                    f"Cannot fingerprint non-contiguous tensor {name!r} without "
                    "an unbounded copy."
                )
            cls._update_digest_field(digest, name)
            cls._update_digest_field(digest, str(value.dtype))
            cls._update_digest_field(digest, repr(tuple(value.shape)))
            byte_view = value.reshape(-1).view(torch.uint8).reshape(-1)
            byte_count = byte_view.numel()
            digest.update(byte_count.to_bytes(8, "big"))
            for start in range(0, byte_count, chunk_bytes):
                raw = byte_view[start : start + chunk_bytes].cpu().numpy().tobytes()
                digest.update(raw)
            numel += value.numel()
        return {
            "sha256": digest.hexdigest(),
            tensor_count_key: len(tensors),
            value_count_key: numel,
        }

    @classmethod
    def _trainable_state_identity(cls, target: torch.nn.Module) -> dict[str, Any]:
        parameters = sorted(
            (
                (name, parameter)
                for name, parameter in target.named_parameters()
                if parameter.requires_grad
            ),
            key=lambda pair: pair[0],
        )
        if not parameters:
            raise RuntimeError(
                "An explicit generation_model must expose trainable PEFT state "
                "so the in-flight adapter can be fingerprinted."
            )

        return cls._tensor_collection_identity(
            parameters,
            tensor_count_key="parameter_tensors",
            value_count_key="parameter_values",
        )

    @classmethod
    def _parameter_state_identity(cls, target: torch.nn.Module) -> dict[str, Any]:
        parameters = sorted(target.named_parameters(), key=lambda pair: pair[0])
        if not parameters:
            raise RuntimeError(
                "An explicit generation_model must expose model parameters so "
                "its complete inference state can be fingerprinted."
            )
        return cls._tensor_collection_identity(
            parameters,
            tensor_count_key="parameter_tensors",
            value_count_key="parameter_values",
        )

    @classmethod
    def _buffer_state_identity(cls, target: torch.nn.Module) -> dict[str, Any]:
        buffers = sorted(target.named_buffers(), key=lambda pair: pair[0])
        return cls._tensor_collection_identity(
            buffers,
            tensor_count_key="buffer_tensors",
            value_count_key="buffer_values",
        )

    def _peft_identity(
        self,
        target: torch.nn.Module,
        base_repo: str,
        base_revision: str,
    ) -> dict[str, Any]:
        get_base_model = getattr(target, "get_base_model", None)
        if not callable(get_base_model) or get_base_model() is not getattr(
            self.model, "model", None
        ):
            raise RuntimeError(
                "generation_model must be a PEFT model wrapping this VLM's live "
                "base model."
            )
        raw_configs = getattr(target, "peft_config", None)
        if not isinstance(raw_configs, Mapping) or not raw_configs:
            raise RuntimeError("generation_model must expose non-empty peft_config.")

        configs = {}
        for adapter_name in sorted(raw_configs):
            if not isinstance(adapter_name, str):
                raise TypeError("PEFT adapter names must be strings.")
            config = raw_configs[adapter_name]
            to_dict = getattr(config, "to_dict", None)
            if not callable(to_dict):
                raise TypeError(
                    f"PEFT config {adapter_name!r} must implement to_dict()."
                )
            payload = self._canonical_config_value(
                to_dict(), f"peft_config.{adapter_name}"
            )
            if not isinstance(payload, dict):
                raise TypeError(
                    f"PEFT config {adapter_name!r} must serialize to a mapping."
                )
            # PEFT may store a machine-local cache path here. The live object
            # binding above plus these immutable wrapper values are the actual
            # base identity and remain stable across equivalent cache layouts.
            payload["base_model_name_or_path"] = base_repo
            payload["revision"] = base_revision
            configs[adapter_name] = payload

        active = getattr(target, "active_adapters", None)
        if active is None:
            active = getattr(target, "active_adapter", None)
        if isinstance(active, str):
            active = [active]
        if not isinstance(active, (list, tuple)) or any(
            not isinstance(name, str) for name in active
        ):
            raise TypeError(
                "generation_model must expose ordered active PEFT adapters."
            )
        if any(name not in configs for name in active):
            raise ValueError("An active PEFT adapter has no matching configuration.")
        adapter_enabled_state = []
        runtime_module_state = []
        runtime_fields = (
            "active_adapters",
            "disable_adapters",
            "merged",
            "merged_adapters",
            "scaling",
        )
        for module_name, module in target.named_modules():
            state = {}
            for field in runtime_fields:
                if not hasattr(module, field):
                    continue
                raw_value = getattr(module, field)
                if field == "disable_adapters" and not isinstance(raw_value, bool):
                    raise TypeError(
                        f"PEFT module {module_name!r} exposes non-boolean "
                        "disable_adapters state."
                    )
                state[field] = self._canonical_config_value(
                    raw_value, f"peft_runtime.{module_name}.{field}"
                )
            if "disable_adapters" in state:
                adapter_enabled_state.append([module_name, state["disable_adapters"]])
            if state:
                runtime_module_state.append({"module": module_name, "state": state})
        return {
            "active_adapters": list(active),
            "disable_adapters": adapter_enabled_state,
            "configs": configs,
            "runtime_modules": runtime_module_state,
        }

    def _checkpoint_identity(
        self,
        target: torch.nn.Module,
        base_repo: str,
        base_revision: str,
    ) -> Optional[dict[str, Any]]:
        if self.generation_model is not None:
            trainable = self._trainable_state_identity(target)
            parameters = self._parameter_state_identity(target)
            buffers = self._buffer_state_identity(target)
            peft = self._peft_identity(target, base_repo, base_revision)
            payload = {
                "trainable_state": trainable,
                "parameter_state": parameters,
                "buffer_state": buffers,
                "peft": peft,
            }
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return {
                "kind": "live_peft_state",
                "sha256": hashlib.sha256(encoded).hexdigest(),
                **payload,
            }

        checkpoint_dir = getattr(self.model, "_checkpoint_dir", None)
        if checkpoint_dir is None:
            return None
        root = Path(checkpoint_dir).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(
                f"Configured VLM checkpoint directory does not exist: {root}"
            )
        digest, file_count = self._directory_sha256(root)
        return {
            "kind": "checkpoint_directory",
            "sha256": digest,
            "files": file_count,
        }

    def _processor_identity(self, base_repo: str, base_revision: str) -> dict[str, Any]:
        processor = getattr(self.model, "processor", None)
        candidates: list[Path] = []
        checkpoint_dir = getattr(self.model, "_checkpoint_dir", None)
        if checkpoint_dir is not None:
            checkpoint_path = Path(checkpoint_dir).expanduser()
            if (checkpoint_path / "preprocessor_config.json").is_file():
                candidates.append(checkpoint_path)

        for owner in (processor, getattr(processor, "tokenizer", None)):
            raw = getattr(owner, "name_or_path", None)
            if isinstance(raw, str) and raw.strip():
                candidates.append(Path(raw).expanduser())

        prefix = getattr(self.model, "FILENAME_PREFIX", "")
        size = getattr(self.model, "size", "")
        if prefix and size:
            candidates.append(Path("weights") / f"{prefix}{size}")

        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or not resolved.is_dir():
                continue
            seen.add(resolved)
            try:
                digest, file_count = self._directory_sha256(
                    resolved, processor_artifacts=True
                )
            except RuntimeError:
                continue
            return {
                "source": base_repo,
                "revision": base_revision,
                "sha256": digest,
                "files": file_count,
                "class": (
                    f"{type(processor).__module__}.{type(processor).__qualname__}"
                    if processor is not None
                    else "unknown"
                ),
            }
        raise RuntimeError(
            "Could not locate local processor content to fingerprint. Load the "
            "pinned processor snapshot before running the confidence gate."
        )

    def _software_identity(self) -> dict[str, str]:
        versions = {}
        for package in ("transformers", "pycocotools"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"Cannot identify required runtime package {package!r}."
                ) from exc
        identity = {
            "python": platform.python_version(),
            "libreyolo": str(libreyolo_version),
            "torch": str(torch.__version__),
            **versions,
        }
        if self.generation_model is not None:
            try:
                identity["peft"] = metadata.version("peft")
            except metadata.PackageNotFoundError as exc:
                raise RuntimeError("Cannot identify the PEFT runtime version.") from exc
        return identity

    @staticmethod
    def _hardware_identity(device: torch.device) -> dict[str, Any]:
        identity: dict[str, Any] = {
            "type": device.type,
            "system": platform.system() or "unknown",
            "machine": platform.machine() or "unknown",
        }
        if device.type == "cuda":
            index = (
                torch.cuda.current_device() if device.index is None else device.index
            )
            properties = torch.cuda.get_device_properties(index)
            identity.update(
                {
                    "index": int(index),
                    "name": properties.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory": int(properties.total_memory),
                    "cuda": str(torch.version.cuda),
                }
            )
        elif device.type == "cpu":
            identity["processor"] = platform.processor() or identity["machine"]
        elif device.type == "mps":
            identity["processor"] = platform.processor() or identity["machine"]
        return identity

    def _build_benchmark_config(self) -> dict[str, Any]:
        target = self._generation_target()
        device, dtype = self._target_device_and_dtype(target)
        self._generation_device = device

        family = getattr(self.model, "FAMILY", None)
        size = getattr(self.model, "size", None)
        repos = getattr(self.model, "HF_REPOS", None)
        revisions = getattr(self.model, "HF_REVISIONS", None)
        if (
            not isinstance(size, str)
            or not isinstance(repos, Mapping)
            or size not in repos
        ):
            raise RuntimeError("Cannot derive the pinned base repository for this VLM.")
        if not isinstance(revisions, Mapping) or size not in revisions:
            raise RuntimeError("Cannot derive the pinned base revision for this VLM.")
        base_repo = str(repos[size])
        base_revision = str(revisions[size])
        backend = f"{type(target).__module__}.{type(target).__qualname__}"
        fallback_score = _probability(
            getattr(self.model, "DEFAULT_SCORE", 1.0), "model.DEFAULT_SCORE"
        )
        actual_imgsz = self._actual_imgsz
        if isinstance(actual_imgsz, tuple):
            actual_imgsz = list(actual_imgsz)

        config = {
            "family": str(family),
            "size": size,
            "base_repo": base_repo,
            "base_revision": base_revision,
            "checkpoint": self._checkpoint_identity(target, base_repo, base_revision),
            "processor": self._processor_identity(base_repo, base_revision),
            "class_names": list(self.class_names or []),
            "confidence_method": _CONFIDENCE_METHOD,
            "generation_kwargs": {
                "max_new_tokens": int(getattr(self.model, "MAX_NEW_TOKENS")),
                "do_sample": False,
                "num_beams": 1,
                "repetition_penalty": float(getattr(self.model, "REPETITION_PENALTY")),
            },
            "confidence_evaluation": {
                "iou_threshold": self.confidence_iou,
                "default_conf": self.default_conf,
                "fallback_score": fallback_score,
                "calibration_bins": _CALIBRATION_BINS,
                "binning": "uniform_left_closed_v1",
                "population": "scored_postprocessed_predictions",
                "matching": "class_aware_max_cardinality_iou_v1",
            },
            "evaluation": {
                "max_det": int(self._coco_max_det()),
                "faster_coco_eval": bool(
                    getattr(self.config, "faster_coco_eval", False)
                ),
                "imgsz": actual_imgsz,
                "label_to_category_id": (
                    {
                        str(key): int(value)
                        for key, value in sorted(
                            (self._coco_label_to_category_id or {}).items()
                        )
                    }
                    if self._coco_label_to_category_id is not None
                    else None
                ),
                # Replaced with the evaluator's actual backend after compute().
                "backend": "pending",
            },
            "seed": self.seed,
            "backend": backend,
            "device": str(device),
            "dtype": dtype,
            "hardware": self._hardware_identity(device),
            "software": self._software_identity(),
        }
        # Exercise the pure contract before the first expensive generation.
        benchmark_manifest_hash(
            self.model._detection_prompt(), {"preflight": True}, config
        )
        return config

    @contextmanager
    def _generation_eval_mode(self):
        target = self._generation_target()
        training_modes = [
            (module, bool(module.training)) for module in target.modules()
        ]
        target.eval()
        try:
            yield
        finally:
            # Assign flags directly: Module.train(mode) recurses and would lose
            # intentionally mixed states such as a training root with an eval
            # dropout/vision child.
            for module, was_training in training_modes:
                module.training = was_training

    @contextmanager
    def _seeded_rng(self):
        if self._generation_device is None:
            raise RuntimeError("Generation device identity was not initialized.")
        device = self._generation_device
        cpu_state = torch.random.get_rng_state()
        accelerator_state = None
        try:
            torch.random.default_generator.manual_seed(self.seed)
            if device.type == "cuda":
                accelerator_state = torch.cuda.get_rng_state(device)
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(self.seed)
            elif device.type == "mps":
                accelerator_state = torch.mps.get_rng_state()
                torch.mps.manual_seed(self.seed)
            elif device.type != "cpu":
                raise NotImplementedError(
                    f"RNG preservation is not implemented for {device.type!r}."
                )
            yield
        finally:
            if accelerator_state is not None:
                if device.type == "cuda":
                    torch.cuda.set_rng_state(accelerator_state, device)
                elif device.type == "mps":
                    torch.mps.set_rng_state(accelerator_state)
            torch.random.set_rng_state(cpu_state)

    def _move_inputs_to_model_device(self, value: Any) -> Any:
        if self._generation_device is None:
            raise RuntimeError("Generation device identity was not initialized.")
        device = self._generation_device
        if hasattr(value, "to"):
            return value.to(device)
        if isinstance(value, Mapping):
            return {
                key: self._move_inputs_to_model_device(nested)
                for key, nested in value.items()
            }
        if isinstance(value, tuple):
            return tuple(self._move_inputs_to_model_device(item) for item in value)
        if isinstance(value, list):
            return [self._move_inputs_to_model_device(item) for item in value]
        return value

    @staticmethod
    def _scalar_image_id(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Each validation image id must be scalar.")
            value = value.item()
        if isinstance(value, np.generic):
            value = value.item()
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"COCO image id must be integer-like, got {value!r}."
            ) from exc

    def _resolve_required_image_path(
        self, dataset, global_index: int, image_id: int
    ) -> Path:
        raw_path = self._resolve_img_path(dataset, global_index, image_id)
        if raw_path is None:
            raise RuntimeError(
                f"Could not resolve the original path for validation image {image_id}."
            )
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Validation image {image_id} does not exist: {path}"
            )
        return path

    @staticmethod
    def _variant_arrays(
        view: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = _to_numpy(view.get("boxes", [])).astype(np.float64, copy=False)
        classes = _to_numpy(view.get("classes", [])).reshape(-1)
        scores = (
            _to_numpy(view.get("scores", [])).astype(np.float64, copy=False).reshape(-1)
        )
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float64)
        elif boxes.ndim != 2 or boxes.shape[1] != 4:
            raise RuntimeError(
                f"VLM score view returned invalid boxes shape {boxes.shape}."
            )
        if not (len(boxes) == len(classes) == len(scores)):
            raise RuntimeError(
                "VLM score view returned misaligned boxes, classes, and scores."
            )
        return boxes, classes, scores

    def _score_independent_prediction_records(
        self,
        variants,
        image_id: str,
    ) -> list[VLMDetection]:
        candidate_boxes, candidate_classes, candidate_scores = self._variant_arrays(
            variants.candidate
        )
        constant_boxes, constant_classes, constant_scores = self._variant_arrays(
            variants.constant
        )

        def keyed(boxes: np.ndarray, classes: np.ndarray) -> list[tuple]:
            return [
                (int(class_id), *(float(value) for value in box))
                for box, class_id in zip(boxes, classes)
            ]

        candidate_keys = keyed(candidate_boxes, candidate_classes)
        constant_keys = keyed(constant_boxes, constant_classes)
        if len(set(candidate_keys)) != len(candidate_keys):
            raise RuntimeError(
                "Candidate score view returned duplicate geometry records."
            )
        if len(set(constant_keys)) != len(constant_keys):
            raise RuntimeError(
                "Constant score view returned duplicate geometry records."
            )
        if candidate_keys != constant_keys:
            raise RuntimeError(
                "Candidate scoring changed the generated detection geometry order; "
                "constant-score COCO tie handling would no longer be comparable."
            )
        if not variants.score_available and not np.array_equal(
            candidate_scores, constant_scores
        ):
            raise RuntimeError(
                "An unavailable candidate score must use the exact constant-score "
                "fallback view."
            )
        return [
            VLMDetection(
                image_id=image_id,
                class_id=key[0],
                xyxy=tuple(key[1:]),
                score=(float(score) if variants.score_available else None),
            )
            for key, score in zip(constant_keys, candidate_scores)
        ]

    def _record_response(self, variants) -> None:
        parsed = int(variants.parsed_items)
        if parsed < 0:
            raise RuntimeError("parsed_items must be non-negative.")
        self._responses += 1
        self._parsed_detections += parsed
        if variants.score_available:
            if variants.item_scores is None or len(variants.item_scores) != parsed:
                raise RuntimeError(
                    "A scored VLM response must provide one score per parsed item."
                )
            self._scored_responses += 1
            self._scored_parsed_detections += parsed
        else:
            reason = variants.fallback_reason or "unknown"
            self.fallback_reasons[str(reason)] += 1

    def _record_generation(self, variants, image_id: str) -> None:
        generation_hash = getattr(variants, "generation_hash", None)
        if not isinstance(generation_hash, str) or not _SHA256_RE.fullmatch(
            generation_hash
        ):
            raise RuntimeError(
                "The VLM confidence hook must return a 64-character "
                "generation_hash for every response."
            )
        self.generation_manifest.append(
            {
                "image_id": image_id,
                "sha256": generation_hash.lower(),
                "parsed_items": int(variants.parsed_items),
                "fallback_reason": (
                    None
                    if variants.score_available
                    else str(variants.fallback_reason or "unknown")
                ),
            }
        )

    def _run_validation(self) -> None:
        self._validate_gate_contract()
        if not self.class_names:
            raise RuntimeError("Dataset vocabulary was not initialized.")
        evaluator_ground_truth = self._evaluator_ground_truth_manifest()
        self._require_plain_ordering_ground_truth(evaluator_ground_truth)
        self._ordering_ground_truth_manifest = evaluator_ground_truth
        self._ground_truth = self._ordering_ground_truth(evaluator_ground_truth)
        self.model.set_classes(list(self.class_names))
        self.benchmark_config = self._build_benchmark_config()
        dataset = self.dataloader.dataset
        progress = tqdm(
            self.dataloader,
            desc="Validating VLM confidence",
            total=len(self.dataloader),
            disable=not self.config.verbose or not sys.stderr.isatty(),
            file=sys.stderr,
        )
        total_start = time.time()
        global_index = 0

        with self._seeded_rng(), self._generation_eval_mode(), torch.no_grad():
            for batch in progress:
                if len(batch) == 5:
                    _, _, _, img_ids, _ = batch
                else:
                    _, _, _, img_ids = batch
                for image_index, raw_image_id in enumerate(img_ids):
                    evaluator_image_id = self._scalar_image_id(raw_image_id)
                    record_image_id = str(evaluator_image_id)
                    path = self._resolve_required_image_path(
                        dataset, global_index, evaluator_image_id
                    )
                    image_hash = self._file_sha256(path)

                    started = time.time()
                    inputs, _, original_size, _ = self.model._preprocess(
                        str(path), color_format="auto", input_size=self._actual_imgsz
                    )
                    inputs = self._move_inputs_to_model_device(inputs)
                    self.speed["preprocess"] += time.time() - started

                    started = time.time()
                    output = self.model._forward_for_confidence_gate(
                        inputs, model=self.generation_model
                    )
                    self.speed["inference"] += time.time() - started

                    started = time.time()
                    variants = self.model._postprocess_score_variants(
                        output, tuple(original_size)
                    )
                    records = self._score_independent_prediction_records(
                        variants, record_image_id
                    )
                    self.speed["postprocess"] += time.time() - started

                    self.candidate_evaluator.update(
                        variants.candidate, evaluator_image_id
                    )
                    self.constant_evaluator.update(
                        variants.constant, evaluator_image_id
                    )
                    self._predictions.extend(records)
                    self._record_response(variants)
                    self._record_generation(variants, record_image_id)
                    if self._file_sha256(path) != image_hash:
                        raise RuntimeError(
                            f"Validation image {evaluator_image_id} changed during "
                            "generation; refusing a mismatched manifest."
                        )
                    orig_w, orig_h = (int(original_size[0]), int(original_size[1]))
                    self._manifest_images.append(
                        {
                            "image_id": record_image_id,
                            "file_name": path.name,
                            "sha256": image_hash,
                            "width": orig_w,
                            "height": orig_h,
                        }
                    )
                    self.seen += 1
                    global_index += 1

        self.speed["total"] = time.time() - total_start

    @staticmethod
    def _optional_metric(value: Optional[float]) -> float:
        return float("nan") if value is None else float(value)

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, np.generic):
            return cls._json_safe(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, Mapping):
            return {str(key): cls._json_safe(nested) for key, nested in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        return value

    def _evaluator_ground_truth_manifest(self) -> dict[str, Any]:
        coco = getattr(self, "_gt_coco_api", None)
        if coco is None:
            raise RuntimeError(
                "The actual COCO ground truth is required for benchmark identity."
            )
        images = getattr(coco, "imgs", None)
        annotations = getattr(coco, "anns", None)
        categories = getattr(coco, "cats", None)
        if not all(
            isinstance(value, Mapping) for value in (images, annotations, categories)
        ):
            raise TypeError("The COCO ground-truth API exposes invalid index mappings.")

        image_records = []
        for key, image in images.items():
            image_records.append(
                {
                    "id": int(image.get("id", key)),
                    "width": int(image.get("width", 0)),
                    "height": int(image.get("height", 0)),
                }
            )
        image_records.sort(key=lambda item: item["id"])

        category_records = []
        for key, category in categories.items():
            category_records.append(
                {
                    "id": int(category.get("id", key)),
                    "name": str(category.get("name", "")),
                }
            )
        category_records.sort(key=lambda item: item["id"])

        annotation_records = []
        for key, annotation in annotations.items():
            bbox = [float(value) for value in annotation.get("bbox", ())]
            if len(bbox) != 4:
                raise ValueError(f"COCO annotation {key!r} has an invalid bbox.")
            annotation_records.append(
                {
                    "id": int(annotation.get("id", key)),
                    "image_id": int(annotation["image_id"]),
                    "category_id": int(annotation["category_id"]),
                    "bbox": bbox,
                    "area": float(annotation.get("area", bbox[2] * bbox[3])),
                    "iscrowd": int(annotation.get("iscrowd", 0)),
                    "ignore": int(annotation.get("ignore", 0)),
                }
            )
        annotation_records.sort(
            key=lambda item: (
                item["image_id"],
                item["category_id"],
                item["id"],
            )
        )
        return {
            "api": f"{type(coco).__module__}.{type(coco).__qualname__}",
            "images": image_records,
            "categories": category_records,
            "annotations": annotation_records,
        }

    @staticmethod
    def _require_plain_ordering_ground_truth(manifest: Mapping[str, Any]) -> None:
        annotations = manifest.get("annotations")
        if not isinstance(annotations, list):
            raise TypeError("Evaluator ground-truth manifest has invalid annotations.")
        unsupported = [
            annotation.get("id")
            for annotation in annotations
            if annotation.get("iscrowd") or annotation.get("ignore")
        ]
        if unsupported:
            preview = ", ".join(str(value) for value in unsupported[:5])
            raise NotImplementedError(
                "VLM confidence ordering metrics do not yet implement COCO "
                "crowd/ignore matching semantics; unsupported annotation ids: "
                f"{preview}. Use a benchmark split without crowd/ignore labels."
            )

    def _ordering_ground_truth(self, manifest: Mapping[str, Any]) -> list[VLMDetection]:
        """Derive ordering labels from canonical COCO data without resize drift."""

        images = manifest.get("images")
        categories = manifest.get("categories")
        annotations = manifest.get("annotations")
        if not all(
            isinstance(value, list) for value in (images, categories, annotations)
        ):
            raise TypeError("Evaluator ground-truth manifest has invalid arrays.")
        image_sizes = {
            int(image["id"]): (int(image["width"]), int(image["height"]))
            for image in images
        }
        category_ids = {int(category["id"]) for category in categories}
        if self._coco_label_to_category_id is None:
            category_to_label = {
                category_id: category_id for category_id in category_ids
            }
        else:
            category_to_label = {
                int(category_id): int(label)
                for label, category_id in self._coco_label_to_category_id.items()
            }
        if set(category_to_label) != category_ids:
            raise RuntimeError(
                "COCO category mapping does not cover the evaluator categories."
            )
        class_count = len(self.class_names or ())
        if any(
            label < 0 or label >= class_count for label in category_to_label.values()
        ):
            raise RuntimeError(
                "COCO category mapping references a class outside the dataset vocabulary."
            )

        result = []
        for annotation in annotations:
            try:
                area = float(annotation["area"])
                bbox = tuple(float(value) for value in annotation["bbox"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Evaluator annotation has invalid numeric fields."
                ) from exc
            if (
                not math.isfinite(area)
                or len(bbox) != 4
                or not all(math.isfinite(value) for value in bbox)
            ):
                raise ValueError("Evaluator annotation has invalid numeric fields.")
            if area <= 0.0:
                continue
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            if image_id not in image_sizes or category_id not in category_to_label:
                raise ValueError("Evaluator annotation references an unknown id.")
            width, height = image_sizes[image_id]
            clean_bbox = clipped_coco_bbox_xyxy(bbox, width, height)
            if clean_bbox is None:
                continue
            result.append(
                VLMDetection(
                    image_id=str(image_id),
                    class_id=category_to_label[category_id],
                    xyxy=clean_bbox,
                )
            )
        return result

    def _write_report(
        self,
        metrics: Mapping[str, float],
        fallback_score: float,
    ) -> None:
        if self.confidence_run is None:
            raise RuntimeError("Confidence metrics were not initialized.")
        if self.benchmark_config is None or self.dataset_manifest is None:
            raise RuntimeError("Benchmark identity was not initialized.")
        diagnostics = self.confidence_run.diagnostics
        calibration = self.confidence_run.calibration
        predictions = []
        for prediction, matched in zip(self._predictions, self.confidence_run.matches):
            predictions.append(
                {
                    "image_id": prediction.image_id,
                    "class_id": prediction.class_id,
                    "xyxy": prediction.xyxy,
                    "candidate_score": prediction.score,
                    "effective_score": (
                        fallback_score if prediction.score is None else prediction.score
                    ),
                    "matched": matched,
                }
            )
        report = {
            "schema": "libreyolo.vlm-confidence-report.v2",
            "prompt": self.model._detection_prompt(),
            "benchmark_config": self.benchmark_config,
            "dataset_manifest": self.dataset_manifest,
            "generation_manifest": self.generation_manifest,
            "hashes": {
                "manifest": self.confidence_run.manifest_hash,
                "configuration": self.confidence_run.configuration_hash,
                "generation": self.confidence_run.generation_hash,
                "prediction_structure": (self.confidence_run.prediction_structure_hash),
            },
            "confidence": {
                "iou_threshold": self.confidence_run.iou_threshold,
                "default_conf": self.confidence_run.default_conf,
                "fallback_score": self.confidence_run.fallback_score,
            },
            "diagnostics": diagnostics.__dict__,
            "calibration": {
                "method": "equal_width",
                "population": "scored_postprocessed_predictions",
                "bin_count": calibration.bin_count,
                "total_predictions": calibration.total_predictions,
                "scored_predictions": calibration.scored_predictions,
                "unscored_predictions": calibration.unscored_predictions,
                "score_coverage": calibration.score_coverage,
                "brier_score": calibration.brier_score,
                "expected_calibration_error": (calibration.expected_calibration_error),
                "maximum_calibration_error": (calibration.maximum_calibration_error),
                "bins": [item.__dict__ for item in calibration.bins],
            },
            "evaluator_metrics": dict(self.confidence_run.evaluator_metrics),
            "fallback_reasons": dict(sorted(self.fallback_reasons.items())),
            "predictions": predictions,
            "metrics": dict(metrics),
            "artifacts": {
                "reliability_plot": (
                    "vlm_confidence_reliability.svg" if self.config.save_plots else None
                )
            },
        }
        destination = self.save_dir / "vlm_confidence_report.json"
        destination.write_text(
            json.dumps(
                self._json_safe(report),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_calibration_plot(self) -> None:
        """Write a dependency-free SVG reliability diagram for the candidate score."""

        if self.confidence_run is None:
            raise RuntimeError("Confidence metrics were not initialized.")
        calibration = self.confidence_run.calibration
        width = 640
        height = 600
        left = 80.0
        top = 62.0
        plot_size = 440.0
        bottom = top + plot_size

        def x(value: float) -> float:
            return left + plot_size * value

        def y(value: float) -> float:
            return top + plot_size * (1.0 - value)

        elements = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}">'
            ),
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            (
                '<text x="320" y="28" text-anchor="middle" '
                'font-family="sans-serif" font-size="18" font-weight="600">'
                "Candidate token probability reliability (diagnostic)</text>"
            ),
        ]
        for tick in range(11):
            value = tick / 10
            x_pos = x(value)
            y_pos = y(value)
            elements.extend(
                (
                    (
                        f'<line x1="{x_pos:.2f}" y1="{top:.2f}" '
                        f'x2="{x_pos:.2f}" y2="{bottom:.2f}" '
                        'stroke="#e6e6e6" stroke-width="1"/>'
                    ),
                    (
                        f'<line x1="{left:.2f}" y1="{y_pos:.2f}" '
                        f'x2="{left + plot_size:.2f}" y2="{y_pos:.2f}" '
                        'stroke="#e6e6e6" stroke-width="1"/>'
                    ),
                )
            )
            if tick % 2 == 0:
                elements.extend(
                    (
                        (
                            f'<text x="{x_pos:.2f}" y="{bottom + 22:.2f}" '
                            'text-anchor="middle" font-family="sans-serif" '
                            f'font-size="11">{value:.1f}</text>'
                        ),
                        (
                            f'<text x="{left - 12:.2f}" y="{y_pos + 4:.2f}" '
                            'text-anchor="end" font-family="sans-serif" '
                            f'font-size="11">{value:.1f}</text>'
                        ),
                    )
                )
        elements.extend(
            (
                (
                    f'<line x1="{left:.2f}" y1="{bottom:.2f}" '
                    f'x2="{left + plot_size:.2f}" y2="{top:.2f}" '
                    'stroke="#666666" stroke-width="1.5" '
                    'stroke-dasharray="6 5"/>'
                ),
                (
                    f'<rect x="{left:.2f}" y="{top:.2f}" '
                    f'width="{plot_size:.2f}" height="{plot_size:.2f}" '
                    'fill="none" stroke="#222222" stroke-width="1.5"/>'
                ),
            )
        )
        for item in calibration.bins:
            if item.count == 0 or item.empirical_accuracy is None:
                continue
            bar_left = x(item.lower) + 3.0
            bar_right = x(item.upper) - 3.0
            bar_top = y(item.empirical_accuracy)
            elements.extend(
                (
                    (
                        f'<rect x="{bar_left:.2f}" y="{bar_top:.2f}" '
                        f'width="{max(1.0, bar_right - bar_left):.2f}" '
                        f'height="{bottom - bar_top:.2f}" fill="#4c78a8" '
                        'fill-opacity="0.35" stroke="#315f86" stroke-width="1"/>'
                    ),
                    (
                        f'<circle cx="{x(item.mean_confidence):.2f}" '
                        f'cy="{bar_top:.2f}" r="4" fill="#d1495b">'
                        f"<title>n={item.count}; confidence="
                        f"{item.mean_confidence:.4f}; accuracy="
                        f"{item.empirical_accuracy:.4f}</title></circle>"
                    ),
                    (
                        f'<text x="{(bar_left + bar_right) / 2:.2f}" '
                        f'y="{max(top + 13.0, bar_top - 6.0):.2f}" '
                        'text-anchor="middle" font-family="sans-serif" '
                        f'font-size="10">n={item.count}</text>'
                    ),
                )
            )
        elements.extend(
            (
                (
                    f'<text x="{left + plot_size / 2:.2f}" y="{bottom + 52:.2f}" '
                    'text-anchor="middle" font-family="sans-serif" '
                    'font-size="13">Mean candidate confidence</text>'
                ),
                (
                    f'<text x="22" y="{top + plot_size / 2:.2f}" '
                    'text-anchor="middle" font-family="sans-serif" font-size="13" '
                    f'transform="rotate(-90 22 {top + plot_size / 2:.2f})">'
                    "Empirical precision at score-independent IoU matching</text>"
                ),
            )
        )
        if calibration.scored_predictions:
            brier = f"{calibration.brier_score:.4f}"
            ece = f"{calibration.expected_calibration_error:.4f}"
            summary = (
                f"N={calibration.scored_predictions}/{calibration.total_predictions} "
                f"scored ({calibration.score_coverage:.1%}); "
                f"Brier={brier}; ECE={ece}; bins={calibration.bin_count}; "
                f"IoU={self.confidence_run.iou_threshold:.2f}"
            )
        else:
            summary = "No score-bearing predictions; calibration is undefined."
        elements.extend(
            (
                (
                    '<text x="320" y="580" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="11">{summary}</text>'
                ),
                "</svg>",
            )
        )
        (self.save_dir / "vlm_confidence_reliability.svg").write_text(
            "\n".join(elements) + "\n", encoding="utf-8"
        )

    def _save_plots(self, metrics: Dict[str, float]) -> None:
        """Suppress inherited detector plots; the reliability SVG is authoritative."""

        del metrics

    def _finalize(self) -> Dict[str, float]:
        """Persist the report after shared finalization adds timing metrics."""

        metrics = super()._finalize()
        fallback_score = float(getattr(self.model, "DEFAULT_SCORE", 1.0))
        self._write_report(metrics, fallback_score)
        return metrics

    def _compute_metrics(self) -> Dict[str, float]:
        evaluator_ground_truth = self._evaluator_ground_truth_manifest()
        if self._ordering_ground_truth_manifest is None:
            raise RuntimeError(
                "Ordering ground truth was not preflighted before generation."
            )
        if evaluator_ground_truth != self._ordering_ground_truth_manifest:
            raise RuntimeError(
                "Evaluator ground truth changed during VLM generation; refusing "
                "a mismatched benchmark identity."
            )
        candidate_json = constant_json = None
        if self.config.save_json:
            candidate_json = str(self.save_dir / "predictions_candidate.json")
            constant_json = str(self.save_dir / "predictions_constant.json")
        candidate = self.candidate_evaluator.compute(save_json=candidate_json)
        constant = self.constant_evaluator.compute(save_json=constant_json)
        self.eval_backend = {
            "candidate": getattr(self.candidate_evaluator, "last_backend", None),
            "constant": getattr(self.constant_evaluator, "last_backend", None),
        }
        if self.benchmark_config is None:
            raise RuntimeError(
                "Benchmark configuration was not derived before generation."
            )
        candidate_backend = self.eval_backend["candidate"] or "not-run:no-predictions"
        constant_backend = self.eval_backend["constant"] or "not-run:no-predictions"
        actual_backend = (
            str(candidate_backend)
            if candidate_backend == constant_backend
            else f"candidate={candidate_backend};constant={constant_backend}"
        )
        self.benchmark_config["evaluation"]["backend"] = actual_backend

        self.dataset_manifest = {
            "split": self.config.split,
            "class_names": list(self.class_names or []),
            "images": self._manifest_images,
            "evaluator_ground_truth": evaluator_ground_truth,
            "ground_truth": [
                {
                    "image_id": item.image_id,
                    "class_id": item.class_id,
                    "xyxy": item.xyxy,
                }
                for item in self._ground_truth
            ],
        }
        if len(self.generation_manifest) != self._responses:
            raise RuntimeError(
                "Generation manifest is incomplete; refusing an unauditable gate result."
            )
        candidate_map = float(candidate["mAP"])
        constant_map = float(constant["mAP"])
        candidate_map50 = float(candidate["mAP50"])
        constant_map50 = float(constant["mAP50"])
        fallback_score = float(getattr(self.model, "DEFAULT_SCORE", 1.0))
        self.confidence_run = build_confidence_run(
            self._predictions,
            self._ground_truth,
            prompt=self.model._detection_prompt(),
            dataset_manifest=self.dataset_manifest,
            benchmark_config=self.benchmark_config,
            generation_manifest=self.generation_manifest,
            evaluator_metrics={
                "candidate_mAP50-95": candidate_map,
                "constant_mAP50-95": constant_map,
                "candidate_mAP50": candidate_map50,
                "constant_mAP50": constant_map50,
            },
            iou_threshold=self.confidence_iou,
            default_conf=self.default_conf,
            fallback_score=fallback_score,
        )
        diagnostics = self.confidence_run.diagnostics
        response_coverage = (
            self._scored_responses / self._responses if self._responses else 0.0
        )
        detection_coverage = (
            self._scored_parsed_detections / self._parsed_detections
            if self._parsed_detections
            else 0.0
        )

        metrics = {
            "metrics/vlm_confidence/candidate_mAP50-95": candidate_map,
            "metrics/vlm_confidence/constant_mAP50-95": constant_map,
            "metrics/vlm_confidence/delta_mAP50-95": candidate_map - constant_map,
            "metrics/vlm_confidence/candidate_mAP50": candidate_map50,
            "metrics/vlm_confidence/constant_mAP50": constant_map50,
            "metrics/vlm_confidence/delta_mAP50": candidate_map50 - constant_map50,
            "metrics/vlm_confidence/auroc": self._optional_metric(
                self.confidence_run.auroc
            ),
            "metrics/vlm_confidence/ranking_ap": self._optional_metric(
                self.confidence_run.ranking_ap
            ),
            "metrics/vlm_confidence/scored_prediction_brier": self._optional_metric(
                self.confidence_run.calibration.brier_score
            ),
            "metrics/vlm_confidence/scored_prediction_ece": (
                self._optional_metric(
                    self.confidence_run.calibration.expected_calibration_error
                )
            ),
            "metrics/vlm_confidence/scored_prediction_mce": (
                self._optional_metric(
                    self.confidence_run.calibration.maximum_calibration_error
                )
            ),
            "metrics/vlm_confidence/default_conf_tp_retention": self._optional_metric(
                diagnostics.correct_retention
            ),
            "metrics/vlm_confidence/default_conf_fp_retention": self._optional_metric(
                diagnostics.incorrect_retention
            ),
            "metrics/vlm_confidence/default_conf_prediction_retention": float(
                diagnostics.default_conf_retention
            ),
            "metrics/vlm_confidence/response_score_coverage": response_coverage,
            "metrics/vlm_confidence/detection_score_coverage": detection_coverage,
            "metrics/vlm_confidence/prediction_score_coverage": float(
                diagnostics.score_coverage
            ),
            "metrics/vlm_confidence/responses": float(self._responses),
            "metrics/vlm_confidence/scored_responses": float(self._scored_responses),
            "metrics/vlm_confidence/parsed_detections": float(self._parsed_detections),
            "metrics/vlm_confidence/scored_parsed_detections": float(
                self._scored_parsed_detections
            ),
            "metrics/vlm_confidence/predictions": float(diagnostics.total_predictions),
            "metrics/vlm_confidence/correct_predictions": float(
                diagnostics.correct_predictions
            ),
            "metrics/vlm_confidence/incorrect_predictions": float(
                diagnostics.incorrect_predictions
            ),
            "metrics/vlm_confidence/retained_correct_predictions": float(
                diagnostics.retained_correct_predictions
            ),
            "metrics/vlm_confidence/retained_incorrect_predictions": float(
                diagnostics.retained_incorrect_predictions
            ),
        }
        if self.config.save_plots:
            self._write_calibration_plot()
        return metrics
