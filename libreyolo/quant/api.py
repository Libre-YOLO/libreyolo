"""PyTorch-native quantization API.

Grammar:

- ``model.quantize(recipe, calib=...)`` transforms the loaded model in place
  (structure swap + calibration) and returns it. No gradients are involved.
- QAT is plain ``model.train(...)`` on a quantized model: the swapped modules
  carry fp32 master weights and STE fake-quant, so the existing trainers work
  unchanged.
- QAD is the same training step with the existing ``distill_model`` kwargs.

Everything runs in PyTorch (simulation tier: numerics-true on any device).
Checkpoints written by ``model.save()`` and by the trainer carry a ``quant``
manifest so ``LibreYOLO(path)`` can rebuild the quantized structure before
loading weights.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .modules import NVFP4Linear, QuantConv2d, QuantLinear

logger = logging.getLogger(__name__)

QUANT_SCHEMA_VERSION = "1.0"
RECIPES = ("fp16", "int8", "nvfp4")
SUPPORTED_FAMILIES = ("yolo9", "rfdetr")

DEFAULT_CALIB_DATA = "coco8.yaml"

# Per-family keep-high-precision defaults (substring match on qualified module
# names). First layers and heads stay in float: standard practice, and the
# YOLO9 DFL conv is a fixed integral-expectation operator that must never be
# quantized.
_FAMILY_KEEP_HIGH_PRECISION: Dict[str, Tuple[str, ...]] = {
    "yolo9": ("head.", "backbone.conv0."),
    "rfdetr": (
        "class_embed",
        "bbox_embed",
        "angle_embed",
        "keypoint_head",
        "segmentation_head",
        "embeddings",
        "ref_point",
    ),
}
_ALWAYS_KEEP = ("dfl",)


class QuantizationError(ValueError):
    """Raised for unsupported or invalid quantization requests."""


def default_keep_high_precision(family: str) -> Tuple[str, ...]:
    return _FAMILY_KEEP_HIGH_PRECISION.get(family, ())


def _check_support(family: str, recipe: str):
    if recipe not in RECIPES:
        raise QuantizationError(
            f"Unknown quantization recipe '{recipe}'. Available: {', '.join(RECIPES)}"
        )
    if family not in SUPPORTED_FAMILIES:
        raise QuantizationError(
            f"Quantization is not supported for model family '{family}' yet. "
            f"Supported families: {', '.join(SUPPORTED_FAMILIES)}"
        )
    if family == "yolo9" and recipe == "nvfp4":
        raise QuantizationError(
            "nvfp4 is not supported for the conv-heavy yolo9 family: FP4 "
            "acceleration is GEMM-only, so convolutions stay in higher "
            "precision. Use recipe='int8' for yolo9, or nvfp4 on the "
            "transformer-based rfdetr family."
        )


def _is_excluded(name: str, keep: Tuple[str, ...]) -> bool:
    for pattern in (*keep, *_ALWAYS_KEEP):
        if pattern and pattern in name:
            return True
    return False


def _select_modules(
    root: nn.Module,
    recipe: str,
    keep: Tuple[str, ...],
) -> Dict[str, nn.Module]:
    """Deterministically select float modules to swap for a recipe."""
    selected: Dict[str, nn.Module] = {}
    for name, module in root.named_modules():
        if not name or _is_excluded(name, keep):
            continue
        if recipe == "int8":
            if type(module) is nn.Conv2d or type(module) is nn.Linear:
                selected[name] = module
        elif recipe == "nvfp4":
            if type(module) is nn.Linear:
                selected[name] = module
    return selected


def _swap_module(root: nn.Module, name: str, new_module: nn.Module):
    parent = root
    *parents, attr = name.split(".")
    for part in parents:
        parent = getattr(parent, part)
    setattr(parent, attr, new_module)


def _swap_selected(root: nn.Module, recipe: str, selected: Dict[str, nn.Module]) -> Dict[str, int]:
    counts = {"conv_int8": 0, "linear_int8": 0, "linear_nvfp4": 0}
    for name, module in selected.items():
        if recipe == "int8":
            if type(module) is nn.Conv2d:
                _swap_module(root, name, QuantConv2d.from_float(module))
                counts["conv_int8"] += 1
            else:
                _swap_module(root, name, QuantLinear.from_float(module))
                counts["linear_int8"] += 1
        elif recipe == "nvfp4":
            _swap_module(root, name, NVFP4Linear.from_float(module))
            counts["linear_nvfp4"] += 1
    return counts


def _quant_modules(root: nn.Module):
    for name, module in root.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear, NVFP4Linear)):
            yield name, module


def _cast_tree(obj, dtype):
    if torch.is_tensor(obj) and obj.is_floating_point():
        return obj.to(dtype)
    if isinstance(obj, dict):
        return {k: _cast_tree(v, dtype) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_cast_tree(v, dtype) for v in obj)
    if isinstance(obj, list):
        return [_cast_tree(v, dtype) for v in obj]
    return obj


def _install_fp16_io_hooks(root: nn.Module):
    """Half the model and keep the float32 I/O contract at the root."""

    def _pre(module, args):
        return tuple(
            a.half() if torch.is_tensor(a) and a.dtype == torch.float32 else a
            for a in args
        )

    def _post(module, args, output):
        return _cast_tree(output, torch.float32)

    root.half()
    root.register_forward_pre_hook(_pre)
    root.register_forward_hook(_post)


def _set_observing(root: nn.Module, flag: bool):
    for _, module in _quant_modules(root):
        if hasattr(module, "_q_observing"):
            module._q_observing = flag


def _run_calibration(wrapper, calib: str, samples: int, batch: int, allow_download_scripts: bool):
    from ..export.calibration import CalibrationDataLoader

    loader = CalibrationDataLoader(
        data=calib,
        imgsz=wrapper._get_input_size(),
        batch=batch,
        fraction=1.0,
        samples=samples,
        preprocess_fn=wrapper._get_preprocess_numpy(),
        allow_download_scripts=allow_download_scripts,
    )
    root = wrapper.model
    was_training = root.training
    root.eval()
    _set_observing(root, True)
    seen = 0
    with torch.no_grad():
        for np_batch in loader:
            x = torch.from_numpy(np_batch).to(wrapper.device)
            root(x)
            seen += x.shape[0]
    _set_observing(root, False)
    if was_training:
        root.train()
    return seen


def quant_info(wrapper) -> Optional[Dict]:
    """Summary of the model's quantization state, or None if float."""
    manifest = getattr(wrapper, "_quant_manifest", None)
    if not manifest:
        return None
    info = dict(manifest)
    counts = {"conv_int8": 0, "linear_int8": 0, "linear_nvfp4": 0}
    calibrated = True
    for _, module in _quant_modules(wrapper.model):
        if isinstance(module, QuantConv2d):
            counts["conv_int8"] += 1
        elif isinstance(module, NVFP4Linear):
            counts["linear_nvfp4"] += 1
        elif isinstance(module, QuantLinear):
            counts["linear_int8"] += 1
        calibrated = calibrated and module.q_calibrated
    info["module_counts"] = counts
    if manifest.get("recipe") != "fp16":
        info["calibrated"] = calibrated
    return info


def quantize_model(
    wrapper,
    recipe: str,
    calib: Optional[str] = DEFAULT_CALIB_DATA,
    samples: int = 128,
    batch: int = 8,
    keep_high_precision: Optional[Tuple[str, ...]] = None,
    allow_download_scripts: bool = False,
    verbose: bool = True,
):
    """Quantize a loaded LibreYOLO model in place and return it."""
    if getattr(wrapper, "_quant_manifest", None):
        raise QuantizationError(
            "Model is already quantized "
            f"(recipe='{wrapper._quant_manifest.get('recipe')}'). Load a float "
            "checkpoint to quantize with a different recipe."
        )

    family = wrapper.FAMILY
    recipe = str(recipe).lower()
    _check_support(family, recipe)

    keep = (
        tuple(keep_high_precision)
        if keep_high_precision is not None
        else default_keep_high_precision(family)
    )

    manifest = {
        "schema": QUANT_SCHEMA_VERSION,
        "recipe": recipe,
        "keep_high_precision": list(keep),
        "execution": "simulated",
        "calibrated": False,
        "calib_data": None,
        "calib_samples": 0,
        "module_count": 0,
    }

    if recipe == "fp16":
        if wrapper.device.type == "cpu":
            logger.warning("fp16 on CPU is functional but slow; use a GPU device.")
        _install_fp16_io_hooks(wrapper.model)
        manifest["execution"] = "native"
        manifest["calibrated"] = True
    else:
        selected = _select_modules(wrapper.model, recipe, keep)
        if not selected:
            raise QuantizationError(
                f"No quantizable modules found for recipe '{recipe}' on family "
                f"'{family}' with keep_high_precision={list(keep)}."
            )
        counts = _swap_selected(wrapper.model, recipe, selected)
        manifest["module_count"] = sum(counts.values())

        if recipe == "int8":
            if calib is not None:
                seen = _run_calibration(
                    wrapper, calib, samples, batch, allow_download_scripts
                )
                manifest["calibrated"] = True
                manifest["calib_data"] = str(calib)
                manifest["calib_samples"] = int(seen)
            else:
                logger.warning(
                    "int8 quantization without calibration: activations stay "
                    "in float (W8 simulation only). Pass calib= to calibrate."
                )
        elif recipe == "nvfp4":
            if calib is not None and calib != DEFAULT_CALIB_DATA:
                logger.info(
                    "nvfp4 activations use dynamic block scaling; calibration "
                    "data is not needed and was ignored."
                )
            manifest["calibrated"] = True

    wrapper._quant_manifest = manifest
    wrapper.model.to(wrapper.device)

    if verbose:
        info = quant_info(wrapper)
        counts = info.get("module_counts", {})
        described = ", ".join(f"{v} {k}" for k, v in counts.items() if v) or "cast only"
        logger.info(
            "Quantized %s to %s (%s; execution=%s, calibrated=%s)",
            wrapper._get_model_name(),
            recipe,
            described,
            info.get("execution"),
            info.get("calibrated"),
        )
    return wrapper


def apply_quant_structure(wrapper, manifest: Dict):
    """Re-apply a checkpoint's quantized structure before loading weights.

    Idempotent: already-swapped modules are left alone. Calibration state is
    restored from the checkpoint buffers by the subsequent load_state_dict.
    """
    recipe = manifest.get("recipe")
    if recipe not in RECIPES:
        raise QuantizationError(
            f"Checkpoint has unknown quantization recipe '{recipe}'."
        )
    _check_support(wrapper.FAMILY, recipe)

    if recipe == "fp16":
        if not getattr(wrapper, "_quant_manifest", None):
            _install_fp16_io_hooks(wrapper.model)
    else:
        keep_raw = manifest.get("keep_high_precision")
        keep = (
            tuple(keep_raw)
            if keep_raw is not None
            else default_keep_high_precision(wrapper.FAMILY)
        )
        selected = _select_modules(wrapper.model, recipe, keep)
        counts = _swap_selected(wrapper.model, recipe, selected)
        swapped = sum(counts.values())
        expected = int(manifest.get("module_count") or 0)
        already = sum(1 for _ in _quant_modules(wrapper.model))
        if expected and already != expected:
            logger.warning(
                "Quantized checkpoint expected %d quantized modules but the "
                "rebuilt model has %d; scales may not fully load.",
                expected,
                already,
            )
        if swapped:
            wrapper.model.to(wrapper.device)

    wrapper._quant_manifest = dict(manifest)
    return wrapper
