"""Special commands: version, checks, models, formats, cfg, info, metadata."""

import sys
from pathlib import Path
from typing import Any, Optional

import typer

from ..command_utils import load_model_or_exit, resolve_model_or_exit
from ..errors import CLIError
from ..output import OutputHandler
from ...utils.model_info import build_model_info, format_model_info


# =========================================================================
# Helpers
# =========================================================================
def _get_output(json_output: bool, quiet: bool) -> OutputHandler:
    return OutputHandler(json_mode=json_output, quiet=quiet)


def _metadata_value_for_cli(value: Any) -> Any:
    """Return a compact JSON-safe representation for raw checkpoint metadata."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
        return {"type": type(value).__name__, "shape": shape, "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [_metadata_value_for_cli(item) for item in value]
    if isinstance(value, dict):
        if len(value) > 200:
            return {"type": "dict", "keys": len(value)}
        return {str(key): _metadata_value_for_cli(item) for key, item in value.items()}
    return str(value)


# =========================================================================
# version
# =========================================================================


def version_cmd(
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """Print LibreYOLO version and environment info."""
    import torch

    from libreyolo import __version__

    cuda_version = torch.version.cuda or None
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    out = _get_output(json_output, quiet)
    data = {
        "version": __version__,
        "python": python_version,
        "torch": torch.__version__,
        "cuda": cuda_version,
    }
    if not json_output:
        data["_human_text"] = (
            f"libreyolo {__version__}\n"
            f"Python {python_version}, torch {torch.__version__}, "
            f"CUDA {cuda_version or 'not available'}"
        )
    out.result(data)


# =========================================================================
# checks
# =========================================================================


def checks_cmd(
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """System info: GPU, CUDA, Python, installed packages."""
    import torch

    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "memory_mb": props.total_memory // (1024 * 1024),
                }
            )

    packages: dict[str, Optional[str]] = {}
    for pkg in (
        "onnx",
        "onnxruntime",
        "tensorrt",
        "openvino",
        "ncnn",
        "onnx2tf",
        "ai-edge-litert",
        "transformers",
        "scipy",
    ):
        try:
            from importlib.metadata import version

            packages[pkg] = version(pkg)
        except Exception:
            packages[pkg] = None

    out = _get_output(json_output, quiet)
    data = {
        "python": python_version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version())
        if torch.backends.cudnn.is_available()
        else None,
        "gpu": gpus,
        "packages": packages,
    }

    if not json_output:
        lines = [
            f"Python:  {python_version}",
            f"Torch:   {torch.__version__}",
            f"CUDA:    {torch.version.cuda or 'not available'}",
            f"cuDNN:   {data['cudnn'] or 'not available'}",
        ]
        if gpus:
            for g in gpus:
                lines.append(f"GPU {g['index']}:   {g['name']} ({g['memory_mb']} MB)")
        else:
            lines.append("GPU:     none detected")
        lines.append("")
        lines.append("Packages:")
        for pkg, ver in packages.items():
            lines.append(f"  {pkg}: {ver or 'not installed'}")
        data["_human_text"] = "\n".join(lines)

    out.result(data)


# =========================================================================
# models
# =========================================================================


def models_cmd(
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """List available model families and sizes."""
    from libreyolo.models.inventory import collect_model_inventory

    families = []
    for family, metadata in collect_model_inventory().items():
        extra = metadata["optional_extra"]
        families.append(
            {
                "name": family,
                "sizes": list(metadata["sizes"]),
                "default_imgsz": metadata["default_imgsz"],
                "task_sizes": metadata["task_sizes"],
                "tasks": metadata["tasks"],
                "default_task": metadata["default_task"],
                "cli_names": metadata["cli_names"],
                "downloadable_cli_names": metadata["downloadable_cli_names"],
                "local_only_cli_names": metadata["local_only_cli_names"],
                "available": metadata["available"],
                "optional_extra": extra,
                "install_hint": f"pip install libreyolo[{extra}]" if extra else None,
                "factory": metadata["factory"],
                "public_entrypoint": metadata["public_entrypoint"],
                "factory_default_model": metadata["factory_default_model"],
                "generic_cli": metadata["generic_cli"],
                "artifacts": metadata["artifacts"],
            }
        )

    out = _get_output(json_output, quiet)
    data = {"families": families}

    if not json_output:
        lines = ["Available models:", ""]
        for f in families:
            lines.append(f"  {f['name']}:")
            lines.append(
                f"    Tasks: {', '.join(f['tasks'])} (default: {f['default_task']})"
            )
            lines.append(f"    Sizes: {', '.join(f['sizes'])}")
            if f["cli_names"]:
                lines.append(f"    Names: {', '.join(f['cli_names'])}")
                if f["downloadable_cli_names"]:
                    lines.append(
                        "    Auto-download: " + ", ".join(f["downloadable_cli_names"])
                    )
                if f["local_only_cli_names"]:
                    lines.append(
                        "    Local checkpoint required: "
                        + ", ".join(f["local_only_cli_names"])
                    )
            else:
                model_states = sorted(
                    {
                        (artifact["factory_model"], artifact["publication"])
                        for artifact in f["artifacts"]
                        if artifact["factory_model"] is not None
                    }
                )
                lines.append(
                    f"    API: {f['public_entrypoint']} (separate {f['factory']} factory)"
                )
                lines.append(
                    "    Models: "
                    + ", ".join(
                        f"{model} [{publication}]"
                        for model, publication in model_states
                    )
                )
                if f["factory_default_model"] is not None:
                    lines.append(f"    Default: {f['factory_default_model']}")
            imgsz_str = ", ".join(f"{s}={v}" for s, v in f["default_imgsz"].items())
            lines.append(f"    Input: {imgsz_str}")
            if not f["available"] and f["install_hint"]:
                lines.append(f"    Unavailable: install with {f['install_hint']}")
            lines.append("")
        data["_human_text"] = "\n".join(lines)

    out.result(data)


# =========================================================================
# formats
# =========================================================================


def formats_cmd(
    family: Optional[str] = typer.Option(
        None, "--family", "--model", help="Show tiers for one model family"
    ),
    task: Optional[str] = typer.Option(None, "--task", help="Canonical model task"),
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """List supported export formats."""
    from libreyolo.export.exporter import BaseExporter
    from libreyolo.export.support import get_support
    from libreyolo.models.inventory import collect_model_inventory
    from libreyolo.tasks import normalize_task

    # Trigger registration of optional exporters
    try:
        from libreyolo.export import tensorrt as _  # noqa: F401
    except ImportError:
        pass
    try:
        from libreyolo.export import openvino as _  # noqa: F401
    except ImportError:
        pass
    try:
        from libreyolo.export import ncnn as _  # noqa: F401
    except ImportError:
        pass

    selected_task = None
    if family is not None:
        inventory = collect_model_inventory()
        family = family.lower()
        if family not in inventory:
            raise typer.BadParameter(f"Unknown model family: {family!r}")
        selected_task = normalize_task(task, default=inventory[family]["default_task"])
        if selected_task not in inventory[family]["tasks"]:
            raise typer.BadParameter(
                f"{family!r} does not support task {selected_task!r}."
            )

    formats = []
    for name, cls in sorted(BaseExporter._registry.items()):
        info: dict = {
            "name": name,
            "extension": cls.suffix,
            "int8": cls.supports_int8,
            "fp16": cls.supports_fp16,
            "requires_onnx": cls.requires_onnx,
        }
        aliases = sorted(
            alias for alias, target in BaseExporter._aliases.items() if target == name
        )
        if aliases:
            info["aliases"] = aliases
        if family is not None and selected_task is not None:
            support = get_support(family, selected_task, name)
            info["tier"] = support.tier
            info["reason"] = support.reason
            info["constraint"] = support.constraint
        formats.append(info)

    out = _get_output(json_output, quiet)
    data = {"formats": formats, "family": family, "task": selected_task}

    if not json_output:
        lines = ["Supported export formats:", ""]
        for f in formats:
            alias = f" (alias: {', '.join(f['aliases'])})" if f.get("aliases") else ""
            lines.append(f"  {f['name']}{alias}")
            lines.append(
                f"    Extension: {f['extension']}, FP16: {f['fp16']}, INT8: {f['int8']}"
            )
            if "tier" in f:
                lines.append(f"    Tier: {f['tier']} ({f['reason']})")
        data["_human_text"] = "\n".join(lines)

    out.result(data)


# =========================================================================
# cfg
# =========================================================================


def cfg_cmd(
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """Print default configuration."""
    from ..config import get_cfg_defaults

    data = get_cfg_defaults()

    out = _get_output(json_output, quiet)

    if not json_output:
        import yaml

        data["_human_text"] = yaml.dump(
            {k: v for k, v in data.items() if not k.startswith("_")},
            default_flow_style=False,
            sort_keys=False,
        )

    out.result(data)


# =========================================================================
# info
# =========================================================================


def info_cmd(
    model: str = typer.Option(..., help="Model name or path to weights"),
    detailed: bool = typer.Option(False, help="Include per-parameter details"),
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """Show model info: family, size, parameters, classes."""
    out = _get_output(json_output, quiet)

    model_path = resolve_model_or_exit(out, model)
    loaded = load_model_or_exit(out, model=model, model_path=model_path, device="cpu")

    info_fn = getattr(loaded, "info", None)
    if callable(info_fn):
        data = info_fn(detailed=detailed, verbose=False)
    else:
        data = build_model_info(loaded, detailed=detailed)
    data["model"] = model
    family = getattr(loaded, "FAMILY", None)
    task = getattr(loaded, "task", "detect")
    if family:
        from libreyolo.export.support import EXPORT_FORMATS, get_support

        data["export_support"] = {
            fmt: get_support(family, task, fmt).tier for fmt in EXPORT_FORMATS
        }

    if not json_output:
        data["_human_text"] = format_model_info(data)

    out.result(data)


# =========================================================================
# metadata
# =========================================================================


def metadata_cmd(
    path: str = typer.Option(..., help="Path to a .pt checkpoint"),
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
) -> None:
    """Inspect raw LibreYOLO checkpoint metadata without constructing a model."""
    from libreyolo.utils.serialization import (
        REQUIRED_CHECKPOINT_METADATA_KEYS,
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    out = _get_output(json_output, quiet)
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        err = CLIError(
            "checkpoint_not_found",
            f"Checkpoint not found: {path}",
        )
        out.error(err)
        raise typer.Exit(err.exit_code)

    loaded = load_untrusted_torch_file(
        checkpoint_path,
        map_location="cpu",
        context="checkpoint metadata",
    )
    errors = validate_checkpoint_metadata(loaded, strict=False)
    metadata = {}
    if isinstance(loaded, dict):
        metadata = {
            key: (
                {"type": "dict", "keys": len(value)}
                if key in {"train_model", "ema", "optimizer"}
                and isinstance(value, dict)
                else _metadata_value_for_cli(value)
            )
            for key, value in loaded.items()
            if key != "model"
        }

    data = {
        "path": str(checkpoint_path),
        "valid": not errors,
        "errors": errors,
        "metadata": metadata,
    }
    if not json_output:
        lines = [
            f"Checkpoint: {checkpoint_path}",
            f"Valid LibreYOLO metadata: {'yes' if not errors else 'no'}",
        ]
        if metadata:
            lines.append("")
            lines.append("Metadata:")
            ordered_keys = [
                key for key in REQUIRED_CHECKPOINT_METADATA_KEYS if key in metadata
            ]
            ordered_keys.extend(key for key in metadata if key not in ordered_keys)
            for key in ordered_keys:
                lines.append(f"  {key}: {metadata[key]}")
        if errors:
            lines.append("")
            lines.append("Errors:")
            lines.extend(f"  - {error}" for error in errors)
        data["_human_text"] = "\n".join(lines)

    out.result(data)
    if errors:
        raise typer.Exit(1)
