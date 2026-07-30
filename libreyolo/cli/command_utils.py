"""Shared helpers for CLI command implementations."""

import json
from pathlib import Path
from typing import Any, Callable, NoReturn, Optional, Set, Tuple, Union

import click
import typer

from .errors import CLIError
from .output import OutputHandler

COREML_VLM_EXPORT_ALIASES = frozenset(
    {
        "florence-2",
        "florence-2-base",
        "florence2",
        "kosmos-2",
        "kosmos2",
        "qwen3-vl-2b",
        "smolvlm2-500m",
    }
)


def exit_with_error(
    out: OutputHandler,
    code: str,
    message: str,
    *,
    suggestion: Optional[str] = None,
) -> NoReturn:
    """Emit a structured CLI error and terminate the command."""
    err = CLIError(code, message, suggestion=suggestion)
    out.error(err)
    raise typer.Exit(code=err.exit_code)


def load_model_or_exit(
    out: OutputHandler,
    *,
    model: str,
    model_path: str,
    device: str,
    task: str | None = None,
    model_factory: Callable[..., Any] | None = None,
) -> Any:
    """Load a model with consistent CLI error handling."""
    if model_factory is None:
        from libreyolo import LibreYOLO

        model_factory = LibreYOLO

    out.progress(f"Loading {model}...")
    try:
        return model_factory(model_path, device=device, task=task)
    except Exception as exc:
        exit_with_error(
            out,
            "model_load_failed",
            f"Failed to load model '{model}': {exc}",
        )


def resolve_export_sibling_factory(
    model: str,
) -> Callable[..., Any] | None:
    """Return the sibling factory for an export alias, if one owns it.

    ``LibreYOLO`` intentionally covers state-dict model families only.
    Promptable SAM and discriminative open-vocabulary models use sibling
    factories and therefore need a narrow CLI export dispatch before the
    normal checkpoint-name resolver runs. Existing filesystem references and
    filename-like values always remain on the generic ``LibreYOLO`` path.
    """
    model_value = str(model).strip()
    path = Path(model_value)
    if path.exists() or path.suffix:
        return None

    key = model_value.lower()
    if key in COREML_VLM_EXPORT_ALIASES:
        from libreyolo import LibreVLM

        return LibreVLM

    sam_candidate = key in {"base", "large", "huge", "b", "l", "h"} or (
        key.startswith(("edge-tam", "edgetam", "mobile-sam", "mobilesam"))
        or key.startswith(("pico-sam3", "picosam3", "sam"))
    )
    if sam_candidate:
        from libreyolo.models.sam.model import LibreSAM, _ALIASES as sam_aliases

        if key in sam_aliases:
            return LibreSAM

    openvocab_key = key.replace("_", "-")
    openvocab_candidate = openvocab_key.startswith(
        ("grounding-dino", "groundingdino", "omdet", "owl-v2", "owlv2")
    ) or openvocab_key.startswith(("ov-deim", "ovdeim"))
    if openvocab_candidate:
        from libreyolo.models.openvocab import (
            LibreOpenVocab,
            _ALIASES as openvocab_aliases,
        )

        if openvocab_key in openvocab_aliases:
            return LibreOpenVocab
    return None


def get_loaded_model_family(loaded_model: Any) -> Optional[str]:
    """Return the model family for native wrappers or exported-runtime backends."""
    family = getattr(loaded_model, "FAMILY", None)
    if family:
        return str(family)

    family = getattr(loaded_model, "model_family", None)
    if family:
        return str(family)

    get_model_name = getattr(loaded_model, "_get_model_name", None)
    if callable(get_model_name):
        try:
            family = get_model_name()
        except Exception:
            family = None
        if family:
            return str(family)

    return None


ImageSize = Union[int, Tuple[int, int]]


def _coerce_input_size(value: Any) -> ImageSize:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"input size must be int or (height, width), got {value}")
        h, w = int(value[0]), int(value[1])
        return h if h == w else (h, w)
    return int(value)


def parse_imgsz_str(imgsz: str | int | None) -> int | tuple[int, int] | None:
    """Parse an imgsz value from CLI string format.

    Accepts:
        "640"      -> 640
        "480x640"  -> (480, 640)
        "480X640"  -> (480, 640)
        None       -> None
        int        -> int (pass-through)
    """
    if imgsz is None:
        return None
    if isinstance(imgsz, int):
        if imgsz <= 0:
            raise ValueError(f"imgsz must be positive, got {imgsz}.")
        return imgsz
    s = str(imgsz).strip()
    if not s:
        return None
    if "x" in s.lower():
        parts = s.lower().split("x", 1)
        try:
            h, w = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid imgsz format: '{imgsz}'. Use 640 (square) or 480x640 (HxW)."
            )
        if h <= 0 or w <= 0:
            raise ValueError(f"imgsz dimensions must be positive, got ({h}, {w}).")
        return h if h == w else (h, w)
    try:
        value = int(s)
    except ValueError:
        raise ValueError(
            f"Invalid imgsz format: '{imgsz}'. Use 640 (square) or 480x640 (HxW)."
        )
    if value <= 0:
        raise ValueError(f"imgsz must be positive, got {value}.")
    return value


def get_loaded_model_input_size(
    loaded_model: Any,
    *,
    imgsz: Optional[ImageSize] = None,
    default: int = 640,
) -> ImageSize:
    """Return the effective input size for wrapper or backend output."""
    if imgsz is not None:
        return _coerce_input_size(imgsz)

    get_input_size = getattr(loaded_model, "_get_input_size", None)
    if callable(get_input_size):
        try:
            return _coerce_input_size(get_input_size())
        except Exception:
            pass

    input_size = getattr(loaded_model, "input_size", None)
    if input_size is not None:
        return _coerce_input_size(input_size)

    backend_imgsz = getattr(loaded_model, "imgsz", None)
    if backend_imgsz is not None:
        return _coerce_input_size(backend_imgsz)

    input_sizes = getattr(loaded_model, "INPUT_SIZES", None)
    size = getattr(loaded_model, "size", None)
    if isinstance(input_sizes, dict) and size is not None:
        return int(input_sizes.get(size, default))

    return default


def resolve_model_or_exit(out: OutputHandler, model: str) -> str:
    """Resolve a model reference or fail with a consistent CLI error."""
    from .config import get_all_cli_names, is_known_weight_filename, resolve_model_name
    from .errors import suggest_key

    model_path = resolve_model_name(model)
    if model_path != model or Path(model).exists() or is_known_weight_filename(model):
        return model_path

    all_names = get_all_cli_names()
    suggestion = suggest_key(model, all_names)
    hint = f" Did you mean '{suggestion}'?" if suggestion else ""
    exit_with_error(
        out,
        "model_not_found",
        f"Unknown model '{model}'.{hint}",
        suggestion=f"Available: {', '.join(all_names)}",
    )


def exit_stage_error(
    out: OutputHandler,
    *,
    stage: str,
    detail: Exception | str,
    code: str = "io_error",
    suggestion: Optional[str] = None,
) -> NoReturn:
    """Emit a stage-specific runtime error and terminate the command."""
    exit_with_error(
        out,
        code,
        f"{stage} failed: {detail}",
        suggestion=suggestion,
    )


def get_user_provided_params() -> Set[str]:
    """Return parameter names explicitly provided on the current command line."""
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        # typer >= 0.26 vendors its own click as typer._click with a separate
        # context stack — fall back to it so the test runner sees the same ctx
        # that KeyValueCommand.parse_args wrote user_provided into.
        try:
            from typer._click.globals import get_current_context as _typer_get_ctx

            ctx = _typer_get_ctx(silent=True)
        except ImportError:
            pass
    if ctx is None:
        return set()
    # KeyValueCommand.parse_args stores the set in ctx.meta for reliable access
    # regardless of click version or test-runner context behaviour.
    if "user_provided" in ctx.meta:
        return ctx.meta["user_provided"]
    # Fallback for plain click commands not using KeyValueCommand.
    return {
        p.name
        for p in ctx.command.params
        if ctx.get_parameter_source(p.name) == click.core.ParameterSource.COMMANDLINE
    }


def help_json_callback(
    ctx: typer.Context,
    param: typer.CallbackParam,
    value: bool,
) -> None:
    """Eager callback for --help-json: dump command schema and exit."""
    del param
    if not value:
        return

    params = []
    flags = []
    for p in ctx.command.params:
        if p.name in ("help_json", "help"):
            continue
        info: dict[str, Any] = {"name": p.name, "type": p.type.name}
        if p.default is not None:
            info["default"] = p.default
        if p.required:
            info["required"] = True
        if p.help:
            info["help"] = p.help
        params.append(info)
        if getattr(p, "is_flag", False):
            for opt in (*p.opts, *p.secondary_opts):
                if opt.startswith("--"):
                    flags.append(opt)

    schema = {
        "schema_version": 1,
        "command": ctx.info_name,
        "parameters": params,
        "flags": sorted(set(flags)),
    }
    print(json.dumps(schema, default=str))
    ctx.exit()
