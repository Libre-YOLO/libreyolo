"""PaddlePaddle export through an intermediate ONNX graph and X2Paddle."""

from __future__ import annotations

import importlib.util
import logging
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SUPPORTED_PADDLE_VERSION = "2.6.2"
_SUPPORTED_X2PADDLE_VERSION = "1.6.0"
_MAX_ONNX_VERSION = (1, 17)
_MAX_ONNX_OPSET = 15


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _major_minor(value: str) -> tuple[int, int]:
    parts = value.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as exc:
        raise ImportError(f"Could not parse dependency version {value!r}.") from exc


def check_paddle_export_available() -> None:
    """Validate the narrow converter stack covered by Paddle parity tests."""
    missing = [
        module
        for module in ("onnx", "paddle", "six", "x2paddle")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise ImportError(
            "Paddle export requires the optional Paddle toolchain "
            f"(missing: {', '.join(missing)}). Install with: "
            "pip install libreyolo[paddle]"
        )

    paddle_version = _package_version("paddlepaddle")
    x2paddle_version = _package_version("x2paddle")
    onnx_version = _package_version("onnx")
    if paddle_version != _SUPPORTED_PADDLE_VERSION:
        raise ImportError(
            "Paddle export is validated with paddlepaddle==2.6.2; got "
            f"{paddle_version or 'an unknown installation'}. Install the "
            "tested stack with: pip install libreyolo[paddle]"
        )
    if x2paddle_version != _SUPPORTED_X2PADDLE_VERSION:
        raise ImportError(
            "Paddle export is validated with x2paddle==1.6.0; got "
            f"{x2paddle_version or 'an unknown installation'}. Install the "
            "tested stack with: pip install libreyolo[paddle]"
        )
    if onnx_version is None or _major_minor(onnx_version) > _MAX_ONNX_VERSION:
        raise ImportError(
            "X2Paddle 1.6.0 requires ONNX <=1.17 for this export path; got "
            f"{onnx_version or 'an unknown installation'}. Install the tested "
            "stack with: pip install libreyolo[paddle]"
        )


def _normalize_onnx_for_x2paddle(onnx_path: str | Path) -> None:
    """Remove redundant ONNX defaults rejected by X2Paddle 1.6.0.

    ONNX defines omitted MaxPool dilation as one. PyTorch writes the explicit
    all-ones attribute, while X2Paddle 1.6.0 rejects it. Removing only that
    redundant default preserves the graph's specified operation.
    """
    import onnx

    path = Path(onnx_path)
    graph = onnx.load(str(path))
    opsets = [entry.version for entry in graph.opset_import if not entry.domain]
    if not opsets:
        raise ValueError("Intermediate ONNX graph does not declare a default opset.")
    if max(opsets) > _MAX_ONNX_OPSET:
        raise NotImplementedError(
            "Paddle export through X2Paddle 1.6.0 supports ONNX opset 15 or "
            f"lower, but the intermediate graph uses opset {max(opsets)}."
        )

    changed = False
    for node in graph.graph.node:
        if node.op_type != "MaxPool":
            continue
        for index in range(len(node.attribute) - 1, -1, -1):
            attribute = node.attribute[index]
            if (
                attribute.name == "dilations"
                and attribute.ints
                and all(value == 1 for value in attribute.ints)
            ):
                del node.attribute[index]
                changed = True
    if changed:
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(metadata, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def export_paddle(
    onnx_path: str,
    output_path: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Convert a static FP32 ONNX graph into a Paddle inference directory."""
    check_paddle_export_available()
    _normalize_onnx_for_x2paddle(onnx_path)

    from x2paddle.convert import onnx2paddle

    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=str(output_dir.parent)
    ) as temporary:
        temporary_root = Path(temporary)
        conversion_dir = temporary_root / "conversion"
        artifact_dir = temporary_root / "artifact"
        onnx2paddle(
            onnx_path,
            str(conversion_dir),
            enable_optim=False,
            disable_feedback=True,
            enable_onnx_checker=True,
        )

        generated = conversion_dir / "inference_model"
        required = (generated / "model.pdmodel", generated / "model.pdiparams")
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "X2Paddle did not produce a runnable Paddle inference model "
                f"(missing: {', '.join(missing)})."
            )

        artifact_dir.mkdir()
        for source in required:
            shutil.copy2(source, artifact_dir / source.name)
        parameters_info = generated / "model.pdiparams.info"
        if parameters_info.is_file():
            shutil.copy2(parameters_info, artifact_dir / parameters_info.name)
        _write_metadata(artifact_dir / "metadata.yaml", metadata or {})

        if output_dir.exists():
            if output_dir.is_dir():
                shutil.rmtree(output_dir)
            else:
                output_dir.unlink()
        shutil.move(str(artifact_dir), str(output_dir))

    logger.info("Paddle export complete: %s", output_dir)
    return str(output_dir)


__all__ = [
    "check_paddle_export_available",
    "export_paddle",
]
