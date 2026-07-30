"""Export command: export a model to a deployment format."""

from pathlib import Path
from typing import Optional

import typer

from ..command_utils import (
    COREML_VLM_EXPORT_ALIASES,
    exit_stage_error,
    exit_with_error,
    get_user_provided_params,
    help_json_callback,
    load_model_or_exit,
    resolve_export_sibling_factory,
    resolve_model_or_exit,
)
from ..output import OutputHandler


def export_cmd(
    model: str = typer.Option(
        ...,
        help="Model weights or a supported LibreSAM/LibreOpenVocab/LibreVLM alias",
    ),
    format: str = typer.Option(
        "onnx",
        help=(
            "Export format: onnx, torchscript, tensorrt, openvino, ncnn, "
            "tflite (alias: litert), coreml, coreai (Apple, macOS only)"
        ),
    ),
    imgsz: Optional[str] = typer.Option(
        None, help="Input image size (e.g. 640 or 640,480)"
    ),
    batch: int = typer.Option(1, help="Export batch size"),
    half: bool = typer.Option(False, help="FP16 precision"),
    int8: bool = typer.Option(False, help="INT8 quantization"),
    dynamic: bool = typer.Option(False, help="Dynamic input shapes (ONNX)"),
    simplify: bool = typer.Option(True, help="ONNX graph simplification"),
    nms: bool = typer.Option(
        False,
        help="Embed NMS in the model (ONNX YOLO9 detection or CoreML)",
    ),
    conf: float = typer.Option(0.25, help="Confidence threshold for embedded NMS"),
    iou: float = typer.Option(0.45, help="IoU threshold for embedded NMS"),
    max_det: int = typer.Option(300, help="Maximum detections for ONNX embedded NMS"),
    rec_max_width: Optional[int] = typer.Option(
        None,
        help=(
            "Finite maximum PP-OCR recognizer crop width for CoreML "
            "(required for PP-OCR CoreML export)"
        ),
    ),
    rec_batch_max: int = typer.Option(
        6,
        help="Maximum PP-OCR recognizer batch for CoreML",
    ),
    opset: Optional[int] = typer.Option(
        None, help="ONNX opset version (auto if omitted)"
    ),
    data: Optional[str] = typer.Option(None, help="Calibration data for INT8"),
    fraction: float = typer.Option(1.0, help="Fraction of calibration data"),
    device: str = typer.Option("auto", help="Device for tracing"),
    compute_units: str = typer.Option(
        "cpu_only",
        help=(
            "CoreML planner: validated, cpu_only, all, cpu_and_gpu, or "
            "cpu_and_ne"
        ),
    ),
    allow_download_scripts: bool = typer.Option(
        False,
        "--allow-download-scripts",
        help="Allow embedded Python in dataset YAML download blocks",
    ),
    # Agent flags
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    verbose: bool = typer.Option(False, help="Verbose export logging"),
    help_json: bool = typer.Option(
        False,
        "--help-json",
        is_eager=True,
        callback=help_json_callback,
        help="Dump command schema as JSON",
    ),
) -> None:
    """Export a model to a deployment format."""
    out = OutputHandler(json_mode=json_output, quiet=quiet)

    # Resolve format aliases (engine -> tensorrt, litert -> tflite) so JSON
    # output and messages always report the canonical format name.
    from libreyolo.export.exporter import BaseExporter

    fmt = format.lower()
    fmt = BaseExporter._aliases.get(fmt, fmt)
    user_provided_params = get_user_provided_params()
    if fmt == "coreml":
        from libreyolo.export.coreml_profiles import (
            normalize_coreml_compute_units,
        )

        try:
            compute_units = normalize_coreml_compute_units(compute_units)
        except ValueError as exc:
            exit_with_error(out, "config_unsupported", str(exc))
    elif "compute_units" in user_provided_params:
        exit_with_error(
            out,
            "config_unsupported",
            "--compute-units applies only to CoreML export.",
        )

    if half and int8:
        out.warning("Both half and int8 were requested. Using INT8 precision.")
        half = False

    if nms and fmt not in {"onnx", "coreml"}:
        exit_with_error(
            out,
            "nms_unsupported_format",
            f"Embedded NMS (--nms) is only supported for ONNX and CoreML, not {fmt!r}.",
        )
    if nms and fmt == "onnx" and dynamic:
        out.warning(
            "Embedded ONNX NMS uses a fixed batch-1 graph. Using dynamic=False."
        )
        dynamic = False
    if nms and fmt == "coreml" and max_det != 300:
        exit_with_error(
            out,
            "config_unsupported",
            "max_det is only supported for ONNX embedded NMS; CoreML embedded "
            "NMS does not expose max_det.",
        )

    sibling_factory = resolve_export_sibling_factory(model)
    coreml_vlm_alias = (
        fmt == "coreml" and str(model).strip().lower() in COREML_VLM_EXPORT_ALIASES
    )
    load_device = device
    if coreml_vlm_alias:
        normalized_device = str(device).strip().lower()
        if normalized_device not in {"auto", "cpu"}:
            exit_with_error(
                out,
                "config_unsupported",
                "Core ML VLM export requires a CPU-loaded FP32 source model; "
                f"device={device!r} is unsupported.",
            )
        # `auto` may select CUDA on a development workstation, but these
        # strict conversion profiles accept CPU FP32 source tensors only.
        load_device = "cpu"
    model_path = (
        model if sibling_factory is not None else resolve_model_or_exit(out, model)
    )

    if allow_download_scripts and data is not None:
        out.warning(
            "Dataset download scripts are enabled. Embedded Python from the dataset YAML may execute locally."
        )

    # Load model
    if sibling_factory is None:
        loaded_model = load_model_or_exit(
            out, model=model, model_path=model_path, device=load_device
        )
    else:
        loaded_model = load_model_or_exit(
            out,
            model=model,
            model_path=model_path,
            device=load_device,
            model_factory=sibling_factory,
        )

    # PP-OCR is the bounded-RangeDim multifunction exception. LibreSAM source
    # capture may be symbolic internally, but its public package materializes
    # exact fixed-P functions and must remain dynamic=False.
    if fmt == "coreml" and loaded_model.FAMILY == "ppocr":
        dynamic = True

    # Build export kwargs. The face component has a deliberately narrow
    # conversion contract; do not inject ONNX-only defaults that its public
    # exporter correctly rejects as irrelevant.
    facerec_coreml = fmt == "coreml" and loaded_model.FAMILY == "facerec"
    export_kwargs: dict = {
        "half": half,
        "int8": int8,
        "dynamic": dynamic,
        "batch": batch,
        "device": device,
    }
    if fmt == "coreml":
        export_kwargs["compute_units"] = compute_units
    if not facerec_coreml:
        export_kwargs.update(
            {
                "simplify": simplify,
                "opset": opset,
                "verbose": verbose,
            }
        )
    if fmt == "coreml" and loaded_model.FAMILY == "ppocr":
        export_kwargs["rec_batch_max"] = rec_batch_max
        if rec_max_width is not None:
            export_kwargs["rec_max_width"] = rec_max_width
    if nms:
        export_kwargs["nms"] = True
        export_kwargs["conf"] = conf
        export_kwargs["iou"] = iou
        if fmt == "onnx":
            export_kwargs["max_det"] = max_det
    if imgsz is not None:
        if "," in imgsz:
            parts = imgsz.split(",")
            if len(parts) != 2:
                exit_with_error(
                    out,
                    "invalid_imgsz",
                    f"Invalid imgsz format: {imgsz}. Use e.g. 640 or 640,480.",
                )
            try:
                export_kwargs["imgsz"] = (int(parts[0]), int(parts[1]))
            except ValueError:
                exit_with_error(
                    out,
                    "invalid_imgsz",
                    f"Invalid imgsz values: {imgsz}. Use integer dimensions.",
                )
        else:
            try:
                export_kwargs["imgsz"] = int(imgsz)
            except ValueError:
                exit_with_error(
                    out,
                    "invalid_imgsz",
                    f"Invalid imgsz: {imgsz}. Use e.g. 640 or 640,480.",
                )
    if data is not None:
        export_kwargs["data"] = data
    if data is not None or (int8 and not facerec_coreml):
        export_kwargs["fraction"] = fraction
        export_kwargs["allow_download_scripts"] = allow_download_scripts

    coreml_vlm = coreml_vlm_alias and loaded_model.FAMILY in {
        "florence2",
        "kosmos2",
        "qwen3vl",
        "smolvlm2",
    }
    if coreml_vlm:
        unsupported = []
        if half:
            unsupported.append("half")
        if int8:
            unsupported.append("int8")
        if dynamic:
            unsupported.append("dynamic")
        if batch != 1:
            unsupported.append("batch")
        if nms:
            unsupported.append("nms")
        if imgsz is not None:
            unsupported.append("imgsz")
        if opset is not None:
            unsupported.append("opset")
        if data is not None:
            unsupported.append("data")
        if verbose:
            unsupported.append("verbose")
        if rec_max_width is not None:
            unsupported.append("rec_max_width")
        if rec_batch_max != 6:
            unsupported.append("rec_batch_max")
        if "compute_units" in user_provided_params:
            unsupported.append("compute_units")
        for name in (
            "allow_download_scripts",
            "conf",
            "fraction",
            "iou",
            "max_det",
            "simplify",
        ):
            if name in user_provided_params:
                unsupported.append(name)
        if unsupported:
            exit_with_error(
                out,
                "config_unsupported",
                "Core ML VLM export does not accept generic graph options: "
                f"{', '.join(unsupported)}.",
            )
        # Stateful VLM exporters own precision, context, state, and package
        # policy. Generic image-graph defaults are semantically inapplicable
        # and their strict public APIs reject them.
        export_kwargs = {}

    # Run export
    out.progress(f"Exporting {model} to {fmt}...")
    try:
        output_path = loaded_model.export(format=fmt, **export_kwargs)
    except ValueError as e:
        if "Unsupported export format" in str(e):
            exit_with_error(
                out,
                "export_format_unknown",
                str(e),
                suggestion="Run: libreyolo formats",
            )
        else:
            exit_stage_error(out, stage="Export", detail=e)
    except ImportError as e:
        exit_with_error(out, "export_dep_missing", str(e))
    except NotImplementedError as e:
        exit_with_error(out, "format_precision_unsupported", str(e))
    except Exception as e:
        exit_stage_error(out, stage="Export", detail=e)

    # File size
    export_path = Path(output_path)
    if export_path.is_file():
        size_mb = export_path.stat().st_size / (1024 * 1024)
    elif export_path.is_dir():
        size_mb = sum(
            f.stat().st_size for f in export_path.rglob("*") if f.is_file()
        ) / (1024 * 1024)
    else:
        size_mb = 0.0

    if imgsz is not None and "," in imgsz:
        parts = imgsz.split(",")
        input_h, input_w = int(parts[0]), int(parts[1])
    elif imgsz is not None:
        input_h = input_w = int(imgsz)
    else:
        native = (
            loaded_model._get_input_size()
            if hasattr(loaded_model, "_get_input_size")
            else loaded_model.INPUT_SIZES.get(loaded_model.size, 640)
        )
        input_h = input_w = native

    if (
        loaded_model.FAMILY == "facerec"
        and str(loaded_model.cfg.layout).upper() == "NHWC"
    ):
        input_shape = [batch, input_h, input_w, 3]
    else:
        input_shape = [batch, 3, input_h, input_w]

    data_out = {
        "source_model": model,
        "model_family": loaded_model.FAMILY,
        "format": fmt,
        "output_path": str(output_path),
        "file_size_mb": round(size_mb, 1),
        "input_shape": input_shape,
        "dynamic": dynamic,
        "half": half,
        "int8": int8,
    }
    if fmt == "coreml":
        data_out["compute_units"] = compute_units

    if not json_output:
        data_out["_human_text"] = (
            f"Exported {loaded_model.FAMILY}-{loaded_model.size} to {fmt.upper()}: "
            f"{output_path} ({size_mb:.1f} MB)\n"
            f"  Input: {input_shape}, "
            f"dynamic={dynamic}, half={half}, int8={int8}"
        )

    out.result(data_out)
