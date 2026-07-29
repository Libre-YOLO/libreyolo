"""Predict command: run inference on images."""

import inspect
import time
from pathlib import Path
from typing import Optional

import typer

from ..command_utils import (
    exit_with_error,
    get_loaded_model_family,
    get_loaded_model_input_size,
    get_user_provided_params,
    help_json_callback,
    load_model_or_exit,
    parse_imgsz_str,
    resolve_model_or_exit,
)
from ..output import OutputHandler


_NATIVE_ONLY_PREDICT_KWARGS = {"tiling", "overlap_ratio", "output_file_format"}


def _call_accepts_kwarg(callable_obj, name: str) -> bool:
    """Return whether a callable accepts a keyword argument by name."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True

    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    return name in signature.parameters


def _build_predict_kwargs(
    loaded_model,
    *,
    out: OutputHandler,
    user_provided: set[str],
    conf: float,
    iou: float,
    imgsz: Optional[int | tuple[int, int]],
    classes: Optional[list[int]],
    max_det: int,
    save: bool,
    batch: int,
    output_path: Optional[str],
    color_format: str,
    tiling: bool,
    overlap_ratio: float,
    output_file_format: Optional[str],
) -> dict:
    """Build inference kwargs without sending native-only options to backends."""
    kwargs = {
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "classes": classes,
        "max_det": max_det,
        "save": save,
        "batch": batch,
        "output_path": output_path,
        "color_format": color_format,
        "tiling": tiling,
        "overlap_ratio": overlap_ratio,
        "output_file_format": output_file_format,
    }

    call = loaded_model.__call__
    unsupported = {
        name
        for name in _NATIVE_ONLY_PREDICT_KWARGS
        if not _call_accepts_kwarg(call, name)
    }
    if hasattr(loaded_model, "_run_inference"):
        unsupported.update(_NATIVE_ONLY_PREDICT_KWARGS)

    requested_unsupported = []
    if "tiling" in unsupported and tiling:
        requested_unsupported.append("tiling")
    if "output_file_format" in unsupported and output_file_format is not None:
        requested_unsupported.append("output_file_format")
    if "overlap_ratio" in unsupported and "overlap_ratio" in user_provided:
        requested_unsupported.append("overlap_ratio")

    if requested_unsupported:
        names = ", ".join(sorted(requested_unsupported))
        exit_with_error(
            out,
            "config_unsupported",
            f"Exported-runtime backends do not support these predict options: {names}.",
            suggestion="Use a native .pt model for tiled/custom-format inference, or omit these options.",
        )

    return {name: value for name, value in kwargs.items() if name not in unsupported}


def predict_cmd(
    source: str = typer.Option(..., help="Image path, directory, or URL"),
    model: str = typer.Option("yolox-s", help="Model name or path"),
    conf: float = typer.Option(0.25, help="Confidence threshold"),
    iou: float = typer.Option(0.45, help="NMS IoU threshold"),
    imgsz: Optional[str] = typer.Option(None, help="Input image size: 640 (square) or 480x640 (HxW)"),
    classes: Optional[str] = typer.Option(
        None, help="Filter by class IDs, e.g. [0,2,5]"
    ),
    max_det: int = typer.Option(300, help="Max detections per image"),
    half: bool = typer.Option(
        False, help="FP16 inference (CUDA only, requires model support)"
    ),
    save: bool = typer.Option(False, help="Save annotated images"),
    batch: int = typer.Option(
        1,
        help="Images per forward pass for directory sources "
        "(batch > 1 runs true batched inference on supported models)",
    ),
    tiling: bool = typer.Option(False, help="Tiled inference for large images"),
    overlap_ratio: float = typer.Option(0.2, help="Tile overlap ratio"),
    output_path: Optional[str] = typer.Option(None, help="Explicit output path"),
    color_format: str = typer.Option("auto", help="Input color: auto, rgb, bgr"),
    output_file_format: Optional[str] = typer.Option(
        None, help="Output format: jpg, png, webp"
    ),
    device: str = typer.Option("auto", help="Device: 0, cpu, mps, auto"),
    face_detector: Optional[str] = typer.Option(
        None,
        "--face-detector",
        help="Face detector model (path or CLI name); required for gaze models",
    ),
    gallery: Optional[str] = typer.Option(
        None,
        "--gallery",
        help="Face gallery (.npz from `libreyolo enroll`) to identify faces "
        "against; facial-recognition models only",
    ),
    gallery_threshold: float = typer.Option(
        0.4, help="Cosine threshold for a gallery identity match"
    ),
    project: str = typer.Option("runs/detect", help="Output directory root"),
    name: str = typer.Option("predict", help="Experiment name"),
    exist_ok: bool = typer.Option(False, help="Reuse existing output directory"),
    # Agent flags
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose stderr output"),
    help_json: bool = typer.Option(
        False,
        "--help-json",
        is_eager=True,
        callback=help_json_callback,
        help="Dump command schema as JSON",
    ),
) -> None:
    """Run inference on images."""
    from libreyolo.utils.general import increment_path

    out = OutputHandler(json_mode=json_output, quiet=quiet)
    user_provided = get_user_provided_params()
    try:
        imgsz = parse_imgsz_str(imgsz) if imgsz is not None else None
    except ValueError as exc:
        exit_with_error(out, "invalid_imgsz", str(exc))

    # Validate source exists
    source_path = Path(source)
    if not source_path.exists() and not source.startswith(("http://", "https://")):
        exit_with_error(out, "source_not_found", f"Source not found: {source}")

    # Resolve CLI model name (yolox-s → LibreYOLOXs.pt)
    model_path = resolve_model_or_exit(out, model)

    # Load model
    loaded_model = load_model_or_exit(
        out, model=model, model_path=model_path, device=device
    )

    # Gaze models (LibreL2CS) are two-stage: they need an upstream face
    # detector. The factory builds them without one, so wire it here from
    # the --face-detector option.
    if getattr(loaded_model, "task", None) == "gaze":
        if face_detector is None:
            exit_with_error(
                out,
                "config_unsupported",
                "Gaze models require a face detector. Pass --face-detector "
                "<model> (a LibreYOLO detector trained on faces).",
                suggestion="e.g. --face-detector path/to/face-detector.pt",
            )
        fd_path = resolve_model_or_exit(out, face_detector)
        fd_model = load_model_or_exit(
            out, model=face_detector, model_path=fd_path, device=device
        )
        from libreyolo.models.l2cs.face import resolve_face_detector

        loaded_model.face_detector = resolve_face_detector(fd_model)

    # Face-embedding models: optional --face-detector override (the family
    # otherwise falls back to its auto-downloaded default detector) and
    # optional --gallery for 1:N identification.
    gallery_obj = None
    if getattr(loaded_model, "task", None) == "embed":
        if face_detector is not None:
            from .special import _resolve_embed_face_detector

            loaded_model.face_detector = _resolve_embed_face_detector(
                out, face_detector, device
            )
        if gallery is not None:
            if not Path(gallery).exists():
                exit_with_error(
                    out,
                    "source_not_found",
                    f"Gallery not found: {gallery}. Build one with "
                    "`libreyolo enroll model=... source=people/ gallery=...`.",
                )
            from libreyolo.models.facerec import FaceGallery

            try:
                gallery_obj = FaceGallery.load(gallery, embedder=loaded_model)
            except ValueError as exc:
                exit_with_error(out, "config_unsupported", str(exc))
    elif gallery is not None:
        exit_with_error(
            out,
            "config_unsupported",
            "--gallery is only supported for embed-task models "
            "(e.g. model=facerec-l, or a CLIP/SigLIP2/DINOv2 model loaded "
            "with task=embed).",
        )

    # FP16 (half) precision: exported runtimes (ONNX, TensorRT, ...) accept the
    # flag and run in FP16, so forward it there. For native PyTorch inference it
    # is not yet wired (model converts to FP16 but input stays FP32 → dtype
    # mismatch), so warn and skip for those.
    is_exported_backend = hasattr(loaded_model, "_run_inference")
    if half and not is_exported_backend:
        out.progress(
            "Warning: half (FP16) is not yet supported for PyTorch inference. "
            "Use exported models (ONNX/TensorRT) for FP16. Ignoring."
        )

    # Resolve output path
    if output_path is None and save:
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok, mkdir=True)
        output_path = str(save_dir)

    # Parse classes list
    parsed_classes = None
    if classes is not None:
        import ast

        try:
            evaluated = ast.literal_eval(classes)
        except (ValueError, SyntaxError):
            exit_with_error(
                out, "config_type_error", f"Invalid classes value: {classes}"
            )
        # Accept a bare int (classes=0), a list (classes=[0,2,5]), or a
        # comma-separated string that parses to a tuple (classes="0,2").
        if isinstance(evaluated, int):
            parsed_classes = [evaluated]
        else:
            try:
                parsed_classes = list(evaluated)
            except TypeError:
                exit_with_error(
                    out, "config_type_error", f"Invalid classes value: {classes}"
                )

    # Run inference
    out.progress(f"Running inference on {source}...")
    t0 = time.time()
    predict_kwargs = _build_predict_kwargs(
        loaded_model,
        out=out,
        user_provided=user_provided,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=parsed_classes,
        max_det=max_det,
        save=save,
        batch=batch,
        output_path=output_path,
        color_format=color_format,
        tiling=tiling,
        overlap_ratio=overlap_ratio,
        output_file_format=output_file_format,
    )
    if half and is_exported_backend:
        predict_kwargs["half"] = half
    if gallery_obj is not None:
        predict_kwargs["gallery"] = gallery_obj
        predict_kwargs["threshold"] = gallery_threshold
    results = loaded_model(source, **predict_kwargs)
    elapsed = time.time() - t0

    # Normalize to list
    if not isinstance(results, list):
        results = [results]
    total_images = len(results)

    # Format output
    result_list = []
    human_lines = []
    import math as _math

    for r in results:
        boxes = r.boxes
        detections = []
        # Count detections per class
        class_counts: dict[str, int] = {}
        h, w = r.orig_shape
        img_name = Path(r.path or source).name
        idx = len(human_lines) + 1
        elapsed_ms = elapsed * 1000 / max(len(results), 1)

        if boxes is None:
            result_data = {
                "path": r.path or str(source),
                "original_shape": list(r.orig_shape),
                "detections": detections,
            }
            if getattr(r, "ocr", None) is not None:
                ocr_np = r.ocr.numpy()
                regions = []
                for i in range(len(ocr_np)):
                    polygon = ocr_np.data[i]
                    regions.append(
                        {
                            "text": ocr_np.texts[i],
                            "confidence": round(float(ocr_np.conf[i]), 4),
                            "det_confidence": round(float(ocr_np.det_conf[i]), 4),
                            "polygon": [
                                [round(float(x), 1), round(float(y), 1)]
                                for x, y in polygon
                            ],
                        }
                    )
                result_data["ocr"] = regions
                n = len(regions)
                summary = f"{n} text region{'s' if n != 1 else ''}"
            elif getattr(r, "points", None) is not None:
                points_np = r.points.numpy()
                for i in range(len(points_np)):
                    cls_id = int(points_np.cls[i])
                    cls_name = r.names.get(cls_id, str(cls_id))
                    detections.append(
                        {
                            "class": cls_name,
                            "class_id": cls_id,
                            "confidence": round(float(points_np.conf[i]), 4),
                            "point_xy": [
                                round(float(points_np.xy[i, 0]), 1),
                                round(float(points_np.xy[i, 1]), 1),
                            ],
                        }
                    )
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                result_data["detections"] = detections
                summary = ", ".join(
                    f"{v} {k}{'s' if v > 1 else ''}"
                    for k, v in class_counts.items()
                ) or "(no detections)"
            elif getattr(r, "probs", None) is not None:
                top_rows = r.summary()
                result_data["classification"] = top_rows[0] if top_rows else None
                result_data["top5"] = top_rows
                top = top_rows[0] if top_rows else None
                if top is not None:
                    summary = f"{top['name']} {top['confidence']:.4f}"
                else:
                    summary = "(no classification)"
            elif getattr(r, "restored", None) is not None:
                restored = r.restored.array
                result_data["restored"] = {
                    "shape": list(restored.shape),
                    "dtype": str(restored.dtype),
                }
                summary = "restored"
            elif getattr(r, "depth_map", None) is not None:
                depth_map = r.depth_map
                result_data["depth"] = {
                    "shape": list(depth_map.data.shape),
                    "min": round(float(depth_map.min), 4),
                    "max": round(float(depth_map.max), 4),
                    "mean": round(float(depth_map.mean), 4),
                }
                summary = (
                    f"depth min={depth_map.min:.4g} "
                    f"max={depth_map.max:.4g} mean={depth_map.mean:.4g}"
                )
            else:
                summary = "(no detections)"

            result_list.append(result_data)
            human_lines.append(
                f"image {idx}/{total_images} {img_name}: "
                f"{w}x{h} "
                f"{summary}, "
                f"{elapsed_ms:.1f}ms avg"
            )
            continue

        gaze_data = r.gaze if getattr(r, "gaze", None) is not None else None
        if gaze_data is not None and hasattr(gaze_data.data, "cpu"):
            gaze_np = gaze_data.numpy()
        else:
            gaze_np = gaze_data
        obb_data = r.obb if getattr(r, "obb", None) is not None else None
        if obb_data is not None and hasattr(obb_data.data, "cpu"):
            obb_np = obb_data.numpy()
        else:
            obb_np = obb_data
        kpts_data = r.keypoints if getattr(r, "keypoints", None) is not None else None
        if kpts_data is not None and hasattr(kpts_data.data, "cpu"):
            kpts_np = kpts_data.numpy()
        else:
            kpts_np = kpts_data
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = r.names.get(cls_id, str(cls_id))
            det = {
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(float(boxes.conf[i]), 4),
                "bbox_xyxy": [round(float(c), 1) for c in boxes.xyxy[i]],
            }
            if obb_np is not None and i < len(obb_np):
                xywhr = obb_np.xywhr[i]
                corners = obb_np.xyxyxyxy[i]
                det["obb"] = {
                    "xywhr": [round(float(v), 4) for v in xywhr],
                    "corners": [
                        [round(float(x), 4), round(float(y), 4)] for x, y in corners
                    ],
                }
            if gaze_np is not None and i < len(gaze_np):
                pitch = float(gaze_np.data[i, 0])
                yaw = float(gaze_np.data[i, 1])
                det["gaze"] = {
                    "pitch_rad": round(pitch, 4),
                    "yaw_rad": round(yaw, 4),
                    "pitch_deg": round(pitch * 180.0 / _math.pi, 2),
                    "yaw_deg": round(yaw * 180.0 / _math.pi, 2),
                }
            if kpts_np is not None and i < len(kpts_np):
                kp_xy = kpts_np.xy[i]
                kp_conf = kpts_np.conf
                keypoints = []
                for k in range(len(kp_xy)):
                    kp = {
                        "xy": [
                            round(float(kp_xy[k, 0]), 1),
                            round(float(kp_xy[k, 1]), 1),
                        ],
                    }
                    if kp_conf is not None:
                        kp["confidence"] = round(float(kp_conf[i, k]), 4)
                    keypoints.append(kp)
                det["keypoints"] = keypoints
            identities = getattr(r, "identities", None)
            if identities is not None and i < len(identities):
                det["identity"] = identities.name[i]
                score = float(identities.score[i])
                det["identity_score"] = round(score, 4) if score == score else None
            detections.append(det)
            count_key = det.get("identity") or cls_name
            class_counts[count_key] = class_counts.get(count_key, 0) + 1

        result_data = {
            "path": r.path or str(source),
            "original_shape": list(r.orig_shape),
            "detections": detections,
        }
        if r.masks is not None:
            result_data["masks"] = {
                "count": len(r.masks),
                "shape": list(r.masks.data.shape),
            }
        result_list.append(result_data)
        # Human summary line:
        # image 1/3 parkour.jpg: 640x480 3 persons, 2 skateboards, 45.2ms
        counts_str = ", ".join(
            f"{v} {k}{'s' if v > 1 else ''}" for k, v in class_counts.items()
        )
        human_lines.append(
            f"image {idx}/{total_images} {img_name}: "
            f"{w}x{h} "
            f"{counts_str or '(no detections)'}, "
            f"{elapsed_ms:.1f}ms avg"
        )

    image_size = get_loaded_model_input_size(loaded_model, imgsz=imgsz)
    if isinstance(image_size, tuple):
        image_shape = [image_size[0], image_size[1]]
    else:
        image_shape = [image_size, image_size]
    data = {
        "source": str(source),
        "model": model,
        "model_family": get_loaded_model_family(loaded_model) or "unknown",
        "image_size": image_shape,
        "device": str(loaded_model.device),
        "results": result_list,
    }
    if save and output_path:
        data["output_path"] = output_path

    if not json_output:
        if save and output_path:
            human_lines.append(f"Results saved to {output_path}")
        data["_human_text"] = "\n".join(human_lines)

    out.result(data)
