"""Train command: train a model on a dataset."""

import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import typer

from ..command_utils import (
    exit_stage_error,
    exit_with_error,
    get_loaded_model_family,
    get_user_provided_params,
    help_json_callback,
    load_model_or_exit,
    parse_imgsz_str,
    resolve_model_or_exit,
)
from ..config import (
    apply_family_defaults,
    build_family_train_kwargs,
    build_vlm_train_kwargs,
    detect_family_from_model_ref,
    get_model_class,
    get_unsupported_train_params,
)
from ..output import OutputHandler
from ...training.freezing import normalize_freeze_selectors, parse_freeze_spec


_LORA_TRAIN_FAMILIES = {
    "rfdetr",
    "dfine",
    "deim",
    "deimv2",
    "rtdetr",
    "rtdetrv2",
    "rtdetrv4",
    "ec",
    "convnext",
}

# Always own their train loader; class_balanced never reaches create_dataloader.
# EC / YOLO-NAS are task-dependent and are rejected at trainer setup instead.
_CLASS_BALANCED_UNSUPPORTED_FAMILIES = {
    "dfine",
    "deim",
    "deimv2",
    "fomo",
    "vjepa2",
}

_VLM_TRAIN_CLI_PARAMS = {
    "allow_download_scripts",
    "batch",
    "data",
    "device",
    "dry_run",
    "epochs",
    "exist_ok",
    "flip_prob",
    "help_json",
    "json_output",
    "lora",
    "lr0",
    "model",
    "name",
    "pretrained",
    "project",
    "quiet",
    "resume",
    "seed",
    "task",
    "val",
    "workers",
}


def _model_ref_exists(model_path: str) -> bool:
    path = Path(model_path)
    if path.exists():
        return True
    return path.parent == Path(".") and (Path("weights") / path.name).exists()


def _create_explicit_task_train_model(
    *,
    family: str | None,
    model_path: str,
    task: str | None,
    resume: bool | str,
    device: str,
    pretrained: bool = True,
    seed: int = 0,
):
    """Build a known architecture when training must not load a checkpoint.

    Scratch training covers every G0/G1/G2 family. The older transfer path below
    remains limited to families that need a task-specific architecture before
    loading their published checkpoint.
    """
    from libreyolo.tasks import normalize_task

    if pretrained is False and not resume:
        from libreyolo.models.registry import group_of

        model_cls = get_model_class(family)
        if model_cls is not None and group_of(family) in {"g0", "g1", "g2"}:
            size = model_cls.detect_size_from_filename(Path(model_path).name)
            if size is None:
                return None
            train_task = (
                normalize_task(task)
                if task is not None
                else model_cls.detect_task_from_filename(Path(model_path).name)
                or model_cls.DEFAULT_TASK
            )
            scratch_kwargs = {}
            variant = model_cls.detect_variant_from_filename(Path(model_path).name)
            if variant is not None:
                scratch_kwargs["weight_variant"] = variant
            return model_cls._from_scratch(
                size=size,
                task=train_task,
                device=device,
                seed=seed,
                **scratch_kwargs,
            )

    if family not in {"yolo9", "rfdetr", "dfine"} or resume:
        return None

    if family == "yolo9":
        from libreyolo.models.yolo9.model import LibreYOLO9 as model_cls
    elif family == "dfine":
        from libreyolo.models.dfine.model import LibreDFINE as model_cls
    else:
        from libreyolo.models.rfdetr.model import LibreRFDETR as model_cls

    filename_task = model_cls.detect_task_from_filename(Path(model_path).name)
    train_task = normalize_task(task) if task is not None else filename_task
    if train_task is None:
        return None
    if family == "dfine" and train_task != "segment":
        return None
    if task is None and filename_task == train_task and _model_ref_exists(model_path):
        return None

    size = model_cls.detect_size_from_filename(Path(model_path).name)
    if size is None:
        return None
    if family == "dfine" and train_task == "segment":
        if not _model_ref_exists(model_path):
            # Published weights (LibreDFINEn-seg.pt or a detect checkpoint used
            # as transfer source) must auto-download here; falling through to
            # the scratch path would silently train uninitialized.
            url = model_cls.get_download_url(Path(model_path).name)
            if url:
                from libreyolo.utils.download import download_weights

                dl_path = Path(model_path)
                if dl_path.parent == Path("."):
                    dl_path = Path("weights") / dl_path.name
                download_weights(str(dl_path), size)
                model_path = str(dl_path)
        if _model_ref_exists(model_path):
            # Detect checkpoints are legal segment-training starting points,
            # but only as an explicit transfer (the mask head starts
            # untrained). Seg checkpoints carry mask keys and load normally;
            # the flag is inert for them.
            return model_cls(
                model_path,
                size=size,
                task=train_task,
                device=device,
                allow_detect_to_segment_transfer=True,
            )
    if family == "rfdetr" and train_task == "obb" and _model_ref_exists(model_path):
        return model_cls(
            model_path,
            size=size,
            task=train_task,
            device=device,
            allow_detect_to_obb_transfer=True,
        )
    if family == "rfdetr" and train_task == "pose" and _model_ref_exists(model_path):
        return model_cls(
            model_path,
            size=size,
            task=train_task,
            device=device,
            allow_detect_to_pose_transfer=True,
        )
    extra = (
        {"allow_detect_to_obb_transfer": True}
        if family == "rfdetr" and train_task == "obb"
        else {}
    )
    return model_cls(None, size=size, task=train_task, device=device, **extra)


def _create_rfdetr_obb_from_loaded_detect_model(
    loaded_model,
    *,
    model_path: str,
    device: str,
):
    """Switch an already-loaded RF-DETR detect checkpoint to OBB architecture."""
    if (
        get_loaded_model_family(loaded_model) != "rfdetr"
        or getattr(loaded_model, "task", "detect") != "detect"
    ):
        return None

    from libreyolo.models.rfdetr.model import LibreRFDETR

    return LibreRFDETR(
        model_path,
        size=getattr(loaded_model, "size", None),
        task="obb",
        device=device,
        allow_detect_to_obb_transfer=True,
    )


def _create_rfdetr_pose_from_loaded_detect_model(
    loaded_model,
    *,
    model_path: str,
    device: str,
):
    """Switch an already-loaded RF-DETR detect checkpoint to pose architecture."""
    if (
        get_loaded_model_family(loaded_model) != "rfdetr"
        or getattr(loaded_model, "task", "detect") != "detect"
    ):
        return None

    from libreyolo.models.rfdetr.model import LibreRFDETR

    return LibreRFDETR(
        model_path,
        size=getattr(loaded_model, "size", None),
        task="pose",
        device=device,
        allow_detect_to_pose_transfer=True,
    )


def _create_dfine_segment_from_loaded_detect_model(
    loaded_model,
    *,
    model_path: str,
    device: str,
):
    """Switch an already-loaded D-FINE detect checkpoint to the segment architecture."""
    if (
        get_loaded_model_family(loaded_model) != "dfine"
        or getattr(loaded_model, "task", "detect") != "detect"
    ):
        return None

    from libreyolo.models.dfine.model import LibreDFINE

    return LibreDFINE(
        model_path,
        size=getattr(loaded_model, "size", None),
        task="segment",
        device=device,
        allow_detect_to_segment_transfer=True,
    )


def _create_yolo9_task_from_loaded_model(loaded_model, task: str, device: str):
    if get_loaded_model_family(loaded_model) != "yolo9":
        return None

    from libreyolo.models.yolo9.model import LibreYOLO9

    if task not in LibreYOLO9.SUPPORTED_TASKS:
        return None
    size = getattr(loaded_model, "size", None)
    if size is None:
        return None
    return LibreYOLO9(None, size=size, task=task, device=device)


def _should_use_yolo9_path_as_transfer(model_path: str, task: str | None) -> bool:
    if task is None or not Path(model_path).exists():
        return False

    from libreyolo.models.yolo9.model import LibreYOLO9

    filename_task = LibreYOLO9.detect_task_from_filename(Path(model_path).name)
    return filename_task != task


def _validate_vlm_train_request(
    out: OutputHandler,
    reference,
    *,
    user_provided: set[str],
    task: str | None,
    pretrained: bool,
    val: bool,
) -> None:
    """Fail before loading weights when the VLM train request is unsupported."""
    if not reference.trainable:
        exit_with_error(
            out,
            "config_unsupported",
            f"VLM training is not supported for family {reference.family!r}.",
            suggestion="Verified VLM training is currently limited to Qwen3-VL 2B and 4B.",
        )
    if reference.size not in reference.trainable_sizes:
        supported = ", ".join(reference.trainable_sizes) or "none"
        exit_with_error(
            out,
            "config_unsupported",
            f"VLM training is not verified for {reference.family} size "
            f"{reference.size!r}. Verified sizes: {supported}.",
        )
    if reference.checkpoint:
        exit_with_error(
            out,
            "config_unsupported",
            "Training directly from a loaded VLM checkpoint is unsafe because "
            "its adapter is merged for inference.",
            suggestion=(
                "Use the corresponding base-model alias and pass "
                "resume=<checkpoint directory>."
            ),
        )
    if task is not None and task != "detect":
        exit_with_error(
            out,
            "config_unsupported",
            "VLM fine-tuning currently supports task='detect' only.",
        )
    unsupported = sorted(user_provided - _VLM_TRAIN_CLI_PARAMS)
    if unsupported:
        exit_with_error(
            out,
            "config_unsupported",
            "VLM training does not support these detector training options: "
            f"{', '.join(unsupported)}.",
            suggestion="Remove the listed options; VLM recipes own optimizer, scheduler, and augmentation settings.",
        )
    if "pretrained" in user_provided and not pretrained:
        exit_with_error(
            out,
            "config_unsupported",
            "VLM training requires its pinned pretrained base; pretrained=false is unsupported.",
        )
    if "val" in user_provided and not val:
        exit_with_error(
            out,
            "config_unsupported",
            "VLM validation-loss evaluation follows the dataset val split and cannot be disabled with val=false.",
        )


def _vlm_dry_run_config(reference, params: dict, train_kwargs: dict) -> dict:
    from libreyolo.models.vlm.training.recipes import get_recipe

    recipe = get_recipe(reference.family)
    lr0 = train_kwargs["lr0"]
    if lr0 is None:
        lr0 = recipe.lr0 if train_kwargs["lora"] else recipe.full_ft_lr0
    return {
        "model": params["model"],
        "data": params["data"],
        "family": reference.family,
        "size": reference.size,
        "task": "detect",
        "epochs": train_kwargs["epochs"],
        "batch": train_kwargs["batch"],
        "accumulate": train_kwargs["accumulate"],
        "workers": train_kwargs["workers"],
        "seed": train_kwargs["seed"],
        "device": train_kwargs["device"],
        "lr0": lr0,
        "lora": train_kwargs["lora"],
        "gradient_checkpointing": train_kwargs["gradient_checkpointing"],
        "vram_check": train_kwargs["vram_check"],
        "hflip": train_kwargs["hflip"],
        "project": train_kwargs["project"],
        "name": train_kwargs["name"],
        "exist_ok": train_kwargs["exist_ok"],
        "resume": train_kwargs["resume"],
        "allow_download_scripts": train_kwargs["allow_download_scripts"],
    }


def _validate_vlm_train_values(
    out: OutputHandler, train_kwargs: dict, *, data: str
) -> None:
    """Validate the VLM subset before a dry run or model download."""

    if not isinstance(data, str) or not data.strip():
        exit_with_error(
            out,
            "config_type_error",
            "VLM data must be a non-empty dataset YAML path or built-in name.",
        )

    def finite_real(value) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            converted = float(value)
        except (OverflowError, TypeError, ValueError):
            return None
        return converted if math.isfinite(converted) else None

    for name in ("epochs", "batch"):
        value = train_kwargs[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            exit_with_error(
                out,
                "config_type_error",
                f"VLM {name} must be an integer >= 1, got {value!r}.",
            )
    workers = train_kwargs["workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        exit_with_error(
            out,
            "config_type_error",
            f"VLM workers must be an integer >= 0, got {workers!r}.",
        )
    raw_hflip = train_kwargs["hflip"]
    hflip = finite_real(raw_hflip)
    if hflip is None or not 0.0 <= hflip <= 1.0:
        exit_with_error(
            out,
            "config_type_error",
            f"VLM flip_prob must be finite and within [0, 1], got {raw_hflip!r}.",
        )
    raw_lr0 = train_kwargs["lr0"]
    lr0 = finite_real(raw_lr0) if raw_lr0 is not None else None
    if raw_lr0 is not None and (lr0 is None or lr0 <= 0.0):
        exit_with_error(
            out,
            "config_type_error",
            f"VLM lr0 must be a finite positive number, got {raw_lr0!r}.",
        )
    for name in ("project", "name"):
        value = train_kwargs[name]
        if not isinstance(value, str) or not value.strip():
            exit_with_error(
                out,
                "config_type_error",
                f"VLM {name} must be a non-empty string.",
            )


def _preflight_vlm_dataset(
    out: OutputHandler,
    data: str,
    *,
    allow_download_scripts: bool,
    prepare: bool,
) -> None:
    """Validate the dataset and prepare it before a real model load."""
    import yaml

    from libreyolo.data import get_img_files, load_data_config
    from libreyolo.models.vlm.training.data import (
        VLMDetectDataset,
        resolve_split_annotation,
        resolve_split_source,
    )
    from libreyolo.models.vlm.training.trainer import _normalize_names

    try:
        config = load_data_config(
            data,
            autodownload=prepare,
            allow_scripts=allow_download_scripts if prepare else False,
        )
        names = _normalize_names(config.get("names"), config.get("nc"))
        train_source = resolve_split_source(config, "train")
        if not train_source:
            raise ValueError(f"Dataset {data!r} has no train split.")
        train_annotation = resolve_split_annotation(config, "train")
        val_source = resolve_split_source(config, "val")
        val_annotation = resolve_split_annotation(config, "val")
        if val_annotation and not val_source:
            raise ValueError(
                "Dataset declares native COCO validation annotations but has "
                "no val image source."
            )
        download_spec = config.get("download")
        can_download = False
        if download_spec is not None:
            if not isinstance(download_spec, str) or not download_spec.strip():
                raise ValueError("Dataset download must be a non-empty string.")
            stripped_download = download_spec.strip().lower()
            can_download = (
                stripped_download.startswith(("http://", "https://"))
                or allow_download_scripts
            )

        split_sources = {"train": train_source}
        if val_source:
            split_sources["val"] = val_source

        available = {}
        annotations = {}
        for split, source in split_sources.items():
            annotations[split] = (
                train_annotation if split == "train" else val_annotation
            )
            source_paths = source if isinstance(source, list) else [source]
            if any(not Path(path).exists() for path in source_paths):
                available[split] = None
                continue
            image_files = get_img_files(source)
            missing_images = [
                str(path) for path in image_files if not Path(path).is_file()
            ]
            if missing_images:
                raise FileNotFoundError(
                    f"Dataset {data!r} {split} split lists missing image files: "
                    f"{missing_images[:3]}."
                )
            available[split] = image_files or None

        # The shared dataset loader downloads only when every configured split
        # is absent. Once any split exists it will not repair another missing
        # split, so fail that partial state before loading model weights.
        annotation_exists = {
            split: annotation is None or Path(annotation).is_file()
            for split, annotation in annotations.items()
        }
        if not (not prepare and can_download and not any(available.values())):
            missing = [
                split
                for split, files in available.items()
                if not files or not annotation_exists[split]
            ]
            if missing:
                raise FileNotFoundError(
                    f"Dataset {data!r} has incomplete image/annotation data "
                    "for split(s): "
                    f"{', '.join(missing)}."
                )
            for split, annotation in annotations.items():
                if annotation is not None:
                    VLMDetectDataset.validate_native_coco_source(
                        split_sources[split], names, annotation
                    )
    except FileNotFoundError as exc:
        exit_with_error(out, "data_not_found", str(exc))
    except NotImplementedError as exc:
        exit_with_error(out, "config_unsupported", str(exc))
    except ImportError as exc:
        exit_with_error(out, "config_unsupported", str(exc))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        exit_with_error(out, "config_type_error", str(exc))


def _preflight_vlm_training_device(out: OutputHandler, requested) -> None:
    """Reject unsupported CUDA training precision before loading weights."""
    from libreyolo.models.vlm.training.trainer import (
        VLMDetectionTrainer,
        resolve_vlm_training_device,
    )

    try:
        device = resolve_vlm_training_device(requested, "cpu")
        VLMDetectionTrainer._training_dtype(device)
    except ValueError as exc:
        exit_with_error(out, "config_type_error", str(exc))
    except (AssertionError, RuntimeError) as exc:
        exit_with_error(out, "config_unsupported", str(exc))


def _vlm_reference_metadata(reference):
    """Build the resume identity for a VLM alias without model construction."""
    from libreyolo.models.vlm import _ALIASES

    match = next(
        (
            family_cls
            for family_cls, size in set(_ALIASES.values())
            if family_cls.FAMILY == reference.family and size == reference.size
        ),
        None,
    )
    if match is None:
        raise ValueError(
            f"No model metadata exists for {reference.family} {reference.size}."
        )
    metadata = SimpleNamespace(
        FAMILY=match.FAMILY,
        size=reference.size,
        HF_REPOS=match.HF_REPOS,
        HF_REVISIONS=match.HF_REVISIONS,
        BBOX_KEY=match.BBOX_KEY,
        COORD_DIVISOR=match.COORD_DIVISOR,
        BOX_FORMAT=match.BOX_FORMAT,
        _checkpoint_dir=None,
        _custom_prompt=None,
    )
    metadata._format_detection_prompt = lambda labels: match._format_detection_prompt(
        metadata, labels
    )
    return metadata


def _preflight_vlm_resume(
    out: OutputHandler,
    reference,
    train_kwargs: dict,
) -> None:
    """Validate a local resume adapter before downloading the base model."""
    resume = train_kwargs["resume"]
    if not resume:
        return
    if not train_kwargs["lora"]:
        exit_with_error(
            out,
            "config_unsupported",
            "VLM resume= is supported only with LoRA training (lora=true).",
        )
    if resume is True:
        if not train_kwargs["exist_ok"]:
            exit_with_error(
                out,
                "config_unsupported",
                "VLM resume=true requires exist_ok=true so the existing run "
                "directory is not incremented.",
            )
        candidate = (
            Path(train_kwargs["project"]) / train_kwargs["name"] / "weights" / "last"
        )
    else:
        candidate = Path(resume)

    from libreyolo.models.vlm.training.trainer import (
        validate_vlm_resume_checkpoint,
    )

    try:
        metadata = _vlm_reference_metadata(reference)
        validate_vlm_resume_checkpoint(candidate, metadata)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        exit_with_error(out, "config_unsupported", str(exc))


def _run_vlm_train(
    out: OutputHandler,
    reference,
    *,
    model: str,
    model_path: str,
    data: str,
    device: str,
    params: dict,
    user_provided: set[str],
    dry_run: bool,
    json_output: bool,
) -> None:
    train_kwargs = build_vlm_train_kwargs(params, user_provided=user_provided)
    _validate_vlm_train_values(out, train_kwargs, data=data)
    _preflight_vlm_training_device(out, train_kwargs["device"])
    _preflight_vlm_resume(out, reference, train_kwargs)
    _preflight_vlm_dataset(
        out,
        data,
        allow_download_scripts=train_kwargs["allow_download_scripts"],
        prepare=False,
    )
    resolved_config = _vlm_dry_run_config(
        reference,
        {"model": model, "data": data},
        train_kwargs,
    )
    if dry_run:
        data_out = {
            "valid": True,
            "mode": "train",
            "model_family": reference.family,
            "resolved_config": resolved_config,
        }
        if not json_output:
            import yaml

            data_out["_human_text"] = (
                f"Dry run: resolved VLM config for {model}:\n"
                + yaml.dump(resolved_config, default_flow_style=False)
            )
        out.result(data_out)
        return

    if train_kwargs["lora"]:
        from libreyolo.models.vlm.training.trainer import (
            require_vlm_lora_dependencies,
        )

        try:
            require_vlm_lora_dependencies()
        except ImportError as exc:
            exit_with_error(out, "config_unsupported", str(exc))

    if train_kwargs["allow_download_scripts"]:
        out.warning(
            "Dataset download scripts are enabled. Embedded Python from the "
            "dataset YAML may execute locally."
        )
    _preflight_vlm_dataset(
        out,
        data,
        allow_download_scripts=train_kwargs["allow_download_scripts"],
        prepare=True,
    )
    loaded_model = load_model_or_exit(
        out,
        model=model,
        model_path=model_path,
        device=device,
        task="detect",
    )
    out.progress(f"Training {model} on {data} for {train_kwargs['epochs']} epochs...")
    t0 = time.time()
    try:
        results = loaded_model.train(data=data, **train_kwargs)
    except FileNotFoundError as exc:
        exit_with_error(
            out,
            "data_not_found",
            str(exc),
            suggestion=f"Check that '{data}' exists and is a valid YOLO-format dataset YAML.",
        )
    except Exception as exc:
        exit_stage_error(out, stage="Training", detail=exc)

    training_hours = (time.time() - t0) / 3600
    epochs_completed = int(results.get("epochs", train_kwargs["epochs"]))
    metric_name = results.get("metric_name")
    best_metric = results.get("best_metric")
    best_epoch = results.get("best_epoch")
    best = results.get("best")
    last = results.get("last")
    save_dir = results.get(
        "save_dir", str(Path(train_kwargs["project"]) / train_kwargs["name"])
    )
    data_out = {
        "status": "complete",
        "model": model,
        "model_family": reference.family,
        "data": data,
        "device": str(loaded_model.device),
        "epochs": epochs_completed,
        "metric_name": metric_name,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "final_loss": results.get("final_loss"),
        "best": best,
        "last": last,
        "training_time_hours": round(training_hours, 2),
        "save_dir": str(save_dir),
    }
    if not json_output:
        lines = [
            f"Training complete: {epochs_completed} epochs in {training_hours:.2f}h"
        ]
        if metric_name is not None and best_metric is not None:
            lines.append(f"Best {metric_name}: {float(best_metric):.6f}")
        if best:
            lines.append(f"Best adapter checkpoint: {best}")
        else:
            lines.append(f"Artifacts saved to: {save_dir}")
        data_out["_human_text"] = "\n".join(lines)
    out.result(data_out)


def train_cmd(
    data: str = typer.Option(
        ..., help="Path to dataset YAML (YOLO format, e.g. coco8.yaml)"
    ),
    model: str = typer.Option("yolox-s", help="Model name or path to weights"),
    task: Optional[str] = typer.Option(
        None,
        help=(
            "Explicit task override: detect, segment, semantic, pose, classify, "
            "gaze, obb, point, depth"
        ),
    ),
    # Training
    epochs: int = typer.Option(300, help="Training epochs"),
    batch: int = typer.Option(16, help="Batch size per device"),
    imgsz: str = typer.Option(
        "640", help="Training image size: 640 (square) or 480x640 (HxW)"
    ),
    device: str = typer.Option("auto", help="Device: 0, cpu, mps, auto"),
    workers: int = typer.Option(4, help="Dataloader workers"),
    cache: str = typer.Option(
        "false", help="Cache images to speed dataloading: ram, disk, true, false"
    ),
    min_samples: int = typer.Option(
        0,
        help="Epoch-length floor for tiny datasets: when the dataset has "
        "fewer images, draw this many samples per epoch with replacement "
        "(0 = off)",
    ),
    class_balanced: bool = typer.Option(
        False,
        "--class-balanced/--no-class-balanced",
        help="LVIS-style repeat-factor sampling for long-tailed datasets "
        "(default: off)",
    ),
    average_best: int = typer.Option(
        0,
        help="Uniform-average the N best checkpoints by the watched metric "
        "into weights/average.pt at the end of training (0 = off)",
    ),
    export_check: bool = typer.Option(
        False,
        "--export-check/--no-export-check",
        help="Export ONNX before epoch 1 and fail the run if export breaks "
        "(default: off)",
    ),
    precise_bn: int = typer.Option(
        0,
        help="Recompute BatchNorm running stats from this many train images "
        "after the last epoch (0 = off)",
    ),
    seed: int = typer.Option(0, help="Random seed"),
    resume: str = typer.Option("", help="Resume training: true, or path to checkpoint"),
    amp: bool = typer.Option(True, help="Automatic Mixed Precision"),
    amp_dtype: str = typer.Option(
        "float16", help="CUDA AMP dtype: float16 or bfloat16"
    ),
    cuda_graph: bool = typer.Option(
        False,
        "--cuda-graph",
        help=(
            "Capture the training forward/backward into CUDA graphs "
            "(single-GPU, supported families only; others run eager)"
        ),
    ),
    pretrained: bool = typer.Option(True, help="Use pretrained weights"),
    lora: bool = typer.Option(
        False,
        "--lora",
        help="Enable LoRA fine-tuning for supported transformer families",
    ),
    freeze: str = typer.Option(
        "",
        help="Freeze layers: int count, list of indices, or module name(s)",
    ),
    # Distillation
    distill_model: str = typer.Option(
        "",
        help="Teacher for knowledge distillation: a detector checkpoint, or a "
        "foundation-teacher id (e.g. 'dinov2') for backbone feature distillation",
    ),
    dis: Optional[float] = typer.Option(
        None,
        help="Distillation loss weight (default: per-loss-type published default)",
    ),
    distill_loss_type: str = typer.Option(
        "mgd",
        help="Distillation feature loss for detector teachers: mgd, cwd "
        "(foundation teachers always use feat_mse)",
    ),
    # Optimizer
    optimizer: str = typer.Option("sgd", help="Optimizer: sgd, adam, adamw"),
    lr0: float = typer.Option(0.01, help="Initial learning rate"),
    momentum: float = typer.Option(0.937, help="SGD momentum / Adam beta1"),
    weight_decay: float = typer.Option(5e-4, help="L2 regularization"),
    nesterov: bool = typer.Option(True, help="Nesterov momentum"),
    # Scheduler
    scheduler: str = typer.Option("yoloxwarmcos", help="LR schedule type"),
    warmup_epochs: int = typer.Option(5, help="Warmup duration"),
    warmup_lr_start: float = typer.Option(0.0, help="Initial warmup LR"),
    min_lr_ratio: float = typer.Option(0.05, help="Minimum LR ratio"),
    lr_drop: int = typer.Option(100, help="RF-DETR step LR drop epoch"),
    # Augmentation
    mosaic: float = typer.Option(1.0, help="Mosaic probability"),
    mixup: float = typer.Option(1.0, help="Mixup probability"),
    hsv_prob: float = typer.Option(1.0, help="HSV jitter probability"),
    flip_prob: float = typer.Option(0.5, help="Horizontal flip probability"),
    degrees: float = typer.Option(10.0, help="Rotation +/- degrees"),
    translate: float = typer.Option(0.1, help="Translation ratio"),
    shear: float = typer.Option(2.0, help="Shear angle"),
    mosaic_scale: str = typer.Option("(0.1,2.0)", help="Mosaic scale range"),
    mixup_scale: str = typer.Option("(0.5,1.5)", help="Mixup scale range"),
    no_aug_epochs: int = typer.Option(
        15, help="Disable augmentation for final N epochs"
    ),
    # EMA
    ema: bool = typer.Option(True, help="Exponential Moving Average"),
    ema_decay: float = typer.Option(0.9998, help="EMA decay factor"),
    # Validation
    val: bool = typer.Option(True, help="Validate during training"),
    eval_interval: int = typer.Option(10, help="Validate every N epochs"),
    max_det: int = typer.Option(
        300, help="Maximum predictions per image after validation NMS"
    ),
    eval_max_det: Optional[int] = typer.Option(
        None,
        help="COCO evaluator cap (default: pycocotools AP@100)",
    ),
    faster_coco_eval: bool = typer.Option(
        True,
        "--faster-coco-eval/--no-faster-coco-eval",
        help="Use the faster-coco-eval C++ backend for validation COCO metrics "
        "(default: on when installed; falls back to pycocotools)",
    ),
    save_plots: bool = typer.Option(
        False, help="Save final validation plots during training"
    ),
    patience: int = typer.Option(50, help="Early stopping patience (0=disabled)"),
    # Output
    project: str = typer.Option("runs/train", help="Output directory root"),
    name: str = typer.Option("exp", help="Experiment name"),
    exist_ok: bool = typer.Option(False, help="Reuse existing output directory"),
    save_period: int = typer.Option(10, help="Save checkpoint every N epochs"),
    log_interval: int = typer.Option(10, help="Log loss every N batches"),
    allow_download_scripts: bool = typer.Option(
        False,
        "--allow-download-scripts",
        help="Allow embedded Python in dataset YAML download blocks",
    ),
    # Agent flags
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without executing"),
    help_json: bool = typer.Option(
        False,
        "--help-json",
        is_eager=True,
        callback=help_json_callback,
        help="Dump command schema as JSON",
    ),
) -> None:
    """Train a detection model on a dataset."""
    import ast

    out = OutputHandler(json_mode=json_output, quiet=quiet)

    user_provided = get_user_provided_params()
    normalized_task = None
    if task is not None:
        from libreyolo.tasks import normalize_task

        try:
            normalized_task = normalize_task(task)
        except ValueError as e:
            exit_with_error(out, "config_type_error", str(e))

    # Parse tuple/list strings
    try:
        from libreyolo.utils.amp import normalize_amp_dtype

        amp_dtype = normalize_amp_dtype(amp_dtype)
        if max_det < 1:
            raise ValueError(f"max_det must be >= 1, got {max_det}")
        if eval_max_det is not None and eval_max_det < 1:
            raise ValueError(f"eval_max_det must be >= 1, got {eval_max_det}")
        mosaic_scale_val = (
            ast.literal_eval(mosaic_scale)
            if isinstance(mosaic_scale, str)
            else mosaic_scale
        )
        mixup_scale_val = (
            ast.literal_eval(mixup_scale)
            if isinstance(mixup_scale, str)
            else mixup_scale
        )
        freeze_val = parse_freeze_spec(freeze)
        normalize_freeze_selectors(freeze_val)
    except (TypeError, ValueError, SyntaxError) as e:
        exit_with_error(out, "config_type_error", f"Invalid train option value: {e}")

    # Parse cache (can be "ram"/"disk" or a bool string)
    cache_val: bool | str = False
    cache_str = cache.strip().lower()
    if cache_str in ("ram", "disk"):
        cache_val = cache_str
    elif cache_str in ("true", "1", "yes"):
        cache_val = True
    elif cache_str in ("false", "0", "no", ""):
        cache_val = False
    else:
        exit_with_error(
            out,
            "config_type_error",
            f"Invalid cache value: {cache}. Use ram, disk, true, or false.",
        )

    # Parse resume (can be "true"/"false" or a path)
    resume_val: bool | str = False
    if resume:
        if resume.lower() == "true":
            resume_val = True
        elif resume.lower() == "false":
            resume_val = False
        else:
            resume_val = resume

    model_path = resolve_model_or_exit(out, model)
    from libreyolo.models.vlm import inspect_vlm_reference

    vlm_reference = inspect_vlm_reference(model_path)
    if vlm_reference is not None:
        _validate_vlm_train_request(
            out,
            vlm_reference,
            user_provided=user_provided,
            task=normalized_task,
            pretrained=pretrained,
            val=val,
        )
        _run_vlm_train(
            out,
            vlm_reference,
            model=model,
            model_path=model_path,
            data=data,
            device=device,
            params={
                "epochs": epochs,
                "batch": batch,
                "workers": workers,
                "seed": seed,
                "device": device,
                "resume": resume_val,
                "lora": lora,
                "lr0": lr0,
                "flip_prob": flip_prob,
                "project": project,
                "name": name,
                "exist_ok": exist_ok,
                "allow_download_scripts": allow_download_scripts,
            },
            user_provided=user_provided,
            dry_run=dry_run,
            json_output=json_output,
        )
        return
    family = detect_family_from_model_ref(model, model_path, inspect_checkpoint=dry_run)
    loaded_model = None
    train_pretrained = pretrained
    if family is None and not dry_run:
        loaded_model = load_model_or_exit(
            out, model=model, model_path=model_path, device=device
        )
        family = get_loaded_model_family(loaded_model)
    if train_pretrained is False and resume_val:
        from libreyolo.models.registry import group_of

        if group_of(family) in {"g0", "g1", "g2"}:
            exit_with_error(
                out,
                "config_unsupported",
                "pretrained=false cannot be combined with resume.",
            )
    if loaded_model is None:
        loaded_model = _create_explicit_task_train_model(
            family=family,
            model_path=model_path,
            task=normalized_task,
            resume=resume_val,
            device=device,
            pretrained=train_pretrained,
            seed=seed,
        )
        if loaded_model is not None and train_pretrained is False:
            # The architecture was seeded and built through _from_scratch().
            # Do not rebuild it a second time in the public train wrapper.
            train_pretrained = None
        if (
            loaded_model is not None
            and family == "yolo9"
            and train_pretrained is True
            and _should_use_yolo9_path_as_transfer(model_path, normalized_task)
        ):
            train_pretrained = model_path
    elif normalized_task is not None:
        loaded_task = getattr(loaded_model, "task", "detect")
        if loaded_task != normalized_task:
            replacement = None
            replacement = _create_yolo9_task_from_loaded_model(
                loaded_model,
                normalized_task,
                device=device,
            )
            if replacement is None and normalized_task == "obb":
                replacement = _create_rfdetr_obb_from_loaded_detect_model(
                    loaded_model,
                    model_path=model_path,
                    device=device,
                )
            if replacement is None and normalized_task == "pose":
                replacement = _create_rfdetr_pose_from_loaded_detect_model(
                    loaded_model,
                    model_path=model_path,
                    device=device,
                )
            if replacement is None and normalized_task == "segment":
                replacement = _create_dfine_segment_from_loaded_detect_model(
                    loaded_model,
                    model_path=model_path,
                    device=device,
                )
            if replacement is None:
                exit_with_error(
                    out,
                    "config_unsupported",
                    f"Loaded model task '{loaded_task}' does not match requested task "
                    f"'{normalized_task}'.",
                )
            loaded_model = replacement
            if (
                train_pretrained is True
                and get_loaded_model_family(loaded_model) == "yolo9"
            ):
                train_pretrained = model_path

    # All training params in CLI-facing names (single source of truth).
    # build_train_kwargs() maps these to TrainConfig field names automatically.
    try:
        parsed_imgsz = parse_imgsz_str(imgsz)
    except ValueError as exc:
        exit_with_error(out, "invalid_imgsz", str(exc))

    params = {
        "epochs": epochs,
        "batch": batch,
        "imgsz": parsed_imgsz,
        "device": device,
        "workers": workers,
        "cache": cache_val,
        "min_samples": min_samples,
        "class_balanced": class_balanced,
        "average_best": average_best,
        "export_check": export_check,
        "precise_bn": precise_bn,
        "seed": seed,
        "resume": resume_val,
        "amp": amp,
        "amp_dtype": amp_dtype,
        "cuda_graph": cuda_graph,
        "lora": lora,
        "freeze": freeze_val,
        "optimizer": optimizer,
        "lr0": lr0,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "nesterov": nesterov,
        "distill_model": distill_model or None,
        "dis": dis,
        "distill_loss_type": distill_loss_type,
        "scheduler": scheduler,
        "warmup_epochs": warmup_epochs,
        "warmup_lr_start": warmup_lr_start,
        "min_lr_ratio": min_lr_ratio,
        "lr_drop": lr_drop,
        "mosaic": mosaic,
        "mixup": mixup,
        "hsv_prob": hsv_prob,
        "flip_prob": flip_prob,
        "degrees": degrees,
        "translate": translate,
        "shear": shear,
        "mosaic_scale": mosaic_scale_val,
        "mixup_scale": mixup_scale_val,
        "no_aug_epochs": no_aug_epochs,
        "ema": ema,
        "ema_decay": ema_decay,
        "eval_interval": eval_interval,
        "max_det": max_det,
        "eval_max_det": eval_max_det,
        "faster_coco_eval": faster_coco_eval,
        "save_plots": save_plots,
        "patience": patience,
        "project": project,
        "name": name,
        "exist_ok": exist_ok,
        "save_period": save_period,
        "log_interval": log_interval,
        "allow_download_scripts": allow_download_scripts,
    }
    if family:
        params = apply_family_defaults(
            params, family, "train", user_provided=user_provided
        )

    if params["lora"] and family is not None and family not in _LORA_TRAIN_FAMILIES:
        exit_with_error(
            out,
            "config_unsupported",
            f"LoRA fine-tuning (lora=True) is not supported for {family}.",
            suggestion=(
                "Use a supported family (RF-DETR, D-FINE, DEIM, DEIMv2, "
                "RT-DETR v1/v2/v4, EC, ConvNeXt) or remove --lora."
            ),
        )

    # Warn when explicitly-set params are ignored by the selected family
    # (spec-driven; see libreyolo/data/augment/spec.py).
    ignored_warnings = []
    unsupported_params = get_unsupported_train_params(family)
    if unsupported_params:
        for param_name in unsupported_params:
            if param_name in user_provided:
                ignored_warnings.append(param_name)
        if ignored_warnings:
            from libreyolo.data.augment.spec import display_name

            out.progress(
                f"Warning: {display_name(family)} ignores these parameters: "
                f"{', '.join(sorted(ignored_warnings))}"
            )

    if (
        family in _CLASS_BALANCED_UNSUPPORTED_FAMILIES
        and "class_balanced" in user_provided
        and params.get("class_balanced")
    ):
        exit_with_error(
            out,
            "config_unsupported",
            f"class_balanced=True is not supported for {family}.",
            suggestion=(
                "This family builds its own dataloader. Use a family that "
                "trains through the shared detection sampler (e.g. YOLO9), "
                "or omit class_balanced."
            ),
        )

    # Dry run: validate and show resolved config
    if dry_run:
        resolved_config = {
            "model": model,
            "data": data,
            "epochs": params["epochs"],
            "batch": params["batch"],
            "imgsz": params["imgsz"],
            "optimizer": params["optimizer"],
            "lr0": params["lr0"],
            "momentum": params["momentum"],
            "scheduler": params["scheduler"],
            "amp": params["amp"],
            "amp_dtype": params["amp_dtype"],
            "max_det": params["max_det"],
            "class_balanced": params["class_balanced"],
            "average_best": params["average_best"],
            "export_check": params["export_check"],
            "precise_bn": params["precise_bn"],
        }
        if params.get("freeze") is not None:
            resolved_config["freeze"] = params["freeze"]
        if params.get("lora"):
            resolved_config["lora"] = True
        if params.get("distill_model"):
            resolved_config["distill_model"] = params["distill_model"]
            resolved_config["distill_loss_type"] = params["distill_loss_type"]
            if params.get("dis") is not None:
                resolved_config["dis"] = params["dis"]
        if normalized_task is not None:
            resolved_config["task"] = normalized_task
        if family == "rfdetr":
            resolved_config = {
                "model": model,
                "data": data,
                "epochs": params["epochs"],
                "batch": params["batch"],
                "lr0": params["lr0"],
                "workers": params["workers"],
                "weight_decay": params["weight_decay"],
                "eval_interval": params["eval_interval"],
                "warmup_epochs": params["warmup_epochs"],
                "lr_drop": params["lr_drop"],
                "ema": params["ema"],
                "ema_decay": params["ema_decay"],
                "amp": params["amp"],
                "amp_dtype": params["amp_dtype"],
                "max_det": params["max_det"],
                "save_period": params["save_period"],
                "lora": params["lora"],
                "class_balanced": params["class_balanced"],
                "average_best": params["average_best"],
                "export_check": params["export_check"],
                "precise_bn": params["precise_bn"],
            }
            if params.get("freeze") is not None:
                resolved_config["freeze"] = params["freeze"]
            if normalized_task is not None:
                resolved_config["task"] = normalized_task
        if params["eval_max_det"] is not None:
            resolved_config["eval_max_det"] = params["eval_max_det"]

        data_out = {
            "valid": True,
            "mode": "train",
            "model_family": family or "auto-detect",
            "resolved_config": resolved_config,
        }
        if not json_output:
            import yaml

            data_out["_human_text"] = (
                f"Dry run: resolved config for {model}:\n"
                + yaml.dump(data_out["resolved_config"], default_flow_style=False)
            )
        out.result(data_out)
        return

    if allow_download_scripts:
        out.warning(
            "Dataset download scripts are enabled. Embedded Python from the dataset YAML may execute locally."
        )

    # Load model
    if loaded_model is None:
        load_kwargs = {
            "out": out,
            "model": model,
            "model_path": model_path,
            "device": device,
        }
        if normalized_task is not None:
            load_kwargs["task"] = normalized_task
        loaded_model = load_model_or_exit(**load_kwargs)
    loaded_family = get_loaded_model_family(loaded_model) or family

    # Build training kwargs, with family-specific translation where needed.
    train_kwargs = build_family_train_kwargs(
        params, family, model_path=model_path, user_provided=user_provided
    )
    if train_pretrained is not None:
        train_kwargs["pretrained"] = train_pretrained  # Not in TrainConfig
    if family == "rfdetr":
        if train_pretrained is not False:
            train_kwargs.pop("pretrained", None)
        if not val and "val" in user_provided:
            out.progress(
                "Warning: RF-DETR does not support disabling validation via val=false. Ignoring."
            )
    elif not val:
        train_kwargs["eval_interval"] = 0

    # Run training
    out.progress(f"Training {model} on {data} for {params['epochs']} epochs...")
    t0 = time.time()
    try:
        results = loaded_model.train(data=data, **train_kwargs)
    except FileNotFoundError as e:
        exit_with_error(
            out,
            "data_not_found",
            str(e),
            suggestion=f"Check that '{data}' exists and is a valid YOLO-format dataset YAML.",
        )
    except Exception as e:
        exit_stage_error(out, stage="Training", detail=e)

    training_hours = (time.time() - t0) / 3600

    # Build output
    epochs_completed = params["epochs"]
    epoch_losses = results.get("epoch_losses")
    if isinstance(epoch_losses, (list, tuple)):
        epochs_completed = len(epoch_losses)
    best_mAP50 = results.get("best_mAP50", None)
    best_mAP50_95 = results.get("best_mAP50_95", None)
    best_epoch = results.get("best_epoch", None)
    save_dir = results.get("save_dir") or results.get(
        "output_dir", f"{project}/{params['name']}"
    )
    best_weights = results.get("best_checkpoint")
    last_weights = results.get("last_checkpoint")

    data_out = {
        "status": "complete",
        "model": model,
        "model_family": loaded_family,
        "data": data,
        "device": str(loaded_model.device),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_metrics": (
            {"mAP50": best_mAP50, "mAP50_95": best_mAP50_95}
            if best_mAP50 is not None
            else None
        ),
        "best_weights": best_weights,
        "last_weights": last_weights,
        "training_time_hours": round(training_hours, 2),
        "save_dir": str(save_dir),
    }

    if not json_output:
        lines = [
            f"Training complete: {epochs_completed} epochs in {training_hours:.2f}h",
        ]
        if best_mAP50 is not None:
            lines.append(
                f"Best results at epoch {best_epoch}:\n"
                f"  mAP50: {best_mAP50:.4f}  mAP50-95: {best_mAP50_95:.4f}"
            )
        if best_weights:
            lines.append(f"Weights saved to: {best_weights}")
        else:
            lines.append(f"Artifacts saved to: {save_dir}")
        data_out["_human_text"] = "\n".join(lines)

    out.result(data_out)
