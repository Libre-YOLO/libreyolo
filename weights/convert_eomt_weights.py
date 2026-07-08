"""Convert DINOv2 EoMT weights (ADE20K semantic / COCO instance / COCO panoptic) into LibreYOLO format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
)


DEFAULT_HF_REPO = "tue-mps/ade20k_semantic_eomt_large_512"
# DINOv2-based COCO instance checkpoints (no _dinov3 suffix = original DINOv2 weights, MIT).
DEFAULT_SEGMENT_HF_REPO = "tue-mps/coco_instance_eomt_large_640"
COCO_HF_REPO = DEFAULT_SEGMENT_HF_REPO
COCO_HF_REPO_1280 = "tue-mps/coco_instance_eomt_large_1280"
# DINOv2-based panoptic checkpoints — approved for --things-only instance conversion.
COCO_PANOPTIC_HF_REPO_S = "tue-mps/coco_panoptic_eomt_small_640_2x"
COCO_PANOPTIC_HF_REPO_B = "tue-mps/coco_panoptic_eomt_base_640_2x"
COCO_PANOPTIC_HF_REPO_L = "tue-mps/coco_panoptic_eomt_large_640"
_APPROVED_HF_REPOS = {
    DEFAULT_HF_REPO,
    DEFAULT_SEGMENT_HF_REPO,
    COCO_HF_REPO_1280,
    COCO_PANOPTIC_HF_REPO_S,
    COCO_PANOPTIC_HF_REPO_B,
    COCO_PANOPTIC_HF_REPO_L,
}
_NC_PANOPTIC = 133  # 80 things + 53 stuff
_NC_THINGS = 80  # COCO panoptic is contiguous: things 0-79, stuff 80-132
_IMGSZ_SEMANTIC = 512
_NC_SEMANTIC = 150
_IMGSZ_SEGMENT = 640

# Legacy aliases kept for backward compat.
_IMGSZ = _IMGSZ_SEMANTIC
_NC = _NC_SEMANTIC

_TASK_TO_IMGSZ = {
    "semantic": _IMGSZ_SEMANTIC,
    "segment": _IMGSZ_SEGMENT,
    "panoptic": _IMGSZ_SEGMENT,
}

# COCO panoptic 133-class contiguous ordering: things 0-79, stuff 80-132.
# Source: EomtConfig.id2label from tue-mps/coco_panoptic_eomt_* HF repos.
_COCO_PANOPTIC_NAMES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup",
    42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
    47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
    52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
    66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
    # stuff classes (80-132)
    80: "banner", 81: "blanket", 82: "bridge", 83: "cardboard", 84: "counter",
    85: "curtain", 86: "door-stuff", 87: "floor-wood", 88: "flower",
    89: "fruit", 90: "gravel", 91: "house", 92: "light", 93: "mirror-stuff",
    94: "net", 95: "pillow", 96: "platform", 97: "playingfield",
    98: "railroad", 99: "river", 100: "road", 101: "roof", 102: "sand",
    103: "sea", 104: "shelf", 105: "snow", 106: "stairs", 107: "tent",
    108: "towel", 109: "wall-brick", 110: "wall-stone", 111: "wall-tile",
    112: "wall-wood", 113: "water-other", 114: "window-blind",
    115: "window-other", 116: "tree-merged", 117: "fence-merged",
    118: "ceiling-merged", 119: "sky-other-merged", 120: "cabinet-merged",
    121: "table-merged", 122: "floor-other-merged", 123: "pavement-merged",
    124: "mountain-merged", 125: "grass-merged", 126: "dirt-merged",
    127: "paper-merged", 128: "food-other-merged", 129: "building-other-merged",
    130: "rock-merged", 131: "wall-other-merged", 132: "rug-merged",
}


def _default_output_path(task: str, size: str, imgsz: int) -> str:
    """Generate default output filename from task, backbone size, and image size."""
    default_imgsz = _TASK_TO_IMGSZ[task]
    task_suffix = {"semantic": "sem", "segment": "seg", "panoptic": "panoptic"}[task]
    imgsz_suffix = f"-{imgsz}" if imgsz != default_imgsz else ""
    return f"weights/LibreEoMT{size}-{task_suffix}{imgsz_suffix}.pt"


def _load_ade20k_names() -> dict[int, str]:
    root = add_repo_root_to_path()
    config_path = root / "libreyolo" / "config" / "datasets" / "ade20k.yaml"
    data = yaml.safe_load(config_path.read_text())
    names = data["names"]
    return {int(k): str(v) for k, v in names.items()}


def _load_coco_names() -> dict[int, str]:
    root = add_repo_root_to_path()
    config_path = root / "libreyolo" / "config" / "datasets" / "coco.yaml"
    data = yaml.safe_load(config_path.read_text())
    names = data["names"]
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}



def _load_hf_state_dict(repo_id: str) -> dict[str, torch.Tensor]:
    try:
        from transformers import EomtForUniversalSegmentation
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Converting from a Hugging Face repo requires transformers with "
            "EoMT support. Install with: pip install 'libreyolo[eomt]'."
        ) from exc

    model = EomtForUniversalSegmentation.from_pretrained(repo_id)
    return model.state_dict()


def _load_local_state_dict(path: Path) -> dict[str, Any]:
    if path.is_dir():
        safetensors_path = path / "model.safetensors"
        torch_path = path / "pytorch_model.bin"
        if safetensors_path.exists():
            path = safetensors_path
        elif torch_path.exists():
            path = torch_path
        else:
            raise FileNotFoundError(
                f"{path} does not contain model.safetensors or pytorch_model.bin."
            )

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except ImportError as exc:
            raise ModuleNotFoundError(
                "Loading safetensors requires safetensors>=0.4.0."
            ) from exc
        return load_safetensors_file(str(path), device="cpu")

    raw = load_checkpoint(path)
    return extract_state_dict(raw)


def _validate_source_provenance(
    source: str,
    *,
    allow_unverified_source: bool = False,
) -> None:
    normalized = source.replace("\\", "/").lower()
    if "dinov3" in normalized:
        raise ValueError(
            "DINOv3 EoMT checkpoints are excluded from LibreYOLO because they "
            "depend on gated non-commercial DINOv3 weights."
        )

    if source in _APPROVED_HF_REPOS:
        return

    path = Path(source)
    if path.exists():
        if allow_unverified_source:
            return
        raise ValueError(
            "Local EoMT sources are not provenance-verifiable by this converter. "
            f"For official LibreYOLO release weights, use {DEFAULT_HF_REPO!r}. "
            "Pass --allow-unverified-source only for private experiments after "
            "confirming the checkpoint is the DINOv2 ADE20K EoMT-L variant."
        )

    raise ValueError(
        f"Unapproved source {source!r}. Approved repos:\n"
        + "\n".join(f"  {r}" for r in sorted(_APPROVED_HF_REPOS))
        + "\nPass --allow-unverified-source for local DINOv2 checkpoints."
    )


def _load_state_dict(
    source: str,
    *,
    allow_unverified_source: bool = False,
) -> dict[str, torch.Tensor]:
    _validate_source_provenance(
        source,
        allow_unverified_source=allow_unverified_source,
    )

    path = Path(source)
    loaded = _load_local_state_dict(path) if path.exists() else _load_hf_state_dict(source)

    if not isinstance(loaded, dict):
        raise TypeError("EoMT weights must load to a state_dict dictionary.")

    add_repo_root_to_path()
    from libreyolo.models.eomt.nn import normalize_eomt_state_dict

    return normalize_eomt_state_dict(loaded)


def _slice_to_things_only(
    state_dict: dict[str, torch.Tensor],
    nc_things: int = 80,
) -> dict[str, torch.Tensor]:
    """Trim a panoptic checkpoint to keep only the things classes.

    COCO panoptic contiguous ordering: indices 0..nc_things-1 are things,
    nc_things..nc_panoptic-1 are stuff, nc_panoptic is the null/no-object class.
    Three tensors depend on nc: class_predictor.weight, class_predictor.bias,
    and criterion.empty_weight. All are sliced [0:nc_things] + [nc_panoptic].
    """
    result = dict(state_dict)
    nc_dependent_keys = (
        "class_predictor.weight",
        "class_predictor.bias",
        "criterion.empty_weight",
    )
    for key in nc_dependent_keys:
        tensor = result.get(key)
        if tensor is None:
            continue
        nc_panoptic = tensor.shape[0] - 1  # last entry is null/no-object class
        if nc_things >= nc_panoptic:
            raise ValueError(
                f"--things-only: nc_things={nc_things} >= nc_panoptic={nc_panoptic}. "
                "Does this checkpoint already have only things classes?"
            )
        null_entry = tensor[nc_panoptic : nc_panoptic + 1]
        result[key] = torch.cat([tensor[:nc_things], null_entry], dim=0)
    return result


def convert_weights(
    input_source: str,
    output_path: str,
    *,
    task: str = "semantic",
    imgsz: int | None = None,
    allow_unverified_source: bool = False,
    things_only: bool = False,
) -> dict[str, Any]:
    if task not in ("semantic", "segment", "panoptic"):
        raise ValueError(f"task must be 'semantic', 'segment', or 'panoptic'; got {task!r}.")
    if things_only and task != "segment":
        raise ValueError("--things-only is only valid with --task segment.")

    add_repo_root_to_path()
    from libreyolo.models.eomt.model import LibreEoMT
    from libreyolo.utils.serialization import (
        validate_checkpoint_metadata,
        wrap_libreyolo_checkpoint,
    )

    state_dict = _load_state_dict(
        input_source,
        allow_unverified_source=allow_unverified_source,
    )
    if not LibreEoMT.can_load(state_dict):
        raise ValueError(
            "This does not look like a DINOv2 EoMT checkpoint "
            "(expected query, mask_head, class_predictor, and embeddings keys)."
        )

    size = LibreEoMT.detect_size(state_dict)
    if size is None:
        raise ValueError(
            "Could not detect EoMT backbone size from checkpoint. "
            "Expected hidden_size of 384 (s), 768 (b), or 1024 (l)."
        )
    nc = LibreEoMT.detect_nb_classes(state_dict)

    if things_only:
        if nc != _NC_PANOPTIC:
            raise ValueError(
                f"--things-only expects a COCO panoptic checkpoint with {_NC_PANOPTIC} "
                f"classes; detected nc={nc}."
            )
        print(f"  Slicing class_predictor: {nc} panoptic classes → 80 COCO things.")
        state_dict = _slice_to_things_only(state_dict, nc_things=80)
        nc = 80

    # Use the actual position-embedding image size as the default imgsz so that
    # non-standard checkpoints (e.g. panoptic at 640px converted as semantic) are
    # embedded correctly without requiring an explicit --imgsz argument.
    detected_imgsz = LibreEoMT.detect_image_size(state_dict)

    if task == "semantic":
        if nc != _NC_SEMANTIC and not allow_unverified_source:
            raise ValueError(
                f"Semantic EoMT requires ADE20K 150-class weights; detected nc={nc}. "
                "Pass --allow-unverified-source to override."
            )
        names = _load_ade20k_names() if nc == _NC_SEMANTIC else {i: f"class_{i}" for i in range(nc)}
        effective_imgsz = imgsz if imgsz is not None else (detected_imgsz or _IMGSZ_SEMANTIC)
        ckpt_task = "semantic"
    elif task == "panoptic":
        if nc != _NC_PANOPTIC and not allow_unverified_source:
            raise ValueError(
                f"Panoptic EoMT requires COCO panoptic 133-class weights; detected nc={nc}. "
                "Pass --allow-unverified-source to override."
            )
        names = _COCO_PANOPTIC_NAMES if nc == _NC_PANOPTIC else {i: f"class_{i}" for i in range(nc)}
        effective_imgsz = imgsz if imgsz is not None else (detected_imgsz or _IMGSZ_SEGMENT)
        ckpt_task = "panoptic"
    else:
        if nc != 80 and not allow_unverified_source:
            raise ValueError(
                f"Segment EoMT requires COCO 80-class instance weights; detected nc={nc}. "
                "Pass --allow-unverified-source to override (e.g. for nc=133 panoptic-as-instance)."
            )
        names = _load_coco_names() if nc == 80 else {i: f"class_{i}" for i in range(nc)}
        effective_imgsz = imgsz if imgsz is not None else (detected_imgsz or _IMGSZ_SEGMENT)
        ckpt_task = "segment"

    # Panoptic label sets need their thing/stuff split carried as category
    # metadata: the panoptic merge fuses stuff categories into one segment each
    # and cannot derive that from the weights. COCO panoptic is contiguous with
    # things first (0-79), stuff after (80-132).
    extra_metadata: dict[str, Any] = {}
    if ckpt_task == "panoptic":
        extra_metadata["thing_class_ids"] = list(range(_NC_THINGS))

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="eomt",
        size=size,
        task=ckpt_task,
        nc=nc,
        names=names,
        imgsz=effective_imgsz,
        **extra_metadata,
    )
    errors = validate_checkpoint_metadata(checkpoint, strict=False)
    if errors:
        raise RuntimeError("; ".join(errors))

    save_checkpoint(checkpoint, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}")
    return checkpoint


def verify_conversion(converted_path: str) -> bool:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    print(f"\nLoading converted weights via LibreYOLO({converted_path})...")
    model = LibreYOLO(converted_path, device="cpu")
    print(
        f"  family={model.FAMILY} size={model.size} task={model.task} "
        f"nc={model.nb_classes}"
    )
    model.model.eval()
    imgsz = model.input_size
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, imgsz, imgsz))
    if not isinstance(out, dict):
        print(f"  forward pass OK - output shape: {tuple(out.shape)}")
        return True
    if model.task in ("segment", "panoptic"):
        class_logits = out.get("class_queries_logits")
        mask_logits = out.get("masks_queries_logits")
        print(
            f"  forward pass OK - class_queries_logits: {tuple(class_logits.shape)}, "
            f"masks_queries_logits: {tuple(mask_logits.shape)}"
        )
    else:
        logits = out.get("semantic_logits", out.get("logits"))
        print(f"  forward pass OK - semantic logits: {tuple(logits.shape)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert EoMT weights (ADE20K semantic / COCO instance / COCO panoptic) to LibreYOLO format.\n\n"
            "Approved DINOv2 sources:\n"
            f"  semantic  l-512 : {DEFAULT_HF_REPO}\n"
            f"  segment   l-640 : {DEFAULT_SEGMENT_HF_REPO}\n"
            f"  segment   l-1280: {COCO_HF_REPO_1280}\n"
            f"  segment   s-640 : {COCO_PANOPTIC_HF_REPO_S}  (--things-only)\n"
            f"  segment   b-640 : {COCO_PANOPTIC_HF_REPO_B}  (--things-only)\n"
            f"  panoptic  s-640 : {COCO_PANOPTIC_HF_REPO_S}\n"
            f"  panoptic  b-640 : {COCO_PANOPTIC_HF_REPO_B}\n"
            f"  panoptic  l-640 : {COCO_PANOPTIC_HF_REPO_L}\n\n"
            "panoptic task: keeps all 133 COCO classes (80 things + 53 stuff) and stores task=panoptic\n"
            "  with the thing/stuff split in thing_class_ids metadata.\n"
            "segment + --things-only: slices to 80 COCO things only (a lossy instance-seg derivation)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _default_input = {
        "semantic": DEFAULT_HF_REPO,
        "segment": DEFAULT_SEGMENT_HF_REPO,
        "panoptic": COCO_PANOPTIC_HF_REPO_L,
    }
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=(
            "HF repo id, local HF directory, model.safetensors, or pytorch_model.bin "
            f"(default for semantic: {DEFAULT_HF_REPO}; "
            f"default for segment: {DEFAULT_SEGMENT_HF_REPO})"
        ),
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help=(
            "Output LibreYOLO checkpoint path. "
            "Default is auto-generated from task/size/imgsz "
            "(e.g. weights/LibreEoMTl-seg.pt, weights/LibreEoMTl-seg-1280.pt)."
        ),
    )
    parser.add_argument(
        "--task",
        choices=("semantic", "segment", "panoptic"),
        default="semantic",
        help=(
            "Target task: 'semantic' (ADE20K 150-class), 'segment' (COCO 80-class instance), "
            "or 'panoptic' (COCO 133-class things+stuff, stored as task=segment). Default: semantic."
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override the embedded image size (default: 512 for semantic, 640 for segment).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify round-trip after conversion.",
    )
    parser.add_argument(
        "--things-only",
        action="store_true",
        help=(
            "Convert a COCO panoptic checkpoint (nc=133) to a COCO instance model (nc=80) "
            "by slicing the class_predictor to keep only the 80 things categories. "
            "Required to convert the s/b backbone panoptic checkpoints for instance seg."
        ),
    )
    parser.add_argument(
        "--allow-unverified-source",
        action="store_true",
        help=(
            "Allow a local/non-allowlisted source after manual DINOv2 "
            "provenance verification. DINOv3 sources remain rejected."
        ),
    )
    args = parser.parse_args()

    input_src = args.input or _default_input.get(args.task, DEFAULT_HF_REPO)
    if args.output:
        output = args.output
    else:
        # Detect size and imgsz for auto-generated filename.
        # We do a lightweight peek at the state dict size before full conversion.
        add_repo_root_to_path()
        from libreyolo.models.eomt.model import LibreEoMT as _EoMT
        _peek_state = _load_state_dict(input_src, allow_unverified_source=args.allow_unverified_source)
        _detected_size = _EoMT.detect_size(_peek_state) or "l"
        _effective_imgsz = args.imgsz if args.imgsz is not None else _TASK_TO_IMGSZ[args.task]
        output = _default_output_path(args.task, _detected_size, _effective_imgsz)
    convert_weights(
        input_src,
        output,
        task=args.task,
        imgsz=args.imgsz,
        allow_unverified_source=args.allow_unverified_source,
        things_only=args.things_only,
    )
    if args.verify:
        verify_conversion(output)
