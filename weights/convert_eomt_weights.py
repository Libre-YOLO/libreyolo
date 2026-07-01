"""Convert DINOv2 EoMT-L ADE20K semantic weights into LibreYOLO format."""

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
_APPROVED_HF_REPOS = {DEFAULT_HF_REPO}
_IMGSZ = 512
_NC = 150


def _load_ade20k_names() -> dict[int, str]:
    root = add_repo_root_to_path()
    config_path = root / "libreyolo" / "config" / "datasets" / "ade20k.yaml"
    data = yaml.safe_load(config_path.read_text())
    names = data["names"]
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
        "LibreEoMT release conversion supports only the approved DINOv2 ADE20K "
        f"source {DEFAULT_HF_REPO!r}. Refusing unapproved source {source!r}."
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


def convert_weights(
    input_source: str,
    output_path: str,
    *,
    allow_unverified_source: bool = False,
) -> dict[str, Any]:
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
    if size != "l":
        raise ValueError(
            f"LibreEoMT v1 ships only size 'l'; detected size={size!r}."
        )
    nc = LibreEoMT.detect_nb_classes(state_dict)
    if nc != _NC:
        raise ValueError(
            f"LibreEoMT v1 ships only ADE20K 150-class weights; detected nc={nc}."
        )

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="eomt",
        size="l",
        task="semantic",
        nc=_NC,
        names=_load_ade20k_names(),
        imgsz=_IMGSZ,
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
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, _IMGSZ, _IMGSZ))
    logits = out["semantic_logits"] if isinstance(out, dict) else out
    print(f"  forward pass OK - semantic logits: {tuple(logits.shape)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert EoMT-L ADE20K semantic weights to LibreYOLO format."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_HF_REPO,
        help=(
            "HF repo id, local HF directory, model.safetensors, or pytorch_model.bin "
            f"(default: {DEFAULT_HF_REPO})"
        ),
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="weights/LibreEoMTl-sem.pt",
        help="Output LibreYOLO checkpoint path.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify round-trip after conversion.",
    )
    parser.add_argument(
        "--allow-unverified-source",
        action="store_true",
        help=(
            "Allow a local/non-allowlisted source after manual DINOv2 ADE20K "
            "provenance verification. DINOv3 sources remain rejected."
        ),
    )
    args = parser.parse_args()

    convert_weights(
        args.input,
        args.output,
        allow_unverified_source=args.allow_unverified_source,
    )
    if args.verify:
        verify_conversion(args.output)
