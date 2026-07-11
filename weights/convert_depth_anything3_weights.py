"""Convert the official Apache-2.0 DA3MONO-LARGE checkpoint to LibreYOLO.

The official Hugging Face artifact is a safetensors state dict whose learnable
keys are nested below ``model.``. LibreYOLO preserves the architecture's
``backbone.*`` and ``head.*`` keys, so conversion removes exactly that one
outer API prefix and wraps the tensors in checkpoint schema v1.0.

Only DA3MONO-LARGE is accepted. Metric checkpoints violate LibreYOLO's
relative inverse-depth contract; any-view and non-commercial DA3 variants are
outside this family's supported surface.

Usage::

    python weights/convert_depth_anything3_weights.py \
        model.safetensors weights/LibreDepthAnything3l-depth.pt --verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from _conversion_utils import add_repo_root_to_path, save_checkpoint

_IMGSZ = 504
_EXPECTED_EMBED_DIM = 1024


def _validate_config(input_path: str, config_path: str | None) -> None:
    path = (
        Path(config_path) if config_path else Path(input_path).with_name("config.json")
    )
    if not path.is_file():
        raise ValueError(
            "DA3 tensor layouts do not encode mono-versus-metric semantics. "
            f"Provide the official DA3MONO-LARGE config.json (expected at {path}) "
            "with --config."
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    model_config = config.get("config", {})
    backbone = model_config.get("net", {})
    head = model_config.get("head", {})
    if not (
        config.get("model_name") == "da3mono-large"
        and backbone.get("name") == "vitl"
        and backbone.get("out_layers") == [4, 11, 17, 23]
        and backbone.get("alt_start") == -1
        and backbone.get("qknorm_start") == -1
        and backbone.get("rope_start") == -1
        and backbone.get("cat_token") is False
        and head.get("dim_in") == _EXPECTED_EMBED_DIM
        and head.get("output_dim") == 1
    ):
        raise ValueError(
            f"{path} is not the supported official DA3MONO-LARGE configuration."
        )


def _load_state_dict(input_path: str) -> dict[str, torch.Tensor]:
    path = Path(input_path)
    if path.suffix.lower() != ".safetensors":
        raise ValueError(
            "Depth Anything 3 official weights must be provided as the "
            "DA3MONO-LARGE model.safetensors artifact."
        )
    return load_file(str(path), device="cpu")


def _validate_and_remap(
    state_dict: dict[str, torch.Tensor], input_path: str
) -> dict[str, torch.Tensor]:
    lower_name = str(input_path).lower()
    forbidden_hints = ("metric", "da3-small", "da3-base", "nested", "giant")
    if any(hint in lower_name for hint in forbidden_hints):
        raise ValueError(
            f"{input_path} does not look like the supported Apache-2.0 "
            "DA3MONO-LARGE checkpoint."
        )

    if not state_dict or not all(key.startswith("model.") for key in state_dict):
        raise ValueError(
            "Expected official DA3MONO-LARGE keys under the outer 'model.' prefix."
        )
    remapped = {key.removeprefix("model."): value for key, value in state_dict.items()}

    cls_token = remapped.get("backbone.pretrained.cls_token")
    if cls_token is None or tuple(cls_token.shape) != (1, 1, _EXPECTED_EMBED_DIM):
        raise ValueError(
            "Checkpoint is not DA3MONO-LARGE: expected a (1, 1, 1024) "
            "backbone class token."
        )
    if not any(key.startswith("head.scratch.sky_output_conv2.") for key in remapped):
        raise ValueError("Checkpoint is missing the DA3MONO-LARGE sky head.")
    if any(key.startswith(("cam_enc.", "cam_dec.", "gs_head.")) for key in remapped):
        raise ValueError("Camera/any-view/3DGS DA3 checkpoints are not supported.")
    return remapped


def convert_weights(
    input_path: str, output_path: str, *, config_path: str | None = None
) -> dict:
    print(f"Loading official DA3MONO-LARGE weights from {input_path}")
    _validate_config(input_path, config_path)
    state_dict = _validate_and_remap(_load_state_dict(input_path), input_path)
    print(f"Found {len(state_dict)} parameter entries")

    add_repo_root_to_path()
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    checkpoint = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="depth_anything3",
        size="l",
        task="depth",
        nc=1,
        names={0: "depth"},
        imgsz=_IMGSZ,
    )
    save_checkpoint(checkpoint, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}")
    return checkpoint


def verify_conversion(converted_path: str) -> bool:
    """Strictly load the converted artifact through the normal factory."""
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    print(f"Loading converted weights via LibreYOLO({converted_path})...")
    model = LibreYOLO(converted_path, device="cpu")
    print(
        f"  family={model.FAMILY} size={model.size} task={model.task} "
        f"nc={model.nb_classes} names={model.names}"
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert official DA3MONO-LARGE weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Official DA3MONO-LARGE model.safetensors")
    parser.add_argument("output", help="Output LibreDepthAnything3l-depth.pt")
    parser.add_argument(
        "--config",
        default=None,
        help="Official DA3MONO-LARGE config.json (default: beside input)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Strictly verify the converted checkpoint"
    )
    args = parser.parse_args()

    convert_weights(args.input, args.output, config_path=args.config)
    if args.verify:
        verify_conversion(args.output)
