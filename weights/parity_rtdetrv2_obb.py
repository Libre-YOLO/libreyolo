"""Compare LibreYOLO RT-DETRv2 OBB raw tensors with the pinned upstream.

The upstream checkout must be RicePasteM/RiO-DETR commit
22d5232a4e0df6ac4bc26ed1c8aac8b4060449c7.  Checkpoints must be the
DOTA-v1.0 single-scale release at Hugging Face revision
f376e9dcedfb9a47a21ac71ef61ad99f8b545698.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

from _conversion_utils import add_repo_root_to_path, extract_state_dict, load_checkpoint

add_repo_root_to_path()


CONFIG_NAMES = {
    size: f"rtdetrv2_obb_hgnetv2_{size}_dota_1_ss.yml"
    for size in ("n", "s", "m", "l", "x")
}
CHECKPOINT_NAMES = {
    size: f"rtdetrv2_obb_hgnetv2_{size}_dota_1_ss.pth"
    for size in ("n", "s", "m", "l", "x")
}


def _release(model: torch.nn.Module) -> None:
    model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cpu_features(features) -> list[torch.Tensor]:
    return [feature.detach().cpu() for feature in features]


def _print_stage_diffs(
    stage: str,
    upstream_features: list[torch.Tensor],
    our_features: list[torch.Tensor],
) -> None:
    diffs = [
        (upstream - ours).abs().max().item()
        for upstream, ours in zip(upstream_features, our_features)
    ]
    print(f"size stage={stage} max_abs_diffs={diffs}")


def _capture_decoder_outputs(decoder: torch.nn.Module):
    captures: dict[str, torch.Tensor] = {}
    handles = []
    wanted = (
        "query_pos_head",
        "decoder.layers.0.self_attn",
        "decoder.layers.0.norm1",
        "decoder.layers.0.cross_attn.sampling_offsets",
        "decoder.layers.0.cross_attn.attention_weights",
        "decoder.layers.0.cross_attn.value_proj",
        "decoder.layers.0.cross_attn.output_proj",
        "decoder.layers.0.cross_attn",
        "decoder.layers.0.norm2",
        "decoder.layers.0.linear1",
        "decoder.layers.0.linear2",
        "decoder.layers.0.norm3",
        "decoder.layers.0",
    )

    def make_hook(name):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captures[name] = tensor.detach().cpu()

        return hook

    for name, module in decoder.named_modules():
        if name in wanted:
            handles.append(module.register_forward_hook(make_hook(name)))
    return captures, handles


def _remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


def run_size(
    *,
    size: str,
    upstream_dir: Path,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[float, float]:
    from engine.core import YAMLConfig
    from libreyolo.models.rtdetr.convert import convert_to_v2
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    config_path = upstream_dir / "configs" / "rtdetrv2_obb" / CONFIG_NAMES[size]
    checkpoint_path = checkpoint_dir / CHECKPOINT_NAMES[size]
    raw = load_checkpoint(checkpoint_path)
    source_state = extract_state_dict(raw)

    config = YAMLConfig(str(config_path))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    # The upstream registry mutates constructor defaults while materializing a
    # config.  Make this omitted-for-L/X field explicit so sequential size
    # checks cannot inherit the preceding N/S/M value.
    config.yaml_cfg["HGNetv2"]["use_lab"] = size in {"n", "s", "m"}
    upstream = config.model
    upstream.load_state_dict(source_state, strict=True)
    upstream.eval().to(device)
    upstream_captures, upstream_hooks = _capture_decoder_outputs(upstream.decoder)

    generator = torch.Generator(device="cpu").manual_seed(20260808)
    image = torch.randn(1, 3, 1024, 1024, generator=generator).to(device)
    with torch.inference_mode():
        upstream_backbone = upstream.backbone(image)
        upstream_encoder = upstream.encoder(upstream_backbone)
        upstream_output = upstream.decoder(upstream_encoder)
        upstream_memory, upstream_shapes = upstream.decoder._get_encoder_input(
            upstream_encoder
        )
        upstream_decoder_input = upstream.decoder._get_decoder_input(
            upstream_memory, upstream_shapes
        )
    _remove_hooks(upstream_hooks)
    upstream_backbone_cpu = _cpu_features(upstream_backbone)
    upstream_encoder_cpu = _cpu_features(upstream_encoder)
    upstream_logits = upstream_output["pred_logits"].cpu()
    upstream_boxes = upstream_output["pred_boxes"].cpu()
    _release(upstream)

    ours = LibreRTDETRv2(
        model_path=None,
        size=size,
        task="obb",
        nb_classes=15,
        device=str(device),
    ).model
    ours.load_state_dict(convert_to_v2(source_state), strict=True)
    ours.eval().to(device)
    our_captures, our_hooks = _capture_decoder_outputs(ours.decoder)
    with torch.inference_mode():
        our_backbone = ours.backbone(image)
        our_encoder = ours.encoder(our_backbone)
        our_output = ours.decoder(our_encoder)
        our_memory, our_shapes = ours.decoder._get_encoder_input(our_encoder)
        our_decoder_input = ours.decoder._get_decoder_input(our_memory, our_shapes)
    _remove_hooks(our_hooks)
    _print_stage_diffs("backbone", upstream_backbone_cpu, _cpu_features(our_backbone))
    _print_stage_diffs("encoder", upstream_encoder_cpu, _cpu_features(our_encoder))
    _print_stage_diffs("decoder-memory", [upstream_memory.cpu()], [our_memory.cpu()])
    _print_stage_diffs(
        "decoder-input",
        [upstream_decoder_input[0].cpu(), upstream_decoder_input[1].cpu()],
        [our_decoder_input[0].cpu(), our_decoder_input[1].cpu()],
    )
    for name in sorted(upstream_captures.keys() & our_captures.keys()):
        diff = (upstream_captures[name] - our_captures[name]).abs().max().item()
        print(f"size stage={name} max_abs_diff={diff:.9g}")
    our_logits = our_output["pred_logits"].cpu()
    our_boxes = our_output["pred_boxes"].cpu()
    _release(ours)

    logits_diff = (upstream_logits - our_logits).abs().max().item()
    boxes_diff = (upstream_boxes - our_boxes).abs().max().item()
    print(
        f"size={size} logits_shape={tuple(our_logits.shape)} "
        f"boxes_shape={tuple(our_boxes.shape)} "
        f"logits_max_abs_diff={logits_diff:.9g} "
        f"boxes_max_abs_diff={boxes_diff:.9g}"
    )
    if logits_diff != 0.0 or boxes_diff != 0.0:
        raise AssertionError(
            f"raw parity failed for size={size}: "
            f"logits={logits_diff}, boxes={boxes_diff}"
        )
    return logits_diff, boxes_diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=list(CONFIG_NAMES),
        choices=list(CONFIG_NAMES),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    upstream_dir = args.upstream_dir.resolve()
    commit = (
        __import__("subprocess")
        .check_output(["git", "-C", str(upstream_dir), "rev-parse", "HEAD"], text=True)
        .strip()
    )
    expected_commit = "22d5232a4e0df6ac4bc26ed1c8aac8b4060449c7"
    if commit != expected_commit:
        raise RuntimeError(f"upstream checkout is {commit}, expected {expected_commit}")
    sys.path.insert(0, str(upstream_dir))

    device = torch.device(args.device)
    for size in args.sizes:
        run_size(
            size=size,
            upstream_dir=upstream_dir,
            checkpoint_dir=args.checkpoint_dir.resolve(),
            device=device,
        )


if __name__ == "__main__":
    main()
