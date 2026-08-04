"""Convert FeyNobg background-removal weights into LibreYOLO format.

LibreFeyNobg keeps the original BiRefNet key schema; the upstream ``nobg``
library stores the same architecture under Hugging-Face-style names (split
``q_proj``/``k_proj``/``v_proj`` attention, ``layernorm_before/after``,
indexed decoder ModuleLists, per-stage ``hidden_states_norms``). Conversion is
therefore a deterministic key remap - fused qkv, renamed modules, list indices
mapped to the port's numbered attributes by tensor shape - with learned
parameters unchanged, wrapped in the LibreYOLO v1.0 checkpoint schema. The
trailing ``bb.swin.layernorm`` is dropped (unused by the backbone forward;
parity is exact without it). This script does not download or redistribute
upstream weights.

Usage::

    python weights/convert_feynobg_weights.py model.safetensors weights/LibreFeyNobgl-matte.pt --verify

FeyNobg code and weights are Apache-2.0 (https://huggingface.co/feyninc/FeyNobg,
https://github.com/feyninc/nobg), Copyright (c) 2026 Feyn Inc. See
weights/LICENSE_NOTICE.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _conversion_utils import add_repo_root_to_path, load_checkpoint, save_checkpoint

_FEYNOBG_MARKER = "bb.layers.2.blocks.23.norm1.weight"
_NOBG_MARKER = "bb.swin.encoder.layers.2.blocks.23.layernorm_before.weight"
_IMGSZ = 1024


def _remap_nobg_state_dict(sd: dict, model_sd: dict) -> dict:
    """Remap nobg (HF-style) tensor names onto the original BiRefNet schema.

    ``model_sd`` supplies the target key set: shapes disambiguate the decoder
    ModuleList ordering, and deterministic buffers missing from the source
    (``relative_position_index``) are filled from the freshly built model.
    """
    import re

    out: dict = {}
    qkv: dict = {}
    dropped = []

    # Indexed decoder/squeeze lists -> numbered attributes. Upstream builds
    # deep-to-shallow (4, 3, 2[, 1]); shape verification below confirms it.
    list_maps = {
        "decoder.decoder_blocks": lambda n: f"decoder.decoder_block{4 - int(n)}",
        "decoder.lateral_blocks": lambda n: f"decoder.lateral_block{4 - int(n)}",
        "decoder.gdt_convs": lambda n: f"decoder.gdt_convs_{4 - int(n)}",
        "decoder.gdt_convs_attn": lambda n: f"decoder.gdt_convs_attn_{4 - int(n)}",
        "decoder.gdt_convs_pred": lambda n: f"decoder.gdt_convs_pred_{4 - int(n)}",
        "decoder.conv_ms_spvn": lambda n: f"decoder.conv_ms_spvn_{4 - int(n)}",
        "decoder.ipt_blks": lambda n: f"decoder.ipt_blk{5 - int(n)}",
    }

    for k, v in sd.items():
        if k.startswith("bb.swin.encoder.layers."):
            m = re.match(r"bb\.swin\.encoder\.layers\.(\d+)\.(.*)", k)
            i, rest = m.group(1), m.group(2)
            bm = re.match(r"blocks\.(\d+)\.(.*)", rest)
            if bm:
                j, tail = bm.group(1), bm.group(2)
                qm = re.match(r"attention\.([qkv])_proj\.(weight|bias)", tail)
                if qm:
                    qkv[(i, j, qm.group(2), qm.group(1))] = v
                    continue
                tail = (
                    tail.replace("layernorm_before", "norm1")
                    .replace("layernorm_after", "norm2")
                    .replace("attention.o_proj", "attn.proj")
                    .replace(
                        "attention.relative_position_bias.relative_position_bias_table",
                        "attn.relative_position_bias_table",
                    )
                )
                out[f"bb.layers.{i}.blocks.{j}.{tail}"] = v
            else:
                out[f"bb.layers.{i}.{rest}"] = v  # downsample.*
        elif k.startswith("bb.hidden_states_norms.stage"):
            m = re.match(r"bb\.hidden_states_norms\.stage(\d)\.(.*)", k)
            out[f"bb.norm{int(m.group(1)) - 1}.{m.group(2)}"] = v
        elif k.startswith("bb.swin.embeddings.patch_embeddings.projection."):
            out[k.replace("bb.swin.embeddings.patch_embeddings.projection.", "bb.patch_embed.proj.")] = v
        elif k.startswith("bb.swin.embeddings.norm."):
            out[k.replace("bb.swin.embeddings.norm.", "bb.patch_embed.norm.")] = v
        elif k.startswith("bb.swin.layernorm."):
            dropped.append(k)  # unused by the backbone forward
        else:
            mapped = False
            for prefix, rename in list_maps.items():
                m = re.match(re.escape(prefix) + r"\.(\d+)\.(.*)", k)
                if m:
                    tgt = rename(m.group(1))
                    tail = m.group(2)
                    # conv_ms_spvn entries are bare convs on the target side.
                    out[f"{tgt}.{tail}"] = v
                    mapped = True
                    break
            if not mapped:
                out[k] = v  # squeeze_module.*, decoder.conv_out1.* are 1:1

    # Fuse q/k/v into the original schema's single qkv projection.
    blocks = sorted({(i, j) for (i, j, _, _) in qkv})
    for i, j in blocks:
        for kind in ("weight", "bias"):
            out[f"bb.layers.{i}.blocks.{j}.attn.qkv.{kind}"] = torch.cat(
                [qkv[(i, j, kind, p)] for p in ("q", "k", "v")], dim=0
            )

    # Deterministic buffers absent from the source checkpoint.
    for k in model_sd:
        if k.endswith("relative_position_index") and k not in out:
            out[k] = model_sd[k]

    # Hard verification: exact key-set match and exact shape match.
    missing = set(model_sd) - set(out)
    unexpected = set(out) - set(model_sd)
    if missing or unexpected:
        raise RuntimeError(
            f"nobg remap mismatch: missing={sorted(missing)[:8]}, "
            f"unexpected={sorted(unexpected)[:8]}"
        )
    bad_shapes = [
        k for k in out if tuple(out[k].shape) != tuple(model_sd[k].shape)
    ]
    if bad_shapes:
        raise RuntimeError(f"nobg remap shape mismatches: {bad_shapes[:8]}")
    print(f"Remapped {len(sd)} nobg tensors -> {len(out)} (dropped unused: {dropped})")
    return out


def _load_state_dict(input_path: str) -> dict:
    if str(input_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return dict(load_file(input_path))
    raw = load_checkpoint(input_path)
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "params", "net"):
            value = raw.get(key)
            if isinstance(value, dict):
                return dict(value)
        return dict(raw)
    if hasattr(raw, "state_dict"):
        return dict(raw.state_dict())
    raise TypeError(f"Unsupported checkpoint object: {type(raw)!r}")


def convert_weights(input_path: str, output_path: str, *, imgsz: int = _IMGSZ) -> dict:
    print(f"Loading FeyNobg weights from {input_path}")
    state_dict = _load_state_dict(input_path)
    print(f"Found {len(state_dict)} parameter entries")

    if _FEYNOBG_MARKER not in state_dict and _NOBG_MARKER not in state_dict:
        raise ValueError(
            "This does not look like a FeyNobg checkpoint (no 24-block stage-3 "
            "marker in either the BiRefNet or nobg key schema). BiRefNet "
            "checkpoints convert with weights/convert_birefnet_weights.py."
        )

    add_repo_root_to_path()
    from libreyolo.models.feynobg import LibreFeyNobg
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    model = LibreFeyNobg(model_path=None, size="l", device="cpu")
    if _NOBG_MARKER in state_dict:
        state_dict = _remap_nobg_state_dict(state_dict, model.model.state_dict())
    result = model.model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "FeyNobg state dict did not load strictly: "
            f"missing={result.missing_keys[:8]}, unexpected={result.unexpected_keys[:8]}"
        )

    checkpoint = wrap_libreyolo_checkpoint(
        model.model.state_dict(),
        model_family="feynobg",
        size="l",
        task="matte",
        nc=1,
        names={0: "matte"},
        supported_tasks=("matte",),
        default_task="matte",
        imgsz=imgsz,
    )
    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(checkpoint, tmp)
    tmp.rename(out)  # atomic
    print(f"Saved LibreYOLO-format checkpoint to {out}")
    return checkpoint


def verify_conversion(converted_path: str) -> bool:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    validate_checkpoint_metadata(converted_path)
    print(f"\nLoading converted weights via LibreYOLO({converted_path})...")
    model = LibreYOLO(converted_path, device="cpu")
    print(f"  family={model.FAMILY} size={model.size} task={model.task} nc={model.nb_classes} names={model.names}")
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, _IMGSZ, _IMGSZ))
    print(f"  forward pass OK - output shape: {tuple(out.shape)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FeyNobg weights to LibreYOLO format")
    parser.add_argument("input", help="Upstream FeyNobg checkpoint (.safetensors/.pth/.pt)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument("--imgsz", type=int, default=_IMGSZ, help="Native input size recorded in metadata")
    parser.add_argument("--verify", action="store_true", help="Verify round-trip + metadata after conversion")
    args = parser.parse_args()

    convert_weights(args.input, args.output, imgsz=args.imgsz)
    if args.verify:
        verify_conversion(args.output)
