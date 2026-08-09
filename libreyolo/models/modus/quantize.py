# SPDX-License-Identifier: MIT

"""Local-only E4M3 weight quantization for LibreMODUS."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_RECIPE = {
    "format": "e4m3fn",
    "granularity": "per-output-channel",
    "compute": "dequantize-to-input-dtype",
    "exempt": [
        "embeddings",
        "lm_head",
        "norms",
        "timestep-and-adaln",
        "first-decoder-block",
        "last-decoder-block",
        "vae",
        "vit",
        "non-decoder-projectors",
    ],
}
_SHARD_BYTES = 1024**3


class WeightOnlyFP8Linear(nn.Module):
    """Linear layer storing E4M3 weights plus one scale per output row."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        device: torch.device | str = "meta",
    ):
        super().__init__()
        if FP8_DTYPE is None:
            raise RuntimeError("This PyTorch build does not expose float8_e4m3fn.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer(
            "weight_fp8",
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=FP8_DTYPE,
                device=device,
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(self.out_features, 1, dtype=torch.float16, device=device),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, dtype=torch.bfloat16, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight_fp8.to(dtype=inputs.dtype)
        weight = weight * self.weight_scale.to(dtype=inputs.dtype)
        bias = self.bias
        if bias is not None and bias.dtype != inputs.dtype:
            bias = bias.to(inputs.dtype)
        return F.linear(inputs, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, storage=e4m3fn"
        )


def eligible_linear_names(model: nn.Module, num_decoder_layers: int) -> tuple[str, ...]:
    """Return decoder-trunk linears covered by the documented FP8 recipe."""
    names = []
    first = 0
    last = int(num_decoder_layers) - 1
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        prefix = "language_model.model.layers."
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix) :]
        layer_text, separator, _ = remainder.partition(".")
        if not separator or not layer_text.isdigit():
            continue
        layer = int(layer_text)
        if layer in {first, last} or "modulation" in name.lower():
            continue
        names.append(name)
    return tuple(names)


def replace_fp8_linears(model: nn.Module, names: Iterable[str]) -> None:
    """Replace selected meta ``nn.Linear`` modules with FP8 storage modules."""
    modules = dict(model.named_modules())
    for name in names:
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            raise KeyError(f"Cannot quantize missing Linear module {name!r}.")
        parent_name, _, child_name = name.rpartition(".")
        parent = modules[parent_name]
        replacement = WeightOnlyFP8Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="meta",
        )
        setattr(parent, child_name, replacement)


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _recipe_hash(names: Iterable[str]) -> str:
    payload = json.dumps(
        {"recipe": FP8_RECIPE, "modules": sorted(names)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if FP8_DTYPE is None:
        raise RuntimeError("This PyTorch build does not expose float8_e4m3fn.")
    work = weight.float()
    fp8_max = float(torch.finfo(FP8_DTYPE).max)
    scale = work.abs().amax(dim=1, keepdim=True).div(fp8_max)
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    quantized = work.div(scale).clamp(-fp8_max, fp8_max).to(FP8_DTYPE)
    return quantized.contiguous(), scale.to(torch.float16).contiguous()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _write_shard(
    directory: Path,
    shard_index: int,
    tensors: dict[str, torch.Tensor],
    weight_map: dict[str, str],
) -> int:
    from safetensors.torch import save_file

    filename = f"model-{shard_index:05d}.safetensors"
    save_file(tensors, str(directory / filename))
    for key in tensors:
        weight_map[key] = filename
    return sum(_tensor_bytes(tensor) for tensor in tensors.values())


def prepare_fp8_checkpoint(
    source_checkpoint: str | Path,
    quantized_modules: Iterable[str],
    *,
    cache_root: str | Path | None = None,
    source_revision: str | None = None,
) -> Path:
    """Stream-convert a BF16 safetensor into a local sharded FP8 cache."""
    source = Path(source_checkpoint).resolve()
    module_names = tuple(sorted(quantized_modules))
    source_id = source_revision or _source_hash(source)
    recipe_id = _recipe_hash(module_names)
    root = Path(cache_root or (Path.home() / ".cache/libreyolo/modus/fp8"))
    destination = root / f"{source_id}-{recipe_id}"
    index_path = destination / "model.safetensors.index.json"
    marker_path = destination / "recipe.json"
    expected_marker = {
        "source_sha_or_revision": source_id,
        "recipe": FP8_RECIPE,
        "recipe_hash": recipe_id,
        "modules": list(module_names),
    }
    if index_path.is_file() and marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            marker = None
            index = None
        shards = (
            set(index.get("weight_map", {}).values())
            if isinstance(index, dict)
            else set()
        )
        if (
            marker == expected_marker
            and shards
            and all((destination / filename).is_file() for filename in shards)
        ):
            logger.info("Using local LibreMODUS FP8 cache at %s", destination)
            return destination

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("FP8 conversion requires safetensors.") from exc

    destination.mkdir(parents=True, exist_ok=True)
    logger.warning(
        "Building a local LibreMODUS FP8 cache from %s. This streams the full "
        "checkpoint once and never uploads the derivative.",
        source,
    )
    quantized_set = set(module_names)
    converted_modules = set()
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0
    weight_map: dict[str, str] = {}
    total_bytes = 0
    shard_index = 1

    with safe_open(str(source), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            tensor = checkpoint.get_tensor(key)
            module_name = key[: -len(".weight")] if key.endswith(".weight") else None
            if module_name in quantized_set:
                converted_modules.add(module_name)
                quantized, scale = _quantize_weight(tensor)
                converted = {
                    f"{module_name}.weight_fp8": quantized,
                    f"{module_name}.weight_scale": scale,
                }
            else:
                converted = {key: tensor.contiguous()}

            converted_bytes = sum(_tensor_bytes(value) for value in converted.values())
            if pending and pending_bytes + converted_bytes > _SHARD_BYTES:
                total_bytes += _write_shard(
                    destination, shard_index, pending, weight_map
                )
                shard_index += 1
                pending = {}
                pending_bytes = 0
            pending.update(converted)
            pending_bytes += converted_bytes

    missing_modules = sorted(quantized_set - converted_modules)
    if missing_modules:
        preview = ", ".join(missing_modules[:10])
        raise RuntimeError(
            "The MODUS checkpoint is missing FP8-recipe Linear weights: "
            f"{preview}{' ...' if len(missing_modules) > 10 else ''}."
        )

    if pending:
        total_bytes += _write_shard(destination, shard_index, pending, weight_map)

    index_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": total_bytes,
                    "source": str(source),
                    "format": "pt",
                },
                "weight_map": weight_map,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    marker_path.write_text(
        json.dumps(expected_marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def theoretical_storage_bytes(model: nn.Module) -> int:
    """Return parameter+buffer storage, useful for toy recipe tests."""
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(
        math.prod(tensor.shape) * tensor.element_size()
        for tensor in tensors
        if tensor.device.type != "meta"
    )
