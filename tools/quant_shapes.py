"""Extract canonical quantization shapes from the flagship live models.

The generated ``shapes.json`` is the shared input for Triton parity and
benchmark gates.  Module selection intentionally routes through the same
policy used by ``model.quantize`` so excluded heads and first layers do not
silently enter the benchmark corpus.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from libreyolo import LibreYOLO
from libreyolo.quant.api import _select_modules, default_keep_high_precision


RFDETR_MODELS = ("LibreRFDETRn.pt", "LibreRFDETRs.pt")
RFDETR_IMAGE_SIZES = (384, 512, 576)
YOLO9_MODELS = ("LibreYOLO9s.pt",)
YOLO9_IMAGE_SIZES = (640,)
BATCH_SIZES = (1, 8)


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _tensor_arg(args: tuple[Any, ...]) -> torch.Tensor:
    for arg in args:
        if torch.is_tensor(arg):
            return arg
    raise RuntimeError("Quantizable module received no tensor positional input")


def _resolve_module(root: nn.Module, name: str) -> nn.Module:
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def _complete_modulelist_observations(
    root: nn.Module,
    selected: dict[str, nn.Module],
    observations: dict[tuple[Any, ...], dict[str, Any]],
) -> None:
    """Fill inactive cloned branches from an observed peer's live shape.

    RF-DETR's training-only group branches are cloned ``ModuleList`` entries.
    Evaluation executes group zero, while training executes every clone with
    that exact input tensor.  Recording the peer relationship preserves every
    quantizable module without inventing a synthetic token count.
    """
    direct: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in observations.values():
        direct[record["module"]].append(record)

    for name, module in selected.items():
        if direct[name]:
            continue
        parent_name, separator, _index = name.rpartition(".")
        if not separator:
            raise RuntimeError(f"No live activation shape observed for {name}")
        parent = _resolve_module(root, parent_name)
        if not isinstance(parent, nn.ModuleList):
            raise RuntimeError(
                f"No live activation shape observed for non-ModuleList module {name}"
            )

        peers = []
        for peer_index, peer in enumerate(parent):
            peer_name = f"{parent_name}.{peer_index}"
            if (
                peer_name in selected
                and direct[peer_name]
                and tuple(peer.weight.shape) == tuple(module.weight.shape)
            ):
                peers.append(peer_name)
        if not peers:
            raise RuntimeError(
                f"No shape-compatible observed ModuleList peer found for {name}"
            )

        source_name = peers[0]
        for source in direct[source_name]:
            copied = {
                **source,
                "module": name,
                "inferred_from": source_name,
            }
            key = (
                name,
                copied["imgsz"],
                copied["batch"],
                tuple(copied["shape"]),
            )
            observations[key] = copied


def _canonical(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, ...], dict[str, Any]] = {}
    for record in records:
        shape = tuple(record["shape"])
        entry = grouped.setdefault(
            shape,
            {
                "shape": list(shape),
                "sources": [],
                "calls": 0,
            },
        )
        source = f"{record['model']}:{record['module']}"
        if source not in entry["sources"]:
            entry["sources"].append(source)
        entry["calls"] += int(record.get("calls", 1))
    return sorted(grouped.values(), key=lambda item: tuple(item["shape"]))


def _extract_model(
    checkpoint: str,
    *,
    recipe: str,
    image_sizes: tuple[int, ...],
    batches: tuple[int, ...],
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    wrapper = LibreYOLO(checkpoint, device=str(device))
    model = wrapper.model.eval()
    selected = _select_modules(
        model,
        recipe,
        default_keep_high_precision(wrapper.FAMILY),
    )

    weights: list[dict[str, Any]] = []
    for name, module in selected.items():
        kind = "conv" if type(module) is nn.Conv2d else "linear"
        record: dict[str, Any] = {
            "model": checkpoint,
            "module": name,
            "shape": list(module.weight.shape),
        }
        if kind == "linear":
            record.update(
                in_features=module.in_features,
                out_features=module.out_features,
            )
        else:
            record.update(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                groups=module.groups,
                kernel_size=list(module.kernel_size),
            )
        weights.append({"kind": kind, **record})

    activation_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    context: dict[str, int] = {}
    handles = []

    def make_hook(name: str, kind: str):
        def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
            tensor = _tensor_arg(args)
            shape = tuple(tensor.shape)
            key = (name, context["imgsz"], context["batch"], shape)
            record = activation_map.setdefault(
                key,
                {
                    "kind": kind,
                    "model": checkpoint,
                    "module": name,
                    "imgsz": context["imgsz"],
                    "batch": context["batch"],
                    "shape": list(shape),
                    "calls": 0,
                },
            )
            record["calls"] += 1

        return hook

    for name, module in selected.items():
        kind = "conv" if type(module) is nn.Conv2d else "linear"
        handles.append(module.register_forward_pre_hook(make_hook(name, kind)))

    try:
        with torch.inference_mode():
            for image_size in image_sizes:
                for batch in batches:
                    context.update(imgsz=image_size, batch=batch)
                    inputs = torch.zeros(
                        batch,
                        3,
                        image_size,
                        image_size,
                        device=device,
                    )
                    model(inputs)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    del inputs
            _complete_modulelist_observations(model, selected, activation_map)
    finally:
        for handle in handles:
            handle.remove()
        del model, wrapper
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "weights": weights,
        "activations": list(activation_map.values()),
    }


def extract_shapes(device: str = "cuda") -> dict[str, Any]:
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    all_weights: list[dict[str, Any]] = []
    all_activations: list[dict[str, Any]] = []
    model_runs: list[dict[str, Any]] = []

    suites = (
        (RFDETR_MODELS, "nvfp4", RFDETR_IMAGE_SIZES),
        (YOLO9_MODELS, "int8", YOLO9_IMAGE_SIZES),
    )
    for checkpoints, recipe, image_sizes in suites:
        for checkpoint in checkpoints:
            extracted = _extract_model(
                checkpoint,
                recipe=recipe,
                image_sizes=image_sizes,
                batches=BATCH_SIZES,
                device=target,
            )
            all_weights.extend(extracted["weights"])
            all_activations.extend(extracted["activations"])
            model_runs.append(
                {
                    "checkpoint": checkpoint,
                    "selection_recipe": recipe,
                    "image_sizes": list(image_sizes),
                    "batches": list(BATCH_SIZES),
                }
            )

    by_kind_weights: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_kind_activations: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_weights:
        by_kind_weights[record["kind"]].append(record)
    for record in all_activations:
        by_kind_activations[record["kind"]].append(record)

    return {
        "schema_version": 1,
        "source_commit": _git_commit(),
        "device": str(target),
        "models": model_runs,
        "modules": {
            "weights": all_weights,
            "activations": all_activations,
        },
        "canonical": {
            "linear_weights": _canonical(by_kind_weights["linear"]),
            "linear_activations": _canonical(by_kind_activations["linear"]),
            "conv_weights": _canonical(by_kind_weights["conv"]),
            "conv_activations": _canonical(by_kind_activations["conv"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("shapes.json"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    shapes = extract_shapes(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(shapes, indent=2) + "\n", encoding="utf-8")
    counts = {name: len(items) for name, items in shapes["canonical"].items()}
    print(f"wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
