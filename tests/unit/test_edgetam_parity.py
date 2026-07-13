"""Gated EdgeTAM parity against the pinned Apache-2.0 implementation."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.sam.edgetam import LibreEdgeTAM
from weights.convert_edgetam_weights import (
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_SOURCE_COMMIT,
    load_official_state_dict,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sam,
    pytest.mark.external_data,
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOLERANCE = 1e-4
# Digests are of the upstream bytes as published (LF). A checkout made with
# git's autocrlf translation active will not match; clone with
# core.autocrlf=false to run this gate.
_UPSTREAM_TREE_SHA256 = (
    "a5ca617d3899c586ea55e82aeaac321ce296ef0bea0877f43a153417196db7ba"
)
_UPSTREAM_FILE_SHA256 = {
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "sam2/configs/edgetam.yaml": (
        "25c2fda8490e7684f924abba130775487ca6eccee87b1d9cf92ddccf2436afe1"
    ),
    "sam2/build_sam.py": (
        "36406726bf6320445927900e1ad1e6336f337eebb0ba724662fad7dbf2e17cb5"
    ),
    "sam2/sam2_image_predictor.py": (
        "f13e5f9d94e5c8d9d2c3622dab20c8f334c089ef2ee5ea8e199da7d332b029ba"
    ),
    "sam2/utils/transforms.py": (
        "ba3a64f4600c62f209206a6df3b40e3fcf133edae32fad658831bb0c2a6d1146"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(upstream: Path) -> str:
    """Hash every executable/config source used from the pinned checkout."""
    digest = hashlib.sha256()
    sources = sorted(
        path
        for path in (upstream / "sam2").rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
    )
    for source in sources:
        digest.update(source.relative_to(upstream).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(source)))
    return digest.hexdigest()


def _require_local_artifacts() -> tuple[Path, Path, Path]:
    upstream_value = os.getenv("LIBREYOLO_EDGETAM_UPSTREAM")
    checkpoint_value = os.getenv("LIBREYOLO_EDGETAM_CHECKPOINT")
    if not upstream_value or not checkpoint_value:
        pytest.skip(
            "set LIBREYOLO_EDGETAM_UPSTREAM and "
            "LIBREYOLO_EDGETAM_CHECKPOINT to run EdgeTAM parity"
        )

    upstream = Path(upstream_value).expanduser().resolve()
    checkpoint = Path(checkpoint_value).expanduser().resolve()
    snapshot_value = os.getenv("LIBREYOLO_EDGETAM_SNAPSHOT")
    snapshot = (
        Path(snapshot_value).expanduser().resolve()
        if snapshot_value
        else (_REPO_ROOT / "_hf_build_LibreEdgeTAM").resolve()
    )

    if not upstream.is_dir():
        pytest.fail(f"configured EdgeTAM upstream checkout is missing: {upstream}")
    for relative, expected in _UPSTREAM_FILE_SHA256.items():
        source = upstream / relative
        if not source.is_file():
            pytest.fail(
                f"EdgeTAM checkout is not pinned at {OFFICIAL_SOURCE_COMMIT}: "
                f"missing {relative}"
            )
        actual = _sha256(source)
        if actual != expected:
            pytest.fail(
                f"EdgeTAM checkout is not pinned at {OFFICIAL_SOURCE_COMMIT}: "
                f"{relative} has SHA-256 {actual}"
            )
    tree_sha256 = _source_tree_sha256(upstream)
    if tree_sha256 != _UPSTREAM_TREE_SHA256:
        pytest.fail(
            f"EdgeTAM sam2 source tree is not pinned at {OFFICIAL_SOURCE_COMMIT}: "
            f"expected {_UPSTREAM_TREE_SHA256}, got {tree_sha256}"
        )

    if not checkpoint.is_file():
        pytest.fail(f"configured official EdgeTAM checkpoint is missing: {checkpoint}")
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != OFFICIAL_CHECKPOINT_SHA256:
        pytest.fail(
            "LIBREYOLO_EDGETAM_CHECKPOINT is not the official checkpoint: "
            f"expected {OFFICIAL_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )

    required_snapshot_files = (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "processor_config.json",
    )
    if not snapshot.is_dir() or any(
        not (snapshot / filename).is_file() for filename in required_snapshot_files
    ):
        if snapshot_value:
            pytest.fail(
                f"configured LIBREYOLO_EDGETAM_SNAPSHOT is incomplete: {snapshot}"
            )
        pytest.skip(
            "set LIBREYOLO_EDGETAM_SNAPSHOT to a local converted snapshot, "
            "or create _hf_build_LibreEdgeTAM"
        )
    return upstream, checkpoint, snapshot


def _iter_hydra_targets(value):
    if isinstance(value, dict):
        target = value.get("_target_")
        if target is not None:
            yield str(target)
        for child in value.values():
            yield from _iter_hydra_targets(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_hydra_targets(child)


def _build_upstream(upstream_root: Path, state_dict):
    hydra = pytest.importorskip("hydra", reason="EdgeTAM parity requires hydra-core")
    pytest.importorskip("omegaconf", reason="EdgeTAM parity requires omegaconf")
    timm_models = pytest.importorskip(
        "timm.models", reason="EdgeTAM parity requires timm"
    )
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if GlobalHydra.instance().is_initialized():
        pytest.fail("Hydra is already initialized; run EdgeTAM parity in isolation")

    existing_sam2 = sys.modules.get("sam2")
    if existing_sam2 is not None:
        module_file = getattr(existing_sam2, "__file__", None)
        if module_file is None or not Path(module_file).resolve().is_relative_to(
            upstream_root
        ):
            pytest.fail("another sam2 package is already imported")

    original_create_model = timm_models.create_model

    def create_model_without_pretrained_download(name, *args, **kwargs):
        # Every TIMM parameter is overwritten by the strict-loaded official
        # checkpoint, so constructing random weights is both offline and exact.
        kwargs["pretrained"] = False
        return original_create_model(name, *args, **kwargs)

    sys.path.insert(0, str(upstream_root))
    try:
        config_dir = upstream_root / "sam2" / "configs"
        with hydra.initialize_config_dir(
            config_dir=str(config_dir), version_base="1.2"
        ):
            config = hydra.compose(
                config_name="edgetam",
                overrides=[
                    "++model.sam_mask_decoder_extra_args."
                    "dynamic_multimask_via_stability=true",
                    "++model.sam_mask_decoder_extra_args."
                    "dynamic_multimask_stability_delta=0.05",
                    "++model.sam_mask_decoder_extra_args."
                    "dynamic_multimask_stability_thresh=0.98",
                ],
            )
            OmegaConf.resolve(config)
            model_config = OmegaConf.to_container(config.model, resolve=True)
            targets = list(_iter_hydra_targets(model_config))
            assert targets and all(target.startswith("sam2.") for target in targets)
            with patch.object(
                timm_models,
                "create_model",
                create_model_without_pretrained_download,
            ):
                model = instantiate(config.model, _recursive_=True)

        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.utils.transforms import SAM2Transforms

        for name, module in tuple(sys.modules.items()):
            if name != "sam2" and not name.startswith("sam2."):
                continue
            module_file = getattr(module, "__file__", None)
            if module_file is not None:
                assert Path(module_file).resolve().is_relative_to(upstream_root)
    finally:
        try:
            sys.path.remove(str(upstream_root))
        except ValueError:
            pass

    incompatible = model.load_state_dict(state_dict, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    return model.eval(), SAM2ImagePredictor, SAM2Transforms


def _load_transformers_snapshot(snapshot: Path):
    transformers = pytest.importorskip(
        "transformers", reason="EdgeTAM parity requires the sam extra"
    )
    edge_model = getattr(transformers, "EdgeTamModel", None)
    processor_cls = getattr(transformers, "Sam2Processor", None)
    if edge_model is None or processor_cls is None:
        pytest.skip("installed transformers does not provide EdgeTAM image inference")

    # Exercise LibreYOLO's config-loading path as part of the gated parity test.
    # It consumes only this local snapshot and avoids Transformers' otherwise
    # redundant lookup for the already-embedded RepViT configuration.
    adapter = object.__new__(LibreEdgeTAM)
    model, processor = adapter._load_pretrained(str(snapshot))
    model.eval()
    return model, processor


def _bare_libre(model, processor) -> LibreEdgeTAM:
    libre = object.__new__(LibreEdgeTAM)
    libre.model = model
    libre.processor = processor
    libre.device = torch.device("cpu")
    libre._model_dtype = torch.float32
    libre._default_multimask = False
    libre.names = {0: "object"}
    libre.size = "edge"
    libre.task = "segment"
    libre._clear_image_state()
    return libre


def _structured_image() -> Image.Image:
    height, width = 96, 128
    yy, xx = np.mgrid[:height, :width]
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = (xx * 2) % 256
    pixels[..., 1] = (yy * 3) % 256
    pixels[..., 2] = ((xx + yy) * 2) % 256
    pixels[20:76, 30:100] = [220, 70, 40]
    return Image.fromarray(pixels, mode="RGB")


def _awkward_image() -> Image.Image:
    pixels = (np.arange(31 * 37 * 3, dtype=np.uint32).reshape(31, 37, 3) % 256).astype(
        np.uint8
    )
    return Image.fromarray(pixels, mode="RGB")


def _assert_max_abs(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    tolerance: float = _TOLERANCE,
) -> None:
    assert actual.shape == expected.shape, (
        f"{label} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    difference = float((actual.float() - expected.float()).abs().max().item())
    assert difference <= tolerance, (
        f"{label} max_abs_diff={difference:.3e} exceeds {tolerance:.1e}"
    )


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    assert actual.shape == expected.shape
    if not torch.equal(actual, expected):
        difference = float((actual.float() - expected.float()).abs().max().item())
        pytest.fail(f"{label} is not exact; max_abs_diff={difference:.3e}")


def _upstream_image_embeddings(upstream, pixel_values, feature_sizes):
    backbone_output = upstream.forward_image(pixel_values)
    _, vision_features, _, _ = upstream._prepare_backbone_features(backbone_output)
    vision_features[-1] = vision_features[-1] + upstream.no_mem_embed
    return [
        feature.permute(1, 2, 0).view(1, -1, *feature_size)
        for feature, feature_size in zip(vision_features[::-1], feature_sizes[::-1])
    ][::-1]


def _upstream_prompt(upstream, encoding):
    if "input_points" in encoding:
        coords = encoding["input_points"].squeeze(1)
        labels = encoding["input_labels"].squeeze(1)
    else:
        coords = encoding["input_boxes"].squeeze(1).reshape(-1, 2, 2)
        labels = torch.tensor([[2, 3]], dtype=torch.int64).repeat(coords.shape[0], 1)
    return upstream.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)


def _compare_prompt_and_decoder(
    label,
    upstream,
    transformers_model,
    upstream_embeddings,
    transformers_embeddings,
    encoding,
):
    upstream_sparse, upstream_dense = _upstream_prompt(upstream, encoding)
    hf_sparse, hf_dense = transformers_model.get_prompt_embeddings(
        input_points=encoding.get("input_points"),
        input_labels=encoding.get("input_labels"),
        input_boxes=encoding.get("input_boxes"),
    )
    assert hf_sparse.shape[1] == 1
    _assert_max_abs(hf_sparse.squeeze(1), upstream_sparse, f"{label} sparse")
    _assert_max_abs(hf_dense, upstream_dense, f"{label} dense")

    upstream_position = upstream.sam_prompt_encoder.get_dense_pe()
    hf_position = transformers_model.get_image_wide_positional_embeddings()
    _assert_max_abs(hf_position, upstream_position, "prompt position")

    upstream_masks, upstream_iou, _, _ = upstream.sam_mask_decoder(
        image_embeddings=upstream_embeddings[-1],
        image_pe=upstream_position,
        sparse_prompt_embeddings=upstream_sparse,
        dense_prompt_embeddings=upstream_dense,
        multimask_output=True,
        repeat_image=False,
        high_res_features=upstream_embeddings[:-1],
    )
    hf_masks, hf_iou, _, _ = transformers_model.mask_decoder(
        image_embeddings=transformers_embeddings[-1],
        image_positional_embeddings=hf_position,
        sparse_prompt_embeddings=hf_sparse,
        dense_prompt_embeddings=hf_dense,
        multimask_output=True,
        high_resolution_features=transformers_embeddings[:-1],
    )
    _assert_max_abs(hf_masks.squeeze(1), upstream_masks, f"{label} mask logits")
    _assert_max_abs(hf_iou.squeeze(1), upstream_iou, f"{label} IoU scores")


def _binary_iou(first, second) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 1.0


@pytest.mark.skipif(
    not os.getenv("LIBREYOLO_EDGETAM_UPSTREAM")
    or not os.getenv("LIBREYOLO_EDGETAM_CHECKPOINT"),
    reason=(
        "set LIBREYOLO_EDGETAM_UPSTREAM and "
        "LIBREYOLO_EDGETAM_CHECKPOINT to run EdgeTAM parity"
    ),
)
def test_edgetam_upstream_parity(monkeypatch):
    upstream_root, checkpoint, snapshot = _require_local_artifacts()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    state_dict = load_official_state_dict(checkpoint)
    upstream, predictor_cls, transform_cls = _build_upstream(upstream_root, state_dict)
    transformers_model, processor = _load_transformers_snapshot(snapshot)
    libre = _bare_libre(transformers_model, processor)
    upstream_transform = transform_cls(resolution=1024, mask_threshold=0.0)
    awkward_image = _awkward_image()
    awkward_point = [11.3, 17.7]
    awkward_box = [2.4, 3.8, 31.2, 27.5]
    awkward_encoding = libre._encode(
        awkward_image,
        points=[[awkward_point]],
        labels=[[1]],
        boxes=[awkward_box],
    )
    upstream_pixels = upstream_transform(awkward_image).unsqueeze(0)
    _assert_exact(
        awkward_encoding["pixel_values"],
        upstream_pixels,
        "LibreEdgeTAM pixel preprocessing",
    )
    assert awkward_encoding["original_sizes"].tolist() == [[31, 37]]

    upstream_points = upstream_transform.transform_coords(
        torch.tensor([[awkward_point]], dtype=torch.float32),
        normalize=True,
        orig_hw=(31, 37),
    )
    upstream_boxes = upstream_transform.transform_boxes(
        torch.tensor([awkward_box], dtype=torch.float32),
        normalize=True,
        orig_hw=(31, 37),
    )
    _assert_exact(
        awkward_encoding["input_points"].squeeze(1),
        upstream_points,
        "point preprocessing",
    )
    _assert_exact(
        awkward_encoding["input_boxes"].squeeze(1).reshape(-1, 2, 2),
        upstream_boxes,
        "box preprocessing",
    )

    image = _structured_image()
    height, width = image.height, image.width
    point = [64.0, 48.0]
    box = [30.0, 20.0, 100.0, 76.0]
    combined_encoding = libre._encode(
        image,
        points=[[point]],
        labels=[[1]],
        boxes=[box],
    )

    with torch.inference_mode():
        transformers_embeddings = transformers_model.get_image_embeddings(
            combined_encoding["pixel_values"]
        )
        feature_sizes = [
            tuple(size) for size in transformers_model.backbone_feature_sizes
        ]
        upstream_embeddings = _upstream_image_embeddings(
            upstream, combined_encoding["pixel_values"], feature_sizes
        )
        assert len(transformers_embeddings) == len(upstream_embeddings) == 3
        for index, (hf_feature, upstream_feature) in enumerate(
            zip(transformers_embeddings, upstream_embeddings)
        ):
            _assert_max_abs(hf_feature, upstream_feature, f"image encoder FPN[{index}]")

        point_encoding = libre._encode(image, points=[[point]], labels=[[1]])
        box_encoding = libre._encode(image, boxes=[box])
        _compare_prompt_and_decoder(
            "point",
            upstream,
            transformers_model,
            upstream_embeddings,
            transformers_embeddings,
            point_encoding,
        )
        _compare_prompt_and_decoder(
            "box",
            upstream,
            transformers_model,
            upstream_embeddings,
            transformers_embeddings,
            box_encoding,
        )

    predictor = predictor_cls(upstream)
    predictor._orig_hw = [(height, width)]
    predictor._features = {
        "image_embed": upstream_embeddings[-1],
        "high_res_feats": upstream_embeddings[:-1],
    }
    predictor._is_image_set = True
    libre._image = image
    libre._image_path = None
    libre._image_embeddings = transformers_embeddings

    cases: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        (
            "point",
            {
                "point_coords": np.array([point], dtype=np.float32),
                "point_labels": np.array([1], dtype=np.int32),
            },
            {"points": point, "labels": [1]},
        ),
        (
            "box",
            {"box": np.array(box, dtype=np.float32)},
            {"bboxes": box},
        ),
        (
            "positive-negative points",
            {
                "point_coords": np.array([point, [10.0, 10.0]], dtype=np.float32),
                "point_labels": np.array([1, 0], dtype=np.int32),
            },
            {
                "points": [[point, [10.0, 10.0]]],
                "labels": [[1, 0]],
            },
        ),
    ]
    for label, upstream_kwargs, libre_kwargs in cases:
        upstream_masks, _, _ = predictor.predict(
            multimask_output=False, **upstream_kwargs
        )
        result = libre.predict(**libre_kwargs)
        assert result.masks is not None and len(result.masks) == 1
        upstream_mask = np.asarray(upstream_masks[0], dtype=bool)
        libre_mask = result.masks.data[0].cpu().numpy().astype(bool)
        iou = _binary_iou(upstream_mask, libre_mask)
        assert iou >= 0.99, f"{label} final mask IoU={iou:.6f}"
