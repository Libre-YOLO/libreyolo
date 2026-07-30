"""Gated production-bundle parity for Qwen3-VL-2B on Apple hardware."""

from __future__ import annotations

import gc
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [
    pytest.mark.coreml,
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
    pytest.mark.general_nightly,
]

if sys.platform != "darwin":  # pragma: no cover - platform gate
    pytest.skip(
        "Qwen3-VL Core ML artifacts require macOS",
        allow_module_level=True,
    )

pytest.importorskip("coremltools")
transformers = pytest.importorskip("transformers")

from libreyolo import SAMPLE_IMAGE  # noqa: E402
from libreyolo.export.coreml_qwen3vl import (  # noqa: E402
    QWEN3VL_COREML_CONTEXT_LENGTH,
    QWEN3VL_COREML_IMAGE_TOKEN_ID,
    QWEN3VL_COREML_MAX_NEW_TOKENS,
)
from libreyolo.models.vlm import LibreVLM  # noqa: E402


def _require_campaign(tmp_path: Path) -> None:
    if os.environ.get("LIBREYOLO_RUN_QWEN3VL_COREML_E2E") != "1":
        pytest.skip(
            "set LIBREYOLO_RUN_QWEN3VL_COREML_E2E=1 for the full "
            "Qwen3-VL conversion/runtime campaign"
        )
    minimum_free = 32 * 1024**3
    available = shutil.disk_usage(tmp_path).free
    if available < minimum_free:
        pytest.skip(
            "Qwen3-VL Core ML E2E requires at least 32 GiB free for the "
            "source checkpoint, compiler temporaries, and deployment bundle"
        )


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    top_left = np.maximum(left[:2], right[:2])
    bottom_right = np.minimum(left[2:], right[2:])
    intersection = float(np.prod(np.maximum(bottom_right - top_left, 0.0)))
    left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    return intersection / max(left_area + right_area - intersection, 1e-12)


def test_qwen3vl_coreml_bundle_matches_pytorch_and_repeats(tmp_path):
    _require_campaign(tmp_path)
    native = LibreVLM(
        "qwen3-vl-2b",
        device="cpu",
        names=["person"],
        max_new_tokens=QWEN3VL_COREML_MAX_NEW_TOKENS,
    )
    snapshot = Path(native._ensure_weights())
    pixels = 448 * 448
    native.processor = transformers.AutoProcessor.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        min_pixels=pixels,
        max_pixels=pixels,
    )
    original = Image.open(SAMPLE_IMAGE).convert("RGB")
    square = original.resize((448, 448), resample=Image.Resampling.BICUBIC)
    inputs = native._prepare_generation_inputs(native._preprocess(square)[0])
    assert inputs["image_grid_thw"].tolist() == [[1, 28, 28]]
    assert tuple(inputs["pixel_values"].shape) == (784, 1536)
    assert int((inputs["input_ids"] == QWEN3VL_COREML_IMAGE_TOKEN_ID).sum()) == 196
    prompt_length = int(inputs["input_ids"].shape[1])
    assert prompt_length + QWEN3VL_COREML_MAX_NEW_TOKENS <= (
        QWEN3VL_COREML_CONTEXT_LENGTH
    )
    with torch.inference_mode():
        generated = native.model.generate(
            **inputs,
            max_new_tokens=QWEN3VL_COREML_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
        )
    native_tokens = generated[:, prompt_length:]
    native_detection = native._postprocess(
        native_tokens,
        conf_thres=0.25,
        iou_thres=0.45,
        original_size=original.size,
        max_det=300,
    )
    assert native_detection["num_detections"] > 0

    bundle = tmp_path / "LibreQwen3VL-2b-448-512.coremlvlm"
    assert native.export(
        format="coreml",
        output_path=bundle,
        context_length=QWEN3VL_COREML_CONTEXT_LENGTH,
        compute_units="cpu_only",
    ) == str(bundle)
    native_boxes = np.asarray(native_detection["boxes"], dtype=np.float32)
    native_classes = np.asarray(native_detection["classes"], dtype=np.int64)
    del generated, inputs, native_tokens, native
    gc.collect()

    deployed = LibreVLM(
        str(bundle),
        names=["person"],
        max_new_tokens=QWEN3VL_COREML_MAX_NEW_TOKENS,
        compute_units="cpu_only",
    )
    first = deployed.predict(original, conf=0.25, iou=0.45, verbose=False)
    repeated = deployed.predict(original, conf=0.25, iou=0.45, verbose=False)
    first_boxes = first.boxes.xyxy.detach().cpu().numpy()
    first_classes = first.boxes.cls.detach().cpu().numpy().astype(np.int64)
    assert first_boxes.shape == native_boxes.shape
    np.testing.assert_array_equal(first_classes, native_classes)
    assert all(
        _iou(expected, actual) >= 0.90
        for expected, actual in zip(native_boxes, first_boxes)
    )
    np.testing.assert_allclose(
        repeated.boxes.xyxy.detach().cpu().numpy(),
        first_boxes,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        repeated.boxes.cls.detach().cpu().numpy().astype(np.int64),
        first_classes,
    )
    deployed.close()
