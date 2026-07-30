"""Queued Florence-2-base conversion and state parity on Apple hardware."""

from __future__ import annotations

import os
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
        "Florence Core ML artifacts only convert and run on macOS",
        allow_module_level=True,
    )

pytest.importorskip(
    "coremltools",
    reason="Florence Core ML parity requires Core ML Tools and runtime",
)
transformers = pytest.importorskip(
    "transformers",
    reason="Florence parity requires the pinned processor/source model",
)


def _image_probe(phase: int) -> Image.Image:
    yy, xx = np.mgrid[:193, :257]
    rgb = np.stack(
        (
            (3 * xx + 5 * yy + phase) % 256,
            (11 * xx + 7 * yy + 2 * phase) % 256,
            (13 * xx + 19 * yy + 3 * phase) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _relative_error(reference: np.ndarray, actual: np.ndarray) -> float:
    reference32 = reference.astype(np.float32)
    actual32 = actual.astype(np.float32)
    scale = max(float(np.abs(reference32).max()), 1e-6)
    return float(np.abs(actual32 - reference32).max()) / scale


def _assert_pair_parity_and_sensitivity(
    references: list[np.ndarray],
    actuals: list[np.ndarray],
    *,
    label: str,
    tolerance: float = 8e-3,
) -> None:
    errors = [
        _relative_error(reference, actual)
        for reference, actual in zip(references, actuals)
    ]
    worst = max(errors)
    scale = max(
        *(float(np.abs(value.astype(np.float32)).max()) for value in references),
        1e-6,
    )
    sensitivity = (
        float(
            np.abs(
                references[1].astype(np.float32) - references[0].astype(np.float32)
            ).max()
        )
        / scale
    )
    print(
        f"{label}: relative_errors={errors}, worst={worst:.8g}, "
        f"relative_sensitivity={sensitivity:.8g}"
    )
    assert worst <= tolerance
    assert sensitivity > max(100.0 * worst, 1e-6)


def _local_checkpoint() -> Path:
    if os.environ.get("LIBREYOLO_RUN_FLORENCE_COREML_E2E") != "1":
        pytest.skip(
            "set LIBREYOLO_RUN_FLORENCE_COREML_E2E=1 for the queued "
            "full Florence conversion/runtime test"
        )
    raw = os.environ.get("LIBREYOLO_FLORENCE2_BASE_DIR")
    if not raw:
        pytest.skip("set LIBREYOLO_FLORENCE2_BASE_DIR to the pinned offline snapshot")
    path = Path(raw).expanduser()
    if not path.is_dir():
        pytest.fail(f"LIBREYOLO_FLORENCE2_BASE_DIR is not a directory: {path}")
    return path.resolve()


def test_florence2_base_coreml_encode_state_decode_and_public_repeat(
    tmp_path,
):
    from libreyolo import LibreVLM
    from libreyolo.backends.coreml_florence import (
        CoreMLFlorenceRuntime,
        _FlorenceDecodeRequest,
        build_coreml_florence_bundle,
    )
    from libreyolo.export.coreml_florence import (
        FLORENCE2_BASE_REVISION,
        FLORENCE2_DECODER_START_TOKEN_ID,
        FLORENCE2_FORCED_BOS_TOKEN_ID,
        FLORENCE_CROSS_ATTENTION_MASK_INPUT,
        FLORENCE_DECODE_FUNCTION,
        FLORENCE_ENCODER_ATTENTION_MASK_INPUT,
        FLORENCE_ENCODER_INPUT_IDS_INPUT,
        FLORENCE_PIXEL_VALUES_INPUT,
        FlorenceDecodeCursor,
        export_florence2_base_coreml_package,
        florence2_base_coreml_profile,
        prepare_florence2_base_processor_batch,
        validate_florence2_base_processor_assets,
        validate_florence2_base_weight_asset,
        wrap_florence2_base_coreml_components,
    )

    checkpoint = _local_checkpoint()
    compute_units = os.environ.get(
        "LIBREYOLO_FLORENCE_COREML_COMPUTE_UNITS",
        "cpu_only",
    ).strip().lower()
    print(f"florence runtime compute_units={compute_units}")
    assert transformers.__version__ == "5.12.1"
    validate_florence2_base_processor_assets(
        checkpoint,
        revision=FLORENCE2_BASE_REVISION,
    )
    validate_florence2_base_weight_asset(
        checkpoint,
        revision=FLORENCE2_BASE_REVISION,
    )
    processor = transformers.AutoProcessor.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = transformers.Florence2ForConditionalGeneration.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()

    existing_bundle = os.environ.get("LIBREYOLO_FLORENCE_COREML_BUNDLE")
    if existing_bundle:
        bundle = str(Path(existing_bundle).expanduser().resolve())
    else:
        package = tmp_path / "florence2-base.mlpackage"
        export_florence2_base_coreml_package(
            model,
            checkpoint_dir=checkpoint,
            processor_revision=FLORENCE2_BASE_REVISION,
            output_path=package,
            compute_units=compute_units,
        )
        bundle = build_coreml_florence_bundle(
            package,
            processor_dir=checkpoint,
            output_path=tmp_path / "florence2-base.coremlvlm",
            move_model=True,
        )

    runtime = CoreMLFlorenceRuntime(
        bundle,
        names=["cat"],
        compute_units=compute_units,
    )
    profile = florence2_base_coreml_profile()
    components = wrap_florence2_base_coreml_components(
        model,
        profile=profile,
    )
    try:
        prepared_probes = [
            prepare_florence2_base_processor_batch(
                processor,
                _image_probe(phase),
                ["cat"],
                profile=profile,
            )
            for phase in (17, 149)
        ]
        source_keys: list[np.ndarray] = []
        source_values: list[np.ndarray] = []
        runtime_keys: list[np.ndarray] = []
        runtime_values: list[np.ndarray] = []
        with torch.inference_mode():
            for prepared in prepared_probes:
                source_key, source_value = components["encode"](
                    torch.from_numpy(
                        prepared[FLORENCE_PIXEL_VALUES_INPUT].astype(np.float32)
                    ),
                    torch.from_numpy(prepared[FLORENCE_ENCODER_INPUT_IDS_INPUT]),
                    torch.from_numpy(
                        prepared[FLORENCE_ENCODER_ATTENTION_MASK_INPUT].astype(
                            np.float32
                        )
                    ),
                )
                actual_key, actual_value = runtime._encode(prepared)
                source_keys.append(source_key.detach().cpu().numpy())
                source_values.append(source_value.detach().cpu().numpy())
                runtime_keys.append(actual_key)
                runtime_values.append(actual_value)
        _assert_pair_parity_and_sensitivity(
            source_keys,
            runtime_keys,
            label="florence encoder cross keys",
        )
        _assert_pair_parity_and_sensitivity(
            source_values,
            runtime_values,
            label="florence encoder cross values",
        )

        source_decoder = components[FLORENCE_DECODE_FUNCTION]
        source_decoder.reset_state()
        source_decoder.initialize_cross_cache(
            torch.from_numpy(runtime_keys[0].astype(np.float32)),
            torch.from_numpy(runtime_values[0].astype(np.float32)),
        )
        request = _FlorenceDecodeRequest(
            runtime._model(FLORENCE_DECODE_FUNCTION),
            profile,
            cross_key_values=runtime_keys[0],
            cross_value_values=runtime_values[0],
        )
        source_cursor = FlorenceDecodeCursor(profile)
        cross_mask = prepared_probes[0][FLORENCE_CROSS_ATTENTION_MASK_INPUT]
        decoder_references: list[np.ndarray] = []
        decoder_actuals: list[np.ndarray] = []
        try:
            for tokens, parents in (
                (
                    np.full(
                        (3, 1),
                        FLORENCE2_DECODER_START_TOKEN_ID,
                        dtype=np.int32,
                    ),
                    np.asarray([0, 1, 2], dtype=np.int32),
                ),
                (
                    np.full(
                        (3, 1),
                        FLORENCE2_FORCED_BOS_TOKEN_ID,
                        dtype=np.int32,
                    ),
                    np.asarray([0, 0, 2], dtype=np.int32),
                ),
            ):
                causal_mask, position_ids = source_cursor.controls()
                with torch.inference_mode():
                    expected = source_decoder(
                        torch.from_numpy(tokens),
                        torch.from_numpy(causal_mask.astype(np.float32)),
                        torch.from_numpy(cross_mask.astype(np.float32)),
                        torch.from_numpy(position_ids),
                        torch.from_numpy(parents),
                    )
                actual = request.predict(
                    tokens,
                    cross_attention_mask=cross_mask,
                    beam_parent_indices=parents,
                )
                decoder_references.append(expected.detach().cpu().numpy())
                decoder_actuals.append(actual)
                source_cursor.commit(
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                )
        finally:
            request.discard()
            source_decoder.reset_state()
        errors = [
            _relative_error(reference, actual)
            for reference, actual in zip(
                decoder_references,
                decoder_actuals,
            )
        ]
        decoder_scale = max(
            float(np.abs(decoder_references[0]).max()),
            float(np.abs(decoder_references[1]).max()),
            1e-6,
        )
        decoder_sensitivity = (
            float(np.abs(decoder_references[1] - decoder_references[0]).max())
            / decoder_scale
        )
        print(
            "florence stateful decoder: "
            f"relative_errors={errors}, worst={max(errors):.8g}, "
            f"relative_sensitivity={decoder_sensitivity:.8g}"
        )
        assert max(errors) <= 8e-3
        assert decoder_sensitivity > max(100.0 * max(errors), 1e-6)

        probe = _image_probe(41)
        first = runtime.predict(probe, max_new_tokens=32)
        second = runtime.predict(probe, max_new_tokens=32)
        assert first == second
        assert set(first) == {
            "boxes",
            "scores",
            "classes",
            "num_detections",
        }
    finally:
        runtime.close()
    assert runtime.closed

    deployed = LibreVLM(
        bundle,
        names=["cat"],
        compute_units=compute_units,
        max_new_tokens=32,
    )
    try:
        public_first = deployed.predict(probe, verbose=False)[0]
        public_second = deployed.predict(probe, verbose=False)[0]
        for name in ("xyxy", "conf", "cls"):
            first_tensor = getattr(public_first.boxes, name)
            second_tensor = getattr(public_second.boxes, name)
            assert torch.equal(first_tensor, second_tensor)
            assert torch.isfinite(first_tensor).all()

        expected_boxes = np.asarray(first["boxes"], dtype=np.float32).reshape(-1, 4)
        expected_scores = np.asarray(first["scores"], dtype=np.float32)
        expected_classes = np.asarray(first["classes"], dtype=np.float32)
        np.testing.assert_array_equal(
            public_first.boxes.xyxy.cpu().numpy(),
            expected_boxes,
        )
        np.testing.assert_array_equal(
            public_first.boxes.conf.cpu().numpy(),
            expected_scores,
        )
        np.testing.assert_array_equal(
            public_first.boxes.cls.cpu().numpy(),
            expected_classes,
        )
        assert deployed._coreml_runtime._active_decode is None
    finally:
        deployed.close()
    assert deployed._coreml_runtime.closed
