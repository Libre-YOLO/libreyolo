"""Stateful SmolVLM2 Core ML conversion and runtime parity on Apple hardware."""

from __future__ import annotations

import sys

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
    pytest.skip("Core ML VLM artifacts only run on macOS", allow_module_level=True)

pytest.importorskip(
    "coremltools",
    reason="Core ML VLM parity requires the Core ML runtime",
)
pytest.importorskip(
    "transformers",
    reason="Core ML VLM parity requires the pinned processor/source model",
)
pytest.importorskip(
    "num2words",
    reason="The pinned SmolVLM2 processor requires num2words",
)


def _image_probe(phase: int) -> Image.Image:
    yy, xx = np.mgrid[:173, :251]
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
    scale = max(float(np.abs(reference).max()), 1e-6)
    return float(np.abs(actual.astype(np.float32) - reference).max()) / scale


def _assert_pair_parity_and_sensitivity(
    references: list[np.ndarray],
    actuals: list[np.ndarray],
    *,
    tolerance: float = 5e-3,
) -> None:
    errors = [
        _relative_error(reference, actual)
        for reference, actual in zip(references, actuals)
    ]
    worst = max(errors)
    assert worst <= tolerance
    sensitivity_scale = max(
        *(float(np.abs(value).max()) for value in references),
        1e-6,
    )
    relative_sensitivity = (
        float(np.abs(references[1] - references[0]).max())
        / sensitivity_scale
    )
    assert relative_sensitivity > max(100.0 * worst, 1e-5)


def test_smolvlm2_500m_coreml_public_bundle_and_state_parity(tmp_path):
    from libreyolo import LibreVLM
    from libreyolo.export.coreml_vlm import (
        COREML_VLM_CAUSAL_MASK_INPUT,
        COREML_VLM_DECODE_FUNCTION,
        COREML_VLM_LAST_LOGITS_OUTPUT,
        COREML_VLM_POSITION_IDS_INPUT,
        COREML_VLM_TOKEN_EMBEDDINGS_INPUT,
        CoreMLVLMDecodeCursor,
        preprocess_smolvlm2_500m_coreml_image,
        wrap_smolvlm2_500m_coreml_components,
    )

    source = LibreVLM(
        "smolvlm2-500m",
        device="cpu",
        names=["cat"],
    )
    bundle = source.export(
        format="coreml",
        context_length=2048,
        output_path=tmp_path / "smolvlm2-500m-2k.coremlvlm",
        compute_units="cpu_only",
    )
    deployed = LibreVLM(
        bundle,
        names=["cat"],
        compute_units="cpu_only",
    )
    runtime = deployed._coreml_runtime
    profile = runtime.profile
    components = wrap_smolvlm2_500m_coreml_components(
        source.model,
        profile=profile,
    )

    pixel_probes = []
    for phase in (17, 149):
        canonical = preprocess_smolvlm2_500m_coreml_image(
            _image_probe(phase)
        )
        batch = source.processor(
            text="<image>",
            images=canonical,
            return_tensors="pt",
        )
        pixel_values = np.asarray(
            batch["pixel_values"],
            dtype=np.float16,
        )
        assert pixel_values.shape == (
            1,
            profile.image_crops,
            profile.image_channels,
            profile.image_height,
            profile.image_width,
        )
        pixel_probes.append(np.ascontiguousarray(pixel_values))

    vision_source = components["encode_image"]
    vision_references = []
    vision_actuals = []
    with torch.inference_mode():
        for probe in pixel_probes:
            vision_references.append(
                vision_source(
                    torch.from_numpy(probe.astype(np.float32))
                )
                .detach()
                .cpu()
                .numpy()
            )
            vision_actuals.append(runtime._encode_image(probe))
    _assert_pair_parity_and_sensitivity(
        vision_references,
        vision_actuals,
    )

    id_probes = [
        np.asarray([[0, 17, 49190, 29]], dtype=np.int32),
        np.asarray([[0, 31, 49190, 47]], dtype=np.int32),
    ]
    embedding_source = components["embed_tokens"]
    embedding_references = []
    embedding_actuals = []
    with torch.inference_mode():
        for probe in id_probes:
            embedding_references.append(
                embedding_source(torch.from_numpy(probe))
                .detach()
                .cpu()
                .numpy()
            )
            embedding_actuals.append(runtime._embed(probe))
    _assert_pair_parity_and_sensitivity(
        embedding_references,
        embedding_actuals,
    )

    decoder_source = components[COREML_VLM_DECODE_FUNCTION]
    decoder_source.reset_state()
    decoder_runtime = runtime._model(COREML_VLM_DECODE_FUNCTION)
    decoder_state = decoder_runtime.make_state()
    cursor = CoreMLVLMDecodeCursor(profile)
    decoder_references = []
    decoder_actuals = []
    with torch.inference_mode():
        for embeddings in embedding_actuals:
            controls = cursor.controls(query_length=embeddings.shape[1])
            causal_mask, position_ids = controls
            decoder_references.append(
                decoder_source(
                    torch.from_numpy(embeddings.astype(np.float32)),
                    torch.from_numpy(causal_mask.astype(np.float32)),
                    torch.from_numpy(position_ids),
                )
                .detach()
                .cpu()
                .numpy()
            )
            prediction = decoder_runtime.predict(
                {
                    COREML_VLM_TOKEN_EMBEDDINGS_INPUT: embeddings,
                    COREML_VLM_CAUSAL_MASK_INPUT: causal_mask,
                    COREML_VLM_POSITION_IDS_INPUT: position_ids,
                },
                state=decoder_state,
            )
            decoder_actuals.append(
                np.asarray(
                    prediction[COREML_VLM_LAST_LOGITS_OUTPUT],
                    dtype=np.float16,
                )
            )
            cursor.commit(
                causal_mask=causal_mask,
                position_ids=position_ids,
            )
    _assert_pair_parity_and_sensitivity(
        decoder_references,
        decoder_actuals,
    )

    first = deployed.chat(
        _image_probe(41),
        "Name the dominant colors.",
        max_new_tokens=8,
    )
    second = deployed.chat(
        _image_probe(41),
        "Name the dominant colors.",
        max_new_tokens=8,
    )
    assert isinstance(first, str)
    assert second == first
    deployed.close()
