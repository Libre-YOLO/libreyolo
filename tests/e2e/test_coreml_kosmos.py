"""Real-M4 component and public-path parity for Kosmos-2 Core ML."""

from __future__ import annotations

import gc
import os
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
]

if sys.platform != "darwin":  # pragma: no cover - platform gate
    pytest.skip("Kosmos-2 Core ML artifacts require macOS", allow_module_level=True)

ct = pytest.importorskip("coremltools")

from libreyolo import SAMPLE_IMAGE  # noqa: E402
from libreyolo.export.coreml_kosmos import (  # noqa: E402
    KOSMOS2_COREML_COMPONENTS,
    KOSMOS2_COREML_CONTEXT_LENGTH,
    Kosmos2CoreMLDecoder,
    Kosmos2CoreMLVision,
)
from libreyolo.models.vlm import LibreVLM  # noqa: E402


def _relative_metrics(references, actuals):
    errors = []
    for expected, actual in zip(references, actuals):
        scale = max(float(np.abs(expected).max()), 1e-8)
        errors.append(float(np.abs(expected - actual).max()) / scale)
    reference_scale = max(
        *(float(np.abs(value).max()) for value in references),
        1e-8,
    )
    sensitivity = (
        float(np.abs(references[0] - references[1]).max()) / reference_scale
    )
    error = max(errors)
    assert error <= 3e-4
    assert sensitivity >= 1e-6
    assert sensitivity / max(error, 1e-12) >= 100.0
    return error, sensitivity


def _component_model(bundle, component):
    model = ct.models.MLModel(
        str(bundle / KOSMOS2_COREML_COMPONENTS[component]),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    assert model.user_defined_metadata["component"] == component
    return model


def _prepared_prefix(source, batch, image_embeddings):
    sequence = batch["input_ids"].detach().cpu().numpy().astype(np.int32)[0]
    image_mask = (
        batch["image_embeds_position_mask"].detach().cpu().numpy().astype(bool)[0]
    )
    padding = KOSMOS2_COREML_CONTEXT_LENGTH - sequence.size
    assert padding >= 0
    input_ids = np.full(
        (1, KOSMOS2_COREML_CONTEXT_LENGTH),
        1,
        dtype=np.int32,
    )
    input_ids[0, padding:] = sequence
    attention_mask = np.zeros(
        (1, KOSMOS2_COREML_CONTEXT_LENGTH),
        dtype=np.float32,
    )
    attention_mask[0, padding:] = 1.0
    position_ids = np.full(
        (1, KOSMOS2_COREML_CONTEXT_LENGTH),
        1,
        dtype=np.int32,
    )
    position_ids[0, padding:] = np.arange(2, sequence.size + 2, dtype=np.int32)
    with torch.no_grad():
        token_embeddings = source.get_input_embeddings()(
            torch.from_numpy(input_ids).to(dtype=torch.long)
        )
    token_embeddings = token_embeddings.detach().cpu().numpy().astype(np.float32)
    token_embeddings[0, padding + np.flatnonzero(image_mask)] = image_embeddings[0]
    return input_ids, attention_mask, position_ids, token_embeddings


@pytest.mark.skipif(
    os.environ.get("LIBREYOLO_RUN_KOSMOS2_COREML_E2E") != "1",
    reason="Kosmos-2 Core ML E2E is an explicit 7GB hardware campaign",
)
def test_kosmos2_coreml_components_and_public_prediction(tmp_path):
    source_wrapper = LibreVLM(
        "kosmos-2",
        device="cpu",
        names=["men"],
        max_new_tokens=24,
    )
    source = source_wrapper.model.eval()
    height, width = 173, 257
    grid = np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3)
    images = (
        Image.fromarray((grid % 251).astype(np.uint8), mode="RGB"),
        Image.fromarray(((grid * 7 + 31) % 253).astype(np.uint8), mode="RGB"),
    )
    batches = [
        source_wrapper._prepare_generation_inputs(
            source_wrapper._preprocess(image)[0]
        )
        for image in images
    ]
    vision_wrapper = Kosmos2CoreMLVision(source).eval()
    pixel_values = [batch["pixel_values"].contiguous() for batch in batches]
    with torch.no_grad():
        vision_references = [
            vision_wrapper(value).detach().cpu().numpy() for value in pixel_values
        ]
    prefixes = [
        _prepared_prefix(source, batch, image_embeddings)
        for batch, image_embeddings in zip(batches, vision_references)
    ]
    decoder_wrapper = Kosmos2CoreMLDecoder(source).eval()
    with torch.no_grad():
        decoder_references = [
            decoder_wrapper(
                torch.from_numpy(prefix[3]),
                torch.from_numpy(prefix[1]),
                torch.from_numpy(prefix[2]),
            )
            .detach()
            .cpu()
            .numpy()
            for prefix in prefixes
        ]

    bundle = tmp_path / "LibreKosmos2-224-128.coremlvlm"
    exported = source_wrapper.export(
        format="coreml",
        output_path=bundle,
        context_length=128,
        compute_units="cpu_only",
    )
    assert exported == str(bundle)

    vision_model = _component_model(bundle, "vision")
    vision_actual = [
        np.asarray(
            vision_model.predict(
                {
                    "pixel_values": value.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                }
            )["image_embeddings"],
            dtype=np.float32,
        )
        for value in pixel_values
    ]
    _relative_metrics(vision_references, vision_actual)
    del vision_model
    gc.collect()

    embedding_model = _component_model(bundle, "token_embedding")
    for prefix in prefixes:
        actual = np.asarray(
            embedding_model.predict({"input_ids": prefix[0]})["token_embeddings"],
            dtype=np.float32,
        )
        expected = (
            source.get_input_embeddings()(
                torch.from_numpy(prefix[0]).to(dtype=torch.long)
            )
            .detach()
            .cpu()
            .numpy()
        )
        scale = max(float(np.abs(expected).max()), 1e-8)
        assert float(np.abs(expected - actual).max()) / scale <= 3e-4
    del embedding_model
    gc.collect()

    decoder_model = _component_model(bundle, "decoder")
    decoder_actual = [
        np.asarray(
            decoder_model.predict(
                {
                    "input_embeddings": prefix[3],
                    "attention_mask": prefix[1],
                    "position_ids": prefix[2],
                }
            )["last_logits"],
            dtype=np.float32,
        )
        for prefix in prefixes
    ]
    _relative_metrics(decoder_references, decoder_actual)
    del decoder_model
    gc.collect()

    native = source_wrapper.predict(SAMPLE_IMAGE, verbose=False)
    assert native.boxes is not None and len(native.boxes) > 0
    native_boxes = native.boxes.xyxy.detach().cpu().numpy()
    native_scores = native.boxes.conf.detach().cpu().numpy()
    native_classes = native.boxes.cls.detach().cpu().numpy()
    del decoder_wrapper, vision_wrapper, source, source_wrapper
    gc.collect()

    deployed_wrapper = LibreVLM(
        str(bundle),
        names=["men"],
        max_new_tokens=24,
        compute_units="cpu_only",
    )
    deployed = deployed_wrapper.predict(SAMPLE_IMAGE, verbose=False)
    repeated = deployed_wrapper.predict(SAMPLE_IMAGE, verbose=False)
    for result in (deployed, repeated):
        assert result.boxes is not None and len(result.boxes) > 0
    np.testing.assert_allclose(
        deployed.boxes.xyxy.detach().cpu().numpy(),
        native_boxes,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        deployed.boxes.conf.detach().cpu().numpy(),
        native_scores,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        deployed.boxes.cls.detach().cpu().numpy(),
        native_classes,
    )
    np.testing.assert_allclose(
        repeated.boxes.xyxy.detach().cpu().numpy(),
        deployed.boxes.xyxy.detach().cpu().numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        repeated.boxes.conf.detach().cpu().numpy(),
        deployed.boxes.conf.detach().cpu().numpy(),
        rtol=0.0,
        atol=0.0,
    )
    deployed_wrapper.close()
