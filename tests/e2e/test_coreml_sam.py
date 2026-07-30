"""Core ML graph and public-path parity for split promptable SAM packages.

This suite is intentionally macOS-only. Linux conversion tests can validate
MIL structure, but only Core ML execution can prove the saved multifunction
package. Each family is converted once, then exercised through every named
function and through the public encode-once/prompt-many API.
"""

from __future__ import annotations

import gc
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
    pytest.mark.sam,
]

if sys.platform != "darwin":  # pragma: no cover - platform gate
    pytest.skip("Core ML artifacts only run on macOS", allow_module_level=True)

ct = pytest.importorskip(
    "coremltools",
    reason="Core ML SAM parity requires the coremltools runtime",
)

REL_TOL = 3e-4
MIN_REL_SENSITIVITY = 1e-6
SENSITIVITY_TO_ERROR_MARGIN = 100.0
PROMPT_MAX_POINTS = 4

SAM_CASES = [
    ("edgetam", "edge", "edgetam"),
    ("mobilesam", "tiny", "mobilesam"),
    ("sam", "base", "base"),
    ("sam2", "tiny", "sam2-tiny"),
    ("sam2", "small", "sam2-small"),
    ("sam2", "base-plus", "sam2-base-plus"),
    ("sam2", "large", "sam2-large"),
    ("sam", "large", "large"),
    ("sam", "huge", "huge"),
    ("sam3", "large", "sam3"),
]


def _byte_probe(width: int, height: int, *, phase: int) -> Image.Image:
    yy, xx = np.mgrid[:height, :width]
    values = np.stack(
        (
            (3 * xx + 5 * yy + 17 + phase) % 256,
            (11 * xx + 7 * yy + 53 + 3 * phase) % 256,
            (13 * xx + 19 * yy + 101 + 5 * phase) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(values, mode="RGB")


def _as_output_tuple(value) -> tuple[torch.Tensor, ...]:
    if torch.is_tensor(value):
        return (value,)
    if isinstance(value, (tuple, list)) and all(
        torch.is_tensor(item) for item in value
    ):
        return tuple(value)
    raise AssertionError(f"Unexpected SAM component output {type(value).__name__}")


def _assert_close(
    expected: torch.Tensor | np.ndarray,
    actual: np.ndarray,
    *,
    label: str,
) -> float:
    reference = (
        expected.detach().cpu().numpy()
        if torch.is_tensor(expected)
        else np.asarray(expected)
    )
    candidate = np.asarray(actual)
    assert candidate.dtype == np.float32, (
        f"{label} returned {candidate.dtype}, expected float32"
    )
    assert candidate.shape == reference.shape
    assert np.isfinite(candidate).all()
    scale = max(float(np.abs(reference).max()), 1e-12)
    error = float(np.abs(candidate - reference).max()) / scale
    assert error <= REL_TOL, (
        f"{label} relative conversion error {error:.3e} exceeds {REL_TOL:.0e}"
    )
    return error


def _runtime_outputs(runtime, inputs, output_names):
    arrays = {
        name: np.ascontiguousarray(
            value.detach().cpu().numpy()
            if torch.is_tensor(value)
            else np.asarray(value)
        )
        for name, value in inputs.items()
    }
    outputs = runtime.predict(arrays)
    assert set(outputs) == set(output_names)
    return tuple(
        np.ascontiguousarray(outputs[name]).copy() for name in output_names
    )


def _owned_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    """Detach an oracle value from the heavyweight eager model lifetime."""
    array = (
        value.detach().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value)
    )
    return np.ascontiguousarray(array).copy()


def _point_prompts(
    image: Image.Image,
    *,
    count: int,
) -> tuple[list[list[float]], list[int]]:
    width, height = image.size
    points = [
        [
            width * (index + 1) / (count + 1),
            height * (((index * 3) % count) + 1) / (count + 1),
        ]
        for index in range(count)
    ]
    labels = [1 if index % 2 == 0 else 0 for index in range(count)]
    return points, labels


def _component_inputs(
    function_name: str,
    *,
    image: Image.Image,
    encoding,
    embeddings,
    profile,
    point_count: int,
):
    from libreyolo.backends.coreml_sam import (
        transform_sam_coreml_box,
        transform_sam_coreml_points,
    )
    from libreyolo.export.coreml_sam import (
        SAM_COREML_BOXES_INPUT,
        SAM_COREML_POINT_COORDS_INPUT,
        SAM_COREML_POINT_LABELS_INPUT,
        sam_coreml_function_contracts,
    )

    values = dict(embeddings)
    if "points" in function_name:
        points, labels = _point_prompts(image, count=point_count)
        values[SAM_COREML_POINT_COORDS_INPUT] = transform_sam_coreml_points(
            points,
            encoding=encoding,
            profile=profile,
        )
        values[SAM_COREML_POINT_LABELS_INPUT] = torch.tensor(
            labels,
            dtype=torch.int32,
        ).reshape(1, 1, -1)
    if "boxes" in function_name:
        width, height = image.size
        values[SAM_COREML_BOXES_INPUT] = transform_sam_coreml_box(
            [
                width * 0.15,
                height * 0.20,
                width * 0.85,
                height * 0.80,
            ],
            encoding=encoding,
            profile=profile,
        )
    expected_names = [
        item["name"]
        for item in sam_coreml_function_contracts(profile)[function_name]["inputs"]
    ]
    assert list(values) == expected_names
    return values


@pytest.fixture(scope="module", params=SAM_CASES)
def converted_sam_case(request, tmp_path_factory):
    family, size, alias = request.param
    if family == "sam3" and os.environ.get(
        "LIBREYOLO_RUN_GATED_SAM3_COREML",
        "",
    ).lower() not in {"1", "true", "yes"}:
        pytest.skip(
            "SAM3 weights are gated/custom-license. Set "
            "LIBREYOLO_RUN_GATED_SAM3_COREML=1 after accepting the terms and "
            "staging credentials to run this local-only gate."
        )

    from libreyolo import LibreSAM
    from libreyolo.export.coreml_sam import validate_sam_coreml_profile
    from libreyolo.export.support import get_support

    model = LibreSAM(alias, device="cpu")
    requested_compute_units = (
        "validated"
        if get_support(family, "segment", "coreml").tier == "validated"
        else "cpu_only"
    )
    profile = validate_sam_coreml_profile(
        family=family,
        size=size,
        prompt_max_points=PROMPT_MAX_POINTS,
    )
    output_dir = tmp_path_factory.mktemp(f"coreml-{family}")
    artifact = Path(
        model.export(
            format="coreml",
            output_path=str(output_dir / f"{family}.mlpackage"),
            prompt_max_points=PROMPT_MAX_POINTS,
            compute_units=requested_compute_units,
        )
    )
    assert artifact.is_dir()
    del model
    gc.collect()
    # Never retain the eager model in this module-scoped fixture. The graph
    # test owns its pristine oracle lifetime and releases it before loading a
    # Core ML proxy; mixing both heavyweight runtimes causes excessive RSS and
    # can trigger native tensor-lifetime failures on macOS.
    return family, alias, profile, artifact, requested_compute_units


def test_saved_sam_multifunction_graph_parity(converted_sam_case):
    from libreyolo import LibreSAM
    from libreyolo.backends.coreml_sam import prepare_sam_coreml_image
    from libreyolo.export.coreml_sam import (
        SAM_COREML_ENCODER_FUNCTION,
        parse_sam_coreml_runtime_function,
        sam_coreml_runtime_function_contracts,
        sam_coreml_runtime_function_names,
        validate_sam_coreml_metadata,
        wrap_sam_coreml_components,
    )

    family, alias, profile, artifact, _requested_compute_units = (
        converted_sam_case
    )
    runtime_names = sam_coreml_runtime_function_names(profile)
    contracts = sam_coreml_runtime_function_contracts(profile)
    images = (
        _byte_probe(79, 61, phase=0),
        _byte_probe(79, 61, phase=97),
    )

    # Build every eager reference before creating any native Core ML proxy.
    # Owned NumPy arrays keep the graph gate independent of exporter mutation
    # while allowing the multi-gigabyte pristine model to be released before
    # runtime compilation and prediction.
    model = LibreSAM(alias, device="cpu")
    components = wrap_sam_coreml_components(
        model.model,
        profile=profile,
    )
    encoder_names = profile.embedding_names
    encoder_cases = []
    decoder_cases = {name: [] for name in runtime_names[1:]}
    for image in images:
        encoding = prepare_sam_coreml_image(image, profile=profile)
        encoder = components[SAM_COREML_ENCODER_FUNCTION]
        with torch.no_grad():
            encoder_expected = _as_output_tuple(encoder(encoding.pixel_values))
        assert len(encoder_expected) == len(encoder_names)
        encoder_reference = tuple(_owned_numpy(value) for value in encoder_expected)
        encoder_cases.append(
            (
                {"pixel_values": _owned_numpy(encoding.pixel_values)},
                encoder_reference,
            )
        )
        embeddings = dict(zip(encoder_names, encoder_expected))
        owned_embeddings = dict(zip(encoder_names, encoder_reference))

        for runtime_name in runtime_names[1:]:
            source_name, fixed_point_count = parse_sam_coreml_runtime_function(
                runtime_name,
                profile=profile,
            )
            inputs = _component_inputs(
                source_name,
                image=image,
                encoding=encoding,
                embeddings=embeddings,
                profile=profile,
                point_count=fixed_point_count or 1,
            )
            input_names = [
                item["name"] for item in contracts[runtime_name]["inputs"]
            ]
            output_names = [
                item["name"] for item in contracts[runtime_name]["outputs"]
            ]
            component_args = tuple(inputs[name] for name in input_names)
            with torch.no_grad():
                expected = _as_output_tuple(
                    components[source_name](*component_args)
                )
            assert len(expected) == len(output_names)
            owned_inputs = {
                name: (
                    owned_embeddings[name]
                    if name in owned_embeddings
                    else _owned_numpy(inputs[name])
                )
                for name in input_names
            }
            decoder_cases[runtime_name].append(
                (
                    owned_inputs,
                    tuple(_owned_numpy(value) for value in expected),
                )
            )

    del (
        component_args,
        components,
        encoder,
        encoder_expected,
        embeddings,
        encoding,
        expected,
        inputs,
        model,
        owned_embeddings,
    )
    gc.collect()

    default_runtime = ct.models.MLModel(
        str(artifact),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    metadata = dict(default_runtime.user_defined_metadata or {})
    validate_sam_coreml_metadata(metadata)
    spec = default_runtime.get_spec()
    assert spec.specificationVersion >= 9
    assert spec.description.defaultFunctionName == SAM_COREML_ENCODER_FUNCTION
    assert [item.name for item in spec.description.functions] == list(runtime_names)

    encoder_references = []
    encoder_conversion_errors = []
    for encoder_inputs, encoder_expected in encoder_cases:
        encoder_actual = _runtime_outputs(
            default_runtime,
            encoder_inputs,
            encoder_names,
        )
        encoder_conversion_errors.append(
            tuple(
                _assert_close(
                    expected,
                    actual,
                    label=f"{family}/{SAM_COREML_ENCODER_FUNCTION}/{name}",
                )
                for name, expected, actual in zip(
                    encoder_names,
                    encoder_expected,
                    encoder_actual,
                )
            )
        )
        encoder_references.append(encoder_expected)

    for index, (name, first, second) in enumerate(
        zip(
            profile.embedding_names,
            encoder_references[0],
            encoder_references[1],
        )
    ):
        scale = max(
            float(np.abs(first).max()),
            float(np.abs(second).max()),
            1e-12,
        )
        sensitivity = float(np.abs(second - first).max()) / scale
        conversion_error = max(
            encoder_conversion_errors[0][index],
            encoder_conversion_errors[1][index],
        )
        required = max(
            MIN_REL_SENSITIVITY,
            SENSITIVITY_TO_ERROR_MARGIN * conversion_error,
        )
        assert sensitivity >= required, (
            f"{family}/encode_image/{name} relative input sensitivity "
            f"{sensitivity:.3e} is below required {required:.3e} "
            f"(conversion error {conversion_error:.3e})"
        )

    del default_runtime
    gc.collect()

    for runtime_name in runtime_names[1:]:
        runtime = ct.models.MLModel(
            str(artifact),
            compute_units=ct.ComputeUnit.CPU_ONLY,
            function_name=runtime_name,
        )
        references = []
        conversion_errors = []
        input_names = [
            item["name"] for item in contracts[runtime_name]["inputs"]
        ]
        output_names = [
            item["name"] for item in contracts[runtime_name]["outputs"]
        ]
        for inputs, expected in decoder_cases[runtime_name]:
            actual = _runtime_outputs(
                runtime,
                inputs,
                output_names,
            )
            conversion_errors.append(
                tuple(
                    _assert_close(
                        reference,
                        candidate,
                        label=f"{family}/{runtime_name}/{name}",
                    )
                    for name, reference, candidate in zip(
                        output_names,
                        expected,
                        actual,
                    )
                )
            )
            references.append(expected)
        for index, (name, first, second) in enumerate(
            zip(
                output_names,
                references[0],
                references[1],
            )
        ):
            scale = max(
                float(np.abs(first).max()),
                float(np.abs(second).max()),
                1e-12,
            )
            sensitivity = float(np.abs(second - first).max()) / scale
            conversion_error = max(
                conversion_errors[0][index],
                conversion_errors[1][index],
            )
            required = max(
                MIN_REL_SENSITIVITY,
                SENSITIVITY_TO_ERROR_MARGIN * conversion_error,
            )
            assert sensitivity >= required, (
                f"{family}/{runtime_name}/{name} relative input sensitivity "
                f"{sensitivity:.3e} is below required {required:.3e} "
                f"(conversion error {conversion_error:.3e})"
            )
        del runtime
        gc.collect()


def test_public_sam_coreml_cached_prompt_path(converted_sam_case):
    from libreyolo import LibreSAM
    from libreyolo.backends.coreml_sam import (
        postprocess_sam_coreml_masks,
        prepare_sam_coreml_image,
        transform_sam_coreml_points,
    )
    from libreyolo.export.coreml_sam import wrap_sam_coreml_components

    _family, alias, profile, artifact, requested_compute_units = (
        converted_sam_case
    )
    model = LibreSAM(alias, device="cpu")
    components = wrap_sam_coreml_components(
        model.model,
        profile=profile,
    )
    image = _byte_probe(83, 67, phase=31)
    point = [[41.25, 33.75]]
    encoding = prepare_sam_coreml_image(image, profile=profile)
    with torch.no_grad():
        encoder_outputs = _as_output_tuple(
            components["encode_image"](encoding.pixel_values)
        )
    embeddings = dict(zip(profile.embedding_names, encoder_outputs))
    point_coords = transform_sam_coreml_points(
        point,
        encoding=encoding,
        profile=profile,
    )
    point_labels = torch.ones((1, 1, 1), dtype=torch.int32)
    direct_inputs = _component_inputs(
        "decode_points_single",
        image=image,
        encoding=encoding,
        embeddings=embeddings,
        profile=profile,
        point_count=1,
    )
    direct_inputs["point_coords"] = point_coords
    direct_inputs["point_labels"] = point_labels
    with torch.no_grad():
        low_res, scores = components["decode_points_single"](
            *(direct_inputs[name] for name in direct_inputs)
        )
    expected_mask = postprocess_sam_coreml_masks(
        low_res,
        encoding=encoding,
        profile=profile,
    )[0]
    expected_score = float(scores[0, 0, 0])
    del (
        components,
        direct_inputs,
        embeddings,
        encoder_outputs,
        encoding,
        low_res,
        model,
        point_coords,
        point_labels,
        scores,
    )
    gc.collect()

    backend = LibreSAM(
        str(artifact),
        compute_units=requested_compute_units,
    )
    backend.set_image(image)
    result = backend.predict(points=point, labels=[1])
    if bool(expected_mask.any()):
        assert len(result) == 1
        assert result.masks is not None
        actual_mask = result.masks.data[0].bool()
        intersection = float((actual_mask & expected_mask).sum())
        union = float((actual_mask | expected_mask).sum())
        assert intersection / max(union, 1.0) >= 0.99
        assert float(result.boxes.conf[0]) == pytest.approx(
            expected_score,
            rel=REL_TOL,
            abs=REL_TOL,
        )
    else:
        assert len(result) == 0

    backend.predict(
        bboxes=[8.0, 7.0, 74.0, 59.0],
        multimask=True,
    )
    points2, labels2 = _point_prompts(image, count=2)
    backend.predict(points=[points2], labels=[labels2])
    cached_p2 = backend._sam_functions["decode_points_single_p2"]
    backend.predict(points=[points2], labels=[labels2])
    assert backend._sam_functions["decode_points_single_p2"] is cached_p2
    del cached_p2

    points4, labels4 = _point_prompts(image, count=4)
    backend.predict(points=[points4], labels=[labels4])
    assert set(backend._sam_functions) == {"decode_points_single_p4"}
    points3, labels3 = _point_prompts(image, count=3)
    backend.predict(points=[points3], labels=[labels3])
    assert set(backend._sam_functions) == {"decode_points_single_p3"}
    backend.predict(points=point, labels=[1])
    assert set(backend._sam_functions) == {"decode_points_single_p1"}
    backend.predict(
        bboxes=[8.0, 7.0, 74.0, 59.0],
        multimask=True,
    )
    assert set(backend._sam_functions) == {"decode_boxes_multimask"}
    backend.predict(
        points=[points4],
        labels=[labels4],
        bboxes=[8.0, 7.0, 74.0, 59.0],
        multimask=True,
    )
    assert set(backend._sam_functions) == {
        "decode_points_boxes_multimask_p4",
    }
    backend.reset_image()
    with pytest.raises(RuntimeError, match="No image set"):
        backend.predict(points=point)
