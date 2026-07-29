"""Core ML graph and public-path parity for split promptable SAM packages.

This suite is intentionally macOS-only. Linux conversion tests can validate
MIL structure, but only Core ML execution can prove the saved multifunction
package. Each family is converted once, then exercised through every named
function and through the public encode-once/prompt-many API.
"""

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
PROMPT_MAX_POINTS = 4

SAM_CASES = [
    ("edgetam", "edge", "edgetam"),
    ("mobilesam", "tiny", "mobilesam"),
    ("sam", "base", "base"),
    ("sam2", "tiny", "sam2-tiny"),
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
    expected: torch.Tensor,
    actual: np.ndarray,
    *,
    label: str,
) -> None:
    reference = expected.detach().cpu().numpy()
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


def _runtime_outputs(runtime, inputs, output_names):
    arrays = {
        name: np.ascontiguousarray(value.detach().cpu().numpy())
        for name, value in inputs.items()
    }
    outputs = runtime.predict(arrays)
    assert set(outputs) == set(output_names)
    return tuple(np.asarray(outputs[name]) for name in output_names)


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
    from libreyolo.export.coreml_sam import (
        validate_sam_coreml_profile,
        wrap_sam_coreml_components,
    )

    model = LibreSAM(alias, device="cpu")
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
            compute_units="cpu_only",
        )
    )
    assert artifact.is_dir()
    components = wrap_sam_coreml_components(
        model.model,
        profile=profile,
    )
    return family, model, profile, artifact, components


def test_saved_sam_multifunction_graph_parity(converted_sam_case):
    from libreyolo.backends.coreml_sam import prepare_sam_coreml_image
    from libreyolo.export.coreml_sam import (
        SAM_COREML_ENCODER_FUNCTION,
        SAM_COREML_FUNCTION_NAMES,
        sam_coreml_function_contracts,
        validate_sam_coreml_metadata,
    )

    family, _model, profile, artifact, components = converted_sam_case
    default_runtime = ct.models.MLModel(
        str(artifact),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    metadata = dict(default_runtime.user_defined_metadata or {})
    validate_sam_coreml_metadata(metadata)
    spec = default_runtime.get_spec()
    assert spec.specificationVersion >= 9
    assert spec.description.defaultFunctionName == SAM_COREML_ENCODER_FUNCTION
    assert [item.name for item in spec.description.functions] == list(
        SAM_COREML_FUNCTION_NAMES
    )

    runtimes = {SAM_COREML_ENCODER_FUNCTION: default_runtime}
    runtimes.update(
        {
            function_name: ct.models.MLModel(
                str(artifact),
                compute_units=ct.ComputeUnit.CPU_ONLY,
                function_name=function_name,
            )
            for function_name in SAM_COREML_FUNCTION_NAMES[1:]
        }
    )
    contracts = sam_coreml_function_contracts(profile)
    images = (
        _byte_probe(79, 61, phase=0),
        _byte_probe(79, 61, phase=97),
    )
    previous_reference: dict[str, tuple[torch.Tensor, ...]] = {}

    for image_index, image in enumerate(images):
        encoding = prepare_sam_coreml_image(image, profile=profile)
        encoder = components[SAM_COREML_ENCODER_FUNCTION]
        with torch.no_grad():
            encoder_expected = _as_output_tuple(encoder(encoding.pixel_values))
        encoder_names = profile.embedding_names
        encoder_actual = _runtime_outputs(
            runtimes[SAM_COREML_ENCODER_FUNCTION],
            {"pixel_values": encoding.pixel_values},
            encoder_names,
        )
        for name, expected, actual in zip(
            encoder_names,
            encoder_expected,
            encoder_actual,
        ):
            _assert_close(
                expected,
                actual,
                label=f"{family}/{SAM_COREML_ENCODER_FUNCTION}/{name}",
            )
        if image_index == 0:
            previous_reference[SAM_COREML_ENCODER_FUNCTION] = encoder_expected
        else:
            first_encoder = previous_reference[SAM_COREML_ENCODER_FUNCTION]
            for name, first, second in zip(
                encoder_names,
                first_encoder,
                encoder_expected,
            ):
                scale = max(
                    float(first.detach().abs().max()),
                    float(second.detach().abs().max()),
                    1e-12,
                )
                sensitivity = (
                    float((second.detach() - first.detach()).abs().max()) / scale
                )
                assert sensitivity >= MIN_REL_SENSITIVITY, (
                    f"{family}/encode_image/{name} relative input sensitivity "
                    f"{sensitivity:.3e} is too small"
                )
        embeddings = dict(zip(encoder_names, encoder_expected))

        for function_name in SAM_COREML_FUNCTION_NAMES[1:]:
            point_counts = (
                (1, 2, PROMPT_MAX_POINTS)
                if "points" in function_name
                else (1,)
            )
            for point_count in point_counts:
                inputs = _component_inputs(
                    function_name,
                    image=image,
                    encoding=encoding,
                    embeddings=embeddings,
                    profile=profile,
                    point_count=point_count,
                )
                input_names = [
                    item["name"] for item in contracts[function_name]["inputs"]
                ]
                output_names = [
                    item["name"] for item in contracts[function_name]["outputs"]
                ]
                component_args = tuple(inputs[name] for name in input_names)
                with torch.no_grad():
                    expected = _as_output_tuple(
                        components[function_name](*component_args)
                    )
                actual = _runtime_outputs(
                    runtimes[function_name],
                    inputs,
                    output_names,
                )
                case_name = f"{function_name}/P={point_count}"
                for name, reference, candidate in zip(
                    output_names,
                    expected,
                    actual,
                ):
                    _assert_close(
                        reference,
                        candidate,
                        label=f"{family}/{case_name}/{name}",
                    )
                if image_index == 0:
                    previous_reference[case_name] = expected
                else:
                    prior = previous_reference[case_name]
                    for name, first, second in zip(output_names, prior, expected):
                        scale = max(
                            float(first.detach().abs().max()),
                            float(second.detach().abs().max()),
                            1e-12,
                        )
                        sensitivity = (
                            float((second.detach() - first.detach()).abs().max())
                            / scale
                        )
                        assert sensitivity >= MIN_REL_SENSITIVITY, (
                            f"{family}/{case_name}/{name} relative input "
                            f"sensitivity {sensitivity:.3e} is too small"
                        )


def test_public_sam_coreml_cached_prompt_path(converted_sam_case):
    from libreyolo import LibreSAM
    from libreyolo.backends.coreml_sam import (
        postprocess_sam_coreml_masks,
        prepare_sam_coreml_image,
        transform_sam_coreml_points,
    )

    family, _model, profile, artifact, components = converted_sam_case
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

    backend = LibreSAM(str(artifact), compute_units="cpu_only")
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
    backend.reset_image()
    with pytest.raises(RuntimeError, match="No image set"):
        backend.predict(points=point)
