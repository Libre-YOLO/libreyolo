from dataclasses import replace

import pytest

from libreyolo.export.coreml_identity import (
    COREML_DEPLOYMENT_ABI_SCHEMA,
)
from libreyolo.export.coreml_profiles import (
    COREML_EXECUTION_PROFILES,
    COREML_EXECUTION_PROFILES_BY_ID,
    COREML_EXECUTION_PROFILE_VERSION,
    COREML_VALIDATED_EXECUTION_PROFILES,
    CoreMLExecutionProfile,
    coreml_execution_profile_metadata,
    finalize_coreml_execution_profile_metadata,
    match_coreml_execution_profile,
    merge_coreml_execution_profile_metadata,
    resolve_coreml_export_compute_units,
    resolve_coreml_runtime_compute_units,
    validate_coreml_execution_profile_metadata,
)

pytestmark = pytest.mark.unit

_TEST_SOURCE_SHA256 = "1" * 64
_TEST_EVIDENCE_SHA256 = "2" * 64
_TEST_ABI_SHA256 = "3" * 64


def _candidate_key(candidate):
    return next(
        key
        for key, profile in COREML_EXECUTION_PROFILES.items()
        if profile is candidate
    )


def _base_metadata(profile):
    height, width = profile.canvas
    output_name = "embedding" if profile.task == "embed" else "prediction"
    output_shape = (
        [1, int(profile.embedding_dim)]
        if profile.embedding_dim is not None
        else [1, 1, 1]
    )
    metadata = {
        "model_family": profile.family,
        "task": profile.task,
        "size": profile.size,
        "model_size": profile.size,
        "imgsz": max(height, width),
        "imgsz_h": height,
        "imgsz_w": width,
        "precision": profile.precision,
        "nms": profile.nms,
        "coreml_io_schema_version": "2",
        "coreml_output_names": [output_name],
        "coreml_io": {
            "input": {
                "name": "image",
                "kind": "tensor",
                "layout": "nchw",
                "color": "rgb",
                "range": "0_1",
                "shape_mode": "fixed",
            },
            "validation": {"color": "rgb", "range": "0_1"},
            "outputs": [
                {
                    "name": output_name,
                    "role": output_name,
                    "rank": len(output_shape),
                    "dtype": "float32",
                    "shape": output_shape,
                }
            ],
        },
    }
    if profile.prompt_max_points is not None:
        metadata["prompt_max_points"] = profile.prompt_max_points
    if profile.class_count is not None:
        metadata["nc"] = profile.class_count
        metadata["nb_classes"] = profile.class_count
    for attribute, metadata_key in (
        ("graph_class_width", "graph_class_width"),
        ("num_keypoints", "num_keypoints"),
        ("keypoint_dim", "keypoint_dim"),
        ("classification_activation", "classification_activation"),
        ("checkpoint_variant", "checkpoint_variant"),
        ("architecture_signature", "architecture_signature"),
        ("restore_scale", "restore_scale"),
        ("embedding_dim", "facerec_embedding_dim"),
    ):
        value = getattr(profile, attribute)
        if value is not None:
            metadata[metadata_key] = value
    if profile.num_keypoints_per_class:
        metadata["num_keypoints_per_class"] = list(
            profile.num_keypoints_per_class
        )
    return metadata


def _promote_for_test(monkeypatch, candidate):
    promoted = replace(
        candidate,
        source_kind="test-source-v1",
        source_sha256=_TEST_SOURCE_SHA256,
        deployment_abi_sha256=_TEST_ABI_SHA256,
        evidence_sha256=_TEST_EVIDENCE_SHA256,
    )
    monkeypatch.setitem(
        COREML_EXECUTION_PROFILES,
        _candidate_key(candidate),
        promoted,
    )
    monkeypatch.setitem(
        COREML_EXECUTION_PROFILES_BY_ID,
        promoted.profile_id,
        promoted,
    )
    return promoted


def _strict_metadata(profile):
    metadata = _base_metadata(profile)
    metadata.update(
        {
            "coreml_profile_source_kind": profile.source_kind,
            "coreml_profile_source_sha256": profile.source_sha256,
            "coreml_profile_abi_schema": COREML_DEPLOYMENT_ABI_SCHEMA,
            "coreml_profile_abi_sha256": profile.deployment_abi_sha256,
        }
    )
    return merge_coreml_execution_profile_metadata(
        metadata,
        profile,
        conversion_compute_units=profile.conversion_compute_units,
    )


def _face_candidate():
    profile = match_coreml_execution_profile(
        "facerec",
        "embed",
        "l",
        112,
        class_count=1,
        embedding_dim=512,
    )
    assert profile is not None and not profile.evidence_complete
    return profile


def _face_metadata(profile):
    preprocess = {
        "layout": "NCHW",
        "color": "rgb",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    }
    metadata = _base_metadata(profile)
    metadata["coreml_io"] = {
        "input": {
            "name": "aligned_face",
            "kind": "tensor",
            **preprocess,
            "range": "standardized",
            "geometry": "host_aligned_face",
            "interpolation": "bilinear",
            "resize_backend": "opencv",
            "pad_value": 0,
            "shape_mode": "fixed",
        },
        "validation": {
            "color": "rgb",
            "range": "standardized",
            "mean": preprocess["mean"],
            "std": preprocess["std"],
        },
        "outputs": [
            {
                "name": "embedding",
                "role": "embedding",
                "encoding": "raw_identity_embedding",
                "rank": 2,
                "dtype": "float32",
                "shape": [1, 512],
            }
        ],
    }
    metadata["coreml_output_names"] = ["embedding"]
    metadata.update(
        {
            "coreml_profile_source_kind": profile.source_kind,
            "coreml_profile_source_sha256": profile.source_sha256,
            "coreml_profile_abi_schema": COREML_DEPLOYMENT_ABI_SCHEMA,
            "coreml_profile_abi_sha256": profile.deployment_abi_sha256,
        }
    )
    return merge_coreml_execution_profile_metadata(
        metadata,
        profile,
        conversion_compute_units="cpu_only",
    )


def test_registry_separates_conversion_candidates_from_promoted_profiles():
    assert len(COREML_EXECUTION_PROFILES) >= 40
    assert COREML_EXECUTION_PROFILE_VERSION == "2"
    ids = {profile.profile_id for profile in COREML_EXECUTION_PROFILES.values()}
    assert len(ids) == len(COREML_EXECUTION_PROFILES)
    assert all(profile_id.startswith("coreml-m4-v2/") for profile_id in ids)
    assert {
        (profile.family, profile.task, profile.size)
        for profile in COREML_VALIDATED_EXECUTION_PROFILES.values()
    } == {
        ("birefnet", "matte", "l"),
        ("clip", "classify", "b32"),
        ("deim", "detect", "n"),
        ("deimv2", "detect", "atto"),
        ("depth_anything", "depth", "s"),
        ("depth_anything3", "depth", "l"),
        ("dfine", "detect", "n"),
        ("dfine", "segment", "n"),
        ("ec", "detect", "s"),
        ("ec", "pose", "s"),
        ("ec", "segment", "s"),
        ("edgetam", "segment", "edge"),
        ("efficientnetv2", "classify", "b0"),
        ("fomo", "point", "s"),
        ("lingbotvision", "semantic", "s"),
        ("mobilenetv4", "classify", "s"),
        ("mobilesam", "segment", "tiny"),
        ("picodet", "detect", "s"),
        ("picosam3", "segment", "pico"),
        ("pidnet", "semantic", "s"),
        ("realesrgan", "restore", "x4t"),
        ("rfdetr", "detect", "n"),
        ("rfdetr", "obb", "n"),
        ("rfdetr", "pose", "x"),
        ("resnet", "classify", "18"),
        ("rtdetr", "detect", "r18"),
        ("rtdetrv2", "detect", "r18"),
        ("rtdetrv4", "detect", "s"),
        ("rtmdet", "detect", "t"),
        ("rtmdet", "segment", "t"),
        ("sam", "segment", "base"),
        ("sam2", "segment", "tiny"),
        ("siglip2", "classify", "b16"),
        ("yolo1", "detect", "b"),
        ("yolo2", "detect", "b"),
        ("yolo3", "detect", "b"),
        ("yolo4", "detect", "b"),
        ("yolo7", "detect", "b"),
        ("yolo9", "detect", "t"),
        ("yolo9_e2e", "detect", "t"),
        ("yolo9_p2", "detect", "t"),
        ("yolonas", "detect", "s"),
        ("yolonas", "pose", "n"),
        ("yolox", "detect", "n"),
        ("zipdepth", "depth", "b"),
    }
    assert all(
        profile.evidence_complete
        for profile in COREML_VALIDATED_EXECUTION_PROFILES.values()
    )


def test_unpromoted_recipe_fails_validated_and_allows_explicit_campaign():
    with pytest.raises(NotImplementedError, match="not yet been promoted"):
        resolve_coreml_export_compute_units(
            "validated",
            family="yolo9",
            task="detect",
            size="s",
            canvas=640,
            precision="fp32",
            nms=False,
            class_count=80,
        )
    with pytest.warns(RuntimeWarning, match="awaiting"):
        units, profile = resolve_coreml_export_compute_units(
            "cpu_only",
            family="yolo9",
            task="detect",
            size="s",
            canvas=640,
            precision="fp32",
            nms=False,
            class_count=80,
        )
    assert (units, profile) == ("cpu_only", None)


def test_promoted_export_requires_exact_source_identity(monkeypatch):
    profile = _promote_for_test(monkeypatch, _face_candidate())
    common = {
        "family": "facerec",
        "task": "embed",
        "size": "l",
        "canvas": 112,
        "precision": "fp32",
        "nms": False,
        "class_count": 1,
        "embedding_dim": 512,
    }
    units, matched = resolve_coreml_export_compute_units(
        "validated",
        **common,
        source_kind=profile.source_kind,
        source_sha256=profile.source_sha256,
    )
    assert (units, matched) == ("cpu_only", profile)

    with pytest.raises(NotImplementedError, match="live source identity"):
        resolve_coreml_export_compute_units("validated", **common)
    with pytest.raises(NotImplementedError, match="exact checkpoint"):
        resolve_coreml_export_compute_units(
            "validated",
            **common,
            source_kind=profile.source_kind,
            source_sha256="0" * 64,
        )
    with pytest.warns(RuntimeWarning, match="does not match"):
        units, matched = resolve_coreml_export_compute_units(
            "cpu_only",
            **common,
            source_kind=profile.source_kind,
            source_sha256="0" * 64,
        )
    assert (units, matched) == ("cpu_only", None)


@pytest.mark.parametrize(
    ("canvas", "class_count"),
    [
        ((640.9, 640), 80),
        ((640, 640.1), 80),
        ((640, 640), 80.9),
        (640.0, 80),
    ],
)
def test_exact_profile_matching_rejects_float_truncation(canvas, class_count):
    with pytest.raises(ValueError, match="must be an integer"):
        resolve_coreml_export_compute_units(
            "validated",
            family="yolo9",
            task="detect",
            size="t",
            canvas=canvas,
            precision="fp32",
            nms=False,
            class_count=class_count,
        )


@pytest.mark.parametrize("prompt_max_points", [1, 4, 16])
def test_each_mobile_sam_prompt_bound_has_a_distinct_candidate(
    prompt_max_points,
):
    profile = match_coreml_execution_profile(
        "mobilesam",
        "segment",
        "tiny",
        1024,
        prompt_max_points=prompt_max_points,
        class_count=1,
    )
    assert profile is not None
    assert profile.prompt_max_points == prompt_max_points
    assert profile.evidence_complete is (prompt_max_points == 4)


@pytest.mark.parametrize(
    ("family", "task", "size", "canvas", "dimensions"),
    [
        (
            "clip",
            "classify",
            "b32",
            224,
            {"class_count": 3, "classification_activation": "softmax"},
        ),
        (
            "ec",
            "pose",
            "s",
            640,
            {
                "class_count": 1,
                "graph_class_width": 2,
                "num_keypoints": 17,
                "keypoint_dim": 2,
            },
        ),
        (
            "nafnet",
            "restore",
            "l",
            256,
            {
                "class_count": 1,
                "checkpoint_variant": "sidd",
                "architecture_signature": "w64-m12-e2.2.4.8-d2.2.2.2",
                "restore_scale": 1,
            },
        ),
        (
            "rfdetr",
            "pose",
            "x",
            576,
            {
                "class_count": 1,
                "graph_class_width": 2,
                "num_keypoints": 17,
                "keypoint_dim": 8,
                "num_keypoints_per_class": (0, 17),
            },
        ),
    ],
)
def test_graph_dimensions_are_part_of_candidate_identity(
    family,
    task,
    size,
    canvas,
    dimensions,
):
    profile = match_coreml_execution_profile(
        family,
        task,
        size,
        canvas,
        **dimensions,
    )
    assert profile is not None
    changed = dict(dimensions)
    dimension = next(key for key in dimensions if key != "class_count")
    value = changed[dimension]
    changed[dimension] = (
        value + 1
        if isinstance(value, int)
        else (*value[:-1], value[-1] + 1)
        if isinstance(value, tuple)
        else f"{value}-other"
    )
    assert (
        match_coreml_execution_profile(
            family,
            task,
            size,
            canvas,
            **changed,
        )
        is None
    )


def test_unpromoted_candidate_cannot_emit_validated_metadata():
    profile = match_coreml_execution_profile(
        "yolo9",
        "detect",
        "s",
        640,
        class_count=80,
    )
    assert profile is not None and not profile.evidence_complete
    with pytest.raises(ValueError, match="unpromoted"):
        coreml_execution_profile_metadata(
            profile,
            conversion_compute_units="cpu_only",
        )


def test_finalizer_rejects_validated_abi_mismatch_and_demotes_explicit(
    monkeypatch,
):
    profile = _promote_for_test(monkeypatch, _face_candidate())
    metadata = _base_metadata(profile)
    metadata.update(
        {
            "coreml_profile_source_kind": profile.source_kind,
            "coreml_profile_source_sha256": profile.source_sha256,
            "coreml_profile_abi_schema": COREML_DEPLOYMENT_ABI_SCHEMA,
            "coreml_profile_abi_sha256": "4" * 64,
        }
    )
    with pytest.raises(RuntimeError, match="protobuf ABI"):
        finalize_coreml_execution_profile_metadata(
            metadata,
            profile,
            requested_compute_units="validated",
            conversion_compute_units="cpu_only",
            deployment_abi_sha256="4" * 64,
        )

    with pytest.warns(RuntimeWarning, match="experimental"):
        finalized, matched = finalize_coreml_execution_profile_metadata(
            metadata,
            profile,
            requested_compute_units="cpu_only",
            conversion_compute_units="cpu_only",
            deployment_abi_sha256="4" * 64,
        )
    assert matched is None
    assert finalized["coreml_execution_profile_status"] == "experimental"
    assert "coreml_execution_profile" not in finalized


def test_face_v2_metadata_round_trips_and_routes_before_proxy(monkeypatch):
    profile = _promote_for_test(monkeypatch, _face_candidate())
    metadata = _face_metadata(profile)
    assert validate_coreml_execution_profile_metadata(metadata) is profile
    assert (
        resolve_coreml_runtime_compute_units("validated", metadata)
        == "cpu_only"
    )
    assert (
        resolve_coreml_runtime_compute_units("cpu_only", metadata)
        == "cpu_only"
    )
    with pytest.raises(NotImplementedError, match="validated only"):
        resolve_coreml_runtime_compute_units("all", metadata)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("coreml_execution_profile", "unknown"),
        ("coreml_validation_hardware", "Apple M3"),
        ("coreml_profile_source_kind", "other-source-v1"),
        ("coreml_profile_source_sha256", "0" * 64),
        ("coreml_profile_abi_sha256", "0" * 64),
        ("coreml_validation_evidence_sha256", "0" * 64),
        ("coreml_conversion_compute_units", "all"),
        ("coreml_runtime_compute_units", '["all"]'),
        ("nb_classes", "2"),
        ("imgsz_w", "111"),
    ],
)
def test_tampered_v2_profile_metadata_fails_closed(
    monkeypatch,
    key,
    value,
):
    profile = _promote_for_test(monkeypatch, _face_candidate())
    metadata = _face_metadata(profile)
    metadata[key] = value
    with pytest.raises(ValueError):
        validate_coreml_execution_profile_metadata(metadata)


@pytest.mark.parametrize(
    ("candidate_args", "tamper_key", "tamper_value", "message"),
    [
        (
            (
                "nafnet",
                "restore",
                "l",
                256,
                {
                    "class_count": 1,
                    "checkpoint_variant": "sidd",
                    "architecture_signature": "w64-m12-e2.2.4.8-d2.2.2.2",
                    "restore_scale": 1,
                },
            ),
            "coreml_disabled_passes",
            "[]",
            "pass",
        ),
        (
            (
                "edgetam",
                "segment",
                "edge",
                1024,
                {"prompt_max_points": 4, "class_count": 1},
            ),
            "coreml_capture_decompositions",
            "[]",
            "capture",
        ),
        (
            (
                "rfdetr",
                "pose",
                "x",
                576,
                {
                    "class_count": 1,
                    "graph_class_width": 2,
                    "num_keypoints": 17,
                    "keypoint_dim": 8,
                    "num_keypoints_per_class": (0, 17),
                },
            ),
            "coreml_validated_num_keypoints_per_class",
            "[0, 16]",
            "keypoint",
        ),
    ],
)
def test_promoted_specialized_metadata_still_fails_closed(
    monkeypatch,
    candidate_args,
    tamper_key,
    tamper_value,
    message,
):
    family, task, size, canvas, dimensions = candidate_args
    candidate = match_coreml_execution_profile(
        family,
        task,
        size,
        canvas,
        **dimensions,
    )
    assert candidate is not None
    profile = _promote_for_test(monkeypatch, candidate)
    metadata = _strict_metadata(profile)
    metadata[tamper_key] = tamper_value
    with pytest.raises(ValueError, match=message):
        validate_coreml_execution_profile_metadata(metadata)


def test_v1_validated_artifact_is_legacy_cpu_only_opt_in():
    metadata = {
        "coreml_execution_profile_status": "validated",
        "coreml_execution_profile_version": "1",
        "coreml_execution_profile": (
            "coreml-m4-v1/yolo9/detect/t/640x640/raw-fp32-cpu_only"
        ),
    }
    with pytest.raises(NotImplementedError, match="legacy"):
        resolve_coreml_runtime_compute_units("validated", metadata)
    with pytest.warns(RuntimeWarning, match="legacy"):
        assert (
            resolve_coreml_runtime_compute_units("cpu_only", metadata)
            == "cpu_only"
        )
    with pytest.raises(NotImplementedError, match="cpu_only"):
        resolve_coreml_runtime_compute_units("all", metadata)


def test_markerless_exact_like_artifact_is_never_laundered():
    metadata = {
        "model_family": "yolox",
        "task": "detect",
        "size": "n",
        "imgsz": 416,
        "precision": "fp32",
        "nms": "false",
        "nc": 80,
    }
    with pytest.raises(NotImplementedError, match="legacy"):
        resolve_coreml_runtime_compute_units("validated", metadata)
    with pytest.warns(RuntimeWarning, match="legacy"):
        assert (
            resolve_coreml_runtime_compute_units("cpu_only", metadata)
            == "cpu_only"
        )
    with pytest.raises(NotImplementedError, match="Re-export"):
        resolve_coreml_runtime_compute_units("all", metadata)


def test_experimental_metadata_has_no_validated_profile_id():
    metadata = coreml_execution_profile_metadata(
        None,
        conversion_compute_units="cpu_only",
    )
    assert metadata == {"coreml_execution_profile_status": "experimental"}
    assert validate_coreml_execution_profile_metadata(metadata) is None
    with pytest.raises(NotImplementedError, match="experimental"):
        resolve_coreml_runtime_compute_units("validated", metadata)
    assert (
        resolve_coreml_runtime_compute_units("cpu_only", metadata)
        == "cpu_only"
    )


def test_profile_dataclass_requires_all_evidence_before_promotion():
    candidate = CoreMLExecutionProfile(
        family="test",
        task="detect",
        size="n",
        canvas=(32, 32),
        reference="fixture",
        source_kind="test-source-v1",
    )
    assert not candidate.evidence_complete
