"""Offline contract tests for LibrePPOCR's Core ML multifunction package."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.unit]


def _pipeline(*, det_limit_side_len: int = 960) -> dict:
    return {
        "det_limit_side_len": det_limit_side_len,
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.6,
        "det_db_unclip_ratio": 1.5,
        "rec_image_shape": [3, 48, 320],
    }


def _profile(*, size: str = "t", rec_max_width: int = 2048):
    from libreyolo.export.coreml_ppocr import validate_ppocr_coreml_profile

    return validate_ppocr_coreml_profile(
        size=size,
        det_limit_side_len=960,
        rec_batch_max=6,
        rec_max_width=rec_max_width,
    )


def _metadata():
    from libreyolo.export.coreml_ppocr import ppocr_coreml_metadata

    return ppocr_coreml_metadata(
        profile=_profile(),
        charset=["blank", "a", "字", " "],
        pipeline=_pipeline(),
        rec_num_classes=4,
    )


def _stringify_metadata(metadata: dict) -> dict[str, str]:
    result = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list, tuple)):
            result[key] = json.dumps(value, ensure_ascii=False)
        else:
            result[key] = str(value)
    return result


def _json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_profile_pins_native_bounds_and_rejects_lossy_variants():
    from libreyolo.export.coreml_ppocr import (
        PPOCRCoreMLProfile,
        ppocr_detector_tensor_upper_bound,
        validate_ppocr_coreml_profile,
    )

    assert ppocr_detector_tensor_upper_bound(32) == 32
    assert ppocr_detector_tensor_upper_bound(47) == 32
    assert ppocr_detector_tensor_upper_bound(48) == 64
    assert ppocr_detector_tensor_upper_bound(960) == 960
    assert ppocr_detector_tensor_upper_bound(3999) == 4000
    assert ppocr_detector_tensor_upper_bound(8000) == 4000

    profile = validate_ppocr_coreml_profile(
        size="l",
        precision="fp32",
        det_limit_side_len=961,
        rec_batch_max=12,
        rec_max_width=4096,
    )
    assert profile == PPOCRCoreMLProfile(
        size="l",
        precision="fp32",
        det_limit_side_len=961,
        det_tensor_upper=960,
        rec_batch_max=12,
        rec_max_width=4096,
    )

    with pytest.raises(NotImplementedError, match="sizes 't' and 'l'"):
        validate_ppocr_coreml_profile(
            size="s",
            rec_max_width=320,
        )
    with pytest.raises(NotImplementedError, match="FP32-only"):
        validate_ppocr_coreml_profile(
            size="t",
            precision="fp16",
            rec_max_width=320,
        )
    with pytest.raises(ValueError, match="rec_max_width must be at least 320"):
        validate_ppocr_coreml_profile(
            size="t",
            rec_max_width=319,
        )
    with pytest.raises(ValueError, match="must be an integer"):
        validate_ppocr_coreml_profile(
            size="t",
            rec_batch_max=True,
            rec_max_width=320,
        )
    with pytest.raises(ValueError, match="det_tensor_upper"):
        PPOCRCoreMLProfile(
            size="t",
            precision="fp32",
            det_limit_side_len=960,
            det_tensor_upper=992,
            rec_batch_max=6,
            rec_max_width=2048,
        )


@pytest.mark.parametrize(
    ("width", "timesteps"),
    [(320, 40), (321, 40), (324, 40), (325, 41), (641, 80)],
)
def test_recognizer_timestep_formula_is_exact(width, timesteps):
    from libreyolo.export.coreml_ppocr import ppocr_recognizer_timesteps

    assert ppocr_recognizer_timesteps(width) == timesteps


def test_charset_and_pipeline_contracts_are_strict_and_deterministic():
    from libreyolo.export.coreml_ppocr import (
        ppocr_charset_sha256,
        validate_ppocr_charset,
        validate_ppocr_pipeline_config,
    )

    charset = ["blank", "a", "字", " "]
    assert validate_ppocr_charset(charset, rec_num_classes=4) == charset
    assert (
        ppocr_charset_sha256(charset)
        == "f9c908504edf051dcbf22a679bce4612374cbb60d7ec5c30c4df281cfe1abcda"
    )
    assert validate_ppocr_pipeline_config(_pipeline()) == _pipeline()

    with pytest.raises(ValueError, match="index 0"):
        validate_ppocr_charset(["_", "a", " "])
    with pytest.raises(ValueError, match="end with"):
        validate_ppocr_charset(["blank", "a"])
    with pytest.raises(ValueError, match="charset entry 1"):
        validate_ppocr_charset(["blank", 1, " "])
    with pytest.raises(ValueError, match="classes and charset"):
        validate_ppocr_charset(charset, rec_num_classes=5)

    missing = _pipeline()
    del missing["rec_image_shape"]
    with pytest.raises(ValueError, match="missing"):
        validate_ppocr_pipeline_config(missing)
    extra = {**_pipeline(), "rotate_lines": True}
    with pytest.raises(ValueError, match="unknown"):
        validate_ppocr_pipeline_config(extra)
    wrong_shape = {**_pipeline(), "rec_image_shape": [3, 32, 320]}
    with pytest.raises(ValueError, match=r"\[3, 48, 320\]"):
        validate_ppocr_pipeline_config(wrong_shape)
    bad_threshold = {**_pipeline(), "det_db_thresh": float("nan")}
    with pytest.raises(ValueError, match="finite"):
        validate_ppocr_pipeline_config(bad_threshold)


def test_function_descriptors_pin_both_flexible_tensor_abis():
    from libreyolo.export.coreml_ppocr import ppocr_coreml_function_contracts

    contracts = ppocr_coreml_function_contracts(
        _profile(rec_max_width=4096),
        rec_num_classes=18385,
    )
    assert list(contracts) == ["detector", "recognizer"]

    detector = contracts["detector"]
    assert detector["input"]["name"] == "detector_input"
    assert detector["input"]["kind"] == "tensor"
    assert detector["input"]["dtype"] == "float32"
    assert detector["input"]["color"] == "bgr"
    assert detector["input"]["range"] == "standardized"
    assert detector["input"]["mean"] == [0.485, 0.456, 0.406]
    assert detector["input"]["std"] == [0.229, 0.224, 0.225]
    assert detector["input"]["shape"] == [
        {"axis": "N", "kind": "fixed", "value": 1},
        {"axis": "C", "kind": "fixed", "value": 3},
        {
            "axis": "H",
            "kind": "range",
            "lower_bound": 32,
            "upper_bound": 960,
            "default": 960,
            "multiple_of": 32,
        },
        {
            "axis": "W",
            "kind": "range",
            "lower_bound": 32,
            "upper_bound": 960,
            "default": 960,
            "multiple_of": 32,
        },
    ]
    assert detector["outputs"][0]["shape_relation"] == {
        "batch": "input.N",
        "channels": 1,
        "height": "input.H",
        "width": "input.W",
    }

    recognizer = contracts["recognizer"]
    assert recognizer["input"]["name"] == "recognizer_input"
    assert recognizer["input"]["range"] == "minus_1_1"
    assert recognizer["input"]["crop_width_rounding"] == "ceil"
    assert recognizer["input"]["bucket_width_rounding"] == "floor"
    assert recognizer["input"]["pad_value"] == 0.0
    assert recognizer["input"]["shape"] == [
        {
            "axis": "N",
            "kind": "range",
            "lower_bound": 1,
            "upper_bound": 6,
            "default": 1,
        },
        {"axis": "C", "kind": "fixed", "value": 3},
        {"axis": "H", "kind": "fixed", "value": 48},
        {
            "axis": "W",
            "kind": "range",
            "lower_bound": 320,
            "upper_bound": 4096,
            "default": 320,
        },
    ]
    assert recognizer["outputs"][0]["shape_relation"] == {
        "batch": "input.N",
        "timesteps": {
            "input_axis": "W",
            "add": 3,
            "divisor": 8,
            "rounding": "floor",
        },
        "classes": 18385,
    }


def test_metadata_native_and_coreml_stringified_round_trip():
    from libreyolo.export.coreml_ppocr import validate_ppocr_coreml_metadata

    metadata = _metadata()
    canonical = validate_ppocr_coreml_metadata(metadata)
    assert canonical == metadata
    assert metadata["artifact_scope"] == "host_orchestrated_pipeline_components"
    assert metadata["component_contract"] == "ppocr_det_rec_v1"
    assert metadata["coreml_multifunction"] is True
    assert metadata["coreml_minimum_deployment_targets"] == ["iOS18", "macOS15"]
    assert metadata["coreml_function_names"] == ["detector", "recognizer"]
    assert metadata["charset_sha256"] == _json_sha256(metadata["charset"])
    assert metadata["coreml_functions_sha256"] == _json_sha256(
        metadata["coreml_functions"]
    )

    stringified = _stringify_metadata(metadata)
    stringified.update(
        {
            "model_family": "ppocr",
            "task": "ocr",
            "size": "t",
            "libreyolo_producer": "libreyolo",
        }
    )
    assert validate_ppocr_coreml_metadata(stringified) == metadata


def test_metadata_rejects_tampered_hashes_interfaces_and_bounds():
    from libreyolo.export.coreml_ppocr import validate_ppocr_coreml_metadata

    bad_charset_hash = _metadata()
    bad_charset_hash["charset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="charset_sha256"):
        validate_ppocr_coreml_metadata(bad_charset_hash)

    bad_interface = _metadata()
    bad_interface["coreml_functions"]["recognizer"]["input"]["shape"][3][
        "upper_bound"
    ] = 8192
    bad_interface["coreml_functions_sha256"] = _json_sha256(
        bad_interface["coreml_functions"]
    )
    with pytest.raises(ValueError, match="coreml_functions"):
        validate_ppocr_coreml_metadata(bad_interface)

    bad_profile = _metadata()
    bad_profile["ppocr_coreml_profile"]["det_tensor_upper"] = 992
    with pytest.raises(ValueError, match="det_tensor_upper"):
        validate_ppocr_coreml_metadata(bad_profile)

    bad_pipeline = _metadata()
    bad_pipeline["pipeline"]["det_limit_side_len"] = 640
    with pytest.raises(ValueError, match="conflicts with pipeline"):
        validate_ppocr_coreml_metadata(bad_pipeline)

    bad_target = _metadata()
    bad_target["coreml_minimum_deployment_targets"] = ["iOS17", "macOS14"]
    with pytest.raises(ValueError, match="deployment_targets"):
        validate_ppocr_coreml_metadata(bad_target)

    bad_size = {**_metadata(), "size": "l"}
    with pytest.raises(ValueError, match="size conflicts"):
        validate_ppocr_coreml_metadata(bad_size)


class _ToyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.conv1 = nn.Module()
        self.backbone.conv1.conv = nn.Conv2d(1, 1, 1, bias=False)

    def forward(self, image):
        return image.mean(dim=1, keepdim=True).sigmoid()


class _ToyRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.conv1 = nn.Module()
        self.backbone.conv1.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.head = nn.Module()
        self.head.ctc_head = nn.Module()
        self.head.ctc_head.fc = nn.Linear(3, 3)

    def forward(self, crops):
        return crops.mean(dim=2).transpose(1, 2).softmax(dim=-1)


def test_component_wrappers_are_separate_eval_graphs():
    from libreyolo.export.coreml_ppocr import wrap_ppocr_coreml_components

    composite = nn.Module()
    composite.det = _ToyDetector()
    composite.rec = _ToyRecognizer()
    composite.train()

    profile = _profile()
    wrapped = wrap_ppocr_coreml_components(
        composite,
        profile=profile,
        rec_num_classes=3,
    )
    assert list(wrapped) == ["detector", "recognizer"]
    assert not wrapped["detector"].training
    assert not wrapped["recognizer"].training
    image = torch.randn(1, 3, 32, 64)
    crops = torch.randn(2, 3, 48, 320)
    torch.testing.assert_close(
        wrapped["detector"](image),
        composite.det(image),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        wrapped["recognizer"](crops),
        composite.rec(crops),
        rtol=0.0,
        atol=0.0,
    )

    incomplete = nn.Module()
    incomplete.det = _ToyDetector()
    with pytest.raises(ValueError, match=r"'.det'.*'.rec'"):
        wrap_ppocr_coreml_components(
            incomplete,
            profile=profile,
            rec_num_classes=3,
        )
    with pytest.raises(ValueError, match="CTC width"):
        wrap_ppocr_coreml_components(
            composite,
            profile=profile,
            rec_num_classes=4,
        )
    with pytest.raises(ValueError, match="graph tier"):
        wrap_ppocr_coreml_components(
            composite,
            profile=_profile(size="l"),
            rec_num_classes=3,
        )


def test_io_validators_pin_shapes_ranges_and_probability_semantics():
    from libreyolo.export.coreml_ppocr import (
        ppocr_recognizer_timesteps,
        validate_ppocr_detector_coreml_io,
        validate_ppocr_detector_coreml_shape,
        validate_ppocr_recognizer_coreml_io,
        validate_ppocr_recognizer_coreml_shape,
    )

    profile = _profile(rec_max_width=641)
    detector_input = torch.zeros(1, 3, 64, 96, dtype=torch.float32)
    probability_map = torch.full((1, 1, 64, 96), 0.5, dtype=torch.float32)
    validate_ppocr_detector_coreml_io(
        detector_input,
        probability_map,
        profile=profile,
    )

    recognizer_input = torch.zeros(2, 3, 48, 641, dtype=torch.float32)
    logits = torch.randn(
        2,
        ppocr_recognizer_timesteps(641),
        4,
        dtype=torch.float32,
    )
    probabilities = logits.softmax(dim=-1)
    validate_ppocr_recognizer_coreml_io(
        recognizer_input,
        probabilities,
        profile=profile,
        rec_num_classes=4,
    )
    wide_input = torch.zeros(1, 3, 48, 320, dtype=torch.float32)
    wide_probabilities = torch.full(
        (1, ppocr_recognizer_timesteps(320), 18_385),
        (1.0 + 1e-4) / 18_385,
        dtype=torch.float32,
    )
    validate_ppocr_recognizer_coreml_io(
        wide_input,
        wide_probabilities,
        profile=profile,
        rec_num_classes=18_385,
    )

    with pytest.raises(ValueError, match="stride-32"):
        validate_ppocr_detector_coreml_shape(33, 64, profile=profile)
    with pytest.raises(ValueError, match="stride-32"):
        validate_ppocr_detector_coreml_io(
            torch.zeros(1, 3, 65, 96),
            torch.zeros(1, 1, 65, 96),
            profile=profile,
        )
    with pytest.raises(ValueError, match=r"normalized to \[-1, 1\]"):
        validate_ppocr_recognizer_coreml_io(
            recognizer_input + 2.0,
            probabilities,
            profile=profile,
            rec_num_classes=4,
        )
    with pytest.raises(ValueError, match="overflow policy is 'error'"):
        validate_ppocr_recognizer_coreml_shape(1, 642, profile=profile)
    with pytest.raises(ValueError, match="sum to one"):
        validate_ppocr_recognizer_coreml_io(
            recognizer_input,
            torch.full_like(probabilities, 0.1),
            profile=profile,
            rec_num_classes=4,
        )


def test_native_recognition_bucket_overflow_is_rejected_not_rescaled():
    from libreyolo.export.coreml_ppocr import (
        ppocr_recognizer_required_width,
        validate_ppocr_recognizer_coreml_crop,
        validate_ppocr_recognizer_coreml_shape,
    )
    from libreyolo.models.ppocr.preprocess import rec_batches

    crop = np.zeros((10, 200, 3), dtype=np.uint8)
    assert ppocr_recognizer_required_width(*crop.shape[:2]) == 960
    with pytest.raises(ValueError, match="must not clamp or rescale"):
        validate_ppocr_recognizer_coreml_crop(
            *crop.shape[:2],
            profile=_profile(rec_max_width=641),
        )

    [(_, batch)] = rec_batches([crop], batch_size=6)
    assert batch.shape == (1, 3, 48, 960)

    with pytest.raises(ValueError, match="must not clamp or rescale"):
        validate_ppocr_recognizer_coreml_shape(
            int(batch.shape[0]),
            int(batch.shape[-1]),
            profile=_profile(rec_max_width=641),
        )


@pytest.mark.parametrize("size", ["t", "l"])
def test_real_t_and_l_component_traces_generalize_with_exact_parity(size):
    from libreyolo.export.coreml_ppocr import (
        validate_ppocr_detector_coreml_io,
        validate_ppocr_recognizer_coreml_io,
        wrap_ppocr_coreml_components,
    )
    from libreyolo.models.ppocr.model import LibrePPOCRModel

    torch.manual_seed(20260729)
    composite = LibrePPOCRModel(size=size, num_classes=7).eval()
    profile = _profile(size=size, rec_max_width=641)
    wrapped = wrap_ppocr_coreml_components(
        composite,
        profile=profile,
        rec_num_classes=7,
    )

    det_first = torch.linspace(-2.0, 2.0, 3 * 64 * 96).reshape(1, 3, 64, 96)
    det_second = torch.linspace(2.0, -2.0, 3 * 96 * 64).reshape(1, 3, 96, 64)
    det_trace = torch.jit.trace(
        wrapped["detector"],
        det_first,
        check_trace=True,
        check_inputs=[(det_second,)],
    )

    rec_first = torch.linspace(-1.0, 1.0, 3 * 48 * 320).reshape(1, 3, 48, 320)
    rec_second = torch.linspace(
        1.0,
        -1.0,
        2 * 3 * 48 * 641,
    ).reshape(2, 3, 48, 641)
    rec_trace = torch.jit.trace(
        wrapped["recognizer"],
        rec_first,
        check_trace=True,
        check_inputs=[(rec_second,)],
    )

    with torch.inference_mode():
        expected_det = wrapped["detector"](det_second)
        actual_det = det_trace(det_second)
        changed_det = wrapped["detector"](-det_second)
        expected_rec = wrapped["recognizer"](rec_second)
        actual_rec = rec_trace(rec_second)
        changed_rec = wrapped["recognizer"](-rec_second)
        before_boundary = rec_trace(torch.zeros(1, 3, 48, 324))
        after_boundary = rec_trace(torch.zeros(1, 3, 48, 325))
    torch.testing.assert_close(actual_det, expected_det, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_rec, expected_rec, rtol=0.0, atol=0.0)
    validate_ppocr_detector_coreml_io(
        det_second,
        actual_det,
        profile=profile,
    )
    validate_ppocr_recognizer_coreml_io(
        rec_second,
        actual_rec,
        profile=profile,
        rec_num_classes=7,
    )
    assert float((changed_det - expected_det).abs().max()) > 1e-8
    assert float((changed_rec - expected_rec).abs().max()) > 1e-8
    assert tuple(before_boundary.shape) == (1, 40, 7)
    assert tuple(after_boundary.shape) == (1, 41, 7)


def test_metadata_validator_does_not_mutate_caller_values():
    from libreyolo.export.coreml_ppocr import validate_ppocr_coreml_metadata

    metadata = _metadata()
    before = copy.deepcopy(metadata)
    validate_ppocr_coreml_metadata(metadata)
    assert metadata == before
