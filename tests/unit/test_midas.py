"""Hermetic tests for the MiDaS relative-depth family."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from libreyolo.models.autoconvert import autoconvert_upstream_checkpoint
from libreyolo.models.midas.convert import (
    UPSTREAM_URLS,
    verify_and_wrap_download,
    wrap_upstream_state_dict,
)
from libreyolo.models.midas.model import LibreMiDaS
from libreyolo.models.midas.utils import _resize_shape, preprocess_numpy
from libreyolo.utils.serialization import validate_checkpoint_metadata

pytestmark = [pytest.mark.unit, pytest.mark.midas]


def _midas_signature(size: str) -> dict[str, torch.Tensor]:
    common = {
        "scratch.refinenet1.resConfUnit1.conv1.weight": torch.zeros(1, 1, 1, 1),
        "scratch.output_conv.4.weight": torch.zeros(1, 1, 1, 1),
    }
    if size == "s":
        return {
            **common,
            "pretrained.layer1.0.weight": torch.zeros(32, 3, 3, 3),
            "pretrained.layer1.3.0.conv_dw.weight": torch.zeros(32, 1, 3, 3),
        }
    if size == "l":
        return {
            **common,
            "pretrained.model.cls_token": torch.zeros(1, 1, 1024),
        }
    raise ValueError(size)


def _sibling_depth_signatures() -> list[tuple[type, dict[str, torch.Tensor]]]:
    from libreyolo.models.depth_anything.model import LibreDepthAnythingV2
    from libreyolo.models.depth_anything3.model import LibreDepthAnything3
    from libreyolo.models.zipdepth.model import LibreZipDepth

    return [
        (
            LibreDepthAnythingV2,
            {
                "pretrained.cls_token": torch.zeros(1, 1, 384),
                "depth_head.scratch.output_conv2.0.weight": torch.zeros(1),
            },
        ),
        (
            LibreDepthAnything3,
            {
                "backbone.pretrained.cls_token": torch.zeros(1, 1, 1024),
                "head.scratch.output_conv2.0.weight": torch.zeros(1),
            },
        ),
        (
            LibreZipDepth,
            {
                "encoder.stem_half.conv.weight": torch.zeros(24, 3, 3, 3),
                "decoder.convex_up.weight": torch.zeros(1),
            },
        ),
    ]


def test_family_metadata_and_registration():
    from libreyolo.models.base import BaseModel

    assert LibreMiDaS.FAMILY == "midas"
    assert LibreMiDaS.FILENAME_PREFIX == "LibreMiDaS"
    assert LibreMiDaS.INPUT_SIZES == {"s": 256, "l": 384}
    assert LibreMiDaS.SUPPORTED_TASKS == ("depth",)
    assert LibreMiDaS.DEFAULT_TASK == "depth"
    assert LibreMiDaS.REQUIRE_TASK_SUFFIX is True
    assert LibreMiDaS in BaseModel._registry


@pytest.mark.parametrize("size", ["s", "l"])
def test_can_load_and_detect_size(size: str):
    state = _midas_signature(size)
    assert LibreMiDaS.can_load(state)
    assert LibreMiDaS.detect_size(state) == size
    assert LibreMiDaS.detect_nb_classes(state) == 1


def test_can_load_is_bidirectionally_exclusive_with_depth_siblings():
    for sibling, sibling_state in _sibling_depth_signatures():
        assert sibling.can_load(sibling_state)
        assert not LibreMiDaS.can_load(sibling_state)
        for size in ("s", "l"):
            midas_state = _midas_signature(size)
            assert LibreMiDaS.can_load(midas_state)
            assert not sibling.can_load(midas_state)


def test_filename_and_download_routing():
    assert LibreMiDaS.detect_size_from_filename("LibreMiDaSs-depth.pt") == "s"
    assert LibreMiDaS.detect_size_from_filename("LibreMiDaSl-depth.pt") == "l"
    assert LibreMiDaS.detect_task_from_filename("LibreMiDaSl-depth.pt") == "depth"
    assert LibreMiDaS.detect_size_from_filename("LibreMiDaSs.pt") is None
    assert LibreMiDaS.get_download_url("LibreMiDaSs-depth.pt") == UPSTREAM_URLS["s"]
    assert LibreMiDaS.get_download_url("LibreMiDaSl-depth.pt") == UPSTREAM_URLS["l"]
    assert LibreMiDaS.get_download_url("LibreMiDaSs.pt") is None


@pytest.mark.parametrize(
    "size,expected",
    [("s", (256, 192)), ("l", (512, 384))],
)
def test_resize_geometry_matches_official_rules(size: str, expected: tuple[int, int]):
    assert _resize_shape(640, 480, LibreMiDaS.INPUT_SIZES[size], size) == expected


@pytest.mark.parametrize("size", ["s", "l"])
def test_preprocess_is_rgb_float_and_multiple_of_32(size: str):
    image = np.random.default_rng(7).integers(
        0, 256, size=(321, 517, 3), dtype=np.uint8
    )
    chw, ratio = preprocess_numpy(image, LibreMiDaS.INPUT_SIZES[size], size)

    assert chw.dtype == np.float32
    assert chw.shape[0] == 3
    assert chw.shape[1] % 32 == 0
    assert chw.shape[2] % 32 == 0
    assert np.isfinite(chw).all()
    assert ratio == 1.0


def test_upstream_wrap_has_strict_depth_metadata():
    checkpoint = wrap_upstream_state_dict(_midas_signature("s"), "s")
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert checkpoint["model_family"] == "midas"
    assert checkpoint["size"] == "s"
    assert checkpoint["task"] == "depth"
    assert checkpoint["nc"] == 1
    assert checkpoint["names"] == {0: "depth"}
    assert checkpoint["imgsz"] == 256


def test_upstream_wrap_rejects_contradictory_size():
    with pytest.raises(ValueError, match="expected 'l'"):
        wrap_upstream_state_dict(_midas_signature("s"), "l")


def test_download_verification_rejects_tampering(tmp_path: Path):
    checkpoint = tmp_path / "download.pt.part"
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        verify_and_wrap_download(str(checkpoint), UPSTREAM_URLS["s"])


def test_raw_upstream_state_autoconverts_with_depth_name(tmp_path: Path):
    source = tmp_path / "midas_v21_small_256.pt"
    torch.save(_midas_signature("s"), source)

    converted = autoconvert_upstream_checkpoint(str(source))

    assert converted is not None
    assert Path(converted).name == ("midas_v21_small_256-LibreMiDaSs-depth.pt")
    checkpoint = torch.load(converted, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert checkpoint["model_family"] == "midas"
    assert checkpoint["names"] == {0: "depth"}


def test_small_forward_contract():
    pytest.importorskip("timm")
    from libreyolo.models.midas.nn import MiDaSSmall

    model = MiDaSSmall().eval()
    assert "pixel_mean" not in model.state_dict()
    assert "pixel_std" not in model.state_dict()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 64))
    assert output.shape == (1, 1, 64, 64)
    assert float(output.min()) >= 0.0


def test_bad_imgsz_and_training_fail_explicitly():
    model = LibreMiDaS.__new__(LibreMiDaS)
    with pytest.raises(ValueError, match="divisible by 32"):
        model._preprocess(np.zeros((64, 64, 3), dtype=np.uint8), input_size=100)
    with pytest.raises(NotImplementedError, match="not implemented"):
        model.train(data="depth.yaml")
