"""Real-Mac Core ML parity for hosted permissive size/task variants.

The baseline family/task profiles live in ``test_coreml_roundtrip``.  This
campaign deliberately covers the other released checkpoint shapes without
pretending that one small checkpoint proves every graph in a family.

Only exact, publicly hosted checkpoint names with permissive deployment terms
belong here.  Restricted/local-only weights, unsupported tasks, frozen-vocabulary
specialists, DEIMv2's DINOv3-backed sizes, and already-covered baseline profiles
are intentionally excluded.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytestmark = [
    pytest.mark.coreml,
    pytest.mark.coreml_variant,
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
]

if sys.platform != "darwin":  # pragma: no cover - platform gate
    pytest.skip("Core ML artifacts only run on macOS", allow_module_level=True)

pytest.importorskip(
    "coremltools",
    reason="Core ML variant parity requires the coremltools runtime",
)

from tests.e2e.test_coreml_roundtrip import (  # noqa: E402
    TRAINED_CASES,
    _assert_flagship_public_detection_path,
    _assert_generic_public_path,
    _assert_model_artifact_parity,
    _assert_public_semantic_path,
    _assert_rtmdet_segment_public_path,
)


@dataclass(frozen=True, slots=True)
class HostedVariantProfile:
    """One exact released graph profile to validate on Apple hardware."""

    weights: str
    family: str
    task: str
    size: str
    imgsz: int

    @property
    def node_id(self) -> str:
        return f"{self.family}-{self.task}-{self.size}-img{self.imgsz}"

    @property
    def baseline_key(self) -> tuple[str, str, str, int]:
        return self.weights, self.family, self.task, self.imgsz


def _profile(
    weights: str,
    family: str,
    task: str,
    size: str,
    imgsz: int,
) -> HostedVariantProfile:
    return HostedVariantProfile(weights, family, task, size, imgsz)


# Keep rows grouped by family so a failed pytest node maps directly back to one
# released size/task graph.  Canvases come from the family INPUT_SIZES or
# TASK_INPUT_SIZES registry and are also asserted against the loaded checkpoint.
HOSTED_VARIANT_PROFILES = (
    # YOLO detectors.
    _profile("LibreYOLOXt.pt", "yolox", "detect", "t", 416),
    _profile("LibreYOLOXs.pt", "yolox", "detect", "s", 640),
    _profile("LibreYOLOXm.pt", "yolox", "detect", "m", 640),
    _profile("LibreYOLOXl.pt", "yolox", "detect", "l", 640),
    _profile("LibreYOLOXx.pt", "yolox", "detect", "x", 640),
    _profile("LibreYOLO9s.pt", "yolo9", "detect", "s", 640),
    _profile("LibreYOLO9m.pt", "yolo9", "detect", "m", 640),
    _profile("LibreYOLO9c.pt", "yolo9", "detect", "c", 640),
    _profile("LibreYOLO9E2Es.pt", "yolo9_e2e", "detect", "s", 640),
    _profile("LibreYOLO9E2Em.pt", "yolo9_e2e", "detect", "m", 640),
    _profile("LibreYOLO9E2Ec.pt", "yolo9_e2e", "detect", "c", 640),
    # The YOLOv1 tiny source weights no longer exist; only genuinely hosted
    # public-domain Darknet variants are listed.
    _profile("LibreYOLO2t.pt", "yolo2", "detect", "t", 416),
    _profile("LibreYOLO3t.pt", "yolo3", "detect", "t", 416),
    _profile("LibreYOLO3spp.pt", "yolo3", "detect", "spp", 608),
    _profile("LibreYOLO4t.pt", "yolo4", "detect", "t", 416),
    # D-FINE detect and instance-segmentation releases.
    _profile("LibreDFINEs.pt", "dfine", "detect", "s", 640),
    _profile("LibreDFINEm.pt", "dfine", "detect", "m", 640),
    _profile("LibreDFINEl.pt", "dfine", "detect", "l", 640),
    _profile("LibreDFINEx.pt", "dfine", "detect", "x", 640),
    _profile("LibreDFINEs-seg.pt", "dfine", "segment", "s", 640),
    _profile("LibreDFINEm-seg.pt", "dfine", "segment", "m", 640),
    _profile("LibreDFINEl-seg.pt", "dfine", "segment", "l", 640),
    _profile("LibreDFINEx-seg.pt", "dfine", "segment", "x", 640),
    # DEIM and only the HGNet-backed, permissive DEIMv2 configurations.
    _profile("LibreDEIMs.pt", "deim", "detect", "s", 640),
    _profile("LibreDEIMm.pt", "deim", "detect", "m", 640),
    _profile("LibreDEIMl.pt", "deim", "detect", "l", 640),
    _profile("LibreDEIMx.pt", "deim", "detect", "x", 640),
    _profile("LibreDEIMv2femto.pt", "deimv2", "detect", "femto", 416),
    _profile("LibreDEIMv2pico.pt", "deimv2", "detect", "pico", 640),
    _profile("LibreDEIMv2n.pt", "deimv2", "detect", "n", 640),
    # EdgeCrafter's three task-specific heads.
    _profile("LibreECm.pt", "ec", "detect", "m", 640),
    _profile("LibreECl.pt", "ec", "detect", "l", 640),
    _profile("LibreECx.pt", "ec", "detect", "x", 640),
    _profile("LibreECm-pose.pt", "ec", "pose", "m", 640),
    _profile("LibreECl-pose.pt", "ec", "pose", "l", 640),
    _profile("LibreECx-pose.pt", "ec", "pose", "x", 640),
    _profile("LibreECm-seg.pt", "ec", "segment", "m", 640),
    _profile("LibreECl-seg.pt", "ec", "segment", "l", 640),
    _profile("LibreECx-seg.pt", "ec", "segment", "x", 640),
    # Dense one-stage detector sizes have distinct candidate grids.
    _profile("LibrePICODETm.pt", "picodet", "detect", "m", 416),
    _profile("LibrePICODETl.pt", "picodet", "detect", "l", 640),
    # RT-DETR lineages.
    _profile("LibreRTDETRr34.pt", "rtdetr", "detect", "r34", 640),
    _profile("LibreRTDETRr50.pt", "rtdetr", "detect", "r50", 640),
    _profile("LibreRTDETRr50m.pt", "rtdetr", "detect", "r50m", 640),
    _profile("LibreRTDETRr101.pt", "rtdetr", "detect", "r101", 640),
    _profile("LibreRTDETRl.pt", "rtdetr", "detect", "l", 640),
    _profile("LibreRTDETRx.pt", "rtdetr", "detect", "x", 640),
    _profile("LibreRTDETRv2r34.pt", "rtdetrv2", "detect", "r34", 640),
    _profile("LibreRTDETRv2r50.pt", "rtdetrv2", "detect", "r50", 640),
    _profile("LibreRTDETRv2r50m.pt", "rtdetrv2", "detect", "r50m", 640),
    _profile("LibreRTDETRv2r101.pt", "rtdetrv2", "detect", "r101", 640),
    _profile("LibreRTDETRv4m.pt", "rtdetrv4", "detect", "m", 640),
    _profile("LibreRTDETRv4l.pt", "rtdetrv4", "detect", "l", 640),
    _profile("LibreRTDETRv4x.pt", "rtdetrv4", "detect", "x", 640),
    # RTMDet and RTMDet-Ins.
    _profile("LibreRTMDets.pt", "rtmdet", "detect", "s", 640),
    _profile("LibreRTMDetm.pt", "rtmdet", "detect", "m", 640),
    _profile("LibreRTMDetl.pt", "rtmdet", "detect", "l", 640),
    _profile("LibreRTMDetx.pt", "rtmdet", "detect", "x", 640),
    _profile("LibreRTMDets-seg.pt", "rtmdet", "segment", "s", 640),
    _profile("LibreRTMDetm-seg.pt", "rtmdet", "segment", "m", 640),
    _profile("LibreRTMDetl-seg.pt", "rtmdet", "segment", "l", 640),
    _profile("LibreRTMDetx-seg.pt", "rtmdet", "segment", "x", 640),
    # RF-DETR segmentation is intentionally absent: its M4 runtime parity gate
    # failed. Pose has only the already-covered x registry profile.
    _profile("LibreRFDETRs.pt", "rfdetr", "detect", "s", 512),
    _profile("LibreRFDETRm.pt", "rfdetr", "detect", "m", 576),
    _profile("LibreRFDETRl.pt", "rfdetr", "detect", "l", 704),
    _profile("LibreRFDETRs-obb.pt", "rfdetr", "obb", "s", 512),
    _profile("LibreRFDETRm-obb.pt", "rfdetr", "obb", "m", 576),
    _profile("LibreRFDETRl-obb.pt", "rfdetr", "obb", "l", 704),
    # Permissively hosted semantic checkpoints.
    _profile("LibrePIDNetm-sem.pt", "pidnet", "semantic", "m", 1024),
    _profile("LibrePIDNetl-sem.pt", "pidnet", "semantic", "l", 1024),
    _profile(
        "LibreLingBotVisionb-sem.pt",
        "lingbotvision",
        "semantic",
        "b",
        512,
    ),
    _profile(
        "LibreLingBotVisionl-sem.pt",
        "lingbotvision",
        "semantic",
        "l",
        512,
    ),
    # Fixed-label ImageNet classifiers. Open-vocabulary CLIP/SigLIP variants
    # use separate frozen-vocabulary contracts and do not belong in this file.
    _profile("LibreResNet34-cls.pt", "resnet", "classify", "34", 224),
    _profile("LibreResNet50-cls.pt", "resnet", "classify", "50", 224),
    _profile("LibreResNet101-cls.pt", "resnet", "classify", "101", 224),
    _profile(
        "LibreMobileNetV4m-cls.pt",
        "mobilenetv4",
        "classify",
        "m",
        224,
    ),
    _profile(
        "LibreMobileNetV4l-cls.pt",
        "mobilenetv4",
        "classify",
        "l",
        256,
    ),
    _profile(
        "LibreEfficientNetV2b1-cls.pt",
        "efficientnetv2",
        "classify",
        "b1",
        240,
    ),
    _profile(
        "LibreEfficientNetV2b2-cls.pt",
        "efficientnetv2",
        "classify",
        "b2",
        260,
    ),
    _profile(
        "LibreEfficientNetV2b3-cls.pt",
        "efficientnetv2",
        "classify",
        "b3",
        300,
    ),
    _profile("LibreConvNeXts-cls.pt", "convnext", "classify", "s", 224),
    _profile("LibreConvNeXtb-cls.pt", "convnext", "classify", "b", 224),
    # Depth and restoration variants with distinct exported implementations.
    _profile("LibreZipDepthbnpu-depth.pt", "zipdepth", "depth", "bnpu", 384),
    _profile(
        "LibreRealESRGANx4-restore.pt",
        "realesrgan",
        "restore",
        "x4",
        64,
    ),
    _profile(
        "LibreRealESRGANx2-restore.pt",
        "realesrgan",
        "restore",
        "x2",
        64,
    ),
)


def _assert_exact_hosted_profile(model, profile: HostedVariantProfile) -> None:
    """Fail before conversion if the checkpoint disagrees with its registry row."""
    assert profile.baseline_key not in TRAINED_CASES
    assert model.FAMILY == profile.family
    assert model.task == profile.task
    assert model.size == profile.size
    assert int(model.input_size) == profile.imgsz

    download_url = type(model).get_download_url(profile.weights)
    assert download_url is not None
    path = urlsplit(download_url).path
    expected_suffix = (
        f"/{Path(profile.weights).stem}/resolve/main/{profile.weights}"
    )
    assert path.endswith(expected_suffix), (
        f"{profile.weights} did not resolve to its exact hosted artifact: "
        f"{download_url}"
    )


def _assert_public_profile(
    model,
    artifact,
    profile: HostedVariantProfile,
) -> None:
    """Dispatch variants through the strongest applicable public parity gate."""
    if profile.task == "detect" and profile.family in {"rfdetr", "yolo9"}:
        _assert_flagship_public_detection_path(
            model,
            artifact,
            family=profile.family,
            imgsz=profile.imgsz,
            compute_units="cpu_only",
        )
    elif (profile.family, profile.task) == ("rtmdet", "segment"):
        _assert_rtmdet_segment_public_path(
            model,
            artifact,
            compute_units="cpu_only",
        )
    elif (profile.family, profile.task) == ("pidnet", "semantic"):
        _assert_public_semantic_path(
            model,
            artifact,
            imgsz=profile.imgsz,
            minimum_agreement=0.995,
            require_multiple_classes=True,
            compute_units="cpu_only",
        )
    else:
        _assert_generic_public_path(
            model,
            artifact,
            family=profile.family,
            task=profile.task,
            imgsz=profile.imgsz,
            compute_units="cpu_only",
        )


@pytest.mark.parametrize(
    "profile",
    HOSTED_VARIANT_PROFILES,
    ids=lambda profile: profile.node_id,
)
def test_coreml_hosted_variant_raw_and_public_parity(
    profile: HostedVariantProfile,
    tmp_path,
):
    """Prove one exact hosted checkpoint through raw and public Core ML paths."""
    from libreyolo import LibreYOLO

    model = LibreYOLO(profile.weights, device="cpu")
    _assert_exact_hosted_profile(model, profile)
    artifact = _assert_model_artifact_parity(
        model,
        profile.family,
        profile.task,
        profile.imgsz,
        tmp_path,
        half=False,
    )

    # Export preparation is allowed to mutate modules.  Never use that model as
    # the public PyTorch oracle; reload the exact checkpoint into a pristine
    # instance before comparing user-facing preprocessing/postprocessing.
    del model
    gc.collect()
    pristine = LibreYOLO(profile.weights, device="cpu")
    _assert_exact_hosted_profile(pristine, profile)
    _assert_public_profile(pristine, artifact, profile)
    del pristine
    gc.collect()
