"""Machine-readable Core ML execution-profile candidates and promotions.

The family/task support matrix deliberately answers a broad question:
"is conversion supported for this model family?"  This module answers the
narrower deployment question that matters at runtime: which exact
family/task/size/canvas/precision/NMS/graph-specialization/compute-unit
combination is eligible for a saved-package and public-API campaign. A
candidate becomes a validated profile only when it also carries a complete
source identity, final deployment-ABI identity, and evidence digest.

Profiles describe conversion fidelity, not model accuracy.  A custom
checkpoint with the same graph profile still needs its own accuracy
validation.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping

from .coreml_identity import (
    COREML_DEPLOYMENT_ABI_SCHEMA,
    require_lower_sha256,
)

COREML_EXECUTION_PROFILE_VERSION = "2"
COREML_VALIDATED_COMPUTE_UNITS = "validated"
COREML_NATIVE_COMPUTE_UNITS = frozenset(
    {"all", "cpu_and_gpu", "cpu_and_ne", "cpu_only"}
)
COREML_COMPUTE_UNITS = COREML_NATIVE_COMPUTE_UNITS | {
    COREML_VALIDATED_COMPUTE_UNITS
}


@dataclass(frozen=True)
class CoreMLExecutionProfile:
    """One exact conversion/runtime profile with real-device evidence."""

    family: str
    task: str
    size: str
    canvas: tuple[int, int]
    reference: str
    source_kind: str | None = None
    source_sha256: str | None = None
    deployment_abi_sha256: str | None = None
    evidence_sha256: str | None = None
    runtime_compute_units: tuple[str, ...] = ("cpu_only",)
    precision: str = "fp32"
    conversion_compute_units: str = "cpu_only"
    nms: bool = False
    conversion_pass_profile: str | None = None
    disabled_passes: tuple[str, ...] = ()
    capture_decomposition_profile: str | None = None
    capture_decompositions: tuple[str, ...] = ()
    prompt_max_points: int | None = None
    class_count: int | None = None
    graph_class_width: int | None = None
    num_keypoints: int | None = None
    keypoint_dim: int | None = None
    num_keypoints_per_class: tuple[int, ...] = ()
    classification_activation: str | None = None
    checkpoint_variant: str | None = None
    architecture_signature: str | None = None
    restore_scale: int | None = None
    embedding_dim: int | None = None

    @property
    def evidence_complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.source_kind,
                self.source_sha256,
                self.deployment_abi_sha256,
                self.evidence_sha256,
            )
        )

    @property
    def profile_id(self) -> str:
        height, width = self.canvas
        boundary = "nms" if self.nms else "raw"
        specializations = []
        if self.prompt_max_points is not None:
            specializations.append(f"pmax{self.prompt_max_points}")
        if self.class_count is not None:
            specializations.append(f"classes{self.class_count}")
        if self.graph_class_width is not None:
            specializations.append(f"graphclasses{self.graph_class_width}")
        if self.num_keypoints is not None:
            specializations.append(f"k{self.num_keypoints}")
        if self.keypoint_dim is not None:
            specializations.append(f"d{self.keypoint_dim}")
        if self.num_keypoints_per_class:
            schema = ".".join(
                str(value) for value in self.num_keypoints_per_class
            )
            specializations.append(f"kpc{schema}")
        if self.classification_activation is not None:
            specializations.append(f"act{self.classification_activation}")
        if self.checkpoint_variant is not None:
            specializations.append(f"variant{self.checkpoint_variant}")
        if self.architecture_signature is not None:
            specializations.append(f"arch{self.architecture_signature}")
        if self.restore_scale is not None:
            specializations.append(f"scale{self.restore_scale}")
        if self.embedding_dim is not None:
            specializations.append(f"embed{self.embedding_dim}")
        if self.evidence_complete:
            specializations.extend(
                (
                    f"src{self.source_sha256[:12]}",
                    f"abi{self.deployment_abi_sha256[:12]}",
                )
            )
        specialization = (
            f"/{'-'.join(specializations)}" if specializations else ""
        )
        return (
            f"coreml-m4-v{COREML_EXECUTION_PROFILE_VERSION}/"
            f"{self.family}/{self.task}/{self.size}/"
            f"{height}x{width}{specialization}/{boundary}-{self.precision}-"
            f"{self.conversion_compute_units}"
        )

    @property
    def default_runtime_compute_units(self) -> str:
        return self.runtime_compute_units[0]


def _profile(
    family: str,
    task: str,
    size: str,
    canvas: int | tuple[int, int],
    reference: str,
    *,
    source_kind: str | None = None,
    source_sha256: str | None = None,
    deployment_abi_sha256: str | None = None,
    evidence_sha256: str | None = None,
    runtime_all: bool = False,
    conversion_pass_profile: str | None = None,
    disabled_passes: tuple[str, ...] = (),
    capture_decomposition_profile: str | None = None,
    capture_decompositions: tuple[str, ...] = (),
    prompt_max_points: int | None = None,
    class_count: int | None = None,
    graph_class_width: int | None = None,
    num_keypoints: int | None = None,
    keypoint_dim: int | None = None,
    num_keypoints_per_class: tuple[int, ...] = (),
    classification_activation: str | None = None,
    checkpoint_variant: str | None = None,
    architecture_signature: str | None = None,
    restore_scale: int | None = None,
    embedding_dim: int | None = None,
) -> CoreMLExecutionProfile:
    canvas_hw = (canvas, canvas) if isinstance(canvas, int) else canvas
    runtime_units = (
        ("cpu_only", "all") if runtime_all else ("cpu_only",)
    )
    return CoreMLExecutionProfile(
        family=family,
        task=task,
        size=size,
        canvas=canvas_hw,
        reference=reference,
        source_kind=source_kind,
        source_sha256=source_sha256,
        deployment_abi_sha256=deployment_abi_sha256,
        evidence_sha256=evidence_sha256,
        runtime_compute_units=runtime_units,
        conversion_pass_profile=conversion_pass_profile,
        disabled_passes=disabled_passes,
        capture_decomposition_profile=capture_decomposition_profile,
        capture_decompositions=capture_decompositions,
        prompt_max_points=prompt_max_points,
        class_count=class_count,
        graph_class_width=graph_class_width,
        num_keypoints=num_keypoints,
        keypoint_dim=keypoint_dim,
        num_keypoints_per_class=num_keypoints_per_class,
        classification_activation=classification_activation,
        checkpoint_variant=checkpoint_variant,
        architecture_signature=architecture_signature,
        restore_scale=restore_scale,
        embedding_dim=embedding_dim,
    )


# Keep this list checkpoint-explicit.  A family-level checkmark must never make
# another size, canvas, precision, embedded-NMS graph, or compute planner look
# hardware-validated by implication.
_PROFILES = (
    _profile(
        "birefnet",
        "matte",
        "l",
        1024,
        "LibreBiRefNetl-matte",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "3ab3ef80216176a850b0c47877f310567c3749ec73e1becbfdacd9a2c13a7b39"
        ),
        deployment_abi_sha256=(
            "07edd1cea201e3119f4100f678a218d046f1ecc6d4a7e925e3b3ba448334a891"
        ),
        evidence_sha256=(
            "17e0912e9a3510f0484b76da41dd1f669b02b314966b59fd5cc992c4acb40c53"
        ),
        runtime_all=True,
        class_count=1,
    ),
    _profile(
        "clip",
        "classify",
        "b32",
        224,
        "LibreCLIPb32-cls",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "a1085d1140b26f66471d170d651f3ae"
            "cea2c600dc1cb313cac0cde209f68344e"
        ),
        deployment_abi_sha256=(
            "933de7407ad78a5f80707ac16317b034"
            "c02144f465dbd9a9b1f914fe3728ff10"
        ),
        evidence_sha256=(
            "bfa0a2a4f01daaf2385202331d51661d"
            "b27c759f37f4b3d106de9b41400df9eb"
        ),
        runtime_all=True,
        class_count=3,
        classification_activation="softmax",
    ),
    _profile(
        "convnext",
        "classify",
        "t",
        224,
        "LibreConvNeXtt-cls",
        runtime_all=True,
        class_count=1000,
        classification_activation="softmax",
    ),
    _profile(
        "deim",
        "detect",
        "n",
        640,
        "LibreDEIMn",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "02763b1b4f851f9aa1394eef73beaa63d6b6ce981dfcdd84332d8b1d1b1d473d"
        ),
        deployment_abi_sha256=(
            "5abfd4ab0f2cb5afbea7e51d96f98557425419b56a6a38caf96dde642546d4ac"
        ),
        evidence_sha256=(
            "60a7766cf9f980553eabf22e5c74ab3ab8cbba1f1067db1948a354a1b3fb220c"
        ),
        class_count=80,
    ),
    _profile(
        "deimv2",
        "detect",
        "atto",
        320,
        "LibreDEIMv2atto",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "0fe4ff677dffdc097c5770f2b8f74abd"
            "51d5190e588714bee8a842745cdc16c0"
        ),
        deployment_abi_sha256=(
            "ff7ba4d823e570417e6700da61bf62c8"
            "033ef089e73f6899902ee29d8d812e39"
        ),
        evidence_sha256=(
            "5e647a5197b48b91b309dba93a49f641"
            "c4341e47f3fec00439573bf05c2f7ce4"
        ),
        class_count=80,
    ),
    _profile(
        "depth_anything",
        "depth",
        "s",
        518,
        "LibreDepthAnythingV2s-depth",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "e21eb6c30a03ff2ae8eb76d9b02127ee"
            "bcd51844523cccbf944202cdf7827b24"
        ),
        deployment_abi_sha256=(
            "f49adf13b613f2ae16147e97b4b3e06b"
            "969beeb97046705dc68d649c575a0e7f"
        ),
        evidence_sha256=(
            "bb2de57b7d238577f70cd47cb673eed7"
            "3416466143a0fba5dd15e6ce24e0b683"
        ),
        runtime_all=True,
        class_count=1,
    ),
    _profile(
        "dfine",
        "detect",
        "n",
        640,
        "LibreDFINEn",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "c63e30ed3e0869c257352f0c250ebc8a1cde7e907ac8423935de18d3d66fb2b1"
        ),
        deployment_abi_sha256=(
            "35563ed7a1bedf120b35e4729707f6772a6bd0dbfb6b514b90cf609d843088de"
        ),
        evidence_sha256=(
            "14302a347caa8972fc1a3a6f0485dac2948fb34e50e961f982164e981d845594"
        ),
        class_count=80,
    ),
    _profile(
        "dfine",
        "segment",
        "n",
        640,
        "LibreDFINEn-seg",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "e7024fb6b5bc13ff5275feafd3389e08"
            "bbd79ed6e89625fb6c3e5860fbedc587"
        ),
        deployment_abi_sha256=(
            "88ad60b2eaf06b5ce8b18fedc68d36ca"
            "ebcab48021420d9c56c9a1ee8e4cb53b"
        ),
        evidence_sha256=(
            "fc83f51f842227e0592854f017e5c3e4"
            "85f954155f29bfa6606c86a466b9c07b"
        ),
        class_count=80,
    ),
    _profile(
        "ec",
        "detect",
        "s",
        640,
        "LibreECs",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "11f8fc0626db2932ce15498d98e9fd4b"
            "c9df6b33963626794033df17c4cc1395"
        ),
        deployment_abi_sha256=(
            "4e4b5cf6892563aa2a47500308adca5b"
            "297bb15c1d501a2bc506b2883dec85c2"
        ),
        evidence_sha256=(
            "c8247d0bf50c37d89e9e48161ac3799d"
            "fb001ff25ee9fb7a7e890ca4c7989875"
        ),
        class_count=80,
    ),
    _profile(
        "ec",
        "pose",
        "s",
        640,
        "LibreECs-pose",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "5007ed66d1e78355d764961e7d7888c4"
            "130b04d8387a8e92e418efb27f9f7c77"
        ),
        deployment_abi_sha256=(
            "f918c571e8c889fefbae2dcae1f6af5d"
            "c911bf0c9fadaf35399038f4331c1a88"
        ),
        evidence_sha256=(
            "fc368fbf6d8bcfcc8947dd1bfa7ce56f"
            "0935b6f37d6ec04e0b9efa624233eac7"
        ),
        class_count=1,
        graph_class_width=2,
        num_keypoints=17,
        keypoint_dim=2,
    ),
    _profile(
        "ec",
        "segment",
        "s",
        640,
        "LibreECs-seg",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "2fe10209b8ff7f0e38048ade2c0ee108"
            "543cbc9b3fb3aad087d19d4709a3a9c2"
        ),
        deployment_abi_sha256=(
            "98bced6dc743f34edd581bf34f421d6d"
            "084cf11ff044158903c90e7e039212b7"
        ),
        evidence_sha256=(
            "c177aa10c1526bb23613b1e820f418a6"
            "bd37844061829814f49d9da596fd3683"
        ),
        class_count=80,
    ),
    _profile(
        "edgetam",
        "segment",
        "edge",
        1024,
        "LibreEdgeTAM edge",
        source_kind="pytorch-captured-bundle-state-v1",
        source_sha256=(
            "4848ef1304b9f0853d3541b5410ba309"
            "8b70d687402a27cc8eba7ad5bf527700"
        ),
        deployment_abi_sha256=(
            "8e63255331b255388b9e4d98bdc6a8e31"
            "8be6aaf4956198ac12131174ae09f81"
        ),
        evidence_sha256=(
            "ec9f6fa7b71a3b68e5bbc9734fd939ec"
            "a855af6131aba9193d65e658ca06b15e"
        ),
        capture_decomposition_profile="edgetam_where_scalarother_v1",
        capture_decompositions=("aten.where.ScalarOther",),
        prompt_max_points=4,
        class_count=1,
    ),
    _profile(
        "efficientnetv2",
        "classify",
        "b0",
        224,
        "LibreEfficientNetV2b0-cls",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "f2931b84e0ab0829b86deb3656d0f78e"
            "34cefe5d3095b780af6f07c3d0967c5d"
        ),
        deployment_abi_sha256=(
            "4cb9d2d39570ffc3d351222327c76ee2"
            "83a51f05ff748749d0e62fde72d2f5a1"
        ),
        evidence_sha256=(
            "71d3a87a15d55b031835b952f7ed44ab"
            "130c3afbde5f03be769fdcffd4e2d327"
        ),
        runtime_all=True,
        class_count=1000,
        classification_activation="softmax",
    ),
    _profile(
        "facerec",
        "embed",
        "l",
        112,
        "librefacerec-l",
        class_count=1,
        embedding_dim=512,
    ),
    _profile(
        "fomo",
        "point",
        "s",
        96,
        "deterministic FOMO-S trained fixture",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "430622c3a4231b49a20881bb2137c716"
            "d043cf4812e5525876b3700309852972"
        ),
        deployment_abi_sha256=(
            "e19cbaa5f80e832bf62d0c37a992c554"
            "9faffc5636de761d909f095e5ed73767"
        ),
        evidence_sha256=(
            "344bab22af46d09cdb2ba7de80df438e"
            "a0085c761f002cd4fc21ccb57e7b3a16"
        ),
        runtime_all=True,
        class_count=2,
    ),
    _profile(
        "lingbotvision",
        "semantic",
        "s",
        512,
        "LibreLingBotVisions-sem",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "ee09c9a85ca8a109962ccebf0ec23e5e"
            "9402f107aaec1ce2a25e36c26682bbae"
        ),
        deployment_abi_sha256=(
            "b1a83e6d896013ad2c6927810daee61a"
            "96a0f069d994424f17172d7f1c6272bd"
        ),
        evidence_sha256=(
            "74c063be117f7f01216baaa313fbc0f9"
            "29506c2f8ce9fb6e783e2138e320510b"
        ),
        class_count=150,
    ),
    _profile(
        "mobilenetv4",
        "classify",
        "s",
        224,
        "LibreMobileNetV4s-cls",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "622e5eb795b55a45f14acf54e3842ae9"
            "b9f9bacd39111653c3032569701c9543"
        ),
        deployment_abi_sha256=(
            "14e18285f17fcccfeaba493019df1bdf"
            "a6593f49a2165cb0b78430f57d3a783b"
        ),
        evidence_sha256=(
            "41664db4935547b2b2a253f854049b64"
            "9a94a2acf99603aa61220d83b4cf03cf"
        ),
        runtime_all=True,
        class_count=1000,
        classification_activation="softmax",
    ),
    *(
        _profile(
            "mobilesam",
            "segment",
            "tiny",
            1024,
            "LibreMobileSAM tiny",
            source_kind=(
                "pytorch-captured-bundle-state-v1"
                if prompt_max_points == 4
                else None
            ),
            source_sha256=(
                "4d4c7f724325c515afb96c1dd6dafce9"
                "99c7c1a567e8606cd4e054dbad10b48f"
                if prompt_max_points == 4
                else None
            ),
            deployment_abi_sha256=(
                "4735fa16ee0e76579b5cfc7601cee9cd5"
                "7961e55cf0c47b5a9391fc6ea881888"
                if prompt_max_points == 4
                else None
            ),
            evidence_sha256=(
                "7c7bc483c72cc8c9c2963e16bcd4aad3"
                "aefc2e14afa8442d3f765f63efd133d6"
                if prompt_max_points == 4
                else None
            ),
            prompt_max_points=prompt_max_points,
            class_count=1,
        )
        for prompt_max_points in (1, 4, 16)
    ),
    _profile(
        "nafnet",
        "restore",
        "l",
        256,
        "LibreNAFNetl-restore-sidd",
        runtime_all=True,
        conversion_pass_profile="nafnet_preserve_elementwise_affine_v1",
        disabled_passes=("common::fuse_elementwise_to_batchnorm",),
        class_count=1,
        checkpoint_variant="sidd",
        architecture_signature="w64-m12-e2.2.4.8-d2.2.2.2",
        restore_scale=1,
    ),
    _profile(
        "omdet_turbo",
        "detect",
        "t",
        640,
        "LibreOMDetTurbot",
        runtime_all=True,
        class_count=3,
    ),
    _profile(
        "owlv2",
        "detect",
        "b16",
        960,
        "LibreOWLv2b16",
        runtime_all=True,
        class_count=3,
    ),
    _profile(
        "picodet",
        "detect",
        "s",
        320,
        "LibrePICODETs",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "0612c3fad35a62964e83501490eaf2b320eb5a9f6c8e3200fddb41809e0a84e8"
        ),
        deployment_abi_sha256=(
            "f70c6619a45a3e939b41f47136866f9af16258e2fda5d24588f0719876ab0c63"
        ),
        evidence_sha256=(
            "ab6d3d0a43e1df7d5e990ced1b6a8f13f1fe464a915b4de1cee22a2c6798fee2"
        ),
        class_count=80,
    ),
    _profile(
        "picosam3",
        "segment",
        "pico",
        96,
        "LibrePicoSAM3pico",
        runtime_all=True,
        class_count=1,
    ),
    _profile(
        "pidnet",
        "semantic",
        "s",
        1024,
        "LibrePIDNets-sem",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "58deae12e213f88b95c36635f71a1520"
            "eff83a4e4a5ff879215ed454e7334eb1"
        ),
        deployment_abi_sha256=(
            "de260e60498c6dd6c42d861a415e3d99"
            "06cd7c2c06a3974fd4e0e909d9862e8f"
        ),
        evidence_sha256=(
            "ac4016c03273dc2428b3236aec8f54ff"
            "e3bda38cf52a93acae70f954ee2e2764"
        ),
        class_count=19,
    ),
    _profile(
        "realesrgan",
        "restore",
        "x4t",
        64,
        "LibreRealESRGANx4t-restore",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "84fab39c99d8c4c7d26b67e90d99e2df"
            "d45a46bd0a86ce351088871b37dced9a"
        ),
        deployment_abi_sha256=(
            "e60098998ba93e59a480dbb1706ee61f"
            "aad09b5663f189edc7b38c4be57f276a"
        ),
        evidence_sha256=(
            "b8e3876fec3eff77a1d2b8ee071263be"
            "a06c919fc58f1024bc4c7681ab5e5fef"
        ),
        runtime_all=True,
        class_count=1,
        architecture_signature="srvgg",
        restore_scale=4,
    ),
    _profile(
        "resnet",
        "classify",
        "18",
        224,
        "LibreResNet18-cls",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "8b485a63bdd7be3787d7862cd54693e8"
            "c86476841ef3cf4a77351bac37d2d4f3"
        ),
        deployment_abi_sha256=(
            "1c7e5726e27f280c6afad5b82dc76f1c"
            "7e05e5fa7d09a12f0e38a4506b743e56"
        ),
        evidence_sha256=(
            "bf22dde43add454315903ee93544c3f0"
            "19da6c03ce11f232bcd556e88d503ace"
        ),
        runtime_all=True,
        class_count=1000,
        classification_activation="softmax",
    ),
    _profile(
        "sam",
        "segment",
        "base",
        1024,
        "LibreSAM base",
        prompt_max_points=4,
        class_count=1,
    ),
    _profile(
        "rfdetr",
        "detect",
        "n",
        384,
        "LibreRFDETRn",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "a1340c15e30ff2ca8df3494c8732e254cad00847c90c28532809a755455d09cb"
        ),
        deployment_abi_sha256=(
            "060b75661d63fc4974db049c80eaa1e9bedbc66159f824e57493d59db620f2e6"
        ),
        evidence_sha256=(
            "f6bee6027f5c232aa2f674a99d6a00119120987ff543a913555b91755812470f"
        ),
        runtime_all=True,
        class_count=80,
    ),
    _profile(
        "rfdetr",
        "detect",
        "s",
        512,
        "LibreRFDETRs",
        class_count=80,
    ),
    _profile(
        "rfdetr",
        "detect",
        "m",
        576,
        "LibreRFDETRm",
        class_count=80,
    ),
    _profile(
        "rfdetr",
        "detect",
        "l",
        704,
        "LibreRFDETRl",
        class_count=80,
    ),
    _profile(
        "rfdetr",
        "obb",
        "n",
        384,
        "LibreRFDETRn-obb",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "c572b9a78049f890ba3fe466931d51a1bae2802669ccc67f6e321cb8449efd2d"
        ),
        deployment_abi_sha256=(
            "7825a8120f696d5b616302c7bbfaf87aea9e30ee548d12cf4cde7a9cdc661b42"
        ),
        evidence_sha256=(
            "6bc45f5aa9aa56746230f78b5ad39223d4922da7a36d857269b83ffd6b67bf72"
        ),
        class_count=6,
    ),
    _profile(
        "rfdetr",
        "obb",
        "s",
        512,
        "LibreRFDETRs-obb",
        class_count=80,
    ),
    _profile(
        "rfdetr",
        "obb",
        "m",
        576,
        "LibreRFDETRm-obb",
        class_count=80,
    ),
    _profile(
        "rfdetr",
        "obb",
        "l",
        704,
        "LibreRFDETRl-obb",
        class_count=80,
    ),
    _profile(
        "rfdetr",
        "pose",
        "x",
        576,
        "LibreRFDETRx-pose",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "99eba2e96ae41d7e39d634b29675b0885da25a214fcc7d2e48ec87f96aea887e"
        ),
        deployment_abi_sha256=(
            "622e264da967248fa98cb55aea6c3360196e5c676b5d062b76fc66a363a86ba6"
        ),
        evidence_sha256=(
            "779d8a7bf483956c5a8cccfd9b37eec3a0baf5f1868f5cf67731a07917cf2f9b"
        ),
        conversion_pass_profile="rfdetr_pose_preserve_division_v1",
        disabled_passes=("common::divide_to_multiply",),
        class_count=1,
        graph_class_width=2,
        num_keypoints=17,
        keypoint_dim=8,
        num_keypoints_per_class=(0, 17),
    ),
    _profile(
        "rtdetr",
        "detect",
        "r18",
        640,
        "LibreRTDETRr18",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "a59e8751d228f25ab9a80a697ff4a7db"
            "3f2e63d7070dccc85b1aff75f369b18a"
        ),
        deployment_abi_sha256=(
            "02cd32a8d7fa13ae6c17a49609c08c2f"
            "e17455be50dad6d481702e8064a5ca5f"
        ),
        evidence_sha256=(
            "d26d4b1d8ba3f9270e397eb77870963f"
            "83df203efab6fe51e5da7641e86ac107"
        ),
        class_count=80,
    ),
    _profile(
        "rtdetrv2",
        "detect",
        "r18",
        640,
        "LibreRTDETRv2r18",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "8e25413ccd1aa375cdcb6763b8d0909f"
            "d7f51559516cdc5957f82e66f25cd664"
        ),
        deployment_abi_sha256=(
            "e19822a8d26ad9f391f534a1be54e4bd"
            "1b6ea03119ad1019521f37cda0074d41"
        ),
        evidence_sha256=(
            "ca7e37378710b6ea875fc6a144cfa7d7"
            "c996d19bd0832121b032f5a8c68b1da3"
        ),
        class_count=80,
    ),
    _profile(
        "rtdetrv4",
        "detect",
        "s",
        640,
        "LibreRTDETRv4s",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "86b9561e4e1c72888e288f01ba9680fc"
            "a14833e7db2fa9b08b4e485cc752cd10"
        ),
        deployment_abi_sha256=(
            "15d13de92cd8ae8e1478875d40534d09"
            "c206ed419a36e6eaf613e0f5a7467e42"
        ),
        evidence_sha256=(
            "930ddae9fd65e39e2cd7186b5603dc3e"
            "cd1cd3278f07cf4891e1aaa5fc02ad1e"
        ),
        class_count=80,
    ),
    _profile(
        "rtmdet",
        "detect",
        "t",
        640,
        "LibreRTMDett",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "2dc7117a2da847d7a339d45c012566c3"
            "ba712921822cfaec7753a5cd2002ba88"
        ),
        deployment_abi_sha256=(
            "cd001f68f05631b3d4bde5c449c52eda"
            "0cb0ef6c33bb1b8afc79f215f5131bfa"
        ),
        evidence_sha256=(
            "e9b3ae2e4af56d93bf92a0bcef2354d4"
            "dad628e7615d97a5dc659728143af5a0"
        ),
        class_count=80,
    ),
    _profile(
        "rtmdet",
        "segment",
        "t",
        640,
        "LibreRTMDett-seg",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "000b2db3a03271dd57deae5009dd9ca6"
            "db1549122032fb9b932eca88bf2728a8"
        ),
        deployment_abi_sha256=(
            "dbfa829bd3b16afca89fb28341726365"
            "3e83c2b769a09a2a5e54bd8cee032184"
        ),
        evidence_sha256=(
            "d9526f9d3b3356664033385c30c2a131"
            "ddbae47d35a10a99d090b6de1587939c"
        ),
        class_count=80,
    ),
    _profile(
        "siglip2",
        "classify",
        "b16",
        256,
        "LibreSigLIP2b16-cls",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "992b4cb57a40cfc0a72b9c72fbb37acb"
            "63b3dc895e34c27d5b451c577a1632ea"
        ),
        deployment_abi_sha256=(
            "f7dd2601a06f4882da49e8f10c3cd858"
            "4c175ac05a9e4aaff6dbff415c373420"
        ),
        evidence_sha256=(
            "0bf2b43b8b772296b21d918210bcbbca"
            "25050ea36f79532c06eddb96efbbab52"
        ),
        runtime_all=True,
        class_count=3,
        classification_activation="sigmoid",
    ),
    _profile(
        "yolo1",
        "detect",
        "b",
        448,
        "LibreYOLO1b",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "61faa37ae665783f7fbe1677887521ff"
            "68d6569996cddc2c042618587eb73cf8"
        ),
        deployment_abi_sha256=(
            "f25f470605e4389db6a268c9007af17c"
            "380854296fe2c1b9bf09d3a3e6e3f246"
        ),
        evidence_sha256=(
            "ce65bec5be7ab79bc551b6c4170ce22d"
            "35db7bbe7f34f85cd1f4f6dd5ce660a6"
        ),
        class_count=20,
    ),
    _profile(
        "yolo2",
        "detect",
        "b",
        608,
        "LibreYOLO2b",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "9b83665c767fdb97d554ec340b9d814f"
            "917ab468bb0d14ff463b068ca003daa0"
        ),
        deployment_abi_sha256=(
            "aea221022587c39889190f7dd164db9a"
            "ad641a383bf5efe442a128143c32bf81"
        ),
        evidence_sha256=(
            "9729a3a2c9b11cf6af24009483d3acd8"
            "b880c82a59a9f16c863c8372efb3933d"
        ),
        class_count=80,
    ),
    _profile(
        "yolo3",
        "detect",
        "b",
        416,
        "LibreYOLO3b",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "193b27d3d620c2dc9e3443e33a7c96d7"
            "3ce83e572ca178d10a1124821607843f"
        ),
        deployment_abi_sha256=(
            "459f5e446bb6bf75dfa7f3c07687afb5"
            "403d12adfdd258c16edec8003fcbefd8"
        ),
        evidence_sha256=(
            "453b0f13049aba782d1757c4ea53aac6"
            "089644acabba132f3fb1720a6c84b22d"
        ),
        class_count=80,
    ),
    _profile(
        "yolo4",
        "detect",
        "b",
        608,
        "LibreYOLO4b",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "92a8bbcfc88289ea839486333233821d"
            "4b949ef8fd151846d68314ad9327185a"
        ),
        deployment_abi_sha256=(
            "e33d551df6f715a186223d12a9a6da2e"
            "31640bb8faeae10822fb806de5c2dcf4"
        ),
        evidence_sha256=(
            "d252b4a180e1452a3f7bffd5d506f590"
            "c33623fa19f7fa16915acf38cc807947"
        ),
        class_count=80,
    ),
    _profile(
        "yolo7",
        "detect",
        "b",
        640,
        "LibreYOLO7b",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "1cd3cc6c0f1fde55164f5d5d9721ce1364a1c98c1145c6eba0c83d20290a13b3"
        ),
        deployment_abi_sha256=(
            "6bc1a0c759e2b4c469fecb0582a9396a74422ab7e010121619ab2f6acd19f1cf"
        ),
        evidence_sha256=(
            "c3374239fd3d73e7bb65d12fb333524785378298c39552da922ea13a09a3032e"
        ),
        class_count=80,
    ),
    _profile(
        "yolo9",
        "detect",
        "t",
        640,
        "LibreYOLO9t",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "57478825c607e921b7d7da9788a5b32aec096b496ea431913d7e64c4579a7c18"
        ),
        deployment_abi_sha256=(
            "c2e7a104cac34d5067c003db142f0cc04d7676f04ae0b5f0c64226c262edcc38"
        ),
        evidence_sha256=(
            "bc850d08eac90f33fdda865c58e9a85b41898c21a6964a0ceed7387c12fbda7c"
        ),
        runtime_all=True,
        class_count=80,
    ),
    _profile(
        "yolo9",
        "detect",
        "s",
        640,
        "LibreYOLO9s",
        class_count=80,
    ),
    _profile(
        "yolo9",
        "detect",
        "m",
        640,
        "LibreYOLO9m",
        class_count=80,
    ),
    _profile(
        "yolo9",
        "detect",
        "c",
        640,
        "LibreYOLO9c",
        class_count=80,
    ),
    _profile(
        "yolo9_e2e",
        "detect",
        "t",
        640,
        "LibreYOLO9E2Et",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "1c36935fe4400cbe891d6bde5cf036ab"
            "5100c6671a449903b73a761f099fc25f"
        ),
        deployment_abi_sha256=(
            "255964bf9d9b1e748197236d8059fb5f"
            "fcf9d8dfb6621e2f1d64f71c0144c72c"
        ),
        evidence_sha256=(
            "a7aec37e33a69465812f140cfaedc036"
            "485f591f5042218ace51c54cf4bb8fef"
        ),
        class_count=80,
    ),
    _profile(
        "yolo9_p2",
        "detect",
        "t",
        640,
        "deterministic YOLO9-P2-T transfer fixture",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "96d1e3bc2ff8c410cd81e5c19a8ebc88"
            "f71eb371a11276b48f740bacf1b9fbaa"
        ),
        deployment_abi_sha256=(
            "8e5d97c2e80cd42ee8f6a80d5b631df"
            "4e686b2e029d3a48da96405420f254882"
        ),
        evidence_sha256=(
            "4dbc6c1509554af3922fc83e3f8afeda"
            "9ec597729b9e9e6222d5675107c6919f"
        ),
        runtime_all=True,
        class_count=80,
    ),
    _profile(
        "yolonas",
        "detect",
        "s",
        96,
        "deterministic YOLO-NAS-S fixture",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "d706b9496bab4aa6807090725ae78faeb"
            "95d562267b2b79cbc539d15a6e3897d"
        ),
        deployment_abi_sha256=(
            "14ff0acc295fcc950fbc4ba7de55e143f"
            "a53646a236a5381d5c44892fb23a030"
        ),
        evidence_sha256=(
            "45af6cf61e82bd2ac04bff1bc504c33f"
            "6d7dc166737eba0e82fb52a61ac74061"
        ),
        runtime_all=True,
        class_count=2,
    ),
    _profile(
        "yolonas",
        "pose",
        "n",
        96,
        "deterministic YOLO-NAS-N pose fixture",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "34d2ded2885b7bfb2d9841080ed55bdb"
            "3423bcaa4b6142ac12493d317f406917"
        ),
        deployment_abi_sha256=(
            "9de49a8b1aded5cbaacea205612c3ff3b"
            "1294b46a47b9473f257c7294dce7911"
        ),
        evidence_sha256=(
            "f2fc7353ce0c56c5db95d17f4c0e5bc"
            "73beea18ed86146cccf7d0fb061a69540"
        ),
        runtime_all=True,
        class_count=1,
        graph_class_width=1,
        num_keypoints=17,
        keypoint_dim=3,
    ),
    _profile(
        "yolox",
        "detect",
        "n",
        416,
        "LibreYOLOXn",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "fcf727c52af6a452853c4924307f526288dc8ac62a307d09b4a43431d7d501ac"
        ),
        deployment_abi_sha256=(
            "897d1ba2e2f41506e9c11b76c6bdf487b05f438682c2be44c843dd1ee7264b7d"
        ),
        evidence_sha256=(
            "8f0168f4dcf29626fdcef227dafa0fb81d535e4ae645061ea7d230b9dda8095e"
        ),
        class_count=80,
    ),
    _profile(
        "zipdepth",
        "depth",
        "b",
        384,
        "LibreZipDepthb-depth",
        source_kind="pytorch-traced-graph-state-v2",
        source_sha256=(
            "d27d2704aa03153f0b36bb7a362dc321"
            "2252522d6b3f083953b07d9bfcfb45e0"
        ),
        deployment_abi_sha256=(
            "4b839906e714dd94b4415f79563d3e33"
            "ad8a6e0cbfcaf098dcc57148259ee163"
        ),
        evidence_sha256=(
            "a0dad29fa6757385f205e853f97ec969"
            "f21b545588776dc829080b4434b7c563"
        ),
        runtime_all=True,
        class_count=1,
    ),
)

COREML_EXECUTION_PROFILES = {
    (
        profile.family,
        profile.task,
        profile.size,
        profile.canvas,
        profile.nms,
        profile.prompt_max_points,
        profile.class_count,
        profile.graph_class_width,
        profile.num_keypoints,
        profile.keypoint_dim,
        profile.num_keypoints_per_class,
        profile.classification_activation,
        profile.checkpoint_variant,
        profile.architecture_signature,
        profile.restore_scale,
        profile.embedding_dim,
    ): profile
    for profile in _PROFILES
}
COREML_EXECUTION_PROFILES_BY_ID = {
    profile.profile_id: profile
    for profile in _PROFILES
    if profile.evidence_complete
}
COREML_VALIDATED_EXECUTION_PROFILES = {
    key: profile
    for key, profile in COREML_EXECUTION_PROFILES.items()
    if profile.evidence_complete
}
_VALIDATED_FAMILY_TASKS = frozenset(
    (profile.family, profile.task)
    for profile in _PROFILES
    if profile.evidence_complete
)

if len(COREML_EXECUTION_PROFILES) != len(_PROFILES):
    raise RuntimeError("Duplicate Core ML execution-profile key.")
if len({profile.profile_id for profile in _PROFILES}) != len(_PROFILES):
    raise RuntimeError("Duplicate Core ML execution-profile id.")
if len(COREML_EXECUTION_PROFILES_BY_ID) != len(
    COREML_VALIDATED_EXECUTION_PROFILES
):
    raise RuntimeError("Duplicate Core ML execution-profile id.")
for _profile_entry in _PROFILES:
    evidence_values = (
        _profile_entry.source_kind,
        _profile_entry.source_sha256,
        _profile_entry.deployment_abi_sha256,
        _profile_entry.evidence_sha256,
    )
    if any(value is not None for value in evidence_values) and not (
        _profile_entry.evidence_complete
    ):
        raise RuntimeError(
            "Core ML execution-profile evidence must be all-or-none for "
            f"{_profile_entry.family}/{_profile_entry.task}/"
            f"{_profile_entry.size}."
        )
    if _profile_entry.evidence_complete:
        if not re.fullmatch(
            r"[a-z0-9][a-z0-9._-]*",
            str(_profile_entry.source_kind),
        ):
            raise RuntimeError("Invalid Core ML profile source-kind token.")
        for _digest in (
            _profile_entry.source_sha256,
            _profile_entry.deployment_abi_sha256,
            _profile_entry.evidence_sha256,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", str(_digest)) is None:
                raise RuntimeError("Invalid Core ML profile evidence digest.")


def _strict_integer(value: Any, *, key: str) -> int:
    """Parse an integer without truncating floats or accepting booleans."""
    if isinstance(value, bool):
        raise ValueError(f"Core ML {key} must be an integer.")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        if re.fullmatch(r"[+-]?\d+", token):
            return int(token)
    raise ValueError(f"Core ML {key} must be an integer, got {value!r}.")


def _canvas_hw(canvas: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(canvas, Integral) and not isinstance(canvas, bool):
        values = (int(canvas), int(canvas))
    elif isinstance(canvas, (tuple, list)) and len(canvas) == 2:
        values = (
            _strict_integer(canvas[0], key="canvas height"),
            _strict_integer(canvas[1], key="canvas width"),
        )
    else:
        raise ValueError(
            "Core ML canvas must be an integer or (height, width) pair; "
            f"got {canvas!r}."
        )
    if min(values) <= 0:
        raise ValueError(f"Core ML canvas values must be positive, got {values}.")
    return values


def _optional_positive_int(value: Any, *, key: str) -> int | None:
    if value in (None, ""):
        return None
    parsed = _strict_integer(value, key=key)
    if parsed <= 0:
        raise ValueError(f"Core ML {key} must be positive, got {parsed}.")
    return parsed


def _optional_profile_token(value: Any, *, key: str) -> str | None:
    if value in (None, ""):
        return None
    token = str(value).strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", token):
        raise ValueError(
            f"Core ML {key} must be a lowercase profile token containing only "
            f"letters, digits, '.', '_', or '-'; got {value!r}."
        )
    return token


def _nonnegative_int_tuple(value: Any, *, key: str) -> tuple[int, ...]:
    if value in (None, "", (), []):
        return ()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Core ML {key} must be a JSON list of nonnegative integers."
            ) from exc
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            f"Core ML {key} must be a non-empty list of nonnegative integers."
        )
    parsed = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"Core ML {key} must contain only nonnegative integers; "
                f"item {index} is {item!r}."
            )
        number = item
        if number < 0:
            raise ValueError(
                f"Core ML {key}[{index}] must be nonnegative, got {number}."
            )
        parsed.append(number)
    if not any(parsed):
        raise ValueError(f"Core ML {key} must contain at least one positive value.")
    return tuple(parsed)


def normalize_coreml_compute_units(
    compute_units: Any,
    *,
    allow_validated: bool = True,
) -> str:
    """Normalize and validate a public Core ML compute-unit token."""
    value = str(compute_units).strip().lower()
    admitted = COREML_COMPUTE_UNITS if allow_validated else COREML_NATIVE_COMPUTE_UNITS
    if value not in admitted:
        raise ValueError(
            f"Invalid Core ML compute_units {compute_units!r}; expected one of "
            f"{sorted(admitted)}."
        )
    return value


def match_coreml_execution_profile(
    family: str,
    task: str,
    size: str | None,
    canvas: int | tuple[int, int] | list[int],
    *,
    nms: bool = False,
    prompt_max_points: int | None = None,
    class_count: int | None = None,
    graph_class_width: int | None = None,
    num_keypoints: int | None = None,
    keypoint_dim: int | None = None,
    num_keypoints_per_class: tuple[int, ...] | list[int] | str | None = None,
    classification_activation: str | None = None,
    checkpoint_variant: str | None = None,
    architecture_signature: str | None = None,
    restore_scale: int | None = None,
    embedding_dim: int | None = None,
) -> CoreMLExecutionProfile | None:
    """Return the exact hardware profile, never a family-level approximation."""
    if size in (None, ""):
        return None
    key = (
        str(family).strip().lower(),
        str(task).strip().lower(),
        str(size).strip().lower(),
        _canvas_hw(canvas),
        bool(nms),
        _optional_positive_int(
            prompt_max_points,
            key="prompt_max_points",
        ),
        _optional_positive_int(
            class_count,
            key="class_count",
        ),
        _optional_positive_int(
            graph_class_width,
            key="graph_class_width",
        ),
        _optional_positive_int(
            num_keypoints,
            key="num_keypoints",
        ),
        _optional_positive_int(
            keypoint_dim,
            key="keypoint_dim",
        ),
        _nonnegative_int_tuple(
            num_keypoints_per_class,
            key="num_keypoints_per_class",
        ),
        _optional_profile_token(
            classification_activation,
            key="classification_activation",
        ),
        _optional_profile_token(
            checkpoint_variant,
            key="checkpoint_variant",
        ),
        _optional_profile_token(
            architecture_signature,
            key="architecture_signature",
        ),
        _optional_positive_int(
            restore_scale,
            key="restore_scale",
        ),
        _optional_positive_int(
            embedding_dim,
            key="embedding_dim",
        ),
    )
    return COREML_EXECUTION_PROFILES.get(key)


def resolve_coreml_export_compute_units(
    compute_units: Any,
    *,
    family: str,
    task: str,
    size: str | None,
    canvas: int | tuple[int, int] | list[int],
    precision: str,
    nms: bool,
    prompt_max_points: int | None = None,
    class_count: int | None = None,
    graph_class_width: int | None = None,
    num_keypoints: int | None = None,
    keypoint_dim: int | None = None,
    num_keypoints_per_class: tuple[int, ...] | list[int] | str | None = None,
    classification_activation: str | None = None,
    checkpoint_variant: str | None = None,
    architecture_signature: str | None = None,
    restore_scale: int | None = None,
    embedding_dim: int | None = None,
    source_kind: str | None = None,
    source_sha256: str | None = None,
    defer_source_validation: bool = False,
) -> tuple[str, CoreMLExecutionProfile | None]:
    """Resolve export routing and reject false claims of exact validation."""
    requested = normalize_coreml_compute_units(compute_units)
    family_key = str(family).strip().lower()
    task_key = str(task).strip().lower()
    profile = match_coreml_execution_profile(
        family_key,
        task_key,
        size,
        canvas,
        nms=nms,
        prompt_max_points=prompt_max_points,
        class_count=class_count,
        graph_class_width=graph_class_width,
        num_keypoints=num_keypoints,
        keypoint_dim=keypoint_dim,
        num_keypoints_per_class=num_keypoints_per_class,
        classification_activation=classification_activation,
        checkpoint_variant=checkpoint_variant,
        architecture_signature=architecture_signature,
        restore_scale=restore_scale,
        embedding_dim=embedding_dim,
    )
    if profile is not None and not profile.evidence_complete:
        if requested == COREML_VALIDATED_COMPUTE_UNITS:
            raise NotImplementedError(
                "An exact Core ML conversion recipe exists for this request, "
                "but it has not yet been promoted with checkpoint/source and "
                "post-conversion ABI evidence. Use compute_units='cpu_only' "
                "explicitly only for an experimental campaign run."
            )
        warnings.warn(
            "This Core ML conversion recipe is using the CPU_ONLY "
            "compatibility path while awaiting exact checkpoint/source and "
            "post-conversion ABI evidence. The artifact will not receive a "
            "hash-bound execution-profile marker.",
            RuntimeWarning,
            stacklevel=3,
        )
        return requested, None

    if profile is not None:
        normalized_source_kind = (
            str(source_kind).strip().lower()
            if source_kind not in (None, "")
            else None
        )
        normalized_source_sha256 = (
            str(source_sha256).strip()
            if source_sha256 not in (None, "")
            else None
        )
        if normalized_source_sha256 is None:
            if defer_source_validation:
                if requested != COREML_VALIDATED_COMPUTE_UNITS:
                    return requested, None
            elif requested == COREML_VALIDATED_COMPUTE_UNITS:
                raise NotImplementedError(
                    "compute_units='validated' requires the exact live source "
                    "identity recorded by the Apple-M4 execution profile."
                )
            else:
                warnings.warn(
                    "Core ML source identity is unavailable. The explicit "
                    "CPU_ONLY compatibility path remains available, but the "
                    "artifact will not receive a hash-bound execution-profile "
                    "marker.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return requested, None
        elif (
            normalized_source_kind != profile.source_kind
            or normalized_source_sha256 != profile.source_sha256
        ):
            if requested == COREML_VALIDATED_COMPUTE_UNITS:
                raise NotImplementedError(
                    "compute_units='validated' requires the exact checkpoint "
                    "or source identity recorded by the Apple-M4 execution "
                    f"profile {profile.profile_id}."
                )
            warnings.warn(
                "The Core ML source identity does not match the Apple-M4 "
                "execution profile. The CPU_ONLY compatibility path remains "
                "available, but the artifact will not receive a hash-bound "
                "execution-profile marker.",
                RuntimeWarning,
                stacklevel=3,
            )
            return requested, None
        if str(precision).strip().lower() != profile.precision:
            raise NotImplementedError(
                f"{profile.profile_id} is validated only in "
                f"{profile.precision.upper()}; pass half=False. FP16 remains "
                "an experimental profile."
            )
        resolved = (
            profile.conversion_compute_units
            if requested == COREML_VALIDATED_COMPUTE_UNITS
            else requested
        )
        if resolved != profile.conversion_compute_units:
            raise NotImplementedError(
                f"{profile.profile_id} conversion is validated only with "
                f"compute_units={profile.conversion_compute_units!r}; got "
                f"{requested!r}."
            )
        return resolved, profile

    if (family_key, task_key) in _VALIDATED_FAMILY_TASKS:
        warnings.warn(
            "This Core ML request does not match an exact Apple-M4 validation "
            f"profile: family={family_key!r}, task={task_key!r}, size={size!r}, "
            f"canvas={_canvas_hw(canvas)!r}, precision={precision!r}, "
            f"nms={bool(nms)!r}, prompt_max_points={prompt_max_points!r}, "
            f"class_count={class_count!r}, graph_class_width="
            f"{graph_class_width!r}, num_keypoints={num_keypoints!r}, "
            f"keypoint_dim={keypoint_dim!r}, num_keypoints_per_class="
            f"{num_keypoints_per_class!r}, classification_activation="
            f"{classification_activation!r}, checkpoint_variant="
            f"{checkpoint_variant!r}, architecture_signature="
            f"{architecture_signature!r}, restore_scale={restore_scale!r}, "
            f"embedding_dim={embedding_dim!r}. Conversion remains "
            "experimental for this "
            "combination.",
            RuntimeWarning,
            stacklevel=3,
        )
    if requested == COREML_VALIDATED_COMPUTE_UNITS:
        raise NotImplementedError(
            "compute_units='validated' requires an exact Apple-M4 execution "
            "profile, but this request has none. Pass an explicit native "
            "planner such as compute_units='cpu_only' only if you intend to "
            "run an experimental conversion."
        )
    return requested, None


def coreml_execution_profile_metadata(
    profile: CoreMLExecutionProfile | None,
    *,
    conversion_compute_units: str,
) -> dict[str, Any]:
    """Return strict metadata for a validated or experimental artifact."""
    if profile is None:
        return {"coreml_execution_profile_status": "experimental"}
    if not profile.evidence_complete:
        raise ValueError(
            "Core ML execution-profile metadata cannot be emitted for an "
            "unpromoted conversion recipe."
        )
    if conversion_compute_units != profile.conversion_compute_units:
        raise ValueError(
            "Core ML profile metadata cannot label an unvalidated conversion "
            f"planner: {conversion_compute_units!r}."
        )
    height, width = profile.canvas
    profile_metadata = {
        "model_family": profile.family,
        "task": profile.task,
        "size": profile.size,
        "model_size": profile.size,
        "imgsz": max(height, width),
        "imgsz_h": height,
        "imgsz_w": width,
        "precision": profile.precision,
        "nms": profile.nms,
        "coreml_execution_profile_status": "validated",
        "coreml_execution_profile_version": COREML_EXECUTION_PROFILE_VERSION,
        "coreml_execution_profile": profile.profile_id,
        "coreml_validation_reference": profile.reference,
        "coreml_profile_source_kind": profile.source_kind,
        "coreml_profile_source_sha256": profile.source_sha256,
        "coreml_profile_abi_schema": COREML_DEPLOYMENT_ABI_SCHEMA,
        "coreml_profile_abi_sha256": profile.deployment_abi_sha256,
        "coreml_validation_evidence_sha256": profile.evidence_sha256,
        "coreml_validated_precision": profile.precision,
        "coreml_conversion_compute_units": profile.conversion_compute_units,
        "coreml_runtime_compute_units": list(profile.runtime_compute_units),
        "coreml_default_compute_units": profile.default_runtime_compute_units,
        "coreml_validated_nms": profile.nms,
        "coreml_validation_hardware": "Apple M4",
        "coreml_validation_os": "macOS 27.0",
        "coreml_validation_coremltools": "9.0",
    }
    if profile.prompt_max_points is not None:
        profile_metadata["prompt_max_points"] = profile.prompt_max_points
        profile_metadata["coreml_validated_prompt_max_points"] = (
            profile.prompt_max_points
        )
    if profile.class_count is not None:
        profile_metadata["nc"] = profile.class_count
        profile_metadata["nb_classes"] = profile.class_count
        profile_metadata["coreml_validated_class_count"] = (
            profile.class_count
        )
    scalar_dimensions = (
        (
            "graph_class_width",
            "graph_class_width",
            "coreml_validated_graph_class_width",
        ),
        (
            "num_keypoints",
            "num_keypoints",
            "coreml_validated_num_keypoints",
        ),
        (
            "keypoint_dim",
            "keypoint_dim",
            "coreml_validated_keypoint_dim",
        ),
        (
            "classification_activation",
            "classification_activation",
            "coreml_validated_classification_activation",
        ),
        (
            "checkpoint_variant",
            "checkpoint_variant",
            "coreml_validated_checkpoint_variant",
        ),
        (
            "architecture_signature",
            "architecture_signature",
            "coreml_validated_architecture_signature",
        ),
        (
            "restore_scale",
            "restore_scale",
            "coreml_validated_restore_scale",
        ),
        (
            "embedding_dim",
            "facerec_embedding_dim",
            "coreml_validated_embedding_dim",
        ),
    )
    for attribute, artifact_key, validated_key in scalar_dimensions:
        value = getattr(profile, attribute)
        if value is not None:
            profile_metadata[artifact_key] = value
            profile_metadata[validated_key] = value
    if profile.num_keypoints_per_class:
        profile_metadata["num_keypoints_per_class"] = list(
            profile.num_keypoints_per_class
        )
        profile_metadata["coreml_validated_num_keypoints_per_class"] = list(
            profile.num_keypoints_per_class
        )
    if profile.conversion_pass_profile is not None:
        profile_metadata.update(
            {
                "coreml_conversion_pass_profile": (
                    profile.conversion_pass_profile
                ),
                "coreml_disabled_passes": list(profile.disabled_passes),
            }
        )
    if profile.capture_decomposition_profile is not None:
        profile_metadata.update(
            {
                "coreml_capture_decomposition_profile": (
                    profile.capture_decomposition_profile
                ),
                "coreml_capture_decompositions": list(
                    profile.capture_decompositions
                ),
            }
        )
    return profile_metadata


def _metadata_values_equal(current: Any, expected: Any) -> bool:
    if current == expected:
        return True
    if isinstance(expected, bool):
        return str(current).strip().lower() in {
            "true" if expected else "false",
            "1" if expected else "0",
        }
    if isinstance(expected, int) and not isinstance(expected, bool):
        return str(current).strip() == str(expected)
    if isinstance(expected, (list, tuple)):
        parsed = current
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except json.JSONDecodeError:
                return False
        return isinstance(parsed, (list, tuple)) and list(parsed) == list(
            expected
        )
    return str(current).strip() == str(expected)


def merge_coreml_execution_profile_metadata(
    metadata: Mapping[str, Any] | None,
    profile: CoreMLExecutionProfile | None,
    *,
    conversion_compute_units: str,
) -> dict[str, Any]:
    """Merge profile metadata without silently replacing caller values."""
    merged = dict(metadata or {})
    expected = coreml_execution_profile_metadata(
        profile,
        conversion_compute_units=conversion_compute_units,
    )
    for key, value in expected.items():
        current = merged.get(key)
        if current not in (None, "") and not _metadata_values_equal(
            current,
            value,
        ):
            raise ValueError(
                f"Core ML metadata {key!r}={current!r} conflicts with "
                f"the execution profile value {value!r}."
            )
        merged[key] = value
    return merged


def finalize_coreml_execution_profile_metadata(
    metadata: Mapping[str, Any],
    profile: CoreMLExecutionProfile | None,
    *,
    requested_compute_units: str,
    conversion_compute_units: str,
    deployment_abi_sha256: str,
) -> tuple[dict[str, Any], CoreMLExecutionProfile | None]:
    """Promote only a source-matched profile whose final protobuf ABI matches."""
    requested = normalize_coreml_compute_units(requested_compute_units)
    actual_abi = require_lower_sha256(
        deployment_abi_sha256,
        key="coreml_profile_abi_sha256",
    )
    resolved_profile = profile
    if (
        resolved_profile is not None
        and actual_abi != resolved_profile.deployment_abi_sha256
    ):
        if requested == COREML_VALIDATED_COMPUTE_UNITS:
            raise RuntimeError(
                "The final converted Core ML protobuf ABI does not match the "
                "Apple-M4 execution profile. The staged artifact was not "
                "promoted or installed."
            )
        warnings.warn(
            "The final converted Core ML protobuf ABI does not match the "
            "Apple-M4 execution profile. Keeping this explicit native-planner "
            "artifact experimental.",
            RuntimeWarning,
            stacklevel=3,
        )
        resolved_profile = None
    merged = merge_coreml_execution_profile_metadata(
        metadata,
        resolved_profile,
        conversion_compute_units=conversion_compute_units,
    )
    validate_coreml_execution_profile_metadata(merged)
    return merged, resolved_profile


def _metadata_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "1"}:
        return True
    if token in {"false", "0"}:
        return False
    raise ValueError(f"Core ML metadata {key!r} must be boolean.")


def _metadata_runtime_units(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Core ML metadata 'coreml_runtime_compute_units' must be a "
                "JSON list."
            ) from exc
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "Core ML metadata 'coreml_runtime_compute_units' must be a "
            "non-empty list."
        )
    units = tuple(
        normalize_coreml_compute_units(item, allow_validated=False)
        for item in value
    )
    if len(set(units)) != len(units):
        raise ValueError(
            "Core ML metadata 'coreml_runtime_compute_units' contains duplicates."
        )
    return units


def _metadata_canvas(metadata: Mapping[str, Any]) -> tuple[int, int] | None:
    raw_h = metadata.get("imgsz_h")
    raw_w = metadata.get("imgsz_w")
    if raw_h not in (None, "") or raw_w not in (None, ""):
        if raw_h in (None, "") or raw_w in (None, ""):
            raise ValueError(
                "Core ML execution-profile metadata must declare both "
                "imgsz_h and imgsz_w."
            )
        return _canvas_hw((int(raw_h), int(raw_w)))
    raw_size = metadata.get("imgsz")
    if raw_size in (None, ""):
        return None
    if isinstance(raw_size, str):
        try:
            parsed = json.loads(raw_size)
        except json.JSONDecodeError:
            parsed = raw_size
    else:
        parsed = raw_size
    if isinstance(parsed, (list, tuple)):
        return _canvas_hw(parsed)
    return _canvas_hw(int(parsed))


def _profile_from_basic_metadata(
    metadata: Mapping[str, Any],
) -> CoreMLExecutionProfile | None:
    canvas = _metadata_canvas(metadata)
    if canvas is None:
        return None
    raw_nms = metadata.get("nms", False)
    nms = _metadata_bool(raw_nms, key="nms") if raw_nms not in (None, "") else False
    family = str(metadata.get("model_family", "")).strip().lower()
    prompt_max_points = (
        metadata.get("prompt_max_points")
        if family in {"edgetam", "mobilesam", "sam", "sam2", "sam3"}
        else None
    )
    class_count = metadata.get("nc", metadata.get("nb_classes"))
    return match_coreml_execution_profile(
        family,
        str(metadata.get("task", "")),
        str(metadata.get("size", metadata.get("model_size", ""))),
        canvas,
        nms=nms,
        prompt_max_points=prompt_max_points,
        class_count=class_count,
        graph_class_width=metadata.get("graph_class_width"),
        num_keypoints=metadata.get("num_keypoints"),
        keypoint_dim=metadata.get("keypoint_dim"),
        num_keypoints_per_class=metadata.get("num_keypoints_per_class"),
        classification_activation=metadata.get(
            "classification_activation"
        ),
        checkpoint_variant=metadata.get("checkpoint_variant"),
        architecture_signature=metadata.get("architecture_signature"),
        restore_scale=metadata.get("restore_scale"),
        embedding_dim=metadata.get("facerec_embedding_dim"),
    )


def validate_coreml_execution_profile_metadata(
    metadata: Mapping[str, Any],
) -> CoreMLExecutionProfile | None:
    """Validate a profile marker without compiling a Core ML proxy."""
    status = str(
        metadata.get("coreml_execution_profile_status", "")
    ).strip().lower()
    profile_id = str(metadata.get("coreml_execution_profile", "")).strip()
    if not status and not profile_id:
        return None
    if status == "experimental":
        if profile_id:
            raise ValueError(
                "Experimental Core ML metadata must not declare a validated "
                "execution-profile id."
            )
        return None
    if status != "validated":
        raise ValueError(
            "Core ML metadata 'coreml_execution_profile_status' must be "
            "'validated' or 'experimental'."
        )
    required = {
        "model_family",
        "task",
        "size",
        "precision",
        "nms",
        "coreml_execution_profile_version",
        "coreml_execution_profile",
        "coreml_validation_reference",
        "coreml_profile_source_kind",
        "coreml_profile_source_sha256",
        "coreml_profile_abi_schema",
        "coreml_profile_abi_sha256",
        "coreml_validation_evidence_sha256",
        "coreml_validation_hardware",
        "coreml_validation_os",
        "coreml_validation_coremltools",
        "coreml_validated_precision",
        "coreml_conversion_compute_units",
        "coreml_runtime_compute_units",
        "coreml_default_compute_units",
        "coreml_validated_nms",
    }
    missing = sorted(key for key in required if metadata.get(key) in (None, ""))
    if missing:
        raise ValueError(
            "Validated Core ML execution-profile metadata is incomplete; "
            f"missing {missing}."
        )
    if (
        str(metadata["coreml_execution_profile_version"]).strip()
        != COREML_EXECUTION_PROFILE_VERSION
    ):
        raise ValueError(
            "Unsupported Core ML execution-profile version "
            f"{metadata['coreml_execution_profile_version']!r}."
        )
    profile = COREML_EXECUTION_PROFILES_BY_ID.get(profile_id)
    if profile is None:
        raise ValueError(
            f"Unknown Core ML execution-profile id {profile_id!r}."
        )
    specialized_required = []
    if profile.prompt_max_points is not None:
        specialized_required.extend(
            (
                "prompt_max_points",
                "coreml_validated_prompt_max_points",
            )
        )
    if profile.class_count is not None:
        specialized_required.extend(
            (
                "nc",
                "nb_classes",
                "coreml_validated_class_count",
            )
        )
    specialized_dimension_keys = (
        (
            profile.graph_class_width,
            "graph_class_width",
            "coreml_validated_graph_class_width",
        ),
        (
            profile.num_keypoints,
            "num_keypoints",
            "coreml_validated_num_keypoints",
        ),
        (
            profile.keypoint_dim,
            "keypoint_dim",
            "coreml_validated_keypoint_dim",
        ),
        (
            profile.num_keypoints_per_class,
            "num_keypoints_per_class",
            "coreml_validated_num_keypoints_per_class",
        ),
        (
            profile.classification_activation,
            "classification_activation",
            "coreml_validated_classification_activation",
        ),
        (
            profile.checkpoint_variant,
            "checkpoint_variant",
            "coreml_validated_checkpoint_variant",
        ),
        (
            profile.architecture_signature,
            "architecture_signature",
            "coreml_validated_architecture_signature",
        ),
        (
            profile.restore_scale,
            "restore_scale",
            "coreml_validated_restore_scale",
        ),
        (
            profile.embedding_dim,
            "facerec_embedding_dim",
            "coreml_validated_embedding_dim",
        ),
    )
    for expected, artifact_key, validated_key in specialized_dimension_keys:
        if expected not in (None, (), []):
            specialized_required.extend((artifact_key, validated_key))
    specialized_missing = sorted(
        key
        for key in specialized_required
        if metadata.get(key) in (None, "")
    )
    if specialized_missing:
        raise ValueError(
            "Validated Core ML execution-profile metadata is incomplete; "
            f"missing {specialized_missing}."
        )
    basic_profile = _profile_from_basic_metadata(metadata)
    if basic_profile != profile:
        raise ValueError(
            "Core ML execution-profile id disagrees with the artifact's "
            "family/task/size/canvas/NMS metadata."
        )
    expected_scalars = {
        "model_family": profile.family,
        "task": profile.task,
        "size": profile.size,
        "precision": profile.precision,
        "coreml_validation_reference": profile.reference,
        "coreml_profile_source_kind": profile.source_kind,
        "coreml_profile_source_sha256": profile.source_sha256,
        "coreml_profile_abi_schema": COREML_DEPLOYMENT_ABI_SCHEMA,
        "coreml_profile_abi_sha256": profile.deployment_abi_sha256,
        "coreml_validation_evidence_sha256": profile.evidence_sha256,
        "coreml_validation_hardware": "Apple M4",
        "coreml_validation_os": "macOS 27.0",
        "coreml_validation_coremltools": "9.0",
        "coreml_validated_precision": profile.precision,
        "coreml_conversion_compute_units": profile.conversion_compute_units,
        "coreml_default_compute_units": profile.default_runtime_compute_units,
    }
    if profile.prompt_max_points is not None:
        expected_scalars["prompt_max_points"] = str(
            profile.prompt_max_points
        )
        expected_scalars["coreml_validated_prompt_max_points"] = str(
            profile.prompt_max_points
        )
    if profile.class_count is not None:
        expected_scalars["nc"] = str(profile.class_count)
        expected_scalars["nb_classes"] = str(profile.class_count)
        expected_scalars["coreml_validated_class_count"] = str(
            profile.class_count
        )
    for attribute, artifact_key, validated_key in (
        (
            "graph_class_width",
            "graph_class_width",
            "coreml_validated_graph_class_width",
        ),
        (
            "num_keypoints",
            "num_keypoints",
            "coreml_validated_num_keypoints",
        ),
        (
            "keypoint_dim",
            "keypoint_dim",
            "coreml_validated_keypoint_dim",
        ),
        (
            "classification_activation",
            "classification_activation",
            "coreml_validated_classification_activation",
        ),
        (
            "checkpoint_variant",
            "checkpoint_variant",
            "coreml_validated_checkpoint_variant",
        ),
        (
            "architecture_signature",
            "architecture_signature",
            "coreml_validated_architecture_signature",
        ),
        (
            "restore_scale",
            "restore_scale",
            "coreml_validated_restore_scale",
        ),
        (
            "embedding_dim",
            "facerec_embedding_dim",
            "coreml_validated_embedding_dim",
        ),
    ):
        value = getattr(profile, attribute)
        if value is not None:
            expected_scalars[artifact_key] = str(value)
            expected_scalars[validated_key] = str(value)
    for key, expected in expected_scalars.items():
        if str(metadata[key]).strip() != expected:
            raise ValueError(
                f"Core ML execution-profile metadata {key!r} must be "
                f"{expected!r}."
            )
    if profile.num_keypoints_per_class:
        artifact_schema = _nonnegative_int_tuple(
            metadata["num_keypoints_per_class"],
            key="num_keypoints_per_class",
        )
        validated_schema = _nonnegative_int_tuple(
            metadata["coreml_validated_num_keypoints_per_class"],
            key="coreml_validated_num_keypoints_per_class",
        )
        if (
            artifact_schema != profile.num_keypoints_per_class
            or validated_schema != profile.num_keypoints_per_class
        ):
            raise ValueError(
                "Core ML validated keypoint-schema metadata disagrees with "
                "the execution profile."
            )
    if _metadata_runtime_units(
        metadata["coreml_runtime_compute_units"]
    ) != profile.runtime_compute_units:
        raise ValueError(
            "Core ML runtime compute-unit allow-list disagrees with the "
            "validated execution profile."
        )
    if (
        _metadata_bool(
            metadata["coreml_validated_nms"],
            key="coreml_validated_nms",
        )
        != profile.nms
    ):
        raise ValueError(
            "Core ML validated-NMS metadata disagrees with the execution profile."
        )
    if profile.conversion_pass_profile is not None:
        pass_profile = str(
            metadata.get("coreml_conversion_pass_profile", "")
        ).strip()
        if pass_profile != profile.conversion_pass_profile:
            raise ValueError(
                "Core ML conversion-pass metadata disagrees with the "
                "validated execution profile."
            )
        disabled_passes = metadata.get("coreml_disabled_passes")
        if isinstance(disabled_passes, str):
            try:
                disabled_passes = json.loads(disabled_passes)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Core ML metadata 'coreml_disabled_passes' must be a "
                    "JSON list."
                ) from exc
        if (
            not isinstance(disabled_passes, (list, tuple))
            or tuple(disabled_passes) != profile.disabled_passes
        ):
            raise ValueError(
                "Core ML disabled-pass metadata disagrees with the "
                "validated execution profile."
            )
    if profile.capture_decomposition_profile is not None:
        capture_profile = str(
            metadata.get("coreml_capture_decomposition_profile", "")
        ).strip()
        if capture_profile != profile.capture_decomposition_profile:
            raise ValueError(
                "Core ML capture-decomposition metadata disagrees with the "
                "validated execution profile."
            )
        capture_decompositions = metadata.get(
            "coreml_capture_decompositions"
        )
        if isinstance(capture_decompositions, str):
            try:
                capture_decompositions = json.loads(
                    capture_decompositions
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Core ML metadata 'coreml_capture_decompositions' must "
                    "be a JSON list."
                ) from exc
        if (
            not isinstance(capture_decompositions, (list, tuple))
            or tuple(capture_decompositions)
            != profile.capture_decompositions
        ):
            raise ValueError(
                "Core ML capture decomposition list disagrees with the "
                "validated execution profile."
            )
    artifact_precision = str(metadata.get("precision", "")).strip().lower()
    if artifact_precision != profile.precision:
        raise ValueError(
            "Core ML artifact precision disagrees with the validated execution "
            f"profile: {artifact_precision!r} != {profile.precision!r}."
        )
    return profile


def resolve_coreml_runtime_compute_units(
    compute_units: Any,
    metadata: Mapping[str, Any],
) -> str:
    """Resolve runtime routing from spec metadata before compiling a proxy."""
    requested = normalize_coreml_compute_units(compute_units)
    status = str(
        metadata.get("coreml_execution_profile_status", "")
    ).strip().lower()
    declared_version = str(
        metadata.get("coreml_execution_profile_version", "")
    ).strip()
    if status == "validated" and declared_version != (
        COREML_EXECUTION_PROFILE_VERSION
    ):
        if requested == COREML_VALIDATED_COMPUTE_UNITS:
            raise NotImplementedError(
                "This Core ML artifact carries a legacy execution profile "
                f"(version {declared_version or 'missing'}). Version "
                f"{COREML_EXECUTION_PROFILE_VERSION} checkpoint/source and ABI "
                "identity is required for compute_units='validated'. Re-export "
                "the artifact."
            )
        if requested != "cpu_only":
            raise NotImplementedError(
                "Legacy Core ML execution-profile artifacts may be loaded only "
                "with the explicit unvalidated compute_units='cpu_only' opt-in."
            )
        warnings.warn(
            "Loading a legacy Core ML execution-profile artifact as an "
            "unvalidated CPU_ONLY opt-in. Re-export it to obtain checkpoint/"
            "source and post-conversion ABI identity.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "cpu_only"
    profile = validate_coreml_execution_profile_metadata(metadata)
    if profile is not None:
        resolved = (
            profile.default_runtime_compute_units
            if requested == COREML_VALIDATED_COMPUTE_UNITS
            else requested
        )
        if resolved not in profile.runtime_compute_units:
            raise NotImplementedError(
                f"{profile.profile_id} runtime is validated only for "
                f"compute_units={list(profile.runtime_compute_units)!r}; got "
                f"{requested!r}."
            )
        return resolved

    if status == "experimental":
        if requested == COREML_VALIDATED_COMPUTE_UNITS:
            raise NotImplementedError(
                "compute_units='validated' cannot load an explicitly "
                "experimental Core ML artifact. Pass compute_units='cpu_only' "
                "only if you intend to opt into its unvalidated runtime path."
            )
        return requested

    inferred = _profile_from_basic_metadata(metadata)
    if inferred is not None:
        if requested == COREML_VALIDATED_COMPUTE_UNITS:
            raise NotImplementedError(
                "This legacy Core ML artifact resembles an exact M4 profile "
                "but does not carry its fail-closed execution metadata. "
                "Re-export it, or pass compute_units='cpu_only' explicitly "
                "only as an unvalidated legacy opt-in."
            )
        if requested != "cpu_only":
            raise NotImplementedError(
                "This legacy Core ML artifact matches an exact M4 profile but "
                "does not carry its fail-closed execution metadata. Re-export "
                "before using an accelerator planner."
            )
        warnings.warn(
            "Loading a legacy Core ML artifact without an execution-profile "
            "marker. Routing to CPU_ONLY; re-export to embed the exact M4 "
            "profile.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "cpu_only"

    if requested == COREML_VALIDATED_COMPUTE_UNITS:
        raise NotImplementedError(
            "compute_units='validated' requires exact execution-profile "
            "metadata. This Core ML artifact is unprofiled; re-export it or "
            "pass an explicit native planner only as an experimental opt-in."
        )
    return requested


__all__ = [
    "COREML_COMPUTE_UNITS",
    "COREML_EXECUTION_PROFILES",
    "COREML_EXECUTION_PROFILE_VERSION",
    "COREML_NATIVE_COMPUTE_UNITS",
    "COREML_VALIDATED_COMPUTE_UNITS",
    "CoreMLExecutionProfile",
    "coreml_execution_profile_metadata",
    "finalize_coreml_execution_profile_metadata",
    "match_coreml_execution_profile",
    "merge_coreml_execution_profile_metadata",
    "normalize_coreml_compute_units",
    "resolve_coreml_export_compute_units",
    "resolve_coreml_runtime_compute_units",
    "validate_coreml_execution_profile_metadata",
]
