"""Immutable public model identity and publication manifest.

The runtime class registry remains the compatibility mechanism for sniffing
metadata-less state dictionaries.  Public discovery must not depend on which
optional model modules happened to be imported first, so CLI aliases,
canonical filenames, task/size validation, and download routing use this
declarative manifest instead.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Iterator

from ..tasks import normalize_task, task_to_suffix


class FactoryKind(str, Enum):
    """Public constructor that owns a model family."""

    CHECKPOINT = "libreyolo"
    SAM = "sam"
    VLM = "vlm"
    OPENVOCAB = "openvocab"


class PublicationState(str, Enum):
    """Declared checkpoint/snapshot availability, independent of architecture support."""

    PUBLISHED = "published"
    DIRECT = "direct"
    GATED = "gated"
    CONFIG_ONLY = "config_only"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SizeSpec:
    """One valid size and its native input resolution for a task."""

    code: str
    native_imgsz: int
    publication: PublicationState = PublicationState.PUBLISHED


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One canonical task and all architecture-supported sizes."""

    task: str
    sizes: tuple[SizeSpec, ...]


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """An explicitly named published/configured checkpoint variant."""

    name: str
    task: str
    sizes: tuple[str, ...]
    publication: PublicationState
    native_imgsz: int | None = None
    download_kind: str | None = None


@dataclass(frozen=True, slots=True)
class FactoryModelSpec:
    """One model identifier accepted by a sibling public factory."""

    size: str
    model: str
    aliases: tuple[str, ...]
    repository: str
    revision: str | None = None


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """Stable public identity for one model family."""

    family: str
    class_path: str
    filename_prefix: str
    default_task: str
    tasks: tuple[TaskSpec, ...]
    factory: FactoryKind = FactoryKind.CHECKPOINT
    optional_extra: str | None = None
    dependencies: tuple[str, ...] = ()
    export_override: str = "none"
    weight_ext: str = ".pt"
    suffixless_tasks: tuple[str, ...] = ()
    download_kind: str = "hf"
    variants: tuple[VariantSpec, ...] = ()
    factory_models: tuple[FactoryModelSpec, ...] = ()

    @property
    def public_entrypoint(self) -> str:
        return {
            FactoryKind.CHECKPOINT: "LibreYOLO",
            FactoryKind.SAM: "LibreSAM",
            FactoryKind.VLM: "LibreVLM",
            FactoryKind.OPENVOCAB: "LibreOpenVocab",
        }[self.factory]


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """One valid family/task/size checkpoint or snapshot identity."""

    family: str
    size: str
    task: str
    native_imgsz: int
    factory: FactoryKind
    canonical_filename: str | None
    publication: PublicationState
    download_kind: str
    download_url: str | None
    variant: str | None = None
    aliases: tuple[str, ...] = ()
    factory_model: str | None = None
    repository: str | None = None
    revision: str | None = None

    @property
    def downloadable(self) -> bool:
        return self.download_kind in {"hf", "family", "snapshot"}

    @property
    def invocation(self) -> str:
        entrypoint = {
            FactoryKind.CHECKPOINT: "LibreYOLO",
            FactoryKind.SAM: "LibreSAM",
            FactoryKind.VLM: "LibreVLM",
            FactoryKind.OPENVOCAB: "LibreOpenVocab",
        }[self.factory]
        model = self.canonical_filename or self.factory_model
        return f"{entrypoint}({model!r})" if model is not None else f"{entrypoint}()"


def _task(
    task: str,
    sizes: tuple[tuple[str, int], ...],
    *,
    publication: PublicationState = PublicationState.PUBLISHED,
    overrides: tuple[tuple[str, PublicationState], ...] = (),
) -> TaskSpec:
    override_map = dict(overrides)
    return TaskSpec(
        task=task,
        sizes=tuple(
            SizeSpec(code, imgsz, override_map.get(code, publication))
            for code, imgsz in sizes
        ),
    )


def _family(
    family: str,
    class_path: str,
    prefix: str,
    default_task: str,
    tasks: tuple[TaskSpec, ...],
    *,
    factory: FactoryKind = FactoryKind.CHECKPOINT,
    optional_extra: str | None = None,
    dependency: str | None = None,
    dependencies: tuple[str, ...] = (),
    export_override: str = "none",
    suffixless_tasks: tuple[str, ...] = (),
    download_kind: str = "hf",
    variants: tuple[VariantSpec, ...] = (),
    factory_models: tuple[FactoryModelSpec, ...] = (),
) -> FamilySpec:
    return FamilySpec(
        family=family,
        class_path=class_path,
        filename_prefix=prefix,
        default_task=default_task,
        tasks=tasks,
        factory=factory,
        optional_extra=optional_extra,
        dependencies=((dependency,) if dependency is not None else ()) + dependencies,
        export_override=export_override,
        suffixless_tasks=suffixless_tasks,
        download_kind=download_kind,
        variants=variants,
        factory_models=factory_models,
    )


def _factory_model(
    size: str,
    model: str,
    aliases: tuple[str, ...],
    repository: str,
    *,
    revision: str | None = None,
) -> FactoryModelSpec:
    return FactoryModelSpec(
        size=size,
        model=model,
        aliases=aliases,
        repository=repository,
        revision=revision,
    )


# This tuple is deliberately static.  Do not populate it by importing model
# modules: doing so recreates the initialization-order bug this manifest fixes.
MODEL_FAMILIES: tuple[FamilySpec, ...] = (
    _family(
        "birefnet",
        "libreyolo.models.birefnet.model.LibreBiRefNet",
        "LibreBiRefNet",
        "matte",
        (
            _task(
                "matte",
                (("t", 1024), ("l", 1024)),
                overrides=(("t", PublicationState.CONFIG_ONLY),),
            ),
        ),
    ),
    _family(
        "clip",
        "libreyolo.models.clip.model.LibreCLIP",
        "LibreCLIP",
        "classify",
        (
            _task(
                "classify",
                (("b32", 224), ("b16", 224), ("l14", 224)),
                overrides=(("l14", PublicationState.CONFIG_ONLY),),
            ),
        ),
        optional_extra="clip",
        dependencies=("regex", "ftfy"),
        export_override="custom",
    ),
    _family(
        "convnext",
        "libreyolo.models.convnext.model.LibreConvNeXt",
        "LibreConvNeXt",
        "classify",
        (_task("classify", (("t", 224), ("s", 224), ("b", 224))),),
    ),
    _family(
        "deim",
        "libreyolo.models.deim.model.LibreDEIM",
        "LibreDEIM",
        "detect",
        (
            _task(
                "detect", (("n", 640), ("s", 640), ("m", 640), ("l", 640), ("x", 640))
            ),
        ),
    ),
    _family(
        "deimv2",
        "libreyolo.models.deimv2.model.LibreDEIMv2",
        "LibreDEIMv2",
        "detect",
        (
            _task(
                "detect",
                (
                    ("atto", 320),
                    ("femto", 416),
                    ("pico", 640),
                    ("n", 640),
                    ("s", 640),
                    ("m", 640),
                    ("l", 640),
                    ("x", 640),
                ),
            ),
        ),
    ),
    _family(
        "depth_anything",
        "libreyolo.models.depth_anything.model.LibreDepthAnythingV2",
        "LibreDepthAnythingV2",
        "depth",
        (
            _task(
                "depth",
                (("s", 518), ("b", 518), ("l", 518), ("g", 518)),
                overrides=(
                    ("b", PublicationState.CONFIG_ONLY),
                    ("l", PublicationState.CONFIG_ONLY),
                    ("g", PublicationState.CONFIG_ONLY),
                ),
            ),
        ),
        export_override="custom",
    ),
    _family(
        "depth_anything3",
        "libreyolo.models.depth_anything3.model.LibreDepthAnything3",
        "LibreDepthAnything3",
        "depth",
        (_task("depth", (("l", 504),)),),
        export_override="blocked",
    ),
    _family(
        "dfine",
        "libreyolo.models.dfine.model.LibreDFINE",
        "LibreDFINE",
        "detect",
        (
            _task(
                "detect", (("n", 640), ("s", 640), ("m", 640), ("l", 640), ("x", 640))
            ),
            _task(
                "segment", (("n", 640), ("s", 640), ("m", 640), ("l", 640), ("x", 640))
            ),
        ),
    ),
    _family(
        "dinov2",
        "libreyolo.models.dinov2.model.LibreDINOv2",
        "LibreDINOv2",
        "semantic",
        (
            _task(
                "semantic",
                (("n", 518), ("s", 518), ("m", 518), ("l", 518)),
                publication=PublicationState.UNKNOWN,
            ),
            _task(
                "classify",
                (("n", 224), ("s", 224), ("m", 224), ("l", 224)),
                publication=PublicationState.UNKNOWN,
            ),
        ),
        optional_extra="rfdetr",
        dependency="transformers",
        export_override="custom",
        suffixless_tasks=("semantic",),
    ),
    _family(
        "ec",
        "libreyolo.models.ec.model.LibreEC",
        "LibreEC",
        "detect",
        (
            _task("detect", (("s", 640), ("m", 640), ("l", 640), ("x", 640))),
            _task("pose", (("s", 640), ("m", 640), ("l", 640), ("x", 640))),
            _task("segment", (("s", 640), ("m", 640), ("l", 640), ("x", 640))),
        ),
    ),
    _family(
        "efficientnetv2",
        "libreyolo.models.efficientnetv2.model.LibreEfficientNetV2",
        "LibreEfficientNetV2",
        "classify",
        (_task("classify", (("b0", 224), ("b1", 240), ("b2", 260), ("b3", 300))),),
    ),
    _family(
        "eomt",
        "libreyolo.models.eomt.model.LibreEoMT",
        "LibreEoMT",
        "semantic",
        (
            _task(
                "semantic",
                (("s", 512), ("b", 512), ("l", 512)),
                overrides=(
                    ("s", PublicationState.CONFIG_ONLY),
                    ("b", PublicationState.CONFIG_ONLY),
                ),
            ),
            _task(
                "segment",
                (("s", 640), ("b", 640), ("l", 640)),
                overrides=(
                    ("s", PublicationState.CONFIG_ONLY),
                    ("b", PublicationState.CONFIG_ONLY),
                ),
            ),
            _task("panoptic", (("s", 640), ("b", 640), ("l", 640))),
        ),
        optional_extra="eomt",
        dependency="transformers",
        export_override="custom",
        variants=(
            VariantSpec(
                "1280", "segment", ("l",), PublicationState.PUBLISHED, native_imgsz=1280
            ),
        ),
    ),
    _family(
        "florence2",
        "libreyolo.models.vlm.florence2.LibreFlorence2",
        "LibreFlorence2",
        "detect",
        (_task("detect", (("base", 768), ("large", 768))),),
        factory=FactoryKind.VLM,
        optional_extra="vlm",
        dependency="transformers",
        export_override="blocked",
        download_kind="snapshot",
        factory_models=(
            _factory_model(
                "base",
                "florence-2-base",
                ("florence-2", "florence2", "florence-2-base"),
                "florence-community/Florence-2-base",
            ),
            _factory_model(
                "large",
                "florence-2-large",
                ("florence-2-large",),
                "florence-community/Florence-2-large",
            ),
        ),
    ),
    _family(
        "fomo",
        "libreyolo.models.fomo.model.LibreFOMO",
        "LibreFOMO",
        "point",
        (
            _task(
                "point",
                (("s", 96), ("m", 192), ("l", 224)),
                publication=PublicationState.UNKNOWN,
            ),
        ),
    ),
    _family(
        "grounding_dino",
        "libreyolo.models.openvocab.grounding_dino.LibreGroundingDINO",
        "LibreGroundingDINO",
        "detect",
        (_task("detect", (("t", 800), ("b", 800))),),
        factory=FactoryKind.OPENVOCAB,
        optional_extra="openvocab",
        dependency="transformers",
        export_override="blocked",
        download_kind="snapshot",
        factory_models=(
            _factory_model(
                "t",
                "grounding-dino-tiny",
                (
                    "grounding-dino",
                    "groundingdino",
                    "grounding-dino-tiny",
                    "groundingdino-tiny",
                    "grounding-dino-t",
                    "groundingdino-t",
                ),
                "LibreYOLO/LibreGroundingDINOt",
            ),
            _factory_model(
                "b",
                "grounding-dino-base",
                (
                    "grounding-dino-base",
                    "groundingdino-base",
                    "grounding-dino-b",
                    "groundingdino-b",
                ),
                "LibreYOLO/LibreGroundingDINOb",
            ),
        ),
    ),
    _family(
        "internvl3",
        "libreyolo.models.vlm.internvl3.LibreInternVL3",
        "LibreInternVL3",
        "detect",
        (
            _task(
                "detect",
                (("1b", 448), ("2b", 448), ("8b", 448)),
                publication=PublicationState.GATED,
            ),
        ),
        factory=FactoryKind.VLM,
        optional_extra="vlm",
        dependency="transformers",
        export_override="blocked",
        download_kind="snapshot",
        factory_models=(
            _factory_model(
                "1b",
                "internvl3-1b",
                ("internvl3-1b",),
                "OpenGVLab/InternVL3-1B-hf",
            ),
            _factory_model(
                "2b",
                "internvl3-2b",
                ("internvl3", "internvl3-2b"),
                "OpenGVLab/InternVL3-2B-hf",
            ),
            _factory_model(
                "8b",
                "internvl3-8b",
                ("internvl3-8b",),
                "OpenGVLab/InternVL3-8B-hf",
            ),
        ),
    ),
    _family(
        "kosmos2",
        "libreyolo.models.vlm.kosmos2.LibreKosmos2",
        "LibreKosmos2",
        "detect",
        (_task("detect", (("224", 224),)),),
        factory=FactoryKind.VLM,
        optional_extra="vlm",
        dependency="transformers",
        export_override="blocked",
        download_kind="snapshot",
        factory_models=(
            _factory_model(
                "224",
                "kosmos-2",
                ("kosmos-2", "kosmos2"),
                "microsoft/kosmos-2-patch14-224",
            ),
        ),
    ),
    _family(
        "l2cs",
        "libreyolo.models.l2cs.model.LibreL2CS",
        "LibreL2CS",
        "gaze",
        (
            _task(
                "gaze",
                (
                    ("r18", 448),
                    ("r34", 448),
                    ("r50", 448),
                    ("r101", 448),
                    ("r152", 448),
                ),
                publication=PublicationState.CONFIG_ONLY,
                overrides=(("r50", PublicationState.DIRECT),),
            ),
        ),
        optional_extra="gaze",
        export_override="custom",
        suffixless_tasks=("gaze",),
        download_kind="none",
    ),
    _family(
        "lfm2vl",
        "libreyolo.models.vlm.lfm2.LibreLFM2VL",
        "LibreLFM2VL",
        "detect",
        (
            _task(
                "detect",
                (("450m", 512), ("1.6b", 512)),
                publication=PublicationState.GATED,
            ),
        ),
        factory=FactoryKind.VLM,
        optional_extra="vlm",
        dependency="transformers",
        export_override="blocked",
        download_kind="snapshot",
        factory_models=(
            _factory_model(
                "450m",
                "lfm2-vl-450m",
                ("lfm2-vl", "lfm2-vl-450m"),
                "LiquidAI/LFM2.5-VL-450M",
            ),
            _factory_model(
                "1.6b",
                "lfm2-vl-1.6b",
                ("lfm2-vl-1.6b",),
                "LiquidAI/LFM2.5-VL-1.6B",
            ),
        ),
    ),
    _family(
        "locateanything",
        "libreyolo.models.vlm.locateanything.LibreLocateAnything",
        "LocateAnything",
        "detect",
        (
            _task(
                "detect",
                (("3b", 2500),),
                publication=PublicationState.GATED,
            ),
            _task(
                "point",
                (("3b", 2500),),
                publication=PublicationState.GATED,
            ),
        ),
        factory=FactoryKind.VLM,
        optional_extra="vlm",
        dependency="transformers",
        export_override="blocked",
        download_kind="snapshot",
        dependencies=("decord", "lmdb", "peft"),
        factory_models=(
            _factory_model(
                "3b",
                "locate-anything-3b",
                (
                    "locate-anything",
                    "locateanything",
                    "locate-anything-3b",
                    "locateanything-3b",
                ),
                "nvidia/LocateAnything-3B",
                revision="c32291ca5e996f5a7a485845b4f57a233936bba0",
            ),
        ),
    ),
    _family(
        "mobilenetv4",
        "libreyolo.models.mobilenetv4.model.LibreMobileNetV4",
        "LibreMobileNetV4",
        "classify",
        (_task("classify", (("s", 224), ("m", 224), ("l", 256))),),
    ),
    _family(
        "mobilesam",
        "libreyolo.models.mobilesam.model.LibreMobileSAM",
        "LibreMobileSAM",
        "segment",
        (_task("segment", (("tiny", 1024),)),),
        factory=FactoryKind.SAM,
        optional_extra="sam",
        dependencies=("transformers", "huggingface_hub"),
        export_override="blocked",
        download_kind="snapshot",
        factory_models=(
            _factory_model(
                "tiny",
                "mobilesam",
                (
                    "mobilesam",
                    "mobilesam-tiny",
                    "mobilesam_t",
                    "mobile-sam",
                    "mobile-sam-tiny",
                ),
                "LibreYOLO/LibreMobileSAM",
            ),
        ),
    ),
    _family(
        "nafnet",
        "libreyolo.models.nafnet.model.LibreNAFNet",
        "LibreNAFNet",
        "restore",
        (
            _task(
                "restore",
                (("s", 256), ("l", 256)),
                publication=PublicationState.UNKNOWN,
            ),
        ),
        variants=(VariantSpec("sidd", "restore", ("l",), PublicationState.PUBLISHED),),
    ),
    _family(
        "owlv2",
        "libreyolo.models.openvocab.owlv2.LibreOWLv2",
        "LibreOWLv2",
        "detect",
        (_task("detect", (("b16", 960), ("l14", 1008))),),
        factory=FactoryKind.OPENVOCAB,
        optional_extra="openvocab",
        dependency="transformers",
        export_override="blocked",
        download_kind="snapshot",
        factory_models=(
            _factory_model(
                "b16",
                "owlv2-b16",
                (
                    "owlv2",
                    "owl-v2",
                    "owlv2-base",
                    "owl-v2-base",
                    "owlv2-b16",
                    "owl-v2-b16",
                ),
                "LibreYOLO/LibreOWLv2b16",
            ),
            _factory_model(
                "l14",
                "owlv2-l14",
                (
                    "owlv2-large",
                    "owl-v2-large",
                    "owlv2-l14",
                    "owl-v2-l14",
                ),
                "LibreYOLO/LibreOWLv2l14",
            ),
        ),
    ),
    _family(
        "picodet",
        "libreyolo.models.picodet.model.LibrePICODET",
        "LibrePICODET",
        "detect",
        (_task("detect", (("s", 320), ("m", 416), ("l", 640))),),
    ),
    _family(
        "picosam3",
        "libreyolo.models.picosam3.model.LibrePicoSAM3",
        "LibrePicoSAM3",
        "segment",
        (_task("segment", (("pico", 96),)),),
        factory=FactoryKind.SAM,
        optional_extra="sam",
        dependencies=("huggingface_hub",),
        export_override="custom",
        download_kind="snapshot",
        factory_models=(
            _factory_model(
                "pico",
                "picosam3",
                ("picosam3", "picosam3-pico", "picosam3_pico", "pico-sam3"),
                "LibreYOLO/LibrePicoSAM3",
            ),
        ),
    ),
    _family(
        "pidnet",
        "libreyolo.models.pidnet.model.LibrePIDNet",
        "LibrePIDNet",
        "semantic",
        (_task("semantic", (("s", 1024), ("m", 1024), ("l", 1024))),),
        export_override="custom",
    ),
    _family(
        "ppocr",
        "libreyolo.models.ppocr.model.LibrePPOCR",
        "LibrePPOCR",
        "ocr",
        (_task("ocr", (("t", 960), ("l", 960))),),
    ),
    _family(
        "qwen3vl",
        "libreyolo.models.vlm.qwen3vl.LibreQwen3VL",
        "LibreQwen3VL",
        "detect",
        (_task("detect", (("2b", 1024), ("4b", 1024), ("8b", 1024))),),
        factory=FactoryKind.VLM,
        optional_extra="vlm",
        dependency="transformers",
        export_override="blocked",
        download_kind="snapshot",
        factory_models=(
            _factory_model(
                "2b",
                "qwen3-vl-2b",
                ("qwen3-vl-2b",),
                "Qwen/Qwen3-VL-2B-Instruct",
            ),
            _factory_model(
                "4b",
                "qwen3-vl-4b",
                ("qwen3-vl", "qwen3-vl-4b"),
                "Qwen/Qwen3-VL-4B-Instruct",
            ),
            _factory_model(
                "8b",
                "qwen3-vl-8b",
                ("qwen3-vl-8b",),
                "Qwen/Qwen3-VL-8B-Instruct",
            ),
        ),
    ),
    _family(
        "realesrgan",
        "libreyolo.models.realesrgan.model.LibreRealESRGAN",
        "LibreRealESRGAN",
        "restore",
        (_task("restore", (("x4", 64), ("x2", 64), ("x4t", 64))),),
    ),
    _family(
        "resnet",
        "libreyolo.models.resnet.model.LibreResNet",
        "LibreResNet",
        "classify",
        (_task("classify", (("18", 224), ("34", 224), ("50", 224), ("101", 224))),),
    ),
    _family(
        "rfdetr",
        "libreyolo.models.rfdetr.model.LibreRFDETR",
        "LibreRFDETR",
        "detect",
        (
            _task("detect", (("n", 384), ("s", 512), ("m", 576), ("l", 704))),
            _task(
                "segment",
                (
                    ("n", 312),
                    ("s", 384),
                    ("m", 432),
                    ("l", 504),
                    ("x", 624),
                    ("xx", 768),
                ),
            ),
            _task("pose", (("x", 576),)),
            _task("obb", (("n", 384), ("s", 512), ("m", 576), ("l", 704))),
        ),
        optional_extra="rfdetr",
        dependency="transformers",
        export_override="custom",
    ),
    _family(
        "rtdetr",
        "libreyolo.models.rtdetr.model.LibreRTDETR",
        "LibreRTDETR",
        "detect",
        (
            _task(
                "detect",
                (
                    ("r18", 640),
                    ("r34", 640),
                    ("r50", 640),
                    ("r50m", 640),
                    ("r101", 640),
                    ("l", 640),
                    ("x", 640),
                ),
            ),
        ),
        export_override="custom",
    ),
    _family(
        "rtdetrv2",
        "libreyolo.models.rtdetrv2.model.LibreRTDETRv2",
        "LibreRTDETRv2",
        "detect",
        (
            _task(
                "detect",
                (
                    ("r18", 640),
                    ("r34", 640),
                    ("r50", 640),
                    ("r50m", 640),
                    ("r101", 640),
                ),
            ),
        ),
        export_override="custom",
    ),
    _family(
        "rtdetrv4",
        "libreyolo.models.rtdetrv4.model.LibreRTDETRv4",
        "LibreRTDETRv4",
        "detect",
        (_task("detect", (("s", 640), ("m", 640), ("l", 640), ("x", 640))),),
    ),
    _family(
        "rtmdet",
        "libreyolo.models.rtmdet.model.LibreRTMDet",
        "LibreRTMDet",
        "detect",
        (
            _task(
                "detect", (("t", 640), ("s", 640), ("m", 640), ("l", 640), ("x", 640))
            ),
            _task(
                "segment", (("t", 640), ("s", 640), ("m", 640), ("l", 640), ("x", 640))
            ),
        ),
    ),
    _family(
        "sam",
        "libreyolo.models.sam.model.LibreSAM1",
        "LibreSAM",
        "segment",
        (_task("segment", (("base", 1024), ("large", 1024), ("huge", 1024))),),
        factory=FactoryKind.SAM,
        optional_extra="sam",
        dependency="transformers",
        export_override="blocked",
        download_kind="snapshot",
        factory_models=(
            _factory_model(
                "base",
                "base",
                ("base", "b", "sam-base", "sam_b"),
                "facebook/sam-vit-base",
            ),
            _factory_model(
                "large",
                "large",
                ("large", "l", "sam-large", "sam_l"),
                "facebook/sam-vit-large",
            ),
            _factory_model(
                "huge",
                "huge",
                ("huge", "h", "sam-huge", "sam_h"),
                "facebook/sam-vit-huge",
            ),
        ),
    ),
    _family(
        "sam2",
        "libreyolo.models.sam.sam2.LibreSAM2",
        "LibreSAM2",
        "segment",
        (
            _task(
                "segment",
                (("tiny", 1024), ("small", 1024), ("base-plus", 1024), ("large", 1024)),
            ),
        ),
        factory=FactoryKind.SAM,
        optional_extra="sam",
        dependency="transformers",
        export_override="blocked",
        download_kind="snapshot",
        factory_models=(
            _factory_model(
                "tiny",
                "sam2-tiny",
                ("sam2-tiny", "sam2-t", "sam2_t"),
                "LibreYOLO/LibreSAM2tiny",
            ),
            _factory_model(
                "small",
                "sam2-small",
                ("sam2-small", "sam2-s", "sam2_s"),
                "LibreYOLO/LibreSAM2small",
            ),
            _factory_model(
                "base-plus",
                "sam2-base-plus",
                ("sam2-base-plus", "sam2-baseplus", "sam2-bp", "sam2_bp"),
                "LibreYOLO/LibreSAM2base-plus",
            ),
            _factory_model(
                "large",
                "sam2-large",
                ("sam2-large", "sam2-l", "sam2_l"),
                "LibreYOLO/LibreSAM2large",
            ),
        ),
    ),
    _family(
        "sam3",
        "libreyolo.models.sam.sam3.LibreSAM3",
        "LibreSAM3",
        "segment",
        (_task("segment", (("large", 1008),), publication=PublicationState.GATED),),
        factory=FactoryKind.SAM,
        optional_extra="sam",
        dependency="transformers",
        export_override="blocked",
        download_kind="snapshot",
        factory_models=(
            _factory_model(
                "large",
                "sam3",
                ("sam3", "sam-3", "sam3-large"),
                "facebook/sam3",
            ),
        ),
    ),
    _family(
        "segformer",
        "libreyolo.models.segformer.model.LibreSegformer",
        "LibreSegformer",
        "semantic",
        (
            _task(
                "semantic",
                (
                    ("b0", 512),
                    ("b1", 512),
                    ("b2", 512),
                    ("b3", 512),
                    ("b4", 512),
                    ("b5", 640),
                ),
            ),
        ),
        export_override="blocked",
    ),
    _family(
        "siglip2",
        "libreyolo.models.siglip2.model.LibreSigLIP2",
        "LibreSigLIP2",
        "classify",
        (_task("classify", (("b16", 256), ("so400m", 384))),),
        optional_extra="siglip2",
        dependency="sentencepiece",
        export_override="custom",
    ),
    _family(
        "smolvlm2",
        "libreyolo.models.vlm.smolvlm.LibreSmolVLM2",
        "LibreSmolVLM2",
        "detect",
        (_task("detect", (("2.2b", 512), ("500m", 512))),),
        factory=FactoryKind.VLM,
        optional_extra="vlm",
        dependency="transformers",
        dependencies=("num2words",),
        export_override="blocked",
        download_kind="snapshot",
        factory_models=(
            _factory_model(
                "2.2b",
                "smolvlm2-2.2b",
                ("smolvlm2", "smolvlm2-2.2b"),
                "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            ),
            _factory_model(
                "500m",
                "smolvlm2-500m",
                ("smolvlm2-500m",),
                "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            ),
        ),
    ),
    _family(
        "swinir",
        "libreyolo.models.swinir.model.LibreSwinIR",
        "LibreSwinIR",
        "restore",
        (_task("restore", (("s", 64), ("m", 64), ("l", 64))),),
    ),
    _family(
        "yolo1",
        "libreyolo.models.yolo1.model.LibreYOLO1",
        "LibreYOLO1",
        "detect",
        (
            _task(
                "detect", (("t", 448), ("b", 448)), publication=PublicationState.UNKNOWN
            ),
        ),
    ),
    _family(
        "yolo2",
        "libreyolo.models.yolo2.model.LibreYOLO2",
        "LibreYOLO2",
        "detect",
        (_task("detect", (("t", 416), ("b", 608))),),
    ),
    _family(
        "yolo3",
        "libreyolo.models.yolo3.model.LibreYOLO3",
        "LibreYOLO3",
        "detect",
        (_task("detect", (("t", 416), ("b", 416), ("spp", 608))),),
    ),
    _family(
        "yolo4",
        "libreyolo.models.yolo4.model.LibreYOLO4",
        "LibreYOLO4",
        "detect",
        (_task("detect", (("t", 416), ("b", 608))),),
    ),
    _family(
        "yolo7",
        "libreyolo.models.yolo7.model.LibreYOLO7",
        "LibreYOLO7",
        "detect",
        (_task("detect", (("b", 640),)),),
    ),
    _family(
        "yolo9",
        "libreyolo.models.yolo9.model.LibreYOLO9",
        "LibreYOLO9",
        "detect",
        (_task("detect", (("t", 640), ("s", 640), ("m", 640), ("c", 640))),),
    ),
    _family(
        "yolo9_e2e",
        "libreyolo.models.yolo9_e2e.model.LibreYOLO9E2E",
        "LibreYOLO9E2E",
        "detect",
        (_task("detect", (("t", 640), ("s", 640), ("m", 640), ("c", 640))),),
    ),
    _family(
        "yolo9_p2",
        "libreyolo.models.yolo9_p2.model.LibreYOLO9P2",
        "LibreYOLO9P2",
        "detect",
        (
            _task(
                "detect",
                (("t", 640), ("s", 640)),
                publication=PublicationState.CONFIG_ONLY,
            ),
        ),
        variants=(
            VariantSpec(
                "visdrone",
                "detect",
                ("t", "s"),
                PublicationState.GATED,
                download_kind="none",
            ),
        ),
    ),
    _family(
        "yolonas",
        "libreyolo.models.yolonas.model.LibreYOLONAS",
        "LibreYOLONAS",
        "detect",
        (
            _task(
                "detect",
                (("s", 640), ("m", 640), ("l", 640)),
                publication=PublicationState.DIRECT,
            ),
            _task(
                "pose",
                (("n", 640), ("s", 640), ("m", 640), ("l", 640)),
                publication=PublicationState.DIRECT,
            ),
        ),
        download_kind="family",
    ),
    _family(
        "yolox",
        "libreyolo.models.yolox.model.LibreYOLOX",
        "LibreYOLOX",
        "detect",
        (
            _task(
                "detect",
                (
                    ("n", 416),
                    ("t", 416),
                    ("s", 640),
                    ("m", 640),
                    ("l", 640),
                    ("x", 640),
                ),
            ),
        ),
    ),
    _family(
        "zipdepth",
        "libreyolo.models.zipdepth.model.LibreZipDepth",
        "LibreZipDepth",
        "depth",
        (_task("depth", (("b", 384), ("bnpu", 384))),),
    ),
)


def _canonical_filename(
    family: FamilySpec, task: str, size: str, variant: str | None = None
) -> str | None:
    if family.factory is not FactoryKind.CHECKPOINT:
        return None
    suffix = None if task in family.suffixless_tasks else task_to_suffix(task)
    task_part = f"-{suffix}" if suffix else ""
    variant_part = f"-{variant}" if variant else ""
    return f"{family.filename_prefix}{size}{task_part}{variant_part}{family.weight_ext}"


def _download_details(
    family: FamilySpec,
    filename: str | None,
    publication: PublicationState,
    *,
    override_kind: str | None = None,
    repository: str | None = None,
) -> tuple[str, str | None]:
    if publication in {PublicationState.CONFIG_ONLY, PublicationState.UNKNOWN}:
        return "none", None
    if family.factory is not FactoryKind.CHECKPOINT:
        if repository is None:
            return "none", None
        return "snapshot", f"https://huggingface.co/{repository}"
    kind = override_kind or family.download_kind
    if kind == "none":
        return "none", None
    if kind == "family":
        return "family", None
    if kind == "hf" and filename is not None:
        stem = filename[: -len(family.weight_ext)]
        return (
            "hf",
            f"https://huggingface.co/LibreYOLO/{stem}/resolve/main/{filename}",
        )
    return "none", None


def _build_indices():
    families: dict[str, FamilySpec] = {}
    artifacts: dict[tuple[str, str, str, str | None], ArtifactSpec] = {}
    filenames: dict[str, ArtifactSpec] = {}
    aliases: dict[str, ArtifactSpec] = {}
    factory_aliases: dict[
        tuple[FactoryKind, str], tuple[FamilySpec, FactoryModelSpec]
    ] = {}

    for family in MODEL_FAMILIES:
        if family.family in families:
            raise RuntimeError(f"Duplicate model family in manifest: {family.family!r}")
        families[family.family] = family
        factory_models = {item.size: item for item in family.factory_models}
        if len(factory_models) != len(family.factory_models):
            raise RuntimeError(f"Duplicate sibling-factory size for {family.family!r}")
        if family.factory is FactoryKind.CHECKPOINT and factory_models:
            raise RuntimeError(
                f"Checkpoint family {family.family!r} declares factory models"
            )
        declared_sizes = {size.code for task in family.tasks for size in task.sizes}
        if family.factory is not FactoryKind.CHECKPOINT:
            if set(factory_models) != declared_sizes:
                raise RuntimeError(
                    f"{family.family!r} sibling-factory sizes do not match its "
                    "task size declarations"
                )
            for factory_model in factory_models.values():
                if (
                    factory_model.model not in factory_model.aliases
                    or not factory_model.repository
                ):
                    raise RuntimeError(
                        f"Invalid sibling-factory model for {family.family!r}/"
                        f"{factory_model.size!r}"
                    )
        task_names = {task.task for task in family.tasks}
        if family.default_task not in task_names:
            raise RuntimeError(
                f"{family.family!r} default task {family.default_task!r} is not declared"
            )
        for task in family.tasks:
            normalized_task = normalize_task(task.task)
            if normalized_task != task.task:
                raise RuntimeError(
                    f"{family.family!r} uses noncanonical task {task.task!r}"
                )
            for size in task.sizes:
                if not size.code or size.native_imgsz <= 0:
                    raise RuntimeError(
                        f"Invalid size declaration for {family.family}/{task.task}: {size!r}"
                    )
                factory_model = factory_models.get(size.code)
                if (
                    family.factory is not FactoryKind.CHECKPOINT
                    and factory_model is None
                ):
                    raise RuntimeError(
                        f"{family.family!r} size {size.code!r} has no public "
                        "factory model declaration"
                    )
                filename = _canonical_filename(family, task.task, size.code)
                artifact_aliases: list[str] = []
                if family.factory is FactoryKind.CHECKPOINT:
                    suffix = task_to_suffix(task.task)
                    if task.task == family.default_task:
                        artifact_aliases.append(f"{family.family}-{size.code}")
                    if suffix:
                        artifact_aliases.append(f"{family.family}-{size.code}-{suffix}")
                elif factory_model is not None:
                    artifact_aliases.extend(factory_model.aliases)
                download_kind, download_url = _download_details(
                    family,
                    filename,
                    size.publication,
                    repository=(
                        factory_model.repository if factory_model is not None else None
                    ),
                )
                artifact = ArtifactSpec(
                    family=family.family,
                    size=size.code,
                    task=task.task,
                    native_imgsz=size.native_imgsz,
                    factory=family.factory,
                    canonical_filename=filename,
                    publication=size.publication,
                    download_kind=download_kind,
                    download_url=download_url,
                    aliases=tuple(artifact_aliases),
                    factory_model=(
                        factory_model.model if factory_model is not None else None
                    ),
                    repository=(
                        factory_model.repository if factory_model is not None else None
                    ),
                    revision=(
                        factory_model.revision if factory_model is not None else None
                    ),
                )
                key = (family.family, size.code, task.task, None)
                if key in artifacts:
                    raise RuntimeError(f"Duplicate model artifact in manifest: {key!r}")
                artifacts[key] = artifact
                if filename is not None:
                    filename_key = filename.casefold()
                    if filename_key in filenames:
                        raise RuntimeError(
                            f"Duplicate canonical model filename: {filename!r}"
                        )
                    filenames[filename_key] = artifact
                if family.factory is FactoryKind.CHECKPOINT:
                    for name in artifact_aliases:
                        alias_key = name.casefold()
                        previous = aliases.get(alias_key)
                        if previous is not None and previous != artifact:
                            raise RuntimeError(f"Duplicate CLI model alias: {name!r}")
                        aliases[alias_key] = artifact
                elif factory_model is not None:
                    for name in factory_model.aliases:
                        alias_key = (family.factory, name.casefold())
                        previous = factory_aliases.get(alias_key)
                        selection = (family, factory_model)
                        if previous is not None and previous != selection:
                            raise RuntimeError(
                                f"Duplicate {family.factory.value} model alias: "
                                f"{name!r}"
                            )
                        factory_aliases[alias_key] = selection

        for variant in family.variants:
            for size_code in variant.sizes:
                base = artifacts.get((family.family, size_code, variant.task, None))
                if base is None:
                    raise RuntimeError(
                        f"Variant {family.family}/{variant.name} targets unknown "
                        f"{variant.task}/{size_code} artifact"
                    )
                filename = _canonical_filename(
                    family, variant.task, size_code, variant.name
                )
                download_kind, download_url = _download_details(
                    family,
                    filename,
                    variant.publication,
                    override_kind=variant.download_kind,
                )
                artifact = ArtifactSpec(
                    family=family.family,
                    size=size_code,
                    task=variant.task,
                    native_imgsz=variant.native_imgsz or base.native_imgsz,
                    factory=family.factory,
                    canonical_filename=filename,
                    publication=variant.publication,
                    download_kind=download_kind,
                    download_url=download_url,
                    variant=variant.name,
                )
                key = (family.family, size_code, variant.task, variant.name)
                artifacts[key] = artifact
                if filename is not None:
                    filename_key = filename.casefold()
                    if filename_key in filenames:
                        raise RuntimeError(
                            f"Duplicate canonical model filename: {filename!r}"
                        )
                    filenames[filename_key] = artifact

    return (
        MappingProxyType(families),
        MappingProxyType(artifacts),
        MappingProxyType(filenames),
        MappingProxyType(aliases),
        MappingProxyType(factory_aliases),
    )


(
    FAMILY_BY_ID,
    ARTIFACT_BY_KEY,
    ARTIFACT_BY_FILENAME,
    CLI_MODEL_ALIASES,
    FACTORY_MODEL_ALIASES,
) = _build_indices()

FACTORY_DEFAULT_MODELS = MappingProxyType(
    {
        FactoryKind.SAM: "base",
        FactoryKind.VLM: "qwen3-vl-4b",
        FactoryKind.OPENVOCAB: "grounding-dino-tiny",
    }
)
for _factory, _model in FACTORY_DEFAULT_MODELS.items():
    if (_factory, _model.casefold()) not in FACTORY_MODEL_ALIASES:
        raise RuntimeError(f"Default {_factory.value} model {_model!r} is not declared")


def iter_family_specs(
    factory: FactoryKind | None = None,
) -> Iterator[FamilySpec]:
    """Iterate families in stable manifest order."""
    for family in MODEL_FAMILIES:
        if factory is None or family.factory is factory:
            yield family


def get_family_spec(family: str) -> FamilySpec | None:
    """Return a family declaration without importing its implementation."""
    return FAMILY_BY_ID.get(str(family).strip().lower())


def get_artifact_spec(
    family: str,
    size: str,
    task: str,
    *,
    variant: str | None = None,
) -> ArtifactSpec | None:
    """Return an exact family/task/size artifact declaration."""
    try:
        canonical_task = normalize_task(task)
    except ValueError:
        return None
    if canonical_task is None:
        return None
    normalized_variant = str(variant).lower() if variant is not None else None
    return ARTIFACT_BY_KEY.get(
        (str(family).lower(), str(size).lower(), canonical_task, normalized_variant)
    )


def match_weight_filename(model: str | os.PathLike[str]) -> ArtifactSpec | None:
    """Match a complete canonical filename, case-insensitively."""
    filename = Path(os.fspath(model)).name.casefold()
    return ARTIFACT_BY_FILENAME.get(filename)


def resolve_cli_model(name: str) -> ArtifactSpec | None:
    """Resolve one generic CLI alias without importing model modules."""
    return CLI_MODEL_ALIASES.get(str(name).casefold())


def resolve_factory_model(
    factory: FactoryKind | str,
    model: str | None = None,
) -> tuple[FamilySpec, FactoryModelSpec] | None:
    """Resolve a sibling-factory model identifier without importing it."""
    try:
        factory_kind = FactoryKind(factory)
    except ValueError:
        return None
    if factory_kind is FactoryKind.CHECKPOINT:
        return None
    if model is None:
        model = FACTORY_DEFAULT_MODELS.get(factory_kind)
    if model is None:
        return None
    return FACTORY_MODEL_ALIASES.get((factory_kind, str(model).strip().casefold()))


def load_family_class(spec: FamilySpec | str):
    """Import exactly the implementation class named by a manifest entry."""
    if isinstance(spec, str):
        resolved = get_family_spec(spec)
        if resolved is None:
            raise KeyError(f"Unknown model family: {spec!r}")
        spec = resolved
    module_name, class_name = spec.class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


__all__ = [
    "ARTIFACT_BY_FILENAME",
    "ARTIFACT_BY_KEY",
    "CLI_MODEL_ALIASES",
    "FACTORY_DEFAULT_MODELS",
    "FACTORY_MODEL_ALIASES",
    "FAMILY_BY_ID",
    "MODEL_FAMILIES",
    "ArtifactSpec",
    "FactoryKind",
    "FactoryModelSpec",
    "FamilySpec",
    "PublicationState",
    "SizeSpec",
    "TaskSpec",
    "VariantSpec",
    "get_artifact_spec",
    "get_family_spec",
    "iter_family_specs",
    "load_family_class",
    "match_weight_filename",
    "resolve_cli_model",
    "resolve_factory_model",
]
