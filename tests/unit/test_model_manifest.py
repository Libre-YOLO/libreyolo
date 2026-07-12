"""Regression tests for deterministic public model discovery."""

from dataclasses import FrozenInstanceError

import pytest

from libreyolo.cli.config import (
    detect_family_from_weight_filename,
    resolve_model_name,
)
from libreyolo.models.manifest import (
    ARTIFACT_BY_FILENAME,
    CLI_MODEL_ALIASES,
    FACTORY_DEFAULT_MODELS,
    FACTORY_MODEL_ALIASES,
    FAMILY_BY_ID,
    FactoryKind,
    PublicationState,
    get_artifact_spec,
    match_weight_filename,
    resolve_factory_model,
)

pytestmark = pytest.mark.unit


def test_manifest_indices_and_records_are_immutable():
    with pytest.raises(TypeError):
        FAMILY_BY_ID["new-family"] = object()
    with pytest.raises(TypeError):
        ARTIFACT_BY_FILENAME["new.pt"] = object()
    with pytest.raises(FrozenInstanceError):
        FAMILY_BY_ID["yolox"].family = "changed"


def test_every_generic_alias_round_trips_to_an_exact_canonical_filename():
    for alias, artifact in CLI_MODEL_ALIASES.items():
        assert resolve_model_name(alias) == artifact.canonical_filename
        assert match_weight_filename(artifact.canonical_filename) == artifact
        assert detect_family_from_weight_filename(artifact.canonical_filename) == (
            artifact.family
        )


@pytest.mark.parametrize(
    ("alias", "filename"),
    [
        ("mobilenetv4-s", "LibreMobileNetV4s-cls.pt"),
        ("convnext-t", "LibreConvNeXtt-cls.pt"),
        ("fomo-s", "LibreFOMOs-point.pt"),
        ("nafnet-l", "LibreNAFNetl-restore.pt"),
        ("ppocr-t", "LibrePPOCRt-ocr.pt"),
        ("l2cs-r50", "LibreL2CSr50.pt"),
        ("dinov2-n", "LibreDINOv2n.pt"),
        ("dinov2-n-sem", "LibreDINOv2n.pt"),
        ("dinov2-n-cls", "LibreDINOv2n-cls.pt"),
    ],
)
def test_task_aware_aliases_use_canonical_filenames(alias, filename):
    assert resolve_model_name(alias) == filename


def test_dinov2_inventory_contains_both_task_specific_size_maps():
    semantic = get_artifact_spec("dinov2", "n", "semantic")
    classify = get_artifact_spec("dinov2", "n", "classify")

    assert semantic is not None and semantic.native_imgsz == 518
    assert classify is not None and classify.native_imgsz == 224
    assert semantic.canonical_filename == "LibreDINOv2n.pt"
    assert classify.canonical_filename == "LibreDINOv2n-cls.pt"


def test_canonical_filename_match_is_complete_and_case_insensitive():
    artifact = match_weight_filename("weights/LIBREMOBILENETV4S-CLS.PT")

    assert artifact is not None and artifact.family == "mobilenetv4"
    assert match_weight_filename("LibreMobileNetV4s-cls.pt.bak") is None
    assert match_weight_filename("LibreDINOv2n-seg.pt") is None


def test_separate_factory_aliases_and_defaults_resolve_without_generic_cli():
    expected = {
        FactoryKind.SAM: ("base", "sam", "base"),
        FactoryKind.VLM: ("qwen3-vl-4b", "qwen3vl", "4b"),
        FactoryKind.OPENVOCAB: (
            "grounding-dino-tiny",
            "grounding_dino",
            "t",
        ),
    }
    for factory, (default, family, size) in expected.items():
        assert FACTORY_DEFAULT_MODELS[factory] == default
        selection = resolve_factory_model(factory)
        assert selection is not None
        family_spec, factory_model = selection
        assert family_spec.family == family
        assert factory_model.size == size

    assert resolve_factory_model("vlm", "locateanything-3b")[1].revision == (
        "c32291ca5e996f5a7a485845b4f57a233936bba0"
    )
    assert not any(
        artifact.factory is not FactoryKind.CHECKPOINT
        for artifact in CLI_MODEL_ALIASES.values()
    )


def test_sibling_factory_aliases_have_snapshot_identity():
    for (_factory, _alias), (family, selection) in FACTORY_MODEL_ALIASES.items():
        assert selection.repository
        artifact = get_artifact_spec(
            family.family,
            selection.size,
            family.default_task,
        )
        assert artifact is not None
        assert artifact.factory_model == selection.model
        assert artifact.repository == selection.repository
        assert artifact.invocation.startswith(family.public_entrypoint + "(")


def test_unpublished_or_restricted_checkpoints_have_no_shared_download_route():
    from libreyolo.models.base.model import BaseModel
    from libreyolo.models.manifest import load_family_class

    cases = [
        ("depth_anything", "b", "depth", PublicationState.CONFIG_ONLY),
        ("l2cs", "r50", "gaze", PublicationState.DIRECT),
        ("yolo1", "t", "detect", PublicationState.UNKNOWN),
        ("yolo9_p2", "s", "detect", PublicationState.CONFIG_ONLY),
    ]
    for family, size, task, publication in cases:
        artifact = get_artifact_spec(family, size, task)
        assert artifact is not None
        assert artifact.publication is publication
        assert artifact.downloadable is False
        assert artifact.download_url is None
        cls = load_family_class(family)
        assert (
            BaseModel.get_download_url.__func__(cls, artifact.canonical_filename) is None
        )


def test_public_inventory_does_not_change_with_runtime_registry(monkeypatch):
    from libreyolo.models.base.model import BaseModel
    from libreyolo.models.inventory import collect_model_inventory

    before = collect_model_inventory()
    monkeypatch.setattr(BaseModel, "_registry", [])

    assert collect_model_inventory() == before
    assert resolve_model_name("dinov2-n") == "LibreDINOv2n.pt"
