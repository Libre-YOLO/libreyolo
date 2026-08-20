"""LibreVLM: vision-language models used as open-vocabulary object detectors.

User-facing entry point is the ``LibreVLM(...)`` factory, a sibling to the
``LibreYOLO(...)`` factory. It returns a model instance that behaves like any
YOLO model (predict on image/folder/video, track, identical ``Results``), but
is backed by a generative VLM, so the class list is open vocabulary.

    from libreyolo import LibreVLM
    model = LibreVLM()                       # defaults to Qwen3-VL-4B, autodownloads
    model.set_classes(["pink car", "wheel"]) # open vocabulary: any words
    results = model.predict("image.jpg")     # same Results as a YOLO model
    results = model.predict("folder/")       # folders, video, track() all work
    text = model.chat("image.jpg", "How many cars are pink?")  # raw escape hatch

See ``docs/librevlm_design.md`` and ``docs/adr/0002-librevlm-contract.md``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Type

from .artifact import (
    VLMArtifactError,
    VLMArtifactInfo,
    build_vlm_artifact,
    create_vlm_publication_evidence_template,
    read_vlm_artifact_manifest,
    validate_vlm_artifact,
    validate_vlm_base_snapshot,
)
from .base import LibreVLMModel
from .florence2 import LibreFlorence2
from .gemma4 import LibreGemma4
from .hub import (
    VLMBaseSnapshotInfo,
    VLMHubRef,
    download_vlm_artifact,
    ensure_vlm_base_snapshot,
    inspect_vlm_hub_artifact,
    parse_vlm_hub_uri,
    push_vlm_artifact,
)
from .internvl3 import LibreInternVL3
from .kosmos2 import LibreKosmos2
from .lfm2 import LibreLFM2VL
from .locateanything import LibreLocateAnything
from .moondream import LibreMoondream
from .northmicro import LibreNorthMicroVision
from .qwen3vl import LibreQwen3VL
from .smolvlm import LibreSmolVLM2

# alias -> (family class, size)
_ALIASES: Dict[str, Tuple[Type[LibreVLMModel], str]] = {
    "qwen3-vl": (LibreQwen3VL, "4b"),
    "qwen3-vl-2b": (LibreQwen3VL, "2b"),
    "qwen3-vl-4b": (LibreQwen3VL, "4b"),
    "qwen3-vl-8b": (LibreQwen3VL, "8b"),
    "lfm2-vl": (LibreLFM2VL, "450m"),
    "lfm2-vl-450m": (LibreLFM2VL, "450m"),
    "lfm2-vl-1.6b": (LibreLFM2VL, "1.6b"),
    "lfm2-vl-3b": (LibreLFM2VL, "3b"),
    "north-micro-vision": (LibreNorthMicroVision, "2.4b"),
    "north-micro-vision-2.4b": (LibreNorthMicroVision, "2.4b"),
    "northmicrovision": (LibreNorthMicroVision, "2.4b"),
    "internvl3": (LibreInternVL3, "2b"),
    "internvl3-1b": (LibreInternVL3, "1b"),
    "internvl3-2b": (LibreInternVL3, "2b"),
    "internvl3-8b": (LibreInternVL3, "8b"),
    "smolvlm2": (LibreSmolVLM2, "2.2b"),
    "smolvlm2-2.2b": (LibreSmolVLM2, "2.2b"),
    "smolvlm2-500m": (LibreSmolVLM2, "500m"),
    "florence-2": (LibreFlorence2, "base"),
    "florence2": (LibreFlorence2, "base"),
    "florence-2-base": (LibreFlorence2, "base"),
    "florence-2-large": (LibreFlorence2, "large"),
    "kosmos-2": (LibreKosmos2, "224"),
    "kosmos2": (LibreKosmos2, "224"),
    "locate-anything": (LibreLocateAnything, "3b"),
    "locateanything": (LibreLocateAnything, "3b"),
    "locate-anything-3b": (LibreLocateAnything, "3b"),
    "locateanything-3b": (LibreLocateAnything, "3b"),
    "gemma-4": (LibreGemma4, "e4b"),
    "gemma4": (LibreGemma4, "e4b"),
    "gemma-4-e4b": (LibreGemma4, "e4b"),
    "gemma4-e4b": (LibreGemma4, "e4b"),
    "gemma-4-e2b": (LibreGemma4, "e2b"),
    "gemma4-e2b": (LibreGemma4, "e2b"),
    "moondream": (LibreMoondream, "2"),
    "moondream-2": (LibreMoondream, "2"),
    "moondream2": (LibreMoondream, "2"),
    "moondream-3": (LibreMoondream, "3"),
    "moondream3": (LibreMoondream, "3"),
}

# SenseNova-Vision lives in ``models/sensenova`` (it vendors its own
# architecture) and imports this package's base class, so its aliases resolve
# lazily to avoid a circular import at package-init time.
_LAZY_ALIASES: Dict[str, str] = {
    "sensenova-vision": "7b",
    "sensenova-vision-7b": "7b",
    "sensenovavision": "7b",
}
_MODUS_ALIASES: Dict[str, str] = {
    "libremodus": "14b-a7b",
    "libremodus-14b-a7b": "14b-a7b",
    "modus": "14b-a7b",
    "modus-14b-a7b": "14b-a7b",
}

_DEFAULT_MODEL = "qwen3-vl-4b"


@dataclass(frozen=True)
class VLMReference:
    """Side-effect-free metadata for a recognized LibreVLM reference."""

    family: str | None
    size: str | None
    trainable: bool
    trainable_sizes: tuple[str, ...]
    checkpoint: bool
    hub: VLMHubRef | None = None

    @property
    def remote(self) -> bool:
        """Whether this reference names an immutable remote artifact."""
        return self.hub is not None


def _family_training_metadata(family: str) -> tuple[bool, tuple[str, ...]]:
    """Return training capability without constructing a model."""
    for family_cls, _size in _ALIASES.values():
        if family_cls.FAMILY == family:
            return bool(family_cls.TRAINABLE), tuple(family_cls.TRAINABLE_SIZES)

    # These families are imported lazily to avoid circular package imports.
    # Both currently inherit the non-trainable LibreVLMModel defaults.
    if family in {"sensenovavision", "libremodus"}:
        return False, ()
    return False, ()


def _reference(family: str, size: str, *, checkpoint: bool) -> VLMReference:
    trainable, trainable_sizes = _family_training_metadata(family)
    return VLMReference(
        family=family,
        size=size,
        trainable=trainable,
        trainable_sizes=trainable_sizes,
        checkpoint=checkpoint,
    )


def get_vlm_aliases() -> tuple[str, ...]:
    """Return every accepted LibreVLM alias in stable sorted order."""
    return tuple(sorted(set(_ALIASES) | set(_LAZY_ALIASES) | set(_MODUS_ALIASES)))


def inspect_vlm_reference(model) -> VLMReference | None:
    """Inspect a VLM alias, checkpoint, or immutable VLM Hub URI without loading.

    Unknown aliases and directories without a VLM contract return ``None``.
    A directory that carries ``libreyolo_vlm.json`` is always parsed through
    the strict checkpoint-contract reader, so malformed contracts raise their
    validation error instead of falling through to another model factory.

    This function only parses remote URI syntax or reads the small local
    contract file when present. It never constructs a model, resolves a
    Hugging Face repository, or downloads weights.
    """
    from .hub import VLM_HUB_URI_PREFIX, parse_vlm_hub_uri
    from .training.checkpoint import (
        CONTRACT_FILENAME,
        read_contract,
        validate_vlm_checkpoint_artifact,
    )

    if isinstance(model, str) and model.startswith(VLM_HUB_URI_PREFIX):
        hub = parse_vlm_hub_uri(model)
        return VLMReference(
            family=None,
            size=None,
            trainable=False,
            trainable_sizes=(),
            checkpoint=True,
            hub=hub,
        )

    try:
        path = Path(model)
    except (TypeError, ValueError):
        return None

    contract_path = path / CONTRACT_FILENAME
    if path.is_dir() and (contract_path.exists() or contract_path.is_symlink()):
        contract = read_contract(path)
        validate_vlm_checkpoint_artifact(path)
        return _reference(contract["family"], contract["size"], checkpoint=True)

    key = str(model).strip().lower()
    match = _ALIASES.get(key)
    if match is not None:
        family_cls, size = match
        return _reference(family_cls.FAMILY, size, checkpoint=False)
    if key in _LAZY_ALIASES:
        return _reference("sensenovavision", _LAZY_ALIASES[key], checkpoint=False)
    if key in _MODUS_ALIASES:
        return _reference("libremodus", _MODUS_ALIASES[key], checkpoint=False)
    return None


def _load_checkpoint(path, **kwargs) -> LibreVLMModel:
    """Load a fine-tune checkpoint directory produced by ``train()``."""
    from .training.checkpoint import read_contract, validate_vlm_checkpoint_artifact

    contract = read_contract(path)
    validate_vlm_checkpoint_artifact(path)
    family_classes = {cls.FAMILY: cls for cls, _size in _ALIASES.values()}
    family_cls = family_classes.get(contract["family"])
    if family_cls is None:
        raise ValueError(
            f"VLM checkpoint {path} was trained on unknown family "
            f"{contract['family']!r}; this libreyolo build knows "
            f"{sorted(family_classes)}."
        )
    kwargs.setdefault("names", list(contract["names"]))
    return family_cls(size=contract["size"], checkpoint_dir=str(path), **kwargs)


def _load_remote_checkpoint(source: str, **kwargs) -> LibreVLMModel:
    """Download one immutable Hub artifact into an isolated lifetime directory."""
    from .artifact import validate_vlm_artifact, validate_vlm_base_snapshot
    from .hub import download_vlm_artifact, ensure_vlm_base_snapshot

    token = kwargs.pop("token", None)
    local_files_only = kwargs.pop("local_files_only", False)
    temporary = tempfile.TemporaryDirectory(prefix="libreyolo-vlm-remote-")
    try:
        temporary_root = Path(temporary.name).resolve(strict=True)
        artifact = download_vlm_artifact(
            source,
            temporary_root / "artifact",
            token=token,
            local_files_only=local_files_only,
        )
        base_snapshot = ensure_vlm_base_snapshot(
            artifact,
            token=token,
            local_files_only=local_files_only,
        )
        validate_vlm_base_snapshot(base_snapshot.root, base_snapshot.identity)
        model = _load_checkpoint(artifact.root, **kwargs)
        revalidated_artifact = validate_vlm_artifact(artifact.root)
        if (
            revalidated_artifact.aggregate_sha256 != artifact.aggregate_sha256
            or revalidated_artifact.files != artifact.files
            or revalidated_artifact.manifest != artifact.manifest
        ):
            raise ValueError("VLM Hub artifact changed during model construction")
        validate_vlm_base_snapshot(base_snapshot.root, base_snapshot.identity)
    except Exception:
        temporary.cleanup()
        raise

    # Checkpoint loading is synchronous today, but retaining the validated
    # directory for the wrapper lifetime also keeps future lazy processor or
    # adapter reads safe. TemporaryDirectory removes it when the model dies.
    model._vlm_remote_artifact = temporary
    model._vlm_remote_source = source
    return model


def LibreVLM(model: str = _DEFAULT_MODEL, **kwargs) -> LibreVLMModel:
    """Load a vision-language detector by name or fine-tune checkpoint path.

    Args:
        model: Model alias (e.g. ``"qwen3-vl-4b"``, ``"lfm2-vl-450m"``), a
            path to a fine-tune checkpoint directory produced by ``train()``,
            or an immutable ``hf+vlm://owner/repo@<commit>`` artifact URI.
            Defaults to Qwen3-VL-4B (Apache-2.0).
        **kwargs: Forwarded to the family constructor: ``device``, ``names``
            (initial class vocabulary, same as calling ``set_classes`` after
            load), ``prompt`` (override the detection prompt), ``max_new_tokens``.
            When loading a checkpoint, ``names`` defaults to the vocabulary the
            fine-tune was trained on.

    Returns:
        A ``LibreVLMModel`` instance with the standard predict/track surface.
    """
    from .training.checkpoint import is_vlm_checkpoint

    reference = inspect_vlm_reference(model)
    if reference is not None and reference.remote:
        return _load_remote_checkpoint(str(model), **kwargs)
    if is_vlm_checkpoint(model):
        return _load_checkpoint(model, **kwargs)
    key = str(model).strip().lower()
    if key in _LAZY_ALIASES:
        from ..sensenova import LibreSenseNovaVision

        return LibreSenseNovaVision(size=_LAZY_ALIASES[key], **kwargs)
    if key in _MODUS_ALIASES:
        from ..modus import LibreMODUS

        return LibreMODUS(size=_MODUS_ALIASES[key], **kwargs)
    match = _ALIASES.get(key)
    if match is None:
        raise ValueError(
            f"Unknown VLM model {model!r}. Known aliases: "
            f"{', '.join(get_vlm_aliases())}."
        )
    family_cls, size = match
    return family_cls(size=size, **kwargs)


def __getattr__(name: str):
    if name == "LibreSenseNovaVision":
        from ..sensenova import LibreSenseNovaVision

        return LibreSenseNovaVision
    if name in {"LibreMODUS", "LibreModus"}:
        from ..modus import LibreMODUS

        return LibreMODUS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LibreVLM",
    "LibreVLMModel",
    "VLMArtifactError",
    "VLMArtifactInfo",
    "VLMBaseSnapshotInfo",
    "VLMHubRef",
    "VLMReference",
    "build_vlm_artifact",
    "create_vlm_publication_evidence_template",
    "download_vlm_artifact",
    "ensure_vlm_base_snapshot",
    "get_vlm_aliases",
    "inspect_vlm_hub_artifact",
    "inspect_vlm_reference",
    "parse_vlm_hub_uri",
    "push_vlm_artifact",
    "read_vlm_artifact_manifest",
    "validate_vlm_artifact",
    "validate_vlm_base_snapshot",
    "LibreLFM2VL",
    "LibreQwen3VL",
    "LibreSmolVLM2",
    "LibreInternVL3",
    "LibreFlorence2",
    "LibreKosmos2",
    "LibreLocateAnything",
    "LibreGemma4",
    "LibreMoondream",
    "LibreNorthMicroVision",
    "LibreSenseNovaVision",
    "LibreMODUS",
    "LibreModus",
]
