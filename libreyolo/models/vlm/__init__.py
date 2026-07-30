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

import json
from pathlib import Path
from typing import Dict, Tuple, Type

from .base import LibreVLMModel
from .florence2 import CoreMLFlorence2, LibreFlorence2
from .internvl3 import LibreInternVL3
from .kosmos2 import CoreMLKosmos2, LibreKosmos2
from .lfm2 import LibreLFM2VL
from .locateanything import LibreLocateAnything
from .qwen3vl import CoreMLQwen3VL, LibreQwen3VL
from .smolvlm import CoreMLSmolVLM2, LibreSmolVLM2

# alias -> (family class, size)
_ALIASES: Dict[str, Tuple[Type[LibreVLMModel], str]] = {
    "qwen3-vl": (LibreQwen3VL, "4b"),
    "qwen3-vl-2b": (LibreQwen3VL, "2b"),
    "qwen3-vl-4b": (LibreQwen3VL, "4b"),
    "qwen3-vl-8b": (LibreQwen3VL, "8b"),
    "lfm2-vl": (LibreLFM2VL, "450m"),
    "lfm2-vl-450m": (LibreLFM2VL, "450m"),
    "lfm2-vl-1.6b": (LibreLFM2VL, "1.6b"),
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
}

# SenseNova-Vision lives in ``models/sensenova`` (it vendors its own
# architecture) and imports this package's base class, so its aliases resolve
# lazily to avoid a circular import at package-init time.
_LAZY_ALIASES: Dict[str, str] = {
    "sensenova-vision": "7b",
    "sensenova-vision-7b": "7b",
    "sensenovavision": "7b",
}

_DEFAULT_MODEL = "qwen3-vl-4b"
_COREML_VLM_MANIFEST_MAX_BYTES = 1024 * 1024


def _coreml_vlm_bundle_format(path: Path) -> str:
    """Read only the strict dispatch discriminator from a portable bundle."""

    if path.is_symlink():
        raise ValueError("Core ML VLM bundle root must not be a symbolic link.")
    manifest = path / "manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise FileNotFoundError(
            f"Core ML VLM bundle manifest does not exist: {manifest}."
        )
    try:
        size = int(manifest.stat().st_size)
    except OSError as exc:
        raise ValueError("Core ML VLM bundle manifest cannot be inspected.") from exc
    if size <= 0 or size > _COREML_VLM_MANIFEST_MAX_BYTES:
        raise ValueError(
            "Core ML VLM bundle manifest must be between 1 byte and "
            f"{_COREML_VLM_MANIFEST_MAX_BYTES} bytes."
        )

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Core ML VLM bundle manifest repeats key {key!r}.")
            result[key] = value
        return result

    try:
        value = json.loads(
            manifest.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Core ML VLM bundle manifest is not valid UTF-8 JSON."
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("Core ML VLM bundle manifest must be a JSON object.")
    bundle_format = value.get("bundle_format")
    if not isinstance(bundle_format, str) or not bundle_format:
        raise ValueError("Core ML VLM bundle manifest has no valid bundle_format.")
    return bundle_format


def LibreVLM(model: str = _DEFAULT_MODEL, **kwargs) -> LibreVLMModel:
    """Load a vision-language detector by name.

    Args:
        model: Model alias (e.g. ``"qwen3-vl-4b"``, ``"lfm2-vl-450m"``)
            or a portable SmolVLM2 ``.coremlvlm`` directory. Defaults to
            Qwen3-VL-4B (Apache-2.0).
        **kwargs: Forwarded to the family constructor: ``device``, ``names``
            (initial class vocabulary, same as calling ``set_classes`` after
            load), ``prompt`` (override the detection prompt),
            ``max_new_tokens``. Core ML bundles additionally accept
            ``compute_units``. Experimental VLM bundles fail closed under the
            default ``"validated"`` policy; pass ``"cpu_only"`` explicitly
            for the recommended experimental runtime opt-in.

    Returns:
        A ``LibreVLMModel`` instance with the standard predict/track surface.
    """
    model_value = str(model).strip()
    candidate = Path(model_value)
    if candidate.suffix.lower() == ".coremlvlm":
        if not candidate.is_dir():
            raise FileNotFoundError(f"Core ML VLM bundle does not exist: {candidate}")
        bundle_format = _coreml_vlm_bundle_format(candidate)
        from ...backends.coreml_florence import COREML_FLORENCE_BUNDLE_FORMAT
        from ...backends.coreml_kosmos import COREML_KOSMOS2_BUNDLE_FORMAT
        from ...backends.coreml_qwen3vl import COREML_QWEN3VL_BUNDLE_FORMAT
        from ...backends.coreml_vlm import COREML_VLM_BUNDLE_FORMAT

        if bundle_format == COREML_VLM_BUNDLE_FORMAT:
            return CoreMLSmolVLM2(str(candidate), **kwargs)
        if bundle_format == COREML_FLORENCE_BUNDLE_FORMAT:
            return CoreMLFlorence2(str(candidate), **kwargs)
        if bundle_format == COREML_KOSMOS2_BUNDLE_FORMAT:
            return CoreMLKosmos2(str(candidate), **kwargs)
        if bundle_format == COREML_QWEN3VL_BUNDLE_FORMAT:
            return CoreMLQwen3VL(str(candidate), **kwargs)
        raise ValueError(f"Unsupported Core ML VLM bundle_format {bundle_format!r}.")

    key = model_value.lower()
    if key in _LAZY_ALIASES:
        from ..sensenova import LibreSenseNovaVision

        return LibreSenseNovaVision(size=_LAZY_ALIASES[key], **kwargs)
    match = _ALIASES.get(key)
    if match is None:
        raise ValueError(
            f"Unknown VLM model {model!r}. Known aliases: "
            f"{', '.join(sorted(set(_ALIASES) | set(_LAZY_ALIASES)))}."
        )
    family_cls, size = match
    return family_cls(size=size, **kwargs)


def __getattr__(name: str):
    if name == "LibreSenseNovaVision":
        from ..sensenova import LibreSenseNovaVision

        return LibreSenseNovaVision
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LibreVLM",
    "LibreVLMModel",
    "LibreLFM2VL",
    "LibreQwen3VL",
    "CoreMLQwen3VL",
    "LibreSmolVLM2",
    "CoreMLSmolVLM2",
    "CoreMLFlorence2",
    "LibreInternVL3",
    "LibreFlorence2",
    "LibreKosmos2",
    "CoreMLKosmos2",
    "LibreLocateAnything",
    "LibreSenseNovaVision",
]
