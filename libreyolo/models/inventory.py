"""Deterministic model-family inventory for CLI and generated documentation."""

from __future__ import annotations

import importlib.util

from .manifest import (
    ARTIFACT_BY_KEY,
    CLI_MODEL_ALIASES,
    FACTORY_DEFAULT_MODELS,
    FactoryKind,
    iter_family_specs,
)


def _is_available(dependencies: tuple[str, ...]) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in dependencies)


def collect_model_inventory() -> dict[str, dict]:
    """Return the complete public inventory without importing model modules."""
    inventory: dict[str, dict] = {}
    for family in iter_family_specs():
        task_sizes = {
            task.task: {size.code: size.native_imgsz for size in task.sizes}
            for task in family.tasks
        }
        default_imgsz = dict(task_sizes[family.default_task])
        # Retain the legacy flattened mapping for report consumers.  Task-aware
        # consumers must use task_sizes because the same size code can have a
        # different native resolution for another task.
        all_sizes = dict(default_imgsz)
        for sizes in task_sizes.values():
            all_sizes.update(sizes)

        artifacts = []
        for key, artifact in ARTIFACT_BY_KEY.items():
            if key[0] != family.family:
                continue
            artifacts.append(
                {
                    "size": artifact.size,
                    "task": artifact.task,
                    "variant": artifact.variant,
                    "imgsz": artifact.native_imgsz,
                    "filename": artifact.canonical_filename,
                    "publication": artifact.publication.value,
                    "downloadable": artifact.downloadable,
                    "download_kind": artifact.download_kind,
                    "download_url": artifact.download_url,
                    "aliases": list(artifact.aliases),
                    "factory_model": artifact.factory_model,
                    "invocation": artifact.invocation,
                    "repository": artifact.repository,
                    "revision": artifact.revision,
                }
            )

        cli_names = sorted(
            alias
            for alias, artifact in CLI_MODEL_ALIASES.items()
            if artifact.family == family.family
        )
        downloadable_cli_names = sorted(
            alias
            for alias, artifact in CLI_MODEL_ALIASES.items()
            if artifact.family == family.family and artifact.downloadable
        )
        inventory[family.family] = {
            "class": family.class_path,
            "tasks": [task.task for task in family.tasks],
            "default_task": family.default_task,
            "sizes": all_sizes,
            "default_imgsz": default_imgsz,
            "task_sizes": task_sizes,
            "export_override": family.export_override,
            "optional_extra": family.optional_extra,
            "available": _is_available(family.dependencies),
            "factory": family.factory.value,
            "public_entrypoint": family.public_entrypoint,
            "factory_default_model": FACTORY_DEFAULT_MODELS.get(family.factory),
            "generic_cli": family.factory is FactoryKind.CHECKPOINT,
            "cli_names": cli_names,
            "downloadable_cli_names": downloadable_cli_names,
            "local_only_cli_names": sorted(
                set(cli_names) - set(downloadable_cli_names)
            ),
            "artifacts": artifacts,
            "dependencies": list(family.dependencies),
        }
    return dict(sorted(inventory.items()))


__all__ = ["collect_model_inventory"]
