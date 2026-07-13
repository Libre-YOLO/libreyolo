"""Model-family inventory used by CLI and generated export documentation."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import textwrap


OPTIONAL_MODELS = (
    ("libreyolo.models.sam.model", "LibreSAM1", "sam", "transformers"),
    ("libreyolo.models.sam.sam2", "LibreSAM2", "sam", "transformers"),
    ("libreyolo.models.sam.edgetam", "LibreEdgeTAM", "sam", "transformers"),
    ("libreyolo.models.sam.sam3", "LibreSAM3", "sam", "transformers"),
    ("libreyolo.models.mobilesam.model", "LibreMobileSAM", "sam", None),
    ("libreyolo.models.picosam3.model", "LibrePicoSAM3", "sam", None),
    ("libreyolo.models.vlm.florence2", "LibreFlorence2", "vlm", "transformers"),
    ("libreyolo.models.vlm.kosmos2", "LibreKosmos2", "vlm", "transformers"),
    ("libreyolo.models.vlm.internvl3", "LibreInternVL3", "vlm", "transformers"),
    ("libreyolo.models.vlm.lfm2", "LibreLFM2VL", "vlm", "transformers"),
    (
        "libreyolo.models.vlm.locateanything",
        "LibreLocateAnything",
        "vlm",
        "transformers",
    ),
    ("libreyolo.models.vlm.qwen3vl", "LibreQwen3VL", "vlm", "transformers"),
    ("libreyolo.models.vlm.smolvlm", "LibreSmolVLM2", "vlm", "transformers"),
    (
        "libreyolo.models.openvocab.grounding_dino",
        "LibreGroundingDINO",
        "openvocab",
        "transformers",
    ),
    ("libreyolo.models.openvocab.owlv2", "LibreOWLv2", "openvocab", "transformers"),
    (
        "libreyolo.models.openvocab.omdet_turbo",
        "LibreOMDetTurbo",
        "openvocab",
        "transformers",
    ),
)


def _export_override(cls, base_cls) -> str:
    """Classify a family's ``export``: ``none``, ``custom``, or ``blocked``.

    ``blocked`` means every path raises. A family that exports some formats
    and raises for the rest (PicoSAM3 ships ONNX only) is ``custom``, so walk
    the AST for an actual export call instead of matching source text: the
    old ``"export_" not in source`` check missed ``torch.onnx.export``.
    """
    if cls.export is base_cls.export:
        return "none"
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.export)))
    raises_not_implemented = False
    performs_export = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            raised = node.exc
            if isinstance(raised, ast.Call):
                raised = raised.func
            if isinstance(raised, ast.Name) and raised.id == "NotImplementedError":
                raises_not_implemented = True
        elif isinstance(node, ast.Call):
            called = node.func
            name = ""
            if isinstance(called, ast.Attribute):
                name = called.attr
            elif isinstance(called, ast.Name):
                name = called.id
            if "export" in name.lower():
                performs_export = True
    if raises_not_implemented and not performs_export:
        return "blocked"
    return "custom"


def collect_model_inventory() -> dict[str, dict]:
    """Return eager and lazy family metadata without constructing models."""
    from libreyolo.models import try_ensure_rfdetr
    from libreyolo.models.base.model import BaseModel

    optional: dict[str, tuple[str, bool]] = {}
    try_ensure_rfdetr()
    classes = list(BaseModel._registry)
    if any(cls.FAMILY == "rfdetr" for cls in BaseModel._registry):
        optional["rfdetr"] = ("rfdetr", True)
        optional["dinov2"] = ("rfdetr", True)

    for module_name, class_name, extra, requirement in OPTIONAL_MODELS:
        available = (
            requirement is None or importlib.util.find_spec(requirement) is not None
        )
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, ModuleNotFoundError):
            continue
        optional[cls.FAMILY] = (extra, available)
        if cls not in classes:
            classes.append(cls)

    inventory: dict[str, dict] = {}
    for cls in classes:
        family = str(getattr(cls, "FAMILY", ""))
        if not family:
            continue
        extra, available = optional.get(family, (None, True))
        task_sizes = {task: dict(sizes) for task, sizes in cls.TASK_INPUT_SIZES.items()}
        all_sizes = dict(cls.INPUT_SIZES)
        for sizes in task_sizes.values():
            all_sizes.update(sizes)
        inventory[family] = {
            "class": f"{cls.__module__}.{cls.__name__}",
            "tasks": list(cls.SUPPORTED_TASKS),
            "default_task": cls.DEFAULT_TASK,
            "sizes": all_sizes,
            "default_imgsz": dict(cls.INPUT_SIZES),
            "task_sizes": task_sizes,
            "export_override": _export_override(cls, BaseModel),
            "optional_extra": extra,
            "available": available,
        }
    return dict(sorted(inventory.items()))


__all__ = ["OPTIONAL_MODELS", "collect_model_inventory"]
