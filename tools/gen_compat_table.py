"""Generate README and detailed documentation from the export support matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


START = "<!-- export-support:start -->"
END = "<!-- export-support:end -->"
# Repository-owned paths are anchored to this script's location so the tool
# works no matter which directory it is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "reports" / "export_inventory.json"
README_PATH = REPO_ROOT / "README.md"
DOCS_PATH = REPO_ROOT / "docs" / "export_support.md"
MARKERS = {"validated": "✓", "experimental": "exp", "blocked": ""}

# The table is a shop window, not an index: readers come for the flagship
# detectors, so those lead, and the remaining families are grouped by how
# people scan the library (detection first, niche tasks last) instead of
# alphabetically — which used to open the table with `birefnet` (matting)
# while yolo9/yolonas/rfdetr sat at the bottom.
FLAGSHIP_FAMILIES = ("yolo9", "rfdetr")
TASK_ORDER = (
    "detect",
    "segment",
    "pose",
    "obb",
    "point",
    "classify",
    "semantic",
    "panoptic",
    "depth",
    "restore",
    "matte",
    "ocr",
    "gaze",
)


def _family_order(inventory: dict) -> list[str]:
    """Return inventory families in reader-facing order.

    Flagships lead in their documented order. The rest sort by the earliest
    task tier they serve (per TASK_ORDER), then by parity-validated export
    count (most validated first, so heavily tested families outrank
    export-blocked ones), then by name for determinism.
    """
    from libreyolo.export.support import EXPORT_FORMATS, get_support

    def key(family: str) -> tuple:
        if family in FLAGSHIP_FAMILIES:
            return (0, FLAGSHIP_FAMILIES.index(family), 0, family)
        metadata = inventory[family]
        tiers = [
            TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER)
            for task in metadata["tasks"]
        ]
        validated = sum(
            1
            for task in metadata["tasks"]
            for fmt in EXPORT_FORMATS
            if get_support(family, task, fmt).tier == "validated"
        )
        return (1, min(tiers), -validated, family)

    return sorted(inventory, key=key)


def _rows() -> tuple[list[str], list[str]]:
    from libreyolo.export.support import EXPORT_FORMATS, get_support

    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Canonical model inventory is missing: {INVENTORY_PATH}. "
            "Run tools/dump_model_inventory.py in the fully provisioned environment."
        )
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))

    header = ["Family", "Task", *EXPORT_FORMATS]
    readme = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", "---", *(["---:"] * len(EXPORT_FORMATS))]) + " |",
    ]
    blocked = []
    for family in _family_order(inventory):
        metadata = inventory[family]
        for task in metadata["tasks"]:
            entries = [get_support(family, task, fmt) for fmt in EXPORT_FORMATS]
            readme.append(
                "| "
                + " | ".join(
                    [family, task, *(MARKERS[entry.tier] for entry in entries)]
                )
                + " |"
            )
            for fmt, entry in zip(EXPORT_FORMATS, entries):
                if entry.tier == "blocked":
                    blocked.append(f"- `{family}` / `{task}` / `{fmt}`: {entry.reason}")
    return readme, blocked


def render_readme_table() -> str:
    rows, _ = _rows()
    return "\n".join([START, *rows, END])


def render_docs() -> str:
    rows, blocked = _rows()
    return "\n".join(
        [
            "# Export support",
            "",
            "This document is generated from `libreyolo/export/support.py`.",
            "Do not edit the matrix by hand.",
            "",
            "`✓` means parity-validated, `exp` means conversion is available without a",
            "numeric parity guarantee, and an empty cell is blocked in preflight.",
            "",
            *rows,
            "",
            "## Parity thresholds",
            "",
            "- Detection and OBB: matched box IoU above 0.95 and score MAE below 0.01.",
            "- Segmentation and panoptic: mask IoU above 0.95.",
            "- Pose: keypoint L2 below 2 pixels at native resolution.",
            "- Classification: logits cosine above 0.999 and equal top-1 class.",
            "- Depth and restoration: PSNR above 40 dB against native output.",
            "- Point: peak locations equal within one output cell.",
            "",
            "## Blocked combinations",
            "",
            *blocked,
            "",
        ]
    )


def _replace_readme(text: str, generated: str) -> str:
    if START in text and END in text:
        start = text.index(START)
        end = text.index(END, start) + len(END)
        return text[:start] + generated + text[end:]
    heading = text.find("## Compatibility")
    if heading < 0:
        raise ValueError(
            "README.md has neither export-support markers nor a "
            "'## Compatibility' section."
        )
    table_start = text.find("<table>", heading)
    table_end_start = text.find("</table>", table_start)
    if table_start < 0 or table_end_start < 0:
        raise ValueError(
            "README.md compatibility section has no replaceable table; "
            "restore the export-support markers."
        )
    table_end = table_end_start + len("</table>")
    return text[:table_start] + generated + text[table_end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    readme_path = README_PATH
    docs_path = DOCS_PATH
    readme = _replace_readme(
        readme_path.read_text(encoding="utf-8"), render_readme_table()
    )
    docs = render_docs()
    if args.check:
        stale = []
        if readme_path.read_text(encoding="utf-8") != readme:
            stale.append(str(readme_path))
        if not docs_path.exists() or docs_path.read_text(encoding="utf-8") != docs:
            stale.append(str(docs_path))
        if stale:
            raise SystemExit("Generated export docs are stale: " + ", ".join(stale))
        return
    readme_path.write_text(readme, encoding="utf-8")
    docs_path.write_text(docs, encoding="utf-8")


if __name__ == "__main__":
    main()
