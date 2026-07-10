"""Generate README and detailed documentation from the export support matrix."""

from __future__ import annotations

import argparse
from pathlib import Path


START = "<!-- export-support:start -->"
END = "<!-- export-support:end -->"
MARKERS = {"validated": "✓", "experimental": "exp", "blocked": ""}


def _rows() -> tuple[list[str], list[str]]:
    from libreyolo.export.support import EXPORT_FORMATS, get_support
    from libreyolo.models.inventory import collect_model_inventory

    header = ["Family", "Task", *EXPORT_FORMATS]
    readme = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", "---", *(["---:"] * len(EXPORT_FORMATS))]) + " |",
    ]
    blocked = []
    for family, metadata in collect_model_inventory().items():
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
    heading = text.index("## Compatibility")
    table_start = text.index("<table>", heading)
    table_end = text.index("</table>", table_start) + len("</table>")
    return text[:table_start] + generated + text[table_end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    readme_path = Path("README.md")
    docs_path = Path("docs/export_support.md")
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
