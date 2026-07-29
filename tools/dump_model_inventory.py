"""Dump the canonical eager and optional model-family inventory as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "export_inventory.json"


def write_inventory(output: Path, *, allow_family_removal: bool = False) -> dict:
    """Collect the runtime inventory and write it to ``output``.

    The committed snapshot is canonical: it is collected in a fully
    provisioned environment where every optional tier imports. When optional
    dependencies are missing, ``collect_model_inventory()`` can return a
    subset, and writing that subset would drop families, tasks, or sizes from
    generated compatibility tables. Refuse to shrink the canonical case set
    unless the caller explicitly allows it for an intentional removal.
    """
    from libreyolo.models.inventory import collect_model_inventory, iter_model_cases

    inventory = collect_model_inventory()
    if output.exists() and not allow_family_removal:
        existing = json.loads(output.read_text(encoding="utf-8"))
        missing = sorted(set(existing) - set(inventory))
        if missing:
            raise SystemExit(
                f"Refusing to overwrite {output}: the fresh inventory is "
                f"missing families present in the committed snapshot: "
                f"{', '.join(missing)}. This usually means optional "
                "dependencies (for example libreyolo[rfdetr]) are not "
                "installed in this environment. Install them and rerun, or "
                "pass --allow-family-removal for an intentional removal."
            )
        existing_cases = {
            (family, task, size) for family, task, size, _ in iter_model_cases(existing)
        }
        fresh_cases = {
            (family, task, size)
            for family, task, size, _ in iter_model_cases(inventory)
        }
        missing_cases = sorted(existing_cases - fresh_cases)
        if missing_cases:
            rendered = ", ".join("/".join(case) for case in missing_cases)
            raise SystemExit(
                f"Refusing to overwrite {output}: the fresh inventory is "
                f"missing canonical family/task/size cases present in the "
                f"committed snapshot: {rendered}. This usually means an "
                "optional model registered only partially. Install the full "
                "dependency set and rerun, or pass --allow-family-removal "
                "for an intentional removal."
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-family-removal",
        action="store_true",
        help="Permit writing an inventory that drops families, tasks, or sizes "
        "present in the existing snapshot.",
    )
    args = parser.parse_args()
    inventory = write_inventory(
        args.output, allow_family_removal=args.allow_family_removal
    )
    print(f"Wrote {len(inventory)} families to {args.output}")


if __name__ == "__main__":
    main()
