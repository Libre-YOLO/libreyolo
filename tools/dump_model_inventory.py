"""Dump the canonical eager and optional model-family inventory as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "export_inventory.json"


def write_inventory(output: Path, *, allow_family_removal: bool = False) -> dict:
    """Collect the runtime inventory and write it to ``output``.

    The committed snapshot is canonical. Model-family membership comes from
    the static manifest and is complete even when optional dependencies are
    absent. Refuse to shrink the family set unless the caller explicitly
    allows it for an intentional family removal.
    """
    from libreyolo.models.inventory import collect_model_inventory

    inventory = collect_model_inventory()
    if output.exists() and not allow_family_removal:
        existing = json.loads(output.read_text(encoding="utf-8"))
        missing = sorted(set(existing) - set(inventory))
        if missing:
            raise SystemExit(
                f"Refusing to overwrite {output}: the fresh inventory is "
                f"missing families present in the committed snapshot: "
                f"{', '.join(missing)}. Update the static manifest, then pass "
                "--allow-family-removal only for an intentional removal."
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
        help="Permit writing an inventory that drops families present in the "
        "existing snapshot.",
    )
    args = parser.parse_args()
    inventory = write_inventory(
        args.output, allow_family_removal=args.allow_family_removal
    )
    print(f"Wrote {len(inventory)} families to {args.output}")


if __name__ == "__main__":
    main()
