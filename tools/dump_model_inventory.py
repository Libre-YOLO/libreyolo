"""Dump the canonical eager and optional model-family inventory as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    from libreyolo.models.inventory import collect_model_inventory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/export_inventory.json"),
    )
    args = parser.parse_args()
    inventory = collect_model_inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(inventory)} families to {args.output}")


if __name__ == "__main__":
    main()
