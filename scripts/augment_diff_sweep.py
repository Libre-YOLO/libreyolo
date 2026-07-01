"""Differential-RNG sweep for the augmentations refactor (issue #480).

Runs every parity case from ``tests/unit/test_augment_parity.py`` across many
seeds against TWO checkouts of the repo (pre-refactor and post-refactor) and
asserts byte-identical outputs. This catches RNG-call-order drift on seeds the
committed golden fixtures don't pin.

The two runs MUST happen in separate processes: both trees export a top-level
``libreyolo`` package, and a single process would cache whichever wins
``sys.path`` in ``sys.modules`` — silently comparing a tree against itself.
Each dump run therefore asserts ``libreyolo.__file__`` lives under the
requested tree and records it in the output.

Usage:
  python augment_diff_sweep.py dump --tree <repo-root> --harness <path-to-test_augment_parity.py> --out <dir> [--seeds 40]
  python augment_diff_sweep.py compare <dir-a> <dir-b>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_harness(path: Path):
    spec = importlib.util.spec_from_file_location("augment_parity_harness", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["augment_parity_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


def dump(args) -> None:
    tree = str(Path(args.tree).resolve())
    sys.path.insert(0, tree)

    import libreyolo

    lib_file = str(Path(libreyolo.__file__).resolve()).replace("\\", "/")
    assert lib_file.startswith(tree.replace("\\", "/")), (
        f"libreyolo resolved to {lib_file}, expected under {tree} — "
        "sys.path leak; results would be meaningless"
    )

    harness = _load_harness(Path(args.harness).resolve())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_seed = harness.SEED
    for i in range(args.seeds):
        harness.SEED = base_seed + i * 7919  # spread seeds; prime stride
        for name, fn in sorted(harness.CASES.items()):
            result = fn()
            np.savez_compressed(out / f"{name}__seed{i}.npz", **result)
    harness.SEED = base_seed

    meta = {"tree": tree, "libreyolo_file": lib_file, "seeds": args.seeds}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"dumped {args.seeds} seeds x {len(harness.CASES)} cases from {lib_file}")


def compare(args) -> None:
    dir_a, dir_b = Path(args.dirs[0]), Path(args.dirs[1])
    meta_a = json.loads((dir_a / "meta.json").read_text())
    meta_b = json.loads((dir_b / "meta.json").read_text())
    assert meta_a["libreyolo_file"] != meta_b["libreyolo_file"], (
        "both dumps came from the SAME libreyolo — the sweep compared a tree "
        "against itself and proves nothing"
    )

    files_a = sorted(p.name for p in dir_a.glob("*.npz"))
    files_b = sorted(p.name for p in dir_b.glob("*.npz"))
    assert files_a == files_b, f"case sets differ: {set(files_a) ^ set(files_b)}"

    checked = 0
    for name in files_a:
        a = np.load(dir_a / name, allow_pickle=True)
        b = np.load(dir_b / name, allow_pickle=True)
        assert set(a.files) == set(b.files), f"{name}: key sets differ"
        for key in a.files:
            np.testing.assert_array_equal(
                a[key], b[key], err_msg=f"{name}/{key} diverged", strict=True
            )
            checked += 1
    print(
        f"OK: {len(files_a)} case-seed dumps byte-identical ({checked} arrays)\n"
        f"  A: {meta_a['libreyolo_file']}\n"
        f"  B: {meta_b['libreyolo_file']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump")
    d.add_argument("--tree", required=True)
    d.add_argument("--harness", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--seeds", type=int, default=40)
    d.set_defaults(func=dump)

    c = sub.add_parser("compare")
    c.add_argument("dirs", nargs=2)
    c.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
