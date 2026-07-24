"""Convert the valid upstream PicoSAM3 checkpoint to LibreYOLO schema.

Only ``PicoSAM3_SAM3_student_best.pt`` matches the published PicoSAM3
architecture. The two epoch-1 files currently contain the older PicoSAM2
network and are rejected deliberately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _conversion_utils import (
        add_repo_root_to_path,
        load_checkpoint,
        save_checkpoint,
        wrap_libreyolo_checkpoint,
    )
except ModuleNotFoundError:  # imported as weights.convert_picosam3_weights
    from weights._conversion_utils import (
        add_repo_root_to_path,
        load_checkpoint,
        save_checkpoint,
        wrap_libreyolo_checkpoint,
    )

UPSTREAM_REPO = "https://github.com/pbonazzi/picosam3"
UPSTREAM_COMMIT = "1b03949e43472953bb0021685c7fc3f5fdf48fde"
WEIGHTS_REPO = "pietrobonazzi/picosam3"
WEIGHTS_REVISION = "af49e4322b6b7cf448499fee5c073d4576f59444"


def _unwrap(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def convert(input_path: str | Path, output_path: str | Path) -> Path:
    add_repo_root_to_path()
    from libreyolo.models.picosam3.nn import PicoSAM3Network

    state_dict = _unwrap(load_checkpoint(input_path))
    if any(key.startswith("output_head.") for key in state_dict):
        raise ValueError(
            "The supplied checkpoint is the older PicoSAM2 architecture. "
            "Convert PicoSAM3_SAM3_student_best.pt instead."
        )
    model = PicoSAM3Network()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    wrapped = wrap_libreyolo_checkpoint(
        model.state_dict(),
        model_family="picosam3",
        size="pico",
        task="segment",
        nc=1,
        names={0: "object"},
        imgsz=96,
    )
    wrapped.update(
        source_repo=UPSTREAM_REPO,
        source_revision=UPSTREAM_COMMIT,
        weights_repo=WEIGHTS_REPO,
        weights_revision=WEIGHTS_REVISION,
        license="Apache-2.0",
        teacher_chain="SAM 2.1 and SAM 3 (distillation only)",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    save_checkpoint(wrapped, temporary)
    temporary.replace(output_path)
    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="weights/LibrePicoSAM3pico.pt")
    args = parser.parse_args()
    convert(args.input, args.output)
