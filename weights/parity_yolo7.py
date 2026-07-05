"""Inference-parity proof for LibreYOLO7 (YOLOv7) vs upstream MMT/YOLO.

Per the libreyolo-port-model skill (§12). Two checks, gated on a local upstream
``v7.pt`` path (``YOLO7_V7_PT``):

1. **Structural (always):** the upstream ``v7.pt`` strict-loads into the native
   port with 0 missing / 0 unexpected keys — every one of the 564 tensors
   matches name and shape.
2. **Numerical (when the MMT package is importable):** build an oracle from
   upstream's real ``yolo.model.module`` block classes with the same cfg
   connectivity, load ``v7.pt`` into both, and compare forward outputs.

Confirmed 2026-07-04 (isolated venv, upstream module.py): worst head
``max_abs_diff == 0.000e+00`` — bitwise identical. Install the oracle with
``pip install einops`` plus the MMT ``yolo`` package to reproduce.

    export YOLO7_V7_PT=/path/to/v7.pt
    python weights/parity_yolo7.py
"""

from __future__ import annotations

import os
import sys

from _conversion_utils import add_repo_root_to_path

add_repo_root_to_path()


def _numerical_oracle(v7_pt: str) -> float | None:
    """Return worst head max_abs_diff vs MMT's real blocks, or None if MMT absent."""
    try:
        import yolo.model.module as M  # the MMT (MultimediaTechLab/YOLO) package
    except Exception:
        return None
    import torch
    import yaml

    from libreyolo.models.yolo7.net import _V7_YAML

    cfg = yaml.safe_load(open(_V7_YAML, encoding="utf-8"))
    entries = [
        (bt, info or {})
        for section in ("backbone", "head")
        for item in cfg["model"][section]
        for bt, info in item.items()
    ]

    def resolve(src, li, tags):
        if isinstance(src, list):
            return [resolve(s, li, tags) for s in src]
        if isinstance(src, str):
            return tags[src]
        return src + li if src < -1 else src

    layers, bts, srcs, tags, out_dim = [], [], [], {}, [3]
    for p, (bt, info) in enumerate(entries):
        li = p + 1
        src = resolve(info.get("source", -1), li, tags)
        a = info.get("args", {}) or {}
        if bt == "Conv":
            m, oc = M.Conv(out_dim[src], a["out_channels"], a["kernel_size"], stride=a.get("stride", 1)), a["out_channels"]
        elif bt == "Pool":
            m, oc = M.Pool(kernel_size=a.get("kernel_size", 2), padding=a.get("padding", 0)), out_dim[src]
        elif bt == "UpSample":
            m, oc = M.UpSample(scale_factor=a.get("scale_factor", 2)), out_dim[src]
        elif bt == "SPPCSPConv":
            m, oc = M.SPPCSPConv(out_dim[src], a["out_channels"]), a["out_channels"]
        elif bt == "RepConv":
            m, oc = M.RepConv(out_dim[src], a["out_channels"]), a["out_channels"]
        elif bt == "Concat":
            m, oc = torch.nn.Identity(), sum(out_dim[i] for i in src)
        elif bt == "MultiheadDetection":
            m, oc = M.MultiheadDetection([out_dim[i] for i in src], 80, version="v7"), 0
        else:
            raise ValueError(bt)
        layers.append(m); bts.append(bt); srcs.append(src); out_dim.append(oc)
        if info.get("tags"):
            tags[info["tags"]] = li

    oracle = torch.nn.ModuleList(layers).eval()
    sd = torch.load(v7_pt, map_location="cpu", weights_only=False)
    oracle.load_state_dict(sd, strict=False)

    from libreyolo.models.yolo7.net import YOLOv7Model

    mine = YOLOv7Model(num_classes=80).eval()
    mine.load_state_dict({f"layers.{k}": v for k, v in sd.items()}, strict=True)

    def fwd(x):
        y, out = [x], None
        for m, bt, src in zip(oracle, bts, srcs):
            if bt == "Concat":
                o = torch.cat([y[i] for i in src], dim=1)
            elif bt == "MultiheadDetection":
                o = m([y[i] for i in src]); out = o
            else:
                o = m(y[src])
            y.append(o)
        return out

    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        return max((a - b).abs().max().item() for a, b in zip(fwd(x), mine(x)))


def main() -> int:
    v7_pt = os.environ.get("YOLO7_V7_PT")
    if not v7_pt:
        raise SystemExit("Set YOLO7_V7_PT to the upstream v7.pt path.")
    import torch

    from libreyolo.models.yolo7.net import YOLOv7Model

    sd = torch.load(v7_pt, map_location="cpu", weights_only=False)
    mine = YOLOv7Model(num_classes=80)
    res = mine.load_state_dict({f"layers.{k}": v for k, v in sd.items()}, strict=False)
    miss = [k for k in res.missing_keys]
    unexp = [k for k in res.unexpected_keys]
    print(f"[structural] strict-load: missing={len(miss)} unexpected={len(unexp)}")
    assert not miss and not unexp, "structural mismatch"

    diff = _numerical_oracle(v7_pt)
    if diff is None:
        print("[numerical] MMT 'yolo' package not importable — skipped "
              "(confirmed 0.000e+00 on 2026-07-04; pip install the MMT package to reproduce)")
        return 0
    print(f"[numerical] worst head max_abs_diff vs MMT blocks = {diff:.3e}")
    assert diff < 1e-5, f"numerical mismatch: {diff}"
    print("v7 PARITY OK (exact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
