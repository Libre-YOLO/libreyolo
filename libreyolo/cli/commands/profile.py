"""profile: analyse training profiles from ``model.train(profile=True)``.

Built for agents. Every subcommand prints results to stdout and supports
``--json``, so an LLM can drive the loop:

    libreyolo profile run  --data coco1000 --weights LibreYOLO9t.pt --size t
    libreyolo profile summary <trace>                # what's the bottleneck?
    libreyolo profile kernels <trace> --top 20       # drill to the kernels
    libreyolo profile ops     <trace> --top 20       # framework/host ops
    libreyolo profile compare <before> <after>       # did my change help?

...read insight, change the training/config/code, re-run, ``compare``, repeat
until images/sec is maxed.
"""

from __future__ import annotations

import json as _json
import re as _re
from pathlib import Path
from typing import Optional

import typer

profile_app = typer.Typer(
    name="profile",
    help="Analyse training profiles (model.train(profile=True)) for speed tuning.",
    no_args_is_help=True,
    add_completion=False,
)


def _load(trace: str) -> dict:
    from libreyolo.training.profiler import analyze_trace

    p = Path(trace)
    if not p.exists():
        typer.echo(f"trace not found: {trace}", err=True)
        raise typer.Exit(2)
    try:
        return analyze_trace(p)
    except Exception as exc:  # pragma: no cover - defensive
        typer.echo(f"failed to analyse {trace}: {exc}", err=True)
        raise typer.Exit(1)


def _pct(before, after) -> str:
    if not before:
        return ""
    return f" ({(after - before) / before * 100:+.0f}%)"


@profile_app.command("run")
def run_cmd(
    data: str = typer.Argument(..., help="Dataset yaml/name (e.g. coco1000)"),
    weights: str = typer.Option("LibreYOLO9t.pt", "--weights", help="Model weights file / name"),
    size: str = typer.Option("t", "--size", help="Model size variant"),
    batch: int = typer.Option(16, "--batch"),
    imgsz: int = typer.Option(640, "--imgsz"),
    workers: int = typer.Option(8, "--workers"),
    amp: bool = typer.Option(False, "--amp", help="Use the family's AMP path"),
    steps: int = typer.Option(20, "--steps", help="Profiled (measured) steps"),
    device: str = typer.Option("0", "--device"),
    project: str = typer.Option("runs/profile", "--project"),
    json_output: bool = typer.Option(False, "--json", help="JSON {trace, summary} to stdout"),
) -> None:
    """Launch a short profiled training and emit the trace (no browser)."""
    from libreyolo import LibreYOLO

    m = LibreYOLO(model_path=weights, size=size, device=device)
    m.train(
        data=data, epochs=1, batch=batch, imgsz=imgsz, workers=workers, amp=amp,
        device=device, profile=True, profile_steps=steps, profile_open=False,
        no_aug_epochs=0, project=project, name="prof", exist_ok=True,
    )
    trace = Path(project) / "prof" / "profile_trace.json"
    if json_output:
        a = _load(str(trace)) if trace.exists() else {}
        print(_json.dumps({"trace": str(trace), "summary": {
            k: a.get(k) for k in
            ("bound", "gpu_util", "img_per_s", "step_ms", "tensorcore_pct", "peak_vram_mb")
        }}, indent=2))
    else:
        print(f"trace: {trace}")
        print(f"next:  libreyolo profile summary {trace}")


@profile_app.command("summary")
def summary_cmd(
    trace: str = typer.Argument(..., help="Path to profile_trace.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """High-level diagnosis: utilisation, verdict, kernel mix, top kernels."""
    a = _load(trace)
    if json_output:
        keep = ("trace", "model", "config", "bound", "bound_why", "step_ms", "img_per_s",
                "gpu_util", "gpu_busy_ms_per_step", "mean_kernel_us", "kernels_per_step",
                "unique_kernels", "memcpy_ms_per_step", "tensorcore_pct", "peak_vram_mb",
                "dataload_ms", "dataload_frac", "window", "categories", "phases_gpu",
                "top_kernels")
        print(_json.dumps({k: a[k] for k in keep}, indent=2))
        return
    print(f"model {a.get('model') or '?'}  |  {a.get('img_per_s') or '?'} img/s  |  "
          f"step {a['step_ms']} ms  |  {a['kernels_per_step']} kernels/step @ ~{a['mean_kernel_us']:.0f}us")
    print(f"GPU util {a['gpu_util'] * 100:.0f}%  ({a['gpu_busy_ms_per_step']} ms busy)  |  "
          f"Tensor Cores {a['tensorcore_pct']:.0f}%  |  peak VRAM {a.get('peak_vram_mb') or '?'} MB  |  "
          f"memcpy {a['memcpy_ms_per_step']} ms")
    print(f">> {str(a['bound']).upper()} — {a['bound_why']}")
    print("kernel mix:")
    for c in a["categories"]:
        print(f"  {c['name']:<17} {c['pct']:5.1f}%   {c['ms_per_step']:.1f} ms/step")
    print("top kernels/step:")
    for k in a["top_kernels"]:
        tc = " [TC]" if k["tensorcore"] else ""
        print(f"  {k['pct_of_gpu']:5.1f}%  {k['ms_per_step']:6.3f} ms  x{k['count_per_step']:<4} {k['kernel']}{tc}")


@profile_app.command("kernels")
def kernels_cmd(
    trace: str = typer.Argument(..., help="Path to profile_trace.json"),
    top: int = typer.Option(20, "--top", help="Show top N by GPU time"),
    category: Optional[str] = typer.Option(None, "--category", help="Filter by category substring (gemm, layout, norm, elementwise)"),
    grep: Optional[str] = typer.Option(None, "--grep", help="Filter by kernel-name regex"),
    tensorcore: bool = typer.Option(False, "--tensorcore", help="Only Tensor-Core kernels"),
    sort: str = typer.Option("time", "--sort", help="time | count | name"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Drill to individual GPU kernels — the bottom of the analysis."""
    a = _load(trace)
    ks = a["kernels"]
    if category:
        ks = [k for k in ks if category.lower() in k["category"].lower()]
    if grep:
        rx = _re.compile(grep, _re.I)
        ks = [k for k in ks if rx.search(k["raw_name"]) or rx.search(k["kernel"])]
    if tensorcore:
        ks = [k for k in ks if k["tensorcore"]]
    keyf = {"time": lambda k: -k["pct_of_gpu"], "count": lambda k: -k["count_per_step"],
            "name": lambda k: k["kernel"]}.get(sort, lambda k: -k["pct_of_gpu"])
    ks = sorted(ks, key=keyf)[:top]
    if json_output:
        print(_json.dumps({"trace": a["trace"], "matched": len(ks), "kernels": ks}, indent=2))
        return
    print(f"{'%GPU':>6} {'ms/step':>9} {'x/step':>7} TC  kernel  [category]")
    for k in ks:
        print(f"{k['pct_of_gpu']:6.2f} {k['ms_per_step']:9.3f} {k['count_per_step']:7d}  "
              f"{'Y' if k['tensorcore'] else ' '}  {k['kernel']}  [{k['category']}]")


@profile_app.command("ops")
def ops_cmd(
    trace: str = typer.Argument(..., help="Path to profile_trace.json"),
    top: int = typer.Option(20, "--top"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Framework view: aten/autograd ops by CPU time (host-launch culprits)."""
    a = _load(trace)
    ops = a["ops"][:top]
    if json_output:
        print(_json.dumps({"trace": a["trace"], "ops": ops}, indent=2))
        return
    print(f"{'%cpu':>6} {'ms/step':>9} {'x/step':>7}  op")
    for o in ops:
        print(f"{o['pct_of_cpu_ops']:6.2f} {o['ms_per_step']:9.3f} {o['count_per_step']:7d}  {o['op']}")


@profile_app.command("compare")
def compare_cmd(
    before: str = typer.Argument(..., help="baseline profile_trace.json"),
    after: str = typer.Argument(..., help="new profile_trace.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Diff two profiles — did the change help? (the optimise-loop closer)."""
    a, b = _load(before), _load(after)

    def delta(x, y):
        return None if (x is None or y is None) else round(y - x, 3)

    catA = {c["name"]: c["ms_per_step"] for c in a["categories"]}
    catB = {c["name"]: c["ms_per_step"] for c in b["categories"]}
    cat_delta = {k: round(catB.get(k, 0.0) - catA.get(k, 0.0), 2)
                 for k in set(catA) | set(catB)}
    res = {
        "before": before, "after": after,
        "img_per_s": {"before": a["img_per_s"], "after": b["img_per_s"],
                      "delta": delta(a["img_per_s"], b["img_per_s"])},
        "step_ms": {"before": a["step_ms"], "after": b["step_ms"],
                    "delta": delta(a["step_ms"], b["step_ms"])},
        "gpu_util": {"before": a["gpu_util"], "after": b["gpu_util"],
                     "delta": delta(a["gpu_util"], b["gpu_util"])},
        "gpu_busy_ms_per_step": {"before": a["gpu_busy_ms_per_step"],
                                 "after": b["gpu_busy_ms_per_step"]},
        "tensorcore_pct": {"before": a["tensorcore_pct"], "after": b["tensorcore_pct"]},
        "bound": {"before": a["bound"], "after": b["bound"]},
        "category_ms_delta": cat_delta,
    }
    if json_output:
        print(_json.dumps(res, indent=2))
        return
    print(f"img/s    {a.get('img_per_s')} -> {b.get('img_per_s')}"
          f"{_pct(a.get('img_per_s') or 0, b.get('img_per_s') or 0)}")
    print(f"step ms  {a['step_ms']} -> {b['step_ms']}{_pct(a['step_ms'], b['step_ms'])}")
    print(f"GPU util {a['gpu_util'] * 100:.0f}% -> {b['gpu_util'] * 100:.0f}%")
    print(f"TC %     {a['tensorcore_pct']:.0f}% -> {b['tensorcore_pct']:.0f}%")
    print(f"verdict  {a['bound']} -> {b['bound']}")
    print("category ms/step delta (negative = faster):")
    for k, v in sorted(cat_delta.items(), key=lambda kv: kv[1]):
        print(f"  {k:<17} {v:+.2f}")
