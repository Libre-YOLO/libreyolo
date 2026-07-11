"""Lightweight built-in training profiler.

Answers one question fast: **where does each training step's time go, and is the
GPU starved?** It profiles a short window of *real* training steps and reports
two complementary views:

1. **Real timing (unsynced).** The first half of the window runs with no extra
   synchronization, so the measured step time and the dataloader stall reflect
   true overlap behavior. This drives the honest verdict (dataloader-bound vs
   compute-bound), the real img/s, and the GPU-idle estimate.
2. **Compute composition (synced).** The second half brackets each phase with a
   CUDA sync to attribute GPU time to forward / backward / optimizer / to_device.

Splitting the window matters: syncing every phase serializes the GPU and hands
the dataloader workers slack, which *hides* starvation. So we never derive the
verdict from synced numbers — only the composition.

It writes a self-contained ``profile_report.html`` (auto-opened in the browser)
and a Chrome/Perfetto trace (``profile_trace.json``), plus ``profile_summary.json``.

Zero overhead when disabled: the trainer holds ``None`` and the hot-loop calls
short-circuit to a shared no-op context.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from pathlib import Path
from typing import Optional

import torch
from torch.profiler import record_function

PHASES = ("to_device", "forward", "backward", "optimizer")
_NULL = contextlib.nullcontext()

# Kernel-name -> coarse category. Order matters (first match wins): norm before
# gemm/conv (cudnn batchnorm), layout before gemm/conv (cudnn nchw<->nhwc).
# First match wins. Explicit transposes are checked BEFORE conv (cudnn's
# nchwToNhwc kernel also contains "cudnn"); conv/gemm is checked BEFORE generic
# copies (cutlass conv epilogues mention "copy"); bare nchw/nhwc are NOT layout
# tokens (they appear in conv kernel names as layout specialisations).
_KERNEL_CATS = (
    ("reduction / norm", r"bn_bw|bn_fw|batchnorm|batch_norm|layer_norm|group_norm|instance_norm|softmax|welford|\breduce|variance|rms_norm"),
    ("layout / copy", r"nchwtonhwc|nhwctonchw"),
    ("gemm / conv", r"gemm|conv|implicit|cutlass|cudnn|wgrad|dgrad|winograd"),
    ("layout / copy", r"memcpy|memset|\bcopy|catarray|concat|contiguous|\bgather|\bscatter|\bpermute|\btranspose"),
    ("elementwise", r"elementwise|vectorized|pointwise|activation|silu|relu|sigmoid|unary|binary|clamp|fill|arange"),
)

# Friendly labels for the most common mangled kernel names.
_KERNEL_FRIENDLY = (
    ("bn_bw", "cudnn batchnorm (bwd)"),
    ("bn_fw", "cudnn batchnorm (fwd)"),
    ("nchwtonhwc", "layout nchw->nhwc"),
    ("nhwctonchw", "layout nhwc->nchw"),
    ("implicitgemm", "cutlass conv (implicit gemm)"),
    ("implicit_convolve", "implicit conv"),
    ("wgrad", "conv wgrad"),
    ("dgrad", "conv dgrad"),
    ("cgemm", "gemm"),
    ("sgemm", "gemm"),
    ("gemm", "gemm"),
    ("vectorized_elementwise", "elementwise"),
    ("elementwise", "elementwise"),
    ("softmax", "softmax"),
    ("reduce", "reduction"),
    ("cat", "concat"),
)


def _kernel_category(name: str) -> str:
    nl = name.lower()
    for cat, pat in _KERNEL_CATS:
        if re.search(pat, nl):
            return cat
    return "other"


def _friendly_kernel(name: str) -> str:
    nl = name.lower()
    for key, friendly in _KERNEL_FRIENDLY:
        if key in nl:
            return friendly
    return name[:40]


# Tensor-Core kernel signatures (Volta h884, Ampere/Ada 16816/16832, tf32 s1688,
# int8 imma, cutlass tensorop, wmma).
_TC_PAT = r"h884|h1688|i16832|i8816|s16816|s1688|hmma|imma|tensorop|wmma|16816|16832|884gemm|1688gemm"


def _is_tensorcore(name: str) -> bool:
    return re.search(_TC_PAT, name.lower()) is not None


def _pick_window(steps):
    """Return ``(w0, w1, n_real)`` for a steady multi-step analysis window.

    Skips the first two (cold) ProfilerStep markers; spanning several steps is
    immune to torch's duplicate big/small ProfilerStep markers.
    """
    if len(steps) >= 4:
        w0 = steps[2]["ts"]
        w1 = steps[min(len(steps) - 1, 8)]["ts"]
    elif steps:
        w0 = steps[0]["ts"]
        w1 = steps[-1]["ts"] + (steps[-1].get("dur", 0) or 0)
    else:
        return None, None, 1
    if w1 <= w0:
        w1 = w0 + 1.0
    n_real = _count_steps_in_window(steps, w0, w1)
    return w0, w1, n_real


def _count_steps_in_window(steps, w0, w1):
    if w0 is None or w1 is None:
        return 1
    in_window = [s for s in steps if w0 <= s.get("ts", 0) < w1]
    if not in_window:
        return 1
    numbered = set()
    for step in in_window:
        match = re.search(r"#(\d+)", str(step.get("name", "")))
        if match:
            numbered.add(match.group(1))
    if numbered:
        return max(len(numbered), 1)
    starts = sorted(float(s.get("ts", 0)) for s in in_window)
    clusters = 0
    last = None
    for ts in starts:
        if last is None or ts - last > 1.0:
            clusters += 1
        last = ts
    return max(clusters, 1)


def analyze_trace(trace_path, summary_path=None):
    """Rich, CLI-facing analysis of a torch ``profile_trace.json``.

    Mines the GPU truth at every abstraction level (utilisation, kernel mix,
    every kernel, every aten op, per-phase GPU time, Tensor-Core %). When a
    sibling ``profile_summary.json`` is present it is used for the honest
    unsynced step time / dataloader stall / verdict; otherwise those are
    derived from the trace (with dataloader-vs-host noted as approximate).

    Returns a plain dict — safe to ``json.dumps`` — so an agent can drive a
    train -> profile -> drill-down -> change -> compare loop over the CLI.
    """
    trace_path = Path(trace_path)
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    events = data.get("traceEvents", []) if isinstance(data, dict) else data

    steps = sorted(
        (e for e in events if e.get("ph") == "X"
         and str(e.get("name", "")).startswith("ProfilerStep")),
        key=lambda e: e.get("ts", 0),
    )
    w0, w1, n_real = _pick_window(steps)

    def in_win(ts):
        return w0 is None or (w0 <= ts < w1)

    kern, opagg, phase_gpu = {}, {}, {}
    kernel_list, op_list = [], []
    cpu_intervals = {}
    gpu_busy_us = memcpy_us = tc_us = 0.0
    n_kern = n_malloc = 0
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        ts = e.get("ts")
        dur = e.get("dur", 0) or 0
        if ts is None or not in_win(ts):
            continue
        name = str(e.get("name", ""))
        if cat == "kernel":
            gpu_busy_us += dur
            n_kern += 1
            if _is_tensorcore(name):
                tc_us += dur
            k = kern.setdefault(name, [0.0, 0])
            k[0] += dur
            k[1] += 1
            kernel_list.append((name, dur, ts))
        elif cat in ("gpu_memcpy", "gpu_memset"):
            memcpy_us += dur
        elif cat == "cuda_runtime" and ("Malloc" in name or "Free" in name):
            n_malloc += 1
        elif cat == "cpu_op":
            o = opagg.setdefault(name, [0.0, 0])
            o[0] += dur
            o[1] += 1
            op_list.append((name, dur, ts))
        elif cat == "gpu_user_annotation" and name.startswith("step/"):
            phase_gpu[name[5:]] = phase_gpu.get(name[5:], 0.0) + dur
        elif cat == "user_annotation" and name.startswith("step/"):
            cpu_intervals.setdefault(name[5:], []).append((ts, ts + dur))

    # CPU phase launch spans, for tagging kernels/ops to a phase (`phases`,
    # `--phase` drill-down).
    cpu_iv = sorted((s, en, p) for p, ivs in cpu_intervals.items() for (s, en) in ivs)

    def _phase_of(ts, iv):
        for s, en, p in iv:
            if s <= ts < en:
                return p
        return "unphased"

    phase_kern, phase_kcount, phase_op, phase_ocount = {}, {}, {}, {}
    phase_gpu_us = {}
    # Tag GPU kernels by the CPU phase span that launched them. The gpu-side
    # annotation spans overlap and mis-attribute (forward swallows backward);
    # the CPU launch spans are clean, and for a host-bound step each kernel
    # runs inside its launch span.
    for name, dur, ts in kernel_list:
        p = _phase_of(ts, cpu_iv)
        d = phase_kern.setdefault(p, {}).setdefault(name, [0.0, 0])
        d[0] += dur
        d[1] += 1
        phase_kcount[p] = phase_kcount.get(p, 0) + 1
        phase_gpu_us[p] = phase_gpu_us.get(p, 0.0) + dur
    for name, dur, ts in op_list:
        p = _phase_of(ts, cpu_iv)
        d = phase_op.setdefault(p, {}).setdefault(name, [0.0, 0])
        d[0] += dur
        d[1] += 1
        phase_ocount[p] = phase_ocount.get(p, 0) + 1

    wall_ms = (w1 - w0) / 1000.0 if w0 is not None else 0.0
    tot = sum(v[0] for v in kern.values()) or 1.0

    summary = {}
    sp = Path(summary_path) if summary_path else trace_path.with_name("profile_summary.json")
    if sp.exists():
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    real = summary.get("real", {})
    comp = summary.get("composition_ms", {})
    step_ms = real.get("step_ms") or (wall_ms / n_real if n_real else wall_ms)
    gpu_busy_ms = gpu_busy_us / 1000.0 / n_real
    util_raw = (gpu_busy_ms / step_ms) if step_ms else 0.0
    util = min(util_raw, 1.0)
    mallocs_per_step = int(n_malloc // n_real) if n_real else n_malloc
    # util > ~100% or many allocations/step = allocator thrash / VRAM pressure,
    # not real saturation — flag it so it stops reading as "compute-bound".
    memory_pressure = mallocs_per_step >= 5 or util_raw > 1.05
    dl_ms = real.get("dataload_ms") or 0.0
    host_overhead_ms = max(step_ms - gpu_busy_ms - dl_ms, 0.0)
    launches_per_step = int(n_kern // n_real)

    def krow(name, td, denom):
        return {
            "kernel": _friendly_kernel(name),
            "raw_name": name[:90],
            "category": _kernel_category(name),
            "tensorcore": _is_tensorcore(name),
            "ms_per_step": round(td[0] / 1000.0 / n_real, 4),
            "pct_of_gpu": round(td[0] / denom * 100, 2),
            "count_per_step": max(td[1] // n_real, 1),
        }

    def orow(name, td, denom):
        return {"op": name[:70], "ms_per_step": round(td[0] / 1000.0 / n_real, 4),
                "pct_of_cpu_ops": round(td[0] / denom * 100, 2),
                "count_per_step": max(td[1] // n_real, 1)}

    kernels = sorted((krow(n, td, tot) for n, td in kern.items()),
                     key=lambda r: r["pct_of_gpu"], reverse=True)
    cats = {}
    for n, td in kern.items():
        c = _kernel_category(n)
        cats[c] = cats.get(c, 0.0) + td[0]
    categories = [{"name": c, "pct": round(v / tot * 100, 1),
                   "ms_per_step": round(v / 1000.0 / n_real, 3)}
                  for c, v in sorted(cats.items(), key=lambda kv: kv[1], reverse=True) if v > 0]
    optot = sum(v[0] for v in opagg.values()) or 1.0
    ops = sorted((orow(n, td, optot) for n, td in opagg.items()),
                 key=lambda r: r["pct_of_cpu_ops"], reverse=True)

    # Per-phase rollup + drill lists. Includes inference stages (preprocess /
    # postprocess) so `profile infer` phase tables read pre -> forward -> post;
    # they simply never appear for training traces.
    _ORDER = ["dataload", "to_device", "preprocess", "forward", "backward",
              "postprocess", "optimizer"]
    phase_names = [p for p in _ORDER if p in phase_gpu or p in phase_kern or p in phase_op]
    phase_names += [p for p in phase_kern if p not in phase_names]
    phases, kernels_by_phase, ops_by_phase = [], {}, {}
    for p in phase_names:
        ptot = sum(v[0] for v in phase_kern.get(p, {}).values()) or 1.0
        potot = sum(v[0] for v in phase_op.get(p, {}).values()) or 1.0
        kernels_by_phase[p] = sorted((krow(n, td, ptot) for n, td in phase_kern.get(p, {}).items()),
                                     key=lambda r: r["pct_of_gpu"], reverse=True)[:40]
        ops_by_phase[p] = sorted((orow(n, td, potot) for n, td in phase_op.get(p, {}).items()),
                                 key=lambda r: r["pct_of_cpu_ops"], reverse=True)[:40]
        wall = real.get("dataload_ms") if p == "dataload" else comp.get(p)
        phases.append({
            "phase": p,
            "gpu_ms_per_step": round(phase_gpu_us.get(p, 0.0) / 1000.0 / n_real, 3),
            "wall_ms_per_step": round(wall, 2) if wall is not None else None,
            "kernels_per_step": int(phase_kcount.get(p, 0) // n_real),
            "ops_per_step": int(phase_ocount.get(p, 0) // n_real),
            "unique_kernels": len(phase_kern.get(p, {})),
        })

    bound = summary.get("bound")
    why = summary.get("bound_why", "")
    if not bound:
        bound = "compute" if util >= 0.8 else "host / launch"
        why = (f"GPU ~{util * 100:.0f}% busy"
               if bound == "compute" else
               f"GPU only ~{util * 100:.0f}% busy — tiny kernels / host overhead "
               "(dataloader stall not measurable from trace alone)")
    if memory_pressure:
        bound = "memory-pressure"
        why = (f"allocator thrash (~{mallocs_per_step} cudaMalloc/free per step"
               + (f"; util read {util_raw * 100:.0f}%" if util_raw > 1.05 else "")
               + ") — VRAM pressure; throughput and utilisation here are unreliable")

    tc_pct = round(tc_us / gpu_busy_us * 100, 1) if gpu_busy_us else 0.0
    peak_vram = (summary.get("analysis") or {}).get("peak_vram_mb")

    # Flat metrics for `profile get <field>` (atomic, minimal-output queries).
    metrics = {
        "img_per_s": real.get("img_per_s"),
        "step_ms": round(step_ms, 2),
        "dataload_ms": real.get("dataload_ms"),
        "dataload_frac": real.get("dataload_frac"),
        "gpu_util": round(util, 3),
        "gpu_busy_ms": round(gpu_busy_ms, 2),
        "mean_kernel_us": round(gpu_busy_us / n_kern, 2) if n_kern else 0.0,
        "kernels_per_step": int(n_kern // n_real),
        "unique_kernels": len(kern),
        "ops_per_step": int(sum(phase_ocount.values()) // n_real),
        "memcpy_ms": round(memcpy_us / 1000.0 / n_real, 3),
        "tensorcore_pct": tc_pct,
        "peak_vram_mb": peak_vram,
        "host_overhead_ms": round(host_overhead_ms, 2),
        "launches_per_step": launches_per_step,
        "cuda_mallocs_per_step": mallocs_per_step,
        "memory_pressure": memory_pressure,
        "bound": bound,
    }
    for ph in phases:
        p = ph["phase"]
        metrics[f"{p}_gpu_ms"] = ph["gpu_ms_per_step"]
        metrics[f"{p}_wall_ms"] = ph["wall_ms_per_step"]
        metrics[f"{p}_kernels"] = ph["kernels_per_step"]
        metrics[f"{p}_ops"] = ph["ops_per_step"]

    return {
        "schema": "libreyolo.profile.analysis/v1",
        "trace": str(trace_path),
        "model": (summary.get("meta") or {}).get("model"),
        "config": summary.get("meta") or {},
        "bound": bound,
        "bound_why": why,
        "step_ms": round(step_ms, 2),
        "img_per_s": real.get("img_per_s"),
        "dataload_ms": real.get("dataload_ms"),
        "dataload_frac": real.get("dataload_frac"),
        "gpu_util": round(util, 3),
        "gpu_util_raw": round(util_raw, 3),
        "memory_pressure": memory_pressure,
        "cuda_mallocs_per_step": mallocs_per_step,
        "host_overhead_ms_per_step": round(host_overhead_ms, 2),
        "launches_per_step": launches_per_step,
        "gpu_busy_ms_per_step": round(gpu_busy_ms, 2),
        "mean_kernel_us": round(gpu_busy_us / n_kern, 2) if n_kern else 0.0,
        "kernels_per_step": int(n_kern // n_real),
        "unique_kernels": len(kern),
        "memcpy_ms_per_step": round(memcpy_us / 1000.0 / n_real, 3),
        "tensorcore_pct": tc_pct,
        "peak_vram_mb": peak_vram,
        "window": {"wall_ms": round(wall_ms, 1), "real_steps": n_real},
        "metrics": metrics,
        "categories": categories,
        "phases": phases,
        "phases_gpu": [{"phase": ph["phase"], "gpu_ms_per_step": ph["gpu_ms_per_step"]}
                       for ph in phases],
        "kernels_by_phase": kernels_by_phase,
        "ops_by_phase": ops_by_phase,
        "top_kernels": kernels[:8],
        "kernels": kernels,
        "ops": ops,
    }


class TrainStepProfiler:
    """Profiles a short window of training steps and reports the breakdown."""

    def __init__(
        self,
        *,
        device: torch.device,
        warmup: int = 5,
        active: int = 20,
        trace: bool = True,
        open_report: bool = True,
        save_dir: Optional[Path] = None,
        logger=None,
        meta: Optional[dict] = None,
    ) -> None:
        self.device = device
        self.warmup = max(0, int(warmup))
        self.active = max(2, int(active))
        self.trace = trace
        self.open_report = open_report
        self.save_dir = Path(save_dir) if save_dir else None
        self.logger = logger
        self.meta = meta or {}

        self._is_cuda = getattr(device, "type", str(device)) == "cuda"
        if self._is_cuda:
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass
        # First half of the active window is unsynced (real timing); second half
        # is synced (compute composition).
        self._half = max(1, self.active // 2)
        self._step_i = 0          # total iterations seen (incl. warmup)
        self._m = 0               # measured iterations (0-based within window)

        # Real (unsynced) timing.
        self._real_step_ms = 0.0
        self._real_step_n = 0
        self._real_dl_ms = 0.0
        self._real_dl_n = 0
        self._step_t0: Optional[float] = None

        # Synced compute composition.
        self._sums = {p: 0.0 for p in PHASES}
        self._synced_n = 0

        self.finished = False
        self.summary: Optional[dict] = None

        self._torch_prof = None
        if self.trace:
            self._start_torch_profiler()

    # -- internals -----------------------------------------------------------

    def _sync(self) -> None:
        if self._is_cuda:
            torch.cuda.synchronize(self.device)

    def _active_now(self) -> bool:
        return (not self.finished) and self._step_i >= self.warmup

    def _unsynced_phase(self) -> bool:
        return self._active_now() and self._m < self._half

    def _synced_phase(self) -> bool:
        return self._active_now() and self._m >= self._half

    def _start_torch_profiler(self) -> None:
        try:
            from torch.profiler import ProfilerActivity, profile, schedule

            acts = [ProfilerActivity.CPU]
            if self._is_cuda:
                acts.append(ProfilerActivity.CUDA)
            self._torch_prof = profile(
                activities=acts,
                schedule=schedule(
                    wait=0, warmup=self.warmup, active=self.active, repeat=1
                ),
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
            )
            self._torch_prof.start()
        except Exception:
            self._torch_prof = None  # trace is best-effort; never break training

    # -- public hooks (called from the training loop) ------------------------

    def phase(self, name: str):
        """Context for one phase: labels the trace across the whole active
        window, and additionally sync-times it during the synced half."""
        if not self._active_now():
            return _NULL
        return _PhaseTimer(self, name)

    def wrap_loader(self, iterable):
        """Yield batches while measuring real step time + dataload stall."""
        it = iter(iterable)
        while True:
            if self.finished:
                # Window closed but training continues: get out of the way so
                # the rest of the epoch pays no per-batch instrumentation.
                yield from it
                return
            t0 = time.perf_counter()
            try:
                with record_function("step/dataload"):
                    batch = next(it)
            except StopIteration:
                return
            t1 = time.perf_counter()
            if self._unsynced_phase():
                self._real_dl_ms += (t1 - t0) * 1000.0
                self._real_dl_n += 1
                self._step_t0 = t0
            else:
                self._step_t0 = None
            yield batch

    def step(self) -> None:
        """Advance one iteration; finalizes + reports when the window closes."""
        if self.finished:
            return
        t_done = time.perf_counter()
        if self._unsynced_phase() and self._step_t0 is not None:
            self._real_step_ms += (t_done - self._step_t0) * 1000.0
            self._real_step_n += 1
            self._step_t0 = None
        if self._torch_prof is not None:
            try:
                self._torch_prof.step()
            except Exception:
                pass
        if self._synced_phase():
            self._synced_n += 1
        if self._step_i >= self.warmup:
            self._m += 1
        self._step_i += 1
        if self._m >= self.active:
            self._finish()

    # -- finalize ------------------------------------------------------------

    def _finish(self) -> None:
        self.finished = True
        trace_path = None
        if self._torch_prof is not None:
            try:
                self._torch_prof.stop()
                if self.trace and self.save_dir is not None:
                    self.save_dir.mkdir(parents=True, exist_ok=True)
                    trace_path = self.save_dir / "profile_trace.json"
                    self._torch_prof.export_chrome_trace(str(trace_path))
            except Exception:
                trace_path = None
        self._build_summary(trace_path)
        timeline_path = None
        if trace_path is not None:
            curated, analysis = self._analyze_trace(trace_path)
            if analysis:
                self.summary["analysis"] = analysis
                if self._is_cuda:
                    try:
                        analysis["peak_vram_mb"] = round(
                            torch.cuda.max_memory_allocated(self.device) / 1e6, 1)
                    except Exception:
                        pass
                self._refine_verdict(analysis)
                if self.save_dir is not None:
                    try:
                        (self.save_dir / "profile_summary.json").write_text(
                            json.dumps(self.summary, indent=2)
                        )
                    except Exception:
                        pass
            if curated:
                if analysis:
                    analysis["bound"] = self.summary.get("bound")
                    analysis["bound_why"] = self.summary.get("bound_why", "")
                    analysis["step_ms"] = self.summary["real"]["step_ms"]
                    analysis["img_per_s"] = self.summary["real"]["img_per_s"]
                    analysis["dataload_ms"] = self.summary["real"]["dataload_ms"]
                timeline_path = self._write_timeline_html(curated)
            # Self-contained analysis artifact — one copyable file the CLI reads
            # (summary/get/phases/compare); the raw trace stays for kernel drill.
            if self.save_dir is not None:
                try:
                    full = analyze_trace(
                        trace_path,
                        summary_path=self.save_dir / "profile_summary.json",
                    )
                    (self.save_dir / "profile.json").write_text(json.dumps(full))
                except Exception:
                    pass
        self._report(trace_path, timeline_path)
        if self.open_report and timeline_path is not None:
            try:
                import webbrowser

                webbrowser.open(timeline_path.as_uri())
            except Exception:
                pass

    def _build_summary(self, trace_path) -> None:
        synced_n = max(self._synced_n, 1)
        comp = {p: self._sums[p] / synced_n for p in PHASES}
        comp_total = sum(comp.values()) or 1.0

        real_step = self._real_step_ms / self._real_step_n if self._real_step_n else 0.0
        real_dl = self._real_dl_ms / self._real_dl_n if self._real_dl_n else 0.0
        real_compute = max(real_step - real_dl, 0.0)
        idle_frac = (real_dl / real_step) if real_step > 0 else 0.0
        batch = self.meta.get("batch", 0) or 0
        img_s = (batch / (real_step / 1000.0)) if real_step > 0 else 0.0

        self.summary = {
            "meta": self.meta,
            "window": {
                "warmup": self.warmup,
                "real_steps": self._real_step_n,
                "synced_steps": self._synced_n,
            },
            "real": {
                "step_ms": round(real_step, 3),
                "dataload_ms": round(real_dl, 3),
                "compute_ms": round(real_compute, 3),
                "img_per_s": round(img_s, 2),
                "dataload_frac": round(idle_frac, 3),
            },
            "composition_ms": {p: round(comp[p], 3) for p in PHASES},
            "composition_total_ms": round(comp_total, 3),
            "bound": "dataloader" if idle_frac >= 0.2 else "compute",
            "bound_why": "",
            "trace": str(trace_path) if trace_path else None,
        }
        if self.save_dir is not None:
            try:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                (self.save_dir / "profile_summary.json").write_text(
                    json.dumps(self.summary, indent=2)
                )
            except Exception:
                pass

    def _refine_verdict(self, analysis) -> None:
        """3-way bottleneck classification using the trace-mined GPU util."""
        dl = self.summary["real"].get("dataload_frac", 0.0)
        # Tie utilisation to our honest unsynced step time (the trace window's
        # wall can include the profiler's own sync overhead); GPU-busy is stable.
        real_step = self.summary["real"].get("step_ms", 0.0)
        busy = analysis.get("gpu_busy_ms_per_step", 0.0)
        util = (busy / real_step) if real_step > 0 else analysis.get("gpu_util", 0.0)
        analysis["gpu_util_raw"] = round(util, 3)
        analysis["gpu_util"] = round(min(util, 1.0), 3)
        memory_pressure = (
            analysis.get("cuda_mallocs_per_step", 0) >= 5
            or util > 1.05
        )
        analysis["memory_pressure"] = memory_pressure
        if memory_pressure:
            bound = "memory-pressure"
            why = (
                f"allocator/VRAM pressure (~{analysis.get('cuda_mallocs_per_step', 0)} "
                "cudaMalloc/free per step); throughput and utilisation are unreliable"
            )
        elif dl >= 0.2:
            bound = "dataloader"
            why = f"the GPU waits on input data (~{dl * 100:.0f}% of the step)"
        elif analysis["gpu_util"] >= 0.8:
            bound = "compute"
            why = f"the GPU is saturated (~{analysis['gpu_util'] * 100:.0f}% busy)"
        else:
            bound = "host / launch"
            why = (f"GPU only ~{analysis['gpu_util'] * 100:.0f}% busy — fed too slowly: "
                   f"{analysis['kernels_per_step']} kernels/step at "
                   f"~{analysis['mean_kernel_us']:.0f}us each plus host overhead")
        self.summary["bound"] = bound
        self.summary["bound_why"] = why

    def _emit(self, line: str) -> None:
        if self.logger is not None:
            self.logger.info(line)
        else:
            print(line)

    def _report(self, trace_path, timeline_path=None) -> None:
        s = self.summary
        r = s["real"]
        m = self.meta
        an = s.get("analysis")
        bar_w = 20

        self._emit("=" * 64)
        self._emit(f"  LibreYOLO training profile — {m.get('model', '?')}")
        self._emit(
            f"  device={m.get('device','?')}  batch={m.get('batch','?')}  "
            f"imgsz={m.get('imgsz','?')}  amp={m.get('amp','?')}  workers={m.get('workers','?')}"
        )
        self._emit(
            f"  REAL step {r['step_ms']:.1f} ms = dataload {r['dataload_ms']:.1f} ms "
            f"+ compute {r['compute_ms']:.1f} ms  ->  {r['img_per_s']:.1f} img/s"
        )
        if an:
            self._emit(
                f"  GPU util {an['gpu_util'] * 100:.0f}%  "
                f"({an['gpu_busy_ms_per_step']:.0f} ms GPU-busy / {r['step_ms']:.0f} ms step)  |  "
                f"{an['kernels_per_step']} kernels/step @ ~{an['mean_kernel_us']:.0f}us"
            )
            bits = []
            if an.get("peak_vram_mb"):
                bits.append(f"peak VRAM {an['peak_vram_mb']:.0f} MB")
            bits.append(f"Tensor Cores {an.get('tensorcore_pct', 0):.0f}% of GPU time")
            if an.get("memcpy_ms_per_step"):
                bits.append(f"memcpy {an['memcpy_ms_per_step']:.1f} ms/step")
            self._emit("  " + "  |  ".join(bits))
        label = {"dataloader": "DATALOADER-BOUND", "host / launch": "HOST/LAUNCH-BOUND",
                 "compute": "COMPUTE-BOUND", "memory-pressure": "MEMORY-PRESSURE"}.get(
                     s["bound"], str(s["bound"]).upper())
        self._emit(f"  >> VERDICT: {label} — {s.get('bound_why', '')}.")
        levers = {
            "dataloader": "workers↑, cache='ram'/'disk', lighter aug, larger batch",
            "host / launch": "larger batch (amortize launches), fewer per-step .item()/syncs, CUDA graphs, op fusion",
            "compute": "already GPU-bound — try AMP/bf16 or a larger model",
            "memory-pressure": "lower batch, reduce activation memory, avoid running at the OOM edge",
        }
        self._emit(f"     Levers: {levers.get(s['bound'], '')}.")
        if an and not self.meta.get("amp"):
            self._emit("     note: running fp32 — AMP/bf16 would push gemm/conv onto "
                       f"Tensor Cores (now {an.get('tensorcore_pct', 0):.0f}%).")
        if an and an.get("categories"):
            self._emit("  GPU kernel mix:")
            for c in an["categories"]:
                fill = int(round(c["pct"] / 100.0 * bar_w))
                self._emit(f"    {c['name']:<17} {c['pct']:5.1f}%  |"
                           + "#" * fill + "." * (bar_w - fill)
                           + f"|  {c['ms']:.1f} ms/step")
            self._emit("  top kernels (per step):")
            for k in an["top_kernels"]:
                self._emit(f"    {k['pct']:5.1f}%  {k['ms']:6.2f} ms  x{k['count']:<4} {k['name']}")
        self._emit("  per-phase wall (synced): "
                   + " · ".join(f"{p} {s['composition_ms'][p]:.0f}ms" for p in PHASES))
        if timeline_path:
            self._emit(f"  timeline:  {timeline_path}")
            if self.open_report:
                self._emit("             (opening in your browser…)")
        if trace_path:
            self._emit(f"  raw trace: {trace_path}  (load in Perfetto/Nsight if you want)")
        self._emit("=" * 64)

    def _analyze_trace(self, trace_path):
        """Parse the trace once; return ``(curated_timeline, analysis)``.

        ``analysis`` mines the GPU truth — achieved utilization (kernel busy
        time / wall), kernel mix and top kernels, mean kernel size — over a
        steady multi-step window. ``curated_timeline`` is the bounded event set
        for the viewer (phase bands, capped cpu/gpu lanes, CPU->GPU flows).
        """
        try:
            data = json.loads(Path(trace_path).read_text(encoding="utf-8"))
        except Exception:
            return None, None
        events = data.get("traceEvents", []) if isinstance(data, dict) else data

        steps = sorted(
            (e for e in events
             if e.get("ph") == "X" and str(e.get("name", "")).startswith("ProfilerStep")),
            key=lambda e: e.get("ts", 0),
        )
        # Robust window: a contiguous span of several step markers, skipping the
        # first two (cold). Spanning multiple steps is immune to torch's
        # duplicate big/small ProfilerStep markers.
        if len(steps) >= 4:
            w0 = steps[2]["ts"]
            w1 = steps[min(len(steps) - 1, 8)]["ts"]
        elif steps:
            w0 = steps[0]["ts"]
            w1 = steps[-1]["ts"] + (steps[-1].get("dur", 0) or 0)
        else:
            w0 = w1 = None
        if w0 is not None and w1 is not None and w1 <= w0:
            w1 = w0 + 1.0

        def in_win(ts):
            return w0 is None or (w0 <= ts < w1)

        LANE = {"pcpu": 0, "cpu": 1, "pgpu": 2, "gpu": 3, "mem": 4}
        kept = []
        kern_total, kern_count = {}, {}
        gpu_busy_us = memcpy_us = tc_us = 0.0
        n_kern = n_malloc = 0
        for e in events:
            if e.get("ph") != "X":
                continue
            cat = e.get("cat", "")
            ts = e.get("ts")
            dur = e.get("dur", 0) or 0
            if ts is None or not in_win(ts):
                continue
            name = str(e.get("name", ""))
            if cat == "kernel":
                gpu_busy_us += dur
                n_kern += 1
                if _is_tensorcore(name):
                    tc_us += dur
                kern_total[name] = kern_total.get(name, 0.0) + dur
                kern_count[name] = kern_count.get(name, 0) + 1
                if dur >= 2:
                    kept.append({"name": name[:64], "_ts": ts, "_dur": dur,
                                 "lane": LANE["gpu"], "cat": "gpu"})
            elif cat in ("gpu_memcpy", "gpu_memset"):
                memcpy_us += dur
                kept.append({"name": name[:64], "_ts": ts, "_dur": dur,
                             "lane": LANE["mem"], "cat": "mem"})
            elif cat == "cuda_runtime" and ("Malloc" in name or "Free" in name):
                n_malloc += 1
            elif cat == "user_annotation" and name.startswith("step/"):
                kept.append({"name": name[5:], "_ts": ts, "_dur": dur,
                             "lane": LANE["pcpu"], "cat": "pcpu"})
            elif cat == "gpu_user_annotation" and name.startswith("step/"):
                kept.append({"name": name[5:], "_ts": ts, "_dur": dur,
                             "lane": LANE["pgpu"], "cat": "pgpu"})
            elif cat == "cpu_op" and dur >= 20:
                kept.append({"name": name[:64], "_ts": ts, "_dur": dur,
                             "lane": LANE["cpu"], "cat": "cpu"})
        if not kept:
            return None, None

        # ---- analysis: the GPU truth ----
        wall_ms = (w1 - w0) / 1000.0 if (w0 is not None and w1 is not None) else 0.0
        n_real = _count_steps_in_window(steps, w0, w1)
        gpu_busy_ms = gpu_busy_us / 1000.0
        util = (gpu_busy_ms / wall_ms) if wall_ms > 0 else 0.0
        mallocs_per_step = int(n_malloc // n_real) if n_real else n_malloc
        tot = sum(kern_total.values()) or 1.0
        top = sorted(kern_total.items(), key=lambda kv: kv[1], reverse=True)[:6]
        top_kernels = [{"name": _friendly_kernel(n), "pct": round(v / tot * 100, 1),
                        "ms": round(v / 1000.0 / n_real, 3),
                        "count": max(kern_count[n] // n_real, 1)} for n, v in top]
        cats = {}
        for n, v in kern_total.items():
            c = _kernel_category(n)
            cats[c] = cats.get(c, 0.0) + v
        categories = [{"name": c, "pct": round(v / tot * 100, 1),
                       "ms": round(v / 1000.0 / n_real, 3)}
                      for c, v in sorted(cats.items(), key=lambda kv: kv[1], reverse=True) if v > 0]
        analysis = {
            "gpu_util": round(util, 3),
            "gpu_busy_ms_per_step": round(gpu_busy_ms / n_real, 2),
            "mean_kernel_us": round((gpu_busy_us / n_kern) if n_kern else 0.0, 2),
            "kernels_per_step": int(n_kern // n_real),
            "memcpy_ms_per_step": round(memcpy_us / 1000.0 / n_real, 3),
            "tensorcore_pct": round(tc_us / gpu_busy_us * 100, 1) if gpu_busy_us else 0.0,
            "cuda_mallocs_per_step": mallocs_per_step,
            "gpu_util_raw": round(util, 3),
            "memory_pressure": mallocs_per_step >= 5 or util > 1.05,
            "window_real_steps": n_real,
            "top_kernels": top_kernels,
            "categories": categories,
        }

        # ---- curate: bounded event set for the viewer ----
        def _longest(c, n):
            evs = [e for e in kept if e["cat"] == c]
            return evs if len(evs) <= n else sorted(
                evs, key=lambda e: e["_dur"], reverse=True)[:n]

        kept = ([e for e in kept if e["cat"] in ("pcpu", "pgpu", "mem")]
                + _longest("cpu", 1500) + _longest("gpu", 2500))
        base = min(e["_ts"] for e in kept)
        for e in kept:
            e["t"] = round((e["_ts"] - base) / 1000.0, 3)
            e["d"] = round(e["_dur"] / 1000.0, 4)
            del e["_ts"]
            del e["_dur"]

        lane_rows = {}
        for li in range(5):
            evs = sorted([e for e in kept if e["lane"] == li], key=lambda x: x["t"])
            ends = []
            for e in evs:
                placed = False
                for i, end in enumerate(ends):
                    if e["t"] >= end:
                        e["row"] = i
                        ends[i] = e["t"] + max(e["d"], 0.001)
                        placed = True
                        break
                if not placed:
                    e["row"] = len(ends)
                    ends.append(e["t"] + max(e["d"], 0.001))
            lane_rows[li] = max(len(ends), 1)

        s_by_id, f_by_id = {}, {}
        for e in events:
            if e.get("cat") != "ac2g":
                continue
            ph, i, ts = e.get("ph"), e.get("id"), e.get("ts")
            if i is None or ts is None or not in_win(ts):
                continue
            if ph == "s":
                s_by_id[i] = ts
            elif ph == "f":
                f_by_id[i] = ts
        flows = []
        for i, sts in s_by_id.items():
            fts = f_by_id.get(i)
            if fts is None:
                continue
            flows.append([round((sts - base) / 1000.0, 3),
                          round((fts - base) / 1000.0, 3)])
            if len(flows) >= 4000:
                break

        total = max((e["t"] + e["d"] for e in kept), default=1.0)
        curated = {"events": kept, "flows": flows, "lane_rows": lane_rows,
                   "total_ms": round(total, 3), "meta": self.meta, "analysis": analysis}
        return curated, analysis

    def _write_timeline_html(self, curated):
        """Render the curated trace into a self-contained timeline.html."""
        if self.save_dir is None or not curated:
            return None
        import json
        try:
            payload = json.dumps(curated).replace("</", "<\\/")
            page = _TIMELINE_HTML.replace("/*__DATA__*/", payload)
            path = self.save_dir / "timeline.html"
            path.write_text(page, encoding="utf-8")
            return path
        except Exception:
            return None


class _PhaseTimer:
    """Labels the trace via record_function; sync-times only in the synced half."""

    __slots__ = ("prof", "name", "_t0", "_rf")

    def __init__(self, prof: TrainStepProfiler, name: str) -> None:
        self.prof = prof
        self.name = name
        self._t0 = None
        self._rf = None

    def __enter__(self):
        self._rf = record_function("step/" + self.name)
        self._rf.__enter__()
        if self.prof._synced_phase():
            self.prof._sync()
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._t0 is not None:
            self.prof._sync()
            self.prof._sums[self.name] += (time.perf_counter() - self._t0) * 1000.0
        if self._rf is not None:
            self._rf.__exit__(*exc)
        return False


_TIMELINE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>LibreYOLO profiler - timeline</title>
<style>
 html,body{margin:0;background:#0f1115;color:#e6e6e6;font-family:Segoe UI,Roboto,sans-serif;overflow:hidden}
 #hdr{padding:7px 12px;font-size:13px;color:#cbd3df;border-bottom:1px solid #20242c;white-space:nowrap;overflow:hidden}
 #hdr b{color:#fff}
 .sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin:0 3px 0 9px;vertical-align:middle}
 #tip{position:fixed;pointer-events:none;background:rgba(0,0,0,.88);border:1px solid #333;
      padding:4px 8px;font-size:12px;border-radius:5px;display:none;white-space:nowrap;z-index:9;max-width:60vw;overflow:hidden}
 canvas{display:block;cursor:grab}
 #ptoggle{position:fixed;right:8px;top:5px;z-index:10;background:#262b36;color:#cbd3df;border:0;
      border-radius:5px;padding:3px 10px;font-size:12px;cursor:pointer}
 #panel{position:fixed;right:0;top:0;width:332px;height:100vh;overflow:auto;box-sizing:border-box;
      background:rgba(18,21,28,.96);border-left:1px solid #262b36;padding:34px 14px 30px;font-size:12px;z-index:8}
 #panel.hidden{display:none}
 #panel .verdict{font-weight:700;color:#fff;padding:8px 10px;border-radius:7px;margin:2px 0 10px;text-align:center}
 #panel .util{font-size:26px;font-weight:700;color:#fff}
 #panel .sub{color:#8b95a5;margin:2px 0}
 #panel .why{color:#9aa4b2;font-size:11px;margin:8px 0 2px;line-height:1.35}
 #panel h4{margin:14px 0 6px;color:#cbd3df;font-size:12px}
 #panel .row{display:flex;align-items:center;gap:6px;margin:3px 0}
 #panel .nm{flex:0 0 118px;color:#cbd3df;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 #panel .barwrap{flex:1;background:#20242c;border-radius:3px;height:11px;overflow:hidden}
 #panel .barfill{height:11px}
 #panel .pc{flex:0 0 42px;text-align:right;color:#8b95a5}
</style></head><body>
<div id="hdr"></div><canvas id="c"></canvas><div id="tip"></div>
<button id="ptoggle" onclick="togglePanel()">hide analysis</button><div id="panel"></div>
<script>
var DATA = /*__DATA__*/;
var cv=document.getElementById('c'),ctx=cv.getContext('2d'),tip=document.getElementById('tip'),
    hdr=document.getElementById('hdr');
var LEFT=128,TOP=28,ROW=12,GAP=12;
var LANES=['phase · CPU','CPU ops','phase · GPU','GPU kernels','GPU memcpy'];
var PH={dataload:'#e67e22',to_device:'#f1c40f',forward:'#3498db',backward:'#9b59b6',optimizer:'#2ecc71'};
function col(e){if(e.cat==='pcpu'||e.cat==='pgpu')return PH[e.name]||'#7f8c8d';
 if(e.cat==='cpu')return '#5d6d7e';if(e.cat==='gpu')return '#16a085';return '#e74c3c';}
var laneY=[],yy=TOP+8;
for(var i=0;i<5;i++){laneY[i]=yy;yy+=(DATA.lane_rows[i]||1)*ROW+GAP;}
var pxPerMs=1,viewStart=0,W=0,H=0,fitted=false;
function xOf(t){return LEFT+(t-viewStart)*pxPerMs;}
function fit(){pxPerMs=(W-LEFT-20)/(DATA.total_ms||1);viewStart=0;}
function resize(){W=cv.width=window.innerWidth;H=cv.height=window.innerHeight-hdr.offsetHeight;
 if(!fitted){fit();fitted=true;}draw();}
function niceStep(span){var raw=span/10,p=Math.pow(10,Math.floor(Math.log10(raw||1))),n=raw/p;
 n=n<1.5?1:n<3?2:n<7?5:10;return Math.max(n*p,0.001);}
function draw(){
 ctx.clearRect(0,0,W,H);
 ctx.strokeStyle='rgba(130,170,210,.08)';ctx.lineWidth=1;ctx.beginPath();
 var cy=laneY[1]+(DATA.lane_rows[1]||1)*ROW,gy=laneY[3];
 for(var f=0;f<DATA.flows.length;f++){var a=DATA.flows[f],xa=xOf(a[0]),xb=xOf(a[1]);
  if((xa<LEFT&&xb<LEFT)||(xa>W&&xb>W))continue;ctx.moveTo(xa,cy);ctx.lineTo(xb,gy);}
 ctx.stroke();
 for(var k=0;k<DATA.events.length;k++){var e=DATA.events[k],ex=xOf(e.t),ew=Math.max(e.d*pxPerMs,1);
  if(ex+ew<LEFT||ex>W)continue;var ey=laneY[e.lane]+e.row*ROW;
  ctx.fillStyle=col(e);ctx.fillRect(ex<LEFT?LEFT:ex,ey,ew,ROW-1);}
 ctx.fillStyle='#0f1115';ctx.fillRect(0,0,LEFT,H);
 ctx.fillStyle='#cbd3df';ctx.font='11px Segoe UI';ctx.textBaseline='top';
 for(var i=0;i<5;i++)ctx.fillText(LANES[i],8,laneY[i]);
 ctx.fillStyle='#0f1115';ctx.fillRect(0,0,W,TOP);
 ctx.strokeStyle='#222';ctx.beginPath();ctx.moveTo(0,TOP);ctx.lineTo(W,TOP);ctx.stroke();
 var span=(W-LEFT)/pxPerMs,st=niceStep(span),t0=Math.floor(viewStart/st)*st;
 ctx.font='10px Segoe UI';
 for(var t=t0;xOf(t)<W;t+=st){var px=xOf(t);if(px<LEFT)continue;
  ctx.strokeStyle='#171a21';ctx.beginPath();ctx.moveTo(px,TOP);ctx.lineTo(px,H);ctx.stroke();
  ctx.fillStyle='#8b95a5';ctx.fillText((st<1?t.toFixed(2):t.toFixed(0))+' ms',px+3,9);}
}
var drag=false,lx=0;
cv.addEventListener('mousedown',function(e){drag=true;lx=e.clientX;cv.style.cursor='grabbing';});
window.addEventListener('mouseup',function(){drag=false;cv.style.cursor='grab';});
window.addEventListener('mousemove',function(e){
 if(drag){viewStart-=(e.clientX-lx)/pxPerMs;lx=e.clientX;draw();return;}
 var r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,found=null;
 if(mx>LEFT)for(var k=0;k<DATA.events.length;k++){var ev=DATA.events[k],ex=xOf(ev.t),
  ew=Math.max(ev.d*pxPerMs,1),ey=laneY[ev.lane]+ev.row*ROW;
  if(mx>=ex&&mx<=ex+ew&&my>=ey&&my<=ey+ROW)found=ev;}
 if(found){tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+14)+'px';
  tip.innerHTML='<b>'+found.name+'</b> &mdash; '+found.d.toFixed(found.d<1?3:2)+' ms';}
 else tip.style.display='none';
});
cv.addEventListener('wheel',function(e){e.preventDefault();var r=cv.getBoundingClientRect(),
 mx=e.clientX-r.left,mt=viewStart+(mx-LEFT)/pxPerMs,fac=e.deltaY<0?1.2:1/1.2;
 pxPerMs*=fac;viewStart=mt-(mx-LEFT)/pxPerMs;draw();},{passive:false});
hdr.innerHTML='LibreYOLO profiler &mdash; <b>'+(DATA.meta.model||'')+'</b> &middot; batch '+
 (DATA.meta.batch||'?')+' &middot; imgsz '+(DATA.meta.imgsz||'?')+' &middot; '+DATA.events.length+' events'+
 '<span style="color:#8b95a5;font-size:11px">'+
 '<i class=sw style="background:#3498db"></i>forward<i class=sw style="background:#9b59b6"></i>backward'+
 '<i class=sw style="background:#e67e22"></i>dataload<i class=sw style="background:#16a085"></i>kernel'+
 '<i class=sw style="background:#e74c3c"></i>memcpy &middot; scroll=zoom drag=pan</span>';
var A=DATA.analysis||{};
function boundColor(b){return b==='dataloader'?'#c0392b':(b==='compute'?'#1e8449':'#d35400');}
function buildPanel(){
 var p=document.getElementById('panel'),tg=document.getElementById('ptoggle');
 if(A.gpu_util===undefined){p.style.display='none';tg.style.display='none';return;}
 var CC={'gemm / conv':'#3498db','layout / copy':'#e67e22','elementwise':'#9b59b6','reduction / norm':'#1abc9c','other':'#7f8c8d'};
 var h='<div class="verdict" style="background:'+boundColor(A.bound)+'">'+String(A.bound||'').toUpperCase()+'</div>';
 h+='<div class="util">GPU '+Math.round(A.gpu_util*100)+'%</div>';
 h+='<div class="sub">'+A.gpu_busy_ms_per_step+' ms busy / '+Math.round(A.step_ms||0)+' ms step &middot; '+(A.img_per_s||'?')+' img/s</div>';
 h+='<div class="sub">'+A.kernels_per_step+' kernels/step @ ~'+Math.round(A.mean_kernel_us)+' &micro;s &middot; memcpy '+A.memcpy_ms_per_step+' ms</div>';
 h+='<div class="sub">peak VRAM '+(A.peak_vram_mb||'?')+' MB &middot; Tensor Cores '+(A.tensorcore_pct||0)+'% of GPU</div>';
 h+='<div class="why">'+(A.bound_why||'')+'</div>';
 h+='<h4>GPU kernel mix</h4>';
 (A.categories||[]).forEach(function(c){h+='<div class="row"><div class="nm">'+c.name+'</div><div class="barwrap"><div class="barfill" style="width:'+c.pct+'%;background:'+(CC[c.name]||'#7f8c8d')+'"></div></div><div class="pc">'+c.pct+'%</div></div>';});
 h+='<h4>top kernels / step</h4>';
 (A.top_kernels||[]).forEach(function(k){h+='<div class="row"><div class="nm" title="'+k.name+'">'+k.name+'</div><div class="pc">'+k.ms.toFixed(2)+'ms</div><div class="pc">'+k.pct+'%</div></div>';});
 p.innerHTML=h;
}
function togglePanel(){var p=document.getElementById('panel'),tg=document.getElementById('ptoggle');
 p.classList.toggle('hidden');tg.textContent=p.classList.contains('hidden')?'show analysis':'hide analysis';}
buildPanel();
window.addEventListener('resize',resize);resize();
</script></body></html>"""
