"""Unit tests for the training profiler analysis (no GPU required).

The analysis logic is exercised against a small synthetic torch chrome-trace so
it runs in CI without a GPU or a real training step.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import pytest
import torch

from libreyolo.training.config import TrainConfig
from libreyolo.training.profiler import (
    TrainStepProfiler,
    _friendly_kernel,
    _is_tensorcore,
    _kernel_category,
    _pick_window,
    analyze_trace,
)

pytestmark = pytest.mark.unit


def _trace_events():
    """A tiny trace: 6 steps, 3 analysed steps, 5 kernels/step, 2 ops/step."""
    ev = [
        {"ph": "X", "cat": "user_annotation", "name": f"ProfilerStep#{i}",
         "ts": i * 100, "dur": 100, "pid": 1, "tid": 1}
        for i in range(6)
    ]
    # CPU launch spans inside the analysis window [200, 500).
    for base in (200, 300, 400):
        ev.append({"ph": "X", "cat": "user_annotation", "name": "step/forward",
                   "ts": base + 10, "dur": 35, "pid": 1, "tid": 1})
        ev.append({"ph": "X", "cat": "user_annotation", "name": "step/backward",
                   "ts": base + 50, "dur": 40, "pid": 1, "tid": 1})
        # kernels: fwd: sgemm, elementwise, nchwToNhwc; bwd: bn_bw, h16816gemm(TC)
        for name, offset, dur in [
            ("ampere_sgemm_128x128_nt", 12, 10),
            ("void at::native::vectorized_elementwise_kernel<f>", 23, 5),
            ("_ZN5cudnn19engines16nchwToNhwcKernel", 31, 8),
            ("_ZN5cudnn21bn_bw_1C11_kernel_new", 55, 20),
            ("ampere_h16816gemm_128x128", 78, 15),
        ]:
            ev.append({"ph": "X", "cat": "kernel", "name": name,
                       "ts": base + offset, "dur": dur, "pid": 0, "tid": 7})
        ev.append({"ph": "X", "cat": "cpu_op", "name": "aten::conv2d",
                   "ts": base + 15, "dur": 40, "pid": 1, "tid": 1})
        ev.append({"ph": "X", "cat": "cpu_op", "name": "aten::convolution_backward",
                   "ts": base + 55, "dur": 50, "pid": 1, "tid": 1})
        ev.append({"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD",
                   "ts": base + 5, "dur": 3, "pid": 0, "tid": 7})
    return ev


def write_synthetic_trace(directory: Path, *, with_summary: bool = False) -> Path:
    directory = Path(directory)
    trace = directory / "profile_trace.json"
    trace.write_text(json.dumps({"traceEvents": _trace_events()}))
    if with_summary:
        (directory / "profile_summary.json").write_text(json.dumps({
            "meta": {"model": "YOLOv9-t", "batch": 16},
            "real": {"step_ms": 100.0, "img_per_s": 10.0, "dataload_ms": 1.0,
                     "dataload_frac": 0.01},
            "composition_ms": {"forward": 0.05, "backward": 0.06,
                               "to_device": 0.01, "optimizer": 0.005},
            "bound": "dataloader", "bound_why": "from summary",
            "analysis": {"peak_vram_mb": 1234.0},
        }))
    return trace


# --- pure helpers -----------------------------------------------------------

def test_kernel_category():
    assert _kernel_category("_ZN5cudnn21bn_bw_1C11_kernel") == "reduction / norm"
    assert _kernel_category("ampere_sgemm_128x128") == "gemm / conv"
    assert _kernel_category("_ZN5cudnn16nchwToNhwcKernel") == "layout / copy"
    assert _kernel_category("vectorized_elementwise_kernel") == "elementwise"
    # a conv whose mangled name embeds nhwc must NOT be miscounted as layout
    assert _kernel_category("cudnn_implicit_convolution_nhwc") == "gemm / conv"


def test_is_tensorcore():
    assert _is_tensorcore("ampere_h16816gemm_128x128") is True
    assert _is_tensorcore("sm80_xmma_tensorop_conv") is True
    assert _is_tensorcore("ampere_sgemm_128x128") is False


def test_friendly_kernel():
    assert _friendly_kernel("_ZN5cudnn21bn_bw_x") == "cudnn batchnorm (bwd)"
    assert _friendly_kernel("ampere_h16816gemm") == "gemm"


def test_pick_window():
    steps = [{"ts": i * 100, "dur": 100} for i in range(6)]
    w0, w1, n_real = _pick_window(steps)
    assert (w0, w1) == (200, 500)
    assert n_real == 3


def test_pick_window_counts_equal_duration_steps():
    steps = [
        {"name": f"ProfilerStep#{i}", "ts": i * 100, "dur": 100}
        for i in range(6)
    ]
    assert _pick_window(steps) == (200, 500, 3)


# --- analyze_trace ----------------------------------------------------------

def test_analyze_trace_metrics(tmp_path):
    a = analyze_trace(write_synthetic_trace(tmp_path))
    assert a["kernels_per_step"] == 5
    assert a["mean_kernel_us"] == pytest.approx(58 / 5, rel=0.05)
    assert a["tensorcore_pct"] == pytest.approx(15 / 58 * 100, abs=1.0)
    assert a["bound"] == "host / launch"
    cats = {c["name"]: c["pct"] for c in a["categories"]}
    assert cats["gemm / conv"] == pytest.approx(25 / 58 * 100, abs=1.0)
    assert "reduction / norm" in cats
    assert a["top_kernels"][0]["kernel"] == "cudnn batchnorm (bwd)"


def test_analyze_trace_phases(tmp_path):
    a = analyze_trace(write_synthetic_trace(tmp_path))
    ph = {p["phase"]: p for p in a["phases"]}
    assert ph["forward"]["kernels_per_step"] == 3
    assert ph["backward"]["kernels_per_step"] == 2
    assert ph["forward"]["ops_per_step"] == 1
    assert ph["backward"]["ops_per_step"] == 1
    # per-phase GPU time reconciles to total GPU-busy
    total = sum(p["gpu_ms_per_step"] for p in a["phases"])
    assert total == pytest.approx(a["gpu_busy_ms_per_step"], rel=0.05)
    assert a["metrics"]["forward_kernels"] == 3
    assert a["kernels_by_phase"]["backward"][0]["kernel"] == "cudnn batchnorm (bwd)"


def test_analyze_trace_uses_summary(tmp_path):
    a = analyze_trace(write_synthetic_trace(tmp_path, with_summary=True))
    assert a["bound"] == "dataloader"
    assert a["img_per_s"] == 10.0
    assert a["step_ms"] == 100.0
    assert a["peak_vram_mb"] == 1234.0


# --- TrainStepProfiler core (no real model, no trace) -----------------------

def test_profiler_disabled_by_default():
    cfg = TrainConfig()
    assert cfg.profile is False
    assert cfg.profile_open is True


def test_trainstepprofiler_runs_without_trace(tmp_path):
    prof = TrainStepProfiler(
        device=torch.device("cpu"), warmup=1, active=2, trace=False,
        open_report=False, save_dir=tmp_path, meta={"model": "t", "batch": 4},
    )
    for _ in prof.wrap_loader(range(20)):
        for name in ("to_device", "forward", "backward", "optimizer"):
            with prof.phase(name):
                pass
        prof.step()
        if prof.finished:
            break
    assert prof.finished
    assert prof.summary is not None
    assert prof.summary["real"]["step_ms"] >= 0


def test_profiler_real_timing_excludes_warmup_work(tmp_path):
    # Use a large warmup sleep vs. a negligible active sleep so the assertion is
    # relative to the warmup cost, not a tight absolute wall-clock budget: the
    # "real" step time must reflect only the active steps and therefore stay well
    # below the warmup sleep, even on slow/loaded CI runners (the previous fixed
    # 30ms threshold flaked on macOS where sleep/scheduling jitter alone neared it).
    warmup_sleep_s = 0.3
    prof = TrainStepProfiler(
        device=torch.device("cpu"), warmup=1, active=2, trace=False,
        open_report=False, save_dir=tmp_path, meta={"model": "t", "batch": 1},
    )
    for idx, _ in enumerate(prof.wrap_loader(range(3))):
        time.sleep(warmup_sleep_s if idx == 0 else 0.001)
        prof.step()
        if prof.finished:
            break

    assert prof.summary is not None
    # Warmup excluded -> real step time is far below the warmup sleep.
    assert prof.summary["real"]["step_ms"] < warmup_sleep_s * 1000 / 2


def test_analyze_trace_host_overhead_and_schema(tmp_path):
    a = analyze_trace(write_synthetic_trace(tmp_path, with_summary=True))
    assert a["schema"] == "libreyolo.profile.analysis/v1"
    assert a["launches_per_step"] == 5
    assert a["memory_pressure"] is False
    # step 100 ms, GPU busy ~0.06 ms, dataload 1 ms -> host overhead dominates
    assert a["host_overhead_ms_per_step"] > 90
    assert a["metrics"]["host_overhead_ms"] == a["host_overhead_ms_per_step"]
    assert a["metrics"]["launches_per_step"] == 5


def test_memory_pressure_detected(tmp_path):
    ev = _trace_events()
    for i in range(20):
        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaMalloc",
                   "ts": 250 + i, "dur": 1, "pid": 1, "tid": 1})
    trace = tmp_path / "profile_trace.json"
    trace.write_text(json.dumps({"traceEvents": ev}))
    a = analyze_trace(trace)
    assert a["memory_pressure"] is True
    assert a["bound"] == "memory-pressure"
    assert a["cuda_mallocs_per_step"] >= 5
