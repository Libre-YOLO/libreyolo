"""CLI tests for the ``libreyolo profile`` command group.

Exercises every lens (summary/get/phases/kernels/ops/compare) against a small
synthetic trace, so it runs in CI without a GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

pytestmark = pytest.mark.unit
runner = CliRunner()


def _write_trace(directory: Path) -> Path:
    ev = [
        {"ph": "X", "cat": "user_annotation", "name": f"ProfilerStep#{i}",
         "ts": i * 100, "dur": 100, "pid": 1, "tid": 1}
        for i in range(6)
    ]
    for base in (200, 300, 400):
        ev.append({"ph": "X", "cat": "user_annotation", "name": "step/forward",
                   "ts": base + 10, "dur": 35, "pid": 1, "tid": 1})
        ev.append({"ph": "X", "cat": "user_annotation", "name": "step/backward",
                   "ts": base + 50, "dur": 40, "pid": 1, "tid": 1})
        for name, offset, dur in [
            ("ampere_sgemm_128x128_nt", 20, 10),
            ("_ZN5cudnn21bn_bw_1C11_kernel_new", 60, 20),
            ("ampere_h16816gemm_128x128", 80, 15),
        ]:
            ev.append({"ph": "X", "cat": "kernel", "name": name,
                       "ts": base + offset, "dur": dur, "pid": 0, "tid": 7})
        ev.append({"ph": "X", "cat": "cpu_op", "name": "aten::conv2d",
                   "ts": base + 15, "dur": 40, "pid": 1, "tid": 1})
        ev.append({"ph": "X", "cat": "cpu_op", "name": "aten::convolution_backward",
                   "ts": base + 55, "dur": 50, "pid": 1, "tid": 1})
    trace = Path(directory) / "profile_trace.json"
    trace.write_text(json.dumps({"traceEvents": ev}))
    (Path(directory) / "profile_summary.json").write_text(json.dumps({
        "meta": {"model": "YOLOv9-t", "batch": 16},
        "real": {"step_ms": 0.3, "img_per_s": 53.3, "dataload_ms": 0.0,
                 "dataload_frac": 0.0},
        "composition_ms": {"forward": 0.05, "backward": 0.06},
        "bound": "host / launch", "bound_why": "test",
    }))
    return trace


def _app() -> typer.Typer:
    from libreyolo.cli.commands import profile

    app = typer.Typer(add_completion=False)
    app.add_typer(profile.profile_app, name="profile")
    return app


def test_summary(tmp_path):
    r = runner.invoke(_app(), ["profile", "summary", str(_write_trace(tmp_path))])
    assert r.exit_code == 0
    assert "kernel mix" in r.stdout
    assert "cudnn batchnorm (bwd)" in r.stdout


def test_get_atomic(tmp_path):
    r = runner.invoke(_app(), ["profile", "get", str(_write_trace(tmp_path)), "kernels_per_step"])
    assert r.exit_code == 0
    assert r.stdout.strip() == "3"


def test_get_json(tmp_path):
    r = runner.invoke(_app(), ["profile", "get", str(_write_trace(tmp_path)),
                               "forward_kernels", "--json"])
    assert r.exit_code == 0
    assert json.loads(r.stdout.strip()) == {"forward_kernels": 1}


def test_get_unknown_field(tmp_path):
    r = runner.invoke(_app(), ["profile", "get", str(_write_trace(tmp_path)), "nonsense"])
    assert r.exit_code == 2


def test_phases(tmp_path):
    r = runner.invoke(_app(), ["profile", "phases", str(_write_trace(tmp_path))])
    assert r.exit_code == 0
    assert "forward" in r.stdout and "backward" in r.stdout


def test_kernels_phase_filter(tmp_path):
    r = runner.invoke(_app(), ["profile", "kernels", str(_write_trace(tmp_path)),
                               "--phase", "backward", "--json"])
    assert r.exit_code == 0
    names = [k["kernel"] for k in json.loads(r.stdout)["kernels"]]
    assert "cudnn batchnorm (bwd)" in names


def test_kernels_unknown_phase(tmp_path):
    r = runner.invoke(_app(), ["profile", "kernels", str(_write_trace(tmp_path)),
                               "--phase", "nope"])
    assert r.exit_code == 2


def test_ops(tmp_path):
    r = runner.invoke(_app(), ["profile", "ops", str(_write_trace(tmp_path)), "--top", "3"])
    assert r.exit_code == 0
    assert "aten::conv" in r.stdout


def test_compare_self(tmp_path):
    t = str(_write_trace(tmp_path))
    r = runner.invoke(_app(), ["profile", "compare", t, t])
    assert r.exit_code == 0
    assert "img/s" in r.stdout


def test_missing_trace(tmp_path):
    r = runner.invoke(_app(), ["profile", "summary", str(tmp_path / "nope.json")])
    assert r.exit_code == 2


def test_summary_shows_host_overhead(tmp_path):
    r = runner.invoke(_app(), ["profile", "summary", str(_write_trace(tmp_path))])
    assert r.exit_code == 0
    assert "host overhead" in r.stdout and "launches/step" in r.stdout


def test_whatif(tmp_path):
    r = runner.invoke(_app(), ["profile", "what-if", str(_write_trace(tmp_path)),
                               "--remove-launches", "2"])
    assert r.exit_code == 0
    assert "what-if" in r.stdout and "img/s" in r.stdout


def test_whatif_requires_arg(tmp_path):
    r = runner.invoke(_app(), ["profile", "what-if", str(_write_trace(tmp_path))])
    assert r.exit_code == 2


def test_compare_significance_note(tmp_path):
    t = str(_write_trace(tmp_path))
    r = runner.invoke(_app(), ["profile", "compare", t, t])
    assert r.exit_code == 0
    assert "single run" in r.stdout
    assert "ms/image" in r.stdout


def test_run_forwards_profile_settings(tmp_path, monkeypatch):
    """The profile command reaches the ordinary public training path."""
    import libreyolo

    calls: list[dict] = []

    class _StubModel:
        def __init__(self, **kw):
            pass

        def train(self, **kw):
            calls.append(kw)

    monkeypatch.setattr(libreyolo, "LibreYOLO", _StubModel)

    r = runner.invoke(_app(), ["profile", "run", "coco128",
                               "--project", str(tmp_path)])
    assert r.exit_code == 3  # stub trains but produces no profile.json
    assert calls
    assert calls[0]["profile"] is True
    assert calls[0]["profile_then_stop"] is True
