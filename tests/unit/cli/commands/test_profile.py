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
    ev.append({"ph": "X", "cat": "user_annotation", "name": "step/forward",
               "ts": 210, "dur": 130, "pid": 1, "tid": 1})
    ev.append({"ph": "X", "cat": "user_annotation", "name": "step/backward",
               "ts": 350, "dur": 140, "pid": 1, "tid": 1})
    for name, ts, dur in [
        ("ampere_sgemm_128x128_nt", 220, 10),
        ("_ZN5cudnn21bn_bw_1C11_kernel_new", 360, 20),
        ("ampere_h16816gemm_128x128", 380, 15),
    ]:
        ev.append({"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur,
                   "pid": 0, "tid": 7})
    ev.append({"ph": "X", "cat": "cpu_op", "name": "aten::conv2d",
               "ts": 215, "dur": 40, "pid": 1, "tid": 1})
    ev.append({"ph": "X", "cat": "cpu_op", "name": "aten::convolution_backward",
               "ts": 355, "dur": 50, "pid": 1, "tid": 1})
    trace = Path(directory) / "profile_trace.json"
    trace.write_text(json.dumps({"traceEvents": ev}))
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
