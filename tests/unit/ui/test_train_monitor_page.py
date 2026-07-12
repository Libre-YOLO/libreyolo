from __future__ import annotations

import json

import pytest

from libreyolo.ui import train_monitor
from libreyolo.ui.train_monitor_page import INDEX_HTML

pytestmark = pytest.mark.unit


def test_run_summary_exposes_absolute_completed_epoch_count(tmp_path):
    root = tmp_path / "runs"
    run = root / "train" / "exp"
    run.mkdir(parents=True)
    (run / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "state": "running",
                "current_epoch": 4,
                "completed_epochs": 5,
                "total_epochs": 10,
                "progress": 0.5,
            }
        )
    )

    summary = train_monitor._run_summary(root, run)

    assert summary["current_epoch"] == 4
    assert summary["completed_epochs"] == 5
    assert summary["progress"] == pytest.approx(0.5)
    assert summary["schema_version"] == 2
    assert summary["display_epoch"] == 5
    assert summary["display_progress"] == pytest.approx(0.5)


def test_status_view_repairs_legacy_running_off_by_one():
    status = train_monitor._normalize_status_view(
        {
            "schema_version": 1,
            "state": "running",
            "start_epoch": 1,
            "current_epoch": 1,
            "completed_epochs": 2,
            "total_epochs": 4,
            "progress": 0.5,
        }
    )

    assert status["display_epoch"] == 1
    assert status["display_progress"] == pytest.approx(0.25)


def test_status_view_repairs_legacy_resume_completion():
    status = train_monitor._normalize_status_view(
        {
            "schema_version": 1,
            "state": "completed",
            "start_epoch": 3,
            "completed_epochs": 2,
            "total_epochs": 4,
            "progress": 0.5,
        }
    )

    assert status["display_epoch"] == 4
    assert status["display_progress"] == pytest.approx(1.0)


def test_status_view_shows_attempted_epoch_on_v2_failure():
    status = train_monitor._normalize_status_view(
        {
            "schema_version": 2,
            "state": "failed",
            "current_epoch": 2,
            "completed_epochs": 2,
            "total_epochs": 4,
            "progress": 0.5,
        }
    )

    assert status["display_epoch"] == 3
    assert status["display_progress"] == pytest.approx(0.5)


def test_monitor_page_uses_zero_based_status_and_one_based_metric_contracts():
    assert "s.display_epoch != null" in INDEX_HTML
    assert "s.display_progress != null" in INDEX_HTML
    assert "return Math.max(0, modern ? epoch + 1 : epoch);" in INDEX_HTML
    assert "const bestEpoch = bestEpochNumber(s);" in INDEX_HTML
    assert "const xs = rows.map(r => r.epoch != null ? r.epoch : 0);" in INDEX_HTML
    assert "r.epoch != null ? r.epoch + 1 : 0" not in INDEX_HTML


def test_monitor_page_bounds_progress_and_can_derive_it_from_completed_epochs():
    assert "completed / total" in INDEX_HTML
    assert "Math.max(0, Math.min(1, value))" in INDEX_HTML
