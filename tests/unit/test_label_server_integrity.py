"""LibreLabel's project/session transaction and HTTP save contracts."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

pytestmark = pytest.mark.unit


def test_label_client_rejects_partial_save_before_posting():
    from libreyolo.label.page import INDEX_HTML

    save_start = INDEX_HTML.index("async function save()")
    preflight = INDEX_HTML.index("if(invalidShapes)", save_start)
    post = INDEX_HTML.index("fetch(`/api/label/", save_start)

    assert save_start < preflight < post
    assert "if(!clipToImage(b)) invalidShapes++" in INDEX_HTML
    assert "if(!clipPoly(p)) invalidShapes++" in INDEX_HTML
    assert "boxes.map(pxToNorm).filter" not in INDEX_HTML


def _make_dataset(root: Path, *, image_name: str = "a.jpg") -> Path:
    image_dir = root / "images" / "train"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (20, 12), color=(12, 34, 56)).save(image_dir / image_name)
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nnc: 1\nnames:\n  0: thing\n",
        encoding="utf-8",
    )
    return yaml_path


@contextmanager
def _label_server(yaml_path: Path):
    from libreyolo.label import server

    with mock.patch.object(server._LabelState, "register_current", lambda *a, **k: None):
        httpd, url, _session = server.serve(
            str(yaml_path), port=0, open_browser=False, assist=False
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield url
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)


def _get_json(url: str, path: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url + path, timeout=5) as response:
        return response.status, json.load(response)


def _post(url: str, path: str, body: bytes) -> tuple[int, dict]:
    request = urllib.request.Request(
        url + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, json.load(exc)


def _box(cx: float = 0.5) -> dict:
    return {"type": "box", "cls": 0, "cx": cx, "cy": 0.5, "w": 0.2, "h": 0.2}


def test_label_post_requires_explicit_payload_epoch_and_revision(tmp_path):
    from libreyolo.label.dataset import DatasetSession

    yaml_path = _make_dataset(tmp_path)
    session = DatasetSession(str(yaml_path))
    session.write_label(0, [_box()])
    label_path = tmp_path / "labels" / "train" / "a.txt"
    original = label_path.read_bytes()

    with _label_server(yaml_path) as url:
        _, loaded = _get_json(url, "/api/label/0")
        assert loaded["epoch"] == 0
        rev = loaded["rev"]
        malformed = [
            (f"/api/label/0?epoch=0&rev={rev}", b""),
            (f"/api/label/0?epoch=0&rev={rev}", b"{}"),
            (f"/api/label/0?epoch=0&rev={rev}", b"[]"),
            (f"/api/label/0?epoch=0&rev={rev}", b'{"annotations":null}'),
            (
                f"/api/label/0?epoch=0&rev={rev}",
                b'{"annotations":[{"type":"box"}],"annotations":[]}',
            ),
            (f"/api/label/0?epoch=0&rev={rev}", b"{"),
            ("/api/label/0?epoch=0", b'{"annotations":[]}'),
            (f"/api/label/0?rev={rev}", b'{"annotations":[]}'),
            (f"/api/label/0?epoch=0&rev={rev}&rev={rev}", b'{"annotations":[]}'),
        ]
        for path, body in malformed:
            status, _response = _post(url, path, body)
            assert status == 400
            assert label_path.read_bytes() == original

        # Empty annotations are still a valid, deliberate background-label save;
        # they erase only when the payload, project epoch, and loaded revision are
        # all supplied explicitly.
        status, response = _post(
            url,
            f"/api/label/0?epoch=0&rev={rev}",
            b'{"annotations":[]}',
        )
        assert status == 200 and response["count"] == 0
        assert label_path.read_text(encoding="utf-8") == ""


def test_label_post_rejects_stale_revision_without_clobbering(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    with _label_server(yaml_path) as url:
        _, loaded = _get_json(url, "/api/label/0")
        rev = loaded["rev"]
        first = json.dumps({"annotations": [_box(0.3)]}).encode()
        second = json.dumps({"annotations": [_box(0.7)]}).encode()
        status, _ = _post(url, f"/api/label/0?epoch=0&rev={rev}", first)
        assert status == 200
        status, conflict = _post(url, f"/api/label/0?epoch=0&rev={rev}", second)
        assert status == 409
        assert "changed by someone else" in conflict["error"]
        _, saved = _get_json(url, "/api/label/0")
        assert saved["annotations"][0]["cx"] == pytest.approx(0.3)


def test_server_upload_transaction_prevents_duplicate_temp_race(tmp_path):
    from libreyolo.label.server import _LabelState

    state = _LabelState(None, assist=False)
    barrier = threading.Barrier(3)
    successes = []
    failures = []

    def upload(data: bytes):
        barrier.wait()
        try:
            state.save_upload(tmp_path, "same.jpg", data)
            successes.append(data)
        except Exception as exc:  # the losing duplicate must be a clean refusal
            failures.append(exc)

    threads = [
        threading.Thread(target=upload, args=(b"first",)),
        threading.Thread(target=upload, args=(b"second",)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert len(successes) == 1
    assert len(failures) == 1 and isinstance(failures[0], FileExistsError)
    assert (tmp_path / "images" / "train" / "same.jpg").read_bytes() == successes[0]


def test_export_and_project_switch_share_one_transaction(monkeypatch, tmp_path):
    from libreyolo.label import export
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    first = _make_dataset(tmp_path / "first")
    second = _make_dataset(tmp_path / "second")
    state = _LabelState(DatasetSession(str(first)), assist=False)
    entered = threading.Event()
    release = threading.Event()
    switched = threading.Event()
    monkeypatch.setattr(state, "register_current", lambda *a, **k: None)

    def blocked_export(session, **kwargs):
        assert session is state.session
        entered.set()
        assert release.wait(5)
        return {"in_place": False, "out": "unused"}

    monkeypatch.setattr(export, "export_dataset", blocked_export)
    export_thread = threading.Thread(
        target=lambda: state.export_project(expected_epoch=0, dst="unused")
    )

    def switch():
        state.open_project(str(second))
        switched.set()

    export_thread.start()
    assert entered.wait(5)
    switch_thread = threading.Thread(target=switch)
    switch_thread.start()
    assert not switched.wait(0.15), "project switch interleaved with an active export"
    release.set()
    export_thread.join(timeout=5)
    switch_thread.join(timeout=5)
    assert switched.is_set()
    assert Path(state.session.yaml_file) == second
    assert state.epoch == 1


def test_class_mutation_checks_epoch_inside_transaction(monkeypatch, tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState, _ProjectConflict

    first = _make_dataset(tmp_path / "first")
    second = _make_dataset(tmp_path / "second")
    state = _LabelState(DatasetSession(str(first)), assist=False)
    monkeypatch.setattr(state, "register_current", lambda *a, **k: None)
    state.open_project(str(second))
    with pytest.raises(_ProjectConflict):
        state.set_class_names(["renamed"], expected_epoch=0)
    assert state.session.names == ["thing"]


def test_class_rename_creates_new_generation_and_invalidates_old_session(
    monkeypatch, tmp_path
):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState, _ProjectConflict

    yaml_path = _make_dataset(tmp_path)
    original = DatasetSession(str(yaml_path))
    state = _LabelState(original, assist=False)
    state._data = str(yaml_path)
    monkeypatch.setattr(state, "register_current", lambda *a, **k: None)

    meta = state.set_class_names(["renamed"], expected_epoch=0)

    assert meta["epoch"] == 1
    assert state.session is not original
    assert state.session.names == ["renamed"]
    assert state.store_pending(0, [_box()], original) is False
    with pytest.raises(_ProjectConflict):
        state.write_label(0, [_box()], epoch=0, expected_rev=0)


def test_class_rename_schedules_registry_for_new_session(monkeypatch, tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    yaml_path = _make_dataset(tmp_path)
    state = _LabelState(DatasetSession(str(yaml_path)), assist=False)
    state._data = str(yaml_path)
    calls = []
    monkeypatch.setattr(
        state,
        "register_current",
        lambda data, **kwargs: calls.append((data, kwargs)),
    )

    state.set_class_names(["renamed"], expected_epoch=0)

    assert len(calls) == 1
    assert calls[0][1]["session"] is state.session
    assert calls[0][1]["epoch"] == state.epoch == 1


def test_stale_background_publishers_cannot_mutate_new_project(monkeypatch, tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState, _ProjectConflict

    first = DatasetSession(str(_make_dataset(tmp_path / "first")))
    second_path = _make_dataset(tmp_path / "second")
    state = _LabelState(first, assist=False)
    monkeypatch.setattr(state, "register_current", lambda *a, **k: None)
    state.open_project(str(second_path))
    current_suggestion = [_box(0.4)]
    state.engine.set_pending(0, current_suggestion)
    state.radar_findings = {0: [{"type": "miss"}]}
    state.embed_points = [{"id": 0, "x": 1.0, "y": 2.0}]

    with pytest.raises(_ProjectConflict):
        state.capture_session(0, "auto-labeling")
    assert state.clear_pending(first) is False
    assert state.store_radar_findings({9: []}, first) is False
    assert state.store_embed_points([{"id": 9}], first) is False
    assert state.engine.get_pending(0) == current_suggestion
    assert state.radar_findings == {0: [{"type": "miss"}]}
    assert state.embed_points == [{"id": 0, "x": 1.0, "y": 2.0}]


def test_delete_matches_current_project_by_root_and_detaches(monkeypatch, tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    yaml_path = _make_dataset(tmp_path)
    state = _LabelState(DatasetSession(str(yaml_path)), assist=False)
    state._data = str(yaml_path)
    monkeypatch.setattr("libreyolo.label.server.trash_project", lambda data: "trash/project")
    forgotten = []
    monkeypatch.setattr("libreyolo.label.server.projects.forget", forgotten.append)
    result = state.delete_project(str(tmp_path), expected_epoch=0)
    assert result == {"trash": "trash/project", "closed": True, "epoch": 1}
    assert state.session is None and state._data is None
    assert {Path(value) for value in forgotten} == {tmp_path, yaml_path}


def test_boost_start_requires_current_epoch(monkeypatch, tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState, _ProjectConflict

    first = _make_dataset(tmp_path / "first")
    second = _make_dataset(tmp_path / "second")
    state = _LabelState(DatasetSession(str(first)), assist=False)
    monkeypatch.setattr(state, "register_current", lambda *a, **k: None)
    state.open_project(str(second))
    called = []
    monkeypatch.setattr(state.boost, "start", lambda **kwargs: called.append(kwargs))

    with pytest.raises(_ProjectConflict):
        state.start_boost(expected_epoch=0)
    assert called == []


def test_project_switch_cannot_leave_old_boost_model_published(monkeypatch, tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    first = DatasetSession(str(_make_dataset(tmp_path / "first")))
    second_path = _make_dataset(tmp_path / "second")
    state = _LabelState(first, assist=False)
    monkeypatch.setattr(state, "register_current", lambda *a, **k: None)
    entered = threading.Event()
    release = threading.Event()

    def publish_like_finishing_worker():
        with state.engine._lock:
            entered.set()
            assert release.wait(5)
            if state.boost.session is first:
                state.engine._models["boosted"] = object()

    worker = threading.Thread(target=publish_like_finishing_worker)
    worker.start()
    assert entered.wait(5)
    switched = threading.Event()
    switcher = threading.Thread(
        target=lambda: (state.open_project(str(second_path)), switched.set())
    )
    switcher.start()
    assert not switched.wait(0.1)
    release.set()
    worker.join(timeout=5)
    switcher.join(timeout=5)

    assert switched.is_set()
    assert state.boost.session is state.session
    assert "boosted" not in state.engine._models


def test_old_boost_worker_cannot_publish_status_after_switch(monkeypatch, tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    first = DatasetSession(str(_make_dataset(tmp_path / "first")))
    second_path = _make_dataset(tmp_path / "second")
    state = _LabelState(first, assist=False)
    monkeypatch.setattr(state, "register_current", lambda *a, **k: None)
    generation = state.boost._generation

    state.open_project(str(second_path))

    assert state.boost._set_for_run(
        first, generation, state="done", boosted_agreement=1.0
    ) is False
    assert state.boost.status() == {"state": "idle"}


def test_project_root_key_preserves_yaml_suffixed_directory(tmp_path):
    from libreyolo.label.server import _project_root_key

    project = tmp_path / "project.yaml"
    project.mkdir()
    yaml_path = project / "data.yaml"
    yaml_path.write_text("names: []\n")

    assert _project_root_key(project) == _project_root_key(yaml_path)


def test_stale_registry_worker_cannot_resurrect_deleted_project(monkeypatch, tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    yaml_path = _make_dataset(tmp_path)
    session = DatasetSession(str(yaml_path))
    state = _LabelState(session, assist=False)
    entered = threading.Event()
    release = threading.Event()
    stats_done = threading.Event()
    finished = threading.Event()
    registered = []
    real_stats = session.stats

    def slow_stats():
        entered.set()
        assert release.wait(5)
        result = real_stats()
        stats_done.set()
        return result

    def register(*args, **kwargs):
        registered.append((args, kwargs))
        finished.set()

    monkeypatch.setattr(session, "stats", slow_stats)
    monkeypatch.setattr("libreyolo.label.server.projects.register", register)
    monkeypatch.setattr("libreyolo.label.server.projects.forget", lambda data: None)
    monkeypatch.setattr("libreyolo.label.server.trash_project", lambda data: "trash/project")
    state.register_current(str(yaml_path))
    assert entered.wait(5)
    state.delete_project(str(tmp_path), expected_epoch=0)
    release.set()
    assert stats_done.wait(5)
    assert not finished.wait(0.1)
    assert registered == []


def test_registry_worker_uses_latest_sidecar_name(monkeypatch, tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    yaml_path = _make_dataset(tmp_path)
    session = DatasetSession(str(yaml_path))
    state = _LabelState(session, assist=False)
    entered = threading.Event()
    release = threading.Event()
    registered = threading.Event()
    captured = []
    real_stats = session.stats

    def slow_stats():
        entered.set()
        assert release.wait(5)
        return real_stats()

    def register(*args, **kwargs):
        captured.append((args, kwargs))
        registered.set()

    monkeypatch.setattr(session, "stats", slow_stats)
    monkeypatch.setattr("libreyolo.label.server.projects.register", register)
    monkeypatch.setattr("libreyolo.label.server.projects.rename", lambda *a: None)
    state.register_current(str(yaml_path))
    assert entered.wait(5)
    state.update_project_meta({"name": "Latest name"}, expected_epoch=0)
    release.set()
    assert registered.wait(5)

    assert captured[0][1]["name"] == "Latest name"


def test_session_aliases_do_not_include_external_dataset_root(tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    external = tmp_path / "external"
    image_dir = external / "images" / "train"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (20, 12), color=(12, 34, 56)).save(image_dir / "a.jpg")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    yaml_path = config_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {external.as_posix()}\ntrain: images/train\nnames: [thing]\nnc: 1\n",
        encoding="utf-8",
    )
    session = DatasetSession(str(yaml_path))
    state = _LabelState(session, assist=False)
    state._data = str(yaml_path)

    with state._lock:
        aliases = state._session_aliases_locked(session, yaml_path)

    assert str(external) not in aliases
    assert str(config_dir) in aliases
