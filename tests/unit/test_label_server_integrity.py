"""LibreLabel's project/session transaction and HTTP save contracts."""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit
from unittest import mock

import pytest
from PIL import Image

pytestmark = pytest.mark.unit

_REJECTED_POST_BODY = b"x" * (128 * 1024)


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
def _label_server(yaml_path: Path, *, host: str = "127.0.0.1"):
    from libreyolo.label import server

    with mock.patch.object(server._LabelState, "register_current", lambda *a, **k: None):
        httpd, url, _session = server.serve(
            str(yaml_path), host=host, port=0, open_browser=False, assist=False
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            if host in ("0.0.0.0", "::", ""):
                url = f"http://127.0.0.1:{httpd.server_address[1]}"
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


def _request_json(
    url: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict | None = None,
) -> tuple[int, dict]:
    request = urllib.request.Request(
        url + path,
        data=body,
        headers=headers or {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, json.load(exc)


def _raw_status(url: str, request: str) -> int:
    """Send an exact HTTP request so duplicate authority headers stay intact."""
    target = urlsplit(url)
    with socket.create_connection((target.hostname, target.port), timeout=5) as peer:
        peer.sendall(request.encode("ascii"))
        response = peer.makefile("rb")
        status_line = response.readline().decode("ascii", errors="replace")
    return int(status_line.split()[1])


def _box(cx: float = 0.5) -> dict:
    return {"type": "box", "cls": 0, "cx": cx, "cy": 0.5, "w": 0.2, "h": 0.2}


def test_loopback_host_header_matrix_blocks_dns_rebinding_on_get(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    with _label_server(yaml_path) as url:
        port = urlsplit(url).port
        cases = (
            (f"127.0.0.1:{port}", 200),
            (f"LOCALHOST:{port}", 200),
            (f"127.0.0.42:{port}", 200),
            (f"[::1]:{port}", 200),
            (f"localhost.:{port}", 200),
            (f"attacker.example:{port}", 403),
            (f"192.168.1.20:{port}", 403),
            ("127.0.0.1:80", 403),
            (f"[::]:{port}", 403),
        )
        for host, expected in cases:
            status, _response = _request_json(
                url, "/api/dataset", headers={"Host": host}
            )
            assert status == expected, host


def test_share_host_header_matrix_allows_numeric_lan_hosts_only(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    with _label_server(yaml_path, host="0.0.0.0") as url:
        port = urlsplit(url).port
        cases = (
            (f"127.0.0.1:{port}", 200),
            (f"localhost:{port}", 200),
            (f"192.168.1.20:{port}", 200),
            (f"10.12.0.8:{port}", 200),
            (f"[fd00::1234]:{port}", 200),
            (f"attacker.example:{port}", 403),
            (f"0.0.0.0:{port}", 403),
            (f"224.0.0.1:{port}", 403),
            (f"[::]:{port}", 403),
        )
        for host, expected in cases:
            status, _response = _request_json(
                url, "/api/dataset", headers={"Host": host}
            )
            assert status == expected, host


def test_post_origin_matrix_normalizes_authority_and_rejects_opaque_origins(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    with _label_server(yaml_path) as url:
        port = urlsplit(url).port
        cases = (
            (f"LOCALHOST:{port}", None, 404),
            (f"LOCALHOST:{port}", f"HTTP://localhost:{port}", 404),
            (f"localhost.:{port}", f"http://LOCALHOST:{port}", 404),
            (f"localhost:{port}", "null", 403),
            (f"localhost:{port}", "file://localhost", 403),
            (f"localhost:{port}", f"https://localhost:{port}", 403),
            (f"localhost:{port}", "http://localhost", 403),
            (f"localhost:{port}", f"http://localhost:{port}/path", 403),
            (f"localhost:{port}", f"http://localhost:{port}?", 403),
            (f"localhost:{port}", f"http://localhost:{port}#", 403),
            (f"localhost:{port}", f"http://attacker.example:{port}", 403),
            (f"attacker.example:{port}", f"http://attacker.example:{port}", 403),
        )
        for host, origin, expected in cases:
            headers = {"Host": host, "Content-Type": "application/json"}
            if origin is not None:
                headers["Origin"] = origin
            status, _response = _request_json(
                url,
                "/not-found",
                method="POST",
                body=_REJECTED_POST_BODY,
                headers=headers,
            )
            assert status == expected, (host, origin)


def test_assist_disabled_rejection_consumes_nonempty_post_body(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    body = b'{"epoch":0,"padding":"' + _REJECTED_POST_BODY + b'"}'
    with _label_server(yaml_path) as url:
        for _ in range(8):
            status, response = _post(url, "/api/boost", body)
            assert status == 403
            assert "AI assist is disabled" in response["error"]


def test_assist_disabled_rejection_consumes_maximum_json_sized_body(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    body = b"x" * (8 * 1024 * 1024)
    with _label_server(yaml_path) as url:
        for _ in range(3):
            status, response = _post(url, "/api/boost", body)
            assert status == 403
            assert "AI assist is disabled" in response["error"]


def test_json_body_limit_discards_oversize_payload_before_413(monkeypatch, tmp_path):
    from libreyolo.label import server

    monkeypatch.setattr(server, "_MAX_JSON_BODY_BYTES", 1024)
    yaml_path = _make_dataset(tmp_path)
    with _label_server(yaml_path) as url:
        for _ in range(8):
            status, response = _post(url, "/api/label/0?epoch=0&rev=0", _REJECTED_POST_BODY)
            assert status == 413
            assert response == {"error": "request body too large"}


@pytest.mark.parametrize(
    "path",
    (
        "/api/boost",
        "/api/label/0?epoch=0&rev=0",
    ),
)
def test_global_body_limit_discards_oversize_payload_before_413(
    monkeypatch, tmp_path, path
):
    from libreyolo.label import server

    monkeypatch.setattr(server, "_MAX_REQUEST_BODY_BYTES", 1024)
    yaml_path = _make_dataset(tmp_path)
    with _label_server(yaml_path) as url:
        for _ in range(8):
            status, response = _post(url, path, _REJECTED_POST_BODY)
            assert status == 413
            assert response == {"error": "request body too large"}


@pytest.mark.parametrize(
    ("framing", "expected"),
    (
        ("Content-Length: invalid\r\n", 400),
        ("Content-Length: 0\r\nContent-Length: 0\r\n", 400),
        ("Transfer-Encoding: chunked\r\n", 400),
        (f"Content-Length: {64 * 1024 * 1024 + 1}\r\n", 413),
        ("Content-Length: 1\r\n", 400),
    ),
)
def test_rejected_post_body_framing_is_bounded(tmp_path, framing, expected):
    yaml_path = _make_dataset(tmp_path)
    with _label_server(yaml_path) as url:
        port = urlsplit(url).port
        assert _raw_status(
            url,
            "POST /not-found HTTP/1.1\r\n"
            f"Host: localhost:{port}\r\nOrigin: http://localhost:{port}\r\n"
            f"{framing}Connection: close\r\n\r\n",
        ) == expected


def test_duplicate_authorities_and_absolute_targets_are_unambiguous(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    with _label_server(yaml_path) as url:
        port = urlsplit(url).port
        valid = f"localhost:{port}"
        attacker = f"attacker.example:{port}"

        assert _raw_status(
            url,
            "GET /api/dataset HTTP/1.1\r\n"
            f"Host: {valid}\r\nHost: {valid}\r\nConnection: close\r\n\r\n",
        ) == 403
        assert _raw_status(
            url,
            "POST /not-found HTTP/1.1\r\n"
            f"Host: {valid}\r\nOrigin: http://{valid}\r\n"
            f"Origin: http://{valid}\r\nContent-Length: 0\r\n"
            "Connection: close\r\n\r\n",
        ) == 403
        assert _raw_status(
            url,
            f"GET http://{attacker}/api/dataset HTTP/1.1\r\n"
            f"Host: {valid}\r\nConnection: close\r\n\r\n",
        ) == 403
        assert _raw_status(
            url,
            f"GET http://{valid}/api/dataset HTTP/1.1\r\n"
            f"Host: {valid}\r\nConnection: close\r\n\r\n",
        ) == 200


def test_authority_parser_applies_http_effective_ports_and_ipv6_rules():
    from libreyolo.label.server import _parse_authority

    assert _parse_authority("LOCALHOST.", default_port=80) == ("localhost", 80)
    assert _parse_authority("[0:0::1]:8000", default_port=80) == ("::1", 8000)
    for authority in (
        "::1",
        "localhost:",
        "localhost:65536",
        "localhost:80,attacker.example",
        "user@localhost:80",
        "[::1]:80:90",
    ):
        with pytest.raises(ValueError):
            _parse_authority(authority, default_port=80)


def test_serve_formats_ipv6_and_empty_wildcard_urls():
    from libreyolo.label.server import _lan_url_host, _local_access_host, serve

    assert _local_access_host("0.0.0.0") == "127.0.0.1"
    assert _local_access_host("::") == "::1"
    assert _local_access_host("fd00::1234") == "fd00::1234"
    assert _lan_url_host("0.0.0.0", "192.168.1.5") == "192.168.1.5"
    assert _lan_url_host("::", "fd00::1234") == "fd00::1234"
    assert _lan_url_host("::", "fe80::1234") is None
    assert _lan_url_host("0.0.0.0", "127.0.0.1") is None
    assert _lan_url_host("192.168.1.5", "127.0.0.1") == "192.168.1.5"
    assert _lan_url_host("label-host", "127.0.0.1") == "label-host"
    assert _lan_url_host("localhost", "192.168.1.5") is None

    httpd, url, _session = serve(
        None, host="", port=0, open_browser=False, assist=False
    )
    try:
        assert url == f"http://0.0.0.0:{httpd.server_address[1]}"
    finally:
        httpd.server_close()

    try:
        httpd, url, _session = serve(
            None, host="::1", port=0, open_browser=False, assist=False
        )
    except OSError as exc:
        pytest.skip(f"IPv6 loopback unavailable: {exc}")
    try:
        assert url == f"http://[::1]:{httpd.server_address[1]}"
        assert httpd.address_family == socket.AF_INET6
    finally:
        httpd.server_close()


def test_server_info_advertises_only_a_usable_lan_url(monkeypatch, tmp_path):
    from libreyolo.label import server

    yaml_path = _make_dataset(tmp_path)
    monkeypatch.setattr(server, "_lan_ip", lambda family: "192.168.1.5")
    with _label_server(yaml_path, host="0.0.0.0") as url:
        status, info = _get_json(url, "/api/server")
        port = urlsplit(url).port

    assert status == 200
    assert info == {
        "host": "0.0.0.0",
        "port": port,
        "local_url": f"http://127.0.0.1:{port}",
        "lan_url": f"http://192.168.1.5:{port}",
        "shareable": True,
    }


@pytest.mark.parametrize(
    ("bind_host", "peer", "local", "expected"),
    (
        ("0.0.0.0", "127.0.0.1", "127.0.0.1", True),
        ("0.0.0.0", "192.168.1.5", "192.168.1.5", False),
        ("::", "::ffff:127.0.0.1", "::1", True),
        ("192.168.1.5", "192.168.1.5", "192.168.1.5", True),
        ("192.168.1.5", "192.168.1.44", "192.168.1.5", False),
        ("label-host", "192.168.1.5", "192.168.1.5", True),
        ("label-host", "192.168.1.44", "192.168.1.5", False),
        ("fd00::5", "fd00::5", "fd00::5", True),
        ("fd00::5", "fd00::44", "fd00::5", False),
    ),
)
def test_local_admin_uses_the_tcp_peer_boundary(bind_host, peer, local, expected):
    from libreyolo.label.server import _Handler

    class _Connection:
        @staticmethod
        def getsockname():
            return (local, 8000)

    handler = object.__new__(_Handler)
    handler.state = mock.Mock(host=bind_host)
    handler.client_address = (peer, 50000)
    handler.connection = _Connection()

    assert handler._local_admin() is expected


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


def test_delete_trash_failure_keeps_live_session_and_registry(monkeypatch, tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    yaml_path = _make_dataset(tmp_path)
    session = DatasetSession(str(yaml_path))
    state = _LabelState(session, assist=False)
    state._data = str(yaml_path)
    forgotten = []

    def fail_trash(data):
        raise RuntimeError("unexpected directory quarantined")

    monkeypatch.setattr("libreyolo.label.server.trash_project", fail_trash)
    monkeypatch.setattr("libreyolo.label.server.projects.forget", forgotten.append)

    with pytest.raises(RuntimeError, match="quarantined"):
        state.delete_project(str(tmp_path), expected_epoch=0)

    assert state.session is session
    assert state._data == str(yaml_path)
    assert state.epoch == 0
    assert forgotten == []


def test_delete_rejects_project_subtree_and_keeps_live_session(monkeypatch, tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    yaml_path = _make_dataset(tmp_path)
    session = DatasetSession(str(yaml_path))
    state = _LabelState(session, assist=False)
    state._data = str(yaml_path)
    trashed = []
    monkeypatch.setattr(
        "libreyolo.label.server.trash_project",
        lambda data: trashed.append(data) or "trash/project",
    )
    monkeypatch.setattr("libreyolo.label.server.projects.list_projects", lambda: [])

    with pytest.raises(ValueError, match="exact registered project root"):
        state.delete_project(str(tmp_path / "images"), expected_epoch=0)

    assert trashed == []
    assert state.session is session
    assert state._data == str(yaml_path)
    assert state.epoch == 0


def test_delete_rejects_missing_yaml_alias_inside_current_project(
    monkeypatch, tmp_path
):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    yaml_path = _make_dataset(tmp_path)
    session = DatasetSession(str(yaml_path))
    state = _LabelState(session, assist=False)
    state._data = str(yaml_path)
    trashed = []
    monkeypatch.setattr(
        "libreyolo.label.server.trash_project",
        lambda data: trashed.append(data) or "trash/project",
    )

    with pytest.raises(ValueError, match="exact registered project root"):
        state.delete_project(str(tmp_path / "missing.yaml"), expected_epoch=0)

    assert trashed == []
    assert state.session is session
    assert state.epoch == 0


def test_delete_allows_exact_registered_project_root(monkeypatch, tmp_path):
    from libreyolo.label.server import _LabelState

    yaml_path = _make_dataset(tmp_path)
    state = _LabelState(None, assist=False)
    trashed = []
    forgotten = []
    monkeypatch.setattr(
        "libreyolo.label.server.projects.list_projects",
        lambda: [{"data": str(yaml_path)}],
    )
    monkeypatch.setattr(
        "libreyolo.label.server.trash_project",
        lambda data: trashed.append(data) or "trash/project",
    )
    monkeypatch.setattr("libreyolo.label.server.projects.forget", forgotten.append)

    result = state.delete_project(str(tmp_path))

    assert result == {"trash": "trash/project", "closed": False, "epoch": 0}
    assert trashed == [str(tmp_path.resolve())]
    assert {Path(value) for value in forgotten} == {tmp_path, yaml_path}


def test_delete_rejects_stale_registry_directory_without_dataset(
    monkeypatch, tmp_path
):
    from libreyolo.label.server import _LabelState

    root = tmp_path / "stale-project"
    root.mkdir()
    (root / "unrelated.txt").write_text("keep", encoding="utf-8")
    state = _LabelState(None, assist=False)
    trashed = []
    monkeypatch.setattr(
        "libreyolo.label.server.projects.list_projects",
        lambda: [{"data": str(root)}],
    )
    monkeypatch.setattr(
        "libreyolo.label.server.trash_project",
        lambda data: trashed.append(data) or "trash/project",
    )

    with pytest.raises(ValueError, match="exact registered project root"):
        state.delete_project(str(root))

    assert trashed == []
    assert (root / "unrelated.txt").read_text(encoding="utf-8") == "keep"


def test_delete_detaches_live_session_even_if_registry_cleanup_fails(
    monkeypatch, tmp_path
):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    yaml_path = _make_dataset(tmp_path)
    state = _LabelState(DatasetSession(str(yaml_path)), assist=False)
    state._data = str(yaml_path)
    monkeypatch.setattr(
        "libreyolo.label.server.trash_project", lambda data: "trash/project"
    )
    monkeypatch.setattr(
        "libreyolo.label.server.projects.forget",
        lambda data: (_ for _ in ()).throw(OSError("registry unavailable")),
    )

    result = state.delete_project(str(tmp_path), expected_epoch=0)

    assert result == {"trash": "trash/project", "closed": True, "epoch": 1}
    assert state.session is None and state._data is None


def test_project_meta_write_failure_rolls_back_memory_and_registry(monkeypatch, tmp_path):
    from libreyolo.label.dataset import DatasetSession
    from libreyolo.label.server import _LabelState

    yaml_path = _make_dataset(tmp_path)
    sidecar = tmp_path / "librelabel.json"
    sidecar.write_text('{"name": "Original"}', encoding="utf-8")
    session = DatasetSession(str(yaml_path))
    state = _LabelState(session, assist=False)
    state._data = str(yaml_path)
    before_memory = dict(session._sidecar)
    before_disk = sidecar.read_bytes()
    renamed = []
    monkeypatch.setattr(
        "libreyolo.label.dataset._write_json_atomic",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("read only")),
    )
    monkeypatch.setattr(
        "libreyolo.label.server.projects.rename",
        lambda *args: renamed.append(args),
    )

    with pytest.raises(PermissionError, match="read only"):
        state.update_project_meta({"name": "Changed"}, expected_epoch=0)

    assert session._sidecar == before_memory
    assert sidecar.read_bytes() == before_disk
    assert renamed == []


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
