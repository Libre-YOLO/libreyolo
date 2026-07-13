"""Standard-library HTTP server backing ``libreyolo monitor``.

Zero third-party dependencies. The server is a thin, read-only relay over a
*runs root* directory: it never touches any training process, so it can attach
to a live run, re-open a finished one, or watch a crashed one, and it keeps
working even if a trainer dies. Everything it serves is written to disk by
:class:`libreyolo.training.artifacts.TrainingStatusCallback` and the family
artifact writer (``status.json``, ``metrics.jsonl``, ``train.log``, images).

A single server watches many runs: each run is addressed by URL via a ``run``
id (its path relative to the root), so several browser tabs (one per run)
share one port, and ``/`` lists every discovered run for an overview.

Endpoints
---------
``GET /``                   the monitor page (index + per-run view, one HTML app)
``GET /api/runs``           discovered runs with live state, for the index
``GET /api/status?run=``    parsed ``status.json`` (``{"state": "missing"}`` if none)
``GET /api/metrics?run=``   metric history as ``{"columns": [...], "rows": [...]}``
``GET /api/log?run=``       tail of ``train.log`` (``?bytes=`` to size the window)
``GET /api/images?run=``    list of plot/sample images in the run dir
``GET /api/image?run=&path=`` a single image file (sandboxed to the run dir)
"""

from __future__ import annotations

import csv
import json
import logging
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from .train_monitor_page import INDEX_HTML

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_RUN_MARKERS = ("status.json", "metrics.jsonl", "results.csv")
_MAX_LOG_BYTES = 200_000


def _finite_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _normalize_status_view(status: dict) -> dict:
    """Add tested display fields across status schema v1 and v2."""
    view = dict(status)
    schema = _finite_number(view.get("schema_version")) or 1
    state = str(view.get("state") or "")
    total = _finite_number(view.get("total_epochs"))
    current = _finite_number(view.get("current_epoch"))
    completed = _finite_number(view.get("completed_epochs"))
    start = _finite_number(view.get("start_epoch")) or 1
    raw_progress = _finite_number(view.get("progress"))

    if schema >= 2:
        display_epoch = current + 1 if state == "failed" and current is not None else completed
        if display_epoch is None and current is not None:
            display_epoch = current + 1
        progress = raw_progress
        if progress is None and completed is not None and total and total > 0:
            progress = completed / total
        best = _finite_number(view.get("best_epoch"))
        display_best = best + 1 if best is not None else None
    else:
        if state == "completed" and completed is not None:
            # Schema v1 TrainEndEvent stored invocation-local completion on resume.
            display_epoch = max(start - 1, 0) + completed
        elif current is not None:
            # Schema v1 current_epoch came from an already one-based public event.
            display_epoch = current
        else:
            display_epoch = completed
        if total and total > 0:
            if state == "completed" and completed is not None:
                progress = (max(start - 1, 0) + completed) / total
            elif state == "running" and current is not None:
                progress = current / total
            elif state == "failed" and current is not None:
                progress = max(current - 1, start - 1, 0) / total
            elif completed is not None:
                progress = completed / total
            else:
                progress = raw_progress
        else:
            progress = raw_progress
        display_best = _finite_number(view.get("best_epoch"))

    if total is not None and display_epoch is not None:
        display_epoch = min(display_epoch, max(total, 0))
    view["display_epoch"] = max(0, int(display_epoch or 0))
    view["display_progress"] = min(max(float(progress or 0.0), 0.0), 1.0)
    view["display_best_epoch"] = (
        max(0, int(display_best)) if display_best is not None else None
    )
    return view


def is_run_dir(path: Path) -> bool:
    """True if ``path`` looks like a training run (carries a known marker)."""
    return path.is_dir() and any((path / m).exists() for m in _RUN_MARKERS)


def discover_runs(root: Path) -> list[Path]:
    """Return every training run at or beneath ``root``, newest first.

    A run is any directory carrying a status/metrics/results marker. ``root``
    itself counts if it is a run (i.e. the user pointed straight at one).
    """
    found: dict[Path, float] = {}
    if is_run_dir(root):
        found[root] = _dir_mtime(root)
    for marker in _RUN_MARKERS:
        for path in root.rglob(marker):
            run_dir = path.parent
            if run_dir not in found:
                found[run_dir] = _dir_mtime(run_dir)
    return [d for d, _ in sorted(found.items(), key=lambda kv: kv[1], reverse=True)]


def _dir_mtime(run_dir: Path) -> float:
    """Freshness of a run: newest of its marker files, else the dir itself."""
    mtimes = [
        (run_dir / m).stat().st_mtime
        for m in _RUN_MARKERS
        if (run_dir / m).exists()
    ]
    return max(mtimes) if mtimes else run_dir.stat().st_mtime


def find_latest_run(root: str | Path = "runs") -> Path | None:
    """Most recently active run under ``root`` (or ``None`` if there are none)."""
    root_path = Path(root)
    if not root_path.exists():
        return None
    runs = discover_runs(root_path)
    return runs[0] if runs else None


def run_id_for(root: Path, run_dir: Path) -> str:
    """URL id for a run: its path relative to ``root`` (``"."`` if it is root)."""
    try:
        rel = run_dir.resolve().relative_to(root.resolve())
    except ValueError:
        return "."
    return rel.as_posix() or "."


def _run_dir(root: Path, run_id: str | None) -> Path | None:
    """Resolve a ``run`` id to a directory inside ``root``, or ``None``.

    Refuses any id that escapes the root (``..`` / absolute paths) or does not
    point at an existing directory.
    """
    if not run_id:
        return None
    try:
        target = (root / unquote(run_id)).resolve()
        root_r = root.resolve()
        if target != root_r and root_r not in target.parents:
            return None
    except (ValueError, OSError):
        return None
    return target if target.is_dir() else None


def _read_status(run_dir: Path) -> dict:
    path = run_dir / "status.json"
    if not path.exists():
        return {"state": "missing", "save_dir": str(run_dir)}
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(status, dict):
            raise ValueError("status must be an object")
        return _normalize_status_view(status)
    except (OSError, ValueError):
        return {"state": "unreadable", "save_dir": str(run_dir)}


def _read_metrics(run_dir: Path) -> dict:
    # metrics.jsonl is the universal history written for every family; the
    # family-gated results.csv is the fallback for older/third-party runs.
    jsonl = run_dir / "metrics.jsonl"
    if jsonl.exists():
        return _read_metrics_jsonl(jsonl)
    path = run_dir / "results.csv"
    if not path.exists():
        return {"columns": [], "rows": []}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            rows = []
            for raw in reader:
                row: dict[str, object] = {}
                for key, value in raw.items():
                    row[key] = _coerce_number(value)
                rows.append(row)
        return {"columns": columns, "rows": rows}
    except OSError:
        return {"columns": [], "rows": []}


def _read_metrics_jsonl(path: Path) -> dict:
    """Parse an append-only metrics.jsonl into ``{columns, rows}``.

    Columns are the union of all row keys, preserving first-seen order so the
    epoch column stays leftmost. Bad lines (e.g. a torn final append) are
    skipped rather than failing the whole read.
    """
    columns: list[str] = []
    seen: set[str] = set()
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                for key in row:
                    if key not in seen:
                        seen.add(key)
                        columns.append(key)
                rows.append(row)
    except OSError:
        return {"columns": [], "rows": []}
    return {"columns": columns, "rows": rows}


def _coerce_number(value: object) -> object:
    if value is None or value == "":
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return value


def _read_log_tail(run_dir: Path, max_bytes: int = _MAX_LOG_BYTES) -> str:
    path = run_dir / "train.log"
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # drop the partial first line
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _list_images(run_dir: Path, run_id: str) -> list[dict]:
    images: list[dict] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            rel = path.relative_to(run_dir).as_posix()
            images.append(
                {
                    "path": rel,
                    "name": path.name,
                    "mtime": path.stat().st_mtime,
                    "url": f"/api/image?run={quote(run_id)}&path={quote(rel)}",
                }
            )
    images.sort(key=lambda item: item["mtime"], reverse=True)
    return images


def _resolve_in_run(run_dir: Path, rel: str) -> Path | None:
    """Resolve ``rel`` against ``run_dir``, refusing to escape the run dir."""
    try:
        target = (run_dir / unquote(rel)).resolve()
        base = run_dir.resolve()
        target.relative_to(base)
    except (ValueError, OSError):
        return None
    return target


def _run_summary(root: Path, run_dir: Path) -> dict:
    """Compact per-run record for the index page."""
    status = _read_status(run_dir)
    return {
        "schema_version": status.get("schema_version"),
        "id": run_id_for(root, run_dir),
        "name": run_id_for(root, run_dir),
        "state": status.get("state", "unknown"),
        "model": (status.get("model_family") or "")
        + (status.get("model_size") or ""),
        "task": status.get("task"),
        "progress": status.get("progress", 0.0),
        "display_progress": status.get("display_progress"),
        "current_epoch": status.get("current_epoch"),
        "completed_epochs": status.get("completed_epochs"),
        "display_epoch": status.get("display_epoch"),
        "total_epochs": status.get("total_epochs"),
        "best_metric": status.get("best_metric"),
        "best_metric_name": status.get("best_metric_name"),
        "updated_at": status.get("updated_at"),
        "mtime": _dir_mtime(run_dir),
    }


class _Handler(BaseHTTPRequestHandler):
    root: Path  # bound on the subclass created in serve()
    server_version = "LibreYOLO-Monitor"

    def log_message(self, *args):  # keep the console quiet
        pass

    def _send(self, code: int, body, ctype: str = "application/json") -> None:
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _resolve_run_param(self, qs: dict) -> Path | None:
        return _run_dir(self.root, qs.get("run", [""])[0])

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML, "text/html; charset=utf-8")
            elif path == "/api/runs":
                runs = [_run_summary(self.root, d) for d in discover_runs(self.root)]
                self._send(200, {"root": str(self.root), "runs": runs})
            elif path in ("/api/status", "/api/metrics", "/api/log", "/api/images"):
                self._handle_run_api(path, qs)
            elif path == "/api/image":
                self._serve_image(qs)
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:  # never crash the monitor on a bad request
            logger.exception("Monitor request failed: %s", path)
            self._send(500, {"error": str(exc)})

    def _handle_run_api(self, path: str, qs: dict) -> None:
        run_dir = self._resolve_run_param(qs)
        if run_dir is None:
            self._send(404, {"error": "unknown run"})
            return
        run_id = qs.get("run", [""])[0]
        if path == "/api/status":
            self._send(200, _read_status(run_dir))
        elif path == "/api/metrics":
            self._send(200, _read_metrics(run_dir))
        elif path == "/api/log":
            try:
                max_bytes = min(int(qs.get("bytes", ["0"])[0]), _MAX_LOG_BYTES)
            except ValueError:
                max_bytes = 0
            max_bytes = max_bytes or _MAX_LOG_BYTES
            self._send(200, {"log": _read_log_tail(run_dir, max_bytes)})
        elif path == "/api/images":
            self._send(200, {"images": _list_images(run_dir, run_id)})

    def _serve_image(self, qs: dict) -> None:
        run_dir = self._resolve_run_param(qs)
        rel = qs.get("path", [""])[0]
        if run_dir is None or not rel:
            self._send(404, {"error": "not found"})
            return
        target = _resolve_in_run(run_dir, rel)
        if target is None or not target.is_file():
            self._send(404, {"error": "not found"})
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), ctype)


def serve(
    root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8420,
    open_browser: bool = True,
    open_run: str | Path | None = None,
) -> tuple[ThreadingHTTPServer, str]:
    """Start the monitor server rooted at ``root``. Raises ``OSError`` if the
    port is taken (the CLI retries with the next port).

    ``open_run`` optionally deep-links the opened browser tab to one run.
    """
    resolved = Path(root)
    handler = type("_BoundHandler", (_Handler,), {"root": resolved})
    httpd = ThreadingHTTPServer((host, port), handler)
    shown_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{shown_host}:{port}"
    open_url = url
    if open_run is not None:
        open_url = url + "/?run=" + quote(run_id_for(resolved, Path(open_run)))
    if open_browser:
        try:
            webbrowser.open(open_url)
        except Exception:
            pass
    return httpd, url
