"""Standard-library HTTP server backing ``libreyolo label`` (v1: boxes only).

Zero third-party dependencies -- the same shape as :mod:`libreyolo.ui.server`:
a ``ThreadingHTTPServer`` with a runtime-subclassed handler that binds the
per-server :class:`~libreyolo.label.dataset.DatasetSession`, hand-rolled routing
on ``urlparse(path)``, and JSON via a tiny ``_send`` helper. The page itself is
the embedded string in :mod:`libreyolo.label.page`.

Endpoints
---------
``GET  /``                     the annotator page (HTML)
``GET  /api/dataset``          ``{root, names, nc, count, writable, reason}``
``GET  /api/images``           ``{images:[{id,name,split,status}]}`` (paged)
``GET  /api/image/<id>``       raw image bytes
``GET  /api/label/<id>``       ``{boxes:[{cls,cx,cy,w,h}], editable}``
``POST /api/label/<id>``       body ``{boxes:[...]}`` -> writes the ``.txt``
``GET  /api/assist/status``    ``{available, models, default}``
``POST /api/assist/prelabel/<id>``  query ``model``/``conf`` -> ``{suggestions:[...]}``
``POST /api/assist/autolabel`` query ``model``/``conf`` -> NDJSON progress stream
"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .assist import AssistEngine
from .dataset import DatasetSession
from .page import INDEX_HTML
from .sam import SamEngine

logger = logging.getLogger(__name__)


class _LabelState:
    """Per-server state: the open dataset, a write lock, and the assist engine."""

    def __init__(self, session: DatasetSession, device: str = "auto", assist: bool = True):
        self.session = session
        self._lock = threading.Lock()
        self.engine = AssistEngine(device=device, enabled=assist)
        self.sam = SamEngine(enabled=assist, device=device)
        self._thumbs: dict = {}
        self._thumb_lock = threading.Lock()

    def write_label(self, idx: int, boxes) -> int:
        with self._lock:  # serialize concurrent saves to the same tree
            return self.session.write_label(idx, boxes)

    def thumb(self, idx: int, path: Path) -> bytes:
        with self._thumb_lock:
            cached = self._thumbs.get(idx)
        if cached is not None:
            return cached
        import io

        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((220, 220))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=82)
        data = buf.getvalue()
        with self._thumb_lock:
            self._thumbs[idx] = data
        return data


class _Handler(BaseHTTPRequestHandler):
    state: _LabelState  # bound on the subclass created in serve()
    server_version = "LibreYOLO-Label"

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
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- GET ---------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML, "text/html; charset=utf-8")
            elif path == "/api/dataset":
                self._send(200, self.state.session.meta())
            elif path == "/api/images":
                self._send(200, {"images": self.state.session.list_images()})
            elif path == "/api/stats":
                self._send(200, self.state.session.stats())
            elif path == "/api/insights":
                self._send(200, self.state.session.insights())
            elif path.startswith("/api/image/"):
                self._serve_image(int(path.rsplit("/", 1)[-1]))
            elif path.startswith("/api/thumb/"):
                self._serve_thumb(int(path.rsplit("/", 1)[-1]))
            elif path.startswith("/api/label/"):
                idx = int(path.rsplit("/", 1)[-1])
                annotations, editable = self.state.session.read_label(idx)
                self._send(200, {"annotations": annotations, "editable": editable})
            elif path == "/api/assist/status":
                st = self.state.engine.status()
                st["sam"] = self.state.sam.available()
                self._send(200, st)
            elif path.startswith("/api/assist/pending/"):
                idx = int(path.rsplit("/", 1)[-1])
                self._send(200, {"suggestions": self.state.engine.get_pending(idx)})
            else:
                self._send(404, {"error": "not found"})
        except (IndexError, ValueError) as exc:
            self._send(404, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("label GET failed: %s", path)
            self._send(500, {"error": str(exc)})

    def _serve_image(self, idx: int) -> None:
        p: Path = self.state.session.image_path(idx)
        if not p.exists():
            self._send(404, {"error": "image missing"})
            return
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_thumb(self, idx: int) -> None:
        p: Path = self.state.session.image_path(idx)
        if not p.exists():
            self._send(404, {"error": "image missing"})
            return
        try:
            data = self.state.thumb(idx, p)
        except Exception as exc:  # noqa: BLE001 - unreadable/odd image; let client fall back
            self._send(415, {"error": str(exc)})
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- POST --------------------------------------------------------------
    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path.startswith("/api/label/"):
                idx = int(path.rsplit("/", 1)[-1])
                payload = self._read_json()
                anns = payload.get("annotations", []) if isinstance(payload, dict) else []
                count = self.state.write_label(idx, anns)
                self._send(200, {"ok": True, "count": count})
            elif path.startswith("/api/assist/prelabel/"):
                self._handle_prelabel(int(path.rsplit("/", 1)[-1]), parse_qs(parsed.query))
            elif path.startswith("/api/assist/segment/"):
                self._handle_segment(int(path.rsplit("/", 1)[-1]))
            elif path == "/api/assist/autolabel":
                self._handle_autolabel_stream(parse_qs(parsed.query))
            else:
                self._send(404, {"error": "not found"})
        except (IndexError, ValueError) as exc:
            self._send(400, {"error": str(exc)})
        except RuntimeError as exc:  # read-only / non-box file -> 409
            self._send(409, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("label POST failed: %s", path)
            self._send(500, {"error": str(exc)})

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        data = self.rfile.read(length) if length else b""
        return json.loads(data.decode("utf-8")) if data else {}

    @staticmethod
    def _model_conf(qs: dict) -> tuple:
        model = (qs.get("model") or [None])[0] or None
        try:
            conf = float((qs.get("conf") or ["0.25"])[0])
        except (TypeError, ValueError):
            conf = 0.25
        return model, conf

    def _handle_prelabel(self, idx: int, qs: dict) -> None:
        self._read_json()  # drain any body
        model, conf = self._model_conf(qs)
        # Don't suggest on polygon/OBB-locked images (box-only mode can't accept).
        _, editable = self.state.session.read_label(idx)
        if not editable:
            self._send(200, {"editable": False, "suggestions": []})
            return
        try:
            sugg = self.state.engine.predict_image(
                self.state.session.image_path(idx), self.state.session.names, model, conf
            )
        except Exception as exc:  # noqa: BLE001 - model load/inference problem
            logger.exception("prelabel failed")
            self._send(503, {"error": str(exc)})
            return
        self.state.engine.set_pending(idx, sugg)
        self._send(200, {"editable": True, "suggestions": sugg})

    def _handle_segment(self, idx: int) -> None:
        payload = self._read_json() or {}
        if not isinstance(payload, dict):
            payload = {}
        box = payload.get("box")
        points = payload.get("points")
        try:
            if box and len(box) == 4:
                kw = {"box": [float(v) for v in box]}
            elif points:
                kw = {
                    "points": [[float(p[0]), float(p[1])] for p in points],
                    "labels": [int(v) for v in (payload.get("labels") or [1] * len(points))],
                }
            else:
                kw = {"point": (float(payload["x"]), float(payload["y"]))}
        except (TypeError, ValueError, KeyError, IndexError):
            self._send(400, {"error": "point {x,y}, points [[x,y],...], or box [x1,y1,x2,y2] required"})
            return
        try:
            poly = self.state.sam.segment(self.state.session.image_path(idx), **kw)
        except Exception as exc:  # noqa: BLE001 - SAM load/inference problem
            logger.exception("segment failed")
            self._send(503, {"error": str(exc)})
            return
        self._send(200, {"polygon": poly})

    def _handle_autolabel_stream(self, qs: dict) -> None:
        """Stream NDJSON: one ``{"type":"progress"}`` per image, then a final
        ``{"type":"done"}`` (or ``{"type":"error"}``). No Content-Length so each
        flushed line reaches the browser's fetch stream immediately."""
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        model, conf = self._model_conf(qs)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        write_lock = threading.Lock()

        def emit(obj: dict) -> None:
            line = (json.dumps(obj) + "\n").encode("utf-8")
            with write_lock:
                try:
                    self.wfile.write(line)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        try:
            self.state.engine.clear_pending()  # fresh run replaces stale suggestions
            summary = self.state.engine.autolabel_dataset(
                self.state.session, model_name=model, conf=conf, progress=emit
            )
            emit(summary)
        except Exception as exc:  # noqa: BLE001
            logger.exception("auto-label failed")
            emit({"type": "error", "error": str(exc)})


def serve(
    data: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    device: str = "auto",
    assist: bool = True,
) -> tuple[ThreadingHTTPServer, str, DatasetSession]:
    """Open ``data`` and bind the annotator server.

    The dataset is loaded eagerly (so a bad path raises here) and the port is
    bound eagerly (so an in-use port raises ``OSError`` for the caller to retry).
    Returns ``(httpd, url, session)``; the caller runs ``serve_forever()``.
    """
    session = DatasetSession(data)
    state = _LabelState(session, device=device, assist=assist)
    handler = type("BoundLabelHandler", (_Handler,), {"state": state})
    httpd = ThreadingHTTPServer((host, port), handler)
    url = "http://%s:%d" % (host, port)
    if open_browser:
        import webbrowser

        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    return httpd, url, session
