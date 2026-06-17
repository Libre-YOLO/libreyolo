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
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from . import projects
from .assist import AssistEngine
from .boost import BoostEngine
from .dataset import (
    DatasetSession,
    count_images,
    folder_yaml,
    scaffold_data_yaml,
    update_class_names,
)
from .embed import EmbedEngine
from .page import INDEX_HTML
from .radar import scan_dataset
from .sam import SamEngine

logger = logging.getLogger(__name__)

# Absolute filesystem paths (Windows ``C:\...`` or POSIX ``/...``) we must not leak
# to LAN clients in error strings on a shared server.
_ABS_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/)[^\s\"']+")


def _redact_paths(msg: str) -> str:
    return _ABS_PATH_RE.sub("<path>", str(msg))


def _lan_ip() -> str:
    """Best-effort primary LAN IP (for the teammate share URL). Sends no packets."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))   # just selects the routable interface
        return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        s.close()


class _LabelState:
    """Per-server state: the open dataset, a write lock, and the assist engine."""

    def __init__(self, session, device: str = "auto", assist: bool = True):
        self.session = session                # may be None -> project home screen
        self._data = None                     # path of the open project (for the registry)
        self.host = "127.0.0.1"               # bind address (for the teammate share URL)
        self.port = 0
        self.epoch = 0                        # bumps on every project switch (stale-save guard)
        self._lock = threading.Lock()
        self.engine = AssistEngine(device=device, enabled=assist)
        self.sam = SamEngine(enabled=assist, device=device)
        self.embed = EmbedEngine(device=device, enabled=assist)
        self.boost = BoostEngine(session, model_name=self.engine.default_model,
                                 enabled=assist, engine=self.engine)
        self._thumbs: dict = {}
        self._thumb_lock = threading.Lock()
        self.radar_findings: dict = {}        # idx -> [finding, ...] from the last scan
        self._radar_lock = threading.Lock()
        self.embed_points = None              # cached 2-D embedding scatter

    def register_current(self, data) -> None:
        """Record the open dataset in the registry. The ``labeled`` count needs a
        full label scan, so it runs on a background thread -- never blocking the
        open/switch action (which would be seconds of dead UI on big datasets)."""
        if self.session is None:
            return
        self._data = str(data)
        session = self.session

        def _work():
            try:
                projects.register(data, root=session.root or None,
                                  count=session.meta().get("count"),
                                  labeled=session.stats().get("labeled"))
            except Exception:  # noqa: BLE001 - registry is a convenience, never fatal
                logger.exception("project registry update failed")

        threading.Thread(target=_work, name="librelabel-registry", daemon=True).start()

    def open_project(self, data) -> dict:
        """Switch the live session to ``data`` (a data.yaml/dir), resetting all
        per-project state the engines hold. Raises on a bad dataset path."""
        session = DatasetSession(data)        # validate before mutating anything
        with self._lock:
            self.session = session
            self.epoch += 1                   # invalidate in-flight saves from the old project
            self.engine.clear_pending()
            self.radar_findings = {}
            self.embed_points = None
            self.embed._cache.clear()
            self.sam._cur = None
            with self.engine._lock:                     # mutually exclusive with the boost write
                self.engine._models.pop("boosted", None)   # boosted model is per-project
            self.boost.on_project_switch(session)       # keeps a running boost intact
            with self._thumb_lock:
                self._thumbs.clear()
        self.register_current(data)
        return session.meta()

    def set_class_names(self, names) -> dict:
        """Rename and/or append dataset classes -- never delete or reorder, so
        existing label class ids keep their meaning -- rewriting the YAML and
        patching the live session in place. No epoch bump: ids are preserved, so
        in-flight saves from this project stay valid."""
        with self._lock:
            if self.session is None:
                raise RuntimeError("no project open")
            cleaned = [str(n).strip() for n in names]
            if any(not n for n in cleaned):
                raise ValueError("class names can't be empty")
            if len({n.lower() for n in cleaned}) != len(cleaned):
                raise ValueError("class names must be unique")
            if len(cleaned) < self.session.nc:
                raise ValueError("classes can be renamed or added here, not removed")
            update_class_names(self.session.yaml_file, cleaned)
            self.session.names = cleaned
            self.session.nc = len(cleaned)
            return self.session.meta()

    def read_label_with_rev(self, idx: int) -> tuple:
        """Read annotations + revision under the save lock, so a save can't land
        between the two and hand a stale client old annotations with a NEW rev
        (which its next save would then use to pass the conflict check)."""
        with self._lock:
            anns, editable = self.session.read_label(idx)
            return anns, editable, self.session.label_rev(idx)

    def write_label(self, idx: int, boxes, epoch=None, expected_rev=None) -> tuple:
        with self._lock:  # serialize concurrent saves to the same tree (check+write atomic)
            if epoch is not None and epoch != self.epoch:
                raise RuntimeError("project changed; reload before saving")
            count = self.session.write_label(idx, boxes, expected_rev=expected_rev)
            return count, self.session.label_rev(idx)

    def store_pending(self, idx: int, sugg, sess) -> bool:
        """Store prelabel suggestions iff still on ``sess`` -- atomically with the
        project switch. open_project() clears pending + rebinds the session under
        the same ``_lock``, so this check-then-write can't interleave with it and
        leave stale suggestions under a new project's numeric id."""
        with self._lock:
            if self.session is not sess:
                return False
            self.engine.set_pending(idx, sugg)
            return True

    def resolve_duplicates(self, ids, purge: bool = False, epoch=None) -> dict:
        with self._lock:  # serialize against label writes on the same tree
            # Same stale-guard as write_label: a Fix request carrying a since-switched
            # project's epoch must not quarantine/purge same-id images in the new one.
            if epoch is not None and epoch != self.epoch:
                raise RuntimeError("project changed; reload before fixing duplicates")
            return self.session.resolve_duplicates(ids, purge=purge)

    def thumb(self, idx: int, path: Path) -> bytes:
        # Key by the absolute image path, not the numeric id: after a project switch
        # (open_project clears _thumbs) an old in-flight request could otherwise store
        # bytes under a reused id like 0 and show the previous dataset's thumbnail.
        key = str(path)
        with self._thumb_lock:
            cached = self._thumbs.get(key)
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
            self._thumbs[key] = data
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
            sessionless = path in ("/", "/index.html", "/api/dataset", "/api/projects",
                                   "/api/server", "/api/assist/status", "/api/boost/status")
            if self.state.session is None and path.startswith("/api/") and not sessionless:
                self._send(409, {"error": "no project open"})
                return
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML, "text/html; charset=utf-8")
            elif path == "/api/projects":
                # The registry holds host-local dataset paths; don't enumerate them
                # for LAN teammates who can only label the already-open project.
                if self._local_admin():
                    self._send(200, {"projects": projects.list_projects(),
                                     "open": self.state._data})
                else:
                    self._send(200, {"projects": [], "open": None})
            elif path == "/api/server":
                ip = _lan_ip()
                host = self.state.host
                shareable = host in ("0.0.0.0", "::", "") or host == ip
                self._send(200, {
                    "host": host, "port": self.state.port,
                    "local_url": "http://127.0.0.1:%d" % self.state.port,
                    "lan_url": ("http://%s:%d" % (ip, self.state.port)) if shareable else None,
                    "shareable": shareable,
                })
            elif path == "/api/dataset":
                if self.state.session is None:
                    self._send(200, {"open": False})
                else:
                    meta = self.state.session.meta()
                    meta["open"] = True
                    meta["epoch"] = self.state.epoch
                    if not self._local_admin():
                        # root/yaml are host-local paths; `reason` can also embed an
                        # absolute path ("Could not derive a label path for <abs>").
                        # LAN teammates only need names/count/writable to label.
                        meta.pop("root", None)
                        meta.pop("yaml", None)
                        meta.pop("reason", None)
                    self._send(200, meta)
            elif path == "/api/images":
                self._send(200, {"images": self.state.session.list_images()})
            elif path == "/api/stats":
                self._send(200, self.state.session.stats())
            elif path == "/api/insights":
                self._send(200, self.state.session.insights())
            elif path == "/api/quality":
                try:
                    imgsz = int((parse_qs(parsed.query).get("imgsz") or ["640"])[0])
                except (TypeError, ValueError):
                    imgsz = 640
                self._send(200, self.state.session.quality(imgsz))
            elif path.startswith("/api/image/"):
                self._serve_image(int(path.rsplit("/", 1)[-1]))
            elif path.startswith("/api/thumb/"):
                self._serve_thumb(int(path.rsplit("/", 1)[-1]))
            elif path.startswith("/api/label/"):
                idx = int(path.rsplit("/", 1)[-1])
                annotations, editable, rev = self.state.read_label_with_rev(idx)
                # rev as a STRING: nanosecond mtimes (~1e18) exceed JS Number.MAX_SAFE_INTEGER,
                # so a numeric token would be rounded by the browser and every save would 409.
                self._send(200, {"annotations": annotations, "editable": editable, "rev": str(rev)})
            elif path == "/api/assist/status":
                st = self.state.engine.status()
                st["sam"] = self.state.sam.available()
                st["embed"] = self.state.embed.available()
                st["boost"] = bool(st.get("available"))
                st["boosted"] = self.state.engine.has_model("boosted")
                self._send(200, st)
            elif path.startswith("/api/assist/pending/"):
                idx = int(path.rsplit("/", 1)[-1])
                self._send(200, {"suggestions": self.state.engine.get_pending(idx)})
            elif path.startswith("/api/assist/radar/"):
                idx = int(path.rsplit("/", 1)[-1])
                with self.state._radar_lock:
                    findings = self.state.radar_findings.get(idx, [])
                self._send(200, {"findings": findings})
            elif path == "/api/boost/status":
                self._send(200, self.state.boost.status())
            else:
                self._send(404, {"error": "not found"})
        except (IndexError, ValueError) as exc:
            self._send(404, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("label GET failed: %s", path)
            # Exception messages can embed absolute host paths (e.g. OSError on a
            # label file); the host still sees detail via the log above.
            self._send(500, {"error": str(exc) if self._local_admin() else "internal error"})

    def _serve_image(self, idx: int) -> None:
        p: Path = self.state.session.image_path(idx)
        if not p.exists():
            self._send(404, {"error": "image missing"})
            return
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        try:  # TOCTOU: a concurrent dup-fix may remove the file after exists()
            data = p.read_bytes()
        except OSError:
            self._send(404, {"error": "image missing"})
            return
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
            # PIL errors usually embed the absolute filename; redact for LAN clients.
            self._send(415, {"error": str(exc) if self._local_admin() else "unreadable image"})
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
            if not self._same_origin():
                self._send(403, {"error": "cross-origin request blocked"})
                return
            if not self._host_allowed():
                self._send(403, {"error": "host not allowed"})
                return
            # Host-admin only: switching the project, pruning duplicates, and the
            # heavy full-dataset compute streams all rebind/clobber server-global
            # state (session, files, the shared pending/findings/embed maps, a host
            # CPU training job) for every teammate -- gate them to the loopback host.
            if path in ("/api/projects/open", "/api/projects/create", "/api/projects/inspect",
                        "/api/projects/forget", "/api/pick-folder", "/api/example", "/api/classes", "/api/insights/fix",
                        "/api/boost", "/api/assist/autolabel", "/api/assist/radar",
                        "/api/embeddings") and not self._local_admin():
                self._send(403, {"error": "This action (create / switch project, edit classes, "
                                          "prune duplicates, full-dataset auto-label, Radar, "
                                          "embeddings, Boost) is only allowed from the host "
                                          "machine on a shared server."})
                return
            if self.state.session is None and path not in (
                    "/api/projects/open", "/api/projects/create",
                    "/api/projects/inspect", "/api/projects/forget", "/api/pick-folder", "/api/example"):
                self._send(409, {"error": "no project open"})
                return
            if (path.startswith("/api/assist/") or path in ("/api/embeddings", "/api/boost")) \
                    and not self.state.engine.enabled:
                self._send(403, {"error": "AI assist is disabled (started with --no-assist)."})
                return
            if path == "/api/projects/open":
                payload = self._read_json()
                data = payload.get("data") if isinstance(payload, dict) else None
                if not data:
                    self._send(400, {"error": "data path required"})
                    return
                try:
                    meta = self.state.open_project(str(data))
                    meta["open"] = True
                    meta["epoch"] = self.state.epoch
                    self._send(200, meta)
                except Exception as exc:  # noqa: BLE001 - bad dataset path/config
                    logger.exception("open project failed")
                    if isinstance(exc, FileNotFoundError):
                        msg = "No dataset YAML found at that path."
                    elif isinstance(exc, UnicodeDecodeError):
                        msg = "That file is not a dataset YAML (couldn't read it as text)."
                    else:
                        msg = str(exc).splitlines()[0][:140] or "Could not open that dataset."
                    self._send(400, {"error": msg})
            elif path == "/api/projects/inspect":
                # Report what's at a host path so the home screen can route: open an
                # existing yaml, or offer to create a project from a bare image folder.
                payload = self._read_json()
                folder = payload.get("folder") if isinstance(payload, dict) else None
                if not folder:
                    self._send(400, {"error": "folder path required"})
                    return
                yp = folder_yaml(str(folder))
                self._send(200, {"folder": str(folder), "is_dir": Path(folder).is_dir(),
                                 "images": count_images(str(folder)),
                                 "yaml": yp, "has_yaml": yp is not None})
            elif path == "/api/projects/create":
                # The keystone of the cold start: a folder of images becomes a project.
                # scaffold_data_yaml writes a minimal data.yaml beside the images (the
                # exact layout the trainer reads), then we open it like any other.
                payload = self._read_json()
                folder = payload.get("folder") if isinstance(payload, dict) else None
                classes = payload.get("classes") if isinstance(payload, dict) else None
                task = payload.get("task") if isinstance(payload, dict) else None
                if task not in ("detect", "segment", "obb", "classify"):
                    task = None   # ignore anything unexpected -> default detection
                if not folder:
                    self._send(400, {"error": "folder path required"})
                    return
                try:
                    existing = folder_yaml(str(folder))   # don't clobber a real dataset
                    target = existing or scaffold_data_yaml(str(folder), classes or [], task=task)
                    meta = self.state.open_project(target)
                    meta["open"] = True
                    meta["epoch"] = self.state.epoch
                    meta["created"] = existing is None
                    self._send(200, meta)
                except FileNotFoundError as exc:
                    self._send(400, {"error": str(exc).splitlines()[0][:140]
                                     or "No images found in that folder."})
                except Exception as exc:  # noqa: BLE001 - bad folder / unwritable path
                    logger.exception("create project failed")
                    self._send(400, {"error": str(exc).splitlines()[0][:140]
                                     or "Could not create that project."})
            elif path == "/api/classes":
                payload = self._read_json()
                names = payload.get("names") if isinstance(payload, dict) else None
                ep = payload.get("epoch") if isinstance(payload, dict) else None
                if not isinstance(names, list):
                    self._send(400, {"error": "names list required"})
                    return
                if ep is not None and int(ep) != self.state.epoch:
                    self._send(409, {"error": "project changed — reopen it before editing classes"})
                    return
                meta = self.state.set_class_names(names)
                meta["open"] = True
                meta["epoch"] = self.state.epoch
                self._send(200, meta)
            elif path == "/api/pick-folder":
                # Pop a NATIVE OS "choose folder" dialog on the host and return the
                # absolute path. Works because the server runs on the user's machine;
                # gated to host-admin (a dialog on the host is meaningless to LAN peers).
                # tkinter is stdlib and imported lazily so plain installs never load it.
                self._read_json()
                try:
                    folder = _native_pick_folder()
                except Exception as exc:  # noqa: BLE001 - headless / no display / Tk missing
                    logger.info("native folder dialog unavailable: %s", exc)
                    self._send(200, {"folder": None, "unavailable": True})
                else:
                    self._send(200, {"folder": folder or None})
            elif path == "/api/example":
                # Open the bundled example project (128 CC0 demo images), fetching it
                # from Hugging Face on first use. Host-admin only (it writes to a local
                # cache + opens a project); allowed before any project is open.
                self._read_json()
                try:
                    folder = _example_dir()
                    existing = folder_yaml(folder)
                    target = existing or scaffold_data_yaml(
                        folder, ["person", "dog", "cat", "car", "bicycle"])
                    meta = self.state.open_project(target)
                    meta["open"] = True
                    meta["epoch"] = self.state.epoch
                    meta["created"] = existing is None
                    self._send(200, meta)
                except Exception as exc:  # noqa: BLE001 - no network / hf missing / bad cache
                    logger.exception("open example failed")
                    self._send(400, {"error": str(exc).splitlines()[0][:160]
                                     or "Could not open the example project."})
            elif path == "/api/projects/forget":
                payload = self._read_json()
                data = payload.get("data") if isinstance(payload, dict) else None
                if data:
                    projects.forget(str(data))
                self._send(200, {"ok": True})
            elif path.startswith("/api/label/"):
                idx = int(path.rsplit("/", 1)[-1])
                payload = self._read_json()
                anns = payload.get("annotations", []) if isinstance(payload, dict) else []
                qs = parse_qs(parsed.query)
                ep = (qs.get("epoch") or [None])[0]
                rev = (qs.get("rev") or [None])[0]
                count, new_rev = self.state.write_label(
                    idx, anns, epoch=int(ep) if ep is not None else None,
                    expected_rev=int(rev) if rev is not None else None)
                self._send(200, {"ok": True, "count": count, "rev": str(new_rev)})
            elif path.startswith("/api/assist/prelabel/"):
                self._handle_prelabel(int(path.rsplit("/", 1)[-1]), parse_qs(parsed.query))
            elif path.startswith("/api/assist/segment/"):
                self._handle_segment(int(path.rsplit("/", 1)[-1]))
            elif path == "/api/assist/autolabel":
                self._handle_autolabel_stream(parse_qs(parsed.query))
            elif path == "/api/assist/radar":
                self._handle_radar_stream(parse_qs(parsed.query))
            elif path == "/api/embeddings":
                self._handle_embeddings_stream()
            elif path == "/api/insights/fix":
                payload = self._read_json()
                ids = payload.get("ids", []) if isinstance(payload, dict) else []
                purge = bool(payload.get("purge")) if isinstance(payload, dict) else False
                ep = payload.get("epoch") if isinstance(payload, dict) else None
                res = self.state.resolve_duplicates(
                    [int(i) for i in ids], purge=purge,
                    epoch=int(ep) if ep is not None else None)
                self._send(200, res)
            elif path == "/api/boost":
                payload = self._read_json()
                kw = payload if isinstance(payload, dict) else {}
                self._send(200, self.state.boost.start(
                    epochs=int(kw.get("epochs", 2)), imgsz=int(kw.get("imgsz", 512)),
                    batch=int(kw.get("batch", 4))))
            else:
                self._send(404, {"error": "not found"})
        except (IndexError, ValueError) as exc:
            self._send(400, {"error": str(exc)})
        except RuntimeError as exc:  # read-only / non-box file -> 409
            # 409 reasons are mostly path-free (read-only/conflict), but one can carry
            # an absolute path; scrub it for non-admin LAN clients.
            msg = str(exc) if self._local_admin() else _redact_paths(str(exc))
            self._send(409, {"error": msg})
        except Exception as exc:  # noqa: BLE001
            logger.exception("label POST failed: %s", path)
            self._send(500, {"error": str(exc) if self._local_admin() else "internal error"})

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        data = self.rfile.read(length) if length else b""
        return json.loads(data.decode("utf-8")) if data else {}

    @staticmethod
    def _is_loopback(addr: str) -> bool:
        a = (addr or "").strip().lower()
        return a in ("127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1") or a.startswith("127.")

    def _local_admin(self) -> bool:
        """Whether this client may perform host-level admin (switch project, prune
        duplicates, list the project registry) -- actions that rebind the global
        session, move/delete files, or expose host-local paths.

        Only a *wildcard* bind (0.0.0.0/:: -- what ``--share`` uses) is both
        LAN-reachable AND still serves loopback, so there the host connects via
        127.0.0.1 while teammates use the LAN IP: require a loopback CLIENT to gate
        teammates out. A loopback bind is local-only, and a concrete NIC bind
        (``host=192.168.x.y``) forces the host itself to connect via that address --
        indistinguishable from a teammate by IP -- so admin is allowed there (it's an
        explicit, advanced exposure, not the default share path).
        """
        host = (self.state.host or "").strip().lower()
        if host in ("0.0.0.0", "::", ""):
            return self._is_loopback(self.client_address[0] if self.client_address else "")
        return True

    def _same_origin(self) -> bool:
        """Block cross-origin state-changing requests (CSRF). A browser always sets
        ``Origin`` on POST; a foreign site's Origin won't match our Host, so we
        reject it before any write. Non-browser tools (no Origin) are allowed."""
        origin = self.headers.get("Origin")
        if not origin:
            return True
        try:
            return urlparse(origin).netloc == (self.headers.get("Host") or "")
        except Exception:  # noqa: BLE001
            return False

    def _host_allowed(self) -> bool:
        """Defend against DNS rebinding: ``_same_origin`` passes when a malicious page
        rebinds its own hostname to 127.0.0.1 (both Origin and Host read
        ``attacker.example``). On a loopback bind the only legitimate Host is a
        loopback name, so reject anything else. Wildcard / NIC binds are explicit
        exposures and keep their LAN Host."""
        bind = (self.state.host or "").strip().lower()
        if bind not in ("127.0.0.1", "::1", "localhost", ""):
            return True   # wildcard / concrete-NIC bind: a LAN Host is legitimate
        host_hdr = (self.headers.get("Host") or "").strip().lower()
        if not host_hdr:
            return True   # non-browser client without a Host header
        name = host_hdr.rsplit(":", 1)[0].strip("[]") if not host_hdr.endswith("]") else host_hdr.strip("[]")
        return self._is_loopback(name)

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
        # Snapshot the session up front: the predict below is slow, and if the user
        # switches projects mid-flight we must not store these suggestions into the
        # shared pending map (keyed by numeric id) for the now-current project.
        sess = self.state.session
        # Don't suggest on polygon/OBB-locked images (box-only mode can't accept).
        _, editable = sess.read_label(idx)
        if not editable:
            self._send(200, {"editable": False, "suggestions": []})
            return
        engine = (qs.get("engine") or ["yolo"])[0]
        try:
            if engine == "locate":
                classes = (qs.get("classes") or [""])[0].split(",")
                sugg = self.state.engine.predict_locate(
                    sess.image_path(idx), sess.names, classes
                )
            else:
                sugg = self.state.engine.predict_image(
                    sess.image_path(idx), sess.names, model, conf
                )
        except Exception as exc:  # noqa: BLE001 - model load/inference problem
            logger.exception("prelabel failed")
            self._send(503, {"error": str(exc)})
            return
        if not self.state.store_pending(idx, sugg, sess):   # atomic with project switch
            self._send(409, {"error": "project changed; reopen and retry"})
            return
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
        engine = (qs.get("engine") or ["yolo"])[0]
        classes = [c for c in (qs.get("classes") or [""])[0].split(",") if c.strip()]
        ep = (qs.get("epoch") or [None])[0]
        if ep is not None and int(ep) != self.state.epoch:
            # a stale tab (project switched in another tab) must not populate the
            # current project's pending suggestions with the old dataset's run
            self._send(409, {"error": "project changed — reload before auto-labeling"})
            return

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

        sess = self.state.session
        try:
            self.state.engine.clear_pending()  # fresh run replaces stale suggestions
            summary = self.state.engine.autolabel_dataset(
                sess, model_name=model, conf=conf, progress=emit,
                engine=engine, classes=classes,
                current=lambda: self.state.session is sess,  # stop if the project switches
                # publish each image's suggestions atomically with the project switch
                store=lambda i, s: self.state.store_pending(i, s, sess),
            )
            emit(summary)
        except Exception as exc:  # noqa: BLE001
            logger.exception("auto-label failed")
            emit({"type": "error", "error": str(exc)})

    def _ndjson_begin(self):
        """Open a streaming NDJSON response and return a thread-safe ``emit``."""
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

        return emit

    def _handle_radar_stream(self, qs: dict) -> None:
        """Audit accepted labels with the model; stream per-image progress then a
        sorted ``deck`` of disagreements. Per-image findings are parked in state
        for the UI to overlay (``GET /api/assist/radar/<id>``)."""
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        model, conf = self._model_conf(qs)
        if not self.state.engine.status().get("available"):
            self._send(503, {"error": "No model available for Radar "
                                      "(assist disabled or no weights)."})
            return
        sess = self.state.session
        emit = self._ndjson_begin()
        try:
            result = scan_dataset(self.state.engine.predict_image, sess,
                                  model_name=model, conf=conf, progress=emit)
            with self.state._radar_lock:
                if self.state.session is sess:   # drop results if the project changed mid-scan
                    self.state.radar_findings = result.pop("findings", {})
            emit(result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("radar scan failed")
            emit({"type": "error", "error": str(exc)})

    def _handle_embeddings_stream(self) -> None:
        """Embed every image and stream the 2-D PCA scatter (``{id,x,y}``)."""
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        if not self.state.embed.available():
            self._send(503, {"error": "Embeddings unavailable (assist disabled)."})
            return
        sess = self.state.session
        emit = self._ndjson_begin()
        try:
            points = self.state.embed.scatter(sess, progress=emit)
            if self.state.session is sess:       # drop if the project changed mid-embed
                self.state.embed_points = points
            emit({"type": "done", "points": points})
        except Exception as exc:  # noqa: BLE001
            logger.exception("embeddings failed")
            emit({"type": "error", "error": str(exc)})


def _example_dir() -> str:
    """Return a writable local folder of the LibreLabel example images, fetching it
    on first use. Resolution order: an existing local cache (so labels drawn in the
    example persist) -> a folder named by LIBRELABEL_EXAMPLE_DIR (dev / pre-upload)
    -> a Hugging Face dataset download. Raises if none can be obtained."""
    import shutil

    name = "tutorial-128"
    cache = Path.home() / ".librelabel" / "examples" / name
    if cache.is_dir() and any(cache.glob("*.jpg")):
        return str(cache)                       # already fetched -> reuse (keeps any labels)
    cache.mkdir(parents=True, exist_ok=True)

    src = os.environ.get("LIBRELABEL_EXAMPLE_DIR")
    if src and Path(src).is_dir():
        for f in Path(src).iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                shutil.copy2(f, cache / f.name)
        if any(cache.glob("*.jpg")) or any(cache.glob("*.png")):
            return str(cache)

    repo = os.environ.get("LIBRELABEL_EXAMPLE_REPO", "LibreYOLO/librelabel-tutorial-128")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download the example "
                           "dataset; install it or set LIBRELABEL_EXAMPLE_DIR.") from exc
    logger.info("Downloading example dataset %s -> %s ...", repo, cache)
    snapshot_download(repo, repo_type="dataset", local_dir=str(cache),
                      allow_patterns=["*.jpg", "*.jpeg", "*.png"])
    if not (any(cache.glob("*.jpg")) or any(cache.glob("*.png"))):
        raise FileNotFoundError(f"Example dataset {repo} has no images.")
    return str(cache)


def _native_pick_folder() -> str:
    """Open a native OS 'choose folder' dialog on the host and return the chosen
    absolute path ('' if cancelled). Local-first convenience for the home screen;
    raises if there is no GUI/display (caller falls back to the text input)."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:  # noqa: BLE001 - cosmetic only
        pass
    try:
        path = filedialog.askdirectory(title="Choose an image folder for LibreLabel")
    finally:
        root.destroy()
    return path or ""


def serve(
    data: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    device: str = "auto",
    assist: bool = True,
) -> tuple:
    """Bind the annotator server.

    With ``data`` set, the dataset is loaded eagerly (a bad path raises here) and
    registered as a project. With ``data=None`` the server starts on the project
    home screen, where any dataset can be opened from the browser. The port is
    bound eagerly (an in-use port raises ``OSError`` to retry). Returns
    ``(httpd, url, session)`` (``session`` is ``None`` in home mode).
    """
    session = DatasetSession(data) if data else None
    state = _LabelState(session, device=device, assist=assist)
    state.host = host
    handler = type("BoundLabelHandler", (_Handler,), {"state": state})
    httpd = ThreadingHTTPServer((host, port), handler)
    # Publish the *actual* bound port: with port=0 the OS assigns one, so the
    # requested value would yield unusable ":0" URLs and poison /api/server.
    bound_port = httpd.server_address[1]
    state.port = bound_port
    if session is not None:
        state.register_current(data)
    url = "http://%s:%d" % (host, bound_port)
    if open_browser:
        import webbrowser

        browse = "http://127.0.0.1:%d" % bound_port if host in ("0.0.0.0", "::", "") else url
        threading.Timer(0.7, lambda: webbrowser.open(browse)).start()
    return httpd, url, session
