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
``GET  /api/label/<id>``       ``{annotations:[...], editable, rev, epoch}``
``POST /api/label/<id>``       query ``epoch``/``rev`` + body ``{annotations:[...]}``
``GET  /api/assist/status``    ``{available, models, default}``
``POST /api/assist/prelabel/<id>``  query ``model``/``conf`` -> ``{suggestions:[...]}``
``POST /api/assist/autolabel`` query ``model``/``conf`` -> NDJSON progress stream
"""

from __future__ import annotations

import ipaddress
import json
import logging
import mimetypes
import os
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse, urlsplit

from . import projects
from .assist import AssistEngine
from .boost import BoostEngine
from .dataset import (
    DatasetSession,
    count_images,
    create_linked_project,
    create_uploaded_project,
    folder_yaml,
    save_uploaded_image,
    scaffold_data_yaml,
    set_sidecar_name,
    trash_project,
    update_class_names,
    update_sidecar,
)
from .embed import EmbedEngine
from .page import INDEX_HTML
from .radar import scan_dataset
from .sam import SamEngine

logger = logging.getLogger(__name__)

# Absolute filesystem paths (Windows ``C:\...`` or POSIX ``/...``) we must not leak
# to LAN clients in error strings on a shared server.
_ABS_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/)[^\s\"']+")


class _ProjectConflict(RuntimeError):
    """A request was bound to a project generation that is no longer current."""


def _nonnegative_int(value, name: str) -> int:
    """Parse an explicit integer token without silently truncating floats/bools."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(value, int):
        out = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        out = int(value.strip())
    else:
        raise ValueError(f"{name} must be a non-negative integer")
    if out < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return out


def _project_root_entry_path(data, *, require_existing: bool = False) -> Path:
    """Return an absolute project root without following its final directory link."""
    p = Path(str(data)).expanduser()
    is_directory = p.is_dir()
    is_yaml = p.is_file() and p.suffix.lower() in (".yaml", ".yml")
    if require_existing and not (is_directory or is_yaml):
        raise FileNotFoundError(f"Not an existing project root or dataset YAML: {p}")
    # A directory may legitimately end in .yaml/.yml.  Only strip the filename
    # suffix when the supplied path is not an existing directory.
    if not is_directory and p.suffix.lower() in (".yaml", ".yml"):
        p = p.parent
    return Path(os.path.abspath(os.path.normpath(str(p))))


def _project_root_path(data, *, require_existing: bool = False) -> Path:
    """Return the canonical project root for either a dataset dir or YAML path."""
    p = _project_root_entry_path(data, require_existing=require_existing)
    try:
        p = p.resolve(strict=False)
    except (OSError, RuntimeError):
        p = Path(os.path.abspath(str(p)))
    return p


def _project_root_key(data, *, require_existing: bool = False) -> str:
    """Canonical project-root identity for either a dataset dir or YAML path."""
    return os.path.normcase(
        str(_project_root_path(data, require_existing=require_existing))
    )


def _unique_json_object(pairs):
    """Reject duplicate JSON keys instead of silently keeping the last value."""
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _redact_paths(msg: str) -> str:
    return _ABS_PATH_RE.sub("<path>", str(msg))


def _normalize_host_name(value: str) -> str:
    """Normalize one DNS name or IP literal without resolving DNS."""
    name = str(value).strip().lower()
    if name.endswith("."):
        name = name[:-1]
    if not name or len(name) > 253 or "%" in name:
        raise ValueError("invalid host")
    try:
        return ipaddress.ip_address(name).compressed.lower()
    except ValueError:
        labels = name.split(".")
        if any(
            not label
            or len(label) > 63
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
            for label in labels
        ):
            raise ValueError("invalid host") from None
        return name


def _parse_authority(authority: str, *, default_port: int) -> tuple[str, int]:
    """Parse and normalize an HTTP authority, rejecting ambiguous spellings."""
    raw = str(authority)
    if not raw or raw != raw.strip() or any(char.isspace() for char in raw):
        raise ValueError("invalid authority")
    if any(char in raw for char in "/?#@"):
        raise ValueError("invalid authority")
    try:
        parsed = urlsplit("//" + raw)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("invalid authority")
        host = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        raise ValueError("invalid authority") from None
    if not host:
        raise ValueError("invalid authority")
    if ":" in host and not raw.startswith("["):
        raise ValueError("IPv6 host literals must be bracketed")
    if raw.endswith(":"):
        raise ValueError("invalid authority")
    return _normalize_host_name(host), default_port if port is None else port


def _ip_is_loopback(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped and mapped.is_loopback)


def _url_host(host: str) -> str:
    """Render one host literal for an HTTP URL."""
    value = str(host).strip().strip("[]")
    try:
        return f"[{value}]" if ipaddress.ip_address(value).version == 6 else value
    except ValueError:
        return value


def _local_access_host(bind_host: str) -> str:
    """Return the address a browser on the server host can actually reach."""
    host = str(bind_host).strip()
    if host in ("", "0.0.0.0"):
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def _lan_url_host(bind_host: str, candidate: str) -> Optional[str]:
    """Return a usable teammate host, or ``None`` when only local access exists."""
    raw_bind = str(bind_host).strip().strip("[]")
    try:
        bind_name = _normalize_host_name(raw_bind) if raw_bind else ""
    except ValueError:
        return None
    try:
        bind_address = ipaddress.ip_address(bind_name) if bind_name else None
    except ValueError:
        bind_address = None
    wildcard = not bind_name or bool(bind_address and bind_address.is_unspecified)
    if not wildcard:
        if bind_name == "localhost" or bool(
            bind_address and (_ip_is_loopback(bind_address) or bind_address.is_link_local)
        ):
            return None
        return bind_name
    try:
        address = ipaddress.ip_address(str(candidate).strip().strip("[]"))
    except ValueError:
        return None
    if (
        _ip_is_loopback(address)
        or address.is_unspecified
        or address.is_multicast
        or address.is_link_local
    ):
        return None
    return address.compressed.lower()


def _lan_ip(family: int = socket.AF_INET) -> str:
    """Best-effort primary LAN IP (for the teammate share URL). Sends no packets."""
    probe = (
        ("2001:4860:4860::8888", 80, 0, 0)
        if family == socket.AF_INET6
        else ("8.8.8.8", 80)
    )
    s = socket.socket(family, socket.SOCK_DGRAM)
    try:
        s.connect(probe)   # just selects the routable interface
        return s.getsockname()[0]
    except OSError:
        try:
            candidates = socket.getaddrinfo(
                socket.gethostname(), None, family, socket.SOCK_DGRAM
            )
            for candidate in candidates:
                address = candidate[4][0]
                try:
                    parsed = ipaddress.ip_address(address)
                except ValueError:
                    continue
                if not parsed.is_unspecified:
                    return address
        except OSError:
            pass
        return "::1" if family == socket.AF_INET6 else "127.0.0.1"
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
        # One project transaction lock covers session rebinding and every mutation
        # whose target is derived from the live session.  It is re-entrant because an
        # in-place export installs its freshly rebuilt DatasetSession before releasing
        # the transaction.
        self._lock = threading.RLock()
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

    def _session_for_epoch_locked(self, expected_epoch, action: str):
        if self.session is None:
            raise RuntimeError("no project open")
        if expected_epoch is not None and expected_epoch != self.epoch:
            raise _ProjectConflict(f"project changed - reload before {action}")
        return self.session

    def capture_session(self, expected_epoch: int, action: str):
        """Atomically bind a long-running request to its asserted project epoch."""
        with self._lock:
            return self._session_for_epoch_locked(expected_epoch, action)

    def session_is_current(self, session) -> bool:
        """Return whether ``session`` still owns the live project generation."""
        with self._lock:
            return self.session is session

    def _clear_project_caches_locked(self, session) -> None:
        self.engine.clear_pending()
        with self._radar_lock:
            self.radar_findings = {}
        self.embed_points = None
        self.embed._cache.clear()
        self.sam._cur = None
        with self.engine._lock:                       # mutually exclusive with boost publication
            # Rebind before removing the model while still holding the publication
            # lock.  A finishing worker must not observe its old session between
            # these operations and republish that project's model after the switch.
            self.boost.on_project_switch(session)
            self.engine._models.pop("boosted", None)  # boosted model is per-project
        with self._thumb_lock:
            self._thumbs.clear()

    def _install_session_locked(self, session, data) -> dict:
        self.session = session
        self._data = str(data)
        self.epoch += 1
        self._clear_project_caches_locked(session)
        meta = session.meta()
        meta["open"] = True
        meta["epoch"] = self.epoch
        return meta

    def _drop_session_locked(self) -> None:
        self.session = None
        self._data = None
        self.epoch += 1
        self._clear_project_caches_locked(None)

    def register_current(self, data, *, session=None, epoch=None) -> None:
        """Record the open dataset in the registry. The ``labeled`` count needs a
        full label scan, so it runs on a background thread -- never blocking the
        open/switch action (which would be seconds of dead UI on big datasets)."""
        with self._lock:
            session = self.session if session is None else session
            if session is None or self.session is not session:
                return
            epoch = self.epoch if epoch is None else epoch
            if epoch != self.epoch:
                return
            self._data = str(data)
        data = str(data)

        def _work():
            try:
                count = session.meta().get("count")
                labeled = session.stats().get("labeled")
                # The scan above can be slow.  Re-check identity and register while
                # holding the project lock so a later delete cannot forget the entry
                # and then have this stale worker resurrect it.
                with self._lock:
                    if self.session is not session or self.epoch != epoch:
                        return
                    if self._data is None or _project_root_key(self._data) != _project_root_key(data):
                        return
                    name = session.meta().get("name") or None
                    projects.register(data, name=name, root=session.root or None,
                                      count=count, labeled=labeled)
            except Exception:  # noqa: BLE001 - registry is a convenience, never fatal
                logger.exception("project registry update failed")

        threading.Thread(target=_work, name="librelabel-registry", daemon=True).start()

    def open_project(self, data) -> dict:
        """Switch the live session to ``data`` (a data.yaml/dir), resetting all
        per-project state the engines hold. Raises on a bad dataset path."""
        with self._lock:
            # Construct under the same lock as export/delete.  Otherwise an open of
            # the current path can validate a pre-export tree and install that stale
            # view after an in-place export has reorganised it.
            session = DatasetSession(data)
            meta = self._install_session_locked(session, data)
            epoch = self.epoch
        self.register_current(data, session=session, epoch=epoch)
        return meta

    def create_project(self, folder, *, classes, colors, task, link, name) -> dict:
        """Create/open a folder project as one serialized project transaction."""
        with self._lock:
            existing = folder_yaml(str(folder))
            if existing:
                target = existing
            elif link:
                target = create_linked_project(
                    str(folder), name=name or None, classes=classes or [],
                    colors=colors or [], task=task)
            else:
                target = scaffold_data_yaml(str(folder), classes or [], task=task)
            session = DatasetSession(target)
            meta = self._install_session_locked(session, target)
            meta["created"] = existing is None
            epoch = self.epoch
        self.register_current(target, session=session, epoch=epoch)
        return meta

    def save_upload(self, dst, name, data) -> str:
        """Serialize upload reserve/write and New Project finalization."""
        with self._lock:
            return save_uploaded_image(str(dst), str(name), data)

    def create_uploaded_project(self, dst, **kwargs) -> dict:
        with self._lock:
            target = create_uploaded_project(str(dst), **kwargs)
            session = DatasetSession(target)
            meta = self._install_session_locked(session, target)
            meta["created"] = True
            epoch = self.epoch
        self.register_current(target, session=session, epoch=epoch)
        return meta

    def set_class_names(self, names, *, expected_epoch=None) -> dict:
        """Rename and/or append dataset classes -- never delete or reorder, so
        existing label class ids keep their meaning -- rewriting the YAML and
        rebuilding the live session as a new project generation.  A rename changes
        the semantic meaning of an id, so stale tabs and in-flight suggestions must
        not remain writable merely because the integer positions are stable."""
        reopened = None
        data = None
        epoch = None
        with self._lock:
            session = self._session_for_epoch_locked(expected_epoch, "editing classes")
            cleaned = [str(n).strip() for n in names]
            if any(not n for n in cleaned):
                raise ValueError("class names can't be empty")
            if len({n.lower() for n in cleaned}) != len(cleaned):
                raise ValueError("class names must be unique")
            if len(cleaned) < session.nc:
                raise ValueError("classes can be renamed or added here, not removed")
            if cleaned == list(session.names):
                meta = session.meta()
                meta["open"] = True
                meta["epoch"] = self.epoch
                return meta
            # Validate/rebuild the session snapshot before touching the YAML. The
            # class edit itself cannot then leave disk changed while state remains
            # bound to an old generation if dataset reopening fails.
            reopened = DatasetSession(session.yaml_file)
            update_class_names(session.yaml_file, cleaned)
            data = self._data or session.yaml_file
            reopened.names = cleaned
            reopened.nc = len(cleaned)
            meta = self._install_session_locked(reopened, data)
            epoch = self.epoch
        self.register_current(data, session=reopened, epoch=epoch)
        return meta

    def _session_aliases_locked(self, session, submitted=None) -> set[str]:
        values = {
            str(value)
            for value in (
                submitted,
                self._data,
                session.yaml_file,
                Path(session.yaml_file).parent,
            )
            if value
        }
        if session.root and _project_root_key(session.root) == _project_root_key(
            session.yaml_file
        ):
            values.add(str(session.root))
        return values

    def read_label_with_rev(self, idx: int) -> tuple:
        """Read annotations + revision under the save lock, so a save can't land
        between the two and hand a stale client old annotations with a NEW rev
        (which its next save would then use to pass the conflict check)."""
        with self._lock:
            session = self._session_for_epoch_locked(None, "reading labels")
            anns, editable, revision = session.read_label_with_rev(idx)
            return anns, editable, revision, self.epoch

    def dataset_meta(self):
        """Return metadata and its matching epoch from one session snapshot."""
        with self._lock:
            if self.session is None:
                return None
            meta = self.session.meta()
            meta["open"] = True
            meta["epoch"] = self.epoch
            return meta

    def write_label(self, idx: int, boxes, epoch=None, expected_rev=None) -> tuple:
        with self._lock:  # serialize concurrent saves to the same tree (check+write atomic)
            session = self._session_for_epoch_locked(epoch, "saving")
            return session.write_label_with_rev(
                idx, boxes, expected_rev=expected_rev
            )

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

    def clear_pending(self, sess) -> bool:
        """Clear suggestions only if the run still owns the live project."""
        with self._lock:
            if self.session is not sess:
                return False
            self.engine.clear_pending()
            return True

    def store_radar_findings(self, findings, sess) -> bool:
        with self._lock:
            if self.session is not sess:
                return False
            with self._radar_lock:
                self.radar_findings = findings
            return True

    def store_embed_points(self, points, sess) -> bool:
        with self._lock:
            if self.session is not sess:
                return False
            self.embed_points = points
            return True

    def resolve_duplicates(self, ids, purge: bool = False, epoch=None) -> dict:
        with self._lock:  # serialize against label writes on the same tree
            # Same stale-guard as write_label: a Fix request carrying a since-switched
            # project's epoch must not quarantine/purge same-id images in the new one.
            session = self._session_for_epoch_locked(epoch, "fixing duplicates")
            return session.resolve_duplicates(ids, purge=purge)

    def update_project_meta(self, fields: dict, *, expected_epoch: int) -> dict:
        """Update the open project's sidecar without a switch interleaving."""
        with self._lock:
            session = self._session_for_epoch_locked(expected_epoch, "editing settings")
            target = self._data or session.yaml_file
            update_sidecar(str(target), **fields)
            if isinstance(session._sidecar, dict):
                session._sidecar.update(fields)
            else:
                session._sidecar = dict(fields)
            if "name" in fields:
                for alias in self._session_aliases_locked(session, target):
                    projects.rename(alias, fields["name"])
            meta = session.meta()
            meta["open"] = True
            meta["epoch"] = self.epoch
            return meta

    def rename_project(self, data, name: str) -> None:
        with self._lock:
            set_sidecar_name(str(data), name)
            aliases = {str(data)}
            if self.session is not None and _project_root_key(data) == _project_root_key(self.session.yaml_file):
                aliases.update(self._session_aliases_locked(self.session, data))
                if not isinstance(self.session._sidecar, dict):
                    self.session._sidecar = {}
                self.session._sidecar["name"] = name
            for alias in aliases:
                projects.rename(alias, name)

    def forget_project(self, data) -> None:
        with self._lock:
            projects.forget(str(data))

    def delete_project(self, data, *, expected_epoch=None) -> dict:
        """Trash a project and atomically detach it if it is the live session."""
        with self._lock:
            try:
                submitted_key = _project_root_key(data, require_existing=True)
            except FileNotFoundError:
                raise ValueError(
                    "only an open project or an exact registered project root can be deleted"
                ) from None
            current_target = (
                _project_root_entry_path(
                    self.session.yaml_file, require_existing=True
                )
                if self.session is not None
                else None
            )
            current_key = (
                _project_root_key(current_target)
                if current_target is not None
                else None
            )
            registered_target = None
            registered_aliases = set()
            for entry in projects.list_projects():
                registered_data = entry.get("data") if isinstance(entry, dict) else None
                if not registered_data:
                    continue
                try:
                    registered_key = _project_root_key(
                        registered_data, require_existing=True
                    )
                except FileNotFoundError:
                    continue
                if registered_key != submitted_key:
                    continue
                registered_aliases.add(str(registered_data))
                if registered_target is not None:
                    continue
                # The registry is convenience data, not filesystem authorization.
                # A stale/malformed entry must not authorize moving its containing
                # directory after the YAML disappeared.
                try:
                    registered_session = DatasetSession(str(registered_data))
                except Exception:  # noqa: BLE001 - stale/corrupt registry entry
                    continue
                candidate = _project_root_entry_path(
                    registered_session.yaml_file, require_existing=True
                )
                if _project_root_key(candidate) == submitted_key:
                    registered_target = candidate

            is_current = current_key is not None and submitted_key == current_key
            if is_current:
                target = current_target
            else:
                target = registered_target
            if target is None:
                raise ValueError(
                    "only an open project or an exact registered project root can be deleted"
                )
            if is_current:
                if expected_epoch is None:
                    raise ValueError("epoch is required when deleting the open project")
                if expected_epoch != self.epoch:
                    raise _ProjectConflict("project changed - reload before deleting")
            aliases = {str(data), *registered_aliases}
            if is_current:
                aliases.update(self._session_aliases_locked(self.session, data))
            trashed = trash_project(str(target))
            if is_current:
                # The filesystem move succeeded, so the old path must stop being a
                # live session before registry convenience cleanup can run.
                self._drop_session_locked()
            for alias in aliases:
                try:
                    projects.forget(alias)
                except Exception:  # noqa: BLE001 - registry is convenience data
                    logger.exception("project registry cleanup failed")
            return {"trash": trashed, "closed": is_current, "epoch": self.epoch}

    def start_boost(self, *, expected_epoch: int, **kwargs) -> dict:
        """Start Boost only for the project generation asserted by the client."""
        with self._lock:
            session = self._session_for_epoch_locked(expected_epoch, "starting Boost")
            if self.boost.session is not session:
                with self.engine._lock:
                    self.boost.on_project_switch(session)
            return self.boost.start(**kwargs)

    def export_project(self, *, expected_epoch: int, **kwargs) -> dict:
        """Export a stable session snapshot; rebind atomically after in-place work."""
        from . import export as _export

        reopened = None
        reopen_data = None
        reopen_epoch = None
        with self._lock:
            session = self._session_for_epoch_locked(expected_epoch, "exporting")
            res = _export.export_dataset(
                session, _in_place_validator=DatasetSession, **kwargs
            )
            if res.get("in_place"):
                reopen_data = res.get("yaml") or self._data
                if not reopen_data:
                    raise RuntimeError("in-place export did not return a dataset path")
                reopened = res.pop("_reopened_session", None)
                if reopened is None:
                    raise RuntimeError("in-place export did not validate its new session")
                meta = self._install_session_locked(reopened, reopen_data)
                reopen_epoch = self.epoch
                res["epoch"] = reopen_epoch
                res["reopened"] = True
                res["dataset"] = meta
        if reopened is not None:
            self.register_current(reopen_data, session=reopened, epoch=reopen_epoch)
        return res

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
            if not self._host_allowed():
                self._send(403, {"error": "host not allowed"})
                return
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
                host = self.state.host
                try:
                    family = (
                        socket.AF_INET6
                        if ipaddress.ip_address(host.strip("[]")).version == 6
                        else socket.AF_INET
                    )
                except ValueError:
                    family = socket.AF_INET
                ip = _lan_ip(family)
                lan_host = _lan_url_host(host, ip)
                shareable = lan_host is not None
                local_host = _local_access_host(host)
                self._send(200, {
                    "host": host, "port": self.state.port,
                    "local_url": "http://%s:%d" % (_url_host(local_host), self.state.port),
                    "lan_url": (
                        "http://%s:%d" % (_url_host(lan_host), self.state.port)
                        if lan_host is not None
                        else None
                    ),
                    "shareable": shareable,
                })
            elif path == "/api/dataset":
                meta = self.state.dataset_meta()
                if meta is None:
                    self._send(200, {"open": False})
                else:
                    if not self._local_admin():
                        # root/yaml are host-local paths; `reason` can also embed an
                        # absolute path ("Could not derive a label path for <abs>").
                        # LAN teammates only need names/count/writable to label.
                        meta.pop("root", None)
                        meta.pop("yaml", None)
                        meta.pop("reason", None)
                        meta.pop("source", None)   # a linked project's source is a host path
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
                annotations, editable, rev, epoch = self.state.read_label_with_rev(idx)
                # rev as a STRING: nanosecond mtimes (~1e18) exceed JS Number.MAX_SAFE_INTEGER,
                # so a numeric token would be rounded by the browser and every save would 409.
                self._send(200, {
                    "annotations": annotations, "editable": editable,
                    "rev": str(rev), "epoch": epoch,
                })
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
            if not self._host_allowed():
                self._send(403, {"error": "host not allowed"})
                return
            if not self._same_origin():
                self._send(403, {"error": "cross-origin request blocked"})
                return
            # Host-admin only: switching the project, pruning duplicates, and the
            # heavy full-dataset compute streams all rebind/clobber server-global
            # state (session, files, the shared pending/findings/embed maps, a host
            # CPU training job) for every teammate -- gate them to the loopback host.
            if path in ("/api/projects/open", "/api/projects/create", "/api/projects/new",
                        "/api/upload", "/api/projects/inspect", "/api/projects/rename", "/api/projects/delete",
                        "/api/projects/meta", "/api/export",
                        "/api/projects/forget", "/api/pick-folder", "/api/classes", "/api/insights/fix",
                        "/api/boost", "/api/assist/autolabel", "/api/assist/radar",
                        "/api/embeddings") and not self._local_admin():
                self._send(403, {"error": "This action (create / switch project, edit classes, "
                                          "prune duplicates, full-dataset auto-label, Radar, "
                                          "embeddings, Boost) is only allowed from the host "
                                          "machine on a shared server."})
                return
            if self.state.session is None and path not in (
                    "/api/projects/open", "/api/projects/create", "/api/projects/new", "/api/upload",
                    "/api/projects/inspect", "/api/projects/forget", "/api/projects/rename",
                    "/api/projects/delete", "/api/pick-folder"):
                self._send(409, {"error": "no project open"})
                return
            if (path.startswith("/api/assist/") or path in ("/api/embeddings", "/api/boost")) \
                    and not self.state.engine.enabled:
                self._send(403, {"error": "AI assist is disabled (started with --no-assist)."})
                return
            # Task-gate the assist stack. OBB: everything it emits (axis-aligned
            # boxes from prelabel/autolabel/Radar/Boost, free polygons from SAM)
            # would corrupt 9-field oriented-box labels -> refuse all of it.
            # Segment: the BOX producers (prelabel/autolabel/Boost) would write
            # 5-field rows into a polygon dataset -> refuse those; SAM (polygons)
            # and Radar (read-only audit) stay available.
            task = getattr(self.state.session, "_task", "") if self.state.session else ""
            if task == "obb" and (
                    path.startswith(("/api/assist/prelabel", "/api/assist/segment"))
                    or path in ("/api/assist/autolabel", "/api/assist/radar", "/api/boost")):
                self._send(409, {"error": "AI assist works with boxes and masks, not oriented "
                                          "boxes - it is disabled for OBB projects."})
                return
            if task == "segment" and (
                    path.startswith("/api/assist/prelabel")
                    or path in ("/api/assist/autolabel", "/api/boost")):
                self._send(409, {"error": "Box auto-label is disabled for segmentation projects "
                                          "- use SAM (S) or the polygon tool instead."})
                return
            if path == "/api/projects/open":
                payload = self._read_json()
                data = payload.get("data") if isinstance(payload, dict) else None
                if not data:
                    self._send(400, {"error": "data path required"})
                    return
                try:
                    meta = self.state.open_project(str(data))
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
                # Default: scaffold_data_yaml writes a minimal data.yaml beside the
                # images (the exact layout the trainer reads). With ``link: true`` the
                # source folder is never written to at all -- config + labels live in
                # a managed project dir under ~/.librelabel/projects.
                payload = self._read_json()
                folder = payload.get("folder") if isinstance(payload, dict) else None
                classes = payload.get("classes") if isinstance(payload, dict) else None
                task = payload.get("task") if isinstance(payload, dict) else None
                link = bool(payload.get("link")) if isinstance(payload, dict) else False
                name = payload.get("name") if isinstance(payload, dict) else None
                if task not in ("detect", "segment", "obb", "classify"):
                    task = None   # ignore anything unexpected -> default detection
                if not folder:
                    self._send(400, {"error": "folder path required"})
                    return
                try:
                    meta = self.state.create_project(
                        str(folder), classes=classes or [], colors=payload.get("colors") or [],
                        task=task, link=link, name=name)
                    self._send(200, meta)
                except FileNotFoundError as exc:
                    self._send(400, {"error": str(exc).splitlines()[0][:140]
                                     or "No images found in that folder."})
                except Exception as exc:  # noqa: BLE001 - bad folder / unwritable path
                    logger.exception("create project failed")
                    self._send(400, {"error": str(exc).splitlines()[0][:140]
                                     or "Could not create that project."})
            elif path == "/api/upload":
                # One browser-uploaded image -> <dst>/images/train/. Raw bytes in the
                # body (no multipart, stdlib-only); admin-gated like every mutation.
                qs = parse_qs(parsed.query)
                dst = (qs.get("dst") or [None])[0]
                name = (qs.get("name") or [None])[0]
                data = self._read_body_bytes()
                if not dst or not name:
                    self._send(400, {"error": "dst and name are required"})
                    return
                try:
                    saved = self.state.save_upload(str(dst), str(name), data)
                    self._send(200, {"ok": True, "saved": Path(saved).name})
                except Exception as exc:  # noqa: BLE001 - bad name / unwritable dst
                    self._send(400, {"error": str(exc).splitlines()[0][:140] or "upload failed"})
            elif path == "/api/projects/new":
                # The New Project wizard: build a real dataset from uploaded images
                # (optional val split + per-class colors), then open it.
                payload = self._read_json()
                if not isinstance(payload, dict):
                    self._send(400, {"error": "bad payload"})
                    return
                dst = payload.get("dst")
                if not dst:
                    self._send(400, {"error": "destination folder required"})
                    return
                task = payload.get("task")
                if task not in ("detect", "segment", "obb", "classify"):
                    task = None
                try:
                    meta = self.state.create_uploaded_project(
                        str(dst),
                        name=payload.get("name") or None,
                        description=payload.get("description") or "",
                        color=payload.get("color") or "",
                        classes=payload.get("classes") or [],
                        colors=payload.get("colors") or [],
                        task=task,
                        make_val=bool(payload.get("make_val")),
                        val_frac=(
                            0.2
                            if payload.get("val_frac") in (None, "")
                            else payload.get("val_frac")
                        ),
                    )
                    self._send(200, meta)
                except FileNotFoundError as exc:
                    self._send(400, {"error": str(exc).splitlines()[0][:140] or "No images uploaded yet."})
                except Exception as exc:  # noqa: BLE001 - bad dst / unwritable path
                    logger.exception("new project failed")
                    self._send(400, {"error": str(exc).splitlines()[0][:140]
                                     or "Could not create the project."})
            elif path == "/api/classes":
                payload = self._read_json()
                names = payload.get("names") if isinstance(payload, dict) else None
                ep = payload.get("epoch") if isinstance(payload, dict) else None
                if not isinstance(names, list):
                    self._send(400, {"error": "names list required"})
                    return
                ep = _nonnegative_int(ep, "epoch")
                meta = self.state.set_class_names(names, expected_epoch=ep)
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
            elif path == "/api/projects/forget":
                payload = self._read_json()
                data = payload.get("data") if isinstance(payload, dict) else None
                if data:
                    self.state.forget_project(str(data))
                self._send(200, {"ok": True})
            elif path == "/api/projects/meta":
                # Project Settings for the OPEN project: display name, description,
                # and labeling instructions -- all sidecar-only (data.yaml untouched).
                payload = self._read_json()
                if not isinstance(payload, dict):
                    self._send(400, {"error": "bad payload"})
                    return
                ep = _nonnegative_int(payload.get("epoch"), "epoch")
                name = payload.get("name")
                if name is not None and not str(name).strip():
                    self._send(400, {"error": "the project name can't be empty"})
                    return
                try:
                    fields = {k: (str(payload[k]).strip() if k == "name" else str(payload[k]))
                              for k in ("name", "description", "instructions")
                              if payload.get(k) is not None}
                    meta = self.state.update_project_meta(fields, expected_epoch=ep)
                    self._send(200, meta)
                except _ProjectConflict:
                    raise
                except Exception as exc:  # noqa: BLE001 - unwritable sidecar
                    logger.exception("settings update failed")
                    self._send(400, {"error": str(exc).splitlines()[0][:140] or "could not save settings"})
            elif path == "/api/projects/rename":
                payload = self._read_json()
                data = payload.get("data") if isinstance(payload, dict) else None
                name = payload.get("name") if isinstance(payload, dict) else None
                if not data or not name or not str(name).strip():
                    self._send(400, {"error": "data and name are required"})
                    return
                try:
                    self.state.rename_project(str(data), str(name).strip())
                    self._send(200, {"ok": True})
                except Exception as exc:  # noqa: BLE001 - bad path / unwritable
                    self._send(400, {"error": str(exc).splitlines()[0][:140] or "rename failed"})
            elif path == "/api/projects/delete":
                # Soft-delete: move the whole project folder to ~/.librelabel/trash
                # (recoverable). Never erases user files outright.
                payload = self._read_json()
                data = payload.get("data") if isinstance(payload, dict) else None
                if not data:
                    self._send(400, {"error": "data is required"})
                    return
                try:
                    raw_ep = payload.get("epoch") if isinstance(payload, dict) else None
                    ep = _nonnegative_int(raw_ep, "epoch") if raw_ep is not None else None
                    result = self.state.delete_project(str(data), expected_epoch=ep)
                    self._send(200, {"ok": True, **result})
                except _ProjectConflict:
                    raise
                except FileNotFoundError as exc:
                    self._send(400, {"error": str(exc).splitlines()[0][:140] or "project folder not found"})
                except Exception as exc:  # noqa: BLE001 - move/permission failure
                    logger.exception("delete project failed")
                    self._send(400, {"error": str(exc).splitlines()[0][:140] or "delete failed"})
            elif path == "/api/export":
                payload = self._read_json()
                if not isinstance(payload, dict):
                    self._send(400, {"error": "bad payload"})
                    return
                ep = _nonnegative_int(payload.get("epoch"), "epoch")
                try:
                    res = self.state.export_project(
                        expected_epoch=ep,
                        dst=payload.get("dst") or None,
                        formats=tuple(payload.get("formats") or ["yolo"]),
                        split=payload.get("split") or "trainval",
                        val_frac=(
                            0.2
                            if payload.get("val_frac") in (None, "")
                            else payload.get("val_frac")
                        ),
                        test_frac=(
                            0.0
                            if payload.get("test_frac") in (None, "")
                            else payload.get("test_frac")
                        ),
                        seed=(
                            1234
                            if payload.get("seed") in (None, "")
                            else int(payload.get("seed"))
                        ),
                        in_place=bool(payload.get("in_place")),
                        make_zip=bool(payload.get("make_zip")),
                    )
                    self._send(200, {"ok": True, **res})
                except _ProjectConflict:
                    raise
                except Exception as exc:  # noqa: BLE001 - bad dst / unreadable images
                    logger.exception("export failed")
                    self._send(400, {"error": str(exc).splitlines()[0][:160] or "export failed"})
            elif path.startswith("/api/label/"):
                idx = int(path.rsplit("/", 1)[-1])
                payload = self._read_json()
                if not isinstance(payload, dict) or "annotations" not in payload:
                    raise ValueError("JSON object with an explicit annotations list required")
                anns = payload["annotations"]
                if not isinstance(anns, list):
                    raise ValueError("annotations must be a list")
                qs = parse_qs(parsed.query)
                ep = self._required_query_int(qs, "epoch")
                rev = self._required_query_int(qs, "rev")
                count, new_rev = self.state.write_label(
                    idx, anns, epoch=ep, expected_rev=rev)
                self._send(200, {"ok": True, "count": count, "rev": str(new_rev)})
            elif path.startswith("/api/assist/prelabel/"):
                self._handle_prelabel(int(path.rsplit("/", 1)[-1]), parse_qs(parsed.query))
            elif path.startswith("/api/assist/segment/"):
                self._handle_segment(
                    int(path.rsplit("/", 1)[-1]), parse_qs(parsed.query)
                )
            elif path == "/api/assist/autolabel":
                self._handle_autolabel_stream(parse_qs(parsed.query))
            elif path == "/api/assist/radar":
                self._handle_radar_stream(parse_qs(parsed.query))
            elif path == "/api/embeddings":
                self._handle_embeddings_stream(parse_qs(parsed.query))
            elif path == "/api/insights/fix":
                payload = self._read_json()
                ids = payload.get("ids", []) if isinstance(payload, dict) else []
                purge = bool(payload.get("purge")) if isinstance(payload, dict) else False
                ep = payload.get("epoch") if isinstance(payload, dict) else None
                res = self.state.resolve_duplicates(
                    [int(i) for i in ids], purge=purge,
                    epoch=_nonnegative_int(ep, "epoch"))
                self._send(200, res)
            elif path == "/api/boost":
                payload = self._read_json()
                kw = payload if isinstance(payload, dict) else {}
                epoch = _nonnegative_int(kw.get("epoch"), "epoch")
                self._send(200, self.state.start_boost(
                    expected_epoch=epoch,
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
        return (
            json.loads(data.decode("utf-8"), object_pairs_hook=_unique_json_object)
            if data else {}
        )

    def _read_body_bytes(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(length) if length else b""

    @staticmethod
    def _required_query_int(qs: dict, name: str) -> int:
        values = qs.get(name)
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"exactly one {name} query parameter is required")
        return _nonnegative_int(values[0], name)

    @staticmethod
    def _is_loopback(addr: str) -> bool:
        a = (addr or "").strip().lower().strip("[]")
        if a.rstrip(".") == "localhost":
            return True
        try:
            return _ip_is_loopback(ipaddress.ip_address(a))
        except ValueError:
            return False

    def _local_admin(self) -> bool:
        """Whether this client may perform host-level admin (switch project, prune
        duplicates, list the project registry) -- actions that rebind the global
        session, move/delete files, or expose host-local paths.

        A wildcard bind (what ``--share`` uses) reserves admin for loopback peers.
        On a concrete NIC bind, a host browser normally connects with the same
        source address as the accepted socket's local endpoint; a LAN teammate has
        a different peer address. This preserves local admin without granting every
        client on that interface host filesystem authority.
        """
        peer_text = self.client_address[0] if self.client_address else ""
        if self._is_loopback(peer_text):
            return True
        host = (self.state.host or "").strip().lower().strip("[]")
        try:
            bind_address = ipaddress.ip_address(host)
        except ValueError:
            bind_address = None
        if not host or bool(bind_address and bind_address.is_unspecified):
            return False
        try:
            peer = ipaddress.ip_address(str(peer_text).strip().strip("[]"))
            local = ipaddress.ip_address(
                str(self.connection.getsockname()[0]).strip().strip("[]")
            )
        except (AttributeError, OSError, ValueError):
            return False
        peer_mapped = getattr(peer, "ipv4_mapped", None) or peer
        local_mapped = getattr(local, "ipv4_mapped", None) or local
        return peer_mapped == local_mapped

    def _same_origin(self) -> bool:
        """Compare a browser Origin to the normalized HTTP request authority."""
        origins = self.headers.get_all("Origin") or []
        if not origins:
            return True
        if len(origins) != 1:
            return False
        origin = origins[0]
        if not origin or origin != origin.strip() or origin.lower() == "null":
            return False
        try:
            parsed = urlsplit(origin)
            if (
                parsed.scheme.lower() != "http"
                or not parsed.netloc
                or parsed.path
                or parsed.query
                or parsed.fragment
                or "?" in origin
                or "#" in origin
                or parsed.username is not None
                or parsed.password is not None
            ):
                return False
            origin_authority = _parse_authority(parsed.netloc, default_port=80)
        except (TypeError, ValueError):
            return False
        return origin_authority == self._request_authority()

    def _request_authority(self) -> Optional[tuple[str, int]]:
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1:
            return None
        try:
            authority = _parse_authority(hosts[0], default_port=80)
        except (TypeError, ValueError):
            return None
        if authority[1] != self.state.port:
            return None
        target = urlsplit(self.path)
        if target.scheme or target.netloc:
            if target.scheme.lower() != "http" or not target.netloc:
                return None
            try:
                target_authority = _parse_authority(
                    target.netloc, default_port=80
                )
            except (TypeError, ValueError):
                return None
            if target_authority != authority:
                return None
        return authority

    def _host_allowed(self) -> bool:
        """Allow only the configured host or explicit local numeric addresses."""
        authority = self._request_authority()
        if authority is None:
            return False
        name, _port = authority
        try:
            address = ipaddress.ip_address(name)
        except ValueError:
            address = None

        bind_raw = (self.state.host or "").strip().lower().strip("[]")
        try:
            bind_name = _normalize_host_name(bind_raw) if bind_raw else ""
        except ValueError:
            return False
        try:
            bind_address = ipaddress.ip_address(bind_name) if bind_name else None
        except ValueError:
            bind_address = None

        wildcard = not bind_name or bool(bind_address and bind_address.is_unspecified)
        if name == "localhost":
            return wildcard or bind_name == "localhost" or bool(
                bind_address and _ip_is_loopback(bind_address)
            )
        if address is None:
            # A deliberately configured DNS bind may use its exact name. Arbitrary
            # DNS Host values are never accepted, including on --share.
            return bool(bind_address is None and bind_name and name == bind_name)
        if address.is_unspecified or address.is_multicast:
            return False
        if _ip_is_loopback(address):
            return wildcard or bind_name == "localhost" or bool(
                bind_address and _ip_is_loopback(bind_address)
            )
        if address.is_reserved:
            return False
        if wildcard:
            return True
        if bind_address is None:
            return False
        if _ip_is_loopback(bind_address):
            return False
        return address == bind_address

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
        epoch = self._required_query_int(qs, "epoch")
        # Epoch validation and session capture are one transaction.  A separate
        # check followed by `self.state.session` can capture the *next* project if
        # a switch lands between the two operations.
        sess = self.state.capture_session(epoch, "auto-labeling")
        model, conf = self._model_conf(qs)
        # Snapshot the session up front: the predict below is slow, and if the user
        # switches projects mid-flight we must not store these suggestions into the
        # shared pending map (keyed by numeric id) for the now-current project.
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
            # PIL/torch errors embed absolute image paths; redact for LAN clients.
            self._send(503, {"error": str(exc) if self._local_admin() else _redact_paths(str(exc))})
            return
        if not self.state.store_pending(idx, sugg, sess):   # atomic with project switch
            self._send(409, {"error": "project changed; reopen and retry"})
            return
        self._send(200, {"editable": True, "suggestions": sugg})

    def _handle_segment(self, idx: int, qs: dict) -> None:
        payload = self._read_json() or {}
        if not isinstance(payload, dict):
            payload = {}
        epoch = self._required_query_int(qs, "epoch")
        sess = self.state.capture_session(epoch, "segmenting")
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
            poly = self.state.sam.segment(sess.image_path(idx), **kw)
        except Exception as exc:  # noqa: BLE001 - SAM load/inference problem
            logger.exception("segment failed")
            # PIL/torch errors embed absolute image paths; redact for LAN clients.
            self._send(503, {"error": str(exc) if self._local_admin() else _redact_paths(str(exc))})
            return
        if not self.state.session_is_current(sess):
            self._send(409, {"error": "project changed; reopen and retry"})
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
        epoch = self._required_query_int(qs, "epoch")
        sess = self.state.capture_session(epoch, "auto-labeling")

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
            # Fresh run replaces old suggestions only while it still owns this
            # project.  An old run resuming after a switch must not clear the new
            # project's pending review deck.
            if not self.state.clear_pending(sess):
                emit({"type": "error", "error": "project changed; reopen and retry"})
                return
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
        epoch = self._required_query_int(qs, "epoch")
        sess = self.state.capture_session(epoch, "running Radar")
        model, conf = self._model_conf(qs)
        if not self.state.engine.status().get("available"):
            self._send(503, {"error": "No model available for Radar "
                                      "(assist disabled or no weights)."})
            return
        emit = self._ndjson_begin()
        try:
            result = scan_dataset(self.state.engine.predict_image, sess,
                                  model_name=model, conf=conf, progress=emit)
            # Publish under the same project lock as open_project(), otherwise a
            # switch can interleave after an identity check and receive stale ids.
            self.state.store_radar_findings(result.pop("findings", {}), sess)
            emit(result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("radar scan failed")
            emit({"type": "error", "error": str(exc)})

    def _handle_embeddings_stream(self, qs: dict) -> None:
        """Embed every image and stream the 2-D PCA scatter (``{id,x,y}``)."""
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        epoch = self._required_query_int(qs, "epoch")
        sess = self.state.capture_session(epoch, "mapping")
        if not self.state.embed.available():
            self._send(503, {"error": "Embeddings unavailable (assist disabled)."})
            return
        emit = self._ndjson_begin()
        try:
            points = self.state.embed.scatter(sess, progress=emit)
            self.state.store_embed_points(points, sess)
            emit({"type": "done", "points": points})
        except Exception as exc:  # noqa: BLE001
            logger.exception("embeddings failed")
            emit({"type": "error", "error": str(exc)})


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
    bind_host = str(host).strip() or "0.0.0.0"
    if bind_host.startswith("[") and bind_host.endswith("]"):
        bind_host = bind_host[1:-1]
    session = DatasetSession(data) if data else None
    state = _LabelState(session, device=device, assist=assist)
    state.host = bind_host
    handler = type("BoundLabelHandler", (_Handler,), {"state": state})
    try:
        ipv6_bind = ipaddress.ip_address(bind_host).version == 6
    except ValueError:
        ipv6_bind = False
    server_type = ThreadingHTTPServer
    if ipv6_bind:
        server_type = type(
            "IPv6ThreadingHTTPServer",
            (ThreadingHTTPServer,),
            {"address_family": socket.AF_INET6},
        )
    httpd = server_type((bind_host, port), handler)
    # Publish the *actual* bound port: with port=0 the OS assigns one, so the
    # requested value would yield unusable ":0" URLs and poison /api/server.
    bound_port = httpd.server_address[1]
    state.port = bound_port
    if session is not None:
        state.register_current(data)
    url = "http://%s:%d" % (_url_host(bind_host), bound_port)
    if open_browser:
        import webbrowser

        if bind_host in ("0.0.0.0", ""):
            browse = "http://127.0.0.1:%d" % bound_port
        elif bind_host == "::":
            browse = "http://[::1]:%d" % bound_port
        else:
            browse = url
        threading.Timer(0.7, lambda: webbrowser.open(browse)).start()
    return httpd, url, session
