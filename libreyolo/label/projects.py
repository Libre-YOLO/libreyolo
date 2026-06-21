"""Project registry for LibreLabel: remember the datasets you label.

A tiny JSON file at ``~/.librelabel/projects.json`` records every dataset opened,
so the home screen can list them with their progress and you can hop between
projects without restarting. No database, no new dependency -- the same
zero-dependency spirit as the rest of LibreLabel. The registry stores only
*paths and cached counts*; the datasets themselves are never copied or moved,
and a missing/corrupt registry is never fatal (it is a convenience, not a store).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

_WLOCK = threading.Lock()   # serialize read-modify-write of the registry file


def registry_dir() -> Path:
    return Path.home() / ".librelabel"


def registry_path() -> Path:
    return registry_dir() / "projects.json"


def _key(data: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(str(data)))
    except Exception:  # noqa: BLE001
        return str(data)


def _default_name(data: str, root: Optional[str]) -> str:
    if root:
        n = Path(root).name
        if n:
            return n
    p = Path(str(data))
    return p.parent.name or p.stem or "dataset"


def load() -> List[dict]:
    p = registry_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save(projects: List[dict]) -> None:
    try:
        registry_dir().mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(registry_dir()), suffix=".tmp")  # unique per write
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(projects, f, indent=1)
            os.replace(tmp, registry_path())
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except OSError:
        pass  # the registry is a convenience; never crash labelling over it


def _last_opened(entry: dict) -> float:
    """``last_opened`` as a float, tolerant of malformed registry entries.

    The registry is convenience data a user (or a stale writer) could corrupt;
    a string/None timestamp must not raise ``TypeError`` and break the listing.
    """
    try:
        return float(entry.get("last_opened", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def list_projects() -> List[dict]:
    projects = load()
    projects.sort(key=_last_opened, reverse=True)
    return projects


def register(data: str, name: Optional[str] = None, *, root: Optional[str] = None,
             count: Optional[int] = None, labeled: Optional[int] = None) -> dict:
    """Insert or update a project entry, bumping ``last_opened`` to now."""
    key = _key(data)
    with _WLOCK:
        projects = load()
        entry = next((e for e in projects if e.get("key") == key), None)
        if entry is None:
            entry = {"key": key, "data": str(data)}
            projects.append(entry)
        entry["name"] = name or entry.get("name") or _default_name(data, root)
        if root is not None:
            entry["root"] = str(root)
        if count is not None:
            entry["count"] = int(count)
        if labeled is not None:
            entry["labeled"] = int(labeled)
        entry["last_opened"] = time.time()
        _save(projects)
    return entry


def forget(data: str) -> None:
    key = _key(data)
    with _WLOCK:
        _save([e for e in load() if e.get("key") != key])


def rename(data: str, name: str) -> None:
    """Set the display name of a registry entry without bumping ``last_opened``."""
    key = _key(data)
    with _WLOCK:
        projects = load()
        for e in projects:
            if e.get("key") == key:
                e["name"] = str(name)
                break
        _save(projects)
