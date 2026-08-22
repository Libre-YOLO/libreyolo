"""Private subprocess coordinator for automatic local DDP training.

The user-facing process writes one short-lived job, then starts this file with
``sys.executable``. Keeping ``torch.multiprocessing.spawn`` here means
spawned ranks import a LibreYOLO module instead of re-importing the user's
``__main__`` module and repeating an unguarded top-level ``model.train()``.

Jobs are trusted, local, ephemeral data created by the current LibreYOLO
process. Like every pickle transport, job payloads must never be accepted
from an untrusted source.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pickle
import secrets
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any


JOB_PROTOCOL = "libreyolo.ddp.local-job"
JOB_PROTOCOL_VERSION = 1
MANIFEST_NAME = "manifest.json"
PAYLOAD_NAME = "payload.pkl"
STATUS_NAME = "status.json"
CLEANUP_SENTINEL_NAME = ".libreyolo-ddp-cleanup.json"


class JobProtocolError(RuntimeError):
    """Raised when a private DDP job descriptor is missing or incompatible."""


class JobTransportError(RuntimeError):
    """Raised when a standard-picklable job cannot use coordinator transport."""


class ParentProcessExited(RuntimeError):
    """Raised when the API process exits while its coordinator is running."""


def _atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    os.replace(temporary, path)


@contextlib.contextmanager
def job_workspace():
    """Yield one launcher-owned job path plus its unpersisted cleanup token."""
    with tempfile.TemporaryDirectory(prefix="libreyolo-ddp-job-") as temp_root:
        root = Path(temp_root).resolve()
        directory = root / "job"
        cleanup_token = secrets.token_hex(32)
        _atomic_json_write(
            root / CLEANUP_SENTINEL_NAME,
            {
                "protocol": JOB_PROTOCOL,
                "version": JOB_PROTOCOL_VERSION,
                "job_id": None,
                "job_dir": str(directory),
                "workspace": str(root),
                "token_sha256": hashlib.sha256(
                    cleanup_token.encode("utf-8")
                ).hexdigest(),
            },
        )
        yield directory, cleanup_token


def _validate_protocol(data: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise JobProtocolError(f"DDP {source} must be a mapping")
    if data.get("protocol") != JOB_PROTOCOL:
        raise JobProtocolError(
            f"Unsupported DDP {source} protocol {data.get('protocol')!r}; "
            f"expected {JOB_PROTOCOL!r}."
        )
    if data.get("version") != JOB_PROTOCOL_VERSION:
        raise JobProtocolError(
            f"Unsupported DDP {source} version {data.get('version')!r}; "
            f"expected {JOB_PROTOCOL_VERSION}."
        )
    return data


def _load_manifest(job_dir: str | Path) -> dict[str, Any]:
    path = Path(job_dir) / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JobProtocolError(
            f"Could not read DDP job manifest {path}: {exc}"
        ) from exc
    data = _validate_protocol(data, source="job manifest")
    nprocs = data.get("nprocs")
    if not isinstance(nprocs, int) or isinstance(nprocs, bool) or nprocs < 1:
        raise JobProtocolError(
            f"DDP job nprocs must be a positive integer, got {nprocs!r}"
        )
    if not isinstance(data.get("job_id"), str) or not data["job_id"]:
        raise JobProtocolError("DDP job manifest is missing job_id")
    import_paths = data.get("import_paths")
    if not isinstance(import_paths, list) or not all(
        isinstance(item, str) and item for item in import_paths
    ):
        raise JobProtocolError(
            "DDP job import_paths must be a list of non-empty strings"
        )
    return data


def _configure_import_paths(manifest: dict[str, Any]) -> None:
    """Prepend the parent's normalized import paths without using PYTHONPATH."""
    requested = manifest["import_paths"]
    requested_keys = {os.path.normcase(os.path.abspath(item)) for item in requested}
    remainder = [
        item
        for item in sys.path
        if not isinstance(item, str)
        or os.path.normcase(os.path.abspath(item or os.getcwd())) not in requested_keys
    ]
    sys.path[:] = [*requested, *remainder]


def _load_payload(job_dir: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    import cloudpickle

    path = Path(job_dir) / PAYLOAD_NAME
    try:
        with path.open("rb") as handle:
            data = cloudpickle.load(handle)
    except Exception as exc:
        raise JobProtocolError(f"Could not read DDP job payload {path}: {exc}") from exc
    data = _validate_protocol(data, source="job payload")
    if data.get("job_id") != manifest["job_id"]:
        raise JobProtocolError("DDP job manifest and payload IDs do not match")
    if data.get("nprocs") != manifest["nprocs"]:
        raise JobProtocolError(
            "DDP job manifest and payload process counts do not match"
        )
    if data.get("import_paths") != manifest["import_paths"]:
        raise JobProtocolError("DDP job manifest and payload import paths do not match")
    if not callable(data.get("worker_fn")):
        raise JobProtocolError("DDP job worker_fn is not callable")
    if not isinstance(data.get("spawn_args"), tuple):
        raise JobProtocolError("DDP job spawn_args must be a tuple")
    return data


def write_job(
    job_dir: str | Path,
    *,
    worker_fn,
    spawn_args: tuple,
    nprocs: int,
    result_path: str,
    master_addr: str,
    master_port: int,
    import_paths: list[str],
) -> None:
    """Write a validated, versioned job for the private coordinator."""
    import cloudpickle

    if not isinstance(nprocs, int) or isinstance(nprocs, bool) or nprocs < 1:
        raise ValueError(f"nprocs must be a positive integer, got {nprocs!r}")
    if not isinstance(spawn_args, tuple):
        raise TypeError(f"spawn_args must be a tuple, got {type(spawn_args).__name__}")
    if not import_paths or not all(
        isinstance(item, str) and item for item in import_paths
    ):
        raise TypeError("import_paths must be a non-empty list of strings")

    # Keep the existing public contract even though cloudpickle is used for
    # transport: lambdas, closures, and other objects rejected by standard
    # pickle must still fail before any child is launched.
    try:
        pickle.dumps((worker_fn, spawn_args), protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise RuntimeError(
            "DDP spawn: worker function and arguments must be picklable with "
            "Python's standard pickle protocol."
        ) from exc

    directory = Path(job_dir)
    directory.mkdir(parents=True, exist_ok=False)
    job_id = uuid.uuid4().hex
    common = {
        "protocol": JOB_PROTOCOL,
        "version": JOB_PROTOCOL_VERSION,
        "job_id": job_id,
        "nprocs": nprocs,
        "import_paths": import_paths,
    }
    payload = {
        **common,
        "worker_fn": worker_fn,
        "spawn_args": spawn_args,
        "result_path": str(result_path),
        "master_addr": str(master_addr),
        "master_port": int(master_port),
    }
    try:
        serialized = cloudpickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise JobTransportError(
            "DDP coordinator transport could not serialize a job that passed "
            "standard pickle validation."
        ) from exc
    (directory / PAYLOAD_NAME).write_bytes(serialized)
    _atomic_json_write(directory / MANIFEST_NAME, common)


def _coordinator_worker(rank: int, job_dir: str) -> None:
    """Load the job independently in each rank and invoke its worker."""
    manifest = _load_manifest(job_dir)
    payload = _load_payload(job_dir, manifest)
    nprocs = payload["nprocs"]

    os.environ["LIBREYOLO_DDP_COORDINATOR_WORKER"] = "1"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(nprocs)
    os.environ["MASTER_ADDR"] = payload["master_addr"]
    os.environ["MASTER_PORT"] = str(payload["master_port"])

    payload["worker_fn"](
        rank,
        nprocs,
        payload["master_addr"],
        payload["master_port"],
        payload["result_path"],
        *payload["spawn_args"],
    )


def _write_status(
    job_dir: Path, manifest: dict[str, Any] | None, **fields: Any
) -> bool:
    status = {
        "protocol": JOB_PROTOCOL,
        "version": JOB_PROTOCOL_VERSION,
        "job_id": manifest.get("job_id") if manifest else None,
        **fields,
    }
    try:
        _atomic_json_write(job_dir / STATUS_NAME, status)
    except OSError:
        # The traceback printed by main remains the fallback if status cannot
        # be persisted (for example, the temp directory was externally removed).
        return False
    return True


def _watch_parent(
    parent_gone: threading.Event, parent_connection: socket.socket
) -> None:
    """Set *parent_gone* when the API process closes its liveness socket."""
    try:
        parent_connection.recv(1)
    except OSError:
        pass
    parent_gone.set()


def _stop_spawned_workers(context) -> None:
    """Terminate every still-live worker owned by a SpawnContext."""
    processes = list(context.processes)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join(timeout=5)


def _safe_cleanup_root(directory: Path) -> Path | None:
    """Return the exact launcher temp root, or None for caller-chosen paths."""
    try:
        resolved_directory = directory.resolve()
        system_temp = Path(tempfile.gettempdir()).resolve()
    except (OSError, RuntimeError):
        return None
    temp_root = resolved_directory.parent
    if (
        resolved_directory.name != "job"
        or temp_root.parent != system_temp
        or not temp_root.name.startswith("libreyolo-ddp-job-")
    ):
        return None
    return temp_root


def _cleanup_abandoned_job(
    directory: Path,
    manifest: dict[str, Any] | None,
    cleanup_token: str | None,
) -> None:
    """Remove only a launcher-authenticated job below the system temp root."""
    if manifest is None or not cleanup_token:
        return
    temp_root = _safe_cleanup_root(directory)
    if temp_root is None:
        return
    sentinel_path = temp_root / CLEANUP_SENTINEL_NAME
    try:
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
        sentinel_job_dir = Path(sentinel["job_dir"]).resolve()
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    if sentinel != {
        "protocol": JOB_PROTOCOL,
        "version": JOB_PROTOCOL_VERSION,
        "job_id": manifest.get("job_id"),
        "job_dir": str(sentinel_job_dir),
        "workspace": str(temp_root),
        "token_sha256": hashlib.sha256(cleanup_token.encode("utf-8")).hexdigest(),
    }:
        return
    if sentinel_job_dir != directory.resolve():
        return
    shutil.rmtree(temp_root, ignore_errors=True)


def _write_cleanup_sentinel(directory: Path, cleanup_token: str) -> None:
    """Authorize abandoned cleanup for one exact launcher-created job."""
    temp_root = _safe_cleanup_root(directory)
    if temp_root is None:
        raise ValueError(
            "DDP coordinator cleanup requires a direct LibreYOLO job directory "
            "under the resolved system temporary directory."
        )
    manifest = _load_manifest(directory)
    sentinel_path = temp_root / CLEANUP_SENTINEL_NAME
    expected_token_hash = hashlib.sha256(cleanup_token.encode("utf-8")).hexdigest()
    try:
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("DDP coordinator cleanup sentinel is unavailable.") from exc
    if sentinel != {
        "protocol": JOB_PROTOCOL,
        "version": JOB_PROTOCOL_VERSION,
        "job_id": None,
        "job_dir": str(directory.resolve()),
        "workspace": str(temp_root),
        "token_sha256": expected_token_hash,
    }:
        raise ValueError("DDP coordinator cleanup sentinel does not match its job.")
    _atomic_json_write(
        sentinel_path,
        {
            "protocol": JOB_PROTOCOL,
            "version": JOB_PROTOCOL_VERSION,
            "job_id": manifest["job_id"],
            "job_dir": str(directory.resolve()),
            "workspace": str(temp_root),
            "token_sha256": expected_token_hash,
        },
    )


def coordinator_main(
    job_dir: str | Path,
    *,
    parent_connection: socket.socket | None = None,
    cleanup_token: str | None = None,
) -> int:
    """Run one descriptor and return a process exit code."""
    directory = Path(job_dir)
    manifest: dict[str, Any] | None = None
    parent_gone = threading.Event()
    try:
        manifest = _load_manifest(directory)
        _configure_import_paths(manifest)

        import torch.multiprocessing as mp

        context = mp.spawn(
            _coordinator_worker,
            args=(str(directory),),
            nprocs=manifest["nprocs"],
            join=False,
        )
        if parent_connection is not None:
            watcher = threading.Thread(
                target=_watch_parent,
                args=(parent_gone, parent_connection),
                daemon=True,
            )
            watcher.start()
        while not context.join(timeout=0.2):
            if parent_gone.is_set():
                _stop_spawned_workers(context)
                raise ParentProcessExited(
                    "The LibreYOLO API process exited while DDP workers were running."
                )
    except BaseException as exc:
        rendered = traceback.format_exc()
        persisted = _write_status(
            directory,
            manifest,
            outcome="error",
            error_type=type(exc).__name__,
            message=str(exc),
            traceback=rendered,
        )
        if not persisted:
            print(rendered, file=sys.stderr, end="")
        if parent_gone.is_set():
            _cleanup_abandoned_job(directory, manifest, cleanup_token)
        return 1

    _write_status(directory, manifest, outcome="ok")
    return 0


def _interrupt_process_tree(process: subprocess.Popen) -> None:
    """Ask a coordinator process group to stop, then escalate if necessary."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=5)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def launch_coordinator(
    job_dir: str | Path, *, env: dict[str, str], cleanup_token: str
) -> None:
    """Launch the packaged coordinator and surface its structured failure."""
    directory = Path(job_dir)
    parent_token = secrets.token_hex(32)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(15)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(directory),
        "--parent-port",
        str(listener.getsockname()[1]),
        "--parent-token",
        parent_token,
        "--cleanup-token",
        cleanup_token,
    ]
    popen_kw: dict[str, Any] = {
        "env": env,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kw["start_new_session"] = True

    process = subprocess.Popen(command, **popen_kw)
    parent_connection: socket.socket | None = None
    try:
        parent_connection, _ = listener.accept()
        parent_connection.settimeout(15)
        received = bytearray()
        expected_size = len(parent_token)
        while len(received) < expected_size:
            chunk = parent_connection.recv(expected_size - len(received))
            if not chunk:
                break
            received.extend(chunk)
        if not secrets.compare_digest(received.decode("ascii"), parent_token):
            raise RuntimeError("LibreYOLO DDP coordinator handshake failed.")
        parent_connection.settimeout(None)
        returncode = process.wait()
    except BaseException:
        if parent_connection is not None:
            parent_connection.close()
            parent_connection = None
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _interrupt_process_tree(process)
        raise
    finally:
        listener.close()
        if parent_connection is not None:
            parent_connection.close()

    if returncode == 0:
        return

    status_path = directory / STATUS_NAME
    detail = ""
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if isinstance(status, dict):
            detail = status.get("traceback") or status.get("message") or ""
    except (OSError, json.JSONDecodeError):
        pass
    suffix = f"\n{detail.rstrip()}" if detail else ""
    raise RuntimeError(
        f"LibreYOLO DDP coordinator exited with code {returncode}.{suffix}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LibreYOLO private DDP coordinator")
    parser.add_argument("job_dir")
    parser.add_argument("--parent-port", type=int, required=True)
    parser.add_argument("--parent-token", required=True)
    parser.add_argument("--cleanup-token", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    with socket.create_connection(
        ("127.0.0.1", args.parent_port), timeout=15
    ) as parent:
        parent.sendall(args.parent_token.encode("ascii"))
        parent.settimeout(None)
        _write_cleanup_sentinel(Path(args.job_dir), args.cleanup_token)
        return coordinator_main(
            args.job_dir,
            parent_connection=parent,
            cleanup_token=args.cleanup_token,
        )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JOB_PROTOCOL",
    "JOB_PROTOCOL_VERSION",
    "JobProtocolError",
    "JobTransportError",
    "coordinator_main",
    "job_workspace",
    "launch_coordinator",
    "write_job",
]
