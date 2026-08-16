"""Unit tests for immutable LibreYOLO VLM Hub transport."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from libreyolo.models.vlm import hub as vlm_hub

pytestmark = pytest.mark.unit

_REVISION = "1" * 40
_INITIAL_REVISION = "9" * 40
_AGGREGATE = "a" * 64
_SOURCE = f"hf+vlm://alice/detector@{_REVISION}"


def _manifest():
    return {
        "schema": "libreyolo.vlm-artifact.v1",
        "representation": "adapter",
        "identity": {"family": "qwen3vl", "size": "2b", "task": "detect"},
        "files": [
            {
                "path": ".gitattributes",
                "role": "hub_config",
                "size": 50,
                "sha256": "1" * 64,
            },
            {
                "path": "adapter/adapter_model.safetensors",
                "role": "weights",
                "size": 7,
                "sha256": "b" * 64,
            },
            {
                "path": "processor.json",
                "role": "processor",
                "size": 9,
                "sha256": "c" * 64,
            },
        ],
        "aggregate_sha256": _AGGREGATE,
    }


def _contents():
    return {
        ".gitattributes": b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        "adapter/adapter_model.safetensors": b"weights",
        "processor.json": b"processor",
    }


def _write_remote(root: Path, manifest=None, contents=None) -> Path:
    manifest = _manifest() if manifest is None else manifest
    contents = _contents() if contents is None else contents
    root.mkdir(parents=True)
    (root / vlm_hub.VLM_ARTIFACT_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for name, payload in contents.items():
        target = root.joinpath(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return root


def _install_artifact_stubs(monkeypatch, expected_contents=None, events=None):
    expected_contents = _contents() if expected_contents is None else expected_contents

    def info(root, *, check_payload):
        root = Path(root)
        data = json.loads(
            (root / vlm_hub.VLM_ARTIFACT_MANIFEST).read_text(encoding="utf-8")
        )
        names = tuple(entry["path"] for entry in data["files"])
        if check_payload:
            actual = {
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            }
            assert actual == {vlm_hub.VLM_ARTIFACT_MANIFEST, *names}
            for name in names:
                if (
                    root.joinpath(*name.split("/")).read_bytes()
                    != expected_contents[name]
                ):
                    raise ValueError(f"hash mismatch for {name}")
        return SimpleNamespace(
            root=root,
            manifest=data,
            aggregate_sha256=data["aggregate_sha256"],
            files=names,
        )

    def read_manifest(path, *, require_payload=False):
        assert require_payload is False
        if events is not None:
            events.append("read_manifest")
        return info(path, check_payload=False)

    def validate(path):
        if events is not None:
            events.append("validate")
        return info(path, check_payload=True)

    monkeypatch.setattr(vlm_hub, "read_vlm_artifact_manifest", read_manifest)
    monkeypatch.setattr(vlm_hub, "validate_vlm_artifact", validate)


class _NoApi:
    def __init__(self, *args, **kwargs):
        raise AssertionError("HfApi must not be constructed")


def _install_download_hub(monkeypatch, remote, calls):
    hub = pytest.importorskip("huggingface_hub")

    def download(**kwargs):
        target = remote.joinpath(*kwargs["filename"].split("/"))
        if kwargs.get("dry_run"):
            return SimpleNamespace(file_size=target.stat().st_size)
        calls.append(dict(kwargs))
        return str(target)

    monkeypatch.setattr(hub, "hf_hub_download", download)
    monkeypatch.setattr(hub, "HfApi", _NoApi)
    return hub


def test_parse_vlm_hub_uri_is_exact_and_immutable():
    parsed = vlm_hub.parse_vlm_hub_uri(_SOURCE)
    assert parsed == vlm_hub.VLMHubRef("alice/detector", _REVISION)
    assert parsed.uri == _SOURCE


def test_parse_vlm_hub_uri_matches_hub_repo_id_edge_rules():
    source = f"hf+vlm://owner/_repo_@{_REVISION}"

    assert vlm_hub.parse_vlm_hub_uri(source).repo_id == "owner/_repo_"

    with pytest.raises(ValueError, match="Invalid Hugging Face repository id"):
        vlm_hub.parse_vlm_hub_uri(f"hf+vlm://owner/repo.git@{_REVISION}")

    long_repo = f"{'o' * 96}/{'r' * 96}"
    assert (
        vlm_hub.parse_vlm_hub_uri(f"hf+vlm://{long_repo}@{_REVISION}").repo_id
        == long_repo
    )


@pytest.mark.parametrize(
    "source",
    [
        "alice/detector",
        "hf://alice/detector@" + _REVISION,
        "hf+vlm://alice/detector@main",
        "hf+vlm://alice/detector@" + _REVISION.upper().replace("1", "A"),
        "hf+vlm://alice/detector@1234567",
        "hf+vlm://alice/detector@" + _REVISION + "/file",
        "hf+vlm://alice/detector@" + _REVISION + "?download=1",
        "hf+vlm://alice/bad--name@" + _REVISION,
        "hf+vlm://alice/extra/repo@" + _REVISION,
    ],
)
def test_parse_vlm_hub_uri_rejects_noncanonical_sources(source):
    with pytest.raises(ValueError):
        vlm_hub.parse_vlm_hub_uri(source)


def test_inspect_fetches_only_manifest_at_exact_revision_offline(tmp_path, monkeypatch):
    remote = _write_remote(tmp_path / "remote")
    calls = []
    _install_artifact_stubs(monkeypatch)
    _install_download_hub(monkeypatch, remote, calls)

    inspected = vlm_hub.inspect_vlm_hub_artifact(
        _SOURCE, token="hf_read_token", local_files_only=True
    )

    assert inspected == _manifest()
    assert [call["filename"] for call in calls] == [vlm_hub.VLM_ARTIFACT_MANIFEST]
    assert calls[0]["revision"] == _REVISION
    assert calls[0]["local_files_only"] is True
    assert calls[0]["repo_id"] == "alice/detector"


def test_inspect_thaws_real_core_manifest_result(tmp_path, monkeypatch):
    # Use the core contract's complete builder fixture so this boundary sees
    # its recursively frozen MappingProxyType/tuple result.
    from libreyolo.models.vlm import artifact as artifact_module
    from tests.unit.test_vlm_artifact import _artifact, _processor_payloads, _sha

    toy = {
        "2b": {"layers": 1, "hidden": 1, "q": 1, "kv": 1, "intermediate": 1},
        "4b": {"layers": 1, "hidden": 1, "q": 1, "kv": 1, "intermediate": 1},
    }
    processor_records = tuple(
        (name, len(payload), _sha(payload))
        for name, payload in sorted(_processor_payloads().items())
    )
    monkeypatch.setattr(artifact_module, "_QWEN_LORA_LAYOUT", toy)
    monkeypatch.setattr(
        artifact_module,
        "_CANONICAL_PROCESSOR_FILES",
        {"2b": processor_records, "4b": processor_records},
    )

    # macOS exposes its pytest temp root through the /var -> /private/var
    # compatibility symlink. The publication contract intentionally rejects
    # linked path components, so pass the canonical temp root to the real
    # builder boundary.
    built = _artifact(tmp_path.resolve())
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "HfApi", _NoApi)
    monkeypatch.setattr(
        hub,
        "hf_hub_download",
        lambda **kwargs: str(built.root / kwargs["filename"]),
    )

    inspected = vlm_hub.inspect_vlm_hub_artifact(_SOURCE, local_files_only=True)

    assert isinstance(inspected, dict)
    assert isinstance(inspected["identity"], dict)
    assert isinstance(inspected["identity"]["base_snapshot"], dict)
    assert isinstance(inspected["identity"]["base_snapshot"]["files"], list)


def test_online_inspect_rejects_unmanifested_repository_files(tmp_path, monkeypatch):
    remote = _write_remote(tmp_path / "remote")
    calls = []
    _install_artifact_stubs(monkeypatch)
    hub = pytest.importorskip("huggingface_hub")

    class Api:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, revision=None, repo_type=None):
            assert repo_id == "alice/detector"
            assert revision == _REVISION
            return [
                vlm_hub.VLM_ARTIFACT_MANIFEST,
                *_contents(),
                "modeling_unmanifested.py",
            ]

    def download(**kwargs):
        target = remote.joinpath(*kwargs["filename"].split("/"))
        if kwargs.get("dry_run"):
            return SimpleNamespace(file_size=target.stat().st_size)
        calls.append(dict(kwargs))
        return str(target)

    monkeypatch.setattr(hub, "HfApi", Api)
    monkeypatch.setattr(hub, "hf_hub_download", download)

    with pytest.raises(ValueError, match="modeling_unmanifested.py"):
        vlm_hub.inspect_vlm_hub_artifact(_SOURCE)
    assert [call["filename"] for call in calls] == [vlm_hub.VLM_ARTIFACT_MANIFEST]


def test_download_fetches_manifest_then_sorted_inventory_without_hf_api(
    tmp_path, monkeypatch
):
    remote = _write_remote(tmp_path / "remote")
    calls = []
    _install_artifact_stubs(monkeypatch)
    _install_download_hub(monkeypatch, remote, calls)
    output = tmp_path / "artifact"

    downloaded = vlm_hub.download_vlm_artifact(_SOURCE, output, local_files_only=True)

    assert downloaded.root == output
    assert [call["filename"] for call in calls] == [
        vlm_hub.VLM_ARTIFACT_MANIFEST,
        ".gitattributes",
        "adapter/adapter_model.safetensors",
        "processor.json",
    ]
    assert all(call["revision"] == _REVISION for call in calls)
    assert all(call["local_files_only"] is True for call in calls)
    assert (output / "processor.json").read_bytes() == b"processor"


def test_download_refuses_existing_destination_before_hub_access(tmp_path, monkeypatch):
    output = tmp_path / "artifact"
    output.mkdir()
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(
        hub,
        "hf_hub_download",
        lambda **kwargs: pytest.fail("download must not start"),
    )

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        vlm_hub.download_vlm_artifact(_SOURCE, output)


def test_manifest_copy_caps_actual_bytes_if_source_grows_after_stat(
    tmp_path, monkeypatch
):
    source = tmp_path / "manifest.json"
    source.write_bytes(b"12345678")
    destination = tmp_path / "copy.json"
    real_stat = Path.stat

    def underreported_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == source:
            fields = list(result)
            fields[6] = 1  # st_size
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(Path, "stat", underreported_stat)

    with pytest.raises(ValueError, match="while reading"):
        vlm_hub._copy_regular_file(source, destination, max_bytes=4)
    assert not destination.exists()


def test_online_download_rejects_oversized_remote_metadata_before_fetch():
    calls = []

    class Hub:
        @staticmethod
        def hf_hub_download(**kwargs):
            calls.append(dict(kwargs))
            if kwargs.get("dry_run"):
                return SimpleNamespace(file_size=9)
            pytest.fail("oversized remote metadata reached the real download")

    with pytest.raises(ValueError, match="remote download limit"):
        vlm_hub._hub_download(
            Hub,
            vlm_hub.VLMHubRef("alice/detector", _REVISION),
            "processor.json",
            token=None,
            local_files_only=False,
            max_bytes=8,
            expected_bytes=8,
        )

    assert len(calls) == 1
    assert calls[0]["dry_run"] is True


def test_download_rejects_declared_limit_before_any_payload_fetch(
    tmp_path, monkeypatch
):
    remote = _write_remote(tmp_path / "remote")
    calls = []
    _install_download_hub(monkeypatch, remote, calls)
    monkeypatch.setattr(
        vlm_hub,
        "read_vlm_artifact_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("declared processor file exceeds limit")
        ),
    )

    with pytest.raises(ValueError, match="declared processor file"):
        vlm_hub.download_vlm_artifact(
            _SOURCE, tmp_path / "artifact", local_files_only=True
        )
    assert [call["filename"] for call in calls] == [vlm_hub.VLM_ARTIFACT_MANIFEST]


def test_download_stream_is_bounded_by_each_declared_file_size(tmp_path, monkeypatch):
    contents = _contents()
    contents[".gitattributes"] += b"x"
    remote = _write_remote(tmp_path / "remote", contents=contents)
    calls = []
    _install_artifact_stubs(monkeypatch)
    _install_download_hub(monkeypatch, remote, calls)
    output = tmp_path / "artifact"

    with pytest.raises(ValueError, match="50-byte safety limit"):
        vlm_hub.download_vlm_artifact(_SOURCE, output, local_files_only=True)
    assert [call["filename"] for call in calls] == [
        vlm_hub.VLM_ARTIFACT_MANIFEST,
        ".gitattributes",
    ]
    assert not output.exists()


def test_download_does_not_publish_tampered_payload(tmp_path, monkeypatch):
    remote = _write_remote(
        tmp_path / "remote",
        contents={**_contents(), "processor.json": b"tampered"},
    )
    calls = []
    _install_artifact_stubs(monkeypatch)
    _install_download_hub(monkeypatch, remote, calls)
    output = tmp_path / "artifact"

    with pytest.raises(ValueError, match="hash mismatch"):
        vlm_hub.download_vlm_artifact(_SOURCE, output, local_files_only=True)
    assert not output.exists()


def test_download_never_replaces_destination_created_during_publication(
    tmp_path, monkeypatch
):
    remote = _write_remote(tmp_path / "remote")
    calls = []
    _install_artifact_stubs(monkeypatch)
    _install_download_hub(monkeypatch, remote, calls)
    output = tmp_path / "artifact"
    real_rename = vlm_hub._atomic_rename_create_only

    def race(source, destination, **kwargs):
        destination.mkdir()
        (destination / "belongs-to-someone-else.txt").write_text(
            "keep", encoding="utf-8"
        )
        return real_rename(source, destination, **kwargs)

    monkeypatch.setattr(vlm_hub, "_atomic_rename_create_only", race)

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        vlm_hub.download_vlm_artifact(_SOURCE, output, local_files_only=True)
    assert (output / "belongs-to-someone-else.txt").read_text(
        encoding="utf-8"
    ) == "keep"
    assert not list(tmp_path.glob(".artifact.publishing-*"))


def test_download_rejects_replaced_valid_staging_directory_before_publication(
    tmp_path, monkeypatch
):
    remote = _write_remote(tmp_path / "remote")
    calls = []
    _install_artifact_stubs(monkeypatch)
    _install_download_hub(monkeypatch, remote, calls)
    output = tmp_path / "artifact"
    real_rename = vlm_hub._atomic_rename_create_only
    displaced = {}

    def replace_with_valid_copy(source, destination, **kwargs):
        original = source.with_name(source.name + "-original")
        source.rename(original)
        shutil.copytree(original, source)
        displaced["original"] = original
        return real_rename(source, destination, **kwargs)

    monkeypatch.setattr(
        vlm_hub,
        "_atomic_rename_create_only",
        replace_with_valid_copy,
    )

    with pytest.raises(ValueError, match="staging directory changed before rename"):
        vlm_hub.download_vlm_artifact(_SOURCE, output, local_files_only=True)
    assert not output.exists()
    assert displaced["original"].is_dir()


def test_download_failure_never_deletes_concurrent_file_after_publication(
    tmp_path, monkeypatch
):
    remote = _write_remote(tmp_path / "remote")
    calls = []
    _install_artifact_stubs(monkeypatch)
    _install_download_hub(monkeypatch, remote, calls)
    output = tmp_path / "artifact"
    real_rename = vlm_hub._atomic_rename_create_only

    def publish_then_add_file(source, destination, **kwargs):
        real_rename(source, destination, **kwargs)
        (destination / "concurrent-owner.txt").write_text("keep", encoding="utf-8")

    monkeypatch.setattr(
        vlm_hub,
        "_atomic_rename_create_only",
        publish_then_add_file,
    )

    with pytest.raises(AssertionError):
        vlm_hub.download_vlm_artifact(_SOURCE, output, local_files_only=True)
    assert (output / "concurrent-owner.txt").read_text(encoding="utf-8") == "keep"


def _base_identity():
    return {
        "schema": "libreyolo.vlm-base-snapshot.v1",
        "source": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "files": [
            {"path": "config.json", "size": 2, "sha256": "d" * 64},
            {
                "path": "model.safetensors",
                "size": 5,
                "sha256": "e" * 64,
            },
            {"path": "README.md", "size": 6, "sha256": "1" * 64},
        ],
        "aggregate_sha256": "f" * 64,
        "sha256": "0" * 64,
    }


def _base_contents():
    return {
        "config.json": b"{}",
        "model.safetensors": b"model",
        "README.md": b"notice",
    }


def _artifact_with_base(tmp_path, monkeypatch):
    root = _write_remote(tmp_path / "artifact")
    manifest = _manifest()
    raw_snapshot = _base_identity()
    manifest["identity"]["base_snapshot"] = raw_snapshot
    (root / vlm_hub.VLM_ARTIFACT_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    frozen_snapshot = MappingProxyType(
        {
            **raw_snapshot,
            "files": tuple(
                MappingProxyType(dict(entry)) for entry in raw_snapshot["files"]
            ),
        }
    )
    manifest["identity"]["base_snapshot"] = frozen_snapshot
    info = SimpleNamespace(
        root=root,
        manifest=manifest,
        aggregate_sha256=_AGGREGATE,
        files=tuple(entry["path"] for entry in manifest["files"]),
        base_snapshot=frozen_snapshot,
    )
    monkeypatch.setattr(vlm_hub, "validate_vlm_artifact", lambda path: info)
    return info


def _install_base_validator(monkeypatch, expected_contents=None):
    expected_contents = (
        _base_contents() if expected_contents is None else expected_contents
    )

    def validate(root, expected):
        root = Path(root)
        expected_names = {entry["path"] for entry in expected["files"]}
        actual_names = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != ".libreyolo_snapshot_complete"
        }
        if actual_names != expected_names:
            raise ValueError("base snapshot inventory mismatch")
        for name in expected_names:
            if root.joinpath(*name.split("/")).read_bytes() != expected_contents[name]:
                raise ValueError(f"base snapshot hash mismatch for {name}")
        return dict(expected)

    monkeypatch.setattr(vlm_hub, "validate_vlm_base_snapshot", validate)


def test_ensure_base_snapshot_downloads_exact_files_offline_and_publishes_atomically(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    info = _artifact_with_base(tmp_path, monkeypatch)
    _install_base_validator(monkeypatch)
    remote = tmp_path / "base-remote"
    _write_remote(
        remote,
        manifest=_manifest(),
        contents=_base_contents(),
    )
    calls = []
    hub = pytest.importorskip("huggingface_hub")

    def download(**kwargs):
        calls.append(dict(kwargs))
        return str(remote.joinpath(*kwargs["filename"].split("/")))

    monkeypatch.setattr(hub, "hf_hub_download", download)
    monkeypatch.setattr(hub, "HfApi", _NoApi)

    snapshot = vlm_hub.ensure_vlm_base_snapshot(
        info, token="hf_read_token", local_files_only=True
    )

    expected_root = tmp_path / "weights" / "LibreQwen3VL2b"
    assert snapshot.root == expected_root
    assert snapshot.identity == _base_identity()
    assert [call["filename"] for call in calls] == [
        "config.json",
        "model.safetensors",
        "README.md",
    ]
    assert all(call["repo_id"] == _base_identity()["source"] for call in calls)
    assert all(call["revision"] == _base_identity()["revision"] for call in calls)
    assert all(call["local_files_only"] is True for call in calls)
    marker = json.loads(
        (expected_root / ".libreyolo_snapshot_complete").read_text(encoding="utf-8")
    )
    assert marker == {
        "repo": _base_identity()["source"],
        "revision": _base_identity()["revision"],
    }
    assert not list((tmp_path / "weights").glob(".*.staging-*"))


def test_ensure_base_snapshot_accepts_valid_existing_root_without_hub(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    info = _artifact_with_base(tmp_path, monkeypatch)
    _install_base_validator(monkeypatch)
    root = tmp_path / "weights" / "LibreQwen3VL2b"
    root.mkdir(parents=True)
    for name, payload in _base_contents().items():
        (root / name).write_bytes(payload)
    (root / ".libreyolo_snapshot_complete").write_text(
        json.dumps(
            {
                "repo": _base_identity()["source"],
                "revision": _base_identity()["revision"],
            }
        ),
        encoding="utf-8",
    )
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "HfApi", _NoApi)
    monkeypatch.setattr(
        hub,
        "hf_hub_download",
        lambda **kwargs: pytest.fail("valid existing root must not download"),
    )

    snapshot = vlm_hub.ensure_vlm_base_snapshot(info, local_files_only=True)

    assert snapshot.root == root


def test_ensure_base_snapshot_refuses_invalid_existing_root_before_hub(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    info = _artifact_with_base(tmp_path, monkeypatch)
    _install_base_validator(monkeypatch)
    root = tmp_path / "weights" / "LibreQwen3VL2b"
    root.mkdir(parents=True)
    (root / "config.json").write_bytes(b"tampered")
    before = (root / "config.json").read_bytes()
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "HfApi", _NoApi)
    monkeypatch.setattr(
        hub,
        "hf_hub_download",
        lambda **kwargs: pytest.fail("invalid existing root must not download"),
    )

    with pytest.raises(ValueError, match="inventory mismatch"):
        vlm_hub.ensure_vlm_base_snapshot(info)
    assert (root / "config.json").read_bytes() == before


def test_ensure_base_snapshot_does_not_publish_tampered_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    info = _artifact_with_base(tmp_path, monkeypatch)
    _install_base_validator(monkeypatch)
    remote = tmp_path / "base-remote"
    _write_remote(
        remote,
        manifest=_manifest(),
        contents={**_base_contents(), "model.safetensors": b"wrong"},
    )
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "HfApi", _NoApi)

    def download(**kwargs):
        target = remote.joinpath(*kwargs["filename"].split("/"))
        if kwargs.get("dry_run"):
            return SimpleNamespace(file_size=target.stat().st_size)
        return str(target)

    monkeypatch.setattr(hub, "hf_hub_download", download)

    with pytest.raises(ValueError, match="base snapshot hash mismatch"):
        vlm_hub.ensure_vlm_base_snapshot(info)
    assert not (tmp_path / "weights" / "LibreQwen3VL2b").exists()


class RepositoryNotFoundError(Exception):
    """Minimal fake recognized by the transport's sanitized error boundary."""


class _OperationAdd:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


def _install_push_hub(
    monkeypatch,
    remote,
    events,
    *,
    identity=None,
    existing_files=None,
    revision=_REVISION,
    persist_commit=False,
    post_commit_extra=(),
    commit_conflict=False,
    initial_commit_ids=None,
):
    hub = pytest.importorskip("huggingface_hub")
    calls = {
        "created": [],
        "commits": [],
        "downloads": [],
        "tokens": [],
        "settings": [],
    }
    identity = {"name": "alice", "orgs": []} if identity is None else identity
    initial_commit_ids = (
        [_INITIAL_REVISION] if initial_commit_ids is None else list(initial_commit_ids)
    )
    state = {"created": False, "committed_files": None}

    class Api:
        def __init__(self, token=None):
            events.append("api_init")
            calls["tokens"].append(token)

        def whoami(self):
            events.append("whoami")
            if isinstance(identity, Exception):
                raise identity
            return identity

        def list_repo_files(self, repo_id, revision=None, repo_type=None):
            events.append("list_repo_files")
            if revision is not None and state["committed_files"] is not None:
                assert revision == _REVISION
                return [*state["committed_files"], *post_commit_extra]
            if state["created"]:
                return [".gitattributes"]
            if existing_files is None:
                raise RepositoryNotFoundError("missing")
            return list(existing_files)

        def create_repo(self, repo_id, **kwargs):
            events.append("create_repo")
            calls["created"].append((repo_id, kwargs))
            state["created"] = True

        def repo_info(self, repo_id, repo_type=None, files_metadata=False):
            events.append("repo_info")
            assert state["created"]
            return SimpleNamespace(sha=_INITIAL_REVISION)

        def list_repo_commits(self, repo_id, repo_type=None):
            events.append("list_repo_commits")
            assert state["created"]
            return [SimpleNamespace(commit_id=value) for value in initial_commit_ids]

        def update_repo_settings(self, repo_id, **kwargs):
            events.append("update_repo_settings")
            calls["settings"].append((repo_id, kwargs))

        def create_commit(self, **kwargs):
            events.append("create_commit")
            calls["commits"].append(kwargs)
            if commit_conflict:
                raise RuntimeError("branch changed after parent snapshot")
            state["committed_files"] = [
                operation.path_in_repo for operation in kwargs["operations"]
            ]
            if persist_commit:
                remote.mkdir(parents=True, exist_ok=True)
                for operation in kwargs["operations"]:
                    target = remote.joinpath(*operation.path_in_repo.split("/"))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    handle = operation.path_or_fileobj
                    position = handle.tell()
                    handle.seek(0)
                    target.write_bytes(handle.read())
                    handle.seek(position)
            return SimpleNamespace(oid=revision)

    def download(**kwargs):
        target = remote.joinpath(*kwargs["filename"].split("/"))
        if kwargs.get("dry_run"):
            return SimpleNamespace(file_size=target.stat().st_size)
        events.append("download")
        calls["downloads"].append(dict(kwargs))
        return str(target)

    monkeypatch.setattr(hub, "HfApi", Api)
    monkeypatch.setattr(hub, "CommitOperationAdd", _OperationAdd)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    return calls


def test_push_validates_then_authenticates_and_creates_one_sorted_commit(
    tmp_path, monkeypatch
):
    local = _write_remote(tmp_path / "local")
    events = []
    _install_artifact_stubs(monkeypatch, events=events)
    calls = _install_push_hub(monkeypatch, local, events)

    uri = vlm_hub.push_vlm_artifact(local, "alice/detector", token="hf_secret")

    assert uri == _SOURCE
    assert events[0:4] == ["validate", "validate", "api_init", "whoami"]
    assert len(calls["created"]) == 1
    repo_id, create_kwargs = calls["created"][0]
    assert repo_id == "alice/detector"
    assert create_kwargs == {
        "private": True,
        "exist_ok": False,
        "repo_type": "model",
    }
    assert len(calls["commits"]) == 1
    commit = calls["commits"][0]
    assert commit["repo_id"] == "alice/detector"
    assert commit["create_pr"] is False
    assert commit["parent_commit"] == _INITIAL_REVISION
    operation_names = [operation.path_in_repo for operation in commit["operations"]]
    assert operation_names == sorted(
        [vlm_hub.VLM_ARTIFACT_MANIFEST, *_contents().keys()], key=str.casefold
    )
    assert len(calls["downloads"]) == 4
    assert all(call["revision"] == _REVISION for call in calls["downloads"])
    assert all(call["force_download"] is True for call in calls["downloads"])
    assert all(call["local_files_only"] is False for call in calls["downloads"])
    assert all("cache_dir" in call for call in calls["downloads"])


def test_push_allows_authenticated_organization_namespace(tmp_path, monkeypatch):
    local = _write_remote(tmp_path / "local")
    events = []
    _install_artifact_stubs(monkeypatch, events=events)
    calls = _install_push_hub(
        monkeypatch,
        local,
        events,
        identity={"name": "alice", "orgs": [{"name": "libreyolo"}]},
    )

    uri = vlm_hub.push_vlm_artifact(local, "libreyolo/detector")

    assert uri == f"hf+vlm://libreyolo/detector@{_REVISION}"
    assert len(calls["commits"]) == 1


def test_public_push_stays_private_until_fresh_verification(tmp_path, monkeypatch):
    local = _write_remote(tmp_path / "local")
    events = []
    _install_artifact_stubs(monkeypatch, events=events)
    calls = _install_push_hub(monkeypatch, local, events)

    uri = vlm_hub.push_vlm_artifact(local, "alice/detector", private=False)

    assert uri == _SOURCE
    assert calls["created"][0][1]["private"] is True
    assert calls["settings"] == [
        ("alice/detector", {"private": False, "repo_type": "model"})
    ]
    assert events.index("update_repo_settings") > max(
        index for index, event in enumerate(events) if event == "download"
    )


def test_push_uses_isolated_validated_bytes_if_original_mutates(tmp_path, monkeypatch):
    local = _write_remote(tmp_path / "local")
    remote = tmp_path / "committed-remote"
    events = []
    _install_artifact_stubs(monkeypatch, events=events)
    calls = _install_push_hub(
        monkeypatch,
        remote,
        events,
        persist_commit=True,
    )
    hub = pytest.importorskip("huggingface_hub")
    mutated = False

    class MutatingOperation(_OperationAdd):
        def __init__(self, path_in_repo, path_or_fileobj):
            nonlocal mutated
            if not mutated:
                (local / "processor.json").write_bytes(b"tampered")
                mutated = True
            super().__init__(path_in_repo, path_or_fileobj)

    monkeypatch.setattr(hub, "CommitOperationAdd", MutatingOperation)

    uri = vlm_hub.push_vlm_artifact(local, "alice/detector")

    assert uri == _SOURCE
    assert (local / "processor.json").read_bytes() == b"tampered"
    assert (remote / "processor.json").read_bytes() == b"processor"
    assert len(calls["commits"]) == 1


def test_push_rejects_foreign_namespace_before_repository_access(tmp_path, monkeypatch):
    local = _write_remote(tmp_path / "local")
    events = []
    _install_artifact_stubs(monkeypatch, events=events)
    calls = _install_push_hub(
        monkeypatch,
        local,
        events,
        identity={"name": "mallory", "orgs": []},
    )

    with pytest.raises(PermissionError, match="authenticated user"):
        vlm_hub.push_vlm_artifact(local, "alice/detector")
    assert "list_repo_files" not in events
    assert calls["commits"] == []


def test_push_rejects_nonempty_repository_without_commit(tmp_path, monkeypatch):
    local = _write_remote(tmp_path / "local")
    events = []
    _install_artifact_stubs(monkeypatch, events=events)
    calls = _install_push_hub(
        monkeypatch,
        local,
        events,
        existing_files=["README.md"],
    )

    with pytest.raises(FileExistsError, match="existing repository"):
        vlm_hub.push_vlm_artifact(local, "alice/detector")
    assert calls["commits"] == []
    assert calls["downloads"] == []


def test_push_private_default_rejects_existing_empty_repository(tmp_path, monkeypatch):
    local = _write_remote(tmp_path / "local")
    events = []
    _install_artifact_stubs(monkeypatch, events=events)
    calls = _install_push_hub(
        monkeypatch,
        local,
        events,
        existing_files=[],
    )

    with pytest.raises(FileExistsError, match="existing repository"):
        vlm_hub.push_vlm_artifact(local, "alice/detector")
    assert calls["created"] == []
    assert calls["commits"] == []


def test_push_rejects_concurrent_initial_commit_before_artifact_commit(
    tmp_path, monkeypatch
):
    local = _write_remote(tmp_path / "local")
    events = []
    _install_artifact_stubs(monkeypatch, events=events)
    calls = _install_push_hub(
        monkeypatch,
        local,
        events,
        initial_commit_ids=["b" * 40, _INITIAL_REVISION],
    )

    with pytest.raises(FileExistsError, match="changed during creation"):
        vlm_hub.push_vlm_artifact(local, "alice/detector")
    assert calls["created"]
    assert calls["commits"] == []
    assert calls["downloads"] == []


def test_push_parent_commit_blocks_concurrent_first_writer(tmp_path, monkeypatch):
    local = _write_remote(tmp_path / "local")
    events = []
    _install_artifact_stubs(monkeypatch, events=events)
    calls = _install_push_hub(
        monkeypatch,
        local,
        events,
        commit_conflict=True,
    )

    with pytest.raises(PermissionError, match="Could not commit"):
        vlm_hub.push_vlm_artifact(local, "alice/detector", private=False)
    assert len(calls["commits"]) == 1
    assert calls["commits"][0]["parent_commit"] == _INITIAL_REVISION
    assert calls["downloads"] == []
    assert calls["settings"] == []


def test_push_invalid_artifact_never_constructs_hub_api(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "HfApi", _NoApi)
    monkeypatch.setattr(
        vlm_hub,
        "validate_vlm_artifact",
        lambda path: (_ for _ in ()).throw(ValueError("invalid artifact")),
    )

    with pytest.raises(ValueError, match="invalid artifact"):
        vlm_hub.push_vlm_artifact(tmp_path / "bad", "alice/detector")


def test_push_rejects_non_sha_commit_without_verification(tmp_path, monkeypatch):
    local = _write_remote(tmp_path / "local")
    events = []
    _install_artifact_stubs(monkeypatch, events=events)
    calls = _install_push_hub(
        monkeypatch,
        local,
        events,
        revision="main",
    )

    with pytest.raises(RuntimeError, match="40-character commit SHA"):
        vlm_hub.push_vlm_artifact(local, "alice/detector")
    assert len(calls["commits"]) == 1
    assert calls["downloads"] == []


def test_push_fresh_verification_rejects_unmanifested_commit_file(
    tmp_path, monkeypatch
):
    local = _write_remote(tmp_path / "local")
    events = []
    _install_artifact_stubs(monkeypatch, events=events)
    calls = _install_push_hub(
        monkeypatch,
        local,
        events,
        post_commit_extra=("unexpected.py",),
    )

    with pytest.raises(ValueError, match="unexpected.py"):
        vlm_hub.push_vlm_artifact(local, "alice/detector")
    assert len(calls["commits"]) == 1
    assert [call["filename"] for call in calls["downloads"]] == [
        vlm_hub.VLM_ARTIFACT_MANIFEST
    ]


def test_download_error_does_not_leak_explicit_token(tmp_path, monkeypatch):
    secret = "hf_top_secret_value"
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "HfApi", _NoApi)
    monkeypatch.setattr(
        hub,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )

    with pytest.raises(ConnectionError) as error:
        vlm_hub.download_vlm_artifact(
            _SOURCE, tmp_path / "artifact", token=secret, local_files_only=True
        )
    assert secret not in str(error.value)
    assert error.value.__cause__ is None


def test_authentication_error_does_not_leak_explicit_token(tmp_path, monkeypatch):
    secret = "hf_top_secret_value"
    local = _write_remote(tmp_path / "local")
    events = []
    _install_artifact_stubs(monkeypatch, events=events)
    _install_push_hub(
        monkeypatch,
        local,
        events,
        identity=RuntimeError(secret),
    )

    with pytest.raises(PermissionError) as error:
        vlm_hub.push_vlm_artifact(local, "alice/detector", token=secret)
    assert secret not in str(error.value)
    assert error.value.__cause__ is None
