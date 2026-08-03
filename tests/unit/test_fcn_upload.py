"""Failure-path tests for FCN Hugging Face publication."""

from __future__ import annotations

import sys
from pathlib import Path

import huggingface_hub
import pytest

pytestmark = pytest.mark.unit

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
if str(WEIGHTS_DIR) not in sys.path:
    sys.path.insert(0, str(WEIGHTS_DIR))

from upload_fcn_hf import _upload  # noqa: E402


class _FakeApi:
    def __init__(self, *, exists: bool, remote_files=()):
        self.exists = exists
        self.remote_files = list(remote_files)
        self.created = []
        self.uploaded = []
        self.collection_items = []

    def repo_exists(self, **kwargs):
        return self.exists

    def list_repo_files(self, **kwargs):
        return self.remote_files

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)

    def add_collection_item(self, **kwargs):
        self.collection_items.append(kwargs)


def _repo_dir(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "LibreFCNr50"
    repo_dir.mkdir()
    for name in (
        ".gitattributes",
        "README.md",
        "LICENSE",
        "NOTICE",
        "LibreFCNr50.pt",
    ):
        (repo_dir / name).write_bytes(b"test")
    return repo_dir


def _install_api(monkeypatch, api: _FakeApi) -> None:
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)


def test_upload_creates_new_repository(monkeypatch, tmp_path):
    api = _FakeApi(exists=False)
    _install_api(monkeypatch, api)

    url = _upload("r50", _repo_dir(tmp_path))

    assert url == "https://huggingface.co/LibreYOLO/LibreFCNr50"
    assert len(api.created) == 1
    assert len(api.uploaded) == 1
    assert len(api.collection_items) == 1


def test_upload_refuses_existing_repository_without_resume(monkeypatch, tmp_path):
    api = _FakeApi(exists=True, remote_files={"README.md"})
    _install_api(monkeypatch, api)

    with pytest.raises(FileExistsError, match="--resume-partial"):
        _upload("r50", _repo_dir(tmp_path))

    assert not api.uploaded
    assert not api.collection_items


def test_upload_resumes_expected_partial_repository(monkeypatch, tmp_path):
    api = _FakeApi(exists=True, remote_files={".gitattributes", "README.md"})
    _install_api(monkeypatch, api)

    _upload("r50", _repo_dir(tmp_path), resume_partial=True)

    assert not api.created
    assert len(api.uploaded) == 1
    assert api.uploaded[0]["commit_message"].startswith("Resume upload:")
    assert len(api.collection_items) == 1


def test_upload_finishes_collection_enrollment_after_complete_upload(
    monkeypatch, tmp_path
):
    api = _FakeApi(
        exists=True,
        remote_files={
            ".gitattributes",
            "README.md",
            "LICENSE",
            "NOTICE",
            "LibreFCNr50.pt",
        },
    )
    _install_api(monkeypatch, api)

    _upload("r50", _repo_dir(tmp_path), resume_partial=True)

    assert not api.created
    assert not api.uploaded
    assert len(api.collection_items) == 1


def test_upload_refuses_unexpected_remote_files(monkeypatch, tmp_path):
    api = _FakeApi(exists=True, remote_files={"README.md", "unrelated.bin"})
    _install_api(monkeypatch, api)

    with pytest.raises(FileExistsError, match="unexpected remote files"):
        _upload("r50", _repo_dir(tmp_path), resume_partial=True)

    assert not api.uploaded
    assert not api.collection_items
