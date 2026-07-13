"""Tests for crash-safe, verified model downloads."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
import requests

from libreyolo.models.base.model import BaseModel
from libreyolo.utils import download

pytestmark = pytest.mark.unit


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.headers = {"content-length": str(len(payload))}
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        del chunk_size
        yield self.payload

    def close(self):
        self.closed = True


def _install_family(monkeypatch, verifier):
    class Family:
        @classmethod
        def get_download_url(cls, filename):
            del cls, filename
            return "https://huggingface.co/LibreYOLO/test/resolve/main/model.pt"

        @classmethod
        def get_download_notice(cls, filename, url):
            del cls, filename, url
            return None

        @classmethod
        def verify_downloaded_file(cls, local_path, source_url):
            del cls
            verifier(Path(local_path), source_url)

    monkeypatch.setattr(BaseModel, "_registry", [Family])


def test_download_verifies_temp_before_atomic_promotion(tmp_path, monkeypatch):
    destination = tmp_path / "model.pt"
    seen = {}

    def verifier(partial, source_url):
        seen["partial"] = partial
        seen["url"] = source_url
        assert partial.read_bytes() == b"valid weights"
        assert not destination.exists()

    _install_family(monkeypatch, verifier)
    response = _Response(b"valid weights")

    def fake_get(url, **kwargs):
        seen["request_url"] = url
        seen["request_kwargs"] = kwargs
        return response

    monkeypatch.setattr(download.requests, "get", fake_get)

    download.download_weights(str(destination), "n")

    assert destination.read_bytes() == b"valid weights"
    assert seen["partial"] != destination
    assert seen["request_kwargs"]["timeout"] == download._DOWNLOAD_TIMEOUT
    assert seen["request_kwargs"]["stream"] is True
    assert response.closed is True
    assert list(tmp_path.glob("*.part")) == []
    assert list(tmp_path.glob("*.lock")) == []


def test_failed_verification_never_publishes_or_caches_file(tmp_path, monkeypatch):
    destination = tmp_path / "model.pt"
    calls = {"requests": 0, "verifier": 0}

    def verifier(partial, source_url):
        del partial, source_url
        calls["verifier"] += 1
        raise RuntimeError("checksum mismatch")

    _install_family(monkeypatch, verifier)

    def fake_get(url, **kwargs):
        del url, kwargs
        calls["requests"] += 1
        return _Response(b"corrupt")

    monkeypatch.setattr(download.requests, "get", fake_get)
    monkeypatch.setattr(download.time, "sleep", lambda delay: None)

    for _ in range(2):
        with pytest.raises(download.WeightVerificationError, match="verification"):
            download.download_weights(str(destination), "n")
        assert not destination.exists()

    assert calls == {
        "requests": 2 * download._DOWNLOAD_ATTEMPTS,
        "verifier": 2 * download._DOWNLOAD_ATTEMPTS,
    }
    assert list(tmp_path.iterdir()) == []


def test_transient_request_failure_is_retried(tmp_path, monkeypatch):
    destination = tmp_path / "model.pt"
    calls = 0
    delays = []

    _install_family(monkeypatch, lambda partial, url: None)

    def fake_get(url, **kwargs):
        nonlocal calls
        del url, kwargs
        calls += 1
        if calls == 1:
            raise requests.Timeout("timed out")
        return _Response(b"weights")

    monkeypatch.setattr(download.requests, "get", fake_get)
    monkeypatch.setattr(download.time, "sleep", delays.append)

    download.download_weights(str(destination), "n")

    assert destination.read_bytes() == b"weights"
    assert calls == 2
    assert delays == [1]


def test_exhausted_transfer_raises_typed_error(tmp_path, monkeypatch):
    destination = tmp_path / "model.pt"
    _install_family(monkeypatch, lambda partial, url: None)

    def fail_get(url, **kwargs):
        del url, kwargs
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(download.requests, "get", fail_get)
    monkeypatch.setattr(download.time, "sleep", lambda delay: None)

    with pytest.raises(download.WeightDownloadError, match="read timed out") as error:
        download.download_weights(str(destination), "n")

    assert not isinstance(error.value, download.WeightVerificationError)
    assert not destination.exists()


def test_existing_user_path_is_not_verified_deleted_or_downloaded(
    tmp_path, monkeypatch
):
    destination = tmp_path / "LibreYOLONASs.pt"
    original = b"user supplied checkpoint"
    destination.write_bytes(original)
    calls = {"requests": 0, "verifier": 0}

    def verifier(path, url):
        del path, url
        calls["verifier"] += 1
        raise RuntimeError("verifier must not inspect an existing path")

    _install_family(monkeypatch, verifier)

    def fake_get(url, **kwargs):
        del url, kwargs
        calls["requests"] += 1
        raise AssertionError("network must not be used for an existing path")

    monkeypatch.setattr(download.requests, "get", fake_get)

    download.download_weights(str(destination), "n")

    assert destination.read_bytes() == original
    assert calls == {"requests": 0, "verifier": 0}
    assert list(tmp_path.glob("*.part")) == []
    assert list(tmp_path.glob("*.lock")) == []


@pytest.mark.parametrize(
    ("source_url", "message"),
    [
        (
            "https://sghub.deci.ai/models/yolo_nas_s_coco.pth",
            "Checksum mismatch",
        ),
        ("https://sghub.deci.ai/models/unpinned.pth", "no pinned checksum"),
    ],
)
def test_yolonas_verifier_failure_does_not_delete_input(tmp_path, source_url, message):
    from libreyolo.models.yolonas.model import LibreYOLONAS

    candidate = tmp_path / "candidate.pth"
    original = b"not official weights"
    candidate.write_bytes(original)

    with pytest.raises(RuntimeError, match=message):
        LibreYOLONAS.verify_downloaded_file(str(candidate), source_url)

    assert candidate.read_bytes() == original


def test_file_created_during_verification_wins_without_replacement(
    tmp_path, monkeypatch
):
    destination = tmp_path / "model.pt"
    user_bytes = b"late user checkpoint"
    requests_made = 0

    def verifier(partial, source_url):
        del partial, source_url
        destination.write_bytes(user_bytes)

    _install_family(monkeypatch, verifier)

    def fake_get(url, **kwargs):
        nonlocal requests_made
        del url, kwargs
        requests_made += 1
        return _Response(b"verified downloaded checkpoint")

    monkeypatch.setattr(download.requests, "get", fake_get)

    download.download_weights(str(destination), "n")

    assert destination.read_bytes() == user_bytes
    assert requests_made == 1
    assert list(tmp_path.glob("*.part")) == []
    assert list(tmp_path.glob("*.lock")) == []


def test_dangling_symlink_created_during_verification_is_preserved(
    tmp_path, monkeypatch
):
    destination = tmp_path / "model.pt"
    missing_target = tmp_path / "missing-user-checkpoint.pt"
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(missing_target)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    else:
        probe.unlink()

    def verifier(partial, source_url):
        del partial, source_url
        destination.symlink_to(missing_target)

    _install_family(monkeypatch, verifier)
    monkeypatch.setattr(
        download.requests,
        "get",
        lambda url, **kwargs: _Response(b"verified downloaded checkpoint"),
    )

    download.download_weights(str(destination), "n")

    assert destination.is_symlink()
    actual_target = os.path.normcase(os.path.normpath(str(destination.readlink())))
    expected_target = os.path.normcase(os.path.normpath(str(missing_target)))
    if os.name == "nt":
        # pathlib may expose Windows symlink targets with the equivalent
        # extended-length ``\\?\`` prefix on hosted runners.
        actual_target = actual_target.removeprefix("\\\\?\\")
        expected_target = expected_target.removeprefix("\\\\?\\")
    assert actual_target == expected_target
    assert os.path.lexists(destination)
    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == []
    assert list(tmp_path.glob("*.lock")) == []


def test_concurrent_callers_share_one_completed_download(tmp_path, monkeypatch):
    destination = tmp_path / "model.pt"
    request_started = Event()
    allow_response = Event()
    calls = 0

    _install_family(monkeypatch, lambda partial, url: None)

    def fake_get(url, **kwargs):
        nonlocal calls
        del url, kwargs
        calls += 1
        request_started.set()
        assert allow_response.wait(timeout=5)
        return _Response(b"weights")

    monkeypatch.setattr(download.requests, "get", fake_get)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(download.download_weights, str(destination), "n")
        assert request_started.wait(timeout=5)
        second = pool.submit(download.download_weights, str(destination), "n")
        allow_response.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert calls == 1
    assert destination.read_bytes() == b"weights"


def test_download_lock_timeout_is_typed(tmp_path, monkeypatch):
    destination = tmp_path / "model.pt"
    destination.with_name("model.pt.lock").write_text("123", encoding="ascii")
    monkeypatch.setattr(download, "_DOWNLOAD_LOCK_TIMEOUT_SECONDS", 0)

    with pytest.raises(download.WeightDownloadLockTimeout):
        with download._download_lock(destination):
            pytest.fail("lock unexpectedly acquired")


@pytest.mark.parametrize(
    ("filename", "size", "status"),
    [
        ("LibreYOLO1t.pt", "t", "unknown"),
        ("LibreDepthAnythingV2b-depth.pt", "b", "config_only"),
        ("LibreL2CSr50.pt", "r50", "direct"),
        ("LibreYOLO9P2s-visdrone.pt", "s", "gated"),
    ],
)
def test_known_unpublished_artifact_fails_before_network(
    tmp_path, monkeypatch, filename, size, status
):
    def unexpected_request(*args, **kwargs):
        del args, kwargs
        pytest.fail("unpublished artifact must fail before a network request")

    monkeypatch.setattr(download.requests, "get", unexpected_request)

    with pytest.raises(download.WeightPublicationError, match=status):
        download.download_weights(str(tmp_path / filename), size)


@pytest.mark.parametrize(
    ("filename", "size"),
    [
        ("LibreYOLOXs.pt", "s"),
        ("LibreRFDETRn.pt", "n"),
    ],
)
def test_canonical_download_uses_manifest_route_not_registry(
    tmp_path, monkeypatch, filename, size
):
    from libreyolo.models import manifest

    destination = tmp_path / filename
    seen = {}

    class ManifestFamily:
        @classmethod
        def get_download_notice(cls, filename, url):
            del cls, filename, url
            return None

        @classmethod
        def verify_downloaded_file(cls, local_path, source_url):
            del cls
            seen["verified"] = (Path(local_path), source_url)

    class RegistryTrap:
        @classmethod
        def get_download_url(cls, filename):
            del cls, filename
            pytest.fail("canonical routing must not inspect the mutable registry")

    monkeypatch.setattr(BaseModel, "_registry", [RegistryTrap])
    monkeypatch.setattr(manifest, "load_family_class", lambda family: ManifestFamily)

    def fake_get(url, **kwargs):
        seen["url"] = url
        seen["kwargs"] = kwargs
        return _Response(b"weights")

    monkeypatch.setattr(download.requests, "get", fake_get)

    download.download_weights(str(destination), size)

    expected = (
        f"https://huggingface.co/LibreYOLO/{destination.stem}/resolve/main/{filename}"
    )
    assert seen["url"] == expected
    assert seen["verified"][1] == expected
    assert destination.read_bytes() == b"weights"
