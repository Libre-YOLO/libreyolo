"""Tests for retrying and resumable weight downloads."""

import requests
import pytest

from libreyolo.models.base.model import BaseModel
from libreyolo.utils import download

pytestmark = pytest.mark.unit


class _DownloadFamily:
    @classmethod
    def get_download_url(cls, _filename):
        return "https://huggingface.co/LibreYOLO/test/resolve/model.pt"

    @classmethod
    def get_download_notice(cls, _filename, _url):
        return None

    @classmethod
    def verify_downloaded_file(cls, _path, _url):
        return None


class _Response:
    def __init__(self, chunks, *, status_code=200, headers=None):
        self._chunks = chunks
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        assert chunk_size == download._DOWNLOAD_CHUNK_SIZE
        for chunk in self._chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


def _prepare(monkeypatch):
    monkeypatch.setattr(BaseModel, "_registry", [_DownloadFamily])
    monkeypatch.setattr(download, "_get_hf_token", lambda: None)
    monkeypatch.setattr(download.time, "sleep", lambda _seconds: None)


def test_interrupted_download_retries_from_partial(monkeypatch, tmp_path):
    _prepare(monkeypatch)
    responses = [
        _Response(
            [b"abcd", requests.ConnectionError("connection dropped")],
            headers={"content-length": "10"},
        ),
        _Response(
            [b"efghij"],
            status_code=206,
            headers={"content-length": "6", "content-range": "bytes 4-9/10"},
        ),
    ]
    calls = []

    def fake_get(url, *, stream, headers, timeout):
        calls.append(
            {"url": url, "stream": stream, "headers": headers, "timeout": timeout}
        )
        return responses.pop(0)

    monkeypatch.setattr(download.requests, "get", fake_get)
    target = tmp_path / "model.pt"

    download.download_weights(str(target), "s")

    assert target.read_bytes() == b"abcdefghij"
    assert calls[0]["headers"].get("Range") is None
    assert calls[1]["headers"]["Range"] == "bytes=4-"
    assert calls[1]["timeout"] == download._DOWNLOAD_TIMEOUT
    assert not target.with_name("model.pt.part").exists()


def test_server_ignoring_range_restarts_partial(monkeypatch, tmp_path):
    _prepare(monkeypatch)
    target = tmp_path / "model.pt"
    partial = target.with_name("model.pt.part")
    partial.write_bytes(b"stale")
    response = _Response([b"complete"], headers={"content-length": "8"})

    monkeypatch.setattr(
        download.requests,
        "get",
        lambda _url, **_kwargs: response,
    )

    download.download_weights(str(target), "s")

    assert target.read_bytes() == b"complete"


def test_complete_partial_is_finalized_after_range_416(monkeypatch, tmp_path):
    _prepare(monkeypatch)
    target = tmp_path / "model.pt"
    partial = target.with_name("model.pt.part")
    partial.write_bytes(b"complete")
    response = _Response(
        [],
        status_code=416,
        headers={"content-range": "bytes */8"},
    )

    monkeypatch.setattr(
        download.requests,
        "get",
        lambda _url, **_kwargs: response,
    )

    download.download_weights(str(target), "s")

    assert target.read_bytes() == b"complete"


def test_exhausted_download_keeps_partial(monkeypatch, tmp_path):
    _prepare(monkeypatch)
    monkeypatch.setattr(download, "_DOWNLOAD_RETRIES", 0)
    response = _Response(
        [b"partial", requests.ConnectionError("still offline")],
        headers={"content-length": "20"},
    )
    monkeypatch.setattr(
        download.requests,
        "get",
        lambda _url, **_kwargs: response,
    )
    target = tmp_path / "model.pt"

    with pytest.raises(RuntimeError, match="Partial download kept"):
        download.download_weights(str(target), "s")

    assert target.with_name("model.pt.part").read_bytes() == b"partial"


def test_factory_reports_and_chains_download_failure(monkeypatch, tmp_path):
    import libreyolo.models as models

    failure = RuntimeError("connection reset while fetching checkpoint")

    def fail_download(_model_path, _size):
        raise failure

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(models, "download_weights", fail_download)

    with pytest.raises(
        FileNotFoundError,
        match="Auto-download failed: connection reset while fetching checkpoint",
    ) as exc_info:
        models.LibreYOLO("LibreYOLO9t.pt")

    assert exc_info.value.__cause__ is failure
