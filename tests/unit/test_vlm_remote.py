"""Unit tests for the remote LibreVLM transport (offline; SDK boundary faked)."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from libreyolo import LibreVLM
from libreyolo.models.vlm import remote as remote_mod
from libreyolo.models.vlm.remote import RemoteVLMModel

pytestmark = [pytest.mark.unit, pytest.mark.vlm]


class FakeOpenAIError(Exception):
    pass


class FakeRateLimitError(FakeOpenAIError):
    pass


class FakeConnectionError(FakeOpenAIError):
    pass


class FakeTimeoutError(FakeOpenAIError):
    pass


class FakeServerError(FakeOpenAIError):
    pass


class FakeAuthError(FakeOpenAIError):
    pass


def _chat_response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _responses_response(text):
    return SimpleNamespace(output_text=text)


class FakeChatCompletions:
    """Returns scripted replies in call order; an Exception entry is raised.

    The last entry repeats once the script is exhausted, so concurrency tests
    can serve any number of requests.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def create(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
            entry = self.script.pop(0) if len(self.script) > 1 else self.script[0]
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if kwargs.get("_sleep") or getattr(self, "sleep", 0):
                time.sleep(getattr(self, "sleep", 0))
            if isinstance(entry, Exception):
                raise entry
            return _chat_response(entry)
        finally:
            with self._lock:
                self.active -= 1


class FakeResponsesAPI:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        entry = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(entry, Exception):
            raise entry
        return _responses_response(entry)


class FakeClient:
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(script))
        self.responses = FakeResponsesAPI(script)


def _fake_openai(client):
    return SimpleNamespace(
        OpenAI=lambda **kwargs: client,
        RateLimitError=FakeRateLimitError,
        APIConnectionError=FakeConnectionError,
        APITimeoutError=FakeTimeoutError,
        InternalServerError=FakeServerError,
        AuthenticationError=FakeAuthError,
    )


@pytest.fixture
def make_remote(monkeypatch):
    """Build a remote model whose SDK boundary is a scripted fake."""

    def factory(script, model="openai/gpt-test", **kwargs):
        client = FakeClient(script)
        monkeypatch.setattr(remote_mod, "_load_openai", lambda: _fake_openai(client))
        return LibreVLM(model, **kwargs), client

    return factory


@pytest.fixture
def image_100x50(tmp_path):
    path = tmp_path / "img.jpg"
    Image.new("RGB", (100, 50), color=(10, 120, 200)).save(path, format="JPEG")
    return path


@pytest.fixture
def image_dir(tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(3):
        Image.new("RGB", (100, 50), color=(i * 40, 0, 0)).save(
            folder / f"frame_{i}.jpg", format="JPEG"
        )
    return folder


BOAT_JSON = '[{"label": "boat", "bbox": [0.1, 0.2, 0.5, 0.8]}]'


# =============================================================================
# Factory routing
# =============================================================================


def test_slash_routes_to_remote(make_remote):
    model, _ = make_remote([BOAT_JSON])
    assert isinstance(model, RemoteVLMModel)
    assert model.provider == "openai"
    assert model.model_id == "gpt-test"


def test_model_id_case_preserved(make_remote):
    model, _ = make_remote([BOAT_JSON], model="openai/GPT-Test-V2")
    assert model.model_id == "GPT-Test-V2"


def test_nested_slug_kept_verbatim(make_remote, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    model, _ = make_remote([BOAT_JSON], model="openrouter/qwen/qwen3-max")
    assert model.provider == "openrouter"
    assert model.model_id == "qwen/qwen3-max"
    assert model.base_url == "https://openrouter.ai/api/v1"


def test_kwarg_provider_form(make_remote):
    model, _ = make_remote([BOAT_JSON], model="gpt-test", provider="openai")
    assert isinstance(model, RemoteVLMModel)
    assert model.model_id == "gpt-test"


def test_unknown_provider_error_covers_all_intents():
    with pytest.raises(ValueError) as excinfo:
        LibreVLM("Qwen/Qwen3-VL-4B")
    message = str(excinfo.value)
    assert "Unknown remote provider 'Qwen'" in message
    assert "openai, openrouter, openai-compat" in message
    assert "Hugging Face repo ids are not accepted" in message
    assert "qwen3-vl-4b" in message
    assert "checkpoint path" in message


def test_windows_drive_path_is_never_a_provider():
    with pytest.raises(FileNotFoundError, match="No VLM checkpoint"):
        LibreVLM(r"C:\nonexistent\finetune_dir")


def test_relative_path_shape_is_never_a_provider():
    with pytest.raises(FileNotFoundError, match="No VLM checkpoint"):
        LibreVLM("./nonexistent_dir")


def test_existing_non_checkpoint_path_raises(tmp_path):
    with pytest.raises(ValueError, match="not a VLM fine-tune checkpoint"):
        LibreVLM(str(tmp_path))


def test_local_alias_rejects_remote_kwargs():
    with pytest.raises(ValueError, match="only applies to remote"):
        LibreVLM("qwen3-vl-4b", api_key="sk-x")


def test_openai_compat_requires_base_url(make_remote):
    with pytest.raises(ValueError, match="base_url="):
        make_remote([BOAT_JSON], model="openai-compat/qwen3-vl-32b")


def test_openai_compat_with_base_url(make_remote):
    model, _ = make_remote(
        [BOAT_JSON],
        model="openai-compat/qwen3-vl-32b",
        base_url="http://localhost:8000/v1",
        api_key="empty",
    )
    assert model.base_url == "http://localhost:8000/v1"


def test_openrouter_without_env_key_raises(make_remote, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        make_remote([BOAT_JSON], model="openrouter/qwen/qwen3-max")


def test_device_kwarg_raises(make_remote):
    with pytest.raises(ValueError, match="provider's servers"):
        make_remote([BOAT_JSON], device="cuda")


def test_unexpected_kwarg_raises(make_remote):
    with pytest.raises(TypeError, match="Unexpected keyword"):
        make_remote([BOAT_JSON], tiling=True)


# =============================================================================
# Detection contract
# =============================================================================


def test_predict_returns_pixel_xyxy(make_remote, image_100x50):
    model, client = make_remote([BOAT_JSON], names=["boat"])
    result = model.predict(str(image_100x50))
    xyxy = result.boxes.xyxy[0].tolist()
    assert xyxy == pytest.approx([10.0, 10.0, 50.0, 40.0])
    assert getattr(result, "remote", None) is None
    # chat.completions is the remote default wire API
    assert client.chat.completions.calls
    assert not client.responses.calls


def test_set_classes_is_sticky_and_in_prompt(make_remote, image_100x50):
    model, client = make_remote([BOAT_JSON])
    model.set_classes(["boat", "pink car"])
    model.predict(str(image_100x50))
    text = client.chat.completions.calls[0]["messages"][0]["content"][0]["text"]
    assert "boat" in text and "pink car" in text


def test_out_of_vocabulary_labels_dropped(make_remote, image_100x50):
    model, _ = make_remote(
        ['[{"label": "zebra", "bbox": [0.1, 0.2, 0.5, 0.8]}]'], names=["boat"]
    )
    result = model.predict(str(image_100x50))
    assert len(result.boxes.xyxy) == 0
    assert getattr(result, "remote", None) is None  # clean: parsed fine


def test_bbox_2d_1000_scale_rescaled(make_remote, image_100x50):
    model, _ = make_remote(
        ['[{"label": "boat", "bbox_2d": [100, 200, 500, 800]}]'], names=["boat"]
    )
    result = model.predict(str(image_100x50))
    assert result.boxes.xyxy[0].tolist() == pytest.approx([10.0, 10.0, 50.0, 40.0])


def test_responses_api_optin(make_remote, image_100x50):
    model, client = make_remote([BOAT_JSON], api="responses", names=["boat"])
    result = model.predict(str(image_100x50))
    assert len(result.boxes.xyxy) == 1
    assert client.responses.calls
    assert not client.chat.completions.calls


def test_max_new_tokens_sent_as_cap(make_remote, image_100x50):
    model, client = make_remote([BOAT_JSON], names=["boat"], max_new_tokens=256)
    model.predict(str(image_100x50))
    assert client.chat.completions.calls[0]["max_completion_tokens"] == 256


# =============================================================================
# Errors: empty is never ambiguous
# =============================================================================


def test_parse_failure_attaches_side_channel(make_remote, image_100x50):
    model, _ = make_remote(["There are two boats near the dock."], names=["boat"])
    result = model.predict(str(image_100x50))
    assert len(result.boxes.xyxy) == 0
    assert result.remote["error"] == "parse"
    assert result.remote["model"] == "openai/gpt-test"


def test_refusal_attaches_side_channel(make_remote, image_100x50):
    model, _ = make_remote(
        ["I'm sorry, I can't help with identifying that."], names=["boat"]
    )
    result = model.predict(str(image_100x50))
    assert result.remote["error"] == "refusal"


def test_clean_empty_has_no_side_channel(make_remote, image_100x50):
    model, _ = make_remote(["[]"], names=["boat"])
    result = model.predict(str(image_100x50))
    assert len(result.boxes.xyxy) == 0
    assert getattr(result, "remote", None) is None


def test_transient_http_error_becomes_empty_result(make_remote, image_100x50):
    model, _ = make_remote([FakeRateLimitError("429")], names=["boat"])
    result = model.predict(str(image_100x50))
    assert len(result.boxes.xyxy) == 0
    assert result.remote["error"] == "http"
    assert "FakeRateLimitError" in result.remote["detail"]


def test_auth_error_raises_loud(make_remote, image_100x50):
    model, _ = make_remote([FakeAuthError("bad key")], names=["boat"])
    with pytest.raises(FakeAuthError):
        model.predict(str(image_100x50))


def test_folder_run_isolates_failures_and_summarizes(
    make_remote, image_dir, caplog
):
    script = [BOAT_JSON, FakeRateLimitError("429"), BOAT_JSON]
    model, _ = make_remote(script, names=["boat"])
    with caplog.at_level(logging.WARNING):
        results = model.predict(str(image_dir), batch=1)
    assert len(results) == 3
    errors = [getattr(r, "remote", None) for r in results]
    assert errors[0] is None and errors[2] is None
    assert errors[1]["error"] == "http"
    assert any("2 ok, 1 failed" in rec.getMessage() for rec in caplog.records)


# =============================================================================
# Cost safety: guards, banner, concurrency
# =============================================================================


@pytest.mark.parametrize("source", [0, "screen", "rtsp://cam.local/stream"])
def test_live_sources_raise(make_remote, source):
    model, _ = make_remote([BOAT_JSON])
    with pytest.raises(ValueError, match="metered API call"):
        model.predict(source)


def test_track_live_source_raises(make_remote):
    model, _ = make_remote([BOAT_JSON])
    with pytest.raises(ValueError, match="metered API call"):
        model.track(0)


def test_metered_banner_before_folder(make_remote, image_dir, caplog):
    model, _ = make_remote([BOAT_JSON], names=["boat"])
    with caplog.at_level(logging.WARNING):
        model.predict(str(image_dir), batch=1)
    banner = [r for r in caplog.records if "paid/metered" in r.getMessage()]
    assert banner
    assert "3 images" in banner[0].getMessage()


def test_folder_requests_run_concurrently(make_remote, image_dir):
    model, client = make_remote([BOAT_JSON], names=["boat"])
    client.chat.completions.sleep = 0.05
    results = model.predict(str(image_dir))  # default concurrency 8
    assert len(results) == 3
    assert client.chat.completions.max_active >= 2


def test_folder_stream_yields_results(make_remote, image_dir):
    model, _ = make_remote([BOAT_JSON], names=["boat"])
    results = list(model.predict(str(image_dir), stream=True))
    assert len(results) == 3


def test_predict_tiling_and_augment_raise(make_remote, image_100x50):
    model, _ = make_remote([BOAT_JSON])
    with pytest.raises(ValueError, match="remote generator"):
        model.predict(str(image_100x50), tiling=True)
    with pytest.raises(ValueError, match="remote generator"):
        model.predict(str(image_100x50), augment=True)


def test_predict_device_raises(make_remote, image_100x50):
    model, _ = make_remote([BOAT_JSON])
    with pytest.raises(ValueError, match="provider's servers"):
        model.predict(str(image_100x50), device="cuda")


# =============================================================================
# chat() and CV verbs
# =============================================================================


def test_chat_returns_plain_string(make_remote, image_100x50):
    model, client = make_remote(["Two boats, one pink."])
    answer = model.chat(str(image_100x50), "How many boats?")
    assert answer == "Two boats, one pink."
    text = client.chat.completions.calls[0]["messages"][0]["content"][0]["text"]
    assert text == "How many boats?"


@pytest.mark.parametrize("verb", ["train", "val", "export"])
def test_cv_verbs_raise_inference_only(make_remote, verb):
    model, _ = make_remote([BOAT_JSON])
    with pytest.raises(NotImplementedError, match="inference-only"):
        getattr(model, verb)()


# =============================================================================
# selftest()
# =============================================================================


def test_selftest_passes_on_grounded_model(make_remote):
    script = [
        '[{"label": "red rectangle", "bbox": [0.2, 0.2, 0.8, 0.8]}]',
        "[]",
    ]
    model, _ = make_remote(script)
    report = model.selftest()
    assert report["passed"] is True
    assert report["iou"] == pytest.approx(1.0)
    assert report["false_positives"] == 0
    assert len(report["raw"]) == 2


def test_selftest_fails_on_ungrounded_model(make_remote):
    model, _ = make_remote(["I see a beautiful red shape in this image."])
    report = model.selftest()
    assert report["passed"] is False
    assert report["iou"] == 0.0


def test_selftest_fails_on_blank_false_positive(make_remote):
    script = [
        '[{"label": "red rectangle", "bbox": [0.2, 0.2, 0.8, 0.8]}]',
        '[{"label": "red rectangle", "bbox": [0.4, 0.4, 0.6, 0.6]}]',
    ]
    model, _ = make_remote(script)
    report = model.selftest()
    assert report["passed"] is False
    assert report["false_positives"] == 1


def test_selftest_does_not_touch_vocabulary(make_remote):
    model, _ = make_remote(["[]"], names=["helmet"])
    model.selftest()
    assert list(model.names.values()) == ["helmet"]
