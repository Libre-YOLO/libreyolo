"""The Hub kernel pin must stay fetchable, and must fail loudly when it isn't.

``_HUB_REVISION`` pins the compiled multi-scale deformable attention kernel to
an audited commit so a moved Hub branch cannot swap the native binary that
runs in-process. A pin the loader cannot fetch costs every user the
accelerated path, on every platform, behind a single log line inside a
fallback that is *supposed* to be quiet -- which is exactly how an unusable
pin reached ``dev`` unnoticed.

Two layers here:

* the classification and log level, testable offline everywhere;
* a network canary that resolves the live pin, which is the check that would
  have caught it.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

import pytest

from libreyolo.kernels.attention import ms_deform_attn as module

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_provider_state():
    """The loader caches its verdict in module globals; restore them."""
    saved = (module._hub_kernel, module._hub_failed, module._hub_failure_kind)
    module._hub_kernel, module._hub_failed, module._hub_failure_kind = None, False, None
    yield
    module._hub_kernel, module._hub_failed, module._hub_failure_kind = saved


# --- classification -------------------------------------------------------


class _RevisionNotFoundError(Exception):
    """Stands in for huggingface_hub's error without importing it."""


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_RevisionNotFoundError("404 Client Error. Invalid rev id: abc123"), "unresolvable"),
        (Exception("Revision Not Found for url: .../tree/abc123/build"), "unresolvable"),
        (FileNotFoundError("Cannot find a build variant for this system"), "unsupported"),
        (ImportError("No module named 'kernels'"), "unsupported"),
        (RuntimeError("CUDA driver version is insufficient"), "unsupported"),
    ],
)
def test_load_failures_are_classified(exc, expected):
    assert module._classify_load_failure(exc) == expected


def _force_load_failure(monkeypatch, exc):
    """Make ``from kernels import get_kernel`` succeed but the call raise."""

    def boom(*args, **kwargs):
        raise exc

    fake = type("FakeKernels", (), {"get_kernel": staticmethod(boom)})
    monkeypatch.setitem(__import__("sys").modules, "kernels", fake)
    return module._load_hub_kernel()


def test_dead_pin_is_logged_as_an_error(monkeypatch, caplog):
    """A pin nobody can fetch is our bug, and must not read as routine fallback."""
    with caplog.at_level(logging.DEBUG, logger=module.logger.name):
        assert _force_load_failure(monkeypatch, _RevisionNotFoundError("Invalid rev id")) is None
    assert module._hub_failure_kind == "unresolvable"
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "an unusable revision pin must be logged at ERROR"
    assert module._HUB_REVISION in errors[0].getMessage()
    assert "packaging bug" in errors[0].getMessage()


def test_missing_build_variant_is_not_an_error(monkeypatch, caplog):
    """No build for this OS/torch/CUDA is normal; the portable path is correct."""
    with caplog.at_level(logging.DEBUG, logger=module.logger.name):
        assert (
            _force_load_failure(
                monkeypatch, FileNotFoundError("Cannot find a build variant for this system")
            )
            is None
        )
    assert module._hub_failure_kind == "unsupported"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_failed_load_never_raises_into_inference(monkeypatch):
    """Whatever goes wrong, callers get None and fall back."""
    assert _force_load_failure(monkeypatch, RuntimeError("anything at all")) is None
    assert module._hub_failed is True


# --- the live canary ------------------------------------------------------


@pytest.mark.network
def test_pinned_revision_is_fetchable():
    """The pin must resolve on the endpoint the ``kernels`` client actually uses.

    Deliberately plain HTTP against ``/api/kernels/...`` rather than going
    through the ``kernels`` package: this has to run wherever there is a
    network, with no optional dependency, no GPU and no compiled build for
    the runner's platform. Resolution happens before variant matching, so a
    runner with no build for its own OS still catches a dead pin.
    """
    url = (
        f"https://huggingface.co/api/kernels/{module._HUB_REPO}"
        f"/tree/{module._HUB_REVISION}/build"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        pytest.fail(
            f"Pinned revision {module._HUB_REVISION} of {module._HUB_REPO} is not "
            f"fetchable ({exc.code} {exc.reason}) from {url}. The accelerated "
            "ms_deform_attn path is dead on every platform until the pin is fixed."
        )
    except urllib.error.URLError as exc:  # offline runner, not a pin problem
        pytest.skip(f"no network: {exc}")
    assert payload, "the pinned revision resolved but exposes no build variants"


@pytest.mark.network
def test_hub_load_failure_is_never_a_dead_pin():
    """End-to-end canary through the real loader, where the package is present.

    Skips when ``kernels`` is not installed. A machine with no compiled build
    is fine and expected; an unresolvable pin is not.
    """
    pytest.importorskip("kernels")
    module._load_hub_kernel()
    assert module._hub_failure_kind != "unresolvable", (
        f"{module._HUB_REPO} pinned at {module._HUB_REVISION} could not be "
        "resolved by the kernels client"
    )
