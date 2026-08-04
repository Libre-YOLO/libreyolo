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
    message = errors[0].getMessage()
    assert module._HUB_REVISION in message, "the error must name the unusable pin"
    assert "hub-kernels" in message, "the error must point at the version cap to check"


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
def test_pinned_revision_exists_in_the_repo():
    """The pinned commit must still exist upstream, with build variants on it.

    Plain HTTP against the models API, which does accept commit SHAs, so this
    runs with no optional dependency, no GPU and no compiled build for the
    runner's platform. It catches a pin that was rewritten or force-pushed
    away; it deliberately does *not* go through ``/api/kernels/``, which
    rejects SHAs outright and would fail even when the pin is perfectly good.
    """
    url = (
        f"https://huggingface.co/api/models/{module._HUB_REPO}"
        f"/tree/{module._HUB_REVISION}/build"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        pytest.fail(
            f"Pinned revision {module._HUB_REVISION} of {module._HUB_REPO} no "
            f"longer exists upstream ({exc.code} {exc.reason}) at {url}."
        )
    except urllib.error.URLError as exc:  # offline runner, not a pin problem
        pytest.skip(f"no network: {exc}")
    assert payload, "the pinned revision exists but exposes no build variants"


@pytest.mark.network
def test_installed_kernels_client_can_resolve_the_pin():
    """The canary that matters: the *installed* client must accept the pin.

    ``kernels`` 0.14 moved revision resolution to an endpoint that rejects
    commit SHAs, which silently removed the accelerated path everywhere. The
    ``hub-kernels`` extra caps the client below that, and this asserts the cap
    is still doing its job. A runner with no compiled build for its platform
    is fine and expected -- that is a different failure kind.
    """
    client = pytest.importorskip("kernels")
    # This directory is itself named `kernels`. If the test tree ever ends up
    # on sys.path as a top-level package again it shadows the real client, and
    # every hub test degrades to a silent skip -- which is exactly how the
    # provider's GPU parity test sat dead for its whole life.
    assert hasattr(client, "get_kernel"), (
        f"`import kernels` resolved to {getattr(client, '__file__', client)!r} "
        "instead of the Hub client. Something (a missing __init__.py in a "
        "parent test package) is shadowing it, and the hub tests below are "
        "not testing what they claim to."
    )
    module._load_hub_kernel()
    assert module._hub_failure_kind != "unresolvable", (
        f"the installed kernels client cannot resolve {module._HUB_REPO} at "
        f"{module._HUB_REVISION}. Check the version cap on the hub-kernels "
        "extra in pyproject.toml."
    )
