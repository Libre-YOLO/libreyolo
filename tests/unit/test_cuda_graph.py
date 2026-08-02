"""Tests for CUDA graph capture of model forward passes.

The CPU-only tests cover the contract (opt-in, guard rails, graceful fallback)
and run everywhere. The parity and replay tests need a real CUDA device and are
skipped otherwise; they are what actually gates the "capture never changes
numerics" claim.
"""

import logging

import pytest
import torch

from libreyolo.models.base.cuda_graph import (
    DEFAULT_AUTO_THRESHOLD,
    CudaGraphUnavailable,
    GraphRunner,
    normalize_cuda_graph_mode,
    with_cuda_graph_scope,
)
from libreyolo.models.base.model import BaseModel
from libreyolo.models.yolo9.model import LibreYOLO9

pytestmark = pytest.mark.unit

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture requires a CUDA device"
)


# ---------------------------------------------------------------------------
# Contract: opt-in flags
# ---------------------------------------------------------------------------


def test_base_model_does_not_opt_in_by_default():
    assert BaseModel.SUPPORTS_CUDA_GRAPH is False


def test_yolo9_opts_in():
    assert LibreYOLO9.SUPPORTS_CUDA_GRAPH is True


# ---------------------------------------------------------------------------
# GraphRunner fallback behavior (no CUDA needed)
# ---------------------------------------------------------------------------


def test_runner_falls_back_to_eager_on_cpu_input():
    calls = []

    def forward(x):
        calls.append(x)
        return x * 2

    runner = GraphRunner(forward_fn=forward, family="fake")
    x = torch.ones(1, 3, 8, 8)
    with torch.no_grad():
        out = runner.run(x)

    torch.testing.assert_close(out, x * 2)
    assert len(calls) == 1
    assert runner.info()["graph_count"] == 0
    reason = runner.info()["fallback_reason"]
    if torch.cuda.is_available():
        assert "cpu" in reason
    else:
        # No CUDA device means the availability guard trips first, so the
        # reason names that rather than the input device.
        assert "CUDA is not available" in reason


def test_runner_refuses_capture_when_grad_enabled():
    runner = GraphRunner(forward_fn=lambda x: x, family="fake")
    x = torch.ones(1, 3, 8, 8)
    # Grad enabled is the default; run() must fall back rather than raise.
    out = runner.run(x)
    torch.testing.assert_close(out, x)
    assert runner.info()["eager_fallbacks"] == 1


def test_runner_warns_once_on_fallback(caplog):
    runner = GraphRunner(forward_fn=lambda x: x, family="fake")
    x = torch.ones(1, 3, 4, 4)
    with caplog.at_level(logging.WARNING):
        with torch.no_grad():
            runner.run(x)
            runner.run(x)
    warnings = [r for r in caplog.records if "cuda_graph" in r.message]
    assert len(warnings) == 1, "fallback should warn once, not per call"
    assert runner.info()["eager_fallbacks"] == 2


def test_release_clears_state():
    runner = GraphRunner(forward_fn=lambda x: x, family="fake")
    with torch.no_grad():
        runner.run(torch.ones(1, 3, 4, 4))
    assert runner.info()["eager_fallbacks"] == 1
    runner.release()
    assert runner.info()["eager_fallbacks"] == 0
    assert runner.info()["fallback_reason"] is None


# ---------------------------------------------------------------------------
# Output structure round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output_factory",
    [
        lambda x: x,
        lambda x: (x, x + 1),
        lambda x: [x, (x + 1, x + 2)],
        lambda x: {"pred": x, "aux": [x + 1]},
        lambda x: (x, {"nested": x + 1}, "not-a-tensor"),
    ],
    ids=["tensor", "tuple", "nested-list", "dict", "mixed"],
)
def test_eager_fallback_preserves_output_structure(output_factory):
    runner = GraphRunner(forward_fn=output_factory, family="fake")
    x = torch.ones(1, 3, 4, 4)
    with torch.no_grad():
        out = runner.run(x)
    assert type(out) is type(output_factory(x))


# ---------------------------------------------------------------------------
# Model-level surface
# ---------------------------------------------------------------------------


def test_unsupported_family_raises_not_implemented(monkeypatch):
    model = LibreYOLO9(model_path=None, size="t", device="cpu")
    monkeypatch.setattr(type(model), "SUPPORTS_CUDA_GRAPH", False)
    with pytest.raises(NotImplementedError, match="not supported"):
        model.capture_graph(imgsz=640)
    with pytest.raises(NotImplementedError, match="not supported"):
        with model.cuda_graph_scope(True):
            pass


def test_graph_info_before_any_capture():
    model = LibreYOLO9(model_path=None, size="t", device="cpu")
    info = model.graph_info()
    assert info["graph_count"] == 0
    assert info["captured"] == []
    assert info["supported"] is True


def test_scope_is_restored_on_exception():
    model = LibreYOLO9(model_path=None, size="t", device="cpu")
    assert model._cuda_graph_mode is None
    with pytest.raises(ValueError):
        with model.cuda_graph_scope(True):
            assert model._cuda_graph_mode == "on"
            raise ValueError("boom")
    assert model._cuda_graph_mode is None


def test_disabled_scope_is_a_noop():
    model = LibreYOLO9(model_path=None, size="t", device="cpu")
    with model.cuda_graph_scope(False):
        assert model._cuda_graph_mode is None


def test_auto_scope_sets_auto_mode():
    model = LibreYOLO9(model_path=None, size="t", device="cpu")
    with model.cuda_graph_scope("auto"):
        assert model._cuda_graph_mode == "auto"
    assert model._cuda_graph_mode is None


@pytest.mark.parametrize(
    "value,expected",
    [
        (False, None),
        (None, None),
        (True, "on"),
        ("auto", "auto"),
        ("AUTO", "auto"),
    ],
)
def test_mode_normalization(value, expected):
    assert normalize_cuda_graph_mode(value) == expected


@pytest.mark.parametrize("value", ["on", "yes", 1, "graph"])
def test_invalid_mode_raises(value):
    with pytest.raises(ValueError, match="Invalid cuda_graph"):
        normalize_cuda_graph_mode(value)


def test_predict_rejects_invalid_mode():
    import numpy as np

    model = LibreYOLO9(model_path=None, size="t", device="cpu")
    model.model.eval()
    img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    with pytest.raises(ValueError, match="Invalid cuda_graph"):
        model.predict(img, cuda_graph="sometimes")


def test_predict_falls_back_on_cpu_without_raising():
    import numpy as np

    model = LibreYOLO9(model_path=None, size="t", device="cpu")
    model.model.eval()
    img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    result = model.predict(img, cuda_graph=True)
    assert result is not None
    assert model.graph_info()["graph_count"] == 0


def test_decorator_passes_through_when_disabled():
    class Fake:
        def __init__(self):
            self.model = object()

        @with_cuda_graph_scope
        def __call__(self, value, *, cuda_graph: bool = False):
            return value * 2

    # No model surface is touched when the flag is off.
    assert Fake()(21) == 42


# ---------------------------------------------------------------------------
# Real capture (CUDA only)
# ---------------------------------------------------------------------------


def _tensors(obj):
    """Flatten a forward output to its tensors, whatever container it uses."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for item in obj for t in _tensors(item)]
    if isinstance(obj, dict):
        return [t for item in obj.values() for t in _tensors(item)]
    return []


@requires_cuda
def test_capture_replay_is_bit_identical():
    """The load-bearing claim: a replayed graph equals eager exactly."""
    model = LibreYOLO9(model_path=None, size="s", device="cuda")
    model.model.eval()
    x = torch.randn(1, 3, 640, 640, device="cuda")

    with torch.no_grad():
        eager = model._forward(x)
        model.capture_graph(imgsz=640, batch=1)
        with model.cuda_graph_scope(True):
            graphed = model._forward_graphed(x)

    eager_tensors, graph_tensors = _tensors(eager), _tensors(graphed)
    assert eager_tensors and len(eager_tensors) == len(graph_tensors)
    for lhs, rhs in zip(eager_tensors, graph_tensors):
        assert lhs.shape == rhs.shape
        assert torch.equal(lhs, rhs), "graph replay diverged from eager"


@requires_cuda
def test_graphs_are_keyed_per_shape():
    model = LibreYOLO9(model_path=None, size="t", device="cuda")
    model.model.eval()
    model.capture_graph(imgsz=640, batch=1)
    model.capture_graph(imgsz=640, batch=2)
    shapes = [tuple(entry["shape"]) for entry in model.graph_info()["captured"]]
    assert (1, 3, 640, 640) in shapes
    assert (2, 3, 640, 640) in shapes
    assert model.graph_info()["graph_count"] == 2


@requires_cuda
def test_replayed_output_survives_a_later_replay():
    """Returned tensors are clones, so holding one across replays is safe."""
    model = LibreYOLO9(model_path=None, size="t", device="cuda")
    model.model.eval()
    first_input = torch.randn(1, 3, 640, 640, device="cuda")
    second_input = torch.randn(1, 3, 640, 640, device="cuda")

    with torch.no_grad(), model.cuda_graph_scope(True):
        first = model._forward_graphed(first_input)
        held = [t.clone() for t in _tensors(first)]
        # Replaying with different data must not mutate what the caller holds.
        model._forward_graphed(second_input)
        still_held = _tensors(first)

    assert held, "forward produced no tensors"
    for expected, actual in zip(held, still_held):
        assert torch.equal(expected, actual), "output aliased the static buffer"


@requires_cuda
def test_cache_cap_falls_back_instead_of_growing():
    model = LibreYOLO9(model_path=None, size="t", device="cuda")
    model.model.eval()
    runner = model._get_graph_runner()
    runner.max_graphs = 1

    with torch.no_grad(), model.cuda_graph_scope(True):
        model._forward_graphed(torch.randn(1, 3, 640, 640, device="cuda"))
        model._forward_graphed(torch.randn(2, 3, 640, 640, device="cuda"))

    info = model.graph_info()
    assert info["graph_count"] == 1
    assert info["eager_fallbacks"] >= 1


@requires_cuda
def test_release_graphs_frees_the_cache():
    model = LibreYOLO9(model_path=None, size="t", device="cuda")
    model.model.eval()
    model.capture_graph(imgsz=640, batch=1)
    assert model.graph_info()["graph_count"] == 1
    model.release_graphs()
    assert model.graph_info()["graph_count"] == 0


@requires_cuda
def test_auto_mode_waits_for_the_shape_to_repeat():
    """One-shot work must never pay capture; a repeated shape must."""
    model = LibreYOLO9(model_path=None, size="t", device="cuda")
    model.model.eval()
    x = torch.randn(1, 3, 640, 640, device="cuda")

    with torch.no_grad(), model.cuda_graph_scope("auto"):
        for _ in range(DEFAULT_AUTO_THRESHOLD - 1):
            model._forward_graphed(x)
        assert model.graph_info()["graph_count"] == 0, "captured too eagerly"
        model._forward_graphed(x)
        assert model.graph_info()["graph_count"] == 1, "never captured"


@requires_cuda
def test_auto_mode_does_not_capture_shape_varying_work():
    model = LibreYOLO9(model_path=None, size="t", device="cuda")
    model.model.eval()
    with torch.no_grad(), model.cuda_graph_scope("auto"):
        for size in (480, 512, 544, 576, 608, 640):
            model._forward_graphed(torch.randn(1, 3, size, size, device="cuda"))
    assert model.graph_info()["graph_count"] == 0


@requires_cuda
def test_auto_mode_output_matches_eager():
    model = LibreYOLO9(model_path=None, size="t", device="cuda")
    model.model.eval()
    x = torch.randn(1, 3, 640, 640, device="cuda")

    with torch.no_grad():
        eager = _tensors(model._forward(x))
        with model.cuda_graph_scope("auto"):
            for _ in range(DEFAULT_AUTO_THRESHOLD + 1):
                out = model._forward_graphed(x)
    assert model.graph_info()["graph_count"] == 1
    for lhs, rhs in zip(eager, _tensors(out)):
        assert torch.equal(lhs, rhs)


@requires_cuda
def test_warns_when_a_precaptured_shape_is_never_used(caplog):
    """capture_graph() restates a shape predict derives; catch a mismatch."""
    model = LibreYOLO9(model_path=None, size="t", device="cuda")
    model.model.eval()
    model.capture_graph(imgsz=640, batch=1)

    with caplog.at_level(logging.WARNING):
        with torch.no_grad(), model.cuda_graph_scope(True):
            model._forward_graphed(torch.randn(2, 3, 640, 640, device="cuda"))

    messages = [r.message for r in caplog.records if "never used" in r.message]
    assert messages, "a pre-captured but unused graph should warn"


@requires_cuda
def test_no_unused_warning_when_precapture_matches(caplog):
    model = LibreYOLO9(model_path=None, size="t", device="cuda")
    model.model.eval()
    model.capture_graph(imgsz=640, batch=1)

    with caplog.at_level(logging.WARNING):
        with torch.no_grad(), model.cuda_graph_scope(True):
            model._forward_graphed(torch.randn(1, 3, 640, 640, device="cuda"))
            model._forward_graphed(torch.randn(2, 3, 640, 640, device="cuda"))

    assert not [r for r in caplog.records if "never used" in r.message]


# ---------------------------------------------------------------------------
# Invalidation: a graph records addresses, so relocating parameters must
# drop the cache (PR #645 review findings 2 and 3).
# ---------------------------------------------------------------------------


@requires_cuda
def test_quantize_invalidates_captured_graphs():
    """quantize() swaps modules, so parameters move and graphs go stale."""
    model = LibreYOLO9(model_path=None, size="t", device="cuda")
    model.model.eval()
    model.capture_graph(imgsz=640, batch=1)
    assert model.graph_info()["graph_count"] == 1

    model._invalidate_cuda_graphs("quantize")
    assert model.graph_info()["graph_count"] == 0


@requires_cuda
def test_device_change_invalidates_captured_graphs():
    """A device move reallocates parameters; the old cache entry is dead.

    Driven through ``_set_device`` rather than ``predict(device=...)`` because
    a cuda-to-cpu predict switch trips an unrelated pre-existing bug: YOLO9
    caches ``anchor_points`` as a plain attribute that ``.to()`` does not move.
    """
    from libreyolo.models.base.inference import InferenceRunner

    model = LibreYOLO9(model_path=None, size="t", device="cuda")
    model.model.eval()
    model.capture_graph(imgsz=640, batch=1)
    assert model.graph_info()["graph_count"] == 1

    InferenceRunner(model)._set_device("cpu")
    assert model.graph_info()["graph_count"] == 0, (
        "moving the model must drop graphs captured on the old device"
    )


def test_invalidate_is_a_noop_without_a_runner():
    model = LibreYOLO9(model_path=None, size="t", device="cpu")
    model._invalidate_cuda_graphs("nothing captured yet")
    assert model.graph_info()["graph_count"] == 0


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs a second CUDA device"
)
def test_capture_binds_to_the_models_device_not_the_current_one():
    """Capture must target the model's device, not whatever is current."""
    model = LibreYOLO9(model_path=None, size="t", device="cuda:1")
    model.model.eval()
    with torch.cuda.device(0):  # current device deliberately differs
        model.capture_graph(imgsz=640, batch=1)
        with torch.no_grad(), model.cuda_graph_scope(True):
            out = model._forward_graphed(
                torch.randn(1, 3, 640, 640, device="cuda:1")
            )
    assert model.graph_info()["graph_count"] == 1
    assert model.graph_info()["eager_fallbacks"] == 0
    assert all(t.device.index == 1 for t in _tensors(out))


@requires_cuda
def test_capture_raises_rather_than_falling_back():
    """Explicit capture() must fail loudly; only run() degrades silently."""

    def bad_forward(_x):
        raise RuntimeError("synthetic capture failure")

    runner = GraphRunner(forward_fn=bad_forward, family="fake")
    with torch.no_grad():
        with pytest.raises(CudaGraphUnavailable, match="capture failed"):
            runner.capture(torch.randn(1, 3, 32, 32, device="cuda"))


@requires_cuda
def test_capture_survives_concurrent_pinned_allocations():
    """Capture must use capture_error_mode="thread_local".

    Under the default "global" mode a cudaHostAlloc from any other host
    thread — which is exactly what a DataLoader pin-memory thread does while
    staging the next batch — invalidates the capture and poisons that
    thread. Seen in the wild during training-time validation on the first
    RF100-VL Vast campaign.
    """
    import threading

    net = torch.nn.Conv2d(3, 8, 3, padding=1).cuda().eval()
    runner = GraphRunner(forward_fn=net, family="fake")
    x = torch.randn(1, 3, 32, 32, device="cuda")

    stop = threading.Event()
    errors: list[BaseException] = []

    def hammer():
        held = []
        size = 0
        while not stop.is_set():
            try:
                # Strictly growing sizes defeat the caching host allocator,
                # so every iteration is a real cudaHostAlloc.
                held.append(
                    torch.empty(4096 + size * 640, dtype=torch.uint8).pin_memory()
                )
                size += 1
            except BaseException as exc:  # must never happen
                errors.append(exc)
                return

    thread = threading.Thread(target=hammer, daemon=True)
    thread.start()
    try:
        with torch.no_grad():
            out = runner.run(x)
    finally:
        stop.set()
        thread.join(timeout=30)

    assert not errors, f"pin-memory stand-in thread was poisoned: {errors[0]!r}"
    assert runner.info()["graph_count"] == 1
    assert runner.info()["fallback_reason"] is None
    torch.testing.assert_close(out, net(x))
