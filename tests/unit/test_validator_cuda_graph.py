"""Validation forward through captured CUDA graphs (ValidationConfig.cuda_graph).

The wiring under test: run() opens the model's graph scope when the config
asks for it, _inference routes through forward_maybe_graphed, and everything
degrades to the plain eager forward when graphs are off, unsupported, or
uncapturable. Bit-identity of replay itself is gated by test_cuda_graph.py.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from libreyolo.validation.base import BaseValidator
from libreyolo.validation.config import ValidationConfig

pytestmark = pytest.mark.unit


class _Runner:
    def __init__(self, forward):
        self.forward = forward
        self.calls = 0

    def run(self, x, auto=False):
        self.calls += 1
        return self.forward(x)


class _GraphAwareModel:
    """Model double implementing the graph-scope contract."""

    SUPPORTS_CUDA_GRAPH = True

    def __init__(self):
        self._cuda_graph_mode = None
        self.eager_calls = 0
        self.scope_modes = []
        self.support_checks = 0
        self.runner = _Runner(self._forward)

    def _forward(self, x):
        self.eager_calls += 1
        return x * 2

    def _get_graph_runner(self):
        return self.runner

    def _require_cuda_graph_support(self):
        self.support_checks += 1
        if not self.SUPPORTS_CUDA_GRAPH:
            raise NotImplementedError

    @contextmanager
    def cuda_graph_scope(self, mode):
        self.scope_modes.append(mode)
        self._cuda_graph_mode = "auto" if mode == "auto" else "on"
        try:
            yield
        finally:
            self._cuda_graph_mode = None


class _PlainModel:
    """Model double that never opted into anything graph related."""

    def __init__(self):
        self.eager_calls = 0

    def _forward(self, x):
        self.eager_calls += 1
        return x * 2


class _OneBatchValidator(BaseValidator):
    """Runs _inference on one fixed CPU batch; counts nothing else."""

    imgsz = 8  # doubles accept anything; real models need >= their stride

    def _setup_dataloader(self):
        return object()

    def _warmup_model(self, n_warmup: int = 3):
        pass

    def _init_metrics(self):
        pass

    def _run_validation(self):
        # Mirror the real loop's execution context: eval mode, no autograd.
        model = getattr(self.model, "model", None)
        if hasattr(model, "eval"):
            model.eval()
        size = type(self).imgsz
        with torch.no_grad():
            self.last_preds = self._inference(torch.ones(2, 3, size, size))

    def _preprocess_batch(self, batch):
        raise AssertionError("not used")

    def _postprocess_predictions(self, preds, batch):
        raise AssertionError("not used")

    def _update_metrics(self, preds, targets, img_info, img_ids=None):
        raise AssertionError("not used")

    def _compute_metrics(self):
        return {}


def _validator(model, tmp_path, **config_kwargs):
    config = ValidationConfig(
        data="x.yaml",
        device="cpu",
        save_dir=str(tmp_path / "val"),
        verbose=False,
        **config_kwargs,
    )
    return _OneBatchValidator(model=model, config=config)


def _assert_outputs_equal(left, right):
    """Structure-aware bit-equality for family forward outputs."""
    from libreyolo.models.base.cuda_graph import _flatten

    left_tensors, left_skeleton = _flatten(left)
    right_tensors, right_skeleton = _flatten(right)
    assert left_skeleton == right_skeleton
    assert len(left_tensors) == len(right_tensors)
    for index, (a, b) in enumerate(zip(left_tensors, right_tensors)):
        assert torch.equal(a, b), f"tensor {index} differs"


def test_config_defaults_off():
    assert ValidationConfig(data="x.yaml").cuda_graph is False


def test_disabled_takes_the_plain_eager_path(tmp_path):
    """Duck-typed doubles without any graph surface must keep working."""
    model = _PlainModel()
    v = _validator(model, tmp_path)
    v.run()
    assert model.eager_calls == 1
    assert torch.equal(v.last_preds, torch.ones(2, 3, 8, 8) * 2)


def test_enabled_opens_scope_and_routes_through_runner(tmp_path):
    model = _GraphAwareModel()
    v = _validator(model, tmp_path, cuda_graph=True)
    v.run()

    assert model.support_checks == 1
    assert model.scope_modes == [True]
    assert model.runner.calls == 1, "_inference must go through the graph runner"
    assert model.eager_calls == 1  # the runner double delegates to _forward
    assert model._cuda_graph_mode is None, "scope must close after the run"
    assert torch.equal(v.last_preds, torch.ones(2, 3, 8, 8) * 2)


def test_auto_mode_is_passed_through(tmp_path):
    model = _GraphAwareModel()
    v = _validator(model, tmp_path, cuda_graph="auto")
    v.run()
    assert model.scope_modes == ["auto"]


def test_unsupported_family_fails_loudly(tmp_path):
    model = _GraphAwareModel()
    model.SUPPORTS_CUDA_GRAPH = False
    v = _validator(model, tmp_path, cuda_graph=True)
    with pytest.raises(NotImplementedError):
        v.run()


def test_cpu_model_falls_back_to_eager_without_raising(tmp_path):
    """Real yolo9 on CPU: scope opens, the runner rejects CPU input, output
    must equal the eager forward."""
    from libreyolo import LibreYOLO9

    class _RealSizeValidator(_OneBatchValidator):
        imgsz = 64

    model = LibreYOLO9(model_path=None, size="t", device="cpu")
    config = ValidationConfig(
        data="x.yaml",
        device="cpu",
        save_dir=str(tmp_path / "val"),
        verbose=False,
        cuda_graph=True,
    )
    v = _RealSizeValidator(model=model, config=config)
    v.run()

    model.model.eval()
    with torch.no_grad():
        eager = model._forward(torch.ones(2, 3, 64, 64))
    _assert_outputs_equal(v.last_preds, eager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_validation_forward_is_bit_identical(tmp_path):
    """Our wiring (autocast context + forward_maybe_graphed) must not change
    a single bit of the forward output on the real capture path."""
    from libreyolo import LibreYOLO9

    model = LibreYOLO9(model_path=None, size="t", device="cuda")
    model.model.eval()
    torch.manual_seed(0)
    batch = torch.rand(2, 3, 64, 64, device="cuda")

    class _FixedBatchValidator(_OneBatchValidator):
        def _run_validation(self):
            self.last_preds = self._inference(batch)

    def run_once(cuda_graph):
        config = ValidationConfig(
            data="x.yaml",
            device="cuda",
            save_dir=str(tmp_path / "val"),
            verbose=False,
            cuda_graph=cuda_graph,
        )
        v = _FixedBatchValidator(model=model, config=config)
        with torch.no_grad():
            v.run()
        return v.last_preds

    eager = run_once(False)
    graphed = run_once(True)
    assert model.graph_info()["graph_count"] >= 1, "capture must have happened"
    _assert_outputs_equal(eager, graphed)
