"""Tests for CUDA graph capture of the training step.

CPU tests cover the dispatch machinery (tree flattening, shape counting,
fallback guarantees, trainer routing, family gating). CUDA tests gate the
core promise: enabling ``cuda_graph`` must not change training numerics,
so eager and graphed runs are compared step by step, loss and parameters
both, for YOLO9 and RF-DETR.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from libreyolo.training.cuda_graph import (
    CudaGraphTrainSpec,
    GraphableNetwork,
    TrainGraphManager,
    _thread_local_capture_errors,
    flatten_tree,
    unflatten_tree,
)
from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# =============================================================================
# Tree flattening
# =============================================================================


class TestTreeFlatten:
    def test_roundtrip_nested(self):
        a, b, c = torch.zeros(1), torch.ones(2), torch.full((3,), 2.0)
        tree = {
            "pred": a,
            "aux": [{"x": b}, {"x": c}],
            "meta": (None, 7, "tag"),
        }
        flat, skeleton = flatten_tree(tree)
        assert len(flat) == 3
        rebuilt = unflatten_tree(skeleton, flat)
        # Identity, not equality: autograd connectivity depends on the
        # rebuilt tree containing the same tensor objects.
        assert rebuilt["pred"] is a
        assert rebuilt["aux"][0]["x"] is b
        assert rebuilt["aux"][1]["x"] is c
        assert rebuilt["meta"] == (None, 7, "tag")

    def test_plain_list(self):
        t = [torch.zeros(1), torch.zeros(2)]
        flat, skeleton = flatten_tree(t)
        rebuilt = unflatten_tree(skeleton, flat)
        assert isinstance(rebuilt, list)
        assert rebuilt[0] is t[0] and rebuilt[1] is t[1]

    def test_graphable_network_adapter(self):
        class Toy(nn.Module):
            def forward(self, x):
                return {"a": x * 2, "b": [x + 1, None]}

        net = GraphableNetwork(Toy())
        x = torch.arange(4.0)
        flat = net(x)
        assert isinstance(flat, tuple) and len(flat) == 2
        rebuilt = net.rebuild(flat)
        assert torch.equal(rebuilt["a"], x * 2)
        assert rebuilt["b"][1] is None
        # The wrapped module's parameters are visible through the adapter,
        # which is what lets capture pass them as graph inputs.
        assert isinstance(net.module, Toy)


# =============================================================================
# Manager dispatch
# =============================================================================


def _fake_cuda_batch(shape=(2, 3, 64, 64)):
    """A stand-in for a CUDA batch tensor, usable on CPU-only machines."""
    imgs = MagicMock(spec=torch.Tensor)
    imgs.is_cuda = True
    imgs.shape = torch.Size(shape)
    imgs.dtype = torch.float32
    imgs.device = "cuda:0"
    return imgs


def _spec():
    return CudaGraphTrainSpec(network=MagicMock(), assemble=MagicMock())


class TestTrainGraphManager:
    def test_non_cuda_input_disables(self):
        manager = TrainGraphManager()
        out = manager.run(_spec(), torch.zeros(1, 3, 8, 8))
        assert out is None
        assert manager.disabled

    def test_captures_after_threshold_and_replays(self):
        manager = TrainGraphManager(warmup_threshold=3)
        spec = _spec()
        imgs = _fake_cuda_batch()
        graphed = MagicMock(return_value=("flat",))
        with patch(
            "torch.cuda.make_graphed_callables", return_value=graphed
        ) as make:
            assert manager.run(spec, imgs) is None
            assert manager.run(spec, imgs) is None
            assert not manager.captured
            out = manager.run(spec, imgs)
        assert out == ("flat",)
        assert manager.captured
        make.assert_called_once()
        # Subsequent same-shape batches replay without recapturing.
        with patch("torch.cuda.make_graphed_callables") as make_again:
            assert manager.run(spec, imgs) == ("flat",)
        make_again.assert_not_called()

    def test_shape_mismatch_falls_back_eager(self):
        manager = TrainGraphManager(warmup_threshold=1)
        spec = _spec()
        with patch(
            "torch.cuda.make_graphed_callables",
            return_value=MagicMock(return_value=("flat",)),
        ):
            assert manager.run(spec, _fake_cuda_batch((2, 3, 64, 64))) == ("flat",)
        # A different shape (multi-scale batch, last partial batch) must run
        # eager without disabling the captured graph.
        assert manager.run(spec, _fake_cuda_batch((1, 3, 64, 64))) is None
        assert not manager.disabled
        assert manager.run(spec, _fake_cuda_batch((2, 3, 64, 64))) == ("flat",)

    def test_capture_failure_disables_permanently(self):
        manager = TrainGraphManager(warmup_threshold=1)
        spec = _spec()
        imgs = _fake_cuda_batch()
        with patch(
            "torch.cuda.make_graphed_callables", side_effect=RuntimeError("boom")
        ):
            assert manager.run(spec, imgs) is None
        assert manager.disabled
        # Disabled means no further capture attempts at all.
        with patch("torch.cuda.make_graphed_callables") as make:
            assert manager.run(spec, imgs) is None
        make.assert_not_called()

    def test_replay_failure_disables(self):
        manager = TrainGraphManager(warmup_threshold=1)
        spec = _spec()
        imgs = _fake_cuda_batch()
        graphed = MagicMock(side_effect=[("flat",), RuntimeError("stale")])
        with patch("torch.cuda.make_graphed_callables", return_value=graphed):
            assert manager.run(spec, imgs) == ("flat",)
        assert manager.run(spec, imgs) is None
        assert manager.disabled


# =============================================================================
# Capture error mode (pin-memory thread race)
# =============================================================================
#
# Capture defaults to capture_error_mode="global", under which a cudaHostAlloc
# from a DataLoader pin-memory thread invalidates the capture AND poisons the
# pin thread, killing the run on the next batch fetch (seen twice on the first
# RF100-VL Vast campaign as "AcceleratorError ... in pin memory thread").
# Capture must therefore run in "thread_local" mode, where other threads'
# CUDA calls proceed normally.


class TestThreadLocalCaptureErrors:
    def test_swaps_and_restores_cudagraph_symbol(self):
        original = torch.cuda.CUDAGraph
        with _thread_local_capture_errors():
            assert torch.cuda.CUDAGraph is not original
            assert issubclass(torch.cuda.CUDAGraph, original)
        assert torch.cuda.CUDAGraph is original

    def test_restores_on_exception(self):
        original = torch.cuda.CUDAGraph
        with pytest.raises(RuntimeError, match="boom"):
            with _thread_local_capture_errors():
                raise RuntimeError("boom")
        assert torch.cuda.CUDAGraph is original

    def test_forces_thread_local_at_capture_begin(self):
        class FakeGraph:
            def capture_begin(self, *args, **kwargs):
                self.begin_kwargs = kwargs

        with patch("torch.cuda.CUDAGraph", FakeGraph):
            with _thread_local_capture_errors():
                graph = torch.cuda.CUDAGraph()
                # torch.cuda.graph.__enter__ passes the default explicitly;
                # the subclass must override it.
                graph.capture_begin(capture_error_mode="global")
        assert graph.begin_kwargs["capture_error_mode"] == "thread_local"

    def test_manager_captures_under_the_patch(self):
        """The capture path must instantiate the patched graph class."""
        manager = TrainGraphManager(warmup_threshold=1)
        original = torch.cuda.CUDAGraph
        seen = {}

        def record_class(*args, **kwargs):
            seen["cls"] = torch.cuda.CUDAGraph
            return MagicMock(return_value=("flat",))

        with patch("torch.cuda.make_graphed_callables", side_effect=record_class):
            assert manager.run(_spec(), _fake_cuda_batch()) == ("flat",)
        assert seen["cls"] is not original
        assert issubclass(seen["cls"], original)
        assert torch.cuda.CUDAGraph is original

    @requires_cuda
    def test_capture_survives_concurrent_pinned_allocations(self):
        """Regression for the campaign killer: capture while a stand-in for
        the DataLoader pin-memory thread keeps calling cudaHostAlloc."""
        import threading

        net = GraphableNetwork(nn.Conv2d(3, 8, 3, padding=1).cuda())
        spec = CudaGraphTrainSpec(
            network=net,
            assemble=lambda flat, *args: {"total_loss": flat[0].sum()},
        )
        manager = TrainGraphManager(warmup_threshold=1)
        imgs = torch.randn(2, 3, 32, 32, device="cuda")

        stop = threading.Event()
        errors: list[BaseException] = []

        def hammer():
            held = []
            size = 0
            while not stop.is_set():
                try:
                    # Strictly growing sizes defeat the caching host
                    # allocator, so every iteration is a real cudaHostAlloc,
                    # exactly what the pin-memory thread issues per batch.
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
            out = manager.run(spec, imgs)
        finally:
            stop.set()
            thread.join(timeout=30)

        assert not errors, f"pin-memory stand-in thread was poisoned: {errors[0]!r}"
        assert out is not None
        assert manager.captured and not manager.disabled
        # And the graph still replays.
        assert manager.run(spec, imgs) is not None


# =============================================================================
# Trainer routing
# =============================================================================


class _RoutingHost:
    """Minimal stand-in exercising BaseTrainer._forward_train unbound."""

    def __init__(self, manager, spec):
        self._cuda_graph_manager = manager
        self._cuda_graph_spec = None
        self._cuda_graph_spec_resolved = False
        self._spec_to_return = spec
        self.on_forward_calls = 0

    def cuda_graph_train_spec(self):
        return self._spec_to_return

    def on_forward(self, imgs, targets, polygons=None):
        self.on_forward_calls += 1
        return {"total_loss": torch.zeros(())}


class TestForwardTrainRouting:
    def test_no_manager_goes_eager(self):
        host = _RoutingHost(manager=None, spec=None)
        out = BaseTrainer._forward_train(host, torch.zeros(1), torch.zeros(1))
        assert host.on_forward_calls == 1
        assert "total_loss" in out

    def test_family_without_spec_disables_and_goes_eager(self):
        manager = TrainGraphManager()
        host = _RoutingHost(manager=manager, spec=None)
        BaseTrainer._forward_train(host, torch.zeros(1), torch.zeros(1))
        assert host.on_forward_calls == 1
        assert manager.disabled

    def test_spec_used_when_graph_runs(self):
        manager = TrainGraphManager()
        flat = (torch.ones(2),)
        assembled = {"total_loss": torch.ones(())}
        spec = CudaGraphTrainSpec(
            network=MagicMock(), assemble=MagicMock(return_value=assembled)
        )
        host = _RoutingHost(manager=manager, spec=spec)
        with patch.object(TrainGraphManager, "run", return_value=flat):
            out = BaseTrainer._forward_train(host, torch.zeros(1), torch.zeros(1))
        assert out is assembled
        assert host.on_forward_calls == 0
        spec.assemble.assert_called_once()

    def test_graph_miss_falls_back_to_on_forward(self):
        manager = TrainGraphManager()
        spec = CudaGraphTrainSpec(network=MagicMock(), assemble=MagicMock())
        host = _RoutingHost(manager=manager, spec=spec)
        with patch.object(TrainGraphManager, "run", return_value=None):
            BaseTrainer._forward_train(host, torch.zeros(1), torch.zeros(1))
        assert host.on_forward_calls == 1
        spec.assemble.assert_not_called()

    def test_spec_resolution_exception_disables(self):
        manager = TrainGraphManager()
        host = _RoutingHost(manager=manager, spec=None)
        host.cuda_graph_train_spec = MagicMock(side_effect=RuntimeError("nope"))
        BaseTrainer._forward_train(host, torch.zeros(1), torch.zeros(1))
        assert manager.disabled
        assert host.on_forward_calls == 1


# =============================================================================
# Family gating
# =============================================================================


class TestYolo9SpecGating:
    def _host(self, task="detect"):
        from libreyolo.models.yolo9.nn import LibreYOLO9Model
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        host = SimpleNamespace(
            wrapper_model=SimpleNamespace(task=task),
            model=LibreYOLO9Model(config="t", nb_classes=3),
        )
        return YOLO9Trainer.cuda_graph_train_spec, host

    def test_detect_supported(self):
        fn, host = self._host()
        spec = fn(host)
        assert spec is not None
        assert spec.network.module is host.model

    def test_non_detect_task_unsupported(self):
        fn, host = self._host(task="pose")
        assert fn(host) is None

    def test_derived_head_unsupported(self):
        fn, host = self._host()
        # Subclassed heads (e2e dual assignment) compute loss at a
        # different boundary; the exact-type gate must reject them.
        class DerivedHead(type(host.model.head)):
            pass

        derived = DerivedHead.__new__(DerivedHead)
        derived.__dict__.update(host.model.head.__dict__)
        host.model.head = derived
        assert fn(host) is None


class TestRFDETRSpecGating:
    def _host(self, task="detect", **model_flags):
        from libreyolo.models.rfdetr.nn import LibreRFDETRModel
        from libreyolo.models.rfdetr.trainer import RFDETRTrainer

        # __new__ dodges the heavy DINOv2 build; gating only reads flags.
        model = object.__new__(LibreRFDETRModel)
        flags = {
            "segmentation": False,
            "pose": False,
            "obb": False,
            "classification": False,
            "semantic": False,
        }
        flags.update(model_flags)
        for key, value in flags.items():
            setattr(model, key, value)
        host = SimpleNamespace(
            wrapper_model=SimpleNamespace(task=task),
            model=model,
            criterion=MagicMock(weight_dict={}),
            _targets_to_rfdetr_list=MagicMock(),
        )
        return RFDETRTrainer.cuda_graph_train_spec, host

    def test_detect_supported(self):
        fn, host = self._host()
        spec = fn(host)
        assert spec is not None
        assert spec.network.module is host.model

    def test_task_variants_unsupported(self):
        for flag in ("segmentation", "pose", "obb", "classification"):
            fn, host = self._host(**{flag: True})
            assert fn(host) is None, flag
        fn, host = self._host(task="segment")
        assert fn(host) is None

    def test_missing_criterion_unsupported(self):
        fn, host = self._host()
        host.criterion = None
        assert fn(host) is None


# =============================================================================
# CUDA parity: enabling cuda_graph must not change training numerics
# =============================================================================


def _yolo9_targets(bsz, device, generator):
    t = torch.zeros(bsz, 50, 5)
    n = 10
    cls = torch.randint(0, 3, (bsz, n), generator=generator).float()
    cx = torch.rand(bsz, n, generator=generator) * 0.8 + 0.1
    cy = torch.rand(bsz, n, generator=generator) * 0.8 + 0.1
    w = torch.rand(bsz, n, generator=generator) * 0.2 + 0.05
    h = torch.rand(bsz, n, generator=generator) * 0.2 + 0.05
    t[:, :n, 0] = cls
    t[:, :n, 1] = (cx - w / 2).clamp(0, 1)
    t[:, :n, 2] = (cy - h / 2).clamp(0, 1)
    t[:, :n, 3] = (cx + w / 2).clamp(0, 1)
    t[:, :n, 4] = (cy + h / 2).clamp(0, 1)
    return t.to(device)


def _run_steps(model, forward_fn, imgs, targets, steps):
    """Shared harness: SGD + AMP GradScaler loop, mirrors the trainer."""
    from torch.amp import GradScaler, autocast

    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = GradScaler("cuda")
    losses = []
    for _ in range(steps):
        with autocast("cuda", cache_enabled=False):
            outputs = forward_fn(imgs, targets)
            loss = outputs["total_loss"]
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.item()))
    return losses


@requires_cuda
class TestCudaParityYolo9:
    def test_loss_trajectory_identical(self):
        from libreyolo.models.yolo9.nn import LibreYOLO9Model
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        bsz, size, steps = 2, 128, 6
        gen = torch.Generator().manual_seed(3)
        imgs = torch.randn(bsz, 3, size, size, generator=gen).cuda()
        targets = _yolo9_targets(bsz, "cuda", gen)

        def build():
            torch.manual_seed(11)
            model = LibreYOLO9Model(config="t", nb_classes=3).cuda().train()
            return model

        # Eager arm mirrors on_forward exactly.
        model_e = build()
        eager = _run_steps(
            model_e,
            lambda i, t: model_e(i, targets=t),
            imgs,
            targets,
            steps,
        )

        # Graphed arm goes through the real spec + manager.
        model_g = build()
        host = SimpleNamespace(
            wrapper_model=SimpleNamespace(task="detect"), model=model_g
        )
        spec = YOLO9Trainer.cuda_graph_train_spec(host)
        assert spec is not None
        manager = TrainGraphManager(warmup_threshold=1)

        def graphed_forward(i, t):
            flat = manager.run(spec, i)
            assert flat is not None
            return spec.assemble(flat, i, t)

        graphed = _run_steps(model_g, graphed_forward, imgs, targets, steps)

        assert manager.captured
        assert eager == pytest.approx(graphed, rel=0, abs=0), (
            f"eager {eager} != graphed {graphed}"
        )
        # Parameters must match exactly after the same number of steps.
        for (name, pe), (_, pg) in zip(
            model_e.named_parameters(), model_g.named_parameters()
        ):
            assert torch.equal(pe, pg), name
        # Buffers too: capture warm-up must not leave extra BatchNorm
        # running-stat updates behind (the manager snapshots and restores
        # them), or validation, EMA and checkpoints would drift from eager.
        buffer_names = 0
        for (name, be), (_, bg) in zip(
            model_e.named_buffers(), model_g.named_buffers()
        ):
            assert torch.equal(be, bg), name
            buffer_names += 1
        assert buffer_names > 0, "expected BatchNorm buffers to compare"


@requires_cuda
class TestCudaParityRFDETR:
    """RF-DETR parity, with a tolerance matched to eager's own noise floor.

    Unlike YOLO9, RF-DETR training is not bitwise reproducible even eager
    to eager: the deformable-attention backward accumulates with atomics,
    so two identical seeded eager runs diverge from step 1 (measured max
    relative difference about 4e-4 over 4 steps on an RTX 5070 Ti). The
    graph contract therefore is: the first step, whose forward and loss
    run on identical weights, must match bit for bit, and the trajectory
    must stay within the eager run-to-run noise band. A real gradient bug
    (wrong boundary, stale buffers) diverges orders of magnitude faster.
    """

    # The DINOv2 backbone build fetches pretrained weights.
    @pytest.mark.external_data
    def test_loss_trajectory_identical(self):
        from libreyolo.models.rfdetr.nn import LibreRFDETRModel
        from libreyolo.models.rfdetr.trainer import RFDETRTrainer

        steps = 4

        def build():
            torch.manual_seed(23)
            model = LibreRFDETRModel(config="n", nb_classes=3, device="cuda")
            model = model.cuda().train()
            criterion, _ = model.build_criterion_and_postprocess()
            criterion.to("cuda")
            return model, criterion

        model_e, criterion_e = build()
        size = model_e.resolution
        bsz = 2
        gen = torch.Generator().manual_seed(5)
        imgs = torch.randn(bsz, 3, size, size, generator=gen).cuda()
        # The RF-DETR target converter expects pixel coordinates.
        targets = _yolo9_targets(bsz, "cuda", gen)
        targets[..., 1:5] *= size

        def make_host(model, criterion):
            host = SimpleNamespace(
                wrapper_model=SimpleNamespace(task="detect"),
                model=model,
                criterion=criterion,
                device=torch.device("cuda"),
            )
            host._targets_to_rfdetr_list = (
                lambda *args, **kwargs: RFDETRTrainer._targets_to_rfdetr_list(
                    host, *args, **kwargs
                )
            )
            return host

        host_e = make_host(model_e, criterion_e)

        def eager_forward(i, t):
            return RFDETRTrainer.on_forward(host_e, i, t)

        eager = _run_steps(model_e, eager_forward, imgs, targets, steps)

        model_g, criterion_g = build()
        host_g = make_host(model_g, criterion_g)
        spec = RFDETRTrainer.cuda_graph_train_spec(host_g)
        assert spec is not None
        manager = TrainGraphManager(warmup_threshold=1)

        def graphed_forward(i, t):
            flat = manager.run(spec, i)
            assert flat is not None
            return spec.assemble(flat, i, t)

        graphed = _run_steps(model_g, graphed_forward, imgs, targets, steps)

        assert manager.captured
        assert eager[0] == graphed[0], (
            f"step-0 loss must be bit-identical: {eager[0]} != {graphed[0]}"
        )
        assert eager == pytest.approx(graphed, rel=5e-3), (
            f"trajectory outside eager noise band: eager {eager} != "
            f"graphed {graphed}"
        )
