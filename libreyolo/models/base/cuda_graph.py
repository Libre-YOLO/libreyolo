"""CUDA graph capture for model forward passes.

Small detectors are launch-bound: a LibreYOLO9s forward at 640 issues roughly
1200 kernels for a few milliseconds of GPU work, so the host spends most of the
step submitting launches while the GPU idles. Capturing the forward into a CUDA
graph collapses those launches into a single replay.

Capture is opt-in per call (``predict(..., cuda_graph=True)``) and covers the
forward only. Preprocessing and NMS stay outside: NMS is a fraction of a percent
of the step and has a dynamic output shape, so there is nothing to win and a
correctness contract to lose.

Graphs are keyed on ``(shape, dtype, device)`` because a graph is valid for
exactly the shape it was captured with. Anything that cannot be captured falls
back to eager with a single warning; capture never changes numerics, and
``tests/unit/test_cuda_graph.py`` gates that as bit-identical output.

Invalidation is a caller contract, not something this module can detect. A graph
records memory addresses, so replacing modules or tensors (``quantize()``,
``dequantize()``, a device move, a rebuilt detection head) leaves captured
kernels pointing at stale or freed storage, and the cache key cannot see it.
Every such site must call ``BaseModel._invalidate_cuda_graphs()``. Updating
weights *in place*, as an optimizer step or a state-dict load does, is safe and
needs nothing: replay reads the new values from the same addresses.
"""

from __future__ import annotations

import functools
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# A graph pins its own static input/output buffers, so an unbounded cache on a
# video with varying letterbox shapes would leak device memory steadily.
DEFAULT_MAX_GRAPHS = 8
# The PyTorch capture recipe: run the forward a few times on a side stream so
# lazy allocations, cuDNN benchmark picks and autotuning settle before capture.
DEFAULT_WARMUP_ITERS = 3
# ``cuda_graph="auto"`` waits for this many forwards at one shape before paying
# capture. One-shot and shape-varying workloads then never capture at all, while
# any repeated-shape loop converges after a negligible warmup.
DEFAULT_AUTO_THRESHOLD = 3

# Accepted ``cuda_graph=`` values. "auto" captures a shape once it repeats.
CUDA_GRAPH_MODES = (False, True, "auto")

GraphKey = Tuple[Tuple[int, ...], torch.dtype, str]


class CudaGraphUnavailable(RuntimeError):
    """Raised when graph capture is requested but cannot be performed."""


def _flatten(obj: Any) -> Tuple[List[torch.Tensor], Any]:
    """Split a forward output into its tensors and a rebuildable skeleton.

    Family ``_forward`` implementations return tensors, tuples, lists or dicts
    of tensors, so replay needs a structure-preserving way to clone the static
    output buffers back into the caller's expected shape.
    """
    tensors: List[torch.Tensor] = []

    def walk(node: Any) -> Any:
        if isinstance(node, torch.Tensor):
            tensors.append(node)
            return _Leaf(len(tensors) - 1)
        if isinstance(node, tuple):
            return tuple(walk(item) for item in node)
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, dict):
            return {key: walk(value) for key, value in node.items()}
        return node

    return tensors, walk(obj)


@dataclass(frozen=True)
class _Leaf:
    index: int


def _unflatten(skeleton: Any, tensors: List[torch.Tensor]) -> Any:
    if isinstance(skeleton, _Leaf):
        return tensors[skeleton.index]
    if isinstance(skeleton, tuple):
        return tuple(_unflatten(item, tensors) for item in skeleton)
    if isinstance(skeleton, list):
        return [_unflatten(item, tensors) for item in skeleton]
    if isinstance(skeleton, dict):
        return {key: _unflatten(value, tensors) for key, value in skeleton.items()}
    return skeleton


@dataclass
class _CapturedGraph:
    graph: "torch.cuda.CUDAGraph"
    static_input: torch.Tensor
    static_outputs: List[torch.Tensor]
    skeleton: Any
    replays: int = 0


@dataclass
class GraphRunner:
    """Captures and replays a model's forward pass, one graph per input shape.

    Args:
        forward_fn: The eager forward to capture, taking and returning what
            ``BaseModel._forward`` takes and returns.
        family: Model family name, for diagnostics.
        max_graphs: Cache ceiling. Exceeding it warns once and falls back to
            eager rather than growing device memory without bound.
        warmup_iters: Side-stream warmup iterations before capture.
    """

    forward_fn: Callable[[torch.Tensor], Any]
    family: str = ""
    max_graphs: int = DEFAULT_MAX_GRAPHS
    warmup_iters: int = DEFAULT_WARMUP_ITERS

    auto_threshold: int = DEFAULT_AUTO_THRESHOLD

    # When False, run() never captures on a shape miss and just runs eager;
    # already-captured shapes still replay. Callers that iterate a DataLoader
    # set this around their loop, because the loader's pin-memory threads call
    # cudaHostAlloc while batches are in flight, and any synchronous CUDA
    # allocation on the context invalidates a capture in progress. Such
    # callers capture at a controlled point first (see capture()).
    capture_on_miss: bool = True

    _graphs: Dict[GraphKey, _CapturedGraph] = field(default_factory=dict)
    _pool: Optional[Any] = None
    _misses: int = 0
    _fallback_reason: Optional[str] = None
    _capture_failed: bool = False
    _warned_capacity: bool = False
    _warned_unused: bool = False
    _shape_counts: Dict[GraphKey, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def run(self, input_tensor: torch.Tensor, auto: bool = False) -> Any:
        """Replay the graph for this input's shape, capturing it on first use.

        Falls back to the eager forward whenever capture is impossible. The
        returned tensors are clones of the graph's static output buffers, so
        callers may hold them across subsequent replays.

        Args:
            input_tensor: The forward input.
            auto: Wait for the shape to repeat ``auto_threshold`` times before
                paying capture, so one-shot calls never capture at all.
        """
        try:
            self._check_capturable(input_tensor)
        except CudaGraphUnavailable as exc:
            self._note_fallback(str(exc))
            return self.forward_fn(input_tensor)

        key = self._key(input_tensor)
        captured = self._graphs.get(key)
        if captured is None and auto:
            seen = self._shape_counts.get(key, 0) + 1
            self._shape_counts[key] = seen
            if seen < self.auto_threshold:
                # Not yet proven to be a repeated shape; stay eager silently.
                return self.forward_fn(input_tensor)

        if captured is None:
            if self._capture_failed or not self.capture_on_miss:
                # No capture attempt. Either a previous capture already failed
                # -- one cudaErrorStreamCaptureInvalidated poisons every later
                # capture attempt on the context, and each retry pays warmup
                # before failing the same way (measured: retry-per-batch made
                # validation 2.4x slower than eager) -- or the caller forbade
                # mid-loop capture because concurrent work (DataLoader
                # pin-memory threads calling cudaHostAlloc) would invalidate
                # it. Replays of already-captured shapes are unaffected.
                self._misses += 1
                return self.forward_fn(input_tensor)
            self._warn_if_precaptured_unused(key)
            if len(self._graphs) >= self.max_graphs:
                if not self._warned_capacity:
                    logger.warning(
                        "cuda_graph: reached the %d-graph cache limit for %s; "
                        "further input shapes run eager. Fix the input size "
                        "(imgsz=, batch=) to keep replays, or raise max_graphs.",
                        self.max_graphs,
                        self.family or "model",
                    )
                    self._warned_capacity = True
                self._misses += 1
                return self.forward_fn(input_tensor)
            try:
                captured = self._capture(input_tensor)
            except Exception as exc:  # capture is best-effort by contract
                self._capture_failed = True
                self._note_fallback(f"capture failed: {type(exc).__name__}: {exc}")
                return self.forward_fn(input_tensor)
            self._graphs[key] = captured

        with torch.cuda.device(input_tensor.device):
            captured.static_input.copy_(input_tensor)
            captured.graph.replay()
        captured.replays += 1
        return _unflatten(
            captured.skeleton, [tensor.clone() for tensor in captured.static_outputs]
        )

    def capture(self, input_tensor: torch.Tensor) -> None:
        """Capture a graph for this input shape now.

        Warmup plus capture costs far more than a replay, so servers and
        benchmarks call this up front rather than paying it on the first
        request. Raises :class:`CudaGraphUnavailable` instead of falling back,
        because an explicit request deserves an explicit failure.
        """
        self._check_capturable(input_tensor)
        key = self._key(input_tensor)
        if key in self._graphs:
            return
        if self._capture_failed:
            # A failed capture poisons later capture attempts on this context;
            # retrying costs a full warmup before failing identically. Callers
            # that really want to retry can release() first.
            raise CudaGraphUnavailable(
                f"a previous capture failed for {self.family or 'model'}; "
                "not retrying (call release() to reset)"
            )
        try:
            self._graphs[key] = self._capture(input_tensor)
        except Exception as exc:
            self._capture_failed = True
            raise CudaGraphUnavailable(
                f"CUDA graph capture failed for {self.family or 'model'} at "
                f"shape {tuple(input_tensor.shape)}: {exc}"
            ) from exc

    def info(self) -> Dict[str, Any]:
        """Report what is captured and why anything fell back."""
        return {
            "family": self.family,
            "captured": [
                {
                    "shape": list(key[0]),
                    "dtype": str(key[1]),
                    "device": key[2],
                    "replays": captured.replays,
                }
                for key, captured in self._graphs.items()
            ],
            "graph_count": len(self._graphs),
            "max_graphs": self.max_graphs,
            "eager_fallbacks": self._misses,
            "fallback_reason": self._fallback_reason,
        }

    def release(self) -> None:
        """Drop every captured graph and its static buffers."""
        self._graphs.clear()
        self._shape_counts.clear()
        self._pool = None
        self._misses = 0
        self._fallback_reason = None
        self._capture_failed = False
        self._warned_capacity = False
        self._warned_unused = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _key(input_tensor: torch.Tensor) -> GraphKey:
        return (
            tuple(input_tensor.shape),
            input_tensor.dtype,
            str(input_tensor.device),
        )

    def _check_capturable(self, input_tensor: torch.Tensor) -> None:
        if not torch.cuda.is_available():
            raise CudaGraphUnavailable("CUDA is not available")
        if input_tensor.device.type != "cuda":
            raise CudaGraphUnavailable(
                f"input is on {input_tensor.device.type}, not cuda"
            )
        if torch.is_grad_enabled():
            raise CudaGraphUnavailable(
                "gradients are enabled; capture requires no_grad/inference_mode"
            )

    def _capture(self, input_tensor: torch.Tensor) -> _CapturedGraph:
        # Static input the graph reads from on every replay. Replays copy the
        # caller's tensor in rather than rebinding, because a graph records
        # memory addresses, not values.
        static_input = torch.empty_like(input_tensor)
        static_input.copy_(input_tensor)

        # Stream, pool and graph all bind to the *current* device, which is not
        # necessarily the model's. Without this, a model on cuda:1 captures
        # against cuda:0 and the capture fails.
        with torch.cuda.device(input_tensor.device):
            # Warm up on the SAME stream the capture will record on. Library
            # handles (cuBLASLt for torch._scaled_mm, cuDNN) allocate their
            # workspaces lazily per stream; warming on a throwaway stream
            # leaves the capture stream cold, and the first call inside
            # capture then allocates and invalidates the whole capture with
            # cudaErrorStreamCaptureInvalidated.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(self.warmup_iters):
                    self.forward_fn(static_input)
            torch.cuda.current_stream().wait_stream(stream)

            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()

            # Quiesce the device before capture begins; in-flight work when
            # capture starts can invalidate it (seen on Windows/WDDM).
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            # thread_local: only this thread is restricted during capture.
            # Under the default "global" mode, a DataLoader pin-memory
            # thread staging the next batch (cudaHostAlloc) would both
            # invalidate the capture and be poisoned by it, killing the
            # run on its next batch fetch. The pin thread never touches
            # the capturing stream, so nothing it does belongs in the
            # graph.
            with torch.cuda.graph(
                graph,
                pool=self._pool,
                stream=stream,
                capture_error_mode="thread_local",
            ):
                output = self.forward_fn(static_input)

        static_outputs, skeleton = _flatten(output)
        if not static_outputs:
            raise CudaGraphUnavailable("forward produced no tensors to capture")

        logger.debug(
            "cuda_graph: captured %s at shape %s (%d output tensors)",
            self.family or "model",
            tuple(input_tensor.shape),
            len(static_outputs),
        )
        return _CapturedGraph(
            graph=graph,
            static_input=static_input,
            static_outputs=static_outputs,
            skeleton=skeleton,
        )

    def _warn_if_precaptured_unused(self, wanted: GraphKey) -> None:
        """Flag a pre-captured graph that no forward has ever replayed.

        ``capture_graph(imgsz=..., batch=...)`` restates a shape that predict
        derives independently, so the two can disagree. When they do, the user
        pays capture twice and gets nothing from the first: silent, and only
        visible in ``graph_info()``. Say so instead.
        """
        if self._warned_unused:
            return
        unused = [key for key, graph in self._graphs.items() if graph.replays == 0]
        if not unused:
            return
        self._warned_unused = True
        logger.warning(
            "cuda_graph: capturing %s for %s, but pre-captured shape(s) %s were "
            "never used. capture_graph() was called with a shape predict does "
            "not produce; match imgsz/batch to your predict call.",
            list(wanted[0]),
            self.family or "model",
            [list(key[0]) for key in unused],
        )

    def _note_fallback(self, reason: str) -> None:
        self._misses += 1
        if self._fallback_reason is None:
            self._fallback_reason = reason
            logger.warning(
                "cuda_graph requested but running eager for %s (%s)",
                self.family or "model",
                reason,
            )


def normalize_cuda_graph_mode(value: Any) -> Optional[str]:
    """Map a ``cuda_graph=`` argument to ``None``, ``"on"`` or ``"auto"``."""
    if value is False or value is None:
        return None
    if value is True:
        return "on"
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    raise ValueError(
        f"Invalid cuda_graph={value!r}. Expected True, False or 'auto'."
    )


def forward_maybe_graphed(model: Any, input_tensor: torch.Tensor) -> Any:
    """Run ``model._forward``, replaying a captured graph when one is in scope.

    Kept as a free function so the predict paths do not require anything beyond
    ``_forward`` from a model. Anything that has not opted in, including the
    duck-typed stubs in the test suite, takes the eager path unchanged.
    """
    mode = getattr(model, "_cuda_graph_mode", None)
    if mode is None:
        return model._forward(input_tensor)
    if getattr(model, "GRAPH_DISPATCH_IN_FORWARD", False):
        # Some families capture only part of the forward and finish the rest
        # eagerly, because the tail does data-dependent work that cannot be
        # recorded. Their ``_forward`` decides when to replay, so calling the
        # runner here instead would return the partial result and silently skip
        # the tail.
        return model._forward(input_tensor)
    return model._get_graph_runner().run(input_tensor, auto=(mode == "auto"))


def _scoped_generator(
    model: Any, generator: Iterator[Any], mode: Any
) -> Iterator[Any]:
    """Consume *generator* with the model's graph scope active around each step.

    Video predict returns a generator that the caller drains after ``predict``
    has already returned, so a scope around the call itself would be long gone
    by the time any forward runs.
    """
    while True:
        with model.cuda_graph_scope(mode):
            try:
                item = next(generator)
            except StopIteration:
                return
        yield item


def with_cuda_graph_scope(predict_fn: Callable) -> Callable:
    """Make a ``predict``-style method honor its own ``cuda_graph`` argument.

    Wrapping keeps the flag out of the dispatch body, which has several return
    paths and would otherwise need the whole block reindented. Invalid values
    and unsupported families raise here, before any work happens.
    """
    signature = inspect.signature(predict_fn)

    @functools.wraps(predict_fn)
    def wrapper(self, *args, **kwargs):
        bound = signature.bind_partial(self, *args, **kwargs)
        # Keep the caller's value: the scope owns normalization, so the public
        # vocabulary (True/False/"auto") never round-trips through it twice.
        requested = bound.arguments.get("cuda_graph", False)
        if normalize_cuda_graph_mode(requested) is None:
            return predict_fn(self, *args, **kwargs)

        model = self.model
        model._require_cuda_graph_support()
        with model.cuda_graph_scope(requested):
            result = predict_fn(self, *args, **kwargs)
        if inspect.isgenerator(result):
            return _scoped_generator(model, result, requested)
        return result

    return wrapper


__all__ = [
    "CUDA_GRAPH_MODES",
    "CudaGraphUnavailable",
    "GraphRunner",
    "forward_maybe_graphed",
    "normalize_cuda_graph_mode",
    "with_cuda_graph_scope",
]
