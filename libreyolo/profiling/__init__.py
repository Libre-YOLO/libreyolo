"""LibreYOLO profiling tools.

Speed analysis for the two hot paths an agent optimises:

* **training** — ``libreyolo.training.profiler`` (the ``TrainStepProfiler`` +
  ``analyze_trace`` shipped with ``model.train(profile=True)``).
* **inference** — :class:`~libreyolo.profiling.inference.InferenceProfiler`
  here, driving the predict path for latency + GPU-truth.

Both emit the same self-contained ``profile.json`` (schema
``libreyolo.profile.analysis/v1``) so the ``libreyolo profile`` CLI lenses
(summary/get/phases/kernels/ops/compare) work identically on either.
"""
from __future__ import annotations

from .inference import InferenceProfiler

__all__ = ["InferenceProfiler"]
