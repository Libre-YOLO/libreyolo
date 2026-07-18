"""Kernel registry: pluggable accelerated implementations for quant ops.

Architecture (vLLM-shaped, scaled down to a library that rents its runtime):

- The reference implementations in ``fake_quant.py`` are always registered
  and always available. They are the numerical oracle: any accelerated
  implementation must match them (parity tests gate in-tree kernels, and
  ``packing.py`` never has variants because it is the checkpoint contract).
- Triton kernels live under ``kernels/triton/`` as plain Python (JIT
  compiled at runtime, no build step, no wheels).
- Compiled kernels ship out-of-tree in the optional ``libreyolo-kernels``
  package. If installed, it is imported here and self-registers via
  :func:`register`; the core package never grows a build dependency.
- Selection is per-op: implementations are tried newest-first and the first
  one whose predicate passes wins, falling back to the reference. The
  ``LIBREYOLO_QUANT_KERNELS`` environment variable overrides selection:
  ``off``/``reference`` forces the reference implementations; any other
  value selects only implementations registered under that name.

Registered op slots (callables with the reference signatures):
``fake_quant_int8_per_channel``, ``fake_quant_int8_affine``,
``fake_quant_fp8``, ``fake_quant_int_grouped``, ``fake_quant_nvfp4_weight``,
``fake_quant_nvfp4_dynamic``, ``fake_quant_mxfp4_weight``,
``fake_quant_mxfp4_dynamic``. GEMM slots (``nvfp4_gemm``, ...) may be
registered by external packages; they have no reference implementation and
callers must check :func:`resolve` returns non-None before use.
"""

import importlib.util
import logging
import os
from typing import Callable, Dict, List, Optional

from .. import fake_quant as _reference

logger = logging.getLogger(__name__)

REFERENCE_OPS = (
    "fake_quant_int8_per_channel",
    "fake_quant_int8_affine",
    "fake_quant_fp8",
    "fake_quant_int_grouped",
    "fake_quant_nvfp4_weight",
    "fake_quant_nvfp4_dynamic",
    "fake_quant_mxfp4_weight",
    "fake_quant_mxfp4_dynamic",
)

_REGISTRY: Dict[str, List[dict]] = {}
_RESOLVED: Dict[str, Callable] = {}


def register(
    op: str,
    impl: Callable,
    *,
    name: str,
    predicate: Optional[Callable[[], bool]] = None,
):
    """Register an implementation for an op slot (newest wins when eligible)."""
    _REGISTRY.setdefault(op, []).insert(
        0, {"name": name, "impl": impl, "predicate": predicate}
    )
    _RESOLVED.pop(op, None)


def unregister(op: str, name: str):
    _REGISTRY[op] = [e for e in _REGISTRY.get(op, []) if e["name"] != name]
    _RESOLVED.pop(op, None)


def clear_cache():
    _RESOLVED.clear()


def _forced() -> str:
    return os.environ.get("LIBREYOLO_QUANT_KERNELS", "").strip().lower()


def resolve(op: str) -> Optional[Callable]:
    """Return the active implementation for an op, or None if none eligible."""
    forced = _forced()
    cache_key = f"{op}|{forced}"
    if cache_key in _RESOLVED:
        return _RESOLVED[cache_key]

    impl = None
    for entry in _REGISTRY.get(op, ()):
        if forced in ("off", "reference") and entry["name"] != "reference":
            continue
        if forced and forced not in ("off", "reference") and entry["name"] not in (
            forced,
            "reference",
        ):
            continue
        predicate = entry["predicate"]
        try:
            if predicate is None or predicate():
                impl = entry["impl"]
                break
        except Exception as exc:  # a broken predicate must never break inference
            logger.warning(
                "Kernel predicate failed for %s/%s: %s", op, entry["name"], exc
            )
    _RESOLVED[cache_key] = impl
    return impl


def active() -> Dict[str, str]:
    """Map of op slot to the name of the implementation currently selected."""
    out = {}
    for op, entries in _REGISTRY.items():
        impl = resolve(op)
        out[op] = next(
            (e["name"] for e in entries if e["impl"] is impl), "unavailable"
        )
    return out


def _make_proxy(op: str) -> Callable:
    def proxy(*args, **kwargs):
        impl = resolve(op)
        if impl is None:
            raise RuntimeError(f"No implementation available for quant op '{op}'.")
        return impl(*args, **kwargs)

    proxy.__name__ = op
    return proxy


# Reference implementations: registered first so they sit last in the list
# (always the fallback) and are selectable via the "reference" name.
for _op in REFERENCE_OPS:
    register(_op, getattr(_reference, _op), name="reference")
    globals()[_op] = _make_proxy(_op)


# Optional out-of-tree compiled kernels (e.g. the CUTLASS NVFP4 GEMM).
# The package self-registers on import; absence is the normal case.
if importlib.util.find_spec("libreyolo_kernels") is not None:
    try:
        import libreyolo_kernels  # noqa: F401
    except Exception as exc:
        logger.warning("libreyolo_kernels present but failed to import: %s", exc)
