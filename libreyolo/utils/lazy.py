"""Deferred module imports.

LibreYOLO supports a lightweight ONNX-only deployment that has no torch wheel
installed (``pip install --no-deps libreyolo`` plus numpy/onnxruntime). The
ONNX inference path is numpy-native end to end, but the modules it lives in
also hold torch-based code for the default path. ``LazyModule`` lets those
modules keep writing ``torch.foo(...)`` normally while deferring the actual
import until the first torch-backed attribute is touched.

See https://github.com/LibreYOLO/libreyolo/discussions/711.
"""

from __future__ import annotations

import importlib
from types import ModuleType


class LazyModule(ModuleType):
    """Stand-in for a module that is imported on first attribute access.

    Behaves like the real module for every practical purpose (``torch.Tensor``,
    ``isinstance(x, torch.Tensor)``, ``torch.cuda.is_available()``), but a
    module that merely *imports* it pays nothing. The underlying module is
    resolved once and cached, so steady-state access is two dict lookups.

    Raises ``ModuleNotFoundError`` at first use if the module really is absent,
    which is the same error the caller would have seen from a normal import.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.__dict__["_lazy_target"] = name

    def __getattr__(self, attr: str):
        # Only invoked for names absent from __dict__, so the bookkeeping keys
        # below never recurse.
        target = self.__dict__.get("_lazy_target")
        if target is None or attr.startswith("_lazy_"):
            raise AttributeError(attr)
        module = self.__dict__.get("_lazy_module")
        if module is None:
            module = importlib.import_module(target)
            self.__dict__["_lazy_module"] = module
        return getattr(module, attr)

    def __dir__(self):
        try:
            return dir(importlib.import_module(self.__dict__["_lazy_target"]))
        except ImportError:
            return []


def module_available(name: str) -> bool:
    """Whether ``name`` can be imported, without keeping it loaded on failure."""
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def lazy_module(name: str) -> ModuleType:
    """Return a :class:`LazyModule` proxy for ``name``.

    If the module is already imported, hand back the real thing so callers
    never pay proxy overhead in the common (torch-installed) case.
    """
    import sys

    existing = sys.modules.get(name)
    if existing is not None:
        return existing  # type: ignore[return-value]
    return LazyModule(name)
