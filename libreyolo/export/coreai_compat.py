"""Compatibility shims for third-party Core AI toolchain defects.

Follows the same discipline as ``libreyolo/models/sam/transformers_compat.py``:
patch from our side rather than editing site-packages, verify the defect is
actually present before touching anything, and decline quietly if upstream
restructures so a future release cannot be silently broken by a stale shim.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

logger = logging.getLogger(__name__)

# aten.avg_pool2d signature:
#   (input, kernel_size, stride, padding, ceil_mode, count_include_pad,
#    divisor_override)
_AVG_POOL2D_ARITY = 7
_AFFECTED_COREAI_TORCH_VERSIONS = {"0.4.1"}


def _patch_avg_pool2d() -> bool:
    """Work around an off-by-one in coreai-torch's avg_pool2d resolver.

    ``coreai_torch/_aten_to_core.py`` reads ``count_include_pad`` as::

        node.args[5] if len(node.args) > 4 and node.args[4] is not None else True

    The guard tests element 4 but the read is element 5, so a node carrying
    exactly five arguments passes the guard and then raises ``IndexError:
    tuple index out of range``. RT-DETR emits avg_pool2d with five arguments,
    which killed conversion for the whole family before a single tensor was
    compared.

    Rather than reimplement the resolver, this normalises the node's argument
    tuple to full arity using PyTorch's documented defaults before handing it
    over, which keeps upstream's implementation in charge of the maths.
    """
    try:
        installed = version("coreai-torch")
    except PackageNotFoundError:
        return False
    if installed not in _AFFECTED_COREAI_TORCH_VERSIONS:
        # Never carry a private upstream patch into an unverified release.
        return False

    try:
        from coreai_torch import _aten_to_core as mod
    except ImportError:
        return False

    original = getattr(mod, "replace_avg_pool2d", None)
    if original is None or getattr(original, "_libreyolo_patched", False):
        return False

    def replace_avg_pool2d(values_map, node, loc, *args, **kwargs):
        current = tuple(node.args)
        if len(current) < _AVG_POOL2D_ARITY:
            kernel_size = current[1] if len(current) > 1 else None
            defaults = (
                None,  # 0 input, always present
                None,  # 1 kernel_size, always present
                kernel_size,  # 2 stride defaults to kernel_size
                [0, 0],  # 3 padding
                False,  # 4 ceil_mode
                True,  # 5 count_include_pad
                None,  # 6 divisor_override
            )
            node.args = current + defaults[len(current) :]
        try:
            return original(values_map, node, loc, *args, **kwargs)
        finally:
            node.args = current

    replace_avg_pool2d._libreyolo_patched = True

    mod.replace_avg_pool2d = replace_avg_pool2d
    # The resolver table captured the function by value at import time, so the
    # module attribute alone is not enough.
    table = getattr(mod, "_aten_to_core_resolver", None)
    if isinstance(table, dict):
        for key, value in list(table.items()):
            if value is original:
                table[key] = replace_avg_pool2d
    try:
        from coreai_torch import converter as conv

        conv_table = getattr(conv, "_aten_to_core_resolver", None)
        if isinstance(conv_table, dict):
            for key, value in list(conv_table.items()):
                if value is original:
                    conv_table[key] = replace_avg_pool2d
    except ImportError:
        pass
    return True


def apply() -> bool:
    """Apply every shim; returns True if any patch was installed."""
    try:
        patched = _patch_avg_pool2d()
    except Exception as exc:  # noqa: BLE001 - never break export over a shim
        logger.debug("Core AI compatibility shim declined: %s", exc)
        return False
    if patched:
        logger.debug("Applied Core AI avg_pool2d compatibility shim.")
    return patched


__all__ = ["apply"]
