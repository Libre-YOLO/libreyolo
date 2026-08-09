"""Finalized-only real-precision kernels (no backward, real hardware).

``scaled_mm_fp8`` is stock torch and is imported eagerly by the registry.
The triton modules (``fp8_fusion``, ``unpack_*``) are imported lazily by the
registry on first resolution, so this package must not import them here.
"""
