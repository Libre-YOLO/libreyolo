"""Back-compat shim: the kernel registry moved to :mod:`libreyolo.kernels`.

Kept so ``from libreyolo.quant import kernels`` and
``libreyolo.quant.kernels.active()`` keep working. Attribute access is
forwarded lazily (PEP 562) because this module participates in the
``libreyolo.kernels`` <-> ``libreyolo.quant`` import cycle: at shim import
time the registry module may be mid-initialization, so names must not be
bound eagerly.
"""

import libreyolo.kernels as _registry


def __getattr__(name):
    return getattr(_registry, name)


def __dir__():
    return dir(_registry)
