"""Quantization kernels, split by purpose.

``simulate/`` holds fake-quantization kernels: numerics-true simulation with
STE backward, runnable on any device. They serve QAT/QAD *and* simulated
PTQ/val inference; the enforced boundary is ``is_finalized``, not
train-vs-deploy.

``execute/`` holds finalized-only real-precision paths: no backward, real
hardware (fp8 tensor cores, packed-weight unpack). ``quant/packing.py`` stays
in ``libreyolo/quant/`` because it is the checkpoint contract, never a
swappable kernel.
"""
