"""Gated exact parity against the two pinned official MiDaS checkpoints."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

UPSTREAM_DIR = os.environ.get("MIDAS_UPSTREAM_DIR")
GEN_EFFICIENTNET_DIR = os.environ.get("MIDAS_GEN_EFFICIENTNET_DIR")
CHECKPOINT_DIR = os.environ.get("MIDAS_CHECKPOINT_DIR")

_READY = all(
    value and Path(value).is_dir()
    for value in (UPSTREAM_DIR, GEN_EFFICIENTNET_DIR, CHECKPOINT_DIR)
)

pytestmark = [
    pytest.mark.midas,
    pytest.mark.external_data,
    pytest.mark.skipif(
        not _READY,
        reason=(
            "Set MIDAS_UPSTREAM_DIR, MIDAS_GEN_EFFICIENTNET_DIR, and "
            "MIDAS_CHECKPOINT_DIR to the pinned local artifacts"
        ),
    ),
]


def test_official_midas_s_and_l_exact_parity():
    command = [
        sys.executable,
        "weights/parity_midas.py",
        "--upstream-repo",
        UPSTREAM_DIR,
        "--gen-efficientnet-repo",
        GEN_EFFICIENTNET_DIR,
        "--checkpoint-dir",
        CHECKPOINT_DIR,
        "--sizes",
        "s",
        "l",
    ]
    subprocess.run(command, check=True, cwd=Path(__file__).parents[2])
