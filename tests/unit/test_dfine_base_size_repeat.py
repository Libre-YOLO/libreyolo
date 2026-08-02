"""D-FINE multi-scale ``base_size_repeat``: per-size upstream defaults.

The D-FINE trainer used to hardcode ``base_size_repeat=3`` for every size,
while upstream's custom fine-tune configs pin it per size (n disables
multi-scale, s 20, m 6, l 4, x 3) — only X coincidentally matched. These
tests pin the mapping to the upstream values, the resolution order
(explicit config override > per-size default > legacy 3), and the wiring
from trainer kwargs down to the collate. See issue #675.
"""

from __future__ import annotations

import pytest

from libreyolo.data.augment.detr import DetrMultiScaleCollate
from libreyolo.training.config import (
    DFINE_BASE_SIZE_REPEAT,
    DFINEConfig,
    resolve_dfine_base_size_repeat,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_config_default_is_unset():
    """None means "use the per-size upstream default", decided at trainer time."""
    assert DFINEConfig().base_size_repeat is None


def test_mapping_matches_upstream_custom_configs():
    """Peterande/D-FINE configs/dfine/custom/*.yml, verified 2026-08."""
    assert DFINE_BASE_SIZE_REPEAT == {"n": None, "s": 20, "m": 6, "l": 4, "x": 3}


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "expected"),
    [("n", None), ("s", 20), ("m", 6), ("l", 4), ("x", 3)],
)
def test_per_size_defaults(size, expected):
    assert resolve_dfine_base_size_repeat(size) == expected


def test_size_lookup_is_case_insensitive():
    assert resolve_dfine_base_size_repeat("S") == 20


def test_explicit_override_wins_even_on_n():
    """A user's config value beats the per-size default, including n's None."""
    assert resolve_dfine_base_size_repeat("n", override=7) == 7
    assert resolve_dfine_base_size_repeat("s", override=3) == 3


def test_unknown_size_falls_back_to_legacy_constant():
    """Sizes without an upstream recipe keep the previously hardcoded 3."""
    assert resolve_dfine_base_size_repeat("b") == 3
    assert resolve_dfine_base_size_repeat("") == 3


# ---------------------------------------------------------------------------
# Collate behavior at the two interesting values
# ---------------------------------------------------------------------------


def test_collate_with_none_repeat_keeps_batches_at_base_size():
    """None must mean fixed-size batches (upstream's ``~`` for the N size)."""
    collate = DetrMultiScaleCollate(base_size=640, base_size_repeat=None)
    assert collate.scales is None


def test_collate_with_s_repeat_weights_base_size_twenty_times():
    collate = DetrMultiScaleCollate(base_size=640, base_size_repeat=20)
    assert collate.scales.count(640) == 20
    # ±25% envelope in multiples of 32 must still be present around the base.
    assert min(collate.scales) == 480
    assert max(collate.scales) == 800


# ---------------------------------------------------------------------------
# Trainer wiring (kwargs -> config -> resolver)
# ---------------------------------------------------------------------------


def _make_trainer(**overrides):
    from libreyolo import LibreDFINE
    from libreyolo.models.dfine.trainer import DFINETrainer

    wrapper = LibreDFINE(None, size="n", device="cpu")
    kwargs = dict(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=640,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
    )
    kwargs.update(overrides)
    return DFINETrainer(**kwargs)


def test_trainer_resolves_per_size_default_for_n():
    """n trains at fixed size out of the box, exactly like upstream."""
    trainer = _make_trainer()
    assert trainer._train_base_size_repeat() is None


def test_trainer_honors_explicit_config_override():
    """base_size_repeat passed as a train kwarg flows through to the collate."""
    trainer = _make_trainer(base_size_repeat=7)
    assert trainer._train_base_size_repeat() == 7
