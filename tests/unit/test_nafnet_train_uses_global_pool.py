"""NAFNet trains through plain global pooling, not TLC local pooling (Finding #60).

TLC (Test-time Local Converter) local pooling — ``NAFNetLocal`` — is an
*inference-time* technique whose window is fixed by a warm-up forward. The
published NAFNet recipe trains the plain global-average-pool network and only
applies TLC local pooling at test time. Training through the fixed-window local
pool is therefore a train/inference mismatch.

These tests prove:
  1. The module the trainer forwards through uses global average pooling
     (``nn.AdaptiveAvgPool2d(1)``), not the fixed-window TLC ``AvgPool2d``.
  2. TLC local pooling is restored after training for inference.
  3. A plain-(global-pool)-trained state_dict loads into the TLC/local inference
     model — the switch is weight-preserving (pooling ops carry no parameters).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from libreyolo.models.nafnet.nn import (
    AvgPool2d,
    NAFNet,
    NAFNetLocal,
    restore_local_pooling,
    use_global_pooling,
)
from libreyolo.models.nafnet.trainer import NAFNetTrainer

pytestmark = pytest.mark.unit


_CFG = dict(
    img_channel=3,
    width=8,
    middle_blk_num=1,
    enc_blk_nums=[1, 1],
    dec_blk_nums=[1, 1],
)
_TRAIN_SIZE = (1, 3, 64, 64)
_NUM_SCA = 5  # sum(enc) + middle + sum(dec) = 2 + 1 + 2


def _count(module: nn.Module, cls) -> int:
    return sum(1 for m in module.modules() if isinstance(m, cls))


def _make_local() -> NAFNetLocal:
    return NAFNetLocal(**_CFG, train_size=_TRAIN_SIZE)


def test_local_model_starts_with_tlc_pooling():
    local = _make_local()
    # Sanity: the inference model uses TLC local pooling, not global pooling.
    assert _count(local, AvgPool2d) == _NUM_SCA
    assert _count(local, nn.AdaptiveAvgPool2d) == 0


def test_trainer_forwards_through_global_pooling():
    local = _make_local()
    trainer = NAFNetTrainer(
        model=local, size="s", device="cpu", imgsz=64, epochs=1
    )

    trainer.on_setup()

    # The model the trainer will forward through must use global average pooling.
    assert _count(trainer.model, AvgPool2d) == 0
    assert _count(trainer.model, nn.AdaptiveAvgPool2d) == _NUM_SCA
    pools = [m for m in trainer.model.modules() if isinstance(m, nn.AdaptiveAvgPool2d)]
    assert all(p.output_size == 1 for p in pools)
    # on_setup mutates the shared model in place (so optimizer/EMA see it).
    assert trainer.model is local
    assert len(trainer._tlc_restore) == _NUM_SCA


def test_trainer_restores_tlc_pooling_for_inference():
    local = _make_local()
    trainer = NAFNetTrainer(
        model=local, size="s", device="cpu", imgsz=64, epochs=1
    )
    trainer.on_setup()
    assert _count(local, AvgPool2d) == 0  # switched to global for training

    trainer._restore_inference_pooling()

    # TLC local pooling is back for inference; the exact original modules return.
    assert _count(local, AvgPool2d) == _NUM_SCA
    assert _count(local, nn.AdaptiveAvgPool2d) == 0
    assert trainer._tlc_restore is None


def test_use_global_pooling_is_reversible_and_weight_preserving():
    local = _make_local()
    before = {k: v.clone() for k, v in local.state_dict().items()}

    restore = use_global_pooling(local)
    assert _count(local, AvgPool2d) == 0
    assert _count(local, nn.AdaptiveAvgPool2d) == _NUM_SCA

    restore_local_pooling(restore)
    assert _count(local, AvgPool2d) == _NUM_SCA
    assert _count(local, nn.AdaptiveAvgPool2d) == 0

    # Pooling ops carry no parameters — weights are untouched by the swap.
    after = local.state_dict()
    assert set(before) == set(after)
    for k in before:
        assert torch.equal(before[k], after[k])


def test_plain_trained_state_dict_loads_into_tlc_inference_model():
    plain = NAFNet(**_CFG)  # global-pool training model
    local = _make_local()  # TLC inference model

    # Identical parameter names — TLC only swaps the parameter-free pooling op.
    assert set(plain.state_dict()) == set(local.state_dict())

    # Simulate a plain-trained checkpoint with distinct weights.
    with torch.no_grad():
        for p in plain.parameters():
            p.normal_()

    # The plain-trained state_dict loads cleanly (strict) into the TLC model.
    incompatible = local.load_state_dict(plain.state_dict(), strict=False)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    local.load_state_dict(plain.state_dict(), strict=True)

    # ...and the reverse direction also holds.
    plain.load_state_dict(local.state_dict(), strict=True)


def test_tlc_equals_global_at_train_size_but_can_diverge_larger():
    """At the warm-up/train size TLC local pooling reduces to global pooling
    (base_size = 1.5x train size => kernel spans the whole feature map), so the
    fix is a no-op there. At a larger input the fixed TLC window is smaller than
    the feature map and diverges — exactly the training bug the fix removes."""
    torch.manual_seed(0)
    plain = NAFNet(**_CFG).eval()
    # NAFBlocks are identity at init (beta=gamma=0), which hides the sca pooling.
    # Randomize weights so the channel-attention pooling actually affects output.
    with torch.no_grad():
        for p in plain.parameters():
            p.normal_()
    local = _make_local().eval()
    local.load_state_dict(plain.state_dict(), strict=True)

    x_train = torch.rand(_TRAIN_SIZE)
    with torch.no_grad():
        assert torch.allclose(plain(x_train), local(x_train), atol=1e-5)

    x_big = torch.rand(1, 3, 192, 192)  # > base_size (1.5 * 64 = 96)
    with torch.no_grad():
        assert not torch.allclose(plain(x_big), local(x_big), atol=1e-4)
