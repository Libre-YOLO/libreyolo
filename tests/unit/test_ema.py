import math

import pytest
import torch
import torch.nn as nn

from libreyolo.training.ema import ModelEMA


pytestmark = pytest.mark.unit


class _SharedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        shared = nn.Linear(2, 2, bias=False)
        self.first = shared
        self.second = shared


class _OverlappingBuffers(nn.Module):
    def __init__(self, device):
        super().__init__()
        base = torch.ones(4, device=device)
        self.register_buffer("base", base)
        self.register_buffer("exact_alias", base.view_as(base))
        self.register_buffer("tail", base[1:])


def _require_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")


def test_model_ema_uses_configurable_tau():
    model = nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.993, tau=100)

    assert ema.decay(1) == pytest.approx(0.993 * (1 - math.exp(-1 / 100)))


def test_model_ema_ramped_set_decay_keeps_configured_tau():
    model = nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.993, tau=100)

    ema.set_decay(0.9, ramp=True)

    assert ema.decay(1) == pytest.approx(0.9 * (1 - math.exp(-1 / 100)))


def test_model_ema_tau_zero_uses_constant_decay():
    model = nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.993, tau=0)

    assert ema.decay(1) == pytest.approx(0.993)

    ema.set_decay(0.9, ramp=True)

    assert ema.decay(1) == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        pytest.param("cpu", torch.float32, id="cpu-float32"),
        pytest.param("cpu", torch.float64, id="cpu-float64"),
        pytest.param("cuda", torch.float32, id="cuda-float32"),
        pytest.param("cuda", torch.float64, id="cuda-float64"),
        pytest.param("mps", torch.float32, id="mps-float32"),
    ],
)
def test_model_ema_updates_shared_parameter_once(device, dtype):
    _require_device(device)

    model = _SharedLinear().to(device=device, dtype=dtype)
    with torch.no_grad():
        model.first.weight.fill_(1.0)
    ema = ModelEMA(model, decay=0.5, tau=0)
    with torch.no_grad():
        ema.ema.first.weight.zero_()

    ema.update(model)

    assert ema.ema.first.weight is ema.ema.second.weight
    assert set(ema.ema.state_dict()) == {"first.weight", "second.weight"}
    assert ema.ema.first.weight.device.type == device
    assert ema.ema.first.weight.dtype == dtype
    assert torch.allclose(
        ema.ema.first.weight, torch.full_like(ema.ema.first.weight, 0.5)
    )
    restored = _SharedLinear().to(device=device, dtype=dtype)
    restored.load_state_dict(ema.ema.state_dict())
    assert restored.first.weight is restored.second.weight
    assert torch.equal(restored.first.weight, ema.ema.first.weight)


# PyTorch deep-copies distinct MPS tensor views with clone(), so the EMA
# destination tensors no longer overlap and are safe to update.
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_model_ema_rejects_distinct_overlapping_views(device):
    _require_device(device)
    model = _OverlappingBuffers(device)
    ema = ModelEMA(model, decay=0.5, tau=0)
    ema.ema.base.zero_()

    with pytest.raises(RuntimeError, match="share storage"):
        ema.update(model)


def test_model_ema_updates_rtmdet_shared_head_once():
    from libreyolo.models.rtmdet.nn import RTMDetSepBNHead

    model = RTMDetSepBNHead(
        num_classes=2,
        in_channels=4,
        feat_channels=4,
        stacked_convs=1,
        share_conv=True,
    )
    shared_weight = model.cls_convs[0][0].conv.weight
    with torch.no_grad():
        shared_weight.fill_(1.0)
    ema = ModelEMA(model, decay=0.5, tau=0)
    with torch.no_grad():
        ema.ema.cls_convs[0][0].conv.weight.zero_()

    ema.update(model)

    weights = [ema.ema.cls_convs[level][0].conv.weight for level in range(3)]
    assert weights[0] is weights[1] is weights[2]
    assert torch.allclose(weights[0], torch.full_like(weights[0], 0.5))
