"""RT-DETR (v1/v2/v4) and DEIMv2 LoRA fine-tuning tests.

Wave-2 coverage for ``lora=True``: the shared DETR recipe drops into the
RT-DETR family, RT-DETRv4 takes it by inheritance (its trainer subclasses
DEIMTrainer and its core is the D-FINE architecture), and DEIMv2 gets its own
recipe (SwiGLU ``w12``/``w3`` FFN targets, plus DINOv3 ``qkv`` adapters with a
frozen ViT on the S/M/L/X sizes). Models build offline (random init).
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.deimv2.model import LibreDEIMv2
from libreyolo.models.deimv2.trainer import DEIMv2Trainer
from libreyolo.models.rtdetr.model import LibreRTDETR
from libreyolo.models.rtdetr.trainer import RTDETRTrainer
from libreyolo.models.rtdetrv2.model import LibreRTDETRv2
from libreyolo.models.rtdetrv2.trainer import RTDETRv2Trainer
from libreyolo.models.rtdetrv4.model import LibreRTDETRv4
from libreyolo.models.rtdetrv4.trainer import RTDETRv4Trainer
from libreyolo.training.lora import (
    apply_lora_to_deimv2,
    apply_lora_to_detr,
    module_has_lora,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.rtdetr,
    pytest.mark.deimv2,
    pytest.mark.slow,
]

pytest.importorskip("peft")

IMGSZ = 640

# family -> (wrapper cls, trainer cls, size, apply fn)
_FAMILIES = {
    "rtdetr": (LibreRTDETR, RTDETRTrainer, "r18", apply_lora_to_detr),
    "rtdetrv2": (LibreRTDETRv2, RTDETRv2Trainer, "r18", apply_lora_to_detr),
    "rtdetrv4": (LibreRTDETRv4, RTDETRv4Trainer, "s", apply_lora_to_detr),
    "deimv2-pico": (LibreDEIMv2, DEIMv2Trainer, "pico", apply_lora_to_deimv2),
    "deimv2-s": (LibreDEIMv2, DEIMv2Trainer, "s", apply_lora_to_deimv2),
}


def _build_trainer(family: str, lora: bool = True):
    model_cls, trainer_cls, size, _ = _FAMILIES[family]
    wrapper = model_cls(None, size=size, device="cpu")
    wrapper.model.train()
    trainer = trainer_cls(
        model=wrapper.model,
        wrapper_model=wrapper,
        size=size,
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=IMGSZ,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
        lora=lora,
    )
    return trainer, wrapper


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_lora_injects_freezes_zones_and_reduces_trainable_params(family):
    trainer, wrapper = _build_trainer(family)
    core = wrapper.model

    trainable_before = sum(p.numel() for p in core.parameters() if p.requires_grad)
    trainer.on_setup()
    trainable_after = sum(p.numel() for p in core.parameters() if p.requires_grad)

    assert module_has_lora(core)
    assert trainable_after < trainable_before

    names = dict(core.named_parameters())
    lora_params = {n: p for n, p in names.items() if "lora_" in n}
    assert lora_params, "no LoRA adapters were created"
    assert all(p.requires_grad for p in lora_params.values())

    # transformer block base weights: frozen
    base_in_blocks = [(n, p) for n, p in names.items() if ".base_layer." in n]
    assert base_in_blocks
    assert all(not p.requires_grad for _, p in base_in_blocks)

    # detection head: still trainable
    assert any(p.requires_grad for n, p in names.items() if "dec_score_head" in n)

    if family == "deimv2-s":
        # DINOv3 ViT base: frozen; STA conv pyramid around it: trainable
        dinov3 = [
            (n, p)
            for n, p in names.items()
            if n.startswith("backbone.dinov3.") and "lora_" not in n
        ]
        assert dinov3
        assert all(not p.requires_grad for _, p in dinov3)
        sta = [
            (n, p)
            for n, p in names.items()
            if n.startswith("backbone.") and not n.startswith("backbone.dinov3.")
        ]
        assert sta, "expected STA params outside backbone.dinov3"
        assert any(p.requires_grad for _, p in sta), "STA must stay trainable"
    else:
        backbone = [(n, p) for n, p in names.items() if n.startswith("backbone.")]
        assert backbone
        assert all(not p.requires_grad for _, p in backbone)


def test_deimv2_dino_recipe_targets_qkv_and_swiglu():
    trainer, wrapper = _build_trainer("deimv2-s")
    trainer.on_setup()
    names = [n for n, _ in wrapper.model.named_parameters()]

    # DINOv3 fused attention qkv is adapted inside the frozen ViT
    assert any("backbone.dinov3." in n and ".qkv.lora_A" in n for n in names)
    # SwiGLU FFN Linears are adapted in the decoder
    assert any(".w12.lora_A" in n for n in names)
    assert any(".w3.lora_A" in n for n in names)
    # attention output proj and MHA self-attention are NOT adapted
    assert not any(".proj.lora_A" in n for n in names)
    assert not any("self_attn" in n and "lora_" in n for n in names)
    # plain LoRA, not DoRA
    assert not any("lora_magnitude" in n for n in names)


def test_rtdetrv4_supports_lora_by_inheritance():
    """RTDETRv4Trainer subclasses DEIMTrainer and the core is the D-FINE
    architecture, so lora=True must work through plain inheritance."""
    assert RTDETRv4Trainer.supports_lora is True
    trainer, wrapper = _build_trainer("rtdetrv4")
    trainer.on_setup()
    assert module_has_lora(wrapper.model)


@pytest.mark.parametrize("family", ["rtdetrv2", "deimv2-s"])
def test_lora_backward_updates_adapters_and_leaves_base_frozen(family):
    torch.manual_seed(0)
    trainer, wrapper = _build_trainer(family)
    trainer.on_setup()
    core = wrapper.model
    params = dict(core.named_parameters())

    base_name = next(n for n in params if ".base_layer.weight" in n)
    base_ref = params[base_name].detach().clone()
    lora_B_name = next(n for n in params if "lora_B" in n)
    assert torch.count_nonzero(params[lora_B_name]) == 0

    optimizer = trainer._setup_optimizer()
    imgs = torch.randn(2, 3, IMGSZ, IMGSZ)
    targets = torch.zeros(2, 120, 5)
    if family.startswith("rtdetr"):
        # RT-DETR targets are [cls, x1, y1, x2, y2] in pixels
        targets[0, 0] = torch.tensor([0.0, 100.0, 100.0, 300.0, 260.0])
        targets[1, 0] = torch.tensor([1.0, 200.0, 180.0, 420.0, 400.0])
    else:
        # DEIM-style targets are [cls, cx, cy, w, h] in pixels
        targets[0, 0] = torch.tensor([0.0, 320.0, 320.0, 120.0, 90.0])
        targets[1, 0] = torch.tensor([1.0, 400.0, 330.0, 90.0, 75.0])

    out = trainer.on_forward(imgs, targets)
    loss = out["total_loss"]
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    assert params[lora_B_name].grad is not None
    optimizer.step()

    assert torch.count_nonzero(params[lora_B_name]) > 0
    assert torch.allclose(params[base_name].detach(), base_ref)


@pytest.mark.parametrize("family", ["rtdetr", "deimv2-s"])
def test_lora_checkpoint_reloads_into_fresh_model(family, tmp_path):
    """A lora=True checkpoint must load into a fresh model: the loader replays
    the adapter injection so the peft keys line up (RT-DETR loads non-strictly,
    so without the replay the adapters would be dropped silently)."""
    model_cls, _, size, apply_fn = _FAMILIES[family]

    w1 = model_cls(None, size=size, device="cpu")
    apply_fn(w1.model)
    sd = dict(w1.model.state_dict())
    assert any("lora_" in k for k in sd), "checkpoint should carry adapter keys"
    ckpt_path = tmp_path / "last.pt"
    torch.save({"model": sd}, ckpt_path)

    w2 = model_cls(None, size=size, device="cpu")
    assert not module_has_lora(w2.model)
    w2._load_weights(str(ckpt_path))
    assert module_has_lora(w2.model), (
        "loader did not replay LoRA injection for an adapter checkpoint"
    )
    # the replayed adapters must carry the trained values, not fresh zeros:
    # non-strict loading would silently drop them otherwise
    loaded = dict(w2.model.named_parameters())
    src_A_name, src_A = next(
        (n, p) for n, p in w1.model.named_parameters() if "lora_A" in n
    )
    assert torch.equal(loaded[src_A_name].detach(), src_A.detach())


@pytest.mark.parametrize("family", ["rtdetrv2", "deimv2-pico", "deimv2-s"])
def test_lora_apply_is_idempotent(family):
    model_cls, _, size, apply_fn = _FAMILIES[family]
    wrapper = model_cls(None, size=size, device="cpu")
    apply_fn(wrapper.model)
    n_before = sum(1 for n, _ in wrapper.model.named_parameters() if "lora_" in n)

    result = apply_fn(wrapper.model)

    assert result is wrapper.model
    n_after = sum(1 for n, _ in wrapper.model.named_parameters() if "lora_" in n)
    assert n_after == n_before
