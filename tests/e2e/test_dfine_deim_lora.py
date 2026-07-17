"""D-FINE / DEIM LoRA fine-tuning tests.

Exercise the real user path (``lora=True``) through both trainers: ``on_setup``
injects plain LoRA adapters into the transformer encoder/decoder Linears, the
CNN backbone and the transformer base weights are frozen, the optimizer only
collects trainable params, and a forward + backward step updates adapters while
leaving the base weights untouched. Models build offline (random init, no
downloads).
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.deim.model import LibreDEIM
from libreyolo.models.deim.trainer import DEIMTrainer
from libreyolo.models.dfine.model import LibreDFINE
from libreyolo.models.dfine.trainer import DFINETrainer
from libreyolo.training.lora import (
    apply_lora_to_detr,
    count_trainable_parameters,
    merge_lora_adapters,
    module_has_lora,
)

pytestmark = [pytest.mark.e2e, pytest.mark.dfine, pytest.mark.deim, pytest.mark.slow]

pytest.importorskip("peft")

IMGSZ = 640

_FAMILIES = {
    "dfine": (LibreDFINE, DFINETrainer),
    "deim": (LibreDEIM, DEIMTrainer),
}


def _build_trainer(family: str, lora: bool = True):
    model_cls, trainer_cls = _FAMILIES[family]
    # None model_path -> random init, no downloads. num_classes matches the
    # default 80-class heads so no data-driven rebuild is needed.
    wrapper = model_cls(None, size="n", device="cpu")
    wrapper.model.train()
    trainer = trainer_cls(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
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


def _synthetic_batch() -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.randn(2, 3, IMGSZ, IMGSZ)
    # padded targets (B, max_labels, 5) = [cls, cx, cy, w, h] in pixels
    targets = torch.zeros(2, 120, 5)
    targets[0, 0] = torch.tensor([0.0, 320.0, 320.0, 120.0, 90.0])
    targets[0, 1] = torch.tensor([1.0, 180.0, 150.0, 60.0, 60.0])
    targets[1, 0] = torch.tensor([1.0, 400.0, 330.0, 90.0, 75.0])
    return imgs, targets


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

    # CNN backbone: frozen entirely
    backbone = [(n, p) for n, p in names.items() if n.startswith("backbone.")]
    assert backbone
    assert all(not p.requires_grad for _, p in backbone)

    # transformer block base weights: frozen (dense weight moved under
    # .base_layer. by the injection)
    base_in_blocks = [
        (n, p) for n, p in names.items() if ".base_layer." in n
    ]
    assert base_in_blocks
    assert all(not p.requires_grad for _, p in base_in_blocks)

    # heads and encoder conv fusion: still trainable
    assert any(
        p.requires_grad for n, p in names.items() if "dec_score_head" in n
    ), "detection head must stay trainable"
    assert any(
        p.requires_grad
        for n, p in names.items()
        if n.startswith("encoder.") and "encoder.encoder" not in n
    ), "encoder conv fusion must stay trainable"


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_lora_recipe_targets_transformer_linears_only(family):
    trainer, wrapper = _build_trainer(family)
    trainer.on_setup()
    names = [n for n, _ in wrapper.model.named_parameters()]

    # FFN, gate, and deformable-attention projections are adapted
    assert any(".linear1.lora_A" in n for n in names)
    assert any(".linear2.lora_A" in n for n in names)
    assert any("gateway.gate.lora_A" in n for n in names)
    assert any("sampling_offsets.lora_A" in n for n in names)
    # MHA self-attention is NOT adapted (out_proj would be silently bypassed)
    assert not any("self_attn" in n and "lora_" in n for n in names)
    # heads and backbone are NOT adapted
    assert not any("dec_score_head" in n and "lora_" in n for n in names)
    assert not any(n.startswith("backbone.") and "lora_" in n for n in names)
    # plain LoRA, not DoRA (zero-init decoder Linears break DoRA's norm)
    assert not any("lora_magnitude" in n for n in names)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_lora_optimizer_only_collects_trainable_params(family):
    trainer, wrapper = _build_trainer(family)
    trainer.on_setup()
    optimizer = trainer._setup_optimizer()

    opt_params = [p for g in optimizer.param_groups for p in g["params"]]
    assert all(p.requires_grad for p in opt_params)

    core = wrapper.model
    opt_ids = {id(p) for p in opt_params}
    lora_ids = {id(p) for n, p in core.named_parameters() if "lora_" in n}
    assert opt_ids & lora_ids, "no LoRA params landed in the optimizer"
    frozen_ids = {id(p) for p in core.parameters() if not p.requires_grad}
    assert not (opt_ids & frozen_ids), "frozen params leaked into optimizer"


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_lora_backward_updates_adapters_and_leaves_base_frozen(family):
    torch.manual_seed(0)
    trainer, wrapper = _build_trainer(family)
    trainer.on_setup()
    core = wrapper.model
    params = dict(core.named_parameters())

    base_name = next(n for n in params if n.endswith("linear1.base_layer.weight"))
    base_ref = params[base_name].detach().clone()
    lora_B_name = next(n for n in params if "lora_B" in n)
    assert torch.count_nonzero(params[lora_B_name]) == 0

    optimizer = trainer._setup_optimizer()
    imgs, targets = _synthetic_batch()
    out = trainer.on_forward(imgs, targets)
    loss = out["total_loss"]
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    assert params[lora_B_name].grad is not None
    optimizer.step()

    assert torch.count_nonzero(params[lora_B_name]) > 0
    assert torch.allclose(params[base_name].detach(), base_ref)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_lora_merge_preserves_layer_math_and_removes_adapters(family):
    """Merging must fold adapters into dense weights exactly.

    Layer equivalence is asserted per adapted Linear: full-model output
    equality is not a usable oracle for DETRs, whose top-k query selection
    reorders under any epsilon-level numeric change.
    """
    torch.manual_seed(0)
    trainer, wrapper = _build_trainer(family)
    trainer.on_setup()
    core = wrapper.model

    # one optimizer step so the adapters are non-trivial before merging
    optimizer = trainer._setup_optimizer()
    imgs, targets = _synthetic_batch()
    out = trainer.on_forward(imgs, targets)
    out["total_loss"].backward()
    optimizer.step()

    core.eval()
    from peft.tuners.tuners_utils import BaseTunerLayer

    probes = {}
    with torch.no_grad():
        for name, sub in core.named_modules():
            if isinstance(sub, BaseTunerLayer):
                x = torch.randn(4, sub.get_base_layer().in_features)
                probes[name] = (x, sub(x))
    assert probes, "no adapted layers found to probe"

    n_merged = merge_lora_adapters(core)
    assert n_merged == len(probes)
    assert not any("lora_" in n for n, _ in core.named_parameters())
    assert not any(".base_layer." in n for n, _ in core.named_parameters())

    with torch.no_grad():
        for name, (x, expected) in probes.items():
            dense = core.get_submodule(name)
            assert isinstance(dense, torch.nn.Linear)
            assert not isinstance(dense, BaseTunerLayer)
            torch.testing.assert_close(dense(x), expected, atol=1e-5, rtol=1e-5)

    # the merged model still runs end to end and produces finite outputs
    with torch.no_grad():
        outputs = core(torch.randn(1, 3, IMGSZ, IMGSZ))
    assert torch.isfinite(outputs["pred_logits"]).all()
    assert torch.isfinite(outputs["pred_boxes"]).all()


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_lora_checkpoint_reloads_into_fresh_model(family, tmp_path):
    """A lora=True checkpoint must load into a fresh model: the loader replays
    the adapter injection so the peft keys line up instead of being rejected as
    unexpected."""
    model_cls, _ = _FAMILIES[family]

    w1 = model_cls(None, size="n", device="cpu")
    apply_lora_to_detr(w1.model)
    sd = dict(w1.model.state_dict())
    assert any("lora_" in k for k in sd), "checkpoint should carry adapter keys"
    ckpt_path = tmp_path / "last.pt"
    torch.save({"model": sd}, ckpt_path)

    w2 = model_cls(None, size="n", device="cpu")
    assert not module_has_lora(w2.model)
    w2._load_weights(str(ckpt_path))  # must not raise on the adapter keys
    assert module_has_lora(w2.model), (
        "loader did not replay LoRA injection for an adapter checkpoint"
    )


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_lora_apply_is_idempotent(family):
    model_cls, _ = _FAMILIES[family]
    wrapper = model_cls(None, size="n", device="cpu")
    apply_lora_to_detr(wrapper.model)
    n_before = sum(1 for n, _ in wrapper.model.named_parameters() if "lora_" in n)
    trainable_before, _ = count_trainable_parameters(wrapper.model)

    result = apply_lora_to_detr(wrapper.model)

    assert result is wrapper.model
    n_after = sum(1 for n, _ in wrapper.model.named_parameters() if "lora_" in n)
    trainable_after, _ = count_trainable_parameters(wrapper.model)
    assert n_after == n_before
    assert trainable_after == trainable_before


def test_lora_rejected_for_dfine_segment_task():
    """The mask head path is untested with adapters; segment must hard-error."""
    wrapper = LibreDFINE(None, size="n", device="cpu", task="segment")
    trainer = DFINETrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=IMGSZ,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
        lora=True,
    )
    with pytest.raises(ValueError, match="segment.*lora|lora.*segment"):
        trainer.on_setup()
