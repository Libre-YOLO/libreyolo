"""EC (EdgeCrafter) and ConvNeXt LoRA fine-tuning tests.

EC: DETR with a ViTAdapter backbone (plain ViT with fused ``qkv`` under
``backbone.backbone`` plus a trainable projector pyramid). ConvNeXt: conv
classifier whose blocks carry channels-last ``fc1``/``fc2`` Linears. Both
build offline (random init, no downloads).
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.convnext.model import LibreConvNeXt
from libreyolo.models.convnext.trainer import ConvNeXtTrainer
from libreyolo.models.ec.model import LibreEC
from libreyolo.models.ec.trainer import ECTrainer
from libreyolo.training.lora import (
    apply_lora_to_convnext,
    apply_lora_to_ec,
    merge_lora_adapters,
    module_has_lora,
)

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

pytest.importorskip("peft")

EC_IMGSZ = 640
CNX_IMGSZ = 224


def _build_ec_trainer(lora: bool = True):
    wrapper = LibreEC(None, size="s", device="cpu")
    wrapper.model.train()
    trainer = ECTrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="s",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=EC_IMGSZ,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
        lora=lora,
    )
    return trainer, wrapper


def _build_cnx_trainer(lora: bool = True):
    wrapper = LibreConvNeXt(None, size="t", device="cpu")
    wrapper.model.train()
    trainer = ConvNeXtTrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="t",
        num_classes=wrapper.nb_classes,
        data=None,
        epochs=1,
        batch=2,
        imgsz=CNX_IMGSZ,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
        lora=lora,
    )
    return trainer, wrapper


def test_ec_lora_injects_and_freezes_zones():
    trainer, wrapper = _build_ec_trainer()
    core = wrapper.model

    before = sum(p.numel() for p in core.parameters() if p.requires_grad)
    trainer.on_setup()
    after = sum(p.numel() for p in core.parameters() if p.requires_grad)

    assert module_has_lora(core)
    assert after < before
    names = dict(core.named_parameters())
    # ViT base frozen, qkv adapted inside it
    vit_base = [
        p for n, p in names.items()
        if n.startswith("backbone.backbone.") and "lora_" not in n
    ]
    assert vit_base and all(not p.requires_grad for p in vit_base)
    assert any(
        "backbone.backbone." in n and ".qkv.lora_A" in n for n in names
    ), "ViT qkv must carry adapters"
    # ViTAdapter projector pyramid stays trainable
    projector = [
        p for n, p in names.items()
        if n.startswith("backbone.") and not n.startswith("backbone.backbone.")
    ]
    assert projector and any(p.requires_grad for p in projector)
    # decoder blocks: base frozen, adapters trainable, heads dense-trainable
    assert any(".base_layer." in n and "decoder" in n for n in names)
    assert any(p.requires_grad for n, p in names.items() if "dec_score_head" in n)
    # attention output proj and MHA are NOT adapted
    assert not any(".proj.lora_A" in n for n in names)
    assert not any("self_attn" in n and "lora_" in n for n in names)


def test_ec_lora_backward_updates_adapters_and_leaves_base_frozen():
    torch.manual_seed(0)
    trainer, wrapper = _build_ec_trainer()
    trainer.on_setup()
    core = wrapper.model
    params = dict(core.named_parameters())

    base_name = next(n for n in params if ".base_layer.weight" in n)
    base_ref = params[base_name].detach().clone()
    lora_B_name = next(n for n in params if "lora_B" in n)
    assert torch.count_nonzero(params[lora_B_name]) == 0

    optimizer = trainer._setup_optimizer()
    imgs = torch.randn(2, 3, EC_IMGSZ, EC_IMGSZ)
    targets = torch.zeros(2, 120, 5)
    targets[0, 0] = torch.tensor([0.0, 320.0, 320.0, 120.0, 90.0])
    targets[1, 0] = torch.tensor([1.0, 400.0, 330.0, 90.0, 75.0])

    out = trainer.on_forward(imgs, targets)
    assert torch.isfinite(out["total_loss"])
    optimizer.zero_grad()
    out["total_loss"].backward()
    assert params[lora_B_name].grad is not None
    optimizer.step()

    assert torch.count_nonzero(params[lora_B_name]) > 0
    assert torch.allclose(params[base_name].detach(), base_ref)


def test_ec_lora_checkpoint_reloads_into_fresh_model(tmp_path):
    w1 = LibreEC(None, size="s", device="cpu")
    apply_lora_to_ec(w1.model)
    sd = dict(w1.model.state_dict())
    assert any("lora_" in k for k in sd)
    ckpt_path = tmp_path / "last.pt"
    torch.save({"model": sd}, ckpt_path)

    w2 = LibreEC(None, size="s", device="cpu")
    assert not module_has_lora(w2.model)
    w2._load_weights(str(ckpt_path))
    assert module_has_lora(w2.model)


def test_ec_lora_rejected_for_non_detect_tasks():
    trainer, wrapper = _build_ec_trainer()
    wrapper.task = "segment"
    with pytest.raises(ValueError, match="lora"):
        trainer.on_setup()


def test_convnext_lora_injects_freezes_backbone_and_keeps_head():
    trainer, wrapper = _build_cnx_trainer()
    core = wrapper.model

    before = sum(p.numel() for p in core.parameters() if p.requires_grad)
    trainer.on_setup()
    after = sum(p.numel() for p in core.parameters() if p.requires_grad)

    assert module_has_lora(core)
    assert after < before
    names = dict(core.named_parameters())
    assert any(".fc1.lora_A" in n for n in names)
    assert any(".fc2.lora_A" in n for n in names)
    # convs, norms, layer scale frozen
    stages_base = [
        p for n, p in names.items()
        if n.startswith("stages.") and "lora_" not in n
    ]
    assert stages_base and all(not p.requires_grad for p in stages_base)
    stem = [p for n, p in names.items() if n.startswith("stem.")]
    assert stem and all(not p.requires_grad for p in stem)
    # classification head stays trainable
    head = [p for n, p in names.items() if n.startswith("head.")]
    assert head and all(p.requires_grad for p in head)
    # no adapters outside the block MLPs
    assert not any("conv_dw" in n and "lora_" in n for n in names)
    assert not any(n.startswith("head.") and "lora_" in n for n in names)


def test_convnext_lora_backward_updates_adapters_and_leaves_base_frozen():
    torch.manual_seed(0)
    trainer, wrapper = _build_cnx_trainer()
    trainer.on_setup()
    core = wrapper.model
    params = dict(core.named_parameters())

    base_name = next(n for n in params if ".base_layer.weight" in n)
    base_ref = params[base_name].detach().clone()
    lora_B_name = next(n for n in params if "lora_B" in n)
    assert torch.count_nonzero(params[lora_B_name]) == 0

    optimizer = trainer._setup_optimizer()
    imgs = torch.randn(2, 3, CNX_IMGSZ, CNX_IMGSZ)
    labels = torch.tensor([1, 3])
    out = trainer.on_forward(imgs, labels)
    assert torch.isfinite(out["total_loss"])
    optimizer.zero_grad()
    out["total_loss"].backward()
    assert params[lora_B_name].grad is not None
    optimizer.step()

    assert torch.count_nonzero(params[lora_B_name]) > 0
    assert torch.allclose(params[base_name].detach(), base_ref)


def test_convnext_lora_merge_preserves_layer_math_and_model_runs():
    torch.manual_seed(0)
    trainer, wrapper = _build_cnx_trainer()
    trainer.on_setup()
    core = wrapper.model

    optimizer = trainer._setup_optimizer()
    out = trainer.on_forward(torch.randn(2, 3, CNX_IMGSZ, CNX_IMGSZ), torch.tensor([0, 2]))
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
    assert probes

    n_merged = merge_lora_adapters(core)
    assert n_merged == len(probes)
    assert not any("lora_" in n for n, _ in core.named_parameters())
    with torch.no_grad():
        for name, (x, expected) in probes.items():
            dense = core.get_submodule(name)
            torch.testing.assert_close(dense(x), expected, atol=1e-5, rtol=1e-5)
        logits = core(torch.randn(1, 3, CNX_IMGSZ, CNX_IMGSZ))
    assert torch.isfinite(logits).all()


def test_convnext_lora_checkpoint_reloads_into_fresh_model(tmp_path):
    w1 = LibreConvNeXt(None, size="t", device="cpu")
    apply_lora_to_convnext(w1.model)
    sd = dict(w1.model.state_dict())
    assert any("lora_" in k for k in sd)
    ckpt_path = tmp_path / "last.pt"
    torch.save({"model": sd}, ckpt_path)

    w2 = LibreConvNeXt(None, size="t", device="cpu")
    assert not module_has_lora(w2.model)
    w2._load_weights(str(ckpt_path))
    assert module_has_lora(w2.model)


def test_export_merges_lora_adapters_for_new_families(monkeypatch):
    """export() must fold live PEFT adapters into dense weights before tracing,
    so the exported graph carries no adapter modules / peft dependency."""
    import libreyolo.export as export_mod

    wrapper = LibreConvNeXt(None, size="t", device="cpu")
    apply_lora_to_convnext(wrapper.model)
    assert module_has_lora(wrapper.model)

    # Stub the actual exporter: we only assert the pre-export merge ran.
    captured = {}

    class _StubExporter:
        @staticmethod
        def create(fmt, model):
            captured["had_lora_at_export"] = module_has_lora(model.model)
            return lambda **kw: "stub.onnx"

    monkeypatch.setattr(export_mod, "BaseExporter", _StubExporter)

    out = wrapper.export(format="onnx")
    assert out == "stub.onnx"
    # merged before the exporter saw the model, and no adapters remain
    assert captured["had_lora_at_export"] is False
    assert not module_has_lora(wrapper.model)
    assert not any(".base_layer." in n for n, _ in wrapper.model.named_parameters())


@pytest.mark.parametrize("family", ["ec", "convnext"])
def test_lora_apply_is_idempotent(family):
    if family == "ec":
        wrapper = LibreEC(None, size="s", device="cpu")
        fn = apply_lora_to_ec
    else:
        wrapper = LibreConvNeXt(None, size="t", device="cpu")
        fn = apply_lora_to_convnext
    fn(wrapper.model)
    n_before = sum(1 for n, _ in wrapper.model.named_parameters() if "lora_" in n)
    result = fn(wrapper.model)
    assert result is wrapper.model
    n_after = sum(1 for n, _ in wrapper.model.named_parameters() if "lora_" in n)
    assert n_after == n_before
