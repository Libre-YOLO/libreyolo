"""LibreSwin exact pretrained-logit parity against timm 1.0.28."""

from __future__ import annotations

import gc

import pytest
import torch

pytestmark = [pytest.mark.external_data, pytest.mark.network, pytest.mark.swin]

TAGS = {
    "t": "swin_tiny_patch4_window7_224.ms_in1k",
    "s": "swin_small_patch4_window7_224.ms_in1k",
    "b": "swin_base_patch4_window7_224.ms_in1k",
    "l": "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
}


@pytest.mark.parametrize("size", list(TAGS))
def test_timm_pretrained_parity(size):
    """All shipped classifiers load strictly and produce identical logits."""
    timm = pytest.importorskip("timm")
    from libreyolo.models.swin.classifier import SwinClassifier

    reference = timm.create_model(TAGS[size], pretrained=True).eval()
    for module in reference.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False

    native = SwinClassifier(size=size, num_classes=1000).eval()
    result = native.load_state_dict(reference.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference.to(device)
    native.to(device)
    generator = torch.Generator(device=device).manual_seed(0)
    inputs = torch.randn(1, 3, 224, 224, generator=generator, device=device)
    with torch.no_grad():
        reference_logits = reference(inputs)
        native_logits = native(inputs)
    max_abs_diff = (reference_logits - native_logits).abs().max().item()
    assert max_abs_diff == 0.0, f"{size}: max_abs_diff={max_abs_diff}"

    del reference, native, inputs, reference_logits, native_logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
