"""LibreViT exact pretrained-logit parity against the pinned timm reference."""

from __future__ import annotations

import gc

import pytest
import torch

pytestmark = [pytest.mark.external_data, pytest.mark.network]

TAGS = {
    "ti": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "s": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "b": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "l": "vit_large_patch16_224.augreg_in21k_ft_in1k",
}


@pytest.mark.parametrize("size", list(TAGS))
def test_timm_pretrained_parity(size):
    """Every shipped checkpoint loads strictly and produces identical logits."""
    timm = pytest.importorskip("timm")
    from libreyolo.models.vit.nn import VisionTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = timm.create_model(TAGS[size], pretrained=True).eval()
    native = VisionTransformer(size=size, num_classes=1000, init_weights=False).eval()
    result = native.load_state_dict(reference.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys

    reference = reference.to(device)
    native = native.to(device)
    generator = torch.Generator(device=device).manual_seed(2020)
    image = torch.randn(1, 3, 224, 224, generator=generator, device=device)
    with torch.inference_mode():
        expected = reference(image)
        actual = native(image)

    max_abs_diff = (expected - actual).abs().max().item()
    print(f"size={size} device={device} max_abs_diff={max_abs_diff}")
    assert max_abs_diff == 0.0

    del reference, native, image, expected, actual
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
