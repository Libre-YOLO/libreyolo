"""LibreDeiT to timm exact pretrained inference parity acceptance gate.

The timm checkpoints are external Apache-2.0 data, so this test is excluded
from the offline PR gate. It validates every shipped size against timm 1.0.28
using a strict state-dict load and bit-identical eval logits.
"""

from __future__ import annotations

import gc

import pytest
import torch

pytestmark = [pytest.mark.external_data, pytest.mark.network]

TAGS = {
    "t": "deit_tiny_patch16_224.fb_in1k",
    "s": "deit_small_patch16_224.fb_in1k",
    "b": "deit_base_patch16_224.fb_in1k",
}


@pytest.mark.parametrize("size", list(TAGS))
def test_timm_pretrained_parity(size):
    timm = pytest.importorskip("timm")
    from libreyolo.models.deit.nn import DeiT

    reference = timm.create_model(TAGS[size], pretrained=True).eval()
    assert reference.pretrained_cfg.get("license") == "apache-2.0"
    assert reference.pretrained_cfg.get("crop_pct") == 0.9
    assert reference.pretrained_cfg.get("interpolation") == "bicubic"

    native = DeiT(size=size, num_classes=1000)
    result = native.load_state_dict(reference.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    native.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference.to(device)
    native.to(device)
    torch.manual_seed(0)
    image = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        reference_logits = reference(image)
        native_logits = native(image)

    max_abs_diff = (reference_logits - native_logits).abs().max().item()
    assert max_abs_diff == 0.0, f"{size}: max_abs_diff={max_abs_diff}"

    del reference, native, image, reference_logits, native_logits
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
