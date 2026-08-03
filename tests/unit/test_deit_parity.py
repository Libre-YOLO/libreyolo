"""LibreDeiT to timm exact pretrained inference parity acceptance gate.

The timm checkpoints are external Apache-2.0 data, so this test is excluded
from the offline PR gate. It validates every shipped size against timm 1.0.28
using a strict state-dict load and bit-identical eval logits.
"""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.external_data, pytest.mark.network]

TAGS = {
    "t": "deit_tiny_patch16_224.fb_in1k",
    "s": "deit_small_patch16_224.fb_in1k",
    "b": "deit_base_patch16_224.fb_in1k",
}
SOURCES = {
    "t": (
        "timm/deit_tiny_patch16_224.fb_in1k",
        "80e968688553f219e4a86f940ed945a23709c16f",
        "21d4764d94f6c3ffdb6da3581115a0a1ee2d505537d96883b540e54766407c9e",
    ),
    "s": (
        "timm/deit_small_patch16_224.fb_in1k",
        "91327a9c99f98fe6b524cd4d397b7226b80e1365",
        "1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b",
    ),
    "b": (
        "timm/deit_base_patch16_224.fb_in1k",
        "b78cc5532a69df6bcad9c3a8d76653fd20b31ac6",
        "cd2da27b74ed7f68b599f16c77af3e1e80f01c75f9ad96029d22ce747a247e8e",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.parametrize("size", list(TAGS))
def test_timm_pretrained_parity(size):
    timm = pytest.importorskip("timm")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    safetensors = pytest.importorskip("safetensors.torch")
    from libreyolo.models.deit.nn import DeiT

    repo_id, revision, expected_sha256 = SOURCES[size]
    source_path = Path(
        huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            revision=revision,
            token=False,
        )
    )
    assert _sha256(source_path) == expected_sha256

    reference = timm.create_model(TAGS[size], pretrained=False).eval()
    assert reference.pretrained_cfg.get("license") == "apache-2.0"
    assert reference.pretrained_cfg.get("crop_pct") == 0.9
    assert reference.pretrained_cfg.get("interpolation") == "bicubic"

    reference_result = reference.load_state_dict(
        safetensors.load_file(source_path, device="cpu"), strict=True
    )
    assert not reference_result.missing_keys and not reference_result.unexpected_keys

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
