"""LibreCLIP <-> open_clip parity (the native-port equivalent of max_abs_diff==0).

Skipped unless ``open_clip`` is installed (the ``libreyolo[clip-convert]`` extra)
and the converted LibreCLIP checkpoint is present. These tests download/​load
real LAION-2B weights, so they are marked ``external_data`` + ``network``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

open_clip = pytest.importorskip("open_clip")

from libreyolo.models.clip.tokenizer import SimpleTokenizer  # noqa: E402

pytestmark = [pytest.mark.external_data, pytest.mark.network, pytest.mark.clip]

_REPO_ROOT = Path(__file__).resolve().parents[2]

# size -> (libreyolo checkpoint, open_clip arch, pretrained tag)
_SPECS = {
    "b32": ("LibreCLIPb32-cls.pt", "ViT-B-32", "laion2b_s34b_b79k"),
    "b16": ("LibreCLIPb16-cls.pt", "ViT-B-16", "laion2b_s34b_b88k"),
}

_CORPUS = [
    "a photo of a cat.",
    "an empty aisle",
    "A SPILL on the FLOOR!",
    "tench, Tinca tinca",
    "n01440764",
    "café — naïve façade 2024 #1",
    "it's a bird's-eye view",
]


def test_tokenizer_parity():
    """Vendored BPE must produce byte-exact token IDs vs open_clip."""
    oc = open_clip.get_tokenizer("ViT-B-32")
    mine = SimpleTokenizer()
    assert torch.equal(oc(_CORPUS), mine(_CORPUS))


@pytest.mark.parametrize("size", list(_SPECS))
def test_forward_parity(size):
    ckpt_name, arch, pretrained = _SPECS[size]
    ckpt = _REPO_ROOT / "weights" / ckpt_name
    if not ckpt.exists():
        pytest.skip(f"{ckpt_name} not converted (run weights/convert_clip_weights.py)")

    from libreyolo import LibreCLIP

    oc_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    oc_model.eval()
    oc_tok = open_clip.get_tokenizer(arch)

    model = LibreCLIP(str(ckpt), size=size, device="cpu")
    labels = ["a forklift", "an empty aisle", "a spill", "a cat"]
    model.set_classes(labels)

    img = torch.randn(2, 3, model.input_size, model.input_size)
    tokens = oc_tok([f"a photo of a {l}." for l in labels])

    with torch.no_grad():
        my_img = model.model.encode_image(img)
        oc_img = oc_model.encode_image(img)
        my_txt = model.model.encode_text(tokens)
        oc_txt = oc_model.encode_text(tokens)
        # full zero-shot logit path through the model's own _forward
        my_logits = model._forward(img)
        oc_text = F.normalize(oc_txt, dim=-1)
        oc_image = F.normalize(oc_img, dim=-1)
        oc_logits = oc_model.logit_scale.exp() * oc_image @ oc_text.t()

    assert (my_img - oc_img).abs().max().item() < 1e-4
    assert (my_txt - oc_txt).abs().max().item() < 1e-4
    assert (my_logits - oc_logits).abs().max().item() < 1e-3
