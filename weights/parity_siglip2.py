"""Inference-parity proof for LibreSigLIP2 vs the reference SiglipModel.

Loads the Apache-2.0 google/siglip2-* weights into both the reference
``transformers`` ``SiglipModel`` (forced to eager attention) and our native
``LibreSigLIP2`` towers, feeds identical preprocessed pixel values and identical
tokenized text, and reports per-stage ``max_abs_diff`` at fp32 on CPU.

Result (held as the family's parity bar):

* Text tower and vision **encoder** (pre-pooling last_hidden_state) are
  **bit-identical** (max_abs_diff == 0).
* The pooled image feature and the final logits differ by < 1e-5. This residual
  comes entirely from ``torch.nn.MultiheadAttention``'s fused kernel in the MAP
  pooling head of the reference: its output is not bit-reproducible across module
  instances / call contexts (it depends on whether forward hooks are attached).
  Our port computes the pooling attention with explicit eager math, which is
  deterministic and version-stable. This is the "truly unavoidable" case the
  port-model skill allows: documented and capped well under 1e-5.

Also verifies our SentencePiece tokenizer reproduces the reference token IDs
exactly, which is what keeps the text tower bit-identical.

Requires ``libreyolo[siglip2-convert]`` (transformers + sentencepiece).

Usage::

    python weights/parity_siglip2.py --sizes b16          # fast
    python weights/parity_siglip2.py --sizes b16 so400m   # so400m is heavy on CPU
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from _conversion_utils import add_repo_root_to_path

add_repo_root_to_path()

from libreyolo.models.siglip2.nn import build_siglip2_model  # noqa: E402
from libreyolo.models.siglip2.tokenizer import SigLIP2Tokenizer  # noqa: E402

SIZE_TO_HF = {
    "b16": "google/siglip2-base-patch16-256",
    "so400m": "google/siglip2-so400m-patch14-384",
}

# Text-tower / vision-encoder stages must be exactly bit-identical.
EXACT_TOL = 0.0
# The MAP-pooling head carries an irreducible torch.nn.MultiheadAttention
# fused-kernel residual; cap it well under the skill's 1e-5 ceiling.
POOL_TOL = 1e-5


def _mad(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def check_size(size: str) -> bool:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    repo = SIZE_TO_HF[size]
    print(f"\n=== {size}  ({repo}) ===")
    ref = AutoModel.from_pretrained(repo, attn_implementation="eager").eval()
    mine = build_siglip2_model(size).eval()
    res = mine.load_state_dict(ref.state_dict(), strict=True)
    assert not res.missing_keys and not res.unexpected_keys, (res.missing_keys, res.unexpected_keys)
    print("state_dict load: 0 missing / 0 unexpected")

    proc = AutoProcessor.from_pretrained(repo)
    ref_tok = AutoTokenizer.from_pretrained(repo)
    my_tok = SigLIP2Tokenizer()

    texts = ["a photo of a cat", "a photo of a dog", "un gato", "eine Katze", "un chat"]
    my_ids = my_tok(texts)
    hf_ids = ref_tok(texts, padding="max_length", max_length=64, return_tensors="pt")["input_ids"]
    tok_match = torch.equal(my_ids, hf_ids)
    print(f"tokenizer IDs match reference: {tok_match}")
    assert tok_match, "tokenizer mismatch would break text-tower parity"

    rng = np.random.RandomState(0)
    img = Image.fromarray((rng.rand(340, 500, 3) * 255).astype("uint8"))
    pix = proc(images=img, return_tensors="pt")["pixel_values"]

    ok = True
    with torch.no_grad():
        # Text tower (must be exact).
        ref_txt = ref.get_text_features(input_ids=my_ids).pooler_output
        my_txt = mine.encode_text(my_ids)
        d_txt = _mad(ref_txt, my_txt)
        print(f"text_embeds            max_abs_diff = {d_txt:.3e}  (tol {EXACT_TOL})")
        ok &= d_txt <= EXACT_TOL

        # Vision encoder pre-pooling (must be exact).
        emb = ref.vision_model.embeddings(pix)
        ref_enc = ref.vision_model.post_layernorm(
            ref.vision_model.encoder(inputs_embeds=emb).last_hidden_state
        )
        my_emb = mine.vision_model.embeddings(pix)
        my_enc = mine.vision_model.post_layernorm(mine.vision_model.encoder(my_emb))
        d_enc = _mad(ref_enc, my_enc)
        print(f"vision encoder         max_abs_diff = {d_enc:.3e}  (tol {EXACT_TOL})")
        ok &= d_enc <= EXACT_TOL

        # Pooled image feature (MAP head; < 1e-5, torch MHA fused-kernel residual).
        ref_img = ref.get_image_features(pixel_values=pix).pooler_output
        my_img = mine.encode_image(pix)
        d_img = _mad(ref_img, my_img)
        print(f"image_embeds (pooled)  max_abs_diff = {d_img:.3e}  (tol {POOL_TOL})")
        ok &= d_img <= POOL_TOL

        # Full logits.
        ref_out = ref(input_ids=my_ids, pixel_values=pix)
        mi = torch.nn.functional.normalize(my_img, dim=-1)
        mt = torch.nn.functional.normalize(my_txt, dim=-1)
        my_lpi = (torch.matmul(mt, mi.t()) * mine.logit_scale.exp() + mine.logit_bias).t()
        d_log = _mad(ref_out.logits_per_image, my_lpi)
        print(f"logits_per_image       max_abs_diff = {d_log:.3e}  (tol {POOL_TOL})")
        ok &= d_log <= POOL_TOL

    print(f"[{size}] PARITY {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=["b16"], choices=list(SIZE_TO_HF))
    args = parser.parse_args()

    all_ok = True
    for size in args.sizes:
        all_ok &= check_size(size)
    if not all_ok:
        raise SystemExit("PARITY FAILED")
    print("\nAll parity checks passed.")


if __name__ == "__main__":
    main()
