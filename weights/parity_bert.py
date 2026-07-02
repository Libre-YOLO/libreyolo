"""BERT text-encoder native-port parity vs Grounding DINO's text backbone."""

from __future__ import annotations

import torch

from libreyolo.models.bert.nn import BertModel

REPO = "LibreYOLO/LibreGroundingDINOt"


def main():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proc = AutoProcessor.from_pretrained(REPO)
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(REPO, attn_implementation="eager").eval()
    ref = gd.model.text_backbone.eval()

    native = BertModel(gd.config.text_config).eval()
    sd = {k.replace("model.text_backbone.", ""): v
          for k, v in gd.state_dict().items() if k.startswith("model.text_backbone.")}
    res = native.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys if "position_ids" not in k]
    unexp = [k for k in res.unexpected_keys if "pooler" not in k and "position_ids" not in k]
    print("missing:", miss[:6], "| unexpected:", unexp[:6])
    if miss or unexp:
        raise SystemExit("state_dict key mismatch")

    inputs = proc(images=None, text="a cat. a remote control. a dog.", return_tensors="pt")
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    ttype = inputs.get("token_type_ids")

    with torch.no_grad():
        ref_out = ref(input_ids=ids, attention_mask=mask, token_type_ids=ttype).last_hidden_state
        our_out = native(ids, attention_mask=mask, token_type_ids=ttype)

    diff = (ref_out - our_out).abs().max().item()
    print(f"last_hidden_state shape={tuple(our_out.shape)}  max_abs_diff={diff:.3e}")
    print("\nBERT native parity:", "PASS" if diff < 1e-4 else "FAIL")
    if diff >= 1e-4:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
