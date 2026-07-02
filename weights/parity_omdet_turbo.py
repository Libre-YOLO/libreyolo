"""OMDet-Turbo native-port parity vs transformers (staged, max_abs_diff)."""

from __future__ import annotations

import torch
from PIL import Image

from libreyolo.models.omdet_turbo.nn import OmDetTurboDetectionModel

REPO = "../491-add-omdet-turbo/weights/LibreOMDetTurbot"  # local snapshot


def remap(sd):
    """timm swin flattens stages to layers_N; native uses layers.N."""
    out = {}
    for k, v in sd.items():
        if "._backbone.layers_" in k:
            k = k.replace("._backbone.layers_", "._backbone.layers.", 1)
        out[k] = v
    return out


def main():
    from transformers import AutoProcessor, OmDetTurboForObjectDetection

    proc = AutoProcessor.from_pretrained(REPO)
    hf = OmDetTurboForObjectDetection.from_pretrained(REPO, attn_implementation="eager").eval()

    native = OmDetTurboDetectionModel(hf.config).eval()
    res = native.load_state_dict(remap(hf.state_dict()), strict=False)
    ignore = ("relative_position_index", "attn_mask", "num_batches_tracked", ".attn.")
    miss = [k for k in res.missing_keys if not any(s in k for s in ignore)]
    unexp = [k for k in res.unexpected_keys if not any(s in k for s in ignore)]
    print("missing:", miss[:10], f"(+{len(miss)-10} more)" if len(miss) > 10 else "")
    print("unexpected:", unexp[:10], f"(+{len(unexp)-10} more)" if len(unexp) > 10 else "")
    if miss or unexp:
        raise SystemExit("state_dict key mismatch")

    img = Image.open("libreyolo/assets/parkour.jpg").convert("RGB")
    classes = ["person", "dog", "skateboard"]
    task = "Detect {}.".format(", ".join(classes))
    inputs = proc(images=img, text=classes, task=task, return_tensors="pt")

    with torch.no_grad():
        # ---- stage 1: vision backbone ----
        hf_feats = hf.vision_backbone(inputs["pixel_values"])
        our_feats = native.vision_backbone(inputs["pixel_values"])
        for i, (r, o) in enumerate(zip(hf_feats, our_feats)):
            print(f"[vision {i}] max_abs_diff = {(r - o).abs().max().item():.3e}")
        # ---- stage 2: hybrid encoder ----
        hf_enc = hf.encoder(hf_feats, return_dict=True).extracted_states
        our_enc = native.encoder(our_feats)
        for i, (r, o) in enumerate(zip(hf_enc, our_enc)):
            print(f"[encoder {i}] max_abs_diff = {(r - o).abs().max().item():.3e}")
        # ---- full model ----
        ref = hf(**inputs)
        out = native(**{k: inputs[k] for k in
                        ["pixel_values", "classes_input_ids", "classes_attention_mask",
                         "tasks_input_ids", "tasks_attention_mask", "classes_structure"]})

    print()
    ok = True
    for key in ("decoder_coord_logits", "decoder_class_logits"):
        r, o = getattr(ref, key), out[key]
        diff = (r - o).abs().max().item()
        passed = diff < 1e-4
        ok = ok and passed
        print(f"{key:22s} shape={tuple(o.shape)}  max_abs_diff={diff:.3e}  {'OK' if passed else 'FAIL'}")
    print("\nOMDet-Turbo native parity:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
