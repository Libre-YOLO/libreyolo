"""Shared Swin backbone and standalone classifier exact parity vs timm."""

from __future__ import annotations

import torch

from libreyolo.models.swin.classifier import SwinClassifier
from libreyolo.models.swin.nn import SwinBackbone, SwinDims

CLASSIFIER_TAGS = {
    "t": "swin_tiny_patch4_window7_224.ms_in1k",
    "s": "swin_small_patch4_window7_224.ms_in1k",
    "b": "swin_base_patch4_window7_224.ms_in1k",
    "l": "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
}


def check_shared_backbone(timm):
    """Keep the original 640px shared-backbone parity gate intact."""
    ref = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        img_size=640,
        always_partition=True,
        out_indices=[1, 2, 3],
        features_only=True,
    ).eval()
    # force eager attention so parity compares identical math
    for m in ref.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False

    native = SwinBackbone(SwinDims()).eval()
    # timm features_only names stages layers_0.. ; native uses layers.0..
    remap = {}
    for k, v in ref.state_dict().items():
        remap[k.replace("layers_", "layers.", 1) if k.startswith("layers_") else k] = v
    res = native.load_state_dict(remap, strict=False)
    miss = [k for k in res.missing_keys if "relative_position_index" not in k]
    unexp = [
        k
        for k in res.unexpected_keys
        if "relative_position_index" not in k and "attn_mask" not in k
    ]
    print("missing (non-buffer):", miss[:8], "..." if len(miss) > 8 else "")
    print("unexpected (non-buffer):", unexp[:8], "..." if len(unexp) > 8 else "")
    assert not miss and not unexp, "state_dict key mismatch"

    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        ref_feats = ref(x)
        our_feats = native(x)

    ok = True
    for i, (r, o) in enumerate(zip(ref_feats, our_feats)):
        diff = (r - o).abs().max().item()
        print(
            f"stage[{i + 1}] shape={tuple(o.shape)}  max_abs_diff={diff:.3e}  {'OK' if diff == 0.0 else 'FAIL'}"
        )
        ok = ok and diff == 0.0
    print("\nSwin-T native parity:", "max_abs_diff == 0  PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


def check_classifiers(timm):
    """Verify released weights and logits for every standalone size."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for size, tag in CLASSIFIER_TAGS.items():
        ref = timm.create_model(tag, pretrained=True).eval()
        for module in ref.modules():
            if hasattr(module, "fused_attn"):
                module.fused_attn = False
        native = SwinClassifier(size=size, num_classes=1000).eval()
        result = native.load_state_dict(ref.state_dict(), strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        ref.to(device)
        native.to(device)

        generator = torch.Generator(device=device).manual_seed(0)
        inputs = torch.randn(1, 3, 224, 224, generator=generator, device=device)
        with torch.no_grad():
            ref_logits = ref(inputs)
            native_logits = native(inputs)
        diff = (ref_logits - native_logits).abs().max().item()
        print(
            f"classifier[{size}] shape={tuple(native_logits.shape)} "
            f"max_abs_diff={diff:.3e} {'OK' if diff == 0.0 else 'FAIL'}"
        )
        if diff != 0.0:
            raise SystemExit(1)


def main():
    import timm

    check_shared_backbone(timm)
    check_classifiers(timm)
    print("\nAll shared-backbone and classifier parity gates PASS")


if __name__ == "__main__":
    main()
