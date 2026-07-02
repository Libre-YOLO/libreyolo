"""Swin-T backbone native-port parity vs timm (max_abs_diff)."""

from __future__ import annotations

import torch

from libreyolo.models.swin.nn import SwinBackbone, SwinDims


def main():
    import timm

    ref = timm.create_model(
        "swin_tiny_patch4_window7_224", pretrained=False,
        img_size=640, always_partition=True, out_indices=[1, 2, 3], features_only=True,
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
    unexp = [k for k in res.unexpected_keys if "relative_position_index" not in k and "attn_mask" not in k]
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
        print(f"stage[{i+1}] shape={tuple(o.shape)}  max_abs_diff={diff:.3e}  {'OK' if diff == 0.0 else 'FAIL'}")
        ok = ok and diff == 0.0
    print("\nSwin-T native parity:", "max_abs_diff == 0  PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
