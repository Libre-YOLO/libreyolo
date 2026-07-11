"""ONNX export support for LibreBiRefNet.

BiRefNet's decoder uses ``torchvision.ops.deform_conv2d`` (modulated deformable
convolution), which neither ONNX exporter knows how to translate out of the box.
ONNX opset 19 defines a standard ``DeformConv`` operator, and torchvision's
offset/mask layout matches the ONNX spec, so we register a symbolic that maps
the op to ``DeformConv`` and export at a fixed resolution. ONNX Runtime's CPU
provider does not currently implement this node, so CPU runtime parity remains
experimental even though graph creation succeeds.
"""

from __future__ import annotations

MIN_OPSET = 19  # ONNX DeformConv is defined from opset 19.

_REGISTERED = False


def register_deform_conv2d_onnx_symbolic(opset: int = MIN_OPSET) -> None:
    """Idempotently register a ``torchvision::deform_conv2d`` -> ONNX ``DeformConv`` symbolic."""
    global _REGISTERED
    if _REGISTERED:
        return
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx.symbolic_helper import parse_args

    @parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "i")
    def _deform_conv2d(
        g,
        input,
        weight,
        offset,
        mask,
        bias,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        groups,
        offset_groups,
        use_mask,
    ):
        ksz = weight.type().sizes()
        kh, kw = int(ksz[2]), int(ksz[3])
        return g.op(
            "DeformConv",
            input,
            weight,
            offset,
            bias,
            mask,
            strides_i=[sh, sw],
            pads_i=[ph, pw, ph, pw],
            dilations_i=[dh, dw],
            group_i=groups,
            offset_group_i=offset_groups,
            kernel_shape_i=[kh, kw],
        )

    register_custom_op_symbolic("torchvision::deform_conv2d", _deform_conv2d, opset)
    _REGISTERED = True


__all__ = ["register_deform_conv2d_onnx_symbolic", "MIN_OPSET"]
