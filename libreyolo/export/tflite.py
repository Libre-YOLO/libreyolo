"""TensorFlow Lite export implementation via onnx2tf."""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Generator, Iterable

import numpy as np

logger = logging.getLogger(__name__)

_SUPPORTED_EXPORTS = {
    ("yolo9", "detect"): "YOLO9 detect",
    ("rfdetr", "detect"): "RF-DETR detect",
    ("rfdetr", "segment"): "RF-DETR segmentation",
    ("rfdetr", "pose"): "RF-DETR pose",
    # Darknet families + YOLOv7 bake the decode into the export graph, so they
    # go through the same generic onnx2tf path as YOLO9 detect. Verified
    # end-to-end on yolo3 (native-vs-tflite max_abs_diff ~3e-4); v2/v4/v7 use
    # the identical wrapper->ONNX->TFLite pipeline.
    ("yolo2", "detect"): "YOLOv2 detect",
    ("yolo3", "detect"): "YOLOv3 detect",
    ("yolo4", "detect"): "YOLOv4 detect",
    ("yolo7", "detect"): "YOLOv7 detect",
}
_UNSUPPORTED_FAMILY_REASONS: dict[str, str] = {}


def supported_tflite_exports() -> tuple[tuple[str, str], ...]:
    """Return ``(family, task)`` pairs with validated TFLite export support."""
    return tuple(_SUPPORTED_EXPORTS)


def ensure_tflite_family_supported(
    model_family: str | None,
    task: str | None,
) -> None:
    """Raise a targeted error when a family/task has not been validated."""
    family = (model_family or "").lower()
    task = (task or "detect").lower()
    if (family, task) in _SUPPORTED_EXPORTS:
        return

    supported = ", ".join(_SUPPORTED_EXPORTS.values())
    if family == "yolo9" and task == "segment":
        reason = "YOLO9 segmentation export is not supported; YOLO9 TFLite is detect-only."
    else:
        reason = _UNSUPPORTED_FAMILY_REASONS.get(
            family,
            "This family/task has not been validated through the ONNX-to-TFLite path yet.",
        )
    raise NotImplementedError(
        f"TFLite export currently supports: {supported}. "
        f"Got model family {model_family!r}, task {task!r}. {reason}"
    )


def check_tflite_export_available() -> None:
    """Check whether the optional TFLite export toolchain is available."""
    if sys.version_info < (3, 12):
        raise ImportError(
            "TFLite export requires Python 3.12 or newer because onnx2tf 2.4.x "
            "does not publish wheels for older Python versions.\n\n"
            "Install in a Python 3.12 environment with:\n"
            "  pip install libreyolo[tflite]"
        )

    try:
        import onnx2tf  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "TFLite export requires the optional onnx2tf converter.\n\n"
            "Install with:\n"
            "  pip install libreyolo[tflite]"
        ) from e


# ── RF-DETR GridSample ONNX rewrite ───────────────────────────────────────────
# RF-DETR's deformable cross-attention emits one GridSample node per decoder
# layer (6 total). onnx2tf's default lowering maps GridSample to
# tf.gather_nd(batch_dims=1), which TFLite's GatherNd kernel does not support —
# it silently accepts the model but produces wrong scores at inference time.
# The replace_to_pseudo_operators=["GridSample"] path is also numerically broken.
#
# The fix rewrites each GridSample into four Gather(axis=0) lookups on a
# transposed, zero-padded, flattened (N*(H+2)*(W+2), C) image tensor.
# onnx2tf lowers Gather(axis=0) to TFLite GATHER with no batch_dims —
# the only gather path that is unconditionally correct in TFLite.
#
# Algorithm ported from roboflow/rf-detr (Apache-2.0).


def _replace_single_gridsample(node: Any, graph: Any, *, index: int) -> None:
    """Rewrite one GridSample ONNX node into a TFLite-safe bilinear subgraph."""
    import numpy as _np
    import onnx as _onnx
    import onnx_graphsurgeon as gs

    im = node.inputs[0]    # [N, C, H, W]
    grid = node.inputs[1]  # [N, H_out, W_out, 2]
    out = node.outputs[0]  # [N, C, H_out, W_out]

    mode = node.attrs.get("mode", "bilinear")
    padding_mode = node.attrs.get("padding_mode", "zeros")
    align_corners = node.attrs.get("align_corners", 0)

    if mode != "bilinear" or padding_mode != "zeros" or align_corners != 0:
        raise NotImplementedError(
            f"GridSample TFLite patch only supports mode='bilinear', "
            f"padding_mode='zeros', align_corners=0; "
            f"got mode={mode!r}, padding_mode={padding_mode!r}, align_corners={align_corners}"
        )

    if im.shape is None or len(im.shape) != 4:
        raise ValueError(f"GridSample TFLite patch requires rank-4 im; got im.shape={im.shape}")
    if grid.shape is None or len(grid.shape) != 4:
        raise ValueError(f"GridSample TFLite patch requires rank-4 grid; got grid.shape={grid.shape}")

    pfx = f"_gsrepl{index}"
    uid: list[int] = [0]

    def v(name: str, dtype: Any = _np.float32, shape: list[int] | None = None) -> Any:
        uid[0] += 1
        return gs.Variable(f"{pfx}_{uid[0]}_{name}", dtype=dtype, shape=shape)

    def c(val: Any, dtype: Any = _np.float32, name: str = "") -> Any:
        uid[0] += 1
        return gs.Constant(f"{pfx}_{uid[0]}_c_{name}", _np.array(val, dtype=dtype))

    nodes: list[Any] = []

    def op(kind: str, ins: list[Any], outs: list[Any], attrs: dict[str, Any] | None = None) -> None:
        nodes.append(gs.Node(op=kind, inputs=ins, outputs=outs, attrs=attrs or {}))

    i64 = int(_onnx.TensorProto.INT64)
    f32 = int(_onnx.TensorProto.FLOAT)

    zero_i = c(_np.int64(0), _np.int64, "zero_i")
    one_i = c(_np.int64(1), _np.int64, "one_i")
    two_i = c(_np.int64(2), _np.int64, "two_i")

    zero_f = c(_np.float32(0.0), _np.float32, "zero_f")
    one_f = c(_np.float32(1.0), _np.float32, "one_f")
    two_f = c(_np.float32(2.0), _np.float32, "two_f")
    half_f = c(_np.float32(0.5), _np.float32, "half_f")

    ax0_1d = c([0], _np.int64, "ax0_1d")
    ax3_1d = c([3], _np.int64, "ax3_1d")
    neg1_1d = c([-1], _np.int64, "neg1_1d")

    # Step 0: Runtime shape extraction
    im_shape_t = v("im_shape", dtype=_np.int64, shape=[4])
    op("Shape", [im], [im_shape_t])
    grid_shape_t = v("grid_shape", dtype=_np.int64, shape=[4])
    op("Shape", [grid], [grid_shape_t])

    def gather_dim(shape_t: Any, axis_idx: int, name: str) -> Any:
        result = v(name, dtype=_np.int64)
        op("Gather", [shape_t, c(_np.int64(axis_idx), _np.int64, f"i{axis_idx}_{name}")], [result], {"axis": 0})
        return result

    N_t = gather_dim(im_shape_t, 0, "N")  # noqa: N806
    C_t = gather_dim(im_shape_t, 1, "C")  # noqa: N806
    H_t = gather_dim(im_shape_t, 2, "H")  # noqa: N806
    W_t = gather_dim(im_shape_t, 3, "W")  # noqa: N806
    H_out_t = gather_dim(grid_shape_t, 1, "H_out")  # noqa: N806
    W_out_t = gather_dim(grid_shape_t, 2, "W_out")  # noqa: N806

    pH_t = v("pH", dtype=_np.int64)  # noqa: N806
    op("Add", [H_t, two_i], [pH_t])
    pW_t = v("pW", dtype=_np.int64)  # noqa: N806
    op("Add", [W_t, two_i], [pW_t])
    pH_pW_t = v("pH_pW", dtype=_np.int64)  # noqa: N806
    op("Mul", [pH_t, pW_t], [pH_pW_t])

    N_pH_pW_t = v("N_pH_pW", dtype=_np.int64)  # noqa: N806
    op("Mul", [N_t, pH_pW_t], [N_pH_pW_t])

    # Float casts for coordinate arithmetic
    W_f = v("W_f", dtype=_np.float32)  # noqa: N806
    op("Cast", [W_t], [W_f], {"to": f32})
    H_f = v("H_f", dtype=_np.float32)  # noqa: N806
    op("Cast", [H_t], [H_f], {"to": f32})
    pW_f = v("pW_f", dtype=_np.float32)  # noqa: N806
    op("Cast", [pW_t], [pW_f], {"to": f32})
    pH_f = v("pH_f", dtype=_np.float32)  # noqa: N806
    op("Cast", [pH_t], [pH_f], {"to": f32})

    W_half = v("W_half", dtype=_np.float32)  # noqa: N806
    op("Div", [W_f, two_f], [W_half])
    H_half = v("H_half", dtype=_np.float32)  # noqa: N806
    op("Div", [H_f, two_f], [H_half])

    pW_max_f = v("pW_max_f", dtype=_np.float32)  # noqa: N806
    op("Sub", [pW_f, one_f], [pW_max_f])
    pH_max_f = v("pH_max_f", dtype=_np.float32)  # noqa: N806
    op("Sub", [pH_f, one_f], [pH_max_f])

    def unsq0(scalar_t: Any, name: str) -> Any:
        result = v(name, dtype=_np.int64, shape=[1])
        op("Unsqueeze", [scalar_t, ax0_1d], [result])
        return result

    N_1d = unsq0(N_t, "N_1d")  # noqa: N806
    C_1d = unsq0(C_t, "C_1d")  # noqa: N806
    H_out_1d = unsq0(H_out_t, "H_out_1d")  # noqa: N806
    W_out_1d = unsq0(W_out_t, "W_out_1d")  # noqa: N806
    N_pH_pW_1d = unsq0(N_pH_pW_t, "N_pH_pW_1d")  # noqa: N806

    flat_shape_t = v("flat_shape", dtype=_np.int64, shape=[2])
    op("Concat", [N_pH_pW_1d, C_1d], [flat_shape_t], {"axis": 0})

    nhwc_shape_t = v("nhwc_shape", dtype=_np.int64, shape=[4])
    op("Concat", [N_1d, H_out_1d, W_out_1d, C_1d], [nhwc_shape_t], {"axis": 0})

    # Step 1: Build flat data tensor (N*(H+2)*(W+2), C)
    pads = c([0, 0, 1, 1, 0, 0, 1, 1], _np.int64, "pads")
    im_pad = v("im_pad")
    op("Pad", [im, pads], [im_pad])
    im_nhwc = v("im_nhwc")
    op("Transpose", [im_pad], [im_nhwc], {"perm": [0, 2, 3, 1]})
    im_flat = v("im_flat")
    op("Reshape", [im_nhwc, flat_shape_t], [im_flat])

    # Step 2: Extract gx, gy from grid last dim
    gx_raw = v("gx_raw")
    op("Slice", [grid, c([0], _np.int64, "s0_gx"), c([1], _np.int64, "e1_gx"), ax3_1d], [gx_raw])
    gx = v("gx")
    op("Squeeze", [gx_raw, ax3_1d], [gx])

    gy_raw = v("gy_raw")
    op("Slice", [grid, c([1], _np.int64, "s1_gy"), c([2], _np.int64, "e2_gy"), ax3_1d], [gy_raw])
    gy = v("gy")
    op("Squeeze", [gy_raw, ax3_1d], [gy])

    # Unnormalize (align_corners=False): px = (gx + 1) * W/2 - 0.5
    gxp1 = v("gxp1")
    op("Add", [gx, one_f], [gxp1])
    pxr = v("pxr")
    op("Mul", [gxp1, W_half], [pxr])
    px = v("px")
    op("Sub", [pxr, half_f], [px])

    gyp1 = v("gyp1")
    op("Add", [gy, one_f], [gyp1])
    pyr = v("pyr")
    op("Mul", [gyp1, H_half], [pyr])
    py = v("py")
    op("Sub", [pyr, half_f], [py])

    # Step 3: Floor + bilinear weights
    x0_f = v("x0_f")
    op("Floor", [px], [x0_f])
    y0_f = v("y0_f")
    op("Floor", [py], [y0_f])
    x1_f = v("x1_f")
    op("Add", [x0_f, one_f], [x1_f])
    y1_f = v("y1_f")
    op("Add", [y0_f, one_f], [y1_f])

    wx1 = v("wx1")
    op("Sub", [px, x0_f], [wx1])
    wy1 = v("wy1")
    op("Sub", [py, y0_f], [wy1])
    wx0 = v("wx0")
    op("Sub", [one_f, wx1], [wx0])
    wy0 = v("wy0")
    op("Sub", [one_f, wy1], [wy0])

    # Step 4: Shifted + clamped integer coords in the padded image
    def int_shifted_clamped(coord_f: Any, clamp_max: Any, name: str) -> Any:
        shifted = v(f"sh_{name}")
        op("Add", [coord_f, one_f], [shifted])
        clipped = v(f"cl_{name}")
        op("Clip", [shifted, zero_f, clamp_max], [clipped])
        cast_int = v(f"ca_{name}", dtype=_np.int64)
        op("Cast", [clipped], [cast_int], {"to": i64})
        return cast_int

    x0c = int_shifted_clamped(x0_f, pW_max_f, "x0c")
    x1c = int_shifted_clamped(x1_f, pW_max_f, "x1c")
    y0c = int_shifted_clamped(y0_f, pH_max_f, "y0c")
    y1c = int_shifted_clamped(y1_f, pH_max_f, "y1c")

    # Step 5: Batch offset via Range
    batch_range = v("batch_range", dtype=_np.int64)
    op("Range", [zero_i, N_t, one_i], [batch_range])
    batch_offset_flat = v("batch_offset_flat", dtype=_np.int64)
    op("Mul", [batch_range, pH_pW_t], [batch_offset_flat])
    batch_offset_shape = c([-1, 1, 1], _np.int64, "batch_offset_shape")
    batch_offset = v("batch_offset", dtype=_np.int64)
    op("Reshape", [batch_offset_flat, batch_offset_shape], [batch_offset])

    # Step 6: Global 1-D flat indices per corner
    def flat_global_index(xc: Any, yc: Any, name: str) -> Any:
        ypw = v(f"ypw_{name}", dtype=_np.int64)
        op("Mul", [yc, pW_t], [ypw])
        local = v(f"local_{name}", dtype=_np.int64)
        op("Add", [ypw, xc], [local])
        global_idx = v(f"glob_{name}", dtype=_np.int64)
        op("Add", [local, batch_offset], [global_idx])
        flat1d = v(f"gidx_{name}", dtype=_np.int64)
        op("Reshape", [global_idx, neg1_1d], [flat1d])
        return flat1d

    gidx_aa = flat_global_index(x0c, y0c, "aa")
    gidx_ba = flat_global_index(x1c, y0c, "ba")
    gidx_ab = flat_global_index(x0c, y1c, "ab")
    gidx_bb = flat_global_index(x1c, y1c, "bb")

    # Step 7: Gather(axis=0) on the flat data tensor
    def gather_flat(gidx: Any, name: str) -> Any:
        sampled = v(f"samp_{name}")
        op("Gather", [im_flat, gidx], [sampled], {"axis": 0})
        return sampled

    saa = gather_flat(gidx_aa, "aa")
    sba = gather_flat(gidx_ba, "ba")
    sab = gather_flat(gidx_ab, "ab")
    sbb = gather_flat(gidx_bb, "bb")

    # Step 8: Bilinear weighted sum
    wflat_shape = c([-1, 1], _np.int64, "wflat_shape")

    def contrib(gathered: Any, wx: Any, wy: Any, name: str) -> Any:
        w_2d = v(f"w2d_{name}")
        op("Mul", [wy, wx], [w_2d])
        w_flat = v(f"wflat_{name}")
        op("Reshape", [w_2d, wflat_shape], [w_flat])
        out_contrib = v(f"c_{name}")
        op("Mul", [gathered, w_flat], [out_contrib])
        return out_contrib

    caa = contrib(saa, wx0, wy0, "aa")
    cba = contrib(sba, wx1, wy0, "ba")
    cab = contrib(sab, wx0, wy1, "ab")
    cbb = contrib(sbb, wx1, wy1, "bb")

    s1 = v("s1")
    op("Add", [caa, cba], [s1])
    s2 = v("s2")
    op("Add", [s1, cab], [s2])
    total = v("total")
    op("Add", [s2, cbb], [total])

    # Step 9: Reshape + transpose back to (N, C, H_out, W_out)
    total_nhwc = v("total_nhwc")
    op("Reshape", [total, nhwc_shape_t], [total_nhwc])
    op("Transpose", [total_nhwc], [out], {"perm": [0, 3, 1, 2]})

    node.inputs.clear()
    node.outputs.clear()
    graph.nodes.extend(nodes)


def _replace_gridsample_for_tflite(onnx_path: Path, output_dir: Path) -> Path:
    """Rewrite every GridSample node in the ONNX graph to TFLite-safe ops.

    Returns *onnx_path* unchanged when no GridSample nodes are present,
    otherwise writes a patched ONNX to *output_dir* and returns that path.
    """
    try:
        import onnx
        import onnx.shape_inference
        import onnx_graphsurgeon as gs
    except ImportError as exc:
        raise ImportError(
            "onnx and onnx-graphsurgeon are required for RF-DETR TFLite export. "
            "Install with: pip install libreyolo[tflite]"
        ) from exc

    model = onnx.load(str(onnx_path))
    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)

    gs_nodes = [n for n in graph.nodes if n.op == "GridSample"]
    if not gs_nodes:
        logger.debug("No GridSample nodes found; skipping TFLite-safe patch.")
        return onnx_path

    logger.info("Patching %d GridSample node(s) → TFLite-safe Gather(axis=0) subgraph.", len(gs_nodes))
    for i, node in enumerate(gs_nodes):
        _replace_single_gridsample(node, graph, index=i)

    graph.cleanup().toposort()

    try:
        patched = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
        onnx.checker.check_model(patched)
    except Exception as exc:
        raise RuntimeError(f"GridSample ONNX patch produced an invalid graph: {exc}") from exc

    out_path = output_dir / (onnx_path.stem + "_gs_patched.onnx")
    onnx.save(patched, str(out_path))
    logger.debug("GridSample-patched ONNX saved to: %s", out_path)
    return out_path


# ── RF-DETR onnxsim preprocessing ────────────────────────────────────────────
# RF-DETR's DINOv2 backbone emits Tile(multiples=[4,1,1]) for windowed
# attention's CLS-token expansion (4 windows of 12×12 patches).  When the
# ONNX model has a dynamic batch dimension, onnx2tf auto-transposes some
# decoder tensors as (N,256,300) and others as (N,300,256), which causes
# an Add shape mismatch deep in the decoder.
#
# Running onnxsim with a static batch=1 input propagates concrete shapes
# throughout the graph so onnx2tf applies a uniform transposition strategy
# everywhere.  The windowed-attention Tile node is left untouched — its
# multiples=[4,1,1] reflect the 4-window architecture, not the batch size.

# onnx2tf auto-transposes the patch embedding Concat output to (N,384,577)
# but leaves the static position_embeddings weight at (N,577,384).  These
# two param replacement entries align both inputs to the backbone Add op.
_RFDETR_FIX_JSON: dict = {
    "format_version": 1,
    "operations": [
        {
            "op_name": "wa/model/backbone/backbone.0/encoder/encoder/embeddings/Add",
            "param_target": "inputs",
            "param_name": "wa/model/backbone/backbone.0/encoder/encoder/embeddings/Concat_1_output_0",
            "pre_process_transpose_perm": [0, 1, 2],
        },
        {
            "op_name": "wa/model/backbone/backbone.0/encoder/encoder/embeddings/Add",
            "param_target": "inputs",
            "param_name": "model.backbone.0.encoder.encoder.embeddings.position_embeddings",
            "pre_process_transpose_perm": [0, 2, 1],
        },
    ],
}


def _write_rfdetr_fix_json(output_dir: Path) -> Path:
    fix_path = output_dir / "_rfdetr_fix.json"
    with open(fix_path, "w") as f:
        json.dump(_RFDETR_FIX_JSON, f)
    return fix_path


def _simplify_rfdetr_onnx(onnx_path: Path, c: int, h: int, w: int, output_dir: Path) -> Path:
    """Run onnxsim with a static batch=1 input to propagate concrete shapes for onnx2tf."""
    try:
        import onnx
        import onnxsim
    except ImportError as exc:
        raise ImportError(
            "onnx-simplifier is required for RF-DETR TFLite export. "
            "Install with: pip install libreyolo[tflite]"
        ) from exc

    model = onnx.load(str(onnx_path))
    inp_name = model.graph.input[0].name
    simplified, ok = onnxsim.simplify(
        model,
        overwrite_input_shapes={inp_name: [1, c, h, w]},
    )
    if not ok:
        logger.warning(
            "onnxsim verification failed for RF-DETR model — "
            "proceeding with the unsimplified ONNX (TFLite conversion may fail)."
        )
        return onnx_path

    out = output_dir / (onnx_path.stem + "_sim.onnx")
    onnx.save(simplified, str(out))
    logger.debug("onnxsim-simplified ONNX: %s", out)
    return out


# ── onnx2tf compatibility helpers ─────────────────────────────────────────────


@contextlib.contextmanager
def _numpy_allow_pickle() -> Generator[None, None, None]:
    """Patch numpy.load to allow pickle for the duration of onnx2tf conversion."""
    _original_load = np.load

    def _patched_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("allow_pickle", True)
        return _original_load(*args, **kwargs)

    np.load = _patched_load  # type: ignore[assignment]
    try:
        yield
    finally:
        np.load = _original_load  # type: ignore[assignment]


@contextlib.contextmanager
def _patch_validation_download(npy_path: str) -> Generator[None, None, None]:
    """Redirect onnx2tf's download_test_image_data() to use local calibration data.

    onnx2tf calls this function during conversion to download test images for
    ONNX-vs-TF output validation. Patching it avoids the network dependency.
    """
    # Force-import submodules here so the patch is self-contained and doesn't
    # depend on call-site ordering.
    import onnx2tf.onnx2tf  # noqa: F401
    import onnx2tf.utils.common_functions  # noqa: F401

    calib = np.load(npy_path, allow_pickle=False)

    def _replacement() -> Any:
        return calib

    originals: dict[str, Any] = {}
    for mod_name in ("onnx2tf.utils.common_functions", "onnx2tf.onnx2tf"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "download_test_image_data"):
            originals[mod_name] = getattr(mod, "download_test_image_data")
            setattr(mod, "download_test_image_data", _replacement)
    if not originals:
        logger.warning(
            "onnx2tf.download_test_image_data not found in expected modules — "
            "onnx2tf may attempt a network download during conversion."
        )
    try:
        yield
    finally:
        for mod_name, original in originals.items():
            mod = sys.modules.get(mod_name)
            if mod:
                setattr(mod, "download_test_image_data", original)


_RFDETR_CALIB_DEFAULT = (640, 640, 3)  # fallback when ONNX input dims are dynamic


def _rfdetr_input_info(onnx_path: Path) -> tuple[str, int, int, int]:
    """Return (inp_name, channels, height, width) from the first ONNX input."""
    import onnx

    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    shape = inp.type.tensor_type.shape
    if shape is None:
        raise ValueError(
            "RF-DETR ONNX input has no shape information. "
            "Re-export the model with shape inference enabled."
        )
    dims = [d.dim_value for d in shape.dim]
    if len(dims) != 4:
        raise ValueError(
            f"RF-DETR ONNX input must be rank-4 (N, C, H, W); got {len(dims)} dims."
        )
    _, c, h, w = dims
    if any(d <= 0 for d in (h, w, c)):
        h, w, c = _RFDETR_CALIB_DEFAULT
        logger.warning(
            "ONNX input has dynamic spatial dimensions; using %dx%dx%d.",
            h, w, c,
        )
    return inp.name, c, h, w


def _rfdetr_calib_data(c: int, h: int, w: int, output_dir: Path) -> Path:
    """Generate random NHWC calibration data for the given shape."""
    calib = np.random.rand(20, h, w, c).astype(np.float32)
    npy_path = output_dir / "_rfdetr_calib.npy"
    np.save(str(npy_path), calib)
    return npy_path


def _export_tflite_rfdetr(onnx_path: str, output_path: str, *, verbose: bool = False) -> str:
    """Convert an RF-DETR ONNX model to TFLite via the onnx2tf Python API.

    Conversion pipeline:
      1. onnxsim — propagates static batch=1 shapes so onnx2tf applies a
         uniform NCHW→NHWC transposition throughout the decoder.
      2. param_replacement JSON — aligns the backbone position-embedding Add
         inputs that onnx2tf would otherwise transpose inconsistently.
      3. tf_converter backend — avoids the TFLite TopK_V2 crash that the
         default flatbuffer_direct backend triggers for deformable attention,
         and correctly lowers GridSample without an ONNX-level rewrite.

    The GridSample ONNX rewrite functions in this module remain available as
    a fallback but are not needed for the tf_converter path.
    """
    onnx_file = Path(onnx_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        inp_name, c, h, w = _rfdetr_input_info(onnx_file)

        # Step 1: Propagate static batch=1 shapes with onnxsim.
        simplified_onnx = _simplify_rfdetr_onnx(onnx_file, c, h, w, tmp)

        # Step 2: Calibration data in NHWC layout for onnx2tf's validation.
        calib_npy = _rfdetr_calib_data(c, h, w, tmp)

        # Step 3: Write backbone Add transpose fixes to a temp JSON file.
        fix_json = _write_rfdetr_fix_json(tmp)

        from onnx2tf import convert

        convert_kwargs: dict[str, Any] = {
            "input_onnx_file_path": str(simplified_onnx),
            "output_folder_path": str(tmp),
            "output_signaturedefs": True,
            "non_verbose": not verbose,
            "verbosity": "info" if verbose else "warn",
            # Erf and GeLU must be replaced with TFLite-native pseudo-ops so the
            # produced model does not require the TensorFlow Flex delegate.
            "replace_to_pseudo_operators": ["Erf", "GeLU"],
            # Re-state static batch=1 for onnx2tf (onnxsim already baked it in,
            # but this ensures correct behavior if onnxsim verification failed).
            "overwrite_input_shape": [f"{inp_name}:1,{c},{h},{w}"],
            "param_replacement_file": str(fix_json),
        }

        with _numpy_allow_pickle(), _patch_validation_download(str(calib_npy)):
            # tf_converter avoids the TFLite TopK_V2 kernel crash at inference
            # time that flatbuffer_direct (onnx2tf 2.x default) triggers for
            # RF-DETR's deformable attention.
            try:
                convert(**convert_kwargs, tflite_backend="tf_converter")
            except TypeError as exc:
                if "tflite_backend" not in str(exc) and "unexpected keyword" not in str(exc):
                    raise
                logger.warning(
                    "onnx2tf does not support tflite_backend= — falling back to default "
                    "backend. RF-DETR TFLite inference may crash or produce wrong results. "
                    "Use onnx2tf>=2.4.3."
                )
                convert(**convert_kwargs)

        converted = _find_converted_tflite(tmp, simplified_onnx)
        shutil.copy2(converted, dst)

    logger.info("TFLite export complete: %s", dst)
    return str(dst)


# ── YOLO9 CLI-based path ───────────────────────────────────────────────────────


def _onnx2tf_command() -> list[str]:
    exe = shutil.which("onnx2tf")
    if exe:
        return [exe]
    return [sys.executable, "-m", "onnx2tf"]


def _find_converted_tflite(output_dir: Path, onnx_path: Path) -> Path:
    exact = output_dir / f"{onnx_path.stem}_float32.tflite"
    if exact.exists():
        return exact

    fp32_matches = sorted(output_dir.rglob("*_float32.tflite"))
    if fp32_matches:
        return fp32_matches[0]

    matches = sorted(output_dir.rglob("*.tflite"))
    if matches:
        return matches[0]

    produced = sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*"))
    raise RuntimeError(
        "onnx2tf did not produce a TFLite file. "
        f"Files found: {produced[:20]}"
    )


def _write_metadata_sidecar(output_path: Path, metadata: dict) -> None:
    sidecar_path = Path(str(output_path) + ".json")
    with open(sidecar_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata sidecar: %s", sidecar_path)


def export_tflite(
    onnx_path: str,
    output_path: str,
    *,
    half: bool = False,
    verbose: bool = False,
    onnx2tf_args: Iterable[str] | None = None,
    metadata: dict | None = None,
) -> str:
    """Convert a static ONNX model to TensorFlow Lite using onnx2tf.

    Note: ``onnx2tf_args`` is forwarded only on the YOLO9 CLI path.  It is
    not applicable to the RF-DETR Python-API path and will be ignored there.
    """
    if half:
        raise ValueError(
            "TFLite FP16 export is not supported yet. Omit half=True for FP32."
        )

    check_tflite_export_available()

    model_family = ((metadata or {}).get("model_family") or "").lower()
    if model_family == "rfdetr":
        if onnx2tf_args is not None:
            logger.warning(
                "onnx2tf_args is not supported on the RF-DETR TFLite path "
                "(uses the Python API, not the CLI) and will be ignored."
            )
        result = _export_tflite_rfdetr(onnx_path, output_path, verbose=verbose)
        if metadata is not None:
            _write_metadata_sidecar(Path(output_path), metadata)
        return result

    onnx_file = Path(onnx_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting ONNX to TFLite: %s", onnx_file)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_output = Path(tmpdir)
        cmd = [
            *_onnx2tf_command(),
            "-i",
            str(onnx_file),
            "-o",
            str(tmp_output),
            "-tb",
            "flatbuffer_direct",
            "-v",
            "info" if verbose else "warn",
        ]
        if onnx2tf_args is not None:
            cmd.extend(str(arg) for arg in onnx2tf_args)

        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
        )
        if result.returncode != 0:
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            raise RuntimeError(
                f"onnx2tf failed with exit code {result.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

        converted = _find_converted_tflite(tmp_output, onnx_file)
        shutil.copy2(converted, dst)

    if metadata is not None:
        _write_metadata_sidecar(dst, metadata)

    logger.info("TFLite export complete: %s", dst)
    return str(dst)


__all__ = [
    "check_tflite_export_available",
    "ensure_tflite_family_supported",
    "export_tflite",
    "supported_tflite_exports",
]
