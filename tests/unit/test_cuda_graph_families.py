"""Per-family CUDA graph capture parity.

Every family that sets ``SUPPORTS_CUDA_GRAPH`` must capture and replay
bit-identically. Models are built with random weights (``model_path=None``),
so nothing is downloaded and no checkpoint is required.

Two things this file is deliberate about, both of which produced wrong results
while the support matrix was being established:

* ``model_path=None`` leaves the network in **train mode**, and several
  families take a CPU-building branch while training. ``predict()`` runs in
  eval, so the model must be switched before probing or the test measures a
  path users never hit.
* The first output tensor is an anchor grid for several families and does not
  depend on the input at all. A replay that ignored its input would still
  match on that tensor, so input dependence is asserted across all outputs.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.base.cuda_graph import forward_maybe_graphed

# Not marked ``unit``: a few of these families pull a pretrained backbone when
# constructed, so the module does not meet the "no external weights" contract.
pytestmark = pytest.mark.general_nightly

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture requires a CUDA device"
)

# (import path, class name, task, size, imgsz). Every entry was verified to
# capture and replay bit-identically before its family's flag was enabled.
CAPTURABLE = [
    # detection
    ("libreyolo.models.yolo1.model", "LibreYOLO1", "detect", "t", 448),
    ("libreyolo.models.yolo2.model", "LibreYOLO2", "detect", "t", 640),
    ("libreyolo.models.yolo3.model", "LibreYOLO3", "detect", "t", 640),
    ("libreyolo.models.yolo4.model", "LibreYOLO4", "detect", "t", 640),
    ("libreyolo.models.yolo9.model", "LibreYOLO9", "detect", "t", 640),
    ("libreyolo.models.yolo9_p2.model", "LibreYOLO9P2", "detect", "t", 640),
    ("libreyolo.models.yolo9_e2e.model", "LibreYOLO9E2E", "detect", "t", 640),
    ("libreyolo.models.yolonas.model", "LibreYOLONAS", "detect", "s", 640),
    ("libreyolo.models.picodet.model", "LibrePICODET", "detect", "s", 640),
    ("libreyolo.models.rtmdet.model", "LibreRTMDet", "detect", "t", 640),
    ("libreyolo.models.dfine.model", "LibreDFINE", "detect", "n", 640),
    ("libreyolo.models.deim.model", "LibreDEIM", "detect", "n", 640),
    ("libreyolo.models.deimv2.model", "LibreDEIMv2", "detect", "atto", 640),
    ("libreyolo.models.rtdetr.model", "LibreRTDETR", "detect", "r18", 640),
    ("libreyolo.models.rtdetrv2.model", "LibreRTDETRv2", "detect", "r18", 640),
    ("libreyolo.models.rtdetrv4.model", "LibreRTDETRv4", "detect", "s", 640),
    ("libreyolo.models.rfdetr.model", "LibreRFDETR", "detect", "n", 640),
    ("libreyolo.models.ec.model", "LibreEC", "detect", "s", 640),
    # segmentation, pose, point
    ("libreyolo.models.dfine.model", "LibreDFINE", "segment", "n", 640),
    ("libreyolo.models.rtmdet.model", "LibreRTMDet", "segment", "t", 640),
    ("libreyolo.models.rfdetr.model", "LibreRFDETR", "segment", "n", 636),
    ("libreyolo.models.ec.model", "LibreEC", "segment", "s", 640),
    ("libreyolo.models.ec.model", "LibreEC", "pose", "s", 640),
    ("libreyolo.models.yolonas.model", "LibreYOLONAS", "pose", "s", 640),
    # rfdetr pose only ships size x, and that backbone needs a shape divisible
    # by 24, so this case cannot reuse the 640 the other entries use.
    ("libreyolo.models.rfdetr.model", "LibreRFDETR", "pose", "x", 648),
    ("libreyolo.models.fomo.model", "LibreFOMO", "point", "s", 640),
    # classification
    ("libreyolo.models.resnet.model", "LibreResNet", "classify", "18", 640),
    ("libreyolo.models.convnext.model", "LibreConvNeXt", "classify", "t", 640),
    ("libreyolo.models.mobilenetv4.model", "LibreMobileNetV4", "classify", "s", 640),
    ("libreyolo.models.clip.model", "LibreCLIP", "classify", "b32", 224),
    ("libreyolo.models.dinov2.model", "LibreDINOv2", "classify", "n", 644),
    ("libreyolo.models.siglip2.model", "LibreSigLIP2", "classify", "b16", 256),
    # panoptic/semantic/instance: capturable once the attention-mask schedule
    # is held on the host (LibreEoMTNet._apply).
    ("libreyolo.models.eomt.model", "LibreEoMT", "semantic", "s", 512),
    # depth: the network is captured, the sky step runs eagerly after replay.
    ("libreyolo.models.depth_anything3.model", "LibreDepthAnything3", "depth", "l", 644),
    # semantic segmentation
    ("libreyolo.models.dinov2.model", "LibreDINOv2", "semantic", "n", 644),
    ("libreyolo.models.segformer.model", "LibreSegformer", "semantic", "b0", 640),
    ("libreyolo.models.pidnet.model", "LibrePIDNet", "semantic", "s", 640),
    ("libreyolo.models.lingbotvision.model", "LibreLingBotVision", "semantic", "s", 640),
    # depth and restoration
    ("libreyolo.models.depth_anything.model", "LibreDepthAnythingV2", "depth", "s", 644),
    ("libreyolo.models.zipdepth.model", "LibreZipDepth", "depth", "b", 640),
    ("libreyolo.models.nafnet.model", "LibreNAFNet", "restore", "s", 640),
    ("libreyolo.models.realesrgan.model", "LibreRealESRGAN", "restore", "x4", 640),
    ("libreyolo.models.swinir.model", "LibreSwinIR", "restore", "s", 640),
]


# Families whose forward is very nearly input-independent at random init, so
# comparing replay against eager cannot prove the graph is reading its input:
# a graph that ignored the input entirely would match just as well. Capture is
# still exercised for them below, which shows the forward contains no op that
# capture rejects, but full parity needs real weights and is not claimed here.
#
# Measured relative variation between two contrasting inputs, random init:
#   LibreYOLOX/detect            1.1e-07
#   LibreEfficientNetV2/classify 2.2e-06
#   LibreYOLO7/detect            2.3e-05
# against ~1e-1 to 1e0 for the families in CAPTURABLE.
WEAK_SIGNAL = [
    ("libreyolo.models.yolox.model", "LibreYOLOX", "detect", "n", 640),
    ("libreyolo.models.efficientnetv2.model", "LibreEfficientNetV2", "classify", "b0", 640),
    ("libreyolo.models.yolo7.model", "LibreYOLO7", "detect", "b", 640),
]

# Relative to the output's own scale. Uniform noise washes out through global
# pooling, so the two probes differ in distribution as well as in draw.
MIN_RELATIVE_VARIATION = 1e-3


def _probe_inputs(imgsz):
    return (
        torch.randn(1, 3, imgsz, imgsz, device="cuda"),
        torch.randn(1, 3, imgsz, imgsz, device="cuda") * 3 + 2,
    )


def _relative_variation(first, second):
    best = 0.0
    for a, b in zip(first, second):
        if a.shape != b.shape:
            return 1.0
        scale = a.abs().max().item()
        if scale > 0:
            best = max(best, (a - b).abs().max().item() / scale)
    return best


def _flatten(out):
    if torch.is_tensor(out):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for o in out for t in _flatten(o)]
    if isinstance(out, dict):
        return [t for k in sorted(out) for t in _flatten(out[k])]
    return []


def _build(import_path, cls_name, task, size):
    import importlib

    # Seed before construction. Weights are random here, and with some draws a
    # head saturates hard enough that its output stops depending on the input
    # at all, which trips the input-dependence assertion below. That is a
    # property of the draw rather than of capture, so the draw is pinned.
    torch.manual_seed(0)
    cls = getattr(importlib.import_module(import_path), cls_name)
    model = cls(model_path=None, size=size, device="cuda", task=task)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "eval"):
        inner.eval()
    return model


@requires_cuda
@pytest.mark.parametrize("import_path,cls_name,task,size,imgsz", CAPTURABLE)
def test_family_capture_is_bit_identical(import_path, cls_name, task, size, imgsz):
    model = _build(import_path, cls_name, task, size)
    try:
        x1, x2 = _probe_inputs(imgsz)

        with torch.no_grad():
            eager1 = _flatten(model._forward(x1))
            eager2 = _flatten(model._forward(x2))

            assert eager1, "family produced no output tensors"
            # Relative, not bitwise. Two outputs can differ only at 1e-9 while
            # the model is effectively constant, and parity against a constant
            # proves nothing: a graph ignoring its input would match too.
            variation = _relative_variation(eager1, eager2)
            assert variation > MIN_RELATIVE_VARIATION, (
                f"output barely depends on the input (relative variation "
                f"{variation:.2e}), so replay parity proves nothing; move this "
                f"family to WEAK_SIGNAL"
            )

            model.capture_graph(imgsz=imgsz, batch=1)
            with model.cuda_graph_scope(True):
                graphed1 = _flatten(forward_maybe_graphed(model, x1))
                graphed2 = _flatten(forward_maybe_graphed(model, x2))

        assert model.graph_info()["graph_count"] == 1

        # Both replays must match eager. The second is what catches a graph
        # that returns a stale buffer rather than recomputing.
        for tag, eager, graphed in (("first", eager1, graphed1), ("second", eager2, graphed2)):
            assert len(eager) == len(graphed)
            for i, (a, b) in enumerate(zip(eager, graphed)):
                assert a.shape == b.shape, f"{tag} replay out[{i}] shape drift"
                assert torch.equal(a, b), (
                    f"{tag} replay out[{i}] differs from eager, "
                    f"maxdiff={(a.float() - b.float()).abs().max().item():.3e}"
                )
    finally:
        model.release_graphs()


@requires_cuda
def test_ppocr_detection_stage_captures():
    """PPOCR captures its detection stage rather than the _forward hook.

    The two-stage pipeline leaves ``_forward`` unimplemented on purpose, so
    this family gets its own runner over ``det``. Recognition stays eager
    because its crops vary in width.
    """
    from libreyolo.models.ppocr.model import LibrePPOCR

    model = LibrePPOCR(model_path=None, size="t", device="cuda")
    model.model.eval()
    try:
        x1 = torch.rand(1, 3, 640, 640, device="cuda")
        x2 = torch.rand(1, 3, 640, 640, device="cuda")
        with torch.no_grad():
            eager1 = model.model.det(x1).clone()
            eager2 = model.model.det(x2).clone()
            assert not torch.equal(eager1, eager2), "detection ignored its input"

            # With no scope active the wrapper must stay on the eager path.
            assert torch.equal(model.forward_det(x1), eager1)

            model.capture_graph(imgsz=640, batch=1)
            with model.cuda_graph_scope(True):
                graphed1 = model.forward_det(x1).clone()
                graphed2 = model.forward_det(x2).clone()

        assert torch.equal(eager1, graphed1)
        assert torch.equal(eager2, graphed2)

        # Detection input size follows the source aspect ratio, so a second
        # shape must get its own graph without corrupting the first.
        other = torch.rand(1, 3, 480, 480, device="cuda")
        with torch.no_grad():
            eager_other = model.model.det(other).clone()
            with model.cuda_graph_scope(True):
                graphed_other = model.forward_det(other).clone()
                graphed_first_again = model.forward_det(x1).clone()

        assert torch.equal(eager_other, graphed_other)
        assert torch.equal(eager1, graphed_first_again)
        assert model.graph_info()["graph_count"] == 2
    finally:
        model.release_graphs()


@requires_cuda
def test_sam_image_encoder_captures():
    """SAM captures its image encoder rather than the _forward hook.

    Upstream's vision attention builds its relative-position index on the host,
    which capture rejects; ``sam.transformers_compat`` replaces that with an
    on-device memoised index. Without the shim this test fails at capture.
    """
    from libreyolo.models.sam import transformers_compat
    from libreyolo.models.sam.model import LibreSAM1

    assert transformers_compat.apply() is True, "compat shim did not install"

    model = LibreSAM1(size="base", device="cuda")
    model.model.eval()
    try:
        x1 = torch.rand(1, 3, 1024, 1024, device="cuda")
        x2 = torch.rand(1, 3, 1024, 1024, device="cuda")
        with torch.no_grad():
            eager1 = model.model.get_image_embeddings(x1).clone()
            eager2 = model.model.get_image_embeddings(x2).clone()
            assert not torch.equal(eager1, eager2), "encoder ignored its input"

            # No scope active means the eager path, unchanged.
            assert torch.equal(model.forward_image_embeddings(x1), eager1)

            model.capture_graph(imgsz=1024, batch=1)
            with model.cuda_graph_scope(True):
                graphed1 = model.forward_image_embeddings(x1).clone()
                graphed2 = model.forward_image_embeddings(x2).clone()

        assert model.graph_info()["graph_count"] == 1
        assert torch.equal(eager1, graphed1)
        assert torch.equal(eager2, graphed2)
    finally:
        model.release_graphs()


def test_sam_compat_shim_matches_upstream():
    """The on-device index must equal what upstream computes on the host."""
    from transformers.models.sam import modeling_sam

    from libreyolo.models.sam import transformers_compat

    transformers_compat.apply()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = object.__new__(modeling_sam.SamVisionAttention)

    for q_size, k_size in ((14, 14), (16, 16), (64, 64), (14, 27)):
        rel_pos = torch.randn(2 * max(q_size, k_size) - 1, 8, device=device)
        expected_index = (
            torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
            - torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
            + (k_size - 1) * max(q_size / k_size, 1.0)
        ).long()
        got = transformers_compat._relative_index(q_size, k_size, rel_pos.device)
        assert torch.equal(got.cpu(), expected_index)


@requires_cuda
def test_unsupported_family_still_refuses():
    """Families that never opted in must raise rather than silently capture."""
    import importlib

    cls = getattr(importlib.import_module("libreyolo.models.birefnet.model"), "LibreBiRefNet")
    assert cls.SUPPORTS_CUDA_GRAPH is False
    model = cls(model_path=None, size="t", device="cuda")
    with pytest.raises(NotImplementedError):
        model.capture_graph(imgsz=640, batch=1)


@requires_cuda
@pytest.mark.parametrize("import_path,cls_name,task,size,imgsz", WEAK_SIGNAL)
def test_weak_signal_family_captures(import_path, cls_name, task, size, imgsz):
    """Capture works for these, but parity is not evidence of correctness.

    Their forward is all but input-independent at random init, so eager and
    replay agreeing says little. What this does establish is that capture
    itself succeeds, meaning the forward contains no operation capture
    rejects, and that replay is stable. Proving parity properly needs real
    weights; see docs/cuda_graphs.md.
    """
    model = _build(import_path, cls_name, task, size)
    try:
        x1, _ = _probe_inputs(imgsz)
        with torch.no_grad():
            eager = _flatten(model._forward(x1))
            model.capture_graph(imgsz=imgsz, batch=1)
            with model.cuda_graph_scope(True):
                graphed = _flatten(forward_maybe_graphed(model, x1))
                again = _flatten(forward_maybe_graphed(model, x1))

        assert model.graph_info()["graph_count"] == 1
        assert len(eager) == len(graphed)
        for i, (a, b) in enumerate(zip(eager, graphed)):
            assert a.shape == b.shape
            assert torch.equal(a, b), f"replay out[{i}] differs from eager"
        for i, (a, b) in enumerate(zip(graphed, again)):
            assert torch.equal(a, b), f"repeated replay out[{i}] is unstable"
    finally:
        model.release_graphs()
