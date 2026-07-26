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
    ("libreyolo.models.yolox.model", "LibreYOLOX", "detect", "n", 640),
    ("libreyolo.models.yolo7.model", "LibreYOLO7", "detect", "b", 640),
    ("libreyolo.models.efficientnetv2.model", "LibreEfficientNetV2", "classify", "b0", 640),
    # matte: encoder captured, deformable decoder eager (torchvision's
    # deform_conv2d kernel is not capture-safe).
    ("libreyolo.models.birefnet.model", "LibreBiRefNet", "matte", "t", 640),
]


# Two probes with different distributions. Uniform draws wash out through
# global pooling, and some heads then produce byte-identical output for both,
# which leaves nothing for the parity check to bite on.
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
            # At least one output must differ between the two probes,
            # otherwise parity proves nothing: a graph that ignored its input
            # would match just as well. Bitwise is the right test here, not a
            # magnitude threshold. Several detection heads add a large constant
            # grid to their predictions, so a real input-dependent signal can
            # be tiny relative to the output's scale while still being exactly
            # what a stale replay would get wrong.
            differing = [
                i
                for i, (a, b) in enumerate(zip(eager1, eager2))
                if a.shape != b.shape or not torch.equal(a, b)
            ]
            assert differing, (
                "every output is byte-identical for both probes, so replay "
                "parity cannot detect a graph that ignores its input"
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

        # Guard on the guard: a graph stuck on the first input would return
        # graphed1 for the second, so those must not agree with eager2.
        assert any(
            not torch.equal(a, b) for a, b in zip(graphed1, eager2)
        ), "a stale graph would be indistinguishable here"
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

    cls = getattr(importlib.import_module("libreyolo.models.l2cs.model"), "LibreL2CS")
    assert cls.SUPPORTS_CUDA_GRAPH is False
    model = cls(model_path=None, size="r18", device="cuda")
    with pytest.raises(NotImplementedError):
        model.capture_graph(imgsz=448, batch=1)


@requires_cuda
def test_sensenova_vision_tower_captures():
    """SenseNova's packed vision tower captures at a fixed token count.

    The tower is built from a synthetic config here, so this needs no
    checkpoint. It used to fail capture because the attention fallback read
    ``cu_seqlens`` element by element with ``int()``, syncing the stream once
    per segment per layer; the boundaries are now read on the eager warmup and
    reused during capture.

    This covers the vision half only. SenseNova as a family stays disabled: its
    inference is autoregressive generation over a growing KV cache, which needs
    a static KV cache with graphs bucketed by length.
    """
    from libreyolo.models.sensenova.modeling.siglip_navit import (
        SiglipVisionConfig,
        SiglipVisionModel,
    )

    torch.manual_seed(0)
    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=64,
        patch_size=16,
    )
    model = SiglipVisionModel(config)
    model.vision_model.embeddings.convert_conv2d_to_linear(config)
    model = model.cuda().eval().to(torch.bfloat16)

    tokens = 16
    patch = config.patch_size
    x1 = torch.randn(tokens, 3 * patch * patch, device="cuda", dtype=torch.bfloat16)
    x2 = torch.randn_like(x1) * 3 + 2
    kwargs = dict(
        packed_flattened_position_ids=torch.arange(tokens, device="cuda"),
        cu_seqlens=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
        max_seqlen=tokens,
    )

    with torch.no_grad():
        eager1 = model(packed_pixel_values=x1, **kwargs).clone()
        eager2 = model(packed_pixel_values=x2, **kwargs).clone()
        assert not torch.equal(eager1, eager2), "tower ignored its input"

        static = x1.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model(packed_pixel_values=static, **kwargs)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
            out = model(packed_pixel_values=static, **kwargs)

        static.copy_(x1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(eager1, out.clone())

        static.copy_(x2)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(eager2, out.clone())


@requires_cuda
def test_sensenova_run_vit_dispatch():
    """Exercise the dispatch that reaches SenseNova's tower.

    The family cannot be constructed without its ~15 GB checkpoint, so this
    drives ``Bagel.run_vit`` and ``Bagel.vit_forward_for_graph`` against a stub
    holding a real tower. That covers the routing and the eager fallback, which
    is everything in the dispatch except the wrapper's ``cuda_graph_scope``
    attaching the runner.
    """
    from libreyolo.models.base.cuda_graph import GraphRunner
    from libreyolo.models.sensenova.modeling.bagel import Bagel
    from libreyolo.models.sensenova.modeling.siglip_navit import (
        SiglipVisionConfig,
        SiglipVisionModel,
    )

    torch.manual_seed(0)
    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=64,
        patch_size=16,
    )
    tower = SiglipVisionModel(config)
    tower.vision_model.embeddings.convert_conv2d_to_linear(config)
    tower = tower.cuda().eval().to(torch.bfloat16)

    tokens_n = 16
    patch = config.patch_size
    tokens = torch.randn(tokens_n, 3 * patch * patch, device="cuda", dtype=torch.bfloat16)
    position_ids = torch.arange(tokens_n, device="cuda")
    cu_seqlens = torch.tensor([0, tokens_n], device="cuda", dtype=torch.int32)

    class _Stub:
        pass

    stub = _Stub()
    stub.vit_model = tower

    with torch.no_grad():
        # No runner attached: the original call, untouched.
        eager = Bagel.run_vit(stub, tokens, position_ids, cu_seqlens, tokens_n).clone()

        stub._vit_graph_runner = GraphRunner(
            forward_fn=lambda t: Bagel.vit_forward_for_graph(stub, t),
            family="sensenova",
        )
        stub._vit_graph_auto = False
        graphed = Bagel.run_vit(stub, tokens, position_ids, cu_seqlens, tokens_n).clone()

    assert stub._vit_graph_runner.info()["graph_count"] == 1, (
        "dispatch did not reach the runner"
    )
    assert torch.equal(eager, graphed)

    # A runner that raises must not break inference: run_vit falls back.
    class _Exploding:
        def run(self, *args, **kwargs):
            raise RuntimeError("boom")

    stub._vit_graph_runner = _Exploding()
    with torch.no_grad():
        fallback = Bagel.run_vit(stub, tokens, position_ids, cu_seqlens, tokens_n).clone()
    assert torch.equal(eager, fallback), "fallback did not return the eager result"
