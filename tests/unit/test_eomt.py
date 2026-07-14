"""Unit tests for the LibreEoMT semantic segmentation family."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image

pytestmark = pytest.mark.unit


def _synthetic_eomt_state(nc: int = 150, hidden: int = 1024) -> dict:
    return {
        "embeddings.patch_embeddings.projection.weight": torch.zeros(hidden, 3, 16, 16),
        "query.weight": torch.zeros(100, hidden),
        "mask_head.fc1.weight": torch.zeros(hidden, hidden),
        "mask_head.fc2.weight": torch.zeros(hidden, hidden),
        "mask_head.fc3.weight": torch.zeros(hidden, hidden),
        "class_predictor.weight": torch.zeros(nc + 1, hidden),
        "class_predictor.bias": torch.zeros(nc + 1),
        "criterion.empty_weight": torch.zeros(nc + 1),
    }


class _FakeEoMTNet(nn.Module):
    def __init__(
        self,
        config: str = "l",
        nb_classes: int = 150,
        image_size: int = 512,
        num_queries: int = 100,
    ):
        super().__init__()
        if config not in ("s", "b", "l"):
            raise ValueError(f"test fake: unsupported size {config!r}")
        self.nb_classes = int(nb_classes)
        self.image_size = int(image_size)
        self.num_queries = int(num_queries)
        self.proj = nn.Conv2d(3, self.nb_classes, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"semantic_logits": self.proj(x)}

    def load_state_dict(self, state_dict, strict: bool = True):
        from torch.nn.modules.module import _IncompatibleKeys

        self.loaded_state_dict = dict(state_dict)
        return _IncompatibleKeys([], [])


@pytest.fixture
def fake_eomt_net(monkeypatch):
    import libreyolo.models.eomt.model as eomt_model

    monkeypatch.setattr(eomt_model, "LibreEoMTNet", _FakeEoMTNet)
    return _FakeEoMTNet


def _load_converter_module():
    weights_dir = Path(__file__).resolve().parents[2] / "weights"
    weights_path = str(weights_dir)
    if weights_path not in sys.path:
        sys.path.insert(0, weights_path)
    return importlib.import_module("convert_eomt_weights")


class TestEoMTMetadata:
    def test_task_and_size_metadata(self):
        from libreyolo.models.eomt.model import LibreEoMT

        assert LibreEoMT.FAMILY == "eomt"
        assert LibreEoMT.FILENAME_PREFIX == "LibreEoMT"
        assert LibreEoMT.SUPPORTED_TASKS == ("semantic", "segment", "panoptic")
        assert LibreEoMT.DEFAULT_TASK == "semantic"
        assert LibreEoMT.REQUIRE_TASK_SUFFIX
        assert LibreEoMT.INPUT_SIZES == {"s": 512, "b": 512, "l": 512}
        assert LibreEoMT.TASK_INPUT_SIZES == {
            "semantic": {"s": 512, "b": 512, "l": 512},
            "segment": {"s": 640, "b": 640, "l": 640},
            "panoptic": {"s": 640, "b": 640, "l": 640},
        }
        assert LibreEoMT.semantic_resize_mode == "split"
        assert LibreEoMT.semantic_imgsz_divisor == 16

    def test_registered_in_factory(self):
        from libreyolo.models import LibreEoMT
        from libreyolo.models.base import BaseModel

        assert LibreEoMT in BaseModel._registry

    def test_can_load_detects_signature_size_and_classes(self):
        from libreyolo.models.eomt.model import LibreEoMT

        state = _synthetic_eomt_state(nc=150)
        assert LibreEoMT.can_load(state)
        assert LibreEoMT.detect_size(state) == "l"
        assert LibreEoMT.detect_nb_classes(state) == 150
        assert LibreEoMT.detect_checkpoint_task(state) == "semantic"

    def test_detect_size_small_and_base(self):
        from libreyolo.models.eomt.model import LibreEoMT

        assert LibreEoMT.detect_size(_synthetic_eomt_state(nc=150, hidden=384)) == "s"
        assert LibreEoMT.detect_size(_synthetic_eomt_state(nc=150, hidden=768)) == "b"
        assert LibreEoMT.detect_size(_synthetic_eomt_state(nc=150, hidden=1024)) == "l"

    def test_detect_checkpoint_task_segment(self):
        from libreyolo.models.eomt.model import LibreEoMT

        assert (
            LibreEoMT.detect_checkpoint_task(_synthetic_eomt_state(nc=80)) == "segment"
        )
        assert (
            LibreEoMT.detect_checkpoint_task(_synthetic_eomt_state(nc=150))
            == "semantic"
        )

    def test_detect_num_queries_and_image_size(self):
        from libreyolo.models.eomt.model import LibreEoMT

        # 100 queries, 512px (32*32 patches * 16 px each)
        state_512 = {
            "query.weight": torch.zeros(100, 256),
            "embeddings.position_embeddings.weight": torch.zeros(1024, 1024),
        }
        assert LibreEoMT.detect_num_queries(state_512) == 100
        assert LibreEoMT.detect_image_size(state_512) == 512

        # 200 queries, 640px (40*40 patches)
        state_640 = {
            "query.weight": torch.zeros(200, 768),
            "embeddings.position_embeddings.weight": torch.zeros(1600, 768),
        }
        assert LibreEoMT.detect_num_queries(state_640) == 200
        assert LibreEoMT.detect_image_size(state_640) == 640

        # Missing keys → None
        assert LibreEoMT.detect_num_queries({}) is None
        assert LibreEoMT.detect_image_size({}) is None

        # Non-square position embeddings → None
        state_bad = {"embeddings.position_embeddings.weight": torch.zeros(1023, 768)}
        assert LibreEoMT.detect_image_size(state_bad) is None

    def test_can_load_handles_common_prefixes(self):
        from libreyolo.models.eomt.model import LibreEoMT

        state = {
            f"module.model.eomt.{k}": v for k, v in _synthetic_eomt_state().items()
        }
        assert LibreEoMT.can_load(state)
        assert LibreEoMT.detect_size(state) == "l"

    def test_normalize_strips_compound_prefixes(self):
        """Regression: compound prefixes (torch.compile + DDP -> _orig_mod.module.)
        must be fully stripped, not just the outer layer (single-pass bug)."""
        from libreyolo.models.eomt.nn import normalize_eomt_state_dict

        raw = {
            "_orig_mod.module.eomt.layers.0.weight": torch.zeros(2),
            "module.model.class_predictor.bias": torch.zeros(2),
            "already.clean.key": torch.zeros(2),
        }
        out = normalize_eomt_state_dict(raw)
        assert "layers.0.weight" in out
        assert "class_predictor.bias" in out
        assert "already.clean.key" in out
        assert not any(
            k.startswith(("module.", "_orig_mod.", "model.", "eomt.")) for k in out
        )

    def test_can_load_rejects_other_dense_families(self):
        from libreyolo.models.eomt.model import LibreEoMT

        dinov2_state = {
            "backbone.encoder.proj.weight": torch.zeros(1),
            "predict.weight": torch.zeros(150, 8, 1, 1),
        }
        depth_state = {
            "pretrained.cls_token": torch.zeros(1),
            "depth_head.scratch.output_conv1.weight": torch.zeros(1),
        }
        assert not LibreEoMT.can_load(dinov2_state)
        assert not LibreEoMT.can_load(depth_state)

    def test_filename_and_download_url(self):
        from libreyolo.models.eomt.model import LibreEoMT

        filename = "LibreEoMTl-sem.pt"
        assert LibreEoMT.detect_size_from_filename(filename) == "l"
        assert LibreEoMT.detect_task_from_filename(filename) == "semantic"
        assert LibreEoMT.detect_size_from_filename("LibreEoMTl.pt") is None
        assert LibreEoMT.get_download_url(filename) == (
            "https://huggingface.co/LibreYOLO/LibreEoMTl-sem/resolve/main/"
            "LibreEoMTl-sem.pt"
        )
        assert LibreEoMT.detect_task_from_filename("LibreEoMTl-seg.pt") == "segment"

    def test_wrong_task_raises(self, fake_eomt_net):
        from libreyolo.models.eomt.model import LibreEoMT

        with pytest.raises(ValueError, match="not supported"):
            LibreEoMT(model_path=None, size="l", task="detect", device="cpu")


class TestEoMTPredict:
    def test_preprocess_splits_wide_image_like_hf_processor(
        self,
        fake_eomt_net,
        tmp_path,
    ):
        from libreyolo.models.eomt.model import LibreEoMT

        img_path = tmp_path / "wide.jpg"
        Image.new("RGB", (640, 480), color=(50, 90, 130)).save(img_path)
        model = LibreEoMT(
            model_path=None, size="l", task="semantic", nb_classes=3, device="cpu"
        )

        tensor, _, original_size, _ = model._preprocess(
            str(img_path),
            color_format="rgb",
            input_size=512,
        )

        assert tuple(tensor.shape) == (2, 3, 512, 512)
        assert original_size == (640, 480)
        assert model._last_eomt_resized_shape == (512, 682)
        assert model._last_eomt_patch_offsets == [(0, 0, 512), (0, 170, 682)]

    def test_predict_returns_semantic_mask(self, fake_eomt_net, tmp_path):
        from libreyolo.models.eomt.model import LibreEoMT

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (90, 45), color=(50, 90, 130)).save(img_path)

        model = LibreEoMT(
            model_path=None, size="l", task="semantic", nb_classes=3, device="cpu"
        )
        result = model.predict(str(img_path), imgsz=512)

        assert result.boxes is None
        assert result.masks is None
        assert result.semantic_mask is not None
        assert tuple(result.semantic_mask.data.shape) == (45, 90)
        assert set(torch.unique(result.semantic_mask.data).tolist()) <= {0, 1, 2}

    def test_predict_augment_returns_semantic_mask(self, fake_eomt_net, tmp_path):
        from libreyolo.models.eomt.model import LibreEoMT

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (90, 45), color=(50, 90, 130)).save(img_path)

        model = LibreEoMT(
            model_path=None, size="l", task="semantic", nb_classes=3, device="cpu"
        )
        result = model.predict(str(img_path), imgsz=512, augment=True)

        assert result.boxes is None
        assert result.semantic_mask is not None
        assert tuple(result.semantic_mask.data.shape) == (45, 90)
        assert set(torch.unique(result.semantic_mask.data).tolist()) <= {0, 1, 2}

    def test_predict_rejects_non_patch_imgsz(self, fake_eomt_net, tmp_path):
        from libreyolo.models.eomt.model import LibreEoMT

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (64, 64), color=(10, 20, 30)).save(img_path)
        model = LibreEoMT(
            model_path=None, size="l", task="semantic", nb_classes=2, device="cpu"
        )

        with pytest.raises(ValueError, match="divisible by 16"):
            model.predict(str(img_path), imgsz=66)

    def test_predict_rejects_non_native_imgsz(self, fake_eomt_net, tmp_path):
        from libreyolo.models.eomt.model import LibreEoMT

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (64, 64), color=(10, 20, 30)).save(img_path)
        model = LibreEoMT(
            model_path=None, size="l", task="semantic", nb_classes=2, device="cpu"
        )

        with pytest.raises(ValueError, match="requires imgsz=512"):
            model.predict(str(img_path), imgsz=64)

    def test_train_out_of_scope(self, fake_eomt_net):
        from libreyolo.models.eomt.model import LibreEoMT

        model = LibreEoMT(
            model_path=None, size="l", task="semantic", nb_classes=2, device="cpu"
        )
        with pytest.raises(NotImplementedError):
            model.train(data="ade20k.yaml")

    @pytest.mark.parametrize("format", ["onnx", "torchscript"])
    def test_exported_semantic_parity(self, fake_eomt_net, tmp_path, format):
        if format == "onnx":
            pytest.importorskip("onnx")
            pytest.importorskip("onnxruntime")

        import numpy as np

        from libreyolo import LibreYOLO
        from libreyolo.models.eomt.model import LibreEoMT

        model = LibreEoMT(
            model_path=None, size="s", task="semantic", nb_classes=3, device="cpu"
        )
        model.model.eval()
        image = np.random.default_rng(17).integers(
            0, 256, size=(512, 512, 3), dtype=np.uint8
        )
        native = model.predict(image, imgsz=512).semantic_mask.data
        suffix = ".onnx" if format == "onnx" else ".torchscript"
        artifact = tmp_path / f"eomt_semantic{suffix}"
        model.export(
            format=format,
            output_path=str(artifact),
            imgsz=512,
            dynamic=False,
            simplify=False,
        )
        exported = LibreYOLO(str(artifact), device="cpu").predict(image)
        agreement = (native == exported.semantic_mask.data).float().mean().item()
        assert agreement > 0.95

    @pytest.mark.slow
    @pytest.mark.skipif(
        os.environ.get("LIBREYOLO_RUN_REAL_EXPORT_PARITY") != "1",
        reason="set LIBREYOLO_RUN_REAL_EXPORT_PARITY=1 for real EoMT export parity",
    )
    @pytest.mark.parametrize("format", ["onnx", "torchscript"])
    def test_real_architecture_export_raw_parity(self, tmp_path, format):
        if format == "onnx":
            pytest.importorskip("onnx")
            pytest.importorskip("onnxruntime")

        import numpy as np

        from libreyolo import LibreEoMT, LibreYOLO
        from libreyolo.export.exporter import OnnxExporter

        torch.manual_seed(0)
        model = LibreEoMT(
            model_path=None,
            size="s",
            task="semantic",
            nb_classes=3,
            device="cpu",
        )
        model.model.eval()
        tensor = torch.rand(1, 3, 512, 512)
        exporter = OnnxExporter(model)
        with exporter._model_context("cpu", False, False, 1, (512, 512)) as (
            wrapped,
            _,
        ):
            with torch.no_grad():
                expected = wrapped(tensor)
        if isinstance(expected, torch.Tensor):
            expected = (expected,)

        artifact = model.export(
            format=format,
            imgsz=512,
            dynamic=False,
            simplify=False,
            output_path=str(tmp_path / f"eomt-semantic.{format}"),
        )
        actual = LibreYOLO(artifact, device="cpu")._run_inference(tensor.numpy())

        assert len(actual) == len(expected)
        rtol, atol = (2e-3, 2e-2) if format == "onnx" else (1e-3, 1e-3)
        for actual_output, expected_output in zip(actual, expected):
            np.testing.assert_allclose(
                actual_output,
                expected_output.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
            )


class _FakeEoMTNetSeg(nn.Module):
    """Fake EoMT net that returns query-level outputs for segment task testing."""

    def __init__(
        self,
        config: str = "l",
        nb_classes: int = 80,
        image_size: int = 640,
        num_queries: int = 100,
    ):
        super().__init__()
        if config not in ("s", "b", "l"):
            raise ValueError(f"test fake: unsupported size {config!r}")
        self.nb_classes = int(nb_classes)
        self.image_size = int(image_size)
        self.num_queries = int(num_queries)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = x.shape
        return {
            "semantic_logits": torch.zeros(b, self.nb_classes, h, w),
            "class_queries_logits": torch.zeros(
                b, self.num_queries, self.nb_classes + 1
            ),
            "masks_queries_logits": torch.zeros(b, self.num_queries, h, w),
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        from torch.nn.modules.module import _IncompatibleKeys

        return _IncompatibleKeys([], [])


@pytest.fixture
def fake_eomt_seg_net(monkeypatch):
    import libreyolo.models.eomt.model as eomt_model

    monkeypatch.setattr(eomt_model, "LibreEoMTNet", _FakeEoMTNetSeg)
    return _FakeEoMTNetSeg


class TestEoMTSegment:
    def test_segment_task_construction(self, fake_eomt_seg_net):
        from libreyolo.models.eomt.model import LibreEoMT

        model = LibreEoMT(
            model_path=None, size="l", task="segment", nb_classes=80, device="cpu"
        )
        assert model.task == "segment"
        assert model.input_size == 640

    def test_segment_postprocess_returns_instance_fields(
        self, fake_eomt_seg_net, tmp_path
    ):
        from libreyolo.models.eomt.model import LibreEoMT

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (80, 60), color=(10, 20, 30)).save(img_path)

        model = LibreEoMT(
            model_path=None, size="l", task="segment", nb_classes=80, device="cpu"
        )
        result = model.predict(str(img_path), imgsz=640)

        # With zero logits and conf=0.25, no instances should pass threshold.
        assert result.semantic_mask is None
        assert result.boxes is not None or result.masks is None  # empty or absent
        assert result.boxes is None or len(result.boxes) == 0

    def test_segment_postprocess_segment_direct(self, fake_eomt_seg_net):
        """Call _postprocess_segment directly with synthetic logits above threshold."""
        import torch

        from libreyolo.models.eomt.model import LibreEoMT

        model = LibreEoMT(
            model_path=None, size="l", task="segment", nb_classes=3, device="cpu"
        )
        num_queries, h, w = 4, 64, 64
        # Make query 0 confident on class 1 by giving it a high score.
        class_logits = torch.full((1, num_queries, 4), -10.0)
        class_logits[0, 0, 1] = 10.0  # query 0 → class 1 with high confidence
        mask_logits = torch.zeros(1, num_queries, h, w)
        mask_logits[0, 0, 16:48, 16:48] = 5.0  # query 0 has a mask patch

        output = {
            "class_queries_logits": class_logits,
            "masks_queries_logits": mask_logits,
        }
        det = model._postprocess_segment(
            output, conf_thres=0.1, iou_thres=0.5, original_size=(w, h)
        )
        assert det["num_detections"] >= 1
        assert len(det["boxes"]) == det["num_detections"]
        assert det["masks"].shape[0] == det["num_detections"]
        assert det["masks"].shape[1] == h
        assert det["masks"].shape[2] == w

    def test_single_patch_uses_topk_not_nms(self, fake_eomt_seg_net):
        """Single-patch EoMT must use top-k (DETR axiom), not NMS.
        Two queries with high overlap should NOT be suppressed by NMS."""
        import torch

        from libreyolo.models.eomt.model import LibreEoMT

        model = LibreEoMT(
            model_path=None, size="l", task="segment", nb_classes=3, device="cpu"
        )
        h, w, num_queries = 64, 64, 4
        # Two queries both confident on different classes, with full-image masks
        # that would have IoU=1.0 — NMS at 0.5 would suppress one; top-k keeps both.
        # Remaining queries have uniform logits → score ≈0.25 < conf_thres=0.5.
        class_logits = torch.full((1, num_queries, 4), -10.0)
        class_logits[0, 0, 1] = 10.0  # query 0 → class 1, score ≈1.0
        class_logits[0, 1, 2] = 10.0  # query 1 → class 2, score ≈1.0
        # queries 2 and 3: uniform → score ≈0.25, filtered by conf_thres=0.5
        mask_logits = torch.full(
            (1, num_queries, h, w), 5.0
        )  # all masks fill the image

        output = {
            "class_queries_logits": class_logits,
            "masks_queries_logits": mask_logits,
        }
        # num_patches=1 (single patch) → top-k: both detections survive
        det = model._postprocess_segment(
            output, conf_thres=0.5, iou_thres=0.5, original_size=(w, h)
        )
        assert det["num_detections"] == 2, (
            "Single-patch segment must use top-k (DETR axiom): "
            "two overlapping queries should both survive, NMS would drop one"
        )

    def test_segment_checkpoint_round_trip(self, fake_eomt_seg_net, tmp_path):
        from libreyolo import LibreYOLO
        from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

        ckpt = wrap_libreyolo_checkpoint(
            _synthetic_eomt_state(nc=80),
            model_family="eomt",
            size="l",
            task="segment",
            nc=80,
            names={i: f"coco_{i}" for i in range(80)},
            imgsz=640,
        )
        ckpt_path = tmp_path / "LibreEoMTl-seg.pt"
        torch.save(ckpt, str(ckpt_path))

        loaded = LibreYOLO(str(ckpt_path), device="cpu")
        assert loaded.FAMILY == "eomt"
        assert loaded.task == "segment"
        assert loaded.size == "l"
        assert loaded.nb_classes == 80
        assert loaded.input_size == 640


class TestEoMTSizes:
    """Tests for small/base backbone size support and large-1280 variant."""

    def test_small_semantic_construction(self, fake_eomt_net):
        from libreyolo.models.eomt.model import LibreEoMT

        model = LibreEoMT(
            model_path=None, size="s", task="semantic", nb_classes=150, device="cpu"
        )
        assert model.size == "s"
        assert model.task == "semantic"
        assert model.input_size == 512

    def test_base_semantic_construction(self, fake_eomt_net):
        from libreyolo.models.eomt.model import LibreEoMT

        model = LibreEoMT(
            model_path=None, size="b", task="semantic", nb_classes=150, device="cpu"
        )
        assert model.size == "b"
        assert model.input_size == 512

    def test_small_segment_construction(self, fake_eomt_seg_net):
        from libreyolo.models.eomt.model import LibreEoMT

        model = LibreEoMT(
            model_path=None, size="s", task="segment", nb_classes=80, device="cpu"
        )
        assert model.size == "s"
        assert model.task == "segment"
        assert model.input_size == 640

    def test_base_segment_construction(self, fake_eomt_seg_net):
        from libreyolo.models.eomt.model import LibreEoMT

        model = LibreEoMT(
            model_path=None, size="b", task="segment", nb_classes=80, device="cpu"
        )
        assert model.size == "b"
        assert model.input_size == 640

    def test_checkpoint_round_trip_base_segment(self, fake_eomt_seg_net, tmp_path):
        from libreyolo import LibreYOLO
        from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

        ckpt = wrap_libreyolo_checkpoint(
            _synthetic_eomt_state(nc=80, hidden=768),
            model_family="eomt",
            size="b",
            task="segment",
            nc=80,
            names={i: f"coco_{i}" for i in range(80)},
            imgsz=640,
        )
        ckpt_path = tmp_path / "LibreEoMTb-seg.pt"
        torch.save(ckpt, str(ckpt_path))

        loaded = LibreYOLO(str(ckpt_path), device="cpu")
        assert loaded.size == "b"
        assert loaded.task == "segment"
        assert loaded.input_size == 640

    def test_auto_task_and_size_from_filename(self, fake_eomt_seg_net, tmp_path):
        """LibreEoMT(path) infers task and size from the checkpoint without task= kwarg."""
        from libreyolo.models.eomt.model import LibreEoMT
        from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

        ckpt = wrap_libreyolo_checkpoint(
            _synthetic_eomt_state(nc=80, hidden=768),
            model_family="eomt",
            size="b",
            task="segment",
            nc=80,
            names={i: f"c_{i}" for i in range(80)},
            imgsz=640,
        )
        ckpt_path = tmp_path / "LibreEoMTb-seg.pt"
        torch.save(ckpt, str(ckpt_path))

        # No task= or size= — both auto-detected.
        model = LibreEoMT(str(ckpt_path), device="cpu")
        assert model.size == "b"
        assert model.task == "segment"
        assert model.input_size == 640

    def test_auto_task_from_panoptic_filename(self, fake_eomt_seg_net, tmp_path):
        """-panoptic.pt is a first-class panoptic checkpoint, not segment in disguise."""
        from libreyolo.models.eomt.model import LibreEoMT
        from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

        names = {i: f"pan_{i}" for i in range(133)}
        ckpt = wrap_libreyolo_checkpoint(
            _synthetic_eomt_state(nc=133, hidden=768),
            model_family="eomt",
            size="b",
            task="panoptic",
            nc=133,
            names=names,
            imgsz=640,
            thing_class_ids=list(range(80)),
        )
        ckpt_path = tmp_path / "LibreEoMTb-panoptic.pt"
        torch.save(ckpt, str(ckpt_path))

        model = LibreEoMT(str(ckpt_path), device="cpu")
        assert model.size == "b"
        assert model.task == "panoptic"
        assert model.nb_classes == 133
        # The thing/stuff split rides along as category metadata so the panoptic
        # merge knows which categories to fuse.
        assert model.thing_class_ids == list(range(80))
        assert model._stuff_class_ids() == set(range(80, 133))

    def test_converter_large_1280_segment(self, monkeypatch, tmp_path):
        converter = _load_converter_module()

        def _fake_load(source, *, allow_unverified_source=False):
            return _synthetic_eomt_state(nc=80, hidden=1024)

        monkeypatch.setattr(converter, "_load_state_dict", _fake_load)
        out_path = tmp_path / "LibreEoMTl-seg-1280.pt"

        ckpt = converter.convert_weights(
            converter.COCO_HF_REPO_1280, str(out_path), task="segment", imgsz=1280
        )

        assert ckpt["size"] == "l"
        assert ckpt["task"] == "segment"
        assert ckpt["nc"] == 80
        assert ckpt["imgsz"] == 1280

    def test_converter_default_output_path(self):
        converter = _load_converter_module()

        assert (
            converter._default_output_path("semantic", "l", 512)
            == "weights/LibreEoMTl-sem.pt"
        )
        assert (
            converter._default_output_path("segment", "l", 640)
            == "weights/LibreEoMTl-seg.pt"
        )
        assert (
            converter._default_output_path("segment", "l", 1280)
            == "weights/LibreEoMTl-seg-1280.pt"
        )
        assert (
            converter._default_output_path("segment", "s", 640)
            == "weights/LibreEoMTs-seg.pt"
        )
        assert (
            converter._default_output_path("segment", "b", 640)
            == "weights/LibreEoMTb-seg.pt"
        )
        assert (
            converter._default_output_path("panoptic", "s", 640)
            == "weights/LibreEoMTs-panoptic.pt"
        )
        assert (
            converter._default_output_path("panoptic", "b", 640)
            == "weights/LibreEoMTb-panoptic.pt"
        )
        assert (
            converter._default_output_path("panoptic", "l", 640)
            == "weights/LibreEoMTl-panoptic.pt"
        )

    def test_converter_things_only_slices_panoptic(self, monkeypatch, tmp_path):
        converter = _load_converter_module()

        def _fake_load(source, *, allow_unverified_source=False):
            # Panoptic checkpoint: nc=133, hidden=384 (small backbone)
            return _synthetic_eomt_state(nc=133, hidden=384)

        monkeypatch.setattr(converter, "_load_state_dict", _fake_load)
        out_path = tmp_path / "LibreEoMTs-seg.pt"

        ckpt = converter.convert_weights(
            converter.COCO_PANOPTIC_HF_REPO_S,
            str(out_path),
            task="segment",
            things_only=True,
        )

        assert ckpt["size"] == "s"
        assert ckpt["task"] == "segment"
        assert ckpt["nc"] == 80
        assert ckpt["imgsz"] == 640
        # All three nc-dependent tensors sliced to 81 rows (80 things + null).
        assert ckpt["model"]["class_predictor.weight"].shape == (81, 384)
        assert ckpt["model"]["class_predictor.bias"].shape == (81,)
        assert ckpt["model"]["criterion.empty_weight"].shape == (81,)

    def test_converter_things_only_rejects_non_panoptic(self, monkeypatch, tmp_path):
        converter = _load_converter_module()

        def _fake_load(source, *, allow_unverified_source=False):
            return _synthetic_eomt_state(nc=80, hidden=1024)

        monkeypatch.setattr(converter, "_load_state_dict", _fake_load)

        with pytest.raises(ValueError, match="133"):
            converter.convert_weights(
                converter.COCO_HF_REPO,
                str(tmp_path / "out.pt"),
                task="segment",
                things_only=True,
            )

    def test_converter_things_only_rejects_semantic_task(self, tmp_path):
        converter = _load_converter_module()

        with pytest.raises(ValueError, match="--things-only"):
            converter.convert_weights(
                converter.DEFAULT_HF_REPO,
                str(tmp_path / "out.pt"),
                task="semantic",
                things_only=True,
            )


def test_val_smoke_uses_split_inference_path(fake_eomt_net, tmp_path):
    from libreyolo.models.eomt.model import LibreEoMT

    for i in range(2):
        img_dir = tmp_path / "images" / "val"
        mask_dir = tmp_path / "masks" / "val"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(20 + i, 30, 40)).save(img_dir / f"img{i}.jpg")
        Image.new("L", (64, 64), color=i % 2).save(mask_dir / f"img{i}.png")

    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val",
                "masks_dir": "masks",
                "nc": 2,
                "names": {0: "left", 1: "right"},
            }
        )
    )
    model = LibreEoMT(
        model_path=None, size="l", task="semantic", nb_classes=2, device="cpu"
    )

    metrics = model.val(
        data=str(yaml_path),
        imgsz=512,
        batch=2,
        workers=4,
        save_dir=str(tmp_path / "val_out"),
        save_plots=True,
        data_dir=None,
        max_det=300,
        half=False,
        verbose=False,
    )

    assert "metrics/mIoU" in metrics
    assert 0.0 <= metrics["metrics/mIoU"] <= 1.0
    assert "speed/preprocess_ms" in metrics
    assert (tmp_path / "val_out" / "config.yaml").exists()


def test_val_augment_smoke(fake_eomt_net, tmp_path):
    """augment=True must run LibreEoMT's own flip-TTA branch, not raise."""
    from libreyolo.models.eomt.model import LibreEoMT

    for i in range(2):
        img_dir = tmp_path / "images" / "val"
        mask_dir = tmp_path / "masks" / "val"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(20 + i, 30, 40)).save(img_dir / f"img{i}.jpg")
        Image.new("L", (64, 64), color=i % 2).save(mask_dir / f"img{i}.png")

    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val",
                "masks_dir": "masks",
                "nc": 2,
                "names": {0: "left", 1: "right"},
            }
        )
    )
    model = LibreEoMT(
        model_path=None, size="l", task="semantic", nb_classes=2, device="cpu"
    )

    metrics = model.val(
        data=str(yaml_path),
        imgsz=512,
        augment=True,
        save_dir=str(tmp_path / "val_out"),
        verbose=False,
    )

    assert "metrics/mIoU" in metrics
    assert 0.0 <= metrics["metrics/mIoU"] <= 1.0


def test_val_segment_routes_to_base_val(fake_eomt_seg_net, monkeypatch):
    """val() on a segment model delegates to BaseModel.val() without hitting
    the semantic-specific code path (which would reject COCO data)."""
    from unittest.mock import MagicMock

    from libreyolo.models.eomt.model import LibreEoMT
    from libreyolo.models.base.model import BaseModel

    model = LibreEoMT(
        model_path=None, size="l", task="segment", nb_classes=80, device="cpu"
    )

    sentinel = MagicMock(return_value={"metrics/mAP50": 0.0})
    monkeypatch.setattr(BaseModel, "val", sentinel)

    model.val(
        data="coco.yaml",
        imgsz=640,
        half=True,
        save_dir="runs/val/exp",
        save_plots=True,
    )

    sentinel.assert_called_once()
    _, kwargs = sentinel.call_args
    assert kwargs.get("data") == "coco.yaml"
    # half/save_dir/save_plots must reach BaseModel.val(), not be silently
    # swallowed by the outer LibreEoMT.val() signature.
    assert kwargs.get("half") is True
    assert kwargs.get("save_dir") == "runs/val/exp"
    assert kwargs.get("save_plots") is True


def test_val_panoptic_routes_to_base_val(fake_eomt_seg_net, monkeypatch):
    """val() on a panoptic model delegates to BaseModel.val(), whose dispatch
    selects PanopticValidator; the semantic dense-mask loop would reject
    COCO-panoptic data with 'requires dense PNG masks'."""
    from unittest.mock import MagicMock

    from libreyolo.models.eomt.model import LibreEoMT
    from libreyolo.models.base.model import BaseModel

    model = LibreEoMT(
        model_path=None, size="s", task="panoptic", nb_classes=133, device="cpu"
    )

    sentinel = MagicMock(return_value={"metrics/PQ": 0.0})
    monkeypatch.setattr(BaseModel, "val", sentinel)

    model.val(data="coco_panoptic.yaml")

    sentinel.assert_called_once()
    _, kwargs = sentinel.call_args
    assert kwargs.get("data") == "coco_panoptic.yaml"


def test_val_rejects_unknown_kwargs(fake_eomt_net):
    from libreyolo.models.eomt.model import LibreEoMT

    model = LibreEoMT(
        model_path=None, size="l", task="semantic", nb_classes=2, device="cpu"
    )

    with pytest.raises(TypeError, match="unexpected keyword"):
        model.val(unused_option=True)


def test_checkpoint_round_trip_through_factory(fake_eomt_net, tmp_path):
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    ckpt = wrap_libreyolo_checkpoint(
        _synthetic_eomt_state(nc=150),
        model_family="eomt",
        size="l",
        task="semantic",
        nc=150,
        names={i: f"ade_{i}" for i in range(150)},
        imgsz=512,
    )
    ckpt_path = tmp_path / "LibreEoMTl-sem.pt"
    torch.save(ckpt, str(ckpt_path))

    loaded = LibreYOLO(str(ckpt_path), device="cpu")
    assert loaded.FAMILY == "eomt"
    assert loaded.task == "semantic"
    assert loaded.size == "l"
    assert loaded.nb_classes == 150
    assert loaded.names[0] == "ade_0"


def test_raw_state_dict_load_requires_converter(fake_eomt_net):
    from libreyolo.models.eomt.model import LibreEoMT

    with pytest.raises(RuntimeError, match="Convert the approved DINOv2 ADE20K"):
        LibreEoMT(
            model_path=_synthetic_eomt_state(),
            size="l",
            task="semantic",
            nb_classes=150,
            device="cpu",
        )


def test_converter_wraps_metadata(monkeypatch, tmp_path):
    converter = _load_converter_module()

    def _fake_load(source, *, allow_unverified_source=False):
        assert source == converter.DEFAULT_HF_REPO
        assert allow_unverified_source is False
        return _synthetic_eomt_state(nc=150)

    monkeypatch.setattr(converter, "_load_state_dict", _fake_load)
    out_path = tmp_path / "LibreEoMTl-sem.pt"

    ckpt = converter.convert_weights(converter.DEFAULT_HF_REPO, str(out_path))

    assert out_path.exists()
    assert ckpt["model_family"] == "eomt"
    assert ckpt["size"] == "l"
    assert ckpt["task"] == "semantic"
    assert ckpt["nc"] == 150
    assert ckpt["imgsz"] == 512
    assert ckpt["names"][0] == "wall"
    assert ckpt["names"][149] == "flag"
    assert set(
        [
            "model",
            "schema_version",
            "libreyolo_version",
            "model_family",
            "size",
            "task",
            "nc",
            "names",
            "imgsz",
        ]
    ).issubset(ckpt)


def test_converter_segment_task(monkeypatch, tmp_path):
    converter = _load_converter_module()

    def _fake_load(source, *, allow_unverified_source=False):
        return _synthetic_eomt_state(nc=80)

    monkeypatch.setattr(converter, "_load_state_dict", _fake_load)
    out_path = tmp_path / "LibreEoMTl-seg.pt"

    ckpt = converter.convert_weights(
        converter.COCO_HF_REPO, str(out_path), task="segment"
    )

    assert out_path.exists()
    assert ckpt["task"] == "segment"
    assert ckpt["nc"] == 80
    assert ckpt["imgsz"] == 640
    assert ckpt["names"][0] == "person"


def test_converter_segment_rejects_wrong_nc(monkeypatch, tmp_path):
    converter = _load_converter_module()

    def _fake_load(source, *, allow_unverified_source=False):
        return _synthetic_eomt_state(nc=150)

    monkeypatch.setattr(converter, "_load_state_dict", _fake_load)

    with pytest.raises(ValueError, match="80-class"):
        converter.convert_weights(
            "fake_source", str(tmp_path / "bad.pt"), task="segment"
        )


def test_converter_panoptic_task(monkeypatch, tmp_path):
    converter = _load_converter_module()

    def _fake_load(source, *, allow_unverified_source=False):
        return _synthetic_eomt_state(nc=133, hidden=384)  # small backbone, panoptic nc

    monkeypatch.setattr(converter, "_load_state_dict", _fake_load)
    out_path = tmp_path / "LibreEoMTs-panoptic.pt"

    ckpt = converter.convert_weights(
        converter.COCO_PANOPTIC_HF_REPO_S, str(out_path), task="panoptic"
    )

    assert out_path.exists()
    # Honest metadata: panoptic checkpoints carry task="panoptic" and the
    # thing/stuff split, not a "segment" label in disguise.
    assert ckpt["task"] == "panoptic"
    assert ckpt["thing_class_ids"] == list(range(80))
    assert ckpt["nc"] == 133
    assert ckpt["size"] == "s"
    assert ckpt["imgsz"] == 640
    # Verify proper class names — things and stuff.
    assert ckpt["names"][0] == "person"
    assert ckpt["names"][79] == "toothbrush"
    assert ckpt["names"][80] == "banner"
    assert ckpt["names"][132] == "rug-merged"


def test_converter_panoptic_rejects_wrong_nc(monkeypatch, tmp_path):
    converter = _load_converter_module()

    def _fake_load(source, *, allow_unverified_source=False):
        return _synthetic_eomt_state(nc=80)

    monkeypatch.setattr(converter, "_load_state_dict", _fake_load)

    with pytest.raises(ValueError, match="133-class"):
        converter.convert_weights(
            "fake_source", str(tmp_path / "bad.pt"), task="panoptic"
        )


def test_converter_panoptic_rejects_things_only(tmp_path):
    converter = _load_converter_module()

    with pytest.raises(ValueError, match="--things-only"):
        converter.convert_weights(
            "fake_source", str(tmp_path / "bad.pt"), task="panoptic", things_only=True
        )


def test_converter_rejects_dinov3_even_with_override(tmp_path):
    converter = _load_converter_module()

    with pytest.raises(ValueError, match="DINOv3"):
        converter.convert_weights(
            "tue-mps/eomt-dinov3-ade-semantic-large-512",
            str(tmp_path / "bad.pt"),
            allow_unverified_source=True,
        )


def test_converter_rejects_unverified_local_source(tmp_path):
    converter = _load_converter_module()
    local = tmp_path / "model.safetensors"
    local.write_bytes(b"not used")

    with pytest.raises(ValueError, match="not provenance-verifiable"):
        converter.convert_weights(str(local), str(tmp_path / "bad.pt"))


def test_builtin_ade20k_config_is_complete():
    from libreyolo.data import load_data_config

    config = load_data_config("ade20k", autodownload=False)

    assert config["nc"] == 150
    assert config["masks_dir"] == "annotations"
    assert config["ignore_index"] == 255
    assert len(config["names"]) == 150
    assert config["names"][0] == "wall"
    assert config["names"][149] == "flag"
    assert config["label_mapping"][1] == 0
    assert config["label_mapping"][150] == 149


# ---------------------------------------------------------------------------
# Panoptic merge (Mask2Former inference recipe)
# ---------------------------------------------------------------------------


def _panoptic_stub(nc: int, thing_class_ids):
    """Minimal stand-in exposing exactly what _postprocess_panoptic touches."""
    from types import SimpleNamespace

    from libreyolo.models.eomt.model import LibreEoMT

    stub = SimpleNamespace(
        PANOPTIC_SCORE_THRESHOLD=LibreEoMT.PANOPTIC_SCORE_THRESHOLD,
        PANOPTIC_MASK_THRESHOLD=LibreEoMT.PANOPTIC_MASK_THRESHOLD,
        PANOPTIC_OVERLAP_THRESHOLD=LibreEoMT.PANOPTIC_OVERLAP_THRESHOLD,
        nb_classes=nc,
        thing_class_ids=thing_class_ids,
        _last_eomt_patch_offsets=None,
        _last_eomt_resized_shape=None,
        _last_eomt_content_size=None,  # unpadded already
    )
    stub._stuff_class_ids = lambda: LibreEoMT._stuff_class_ids(stub)
    stub._unpad_and_resize_mask_logits = lambda ml, osz: (
        LibreEoMT._unpad_and_resize_mask_logits(stub, ml, osz)
    )
    stub._panoptic_queries = lambda output, osz: LibreEoMT._panoptic_queries(
        stub, output, osz
    )
    stub._fuse_panoptic_queries = lambda scores, labels, mask_probs, osz: (
        LibreEoMT._fuse_panoptic_queries(stub, scores, labels, mask_probs, osz)
    )
    stub.PANOPTIC_TTA_DEDUP_IOU = LibreEoMT.PANOPTIC_TTA_DEDUP_IOU
    stub._dedup_panoptic_queries = lambda scores, labels, mask_probs, view_ids=None: (
        LibreEoMT._dedup_panoptic_queries(stub, scores, labels, mask_probs, view_ids)
    )
    return stub


def _quadrant_panoptic_output(nc: int = 4):
    """4x4 canvas, 4 disjoint 2x2 quadrant queries + a null + a low-conf query.

    q0/q1 -> class 0 (thing) in separate quadrants  -> two distinct segments
    q2/q3 -> class 2 (stuff) in separate quadrants  -> fused into one segment
    q4    -> argmax is the null class               -> dropped
    q5    -> uniform logits (score 0.2)             -> dropped by conf_thres
    """
    quadrants = [
        (slice(0, 2), slice(0, 2)),
        (slice(0, 2), slice(2, 4)),
        (slice(2, 4), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
    ]
    classes = [0, 0, 2, 2]

    class_logits = torch.full((1, 6, nc + 1), -10.0)
    for q, c in enumerate(classes):
        class_logits[0, q, c] = 5.0
    class_logits[0, 4, nc] = 5.0  # null/no-object wins
    class_logits[0, 5, :] = 0.0  # uniform -> score 1/(nc+1) = 0.2

    mask_logits = torch.full((1, 6, 4, 4), -4.0)  # sigmoid ~0.018 outside
    for q, (rows, cols) in enumerate(quadrants):
        mask_logits[0, q, rows, cols] = 4.0  # sigmoid ~0.982 inside
    mask_logits[0, 4] = 4.0
    mask_logits[0, 5] = 4.0
    return {"class_queries_logits": class_logits, "masks_queries_logits": mask_logits}


def test_panoptic_merge_fuses_stuff_and_separates_things():
    from libreyolo.models.eomt.model import LibreEoMT

    stub = _panoptic_stub(nc=4, thing_class_ids=[0, 1])
    out = LibreEoMT._postprocess_panoptic(
        stub, _quadrant_panoptic_output(nc=4), 0.5, (4, 4)
    )
    seg, info = out["panoptic"], out["segments_info"]

    assert seg.shape == (4, 4)
    # Two thing segments (same category, separate instances) + one fused stuff.
    assert len(info) == 3
    assert sorted(e["category_id"] for e in info) == [0, 0, 2]
    assert [e["isthing"] for e in info if e["category_id"] == 0] == [True, True]
    assert [e["isthing"] for e in info if e["category_id"] == 2] == [False]

    ids = {e["id"] for e in info}
    assert 0 not in ids  # 0 is reserved for void
    assert len(ids) == 3  # one entry per distinct segment id
    assert set(seg.unique().tolist()) == ids  # every pixel labeled, no void left

    # The two stuff quadrants share one segment id (fused).
    stuff_id = next(e["id"] for e in info if e["category_id"] == 2)
    assert int((seg == stuff_id).sum()) == 8  # both bottom quadrants
    # Things stay separate: 4 pixels each.
    for e in (e for e in info if e["category_id"] == 0):
        assert int((seg == e["id"]).sum()) == 4


def test_panoptic_merge_is_non_overlapping_and_drops_null_queries():
    from libreyolo.models.eomt.model import LibreEoMT

    stub = _panoptic_stub(nc=4, thing_class_ids=[0, 1])
    out = LibreEoMT._postprocess_panoptic(
        stub, _quadrant_panoptic_output(nc=4), 0.5, (4, 4)
    )
    seg = out["panoptic"]
    # Every pixel carries exactly one id: a dense map trivially cannot overlap,
    # so the real assertion is that the null/low-conf queries never claimed one.
    assert int((seg == 0).sum()) == 0
    assert seg.dtype == torch.int32
    # 6 queries in, at most 4 could survive; null + low-conf were removed.
    assert len(out["segments_info"]) == 3


def test_panoptic_merge_without_thing_class_ids_fuses_nothing():
    from libreyolo.models.eomt.model import LibreEoMT

    stub = _panoptic_stub(nc=4, thing_class_ids=None)
    out = LibreEoMT._postprocess_panoptic(
        stub, _quadrant_panoptic_output(nc=4), 0.5, (4, 4)
    )
    # No category metadata -> nothing is stuff -> the two class-2 quadrants stay
    # separate segments instead of being silently fused.
    assert len(out["segments_info"]) == 4
    assert all(e["isthing"] for e in out["segments_info"])


def test_panoptic_merge_empty_when_all_queries_are_null():
    from libreyolo.models.eomt.model import LibreEoMT

    stub = _panoptic_stub(nc=4, thing_class_ids=[0, 1])
    class_logits = torch.full((1, 3, 5), -10.0)
    class_logits[0, :, 4] = 5.0  # every query is no-object
    output = {
        "class_queries_logits": class_logits,
        "masks_queries_logits": torch.full((1, 3, 4, 4), 4.0),
    }
    out = LibreEoMT._postprocess_panoptic(stub, output, 0.5, (4, 4))
    assert out["segments_info"] == []
    assert int(out["panoptic"].sum()) == 0


def _single_query_panoptic_output(nc, quadrant, cls):
    """A single high-confidence query occupying one quadrant of a 4x4 canvas."""
    class_logits = torch.full((1, 1, nc + 1), -10.0)
    class_logits[0, 0, cls] = 5.0
    mask_logits = torch.full((1, 1, 4, 4), -4.0)
    rows, cols = quadrant
    mask_logits[0, 0, rows, cols] = 4.0
    return {"class_queries_logits": class_logits, "masks_queries_logits": mask_logits}


def _panoptic_tta_stub(nc: int, thing_class_ids, forward_outputs: list):
    """_panoptic_stub extended with the predict-path hooks _predict_augment_panoptic
    needs: device, names, _preprocess, _forward (returns forward_outputs[call_idx])."""
    stub = _panoptic_stub(nc=nc, thing_class_ids=thing_class_ids)
    stub.device = torch.device("cpu")
    stub.names = {i: str(i) for i in range(nc)}
    calls = {"n": 0}

    def _preprocess(image, color_format="auto", input_size=None):
        return torch.zeros(1, 3, 4, 4), image, (4, 4), 1.0

    def _forward(tensor):
        out = forward_outputs[calls["n"]]
        calls["n"] += 1
        return out

    stub._preprocess = _preprocess
    stub._forward = _forward
    return stub


TOP_LEFT = (slice(0, 2), slice(0, 2))
BOTTOM_LEFT = (slice(2, 4), slice(0, 2))
BOTTOM_RIGHT = (slice(2, 4), slice(2, 4))


def test_dedup_panoptic_queries_merges_same_class_overlapping_masks():
    from libreyolo.models.eomt.model import LibreEoMT

    stub = _panoptic_stub(nc=4, thing_class_ids=[0, 1])
    scores = torch.tensor([0.9, 0.85])
    labels = torch.tensor([0, 0])
    mask_probs = torch.full((2, 4, 4), 0.02)
    mask_probs[0, 0:2, 0:2] = 0.9
    mask_probs[1, 0:2, 0:2] = 0.9  # identical quadrant -> IoU 1.0

    out_scores, out_labels, out_masks = LibreEoMT._dedup_panoptic_queries(
        stub, scores, labels, mask_probs
    )

    assert out_scores.shape == (1,)
    assert out_labels.tolist() == [0]
    assert out_scores.item() == pytest.approx((0.9 + 0.85) / 2)
    assert torch.allclose(out_masks[0], mask_probs.mean(dim=0))


def test_dedup_panoptic_queries_keeps_different_classes_separate():
    from libreyolo.models.eomt.model import LibreEoMT

    stub = _panoptic_stub(nc=4, thing_class_ids=[0, 1])
    scores = torch.tensor([0.9, 0.85])
    labels = torch.tensor([0, 1])  # different classes, identical masks
    mask_probs = torch.full((2, 4, 4), 0.02)
    mask_probs[:, 0:2, 0:2] = 0.9

    out_scores, out_labels, _ = LibreEoMT._dedup_panoptic_queries(
        stub, scores, labels, mask_probs
    )

    assert out_scores.shape == (2,)
    assert sorted(out_labels.tolist()) == [0, 1]


def test_dedup_panoptic_queries_keeps_non_overlapping_same_class_separate():
    from libreyolo.models.eomt.model import LibreEoMT

    stub = _panoptic_stub(nc=4, thing_class_ids=[0, 1])
    scores = torch.tensor([0.9, 0.85])
    labels = torch.tensor([0, 0])
    mask_probs = torch.full((2, 4, 4), 0.02)
    mask_probs[0, 0:2, 0:2] = 0.9  # top-left
    mask_probs[1, 2:4, 2:4] = 0.9  # bottom-right, disjoint -> IoU 0

    out_scores, _, _ = LibreEoMT._dedup_panoptic_queries(stub, scores, labels, mask_probs)

    assert out_scores.shape == (2,)


def test_dedup_prevents_two_agreeing_views_from_losing_to_each_other():
    """Regression for the exact failure mode measured on a real image: two
    same-class queries whose masks mostly-but-not-perfectly agree split the
    per-pixel winner-take-all between them, so *each* one's own overlap
    ratio can fall below PANOPTIC_OVERLAP_THRESHOLD and get dropped —
    losing an object both views correctly detected. The canvas here is
    exactly the 2-pixel competing region (no other "background" pixels):
    _fuse_panoptic_queries's winner is an argmax with ties resolved to the
    lowest query index, so any padding pixels where two queries are exactly
    tied would spuriously hand one of them a free, unbounded won_area and
    mask the effect under test.
    """
    from libreyolo.models.eomt.model import LibreEoMT

    stub = _panoptic_stub(nc=4, thing_class_ids=[0, 1])
    scores = torch.tensor([0.9, 0.9])
    labels = torch.tensor([0, 0])
    # A 2x1 canvas: query A confident on row 0, only-just-over-threshold on
    # row 1; query B is the mirror image. Both "own" both pixels, but each
    # only wins the pixel where it's more confident.
    mask_probs = torch.tensor([[[0.95], [0.62]], [[0.62], [0.95]]])
    view_ids = torch.tensor([0, 1])  # one query per view, as flip-TTA produces

    undeduped = LibreEoMT._fuse_panoptic_queries(stub, scores, labels, mask_probs, (1, 2))
    assert undeduped["segments_info"] == []  # both queries lose to each other

    d_scores, d_labels, d_masks = LibreEoMT._dedup_panoptic_queries(
        stub, scores, labels, mask_probs, view_ids
    )
    deduped = LibreEoMT._fuse_panoptic_queries(stub, d_scores, d_labels, d_masks, (1, 2))
    assert len(deduped["segments_info"]) == 1
    assert deduped["segments_info"][0]["category_id"] == 0
    assert int((deduped["panoptic"] != 0).sum()) == 2


def test_dedup_panoptic_queries_never_merges_same_view_duplicates():
    """Two distinct, same-class, overlapping-mask queries from the SAME view
    must stay separate. Dedup exists to merge one real object seen twice
    across the two flip-TTA views, not to second-guess a single view's own
    instance separation — e.g. two overlapping people in a crowd, which a
    single forward pass can legitimately emit as two high-IoU same-class
    queries. Without the view_ids guard this would wrongly collapse them
    into one segment (reported as a Greptile review finding on this PR)."""
    from libreyolo.models.eomt.model import LibreEoMT

    stub = _panoptic_stub(nc=4, thing_class_ids=[0, 1])
    scores = torch.tensor([0.9, 0.85])
    labels = torch.tensor([0, 0])
    mask_probs = torch.full((2, 4, 4), 0.02)
    mask_probs[0, 0:2, 0:2] = 0.9
    mask_probs[1, 0:2, 0:2] = 0.9  # identical quadrant -> IoU 1.0
    view_ids = torch.tensor([0, 0])  # both from the same view

    out_scores, out_labels, _ = LibreEoMT._dedup_panoptic_queries(
        stub, scores, labels, mask_probs, view_ids
    )

    assert out_scores.shape == (2,)  # kept separate despite IoU 1.0
    assert out_labels.tolist() == [0, 0]


def test_predict_augment_panoptic_tags_queries_with_their_source_view():
    """Integration check that _predict_augment_panoptic actually wires
    per-query view_ids through to _dedup_panoptic_queries (the unit-level
    fix is tested directly in
    test_dedup_panoptic_queries_never_merges_same_view_duplicates; this
    confirms the caller doesn't drop that information on the way in).
    Two same-class, identical-mask queries both coming from the ORIGINAL
    view must reach fusion un-merged (2 queries in), not folded into 1 by
    dedup, even though their mask IoU is 1.0 -- well above
    PANOPTIC_TTA_DEDUP_IOU."""
    from PIL import Image

    from libreyolo.models.eomt.model import LibreEoMT

    def _two_overlapping_queries_output(nc=4):
        class_logits = torch.full((1, 2, nc + 1), -10.0)
        class_logits[0, :, 0] = 5.0
        mask_logits = torch.full((1, 2, 4, 4), -4.0)
        mask_logits[0, :, 0:2, 0:2] = 4.0
        return {"class_queries_logits": class_logits, "masks_queries_logits": mask_logits}

    all_null = {
        "class_queries_logits": torch.full((1, 2, 5), -10.0).index_fill_(
            -1, torch.tensor([4]), 5.0
        ),
        "masks_queries_logits": torch.full((1, 2, 4, 4), 4.0),
    }
    stub = _panoptic_tta_stub(
        nc=4,
        thing_class_ids=[0, 1],
        forward_outputs=[_two_overlapping_queries_output(nc=4), all_null],
    )
    seen_query_count = {}
    real_fuse = stub._fuse_panoptic_queries

    def _spy_fuse(scores, labels, mask_probs, osz):
        seen_query_count["n"] = mask_probs.shape[0]
        return real_fuse(scores, labels, mask_probs, osz)

    stub._fuse_panoptic_queries = _spy_fuse

    LibreEoMT._predict_augment_panoptic(
        stub,
        Image.new("RGB", (4, 4)),
        image_path=None,
        original_size=(4, 4),
        effective_imgsz=4,
        color_format="auto",
    )

    # Both same-view queries reach fusion un-merged: dedup only collapses
    # cross-view duplicates.
    assert seen_query_count["n"] == 2


def test_predict_augment_panoptic_concatenates_queries_across_views():
    """The flipped view's bottom-left detection must land at bottom-right
    after flip-back, and both views' detections must survive into the
    merged result — the whole point of the concatenate-then-fuse prototype."""
    from PIL import Image

    from libreyolo.models.eomt.model import LibreEoMT

    stub = _panoptic_tta_stub(
        nc=4,
        thing_class_ids=[0, 1],
        forward_outputs=[
            _single_query_panoptic_output(nc=4, quadrant=TOP_LEFT, cls=0),
            _single_query_panoptic_output(nc=4, quadrant=BOTTOM_LEFT, cls=1),
        ],
    )

    result = LibreEoMT._predict_augment_panoptic(
        stub,
        Image.new("RGB", (4, 4)),
        image_path=None,
        original_size=(4, 4),
        effective_imgsz=4,
        color_format="auto",
    )

    seg = result.panoptic.data
    info = result.panoptic.segments_info
    assert len(info) == 2
    assert sorted(e["category_id"] for e in info) == [0, 1]

    top_left_id = next(e["id"] for e in info if e["category_id"] == 0)
    assert int((seg[0:2, 0:2] == top_left_id).sum()) == 4

    bottom_right_id = next(e["id"] for e in info if e["category_id"] == 1)
    assert int((seg[2:4, 2:4] == bottom_right_id).sum()) == 4
    # Nothing survives where the flipped view's raw (pre-flip-back) mask was.
    assert int((seg[2:4, 0:2] != 0).sum()) == 0


def test_predict_augment_panoptic_when_one_view_finds_nothing():
    """One view surviving 0 queries must not break concatenation with the
    other view's real detection."""
    from PIL import Image

    from libreyolo.models.eomt.model import LibreEoMT

    all_null = {
        "class_queries_logits": torch.full((1, 2, 5), -10.0).index_fill_(
            -1, torch.tensor([4]), 5.0
        ),
        "masks_queries_logits": torch.full((1, 2, 4, 4), 4.0),
    }
    stub = _panoptic_tta_stub(
        nc=4,
        thing_class_ids=[0, 1],
        forward_outputs=[
            all_null,
            _single_query_panoptic_output(nc=4, quadrant=TOP_LEFT, cls=0),
        ],
    )

    result = LibreEoMT._predict_augment_panoptic(
        stub,
        Image.new("RGB", (4, 4)),
        image_path=None,
        original_size=(4, 4),
        effective_imgsz=4,
        color_format="auto",
    )

    info = result.panoptic.segments_info
    assert len(info) == 1
    assert info[0]["category_id"] == 0
    # The flipped view's top-left query flips back to top-right.
    seg = result.panoptic.data
    assert int((seg[0:2, 2:4] == info[0]["id"]).sum()) == 4


def test_base_model_predict_augment_panoptic_default_raises():
    """Families that don't implement panoptic flip-TTA get a clear error,
    not a silent no-op or an AttributeError."""
    from types import SimpleNamespace

    from libreyolo.models.base.model import BaseModel

    model = SimpleNamespace(task="panoptic")
    with pytest.raises(NotImplementedError, match="panoptic flip-TTA"):
        BaseModel._predict_augment_panoptic(model, None, None, (1, 1), 1, "auto")


def test_coco_content_size_matches_upstream_aspect_ratio_rule():
    """COCO checkpoints resize the longest edge to 640, preserving aspect ratio.

    Expected values captured from the upstream EoMT image processor's
    get_size_with_aspect_ratio(size, shortest_edge=640, longest_edge=640).
    """
    from libreyolo.models.eomt.model import LibreEoMT

    cases = {
        (576, 768): (480, 640),  # landscape
        (1194, 1536): (498, 640),  # landscape, rounds to 498
        (852, 1280): (426, 640),  # landscape
        (640, 640): (640, 640),  # already square
    }
    for (oh, ow), expected in cases.items():
        assert LibreEoMT._coco_content_size(oh, ow, 640) == expected


def test_preprocess_pads_for_coco_tasks_and_splits_for_semantic(fake_eomt_seg_net):
    """Only the ADE20K semantic checkpoint uses sliding-window patches.

    Splitting a COCO image would hand the same object to two patches as two
    independent queries, which the panoptic overlap check then discards.
    """
    from libreyolo.models.eomt.model import LibreEoMT

    img = Image.new("RGB", (768, 576), color=(120, 30, 200))

    seg = LibreEoMT(
        model_path=None, size="l", task="segment", nb_classes=80, device="cpu"
    )
    tensor, _, orig_size, _ = seg._preprocess(img, "rgb")
    assert tuple(tensor.shape) == (1, 3, 640, 640)  # single padded image
    assert seg._last_eomt_patch_offsets is None
    assert seg._last_eomt_content_size == (480, 640)
    assert orig_size == (768, 576)
    # Padding is zeros in [0, 1] space (the net normalizes afterwards).
    assert float(tensor[0, :, 500:, :].abs().max()) == 0.0
    assert float(tensor[0, :, :480, :640].abs().max()) > 0.0
