"""Unit tests for LibreDINOv2 semantic segmentation.

Structural tests run against a lightweight fake backbone (monkeypatched
``build_backbone``) so they stay hermetic; one real-backbone forward test is
network-marked for nightly runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

pytestmark = pytest.mark.unit


def _fake_backbone_factory(hidden_dim: int, num_levels: int):
    from libreyolo.models.rfdetr.nn import NestedTensor

    class _FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, hidden_dim, 14, stride=14)

        def forward(self, nested):
            x = self.proj(nested.tensors)
            levels = [x]
            for _ in range(num_levels - 1):
                x = F.max_pool2d(x, 2, ceil_mode=True)
                levels.append(x)
            return [NestedTensor(t, None) for t in levels]

    return _FakeBackbone()


@pytest.fixture
def fake_backbone(monkeypatch):
    """Replace the DINOv2 backbone build with a tiny conv pyramid."""
    import libreyolo.models.rfdetr.nn as rfdetr_nn

    def _build(load_dinov2_weights=True, **kwargs):
        backbone = _fake_backbone_factory(
            kwargs["hidden_dim"], len(kwargs["projector_scale"])
        )
        return nn.Sequential(backbone, nn.Identity())

    monkeypatch.setattr(rfdetr_nn, "build_backbone", _build)
    return _build


class TestDINOv2Metadata:
    def test_task_registration(self):
        from libreyolo.models.dinov2.model import LibreDINOv2

        assert "semantic" in LibreDINOv2.SUPPORTED_TASKS
        assert LibreDINOv2.INPUT_SIZES["n"] == 518
        assert LibreDINOv2.semantic_resize_mode == "stretch"

    def test_family_is_dinov2(self):
        from libreyolo.models.dinov2.model import LibreDINOv2

        assert LibreDINOv2.FAMILY == "dinov2"
        assert LibreDINOv2.FILENAME_PREFIX == "LibreDINOv2"

    def test_can_load_recognizes_semantic_signature(self):
        from libreyolo.models.dinov2.model import LibreDINOv2

        state = {
            "backbone.encoder.proj.weight": torch.zeros(1),
            "predict.weight": torch.zeros(3, 8, 1, 1),
        }
        assert LibreDINOv2.can_load(state)

    def test_can_load_rejects_detection_signature(self):
        from libreyolo.models.dinov2.model import LibreDINOv2

        state = {
            "backbone.encoder.proj.weight": torch.zeros(1),
            "class_embed.bias": torch.zeros(81),
        }
        assert not LibreDINOv2.can_load(state)

    def test_rfdetr_no_longer_claims_semantic(self):
        """LibreRFDETR.can_load must return False for semantic-only key sets."""
        from libreyolo.models.rfdetr.model import LibreRFDETR

        state = {
            "backbone.encoder.proj.weight": torch.zeros(1),
            "predict.weight": torch.zeros(3, 8, 1, 1),
        }
        assert not LibreRFDETR.can_load(state)

    def test_rfdetr_supported_tasks_excludes_semantic(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        assert "semantic" not in LibreRFDETR.SUPPORTED_TASKS


class TestDINOv2SemanticSegmenter:
    def test_forward_loss_and_eval_shapes(self, fake_backbone):
        from libreyolo.models.rfdetr.nn import RFDETRSemanticSegmenter

        model = RFDETRSemanticSegmenter(config="n", nb_classes=3)
        x = torch.rand(2, 3, 70, 70)

        model.train()
        targets = torch.randint(0, 3, (2, 70, 70))
        targets[:, :8, :] = 255
        out = model(x, targets=targets)
        assert set(out) == {"total_loss", "sem"}
        assert torch.isfinite(out["total_loss"])
        out["total_loss"].backward()
        assert model.predict.weight.grad is not None

        model.eval()
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 3, 70, 70)

    def test_wrapper_predict_returns_semantic_mask(self, fake_backbone, tmp_path):
        from libreyolo.models.dinov2.model import LibreDINOv2

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (90, 45), color=(50, 90, 130)).save(img_path)

        m = LibreDINOv2(
            model_path=None, size="n", task="semantic", nb_classes=3, device="cpu"
        )
        assert m.task == "semantic"
        assert m.input_size == 518

        result = m.predict(str(img_path), imgsz=70)

        assert result.boxes is None
        assert result.semantic_mask is not None
        assert tuple(result.semantic_mask.data.shape) == (45, 90)

    def test_wrapper_class_rebuild(self, fake_backbone):
        from libreyolo.models.dinov2.model import LibreDINOv2

        m = LibreDINOv2(
            model_path=None, size="n", task="semantic", nb_classes=3, device="cpu"
        )
        m._rebuild_for_new_classes(5)

        m.model.eval()
        with torch.no_grad():
            logits = m.model(torch.rand(1, 3, 70, 70))
        assert logits.shape == (1, 5, 70, 70)

    def test_wrong_task_raises(self):
        from libreyolo.models.dinov2.model import LibreDINOv2

        with pytest.raises(ValueError, match="semantic"):
            LibreDINOv2(
                model_path=None, size="n", task="detect", nb_classes=3, device="cpu"
            )

    @pytest.mark.parametrize("format", ["onnx", "torchscript"])
    def test_exported_semantic_parity(self, fake_backbone, tmp_path, format):
        if format == "onnx":
            pytest.importorskip("onnx")
            pytest.importorskip("onnxruntime")

        from libreyolo import LibreYOLO
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(
            model_path=None, size="n", task="semantic", nb_classes=3, device="cpu"
        )
        model.model.eval()
        image = np.random.default_rng(13).integers(
            0, 256, size=(70, 70, 3), dtype=np.uint8
        )
        native = model.predict(image, imgsz=70).semantic_mask.data
        suffix = ".onnx" if format == "onnx" else ".torchscript"
        artifact = tmp_path / f"dinov2_semantic{suffix}"
        model.export(
            format=format,
            output_path=str(artifact),
            imgsz=70,
            dynamic=False,
            simplify=False,
        )
        exported = LibreYOLO(str(artifact), device="cpu").predict(image)
        agreement = (native == exported.semantic_mask.data).float().mean().item()
        assert agreement > 0.95


def _make_semantic_yaml(root, n_images=4, size=70):
    import yaml as _yaml

    for split in ("train", "val"):
        for i in range(n_images):
            img_dir = root / "images" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            arr[:, : size // 2] = (200, 40, 40)
            arr[:, size // 2 :] = (40, 40, 200)
            Image.fromarray(arr).save(img_dir / f"img{i}.jpg")
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[:, size // 2 :] = 1
            mask_dir = root / "masks" / split
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, mode="L").save(mask_dir / f"img{i}.png")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        _yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "masks_dir": "masks",
                "nc": 2,
                "names": {0: "left", 1: "right"},
            }
        )
    )
    return yaml_path


def test_dinov2_semantic_train_smoke(fake_backbone, tmp_path):
    """One epoch through the shared trainer with the stub backbone."""
    from libreyolo.models.dinov2.model import LibreDINOv2

    yaml_path = _make_semantic_yaml(tmp_path)
    m = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=2, device="cpu"
    )

    res = m.train(
        data=str(yaml_path),
        epochs=1,
        batch=2,
        imgsz=70,
        workers=0,
        eval_interval=1,
        project=str(tmp_path / "runs"),
        name="sem_smoke",
        exist_ok=True,
        amp=False,
        ema=False,
        warmup_epochs=0,
    )

    assert np.isfinite(res["epoch_losses"][0])
    assert res["epoch_metrics"][-1]["val_metrics"].get("metrics/mIoU") is not None


def test_dinov2_checkpoint_family_is_dinov2(fake_backbone, tmp_path):
    """Trainer must save model_family='dinov2' (not 'rfdetr')."""
    from libreyolo.models.dinov2.model import LibreDINOv2
    from libreyolo.utils.serialization import load_trusted_torch_file

    yaml_path = _make_semantic_yaml(tmp_path)
    m = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=2, device="cpu"
    )
    res = m.train(
        data=str(yaml_path),
        epochs=1,
        batch=2,
        imgsz=70,
        workers=0,
        eval_interval=0,
        project=str(tmp_path / "runs"),
        name="ckpt_family",
        exist_ok=True,
        amp=False,
        ema=False,
        warmup_epochs=0,
    )
    ckpt_path = res.get("best_checkpoint") or res.get("last_checkpoint")
    assert ckpt_path is not None
    ckpt = load_trusted_torch_file(ckpt_path, map_location="cpu", context="test")
    assert ckpt.get("model_family") == "dinov2"


def test_dinov2_semantic_rejects_head_only_native_checkpoint(fake_backbone):
    from libreyolo.models.dinov2.model import LibreDINOv2
    from libreyolo.utils.serialization import (
        CheckpointLoadError,
        wrap_libreyolo_checkpoint,
    )

    source = LibreDINOv2(
        model_path=None,
        size="n",
        task="semantic",
        nb_classes=3,
        device="cpu",
    )
    head_only = {
        key: value.detach().clone()
        for key, value in source.model.state_dict().items()
        if key.startswith("predict.")
    }
    checkpoint = wrap_libreyolo_checkpoint(
        head_only,
        model_family="dinov2",
        size="n",
        task="semantic",
        nc=3,
        names=["a", "b", "c"],
        imgsz=518,
    )

    with pytest.raises(CheckpointLoadError, match="required model tensors"):
        LibreDINOv2(
            model_path=checkpoint,
            size="n",
            task="semantic",
            nb_classes=3,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("task", "checkpoint_imgsz"),
    [("semantic", 532), ("classify", 238)],
)
def test_dinov2_custom_loader_adopts_native_checkpoint_imgsz(
    fake_backbone,
    task,
    checkpoint_imgsz,
):
    from libreyolo.models.dinov2.model import LibreDINOv2
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    source = LibreDINOv2(
        model_path=None,
        size="n",
        task=task,
        nb_classes=3,
        device="cpu",
    )
    checkpoint = wrap_libreyolo_checkpoint(
        source.model.state_dict(),
        model_family="dinov2",
        size="n",
        task=task,
        nc=3,
        names=["a", "b", "c"],
        imgsz=checkpoint_imgsz,
    )

    loaded = LibreDINOv2(
        model_path=checkpoint,
        size="n",
        task=task,
        nb_classes=3,
        device="cpu",
    )

    assert loaded.input_size == checkpoint_imgsz


@pytest.mark.external_data
@pytest.mark.network
@pytest.mark.slow
def test_dinov2_semantic_forward_real_backbone():
    """LibreDINOv2 build + forward (DINOv2 backbone; random-init if offline)."""
    from libreyolo.models.dinov2.model import LibreDINOv2

    m = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=4, device="cpu"
    )
    assert m.task == "semantic"
    assert m.input_size == 518

    x = torch.rand(1, 3, 518, 518)
    m.model.train()
    out = m.model(x, targets=torch.randint(0, 4, (1, 518, 518)))
    assert "total_loss" in out

    m.model.eval()
    with torch.no_grad():
        assert m.model(x).shape == (1, 4, 518, 518)


@pytest.mark.external_data
@pytest.mark.network
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("LIBREYOLO_RUN_REAL_EXPORT_PARITY") != "1",
    reason="set LIBREYOLO_RUN_REAL_EXPORT_PARITY=1 for real DINOv2 export parity",
)
@pytest.mark.parametrize(("task", "imgsz"), [("semantic", 518), ("classify", 224)])
@pytest.mark.parametrize("format", ["onnx", "torchscript"])
def test_dinov2_real_export_raw_parity(tmp_path, task, imgsz, format):
    if task == "classify" and format != "onnx":
        pytest.skip("LibreDINOv2 classify export is intentionally ONNX-only")
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    from libreyolo import LibreDINOv2, LibreYOLO
    from libreyolo.export.exporter import OnnxExporter

    torch.manual_seed(0)
    model = LibreDINOv2(
        model_path=None, size="n", task=task, nb_classes=3, device="cpu"
    )
    model.model.eval()
    tensor = torch.rand(1, 3, imgsz, imgsz)
    exporter = OnnxExporter(model)
    with exporter._model_context("cpu", False, False, 1, (imgsz, imgsz)) as (
        wrapped,
        _,
    ):
        with torch.no_grad():
            expected = wrapped(tensor)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)

    artifact = model.export(
        format=format,
        imgsz=imgsz,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / f"dinov2-{task}.{format}"),
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


def test_all_ignore_targets_yield_finite_zero_loss(fake_backbone):
    from libreyolo.models.rfdetr.nn import RFDETRSemanticSegmenter

    model = RFDETRSemanticSegmenter(config="n", nb_classes=3)
    model.train()
    out = model(
        torch.rand(1, 3, 70, 70),
        targets=torch.full((1, 70, 70), 255, dtype=torch.long),
    )

    assert torch.isfinite(out["total_loss"])
    assert float(out["total_loss"]) == 0.0


def test_dinov2_semantic_predict_rejects_non_patch_imgsz(fake_backbone, tmp_path):
    from libreyolo.models.dinov2.model import LibreDINOv2

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(img_path)
    m = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=2, device="cpu"
    )

    with pytest.raises(ValueError, match="divisible by 14"):
        m.predict(str(img_path), imgsz=100)


def test_dinov2_semantic_train_rejects_non_patch_imgsz(
    fake_backbone, tmp_path, monkeypatch
):
    from libreyolo.models.dinov2.model import LibreDINOv2

    monkeypatch.chdir(tmp_path)
    yaml_path = _make_semantic_yaml(tmp_path)
    m = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=2, device="cpu"
    )

    with pytest.raises(ValueError, match="divisible by 14"):
        m.train(
            data=str(yaml_path),
            epochs=1,
            batch=2,
            imgsz=64,
            workers=0,
            eval_interval=0,
            project=str(tmp_path / "runs"),
            name="bad_imgsz",
            exist_ok=True,
            amp=False,
            ema=False,
            warmup_epochs=0,
        )


def test_dinov2_semantic_rejects_lora(fake_backbone, tmp_path, monkeypatch):
    from libreyolo.models.dinov2.model import LibreDINOv2

    monkeypatch.chdir(tmp_path)
    yaml_path = _make_semantic_yaml(tmp_path)
    m = LibreDINOv2(
        model_path=None, size="n", task="semantic", nb_classes=2, device="cpu"
    )

    with pytest.raises(ValueError, match="lora"):
        m.train(
            data=str(yaml_path),
            epochs=1,
            batch=2,
            imgsz=70,
            workers=0,
            eval_interval=0,
            project=str(tmp_path / "runs"),
            name="lora_reject",
            exist_ok=True,
            amp=False,
            ema=False,
            warmup_epochs=0,
            lora=True,
        )
