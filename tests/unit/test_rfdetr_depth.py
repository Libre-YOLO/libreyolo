"""Unit tests for RF-DETR depth estimation.

Structural tests run against a lightweight fake backbone (monkeypatched
``build_backbone``) so they stay hermetic; one real-backbone forward test is
network-marked for nightly runs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml as _yaml
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


def _depth_targets(batch: int, size: int) -> torch.Tensor:
    """Left half near (2.0), right half far (8.0)."""
    targets = torch.full((batch, size, size), 8.0)
    targets[..., : size // 2] = 2.0
    return targets


class TestDepthMetadata:
    def test_task_registration(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        assert "depth" in LibreRFDETR.SUPPORTED_TASKS
        assert LibreRFDETR.TASK_INPUT_SIZES["depth"]["n"] == 518
        assert LibreRFDETR.depth_resize_mode == "stretch"
        assert LibreRFDETR.depth_imgsz_divisor == 14

    def test_can_load_recognizes_depth_signature(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        state = {
            "backbone.encoder.proj.weight": torch.zeros(1),
            "depth_head.weight": torch.zeros(1, 8, 1, 1),
        }
        assert LibreRFDETR.can_load(state)

    def test_detect_task_from_source_prefers_metadata_then_signature(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        assert (
            LibreRFDETR._detect_task_from_source({"task": "depth", "model": {}})
            == "depth"
        )
        signature_ckpt = {
            "model": {
                "backbone.encoder.proj.weight": torch.zeros(1),
                "depth_head.weight": torch.zeros(1, 8, 1, 1),
            }
        }
        assert LibreRFDETR._detect_task_from_source(signature_ckpt) == "depth"


class TestDepthEstimator:
    def test_forward_loss_and_eval_shapes(self, fake_backbone):
        from libreyolo.models.rfdetr.nn import RFDETRDepthEstimator

        model = RFDETRDepthEstimator(config="n")
        x = torch.rand(2, 3, 70, 70)

        model.train()
        targets = _depth_targets(2, 70)
        targets[:, :8, :] = 0.0  # an invalid stripe
        out = model(x, targets=targets)
        assert set(out) == {"total_loss", "depth"}
        assert torch.isfinite(out["total_loss"])
        out["total_loss"].backward()
        assert model.depth_head.weight.grad is not None

        model.eval()
        with torch.no_grad():
            pred = model(x)
        assert pred.shape == (2, 1, 70, 70)
        assert float(pred.min()) >= 0.0  # non-negative inverse depth

    def test_loss_decreases_under_optimization(self, fake_backbone):
        from libreyolo.models.rfdetr.nn import RFDETRDepthEstimator

        torch.manual_seed(0)
        model = RFDETRDepthEstimator(config="n")
        model.train()
        x = torch.rand(2, 3, 70, 70)
        targets = _depth_targets(2, 70)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        first = None
        last = None
        for _ in range(8):
            optimizer.zero_grad()
            out = model(x, targets=targets)
            out["total_loss"].backward()
            optimizer.step()
            last = float(out["total_loss"].detach())
            if first is None:
                first = last
        assert last < first

    def test_loss_is_scale_and_shift_invariant(self, fake_backbone):
        from libreyolo.models.rfdetr.nn import RFDETRDepthEstimator

        model = RFDETRDepthEstimator(config="n")
        pred = torch.rand(40, 40) + 0.1
        gt = torch.rand(40, 40) * 5 + 1.0

        base = model._ssi_loss_single(pred, gt)
        scaled = model._ssi_loss_single(4.0 * pred + 2.0, gt)
        assert float(base) == pytest.approx(float(scaled), abs=1e-5)

    def test_all_invalid_targets_yield_finite_zero_loss(self, fake_backbone):
        from libreyolo.models.rfdetr.nn import RFDETRDepthEstimator

        model = RFDETRDepthEstimator(config="n")
        model.train()
        out = model(
            torch.rand(1, 3, 70, 70),
            targets=torch.zeros((1, 70, 70)),
        )

        assert torch.isfinite(out["total_loss"])
        assert float(out["total_loss"].detach()) == 0.0

    def test_one_task_head_at_a_time(self, fake_backbone):
        from libreyolo.models.rfdetr.nn import LibreRFDETRModel

        with pytest.raises(ValueError, match="one task head"):
            LibreRFDETRModel(config="n", depth=True, classification=True)

    def test_wrapper_predict_returns_depth_map(self, fake_backbone, tmp_path):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (90, 45), color=(50, 90, 130)).save(img_path)

        m = LibreRFDETR(model_path=None, size="n", task="depth", device="cpu")
        assert m.task == "depth"
        assert m.input_size == 518

        result = m.predict(str(img_path), imgsz=70)

        assert result.boxes is None
        assert result.depth_map is not None
        assert tuple(result.depth_map.data.shape) == (45, 90)

    def test_wrapper_predict_save_writes_colormap(self, fake_backbone, tmp_path):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (56, 56), color=(10, 20, 30)).save(img_path)

        m = LibreRFDETR(model_path=None, size="n", task="depth", device="cpu")
        m.predict(
            str(img_path),
            imgsz=70,
            save=True,
            output_path=str(tmp_path / "out.jpg"),
        )

        assert (tmp_path / "out.jpg").exists()

    def test_checkpoint_round_trip(self, fake_backbone, tmp_path):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        m = LibreRFDETR(model_path=None, size="n", task="depth", device="cpu")
        state = {
            "model": m.model.state_dict(),
            "task": "depth",
            "size": "n",
            "nc": 1,
            "names": {0: "depth"},
        }
        ckpt_path = tmp_path / "LibreRFDETRn-depth.pt"
        torch.save(state, str(ckpt_path))

        loaded = LibreRFDETR(model_path=str(ckpt_path), device="cpu")
        assert loaded.task == "depth"
        assert loaded.nb_classes == 1
        assert loaded.names == {0: "depth"}

    def test_detection_checkpoint_rejected_as_depth(self, fake_backbone, tmp_path):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        ckpt_path = tmp_path / "det.pt"
        torch.save({"model": {}, "task": "detect"}, str(ckpt_path))

        with pytest.raises(ValueError, match="task="):
            LibreRFDETR(model_path=str(ckpt_path), size="n", task="depth", device="cpu")


def _make_depth_yaml(root, n_images=4, size=70):
    for split in ("train", "val"):
        for i in range(n_images):
            img_dir = root / "images" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            arr[:, : size // 2] = (200, 40, 40)
            arr[:, size // 2 :] = (40, 40, 200)
            Image.fromarray(arr).save(img_dir / f"img{i}.jpg")
            depth = np.full((size, size), 8.0)
            depth[:, : size // 2] = 2.0
            depth_dir = root / "depths" / split
            depth_dir.mkdir(parents=True, exist_ok=True)
            encoded = (depth * 256.0).round().astype(np.uint16)
            Image.fromarray(encoded).save(depth_dir / f"img{i}.png")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        _yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
            }
        )
    )
    return yaml_path


def test_rfdetr_depth_train_smoke(fake_backbone, tmp_path):
    """One epoch through the shared trainer with the stub backbone."""
    from libreyolo.models.rfdetr.model import LibreRFDETR

    yaml_path = _make_depth_yaml(tmp_path)
    m = LibreRFDETR(model_path=None, size="n", task="depth", device="cpu")

    res = m.train(
        data=str(yaml_path),
        epochs=1,
        batch=2,
        imgsz=70,
        workers=0,
        eval_interval=1,
        project=str(tmp_path / "runs"),
        name="depth_smoke",
        exist_ok=True,
        amp=False,
        ema=False,
        warmup_epochs=0,
    )

    assert np.isfinite(res["epoch_losses"][0])
    assert res["epoch_metrics"][-1]["val_metrics"].get("metrics/delta1") is not None


def test_rfdetr_depth_predict_rejects_non_patch_imgsz(fake_backbone, tmp_path):
    from libreyolo.models.rfdetr.model import LibreRFDETR

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(img_path)
    m = LibreRFDETR(model_path=None, size="n", task="depth", device="cpu")

    with pytest.raises(ValueError, match="divisible by 14"):
        m.predict(str(img_path), imgsz=100)


def test_rfdetr_depth_train_rejects_non_patch_imgsz(fake_backbone, tmp_path):
    from libreyolo.models.rfdetr.model import LibreRFDETR

    yaml_path = _make_depth_yaml(tmp_path)
    m = LibreRFDETR(model_path=None, size="n", task="depth", device="cpu")

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


def test_rfdetr_depth_rejects_lora(fake_backbone, tmp_path):
    from libreyolo.models.rfdetr.model import LibreRFDETR

    yaml_path = _make_depth_yaml(tmp_path)
    m = LibreRFDETR(model_path=None, size="n", task="depth", device="cpu")

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


def test_rfdetr_depth_tta_rejected(fake_backbone, tmp_path):
    from libreyolo.models.rfdetr.model import LibreRFDETR

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (56, 56), color=(10, 20, 30)).save(img_path)
    m = LibreRFDETR(model_path=None, size="n", task="depth", device="cpu")

    with pytest.raises(ValueError, match="augment"):
        m.predict(str(img_path), imgsz=70, augment=True)


def test_rfdetr_depth_tiling_rejected(fake_backbone, tmp_path):
    from libreyolo.models.rfdetr.model import LibreRFDETR

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (128, 128), color=(10, 20, 30)).save(img_path)
    m = LibreRFDETR(model_path=None, size="n", task="depth", device="cpu")

    with pytest.raises(ValueError, match="Tiled inference"):
        m.predict(str(img_path), imgsz=70, tiling=True)


def test_rfdetr_depth_tracking_rejected(fake_backbone, tmp_path):
    from libreyolo.models.rfdetr.model import LibreRFDETR

    m = LibreRFDETR(model_path=None, size="n", task="depth", device="cpu")

    with pytest.raises(NotImplementedError, match="Tracking"):
        list(m.track(tmp_path / "missing.mp4"))


@pytest.mark.external_data
@pytest.mark.network
@pytest.mark.slow
def test_rfdetr_depth_forward_real_backbone():
    """RF-DETR depth build + forward (DINOv2 backbone; random-init if offline)."""
    from libreyolo import LibreRFDETR

    m = LibreRFDETR(model_path=None, size="n", task="depth", device="cpu")
    assert m.task == "depth"
    assert m.input_size == 518
    assert m.model.depth

    x = torch.rand(1, 3, 518, 518)
    m.model.train()
    out = m.model(x, targets=_depth_targets(1, 518))
    assert "total_loss" in out

    m.model.eval()
    with torch.no_grad():
        assert m.model(x).shape == (1, 1, 518, 518)
