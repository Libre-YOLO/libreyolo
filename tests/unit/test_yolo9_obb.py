"""Unit tests for the YOLO9 oriented-bounding-box (OBB) task.

Covers the design contracts of PLAN_obb_yolo9.md: head shapes and decode
bounds, the double-angle encode/decode roundtrip, the Gaussian geometry
helpers, the rotated matcher, loss composition and gradients, and the
factory/transfer rules.
"""

import math
import pickle

import numpy as np
import pytest
import torch

from libreyolo import LibreYOLO
from libreyolo.models.yolo9.loss import (
    RotatedBoxMatcher,
    YOLO9OrientedLoss,
    _bhattacharyya_score,
    _gaussian_kld,
    _rbox_covariance,
)
from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.models.yolo9.nn import LibreYOLO9Model, OrientedDDetect
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint
from libreyolo.validation.preprocessors import YOLO9ValPreprocessor

pytestmark = pytest.mark.unit


class TestOrientedHead:
    def _head(self, nc=2):
        return OrientedDDetect(
            nc=nc, ch=(64, 128, 256), reg_max=16, stride=(8, 16, 32)
        )

    def _feats(self, hw=8):
        return [
            torch.randn(1, 64, hw, hw),
            torch.randn(1, 128, hw // 2, hw // 2),
            torch.randn(1, 256, hw // 4, hw // 4),
        ]

    def test_eval_forward_shapes(self):
        head = self._head()
        head.eval()
        y, preds = head(self._feats())
        assert y.shape == (1, 4 + 1 + 2, 84)
        assert len(preds) == 3

    def test_theta_row_is_bounded(self):
        head = self._head()
        head.eval()
        feats = [f * 100 for f in self._feats()]
        y, _ = head(feats)
        theta = y[:, 4]
        assert torch.all(theta >= -math.pi / 2 - 1e-6)
        assert torch.all(theta <= math.pi / 2 + 1e-6)

    def test_export_mode_returns_single_tensor(self):
        model = LibreYOLO9Model(config="t", nb_classes=2, obb=True)
        model.eval()
        model.head.export = True
        y = model(torch.randn(1, 3, 64, 64))
        assert isinstance(y, torch.Tensor)
        assert y.shape == (1, 7, 84)

    def test_detect_towers_are_reused_unchanged(self):
        """The oriented head must produce the same box/class maps as the
        detect head with identical tower weights (the hidden-feature tap
        cannot change the values)."""
        torch.manual_seed(3)
        from libreyolo.models.yolo9.nn import DDetect

        det = DDetect(nc=2, ch=(64, 128, 256), reg_max=16, stride=(8, 16, 32))
        obb = self._head()
        obb.load_state_dict(det.state_dict(), strict=False)
        det.train()
        obb.train()
        feats = self._feats()
        det_preds = det(feats)
        obb_preds, _angles = obb(feats)
        for a, b in zip(det_preds, obb_preds):
            torch.testing.assert_close(a, b)

    def test_orientation_bias_starts_at_zero(self):
        head = self._head()
        for conv in head.ang:
            assert torch.all(conv.bias == 0)


def test_double_angle_roundtrip():
    for theta in np.linspace(-math.pi / 2, math.pi / 2, 37):
        vec = (math.cos(2 * theta), math.sin(2 * theta))
        recovered = 0.5 * math.atan2(vec[1], vec[0])
        canon = (theta + math.pi / 2) % math.pi - math.pi / 2
        recovered = (recovered + math.pi / 2) % math.pi - math.pi / 2
        assert abs(recovered - canon) < 1e-9


class TestGaussianHelpers:
    def _cov(self, w, h, theta):
        wh = torch.tensor([[float(w), float(h)]])
        tt = torch.tensor([2.0 * theta])
        return _rbox_covariance(wh, tt.cos(), tt.sin())

    def test_kld_zero_at_identity(self):
        mu = torch.tensor([[10.0, 20.0]])
        cov = self._cov(60, 20, 0.5)
        assert float(_gaussian_kld(mu, cov, mu, cov)) == pytest.approx(0.0, abs=1e-5)

    def test_bhattacharyya_one_at_identity_and_monotone(self):
        mu = torch.tensor([[10.0, 20.0]])
        cov = self._cov(60, 20, 0.5)
        assert float(_bhattacharyya_score(mu, cov, mu, cov)) == pytest.approx(
            1.0, abs=1e-5
        )
        previous = 1.0
        for offset in (5.0, 15.0, 40.0):
            score = float(
                _bhattacharyya_score(mu + torch.tensor([[offset, 0.0]]), cov, mu, cov)
            )
            assert score < previous
            previous = score

    def test_square_targets_are_rotation_invariant(self):
        """A square's Gaussian is isotropic: rotating the target must not
        change the loss against any fixed prediction."""
        mu = torch.tensor([[0.0, 0.0]])
        pred_cov = self._cov(40, 25, 0.3)
        base = float(_gaussian_kld(mu, pred_cov, mu, self._cov(30, 30, 0.0)))
        for theta in (0.3, -0.9, 1.2):
            rotated = float(
                _gaussian_kld(mu, pred_cov, mu, self._cov(30, 30, theta))
            )
            assert rotated == pytest.approx(base, abs=1e-5)

    def test_kld_sees_orientation_of_elongated_boxes(self):
        mu = torch.tensor([[0.0, 0.0]])
        target = self._cov(80, 20, 0.0)
        aligned = float(_gaussian_kld(mu, self._cov(80, 20, 0.0), mu, target))
        crossed = float(_gaussian_kld(mu, self._cov(80, 20, math.pi / 2 - 0.01), mu, target))
        assert crossed > aligned + 1.0


class TestRotatedMatcher:
    def _matcher(self, nc=2):
        from libreyolo.models.yolo9.loss import generate_anchors

        anchor_grid, scaler = generate_anchors([64, 64], [8, 16, 32])
        return RotatedBoxMatcher(
            num_classes=nc,
            anchor_grid=anchor_grid,
            scaler=scaler,
            reg_max=16,
        )

    def test_matched_targets_carry_theta(self):
        matcher = self._matcher()
        n_anchors = matcher.anchor_grid.shape[0]
        target = torch.tensor([[[0.0, 8.0, 8.0, 46.0, 26.0, 0.7]]])
        pred_cls = torch.zeros(1, n_anchors, 2)
        pred_rbox = torch.zeros(1, n_anchors, 6)
        pred_rbox[..., :2] = matcher.anchor_grid[None]
        pred_rbox[..., 2:4] = 20.0
        pred_rbox[..., 4] = 1.0
        matched, valid = matcher(target, (pred_cls, pred_rbox))
        assert matched.shape == (1, n_anchors, 2 + 4 + 1)
        assert valid.any()
        matched_theta = matched[0, valid[0], -1]
        torch.testing.assert_close(
            matched_theta, torch.full_like(matched_theta, 0.7)
        )

    def test_empty_targets(self):
        matcher = self._matcher()
        n_anchors = matcher.anchor_grid.shape[0]
        target = torch.zeros(1, 0, 6)
        matched, valid = matcher(
            (target), (torch.zeros(1, n_anchors, 2), torch.zeros(1, n_anchors, 6))
        )
        assert matched.shape == (1, n_anchors, 7)
        assert not valid.any()


class TestOrientedLoss:
    def _loss(self, nc=2):
        return YOLO9OrientedLoss(
            num_classes=nc,
            reg_max=16,
            strides=[8, 16, 32],
            image_size=[64, 64],
            device=torch.device("cpu"),
        )

    def test_rejects_five_column_targets(self):
        model = LibreYOLO9Model(config="t", nb_classes=2, obb=True)
        model.train()
        targets = torch.zeros(1, 10, 5)
        targets[:, :, 0] = -1
        with pytest.raises(ValueError, match="6"):
            model(torch.randn(1, 3, 64, 64), targets=targets)

    def test_components_and_gradients(self):
        model = LibreYOLO9Model(config="t", nb_classes=2, obb=True)
        model.train()
        targets = torch.zeros(2, 100, 6)
        targets[:, :, 0] = -1
        targets[0, 0] = torch.tensor([0, 0.20, 0.30, 0.70, 0.48, 0.9])
        targets[1, 0] = torch.tensor([1, 0.10, 0.10, 0.60, 0.28, -0.6])

        out = model(torch.randn(2, 3, 64, 64), targets=targets)

        assert out["total_loss"].requires_grad
        assert out["angle_loss"].requires_grad
        assert out["box"] >= 0 and out["angle"] >= 0
        out["total_loss"].backward()
        assert model.head.ang[0].weight.grad is not None
        assert model.head.ang[0].weight.grad.abs().sum() > 0
        assert model.head.cv2[0][0].conv.weight.grad is not None

    def test_square_targets_get_no_directional_supervision(self):
        """Aspect weighting: a square target's vector target is zero, so the
        angle loss must be invariant to the direction of the predicted
        vector (it only pulls its norm down). Elongated targets must prefer
        the correct direction."""
        loss_fn = self._loss(nc=1)

        def angle_loss(box_row, direction):
            torch.manual_seed(0)
            preds = [
                torch.randn(1, 1 + 64, 64 // s, 64 // s) for s in (8, 16, 32)
            ]
            angles = [
                torch.full((1, 2, 64 // s, 64 // s), 0.0) for s in (8, 16, 32)
            ]
            for level in angles:
                level[0, 0] = direction[0]
                level[0, 1] = direction[1]
            targets = torch.zeros(1, 1, 6)
            targets[0, 0] = box_row
            return float(loss_fn(preds, targets, angles)["angle_loss"])

        square = torch.tensor([0, 0.25, 0.25, 0.75, 0.75, 0.8])
        elongated = torch.tensor([0, 0.10, 0.35, 0.90, 0.55, 0.0])

        sq_a = angle_loss(square, (1.0, 0.0))
        sq_b = angle_loss(square, (0.0, 1.0))
        assert sq_a == pytest.approx(sq_b, rel=1e-5)

        el_right = angle_loss(elongated, (1.0, 0.0))
        el_wrong = angle_loss(elongated, (-1.0, 0.0))
        assert el_right < el_wrong


class TestOBBFactoryAndTransfer:
    def test_scratch_checkpoint_roundtrip(self, tmp_path):
        model = LibreYOLO9Model(config="t", nb_classes=1, obb=True)
        ckpt = tmp_path / "LibreYOLO9t-obb.pt"
        torch.save(
            wrap_libreyolo_checkpoint(
                model.state_dict(),
                model_family="yolo9",
                size="t",
                task="obb",
                nc=1,
                names={0: "ship"},
                imgsz=64,
            ),
            ckpt,
        )
        loaded = LibreYOLO(str(ckpt), device="cpu")
        assert loaded.FAMILY == "yolo9"
        assert loaded.task == "obb"
        assert loaded.names == {0: "ship"}
        assert isinstance(loaded.model.head, OrientedDDetect)

    def test_metadata_less_task_inference_from_ang_keys(self, tmp_path):
        model = LibreYOLO9Model(config="t", nb_classes=1, obb=True)
        ckpt = tmp_path / "best.pt"
        torch.save(model.state_dict(), ckpt)
        loaded = LibreYOLO(str(ckpt), size="t", device="cpu")
        assert loaded.task == "obb"

    def test_transfer_accepts_detect_checkpoint(self, tmp_path):
        detect = LibreYOLO9Model(config="t", nb_classes=80)
        ckpt = tmp_path / "LibreYOLO9t.pt"
        torch.save(
            wrap_libreyolo_checkpoint(
                detect.state_dict(),
                model_family="yolo9",
                size="t",
                task="detect",
                nc=80,
                imgsz=640,
            ),
            ckpt,
        )
        target = LibreYOLO9(None, size="t", task="obb", nb_classes=6, device="cpu")
        stats = target._load_transfer_weights(ckpt)
        assert stats["loaded"] > 0
        # Only the fresh orientation convs are skipped (weight+bias per scale).
        assert stats["skipped"] == 6
        torch.testing.assert_close(
            target.model.state_dict()["backbone.conv0.conv.weight"],
            detect.state_dict()["backbone.conv0.conv.weight"],
        )

    def test_direct_load_rejects_detect_checkpoint(self, tmp_path):
        detect = LibreYOLO9Model(config="t", nb_classes=80)
        ckpt = tmp_path / "LibreYOLO9t.pt"
        torch.save(
            wrap_libreyolo_checkpoint(
                detect.state_dict(),
                model_family="yolo9",
                size="t",
                task="detect",
                nc=80,
                imgsz=640,
            ),
            ckpt,
        )
        with pytest.raises(RuntimeError, match="task='detect'"):
            LibreYOLO9(str(ckpt), size="t", task="obb", device="cpu")

    def test_predict_returns_obb_container(self):
        model = LibreYOLO9(None, size="t", task="obb", nb_classes=2, device="cpu")
        model.model.eval()
        img = (np.random.rand(96, 128, 3) * 255).astype("uint8")
        result = model(img, conf=0.0001, save=False)
        result = result[0] if isinstance(result, list) else result
        assert result.obb is not None or result.boxes is not None


def test_obb_val_preprocessor_survives_pickle_roundtrip():
    """Windows spawn DataLoader workers pickle the preprocessor; __getattr__
    must not recurse while __dict__ is still empty during unpickling."""
    from libreyolo.validation.obb_validator import _OBBValPreprocessor

    preproc = _OBBValPreprocessor(YOLO9ValPreprocessor((32, 32)))
    restored = pickle.loads(pickle.dumps(preproc))

    assert isinstance(restored.base_preprocessor, YOLO9ValPreprocessor)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    targets = np.zeros((1, 6), dtype=np.float32)
    assert restored(img, targets, (32, 32)) is not None
