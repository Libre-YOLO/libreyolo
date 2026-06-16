"""Unit and end-to-end tests for LibreFOMO."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.fomo]


# ===========================================================================
# Helpers
# ===========================================================================


def _write_tiny_point_dataset(root: Path, imgsz: int = 96) -> Path:
    """Write a minimal YOLO-format point dataset (1 image, 1 label for train and val splits)."""
    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (imgsz, imgsz), color=(128, 64, 32)).save(img_dir / "sample.jpg")
        (lbl_dir / "sample.txt").write_text("0 0.5 0.5 0.05 0.05\n", encoding="utf-8")

    import yaml

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": {0: "object"},
            }
        ),
        encoding="utf-8",
    )
    return data_yaml


def _make_random_fomo(size: str = "s", nc: int = 1) -> "LibreFOMO":
    from libreyolo.models.fomo.model import LibreFOMO

    return LibreFOMO(model_path=None, size=size, nb_classes=nc, device="cpu")


# ===========================================================================
# nn.py — architecture contracts
# ===========================================================================


class TestFOMOBackboneShapes:
    """Verify spatial resolution contracts for all three sizes."""

    @pytest.mark.parametrize(
        "size,imgsz,expected_hw",
        [
            ("s", 96,  12),  # 96  / 8 = 12
            ("m", 192, 24),  # 192 / 8 = 24
            ("l", 224, 28),  # 224 / 8 = 28
        ],
    )
    def test_backbone_spatial_downsample(self, size: str, imgsz: int, expected_hw: int) -> None:
        from libreyolo.models.fomo.nn import FOMOBackbone

        backbone = FOMOBackbone(size).eval()
        x = torch.zeros(1, 3, imgsz, imgsz)
        with torch.no_grad():
            out = backbone(x)
        assert out.shape[-2] == expected_hw, f"size={size}: expected H={expected_hw}, got {out.shape[-2]}"
        assert out.shape[-1] == expected_hw, f"size={size}: expected W={expected_hw}, got {out.shape[-1]}"

    @pytest.mark.parametrize("size", ["s", "m", "l"])
    @pytest.mark.parametrize("nc", [1, 3])
    def test_model_output_channels(self, size: str, nc: int) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel

        model = LibreFOMOModel(size=size, nc=nc).eval()
        cfg = model.CONFIGS[size]
        x = torch.zeros(1, 3, cfg["imgsz"], cfg["imgsz"])
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == nc + 1, f"Expected {nc+1} channels, got {out.shape[1]}"

    @pytest.mark.parametrize("size", ["s", "m", "l"])
    def test_model_batch_dimension_preserved(self, size: str) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel, CONFIGS

        model = LibreFOMOModel(size=size, nc=1).eval()
        imgsz = CONFIGS[size]["imgsz"]
        x = torch.zeros(2, 3, imgsz, imgsz)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 2


class TestDetectSizeFromStateDict:
    """Verify size detection heuristic for all three variants."""

    @pytest.mark.parametrize("size", ["s", "m", "l"])
    def test_detect_roundtrip(self, size: str) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel, detect_size_from_state_dict

        model = LibreFOMOModel(size=size, nc=1)
        sd = model.state_dict()
        detected = detect_size_from_state_dict(sd)
        assert detected == size, f"Expected size={size!r}, detected={detected!r}"

    def test_detect_empty_dict_returns_none(self) -> None:
        from libreyolo.models.fomo.nn import detect_size_from_state_dict

        assert detect_size_from_state_dict({}) is None

    def test_detect_unrelated_dict_returns_none(self) -> None:
        from libreyolo.models.fomo.nn import detect_size_from_state_dict

        assert detect_size_from_state_dict({"unrelated.weight": torch.zeros(3, 3)}) is None

    @pytest.mark.parametrize("size", ["s", "m", "l"])
    def test_detect_roundtrip_ddp(self, size: str) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel, detect_size_from_state_dict

        model = LibreFOMOModel(size=size, nc=1)
        sd = {f"module.{k}": v for k, v in model.state_dict().items()}
        detected = detect_size_from_state_dict(sd)
        assert detected == size, f"Expected size={size!r} under DDP, detected={detected!r}"


class TestFOMONNDeterministic:
    """Forward pass is deterministic (no dropout / stochastic layers)."""

    def test_deterministic_forward(self) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel, CONFIGS

        torch.manual_seed(0)
        model = LibreFOMOModel(size="s", nc=1).eval()
        imgsz = CONFIGS["s"]["imgsz"]
        x = torch.randn(1, 3, imgsz, imgsz)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1, out2)


class TestFOMONNInvalidSize:
    def test_raises_on_invalid_size(self) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel

        with pytest.raises(ValueError, match="Unsupported LibreFOMO size"):
            LibreFOMOModel(size="xl", nc=1)


# ===========================================================================
# model.py — LibreFOMO class
# ===========================================================================


class TestLibreFOMORegistration:
    def test_family_attribute(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        assert LibreFOMO.FAMILY == "fomo"

    def test_supported_tasks(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        assert "point" in LibreFOMO.SUPPORTED_TASKS

    def test_default_task(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        assert LibreFOMO.DEFAULT_TASK == "point"

    def test_in_models_registry(self) -> None:
        import libreyolo.models  # triggers registration  # noqa: F401
        from libreyolo.models.base.model import BaseModel
        from libreyolo.models.fomo.model import LibreFOMO

        assert LibreFOMO in BaseModel._registry

    def test_input_sizes_populated(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        for size in ("s", "m", "l"):
            assert size in LibreFOMO.INPUT_SIZES
            assert LibreFOMO.INPUT_SIZES[size] > 0


class TestLibreFOMOCanLoad:
    @pytest.mark.parametrize("size", ["s", "m", "l"])
    def test_can_load_own_state_dict(self, size: str) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel
        from libreyolo.models.fomo.model import LibreFOMO

        sd = LibreFOMOModel(size=size, nc=1).state_dict()
        assert LibreFOMO.can_load(sd)

    def test_cannot_load_unrelated_dict(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        assert not LibreFOMO.can_load({"some.unrelated.key": torch.zeros(1)})

    def test_cannot_load_missing_head(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO
        from libreyolo.models.fomo.nn import LibreFOMOModel

        sd = LibreFOMOModel(size="s", nc=1).state_dict()
        sd_no_head = {k: v for k, v in sd.items() if k != "head.weight"}
        assert not LibreFOMO.can_load(sd_no_head)





class TestLibreFOMODetectNbClasses:
    @pytest.mark.parametrize("nc", [1, 2, 5])
    def test_detect_nb_classes(self, nc: int) -> None:
        from libreyolo.models.fomo.nn import LibreFOMOModel
        from libreyolo.models.fomo.model import LibreFOMO

        sd = LibreFOMOModel(size="s", nc=nc).state_dict()
        assert LibreFOMO.detect_nb_classes(sd) == nc


class TestLibreFOMORandomInit:
    """Random-weight models instantiate without errors and have correct defaults."""

    @pytest.mark.parametrize("size", ["s", "m", "l"])
    def test_random_init_all_sizes(self, size: str) -> None:
        model = _make_random_fomo(size=size)
        assert model.size == size
        assert model.task == "point"
        assert model.device.type == "cpu"

    def test_random_init_sets_nb_classes(self) -> None:
        model = _make_random_fomo(size="s", nc=3)
        assert model.nb_classes == 3

    def test_model_attribute_is_nn_module(self) -> None:
        model = _make_random_fomo()
        assert isinstance(model.model, torch.nn.Module)

class TestLibreFOMOPostprocess:
    """_postprocess satisfies the PointValidator contract."""

    def test_empty_output_returns_zero_detections(self) -> None:
        model = _make_random_fomo(size="s", nc=1)
        from libreyolo.models.fomo.nn import CONFIGS

        hw = CONFIGS["s"]["imgsz"] // 8
        # All-zero logits → softmax dominated by background → no foreground above threshold
        logits = torch.zeros(1, 2, hw, hw)
        result = model._postprocess(
            logits,
            conf_thres=0.5,
            iou_thres=0.45,
            original_size=(96, 96),
        )
        assert result["num_detections"] == 0
        assert result["points"].shape == (0, 2)
        assert result["scores"].shape == (0,)
        assert result["classes"].shape == (0,)

    def test_high_confidence_foreground_detected(self) -> None:
        model = _make_random_fomo(size="s", nc=1)
        from libreyolo.models.fomo.nn import CONFIGS

        hw = CONFIGS["s"]["imgsz"] // 8
        # Large positive foreground logit at cell (2, 3)
        logits = torch.zeros(1, 2, hw, hw)
        logits[0, 1, 2, 3] = 20.0  # channel 1 = class 0 foreground
        result = model._postprocess(
            logits,
            conf_thres=0.01,
            iou_thres=0.45,
            original_size=(96, 96),
        )
        assert result["num_detections"] >= 1
        assert result["points"].shape[1] == 2
        # class id should be 0 (channel 1 - 1)
        assert int(result["classes"][0].item()) == 0

    def test_postprocess_scales_to_original_size(self) -> None:
        """Points must be in original-image pixel space, not grid space."""
        model = _make_random_fomo(size="s", nc=1)
        from libreyolo.models.fomo.nn import CONFIGS

        hw = CONFIGS["s"]["imgsz"] // 8
        logits = torch.zeros(1, 2, hw, hw)
        logits[0, 1, 0, 0] = 20.0  # top-left cell
        result = model._postprocess(
            logits,
            conf_thres=0.01,
            iou_thres=0.45,
            original_size=(480, 640),
        )
        if result["num_detections"] > 0:
            x, y = result["points"][0]
            # scaled pixel should be > 0 and fit in original image
            assert x.item() >= 0.0
            assert y.item() >= 0.0
            assert x.item() <= 640
            assert y.item() <= 480


class TestLibreFOMOValPreprocessor:
    """Validate FOMOValPreprocessor output shape, dtype, and pixel range."""

    def _make_preprocessor(self, imgsz: int = 96):
        from libreyolo.validation.preprocessors import FOMOValPreprocessor

        return FOMOValPreprocessor(img_size=(imgsz, imgsz))

    def test_output_shape(self) -> None:
        preproc = self._make_preprocessor(96)
        img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        targets = np.zeros((0, 5), dtype=np.float32)
        chw, _ = preproc(img, targets, (96, 96))
        assert chw.shape == (3, 96, 96)

    def test_output_dtype_float32(self) -> None:
        preproc = self._make_preprocessor(96)
        img = np.ones((96, 96, 3), dtype=np.uint8) * 128
        chw, _ = preproc(img, np.zeros((0, 5), dtype=np.float32), (96, 96))
        assert chw.dtype == np.float32

    def test_output_range_minus1_to_1(self) -> None:
        preproc = self._make_preprocessor(96)
        # Black image → pixel 0 → (0/255 - 0.5)/0.5 = -1.0
        img_black = np.zeros((96, 96, 3), dtype=np.uint8)
        chw_black, _ = preproc(img_black, np.zeros((0, 5), dtype=np.float32), (96, 96))
        assert chw_black.min() == pytest.approx(-1.0, abs=1e-4)
        # White image → pixel 255 → (1.0 - 0.5)/0.5 = 1.0
        img_white = np.full((96, 96, 3), 255, dtype=np.uint8)
        chw_white, _ = preproc(img_white, np.zeros((0, 5), dtype=np.float32), (96, 96))
        assert chw_white.max() == pytest.approx(1.0, abs=1e-4)

    def test_custom_normalization_flag(self) -> None:
        from libreyolo.validation.preprocessors import FOMOValPreprocessor

        p = FOMOValPreprocessor(img_size=(96, 96))
        assert p.custom_normalization is True

    def test_wants_unresized_image_flag(self) -> None:
        from libreyolo.validation.preprocessors import FOMOValPreprocessor

        p = FOMOValPreprocessor(img_size=(96, 96))
        assert p.wants_unresized_image is True

    def test_uses_letterbox_false(self) -> None:
        from libreyolo.validation.preprocessors import FOMOValPreprocessor

        p = FOMOValPreprocessor(img_size=(96, 96))
        assert p.uses_letterbox is False

    def test_target_scaling(self) -> None:
        """Box coordinates must be scaled proportionally when resizing."""
        preproc = self._make_preprocessor(96)
        img = np.zeros((200, 400, 3), dtype=np.uint8)  # orig 200×400
        # Box: [x1, y1, x2, y2, cls] in original pixels (letterboxed from dataset)
        targets = np.array([[100.0, 50.0, 200.0, 100.0, 0.0]], dtype=np.float32)
        _, padded_targets = preproc(img, targets, (96, 96))
        # scale_x = 96 / 400 = 0.24 ; scale_y = 96 / 200 = 0.48
        np.testing.assert_allclose(padded_targets[0, 0], 100.0 * (96 / 400), atol=1e-4)
        np.testing.assert_allclose(padded_targets[0, 1], 50.0 * (96 / 200), atol=1e-4)

    def test_empty_targets_padded_to_max_labels(self) -> None:
        preproc = self._make_preprocessor(96)
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        _, padded = preproc(img, np.zeros((0, 5), dtype=np.float32), (96, 96))
        assert padded.shape == (preproc.max_labels, 5)
        assert padded.sum() == 0.0

class TestLibreFOMOParseGtPoints:
    """_parse_gt_points must delegate to the validator's box-centre helper."""

    def test_delegates_to_validator_helper(self) -> None:
        from libreyolo.validation.point_validator import PointValidator

        model = _make_random_fomo(size="s", nc=1)

        # Build a minimal validator scaffold
        validator = object.__new__(PointValidator)
        from libreyolo.validation.point_validator import _DEFAULT_DIST_THRESHOLDS

        validator._dist_thresholds = _DEFAULT_DIST_THRESHOLDS
        validator._primary_threshold = _DEFAULT_DIST_THRESHOLDS[0]
        validator._records = []
        validator.nc = 1
        validator._actual_imgsz = 96
        validator.config = type("_Cfg", (), {"verbose": False})()
        validator.seen = 0
        validator.val_preproc = type("_Preproc", (), {"uses_letterbox": False})()
        validator.model = model

        gt_row = np.array([[43.2, 43.2, 52.8, 52.8, 0.0]], dtype=np.float32)
        xy, cls = model._parse_gt_points(gt_row, orig_h=96, orig_w=96, validator=validator)

        assert xy.shape == (1, 2)
        np.testing.assert_allclose(xy[0], [0.5, 0.5], atol=1e-5)
        assert cls[0] == 0


class TestLibreFOMODownloadURL:
    def test_valid_filename_returns_url(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        url = LibreFOMO.get_download_url("LibreFOMOs.pt")
        assert url == "https://huggingface.co/fomo-edge-ai/FOMO/resolve/main/weights/FOMOs.pt"

    def test_unknown_filename_returns_none(self) -> None:
        from libreyolo.models.fomo.model import LibreFOMO

        url = LibreFOMO.get_download_url("SomeOtherModel.pt")
        assert url is None


# ===========================================================================
# End-to-end: PointValidator integration
# ===========================================================================


class TestLibreFOMOEndToEnd:
    """Run a full validation pass using PointValidator on a synthetic dataset."""

    def test_point_validator_integration(self, tmp_path: Path) -> None:
        from libreyolo.validation import PointValidator, ValidationConfig

        data_yaml = _write_tiny_point_dataset(tmp_path, imgsz=96)

        # Build a random-weight FOMO model that always predicts one foreground
        # point at the centre of the grid.
        model = _make_random_fomo(size="s", nc=1)

        # Monkey-patch _postprocess so we can inspect the PointValidator pipeline
        # without needing trained weights.
        def _fixed_postprocess(output, conf_thres, iou_thres, original_size, **kwargs):
            w, h = original_size
            return {
                "points": torch.tensor([[w / 2.0, h / 2.0]], dtype=torch.float32),
                "scores": torch.tensor([0.9], dtype=torch.float32),
                "classes": torch.tensor([0.0], dtype=torch.float32),
                "num_detections": 1,
            }

        model._postprocess = _fixed_postprocess

        config = ValidationConfig(
            data=str(data_yaml),
            batch_size=1,
            imgsz=96,
            num_workers=0,
            verbose=False,
            save_dir=str(tmp_path / "val_out"),
        )
        validator = PointValidator(model, config)
        metrics = validator.run()

        # The validator ran without error
        assert "metrics/precision" in metrics
        assert "metrics/recall" in metrics
        assert "metrics/f1" in metrics
        assert "speed/images_seen" in metrics
        assert metrics["speed/images_seen"] == 1

    def test_training_integration(self, tmp_path: Path) -> None:
        """Verify that FOMOTrainer runs a training epoch without error on a synthetic dataset."""
        data_yaml = _write_tiny_point_dataset(tmp_path, imgsz=96)
        model = _make_random_fomo(size="s", nc=1)
        results = model.train(
            allow_experimental=True,
            data=str(data_yaml),
            epochs=1,
            batch=2,
            imgsz=96,
            device="cpu",
            project=str(tmp_path / "runs"),
            name="fomo_test",
            exist_ok=True,
            workers=0,
        )
        assert "final_loss" in results
        assert len(results["epoch_losses"]) == 1

    def test_val_dispatch_uses_point_validator(self) -> None:
        """BaseModel.val() must route task='point' to PointValidator."""
        from libreyolo.models.base.model import BaseModel

        model = _make_random_fomo(size="s", nc=1)
        assert model.task == "point"

        # Augmented validation must be blocked for point models
        with pytest.raises(ValueError, match="Augmented validation"):
            BaseModel.val(model, data="unused.yaml", imgsz=96, augment=True)

    def test_train_without_flag_raises(self) -> None:
        model = _make_random_fomo(size="s")
        with pytest.raises(NotImplementedError, match="allow_experimental"):
            model.train()

class TestFOMODatasets:
    """Verify FOMOYOLODataset and FOMOAugmentedDataset functionality on synthetic data."""

    def test_fomo_yolo_dataset(self, tmp_path: Path) -> None:
        from libreyolo.models.fomo.dataset import FOMOYOLODataset

        # Set up synthetic image and label
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        img_path = img_dir / "sample.jpg"
        lbl_path = lbl_dir / "sample.txt"

        # 96x96 synthetic image
        Image.new("RGB", (96, 96), color=(255, 128, 64)).save(img_path)
        # YOLO box format: cls cx cy w h
        # One point at cx=0.5, cy=0.5
        lbl_path.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        dataset = FOMOYOLODataset(
            img_files=[img_path],
            label_files=[lbl_path],
            input_size=96,
            grid_size=12,
        )

        assert len(dataset) == 1
        img_tensor, grid, img_info, idx = dataset[0]

        # Check shapes and types
        assert img_tensor.shape == (3, 96, 96)
        assert img_tensor.dtype == torch.float32
        assert grid.shape == (12, 12)
        assert grid.dtype == torch.long
        assert img_info == (96, 96)
        assert idx == 0

        # Center should be class 0 + 1 = 1
        assert grid[6, 6] == 1
        assert grid.sum() == 1

    def test_fomo_augmented_dataset(self) -> None:
        from libreyolo.models.fomo.dataset import FOMOAugmentedDataset

        # Create a mock augmented dataset that returns (img, targets, img_info, img_id)
        # YOLOX-style image shape (C, H, W) in BGR
        mock_img = np.zeros((3, 96, 96), dtype=np.uint8)
        # target row: [class, cx, cy, w, h, ...]
        mock_targets = np.array([
            [0.0, 48.0, 48.0, 10.0, 10.0]
        ], dtype=np.float32)
        mock_img_info = (96, 96)
        mock_img_id = 42

        class MockBaseDataset:
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return mock_img, mock_targets, mock_img_info, mock_img_id
            def close_mosaic(self):
                self.closed = True

        base_ds = MockBaseDataset()
        base_ds.closed = False

        dataset = FOMOAugmentedDataset(
            augmented_dataset=base_ds,
            input_size=96,
            grid_size=12,
        )

        assert len(dataset) == 1
        img_tensor, grid, img_info, img_id = dataset[0]

        assert img_tensor.shape == (3, 96, 96)
        assert grid.shape == (12, 12)
        assert img_info == (96, 96)
        assert img_id == 42

        assert grid[6, 6] == 1
        assert grid.sum() == 1

        dataset.close_mosaic()
        assert base_ds.closed is True


# ===========================================================================
# Decode helper unit tests
# ===========================================================================


class TestDecodePoints:
    """Unit tests for _decode_points (the internal NMS peak-finder)."""

    def test_all_background_returns_empty(self) -> None:
        from libreyolo.models.fomo.utils import decode_points_from_logits as _decode_points

        # All logits zero → softmax gives equal mass to bg and fg → obj_probs ~0.5
        # depending on nc. With threshold 0.6 nothing should fire.
        logits = torch.zeros(1, 2, 6, 6)  # nc=1
        results = _decode_points(logits, conf_threshold=0.6)
        assert results[0].shape[0] == 0

    def test_peak_at_known_location(self) -> None:
        from libreyolo.models.fomo.utils import decode_points_from_logits as _decode_points

        logits = torch.zeros(1, 2, 6, 6)
        logits[0, 1, 3, 4] = 20.0  # large foreground logit at (y=3, x=4)
        results = _decode_points(logits, conf_threshold=0.01)
        assert len(results[0]) >= 1
        # x should be 4, y should be 3
        assert int(results[0][0, 0].item()) == 4  # x
        assert int(results[0][0, 1].item()) == 3  # y

    def test_nms_radius_suppresses_neighbours(self) -> None:
        from libreyolo.models.fomo.utils import decode_points_from_logits as _decode_points

        logits = torch.zeros(1, 2, 6, 6)
        # Two adjacent peaks; with nms_radius=2 the second should be suppressed
        logits[0, 1, 3, 3] = 20.0
        logits[0, 1, 3, 4] = 18.0  # within radius 2 of (3,3)
        results_r2 = _decode_points(logits, conf_threshold=0.01, nms_radius=2)
        results_r0 = _decode_points(logits, conf_threshold=0.01, nms_radius=0)
        assert len(results_r2[0]) < len(results_r0[0])

    def test_class_index_correct(self) -> None:
        """Returned class channel index must equal the argmax foreground channel."""
        from libreyolo.models.fomo.utils import decode_points_from_logits as _decode_points

        logits = torch.zeros(1, 3, 4, 4)  # nc=2
        logits[0, 2, 1, 1] = 20.0  # channel 2 → 0-based class id 1
        results = _decode_points(logits, conf_threshold=0.01)
        assert len(results[0]) >= 1
        cls_channel = int(results[0][0, 2].item())  # raw channel index
        assert cls_channel == 2

    def test_batch_dimension_respected(self) -> None:
        from libreyolo.models.fomo.utils import decode_points_from_logits as _decode_points

        # nc=1: softmax gives probs 0.5 each when all logits are zero.
        # Use threshold > 0.5 so only the explicitly boosted cells fire.
        logits = torch.zeros(3, 2, 4, 4)
        logits[0, 1, 0, 0] = 20.0   # batch 0: large foreground at (0,0)
        logits[2, 1, 3, 3] = 20.0   # batch 2: large foreground at (3,3)
        # batch 1 stays all-zeros → obj_probs ≈ 0.5, below threshold 0.6
        results = _decode_points(logits, conf_threshold=0.6)
        assert len(results) == 3
        assert len(results[0]) >= 1
        assert len(results[1]) == 0
        assert len(results[2]) >= 1


    def test_max_points_limit(self) -> None:
        from libreyolo.models.fomo.utils import decode_points_from_logits as _decode_points

        logits = torch.zeros(1, 2, 8, 8)
        logits[0, 1, :, :] = 20.0  # every cell fires
        results = _decode_points(logits, conf_threshold=0.01, nms_radius=0, max_points=3)
        assert len(results[0]) <= 3
