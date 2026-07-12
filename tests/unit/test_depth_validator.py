"""Unit tests for the depth validator."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from libreyolo.validation import DepthValidator, ValidationConfig
from libreyolo.validation.depth_validator import align_inverse_depth

pytestmark = pytest.mark.unit

IMGSZ = 32

NEAR_DEPTH = 2.0
FAR_DEPTH = 8.0


def _write_image(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(30, 60, 90)).save(path)


def _write_depth(path: Path, array: np.ndarray, scale: float = 256.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (array * scale).round().astype(np.uint16)
    Image.fromarray(encoded).save(path)


def _make_dataset_yaml(root: Path, n_images: int = 2) -> Path:
    """Square dataset: left half near (2.0), right half far (8.0)."""
    for i in range(n_images):
        _write_image(root / "images" / "val" / f"img{i}.jpg", IMGSZ, IMGSZ)
        depth = np.full((IMGSZ, IMGSZ), FAR_DEPTH, dtype=np.float64)
        depth[:, : IMGSZ // 2] = NEAR_DEPTH
        _write_depth(root / "depths" / "val" / f"img{i}.png", depth)
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "val: images/val",
                "names: [depth]",
                "",
            ]
        )
    )
    return yaml_path


class _StubDepthModel:
    """Minimal model double satisfying the BaseValidator contract."""

    size = "n"
    names = {0: "depth"}
    depth_resize_mode = "stretch"

    def __init__(self, prediction: str = "perfect"):
        self.model = nn.Identity()
        self._prediction = prediction

    def _get_model_name(self) -> str:
        return "stub"

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        pred = torch.full((batch, 1, height, width), 1.0 / FAR_DEPTH)
        pred[..., : width // 2] = 1.0 / NEAR_DEPTH
        if self._prediction == "affine":
            # An affine transform of the true inverse depth must be
            # indistinguishable after scale-and-shift alignment.
            pred = 3.0 * pred + 7.0
        elif self._prediction == "inverted":
            pred = pred.max() + pred.min() - pred
        elif self._prediction == "flat":
            pred = torch.ones((batch, 1, height, width))
        return pred


def _run_validator(tmp_path, prediction: str, **overrides):
    yaml_path = _make_dataset_yaml(tmp_path)
    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        batch_size=2,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
        **overrides,
    )
    validator = DepthValidator(_StubDepthModel(prediction), config)
    return validator.run()


def test_align_inverse_depth_recovers_affine_transform():
    gt = torch.rand(100).double() + 0.1
    pred = 2.5 * gt - 0.7

    aligned = align_inverse_depth(pred, gt)
    assert torch.allclose(aligned, gt, atol=1e-6)


def test_align_inverse_depth_degenerate_prediction_falls_back():
    gt = torch.rand(50).double() + 0.1
    pred = torch.full((50,), 4.0).double()

    aligned = align_inverse_depth(pred, gt)
    assert torch.isfinite(aligned).all()
    assert float(aligned.median()) == pytest.approx(float(gt.median()), abs=1e-6)


def test_perfect_predictions_score_perfect_metrics(tmp_path):
    metrics = _run_validator(tmp_path, prediction="perfect")

    assert metrics["metrics/abs_rel"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["metrics/delta1"] == pytest.approx(1.0)
    assert metrics["fitness"] == pytest.approx(1.0)


def test_affine_transformed_predictions_score_perfect_metrics(tmp_path):
    # Relative depth contract: predictions are affine-invariant, so any
    # scale/shift of the true inverse depth must validate perfectly.
    metrics = _run_validator(tmp_path, prediction="affine")

    assert metrics["metrics/abs_rel"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["metrics/delta1"] == pytest.approx(1.0)


def test_inverted_predictions_do_not_score_perfect_metrics(tmp_path):
    metrics = _run_validator(tmp_path, prediction="inverted")

    assert metrics["metrics/abs_rel"] > 0.05
    assert metrics["metrics/delta1"] < 1.0


def test_flat_prediction_scores_imperfect_metrics(tmp_path):
    metrics = _run_validator(tmp_path, prediction="flat")

    assert np.isfinite(metrics["metrics/abs_rel"])
    assert metrics["metrics/abs_rel"] > 0.05
    assert metrics["metrics/delta1"] < 1.0


def test_low_resolution_predictions_are_upsampled(tmp_path):
    class _StrideFourModel(_StubDepthModel):
        def _forward(self, images: torch.Tensor) -> torch.Tensor:
            batch, _, height, width = images.shape
            pred = torch.full(
                (batch, 1, height // 4, width // 4), 1.0 / FAR_DEPTH
            )
            pred[..., : width // 8] = 1.0 / NEAR_DEPTH
            return pred

    yaml_path = _make_dataset_yaml(tmp_path)
    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )
    metrics = DepthValidator(_StrideFourModel(), config).run()

    # Bilinear upsampling blurs the depth edge; everything else is exact.
    assert metrics["metrics/delta1"] > 0.9


def test_invalid_pixels_are_excluded(tmp_path):
    # GT is valid on the left half only; a model predicting garbage on the
    # invalid half must still score perfectly.
    _write_image(tmp_path / "images" / "val" / "a.jpg", IMGSZ, IMGSZ)
    depth = np.zeros((IMGSZ, IMGSZ), dtype=np.float64)
    depth[:, : IMGSZ // 2] = NEAR_DEPTH
    _write_depth(tmp_path / "depths" / "val" / "a.png", depth)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        f"path: {tmp_path.as_posix()}\nval: images/val\nnames: [depth]\n"
    )

    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )

    class _HalfGarbageModel(_StubDepthModel):
        def _forward(self, images: torch.Tensor) -> torch.Tensor:
            batch, _, height, width = images.shape
            pred = torch.full((batch, 1, height, width), 1.0 / NEAR_DEPTH)
            pred[..., width // 2 :] = 123.0  # garbage where GT is invalid
            return pred

    metrics = DepthValidator(_HalfGarbageModel(), config).run()
    assert metrics["metrics/abs_rel"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["metrics/delta1"] == pytest.approx(1.0)


def test_all_invalid_pixels_raise(tmp_path):
    _write_image(tmp_path / "images" / "val" / "a.jpg", IMGSZ, IMGSZ)
    depth = np.zeros((IMGSZ, IMGSZ), dtype=np.float64)
    _write_depth(tmp_path / "depths" / "val" / "a.png", depth)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        f"path: {tmp_path.as_posix()}\nval: images/val\nnames: [depth]\n"
    )

    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )

    with pytest.raises(ValueError, match="no images with at least two valid pixels"):
        DepthValidator(_StubDepthModel(), config).run()


def test_imgsz_divisor_mismatch_raises(tmp_path):
    yaml_path = _make_dataset_yaml(tmp_path)
    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,  # 32, not divisible by 14
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )
    model = _StubDepthModel()
    model.depth_imgsz_divisor = 14

    with pytest.raises(ValueError, match="divisible by 14"):
        DepthValidator(model, config).run()


def test_native_depth_validation_uses_family_geometry_and_postprocess(tmp_path):
    width, height = 20, 10
    _write_image(tmp_path / "images" / "val" / "a.jpg", width, height)
    depth = np.full((height, width), FAR_DEPTH, dtype=np.float64)
    _write_depth(tmp_path / "depths" / "val" / "a.png", depth)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        f"path: {tmp_path.as_posix()}\nval: images/val\nnames: [depth]\n"
    )

    class _NativeModel(_StubDepthModel):
        depth_resize_mode = "native"

        def _preprocess(self, image, color_format, input_size):
            del image, color_format, input_size
            return torch.zeros(1, 3, 16, 32), None, (width, height), 1.0

        def _postprocess(self, output, **kwargs):
            assert kwargs["original_size"] == (width, height)
            resized = torch.nn.functional.interpolate(
                output,
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )
            return {"depth": resized[0, 0]}

    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=32,
        batch_size=4,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )
    validator = DepthValidator(_NativeModel(), config)
    validator.dataloader = validator._setup_dataloader()
    batch = next(iter(validator.dataloader))

    images, targets, infos, ids = validator._preprocess_batch(batch)
    raw = torch.tensor([[[[0.0, 1.0], [2.0, 4.0]]]])
    decoded = validator._postprocess_predictions(raw, batch)
    align_false = torch.nn.functional.interpolate(
        raw, size=(height, width), mode="bilinear", align_corners=False
    )[:, 0]

    assert images.shape == (1, 3, 16, 32)
    assert targets.shape == (1, height, width)
    assert infos[0]["orig_shape"] == (height, width)
    assert ids == [0]
    assert decoded.shape == (1, height, width)
    assert not torch.allclose(decoded, align_false)
