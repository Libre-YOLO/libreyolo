from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from libreyolo.export.calibration import CalibrationDataLoader


pytestmark = pytest.mark.unit


def _loader(*, family: str | None, task: str | None, shape=(1, 3, 8, 12)):
    loader = CalibrationDataLoader.__new__(CalibrationDataLoader)
    loader.imgsz = (shape[2], shape[3])
    loader.batch = shape[0]
    loader.input_shape = shape
    loader._sample_shape = shape[1:]
    loader.model_family = family
    loader.task = task
    loader.model_size = None
    loader._preprocess_fn = None
    return loader


def _iter_loader(sample_ids, *, batch: int, failing_ids=()):
    loader = CalibrationDataLoader.__new__(CalibrationDataLoader)
    loader.batch = batch
    loader.img_files = [Path(str(sample_id)) for sample_id in sample_ids]
    loader._num_batches = (len(loader.img_files) + batch - 1) // batch
    failing_ids = set(failing_ids)

    def preprocess(path):
        sample_id = int(path.name)
        if sample_id in failing_ids:
            raise ValueError("invalid calibration sample")
        return np.full((1, 1, 1), sample_id, dtype=np.float32)

    loader._preprocess = preprocess
    return loader


def _batch_ids(batch):
    return batch[:, 0, 0, 0].astype(int).tolist()


def test_depth_calibration_uses_fixed_runtime_stretch():
    loader = _loader(family="depth_anything", task="depth")
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    image[..., 0] = 255

    sample = loader._preprocess_array(image)

    assert sample.shape == (3, 8, 12)
    assert sample.dtype == np.float32
    assert np.all(sample[0] == 1.0)
    assert np.all(sample[1:] == 0.0)


@pytest.mark.parametrize("source_shape", [(3, 5), (24, 30)])
def test_restoration_calibration_is_fixed_to_profile_shape(source_shape):
    loader = _loader(family="nafnet", task="restore")
    image = np.full((*source_shape, 3), 127, dtype=np.uint8)

    sample = loader._preprocess_array(image)

    assert sample.shape == (3, 8, 12)
    assert sample.dtype == np.float32


def test_yolonas_pose_calibration_matches_runtime_pose_preprocessor():
    from libreyolo.models.yolonas.utils import preprocess_pose_image

    loader = _loader(family="yolonas", task="pose", shape=(1, 3, 640, 640))
    image = np.zeros((9, 5, 3), dtype=np.uint8)
    image[..., 0] = 255

    actual = loader._preprocess_array(image)
    expected, *_ = preprocess_pose_image(
        image,
        input_size=640,
        color_format="rgb",
    )

    np.testing.assert_allclose(actual, expected.squeeze(0).numpy())


def test_generic_calibration_rejects_variable_preprocess_shape():
    loader = _loader(family="yolo9", task="detect")
    loader._preprocess_fn = lambda image, imgsz: (
        np.zeros((3, 4, 4), dtype=np.float32),
        1.0,
    )

    with pytest.raises(ValueError, match="must match the exported runtime"):
        loader._preprocess_array(np.zeros((4, 4, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("sample_ids", "batch_size", "expected_counts"),
    [
        (range(8), 16, [2] * 8),
        (range(18), 16, [2] * 14 + [1] * 4),
    ],
)
def test_calibration_tail_padding_balances_valid_samples(
    sample_ids, batch_size, expected_counts
):
    batches = list(_iter_loader(sample_ids, batch=batch_size))
    observed = np.bincount(
        [sample_id for batch in batches for sample_id in _batch_ids(batch)],
        minlength=len(expected_counts),
    )

    assert all(batch.shape == (batch_size, 1, 1, 1) for batch in batches)
    assert observed.tolist() == expected_counts


def test_calibration_tail_padding_excludes_invalid_samples():
    batches = list(_iter_loader(range(4), batch=4, failing_ids={1}))

    assert _batch_ids(batches[0]) == [0, 2, 3, 0]


def test_calibration_loader_rejects_all_invalid_samples():
    loader = _iter_loader(range(2), batch=4, failing_ids={0, 1})

    with pytest.raises(RuntimeError, match="No calibration images matched"):
        list(loader)


def test_calibration_exact_batches_are_not_padded():
    batches = list(_iter_loader(range(4), batch=2))

    assert [_batch_ids(batch) for batch in batches] == [[0, 1], [2, 3]]
