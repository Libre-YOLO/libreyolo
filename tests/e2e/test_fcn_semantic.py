"""Real-checkpoint FCN semantic predict, validation, and UI smoke gate."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreYOLO
from libreyolo.ui.server import _UIState

from .conftest import (
    FCN_SEMANTIC_NIGHTLY_PARAMS,
    cuda_cleanup,
    require_test_weights,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.general_nightly,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.fcn,
]


def _write_self_consistency_dataset(tmp_path, model, image_path, mask) -> str:
    image_dir = tmp_path / "images" / "val"
    mask_dir = tmp_path / "masks" / "val"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    shutil.copy2(image_path, image_dir / image_path.name)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(
        mask_dir / f"{image_path.stem}.png"
    )

    names = ["names:"]
    names.extend(f"  {index}: {name}" for index, name in model.names.items())
    yaml_path = tmp_path / "fcn-semantic.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {tmp_path.as_posix()}",
                "val: images/val",
                "masks_dir: masks",
                f"nc: {model.nb_classes}",
                *names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(yaml_path)


@pytest.mark.parametrize("family,size,weights", FCN_SEMANTIC_NIGHTLY_PARAMS)
def test_fcn_real_checkpoint_predict_val_and_ui(family, size, weights, tmp_path):
    weights = require_test_weights(weights, expected_family=family)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreYOLO(weights, size=size, device=device)
    ui_state = None
    try:
        rows = np.arange(64, dtype=np.uint16)[:, None]
        cols = np.arange(64, dtype=np.uint16)[None, :]
        image = np.stack(
            (
                np.broadcast_to(cols * 3, (64, 64)),
                np.broadcast_to(rows * 4, (64, 64)),
                (rows * 5 + cols * 7) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        image_path = tmp_path / "fcn-smoke.png"
        Image.fromarray(image, mode="RGB").save(image_path)

        result = model.predict(str(image_path), imgsz=64)
        assert result.semantic_mask is not None
        assert tuple(result.semantic_mask.data.shape) == (64, 64)
        mask = result.semantic_mask.data.cpu().numpy()

        data = _write_self_consistency_dataset(
            tmp_path / "dataset", model, image_path, mask
        )
        metrics = model.val(
            data=data,
            imgsz=64,
            batch=1,
            workers=0,
            device=device,
            verbose=False,
        )
        assert metrics["metrics/mIoU"] == pytest.approx(1.0)
        assert metrics["metrics/pixel_accuracy"] == pytest.approx(1.0)

        del model
        model = None
        cuda_cleanup()
        ui_state = _UIState(device=device)
        ui_state.run_dir = tmp_path / "ui"
        ui_state.run_dir.mkdir()
        rendered = ui_state.infer(
            f"fcn-{size}",
            0.25,
            image_path.name,
            image_path.read_bytes(),
        )
        assert rendered["task"] == "semantic"
        assert rendered["rendered"].startswith("data:image/")
        assert rendered["saved"]
    finally:
        if ui_state is not None:
            ui_state._models.clear()
            shutil.rmtree(ui_state._input_dir, ignore_errors=True)
        if model is not None:
            del model
        cuda_cleanup()
