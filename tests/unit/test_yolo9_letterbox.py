"""YOLOv9 letterbox stamp + PGI aux: no silent 1.5 → 1.6 geometry flip."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.preprocess.letterbox import (
    DEFAULT_LETTERBOX_PAD,
    LETTERBOX_CENTER,
    LETTERBOX_TOPLEFT,
    apply_letterbox_hwc,
    letterbox_geometry,
    normalize_letterbox_pad,
    unletterbox_xyxy,
)
from libreyolo.preprocess.yolo9 import preprocess_numpy
from libreyolo.postprocess.yolo9 import postprocess
from libreyolo.validation.preprocessors import YOLO9ValPreprocessor


def _uint8_gradient(h=40, w=80):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    img[:, :, 2] = 40
    return img


def test_unmarked_letterbox_pad_defaults_to_topleft():
    assert normalize_letterbox_pad(None) == LETTERBOX_TOPLEFT
    assert normalize_letterbox_pad("") == LETTERBOX_TOPLEFT
    assert normalize_letterbox_pad("nope") == LETTERBOX_TOPLEFT
    assert DEFAULT_LETTERBOX_PAD == LETTERBOX_TOPLEFT


def test_topleft_geometry_has_zero_pad_offsets():
    ratio, new_h, new_w, pad_left, pad_top = letterbox_geometry(40, 80, 64, 64, "topleft")
    assert pad_left == 0
    assert pad_top == 0
    assert new_w == 64
    assert new_h == 32
    assert ratio == pytest.approx(0.8)


def test_center_geometry_splits_pad():
    _ratio, _nh, _nw, pad_left, pad_top = letterbox_geometry(40, 80, 64, 64, "center")
    assert pad_left == 0
    assert pad_top == 16


def test_topleft_preprocess_is_bit_identical_to_historical_placement():
    img = _uint8_gradient()
    historical = np.full((64, 64, 3), 114, dtype=np.uint8)
    import cv2

    resized = cv2.resize(img, (64, 32), interpolation=cv2.INTER_LINEAR)
    historical[:32, :64] = resized

    canvas, ratio, pad_left, pad_top = apply_letterbox_hwc(img, 64, 64, pad="topleft")
    assert pad_left == 0
    assert pad_top == 0
    np.testing.assert_array_equal(canvas, historical)
    assert ratio == pytest.approx(0.8)


def test_default_preprocess_numpy_matches_topleft():
    img = _uint8_gradient()
    default, _ = preprocess_numpy(img, 64)
    topleft, _ = preprocess_numpy(img, 64, letterbox_pad="topleft")
    center, _ = preprocess_numpy(img, 64, letterbox_pad="center")
    np.testing.assert_array_equal(default, topleft)
    assert not np.array_equal(default, center)


def test_val_preprocessor_default_matches_predict_topleft():
    img = _uint8_gradient()
    tensor, _ = preprocess_numpy(img, 64, letterbox_pad=None)
    val, _ = YOLO9ValPreprocessor((64, 64), max_labels=1)(
        img[:, :, ::-1].copy(),
        np.zeros((0, 5), dtype=np.float32),
        (64, 64),
    )
    np.testing.assert_allclose(tensor, val, atol=1e-6)


def test_val_preprocessor_center_moves_content():
    img = _uint8_gradient()
    topleft, _ = YOLO9ValPreprocessor((64, 64), letterbox_pad="topleft")(
        img[:, :, ::-1].copy(),
        np.zeros((0, 5), dtype=np.float32),
        (64, 64),
    )
    center, _ = YOLO9ValPreprocessor((64, 64), letterbox_pad="center")(
        img[:, :, ::-1].copy(),
        np.zeros((0, 5), dtype=np.float32),
        (64, 64),
    )
    assert not np.array_equal(topleft, center)
    r, off_x, off_y = YOLO9ValPreprocessor((64, 64), letterbox_pad="center").letterbox_scale(
        40, 80, 64
    )
    assert off_x == 0.0
    assert off_y == 16.0
    assert r == pytest.approx(0.8)


def test_postprocess_topleft_undo_matches_historical_divide_by_ratio():
    pred = torch.zeros(1, 6, 1)
    pred[0, :4, 0] = torch.tensor([0.0, 0.0, 320.0, 320.0])
    pred[0, 4, 0] = 0.9

    out = postprocess(
        {"predictions": pred},
        input_size=640,
        original_size=(1280, 960),
    )
    torch.testing.assert_close(
        torch.as_tensor(out["boxes"]),
        torch.tensor([[0.0, 0.0, 640.0, 640.0]]),
    )

    explicit = postprocess(
        {"predictions": pred.clone()},
        input_size=640,
        original_size=(1280, 960),
        letterbox_pad="topleft",
    )
    torch.testing.assert_close(
        torch.as_tensor(out["boxes"]),
        torch.as_tensor(explicit["boxes"]),
    )


def test_postprocess_center_subtracts_pad():
    pred = torch.zeros(1, 6, 1)
    # Box sitting in the padded canvas at y=16..336 on a 64-tall content
    # after center pad of 16 on a 40x80 → 64x64 letterbox (ratio 0.8).
    pred[0, :4, 0] = torch.tensor([0.0, 16.0, 64.0, 48.0])
    pred[0, 4, 0] = 0.9
    out = postprocess(
        {"predictions": pred},
        input_size=64,
        original_size=(80, 40),
        letterbox_pad="center",
    )
    torch.testing.assert_close(
        torch.as_tensor(out["boxes"]),
        torch.tensor([[0.0, 0.0, 80.0, 40.0]]),
    )


def test_unletterbox_topleft_is_pure_divide():
    boxes = torch.tensor([[8.0, 4.0, 16.0, 12.0]])
    out = unletterbox_xyxy(boxes, orig_w=80, orig_h=40, input_h=64, input_w=64, pad="topleft")
    torch.testing.assert_close(out, torch.tensor([[10.0, 5.0, 20.0, 15.0]]))


def test_unmarked_yolo9_checkpoint_stays_topleft(tmp_path):
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    raw = LibreYOLO9Model(config="t", nb_classes=2)
    ckpt = wrap_libreyolo_checkpoint(
        raw.state_dict(),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=640,
    )
    path = tmp_path / "old.pt"
    torch.save(ckpt, path)

    loaded = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert loaded.letterbox_pad == LETTERBOX_TOPLEFT
    assert loaded.model.aux is None


def test_stamped_center_checkpoint_is_honored(tmp_path):
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    raw = LibreYOLO9Model(config="t", nb_classes=2)
    ckpt = wrap_libreyolo_checkpoint(
        raw.state_dict(),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=640,
        letterbox_pad="center",
    )
    path = tmp_path / "official.pt"
    torch.save(ckpt, path)

    loaded = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert loaded.letterbox_pad == LETTERBOX_CENTER
    saved = loaded.save(str(tmp_path / "resave.pt"))
    from libreyolo.utils.serialization import load_untrusted_torch_file

    resaved = load_untrusted_torch_file(saved, map_location="cpu", context="resave")
    assert resaved["letterbox_pad"] == LETTERBOX_CENTER


def test_aux_keys_are_ignored_at_inference(tmp_path):
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    raw = LibreYOLO9Model(config="t", nb_classes=2)
    raw.enable_aux(0.25)
    ckpt = wrap_libreyolo_checkpoint(
        raw.state_dict(),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=640,
        letterbox_pad="center",
    )
    path = tmp_path / "with_aux.pt"
    torch.save(ckpt, path)

    loaded = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert loaded.model.aux is None
    assert all(not k.startswith("aux.") for k in loaded.model.state_dict())


def test_enable_aux_is_training_only_and_idempotent():
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    model = LibreYOLO9Model(config="t", nb_classes=2)
    assert model.aux is None
    model.enable_aux(0.25)
    first = model.aux
    model.enable_aux(0.25)
    assert model.aux is first
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    assert "predictions" in out


def test_aux_training_forward_combines_losses():
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    model = LibreYOLO9Model(config="t", nb_classes=2)
    model.enable_aux(0.25)
    model.train()
    targets = torch.zeros(1, 4, 5)
    targets[0, 0] = torch.tensor([0.0, 0.2, 0.2, 0.4, 0.4])
    out = model(torch.zeros(1, 3, 64, 64), targets=targets)
    assert torch.isfinite(out["total_loss"])
    bare = LibreYOLO9Model(config="t", nb_classes=2)
    bare.train()
    main_only = bare(torch.zeros(1, 3, 64, 64), targets=targets)
    # Combined PGI loss is a different tensor than the single-head path.
    assert out["total_loss"].shape == main_only["total_loss"].shape


def test_yolo9_config_recipe_defaults():
    from libreyolo.training.config import YOLO9Config

    cfg = YOLO9Config()
    assert cfg.max_labels == 300
    assert cfg.aux_weight == 0.25
    assert cfg.warmup_momentum == 0.8


def test_linear_scheduler_momentum_warmup():
    from libreyolo.training.scheduler import LinearLRScheduler

    sched = LinearLRScheduler(
        lr=0.01,
        iters_per_epoch=10,
        total_epochs=10,
        warmup_epochs=3,
        warmup_momentum=0.8,
        momentum=0.937,
    )
    assert sched.update_momentum(0) == pytest.approx(0.8)
    assert sched.update_momentum(30) == pytest.approx(0.937)
    mid = sched.update_momentum(15)
    assert 0.8 < mid < 0.937

    plain = LinearLRScheduler(lr=0.01, iters_per_epoch=10, total_epochs=10)
    assert plain.update_momentum(1) is None
