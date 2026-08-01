"""Unit tests for the native LW-DETR family."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit

# (size, input_size, queries, num_select)
LWDETR_SIZE_CASES = [
    ("t", 640, 100, 100),
    ("s", 640, 300, 300),
    ("m", 640, 300, 300),
    ("l", 640, 300, 300),
    ("x", 640, 300, 300),
]


def test_lwdetr_is_registered_and_detects_filenames():
    from libreyolo import LibreLWDETR
    from libreyolo.models.base.model import BaseModel

    assert any(cls.__name__ == "LibreLWDETR" for cls in BaseModel._registry)
    assert LibreLWDETR.FAMILY == "lwdetr"
    assert LibreLWDETR.FILENAME_PREFIX == "LibreLWDETR"
    assert LibreLWDETR.SUPPORTED_TASKS == ("detect",)
    assert LibreLWDETR.DEFAULT_TASK == "detect"
    assert LibreLWDETR.TRAIN_CONFIG is None  # inference-only

    for code in ("t", "s", "m", "l", "x"):
        assert LibreLWDETR.detect_size_from_filename(f"LibreLWDETR{code}.pt") == code
    assert LibreLWDETR.detect_size_from_filename("LWDETR_tiny_60e_coco.pth") == "t"
    assert LibreLWDETR.detect_size_from_filename("LWDETR_xlarge_60e_coco.pth") == "x"
    assert LibreLWDETR.detect_size_from_filename("LibreDFINEs.pt") is None


@pytest.mark.parametrize(
    ("size", "input_size", "queries", "num_select"), LWDETR_SIZE_CASES
)
def test_lwdetr_forward_shapes(size, input_size, queries, num_select):
    from libreyolo import LibreLWDETR

    model = LibreLWDETR(None, size=size, device="cpu")
    # Default COCO build keeps upstream's 91-column head behind the 80-class
    # user interface.
    assert model.nb_classes == 80
    assert model._arch_num_classes == 91
    assert model.model.num_select == num_select

    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, input_size, input_size))

    assert out["pred_logits"].shape == (1, queries, 91)
    assert out["pred_boxes"].shape == (1, queries, 4)


def test_lwdetr_custom_class_count_builds_contiguous_head():
    from libreyolo import LibreLWDETR

    model = LibreLWDETR(None, size="t", nb_classes=7, device="cpu")
    assert model.nb_classes == 7
    assert model._arch_num_classes == 7
    assert model.model.class_embed.out_features == 7


def test_lwdetr_rejects_input_not_divisible_by_64():
    from libreyolo import LibreLWDETR

    model = LibreLWDETR(None, size="t", device="cpu")
    with pytest.raises(ValueError, match="multiple of 64"):
        model._preprocess(np.zeros((32, 32, 3), dtype=np.uint8), input_size=600)


def test_lwdetr_training_is_not_implemented():
    from libreyolo import LibreLWDETR

    model = LibreLWDETR(None, size="t", device="cpu")
    with pytest.raises(NotImplementedError, match="Group-DETR"):
        model.train(data="coco128.yaml")


def test_lwdetr_factory_loads_v1_metadata_checkpoint(tmp_path):
    from libreyolo import LibreLWDETR, LibreYOLO

    src = LibreLWDETR(None, size="t", device="cpu")
    ckpt = tmp_path / "LibreLWDETRt.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            src.model.state_dict(),
            model_family="lwdetr",
            size="t",
            task="detect",
            nc=80,
            names={i: f"class_{i}" for i in range(80)},
            imgsz=640,
        ),
        ckpt,
    )

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "lwdetr"
    assert loaded.size == "t"
    assert loaded.input_size == 640
    assert loaded.nb_classes == 80
    assert loaded._arch_num_classes == 91


def test_lwdetr_factory_detects_upstream_style_checkpoint(tmp_path):
    from libreyolo import LibreLWDETR, LibreYOLO

    src = LibreLWDETR(None, size="t", device="cpu")
    ckpt = tmp_path / "LWDETR_tiny_60e_coco.pth"
    torch.save({"model": src.model.state_dict()}, ckpt)

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "lwdetr"
    assert loaded.size == "t"


@pytest.mark.parametrize(("size", "_i", "_q", "_n"), LWDETR_SIZE_CASES)
def test_lwdetr_detect_size_from_state_dict(size, _i, _q, _n):
    from libreyolo import LibreLWDETR

    src = LibreLWDETR(None, size=size, device="cpu")
    state_dict = src.model.state_dict()
    assert LibreLWDETR.detect_size(state_dict) == size
    assert LibreLWDETR.can_load(state_dict) is True
    # detect_nb_classes reports the user-facing count, which the factory feeds
    # straight into the constructor.
    assert LibreLWDETR.detect_nb_classes(state_dict) == 80


def test_lwdetr_postprocess_contract_and_coco_remap():
    from libreyolo.postprocess.lwdetr import postprocess
    from libreyolo.utils.coco import COCO91_TO_COCO80

    torch.manual_seed(0)
    outputs = {
        "pred_logits": torch.full((1, 4, 91), -10.0),
        "pred_boxes": torch.tensor(
            [[[0.5, 0.5, 0.2, 0.2]] * 4], dtype=torch.float32
        ),
    }
    # Category id 1 ("person") maps to contiguous 0; id 12 is one of the 11
    # unused COCO ids and must be dropped.
    outputs["pred_logits"][0, 0, 1] = 10.0
    outputs["pred_logits"][0, 1, 12] = 10.0

    result = postprocess(
        outputs,
        conf_thres=0.5,
        iou_thres=0.5,
        original_size=(100, 200),
        max_det=10,
        class_map=COCO91_TO_COCO80,
    )

    assert set(result) == {"num_detections", "boxes", "scores", "classes"}
    assert result["num_detections"] == 1
    assert result["classes"].tolist() == [0]
    assert result["boxes"].shape == (1, 4)
    # cxcywh (0.5, 0.5, 0.2, 0.2) -> xyxy scaled by (w=100, h=200)
    np.testing.assert_allclose(result["boxes"][0], [40.0, 80.0, 60.0, 120.0], rtol=1e-6)


def test_lwdetr_val_preprocessor_matches_inference_preprocess():
    from libreyolo.models.lwdetr.utils import preprocess_numpy
    from libreyolo.validation.preprocessors import LWDETRValPreprocessor

    rng = np.random.default_rng(0)
    img_bgr = rng.integers(0, 256, (7, 5, 3), dtype=np.uint8)
    preproc = LWDETRValPreprocessor(img_size=(64, 64))

    out, _ = preproc(img_bgr, np.zeros((0, 5), dtype=np.float32), (64, 64))
    expected, _ = preprocess_numpy(img_bgr[:, :, ::-1], 64)

    np.testing.assert_allclose(out, expected, rtol=0, atol=0)
    assert preproc.custom_normalization is True
    assert preproc.uses_letterbox is False
