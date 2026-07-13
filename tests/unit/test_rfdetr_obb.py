from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image
from types import SimpleNamespace

pytestmark = pytest.mark.unit


def test_rfdetr_obb_transform_refits_angle_after_nonuniform_resize():
    from libreyolo.models.rfdetr.seg_transforms import RFDETRDetTransform

    image = np.zeros((10, 20, 3), dtype=np.uint8)
    targets = np.array([[1.0, 2.0, 5.0, 6.0, 2.0, 0.25]], dtype=np.float32)
    transform = RFDETRDetTransform(
        max_labels=2,
        flip_prob=0.0,
        imgsz=64,
        multi_scale=False,
        crop_resize_prob=0.0,
        target_dim=6,
    )

    _, labels = transform(image, targets, (64, 64))

    assert labels.shape == (2, 6)
    np.testing.assert_allclose(
        labels[0],
        np.array(
            [2.0, 9.6, 25.6, 29.717402, 13.104321, -1.4438124],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    assert labels[1:].sum() == 0


def test_rfdetr_obb_transform_drops_crop_collapsed_boxes(monkeypatch):
    import libreyolo.models.rfdetr.seg_transforms as transforms
    from libreyolo.models.rfdetr.seg_transforms import RFDETRDetTransform

    random_values = iter([1.0, 0.0])
    monkeypatch.setattr(transforms.random, "random", lambda: next(random_values))
    monkeypatch.setattr(transforms.random, "choice", lambda seq: seq[0])

    randint_values = iter([20, 0, 10])
    monkeypatch.setattr(transforms.random, "randint", lambda _a, _b: next(randint_values))

    image = np.zeros((40, 40, 3), dtype=np.uint8)
    targets = np.array([[0, 10, 10, 30, 1, 0.0]], dtype=np.float32)
    transform = RFDETRDetTransform(
        max_labels=4,
        flip_prob=0.0,
        imgsz=40,
        crop_resize_prob=1.0,
        crop_intermediate_sizes=(40,),
        crop_min_size=20,
        crop_max_size=20,
        target_dim=6,
    )

    _, labels = transform(image, targets, (40, 40))

    assert labels.shape == (4, 6)
    assert labels.sum() == 0


def test_rfdetr_obb_crop_intersects_rotated_corners_before_refit(monkeypatch):
    import libreyolo.data.augment.rfdetr as transforms
    from libreyolo.data.augment.rfdetr import RFDETRDetTransform

    random_values = iter([1.0, 0.0])
    monkeypatch.setattr(transforms.random, "random", lambda: next(random_values))
    monkeypatch.setattr(transforms.random, "choice", lambda _seq: 100)
    randint_values = iter([50, 25, 50])
    monkeypatch.setattr(
        transforms.random,
        "randint",
        lambda _minimum, _maximum: next(randint_values),
    )

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    targets = np.array(
        [[20.0, 40.0, 80.0, 60.0, 0.0, np.pi / 4]],
        dtype=np.float32,
    )
    transform = RFDETRDetTransform(
        max_labels=2,
        flip_prob=0.0,
        imgsz=50,
        crop_resize_prob=1.0,
        crop_intermediate_sizes=(100,),
        crop_min_size=50,
        crop_max_size=50,
        target_dim=6,
    )

    _, labels = transform(image, targets, (50, 50))

    np.testing.assert_allclose(
        labels[0],
        [0.0, 7.071068, 32.071068, 40.0, 20.0, np.pi / 4],
        rtol=1e-5,
        atol=1e-5,
    )
    assert labels[1:].sum() == 0


def test_rfdetr_postprocess_returns_obb_payload():
    from libreyolo.models.rfdetr.utils import postprocess

    outputs = {
        "pred_logits": torch.tensor([[[0.0, 10.0], [-10.0, -10.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.25, 0.2, 0.1], [0.1, 0.1, 0.1, 0.1]]]),
        "pred_angles": torch.tensor([[[0.3], [0.0]]]),
    }
    results = postprocess(outputs, torch.tensor([[100.0, 200.0]]), num_select=1)

    assert "obb" in results[0]
    torch.testing.assert_close(
        results[0]["boxes"][0],
        torch.tensor([80.0, 20.0, 120.0, 30.0]),
    )
    torch.testing.assert_close(
        results[0]["obb"][0],
        torch.tensor(
            [100.0, 25.0, 43.048557, 10.344515, 0.15345213, 0.9999546, 1.0]
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_rfdetr_postprocess_tolerates_degenerate_obb_prediction():
    from libreyolo.models.rfdetr.utils import postprocess

    outputs = {
        "pred_logits": torch.tensor([[[10.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.0, 0.2]]]),
        "pred_angles": torch.tensor([[[0.0]]]),
    }

    results = postprocess(outputs, torch.tensor([[100.0, 200.0]]), num_select=1)

    assert "obb" in results[0]
    assert torch.isfinite(results[0]["obb"]).all()
    assert results[0]["obb"][0, 2] > 0.0
    assert results[0]["obb"][0, 3] > 0.0


def test_rfdetr_obb_load_rejects_detect_checkpoint_without_transfer_flag():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    class DummyRFDETR(torch.nn.Module):
        nb_classes = 80

        def load_state_dict(self, state_dict, strict=False, canonical_v1=False):
            return ["angle_embed.weight", "angle_embed.bias"], []

    wrapper = object.__new__(LibreRFDETR)
    wrapper.task = "obb"
    wrapper.model = DummyRFDETR()
    wrapper.nb_classes = 80
    wrapper._model_num_classes = 80
    wrapper._allow_detect_to_obb_transfer = False

    with pytest.raises(RuntimeError, match="task='detect'"):
        wrapper._load_weights(
            {
                "model_family": "rfdetr",
                "task": "detect",
                "nc": 80,
                "names": {i: f"class_{i}" for i in range(80)},
                "model": {},
            }
        )


def test_rfdetr_obb_load_rejects_missing_angle_head_without_metadata():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    class DummyRFDETR(torch.nn.Module):
        nb_classes = 80

        def load_state_dict(self, state_dict, strict=False, canonical_v1=False):
            return ["angle_embed.weight", "angle_embed.bias"], []

    wrapper = object.__new__(LibreRFDETR)
    wrapper.task = "obb"
    wrapper.model = DummyRFDETR()
    wrapper.nb_classes = 80
    wrapper._model_num_classes = 80
    wrapper._allow_detect_to_obb_transfer = False

    with pytest.raises(RuntimeError, match=r"angle_embed\.\*"):
        wrapper._load_weights({"class_embed.bias": torch.zeros(81)})


def test_rfdetr_obb_load_allows_detect_checkpoint_for_training_transfer():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    class DummyRFDETR(torch.nn.Module):
        nb_classes = 80

        def load_state_dict(self, state_dict, strict=False, canonical_v1=False):
            return ["angle_embed.weight", "angle_embed.bias"], []

    wrapper = object.__new__(LibreRFDETR)
    wrapper.task = "obb"
    wrapper.model = DummyRFDETR()
    wrapper.nb_classes = 80
    wrapper._model_num_classes = 80
    wrapper._allow_detect_to_obb_transfer = True

    wrapper._load_weights(
        {
            "model_family": "rfdetr",
            "task": "detect",
            "nc": 80,
            "names": {i: f"class_{i}" for i in range(80)},
            "model": {},
        }
    )


def test_rfdetr_top_level_names_override_nested_args_metadata():
    from libreyolo.models.rfdetr.model import LibreRFDETR
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    class DummyRFDETR(torch.nn.Module):
        nb_classes = 2
        patch_size = 16
        num_windows = 2

        def load_state_dict(self, state_dict, strict=False, canonical_v1=False):
            from torch.nn.modules.module import _IncompatibleKeys

            return _IncompatibleKeys([], [])

    wrapper = object.__new__(LibreRFDETR)
    wrapper.task = "detect"
    wrapper.model = DummyRFDETR()
    wrapper.nb_classes = 2
    wrapper._model_num_classes = 2
    wrapper._allow_detect_to_obb_transfer = False
    wrapper._allow_detect_to_pose_transfer = False
    wrapper.size = "n"
    wrapper.input_size = 384
    checkpoint = wrap_libreyolo_checkpoint(
        {},
        model_family="rfdetr",
        size="n",
        task="detect",
        nc=2,
        names={0: "top-left", 1: "top-right"},
        imgsz=384,
        args={"class_names": ["nested-left", "nested-right"]},
    )

    wrapper._load_weights(checkpoint)

    assert wrapper.names == {0: "top-left", 1: "top-right"}
    assert wrapper.input_size == 384
    assert wrapper.model.training is False


def test_rfdetr_custom_loader_adopts_native_checkpoint_imgsz():
    from libreyolo.models.rfdetr.model import LibreRFDETR
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    class DummyRFDETR(torch.nn.Module):
        nb_classes = 2
        patch_size = 16
        num_windows = 2

        def load_state_dict(self, state_dict, strict=False, canonical_v1=False):
            from torch.nn.modules.module import _IncompatibleKeys

            return _IncompatibleKeys([], [])

    wrapper = object.__new__(LibreRFDETR)
    wrapper.task = "detect"
    wrapper.model = DummyRFDETR()
    wrapper.nb_classes = 2
    wrapper._model_num_classes = 2
    wrapper._allow_detect_to_obb_transfer = False
    wrapper._allow_detect_to_pose_transfer = False
    wrapper.size = "n"
    wrapper.input_size = 384
    checkpoint = wrap_libreyolo_checkpoint(
        {},
        model_family="rfdetr",
        size="n",
        task="detect",
        nc=2,
        names={0: "left", 1: "right"},
        imgsz=416,
    )

    wrapper._load_weights(checkpoint)

    assert wrapper.input_size == 416
    assert wrapper.model.training is False


def test_rfdetr_angle_loss_is_pi_periodic():
    from libreyolo.models.rfdetr.loss import SetCriterion

    criterion = SetCriterion(
        num_classes=2,
        matcher=None,
        weight_dict={"loss_angle": 1.0},
        focal_alpha=0.25,
        losses=["angles"],
    )
    outputs = {"pred_angles": torch.tensor([[[0.25]]])}
    targets = [{"angles": torch.tensor([0.25 + torch.pi])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    losses = criterion.loss_angles(outputs, targets, indices, num_boxes=1.0)

    torch.testing.assert_close(losses["loss_angle"], torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_rfdetr_trainer_forward_passes_obb_angles_to_criterion():
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    class DummyModel(torch.nn.Module):
        def forward(self, imgs, targets=None):
            self.targets = targets
            return {"angle": torch.ones((), device=imgs.device, requires_grad=True)}

    class DummyCriterion:
        weight_dict = {"loss_angle": 2.0}

        def __call__(self, outputs, targets):
            self.targets = targets
            return {"loss_angle": outputs["angle"]}

    trainer = object.__new__(RFDETRTrainer)
    trainer.device = torch.device("cpu")
    trainer.wrapper_model = type("Wrapper", (), {"task": "obb"})()
    trainer.model = DummyModel()
    trainer.criterion = DummyCriterion()

    imgs = torch.zeros(2, 3, 8, 8)
    targets = torch.zeros(2, 2, 6)
    targets[0, 0] = torch.tensor([1.0, 4.0, 4.0, 2.0, 2.0, 0.25])

    out = trainer.on_forward(imgs, targets)

    assert "angles" in trainer.criterion.targets[0]
    torch.testing.assert_close(
        trainer.criterion.targets[0]["boxes"],
        torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    )
    torch.testing.assert_close(
        trainer.criterion.targets[0]["angles"],
        torch.tensor([0.25]),
    )
    assert trainer.criterion.targets[1]["angles"].shape == (0,)
    torch.testing.assert_close(out["total_loss"], torch.tensor(2.0))
    assert trainer.get_loss_components(out)["angle"] == pytest.approx(1.0)


def test_rfdetr_obb_multiscale_does_not_treat_angle_as_keypoints():
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    trainer = object.__new__(RFDETRTrainer)
    trainer.wrapper_model = SimpleNamespace(task="obb")
    trainer._multi_scale_scales = lambda: [128]

    imgs = torch.zeros(1, 3, 64, 64)
    targets = torch.tensor([[[0.0, 16.0, 20.0, 8.0, 10.0, 0.25]]])

    scaled_imgs, scaled_targets, _ = trainer._apply_multi_scale_batch(
        imgs,
        targets,
        None,
        step=1,
    )

    assert scaled_imgs.shape[-2:] == (128, 128)
    torch.testing.assert_close(
        scaled_targets[0, 0],
        torch.tensor([0.0, 32.0, 40.0, 16.0, 20.0, 0.25]),
    )


def test_rfdetr_trainer_derives_classes_from_names_without_nc(tmp_path):
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {tmp_path.as_posix()}",
                "train: train/images",
                "val: valid/images",
                "names:",
                "  0: bike",
                "  1: bus",
                "  2: car",
                "  3: other_vehicle",
                "  4: taxi",
                "  5: truck",
            ]
        ),
        encoding="utf-8",
    )

    trainer = RFDETRTrainer(
        model=torch.nn.Linear(1, 1),
        wrapper_model=SimpleNamespace(task="obb"),
        size="n",
        num_classes=80,
        data=str(data_yaml),
        epochs=1,
        batch=1,
        imgsz=384,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
    )

    assert trainer.config.num_classes == 6
    assert trainer._class_names == {
        0: "bike",
        1: "bus",
        2: "car",
        3: "other_vehicle",
        4: "taxi",
        5: "truck",
    }


def test_rfdetr_trainer_uses_explicit_coco_json_paths_for_obb(tmp_path):
    pytest.importorskip("pycocotools")
    from libreyolo.data.dataset import COCODataset
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    image_dir = tmp_path / "images" / "custom_train"
    ann_dir = tmp_path / "custom_annotations"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")
    (ann_dir / "train.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 10, "file_name": "sample.jpg", "width": 100, "height": 100}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10,
                        "category_id": 42,
                        "bbox": [10, 20, 40, 20],
                        "obb": [10, 20, 50, 20, 50, 40, 10, 40],
                        "area": 800,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 42, "name": "vehicle"}],
            }
        ),
        encoding="utf-8",
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: " + str(tmp_path).replace("\\", "/") + "\n"
        "train: images/custom_train\n"
        "val: images/custom_train\n"
        "annotations:\n"
        "  train: custom_annotations/train.json\n"
        "nc: 1\n"
        "names:\n"
        "  0: vehicle\n",
        encoding="utf-8",
    )
    trainer = RFDETRTrainer(
        model=torch.nn.Linear(1, 1),
        wrapper_model=SimpleNamespace(task="obb"),
        size="n",
        num_classes=80,
        data=str(data_yaml),
        epochs=1,
        batch=1,
        imgsz=384,
        workers=0,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
    )

    train_dataset = trainer._setup_data()

    assert isinstance(train_dataset.dataset, COCODataset)
    assert train_dataset.dataset.load_obb is True
    assert train_dataset.dataset.json_file == str(ann_dir / "train.json")
