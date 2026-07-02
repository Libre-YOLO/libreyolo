"""Unit tests for the LibrePIDNet semantic segmentation family."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


def _synthetic_pidnet_state(size: str, nc: int = 19) -> dict:
    planes = 32 if size == "s" else 64
    head_channels = 256 if size == "l" else 128
    state = {
        "conv1.0.weight": torch.zeros(planes, 3, 3, 3),
        "pag3.f_x.0.weight": torch.zeros(planes, planes * 2, 1, 1),
        "diff3.0.weight": torch.zeros(planes, planes * 4, 3, 3),
        "spp.scale0.2.weight": torch.zeros(1),
        "final_layer.conv2.weight": torch.zeros(nc, head_channels, 1, 1),
        "final_layer.conv2.bias": torch.zeros(nc),
    }
    if size == "l":
        state["layer3.3.conv1.weight"] = torch.zeros(1)
        state["spp.process1.2.weight"] = torch.zeros(1)
        state["dfm.conv.2.weight"] = torch.zeros(1)
    else:
        state["layer3.2.conv1.weight"] = torch.zeros(1)
        state["dfm.conv_p.0.weight"] = torch.zeros(1)
    return state


class TestPIDNetMetadata:
    def test_task_and_size_metadata(self):
        from libreyolo.models.pidnet.model import LibrePIDNet

        assert LibrePIDNet.FAMILY == "pidnet"
        assert LibrePIDNet.FILENAME_PREFIX == "LibrePIDNet"
        assert LibrePIDNet.SUPPORTED_TASKS == ("semantic",)
        assert LibrePIDNet.DEFAULT_TASK == "semantic"
        assert LibrePIDNet.REQUIRE_TASK_SUFFIX
        assert LibrePIDNet.semantic_resize_mode == "letterbox"
        assert LibrePIDNet.semantic_imgsz_divisor == 8
        assert set(LibrePIDNet.INPUT_SIZES) == {"s", "m", "l"}

    def test_registered_in_factory(self):
        from libreyolo.models import LibrePIDNet
        from libreyolo.models.base import BaseModel

        assert LibrePIDNet in BaseModel._registry

    @pytest.mark.parametrize("size", ["s", "m", "l"])
    def test_can_load_detects_size_and_classes(self, size):
        from libreyolo.models.pidnet.model import LibrePIDNet

        state = _synthetic_pidnet_state(size, nc=7)
        assert LibrePIDNet.can_load(state)
        assert LibrePIDNet.detect_size(state) == size
        assert LibrePIDNet.detect_nb_classes(state) == 7

    def test_can_load_handles_common_upstream_prefixes(self):
        from libreyolo.models.pidnet.model import LibrePIDNet

        state = {
            f"module.model.{key}": value
            for key, value in _synthetic_pidnet_state("s").items()
        }
        assert LibrePIDNet.can_load(state)
        assert LibrePIDNet.detect_size(state) == "s"

    def test_detect_size_handles_official_medium_extra_blocks(self):
        from libreyolo.models.pidnet.model import LibrePIDNet

        state = _synthetic_pidnet_state("m")
        state["layer3.3.conv1.weight"] = torch.zeros(256, 256, 3, 3)

        assert LibrePIDNet.detect_size(state) == "m"

    def test_can_load_rejects_dinov2_semantic_signature(self):
        from libreyolo.models.pidnet.model import LibrePIDNet

        state = {
            "backbone.encoder.proj.weight": torch.zeros(1),
            "predict.weight": torch.zeros(19, 8, 1, 1),
        }
        assert not LibrePIDNet.can_load(state)

    def test_filename_size_and_task_detection(self):
        from libreyolo.models.pidnet.model import LibrePIDNet

        assert LibrePIDNet.detect_size_from_filename("LibrePIDNets-sem.pt") == "s"
        assert LibrePIDNet.detect_task_from_filename("LibrePIDNets-sem.pt") == "semantic"
        assert LibrePIDNet.detect_size_from_filename("LibrePIDNets.pt") is None


class TestPIDNetForwardAndPredict:
    def test_net_forward_shape(self):
        from libreyolo.models.pidnet.nn import LibrePIDNetNet

        net = LibrePIDNetNet(size="s", num_classes=3).eval()
        with torch.no_grad():
            out = net(torch.rand(2, 3, 64, 64))
        assert out.shape == (2, 3, 8, 8)

    def test_normalization_buffers_not_in_state_dict(self):
        from libreyolo.models.pidnet.nn import LibrePIDNetNet

        keys = LibrePIDNetNet(size="s", num_classes=3).state_dict().keys()
        assert "pixel_mean" not in keys
        assert "pixel_std" not in keys

    def test_predict_returns_semantic_mask(self, tmp_path):
        from libreyolo.models.pidnet.model import LibrePIDNet

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (90, 45), color=(50, 90, 130)).save(img_path)

        model = LibrePIDNet(
            model_path=None, size="s", task="semantic", nb_classes=3, device="cpu"
        )
        result = model.predict(str(img_path), imgsz=64)[0]

        assert result.boxes is None
        assert result.masks is None
        assert result.semantic_mask is not None
        assert tuple(result.semantic_mask.data.shape) == (45, 90)

    def test_postprocess_crops_letterbox_padding(self):
        from libreyolo.models.pidnet.model import LibrePIDNet

        model = LibrePIDNet(
            model_path=None, size="s", task="semantic", nb_classes=2, device="cpu"
        )
        logits = torch.zeros(1, 2, 8, 8)
        logits[:, 0] = 10
        logits[:, 1, :4] = 20

        result = model._postprocess(
            logits,
            conf_thres=0.25,
            iou_thres=0.45,
            original_size=(80, 40),
            ratio=0.8,
            input_size=64,
        )

        assert result["semantic"].shape == (40, 80)
        assert torch.equal(result["semantic"].unique(), torch.tensor([1]))

    def test_predict_rejects_non_stride_imgsz(self, tmp_path):
        from libreyolo.models.pidnet.model import LibrePIDNet

        img_path = tmp_path / "img.jpg"
        Image.new("RGB", (64, 64), color=(10, 20, 30)).save(img_path)
        model = LibrePIDNet(
            model_path=None, size="s", task="semantic", nb_classes=2, device="cpu"
        )

        with pytest.raises(ValueError, match="divisible by 8"):
            model.predict(str(img_path), imgsz=62)

    def test_wrong_task_raises(self):
        from libreyolo.models.pidnet.model import LibrePIDNet

        with pytest.raises(ValueError, match="semantic"):
            LibrePIDNet(model_path=None, size="s", task="detect", device="cpu")

    def test_train_and_export_out_of_scope(self):
        from libreyolo.models.pidnet.model import LibrePIDNet

        model = LibrePIDNet(
            model_path=None, size="s", task="semantic", nb_classes=2, device="cpu"
        )
        with pytest.raises(NotImplementedError):
            model.train(data="cityscapes.yaml")
        with pytest.raises(NotImplementedError):
            model.export(format="onnx")


def test_checkpoint_round_trip_through_factory(tmp_path):
    from libreyolo import LibreYOLO
    from libreyolo.models.pidnet.model import LibrePIDNet
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    model = LibrePIDNet(
        model_path=None, size="s", task="semantic", nb_classes=3, device="cpu"
    )
    ckpt = wrap_libreyolo_checkpoint(
        model.model.state_dict(),
        model_family="pidnet",
        size="s",
        task="semantic",
        nc=3,
        names={0: "left", 1: "middle", 2: "right"},
        imgsz=1024,
    )
    ckpt_path = tmp_path / "LibrePIDNets-sem.pt"
    torch.save(ckpt, str(ckpt_path))

    loaded = LibreYOLO(str(ckpt_path), device="cpu")
    assert loaded.FAMILY == "pidnet"
    assert loaded.task == "semantic"
    assert loaded.size == "s"
    assert loaded.nb_classes == 3
    assert loaded.names == {0: "left", 1: "middle", 2: "right"}


def test_raw_pidnet_checkpoint_requires_converter(tmp_path):
    from libreyolo import LibreYOLO

    ckpt_path = tmp_path / "PIDNet_S_Cityscapes_val.pt"
    torch.save({"state_dict": _synthetic_pidnet_state("s")}, str(ckpt_path))

    with pytest.raises(ValueError, match="convert_pidnet_weights"):
        LibreYOLO(str(ckpt_path), device="cpu")
