"""Unit coverage for the inference-only LibreDeepLabv3 family."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit

ALL_SIZES = ("r50", "r101", "mv3")


def _synthetic_state(size: str, nc: int = 21, *, auxiliary: bool = False) -> dict:
    in_channels = 960 if size == "mv3" else 2048
    state = {
        "classifier.0.convs.0.0.weight": torch.zeros(1, in_channels, 1, 1),
        "classifier.0.convs.1.0.weight": torch.zeros(1),
        "classifier.0.convs.4.1.weight": torch.zeros(1),
        "classifier.0.project.0.weight": torch.zeros(1),
        "classifier.4.weight": torch.zeros(nc, 1, 1, 1),
    }
    if size == "mv3":
        state["backbone.0.0.weight"] = torch.zeros(1)
    else:
        state["backbone.conv1.weight"] = torch.zeros(1)
        depth = 6 if size == "r50" else 23
        for index in range(depth):
            state[f"backbone.layer3.{index}.conv1.weight"] = torch.zeros(1)
    if auxiliary:
        state["aux_classifier.0.weight"] = torch.zeros(1)
        state["aux_classifier.4.weight"] = torch.zeros(1)
    return state


class TestDeepLabv3Metadata:
    def test_family_contract(self):
        from libreyolo.models.deeplabv3.model import LibreDeepLabv3

        assert LibreDeepLabv3.FAMILY == "deeplabv3"
        assert LibreDeepLabv3.FILENAME_PREFIX == "LibreDeepLabv3"
        assert LibreDeepLabv3.SUPPORTED_TASKS == ("semantic",)
        assert LibreDeepLabv3.DEFAULT_TASK == "semantic"
        assert LibreDeepLabv3.REQUIRE_TASK_SUFFIX
        assert LibreDeepLabv3.INPUT_SIZES == {size: 520 for size in ALL_SIZES}
        assert LibreDeepLabv3.TRAIN_CONFIG is None
        assert LibreDeepLabv3.semantic_resize_mode == "stretch"

    def test_registered_in_factory(self):
        from libreyolo.models import LibreDeepLabv3
        from libreyolo.models.base import BaseModel

        assert LibreDeepLabv3 in BaseModel._registry

    @pytest.mark.parametrize("size", ALL_SIZES)
    def test_can_load_detects_size_and_classes(self, size):
        from libreyolo.models.deeplabv3.model import LibreDeepLabv3

        state = _synthetic_state(size, nc=7)
        assert LibreDeepLabv3.can_load(state)
        assert LibreDeepLabv3.detect_size(state) == size
        assert LibreDeepLabv3.detect_nb_classes(state) == 7

    @pytest.mark.parametrize("size", ALL_SIZES)
    def test_canonical_filename_detection(self, size):
        from libreyolo.models.deeplabv3.model import LibreDeepLabv3

        filename = f"LibreDeepLabv3{size}-sem.pt"
        assert LibreDeepLabv3.detect_size_from_filename(filename) == size
        assert LibreDeepLabv3.detect_task_from_filename(filename) == "semantic"
        assert (
            LibreDeepLabv3.detect_size_from_filename(f"LibreDeepLabv3{size}.pt") is None
        )

    def test_download_url_uses_canonical_repository(self):
        from libreyolo.models.deeplabv3.model import LibreDeepLabv3

        assert LibreDeepLabv3.get_download_url("LibreDeepLabv3mv3-sem.pt") == (
            "https://huggingface.co/LibreYOLO/LibreDeepLabv3mv3-sem/resolve/"
            "main/LibreDeepLabv3mv3-sem.pt"
        )

    def test_parity_validated_export_matrix(self):
        from libreyolo.export.support import get_support

        for format_name in ("onnx", "torchscript", "openvino", "tensorrt"):
            support = get_support("deeplabv3", "semantic", format_name)
            assert support.tier == "validated"
            assert support.constraint
        for format_name in ("executorch", "ncnn", "tflite", "coreml", "coreai"):
            assert get_support("deeplabv3", "semantic", format_name).tier == ("blocked")


class TestDeepLabv3Conversion:
    def test_converter_requires_auxiliary_upstream_fingerprint(self):
        from libreyolo.models.deeplabv3.convert import (
            convert_upstream_deeplabv3_state_dict,
        )

        native = _synthetic_state("r50")
        upstream = _synthetic_state("r50", auxiliary=True)
        converted = convert_upstream_deeplabv3_state_dict(upstream)

        assert convert_upstream_deeplabv3_state_dict(native) is None
        assert converted is not None
        assert converted.keys() == native.keys()

    def test_converter_strips_common_wrappers_and_only_aux_head(self):
        from libreyolo.models.deeplabv3.convert import (
            convert_upstream_deeplabv3_state_dict,
        )

        wrapped = {
            f"module.model.{key}": value
            for key, value in _synthetic_state("mv3", auxiliary=True).items()
        }
        converted = convert_upstream_deeplabv3_state_dict(wrapped)

        assert converted is not None
        assert "classifier.4.weight" in converted
        assert all(not key.startswith("aux_classifier.") for key in converted)

    def test_global_autoconverter_has_one_claim(self):
        from libreyolo.models.autoconvert import _claim_upstream_state

        claims = _claim_upstream_state(
            _synthetic_state("r101", auxiliary=True),
            existing_libreyolo=False,
        )
        assert [cls.FAMILY for cls, _state in claims] == ["deeplabv3"]

    def test_other_families_reject_native_signature(self):
        from libreyolo.models.base.model import BaseModel
        from libreyolo.models.deeplabv3.model import LibreDeepLabv3

        state = _synthetic_state("mv3")
        for cls in BaseModel._registry:
            if cls is LibreDeepLabv3:
                continue
            assert not cls.can_load(state), (
                f"{cls.__name__} incorrectly claims DeepLabv3 weights"
            )


class TestDeepLabv3Inference:
    def test_size_configs_encode_the_two_official_output_strides(self):
        from libreyolo.models.deeplabv3.nn import SIZE_CONFIGS

        assert SIZE_CONFIGS == {
            "r50": {"backbone": "resnet50", "output_stride": 8},
            "r101": {"backbone": "resnet101", "output_stride": 8},
            "mv3": {"backbone": "mobilenet_v3_large", "output_stride": 16},
        }

    def test_forward_upsamples_backbone_logits_to_input_canvas(self):
        import torch.nn as nn

        from libreyolo.models.deeplabv3.nn import LibreDeepLabv3Net

        class ToyBackbone(nn.Module):
            def forward(self, value):
                return {"out": value[..., ::2, ::2]}

        model = LibreDeepLabv3Net.__new__(LibreDeepLabv3Net)
        nn.Module.__init__(model)
        model.backbone = ToyBackbone()
        model.classifier = nn.Conv2d(3, 4, kernel_size=1)
        with torch.inference_mode():
            output = model(torch.rand(1, 3, 32, 48))
        assert output.shape == (1, 4, 32, 48)

    def test_preprocess_stretches_and_normalizes_once(self):
        from libreyolo.models.deeplabv3.utils import (
            IMAGENET_MEAN,
            IMAGENET_STD,
            preprocess_numpy,
        )

        image = np.full((6, 8, 3), (255, 128, 0), dtype=np.uint8)
        chw, ratio = preprocess_numpy(image, (10, 20))
        expected = (np.asarray((1.0, 128 / 255.0, 0.0)) - IMAGENET_MEAN) / IMAGENET_STD

        assert chw.shape == (3, 10, 20)
        assert ratio == 1.0
        np.testing.assert_allclose(chw[:, 0, 0], expected, rtol=0, atol=1e-6)

    def test_postprocess_resizes_logits_before_argmax(self):
        from libreyolo.postprocess.deeplabv3 import postprocess, semantic_logits

        logits = torch.zeros(1, 3, 4, 8)
        logits[:, 2] = 5
        resized = semantic_logits({"out": logits}, (16, 10))
        result = postprocess([logits], (16, 10))["semantic"]

        assert resized.shape == (1, 3, 10, 16)
        assert result.shape == (10, 16)
        assert torch.equal(result.unique(), torch.tensor([2]))

    def test_wrapper_predict_and_flip_tta_return_original_shape(
        self, tmp_path, monkeypatch
    ):
        import torch.nn as nn

        from libreyolo.models.deeplabv3.model import LibreDeepLabv3

        monkeypatch.setattr(
            LibreDeepLabv3,
            "_init_model",
            lambda self: nn.Conv2d(3, self.nb_classes, kernel_size=1),
        )
        image_path = tmp_path / "image.jpg"
        Image.new("RGB", (90, 45), color=(50, 90, 130)).save(image_path)
        model = LibreDeepLabv3(
            model_path=None,
            size="mv3",
            task="semantic",
            nb_classes=3,
            device="cpu",
        )

        def deterministic_forward(input_tensor):
            output = torch.zeros(
                input_tensor.shape[0],
                3,
                input_tensor.shape[-2],
                input_tensor.shape[-1],
                device=input_tensor.device,
            )
            output[:, 1] = 1.0
            return output

        model._forward = deterministic_forward

        for augment in (False, True):
            result = model.predict(str(image_path), imgsz=32, augment=augment)
            assert result.boxes is None
            assert result.semantic_mask is not None
            assert tuple(result.semantic_mask.data.shape) == (45, 90)

    def test_wrong_task_and_training_raise(self, monkeypatch):
        import torch.nn as nn

        from libreyolo.models.deeplabv3.model import LibreDeepLabv3

        with pytest.raises(ValueError, match="semantic"):
            LibreDeepLabv3(task="detect", device="cpu")
        monkeypatch.setattr(
            LibreDeepLabv3,
            "_init_model",
            lambda self: nn.Conv2d(3, self.nb_classes, kernel_size=1),
        )
        model = LibreDeepLabv3(size="mv3", nb_classes=3, device="cpu")
        with pytest.raises(NotImplementedError, match="inference-only"):
            model.train(data="voc.yaml")
