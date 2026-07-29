"""Offline coverage for the LibreEdgeTAM adapter."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.sam.edgetam import LibreEdgeTAM, _load_edgetam_video_config
from libreyolo.models.sam.model import _ALIASES

pytestmark = [pytest.mark.unit, pytest.mark.sam]


class _FakeEdgeTamModel(torch.nn.Module):
    """Deterministic local stand-in for EdgeTAM's image and mask decoders."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.embedding_calls = 0
        self.cached_forward_calls = 0

    def get_image_embeddings(self, pixel_values):
        self.embedding_calls += 1
        feature = pixel_values.mean(dim=1, keepdim=True)
        return [feature, feature[..., ::2, ::2]]

    def forward(
        self,
        *,
        pixel_values=None,
        image_embeddings=None,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        multimask_output=False,
    ):
        assert (pixel_values is None) != (image_embeddings is None)
        if image_embeddings is not None:
            assert isinstance(image_embeddings, list)
            self.cached_forward_calls += 1

        prompt = input_points if input_points is not None else input_boxes
        assert prompt is not None
        if input_points is not None:
            assert input_labels is not None
        prompt_count = prompt.shape[1]
        mask_count = 3 if multimask_output else 1
        pred_masks = torch.zeros((1, prompt_count, mask_count, 4, 4))
        iou_scores = torch.linspace(0.6, 0.8, mask_count).repeat(1, prompt_count, 1)
        return SimpleNamespace(pred_masks=pred_masks, iou_scores=iou_scores)


class _FakeEdgeTamProcessor:
    """Processor with EdgeTAM/SAM-2's two-argument mask postprocess API."""

    def __init__(self):
        self.postprocess_calls = 0

    def __call__(self, images, return_tensors="pt", **kwargs):
        assert return_tensors == "pt"
        width, height = images.size
        enc = {
            "pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            "original_sizes": torch.tensor([[height, width]], dtype=torch.long),
            # Kept deliberately: EdgeTAM must not pass this SAM-1-only value.
            "reshaped_input_sizes": torch.tensor([[8, 8]], dtype=torch.long),
        }
        if "input_points" in kwargs:
            enc["input_points"] = torch.tensor(
                kwargs["input_points"], dtype=torch.float32
            )
            enc["input_labels"] = torch.tensor(kwargs["input_labels"], dtype=torch.long)
        if "input_boxes" in kwargs:
            enc["input_boxes"] = torch.tensor(
                kwargs["input_boxes"], dtype=torch.float32
            )
        return enc

    def post_process_masks(self, masks, original_sizes):
        self.postprocess_calls += 1
        height, width = [int(value) for value in original_sizes[0]]
        output = torch.zeros(
            (masks.shape[1], masks.shape[2], height, width), dtype=torch.bool
        )
        output[
            ...,
            height // 4 : max(height // 4 + 1, 3 * height // 4),
            width // 4 : max(width // 4 + 1, 3 * width // 4),
        ] = True
        return [output]


def _bare_edgetam() -> LibreEdgeTAM:
    model = object.__new__(LibreEdgeTAM)
    model.model = _FakeEdgeTamModel().eval()
    model.processor = _FakeEdgeTamProcessor()
    model.device = torch.device("cpu")
    model._model_dtype = torch.float32
    model._default_multimask = False
    model.names = {0: "object"}
    model.size = "edge"
    model.INPUT_SIZES = {"edge": 8}
    model.task = "segment"
    model._clear_image_state()
    return model


def test_edgetam_public_metadata_and_export():
    from libreyolo import LibreEdgeTAM as Exported

    assert Exported is LibreEdgeTAM
    assert LibreEdgeTAM.FAMILY == "edgetam"
    assert LibreEdgeTAM.FILENAME_PREFIX == "LibreEdgeTAM"
    assert LibreEdgeTAM.HF_REPOS == {"edge": "LibreYOLO/LibreEdgeTAM"}
    assert LibreEdgeTAM.INPUT_SIZES == {"edge": 1024}
    assert LibreEdgeTAM.SUPPORTED_TASKS == ("segment",)
    assert "*.pt" in LibreEdgeTAM.SNAPSHOT_IGNORE_PATTERNS


def test_aliases_resolve_to_edgetam_without_download():
    expected = (LibreEdgeTAM, "edge")
    assert _ALIASES["edgetam"] == expected
    assert _ALIASES["edge-tam"] == expected
    assert _ALIASES["edgetam-edge"] == expected


def test_invalid_size_lists_valid_key_before_weight_loading():
    with pytest.raises(ValueError, match=r"Must be one of: edge"):
        LibreEdgeTAM("tiny", device="cpu")


def test_load_pretrained_builds_embedded_config_without_mutating_global_flags(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """{
  "model_type": "edgetam_video",
  "vision_config": {
    "model_type": "edgetam_vision_model",
    "backbone_config": {
      "model_type": "timm_wrapper",
      "architecture": "repvit_m1"
    }
  }
}
""",
        encoding="utf-8",
    )

    class FakeVideoConfig:
        has_no_defaults_at_init = False

        def __init__(self, *, vision_config, **kwargs):
            self.vision_config = vision_config
            self.kwargs = kwargs

    class FakeVisionConfig:
        has_no_defaults_at_init = False

        def __init__(self, *, backbone_config, **kwargs):
            self.backbone_config = backbone_config
            self.kwargs = kwargs

    class FakeTimmWrapperConfig:
        has_no_defaults_at_init = False

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModel:
        loaded_config = None

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == str(tmp_path)
            config = kwargs.pop("config")
            cls.loaded_config = config
            assert isinstance(config, FakeVideoConfig)
            assert config.has_no_defaults_at_init is True
            assert config.kwargs == {"model_type": "edgetam_video"}
            assert isinstance(config.vision_config, FakeVisionConfig)
            assert config.vision_config.has_no_defaults_at_init is True
            assert config.vision_config.kwargs == {"model_type": "edgetam_vision_model"}
            backbone = config.vision_config.backbone_config
            assert isinstance(backbone, FakeTimmWrapperConfig)
            assert backbone.kwargs == {
                "model_type": "timm_wrapper",
                "architecture": "repvit_m1",
            }
            assert kwargs == {"dtype": torch.float32, "local_files_only": True}
            return SimpleNamespace(config=config)

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == str(tmp_path)
            assert kwargs == {"local_files_only": True}
            return "processor"

    transformers = ModuleType("transformers")
    transformers.EdgeTamModel = FakeModel
    transformers.EdgeTamVideoConfig = FakeVideoConfig
    transformers.Sam2Processor = FakeProcessor
    configuration = ModuleType("transformers.models.edgetam.configuration_edgetam")
    configuration.EdgeTamVisionConfig = FakeVisionConfig
    timm_configuration = ModuleType(
        "transformers.models.timm_wrapper.configuration_timm_wrapper"
    )
    timm_configuration.TimmWrapperConfig = FakeTimmWrapperConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.edgetam.configuration_edgetam",
        configuration,
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.timm_wrapper.configuration_timm_wrapper",
        timm_configuration,
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        configs = list(executor.map(_load_edgetam_video_config, [config_path] * 8))

    assert all(config.has_no_defaults_at_init is True for config in configs)
    assert all(
        config.vision_config.has_no_defaults_at_init is True for config in configs
    )
    assert FakeVideoConfig.has_no_defaults_at_init is False
    assert FakeVisionConfig.has_no_defaults_at_init is False
    assert FakeTimmWrapperConfig.has_no_defaults_at_init is False

    model = object.__new__(LibreEdgeTAM)
    loaded = model._load_pretrained(str(tmp_path))

    assert loaded[1] == "processor"
    assert isinstance(loaded[0], SimpleNamespace)
    assert "has_no_defaults_at_init" not in FakeModel.loaded_config.__dict__
    assert (
        "has_no_defaults_at_init" not in FakeModel.loaded_config.vision_config.__dict__
    )
    assert FakeVideoConfig.has_no_defaults_at_init is False
    assert FakeVisionConfig.has_no_defaults_at_init is False
    assert FakeTimmWrapperConfig.has_no_defaults_at_init is False


def test_move_embeddings_maps_edgetam_feature_list():
    model = object.__new__(LibreEdgeTAM)
    embeddings = [torch.zeros(1), torch.ones(1)]

    moved = model._move_embeddings(embeddings, torch.device("meta"))

    assert isinstance(moved, list)
    assert [embedding.device.type for embedding in moved] == ["meta", "meta"]


def test_post_process_uses_sam2_processor_signature():
    model = object.__new__(LibreEdgeTAM)
    model.processor = _FakeEdgeTamProcessor()
    enc = {
        "original_sizes": torch.tensor([[12, 16]]),
        "reshaped_input_sizes": torch.tensor([[8, 8]]),
    }

    output = model._post_process_masks(torch.zeros((1, 1, 1, 4, 4)), enc)

    assert model.processor.postprocess_calls == 1
    assert output[0].shape == (1, 1, 12, 16)


def test_encode_replaces_processor_pixels_with_official_square_transform():
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as vision_functional

    model = _bare_edgetam()
    image = Image.new("RGB", (13, 9), color=(10, 80, 240))

    enc = model._encode(image, points=[[[6, 4]]], labels=[[1]])

    expected = vision_functional.normalize(
        vision_functional.resize(
            vision_functional.to_tensor(image),
            [8, 8],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ).unsqueeze(0)
    assert torch.equal(enc["pixel_values"], expected)
    expected_points = torch.tensor([[[6.0, 4.0]]], dtype=torch.float32)
    expected_points[..., 0] /= image.width
    expected_points[..., 1] /= image.height
    expected_points *= 8
    assert torch.equal(enc["input_points"], expected_points.unsqueeze(0))


def test_reusing_one_prompt_array_does_not_drift_between_predictions():
    """The prompt scaling is in-place, so caller arrays must never be aliased.

    Prompt normalization already copies (it goes via ``tolist()``), and
    ``_encode`` clones on top of that. Assert the user-visible property rather
    than either mechanism: predicting twice with the same array must give the
    same mask, and must leave the caller's array untouched.
    """
    model = _bare_edgetam()
    image = Image.new("RGB", (20, 16), color="white")
    points = np.array([[10.0, 8.0]], dtype=np.float32)
    boxes = np.array([[4.0, 3.0, 16.0, 13.0]], dtype=np.float32)

    first = model.predict(image, points=points, labels=[1])
    second = model.predict(image, points=points, labels=[1])
    box_first = model.predict(image, bboxes=boxes)
    box_second = model.predict(image, bboxes=boxes)

    assert np.array_equal(points, np.array([[10.0, 8.0]], dtype=np.float32))
    assert np.array_equal(boxes, np.array([[4.0, 3.0, 16.0, 13.0]], dtype=np.float32))
    assert torch.equal(first.masks.data, second.masks.data)
    assert torch.equal(box_first.masks.data, box_second.masks.data)


def test_segment_everything_grid_runs_through_the_edgetam_transform():
    """The grid path slices prompt tensors that _encode rebuilds; cover it."""
    model = _bare_edgetam()
    image = Image.new("RGB", (20, 16), color="white")

    result = model.predict(image, points_per_side=2, conf=0.0)

    assert result.masks is not None
    assert len(result.masks) >= 1
    assert model.model.embedding_calls == 1  # encoded once, then prompted
    assert result.masks.data.shape[-2:] == (16, 20)


def test_fake_edgetam_point_prompt_returns_result():
    model = _bare_edgetam()
    image = Image.new("RGB", (20, 16), color="white")

    result = model.predict(image, points=[10, 8], labels=[1])

    assert result.masks is not None
    assert result.boxes is not None
    assert result.masks.data.shape == (1, 16, 20)
    assert result.boxes.xyxy.shape == (1, 4)
    assert len(result.boxes.conf) == len(result.masks) == 1
    assert result.names[0] == "object"
    assert model.task == "segment"


def test_fake_edgetam_box_prompt_multimask_returns_three_results():
    model = _bare_edgetam()
    image = Image.new("RGB", (20, 16), color="white")

    result = model.predict(image, bboxes=[4, 3, 16, 13], multimask=True)

    assert result.masks is not None
    assert result.boxes is not None
    assert len(result.masks) == 3
    assert len(result.boxes.conf) == 3


def test_interactive_session_caches_list_embeddings_and_reset_clears_state():
    model = _bare_edgetam()
    image = Image.new("RGB", (20, 16), color="white")

    assert model.set_image(image) is model
    cached = model._image_embeddings
    assert isinstance(cached, list)
    assert model.model.embedding_calls == 1

    point_result = model.predict(points=[10, 8], labels=[1])
    box_result = model.predict(bboxes=[4, 3, 16, 13])

    assert point_result.masks is not None and box_result.masks is not None
    assert model._image_embeddings is cached
    assert model.model.embedding_calls == 1
    assert model.model.cached_forward_calls == 2

    assert model.reset_image() is model
    assert model._image is None
    assert model._image_path is None
    assert model._image_embeddings is None
    with pytest.raises(RuntimeError, match="No image set"):
        model.predict(points=[10, 8], labels=[1])


def test_edgetam_scope_methods_raise():
    model = object.__new__(LibreEdgeTAM)

    with pytest.raises(NotImplementedError, match="video tracking"):
        model.track("video.mp4")
    with pytest.raises(NotImplementedError, match="Training is out of scope"):
        model.train("data.yaml")
    with pytest.raises(NotImplementedError, match="validation is not supported"):
        model.val("data.yaml")
    with pytest.raises(NotImplementedError, match="format='coreml'"):
        model.export()
