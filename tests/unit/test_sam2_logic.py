"""Offline coverage for the LibreSAM2 adapter."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from libreyolo.models.sam.model import _ALIASES
from libreyolo.models.sam.sam2 import LibreSAM2

pytestmark = [pytest.mark.unit, pytest.mark.sam]


def _tiny_sam2_model():
    pytest.importorskip("transformers", reason="LibreSAM2 requires transformers")
    from transformers import Sam2Config, Sam2Model
    from transformers.models.sam2.configuration_sam2 import (
        Sam2HieraDetConfig,
        Sam2MaskDecoderConfig,
        Sam2PromptEncoderConfig,
        Sam2VisionConfig,
    )

    backbone = Sam2HieraDetConfig(
        hidden_size=8,
        num_attention_heads=1,
        image_size=[32, 32],
        patch_kernel_size=[3, 3],
        patch_stride=[4, 4],
        patch_padding=[1, 1],
        query_stride=[2, 2],
        window_positional_embedding_background_size=[2, 2],
        num_query_pool_stages=3,
        blocks_per_stage=[1, 1, 1, 1],
        embed_dim_per_stage=[8, 16, 32, 64],
        num_attention_heads_per_stage=[1, 1, 2, 4],
        window_size_per_stage=[2, 2, 2, 2],
        global_attention_blocks=[],
        mlp_ratio=2.0,
    )
    vision = Sam2VisionConfig(
        backbone_config=backbone,
        backbone_channel_list=[64, 32, 16, 8],
        backbone_feature_sizes=[[8, 8], [4, 4], [2, 2]],
        fpn_hidden_size=8,
        num_feature_levels=3,
        fpn_top_down_levels=[1, 2],
    )
    prompt = Sam2PromptEncoderConfig(
        hidden_size=8,
        image_size=32,
        patch_size=16,
        mask_input_channels=4,
    )
    mask = Sam2MaskDecoderConfig(
        hidden_size=8,
        mlp_dim=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        attention_downsample_rate=1,
        iou_head_hidden_dim=8,
    )
    return Sam2Model(
        Sam2Config(
            vision_config=vision,
            prompt_encoder_config=prompt,
            mask_decoder_config=mask,
        )
    ).eval()


class _TinyProcessor:
    def __call__(self, images, return_tensors="pt", **kwargs):
        assert return_tensors == "pt"
        width, height = images.size
        enc = {
            "pixel_values": torch.zeros((1, 3, 32, 32), dtype=torch.float32),
            "original_sizes": torch.tensor([[height, width]], dtype=torch.long),
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
        height, width = [int(v) for v in original_sizes[0]]
        out = torch.zeros(
            (masks.shape[1], masks.shape[2], height, width),
            dtype=torch.bool,
        )
        out[
            ...,
            height // 4 : max(height // 4 + 1, 3 * height // 4),
            width // 4 : max(width // 4 + 1, 3 * width // 4),
        ] = True
        return [out]


def _bare_sam2():
    model = object.__new__(LibreSAM2)
    model.model = _tiny_sam2_model()
    model.processor = _TinyProcessor()
    model.device = torch.device("cpu")
    model._model_dtype = torch.float32
    model._default_multimask = False
    model.names = {0: "object"}
    model.size = "tiny"
    model.task = "segment"
    model._clear_image_state()
    return model


def test_aliases_resolve_to_sam2_without_download():
    assert _ALIASES["sam2-tiny"] == (LibreSAM2, "tiny")
    assert _ALIASES["sam2_t"] == (LibreSAM2, "tiny")
    assert _ALIASES["sam2_bp"] == (LibreSAM2, "base-plus")
    assert _ALIASES["base"][0] is not LibreSAM2


def test_sam2_uses_libreyolo_mirrors_without_downloading_raw_pt():
    assert LibreSAM2.HF_REPOS == {
        "tiny": "LibreYOLO/LibreSAM2tiny",
        "small": "LibreYOLO/LibreSAM2small",
        "base-plus": "LibreYOLO/LibreSAM2base-plus",
        "large": "LibreYOLO/LibreSAM2large",
    }
    assert "*.pt" in LibreSAM2.SNAPSHOT_IGNORE_PATTERNS


def test_invalid_size_raises_before_weight_loading():
    with pytest.raises(ValueError, match="Invalid size"):
        LibreSAM2("medium", device="cpu")


def test_move_embeddings_maps_sam2_feature_list():
    model = object.__new__(LibreSAM2)
    model.device = torch.device("meta")
    model.model = torch.nn.Identity()
    model._image_embeddings = [torch.zeros(1), torch.ones(1)]

    model._set_device("cpu")

    assert isinstance(model._image_embeddings, list)
    assert [emb.device.type for emb in model._image_embeddings] == ["cpu", "cpu"]


def test_post_process_uses_sam2_processor_signature():
    class Processor:
        def __init__(self):
            self.called = False

        def post_process_masks(self, masks, original_sizes):
            self.called = True
            assert tuple(original_sizes.shape) == (1, 2)
            return [masks[0]]

    model = object.__new__(LibreSAM2)
    model.processor = Processor()
    enc = {
        "original_sizes": torch.tensor([[12, 16]]),
        "reshaped_input_sizes": torch.tensor([[32, 32]]),
    }

    out = model._post_process_masks(torch.zeros((1, 1, 1, 4, 4)), enc)

    assert model.processor.called is True
    assert len(out) == 1


def test_random_sam2_point_prompt_returns_results():
    model = _bare_sam2()
    image = Image.new("RGB", (20, 16), color="white")

    result = model.predict(image, points=[10, 8], labels=[1])

    assert result.masks is not None
    assert len(result.masks) == 1
    assert result.boxes.xyxy.shape == (1, 4)
    assert result.names[0] == "object"
    assert model.task == "segment"


def test_random_sam2_multimask_returns_three_masks():
    model = _bare_sam2()
    image = Image.new("RGB", (20, 16), color="white")

    result = model.predict(image, points=[10, 8], labels=[1], multimask=True)

    assert result.masks is not None
    assert len(result.masks) == 3
    assert len(result.boxes.conf) == 3


def test_interactive_session_caches_list_embeddings():
    model = _bare_sam2()
    image = Image.new("RGB", (20, 16), color="white")

    model.set_image(image)
    try:
        assert isinstance(model._image_embeddings, list)
        result = model.predict(points=[10, 8], labels=[1])
        assert result.masks is not None
    finally:
        model.reset_image()


def test_sam2_scope_methods_raise():
    model = object.__new__(LibreSAM2)

    with pytest.raises(NotImplementedError, match="video tracking"):
        model.track("video.mp4")
    with pytest.raises(NotImplementedError):
        model.train("data.yaml")
    with pytest.raises(NotImplementedError):
        model.val("data.yaml")
    with pytest.raises(NotImplementedError):
        model.export()
