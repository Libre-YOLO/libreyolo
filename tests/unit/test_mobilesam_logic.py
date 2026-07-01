"""Offline tests for the native MobileSAM port."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from libreyolo.models.mobilesam.model import LibreMobileSAM, MobileSAMNetwork
from libreyolo.models.mobilesam.preprocess import encode_image_and_prompts
from libreyolo.models.sam.base import _SNAPSHOT_COMPLETE_MARKER
from libreyolo.models.sam.model import _ALIASES, _MOBILE_SAM
from libreyolo.utils.serialization import (
    CheckpointMetadataError,
    wrap_libreyolo_checkpoint,
)

pytestmark = [pytest.mark.unit, pytest.mark.sam]


def _bare_mobilesam():
    model = object.__new__(LibreMobileSAM)
    model.model = MobileSAMNetwork().eval()
    model.model.mask_threshold = -float("inf")
    model.processor = None
    model.device = torch.device("cpu")
    model._model_dtype = torch.float32
    model._default_multimask = False
    model.names = {0: "object"}
    model.size = "tiny"
    model.task = "segment"
    model._clear_image_state()
    return model


def test_top_level_export_resolves_lazily():
    from libreyolo import LibreMobileSAM as Exported

    assert Exported is LibreMobileSAM


def test_aliases_resolve_to_mobilesam_without_download():
    assert _ALIASES["mobilesam"] == (_MOBILE_SAM, "tiny")
    assert _ALIASES["mobile-sam-tiny"] == (_MOBILE_SAM, "tiny")


def test_invalid_size_raises_before_weight_loading():
    with pytest.raises(ValueError, match="Invalid size"):
        LibreMobileSAM("small", device="cpu")


def test_resize_longest_side_transforms_image_points_and_boxes():
    image = Image.new("RGB", (32, 24), color="white")
    enc = encode_image_and_prompts(
        image,
        target_length=1024,
        points=[[[16, 12]]],
        labels=[[1]],
        boxes=[[8, 6, 24, 18]],
    )

    assert enc["pixel_values"].shape == (1, 3, 768, 1024)
    assert enc["reshaped_input_sizes"].tolist() == [[768, 1024]]
    assert enc["original_sizes"].tolist() == [[24, 32]]
    assert enc["input_points"].tolist() == [[[[512.0, 384.0]]]]
    assert enc["input_boxes"].tolist() == [[[256.0, 192.0, 768.0, 576.0]]]


def test_tinyvit_encoder_shape():
    model = MobileSAMNetwork().eval()

    with torch.no_grad():
        embedding = model.image_encoder(torch.zeros((1, 3, 1024, 1024)))

    assert embedding.shape == (1, 256, 64, 64)


def test_point_prompt_returns_results():
    model = _bare_mobilesam()
    image = Image.new("RGB", (32, 24), color="white")

    result = model.predict(image, points=[16, 12], labels=[1])

    assert result.masks is not None
    assert len(result.masks) == 1
    assert result.masks.data.shape[-2:] == (24, 32)
    assert result.boxes.xyxy.shape == (1, 4)
    assert result.names[0] == "object"


def test_box_prompt_and_multimask_return_results():
    model = _bare_mobilesam()
    image = Image.new("RGB", (32, 24), color="white")

    result = model.predict(
        image,
        bboxes=[8, 6, 24, 18],
        multimask=True,
    )

    assert result.masks is not None
    assert len(result.masks) == 3
    assert len(result.boxes.conf) == 3


def test_interactive_session_caches_embedding_tensor():
    model = _bare_mobilesam()
    image = Image.new("RGB", (32, 24), color="white")

    model.set_image(image)
    try:
        assert isinstance(model._image_embeddings, torch.Tensor)
        assert model._image_embeddings.shape == (1, 256, 64, 64)
        result = model.predict(points=[16, 12], labels=[1])
        assert result.masks is not None
    finally:
        model.reset_image()


def test_snapshot_complete_requires_marker_and_weight(tmp_path):
    assert LibreMobileSAM._snapshot_complete(tmp_path) is False
    (tmp_path / _SNAPSHOT_COMPLETE_MARKER).write_text("{}", encoding="utf-8")
    assert LibreMobileSAM._snapshot_complete(tmp_path) is False
    (tmp_path / LibreMobileSAM.WEIGHT_FILE).write_bytes(b"placeholder")
    assert LibreMobileSAM._snapshot_complete(tmp_path) is True


def test_init_model_loads_wrapped_checkpoint(tmp_path):
    wrapped = wrap_libreyolo_checkpoint(
        MobileSAMNetwork().state_dict(),
        model_family="mobilesam",
        size="tiny",
        task="segment",
        nc=1,
        names={0: "object"},
        imgsz=1024,
    )
    torch.save(wrapped, tmp_path / LibreMobileSAM.WEIGHT_FILE)
    model = object.__new__(LibreMobileSAM)
    model.size = "tiny"
    model._ensure_weights = lambda: str(tmp_path)

    loaded = model._init_model()

    assert isinstance(loaded, MobileSAMNetwork)
    assert model._model_dtype == torch.float32


def test_init_model_rejects_raw_state_dict(tmp_path):
    torch.save(MobileSAMNetwork().state_dict(), tmp_path / LibreMobileSAM.WEIGHT_FILE)
    model = object.__new__(LibreMobileSAM)
    model.size = "tiny"
    model._ensure_weights = lambda: str(tmp_path)

    with pytest.raises(CheckpointMetadataError):
        model._init_model()
