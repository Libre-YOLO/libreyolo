"""Offline coverage for the LibreSAM3 adapter."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from libreyolo.models.sam.model import LibreSAM1, _ALIASES
from libreyolo.models.sam.sam3 import LibreSAM3

pytestmark = [pytest.mark.unit, pytest.mark.sam]


class _PCSProcessor:
    def __init__(self):
        self.calls = 0

    def __call__(self, *, images, text, return_tensors):
        self.calls += 1
        assert images.size == (10, 8)
        assert text == "person"
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.zeros((1, 3, 4, 4)),
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "original_sizes": torch.tensor([[8, 10]]),
        }

    def post_process_instance_segmentation(self, outputs, **kwargs):
        assert outputs.ok
        assert kwargs == {
            "threshold": 0.0,
            "mask_threshold": 0.5,
            "target_sizes": [[8, 10]],
        }
        masks = torch.zeros((3, 8, 10), dtype=torch.bool)
        masks[0, :2, :2] = True
        masks[1, 2:5, 2:5] = True
        masks[2, 5:8, 5:10] = True
        return [{"masks": masks, "scores": torch.tensor([0.2, 0.9, 0.7])}]


class _PCSModel(torch.nn.Module):
    def forward(self, **inputs):
        assert inputs["pixel_values"].device.type == "cpu"
        return SimpleNamespace(ok=True)


def _bare_sam3():
    model = object.__new__(LibreSAM3)
    model.model = torch.nn.Identity()
    model.processor = SimpleNamespace()
    model.device = torch.device("cpu")
    model._model_dtype = torch.float32
    model._default_multimask = False
    model.names = {0: "object"}
    model.size = "large"
    model.task = "segment"
    model._snapshot_dir = "unused"
    model._pcs_model = _PCSModel()
    model._pcs_processor = _PCSProcessor()
    model._clear_image_state()
    return model


def test_aliases_resolve_to_sam3_without_download():
    assert _ALIASES["sam3"] == (LibreSAM3, "large")
    assert _ALIASES["sam-3"] == (LibreSAM3, "large")
    assert _ALIASES["sam3-large"] == (LibreSAM3, "large")


def test_sam3_repo_and_input_frame_are_per_size():
    assert LibreSAM3.HF_REPOS == {"large": "facebook/sam3"}
    assert LibreSAM3.INPUT_SIZES == {"large": 1008}
    assert LibreSAM3.DEFAULT_PCS_SCORE_THRESH == 0.3
    assert "*.pt" in LibreSAM3.SNAPSHOT_IGNORE_PATTERNS


def test_move_embeddings_maps_tracker_feature_list():
    model = object.__new__(LibreSAM3)
    embeddings = [torch.zeros(1), torch.ones(1)]
    moved = model._move_embeddings(embeddings, torch.device("cpu"))
    assert isinstance(moved, list)
    assert [item.device.type for item in moved] == ["cpu", "cpu"]


def test_non_sam3_text_prompt_fails_loudly():
    model = object.__new__(LibreSAM1)
    with pytest.raises(NotImplementedError, match=r"LibreSAM\('sam3'\)"):
        model.predict(Image.new("RGB", (4, 4)), text="person")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"points": [1, 1]},
        {"bboxes": [0, 0, 2, 2]},
        {"labels": [1]},
        {"masks": torch.zeros((2, 2))},
    ],
)
def test_text_and_visual_prompts_are_mutually_exclusive(kwargs):
    with pytest.raises(ValueError, match="cannot be combined"):
        _bare_sam3().predict(Image.new("RGB", (10, 8)), text="person", **kwargs)


def test_text_path_filters_by_score_caps_and_sets_concept_name():
    model = _bare_sam3()
    result = model.predict(
        Image.new("RGB", (10, 8)), text=" person ", conf=0.5, max_det=1
    )
    assert len(result.masks) == 1
    assert float(result.boxes.conf[0]) == pytest.approx(0.9)
    assert result.names == {0: "person"}
    assert model.names == {0: "object"}


def test_text_path_default_score_threshold_and_explicit_keep_all():
    model = _bare_sam3()
    image = Image.new("RGB", (10, 8))

    default = model.predict(image, text="person")
    keep_all = model.predict(image, text="person", conf=0.0)

    assert [float(score) for score in default.boxes.conf] == pytest.approx([0.9, 0.7])
    assert [float(score) for score in keep_all.boxes.conf] == pytest.approx(
        [0.2, 0.9, 0.7]
    )


def test_text_path_rejects_nonpositive_max_det_before_inference():
    with pytest.raises(ValueError, match="integer >= 1"):
        _bare_sam3().predict(
            Image.new("RGB", (10, 8)), text="person", max_det=0
        )


def test_lazy_loader_instantiates_pcs_once(monkeypatch):
    model = _bare_sam3()
    model._pcs_model = None
    model._pcs_processor = None
    created = {"models": 0, "processors": 0}

    class ModelFactory:
        @classmethod
        def from_pretrained(cls, path, dtype):
            created["models"] += 1
            return _PCSModel()

    class ProcessorFactory:
        @classmethod
        def from_pretrained(cls, path):
            created["processors"] += 1
            return _PCSProcessor()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(Sam3Model=ModelFactory, Sam3Processor=ProcessorFactory),
    )
    assert model._load_pcs() == model._load_pcs()
    assert created == {"models": 1, "processors": 1}


def test_gated_download_error_is_actionable(monkeypatch, tmp_path):
    class Forbidden(Exception):
        response = SimpleNamespace(status_code=403)

    model = object.__new__(LibreSAM3)
    model.size = "large"
    model.FILENAME_PREFIX = "TestSAM3"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda *args, **kwargs: (_ for _ in ()).throw(
                Forbidden()
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    with pytest.raises(RuntimeError) as caught:
        model._ensure_weights()
    message = str(caught.value)
    assert "accept the terms" in message
    assert "HF_TOKEN" in message
    assert "https://huggingface.co/facebook/sam3" in message
