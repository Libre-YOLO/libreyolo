"""LibreCLIP zero-shot (open-vocabulary) classification — unit tests.

Most tests use a tiny synthetic CLIP (random weights, small towers) so they run
fast on CPU with no external weights or network. Forward/zero-shot *parity*
against open_clip and accuracy live in ``test_clip_parity.py`` (external_data).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

# LibreCLIP's tokenizer needs the optional ``clip`` extra (ftfy + regex).
# Skip cleanly instead of erroring when it isn't installed.
pytest.importorskip("ftfy", reason="LibreCLIP tokenizer needs the 'clip' extra")
pytest.importorskip("regex", reason="LibreCLIP tokenizer needs the 'clip' extra")

from libreyolo.models.base.model import BaseModel
from libreyolo.models.clip import nn as clip_nn
from libreyolo.models.clip.model import LibreCLIP
from libreyolo.models.clip.tokenizer import SimpleTokenizer

pytestmark = [pytest.mark.unit, pytest.mark.clip]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_clip(monkeypatch):
    """A tiny random-weight LibreCLIP (no external weights) for fast tests."""
    tiny = clip_nn.CLIPConfig(
        embed_dim=16,
        image_size=32,
        patch_size=16,
        vision_width=64,
        vision_layers=1,
        text_width=32,
        text_heads=2,
        text_layers=1,
    )
    monkeypatch.setitem(clip_nn.CLIP_CONFIGS, "tiny", tiny)
    monkeypatch.setattr(
        LibreCLIP, "INPUT_SIZES", {**LibreCLIP.INPUT_SIZES, "tiny": tiny.image_size}
    )
    torch.manual_seed(0)
    state = clip_nn.CLIPModel(tiny).state_dict()
    return LibreCLIP(
        model_path=state, size="tiny", classes=["a cat", "a dog"], device="cpu"
    )


# ---------------------------------------------------------------------------
# Registry / detection (pure classmethods — no model)
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered(self):
        assert any(c.__name__ == "LibreCLIP" for c in BaseModel._registry)

    def test_can_load_signature(self):
        good = {
            "logit_scale": torch.tensor(1.0),
            "text_projection": torch.zeros(8, 8),
            "visual.conv1.weight": torch.zeros(8, 3, 32, 32),
        }
        assert LibreCLIP.can_load(good)
        # Missing any one signature key -> not a CLIP checkpoint.
        assert not LibreCLIP.can_load(
            {k: v for k, v in good.items() if k != "logit_scale"}
        )
        assert not LibreCLIP.can_load({"backbone.0.weight": torch.zeros(1)})

    @pytest.mark.parametrize(
        "patch,width,expected",
        [(32, 768, "b32"), (16, 768, "b16"), (14, 1024, "l14"), (8, 768, None)],
    )
    def test_detect_size(self, patch, width, expected):
        sd = {"visual.conv1.weight": torch.zeros(width, 3, patch, patch)}
        assert LibreCLIP.detect_size(sd) == expected

    def test_detect_nb_classes_is_open_vocab(self):
        # Open-vocabulary: no fixed head to count.
        assert LibreCLIP.detect_nb_classes({"logit_scale": torch.tensor(1.0)}) is None

    def test_filename_detection(self):
        assert LibreCLIP.detect_size_from_filename("LibreCLIPb32-cls.pt") == "b32"
        assert LibreCLIP.detect_size_from_filename("LibreCLIPb16-cls.pt") == "b16"
        assert LibreCLIP.detect_task_from_filename("LibreCLIPb32-cls.pt") == "classify"

    def test_download_url(self):
        url = LibreCLIP.get_download_url("LibreCLIPb32-cls.pt")
        assert url == (
            "https://huggingface.co/LibreYOLO/LibreCLIPb32-cls/resolve/main/"
            "LibreCLIPb32-cls.pt"
        )


# ---------------------------------------------------------------------------
# Tokenizer (vendored, no model)
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_shape_and_special_tokens(self):
        tok = SimpleTokenizer()
        out = tok(["a photo of a cat."])
        assert out.shape == (1, 77)
        assert out.dtype == torch.long
        assert out[0, 0].item() == 49406  # SOT
        assert 49407 in out[0].tolist()  # EOT present

    def test_deterministic(self):
        tok = SimpleTokenizer()
        a = tok(["an empty aisle", "a forklift"])
        b = tok(["an empty aisle", "a forklift"])
        assert torch.equal(a, b)

    def test_padding_is_zero(self):
        tok = SimpleTokenizer()
        out = tok(["a"])  # short prompt -> mostly padding
        assert out[0, -1].item() == 0

    def test_decode_roundtrip(self):
        tok = SimpleTokenizer()
        ids = tok.encode("a photo of a forklift")
        assert "forklift" in tok.decode(ids)

    def test_missing_text_deps_message(self, monkeypatch):
        # Native towers need no open_clip, but the tokenizer needs ftfy/regex;
        # a missing one must point at the libreyolo[clip] extra, not crash raw.
        import sys

        import libreyolo.models.clip.tokenizer as tk

        monkeypatch.setitem(sys.modules, "ftfy", None)  # `import ftfy` -> ImportError
        with pytest.raises(ImportError, match=r"libreyolo\[clip\]"):
            tk.basic_clean("hello world")


# ---------------------------------------------------------------------------
# Open-vocabulary head: set_classes
# ---------------------------------------------------------------------------


class TestSetClasses:
    def test_updates_names_and_count(self, tiny_clip):
        tiny_clip.set_classes(["a forklift", "an empty aisle", "a spill"])
        assert tiny_clip.nb_classes == 3
        assert tiny_clip.names == {0: "a forklift", 1: "an empty aisle", 2: "a spill"}

    def test_caches_normalized_embeds(self, tiny_clip):
        tiny_clip.set_classes(["a", "b", "c", "d"])
        emb = tiny_clip._text_embeds
        assert emb.shape == (4, tiny_clip.model.config.embed_dim)
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_embeds_not_recomputed_per_forward(self, tiny_clip):
        tiny_clip.set_classes(["x", "y"])
        before = tiny_clip._text_embeds
        tiny_clip._forward(torch.zeros(1, 3, 32, 32))
        assert tiny_clip._text_embeds is before  # cached, identity preserved

    def test_template_ensemble_averaging(self, tiny_clip):
        # Two templates -> still K embeds, each averaged across templates.
        tiny_clip.set_classes(["cat"], templates=["a photo of a {}.", "a {}."])
        assert tiny_clip._text_embeds.shape[0] == 1

    def test_empty_raises(self, tiny_clip):
        with pytest.raises(ValueError):
            tiny_clip.set_classes([])


# ---------------------------------------------------------------------------
# Forward / postprocess / predict
# ---------------------------------------------------------------------------


class TestInference:
    def test_forward_shape(self, tiny_clip):
        tiny_clip.set_classes(["a", "b", "c"])
        logits = tiny_clip._forward(torch.zeros(2, 3, 32, 32))
        assert logits.shape == (2, 3)

    def test_postprocess_sums_to_one(self, tiny_clip):
        tiny_clip.set_classes(["a", "b", "c"])
        logits = tiny_clip._forward(torch.zeros(1, 3, 32, 32))
        out = tiny_clip._postprocess(logits, 0.0, 0.0, (32, 32))
        probs = out["probs"]
        assert probs.shape == (3,)
        assert pytest.approx(float(probs.sum()), abs=1e-5) == 1.0

    def test_predict_endtoend(self, tiny_clip):
        tiny_clip.set_classes(["a cat", "a dog", "a truck"])
        img = Image.fromarray(np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8))
        r = tiny_clip.predict(img, verbose=False)[0]
        assert r.probs is not None
        assert 0 <= r.probs.top1 < 3
        assert len(r.probs.top5) <= 3
        assert tiny_clip.names[r.probs.top1] in {"a cat", "a dog", "a truck"}

    def test_forward_without_classes_raises(self, tiny_clip):
        tiny_clip._text_embeds = None
        with pytest.raises(RuntimeError):
            tiny_clip._forward(torch.zeros(1, 3, 32, 32))


# ---------------------------------------------------------------------------
# Train is out of scope; export is frozen-class ONNX only
# ---------------------------------------------------------------------------


class TestScope:
    def test_train_raises(self, tiny_clip):
        with pytest.raises(NotImplementedError):
            tiny_clip.train(data="anything")

    def test_export_non_onnx_raises(self, tiny_clip):
        with pytest.raises(NotImplementedError):
            tiny_clip.export(format="torchscript")

    def test_export_accepts_standard_cli_arguments(
        self, tiny_clip, monkeypatch, tmp_path
    ):
        captured = {}

        def fake_export(model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)
            return kwargs["output"]

        monkeypatch.setattr(
            "libreyolo.models.clip.export.export_frozen_onnx", fake_export
        )
        output = tmp_path / "clip.onnx"

        result = tiny_clip.export(
            format="onnx",
            output_path=output,
            opset=None,
            simplify=False,
            dynamic=False,
            half=False,
            int8=False,
            imgsz=(32, 32),
            batch=2,
            device="cpu",
            verbose=True,
        )

        assert result == str(output)
        assert captured == {
            "model": tiny_clip,
            "imgsz": 32,
            "opset": None,
            "output": str(output),
            "batch": 2,
            "dynamic": False,
            "device": "cpu",
            "simplify": False,
            "verbose": True,
        }

    @pytest.mark.parametrize(
        "kwargs,error,match",
        [
            ({"half": True}, NotImplementedError, "half=True"),
            ({"int8": True}, NotImplementedError, "int8=True"),
            (
                {"int8": True, "fraction": 0.5, "allow_download_scripts": False},
                NotImplementedError,
                "int8=True",
            ),
            ({"batch": 0}, ValueError, "positive integer"),
            ({"batch": True}, ValueError, "positive integer"),
            ({"imgsz": 64}, ValueError, "only supports imgsz=32"),
            ({"imgsz": (32, 31)}, ValueError, "only supports imgsz=32"),
        ],
    )
    def test_export_rejects_unsupported_requests(
        self, tiny_clip, tmp_path, kwargs, error, match
    ):
        output = tmp_path / "should-not-exist.onnx"
        with pytest.raises(error, match=match):
            tiny_clip.export(output_path=output, **kwargs)
        assert not output.exists()

    def test_frozen_onnx_roundtrip(self, tiny_clip, tmp_path):
        onnx = pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        tiny_clip.set_classes(["a cat", "a dog", "a truck"])
        out = tiny_clip.export(
            format="onnx",
            output_path=tmp_path / "clip.onnx",
            opset=None,
            batch=2,
            dynamic=False,
            device="cpu",
            simplify=False,
            verbose=False,
        )
        proto = onnx.load(out)
        metadata = {entry.key: entry.value for entry in proto.metadata_props}
        assert proto.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 2
        assert metadata["model_family"] == "clip"
        assert metadata["task"] == "classify"
        assert json.loads(metadata["names"]) == {
            "0": "a cat",
            "1": "a dog",
            "2": "a truck",
        }
        assert json.loads(metadata["classification_mean"]) == pytest.approx(
            [0.48145466, 0.4578275, 0.40821073]
        )
        assert metadata["classification_crop_pct"] == "1.0"
        assert metadata["classification_interpolation"] == "bicubic"
        assert metadata["classification_square_resize"] == "false"
        assert metadata["classification_activation"] == "softmax"
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        x = np.zeros(
            (2, 3, tiny_clip.input_size, tiny_clip.input_size), dtype=np.float32
        )
        (logits,) = sess.run(None, {"images": x})
        assert logits.shape == (2, 3)
        # ONNX frozen head must match the eager zero-shot logits.
        with torch.no_grad():
            eager = tiny_clip._forward(torch.from_numpy(x)).cpu().numpy()
        np.testing.assert_allclose(logits, eager, rtol=1e-3, atol=1e-4)
