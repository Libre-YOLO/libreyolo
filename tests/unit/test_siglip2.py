"""LibreSigLIP2 zero-shot (open-vocabulary) classification - unit tests.

Most tests use a tiny synthetic SigLIP (random weights, small towers) so they
run fast on CPU with no external weights or network. Forward/zero-shot *parity*
against the reference SiglipModel and imagenette accuracy live in
``test_siglip2_parity.py`` (external_data).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

# LibreSigLIP2's tokenizer needs the optional ``siglip2`` extra (sentencepiece).
# Skip cleanly instead of erroring when it isn't installed.
pytest.importorskip("sentencepiece", reason="LibreSigLIP2 tokenizer needs the 'siglip2' extra")

from libreyolo.models.base.model import BaseModel
from libreyolo.models.siglip2 import nn as sig_nn
from libreyolo.models.siglip2.model import LibreSigLIP2, siglip2_logits, siglip2_probs
from libreyolo.models.siglip2.tokenizer import SigLIP2Tokenizer

pytestmark = [pytest.mark.unit, pytest.mark.siglip2]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_siglip2(monkeypatch):
    """A tiny random-weight LibreSigLIP2 (no external weights) for fast tests.

    ``vocab_size`` stays at the real 256k so the (real) SentencePiece tokenizer's
    ids index the token embedding; every other dimension is shrunk.
    """
    # projection_size must equal vision_width (image & text embeds share a dim).
    tiny = sig_nn.SigLIP2Config(
        vision_width=64, vision_layers=1, vision_heads=2, vision_intermediate=128,
        image_size=32, patch_size=16,
        text_width=32, text_layers=1, text_heads=2, text_intermediate=64,
        vocab_size=256000, max_position_embeddings=64, projection_size=64,
    )
    monkeypatch.setitem(sig_nn.SIGLIP2_CONFIGS, "tiny", tiny)
    monkeypatch.setattr(
        LibreSigLIP2, "INPUT_SIZES", {**LibreSigLIP2.INPUT_SIZES, "tiny": tiny.image_size}
    )
    torch.manual_seed(0)
    state = sig_nn.SigLIP2Model(tiny).state_dict()
    return LibreSigLIP2(
        model_path=state, size="tiny", classes=["a cat", "a dog"], device="cpu"
    )


@pytest.fixture
def tiny_siglip2_embed(monkeypatch):
    """Tiny SigLIP2 loaded through the whole-image embed task."""
    tiny = sig_nn.SigLIP2Config(
        vision_width=64,
        vision_layers=1,
        vision_heads=2,
        vision_intermediate=128,
        image_size=32,
        patch_size=16,
        text_width=32,
        text_layers=1,
        text_heads=2,
        text_intermediate=64,
        vocab_size=256000,
        max_position_embeddings=64,
        projection_size=64,
    )
    monkeypatch.setitem(sig_nn.SIGLIP2_CONFIGS, "tiny", tiny)
    monkeypatch.setattr(
        LibreSigLIP2,
        "INPUT_SIZES",
        {**LibreSigLIP2.INPUT_SIZES, "tiny": tiny.image_size},
    )
    torch.manual_seed(0)
    state = sig_nn.SigLIP2Model(tiny).state_dict()
    return LibreSigLIP2(
        model_path=state,
        size="tiny",
        task="embed",
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Registry / detection (pure classmethods - no model)
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered(self):
        assert any(c.__name__ == "LibreSigLIP2" for c in BaseModel._registry)

    def test_can_load_signature(self):
        good = {
            "logit_bias": torch.tensor([1.0]),
            "logit_scale": torch.tensor([1.0]),
            "vision_model.embeddings.patch_embedding.weight": torch.zeros(768, 3, 16, 16),
            "text_model.head.weight": torch.zeros(768, 768),
        }
        assert LibreSigLIP2.can_load(good)
        # Missing any one signature key -> not a SigLIP2 checkpoint.
        for drop in list(good):
            assert not LibreSigLIP2.can_load({k: v for k, v in good.items() if k != drop})
        assert not LibreSigLIP2.can_load({"backbone.0.weight": torch.zeros(1)})

    def test_rejects_clip_and_clip_rejects_siglip2(self):
        # Bidirectional sibling rejection: CLIP has logit_scale but no logit_bias.
        from libreyolo.models.clip.model import LibreCLIP

        clip_sd = {
            "logit_scale": torch.tensor(1.0),
            "text_projection": torch.zeros(8, 8),
            "visual.conv1.weight": torch.zeros(8, 3, 16, 16),
        }
        siglip_sd = {
            "logit_bias": torch.tensor([1.0]),
            "logit_scale": torch.tensor([1.0]),
            "vision_model.embeddings.patch_embedding.weight": torch.zeros(768, 3, 16, 16),
            "text_model.head.weight": torch.zeros(768, 768),
        }
        assert not LibreSigLIP2.can_load(clip_sd)
        assert not LibreCLIP.can_load(siglip_sd)

    @pytest.mark.parametrize(
        "width,expected", [(768, "b16"), (1152, "so400m"), (512, None)]
    )
    def test_detect_size(self, width, expected):
        sd = {"vision_model.embeddings.patch_embedding.weight": torch.zeros(width, 3, 16, 16)}
        assert LibreSigLIP2.detect_size(sd) == expected

    def test_detect_nb_classes_is_open_vocab(self):
        assert LibreSigLIP2.detect_nb_classes({"logit_bias": torch.tensor([1.0])}) is None

    def test_filename_detection(self):
        assert LibreSigLIP2.detect_size_from_filename("LibreSigLIP2b16-cls.pt") == "b16"
        assert LibreSigLIP2.detect_size_from_filename("LibreSigLIP2so400m-cls.pt") == "so400m"
        assert LibreSigLIP2.detect_task_from_filename("LibreSigLIP2b16-cls.pt") == "classify"
        assert (
            LibreSigLIP2.detect_size_from_filename("LibreSigLIP2b16-embed.pt")
            is None
        )

    def test_download_url(self):
        url = LibreSigLIP2.get_download_url("LibreSigLIP2b16-cls.pt")
        assert url == (
            "https://huggingface.co/LibreYOLO/LibreSigLIP2b16-cls/resolve/main/"
            "LibreSigLIP2b16-cls.pt"
        )


# ---------------------------------------------------------------------------
# Tokenizer (vendored SentencePiece; multilingual)
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_shape_and_special_tokens(self):
        tok = SigLIP2Tokenizer()
        out = tok(["a photo of a cat"])
        assert out.shape == (1, 64)
        assert out.dtype == torch.long
        # Every sequence ends the content with the eot id before padding.
        row = out[0].tolist()
        assert tok.eot_token_id in row

    def test_deterministic(self):
        tok = SigLIP2Tokenizer()
        a = tok(["an empty aisle", "a forklift"])
        b = tok(["an empty aisle", "a forklift"])
        assert torch.equal(a, b)

    def test_padding_is_pad_id(self):
        tok = SigLIP2Tokenizer()
        out = tok(["a"])  # short prompt -> mostly padding
        assert out[0, -1].item() == tok.pad_token_id

    def test_multilingual(self):
        # SentencePiece (Gemma vocab) tokenizes many languages, all to 64 tokens.
        tok = SigLIP2Tokenizer()
        out = tok(["a cat", "un gato", "un chat", "eine Katze", "un gatto"])
        assert out.shape == (5, 64)
        # Different languages produce different token sequences.
        assert not torch.equal(out[0], out[1])

    def test_truncation_keeps_eot_last(self):
        tok = SigLIP2Tokenizer(context_length=8)
        out = tok(["one two three four five six seven eight nine ten eleven"])
        assert out.shape == (1, 8)
        assert out[0, -1].item() == tok.eot_token_id

    def test_missing_sentencepiece_message(self, monkeypatch):
        import sys

        import libreyolo.models.siglip2.tokenizer as tk

        monkeypatch.setitem(sys.modules, "sentencepiece", None)  # import -> ImportError
        with pytest.raises(ImportError, match=r"libreyolo\[siglip2\]"):
            tk._require_sentencepiece()


# ---------------------------------------------------------------------------
# Scoring: softmax default, multi_label sigmoid, and logit_bias handling
# ---------------------------------------------------------------------------


class TestScoring:
    def test_logit_bias_is_applied(self):
        # logit_bias shifts every logit by the same scalar.
        torch.manual_seed(0)
        img = torch.randn(1, 8)
        txt = F.normalize(torch.randn(3, 8), dim=-1)
        scale = torch.tensor([0.5])
        l_a = siglip2_logits(img, txt, scale, torch.tensor([0.0]))
        l_b = siglip2_logits(img, txt, scale, torch.tensor([3.0]))
        assert torch.allclose(l_b - l_a, torch.full_like(l_a, 3.0), atol=1e-6)

    def test_softmax_cancels_bias_sigmoid_does_not(self):
        # This is the landmine: softmax mode hides a missing/incorrect logit_bias
        # (a scalar cancels), sigmoid mode does not. So multi_label sigmoid is the
        # mode that actually exercises correct bias handling.
        torch.manual_seed(1)
        img = torch.randn(1, 8)
        txt = F.normalize(torch.randn(4, 8), dim=-1)
        scale = torch.tensor([0.7])
        l_a = siglip2_logits(img, txt, scale, torch.tensor([0.0]))
        l_b = siglip2_logits(img, txt, scale, torch.tensor([-5.0]))
        assert torch.allclose(siglip2_probs(l_a, False), siglip2_probs(l_b, False), atol=1e-6)
        assert not torch.allclose(siglip2_probs(l_a, True), siglip2_probs(l_b, True), atol=1e-3)

    def test_probs_modes(self):
        logits = torch.tensor([[1.0, -2.0, 0.5]])
        softmax = siglip2_probs(logits, False)
        sigmoid = siglip2_probs(logits, True)
        assert pytest.approx(float(softmax.sum()), abs=1e-6) == 1.0
        assert torch.allclose(sigmoid, torch.sigmoid(logits))
        # Independent sigmoid probs generally do not sum to 1.
        assert abs(float(sigmoid.sum()) - 1.0) > 1e-3


# ---------------------------------------------------------------------------
# Open-vocabulary head: set_classes
# ---------------------------------------------------------------------------


class TestSetClasses:
    def test_updates_names_and_count(self, tiny_siglip2):
        tiny_siglip2.set_classes(["a forklift", "an empty aisle", "a spill"])
        assert tiny_siglip2.nb_classes == 3
        assert tiny_siglip2.names == {0: "a forklift", 1: "an empty aisle", 2: "a spill"}

    def test_caches_normalized_embeds(self, tiny_siglip2):
        tiny_siglip2.set_classes(["a", "b", "c", "d"])
        emb = tiny_siglip2._text_embeds
        assert emb.shape == (4, tiny_siglip2.model.config.projection_size)
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_embeds_not_recomputed_per_forward(self, tiny_siglip2):
        tiny_siglip2.set_classes(["x", "y"])
        before = tiny_siglip2._text_embeds
        tiny_siglip2._forward(torch.zeros(1, 3, 32, 32))
        assert tiny_siglip2._text_embeds is before

    def test_multi_label_flag_via_set_classes(self, tiny_siglip2):
        assert tiny_siglip2.multi_label is False
        tiny_siglip2.set_classes(["a", "b"], multi_label=True)
        assert tiny_siglip2.multi_label is True

    def test_empty_raises(self, tiny_siglip2):
        with pytest.raises(ValueError):
            tiny_siglip2.set_classes([])


# ---------------------------------------------------------------------------
# Forward / postprocess / predict
# ---------------------------------------------------------------------------


class TestInference:
    def test_forward_shape(self, tiny_siglip2):
        tiny_siglip2.set_classes(["a", "b", "c"])
        logits = tiny_siglip2._forward(torch.zeros(2, 3, 32, 32))
        assert logits.shape == (2, 3)

    def test_postprocess_softmax_sums_to_one(self, tiny_siglip2):
        tiny_siglip2.set_classes(["a", "b", "c"])
        logits = tiny_siglip2._forward(torch.zeros(1, 3, 32, 32))
        out = tiny_siglip2._postprocess(logits, 0.0, 0.0, (32, 32))
        assert out["probs"].shape == (3,)
        assert pytest.approx(float(out["probs"].sum()), abs=1e-5) == 1.0

    def test_postprocess_multi_label_is_sigmoid(self, tiny_siglip2):
        tiny_siglip2.set_classes(["a", "b", "c"], multi_label=True)
        logits = tiny_siglip2._forward(torch.zeros(1, 3, 32, 32))
        out = tiny_siglip2._postprocess(logits, 0.0, 0.0, (32, 32))
        assert torch.allclose(out["probs"], torch.sigmoid(logits[0].float()), atol=1e-6)

    def test_predict_endtoend(self, tiny_siglip2):
        tiny_siglip2.set_classes(["a cat", "a dog", "a truck"])
        img = Image.fromarray(np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8))
        r = tiny_siglip2.predict(img, verbose=False)[0]
        assert r.probs is not None
        assert 0 <= r.probs.top1 < 3
        assert tiny_siglip2.names[r.probs.top1] in {"a cat", "a dog", "a truck"}

    def test_forward_without_classes_raises(self, tiny_siglip2):
        tiny_siglip2._text_embeds = None
        with pytest.raises(RuntimeError):
            tiny_siglip2._forward(torch.zeros(1, 3, 32, 32))


class TestEmbedTask:
    def test_image_text_shapes_and_norms(self, tiny_siglip2_embed):
        image = Image.fromarray(
            np.random.default_rng(8).integers(
                0, 256, size=(48, 36, 3), dtype=np.uint8
            )
        )
        result = tiny_siglip2_embed(image)
        assert result.boxes is None
        assert result.probs is None
        assert result.embeddings.data.shape == (1, 64)
        assert result.embeddings.data.dtype == torch.float32
        assert torch.allclose(
            result.embeddings.data.norm(dim=-1),
            torch.ones(1),
            atol=1e-5,
        )

        text = tiny_siglip2_embed.embed_text(["a red truck", "a dog"])
        assert text.shape == (2, 64)
        assert text.dtype == torch.float32
        assert torch.allclose(text.norm(dim=-1), torch.ones(2), atol=1e-5)
        assert tiny_siglip2_embed.embed_text("a red truck").shape == (1, 64)
        assert tiny_siglip2_embed.embed_text([]).shape == (0, 64)
        with pytest.raises(TypeError, match="string"):
            tiny_siglip2_embed.embed_text(["a dog", 3])

    def test_embed_verb_stacks(self, tiny_siglip2_embed):
        image_a = Image.new("RGB", (30, 40), color=(200, 30, 10))
        image_b = Image.new("RGB", (30, 40), color=(10, 30, 200))
        vectors = tiny_siglip2_embed.embed([image_a, image_b], batch=2)
        assert vectors.shape == (2, 64)
        assert torch.allclose(vectors.norm(dim=-1), torch.ones(2), atol=1e-5)


# ---------------------------------------------------------------------------
# Train is out of scope; export is frozen-class ONNX only
# ---------------------------------------------------------------------------


class TestScope:
    def test_train_raises(self, tiny_siglip2):
        with pytest.raises(NotImplementedError):
            tiny_siglip2.train(data="anything")

    def test_export_non_onnx_raises(self, tiny_siglip2):
        with pytest.raises(NotImplementedError):
            tiny_siglip2.export(format="torchscript")

    @pytest.mark.parametrize("format", ["onnx", "tflite"])
    def test_embed_export_routes_to_shared_export(
        self, tiny_siglip2_embed, monkeypatch, format
    ):
        captured = {}

        def fake_export(self, format="onnx", **kwargs):
            captured.update(format=format, **kwargs)
            return f"siglip2-embed.{format}"

        monkeypatch.setattr(BaseModel, "export", fake_export)
        assert tiny_siglip2_embed.export(format, dynamic=False) == (
            f"siglip2-embed.{format}"
        )
        assert captured == {"format": format, "opset": 17, "dynamic": False}

    def test_frozen_onnx_roundtrip(self, tiny_siglip2, tmp_path):
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        tiny_siglip2.set_classes(["a cat", "a dog", "a truck"])
        out = tiny_siglip2.export(format="onnx", output=str(tmp_path / "siglip2.onnx"))
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        x = np.zeros((1, 3, tiny_siglip2.input_size, tiny_siglip2.input_size), dtype=np.float32)
        (logits,) = sess.run(None, {"images": x})
        assert logits.shape == (1, 3)
        # ONNX frozen head (with logit_bias) must match the eager zero-shot logits.
        with torch.no_grad():
            eager = tiny_siglip2._forward(torch.from_numpy(x)).cpu().numpy()
        np.testing.assert_allclose(logits, eager, rtol=1e-3, atol=1e-4)
