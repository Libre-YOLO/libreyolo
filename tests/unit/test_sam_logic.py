"""Offline tests for LibreSAM logic that needs no model weights.

Covers mask selection, the segment-everything dispatch guard, and the factory
alias errors — paths that were bug-prone in review but are reachable without
downloading a checkpoint.
"""

import threading

import pytest
import torch
from PIL import Image

from libreyolo.models.sam.base import LibreSAMModel, _is_empty_prompt
from libreyolo.models.sam.model import LibreSAM, LibreSAM1

pytestmark = pytest.mark.unit


def _bare_model():
    """A LibreSAM instance with no weights — just enough state for the pure
    result-assembly logic (built via ``__new__`` to skip the download)."""
    m = object.__new__(LibreSAM1)
    m.names = {0: "object"}
    return m


class TestSelectBest:
    def _candidates(self):
        # Two prompts, three candidate masks each (2, 3, 2, 2).
        masks = torch.zeros((2, 3, 2, 2), dtype=torch.bool)
        masks[0, 1, 0, 0] = True  # prompt 0 best is candidate 1
        masks[1, 2, 1, 1] = True  # prompt 1 best is candidate 2
        scores = torch.tensor([[0.1, 0.9, 0.2], [0.3, 0.4, 0.8]])
        return masks, scores

    def test_single_best_picks_argmax_per_prompt(self):
        masks, scores = self._candidates()
        sel, conf = LibreSAMModel._select_best(masks, scores, multimask=False)
        # One mask per prompt, each the highest-IoU candidate.
        assert sel.shape == (2, 2, 2)
        assert torch.allclose(conf, torch.tensor([0.9, 0.8]))
        assert bool(sel[0, 0, 0]) and bool(sel[1, 1, 1])

    def test_multimask_returns_all_candidates_flattened(self):
        masks, scores = self._candidates()
        sel, conf = LibreSAMModel._select_best(masks, scores, multimask=True)
        # All 2*3 = 6 masks surfaced (the documented "3 ambiguity masks").
        assert sel.shape == (6, 2, 2)
        assert torch.allclose(conf, scores.reshape(6))


class TestAssembleResults:
    """The conf/empty/max_det filtering in _assemble_results (no weights)."""

    def test_conf_filter_then_empty_drop(self):
        m = _bare_model()
        masks = torch.zeros((3, 8, 8), dtype=torch.bool)
        masks[0, 0:4, 0:4] = True  # conf 0.9 -> kept
        masks[1, 5:7, 5:7] = True  # conf 0.5 -> below conf_thres, dropped
        # masks[2] stays empty -> dropped even though its conf (0.7) passes
        conf = torch.tensor([0.9, 0.5, 0.7])
        r = m._assemble_results(
            masks, conf, Image.new("RGB", (8, 8)), None, conf_thres=0.6, max_det=10
        )
        assert len(r.masks) == 1
        assert float(r.boxes.conf[0]) == pytest.approx(0.9)
        assert tuple(r.orig_shape) == (8, 8)
        # Box is the tight extent of the kept mask (cols/rows 0..3).
        assert r.boxes.xyxy[0].tolist() == [0.0, 0.0, 3.0, 3.0]

    def test_max_det_keeps_highest_conf_in_order(self):
        m = _bare_model()
        masks = torch.zeros((3, 8, 8), dtype=torch.bool)
        masks[0, 0:2, 0:2] = True
        masks[1, 2:4, 2:4] = True
        masks[2, 4:6, 4:6] = True
        conf = torch.tensor([0.3, 0.9, 0.6])
        r = m._assemble_results(
            masks, conf, Image.new("RGB", (8, 8)), None, conf_thres=0.0, max_det=2
        )
        assert len(r.masks) == 2
        assert [round(float(c), 1) for c in r.boxes.conf] == [0.9, 0.6]

    def test_all_filtered_returns_empty_contract(self):
        m = _bare_model()
        masks = torch.zeros((2, 8, 8), dtype=torch.bool)
        masks[0, 0, 0] = True
        masks[1, 1, 1] = True
        conf = torch.tensor([0.1, 0.2])
        r = m._assemble_results(
            masks, conf, Image.new("RGB", (8, 8)), None, conf_thres=0.9, max_det=10
        )
        # Empty result: no masks, boxes is a well-formed (0, 4) tensor.
        assert r.masks is None
        assert tuple(r.boxes.xyxy.shape) == (0, 4)
        assert len(r.boxes.conf) == 0


class TestResolveDtype:
    def test_fp32_on_cpu(self):
        m = object.__new__(LibreSAM1)
        m.device = torch.device("cpu")
        assert m._resolve_dtype() == torch.float32

    def test_fp32_even_on_cuda(self):
        # torch.device("cuda") constructs without a GPU present; the point is
        # that SAM never silently downcasts to half on GPU.
        m = object.__new__(LibreSAM1)
        m.device = torch.device("cuda")
        assert m._resolve_dtype() == torch.float32


class TestSnapshotIgnore:
    def test_bin_is_ignored(self):
        # SAM repos ship a duplicate pytorch_model.bin next to safetensors;
        # transformers prefers safetensors, so the pickle must not be fetched.
        assert "*.bin" in LibreSAM1.SNAPSHOT_IGNORE_PATTERNS


class TestSetDevice:
    def test_auto_and_empty_are_noop(self):
        # 'auto'/'' are sentinels meaning "keep current device" — they must not
        # be fed to torch.device() (which would raise), matching the runner.
        m = object.__new__(LibreSAM1)
        m.device = torch.device("cpu")
        assert m._set_device("auto").device == torch.device("cpu")
        assert m._set_device("").device == torch.device("cpu")


class TestEmptyPromptDispatch:
    def test_none_is_empty(self):
        assert _is_empty_prompt(None) is True

    def test_empty_list_is_empty(self):
        # A dynamically-built empty prompt routes to "segment everything",
        # not a cryptic normalization error.
        assert _is_empty_prompt([]) is True

    def test_real_point_is_not_empty(self):
        assert _is_empty_prompt([900, 370]) is False

    def test_numpy_point_is_not_empty(self):
        import numpy as np

        assert _is_empty_prompt(np.array([900, 370])) is False


@pytest.mark.parametrize("max_det", [-1, 0, 1.5, True])
def test_predict_rejects_invalid_max_det_before_inference(max_det):
    model = _bare_model()

    with pytest.raises(ValueError, match="integer >= 1"):
        model.predict(Image.new("RGB", (8, 8)), points=[1, 1], max_det=max_det)


def test_set_image_and_cache_read_are_one_locked_transaction():
    model = _bare_model()
    model._clear_image_state()
    model._image = Image.new("RGB", (4, 4), color="red")
    model._image_embeddings = "red-embedding"

    encode_started = threading.Event()
    allow_encode = threading.Event()
    read_started = threading.Event()
    observed = {}

    def encode(image, **kwargs):
        encode_started.set()
        assert allow_encode.wait(timeout=2)
        return {"embedding": f"{image.getpixel((0, 0))}-embedding"}

    model._encode = encode
    model._image_embeddings_from_encoding = lambda enc: enc["embedding"]
    new_image = Image.new("RGB", (4, 4), color="blue")

    setter = threading.Thread(target=model.set_image, args=(new_image,))
    setter.start()
    assert encode_started.wait(timeout=2)

    def read_cache():
        read_started.set()
        observed["state"] = model._resolve_image(None, "auto")

    reader = threading.Thread(target=read_cache)
    reader.start()
    assert read_started.wait(timeout=2)
    reader.join(timeout=0.05)
    assert reader.is_alive(), "cache read escaped while set_image held the session lock"

    allow_encode.set()
    setter.join(timeout=2)
    reader.join(timeout=2)
    assert not setter.is_alive()
    assert not reader.is_alive()

    image, path, embedding = observed["state"]
    assert image.getpixel((0, 0)) == (0, 0, 255)
    assert path is None
    assert embedding == "(0, 0, 255)-embedding"


class TestFactoryAliases:
    def test_unknown_alias_raises_without_download(self):
        # Alias resolution happens before any model construction/download.
        with pytest.raises(ValueError, match="Unknown SAM model"):
            LibreSAM("xl")

    def test_case_and_whitespace_insensitive(self):
        from libreyolo.models.sam.model import _ALIASES

        # The factory lowercases/strips before lookup; both resolve.
        assert "sam_b" in _ALIASES
        assert _ALIASES["b"][1] == "base"
