"""Unit tests for BaseModel._predict_augment_semantic (flip TTA).

Locks two behaviors: (1) the flipped view's logits are correctly flipped back
into spatial alignment with the original view before merging, and (2) views
are merged by averaging *softmax probabilities*, not raw logits — the two
differ whenever per-view logit magnitudes diverge (see
``test_merge_averages_softmax_not_raw_logits``), and softmax-then-average is
the standard flip-TTA convention (e.g. mmsegmentation), so this is the
contract worth pinning.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.base.model import BaseModel

pytestmark = pytest.mark.unit


def test_predict_augment_semantic_flips_logits_back_into_alignment():
    """A content-aware stub: flipping the image must flip its raw
    prediction, and _predict_augment_semantic must flip it back before
    merging — otherwise the two views disagree and the merge degrades to
    a near-uniform, ambiguous distribution instead of a confident one.
    """

    class _ContentAwareModel:
        task = "semantic"
        names = {0: "dark", 1: "light"}
        device = torch.device("cpu")

        def _preprocess(self, image, color_format="auto", input_size=None):
            arr = torch.from_numpy(
                np.asarray(image.convert("L"), dtype=np.float32).copy()
            ).view(1, 1, image.height, image.width)
            return arr, image, image.size, 1.0

        def _forward(self, tensor):
            return tensor

        def _postprocess_semantic_logits(self, output, original_size, **kwargs):
            dark = output < 128
            logits = torch.zeros(1, 2, output.shape[-2], output.shape[-1])
            logits[:, 0:1][dark] = 10.0
            logits[:, 1:2][~dark] = 10.0
            return logits

    # Left half dark (class 0), right half light (class 1).
    img = Image.new("L", (4, 1))
    img.putdata([0, 0, 255, 255])
    img = img.convert("RGB")

    model = _ContentAwareModel()
    result = BaseModel._predict_augment_semantic(
        model,
        img,
        image_path=None,
        original_size=(4, 1),
        effective_imgsz=4,
        color_format="auto",
    )

    assert result.semantic_mask.data.tolist() == [[0, 0, 1, 1]]


def test_merge_averages_softmax_not_raw_logits():
    """Constructed so raw-logit averaging and softmax averaging disagree.

    view_a = [-11.48, -1.42, 9.97] -> raw argmax with view_b averages to
    class 1, but softmax(view_a) + softmax(view_b) averages to class 2.
    A regression to "average logits, then argmax" would silently flip this
    test's expected class.
    """

    class _SequencedLogitsModel:
        task = "semantic"
        names = {0: "a", 1: "b", 2: "c"}
        device = torch.device("cpu")

        def __init__(self, views):
            self._views = views
            self._call = 0

        def _preprocess(self, image, color_format="auto", input_size=None):
            return torch.zeros(1, 1, 1, 1), image, image.size, 1.0

        def _forward(self, tensor):
            return tensor

        def _postprocess_semantic_logits(self, output, original_size, **kwargs):
            logits = self._views[self._call].view(1, -1, 1, 1)
            self._call += 1
            return logits

    view_a = torch.tensor([-11.48, -1.42, 9.97])
    view_b = torch.tensor([7.32, 7.32, -10.03])
    raw_avg = (view_a + view_b) / 2
    softmax_avg = (torch.softmax(view_a, dim=0) + torch.softmax(view_b, dim=0)) / 2
    assert raw_avg.argmax().item() != softmax_avg.argmax().item()

    # 1x1 image: flip(-1) on the flipped view's logits is a spatial no-op,
    # so the call order alone controls which view is which.
    img = Image.new("RGB", (1, 1))
    model = _SequencedLogitsModel([view_a, view_b])

    result = BaseModel._predict_augment_semantic(
        model,
        img,
        image_path=None,
        original_size=(1, 1),
        effective_imgsz=1,
        color_format="auto",
    )

    assert result.semantic_mask.data.item() == softmax_avg.argmax().item()
