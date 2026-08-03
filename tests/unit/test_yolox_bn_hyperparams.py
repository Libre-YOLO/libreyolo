"""YOLOX BatchNorm hyperparameters must match official YOLOX and survive rebuilds.

Official YOLOX (Exp.get_model() in yolox_base.py) sets BN eps=1e-3 and
momentum=0.03 on every size. A wrapper-level post-construction fixup was lost
whenever training rebuilt the model for a new dataset class count
(_rebuild_for_new_classes), so models trained (and in-training validated) at
torch's default eps=1e-5 were later reloaded and evaluated at 1e-3. For the
depthwise "n" size, whose per-channel running_var is small enough for eps to
dominate, that silently corrupted reload-time metrics (RF100-VL "ball": 0.566
mAP at the trained eps vs 0.151 after reload).

These tests pin the invariant at every construction path.
"""

import pytest
import torch.nn as nn

from libreyolo.models.yolox.model import LibreYOLOX
from libreyolo.models.yolox.nn import LibreYOLOXModel

pytestmark = pytest.mark.unit

OFFICIAL_EPS = 1e-3
OFFICIAL_MOMENTUM = 0.03


def _bn_modules(module: nn.Module):
    bns = [m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bns, "expected BatchNorm2d modules in a YOLOX model"
    return bns


def _assert_official_bn(module: nn.Module):
    for bn in _bn_modules(module):
        assert bn.eps == OFFICIAL_EPS
        assert bn.momentum == OFFICIAL_MOMENTUM


class TestYOLOXBNHyperparams:
    @pytest.mark.parametrize("size", ["n", "t", "s"])
    def test_bare_model_construction(self, size):
        """LibreYOLOXModel itself must apply official BN settings."""
        _assert_official_bn(LibreYOLOXModel(config=size, nb_classes=3))

    def test_wrapper_construction(self):
        _assert_official_bn(LibreYOLOX(size="n", nb_classes=80).model)

    def test_survives_class_count_rebuild(self):
        """The regression: _rebuild_for_new_classes constructs a fresh
        LibreYOLOXModel; official BN settings must survive it."""
        wrapper = LibreYOLOX(size="n", nb_classes=80)
        wrapper._rebuild_for_new_classes(18)
        assert wrapper.nb_classes == 18
        _assert_official_bn(wrapper.model)
