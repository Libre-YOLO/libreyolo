"""Factory and checkpoint-recognition tests for LibreFCN."""

from __future__ import annotations

import pytest
import torch

from libreyolo import LibreFCN
from libreyolo.models.registry import group_of


def _fcn_state(depth: int = 50, nc: int = 21) -> dict[str, torch.Tensor]:
    last_block = 5 if depth == 50 else 22
    return {
        "backbone.conv1.weight": torch.empty(64, 3, 7, 7),
        "backbone.layer4.0.conv2.weight": torch.empty(512, 512, 3, 3),
        f"backbone.layer3.{last_block}.conv3.weight": torch.empty(1024, 256, 1, 1),
        "classifier.0.weight": torch.empty(512, 2048, 3, 3),
        "classifier.1.running_mean": torch.empty(512),
        "classifier.4.weight": torch.empty(nc, 512, 1, 1),
        "aux_classifier.0.weight": torch.empty(256, 1024, 3, 3),
        "aux_classifier.4.weight": torch.empty(nc, 256, 1, 1),
    }


def test_fcn_public_factory_shell():
    model = LibreFCN(size="r50", nb_classes=21, device="cpu")
    assert model.family == "fcn"
    assert model.task == "semantic"
    assert model.input_size == 520
    assert model.names[0] == "__background__"
    assert model.names[20] == "tvmonitor"
    assert group_of("fcn") == "g3"


@pytest.mark.parametrize(("depth", "size"), [(50, "r50"), (101, "r101")])
def test_fcn_checkpoint_recognition(depth, size):
    state = _fcn_state(depth)
    assert LibreFCN.can_load(state)
    assert LibreFCN.detect_size(state) == size
    assert LibreFCN.detect_nb_classes(state) == 21


def test_fcn_canonical_default_task_filename_is_suffixless():
    assert LibreFCN.detect_size_from_filename("LibreFCNr50.pt") == "r50"
    assert LibreFCN.get_download_url("LibreFCNr50.pt") == (
        "https://huggingface.co/LibreYOLO/LibreFCNr50/resolve/main/LibreFCNr50.pt"
    )


def test_fcn_rejects_generic_resnet_backbone():
    state = {
        "backbone.conv1.weight": torch.empty(64, 3, 7, 7),
        "classifier.4.weight": torch.empty(21, 512, 1, 1),
    }
    assert not LibreFCN.can_load(state)


def test_fcn_training_is_explicitly_out_of_scope():
    model = LibreFCN(size="r50", device="cpu")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train()
