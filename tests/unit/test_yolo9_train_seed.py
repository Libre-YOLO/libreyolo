"""YOLO9 training seed-order contracts."""

from types import MethodType

import pytest
import torch

from libreyolo.models.yolo9.model import LibreYOLO9


@pytest.mark.unit
def test_seed_is_applied_before_class_head_rebuild(monkeypatch):
    """The same train seed must produce the same new-head initialization."""

    class _HeadObserved(Exception):
        pass

    samples = []
    model = object.__new__(LibreYOLO9)
    model.nb_classes = 80

    def observe_rebuild(self, num_classes):
        assert num_classes == 2
        samples.append(torch.rand(4))
        raise _HeadObserved

    model._rebuild_for_new_classes = MethodType(observe_rebuild, model)
    monkeypatch.setattr(
        "libreyolo.data.load_data_config",
        lambda *args, **kwargs: {
            "nc": 2,
            "names": {0: "a", 1: "b"},
            "yaml_file": "fixture.yaml",
        },
    )

    for ambient_seed in (1, 999):
        torch.manual_seed(ambient_seed)
        with pytest.raises(_HeadObserved):
            LibreYOLO9.train.__wrapped__(
                model,
                "fixture.yaml",
                seed=17,
                device="cpu",
            )

    assert torch.equal(samples[0], samples[1])
