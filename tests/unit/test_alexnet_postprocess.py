"""Unit tests for centralized AlexNet classification postprocessing."""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.alexnet import LibreAlexNet
from libreyolo.postprocess.alexnet import postprocess


pytestmark = [pytest.mark.unit, pytest.mark.alexnet]


@pytest.mark.parametrize(
    "wrapped",
    [
        lambda logits: logits,
        lambda logits: [logits],
        lambda logits: (logits,),
        lambda logits: {"logits": logits},
        lambda logits: {"predictions": logits},
    ],
)
def test_postprocess_normalizes_supported_output_wrappers(wrapped):
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    result = postprocess(wrapped(logits))

    assert set(result) == {"probs"}
    assert result["probs"].shape == (3,)
    torch.testing.assert_close(result["probs"].sum(), torch.tensor(1.0))
    assert result["probs"].argmax().item() == 2


def test_model_wrapper_delegates_to_central_postprocess():
    model = LibreAlexNet(size="b", nc=3)

    result = model._postprocess(
        torch.tensor([[0.5, 1.5, -2.0]]),
        conf_thres=0.25,
        iou_thres=0.7,
        original_size=(224, 224),
    )

    torch.testing.assert_close(
        result["probs"], torch.softmax(torch.tensor([0.5, 1.5, -2.0]), dim=0)
    )
