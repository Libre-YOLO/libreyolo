from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.data.anomaly_dataset import (
    resolve_anomaly_test_samples,
    resolve_good_training_images,
)
from libreyolo.models.patchcore.model import LibrePatchCore
from libreyolo.models.patchcore.nn import PatchCoreNet, greedy_coreset
from libreyolo.postprocess.patchcore import postprocess
from libreyolo.utils.drawing import draw_anomaly_map
from libreyolo.utils.results import AnomalyMap, Results
from libreyolo.validation.anomaly_validator import best_f1, binary_auroc


def test_anomaly_results_contract_and_rendering(tmp_path):
    heatmap = np.arange(24, dtype=np.float32).reshape(4, 6)
    result = Results(
        boxes=None,
        orig_shape=(4, 6),
        anomaly_map=AnomalyMap(heatmap),
        anomaly_score=2.5,
        is_anomalous=True,
        path=str(tmp_path / "source.png"),
    )
    Image.new("RGB", (6, 4), "gray").save(result.path)
    assert len(result) == 1
    assert result.anomaly_map.array.dtype == np.float32
    assert result.summary() == [
        {
            "name": "anomaly",
            "score": 2.5,
            "is_anomalous": True,
            "map_min": 0.0,
            "map_max": 23.0,
        }
    ]
    assert json.loads(result.to_json())[0]["is_anomalous"] is True
    rendered = draw_anomaly_map(Image.new("RGB", (6, 4), "gray"), heatmap)
    assert rendered.size == (6, 4)
    assert result.plot().size == (6, 4)
    assert "anomalous" in result.verbose()
    assert result.save(tmp_path / "anomaly.png") == str(tmp_path / "anomaly.png")


def test_anomaly_dataset_layout_and_optional_mask(tmp_path):
    for relative in (
        "train/good/train.png",
        "test/good/good.png",
        "test/crack/bad.png",
        "ground_truth/crack/bad_mask.png",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L" if "mask" in relative else "RGB", (8, 8), 255).save(path)
    assert [path.name for path in resolve_good_training_images(tmp_path)] == ["train.png"]
    samples = resolve_anomaly_test_samples(tmp_path)
    assert [(path.parent.name, label) for path, label, _ in samples] == [
        ("crack", 1),
        ("good", 0),
    ]
    assert samples[0][2].name == "bad_mask.png"
    assert samples[1][2] is None


def test_anomaly_metrics_and_ties():
    assert binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert binary_auroc([0, 1], [0.5, 0.5]) == 0.5
    f1, threshold = best_f1([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert f1 == 1.0
    assert threshold == 0.8


def test_greedy_coreset_is_deterministic_and_bounded():
    features = torch.arange(100 * 12, dtype=torch.float32).reshape(100, 12)
    first = greedy_coreset(features, percent=10, projection_dim=4, seed=7)
    second = greedy_coreset(features, percent=10, projection_dim=4, seed=7)
    assert first.shape == (10, 12)
    assert torch.equal(first, second)
    with pytest.raises(ValueError, match=r"\[1, 25\]"):
        greedy_coreset(features, percent=0.5)


def test_patchcore_nearest_neighbor_scoring_without_backbone():
    net = PatchCoreNet.__new__(PatchCoreNet)
    torch.nn.Module.__init__(net)
    net.register_buffer("memory_bank", torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]))
    net.register_buffer("fitted", torch.tensor(True))
    net.reweight_neighbors = 3
    net.query_chunk_size = 2
    queries = torch.tensor([[0.1, 0.0], [1.8, 0.0], [3.0, 3.0]])
    distances, indices = net.nearest_distances(queries)
    assert distances.tolist() == pytest.approx([0.1, 0.2, np.sqrt(10.0)], rel=1e-5)
    assert indices.tolist() == [0, 1, 1]
    assert float(net._image_score(queries, distances, indices)) > 0


def test_patchcore_postprocess_and_checkpoint_fingerprint():
    output = {
        "patch_scores": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        "image_scores": torch.tensor(2.75),
    }
    parsed = postprocess(output, original_size=(10, 8), threshold=2.0, sigma=0)
    assert parsed["anomaly_map"].shape == (8, 10)
    assert parsed["anomaly_score"] == pytest.approx(2.75)
    assert parsed["is_anomalous"] is True
    state = {
        "memory_bank": torch.empty(3, 1536),
        "anomaly_threshold": torch.tensor(1.0),
        "fitted": torch.tensor(True),
    }
    assert LibrePatchCore.can_load(state)
    assert LibrePatchCore.detect_size(state) == "b"
