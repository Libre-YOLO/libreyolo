"""Equivalence tests for the vectorized GO-LSD match-union (`_get_go_indices`).

The upstream implementation walked the unique (query, gt) pairs with two
`.item()` calls per pair — about 1,200 device syncs per training step for the
DETR families. The vectorized form batches that into one `.tolist()` transfer.
These tests pin the output to a reference re-implementation of the original
per-element loop, over randomized inputs, for every class that carries the
method (DFINECriterion, DEIMCriterion, ECPoseCriterion — ECCriterion inherits
D-FINE's).
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.deim.loss import DEIMCriterion
from libreyolo.models.dfine.loss import DFINECriterion
from libreyolo.models.ec.pose_loss import ECPoseCriterion

pytestmark = pytest.mark.unit


def _reference_go_indices(indices, indices_aux_list):
    """Upstream D-FINE loop, verbatim semantics: first col per row wins,
    iterating unique pairs in count-descending order."""
    results = []
    for indices_aux in indices_aux_list:
        indices = [
            (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
            for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
        ]
    for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
        unique, counts = torch.unique(ind, return_counts=True, dim=0)
        unique_sorted = unique[torch.argsort(counts, descending=True)]
        column_to_row = {}
        for idx in unique_sorted:
            row_idx, col_idx = idx[0].item(), idx[1].item()
            if row_idx not in column_to_row:
                column_to_row[row_idx] = col_idx
        results.append(
            (
                torch.tensor(list(column_to_row.keys()), device=ind.device).long(),
                torch.tensor(list(column_to_row.values()), device=ind.device).long(),
            )
        )
    return results


def _random_match(generator, num_queries=30, max_gt=6):
    n = int(torch.randint(0, max_gt + 1, (1,), generator=generator))
    rows = torch.randperm(num_queries, generator=generator)[:n]
    cols = torch.randperm(max(n, 1), generator=generator)[:n]
    return rows.long(), cols.long()


def _go_indices_fn(criterion_cls):
    # _get_go_indices does not touch self, so bind-free invocation keeps the
    # test independent of each criterion's constructor plumbing.
    return lambda indices, aux: criterion_cls._get_go_indices(None, indices, aux)


@pytest.mark.parametrize(
    "criterion_cls",
    [DFINECriterion, DEIMCriterion, ECPoseCriterion],
    ids=["dfine", "deim", "ecpose"],
)
def test_go_indices_matches_upstream_loop(criterion_cls):
    generator = torch.Generator().manual_seed(0)
    fn = _go_indices_fn(criterion_cls)
    for _ in range(25):
        batch = int(torch.randint(1, 5, (1,), generator=generator))
        num_layers = int(torch.randint(1, 8, (1,), generator=generator))
        indices = [_random_match(generator) for _ in range(batch)]
        aux_list = [
            [_random_match(generator) for _ in range(batch)]
            for _ in range(num_layers)
        ]
        got = fn(indices, aux_list)
        expected = _reference_go_indices(indices, aux_list)
        assert len(got) == len(expected) == batch
        for (g_rows, g_cols), (e_rows, e_cols) in zip(got, expected):
            assert torch.equal(g_rows, e_rows)
            assert torch.equal(g_cols, e_cols)
            assert g_rows.dtype == torch.int64
            assert g_cols.dtype == torch.int64


@pytest.mark.parametrize(
    "criterion_cls",
    [DFINECriterion, DEIMCriterion, ECPoseCriterion],
    ids=["dfine", "deim", "ecpose"],
)
def test_go_indices_empty_matches(criterion_cls):
    empty = (torch.zeros(0).long(), torch.zeros(0).long())
    got = _go_indices_fn(criterion_cls)([empty], [[empty]])
    assert len(got) == 1
    assert got[0][0].numel() == 0
    assert got[0][1].numel() == 0


def test_go_indices_does_not_sync_per_element(monkeypatch):
    """Guard the fix: a per-element `.item()` regression re-introduces ~1,200
    device syncs per step. The batched form calls `.item()` zero times."""
    calls = {"n": 0}
    original = torch.Tensor.item

    def counting_item(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(torch.Tensor, "item", counting_item)
    generator = torch.Generator().manual_seed(1)
    indices = [_random_match(generator) for _ in range(4)]
    aux_list = [[_random_match(generator) for _ in range(4)] for _ in range(6)]
    DFINECriterion._get_go_indices(None, indices, aux_list)
    assert calls["n"] == 0, (
        f"_get_go_indices called .item() {calls['n']} times; the transfer "
        "must stay batched (.tolist once per image)"
    )
