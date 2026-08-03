"""Tests for the canonical ImageNet-1k class-name data used by classifiers."""

import pytest

from libreyolo.data.imagenet import (
    IMAGENET1K_NUM_CLASSES,
    imagenet1k_class_list,
    imagenet1k_names,
    imagenet1k_synset_list,
    imagenet1k_synset_to_index,
)

pytestmark = pytest.mark.unit


def test_class_list_has_1000_unique_indices():
    names = imagenet1k_class_list()
    assert len(names) == IMAGENET1K_NUM_CLASSES == 1000
    assert all(isinstance(n, str) and n for n in names)


def test_names_map_keys_are_contiguous_ids():
    names = imagenet1k_names()
    assert set(names) == set(range(1000))
    assert all(isinstance(v, str) and v for v in names.values())


def test_known_indices_match_standard_ordering():
    # Index order must match the wnid-sorted ordering of the timm-derived
    # checkpoints, anchored by a few well-known class ids.
    names = imagenet1k_names()
    assert names[0] == "tench"
    assert names[258] == "Samoyed"
    assert names[999] == "toilet tissue"


def test_names_returns_fresh_mutable_copy():
    a = imagenet1k_names()
    a[258] = "mutated"
    b = imagenet1k_names()
    assert b[258] == "Samoyed"


def test_synsets_match_standard_classifier_indices():
    synsets = imagenet1k_synset_list()
    assert len(synsets) == len(set(synsets)) == IMAGENET1K_NUM_CLASSES
    assert synsets[0] == "n01440764"
    assert synsets[258] == "n02111889"
    assert synsets[999] == "n15075141"

    mapping = imagenet1k_synset_to_index()
    assert mapping["n01440764"] == 0
    assert mapping["n02111889"] == 258
    assert mapping["n15075141"] == 999
