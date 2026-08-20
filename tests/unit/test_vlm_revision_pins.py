"""Offline contract tests for immutable LibreVLM base snapshots."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from libreyolo.models.modus.model import LibreMODUS
from libreyolo.models.sensenova.model import LibreSenseNovaVision
from libreyolo.models.vlm import _ALIASES, _LAZY_ALIASES, _MODUS_ALIASES
from libreyolo.models.vlm.florence2 import LibreFlorence2
from libreyolo.models.vlm.gemma4 import LibreGemma4
from libreyolo.models.vlm.internvl3 import LibreInternVL3
from libreyolo.models.vlm.kosmos2 import LibreKosmos2
from libreyolo.models.vlm.lfm2 import LibreLFM2VL
from libreyolo.models.vlm.locateanything import LibreLocateAnything
from libreyolo.models.vlm.moondream import LibreMoondream
from libreyolo.models.vlm.northmicro import LibreNorthMicroVision
from libreyolo.models.vlm.qwen3vl import LibreQwen3VL
from libreyolo.models.vlm.smolvlm import LibreSmolVLM2

pytestmark = pytest.mark.unit

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

# MODUS is deliberately absent: its research-only checkpoint uses the separate
# authenticated external resolver in ``models/modus/weights.py``.
_PINNED_VARIANTS = (
    (LibreQwen3VL, "2b", "89644892e4d85e24eaac8bacfd4f463576704203"),
    (LibreQwen3VL, "4b", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
    (LibreQwen3VL, "8b", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
    (LibreLFM2VL, "450m", "fc6221ca597f3315e4f82fc2df606783267b34ba"),
    (LibreLFM2VL, "1.6b", "919fde3d022e3f90a4716006f993938ee8c2eb97"),
    (LibreLFM2VL, "3b", "5a414ead75d45db003906d06fb62bd5b6846cec0"),
    (
        LibreNorthMicroVision,
        "2.4b",
        "8be3368e3ad675d84c162d458b4499aadefc3aeb",
    ),
    (LibreInternVL3, "1b", "014c0583a0d4bedf29fbe2dbff4f865eb998e171"),
    (LibreInternVL3, "2b", "cb57a075cb75a2e6d1b668b128d48bb00ae321d2"),
    (LibreInternVL3, "8b", "259a3b64a14623c0ec91a045cb43f7c5af5fa6af"),
    (LibreSmolVLM2, "2.2b", "482adb537c021c86670beed01cd58990d01e72e4"),
    (LibreSmolVLM2, "500m", "7b375e1b73b11138ff12fe22c8f2822d8fe03467"),
    (LibreFlorence2, "base", "00921df66db728a9ceb750f5eca43e5c203a2051"),
    (LibreFlorence2, "large", "4271c66b88cdbc05735372ec13b2360108de5317"),
    (LibreKosmos2, "224", "e91cfbcb4ce051b6a55bfb5f96165a3bbf5eb82c"),
    (
        LibreLocateAnything,
        "3b",
        "c32291ca5e996f5a7a485845b4f57a233936bba0",
    ),
    (LibreGemma4, "e2b", "507bec1827280677ff6332406cc6e2500670b0a6"),
    (LibreGemma4, "e4b", "80662d5dc7f69bfe5d1b5be53da3558b8d481998"),
    (LibreMoondream, "2", "148fe3489ad456f3b0e5301d684116eb3ad2bece"),
    (LibreMoondream, "3", "27c8082b22b45de224c30431b3961ef8ee6c740e"),
    (
        LibreSenseNovaVision,
        "7b",
        "bd67637b1536f1149fb31a05568c4b6b2cf2d032",
    ),
)
_EXTERNAL_VARIANTS = ((LibreMODUS, "14b-a7b"),)


def _variant_id(case) -> str:
    family_cls, size, _revision = case
    return f"{family_cls.FAMILY}-{size}"


def test_revision_roster_covers_all_22_registered_variants():
    registered = set(_ALIASES.values())
    registered.update((LibreSenseNovaVision, size) for size in _LAZY_ALIASES.values())
    registered.update((LibreMODUS, size) for size in _MODUS_ALIASES.values())
    expected = {(family_cls, size) for family_cls, size, _ in _PINNED_VARIANTS}
    expected.update(_EXTERNAL_VARIANTS)

    assert len(registered) == 22
    assert registered == expected


@pytest.mark.parametrize(
    ("family_cls", "size", "expected_revision"),
    _PINNED_VARIANTS,
    ids=[_variant_id(case) for case in _PINNED_VARIANTS],
)
def test_every_managed_base_snapshot_has_immutable_revision(
    family_cls, size, expected_revision
):
    assert set(family_cls.HF_REVISIONS) == set(family_cls.HF_REPOS)
    assert family_cls.HF_REVISIONS[size] == expected_revision
    assert _COMMIT_SHA.fullmatch(expected_revision)


def test_modus_keeps_its_external_only_download_path():
    from libreyolo.models.vlm.base import LibreVLMModel

    assert LibreMODUS._ensure_weights is not LibreVLMModel._ensure_weights


@pytest.mark.parametrize(
    ("family_cls", "size", "expected_revision"),
    _PINNED_VARIANTS,
    ids=[_variant_id(case) for case in _PINNED_VARIANTS],
)
def test_snapshot_download_receives_each_pin_without_model_construction(
    family_cls, size, expected_revision, tmp_path, monkeypatch
):
    import huggingface_hub

    observed = {}

    def fake_snapshot_download(repo, *, local_dir, ignore_patterns, revision):
        observed.update(
            repo=repo,
            local_dir=local_dir,
            ignore_patterns=ignore_patterns,
            revision=revision,
        )
        Path(local_dir).mkdir(parents=True)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.chdir(tmp_path)

    model = object.__new__(family_cls)
    model.size = size
    model._notify_license_once = lambda: None
    completeness = iter((False, True))
    model._snapshot_complete = lambda *_args, **_kwargs: next(completeness)

    resolved = Path(model._ensure_weights())

    assert observed["repo"] == family_cls.HF_REPOS[size]
    assert observed["revision"] == expected_revision
    assert observed["local_dir"] == str(resolved)
