"""Behavior tests for the `enroll` (face-gallery build) command."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import typer
from PIL import Image
from typer.testing import CliRunner

from libreyolo.cli.commands import special as special_module
from libreyolo.cli.commands.special import enroll_cmd
from libreyolo.cli.parsing import KeyValueCommand
from libreyolo.utils.results import Embeddings

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer()
    app.command("enroll", cls=KeyValueCommand)(enroll_cmd)
    return app


class _FakeEmbedModel:
    """Embeds each image as a deterministic unit vector per identity folder."""

    task = "embed"
    family = "facerec"
    model_path = None

    def __init__(self):
        self.face_detector = None

    def __call__(self, source, **_):
        identity = Path(source).parent.name
        rng = np.random.RandomState(abs(hash(identity)) % (2**31))
        vec = rng.rand(8).astype(np.float32)
        return SimpleNamespace(
            embeddings=Embeddings(vec[None, :]), boxes=None
        )


def _patch(monkeypatch, fake):
    monkeypatch.setattr(special_module, "resolve_model_or_exit", lambda out, m: m)
    monkeypatch.setattr(
        special_module,
        "load_model_or_exit",
        lambda out, *, model, model_path, device, task=None: fake,
    )


def _people_tree(tmp_path):
    people = tmp_path / "people"
    for identity, count in (("alice", 2), ("bob", 1)):
        d = people / identity
        d.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (20, 20)).save(d / f"{i}.jpg")
    (people / "carol").mkdir()  # no images -> skipped
    return people


def test_enroll_builds_gallery(monkeypatch, tmp_path):
    from libreyolo.models.facerec import FaceGallery

    people = _people_tree(tmp_path)
    gallery_path = tmp_path / "team.gallery.npz"
    _patch(monkeypatch, _FakeEmbedModel())

    result = runner.invoke(
        _make_app(),
        [f"model=facerec-l", f"source={people}", f"gallery={gallery_path}", "--json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["identities"] == 2
    assert data["references"] == 3
    assert data["enrolled"] == {"alice": 2, "bob": 1}
    assert data["skipped"] == ["carol"]

    loaded = FaceGallery.load(gallery_path)
    assert sorted(loaded.identities) == ["alice", "bob"]


def test_enroll_appends_to_existing_gallery(monkeypatch, tmp_path):
    from libreyolo.models.facerec import FaceGallery

    people = _people_tree(tmp_path)
    gallery_path = tmp_path / "team.gallery.npz"
    _patch(monkeypatch, _FakeEmbedModel())
    app = _make_app()

    first = runner.invoke(
        app, [f"model=facerec-l", f"source={people}", f"gallery={gallery_path}", "--json"]
    )
    assert first.exit_code == 0, first.output

    dave = people / "dave"
    dave.mkdir()
    Image.new("RGB", (20, 20)).save(dave / "0.jpg")
    second = runner.invoke(
        app, [f"model=facerec-l", f"source={people}", f"gallery={gallery_path}", "--json"]
    )
    assert second.exit_code == 0, second.output

    loaded = FaceGallery.load(gallery_path)
    assert "dave" in loaded.identities
    assert len(loaded.identities) == 3


def test_enroll_rejects_flat_folder(monkeypatch, tmp_path):
    flat = tmp_path / "flat"
    flat.mkdir()
    Image.new("RGB", (20, 20)).save(flat / "img.jpg")
    _patch(monkeypatch, _FakeEmbedModel())

    result = runner.invoke(
        _make_app(),
        [f"model=facerec-l", f"source={flat}", f"gallery={tmp_path / 'g.npz'}", "--json"],
    )
    assert result.exit_code != 0
