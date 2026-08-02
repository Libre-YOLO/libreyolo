"""Bidirectional ``can_load`` rejection between LW-DETR and its relatives.

LW-DETR is RF-DETR's direct ancestor: Roboflow forked this architecture and
swapped the plain-ViT encoder for DINOv2, so the two families share decoder,
projector, and two-stage head key names. D-FINE, DEIM, and RT-DETR are further
cousins that share the broader DETR vocabulary. Every pair is checked in both
directions — a one-way check has historically let a family quietly steal
another's checkpoints.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit


def _lwdetr_state_dict(size: str = "t") -> dict:
    from libreyolo import LibreLWDETR

    return LibreLWDETR(None, size=size, device="cpu").model.state_dict()


def _rfdetr_state_dict() -> dict:
    """Minimal RF-DETR key set.

    Built by hand rather than by constructing ``LibreRFDETR``: that pulls the
    DINOv2 backbone over the network. ``can_load`` only inspects key names, and
    these mirror a real RF-DETR checkpoint (note the doubled ``encoder.encoder``
    from the HuggingFace backbone, which LW-DETR never produces).
    """
    return {
        "backbone.0.encoder.encoder.embeddings.cls_token": torch.zeros(1, 1, 256),
        "backbone.0.encoder.encoder.embeddings.position_embeddings": torch.zeros(
            1, 1370, 256
        ),
        "backbone.0.projector.stages.0.0.cv1.conv.weight": torch.zeros(256, 768, 1, 1),
        "transformer.decoder.query_embed.weight": torch.zeros(300, 256),
        "transformer.enc_out_class_embed.0.weight": torch.zeros(91, 256),
        "transformer.enc_out_bbox_embed.0.layers.0.weight": torch.zeros(256, 256),
        "class_embed.weight": torch.zeros(91, 256),
        "class_embed.bias": torch.zeros(91),
        "bbox_embed.layers.0.weight": torch.zeros(256, 256),
    }


# Sibling families that construct offline, keyed by the size to build.
_OFFLINE_SIBLINGS = [
    ("LibreDFINE", "n"),
    ("LibreDEIM", "n"),
    ("LibreRTDETR", "r18"),
]


def _sibling_state_dict(class_name: str, size: str) -> dict:
    import libreyolo

    cls = getattr(libreyolo, class_name)
    return cls(None, size=size, device="cpu").model.state_dict()


@pytest.mark.parametrize(("class_name", "size"), _OFFLINE_SIBLINGS)
def test_lwdetr_rejects_sibling_state_dict(class_name, size):
    from libreyolo import LibreLWDETR

    assert LibreLWDETR.can_load(_sibling_state_dict(class_name, size)) is False


@pytest.mark.parametrize(("class_name", "size"), _OFFLINE_SIBLINGS)
def test_sibling_rejects_lwdetr_state_dict(class_name, size):
    import libreyolo

    cls = getattr(libreyolo, class_name)
    assert cls.can_load(_lwdetr_state_dict()) is False


def test_lwdetr_rejects_rfdetr_state_dict():
    from libreyolo import LibreLWDETR

    assert LibreLWDETR.can_load(_rfdetr_state_dict()) is False


def test_rfdetr_rejects_lwdetr_state_dict():
    pytest.importorskip("transformers")
    from libreyolo.models.rfdetr.model import LibreRFDETR

    assert LibreRFDETR.can_load(_lwdetr_state_dict()) is False
    # Guard the fixture itself: it must still look like RF-DETR, or the
    # rejection above would pass for the wrong reason.
    assert LibreRFDETR.can_load(_rfdetr_state_dict()) is True


def test_lwdetr_weights_do_not_trigger_rfdetr_registration():
    """LW-DETR must not drag in the optional ``transformers`` dependency.

    Its checkpoints carry ``enc_out_class_embed`` / ``enc_out_bbox_embed``,
    which RF-DETR forked from it and which the lazy-registration probe keys on.
    Without an explicit guard, loading LW-DETR weights on an install without the
    ``rfdetr`` extra raises ModuleNotFoundError.
    """
    from libreyolo.models import _needs_rfdetr_registration

    state_dict = _lwdetr_state_dict()
    assert any("enc_out_class_embed" in k for k in state_dict)
    assert _needs_rfdetr_registration(state_dict) is False
    assert _needs_rfdetr_registration(_rfdetr_state_dict()) is True


@pytest.mark.parametrize("size", ["t", "s"])
def test_factory_routes_lwdetr_checkpoint_to_lwdetr(tmp_path, size):
    from libreyolo import LibreLWDETR, LibreYOLO

    src = LibreLWDETR(None, size=size, device="cpu")
    ckpt = tmp_path / f"LibreLWDETR{size}.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            src.model.state_dict(),
            model_family="lwdetr",
            size=size,
            task="detect",
            nc=80,
            names={i: f"class_{i}" for i in range(80)},
            imgsz=640,
        ),
        ckpt,
    )

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "lwdetr"
    assert loaded.size == size
