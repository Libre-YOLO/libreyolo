"""LibreDOMEDETR unit suite: routing, shapes, and the MWAS reformulation.

The upstream-parity proof lives in ``weights/parity_domedetr.py`` (it needs the
upstream checkout and the published checkpoints, so it is not a CI test).
``tests/e2e/test_val_coco128.py``'s mAP gate does not apply to this family:
Dome-DETR has no COCO checkpoint, only AI-TOD-V2 and VisDrone.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.deim.model import LibreDEIM
from libreyolo.models.deimv2.model import LibreDEIMv2
from libreyolo.models.dfine.model import LibreDFINE
from libreyolo.models.domedetr.model import LibreDOMEDETR
from libreyolo.models.domedetr.nn import DEC_NUM_LAYERS, LibreDOMEDETRModel
from libreyolo.models.rtdetrv4.model import LibreRTDETRv4


pytestmark = [pytest.mark.unit, pytest.mark.domedetr]


def _domedetr_state_dict(size: str = "s", variant: str = "aitod") -> dict:
    nc = 9 if variant == "aitod" else 12
    return LibreDOMEDETRModel(config=size, nb_classes=nc, variant=variant).state_dict()


def _lazy_rfdetr():
    """RF-DETR registers lazily behind the transformers dep.

    Its discriminator is deliberately broad and matches Dome-DETR's
    ``decoder.denoising_class_embed.weight``, so the rejection has to be
    explicit rather than left to registry order.
    """
    from libreyolo.models.rfdetr.model import LibreRFDETR

    return LibreRFDETR


def _dfine_like_state_dict() -> dict:
    """A D-FINE-lineage state dict: carries pre_bbox_head but no DeFE."""
    from libreyolo.models.dfine.nn import LibreDFINEModel

    return LibreDFINEModel(config="s", nb_classes=80).state_dict()


# -- routing ------------------------------------------------------------------


def test_can_load_accepts_domedetr():
    assert LibreDOMEDETR.can_load(_domedetr_state_dict()) is True


def test_can_load_rejects_dfine():
    assert LibreDOMEDETR.can_load(_dfine_like_state_dict()) is False


@pytest.mark.parametrize(
    "sibling",
    [LibreDFINE, LibreDEIM, LibreDEIMv2, LibreRTDETRv4, _lazy_rfdetr],
    ids=["dfine", "deim", "deimv2", "rtdetrv4", "rfdetr"],
)
def test_dfine_lineage_rejects_domedetr(sibling):
    """The whole D-FINE lineage must refuse Dome-DETR checkpoints.

    Dome-DETR is a D-FINE derivative and carries ``decoder.pre_bbox_head.``,
    the key D-FINE discriminates on. Without an explicit rejection LibreDFINE
    would claim these files and load a subset of their tensors.
    """
    cls = sibling() if callable(sibling) and not hasattr(sibling, "can_load") else sibling
    assert cls.can_load(_domedetr_state_dict()) is False


def test_only_domedetr_claims_a_domedetr_checkpoint():
    """Exactly one registered family may claim these tensors.

    Registry order cannot be the safeguard here: importing LibreDOMEDETR pulls
    in ``models.dfine`` for the shared decoder stack, so LibreDFINE registers
    first no matter where the import sits. The bidirectional ``can_load``
    rejection is what actually decides it, so assert the outcome directly.
    """
    import libreyolo.models  # noqa: F401  (registers every family)
    from libreyolo.models.base import BaseModel

    state_dict = _domedetr_state_dict()
    claimants = {
        cls.FAMILY for cls in BaseModel._registry if cls.can_load(state_dict)
    }
    assert claimants == {"domedetr"}


def test_filename_detection_with_variant_suffix():
    assert LibreDOMEDETR.detect_size_from_filename("LibreDOMEDETRs-visdrone.pt") == "s"
    assert LibreDOMEDETR.detect_size_from_filename("LibreDOMEDETRl-aitod.pt") == "l"
    # Detect has no task suffix, so the variant must not be read as one.
    assert LibreDOMEDETR.detect_task_from_filename("LibreDOMEDETRm-aitod.pt") in (
        None,
        "detect",
    )


@pytest.mark.parametrize(
    ("filename", "nb_classes", "expected"),
    [
        ("LibreDOMEDETRs-visdrone.pt", 80, "visdrone"),
        ("LibreDOMEDETRs-aitod.pt", 80, "aitod"),
        (None, 12, "visdrone"),
        (None, 9, "aitod"),
    ],
)
def test_weight_variant_resolution(filename, nb_classes, expected):
    resolved = LibreDOMEDETR._resolve_weight_variant(
        explicit=None, model_path=filename, nb_classes=nb_classes
    )
    assert resolved == expected


def test_weight_variant_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown weight_variant"):
        LibreDOMEDETR._resolve_weight_variant(
            explicit="coco", model_path=None, nb_classes=9
        )


# -- shapes -------------------------------------------------------------------


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_size_detection_from_state_dict(size):
    assert LibreDOMEDETR.detect_size(_domedetr_state_dict(size)) == size


@pytest.mark.parametrize(("variant", "nc"), [("aitod", 9), ("visdrone", 12)])
def test_nb_classes_detection(variant, nc):
    assert LibreDOMEDETR.detect_nb_classes(_domedetr_state_dict("s", variant)) == nc


def test_decoder_depth_is_per_size_and_variant():
    """L is 4 decoder layers on AI-TOD-V2 but 6 on VisDrone.

    Keying depth off the size alone silently builds the wrong model, and the
    state dict then loads non-strictly with two layers left at init.
    """
    assert DEC_NUM_LAYERS[("l", "aitod")] == 4
    assert DEC_NUM_LAYERS[("l", "visdrone")] == 6

    aitod = LibreDOMEDETRModel(config="l", nb_classes=9, variant="aitod")
    visdrone = LibreDOMEDETRModel(config="l", nb_classes=12, variant="visdrone")
    assert len(aitod.decoder.decoder.layers) == 4
    assert len(visdrone.decoder.decoder.layers) == 6


def test_forward_shape_and_query_budget():
    torch.manual_seed(0)
    model = LibreDOMEDETRModel(config="s", nb_classes=12, variant="visdrone").eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 800, 800))

    assert set(out) >= {"pred_logits", "pred_boxes"}
    n_queries = out["pred_logits"].shape[1]
    assert out["pred_logits"].shape == (1, n_queries, 12)
    assert out["pred_boxes"].shape == (1, n_queries, 4)
    # PAQI keeps the top min_num_select unconditionally and never exceeds
    # max_num_select, whatever the density map says.
    assert 250 <= n_queries <= 500
    assert out["batch_queries_num"] == [n_queries]


def test_batch_padding_uses_negative_logits():
    """Short rows in a mixed batch must not surface as 0.5-confidence boxes."""
    torch.manual_seed(0)
    model = LibreDOMEDETRModel(config="s", nb_classes=12, variant="visdrone").eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 800, 800))

    counts = out["batch_queries_num"]
    n_queries = out["pred_logits"].shape[1]
    assert n_queries == max(counts)
    for b, count in enumerate(counts):
        if count < n_queries:
            padded = out["pred_logits"][b, count:]
            assert torch.all(padded < -1e30), "padding must be far below any real logit"


# -- MWAS reformulation -------------------------------------------------------


def test_mwas_static_path_matches_gather_path():
    """The static path is algebraically the same computation, not a fallback.

    It keeps every window in the tensor and hides the empty ones from the
    cross-window attention with a key-padding mask instead of gathering the
    occupied ones. Softmax over the padded key set reassociates the
    floating-point sums, so this pins the gap at ~1e-5 rather than asserting
    bit-equality.
    """
    torch.manual_seed(0)
    model = LibreDOMEDETRModel(config="s", nb_classes=9, variant="aitod").eval()
    x = torch.randn(1, 3, 800, 800)
    processor = model.encoder.mwas_processor

    with torch.no_grad():
        processor.force_static_path = False
        gather = model.encoder(model.backbone(x), img_inputs=x)
        processor.force_static_path = True
        static = model.encoder(model.backbone(x), img_inputs=x)

    assert torch.equal(
        gather["defe"]["defe_window_mask"], static["defe"]["defe_window_mask"]
    )
    worst = max(
        (a - b).abs().max().item()
        for a, b in zip(gather["feats"], static["feats"])
    )
    assert worst < 1e-4, f"static MWAS path diverged by {worst}"


# -- scope --------------------------------------------------------------------


def test_training_raises_with_a_reason():
    model = LibreDOMEDETR(model_path=None, size="s", nb_classes=9)
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="coco128.yaml")


def test_export_is_blocked_with_a_reason():
    """Better a clear refusal than a graph that is only valid for one image."""
    model = LibreDOMEDETR(model_path=None, size="s", nb_classes=9)
    with pytest.raises(NotImplementedError, match="query count per image"):
        model.export(format="onnx")


def test_enrolled_in_a_rollout_group():
    from libreyolo.models.registry import MODEL_GROUPS

    assert MODEL_GROUPS["domedetr"] == "g3"


def test_is_nms_free_family():
    """PAQI's NMS runs inside the decoder; backends must not re-suppress."""
    from libreyolo.backends.base import _is_nms_free_family

    assert _is_nms_free_family("domedetr") is True
