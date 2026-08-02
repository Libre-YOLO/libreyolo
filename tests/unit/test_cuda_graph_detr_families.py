"""Per-family CUDA-graph parity gates for the DETR-line eval forwards.

``SUPPORTS_CUDA_GRAPH`` is an opt-in that every family must re-earn with its
own parity evidence (models/base/model.py). This module is that evidence for
the RF100-VL campaign families whose validation is launch-bound: RF-DETR,
D-FINE, DEIMv2 and EC, plus YOLOv9-E2E, which inherits the flag from
LibreYOLO9 and therefore needs its own proof that the inherited capture path
is safe for the dual-assignment head's eval forward.

The CPU tests pin the opt-in surface and run everywhere. The CUDA tests gate
the load-bearing claim -- capture and replay change nothing, bit for bit --
at both the model level (capture_graph + _forward_graphed) and the validator
level (the exact wiring training-time validation uses: up-front capture,
replay-only loop policy, graph scope).
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.validation.base import BaseValidator
from libreyolo.validation.config import ValidationConfig

pytestmark = pytest.mark.unit

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture requires a CUDA device"
)


# Model classes are imported inside the builders, not at module scope: RF-DETR's
# DINOv2 backbone imports ``transformers``, which ships only with the ``rfdetr``
# extra, and a module-scope import would fail collection for every family in an
# environment without it.
def _build_rfdetr(device):
    pytest.importorskip(
        "transformers",
        reason="RF-DETR needs the rfdetr extra",
        exc_type=ImportError,
    )
    from libreyolo.models.rfdetr.model import LibreRFDETR

    # ``model_path={}`` is the random-init convention RF-DETR's other unit tests
    # use; ``None`` means "download pretrained" for this family.
    return LibreRFDETR(model_path={}, size="n", nb_classes=2, device=device)


def _build_dfine(device):
    from libreyolo.models.dfine.model import LibreDFINE

    return LibreDFINE(None, size="n", nb_classes=2, device=device)


def _build_deimv2(device):
    from libreyolo.models.deimv2.model import LibreDEIMv2

    return LibreDEIMv2(None, size="atto", nb_classes=2, device=device)


def _build_ec(device):
    from libreyolo.models.ec.model import LibreEC

    return LibreEC(None, size="s", nb_classes=2, device=device)


def _build_yolo9_e2e(device):
    from libreyolo.models.yolo9_e2e.model import LibreYOLO9E2E

    return LibreYOLO9E2E(None, size="t", device=device)


# (family, constructor, capture imgsz). Sizes are each family's smallest
# variant at its native eval resolution, so the captured shape is the one a
# real campaign validation would replay.
FAMILY_CASES = [
    ("rfdetr", _build_rfdetr, 384),
    ("dfine", _build_dfine, 640),
    ("deimv2", _build_deimv2, 320),
    ("ec", _build_ec, 640),
    ("yolo9_e2e", _build_yolo9_e2e, 320),
]

CASE_IDS = [case[0] for case in FAMILY_CASES]


def _tensors(obj):
    """Flatten a forward output to its tensors, whatever container it uses."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for item in obj for t in _tensors(item)]
    if isinstance(obj, dict):
        return [t for item in obj.values() for t in _tensors(item)]
    return []


def _assert_bit_identical(eager, graphed, family):
    eager_tensors, graph_tensors = _tensors(eager), _tensors(graphed)
    assert eager_tensors, f"{family} forward produced no tensors"
    assert len(eager_tensors) == len(graph_tensors)
    for index, (lhs, rhs) in enumerate(zip(eager_tensors, graph_tensors)):
        assert lhs.shape == rhs.shape
        assert torch.equal(lhs, rhs), (
            f"{family} graph replay diverged from eager at tensor {index}"
        )


# ---------------------------------------------------------------------------
# Opt-in surface (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family,build,imgsz", FAMILY_CASES, ids=CASE_IDS)
def test_family_opts_in(family, build, imgsz):
    model = build("cpu")
    assert type(model).SUPPORTS_CUDA_GRAPH is True
    assert model.graph_info()["supported"] is True


@pytest.mark.parametrize("family,build,imgsz", FAMILY_CASES, ids=CASE_IDS)
def test_cpu_forward_falls_back_without_raising(family, build, imgsz):
    """On CPU the runner must reject capture and return the eager result."""
    model = build("cpu")
    model.model.eval()
    torch.manual_seed(0)
    x = torch.rand(1, 3, imgsz, imgsz)
    with torch.no_grad():
        eager = model._forward(x)
        with model.cuda_graph_scope(True):
            fallback = model._forward_graphed(x)
    assert model.graph_info()["graph_count"] == 0
    _assert_bit_identical(eager, fallback, family)


# ---------------------------------------------------------------------------
# Capture/replay parity (CUDA)
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("family,build,imgsz", FAMILY_CASES, ids=CASE_IDS)
def test_capture_replay_is_bit_identical(family, build, imgsz):
    """The load-bearing claim, per family: a replayed graph equals eager."""
    model = build("cuda")
    model.model.eval()
    torch.manual_seed(0)
    x = torch.rand(2, 3, imgsz, imgsz, device="cuda")

    with torch.no_grad():
        eager = model._forward(x)
        model.capture_graph(imgsz=imgsz, batch=2)
        with model.cuda_graph_scope(True):
            graphed = model._forward_graphed(x)

    info = model.graph_info()
    assert info["graph_count"] == 1, f"{family} capture did not happen"
    assert sum(c["replays"] for c in info["captured"]) >= 1, (
        f"{family} ran eager instead of replaying"
    )
    _assert_bit_identical(eager, graphed, family)
    model.release_graphs()


# ---------------------------------------------------------------------------
# Validator wiring parity (CUDA): the path training-time validation takes.
# ---------------------------------------------------------------------------


class _FixedBatchValidator(BaseValidator):
    """Feeds one fixed CUDA batch through _inference; nothing else."""

    def __init__(self, model, config, batch):
        super().__init__(model=model, config=config)
        self._batch = batch
        self.last_preds = None

    def _setup_dataloader(self):
        return object()

    def _warmup_model(self, n_warmup: int = 3):
        pass

    def _init_metrics(self):
        pass

    def _run_validation(self):
        model = getattr(self.model, "model", None)
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            self.last_preds = self._inference(self._batch)

    def _preprocess_batch(self, batch):
        raise AssertionError("not used")

    def _postprocess_predictions(self, preds, batch):
        raise AssertionError("not used")

    def _update_metrics(self, preds, targets, img_info, img_ids=None):
        raise AssertionError("not used")

    def _compute_metrics(self):
        return {}


@requires_cuda
@pytest.mark.parametrize("family,build,imgsz", FAMILY_CASES, ids=CASE_IDS)
def test_validator_wiring_is_bit_identical(family, build, imgsz, tmp_path):
    """run() with cuda_graph=True (up-front capture, replay-only loop) must
    produce the exact eager forward output for every campaign family."""
    model = build("cuda")
    model.model.eval()
    torch.manual_seed(0)
    batch = torch.rand(2, 3, imgsz, imgsz, device="cuda")

    def run_once(cuda_graph):
        config = ValidationConfig(
            data="x.yaml",
            device="cuda",
            save_dir=str(tmp_path / "val"),
            verbose=False,
            cuda_graph=cuda_graph,
            batch_size=2,
            imgsz=imgsz,
        )
        validator = _FixedBatchValidator(model=model, config=config, batch=batch)
        with torch.no_grad():
            validator.run()
        return validator.last_preds

    eager = run_once(False)
    graphed = run_once(True)
    info = model.graph_info()
    assert info["graph_count"] >= 1, f"{family} capture must have happened"
    assert sum(c["replays"] for c in info["captured"]) >= 1, (
        f"{family} loop must replay the up-front capture, not run eager"
    )
    _assert_bit_identical(eager, graphed, family)
    model.release_graphs()
