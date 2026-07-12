"""Offline tests for the native PicoSAM3 port."""

from __future__ import annotations

from dataclasses import replace
import json
import threading

import pytest
import torch
from PIL import Image

from libreyolo.models.picosam3.model import LibrePicoSAM3
from libreyolo.models.picosam3.nn import PicoSAM3Network
from libreyolo.models.picosam3.preprocess import padded_square_roi, preprocess_roi
from libreyolo.models.sam.model import _ALIASES, _PICO_SAM3

pytestmark = [pytest.mark.unit, pytest.mark.sam]


def _bare_picosam3() -> LibrePicoSAM3:
    model = object.__new__(LibrePicoSAM3)
    model.model = PicoSAM3Network().eval()
    for parameter in model.model.parameters():
        parameter.data.zero_()
    model.model.refine[3].bias.data.fill_(1.0)
    model.processor = None
    model.device = torch.device("cpu")
    model._model_dtype = torch.float32
    model._default_multimask = False
    model.names = {0: "object"}
    model.size = "pico"
    model.task = "segment"
    model._clear_image_state()
    return model


def test_top_level_export_resolves_lazily():
    from libreyolo import LibrePicoSAM3 as Exported

    assert Exported is LibrePicoSAM3


def test_aliases_resolve_without_download():
    assert _ALIASES["picosam3"] == (_PICO_SAM3, "pico")
    assert _ALIASES["picosam3-pico"] == (_PICO_SAM3, "pico")


def test_network_shape_and_parameter_count():
    model = PicoSAM3Network().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_371_418
    with torch.no_grad():
        output = model(torch.zeros((2, 3, 96, 96)))
    assert output.shape == (2, 1, 96, 96)


def test_upstream_roi_geometry_and_preprocess_are_deterministic():
    image = Image.new("RGB", (32, 24), color=(10, 20, 30))
    roi = padded_square_roi([8, 6, 24, 18], 32, 24)
    tensor = preprocess_roi(image, roi)

    assert roi == (6, 2, 25, 21)
    assert tensor.shape == (3, 96, 96)
    assert tensor[:, 0, 0].tolist() == pytest.approx(
        [-1.9466564655, -1.6855741739, -1.2815686464]
    )


def test_box_prompt_places_mask_back_into_original_image():
    model = _bare_picosam3()
    image = Image.new("RGB", (32, 24), color="white")

    result = model.predict(image, bboxes=[8, 6, 24, 18], conf=0.0)

    assert result.masks.data.shape == (1, 24, 32)
    assert result.masks.data[0, 2:21, 6:25].all()
    assert not result.masks.data[0, :2].any()
    assert result.boxes.xyxy.tolist() == [[6.0, 2.0, 24.0, 20.0]]


def test_multiple_boxes_are_batched_and_set_image_is_reused():
    model = _bare_picosam3().set_image(Image.new("RGB", (40, 30), color="white"))
    try:
        result = model.predict(bboxes=[[2, 2, 15, 20], [20, 5, 35, 25]])
        assert len(result.masks) == 2
    finally:
        model.reset_image()


@pytest.mark.parametrize("max_det", [-1, 0, 1.5, True])
def test_predict_rejects_invalid_max_det_before_inference(max_det):
    with pytest.raises(ValueError, match="integer >= 1"):
        _bare_picosam3().predict(
            Image.new("RGB", (16, 16)),
            bboxes=[1, 1, 8, 8],
            max_det=max_det,
        )


def test_predict_serializes_forward_with_device_changes():
    model = _bare_picosam3()
    forward_started = threading.Event()
    allow_forward = threading.Event()
    device_call_started = threading.Event()
    device_call_done = threading.Event()
    errors = []

    class BlockingModel(torch.nn.Module):
        def forward(self, batch):
            forward_started.set()
            if not allow_forward.wait(timeout=2):
                raise TimeoutError("test did not release PicoSAM3 forward")
            return torch.ones(
                (batch.shape[0], 1, 96, 96),
                device=batch.device,
                dtype=batch.dtype,
            )

    model.model = BlockingModel().eval()

    def run_predict():
        try:
            model.predict(
                Image.new("RGB", (16, 16)),
                bboxes=[1, 1, 8, 8],
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def set_device():
        device_call_started.set()
        try:
            model._set_device("cpu")
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            device_call_done.set()

    prediction = threading.Thread(target=run_predict)
    prediction.start()
    assert forward_started.wait(timeout=2)

    device_change = threading.Thread(target=set_device)
    device_change.start()
    assert device_call_started.wait(timeout=2)
    assert not device_call_done.wait(timeout=0.05), (
        "device mutation escaped while PicoSAM3 prediction held the session lock"
    )

    allow_forward.set()
    prediction.join(timeout=2)
    device_change.join(timeout=2)
    assert not prediction.is_alive()
    assert not device_change.is_alive()
    assert not errors


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"points": [4, 4]}, "supports only bboxes"),
        ({"text": "dog"}, "supports only bboxes"),
        ({"masks": torch.zeros(1)}, "supports only bboxes"),
        ({"multimask": True}, "does not support multimask"),
        ({}, "requires bboxes"),
    ],
)
def test_unsupported_prompt_contract_fails_clearly(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _bare_picosam3().predict(Image.new("RGB", (16, 16)), **kwargs)


def test_epoch1_picosam2_checkpoint_is_rejected(tmp_path):
    checkpoint = {"output_head.weight": torch.zeros((1, 40, 1, 1))}
    torch.save(checkpoint, tmp_path / LibrePicoSAM3.WEIGHT_FILE)
    model = object.__new__(LibrePicoSAM3)
    model.size = "pico"
    model._ensure_weights = lambda: str(tmp_path)

    with pytest.raises(ValueError, match="older PicoSAM2 architecture"):
        model._init_model()


def test_missing_snapshot_honors_manifest_publication(monkeypatch, tmp_path):
    from libreyolo.models.manifest import PublicationState, get_artifact_spec
    from libreyolo.utils.download import WeightPublicationError

    artifact = get_artifact_spec("picosam3", "pico", "segment")
    assert artifact is not None
    local_only = replace(
        artifact,
        publication=PublicationState.CONFIG_ONLY,
        download_kind="none",
        download_url=None,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "libreyolo.models.picosam3.model.get_artifact_spec",
        lambda *args, **kwargs: local_only,
    )
    model = object.__new__(LibrePicoSAM3)
    model.size = "pico"

    with pytest.raises(WeightPublicationError, match="public snapshot route"):
        model._ensure_weights()


def test_conversion_round_trip_writes_provenance_metadata(tmp_path):
    from weights.convert_picosam3_weights import convert

    source = tmp_path / "PicoSAM3_SAM3_student_best.pt"
    output = tmp_path / "LibrePicoSAM3pico.pt"
    torch.save(PicoSAM3Network().state_dict(), source)
    convert(source, output)
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)

    assert checkpoint["model_family"] == "picosam3"
    assert checkpoint["size"] == "pico"
    assert checkpoint["task"] == "segment"
    assert checkpoint["imgsz"] == 96
    assert checkpoint["license"] == "Apache-2.0"
    PicoSAM3Network().load_state_dict(checkpoint["model"], strict=True)


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"half": True}, NotImplementedError, "half=True"),
        ({"int8": True}, NotImplementedError, "int8=True"),
        (
            {"int8": True, "fraction": 0.5, "allow_download_scripts": False},
            NotImplementedError,
            "int8=True",
        ),
        ({"batch": 0}, ValueError, "positive integer"),
        ({"batch": True}, ValueError, "positive integer"),
        ({"imgsz": 128}, ValueError, "requires imgsz=96"),
        ({"imgsz": (96, 95)}, ValueError, "requires imgsz=96"),
    ],
)
def test_export_rejects_unsupported_requests(tmp_path, kwargs, error, match):
    output = tmp_path / "should-not-exist.onnx"
    with pytest.raises(error, match=match):
        _bare_picosam3().export(output_path=output, **kwargs)
    assert not output.exists()


def test_raw_onnx_export_matches_pytorch(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = _bare_picosam3()
    output = tmp_path / "picosam3.onnx"
    result = model.export(
        output_path=output,
        opset=None,
        simplify=False,
        dynamic=False,
        half=False,
        int8=False,
        imgsz=(96, 96),
        batch=2,
        device="cpu",
        verbose=False,
    )

    assert result == str(output)
    proto = onnx.load(output)
    metadata = {entry.key: entry.value for entry in proto.metadata_props}
    assert proto.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 2
    assert metadata["model_family"] == "picosam3"
    assert metadata["task"] == "segment"
    assert json.loads(metadata["supported_tasks"]) == ["segment"]
    assert metadata["imgsz"] == "96"
    assert metadata["dynamic"] == "False"

    inputs = torch.randn((2, 3, 96, 96))
    with torch.no_grad():
        expected = model.model(inputs).numpy()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(["mask_logits"], {"roi_image": inputs.numpy()})[0]
    torch.testing.assert_close(
        torch.from_numpy(actual), torch.from_numpy(expected), atol=1e-5, rtol=1e-5
    )
