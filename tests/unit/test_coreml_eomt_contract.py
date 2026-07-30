"""EoMT compact-query Core ML export and host-orchestration contracts."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.backends.coreml import (
    CoreMLBackend,
    _parse_io_contract,
    _validate_eomt_metadata,
)
from libreyolo.export.coreml import (
    _input_contract,
    _output_contract,
    _validate_output_semantics,
    _validation_contract,
    supported_coreml_exports,
)
from libreyolo.export.coreml_eomt import (
    EOMT_COREML_CONTRACT,
    EoMTCoreMLAdapter,
    eomt_coreml_input_contract,
    eomt_coreml_metadata,
    eomt_coreml_output_contract,
    eomt_coreml_validation_contract,
    expected_eomt_coreml_shapes,
    reconstruct_eomt_full_outputs,
    validate_eomt_coreml_profile,
)
from libreyolo.models.eomt.model import LibreEoMT
from libreyolo.models.eomt.nn import LibreEoMTNet


pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def tiny_eomt() -> LibreEoMTNet:
    torch.manual_seed(20260729)
    return LibreEoMTNet(
        config="s",
        nb_classes=2,
        image_size=32,
        num_queries=4,
    ).eval()


def _decoder(task: str) -> LibreEoMT:
    decoder = object.__new__(LibreEoMT)
    decoder.task = task
    decoder.input_size = 32
    decoder.num_queries = 4
    decoder.nb_classes = 2
    decoder.names = {0: "thing", 1: "stuff"}
    decoder.thing_class_ids = {0} if task == "panoptic" else None
    return decoder


def _backend(task: str, decoder: LibreEoMT) -> CoreMLBackend:
    backend = CoreMLBackend.__new__(CoreMLBackend)
    backend.model_family = "eomt"
    backend.task = task
    backend.imgsz = 32
    backend.nb_classes = 2
    backend.names = decoder.names
    backend.output_names = [
        "class_queries_logits",
        "masks_queries_logits",
    ]
    backend._has_embedded_nms = False
    backend._eomt_decoder = decoder
    return backend


def _gradient_image(width: int, height: int) -> Image.Image:
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (11 * xx + 3 * yy + 17) % 256,
            (5 * xx + 19 * yy + 31) % 256,
            (23 * xx + 7 * yy + 53) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _contract_io(task: str, *, thing_ids=(0,)):
    outputs = eomt_coreml_output_contract()
    shapes = expected_eomt_coreml_shapes(
        nc=2,
        num_queries=4,
        canvas_hw=(32, 32),
    )
    enriched = []
    for output in outputs:
        item = dict(output)
        item["dtype"] = "float32"
        item["shape"] = list(shapes[item["name"]])
        enriched.append(item)
    io = _parse_io_contract(
        {
            "coreml_io": {
                "input": eomt_coreml_input_contract(task),
                "validation": eomt_coreml_validation_contract(task),
                "outputs": enriched,
            }
        }
    )
    metadata = eomt_coreml_metadata(
        task=task,
        num_queries=4,
        image_size=32,
    )
    metadata["num_queries"] = 4
    if task == "panoptic":
        metadata.update(
            {
                "thing_class_ids": list(thing_ids),
                "eomt_panoptic_score_threshold": 0.8,
                "eomt_panoptic_mask_threshold": 0.5,
                "eomt_panoptic_overlap_threshold": 0.8,
            }
        )
    return metadata, io


@pytest.mark.parametrize("task", ["semantic", "segment", "panoptic"])
def test_eomt_contract_is_routed_for_every_task(task):
    assert ("eomt", task) in supported_coreml_exports()
    assert _input_contract("eomt", task, "s") == eomt_coreml_input_contract(task)
    assert _output_contract("eomt", task, nms=False) == (eomt_coreml_output_contract())
    assert _validation_contract("eomt", task) == (eomt_coreml_validation_contract(task))
    assert (
        eomt_coreml_metadata(
            task=task,
            num_queries=4,
            image_size=32,
        )["eomt_contract"]
        == EOMT_COREML_CONTRACT
    )


def test_functional_adapter_and_host_reconstruction_are_native_exact(tiny_eomt):
    adapter = EoMTCoreMLAdapter(tiny_eomt).eval()
    first = torch.linspace(0.0, 1.0, 32 * 32).reshape(1, 1, 32, 32)
    first = torch.cat((first, first.flip(-1), first.flip(-2)), dim=1)
    second = 1.0 - first

    with torch.no_grad():
        normalized_first = (first - tiny_eomt.pixel_mean) / tiny_eomt.pixel_std
        normalized_second = (second - tiny_eomt.pixel_mean) / tiny_eomt.pixel_std
        raw_first = tiny_eomt.eomt(pixel_values=normalized_first)
        raw_second = tiny_eomt.eomt(pixel_values=normalized_second)
        expected_first = adapter(first)
        expected_second = adapter(second)
        native_first = tiny_eomt(first)
        native_second = tiny_eomt(second)
        traced = torch.jit.trace(
            adapter,
            first,
            check_trace=True,
            check_inputs=[(second,)],
        )
        actual_first = traced(first)
        actual_second = traced(second)

    torch.testing.assert_close(
        expected_first[0],
        raw_first.class_queries_logits,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        expected_first[1],
        raw_first.masks_queries_logits,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        expected_second[0],
        raw_second.class_queries_logits,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        expected_second[1],
        raw_second.masks_queries_logits,
        rtol=0.0,
        atol=0.0,
    )
    for expected, actual, expected_alt, actual_alt in zip(
        expected_first,
        actual_first,
        expected_second,
        actual_second,
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_alt, expected_alt, rtol=0.0, atol=0.0)
        assert float((expected_alt - expected).abs().max()) > 1e-8

    rebuilt_first = reconstruct_eomt_full_outputs(
        *expected_first,
        nc=2,
        canvas_hw=(32, 32),
    )
    rebuilt_second = reconstruct_eomt_full_outputs(
        *expected_second,
        nc=2,
        canvas_hw=(32, 32),
    )
    for name in rebuilt_first:
        torch.testing.assert_close(
            rebuilt_first[name],
            native_first[name],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            rebuilt_second[name],
            native_second[name],
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("size", "canvas", "error"),
    [
        ("x", (32, 32), "sizes s/b/l"),
        ("s", (32, 48), "square"),
        ("s", (48, 48), "position embeddings"),
    ],
)
def test_profile_rejects_incompatible_eomt_graphs(
    tiny_eomt,
    size,
    canvas,
    error,
):
    with pytest.raises(NotImplementedError, match=error):
        validate_eomt_coreml_profile(
            tiny_eomt,
            task="semantic",
            size=size,
            canvas_hw=canvas,
        )


def test_profile_rejects_random_attention_mask_configuration(tiny_eomt):
    original = tiny_eomt.eomt.attn_mask_probs.detach().clone()
    try:
        tiny_eomt.eomt.attn_mask_probs[0] = 0.5
        with pytest.raises(RuntimeError, match="exactly 0 or 1"):
            validate_eomt_coreml_profile(
                tiny_eomt,
                task="semantic",
                size="s",
                canvas_hw=(32, 32),
            )
    finally:
        tiny_eomt.eomt.attn_mask_probs.copy_(original)


@pytest.mark.parametrize("schedule", [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)])
def test_adapter_preserves_deterministic_binary_attention_schedule(
    tiny_eomt,
    schedule,
):
    original = tiny_eomt.eomt.attn_mask_probs.detach().clone()
    image = torch.linspace(0.0, 1.0, 3 * 32 * 32).reshape(1, 3, 32, 32)
    try:
        tiny_eomt.eomt.attn_mask_probs.copy_(torch.tensor(schedule))
        validate_eomt_coreml_profile(
            tiny_eomt,
            task="semantic",
            size="s",
            canvas_hw=(32, 32),
        )
        adapter = EoMTCoreMLAdapter(tiny_eomt).eval()
        normalized = (image - tiny_eomt.pixel_mean) / tiny_eomt.pixel_std
        with torch.no_grad():
            native = tiny_eomt.eomt(pixel_values=normalized)
            actual = adapter(image)
        torch.testing.assert_close(
            actual[0],
            native.class_queries_logits,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            actual[1],
            native.masks_queries_logits,
            rtol=0.0,
            atol=0.0,
        )
    finally:
        tiny_eomt.eomt.attn_mask_probs.copy_(original)


@pytest.mark.parametrize("task", ["semantic", "segment", "panoptic"])
def test_strict_metadata_accepts_exact_eomt_contract(task):
    metadata, io = _contract_io(task)
    parsed = _validate_eomt_metadata(
        metadata,
        task=task,
        nc=2,
        imgsz=32,
        io_contract=io,
    )
    assert parsed["num_queries"] == 4
    assert parsed["image_size"] == 32
    assert parsed["thing_class_ids"] == ([0] if task == "panoptic" else None)


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("eomt_contract", "wrong", "eomt_contract"),
        ("eomt_preprocess", "stretch", "eomt_preprocess"),
        ("eomt_num_queries", 5, "aliases disagree"),
        ("eomt_mask_stride", 8, "eomt_mask_stride"),
        ("eomt_mask_align_corners", True, "align_corners=false"),
        ("eomt_antialias", False, "antialias=true"),
        ("thing_class_ids", [0, 0], "sorted, unique"),
        ("eomt_panoptic_overlap_threshold", 0.5, "overlap_threshold"),
    ],
)
def test_strict_panoptic_metadata_rejects_tampering(key, value, error):
    metadata, io = _contract_io("panoptic")
    tampered = deepcopy(metadata)
    tampered[key] = value
    with pytest.raises(ValueError, match=error):
        _validate_eomt_metadata(
            tampered,
            task="panoptic",
            nc=2,
            imgsz=32,
            io_contract=io,
        )


def test_semantic_host_split_and_stitch_matches_native(tiny_eomt):
    source = _gradient_image(51, 20)
    native_decoder = _decoder("semantic")
    native_input, _, original_size, _ = native_decoder._preprocess(
        source,
        "rgb",
        input_size=32,
    )
    with torch.no_grad():
        native_output = tiny_eomt(native_input)
    expected = native_decoder._postprocess_semantic(
        native_output,
        original_size,
    )["semantic"]

    runtime_decoder = _decoder("semantic")
    backend = _backend("semantic", runtime_decoder)
    canonical, _, backend_original_size, ratio = backend._preprocess(
        source,
        32,
        "rgb",
    )
    assert canonical.shape[0] > 1
    with torch.no_grad():
        raw = EoMTCoreMLAdapter(tiny_eomt)(canonical / 255.0)
    actual = backend._parse_semantic_output(
        [raw[0].numpy(), raw[1].numpy()],
        backend_original_size,
        32,
        ratio,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_instance_host_decode_matches_native(tiny_eomt):
    source = _gradient_image(51, 20)
    decoder = _decoder("segment")
    backend = _backend("segment", decoder)
    canonical, _, original_size, ratio = backend._preprocess(source, 32, "rgb")
    with torch.no_grad():
        native = tiny_eomt(canonical / 255.0)
        raw = EoMTCoreMLAdapter(tiny_eomt)(canonical / 255.0)
    expected = decoder._postprocess_segment(
        native,
        conf_thres=0.0,
        iou_thres=0.6,
        original_size=original_size,
        max_det=4,
    )
    boxes, scores, classes, masks = backend._parse_outputs(
        [raw[0].numpy(), raw[1].numpy()],
        32,
        original_size,
        0.0,
        ratio=ratio,
        iou=0.6,
        max_det=4,
    )
    np.testing.assert_allclose(boxes, expected["boxes"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scores, expected["scores"], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(classes, expected["classes"])
    torch.testing.assert_close(masks, expected["masks"], rtol=0.0, atol=0.0)


def test_panoptic_host_decode_matches_native_fixed_thresholds():
    source = _gradient_image(47, 23)
    decoder = _decoder("panoptic")
    backend = _backend("panoptic", decoder)
    _, _, original_size, ratio = backend._preprocess(source, 32, "rgb")

    class_logits = torch.full((1, 4, 3), -20.0)
    class_logits[0, 0, 0] = 20.0
    class_logits[0, 1, 1] = 18.0
    mask_logits = torch.full((1, 4, 8, 8), -20.0)
    mask_logits[0, 0, :, :4] = 20.0
    mask_logits[0, 1, :, 4:] = 20.0
    reconstructed = reconstruct_eomt_full_outputs(
        class_logits,
        mask_logits,
        nc=2,
        canvas_hw=(32, 32),
    )
    expected = decoder._postprocess_panoptic(
        reconstructed,
        conf_thres=0.01,
        original_size=original_size,
    )
    actual = backend._parse_panoptic_output(
        [class_logits.numpy(), mask_logits.numpy()],
        original_size,
        32,
        conf=0.99,
        iou=0.1,
        max_det=1,
        ratio=ratio,
    )
    torch.testing.assert_close(
        actual["panoptic"],
        expected["panoptic"],
        rtol=0.0,
        atol=0.0,
    )
    assert actual["segments_info"] == expected["segments_info"]


def test_output_semantics_pins_compact_query_shapes():
    outputs = eomt_coreml_output_contract()
    shapes = expected_eomt_coreml_shapes(
        nc=2,
        num_queries=4,
        canvas_hw=(32, 32),
    )
    tensors = [torch.zeros(shapes[output["name"]]) for output in outputs]
    metadata = eomt_coreml_metadata(
        task="semantic",
        num_queries=4,
        image_size=32,
    )
    _validate_output_semantics(
        outputs,
        tensors,
        family="eomt",
        task="semantic",
        nc=2,
        input_hw=(32, 32),
        size="s",
        nms=False,
        metadata=metadata,
    )
    tensors[1] = torch.zeros(1, 4, 7, 8)
    with pytest.raises(RuntimeError, match="masks_queries_logits"):
        _validate_output_semantics(
            outputs,
            tensors,
            family="eomt",
            task="semantic",
            nc=2,
            input_hw=(32, 32),
            size="s",
            nms=False,
            metadata=metadata,
        )


@pytest.mark.parametrize("task", ["semantic", "segment", "panoptic"])
@pytest.mark.parametrize("format_name", ["coreml", "mlpackage", " CoreML "])
def test_model_exposes_coreml_export_route_for_all_tasks(
    monkeypatch,
    task,
    format_name,
):
    from libreyolo.models.base.model import BaseModel

    calls = []

    def fake_export(self, format="onnx", **kwargs):
        calls.append((self, format, kwargs))
        return "eomt.mlpackage"

    monkeypatch.setattr(BaseModel, "export", fake_export)
    model = object.__new__(LibreEoMT)
    model.task = task
    assert model.export(format=format_name, dynamic=False) == "eomt.mlpackage"
    assert calls == [(model, "coreml", {"opset": 17, "dynamic": False})]


def test_query_tasks_keep_non_coreml_formats_blocked():
    model = object.__new__(LibreEoMT)
    model.task = "segment"
    with pytest.raises(NotImplementedError, match="query-mask runtime contracts"):
        model.export(format="onnx")
