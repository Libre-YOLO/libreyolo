from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from libreyolo.export.coreml_omdet_turbo import (
    OMDET_TURBO_COREML_CONTRACT,
    OMDET_TURBO_COREML_MEAN,
    OMDET_TURBO_COREML_POSTPROCESS,
    OMDET_TURBO_COREML_PREPROCESS,
    OMDET_TURBO_COREML_STD,
    OmDetTurboDeformableAttentionCoreMLAdapter,
    OmDetTurboFixedDecoderCoreMLAdapter,
    OmDetTurboFrozenCoreMLAdapter,
    expected_omdet_turbo_coreml_shapes,
    export_omdet_turbo_coreml,
    freeze_omdet_turbo_language_embeddings,
    omdet_turbo_coreml_input_contract,
    omdet_turbo_coreml_metadata,
    omdet_turbo_coreml_output_contract,
    omdet_turbo_coreml_task,
    omdet_turbo_coreml_validation_contract,
    omdet_turbo_coreml_vocabulary_hash,
    postprocess_omdet_turbo_coreml_outputs,
    prepare_omdet_turbo_coreml_export,
    preprocess_omdet_turbo_coreml_image,
    require_omdet_turbo_coremltools_toolchain,
    require_omdet_turbo_transformers_toolchain,
    validate_omdet_turbo_coreml_metadata,
    validate_omdet_turbo_coreml_profile,
)
from libreyolo.models.openvocab.omdet_turbo import LibreOMDetTurbo

pytestmark = pytest.mark.unit


def _gradient(width: int, height: int) -> Image.Image:
    y, x = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (x * 17 + y * 3) % 256,
            (x * 5 + y * 19) % 256,
            (x * 11 + y * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def test_export_toolchain_versions_are_fail_closed():
    require_omdet_turbo_transformers_toolchain(
        SimpleNamespace(__version__="5.12.1")
    )
    with pytest.raises(RuntimeError, match="transformers==5.12.1"):
        require_omdet_turbo_transformers_toolchain(
            SimpleNamespace(__version__="5.13.0")
        )

    require_omdet_turbo_coremltools_toolchain(
        SimpleNamespace(__version__="9.0")
    )
    with pytest.raises(RuntimeError, match="coremltools 9.x"):
        require_omdet_turbo_coremltools_toolchain(
            SimpleNamespace(__version__="8.3")
        )


def test_profile_and_raw_io_contract_are_fixed():
    profile = validate_omdet_turbo_coreml_profile(
        size="t",
        canvas_hw=(640, 640),
    )
    assert profile.num_queries == 900
    assert expected_omdet_turbo_coreml_shapes(size="t", nc=3) == {
        "pred_logits": (1, 900, 3),
        "pred_boxes": (1, 900, 4),
    }
    assert omdet_turbo_coreml_input_contract() == {
        "name": "image",
        "kind": "tensor",
        "layout": "NCHW",
        "color": "rgb",
        "range": "0_255",
        "geometry": "stretch",
        "interpolation": "bilinear",
        "resize_backend": "torchvision",
        "pad_value": 0,
    }
    assert [
        (item["name"], item["role"], item["encoding"], item["rank"])
        for item in omdet_turbo_coreml_output_contract()
    ] == [
        ("pred_logits", "class_logits", "raw_logits", 3),
        ("pred_boxes", "boxes", "cxcywh_normalized", 3),
    ]
    assert omdet_turbo_coreml_validation_contract() == {
        "color": "rgb",
        "range": "0_255",
    }


@pytest.mark.parametrize(
    ("size", "canvas", "error"),
    [
        ("s", None, "size='t'"),
        ("t", (320, 320), "640x640"),
        ("t", (640, 608), "640x640"),
    ],
)
def test_profile_rejects_unknown_or_interpolated_graphs(size, canvas, error):
    with pytest.raises(NotImplementedError, match=error):
        validate_omdet_turbo_coreml_profile(size=size, canvas_hw=canvas)


def test_frozen_vocabulary_task_hash_and_metadata_are_strict():
    names = {0: "Red Fox", 1: "Fire hydrant"}
    assert omdet_turbo_coreml_task(names) == (
        "Detect Red Fox, Fire hydrant."
    )
    first = omdet_turbo_coreml_vocabulary_hash(names)
    assert first == omdet_turbo_coreml_vocabulary_hash(
        ["Red Fox", "Fire hydrant"]
    )
    assert first != omdet_turbo_coreml_vocabulary_hash(
        ["Fire hydrant", "Red Fox"]
    )

    metadata = omdet_turbo_coreml_metadata(size="t", names=names)
    assert metadata == {
        "frozen_classes": True,
        "omdet_turbo_contract": OMDET_TURBO_COREML_CONTRACT,
        "omdet_turbo_preprocess": OMDET_TURBO_COREML_PREPROCESS,
        "omdet_turbo_postprocess": OMDET_TURBO_COREML_POSTPROCESS,
        "omdet_turbo_task_template": "Detect {}.",
        "omdet_turbo_task": "Detect Red Fox, Fire hydrant.",
        "omdet_turbo_vocabulary_sha256": first,
        "omdet_turbo_image_size": 640,
        "omdet_turbo_num_queries": 900,
        "omdet_turbo_num_classes": 2,
    }
    validate_omdet_turbo_coreml_metadata(
        {key: str(value) for key, value in metadata.items()},
        size="t",
        names=names,
    )

    tampered = deepcopy(metadata)
    tampered["omdet_turbo_num_queries"] = 899
    with pytest.raises(ValueError, match="num_queries"):
        validate_omdet_turbo_coreml_metadata(
            tampered,
            size="t",
            names=names,
        )
    tampered = deepcopy(metadata)
    tampered["omdet_turbo_vocabulary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="vocabulary"):
        validate_omdet_turbo_coreml_metadata(
            tampered,
            size="t",
            names=names,
        )


@pytest.mark.parametrize(
    ("classes", "error"),
    [
        ([], "at least one"),
        ({1: "cat"}, "contiguous"),
        (["cat", "CAT"], "unique"),
        (["cat", " "], "blank"),
        ("cat", "mapping or"),
    ],
)
def test_frozen_vocabulary_rejects_invalid_text(classes, error):
    with pytest.raises((TypeError, ValueError), match=error):
        omdet_turbo_coreml_metadata(size="t", names=classes)


@pytest.mark.parametrize(
    ("width", "height", "image_size"),
    [
        (37, 21, 32),
        (23, 61, 48),
        (19, 17, 64),
    ],
)
def test_host_preprocessing_matches_pinned_transformers(
    width,
    height,
    image_size,
):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.12.1":
        pytest.skip("reference parity is pinned to transformers 5.12.1")
    from transformers import DetrImageProcessor

    image = _gradient(width, height)
    processor = DetrImageProcessor(
        do_resize=True,
        size={"height": image_size, "width": image_size},
        resample=Image.Resampling.BILINEAR,
        do_rescale=False,
        do_normalize=True,
        image_mean=OMDET_TURBO_COREML_MEAN,
        image_std=OMDET_TURBO_COREML_STD,
        do_pad=False,
    )
    expected = processor(images=image, return_tensors="pt")["pixel_values"]
    canonical = preprocess_omdet_turbo_coreml_image(
        image,
        image_size=image_size,
    )
    mean = torch.tensor(OMDET_TURBO_COREML_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(OMDET_TURBO_COREML_STD).view(1, 3, 1, 1)
    actual = (canonical - mean) / std
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


class _FrozenLanguageProcessor:
    def __init__(self):
        self.call = None

    def __call__(self, **kwargs):
        self.call = kwargs
        return {
            "pixel_values": torch.zeros(1, 3, 1, 1),
            "classes_input_ids": torch.tensor([[1, 2], [3, 4]]),
            "classes_attention_mask": torch.ones(2, 2, dtype=torch.long),
            "tasks_input_ids": torch.tensor([[5, 6, 7]]),
            "tasks_attention_mask": torch.ones(1, 3, dtype=torch.long),
            "classes_structure": torch.tensor([2]),
        }


class _FrozenLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.received = None

    def get_language_embedding(self, *values):
        self.received = values
        return (
            torch.arange(16, dtype=torch.float32).reshape(2, 1, 8),
            torch.arange(24, dtype=torch.float32).reshape(3, 1, 8),
            torch.tensor([[1, 1, 0]], dtype=torch.long),
        )


def test_language_tower_is_frozen_once_with_exact_task():
    model = _FrozenLanguageModel()
    processor = _FrozenLanguageProcessor()
    class_features, task_features, task_mask = (
        freeze_omdet_turbo_language_embeddings(
            model,
            processor,
            {0: "cat", 1: "remote control"},
        )
    )
    assert processor.call["text"] == ["cat", "remote control"]
    assert processor.call["task"] == "Detect cat, remote control."
    assert processor.call["return_tensors"] == "pt"
    assert len(model.received) == 5
    assert class_features.shape == (2, 1, 8)
    assert task_features.shape == (3, 1, 8)
    assert task_mask.tolist() == [[1, 1, 0]]


class _FakeVision(nn.Module):
    def forward(self, pixel_values):
        return [pixel_values]


class _FakeEncoder(nn.Module):
    def forward(self, features, **kwargs):
        assert kwargs == {
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": False,
        }
        return (features[-1], None, None, features)


class _FakeDecoder(nn.Module):
    def forward(
        self,
        features,
        class_features,
        task_features,
        task_mask,
        **kwargs,
    ):
        assert kwargs == {
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": False,
        }
        batch = features[0].shape[0]
        classes = class_features.shape[0]
        logits = torch.ones(1, batch, 4, classes) * features[0].mean()
        boxes = torch.full((1, batch, 4, 4), 0.5)
        return (None, None, None, boxes, logits, None, None, None, None)


def test_image_only_adapter_normalizes_and_excludes_language_tower():
    model = nn.Module()
    model.vision_backbone = _FakeVision()
    model.encoder = _FakeEncoder()
    model.decoder = _FakeDecoder()
    model.language_backbone = nn.Linear(2, 2)
    adapter = OmDetTurboFrozenCoreMLAdapter(
        model,
        torch.zeros(2, 1, 8),
        torch.zeros(3, 1, 8),
        torch.ones(1, 3, dtype=torch.long),
    ).eval()
    assert not any("language_backbone" in key for key in adapter.state_dict())

    image = torch.full((1, 3, 8, 8), 0.5)
    logits, boxes = adapter(image)
    expected_mean = (
        (image - adapter.image_mean) / adapter.image_std
    ).mean()
    torch.testing.assert_close(
        logits,
        torch.ones_like(logits) * expected_mean,
    )
    assert logits.shape == (1, 4, 2)
    assert boxes.shape == (1, 4, 4)


def test_rank_five_deformable_attention_is_bit_exact_and_traces():
    pytest.importorskip("transformers")
    from transformers.models.omdet_turbo.modeling_omdet_turbo import (
        OmDetTurboMultiscaleDeformableAttention,
    )

    config = SimpleNamespace(
        d_model=32,
        num_feature_levels=3,
        disable_custom_kernels=True,
    )
    torch.manual_seed(31)
    reference = OmDetTurboMultiscaleDeformableAttention(
        config,
        num_heads=4,
        n_points=2,
    ).eval()
    adapter = OmDetTurboDeformableAttentionCoreMLAdapter(reference).eval()
    hidden = torch.randn(1, 5, 32)
    encoded = torch.randn(1, 21, 32)
    shapes = torch.tensor([[4, 4], [2, 2], [1, 1]], dtype=torch.long)
    shape_list = [(4, 4), (2, 2), (1, 1)]
    starts = torch.tensor([0, 16, 20], dtype=torch.long)
    points = torch.rand(1, 5, 3, 4)
    args = (
        hidden,
        None,
        encoded,
        None,
        None,
        points,
        shapes,
        shape_list,
        starts,
    )
    with torch.inference_mode():
        expected = reference(*args)
        actual = adapter(*args)
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)

    class TraceBoundary(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.register_buffer("shapes", shapes)
            self.register_buffer("starts", starts)

        def forward(self, hidden_states, encoder_states, reference_points):
            return self.module(
                hidden_states,
                encoder_hidden_states=encoder_states,
                reference_points=reference_points,
                spatial_shapes=self.shapes,
                spatial_shapes_list=shape_list,
                level_start_index=self.starts,
            )

    boundary = TraceBoundary(adapter).eval()
    trace_args = (hidden, encoded, points)
    traced = torch.jit.trace(boundary, trace_args, strict=True)
    traced_outputs = traced(*trace_args)
    torch.testing.assert_close(
        traced_outputs[0],
        expected[0],
        rtol=0.0,
        atol=0.0,
    )
    for node in traced.inlined_graph.nodes():
        for output in node.outputs():
            try:
                sizes = output.type().sizes()
            except RuntimeError:
                continue
            if sizes is not None:
                assert len(sizes) <= 5


def test_selective_decoder_clone_preserves_native_query_order():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.12.1":
        pytest.skip("decoder parity is pinned to transformers 5.12.1")
    from transformers import OmDetTurboConfig
    from transformers.models.omdet_turbo.modeling_omdet_turbo import (
        OmDetTurboDecoder,
    )

    config = OmDetTurboConfig(
        class_embed_dim=48,
        task_encoder_hidden_dim=64,
        encoder_hidden_dim=32,
        decoder_hidden_dim=32,
        decoder_num_heads=4,
        decoder_num_layers=2,
        decoder_dim_feedforward=64,
        decoder_num_points=2,
        num_feature_levels=3,
        num_queries=5,
        vision_features_channels=(16, 24, 32),
        disable_custom_kernels=True,
    )
    torch.manual_seed(37)
    native = OmDetTurboDecoder(config).eval()
    task_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    deployment = OmDetTurboFixedDecoderCoreMLAdapter(
        native,
        task_mask,
    ).eval()
    vision_features = [
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 24, 2, 2),
        torch.randn(1, 32, 1, 1),
    ]
    class_features = torch.randn(3, 1, 48)
    task_features = torch.randn(4, 1, 48)

    with torch.inference_mode():
        expected = native(
            vision_features,
            class_features,
            task_features,
            task_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        actual = deployment(
            vision_features,
            class_features,
            task_features,
            task_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

    # Exact equality also proves the top-k query ordering was retained.
    torch.testing.assert_close(actual[4], expected[4], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0.0, atol=0.0)


def test_postprocess_matches_pinned_transformers_topk_and_nms():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.12.1":
        pytest.skip("reference parity is pinned to transformers 5.12.1")
    from transformers.models.omdet_turbo.processing_omdet_turbo import (
        _post_process_boxes_for_image,
        compute_score,
    )

    logits = torch.tensor(
        [
            [
                [4.0, -2.0, -3.0],
                [3.5, -1.0, -4.0],
                [-2.0, 3.0, -1.0],
                [-2.0, 2.5, -3.0],
                [-4.0, -3.0, 4.5],
                [0.0, -4.0, -2.0],
            ]
        ]
    )
    boxes = torch.tensor(
        [
            [
                [0.20, 0.20, 0.20, 0.20],
                [0.22, 0.22, 0.20, 0.20],
                [0.55, 0.25, 0.15, 0.20],
                [0.57, 0.25, 0.15, 0.20],
                [0.80, 0.75, 0.25, 0.30],
                [0.45, 0.75, 0.10, 0.10],
            ]
        ]
    )
    conf = 0.3
    iou = 0.5
    width, height = 200, 100
    batch_scores, labels = compute_score(logits)
    expected_boxes, expected_scores, expected_labels = (
        _post_process_boxes_for_image(
            boxes=boxes[0],
            scores=batch_scores[0],
            labels=labels,
            image_num_classes=3,
            image_size=(height, width),
            threshold=conf,
            nms_threshold=iou,
        )
    )
    valid = (
        (expected_boxes[:, 2] > expected_boxes[:, 0])
        & (expected_boxes[:, 3] > expected_boxes[:, 1])
    )
    expected_boxes = expected_boxes[valid]
    expected_scores = expected_scores[valid]
    expected_labels = expected_labels[valid]

    actual = postprocess_omdet_turbo_coreml_outputs(
        logits,
        boxes,
        original_size=(width, height),
        conf=conf,
        iou=iou,
        max_det=300,
    )
    torch.testing.assert_close(actual["boxes"], expected_boxes)
    torch.testing.assert_close(actual["scores"], expected_scores)
    torch.testing.assert_close(actual["classes"], expected_labels)
    assert actual["num_detections"] == len(expected_scores)

    filtered = postprocess_omdet_turbo_coreml_outputs(
        logits,
        boxes,
        original_size=(width, height),
        conf=conf,
        iou=iou,
        max_det=300,
        classes=[2],
    )
    assert filtered["classes"].tolist() == [2]


@pytest.mark.parametrize(
    ("logits", "boxes", "error"),
    [
        (torch.zeros(4, 2), torch.zeros(1, 4, 4), "pred_logits"),
        (torch.zeros(1, 4, 2), torch.zeros(1, 3, 4), "pred_boxes"),
        (
            torch.tensor([[[float("nan")]]]),
            torch.zeros(1, 1, 4),
            "NaN",
        ),
    ],
)
def test_postprocess_rejects_invalid_runtime_outputs(logits, boxes, error):
    with pytest.raises((ValueError, RuntimeError), match=error):
        postprocess_omdet_turbo_coreml_outputs(
            logits,
            boxes,
            original_size=(100, 50),
            conf=0.3,
            iou=0.5,
            max_det=10,
        )


def test_public_model_exposes_only_frozen_coreml_export(monkeypatch):
    calls = []

    def fake_export(model, kwargs):
        calls.append((model, kwargs))
        return "omdet.mlpackage"

    monkeypatch.setattr(
        "libreyolo.export.coreml_omdet_turbo.export_omdet_turbo_coreml",
        fake_export,
    )
    model = object.__new__(LibreOMDetTurbo)
    assert model.export(format="COREML", dynamic=False) == "omdet.mlpackage"
    assert calls == [(model, {"dynamic": False})]

    with pytest.raises(NotImplementedError, match="open-vocabulary detector"):
        model.export(format="onnx")


def test_direct_export_preparation_pins_canvas_and_metadata(monkeypatch):
    from libreyolo.export.exporter import CoreMLExporter

    calls = []
    monkeypatch.setattr(
        CoreMLExporter,
        "_validate",
        lambda self, half, int8, data: (half, int8),
    )

    def fake_preflight(self, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(CoreMLExporter, "_preflight", fake_preflight)
    monkeypatch.setattr(
        CoreMLExporter,
        "_build_metadata",
        lambda self, precision, dynamic, onnx_path, imgsz=None: {
            "model_family": "omdet_turbo",
            "size": "t",
            "task": "detect",
            "names": {"0": "cat", "1": "dog"},
            "nc": 2,
            "precision": precision,
            "dynamic": dynamic,
            "imgsz": imgsz[0],
        },
    )
    model = SimpleNamespace(
        size="t",
        names={0: "cat", 1: "dog"},
        nb_classes=2,
    )
    prepared = prepare_omdet_turbo_coreml_export(
        model,
        {
            "output": "frozen.OTHER",
            "imgsz": 640,
            "half": False,
            "compute_units": "cpu_only",
        },
    )
    image_size, output, metadata, precision, compute_units = prepared
    assert image_size == 640
    assert output.endswith("frozen.mlpackage")
    assert precision == "fp32"
    assert compute_units == "cpu_only"
    assert metadata["frozen_classes"] is True
    assert metadata["omdet_turbo_num_classes"] == 2
    assert metadata["omdet_turbo_vocabulary_sha256"] == (
        omdet_turbo_coreml_vocabulary_hash(model.names)
    )
    assert calls == [
        {
            "half": False,
            "int8": False,
            "data": None,
            "nms": False,
            "compute_units": "cpu_only",
            "conf": 0.3,
            "iou": 0.5,
            "max_det": 300,
        }
    ]


@pytest.mark.parametrize("conversion_fails", [False, True])
def test_direct_export_restores_live_model_state(
    monkeypatch,
    conversion_fails,
):
    live_model = nn.Linear(2, 2).to(dtype=torch.float64).train()
    wrapper = SimpleNamespace(
        model=live_model,
        processor=object(),
        size="t",
        names={0: "cat"},
    )
    monkeypatch.setattr(
        "libreyolo.export.coreml_omdet_turbo."
        "prepare_omdet_turbo_coreml_export",
        lambda model, kwargs: (
            640,
            "frozen.mlpackage",
            {"model_family": "omdet_turbo"},
            "fp16",
            "all",
        ),
    )

    def fake_build(model, processor, **kwargs):
        model.to(dtype=torch.float32).eval()
        return nn.Identity()

    monkeypatch.setattr(
        "libreyolo.export.coreml_omdet_turbo."
        "build_omdet_turbo_frozen_coreml_adapter",
        fake_build,
    )
    monkeypatch.setattr(
        "libreyolo.export.coreml_omdet_turbo."
        "require_omdet_turbo_coremltools_toolchain",
        lambda: None,
    )

    def fake_export(*args, **kwargs):
        if conversion_fails:
            raise RuntimeError("conversion failed")
        return "frozen.mlpackage"

    monkeypatch.setattr(
        "libreyolo.export.coreml.export_coreml",
        fake_export,
    )

    if conversion_fails:
        with pytest.raises(RuntimeError, match="conversion failed"):
            export_omdet_turbo_coreml(wrapper, {})
    else:
        assert export_omdet_turbo_coreml(wrapper, {}) == (
            "frozen.mlpackage"
        )

    assert next(live_model.parameters()).dtype == torch.float64
    assert live_model.training is True


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"dynamic": True}, "dynamic"),
        ({"batch": 2}, "batch=1"),
        ({"nms": True}, "host"),
        ({"imgsz": 608}, "640x640"),
        ({"device": "cuda"}, "traces on CPU"),
        ({"half": True}, "FP32-only"),
        ({"text": ["cat"]}, "Unsupported"),
    ],
)
def test_direct_export_preparation_rejects_wrong_profiles(kwargs, error):
    model = SimpleNamespace(
        size="t",
        names={0: "cat"},
        nb_classes=1,
    )
    with pytest.raises(
        (TypeError, ValueError, NotImplementedError),
        match=error,
    ):
        prepare_omdet_turbo_coreml_export(model, kwargs)
