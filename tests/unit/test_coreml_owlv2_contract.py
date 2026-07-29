from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.export.coreml_owlv2 import (
    OWLV2_COREML_CONTRACT,
    OWLV2_COREML_MEAN,
    OWLV2_COREML_POSTPROCESS,
    OWLV2_COREML_PREPROCESS,
    OWLV2_COREML_STD,
    Owlv2FrozenCoreMLAdapter,
    _owlv2_dims_from_config,
    expected_owlv2_coreml_shapes,
    freeze_owlv2_text_embeddings,
    owlv2_coreml_input_contract,
    owlv2_coreml_metadata,
    owlv2_coreml_output_contract,
    owlv2_coreml_prompts,
    owlv2_coreml_validation_contract,
    owlv2_coreml_vocabulary_hash,
    postprocess_owlv2_coreml_outputs,
    prepare_owlv2_coreml_export,
    preprocess_owlv2_coreml_image,
    validate_owlv2_coreml_metadata,
    validate_owlv2_coreml_profile,
)
from libreyolo.models.openvocab.owlv2 import LibreOWLv2
from libreyolo.models.owlv2.nn import Owlv2DetectionModel

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


def test_profile_and_raw_io_contract_are_fixed():
    from libreyolo.export import coreml

    base = validate_owlv2_coreml_profile(
        size="b16",
        canvas_hw=(960, 960),
    )
    assert base.num_patches == 3600
    assert expected_owlv2_coreml_shapes(size="b16", nc=3) == {
        "pred_logits": (1, 3600, 3),
        "pred_boxes": (1, 3600, 4),
    }
    assert expected_owlv2_coreml_shapes(size="l14", nc=2) == {
        "pred_logits": (1, 5184, 2),
        "pred_boxes": (1, 5184, 4),
    }
    assert owlv2_coreml_input_contract() == {
        "name": "image",
        "kind": "tensor",
        "layout": "NCHW",
        "color": "rgb",
        "range": "0_1",
        "geometry": "owlv2_pad_square",
        "interpolation": "bilinear",
        "resize_backend": "torchvision",
        "resize_rounding": "floor",
        "pad_value": 0,
    }
    assert [
        (item["name"], item["role"], item["encoding"], item["rank"])
        for item in owlv2_coreml_output_contract()
    ] == [
        ("pred_logits", "class_logits", "raw_logits", 3),
        ("pred_boxes", "boxes", "cxcywh_normalized_square", 3),
    ]
    assert ("owlv2", "detect") in coreml.supported_coreml_exports()
    assert coreml._input_contract("owlv2", "detect", "b16") == (
        owlv2_coreml_input_contract()
    )
    assert coreml._output_contract("owlv2", "detect", nms=False) == (
        owlv2_coreml_output_contract()
    )
    assert coreml._validation_contract("owlv2", "detect") == (
        owlv2_coreml_validation_contract()
    )


@pytest.mark.parametrize(
    ("size", "canvas", "error"),
    [
        ("x", None, "b16 and l14"),
        ("b16", (1008, 1008), "native"),
        ("l14", (1008, 960), "native"),
    ],
)
def test_profile_rejects_unknown_or_interpolated_graphs(size, canvas, error):
    with pytest.raises(NotImplementedError, match=error):
        validate_owlv2_coreml_profile(size=size, canvas_hw=canvas)


def test_frozen_vocabulary_is_ordered_hashed_and_strict():
    names = {0: "Red Fox", 1: "Fire hydrant"}
    assert owlv2_coreml_prompts(names) == [
        "a photo of a red fox",
        "a photo of a fire hydrant",
    ]
    first = owlv2_coreml_vocabulary_hash(names)
    assert first == owlv2_coreml_vocabulary_hash(["Red Fox", "Fire hydrant"])
    assert first != owlv2_coreml_vocabulary_hash(["Fire hydrant", "Red Fox"])

    metadata = owlv2_coreml_metadata(size="b16", names=names)
    assert metadata == {
        "frozen_classes": True,
        "owlv2_contract": OWLV2_COREML_CONTRACT,
        "owlv2_preprocess": OWLV2_COREML_PREPROCESS,
        "owlv2_postprocess": OWLV2_COREML_POSTPROCESS,
        "owlv2_prompt_template": "a photo of a {}",
        "owlv2_vocabulary_sha256": first,
        "owlv2_image_size": 960,
        "owlv2_patch_size": 16,
        "owlv2_num_patches": 3600,
        "owlv2_num_classes": 2,
    }
    validate_owlv2_coreml_metadata(
        {key: str(value) for key, value in metadata.items()},
        size="b16",
        names=names,
    )

    tampered = deepcopy(metadata)
    tampered["owlv2_num_classes"] = 3
    with pytest.raises(ValueError, match="owlv2_num_classes"):
        validate_owlv2_coreml_metadata(
            tampered,
            size="b16",
            names=names,
        )
    tampered = deepcopy(metadata)
    tampered["owlv2_vocabulary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="vocabulary"):
        validate_owlv2_coreml_metadata(
            tampered,
            size="b16",
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
def test_frozen_vocabulary_rejects_invalid_or_unbounded_text(classes, error):
    with pytest.raises((TypeError, ValueError), match=error):
        owlv2_coreml_metadata(size="b16", names=classes)


@pytest.mark.parametrize(
    ("width", "height", "image_size"),
    [
        (37, 21, 32),
        (23, 61, 32),
        (19, 17, 48),
        (40, 80, 32),
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
    from transformers import Owlv2ImageProcessor

    image = _gradient(width, height)
    processor = Owlv2ImageProcessor(
        size={"height": image_size, "width": image_size}
    )
    expected = processor(images=image, return_tensors="pt")["pixel_values"]
    canonical = preprocess_owlv2_coreml_image(
        image,
        image_size=image_size,
    )
    mean = torch.tensor(OWLV2_COREML_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(OWLV2_COREML_STD).view(1, 3, 1, 1)
    actual = (canonical - mean) / std
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


class _FrozenTextProcessor:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.text = None

    def __call__(self, *, text, return_tensors):
        self.text = text
        assert return_tensors == "pt"
        return {
            "input_ids": self.input_ids.clone(),
            "attention_mask": self.attention_mask.clone(),
        }


def _tiny_hf_owlv2():
    transformers = pytest.importorskip("transformers")
    from transformers import (
        Owlv2Config,
        Owlv2ForObjectDetection,
        Owlv2TextConfig,
        Owlv2VisionConfig,
    )

    vision = Owlv2VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=32,
        patch_size=8,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
    )
    text = Owlv2TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=64,
        max_position_embeddings=8,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        bos_token_id=62,
        eos_token_id=63,
        pad_token_id=0,
    )
    config = Owlv2Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        projection_dim=32,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(4)
    return Owlv2ForObjectDetection(config).eval(), transformers.__version__


def test_text_free_graph_matches_full_tower_and_traces():
    hf_model, _ = _tiny_hf_owlv2()
    input_ids = torch.tensor(
        [
            [62, 4, 9, 63, 0, 0, 0, 0],
            [62, 8, 7, 5, 63, 0, 0, 0],
            [62, 3, 15, 2, 11, 63, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids.ne(0).long()
    processor = _FrozenTextProcessor(input_ids, attention_mask)
    names = ["cat", "dog", "remote control"]
    query = freeze_owlv2_text_embeddings(
        hf_model,
        processor,
        names,
    )
    assert processor.text == [[
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a remote control",
    ]]
    assert query.shape == (1, 3, 32)

    dims = _owlv2_dims_from_config(hf_model.config)
    native = Owlv2DetectionModel(dims).eval()
    incompatible = native.load_state_dict(hf_model.state_dict(), strict=False)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    adapter = Owlv2FrozenCoreMLAdapter(native, query).eval()
    assert not any("text_model" in key for key in adapter.state_dict())

    canonical = preprocess_owlv2_coreml_image(
        _gradient(29, 17),
        image_size=32,
    )
    mean = torch.tensor(OWLV2_COREML_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(OWLV2_COREML_STD).view(1, 3, 1, 1)
    with torch.inference_mode():
        reference = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=(canonical - mean) / std,
        )
        actual = adapter(canonical)
    torch.testing.assert_close(
        actual[0],
        reference.logits,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        actual[1],
        reference.pred_boxes,
        rtol=1e-5,
        atol=1e-5,
    )

    traced = torch.jit.trace(adapter, canonical, strict=True)
    traced_graph = str(traced.inlined_graph)
    assert "aten::broadcast_to" not in traced_graph
    assert "aten::einsum" not in traced_graph
    traced_first = traced(canonical)
    traced_second = traced(torch.flip(canonical, dims=(-1,)))
    torch.testing.assert_close(traced_first[0], actual[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(traced_first[1], actual[1], rtol=0.0, atol=0.0)
    assert not torch.equal(traced_first[0], traced_second[0])
    assert list(traced.graph.inputs())[1].debugName() == "image"


def test_postprocess_uses_square_scale_strict_threshold_and_class_filter():
    logits = torch.tensor(
        [
            [
                [2.0, -3.0],
                [0.0, -2.0],
                [-4.0, 3.0],
                [4.0, -5.0],
            ]
        ]
    )
    boxes = torch.tensor(
        [
            [
                [0.25, 0.25, 0.2, 0.2],
                [0.50, 0.25, 0.2, 0.2],
                [0.50, 0.80, 0.2, 0.2],
                [0.95, 0.25, 0.2, 0.2],
            ]
        ]
    )
    result = postprocess_owlv2_coreml_outputs(
        logits,
        boxes,
        original_size=(100, 50),
        conf=0.5,
        max_det=10,
    )
    # Candidate 1 has score exactly 0.5 and is excluded. Candidate 2 is wholly
    # inside bottom padding after max-side scaling and clips to zero height.
    assert result["num_detections"] == 2
    torch.testing.assert_close(
        result["boxes"],
        torch.tensor(
            [
                [85.0, 15.0, 100.0, 35.0],
                [15.0, 15.0, 35.0, 35.0],
            ]
        ),
    )
    assert result["classes"].tolist() == [0, 0]

    filtered = postprocess_owlv2_coreml_outputs(
        logits,
        boxes,
        original_size=(100, 50),
        conf=0.5,
        max_det=10,
        classes=[1],
    )
    assert filtered["num_detections"] == 0


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
        postprocess_owlv2_coreml_outputs(
            logits,
            boxes,
            original_size=(100, 50),
            conf=0.1,
            max_det=10,
        )


def test_public_model_exposes_only_frozen_coreml_export(monkeypatch):
    calls = []

    def fake_export(model, kwargs):
        calls.append((model, kwargs))
        return "owlv2.mlpackage"

    monkeypatch.setattr(
        "libreyolo.export.coreml_owlv2.export_owlv2_coreml",
        fake_export,
    )
    model = object.__new__(LibreOWLv2)
    assert model.export(format="COREML", dynamic=False) == "owlv2.mlpackage"
    assert calls == [(model, {"dynamic": False})]

    with pytest.raises(NotImplementedError, match="open-vocabulary detector"):
        model.export(format="onnx")


def test_direct_export_preparation_pins_native_canvas_and_metadata(
    monkeypatch,
):
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
            "model_family": "owlv2",
            "size": "b16",
            "task": "detect",
            "names": {"0": "cat", "1": "dog"},
            "nc": 2,
            "precision": precision,
            "dynamic": dynamic,
            "imgsz": imgsz[0],
        },
    )
    model = SimpleNamespace(
        size="b16",
        names={0: "cat", 1: "dog"},
        nb_classes=2,
    )
    prepared = prepare_owlv2_coreml_export(
        model,
        {
            "output": "frozen.OTHER",
            "imgsz": 960,
            "half": True,
            "compute_units": "cpu_only",
        },
    )
    image_size, output, metadata, precision, compute_units = prepared
    assert image_size == 960
    assert output.endswith("frozen.mlpackage")
    assert precision == "fp16"
    assert compute_units == "cpu_only"
    assert metadata["frozen_classes"] is True
    assert metadata["owlv2_num_classes"] == 2
    assert metadata["owlv2_vocabulary_sha256"] == (
        owlv2_coreml_vocabulary_hash(model.names)
    )
    assert calls == [
        {
            "half": True,
            "int8": False,
            "data": None,
            "nms": False,
            "compute_units": "cpu_only",
            "conf": 0.1,
            "iou": 0.45,
            "max_det": 300,
        }
    ]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"dynamic": True}, "dynamic"),
        ({"batch": 2}, "batch=1"),
        ({"nms": True}, "does not run NMS"),
        ({"imgsz": 1008}, "native"),
        ({"device": "cuda"}, "traces on CPU"),
        ({"text": ["cat"]}, "Unsupported"),
    ],
)
def test_direct_export_preparation_rejects_unbounded_or_wrong_profiles(
    kwargs,
    error,
):
    model = SimpleNamespace(
        size="b16",
        names={0: "cat"},
        nb_classes=1,
    )
    with pytest.raises((TypeError, ValueError, NotImplementedError), match=error):
        prepare_owlv2_coreml_export(model, kwargs)
