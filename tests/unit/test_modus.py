"""CPU-only contract tests for the external-weight LibreMODUS family."""

from __future__ import annotations

from contextlib import nullcontext
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from libreyolo.models.modus.decode import (
    detection_payload,
    decode_cocodet_tokens,
    decode_grounding_tokens,
    image_to_payload,
    input_to_image,
)
from libreyolo.models.modus.inference import ModusInferencer
from libreyolo.models.modus.modality import CodeCondition
from libreyolo.models.modus.model import LibreMODUS, _mapped_device
from libreyolo.models.modus.prompts import (
    BASE_MODALITY_REGISTRY,
    GROUNDING_PROMPT,
    PUBLIC_TARGETS,
    validate_any2any_request,
)
from libreyolo.models.modus.quantize import (
    WeightOnlyFP8Linear,
    _quantize_weight,
    prepare_fp8_checkpoint,
)
from libreyolo.models.modus.tokenizer import (
    assert_checkpoint_vocabulary,
    build_modus_tokenizer,
)
from libreyolo.models.modus.weights import (
    REQUIRED_FILES,
    resolve_modus_snapshot,
    validate_snapshot,
)
from libreyolo.models.modus.nn import ModusBagel
from libreyolo.models.sensenova.modeling.autoencoder import AutoEncoderParams
from libreyolo.models.sensenova.modeling.bagel import BagelConfig
from libreyolo.models.sensenova.modeling.layers import LearnableEmbedding
from libreyolo.models.sensenova.modeling.qwen2_navit import (
    NaiveCache,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from libreyolo.models.sensenova.modeling.siglip_navit import (
    SiglipVisionConfig,
    SiglipVisionModel,
)

pytestmark = [pytest.mark.unit, pytest.mark.modus]


class _CheckpointTokenizer:
    """Small fake with the released Qwen added-token starting offset."""

    def __init__(self):
        self.vocab = {
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
            "<|box_start|>": 151648,
            "<|box_end|>": 151649,
            "<|vision_start|>": 151652,
            "<|vision_end|>": 151653,
        }
        self.next_id = 151665

    def __len__(self):
        return self.next_id

    def add_tokens(self, tokens):
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.next_id += 1
                added += 1
        return added

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, -1)

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return [42] if text else []

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        inverse = {value: key for key, value in self.vocab.items()}
        return "".join(inverse.get(int(value), "?") for value in token_ids)


@pytest.fixture(scope="module")
def tokenizer_artifacts():
    return build_modus_tokenizer(_CheckpointTokenizer())


def test_released_registry_and_token_order_are_checkpoint_exact(tokenizer_artifacts):
    artifacts = tokenizer_artifacts
    assert len(BASE_MODALITY_REGISTRY) == 16
    assert BASE_MODALITY_REGISTRY.names == (
        "text",
        "caption",
        "rgb",
        "depth",
        "normal",
        "det",
        "seg",
        "canny",
        "dino",
        "dinolocal",
        "clip",
        "imagebind",
        "imagebindlocal",
        "cocodet",
        "samseg",
        "samedge",
    )
    assert len(artifacts.tokenizer) == 196840
    assert artifacts.token_ranges == {
        "det": (151671, 4100),
        "dino": (155773, 8192),
        "dinolocal": (163967, 8192),
        "clip": (172161, 8192),
        "imagebind": (180355, 8192),
        "imagebindlocal": (188549, 8192),
    }
    assert artifacts.new_token_ids["end_of_det"] == 155772
    assert artifacts.new_token_ids["start_of_cocodet"] == 196834
    assert artifacts.new_token_ids["end_of_cocodet"] == 196835
    assert artifacts.new_token_ids["start_of_canny"] == 196837
    assert artifacts.new_token_ids["start_of_samedge"] == 196839
    assert len(artifacts.code_token_ids["cocodet"]) == 4092
    assert_checkpoint_vocabulary(artifacts, 196840)
    with pytest.raises(RuntimeError, match="tokenizer/checkpoint mismatch"):
        assert_checkpoint_vocabulary(artifacts, 152064)


def test_learnable_embeddings_use_released_parameter_names(tokenizer_artifacts):
    layer = LearnableEmbedding(16, 8)
    assert list(layer.state_dict()) == ["embedding"]
    assert layer(torch.tensor([0, 3])).shape == (2, 8)
    runtime = tokenizer_artifacts.modality_registry
    assert tuple(
        (spec.name, spec.pos_embed_size)
        for spec in runtime.modalities_with_forward_pos_embed()
    ) == (
        ("dino", 16),
        ("dinolocal", 1024),
        ("clip", 784),
        ("imagebind", 16),
        ("imagebindlocal", 1024),
    )


def _build_tiny_modus_bagel(*, empty: bool = False):
    if empty:
        from accelerate import init_empty_weights

        construction = init_empty_weights(include_buffers=False)
    else:
        construction = nullcontext()
    llm_config = Qwen2Config(
        vocab_size=5000,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        qk_norm=True,
        layer_module="Qwen2MoTDecoderLayer",
    )
    vit_config = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
        rope=False,
    )
    vae_config = AutoEncoderParams(
        resolution=64,
        in_channels=3,
        downsample=8,
        ch=32,
        out_ch=3,
        ch_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        z_channels=4,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=2,
        max_latent_size=8,
        vit_max_num_patch_per_side=2,
        connector_act="gelu_pytorch_tanh",
    )
    with construction:
        torch.manual_seed(0)
        model = ModusBagel(
            Qwen2ForCausalLM(llm_config),
            SiglipVisionModel(vit_config),
            config,
            modality_registry=BASE_MODALITY_REGISTRY,
        )
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=empty
        )
    return model if empty else model.to(torch.bfloat16).eval()


@pytest.fixture(scope="module")
def tiny_modus_bagel():
    return _build_tiny_modus_bagel()


def test_two_layer_toy_has_checkpoint_surface_and_modality_boundaries(
    tiny_modus_bagel,
):
    state = tiny_modus_bagel.state_dict()
    assert "latent_pos_embed.pos_embed" not in state
    assert "vit_pos_embed.pos_embed" not in state
    for name in ("dino", "dinolocal", "clip", "imagebind", "imagebindlocal"):
        assert f"{name}_pos_embed.embedding" in state

    class Tokens:
        @staticmethod
        def encode(_text):
            return [7, 8]

    token_ids = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "start_of_det": 3,
        "end_of_det": 4,
    }
    packed, lengths, ropes = tiny_modus_bagel.prepare_prompts(
        [0],
        [0],
        ["box"],
        Tokens(),
        token_ids,
        modality_type="det",
    )
    assert packed["packed_text_ids"].tolist() == [3, 7, 8, 4]
    assert lengths == [4]
    assert ropes == [4]


def test_two_layer_toy_dispatches_without_checkpoint_static_tables(
    tiny_modus_bagel,
    tmp_path,
):
    accelerate = pytest.importorskip("accelerate")
    checkpoint = tmp_path / "toy.safetensors"
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in tiny_modus_bagel.state_dict().items()
        },
        str(checkpoint),
    )
    skeleton = _build_tiny_modus_bagel(empty=True)
    skeleton.materialize_static_position_embeddings({"": "cpu"})
    loaded = accelerate.load_checkpoint_and_dispatch(
        skeleton,
        checkpoint=str(checkpoint),
        device_map={"": "cpu"},
        dtype=torch.bfloat16,
    )
    assert not any(tensor.device.type == "meta" for tensor in loaded.parameters())
    assert loaded.latent_pos_embed.pos_embed.device.type == "cpu"
    assert loaded.vit_pos_embed.pos_embed.device.type == "cpu"


def test_accelerate_device_map_uses_most_specific_parent():
    device_map = {"": "cpu", "language_model": 0, "language_model.model": 1}
    assert (
        _mapped_device(device_map, "language_model.model.embed_tokens", "fallback") == 1
    )
    assert _mapped_device(device_map, "connector", "fallback") == "cpu"


def test_two_layer_toy_grounding_decoder_enforces_coordinate_slots(
    tiny_modus_bagel,
):
    start = tiny_modus_bagel.prepare_start_tokens(
        [0],
        [0],
        {"bos_token_id": 1, "start_of_det": 4500},
        modality_type="det",
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        tokens, probs = tiny_modus_bagel.generate_detection_coordonly(
            past_key_values=NaiveCache(2),
            max_length=5,
            x1_base=100,
            y1_base=1100,
            x2_base=2100,
            y2_base=3100,
            det_end_token=4501,
            **start,
        )
    values = tokens[:, 0].tolist()
    assert values[0] == 4500
    assert 100 <= values[1] < 1100
    assert 1100 <= values[2] < 2100
    assert 2100 <= values[3] < 3100
    assert 3100 <= values[4] < 4100
    assert probs.shape == (5,)


@pytest.mark.parametrize(
    ("inputs", "target", "expected"),
    [
        (("image",), "edge", (("rgb",), "canny")),
        (("depth", "text"), "grounding", (("depth", "text"), "det")),
        (("normal", "edges"), "detect", (("normal", "canny"), "cocodet")),
    ],
)
def test_public_any2any_matrix_normalizes_aliases(inputs, target, expected):
    assert validate_any2any_request(inputs, target) == expected


def test_public_any2any_matrix_rejects_generation_and_text_only():
    with pytest.raises(ValueError, match="image-derived"):
        validate_any2any_request(("text",), "depth")
    with pytest.raises(ValueError, match="Unsupported target"):
        validate_any2any_request(("rgb",), "rgb")
    with pytest.raises(ValueError, match="1..3 image-derived"):
        validate_any2any_request(("rgb", "depth", "normal", "canny"), "edge")


def test_released_public_prompts_are_instruction_free_except_grounding():
    assert all(
        not BASE_MODALITY_REGISTRY.get(target).inference_add_instruction
        for target in PUBLIC_TARGETS
    )
    assert GROUNDING_PROMPT.format(phrase="red bus") == (
        "[start grounding the phrase] red bus"
    )


def test_chained_code_condition_uses_upstream_cfg_branch():
    inferencer = object.__new__(ModusInferencer)
    inferencer.init_context = lambda: {"events": []}
    inferencer.vae_transform = type(
        "_IdentityTransform",
        (),
        {"resize_transform": staticmethod(lambda image: image)},
    )()

    def update_text(text, context, *, modality=None):
        context["events"].append(("text", str(text), modality))
        return context

    def update_image(image, context, *, modality, vae=True, vit=True):
        del image
        context["events"].append(("image", modality, vae, vit))
        return context

    inferencer.update_context_text = update_text
    inferencer.update_context_image = update_image
    full, without_text, without_image, _ = inferencer._build_contexts(
        [
            ("rgb", Image.new("RGB", (4, 3))),
            ("text", "red bus"),
            (
                "det",
                CodeCondition(
                    modality="det",
                    text="<box tokens>",
                    prefix="red bus",
                ),
            ),
        ]
    )
    assert full["events"] == [
        ("image", "rgb", True, True),
        ("text", "red bus", None),
        ("text", "red bus", None),
        ("text", "<box tokens>", "det"),
    ]
    assert without_text["events"] == [("image", "rgb", True, True)]
    assert without_image["events"] == [
        ("text", "red bus", None),
        ("text", "red bus", None),
        ("text", "<box tokens>", "det"),
    ]

    _, verification_without_text, _, _ = inferencer._build_contexts(
        [("rgb", Image.new("RGB", (4, 3)))],
        vae_conditioning=False,
    )
    assert verification_without_text["events"] == [("image", "rgb", False, True)]


def test_image_generation_uses_released_flow_recipe():
    class _FlowModel:
        latent_downsample = 8
        latent_patch_size = 2
        latent_channel = 4

        def __init__(self):
            self.generated = None

        @staticmethod
        def prepare_vae_latent(**kwargs):
            del kwargs
            return {}

        @staticmethod
        def prepare_vae_latent_cfg(*args):
            del args
            return {
                "cfg_packed_position_ids": torch.tensor([0]),
                "cfg_packed_query_indexes": torch.tensor([0]),
                "cfg_key_values_lens": torch.tensor([1]),
                "cfg_packed_key_value_indexes": torch.tensor([0]),
            }

        def generate_image(self, **kwargs):
            self.generated = kwargs
            return [torch.zeros(1)]

    inferencer = object.__new__(ModusInferencer)
    inferencer.model = _FlowModel()
    inferencer.new_token_ids = {}
    inferencer.device = torch.device("cpu")
    inferencer.decode_image = lambda latent, image_shape, target: (
        latent,
        image_shape,
        target,
    )
    contexts = tuple(
        {
            "kv_lens": [1],
            "ropes": [1],
            "past_key_values": marker,
        }
        for marker in ("full", "without-text", "without-image")
    )
    decoded = inferencer._generate_image(
        target="normal",
        contexts=contexts,
        image_shape=(32, 48),
        steps=10,
        cfg=4.0,
        cfg_img=2.0,
    )
    assert decoded[1:] == ((32, 48), "normal")
    generated = inferencer.model.generated
    assert generated["num_timesteps"] == 11
    assert generated["timestep_shift"] == 3.0
    assert generated["cfg_interval"] == (0.0, 1.0)
    assert generated["cfg_renorm_type"] == "text_channel"
    assert generated["cfg_text_scale"] == 4.0
    assert generated["cfg_img_scale"] == 2.0


def test_ar_generation_uses_text_unconditional_context_for_cfg():
    inferencer = object.__new__(ModusInferencer)
    inferencer._generation_start = lambda context, target: {
        "key_values_lens": (context["name"], target, "lens"),
        "packed_key_value_indexes": (context["name"], target, "indexes"),
        "packed_query_position_ids": (context["name"], target, "positions"),
    }
    context = {"name": "without-text", "past_key_values": "cache"}
    assert inferencer._ar_cfg_args(context, "det", 2.0) == {
        "cfg_scale": 2.0,
        "cfg_past_key_values": "cache",
        "cfg_key_values_lens": ("without-text", "det", "lens"),
        "cfg_packed_key_value_indexes": ("without-text", "det", "indexes"),
        "cfg_packed_query_position_ids": ("without-text", "det", "positions"),
    }
    assert inferencer._ar_cfg_args(context, "det", 1.0) == {"cfg_scale": 1.0}


def test_cocodet_golden_grammar_maps_sparse_coco_ids():
    tokens = torch.tensor(
        [
            9000,
            100 + 100,
            1100 + 200,
            2100 + 700,
            3100 + 800,
            4100 + 1,
            100 + 300,
            1100 + 400,
            2100 + 900,
            3100 + 950,
            4100 + 12,  # sparse/missing COCO id: skipped
        ]
    )
    decoded = decode_cocodet_tokens(
        tokens,
        x1_base=100,
        y1_base=1100,
        x2_base=2100,
        y2_base=3100,
        cls_base=4100,
        start_token=9000,
        step_probs=[0.9] * 10,
    )
    assert decoded == [
        {
            "bbox": [0.1, 0.2, 0.7, 0.8],
            "label": 0,
            "score": pytest.approx(0.9),
        }
    ]


def test_grounding_golden_grammar_attaches_phrase_and_confidence():
    decoded = decode_grounding_tokens(
        [99, 10, 1010, 2020, 3030],
        x1_base=0,
        y1_base=1000,
        x2_base=2000,
        y2_base=3000,
        label="red bus",
        start_token=99,
        step_probs=[0.8, 0.7, 0.9, 0.6],
    )
    assert decoded == [
        {
            "bbox": [0.01, 0.01, 0.02, 0.03],
            "label": "red bus",
            "score": pytest.approx(0.6),
        }
    ]


def test_detection_payload_sorts_before_nms_and_max_det():
    payload = detection_payload(
        [
            {"bbox": [0.0, 0.0, 0.8, 0.8], "label": 0, "score": 0.2},
            {"bbox": [0.1, 0.1, 0.9, 0.9], "label": 0, "score": 0.9},
        ],
        (100, 50),
        iou=0.5,
        max_det=1,
    )
    assert payload["scores"] == [0.9]
    assert payload["boxes"] == [[10.0, 5.0, 90.0, 45.0]]


def test_dense_payloads_are_normalized():
    image = Image.fromarray(
        np.array([[[255, 127, 0], [127, 127, 127]]], dtype=np.uint8)
    )
    normals = image_to_payload(image, "normal", image.size)["normal"]
    np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-5)
    edge = image_to_payload(image, "canny", image.size)["edges"]
    assert edge.dtype == np.float32
    assert edge.min() >= 0 and edge.max() <= 1


def test_normal_boundary_uses_libreyolo_camera_facing_convention():
    modus_flat = Image.fromarray(
        np.array([[[128, 128, 255]]], dtype=np.uint8),
        mode="RGB",
    )
    decoded = image_to_payload(modus_flat, "normal", modus_flat.size)["normal"]
    np.testing.assert_allclose(decoded[0, 0], [0.0, 0.0, -1.0], atol=0.005)

    public_flat = np.array([[[0.0, 0.0, -1.0]]], dtype=np.float32)
    encoded = input_to_image(public_flat, "normal")
    round_trip = image_to_payload(encoded, "normal", encoded.size)["normal"]
    np.testing.assert_allclose(round_trip[0, 0], public_flat[0, 0], atol=0.005)


def test_dense_inputs_validate_ranges_and_normalize_relative_depth():
    depth = input_to_image(
        np.array([[10.0, 20.0]], dtype=np.float32),
        "depth",
    )
    np.testing.assert_array_equal(
        np.asarray(depth.convert("L")),
        np.array([[0, 255]], dtype=np.uint8),
    )
    with pytest.raises(ValueError, match=r"canny input must be in \[0, 1\]"):
        input_to_image(np.array([[0.0, 2.0]], dtype=np.float32), "canny")
    with pytest.raises(ValueError, match="NaN"):
        input_to_image(
            np.array([[[0.0, 0.0, np.nan]]], dtype=np.float32),
            "normal",
        )
    with pytest.raises(ValueError, match=r"rgb input must be in \[0, 1\]"):
        input_to_image(np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32), "rgb")


def test_external_snapshot_validation_requires_every_file(tmp_path):
    for name in REQUIRED_FILES:
        path = tmp_path / name
        path.write_bytes(b"x")
    assert validate_snapshot(tmp_path) == tmp_path.resolve()
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        validate_snapshot(tmp_path)


def test_external_download_requires_user_authentication(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    with pytest.raises(PermissionError, match="user's own Hugging Face account"):
        resolve_modus_snapshot(download_dir=tmp_path / "download")


@pytest.mark.skipif(
    getattr(torch, "float8_e4m3fn", None) is None,
    reason="PyTorch build has no E4M3 dtype",
)
def test_fp8_cache_is_local_weight_only_and_numerically_bounded(tmp_path):
    source = tmp_path / "source.safetensors"
    weight = torch.tensor(
        [[-2.0, -0.5, 0.25, 1.0], [0.1, 0.2, 0.3, 0.4]],
        dtype=torch.bfloat16,
    )
    bias = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
    save_file({"linear.weight": weight, "linear.bias": bias}, str(source))
    cache = prepare_fp8_checkpoint(
        source,
        ("linear",),
        cache_root=tmp_path / "cache",
    )
    index = json.loads((cache / "model.safetensors.index.json").read_text())
    assert "linear.weight" not in index["weight_map"]
    assert "linear.weight_fp8" in index["weight_map"]
    assert "linear.weight_scale" in index["weight_map"]

    tensors = {}
    for filename in set(index["weight_map"].values()):
        tensors.update(load_file(str(cache / filename)))
    layer = WeightOnlyFP8Linear(4, 2, bias=True, device="cpu")
    layer.weight_fp8.copy_(tensors["linear.weight_fp8"])
    layer.weight_scale.copy_(tensors["linear.weight_scale"])
    layer.bias.data.copy_(tensors["linear.bias"])
    inputs = torch.tensor([[1.0, 2.0, -1.0, 0.5]], dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(inputs, weight, bias)
    torch.testing.assert_close(layer(inputs), expected, atol=0.03, rtol=0.03)

    quantized, scale = _quantize_weight(weight)
    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 1)


class _FakeInferencer:
    def __init__(self):
        self.calls = []

    def run(
        self,
        conditions,
        *,
        target,
        steps,
        cfg,
        seed,
        grounding_phrase=None,
        cfg_img=None,
    ):
        self.calls.append(
            (tuple(conditions), target, steps, cfg, seed, grounding_phrase, cfg_img)
        )
        if target in {"depth", "normal", "canny", "samedge"}:
            color = {
                "depth": (128, 128, 128),
                "normal": (128, 128, 0),
                "canny": (255, 255, 255),
                "samedge": (64, 64, 64),
            }[target]
            return {"image": Image.new("RGB", (4, 3), color)}
        return {
            "boxes": [
                {
                    "bbox": [0.1, 0.2, 0.8, 0.9],
                    "label": grounding_phrase if target == "det" else 0,
                    "score": 0.9,
                }
            ],
            "token_text": "<tokens>",
        }

    def verification_score(self, conditions, *, candidate, target):
        del conditions, candidate, target
        return float(len(self.calls))


def _unloaded_modus():
    model = object.__new__(LibreMODUS)
    model.inferencer = _FakeInferencer()
    model.names = {0: "person"}
    model.nb_classes = 1
    model._user_vocab = False
    model.inference_steps = 10
    model.inference_cfg = 4.0
    model.inference_image_cfg = 2.0
    model.seed = 0
    return model


def test_factory_aliases_and_lazy_exports(monkeypatch):
    import libreyolo
    import libreyolo.models.modus as modus
    from libreyolo.models import vlm

    assert libreyolo.LibreMODUS is LibreMODUS
    assert libreyolo.LibreModus is LibreMODUS
    assert vlm.LibreMODUS is LibreMODUS
    assert vlm.LibreModus is LibreMODUS
    assert vlm._MODUS_ALIASES["libremodus-14b-a7b"] == "14b-a7b"

    class Sentinel:
        def __init__(self, size, **kwargs):
            self.size = size
            self.kwargs = kwargs

    monkeypatch.setattr(modus, "LibreMODUS", Sentinel)
    resolved = vlm.LibreVLM("modus", checkpoint_path="local")
    assert resolved.size == "14b-a7b"
    assert resolved.kwargs == {"checkpoint_path": "local"}


def test_any2any_returns_standard_dense_results_and_chains():
    model = _unloaded_modus()
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    result = model.any2any(
        {"rgb": image},
        "normal",
        chain=("edge",),
        steps=7,
        cfg=1.5,
        seed=4,
    )
    assert result.normal_map is not None
    result.normal_map.assert_normalized()
    assert [call[1] for call in model.inferencer.calls] == ["canny", "normal"]
    chained_condition = model.inferencer.calls[1][0][-1]
    assert chained_condition[0] == "canny"
    assert isinstance(chained_condition[1], Image.Image)


def test_standard_api_uses_released_text_and_image_guidance_defaults():
    signature = inspect.signature(LibreMODUS.__init__)
    assert signature.parameters["inference_cfg"].default == 4.0
    assert signature.parameters["inference_image_cfg"].default == 2.0

    model = _unloaded_modus()
    model._forward(
        SimpleNamespace(
            image=Image.new("RGB", (4, 3)),
            target="normal",
            grounding_phrases=(),
        )
    )
    call = model.inferencer.calls[-1]
    assert call[3] == 4.0
    assert call[6] == 2.0


def test_any2any_requires_one_aligned_image_canvas():
    model = _unloaded_modus()
    with pytest.raises(ValueError, match="share one aligned canvas"):
        model._prepare_any2any_inputs(
            {
                "rgb": Image.new("RGB", (4, 3)),
                "depth": np.zeros((2, 2), dtype=np.float32),
            }
        )


def test_any2any_rejects_string_chain():
    model = _unloaded_modus()
    with pytest.raises(TypeError, match="sequence"):
        model.any2any({"rgb": Image.new("RGB", (4, 3))}, "normal", chain="edge")


def test_any2any_cocodet_uses_coco_names_without_mutating_user_vocab():
    model = _unloaded_modus()
    model.names = {0: "custom"}
    model.nb_classes = 1
    model._user_vocab = True
    result = model.any2any({"rgb": Image.new("RGB", (4, 3))}, "detect")
    assert len(result.names) == 80
    assert result.names[0] == "person"
    assert model.names == {0: "custom"}


def test_any2any_grounding_and_best_of_n_verification():
    model = _unloaded_modus()
    result = model.any2any(
        {"rgb": Image.new("RGB", (10, 8)), "text": "red bus"},
        "grounding",
        verify=2,
    )
    assert len(result.boxes) == 1
    assert result.names == {0: "red bus"}
    assert result.verification_candidates == 2
    assert result.verification_score > 0
    assert model.names == {0: "person"}
    assert all(
        isinstance(call[0][0][1], Image.Image) for call in model.inferencer.calls
    )
