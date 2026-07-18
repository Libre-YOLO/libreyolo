"""Offline integration test for the vendored SenseNova (Bagel-MoT) stack.

Builds a miniature random-weight Bagel and drives the real
InterleaveInferencer through all three generation modes on CPU. This
exercises the packed SDPA attention fallback (both MoT modes), KV-cache
merging, rotary embeddings, CFG flow sampling, and VAE decode, none of which
the parser-level tests touch.
"""

import pytest
import torch
from PIL import Image

from libreyolo.models.sensenova.inferencer import InterleaveInferencer
from libreyolo.models.sensenova.modeling.autoencoder import AutoEncoder, AutoEncoderParams
from libreyolo.models.sensenova.modeling.bagel import Bagel, BagelConfig
from libreyolo.models.sensenova.modeling.qwen2_navit import Qwen2Config, Qwen2ForCausalLM
from libreyolo.models.sensenova.modeling.siglip_navit import (
    SiglipVisionConfig,
    SiglipVisionModel,
)
from libreyolo.models.sensenova.transforms import ImageTransform

pytestmark = pytest.mark.unit


class _FakeTokenizer:
    """Minimal tokenizer surface the inferencer and Bagel.prepare_* use."""

    special_tokens_map = {}

    def add_tokens(self, tokens):
        self._extra = {t: 500 + i for i, t in enumerate(tokens)}
        return len(tokens)

    def convert_tokens_to_ids(self, token):
        return self._extra[token]

    def encode(self, text):
        return [(7 + ord(c)) % 400 for c in text[:12]]

    def decode(self, ids):
        return "<|im_start|>tiny output text<|im_end|>"


@pytest.fixture(scope="module")
def tiny_inferencer():
    llm_config = Qwen2Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        qk_norm=True,
        layer_module="Qwen2MoTDecoderLayer",
    )
    vit_config = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=64,
        patch_size=16,
        rope=False,
    )
    vae_params = AutoEncoderParams(
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
    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_params,
        latent_patch_size=2,
        max_latent_size=64,
        # Must equal image_size // patch_size so flattened position ids stay
        # in range of the tower's embedding table (real model: 980 // 14 = 70).
        vit_max_num_patch_per_side=4,
        connector_act="gelu_pytorch_tanh",
    )

    torch.manual_seed(0)
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, bagel_config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    model = model.to(torch.bfloat16).eval()
    vae = AutoEncoder(vae_params).to(torch.bfloat16).eval()

    tokenizer = _FakeTokenizer()
    tokenizer.add_tokens(
        ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]
    )
    new_token_ids = dict(
        bos_token_id=500, eos_token_id=501, start_of_image=502, end_of_image=503
    )

    return InterleaveInferencer(
        model=model,
        vae_model=vae,
        tokenizer=tokenizer,
        vae_transform=ImageTransform(64, 32, 16),
        vit_transform=ImageTransform(64, 32, 16),
        new_token_ids=new_token_ids,
        device="cpu",
    )


@pytest.fixture()
def image():
    return Image.new("RGB", (64, 64), (120, 40, 200))


def test_understanding_mode_generates_text(tiny_inferencer, image):
    out = tiny_inferencer.interleave_inference(
        [image, "describe"],
        understanding_output=True,
        max_think_token_n=5,
        do_sample=False,
    )
    assert any(isinstance(o, str) for o in out)


def test_dense_perception_mode_generates_image(tiny_inferencer, image):
    out = tiny_inferencer.interleave_inference(
        [image, "estimate depth"],
        understanding_output=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=4,
        cfg_renorm_min=1.0,
        cfg_renorm_type="text_channel",
    )
    images = [o for o in out if isinstance(o, Image.Image)]
    assert images and images[0].size == (64, 64)


def test_caption_generate_mode_yields_text_then_image(tiny_inferencer, image):
    out = tiny_inferencer.interleave_inference(
        [image, "caption then mask"],
        understanding_output=False,
        caption=True,
        max_think_token_n=5,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=4.0,
        num_timesteps=4,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
    )
    assert any(isinstance(o, str) for o in out)
    assert any(isinstance(o, Image.Image) for o in out)
