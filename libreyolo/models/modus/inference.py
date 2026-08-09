# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2026 EPFL Visual Intelligence and Learning Lab (VILAB) and the
# MODUS authors.
# SPDX-License-Identifier: Apache-2.0
#
# Inference-only adaptation of Modus any2any/inferencer.py at
# c299ef0fbba1cfe7c93336c45d7085afd770c0fa.  Feature-token modalities,
# training/demo code, segmentation, and content-generation paths are omitted.

"""Small, analysis-only any-to-any inference engine for LibreMODUS."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Optional

import torch
from PIL import Image

from ..sensenova.data_utils import pil_img2rgb
from ..sensenova.modeling.qwen2_navit import NaiveCache
from .decode import decode_cocodet_tokens, decode_grounding_tokens
from .modality import CodeCondition
from .prompts import GROUNDING_PROMPT


def _move_tensors(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    return value


def _module_device(module) -> torch.device:
    for parameter in module.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


class ModusInferencer:
    """Context assembly, flow sampling, and constrained 1D decoding."""

    def __init__(
        self,
        *,
        model,
        vae_model,
        tokenizer,
        vae_transform,
        vit_transform,
        token_artifacts,
    ):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.token_artifacts = token_artifacts
        self.new_token_ids = token_artifacts.new_token_ids
        self.modality_registry = token_artifacts.modality_registry
        self.device = _module_device(self.model.language_model.model.embed_tokens)
        self.vae_device = _module_device(self.vae_model)

    def init_context(self) -> dict:
        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(
                self.model.config.llm_config.num_hidden_layers
            ),
        }

    @torch.no_grad()
    def update_context_text(
        self,
        text: str,
        context: dict,
        *,
        modality: Optional[str] = None,
    ) -> dict:
        inputs, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=context["kv_lens"],
            curr_rope=context["ropes"],
            prompts=[str(text)],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
            modality_type=modality,
        )
        context["past_key_values"] = self.model.forward_cache_update_text(
            context["past_key_values"], **_move_tensors(inputs, self.device)
        )
        context["kv_lens"] = kv_lens
        context["ropes"] = ropes
        return context

    @torch.no_grad()
    def update_context_image(
        self,
        image: Image.Image,
        context: dict,
        *,
        modality: str,
        vae: bool = True,
        vit: bool = True,
    ) -> dict:
        if not (vae or vit):
            raise ValueError("Image conditioning requires VAE, ViT, or both.")
        kv_lens, ropes = context["kv_lens"], context["ropes"]
        if vae:
            vae_inputs, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
                modality_type=modality,
            )
            context["past_key_values"] = self.model.forward_cache_update_vae(
                self.vae_model,
                context["past_key_values"],
                **_move_tensors(vae_inputs, self.device),
            )

        if vit:
            vit_inputs, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
                modality_type=modality,
            )
            context["past_key_values"] = self.model.forward_cache_update_vit(
                context["past_key_values"],
                **_move_tensors(vit_inputs, self.device),
            )
        context["kv_lens"] = kv_lens
        context["ropes"] = ropes
        return context

    def _build_contexts(
        self,
        conditions: Iterable[tuple[str, Any]],
        *,
        vae_conditioning: bool = True,
    ):
        full = self.init_context()
        without_text = deepcopy(full)
        without_image = deepcopy(full)
        image_shape = None

        for modality, value in conditions:
            if modality == "text":
                full = self.update_context_text(value, full)
                without_image = self.update_context_text(value, without_image)
                continue
            if isinstance(value, CodeCondition):
                # Chained autoregressive codes are text-cache inputs in the
                # released inferencer: keep them in the image-unconditional
                # branch and leave the text-unconditional branch at its clean
                # image-only baseline.
                if value.prefix:
                    full = self.update_context_text(value.prefix, full)
                    without_image = self.update_context_text(
                        value.prefix, without_image
                    )
                full = self.update_context_text(
                    value.text, full, modality=value.modality
                )
                without_image = self.update_context_text(
                    value.text, without_image, modality=value.modality
                )
                continue
            image = self.vae_transform.resize_transform(pil_img2rgb(value))
            image_shape = image_shape or image.size[::-1]
            full = self.update_context_image(
                image,
                full,
                modality=modality,
                vae=vae_conditioning,
            )
            without_text = self.update_context_image(
                image,
                without_text,
                modality=modality,
                vae=vae_conditioning,
            )
        if image_shape is None:
            raise ValueError(
                "MODUS inference needs at least one image-derived condition."
            )
        return full, without_text, without_image, image_shape

    @torch.no_grad()
    def _generate_image(
        self,
        *,
        target: str,
        contexts,
        image_shape,
        steps: int,
        cfg: float,
        cfg_img: Optional[float] = None,
    ) -> Image.Image:
        full, without_text, without_image = contexts
        image_guidance = float(cfg if cfg_img is None else cfg_img)
        inputs = self.model.prepare_vae_latent(
            curr_kvlens=full["kv_lens"],
            curr_rope=full["ropes"],
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
            modality_type=target,
        )
        inputs = _move_tensors(inputs, self.device)
        cfg_text = _move_tensors(
            self.model.prepare_vae_latent_cfg(
                without_text["kv_lens"], without_text["ropes"], [image_shape]
            ),
            self.device,
        )
        cfg_image = _move_tensors(
            self.model.prepare_vae_latent_cfg(
                without_image["kv_lens"], without_image["ropes"], [image_shape]
            ),
            self.device,
        )
        unpacked = self.model.generate_image(
            past_key_values=full["past_key_values"],
            cfg_text_past_key_values=without_text["past_key_values"],
            cfg_img_past_key_values=without_image["past_key_values"],
            # Bagel's sampler consumes adjacent timestep pairs, so N+1 points
            # execute exactly N flow updates.
            num_timesteps=steps + 1,
            cfg_text_scale=cfg,
            cfg_img_scale=image_guidance,
            cfg_interval=(0.0, 1.0),
            cfg_renorm_min=0.0,
            # The upstream image-conditioned recipe renormalizes each text
            # guidance channel before applying image guidance.
            cfg_renorm_type="text_channel",
            timestep_shift=3.0,
            **inputs,
            cfg_text_packed_position_ids=cfg_text["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=cfg_text["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=cfg_text["cfg_packed_key_value_indexes"],
            cfg_img_packed_position_ids=cfg_image["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=cfg_image["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=cfg_image["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=cfg_image["cfg_packed_key_value_indexes"],
        )
        return self.decode_image(unpacked[0], image_shape, target)

    def decode_image(
        self, latent: torch.Tensor, image_shape, target: str
    ) -> Image.Image:
        height, width = image_shape
        latent_h = height // self.model.latent_downsample
        latent_w = width // self.model.latent_downsample
        patch = self.model.latent_patch_size
        latent = latent.reshape(
            1,
            latent_h,
            latent_w,
            patch,
            patch,
            self.model.latent_channel,
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent).reshape(
            1,
            self.model.latent_channel,
            latent_h * patch,
            latent_w * patch,
        )
        vae_dtype = next(self.vae_model.parameters()).dtype
        latent = latent.to(device=self.vae_device, dtype=vae_dtype)
        image = self.vae_model.decode(latent)
        pixels = (
            (image * 0.5 + 0.5)
            .clamp(0, 1)[0]
            .permute(1, 2, 0)
            .mul(255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        return Image.fromarray(pixels)

    def _generation_start(self, context: dict, target: str) -> dict:
        return _move_tensors(
            self.model.prepare_start_tokens(
                context["kv_lens"],
                context["ropes"],
                self.new_token_ids,
                target,
            ),
            self.device,
        )

    def _ar_cfg_args(
        self,
        context: Optional[dict],
        target: str,
        cfg: float,
    ) -> dict:
        if context is None or cfg <= 1.0:
            return {"cfg_scale": 1.0}
        start = self._generation_start(context, target)
        return {
            "cfg_scale": float(cfg),
            "cfg_past_key_values": context["past_key_values"],
            "cfg_key_values_lens": start["key_values_lens"],
            "cfg_packed_key_value_indexes": start["packed_key_value_indexes"],
            "cfg_packed_query_position_ids": start["packed_query_position_ids"],
        }

    @torch.no_grad()
    def verification_score(
        self,
        conditions: Iterable[tuple[str, Any]],
        *,
        candidate: tuple[str, Any],
        target: str,
    ) -> float:
        """Use the model's own yes/no VQA logits to rank a generated candidate."""
        # The released VQA path was trained with ViT-only image conditions;
        # image-producing and structured targets use VAE+ViT.
        full, _, _, _ = self._build_contexts(
            [*conditions, candidate],
            vae_conditioning=False,
        )
        question = (
            f"Does the generated {target} agree with all provided conditions? "
            "Answer yes or no."
        )
        full = self.update_context_text(question, full)
        start = self._generation_start(full, "text")
        embedding = self.model.language_model.model.embed_tokens(
            start["packed_start_tokens"]
        )
        indexes = self.model._reindex_kv(
            start["packed_key_value_indexes"], start["key_values_lens"]
        )
        _, logits = self.model._ar_forward_step(
            embedding,
            start["packed_query_position_ids"],
            start["key_values_lens"],
            indexes,
            full["past_key_values"],
        )

        def first_ids(texts):
            result = []
            for text in texts:
                values = self.tokenizer.encode(text, add_special_tokens=False)
                if values:
                    result.append(int(values[0]))
            return tuple(dict.fromkeys(result))

        yes_ids = first_ids((" yes", "yes", " Yes", "Yes"))
        no_ids = first_ids((" no", "no", " No", "No"))
        if not yes_ids or not no_ids:
            raise RuntimeError("Tokenizer cannot encode yes/no verification answers.")
        yes_logit = torch.logsumexp(logits[0, list(yes_ids)].float(), dim=0)
        no_logit = torch.logsumexp(logits[0, list(no_ids)].float(), dim=0)
        return float(torch.sigmoid(yes_logit - no_logit).item())

    @torch.no_grad()
    def _generate_cocodet(
        self,
        context: dict,
        *,
        cfg_context: Optional[dict] = None,
        cfg: float = 1.0,
    ) -> dict:
        start = self._generation_start(context, "cocodet")
        token = self.tokenizer.convert_tokens_to_ids
        x1_base = int(token("<|x1_000|>"))
        y1_base = int(token("<|y1_000|>"))
        x2_base = int(token("<|x2_000|>"))
        y2_base = int(token("<|y2_000|>"))
        cls_base = int(token("<|coco_cls_00|>"))
        tokens, probs = self.model.generate_cocodet(
            past_key_values=context["past_key_values"],
            max_length=1000,
            x1_base=x1_base,
            y1_base=y1_base,
            x2_base=x2_base,
            y2_base=y2_base,
            cls_base=cls_base,
            n_cls=91,
            cocodet_end_token=self.new_token_ids["end_of_cocodet"],
            do_sample=False,
            temperature=1.0,
            **self._ar_cfg_args(cfg_context, "cocodet", cfg),
            **start,
        )
        flattened = tokens[:, 0]
        boxes = decode_cocodet_tokens(
            flattened,
            x1_base=x1_base,
            y1_base=y1_base,
            x2_base=x2_base,
            y2_base=y2_base,
            cls_base=cls_base,
            start_token=self.new_token_ids["start_of_cocodet"],
            end_token=self.new_token_ids["end_of_cocodet"],
            step_probs=probs,
        )
        return {
            "boxes": boxes,
            "tokens": flattened,
            "token_text": self.tokenizer.decode(
                flattened[1:], skip_special_tokens=False
            ),
        }

    @torch.no_grad()
    def _generate_grounding(
        self,
        context: dict,
        phrase: str,
        *,
        cfg_context: Optional[dict] = None,
        cfg: float = 1.0,
    ) -> dict:
        start = self._generation_start(context, "det")
        token = self.tokenizer.convert_tokens_to_ids
        x1_base = int(token("<|x1_000|>"))
        y1_base = int(token("<|y1_000|>"))
        x2_base = int(token("<|x2_000|>"))
        y2_base = int(token("<|y2_000|>"))
        tokens, probs = self.model.generate_detection_coordonly(
            past_key_values=context["past_key_values"],
            max_length=5,
            x1_base=x1_base,
            y1_base=y1_base,
            x2_base=x2_base,
            y2_base=y2_base,
            det_end_token=self.new_token_ids["end_of_det"],
            do_sample=False,
            temperature=0.03,
            **self._ar_cfg_args(cfg_context, "det", cfg),
            **start,
        )
        flattened = tokens[:, 0]
        boxes = decode_grounding_tokens(
            flattened,
            x1_base=x1_base,
            y1_base=y1_base,
            x2_base=x2_base,
            y2_base=y2_base,
            label=phrase,
            start_token=self.new_token_ids["start_of_det"],
            end_token=self.new_token_ids["end_of_det"],
            step_probs=probs,
        )
        return {
            "boxes": boxes,
            "tokens": flattened,
            "token_text": self.tokenizer.decode(
                flattened[1:], skip_special_tokens=False
            ),
        }

    @torch.no_grad()
    def run(
        self,
        conditions: Iterable[tuple[str, Any]],
        *,
        target: str,
        steps: int = 10,
        cfg: float = 2.0,
        cfg_img: Optional[float] = None,
        seed: int = 0,
        grounding_phrase: Optional[str] = None,
    ) -> dict:
        if not isinstance(steps, int) or steps < 1:
            raise ValueError(f"steps must be a positive integer, got {steps!r}.")
        if not 1.0 <= float(cfg) <= 20.0:
            raise ValueError(f"cfg must be in [1, 20], got {cfg!r}.")
        if cfg_img is not None and not 1.0 <= float(cfg_img) <= 20.0:
            raise ValueError(f"cfg_img must be in [1, 20], got {cfg_img!r}.")
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        condition_list = list(conditions)
        if target == "det":
            phrase = str(grounding_phrase or "").strip()
            if not phrase:
                raise ValueError(
                    "grounding requires a phrase; pass inputs={'text': '<phrase>'} "
                    "or call set_classes(['<phrase>'])."
                )
            condition_list.append(("text", GROUNDING_PROMPT.format(phrase=phrase)))

        autocast_enabled = self.device.type in {"cuda", "cpu"}
        with torch.autocast(
            device_type=self.device.type,
            enabled=autocast_enabled,
            dtype=torch.bfloat16,
        ):
            full, without_text, without_image, image_shape = self._build_contexts(
                condition_list
            )
            if target in {"depth", "normal", "canny", "samedge"}:
                image = self._generate_image(
                    target=target,
                    contexts=(full, without_text, without_image),
                    image_shape=image_shape,
                    steps=steps,
                    cfg=float(cfg),
                    cfg_img=cfg_img,
                )
                return {"image": image, "image_shape": image_shape}
            if target == "cocodet":
                result = self._generate_cocodet(
                    full,
                    cfg_context=without_text,
                    cfg=float(cfg),
                )
                result["image_shape"] = image_shape
                return result
            if target == "det":
                result = self._generate_grounding(
                    full,
                    phrase,
                    cfg_context=without_text,
                    cfg=float(cfg),
                )
                result["image_shape"] = image_shape
                return result
        raise ValueError(f"Unsupported inference target {target!r}.")
