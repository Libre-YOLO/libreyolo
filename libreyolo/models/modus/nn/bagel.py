# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2026 EPFL Visual Intelligence and Learning Lab (VILAB) and the
# MODUS authors.
# SPDX-License-Identifier: Apache-2.0
#
# MODUS-specific inference extensions adapted from modeling/bagel/bagel.py at
# c299ef0fbba1cfe7c93336c45d7085afd770c0fa.  The shared Bagel trunk comes
# from LibreYOLO's audited Apache-2.0 SenseNova port.  The upstream
# CC-BY-NC modeling_utils.py file is not imported, copied, or adapted.

"""Modality-aware Bagel inference required by the MODUS checkpoint."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ...sensenova.modeling.bagel import Bagel
from ...sensenova.modeling.layers import (
    LearnableEmbedding,
    sincos_position_embedding_2d,
)
from ...sensenova.modeling.qwen2_navit import NaiveCache


class ModusBagel(Bagel):
    """Bagel with the released MODUS modality registry and constrained heads."""

    def __init__(self, language_model, vit_model, config, modality_registry=None):
        self.modality_registry = None
        super().__init__(language_model, vit_model, config)
        # The released MODUS safetensor omits these two deterministic sine/cos
        # tables. Keep them out of state_dict and materialize them after
        # accelerate dispatch; every learned tensor then matches the checkpoint
        # one-for-one.
        for module in (self.latent_pos_embed, self.vit_pos_embed):
            table = module._parameters.pop("pos_embed")
            module.register_buffer("pos_embed", table, persistent=False)
        if modality_registry is not None:
            self.set_modality_registry(modality_registry)

    def materialize_static_position_embeddings(self, device_map) -> None:
        """Rebuild deterministic tables omitted from the released checkpoint."""
        for name, module in (
            ("latent_pos_embed", self.latent_pos_embed),
            ("vit_pos_embed", self.vit_pos_embed),
        ):
            matches = (
                (prefix, target)
                for prefix, target in device_map.items()
                if not prefix or name == prefix or name.startswith(f"{prefix}.")
            )
            target = max(matches, key=lambda item: len(item[0]), default=("", "cpu"))[1]
            if target == "disk":
                target = "cpu"
            if isinstance(target, int):
                target = f"cuda:{target}"
            table = sincos_position_embedding_2d(
                module.hidden_size, module.max_num_patch_per_side
            )
            module.pos_embed = table.to(target)

    def set_modality_registry(self, modality_registry) -> None:
        """Attach runtime token ranges and create checkpoint-named embeddings."""
        self.modality_registry = modality_registry
        if modality_registry is None:
            return
        for spec in modality_registry.modalities_with_forward_pos_embed():
            attr_name = f"{spec.pos_embed_name or spec.name}_pos_embed"
            if not hasattr(self, attr_name):
                setattr(
                    self,
                    attr_name,
                    LearnableEmbedding(int(spec.pos_embed_size), self.hidden_size),
                )

    def _resolve_token_keys(self, modality_type: Optional[str]) -> tuple[str, str]:
        if modality_type is None:
            return "bos_token_id", "eos_token_id"
        spec = self.modality_registry.get(modality_type)
        return spec.start_token_key, spec.end_token_key

    def prepare_prompts(
        self,
        curr_kvlens,
        curr_rope,
        prompts,
        tokenizer,
        new_token_ids,
        modality_type=None,
    ):
        inputs, new_lens, new_rope = super().prepare_prompts(
            curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids
        )
        start_key, end_key = self._resolve_token_keys(modality_type)
        offset = 0
        for length in inputs["text_token_lens"].tolist():
            inputs["packed_text_ids"][offset] = new_token_ids[start_key]
            inputs["packed_text_ids"][offset + int(length) - 1] = new_token_ids[end_key]
            offset += int(length)
        return inputs, new_lens, new_rope

    def _replace_image_start(self, inputs, new_token_ids, modality_type):
        start_key, _ = self._resolve_token_keys(modality_type)
        inputs["packed_text_ids"][0::2] = int(new_token_ids[start_key])
        return inputs

    def prepare_vit_images(
        self,
        curr_kvlens,
        curr_rope,
        images,
        transforms,
        new_token_ids,
        modality_type=None,
    ):
        inputs, new_lens, new_rope = super().prepare_vit_images(
            curr_kvlens, curr_rope, images, transforms, new_token_ids
        )
        return (
            self._replace_image_start(inputs, new_token_ids, modality_type),
            new_lens,
            new_rope,
        )

    def prepare_vae_images(
        self,
        curr_kvlens,
        curr_rope,
        images,
        transforms,
        new_token_ids,
        timestep=0,
        modality_type=None,
    ):
        inputs, new_lens, new_rope = super().prepare_vae_images(
            curr_kvlens,
            curr_rope,
            images,
            transforms,
            new_token_ids,
            timestep=timestep,
        )
        return (
            self._replace_image_start(inputs, new_token_ids, modality_type),
            new_lens,
            new_rope,
        )

    def prepare_vae_latent(
        self,
        curr_kvlens,
        curr_rope,
        image_sizes,
        new_token_ids,
        modality_type=None,
    ):
        inputs = super().prepare_vae_latent(
            curr_kvlens, curr_rope, image_sizes, new_token_ids
        )
        return self._replace_image_start(inputs, new_token_ids, modality_type)

    def prepare_start_tokens(
        self,
        curr_kvlens,
        curr_rope,
        new_token_ids,
        modality_type=None,
    ):
        inputs = super().prepare_start_tokens(curr_kvlens, curr_rope, new_token_ids)
        start_key, _ = self._resolve_token_keys(modality_type)
        inputs["packed_start_tokens"].fill_(int(new_token_ids[start_key]))
        return inputs

    @torch.no_grad()
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        pos_embed_key: Optional[str] = None,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        specs = ()
        if self.modality_registry is not None:
            if pos_embed_key is None:
                specs = self.modality_registry.modalities_with_forward_pos_embed()
            else:
                spec = self.modality_registry.get(pos_embed_key)
                specs = (spec,) if spec.apply_pos_embed_in_forward else ()

        text_lens = [int(value) for value in text_token_lens.detach().cpu().tolist()]
        for spec in specs:
            if spec.code_token_range is None:
                continue
            base, length = spec.code_token_range
            token_mask = (packed_text_ids >= base) & (packed_text_ids < base + length)
            code_indices = torch.nonzero(token_mask, as_tuple=False).flatten()
            if code_indices.numel() == 0:
                continue
            code_positions = torch.zeros_like(code_indices)
            seq_start = 0
            for seq_len in text_lens:
                in_sequence = (code_indices >= seq_start) & (
                    code_indices < seq_start + seq_len
                )
                count = int(in_sequence.sum().item())
                if count:
                    code_positions[in_sequence] = torch.arange(
                        count,
                        device=packed_text_ids.device,
                        dtype=torch.long,
                    )
                seq_start += seq_len
            position_embedding = getattr(
                self, f"{spec.pos_embed_name or spec.name}_pos_embed"
            )(code_positions)
            packed_text_embedding[code_indices] += position_embedding.to(
                packed_text_embedding.dtype
            )

        extra_inputs = {"mode": "und"} if self.use_moe else {}
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        return output.past_key_values

    @staticmethod
    def _reindex_kv(packed_key_value_indexes, key_values_lens):
        unpacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
        for index, values in enumerate(unpacked):
            unpacked[index] = values + index
        return torch.cat(unpacked, dim=0)

    @staticmethod
    def _advance_kv(packed_key_value_indexes, key_values_lens):
        unpacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
        for index, values in enumerate(unpacked):
            next_index = (
                values[-1] + 1
                if values.numel()
                else torch.tensor(0, device=values.device, dtype=values.dtype)
            )
            unpacked[index] = torch.cat((values, next_index.reshape(1)))
        return torch.cat(unpacked, dim=0), key_values_lens + 1

    def _ar_forward_step(
        self,
        packed_text_embedding,
        packed_query_position_ids,
        key_values_lens,
        packed_key_value_indexes,
        past_key_values,
    ):
        query_lens = torch.ones(
            packed_text_embedding.shape[0],
            device=packed_text_embedding.device,
            dtype=torch.long,
        )
        packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
            len(key_values_lens),
            device=key_values_lens.device,
            dtype=key_values_lens.dtype,
        )
        extra_inputs = {"mode": "und"} if self.use_moe else {}
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        return output.past_key_values, self.language_model.lm_head(
            output.packed_query_sequence
        )

    def _cfg_forward_and_merge(
        self,
        packed_text_embedding,
        pred_logits,
        cfg_scale,
        cfg_past_key_values,
        cfg_key_values_lens,
        cfg_packed_key_value_indexes,
        cfg_packed_query_position_ids,
    ):
        enabled = (
            cfg_scale > 1.0
            and cfg_past_key_values is not None
            and cfg_key_values_lens is not None
            and cfg_packed_key_value_indexes is not None
            and cfg_packed_query_position_ids is not None
            and cfg_key_values_lens.numel() == packed_text_embedding.shape[0]
        )
        if not enabled:
            return cfg_past_key_values, cfg_packed_key_value_indexes, pred_logits
        cfg_packed_key_value_indexes = self._reindex_kv(
            cfg_packed_key_value_indexes, cfg_key_values_lens
        )
        cfg_past_key_values, cfg_pred_logits = self._ar_forward_step(
            packed_text_embedding,
            cfg_packed_query_position_ids,
            cfg_key_values_lens,
            cfg_packed_key_value_indexes,
            cfg_past_key_values,
        )
        merged = pred_logits + cfg_scale * (pred_logits - cfg_pred_logits)
        return cfg_past_key_values, cfg_packed_key_value_indexes, merged

    @staticmethod
    def _choose_token(pred_logits, *, do_sample, temperature):
        scaled = pred_logits / temperature if do_sample else pred_logits
        probs = nn.functional.softmax(scaled, dim=-1)
        if do_sample:
            token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            token = torch.argmax(pred_logits, dim=-1)
        chosen_prob = probs.gather(1, token[:, None]).squeeze(1)
        return token, chosen_prob

    @torch.no_grad()
    def generate_detection_coordonly(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        *,
        x1_base: int,
        y1_base: int,
        x2_base: int,
        y2_base: int,
        det_end_token: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        cfg_past_key_values: Optional[NaiveCache] = None,
        cfg_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_packed_query_position_ids: Optional[torch.LongTensor] = None,
    ):
        """Constrained ``x1 -> y1 -> x2 -> y2 -> end`` grounding decode."""
        generated, chosen_probs = [], []
        current = packed_start_tokens
        state = 0
        lower_tokens = [None, None]
        bases = (x1_base, y1_base, x2_base, y2_base)

        for _ in range(max_length):
            generated.append(current)
            embedding = self.language_model.model.embed_tokens(current)
            packed_key_value_indexes = self._reindex_kv(
                packed_key_value_indexes, key_values_lens
            )
            past_key_values, logits = self._ar_forward_step(
                embedding,
                packed_query_position_ids,
                key_values_lens,
                packed_key_value_indexes,
                past_key_values,
            )
            cfg_past_key_values, cfg_packed_key_value_indexes, logits = (
                self._cfg_forward_and_merge(
                    embedding,
                    logits,
                    cfg_scale,
                    cfg_past_key_values,
                    cfg_key_values_lens,
                    cfg_packed_key_value_indexes,
                    cfg_packed_query_position_ids,
                )
            )
            logits = logits.float()
            mask = torch.full_like(logits, float("-inf"))
            if state in (0, 1):
                mask[:, bases[state] : bases[state] + 1000] = 0
            elif state in (2, 3):
                offset = int(lower_tokens[state - 2]) - bases[state - 2]
                mask[:, bases[state] + offset : bases[state] + 1000] = 0
            else:
                mask[:, det_end_token] = 0
            current, probability = self._choose_token(
                logits + mask, do_sample=do_sample, temperature=temperature
            )
            chosen_probs.append(probability[0])
            if state == 0:
                lower_tokens[0] = current[0]
            elif state == 1:
                lower_tokens[1] = current[0]
            state += 1

            packed_key_value_indexes, key_values_lens = self._advance_kv(
                packed_key_value_indexes, key_values_lens
            )
            packed_query_position_ids += 1
            if cfg_scale > 1.0 and cfg_key_values_lens is not None:
                cfg_packed_key_value_indexes, cfg_key_values_lens = self._advance_kv(
                    cfg_packed_key_value_indexes, cfg_key_values_lens
                )
                cfg_packed_query_position_ids += 1
            if state > 4:
                break

        device = generated[0].device
        return (
            torch.stack([value.to(device) for value in generated]),
            torch.stack(chosen_probs).to(device),
        )

    @torch.no_grad()
    def generate_cocodet(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        *,
        x1_base: int,
        y1_base: int,
        x2_base: int,
        y2_base: int,
        cls_base: int,
        n_cls: int,
        cocodet_end_token: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        cfg_past_key_values: Optional[NaiveCache] = None,
        cfg_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_packed_query_position_ids: Optional[torch.LongTensor] = None,
    ):
        """Constrained Pix2Seq-style ``coords,class`` box generation."""
        generated, chosen_probs = [], []
        current = packed_start_tokens
        state = 0
        lower_tokens = [None, None]
        bases = (x1_base, y1_base, x2_base, y2_base)

        for _ in range(max_length):
            generated.append(current)
            embedding = self.language_model.model.embed_tokens(current)
            packed_key_value_indexes = self._reindex_kv(
                packed_key_value_indexes, key_values_lens
            )
            past_key_values, logits = self._ar_forward_step(
                embedding,
                packed_query_position_ids,
                key_values_lens,
                packed_key_value_indexes,
                past_key_values,
            )
            cfg_past_key_values, cfg_packed_key_value_indexes, logits = (
                self._cfg_forward_and_merge(
                    embedding,
                    logits,
                    cfg_scale,
                    cfg_past_key_values,
                    cfg_key_values_lens,
                    cfg_packed_key_value_indexes,
                    cfg_packed_query_position_ids,
                )
            )
            logits = logits.float()
            mask = torch.full_like(logits, float("-inf"))
            if state in (0, 1):
                mask[:, bases[state] : bases[state] + 1000] = 0
            elif state in (2, 3):
                offset = int(lower_tokens[state - 2]) - bases[state - 2]
                mask[:, bases[state] + offset : bases[state] + 1000] = 0
            elif state == 4:
                mask[:, cls_base : cls_base + n_cls] = 0
            else:
                mask[:, x1_base : x1_base + 1000] = 0
                mask[:, cocodet_end_token] = 0
            current, probability = self._choose_token(
                logits + mask, do_sample=do_sample, temperature=temperature
            )
            chosen_probs.append(probability[0])
            if state == 0:
                lower_tokens[0] = current[0]
            elif state == 1:
                lower_tokens[1] = current[0]

            if state < 4:
                state += 1
            elif state == 4:
                state = 9
            elif int(current[0]) == cocodet_end_token:
                state = 10
            else:
                lower_tokens[0] = current[0]
                state = 1

            packed_key_value_indexes, key_values_lens = self._advance_kv(
                packed_key_value_indexes, key_values_lens
            )
            packed_query_position_ids += 1
            if cfg_scale > 1.0 and cfg_key_values_lens is not None:
                cfg_packed_key_value_indexes, cfg_key_values_lens = self._advance_kv(
                    cfg_packed_key_value_indexes, cfg_key_values_lens
                )
                cfg_packed_query_position_ids += 1
            if state == 10:
                break

        device = generated[0].device
        return (
            torch.stack([value.to(device) for value in generated]),
            torch.stack(chosen_probs).to(device),
        )
