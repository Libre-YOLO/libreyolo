# Copyright 2026 EPFL Visual Intelligence and Learning Lab (VILAB) and the
# MODUS authors.
# SPDX-License-Identifier: Apache-2.0
#
# Inference-only adaptation of Modus core/modality.py at
# c299ef0fbba1cfe7c93336c45d7085afd770c0fa.  Training loss configuration,
# dataset sampling fields, and external feature-tokenizer support are omitted.

"""Typed modality metadata used by the LibreMODUS inference path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Optional, Tuple

TokenRange = Tuple[int, int]


@dataclass(frozen=True)
class CodeCondition:
    """Generated codebook text fed back during chained inference."""

    modality: str
    text: str
    prefix: Optional[str] = None


@dataclass(frozen=True)
class CodeTokenGroup:
    """One formatted, inclusive token interval."""

    token_format: str
    start: int
    end: int

    def tokens(self) -> tuple[str, ...]:
        return tuple(
            self.token_format.format(i=i) for i in range(self.start, self.end + 1)
        )


@dataclass(frozen=True)
class ModalitySpec:
    """Inference fields for one entry in the released 16-modality registry."""

    name: str
    id: int
    kind: str
    start_token_key: str
    end_token_key: str
    start_token: Optional[str] = None
    end_token: Optional[str] = None
    extra_tokens: tuple[str, ...] = ()
    code_token_groups: tuple[CodeTokenGroup, ...] = ()
    code_vocab_size: Optional[int] = None
    code_token_format: str = "<|{prefix}_{i:04d}|>"
    code_token_prefix: Optional[str] = None
    dispersed_code_tokens: bool = False
    code_token_range: Optional[TokenRange] = None
    code_token_ids: Optional[tuple[int, ...]] = None
    pos_embed_size: Optional[int] = None
    apply_pos_embed_in_forward: bool = False
    pos_embed_name: Optional[str] = None
    represent_vit: bool = True
    represent_vae: bool = False
    inference_decode_method: str = "auto"
    inference_max_tokens: Optional[int] = None
    inference_cfg_uncond: str = "auto"
    inference_add_instruction: bool = True
    inference_cfg_img_scale: Optional[float] = None

    def code_tokens(self) -> tuple[str, ...]:
        """Return this modality's code-token strings in checkpoint order."""
        if self.kind != "codebook":
            return ()
        if self.code_token_groups:
            return tuple(
                token for group in self.code_token_groups for token in group.tokens()
            )
        if self.code_vocab_size is None:
            raise ValueError(
                f"Codebook modality {self.name!r} has neither token groups nor a vocab size."
            )
        prefix = self.code_token_prefix or self.name
        return tuple(
            self.code_token_format.format(prefix=prefix, i=i)
            for i in range(self.code_vocab_size)
        )


class ModalityRegistry:
    """Validated name/id lookup for MODUS inference."""

    def __init__(self, specs: Iterable[ModalitySpec]):
        ordered = tuple(specs)
        by_name = {spec.name: spec for spec in ordered}
        by_id = {spec.id: spec for spec in ordered}
        if len(by_name) != len(ordered):
            raise ValueError("Modality names must be unique.")
        if len(by_id) != len(ordered):
            raise ValueError("Modality ids must be unique.")
        if any(spec.kind not in {"text", "image", "codebook"} for spec in ordered):
            raise ValueError("Modality kind must be text, image, or codebook.")
        self._ordered = ordered
        self._by_name = by_name
        self._by_id = by_id

    def __len__(self) -> int:
        return len(self._ordered)

    def __iter__(self):
        return iter(self._ordered)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._ordered)

    def get(self, name: str) -> ModalitySpec:
        try:
            return self._by_name[str(name)]
        except KeyError as exc:
            raise KeyError(
                f"Unknown modality {name!r}; expected one of: {', '.join(self.names)}."
            ) from exc

    def by_id(self, modality_id: int) -> ModalitySpec:
        try:
            return self._by_id[int(modality_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown modality id {modality_id!r}.") from exc

    def modalities_with_forward_pos_embed(self) -> tuple[ModalitySpec, ...]:
        return tuple(
            spec
            for spec in self._ordered
            if spec.pos_embed_size is not None and spec.apply_pos_embed_in_forward
        )

    def resolve_decode_method(self, name: str) -> str:
        spec = self.get(name)
        if spec.inference_decode_method != "auto":
            return spec.inference_decode_method
        return {"text": "text", "image": "image", "codebook": "text"}[spec.kind]

    def resolve_token_key(
        self,
        name: str,
        token_ids: Mapping[str, int],
        *,
        end: bool = False,
    ) -> int:
        spec = self.get(name)
        key = spec.end_token_key if end else spec.start_token_key
        try:
            return int(token_ids[key])
        except KeyError as exc:
            raise KeyError(
                f"Tokenizer artifacts are missing {key!r} for modality {name!r}."
            ) from exc

    def with_runtime_tokens(
        self,
        token_ranges: Mapping[str, TokenRange],
        code_token_ids: Mapping[str, tuple[int, ...]],
    ) -> "ModalityRegistry":
        """Return a registry carrying token ids computed from the loaded tokenizer."""
        return ModalityRegistry(
            replace(
                spec,
                code_token_range=token_ranges.get(spec.name, spec.code_token_range),
                code_token_ids=code_token_ids.get(spec.name, spec.code_token_ids),
            )
            for spec in self._ordered
        )


def infer_contiguous_token_range(token_ids: Iterable[int]) -> TokenRange:
    """Validate contiguity and return ``(base, length)``."""
    values = sorted(int(token_id) for token_id in token_ids)
    if not values:
        raise ValueError("token_ids cannot be empty.")
    expected = list(range(values[0], values[0] + len(values)))
    if values != expected:
        raise ValueError(
            "Token ids are not contiguous; "
            f"base={values[0]}, length={len(values)}, first={values[:10]}."
        )
    return values[0], len(values)
