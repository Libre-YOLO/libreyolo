# Copyright 2026 EPFL Visual Intelligence and Learning Lab (VILAB) and the
# MODUS authors.
# SPDX-License-Identifier: Apache-2.0
#
# Inference-only adaptation of Modus core/tokenizer_utils.py and
# data/data_utils.py at c299ef0fbba1cfe7c93336c45d7085afd770c0fa.

"""Checkpoint-compatible MODUS tokenizer assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .modality import ModalityRegistry, TokenRange, infer_contiguous_token_range
from .prompts import BASE_MODALITY_REGISTRY

_UNIVERSAL_TOKENS = (
    ("bos_token_id", "<|im_start|>"),
    ("eos_token_id", "<|im_end|>"),
    ("start_of_image", "<|vision_start|>"),
    ("end_of_image", "<|vision_end|>"),
    ("start_of_caption", "<|caption_start|>"),
    ("end_of_caption", "<|caption_end|>"),
    ("start_of_depth", "<|depth_start|>"),
    ("end_of_depth", "<|depth_end|>"),
    ("start_of_normal", "<|normal_start|>"),
    ("end_of_normal", "<|normal_end|>"),
)


@dataclass(frozen=True)
class TokenizerArtifacts:
    tokenizer: Any
    new_token_ids: dict[str, int]
    token_ranges: dict[str, TokenRange]
    code_token_ids: dict[str, tuple[int, ...]]
    modality_registry: ModalityRegistry
    num_new_tokens: int


def _unique(tokens: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(token) for token in tokens))


def _vocab(tokenizer) -> dict[str, int]:
    vocab = tokenizer.get_vocab()
    return {str(token): int(token_id) for token, token_id in vocab.items()}


def _token_id(tokenizer, token: str, vocab: dict[str, int] | None = None) -> int:
    vocab = _vocab(tokenizer) if vocab is None else vocab
    token_id = int(vocab.get(token, -1))
    if token_id < 0 or int(tokenizer.convert_tokens_to_ids(token)) != token_id:
        raise RuntimeError(
            f"Tokenizer did not register required MODUS token {token!r}."
        )
    return token_id


def _add_tokens(tokenizer, tokens: Iterable[str]) -> int:
    ordered = _unique(tokens)
    if not ordered:
        return 0
    return int(tokenizer.add_tokens(ordered))


def build_modus_tokenizer(
    tokenizer,
    registry: ModalityRegistry = BASE_MODALITY_REGISTRY,
) -> TokenizerArtifacts:
    """Add/resolve all 16-modality tokens in released-checkpoint order."""
    before = len(tokenizer)
    _add_tokens(tokenizer, (token for _, token in _UNIVERSAL_TOKENS))
    vocab = _vocab(tokenizer)
    token_ids = {
        key: _token_id(tokenizer, token, vocab) for key, token in _UNIVERSAL_TOKENS
    }
    ranges: dict[str, TokenRange] = {}
    dispersed: dict[str, tuple[int, ...]] = {}
    deferred: list[tuple[str, str]] = []

    for spec in registry:
        if spec.kind != "codebook":
            vocab = _vocab(tokenizer)
            for key, token in (
                (spec.start_token_key, spec.start_token),
                (spec.end_token_key, spec.end_token),
            ):
                if key in token_ids:
                    continue
                if token is None:
                    token_id = getattr(tokenizer, key, None)
                    if token_id is None:
                        raise KeyError(
                            f"Cannot resolve tokenizer field {key!r} for {spec.name!r}."
                        )
                    token_ids[key] = int(token_id)
                elif token in vocab:
                    token_ids[key] = int(vocab[token])
                else:
                    deferred.append((key, token))
            continue

        code_tokens = spec.code_tokens()
        delimiters = tuple(
            token
            for token in (spec.start_token, spec.end_token, *spec.extra_tokens)
            if token is not None
        )
        _add_tokens(tokenizer, (*code_tokens, *delimiters))
        vocab = _vocab(tokenizer)
        if spec.start_token is not None:
            token_ids[spec.start_token_key] = _token_id(
                tokenizer, spec.start_token, vocab
            )
        if spec.end_token is not None:
            token_ids[spec.end_token_key] = _token_id(tokenizer, spec.end_token, vocab)

        code_ids = tuple(_token_id(tokenizer, token, vocab) for token in code_tokens)
        if spec.dispersed_code_tokens:
            end_id = token_ids.get(spec.end_token_key)
            dispersed[spec.name] = code_ids + (() if end_id is None else (end_id,))
        else:
            ranges[spec.name] = infer_contiguous_token_range(code_ids)

    # New image delimiters are deliberately last so they cannot shift any
    # checkpoint-trained codebook interval.
    for key, token in deferred:
        if key in token_ids:
            continue
        _add_tokens(tokenizer, (token,))
        vocab = _vocab(tokenizer)
        token_ids[key] = _token_id(tokenizer, token, vocab)

    runtime_registry = registry.with_runtime_tokens(ranges, dispersed)
    return TokenizerArtifacts(
        tokenizer=tokenizer,
        new_token_ids=token_ids,
        token_ranges=ranges,
        code_token_ids=dispersed,
        modality_registry=runtime_registry,
        num_new_tokens=len(tokenizer) - before,
    )


def assert_checkpoint_vocabulary(
    artifacts: TokenizerArtifacts, vocab_size: int
) -> None:
    """Fail before loading if tokenizer ids cannot index the checkpoint tables."""
    max_id = max(
        [*artifacts.new_token_ids.values()]
        + [base + length - 1 for base, length in artifacts.token_ranges.values()]
        + [token for values in artifacts.code_token_ids.values() for token in values]
    )
    if len(artifacts.tokenizer) > vocab_size or max_id >= vocab_size:
        raise RuntimeError(
            "MODUS tokenizer/checkpoint mismatch: "
            f"tokenizer length={len(artifacts.tokenizer)}, max token id={max_id}, "
            f"checkpoint vocab_size={vocab_size}."
        )
