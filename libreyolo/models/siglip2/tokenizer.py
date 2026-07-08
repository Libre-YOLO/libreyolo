"""SigLIP multilingual SentencePiece tokenizer (vendored, native).

SigLIP 2 tokenizes text with a multilingual SentencePiece model (the Gemma
vocabulary, 256k pieces), not CLIP's BPE. The model file
(``siglip2_tokenizer.model``) is shipped alongside this module from the
Apache-2.0 ``google/siglip2-*`` release so LibreSigLIP2 tokenizes without a
``transformers`` dependency at runtime.

``sentencepiece`` is the only runtime dependency, pulled in via the optional
``libreyolo[siglip2]`` extra and imported lazily with a clear install message,
mirroring how LibreCLIP defers ``ftfy``/``regex``.

Tokenization exactly matches the reference processor (verified token-for-token):
SentencePiece encode, append the end-of-text id, right-pad with the pad id to a
fixed 64-token length. No lowercasing is applied - the reference fast tokenizer
does not lowercase despite the ``do_lower_case`` config flag, and matching it is
what keeps the text tower bit-identical.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Union

import torch

# SigLIP was trained with a fixed 64-token context and ``padding="max_length"``.
DEFAULT_CONTEXT_LENGTH = 64

_SIGLIP2_EXTRA_HINT = (
    "LibreSigLIP2's text tokenizer needs 'sentencepiece'. Install it with:\n"
    '    pip install "libreyolo[siglip2]"'
)


def _require_sentencepiece():
    try:
        import sentencepiece
    except ImportError as exc:  # pragma: no cover - exercised via install hint
        raise ImportError(_SIGLIP2_EXTRA_HINT) from exc
    return sentencepiece


@lru_cache()
def default_spm_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "siglip2_tokenizer.model"
    )


class SigLIP2Tokenizer:
    """SentencePiece tokenizer producing SigLIP's exact 64-length token IDs.

    Padding uses the model's pad id (0) on the right; every sequence ends with
    the end-of-text id (1) before padding, and is truncated to ``context_length``
    keeping the end-of-text id last.
    """

    def __init__(
        self,
        spm_path: str | None = None,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> None:
        spm = _require_sentencepiece()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path or default_spm_path())
        self.context_length = context_length
        eos = self.sp.eos_id()
        pad = self.sp.pad_id()
        # Fall back to the known SigLIP/Gemma ids if the model omits them.
        self.eot_token_id = eos if eos >= 0 else 1
        self.pad_token_id = pad if pad >= 0 else 0
        self.vocab_size = self.sp.GetPieceSize()

    def encode(self, text: str) -> List[int]:
        return list(self.sp.EncodeAsIds(text))

    def __call__(
        self,
        texts: Union[str, List[str]],
        context_length: int | None = None,
    ) -> torch.LongTensor:
        """Tokenize string(s) into a ``[N, context_length]`` long tensor."""
        if isinstance(texts, str):
            texts = [texts]
        context_length = context_length or self.context_length

        all_tokens = [self.encode(text) + [self.eot_token_id] for text in texts]
        result = torch.full(
            (len(all_tokens), context_length), self.pad_token_id, dtype=torch.long
        )
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = self.eot_token_id
            result[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return result
