"""CLIP byte-pair-encoding tokenizer (vendored, native — no open_clip at runtime).

This is a clean-room-faithful vendoring of the original CLIP ``SimpleTokenizer``
(OpenAI CLIP / open_clip, both MIT licensed) so LibreCLIP can tokenize text
without depending on ``open_clip_torch`` at runtime. The BPE merge table
(``bpe_simple_vocab_16e6.txt.gz``) is shipped alongside this module and is the
same 49,408-entry vocabulary every CLIP model uses.

The two text-cleaning dependencies (``ftfy`` and ``regex``) are required to
reproduce CLIP's exact token IDs (the pattern uses Unicode property classes
``\\p{L}``/``\\p{N}`` that the stdlib ``re`` module cannot express, and
``ftfy.fix_text`` is part of the reference ``basic_clean``). They are pulled in
via the optional ``libreyolo[clip]`` extra and imported lazily with a clear
install message — keeping the base library dependency-free.

Original copyright: (c) 2021 OpenAI, MIT License. Vendored for LibreYOLO.
"""

from __future__ import annotations

import gzip
import html
import os
from functools import lru_cache
from typing import List, Union

import torch

DEFAULT_CONTEXT_LENGTH = 77  # every CLIP model uses 77
SOT_TOKEN = "<start_of_text>"
EOT_TOKEN = "<end_of_text>"

_CLIP_EXTRA_HINT = (
    "LibreCLIP's text tokenizer needs 'ftfy' and 'regex'. Install them with:\n"
    '    pip install "libreyolo[clip]"'
)


def _require_regex():
    try:
        import regex
    except ImportError as exc:  # pragma: no cover - exercised via install hint
        raise ImportError(_CLIP_EXTRA_HINT) from exc
    return regex


def _require_ftfy():
    try:
        import ftfy
    except ImportError as exc:  # pragma: no cover - exercised via install hint
        raise ImportError(_CLIP_EXTRA_HINT) from exc
    return ftfy


@lru_cache()
def default_bpe() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz"
    )


@lru_cache()
def bytes_to_unicode():
    """Reversible map from utf-8 bytes to unicode strings the BPE works on."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return the set of adjacent symbol pairs in a tuple-of-symbols word."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text: str) -> str:
    ftfy = _require_ftfy()
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    regex = _require_regex()
    text = regex.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer:
    """Byte-pair-encoding tokenizer producing CLIP's exact 77-length token IDs.

    Cleaning is ``lower`` (CLIP's default): ``basic_clean`` (ftfy + html
    unescape) then whitespace collapse then lowercase. SOT/EOT are 49406/49407
    and padding is 0, identical to every CLIP/open_clip checkpoint.
    """

    def __init__(
        self,
        bpe_path: str | None = None,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> None:
        regex = _require_regex()
        bpe_path = bpe_path or default_bpe()
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with gzip.open(bpe_path) as fh:
            merges = fh.read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        special_tokens = [SOT_TOKEN, EOT_TOKEN]
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = regex.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE,
        )
        self.vocab_size = len(self.encoder)
        self.sot_token_id = self.encoder[SOT_TOKEN]
        self.eot_token_id = self.encoder[EOT_TOKEN]
        self.context_length = context_length

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        regex = _require_regex()
        bpe_tokens: List[int] = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in regex.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        text = "".join(self.decoder[token] for token in tokens)
        return (
            bytearray(self.byte_decoder[c] for c in text)
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )

    def __call__(
        self,
        texts: Union[str, List[str]],
        context_length: int | None = None,
    ) -> torch.LongTensor:
        """Tokenize string(s) into a ``[N, context_length]`` long tensor."""
        if isinstance(texts, str):
            texts = [texts]
        context_length = context_length or self.context_length

        all_tokens = [
            [self.sot_token_id] + self.encode(text) + [self.eot_token_id]
            for text in texts
        ]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = self.eot_token_id
            result[i, : len(tokens)] = torch.tensor(tokens)
        return result
