"""Exact host-side three-beam search for the Florence-2 base Core ML profile.

The scoring contract is derived from Hugging Face Transformers 5.12.1,
``src/transformers/generation/{utils,logits_process,stopping_criteria}.py`` at
commit ``ddb849abe009d1089e6c691bfc897f27211c663c`` (Apache-2.0). This module
specializes that permissively licensed batch-generic implementation to one
Florence-2 request with exactly three deterministic beams.

Core ML owns decoder execution and KV state. This scorer owns the host-only
parts of generation: log-softmax, forced tokens, the no-repeat trigram rule,
finished-hypothesis ranking, early stopping, and the parent indices required
to reorder the three decoder states before their next token is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

FLORENCE2_BASE_VOCAB_SIZE = 51_328
FLORENCE2_NUM_BEAMS = 3
FLORENCE2_DECODER_START_TOKEN_ID = 2
FLORENCE2_BOS_TOKEN_ID = 0
FLORENCE2_EOS_TOKEN_ID = 2
FLORENCE2_PAD_TOKEN_ID = 1
FLORENCE2_NO_REPEAT_NGRAM_SIZE = 3
FLORENCE2_LENGTH_PENALTY = 1.0
FLORENCE2_MAX_DECODER_POSITIONS = 1_024

_NEGATIVE_SENTINEL = -1.0e9


@dataclass(frozen=True)
class Florence2BeamStep:
    """The live continuation selected after one three-beam scoring step."""

    next_token_ids: tuple[int, int, int] | None
    parent_indices: tuple[int, int, int] | None
    done: bool


def _require_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


def _banned_trigram_tokens(sequence: tuple[int, ...]) -> tuple[int, ...]:
    """Return followers that would repeat a trigram in ``sequence``."""

    if len(sequence) + 1 < FLORENCE2_NO_REPEAT_NGRAM_SIZE:
        return ()
    followers: dict[tuple[int, int], list[int]] = {}
    for index in range(len(sequence) - 2):
        prefix = (sequence[index], sequence[index + 1])
        followers.setdefault(prefix, []).append(sequence[index + 2])
    return tuple(followers.get((sequence[-2], sequence[-1]), ()))


class Florence2BeamSearch:
    """Transformers-compatible Florence-2 beam state for one batch item."""

    def __init__(
        self,
        *,
        max_new_tokens: int,
        vocab_size: int = FLORENCE2_BASE_VOCAB_SIZE,
    ) -> None:
        budget = _require_int("max_new_tokens", max_new_tokens)
        vocabulary = _require_int("vocab_size", vocab_size)
        if budget <= 0 or budget > FLORENCE2_MAX_DECODER_POSITIONS:
            raise ValueError(
                "Florence-2 max_new_tokens must be in the inclusive range "
                f"[1, {FLORENCE2_MAX_DECODER_POSITIONS}]."
            )
        if vocabulary <= max(
            FLORENCE2_BOS_TOKEN_ID,
            FLORENCE2_EOS_TOKEN_ID,
            FLORENCE2_PAD_TOKEN_ID,
        ):
            raise ValueError(
                "Florence-2 vocab_size does not contain its special tokens."
            )

        self.max_new_tokens = budget
        self.max_length = budget + 1
        self.vocab_size = vocabulary
        self._cur_len = 1
        self._done = False
        self._running_sequences: list[tuple[int, ...]] = [
            (FLORENCE2_DECODER_START_TOKEN_ID,) for _ in range(FLORENCE2_NUM_BEAMS)
        ]
        self._running_scores = torch.zeros(
            FLORENCE2_NUM_BEAMS,
            dtype=torch.float32,
        )
        self._running_scores[1:] = _NEGATIVE_SENTINEL
        self._finished_sequences = list(self._running_sequences)
        self._finished_scores = torch.full(
            (FLORENCE2_NUM_BEAMS,),
            _NEGATIVE_SENTINEL,
            dtype=torch.float32,
        )
        self._finished = torch.zeros(FLORENCE2_NUM_BEAMS, dtype=torch.bool)
        self._early_stop_heuristic_unsatisfied = True

    @property
    def done(self) -> bool:
        return self._done

    @property
    def generated_tokens(self) -> int:
        return self._cur_len - 1

    @property
    def running_sequences(self) -> tuple[tuple[int, ...], ...]:
        """Return the three decoder histories whose logits are expected next."""

        return tuple(self._running_sequences)

    def decoder_input_ids(self) -> np.ndarray:
        """Return the current three decoder histories as contiguous INT32 IDs."""

        if self._done:
            raise RuntimeError("Florence-2 beam search is already complete.")
        return np.ascontiguousarray(np.asarray(self._running_sequences, dtype=np.int32))

    @property
    def output_sequence(self) -> tuple[int, ...]:
        """Return the best finalized sequence, including decoder-start token."""

        if not self._done:
            raise RuntimeError("Florence-2 beam search has not completed.")
        if not bool(self._finished[0]):
            raise RuntimeError("Florence-2 beam search ended without a hypothesis.")
        return self._finished_sequences[0]

    @property
    def output_score(self) -> float:
        """Return the normalized log-probability of ``output_sequence``."""

        if not self._done:
            raise RuntimeError("Florence-2 beam search has not completed.")
        if not bool(self._finished[0]):
            raise RuntimeError("Florence-2 beam search ended without a hypothesis.")
        return float(self._finished_scores[0].item())

    def _validated_logits(self, logits: Any) -> torch.Tensor:
        try:
            values = torch.as_tensor(logits, device="cpu")
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                "Florence-2 beam logits must be a numeric tensor."
            ) from exc
        expected = (FLORENCE2_NUM_BEAMS, self.vocab_size)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Florence-2 beam logits must have shape {expected}, "
                f"got {tuple(values.shape)}."
            )
        if not values.is_floating_point():
            raise ValueError("Florence-2 beam logits must be floating point.")
        values = values.to(dtype=torch.float32, device="cpu", copy=True)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Florence-2 beam logits contain NaN or infinity.")
        return values

    def _processed_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        scores = torch.nn.functional.log_softmax(logits, dim=-1)
        for beam_index, sequence in enumerate(self._running_sequences):
            banned = _banned_trigram_tokens(sequence)
            if banned:
                scores[beam_index, list(banned)] = -torch.inf

        # This is the processor order in Transformers 5.12.1. In particular,
        # an intentionally tiny one-token budget makes forced EOS override BOS.
        if self._cur_len == 1:
            scores = torch.full_like(scores, -torch.inf)
            scores[:, FLORENCE2_BOS_TOKEN_ID] = 0.0
        if self._cur_len == self.max_length - 1:
            scores = torch.full_like(scores, -torch.inf)
            scores[:, FLORENCE2_EOS_TOKEN_ID] = 0.0
        return scores

    def _update_finished(
        self,
        *,
        candidate_sequences: list[tuple[int, ...]],
        candidate_scores: torch.Tensor,
        candidate_finished: torch.Tensor,
    ) -> None:
        top_beam_mask = torch.tensor(
            [True, True, True, False, False, False],
            dtype=torch.bool,
        )
        newly_finished = candidate_finished & top_beam_mask
        generated_length = self._cur_len + 1 - 1
        normalized_scores = candidate_scores / (
            generated_length**FLORENCE2_LENGTH_PENALTY
        )
        if bool(torch.all(self._finished)):
            normalized_scores += _NEGATIVE_SENTINEL
        if not self._early_stop_heuristic_unsatisfied:
            normalized_scores += _NEGATIVE_SENTINEL
        normalized_scores += (~newly_finished).to(torch.float32) * (_NEGATIVE_SENTINEL)

        merged_scores = torch.cat(
            (self._finished_scores, normalized_scores),
            dim=0,
        )
        merged_sequences = self._finished_sequences + candidate_sequences
        merged_finished = torch.cat((self._finished, newly_finished), dim=0)
        selected = torch.topk(
            merged_scores,
            k=FLORENCE2_NUM_BEAMS,
        ).indices
        selected_indices = [int(index) for index in selected.tolist()]
        self._finished_scores = merged_scores[selected]
        self._finished_sequences = [
            merged_sequences[index] for index in selected_indices
        ]
        self._finished = merged_finished[selected]

    def _update_early_stop_heuristic(self) -> None:
        generated_length = self._cur_len - 1
        best_running_score = self._running_scores[:1] / (
            generated_length**FLORENCE2_LENGTH_PENALTY
        )
        minimum_finished_score = torch.min(self._finished_scores)
        worst_finished_scores = torch.where(
            self._finished,
            minimum_finished_score,
            torch.tensor(_NEGATIVE_SENTINEL, dtype=torch.float32),
        )
        improvement_possible = bool(
            torch.any(best_running_score > worst_finished_scores)
        )
        self._early_stop_heuristic_unsatisfied = (
            self._early_stop_heuristic_unsatisfied and improvement_possible
        )

    def advance(self, logits: Any) -> Florence2BeamStep:
        """Score one decoder result and return tokens/parents for the next call."""

        if self._done:
            raise RuntimeError("Florence-2 beam search is already complete.")
        values = self._validated_logits(logits)
        log_probs = self._processed_log_probs(values)
        accumulated = log_probs + self._running_scores[:, None]
        flat_scores = accumulated.reshape(-1)

        # One EOS token means six candidates are needed to retain three live
        # continuations even when the highest-ranked candidates just finished.
        candidate_scores, flat_indices = torch.topk(flat_scores, k=6)
        parent_indices = flat_indices // self.vocab_size
        token_ids = flat_indices % self.vocab_size
        parent_list = [int(index) for index in parent_indices.tolist()]
        token_list = [int(token) for token in token_ids.tolist()]
        candidate_sequences = [
            self._running_sequences[parent] + (token,)
            for parent, token in zip(parent_list, token_list)
        ]
        candidate_finished = (token_ids == FLORENCE2_EOS_TOKEN_ID) | (
            self._cur_len + 1 >= self.max_length
        )

        live_scores = candidate_scores + candidate_finished.to(torch.float32) * (
            _NEGATIVE_SENTINEL
        )
        live_selection = torch.topk(
            live_scores,
            k=FLORENCE2_NUM_BEAMS,
        ).indices
        live_indices = [int(index) for index in live_selection.tolist()]
        live_parent_indices = parent_indices[live_selection]
        self._running_scores = live_scores[live_selection]
        self._running_sequences = [candidate_sequences[index] for index in live_indices]

        self._update_finished(
            candidate_sequences=candidate_sequences,
            candidate_scores=candidate_scores,
            candidate_finished=candidate_finished,
        )
        self._cur_len += 1
        self._update_early_stop_heuristic()

        improvement_possible = self._early_stop_heuristic_unsatisfied
        exists_open_beam = not bool(torch.all(self._finished))
        valid_continuations = not bool(torch.all(candidate_finished))
        self._done = not (
            improvement_possible and exists_open_beam and valid_continuations
        )
        if self._done:
            return Florence2BeamStep(
                next_token_ids=None,
                parent_indices=None,
                done=True,
            )

        return Florence2BeamStep(
            next_token_ids=tuple(
                int(sequence[-1]) for sequence in self._running_sequences
            ),
            parent_indices=tuple(int(index) for index in live_parent_indices.tolist()),
            done=False,
        )


__all__ = [
    "FLORENCE2_BASE_VOCAB_SIZE",
    "FLORENCE2_BOS_TOKEN_ID",
    "FLORENCE2_DECODER_START_TOKEN_ID",
    "FLORENCE2_EOS_TOKEN_ID",
    "FLORENCE2_LENGTH_PENALTY",
    "FLORENCE2_MAX_DECODER_POSITIONS",
    "FLORENCE2_NO_REPEAT_NGRAM_SIZE",
    "FLORENCE2_NUM_BEAMS",
    "FLORENCE2_PAD_TOKEN_ID",
    "Florence2BeamSearch",
    "Florence2BeamStep",
]
