"""Unit and pinned-reference tests for Florence-2 Core ML beam scoring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.backends.coreml_florence_beam import (
    FLORENCE2_DECODER_START_TOKEN_ID,
    Florence2BeamSearch,
    _banned_trigram_tokens,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vlm,
    pytest.mark.experimental_backend,
]


def _flat_logits(vocab_size: int, value: float = 0.0) -> np.ndarray:
    return np.full((3, vocab_size), value, dtype=np.float32)


def test_forced_tokens_follow_pinned_processor_order():
    search = Florence2BeamSearch(max_new_tokens=2, vocab_size=8)

    first = search.advance(_flat_logits(8))

    assert not first.done
    assert first.next_token_ids == (0, 0, 0)
    assert search.running_sequences == ((2, 0), (2, 0), (2, 0))

    final = search.advance(_flat_logits(8))

    assert final.done
    assert final.next_token_ids is None
    assert final.parent_indices is None
    assert search.output_sequence == (2, 0, 2)


def test_one_token_budget_makes_later_forced_eos_override_bos():
    search = Florence2BeamSearch(max_new_tokens=1, vocab_size=8)

    step = search.advance(_flat_logits(8))

    assert step.done
    assert search.output_sequence == (2, 2)


def test_parent_indices_duplicate_selected_cache_beam():
    search = Florence2BeamSearch(max_new_tokens=5, vocab_size=8)
    search.advance(_flat_logits(8))
    logits = np.full((3, 8), -20.0, dtype=np.float32)
    logits[0, 3:6] = np.asarray([3.0, 2.0, 1.0], dtype=np.float32)

    step = search.advance(logits)

    assert step.next_token_ids == (3, 4, 5)
    assert step.parent_indices == (0, 0, 0)
    assert search.running_sequences == (
        (2, 0, 3),
        (2, 0, 4),
        (2, 0, 5),
    )


def test_no_repeat_trigram_uses_complete_decoder_history():
    sequence = (2, 0, 4, 5, 4, 5)

    assert _banned_trigram_tokens(sequence) == (4,)
    assert _banned_trigram_tokens((2, 0)) == ()


def test_no_repeat_trigram_changes_live_beam_selection():
    search = Florence2BeamSearch(max_new_tokens=8, vocab_size=8)
    search.advance(_flat_logits(8))
    for token_id in (4, 5, 4, 5):
        logits = np.full((3, 8), -100.0, dtype=np.float32)
        logits[:, token_id] = 100.0
        search.advance(logits)
    assert search.running_sequences[0] == (2, 0, 4, 5, 4, 5)

    logits = np.full((3, 8), -100.0, dtype=np.float32)
    logits[:, 4] = 100.0
    logits[:, 6] = 90.0
    step = search.advance(logits)

    assert not step.done
    assert search.running_sequences[0][-1] == 6


def test_contract_rejects_non_integer_profile_value():
    with pytest.raises(TypeError, match="must be an integer"):
        Florence2BeamSearch(max_new_tokens=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_new_tokens": 0}, "inclusive range"),
        ({"max_new_tokens": 1_025}, "inclusive range"),
        ({"max_new_tokens": 2, "vocab_size": 2}, "special tokens"),
    ],
)
def test_contract_rejects_invalid_profile_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Florence2BeamSearch(**kwargs)


@pytest.mark.parametrize(
    ("logits", "match"),
    [
        (np.zeros((1, 8), dtype=np.float32), "shape"),
        (np.zeros((3, 8), dtype=np.int32), "floating point"),
        (
            np.full((3, 8), np.nan, dtype=np.float32),
            "NaN or infinity",
        ),
    ],
)
def test_logits_boundary_is_strict(logits, match):
    search = Florence2BeamSearch(max_new_tokens=2, vocab_size=8)

    with pytest.raises(ValueError, match=match):
        search.advance(logits)


def test_completed_search_rejects_state_reuse():
    search = Florence2BeamSearch(max_new_tokens=1, vocab_size=8)
    search.advance(_flat_logits(8))

    with pytest.raises(RuntimeError, match="already complete"):
        search.advance(_flat_logits(8))
    with pytest.raises(RuntimeError, match="already complete"):
        search.decoder_input_ids()


def test_matches_transformers_5_12_1_on_tiny_deterministic_bart(monkeypatch):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.12.1":
        pytest.skip("reference parity is pinned to Transformers 5.12.1")
    from transformers import BartConfig, BartForConditionalGeneration
    from transformers.cache_utils import EncoderDecoderCache
    from transformers.modeling_outputs import BaseModelOutput

    torch.manual_seed(719)
    vocab_size = 17
    max_new_tokens = 8
    config = BartConfig(
        vocab_size=vocab_size,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=32,
        decoder_start_token_id=2,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        forced_bos_token_id=0,
        forced_eos_token_id=2,
    )
    model = BartForConditionalGeneration(config).eval()
    encoder_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    reference_parents = []
    original_reorder = EncoderDecoderCache.reorder_cache

    def capture_reorder(cache, beam_indices):
        reference_parents.append(tuple(int(index) for index in beam_indices.tolist()))
        return original_reorder(cache, beam_indices)

    monkeypatch.setattr(EncoderDecoderCache, "reorder_cache", capture_reorder)

    with torch.no_grad():
        reference = model.generate(
            input_ids=encoder_ids,
            max_new_tokens=max_new_tokens,
            num_beams=3,
            do_sample=False,
            early_stopping=True,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            forced_bos_token_id=0,
            forced_eos_token_id=2,
            return_dict_in_generate=True,
            output_scores=True,
        )
        encoder_hidden = model.model.encoder(
            input_ids=encoder_ids,
            return_dict=True,
        ).last_hidden_state
        expanded_encoder = BaseModelOutput(
            last_hidden_state=encoder_hidden.repeat(3, 1, 1)
        )
        search = Florence2BeamSearch(
            max_new_tokens=max_new_tokens,
            vocab_size=vocab_size,
        )
        scorer_parents = []
        while not search.done:
            decoder_ids = torch.from_numpy(search.decoder_input_ids()).long()
            output = model(
                encoder_outputs=expanded_encoder,
                decoder_input_ids=decoder_ids,
                use_cache=False,
                return_dict=True,
            )
            step = search.advance(output.logits[:, -1, :])
            if not step.done:
                scorer_parents.append(step.parent_indices)

    assert search.output_sequence == tuple(reference.sequences[0].tolist())
    assert search.output_score == pytest.approx(
        float(reference.sequences_scores[0]),
        abs=1e-6,
    )
    assert scorer_parents == reference_parents[:-1]
    assert search.output_sequence[0] == FLORENCE2_DECODER_START_TOKEN_ID
