"""Offline tests for VLM token-logprob confidence scoring."""

import math
import re

import pytest
import torch

from libreyolo.models.vlm.base import (
    LibreVLMModel,
    _GreedyTokenLogprobRecorder,
    _ScoredGeneration,
)
from libreyolo.models.vlm.confidence import (
    TokenSpan,
    decode_token_spans,
    score_detection_items,
)
from libreyolo.models.vlm.parsing import build_detection_dict, extract_detections
from libreyolo.models.vlm.qwen3vl import LibreQwen3VL

pytestmark = pytest.mark.unit


def _char_spans(text: str, values: dict[int, float]) -> list[TokenSpan]:
    return [
        TokenSpan(index, index + 1, values.get(index, -100.0))
        for index in range(len(text))
    ]


class TestDecodeTokenSpans:
    def test_composable_token_pieces_take_linear_path(self):
        pieces = {1: "red", 2: " car", 3: ""}

        def decode(ids):
            return "".join(pieces[token_id] for token_id in ids)

        text, spans = decode_token_spans(
            [[1, 2, 3]], [[math.log(0.8), math.log(0.6), math.log(0.9)]], decode
        )
        assert text == "red car"
        assert [(span.start, span.end) for span in spans] == [(0, 3), (3, 7), (7, 7)]

    def test_context_dependent_piece_uses_monotonic_prefix_fallback(self):
        prefixes = {(1,): "red", (2,): "car", (1, 2): "red car"}
        text, spans = decode_token_spans(
            [1, 2], [-0.1, -0.2], lambda ids: prefixes[tuple(ids)]
        )
        assert text == "red car"
        assert [(span.start, span.end) for span in spans] == [(0, 3), (3, 7)]

    def test_non_monotonic_decode_fails_closed(self):
        values = {(1,): "a", (2,): "b", (1, 2): "B"}
        text, spans = decode_token_spans(
            [1, 2], [-0.1, -0.2], lambda ids: values[tuple(ids)]
        )
        assert text == "B"
        assert spans == []

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same generated length"):
            decode_token_spans([1, 2], [-0.1], lambda ids: "x" * len(ids))


class TestScoreDetectionItems:
    def test_equal_weights_label_and_coordinate_components(self):
        text = '[{"bbox_2d":[10,20,30,40],"label":"red car"}]'
        items = extract_detections(text)
        values = {}
        for match in re.finditer(r"10|20|30|40", text):
            values.update(
                {index: math.log(0.25) for index in range(match.start(), match.end())}
            )
        label_start = text.index("red car")
        values.update(
            {
                index: math.log(0.81)
                for index in range(label_start, label_start + len("red car"))
            }
        )
        scores = score_detection_items(
            text, items, _char_spans(text, values), bbox_key="bbox_2d"
        )
        assert scores == pytest.approx([0.45])

    def test_punctuation_and_key_logprobs_are_excluded(self):
        text = '[{"bbox_2d":[1,2,3,4],"label":"cat"}]'
        items = extract_detections(text)
        values = {
            index: math.log(0.64)
            for index, char in enumerate(text)
            if char.isdigit()
        }
        start = text.index("cat")
        values.update({index: math.log(0.64) for index in range(start, start + 3)})
        scores = score_detection_items(
            text, items, _char_spans(text, values), bbox_key="bbox_2d"
        )
        assert scores == pytest.approx([0.64])

    def test_repeated_objects_use_successive_token_spans(self):
        obj = '{"bbox_2d":[1,2,3,4],"label":"cat"}'
        text = f"[{obj},{obj}]"
        items = extract_detections(text)
        spans = _char_spans(text, {})
        first_start = text.index(obj)
        second_start = text.index(obj, first_start + 1)
        mutable = list(spans)
        for object_start, probability in ((first_start, 0.8), (second_start, 0.2)):
            for index in range(object_start, object_start + len(obj)):
                mutable[index] = TokenSpan(index, index + 1, math.log(probability))
        scores = score_detection_items(text, items, mutable, bbox_key="bbox_2d")
        assert scores == pytest.approx([0.8, 0.2])

    def test_missing_source_span_returns_none(self):
        scores = score_detection_items(
            "[]",
            [{"bbox_2d": [1, 2, 3, 4], "label": "cat"}],
            [],
            bbox_key="bbox_2d",
        )
        assert scores == [None]

    @pytest.mark.parametrize(
        "text",
        [
            '[{"bbox_2d":[1,2,3,4],"bbox_2d":[5,6,7,8],"label":"cat"}]',
            (
                '[{"bbox_2d":[1,2,3,4],"bbox":[9,9,10,10],'
                '"bbox_2d":[5,6,7,8],"label":"cat"}]'
            ),
            '[{"bbox_2d":[1,2,3,4],"label":"dog","label":"cat"}]',
        ],
    )
    def test_duplicate_scoring_keys_fail_closed(self, text):
        items = extract_detections(text)
        spans = _char_spans(text, {index: math.log(0.8) for index in range(len(text))})
        assert score_detection_items(text, items, spans, bbox_key="bbox_2d") == [
            None
        ]

    def test_key_like_text_inside_string_is_not_a_member(self):
        text = (
            '[{"note":"fake \'bbox_2d\': [1,2,3,4]",'
            '"bbox_2d":[5,6,7,8],"label":"cat"}]'
        )
        items = extract_detections(text)
        values = {}
        actual_box = text.index("[5,6,7,8]")
        for index in range(actual_box + 1, actual_box + len("5,6,7,8") + 1):
            if text[index].isdigit():
                values[index] = math.log(0.25)
        label = text.index("cat")
        values.update({index: math.log(0.25) for index in range(label, label + 3)})
        scores = score_detection_items(
            text, items, _char_spans(text, values), bbox_key="bbox_2d"
        )
        assert scores == pytest.approx([0.25])

    def test_missing_label_or_coordinate_component_fails_closed(self):
        text = '[{"bbox_2d":[1,2,3,4]}]'
        items = extract_detections(text)
        spans = _char_spans(text, {index: math.log(0.8) for index in range(len(text))})
        assert score_detection_items(text, items, spans, bbox_key="bbox_2d") == [
            None
        ]

    def test_non_finite_component_token_fails_closed(self):
        text = '[{"bbox_2d":[1,2,3,4],"label":"cat"}]'
        items = extract_detections(text)
        spans = _char_spans(text, {index: math.log(0.8) for index in range(len(text))})
        label_start = text.index("cat")
        spans[label_start] = TokenSpan(label_start, label_start + 1, float("nan"))
        assert score_detection_items(text, items, spans, bbox_key="bbox_2d") == [
            None
        ]


class TestGreedyRecorder:
    def test_records_normalized_argmax_only(self):
        recorder = _GreedyTokenLogprobRecorder()
        probabilities = torch.tensor([[0.1, 0.6, 0.3]])
        logits = probabilities.log()
        returned = recorder(torch.tensor([[1, 2]]), logits)
        assert returned is logits
        assert recorder.values().shape == (1, 1)
        assert recorder.values()[0, 0].item() == pytest.approx(math.log(0.6))

    def test_empty_recorder_has_bounded_empty_shape(self):
        assert _GreedyTokenLogprobRecorder().values().shape == (1, 0)


class _StubGenerateModel:
    def __init__(self, step_probabilities):
        self.step_probabilities = step_probabilities
        self.kwargs = None

    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        sequence = input_ids
        for probabilities in self.step_probabilities:
            scores = torch.tensor([probabilities], dtype=torch.float32).log()
            for processor in kwargs.get("logits_processor", []):
                scores = processor(sequence, scores)
            selected = scores.argmax(dim=-1, keepdim=True)
            sequence = torch.cat((sequence, selected), dim=1)
        return sequence


class _StubDecodeProcessor:
    pieces = {
        0: '[{"bbox_2d":[',
        1: "10,20,30,40",
        2: '],"label":"cat"}]',
    }

    def __init__(self):
        self.calls = []

    def batch_decode(self, rows, **kwargs):
        self.calls.append(kwargs)
        if isinstance(rows, torch.Tensor):
            rows = rows.tolist()
        return ["".join(self.pieces[token] for token in row) for row in rows]


class TestBaseConfidenceIntegration:
    def _model(self):
        model = object.__new__(LibreVLMModel)
        model.FAMILY = "stub"
        model.TOKEN_LOGPROB_CONFIDENCE = True
        model.MAX_NEW_TOKENS = 3
        model.REPETITION_PENALTY = 1.0
        model.BBOX_KEY = "bbox_2d"
        model.COORD_DIVISOR = 1000.0
        model.BOX_FORMAT = "xyxy"
        model.DEFAULT_SCORE = 1.0
        model._model_dtype = None
        model._name_to_id = {"cat": 0}
        model.processor = _StubDecodeProcessor()
        model.model = _StubGenerateModel(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]
        )
        return model

    def test_forward_returns_only_scalar_per_generated_step(self):
        model = self._model()
        output = model._forward({"input_ids": torch.tensor([[8, 9]])})
        assert isinstance(output, _ScoredGeneration)
        assert output.token_ids.shape == (1, 3)
        assert output.token_logprobs.shape == (1, 3)
        assert model.model.kwargs["num_beams"] == 1
        assert "output_scores" not in model.model.kwargs

    def test_confidence_method_exposes_score_provenance(self):
        assert LibreVLMModel.CONFIDENCE_METHOD == "constant"
        assert LibreQwen3VL.CONFIDENCE_METHOD == "constant"
        assert LibreQwen3VL.TOKEN_LOGPROB_CONFIDENCE is False

    def test_postprocess_emits_per_box_token_score(self):
        model = self._model()
        output = model._forward({"input_ids": torch.tensor([[8, 9]])})
        result = model._postprocess(
            output,
            conf_thres=0.0,
            iou_thres=0.7,
            original_size=(1000, 1000),
        )
        assert result["num_detections"] == 1
        assert 0.0 < result["scores"][0] < 1.0

    def test_unscored_output_keeps_constant_fallback(self):
        model = self._model()
        tokens = torch.tensor([[0, 1, 2]])
        result = model._postprocess(
            tokens,
            conf_thres=0.0,
            iou_thres=0.7,
            original_size=(1000, 1000),
        )
        assert result["scores"] == [1.0]
        assert model.processor.calls == [{"skip_special_tokens": True}]

    def test_one_unaligned_object_falls_whole_response_back_to_constant(self):
        model = self._model()
        model._name_to_id = {"cat": 0, "dog {x}": 1}
        text = (
            '[{"bbox_2d":[10,20,30,40],"label":"cat"},'
            '{"bbox_2d":[50,60,70,80],"label":"dog {x}"}]'
        )
        items = extract_detections(text)
        output = _ScoredGeneration(torch.tensor([[0]]), torch.tensor([[-0.1]]))
        token_spans = _char_spans(
            text, {index: math.log(0.8) for index in range(len(text))}
        )
        item_scores = model._scores_for_detections(
            output, text, items, token_spans
        )
        assert item_scores is None
        result = build_detection_dict(
            items,
            model._name_to_id,
            (1000, 1000),
            default_score=model.DEFAULT_SCORE,
            item_scores=item_scores,
            bbox_key="bbox_2d",
            coord_divisor=1000.0,
        )
        assert result["classes"] == [0, 1]
        assert result["scores"] == [1.0, 1.0]
