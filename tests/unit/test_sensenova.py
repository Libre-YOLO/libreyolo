"""Offline tests for the SenseNova-Vision multi-task VLM family.

Everything here runs without weights: the tagged-text and generated-image
parsers are pure, and the adapter's postprocess hooks are exercised on bare
instances following the LocateAnything test pattern.
"""

import numpy as np
import pytest
from PIL import Image

from libreyolo.models.sensenova import parsing
from libreyolo.models.sensenova.model import (
    _TASK_TO_MODE,
    BASE_PARAMS,
    LibreSenseNovaVision,
)
from libreyolo.models.sensenova.modeling.layers import (
    TimestepEmbedder,
    sincos_position_embedding_2d,
)
from libreyolo.models.sensenova.prompts import COCO_PANOPTIC_NAMES

pytestmark = pytest.mark.unit


def _bare(task="detect", names=None, user_vocab=False):
    m = object.__new__(LibreSenseNovaVision)
    names = names or ["person", "car"]
    m.names = {i: name for i, name in enumerate(names)}
    m.nb_classes = len(m.names)
    m._name_to_id = {name.lower(): i for i, name in m.names.items()}
    m.task = task
    m._custom_prompt = None
    m._user_vocab = user_vocab
    m.noise_seed = 42
    return m


class TestTaggedTextParsing:
    def test_parses_labeled_boxes_normalized(self):
        text = (
            "<p>person</p><bbox>[0.1, 0.2, 0.5, 0.9]</bbox>"
            "<bbox>[0.6,0.1,0.8,0.4]</bbox><p>car</p><bbox>[0.0,0.0,0.3,0.3]</bbox>"
        )
        parsed = parsing.parse_tagged_boxes(text)
        assert parsed["person"] == [[0.1, 0.2, 0.5, 0.9], [0.6, 0.1, 0.8, 0.4]]
        assert parsed["car"] == [[0.0, 0.0, 0.3, 0.3]]

    def test_clips_out_of_range_coordinates(self):
        parsed = parsing.parse_tagged_boxes("<p>dog</p><bbox>[-0.2, 0.5, 1.4, 0.9]</bbox>")
        assert parsed["dog"] == [[0.0, 0.5, 0.999, 0.9]]

    def test_ignores_malformed_fragments(self):
        assert parsing.parse_tagged_boxes("<p>cat</p><bbox>[a, b, c, d]</bbox>") == {}
        assert parsing.parse_tagged_boxes("no tags at all") == {}

    def test_parses_points(self):
        parsed = parsing.parse_tagged_points(
            "<p>bird</p><point>[0.25, 0.75]</point><point>[0.5,0.5]</point>"
        )
        assert parsed["bird"] == [[0.25, 0.75], [0.5, 0.5]]

    def test_parses_keypoints_with_instances_and_unvisible(self):
        text = (
            "<p>person</p><bbox>[0.1,0.1,0.9,0.9]</bbox>"
            "nose<kpt>[0.5,0.2]</kpt>left eye<kpt>unvisible</kpt>"
        )
        parsed = parsing.parse_tagged_keypoints(text)
        assert len(parsed["person"]) == 1
        instance = parsed["person"][0]
        assert instance["bbox"] == [0.1, 0.1, 0.9, 0.9]
        assert instance["keypoints"]["nose"] == [0.5, 0.2]
        assert instance["keypoints"]["left eye"] == [-1.0, -1.0]

    def test_keypoint_array_orders_by_skeleton(self):
        instance = {"keypoints": {"nose": [0.5, 0.2], "left ankle": [-1.0, -1.0]}}
        arr = parsing.keypoint_instance_to_array(
            instance, parsing.HUMAN_KEYPOINT_NAMES, 100, 200
        )
        assert arr.shape == (17, 3)
        assert arr[0].tolist() == [50.0, 40.0, 1.0]  # nose is index 0
        assert arr[15].tolist() == [0.0, 0.0, 0.0]  # invisible left ankle
        assert arr[16].tolist() == [0.0, 0.0, 0.0]  # missing right ankle

    def test_normalize_category_strips_dataset_suffixes(self):
        assert parsing.normalize_category("tree-merged") == "tree"
        assert parsing.normalize_category("wall-other") == "wall"
        assert parsing.normalize_category("Traffic-Light") == "traffic light"


class TestGeneratedImageParsing:
    def test_binary_mask_resizes_and_thresholds(self):
        img = Image.new("L", (10, 10), 0)
        for x in range(5):
            for y in range(10):
                img.putpixel((x, y), 255)
        mask = parsing.binary_mask_from_image(img, (20, 20))
        assert mask.shape == (20, 20)
        assert mask[:, :10].all() and not mask[:, 10:].any()

    def test_mask_to_xyxy(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 3:8] = True
        assert parsing.mask_to_xyxy(mask) == [3.0, 2.0, 8.0, 5.0]
        assert parsing.mask_to_xyxy(np.zeros((4, 4), dtype=bool)) is None

    def test_depth_map_scales_to_unit_range(self):
        img = Image.new("L", (8, 8), 255)
        depth = parsing.depth_from_image(img, (8, 8))
        assert depth.shape == (8, 8)
        assert float(depth.max()) == 1.0

    def test_panoptic_assembly_from_colors_and_caption(self):
        img = Image.new("RGB", (20, 20), (0, 0, 0))
        for x in range(10):
            for y in range(20):
                img.putpixel((x, y), (255, 0, 0))
        for x in range(10, 20):
            for y in range(10):
                img.putpixel((x, y), (0, 0, 255))
        caption = (
            "A person stands under the sky. "
            "<p>person-0<color>(255,0,0)</color></p> "
            "<p>sky<color>(0,0,255)</color></p>"
        )
        id_map, segments, names = parsing.panoptic_from_mask_and_caption(
            img, caption, {0: "person"}, (20, 20)
        )
        assert id_map.shape == (20, 20)
        # person: category 0 -> segment id 1000; sky registered as category 1.
        assert segments[0] == {"id": 1000, "category_id": 0, "score": 1.0}
        assert segments[1]["category_id"] == 1
        assert names[1] == "sky"
        assert (id_map[:, :10] == 1000).all()
        # Bottom-right quadrant stayed black -> void id 0.
        assert (id_map[10:, 10:] == 0).all()

    def test_panoptic_repeated_phrase_gets_new_instance_ids(self):
        img = Image.new("RGB", (4, 4), (255, 0, 0))
        for x in range(2):
            for y in range(4):
                img.putpixel((x, y), (0, 255, 0))
        caption = (
            "<p>car-0<color>(0,255,0)</color></p><p>car-0<color>(255,0,0)</color></p>"
        )
        _, segments, _ = parsing.panoptic_from_mask_and_caption(
            img, caption, {0: "car"}, (4, 4)
        )
        assert [s["id"] for s in segments] == [1000, 1001]


class TestCleanSourceLayers:
    def test_sincos_table_matches_reference_convention(self):
        table = sincos_position_embedding_2d(8, 3)
        assert table.shape == (9, 8)
        # Position 0 encodes sin(0)=0 in each half's first slot, cos(0)=1 after.
        assert table[0, 0] == pytest.approx(0.0)
        assert table[0, 2] == pytest.approx(1.0)
        # Position p = row*grid+col: first half varies with col, second with row.
        col_half = table[:, :4]
        row_half = table[:, 4:]
        assert np.allclose(col_half[0], col_half[3])  # positions 0 and 3 share col 0
        assert np.allclose(row_half[0], row_half[1])  # positions 0 and 1 share row 0
        assert not np.allclose(col_half[0], col_half[1])  # different cols differ

    def test_timestep_embedding_is_cos_first(self):
        import torch

        emb = TimestepEmbedder.timestep_embedding(torch.zeros(2), 8)
        assert emb.shape == (2, 8)
        assert torch.allclose(emb[:, :4], torch.ones(2, 4))  # cos(0) = 1
        assert torch.allclose(emb[:, 4:], torch.zeros(2, 4))  # sin(0) = 0

    def test_position_embedding_module_matches_table(self):
        import torch

        from libreyolo.models.sensenova.modeling.layers import PositionEmbedding

        mod = PositionEmbedding(4, 8)
        table = sincos_position_embedding_2d(8, 4)
        assert torch.allclose(mod.pos_embed.data, table)
        assert not mod.pos_embed.requires_grad


class TestPrompts:
    def test_detect_prompt_wraps_classes_in_p_tags(self):
        prompt = _bare("detect", ["bird", "pink car"])._task_prompt()
        assert "<p>bird</p>, <p>pink car</p>" in prompt
        assert "<bbox>" in prompt

    def test_pose_prompt_selects_human_skeleton(self):
        prompt = _bare("pose", ["person"])._task_prompt()
        assert "left wrist" in prompt and "root of tail" not in prompt

    def test_pose_prompt_selects_animal_skeleton(self):
        prompt = _bare("pose", ["cat"], user_vocab=True)._task_prompt()
        assert "root of tail" in prompt and "left wrist" not in prompt

    def test_pose_prompt_mixed(self):
        prompt = _bare("pose", ["person", "dog"], user_vocab=True)._task_prompt()
        assert "AP-10K" in prompt

    def test_pose_defaults_to_person_without_user_vocab(self):
        model = _bare("pose", ["person", "car"], user_vocab=False)
        assert model._task_names() == {0: "person"}

    def test_segment_requires_user_vocabulary(self):
        with pytest.raises(ValueError, match="set_classes"):
            _bare("segment")._task_prompt()
        prompt = _bare(
            "segment", ["person furthest to the right"], user_vocab=True
        )._task_prompt()
        assert "<p>person furthest to the right</p>" in prompt
        assert "binary" in prompt

    def test_panoptic_defaults_to_coco_panoptic_names(self):
        model = _bare("panoptic")
        assert len(COCO_PANOPTIC_NAMES) == 133
        prompt = model._task_prompt()
        assert "<p>person</p>" in prompt and "panoptic" in prompt
        assert "<color>(R,G,B)</color>" in prompt

    def test_every_supported_task_has_a_mode(self):
        for task in LibreSenseNovaVision.SUPPORTED_TASKS:
            assert task in _TASK_TO_MODE
            assert _TASK_TO_MODE[task] in BASE_PARAMS

    def test_custom_prompt_wins(self):
        model = _bare("detect")
        model._custom_prompt = "my prompt"
        assert model._task_prompt() == "my prompt"


class TestPostprocess:
    def test_detect_maps_labels_and_scales_boxes(self):
        det = _bare("detect")._postprocess(
            {"text": "<p>person</p><bbox>[0.1,0.2,0.5,0.4]</bbox>", "image": None},
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(1000, 500),
        )
        assert det["num_detections"] == 1
        assert det["boxes"] == [[100.0, 100.0, 500.0, 200.0]]
        assert det["classes"] == [0]

    def test_detect_drops_out_of_vocabulary_labels(self):
        det = _bare("detect")._postprocess(
            {"text": "<p>zebra</p><bbox>[0.1,0.2,0.5,0.4]</bbox>", "image": None},
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(100, 100),
        )
        assert det["num_detections"] == 0

    def test_point_task_returns_point_rows(self):
        det = _bare("point")._postprocess(
            {"text": "<p>car</p><point>[0.5,0.5]</point>", "image": None},
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(200, 100),
        )
        assert det["points"] == [[100.0, 50.0, 1.0, 1.0]]

    def test_pose_task_builds_keypoint_tensor(self):
        text = (
            "<p>person</p><bbox>[0.0,0.0,1.0,1.0]</bbox>"
            "nose<kpt>[0.5,0.25]</kpt>left shoulder<kpt>[0.4,0.5]</kpt>"
        )
        det = _bare("pose", ["person"], user_vocab=True)._postprocess(
            {"text": text, "image": None},
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(100, 100),
        )
        assert det["num_detections"] == 1
        assert det["keypoints"].shape == (1, 17, 3)
        assert det["keypoints"][0, 0].tolist() == [50.0, 25.0, 1.0]

    def test_ocr_returns_polygons_and_texts(self):
        det = _bare("ocr")._postprocess(
            {"text": "<p>STOP</p><bbox>[0.1,0.1,0.3,0.2]</bbox>", "image": None},
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(100, 100),
        )
        assert det["ocr"]["texts"] == ["STOP"]
        assert det["ocr"]["polygons"][0][0] == [10.0, 10.0]

    def test_depth_returns_dense_map(self):
        det = _bare("depth")._postprocess(
            {"text": None, "image": Image.new("L", (10, 10), 128)},
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(20, 40),
        )
        assert det["depth"].shape == (40, 20)
        assert det["depth"].max() == pytest.approx(128 / 255)

    def test_segment_returns_single_mask_instance(self):
        mask_img = Image.new("L", (10, 10), 0)
        for x in range(3, 7):
            for y in range(3, 7):
                mask_img.putpixel((x, y), 255)
        det = _bare("segment", ["the box"], user_vocab=True)._postprocess(
            {"text": None, "image": mask_img},
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(10, 10),
        )
        assert det["num_detections"] == 1
        assert det["masks"].shape == (1, 10, 10)
        assert det["boxes"] == [[3.0, 3.0, 7.0, 7.0]]

    def test_panoptic_extends_names_with_registered_phrases(self):
        model = _bare("panoptic", ["person"], user_vocab=True)
        img = Image.new("RGB", (10, 10), (255, 0, 0))
        caption = "<p>surfboard<color>(255,0,0)</color></p>"
        det = model._postprocess(
            {"text": caption, "image": img},
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(10, 10),
        )
        assert det["segments_info"][0]["category_id"] == 1
        assert model.names[1] == "surfboard"

    def test_missing_image_output_degrades_gracefully(self):
        det = _bare("depth")._postprocess(
            {"text": None, "image": None},
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(4, 6),
        )
        assert det["depth"].shape == (6, 4)
        seg = _bare("segment", ["x"], user_vocab=True)._postprocess(
            {"text": None, "image": None},
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(4, 4),
        )
        assert seg["num_detections"] == 0


class _FakeBaseTokenizer:
    """Byte-per-char stand-in for the base BPE."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 1000 for c in text]

    def decode(self, ids):
        return "".join(chr(i) if 31 < i < 127 else "?" for i in ids)


class TestCheckpointTokenizer:
    def _tok(self):
        from libreyolo.models.sensenova.data_utils import CheckpointTokenizer

        overrides = {
            2000: "<|im_start|>",
            2001: "<|im_end|>",
            2002: "<|vision_start|>",
            2003: "<|vision_end|>",
            2004: "<quat>",
        }
        return CheckpointTokenizer(_FakeBaseTokenizer(), overrides)

    def test_override_literals_encode_to_trained_ids(self):
        tok = self._tok()
        assert tok.encode("<|im_start|>hi<quat>") == [2000, ord("h"), ord("i"), 2004]

    def test_decode_maps_override_ids_back_to_literals(self):
        tok = self._tok()
        text = tok.decode([2000, ord("h"), ord("i"), 2001])
        assert text == "<|im_start|>hi<|im_end|>"
        assert text.split("<|im_end|>")[0].split("<|im_start|>")[1] == "hi"

    def test_decode_accepts_tensors(self):
        import torch

        tok = self._tok()
        assert tok.decode(torch.tensor([2000, ord("a")])) == "<|im_start|>a"

    def test_add_special_tokens_resolves_trained_ids(self):
        from libreyolo.models.sensenova.data_utils import add_special_tokens

        tok, ids, n_new = add_special_tokens(self._tok())
        assert n_new == 0
        assert ids == {
            "bos_token_id": 2000,
            "eos_token_id": 2001,
            "start_of_image": 2002,
            "end_of_image": 2003,
        }

    def test_refuses_unknown_token_registration(self):
        tok = self._tok()
        with pytest.raises(ValueError, match="checkpoint layout"):
            tok.add_tokens(["<|brand-new|>"])


class _FakeInferencer:
    """Stands in for InterleaveInferencer, returning canned generations."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def interleave_inference(self, input_lists, **kwargs):
        self.calls.append((input_lists, kwargs))
        return list(self.outputs)


def _loaded(task, names, outputs, user_vocab=True):
    import torch

    m = _bare(task, names, user_vocab=user_vocab)
    m.device = torch.device("cpu")
    m.input_size = 1024
    m.model = torch.nn.Identity()
    m.inferencer = _FakeInferencer(outputs)
    return m


class TestPredictEndToEnd:
    """Full predict-path smoke through the shared InferenceRunner."""

    def test_detect_predict_returns_boxes_results(self):
        model = _loaded(
            "detect",
            ["person", "car"],
            ["<p>person</p><bbox>[0.1,0.2,0.5,0.4]</bbox>"],
        )
        result = model.predict(Image.new("RGB", (200, 100)))
        assert result.boxes is not None
        assert result.boxes.xyxy.tolist() == [[20.0, 20.0, 100.0, 40.0]]
        assert result.names[int(result.boxes.cls[0])] == "person"
        # The prompt sent to the model carries the vocabulary.
        sent = model.inferencer.calls[0][0]
        assert any("<p>person</p>" in str(item) for item in sent)

    def test_depth_predict_returns_depth_map(self):
        model = _loaded("depth", ["person"], [Image.new("RGB", (64, 32), (200, 200, 200))])
        result = model.predict(Image.new("RGB", (64, 32)))
        assert result.depth_map is not None
        assert tuple(result.depth_map.data.shape) == (32, 64)

    def test_panoptic_predict_returns_panoptic_results(self):
        mask = Image.new("RGB", (20, 20), (255, 0, 0))
        caption = "<p>person-0<color>(255,0,0)</color></p>"
        model = _loaded("panoptic", ["person"], [caption, mask])
        result = model.predict(Image.new("RGB", (20, 20)))
        assert result.panoptic is not None
        assert result.panoptic.segments_info[0]["category_id"] == 0

    def test_pose_predict_returns_keypoints(self):
        text = (
            "<p>person</p><bbox>[0.0,0.0,1.0,1.0]</bbox>nose<kpt>[0.5,0.5]</kpt>"
        )
        model = _loaded("pose", ["person"], [text])
        result = model.predict(Image.new("RGB", (100, 100)))
        assert result.keypoints is not None
        assert tuple(result.keypoints.data.shape) == (1, 17, 3)

    def test_segment_predict_returns_masks(self):
        mask_img = Image.new("L", (10, 10), 0)
        for x in range(2, 8):
            for y in range(2, 8):
                mask_img.putpixel((x, y), 255)
        model = _loaded("segment", ["the thing"], [mask_img.convert("RGB")])
        result = model.predict(Image.new("RGB", (10, 10)))
        assert result.masks is not None
        assert result.boxes.xyxy.tolist() == [[2.0, 2.0, 8.0, 8.0]]

    def test_ocr_predict_returns_regions(self):
        model = _loaded("ocr", ["person"], ["<p>EXIT</p><bbox>[0.0,0.0,0.5,0.2]</bbox>"])
        result = model.predict(Image.new("RGB", (100, 100)))
        assert result.ocr is not None
        assert result.ocr.texts == ["EXIT"]


class TestFamilySurface:
    def test_import_order_does_not_break(self):
        # Regression: importing the family package before the vlm package used
        # to raise a circular-import error.
        import importlib
        import sys

        for name in [
            "libreyolo.models.sensenova",
            "libreyolo.models.sensenova.model",
            "libreyolo.models.vlm",
        ]:
            sys.modules.pop(name, None)
        sensenova = importlib.import_module("libreyolo.models.sensenova")
        vlm = importlib.import_module("libreyolo.models.vlm")
        assert vlm.LibreSenseNovaVision is sensenova.LibreSenseNovaVision

    def test_factory_knows_the_alias(self):
        from libreyolo.models.vlm import _LAZY_ALIASES, LibreVLM

        assert _LAZY_ALIASES["sensenova-vision"] == "7b"
        with pytest.raises(ValueError, match="sensenova-vision"):
            LibreVLM("definitely-not-a-model")

    def test_set_task_switches_without_reload(self):
        model = _bare("detect")
        assert model.set_task("depth") is model
        assert model.task == "depth"
        with pytest.raises(ValueError):
            model.set_task("classify")

    def test_supported_tasks_are_canonical(self):
        from libreyolo.tasks import TASKS

        for task in LibreSenseNovaVision.SUPPORTED_TASKS:
            assert task in TASKS

    def test_notice_names_the_license_mirror_and_upstream(self):
        notice = LibreSenseNovaVision._LICENSE_NOTICE
        assert "CC BY-NC 4.0" in notice
        assert "NON-COMMERCIAL" in notice
        assert "huggingface.co/LibreYOLO/SenseNovaVision7b" in notice
        assert "huggingface.co/sensenova/SenseNova-Vision-7B-MoT" in notice

    def test_default_score_below_conf_threshold_drops_everything(self):
        det = _bare("detect")._postprocess(
            {"text": "<p>person</p><bbox>[0.1,0.2,0.5,0.4]</bbox>", "image": None},
            conf_thres=1.5,
            iou_thres=0.5,
            original_size=(100, 100),
        )
        assert det["num_detections"] == 0
