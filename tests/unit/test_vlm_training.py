"""Offline unit tests for VLM detection fine-tuning.

Everything here runs without network, GPU, or model weights: target
serialization (including a full round-trip through the inference parser),
dataset reading, collator label masking with a stub processor, the checkpoint
contract, and the train() gating surface.
"""

import json
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.vlm]

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from libreyolo.models.vlm.parsing import build_detection_dict, extract_detections  # noqa: E402
from libreyolo.models.vlm.training.checkpoint import (  # noqa: E402
    CONTRACT_FILENAME,
    is_vlm_checkpoint,
    read_contract,
    save_vlm_checkpoint,
)
from libreyolo.models.vlm.training.collate import VLMChatCollator  # noqa: E402
from libreyolo.models.vlm.training.data import (  # noqa: E402
    VLMDetectDataset,
    resolve_split_annotation,
    resolve_split_source,
)
from libreyolo.models.vlm.training.recipes import get_recipe  # noqa: E402
from libreyolo.models.vlm.training.targets import (  # noqa: E402
    FamilyFormat,
    serialize_detections,
)

QWEN_FMT = FamilyFormat(
    family="qwen3vl",
    bbox_key="bbox_2d",
    coord_divisor=1000.0,
    box_format="xyxy",
    detection_prompt="Detect all instances of: cat, dog.",
)
UNIT_FMT = FamilyFormat(
    family="unit",
    bbox_key="bbox",
    coord_divisor=1.0,
    box_format="xyxy",
    detection_prompt="Detect.",
)


# ---------------------------------------------------------------------------
# Target serialization
# ---------------------------------------------------------------------------


class TestSerializeDetections:
    def test_qwen_format_scales_to_thousand_ints(self):
        text = serialize_detections([[0.1, 0.2, 0.5, 0.9]], ["cat"], QWEN_FMT)
        assert json.loads(text) == [{"bbox_2d": [100, 200, 500, 900], "label": "cat"}]

    def test_unit_scale_keeps_three_decimal_floats(self):
        text = serialize_detections([[0.1234, 0.2, 0.5, 0.8768]], ["cat"], UNIT_FMT)
        assert json.loads(text) == [{"label": "cat", "bbox": [0.123, 0.2, 0.5, 0.877]}]

    def test_empty_is_empty_array(self):
        assert serialize_detections([], [], QWEN_FMT) == "[]"

    def test_reading_order_is_deterministic(self):
        boxes = [[0.5, 0.8, 0.6, 0.9], [0.1, 0.1, 0.2, 0.2], [0.7, 0.1, 0.8, 0.2]]
        text = serialize_detections(boxes, ["low", "topleft", "topright"], QWEN_FMT)
        labels = [item["label"] for item in json.loads(text)]
        assert labels == ["topleft", "topright", "low"]

    def test_clamps_and_reorders_corners(self):
        text = serialize_detections([[0.9, 1.4, 0.1, -0.2]], ["cat"], QWEN_FMT)
        assert json.loads(text) == [{"bbox_2d": [100, 0, 900, 1000], "label": "cat"}]

    def test_layout_conversion_xywh_and_cxcywh(self):
        xywh = FamilyFormat("f", "bbox", 1.0, "xywh", "p")
        cxcywh = FamilyFormat("f", "bbox", 1.0, "cxcywh", "p")
        box = [[0.2, 0.2, 0.6, 0.8]]
        assert json.loads(serialize_detections(box, ["x"], xywh))[0]["bbox"] == [
            0.2,
            0.2,
            0.4,
            0.6,
        ]
        assert json.loads(serialize_detections(box, ["x"], cxcywh))[0]["bbox"] == [
            0.4,
            0.5,
            0.4,
            0.6,
        ]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            serialize_detections([[0, 0, 1, 1]], [], QWEN_FMT)

    def test_round_trip_through_inference_parser(self):
        """The serializer must be the exact inverse of the predict-time parser."""
        boxes = [[0.10, 0.20, 0.50, 0.90], [0.25, 0.05, 0.75, 0.55]]
        labels = ["cat", "dog"]
        text = serialize_detections(boxes, labels, QWEN_FMT)
        items = extract_detections(text)
        result = build_detection_dict(
            items,
            {"cat": 0, "dog": 1},
            (1000, 1000),  # W, H chosen so pixel coords equal 1000*normalized
            bbox_key=QWEN_FMT.bbox_key,
            coord_divisor=QWEN_FMT.coord_divisor,
            box_format=QWEN_FMT.box_format,
        )
        assert result["num_detections"] == 2
        recovered = sorted(
            (int(c), [round(v) for v in b])
            for c, b in zip(result["classes"], result["boxes"])
        )
        expected = sorted(
            (i, [round(v * 1000) for v in box]) for i, box in zip([0, 1], boxes)
        )
        assert recovered == expected


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _write_dataset(root: Path, rows_by_image: dict) -> Path:
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for stem, rows in rows_by_image.items():
        Image.new("RGB", (64, 48), (120, 130, 140)).save(images / f"{stem}.png")
        if rows is not None:
            (labels / f"{stem}.txt").write_text(rows, encoding="utf-8")
    return images


def _write_coco_dataset(
    root: Path, *, file_name: str = "sample.png"
) -> tuple[Path, Path]:
    images = root / "images" / "train"
    annotations = root / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)
    Image.new("RGB", (100, 50), (120, 130, 140)).save(images / "sample.png")
    payload = {
        "images": [{"id": 11, "file_name": file_name, "width": 100, "height": 50}],
        "categories": [{"id": 7, "name": "cat"}, {"id": 3, "name": "dog"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 11,
                "category_id": 7,
                "bbox": [-10, 5, 30, 20],
                "area": 600,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 11,
                "category_id": 3,
                "bbox": [50, 10, 25, 20],
                "area": 500,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 11,
                "category_id": 7,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 1,
            },
            {
                "id": 4,
                "image_id": 11,
                "category_id": 3,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "ignore": 1,
            },
        ],
    }
    annotation_file = annotations / "train.json"
    annotation_file.write_text(json.dumps(payload), encoding="utf-8")
    return images, annotation_file


class TestVLMDetectDataset:
    def test_renders_prompt_and_target(self, tmp_path):
        images = _write_dataset(tmp_path, {"a": "0 0.5 0.5 0.2 0.4\n"})
        ds = VLMDetectDataset(images, {0: "cat", 1: "dog"}, QWEN_FMT)
        sample = ds[0]
        assert sample["prompt"] == QWEN_FMT.detection_prompt
        parsed = json.loads(sample["target"])
        assert parsed == [{"bbox_2d": [400, 300, 600, 700], "label": "cat"}]
        assert sample["image"].size == (64, 48)

    def test_missing_label_file_teaches_empty_answer(self, tmp_path):
        images = _write_dataset(tmp_path, {"a": None})
        ds = VLMDetectDataset(images, {0: "cat"}, QWEN_FMT)
        assert ds[0]["target"] == "[]"

    def test_malformed_and_unknown_rows_are_skipped(self, tmp_path):
        images = _write_dataset(
            tmp_path, {"a": "not a row\n7 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n"}
        )
        ds = VLMDetectDataset(images, {0: "cat"}, QWEN_FMT)
        parsed = json.loads(ds[0]["target"])
        assert len(parsed) == 1 and parsed[0]["label"] == "cat"

    def test_hflip_mirrors_boxes(self, tmp_path):
        images = _write_dataset(tmp_path, {"a": "0 0.25 0.5 0.1 0.2\n"})
        ds = VLMDetectDataset(images, {0: "cat"}, QWEN_FMT, augment=True, hflip_p=1.0)
        parsed = json.loads(ds[0]["target"])
        # cx 0.25 -> mirrored cx 0.75; y untouched.
        assert parsed[0]["bbox_2d"] == [700, 400, 800, 600]

    def test_empty_image_dir_raises(self, tmp_path):
        empty = tmp_path / "none"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            VLMDetectDataset(empty, {0: "cat"}, QWEN_FMT)

    def test_native_coco_renders_clipped_boxes_and_skips_crowd_ignore(self, tmp_path):
        images, annotation_file = _write_coco_dataset(tmp_path)
        ds = VLMDetectDataset(
            images,
            {0: "cat", 1: "dog"},
            QWEN_FMT,
            annotation_file=annotation_file,
        )

        sample = ds[0]

        assert sample["image"].size == (100, 50)
        assert json.loads(sample["target"]) == [
            {"bbox_2d": [0, 100, 200, 500], "label": "cat"},
            {"bbox_2d": [500, 200, 750, 600], "label": "dog"},
        ]

    def test_native_coco_requires_matching_categories_and_contained_images(
        self, tmp_path
    ):
        images, annotation_file = _write_coco_dataset(tmp_path)
        with pytest.raises(ValueError, match="category name"):
            VLMDetectDataset(
                images,
                {0: "cat"},
                QWEN_FMT,
                annotation_file=annotation_file,
            )

        _, escaped_annotation = _write_coco_dataset(
            tmp_path / "escaped", file_name="../outside.png"
        )
        with pytest.raises(ValueError, match="escapes"):
            VLMDetectDataset(
                tmp_path / "escaped" / "images" / "train",
                {0: "cat", 1: "dog"},
                QWEN_FMT,
                annotation_file=escaped_annotation,
            )

    def test_native_coco_rejects_multiple_categories_for_one_yaml_label(self, tmp_path):
        images, annotation_file = _write_coco_dataset(tmp_path)
        payload = json.loads(annotation_file.read_text(encoding="utf-8"))
        payload["categories"] = [
            {"id": 7, "name": "cat"},
            {"id": 8, "name": "cat"},
        ]
        for annotation in payload["annotations"]:
            if annotation["category_id"] == 3:
                annotation["category_id"] = 8
        annotation_file.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="Multiple COCO categories"):
            VLMDetectDataset(
                images,
                {0: "cat"},
                QWEN_FMT,
                annotation_file=annotation_file,
            )

    @pytest.mark.parametrize(
        ("mutation", "match"),
        [
            ("string_image_id", "exact integer"),
            ("string_category_id", "exact integer"),
            ("duplicate_annotation_id", "duplicate id"),
            ("string_crowd", "integer 0 or 1"),
            ("string_bbox", "invalid bbox"),
        ],
    )
    def test_native_coco_rejects_ambiguous_index_fields(
        self, mutation, match, tmp_path
    ):
        images, annotation_file = _write_coco_dataset(tmp_path)
        payload = json.loads(annotation_file.read_text(encoding="utf-8"))
        if mutation == "string_image_id":
            payload["images"][0]["id"] = "11"
        elif mutation == "string_category_id":
            payload["categories"][0]["id"] = "7"
        elif mutation == "duplicate_annotation_id":
            duplicate = dict(payload["annotations"][0])
            duplicate["bbox"] = [50, 25, 20, 10]
            payload["annotations"].append(duplicate)
        elif mutation == "string_crowd":
            payload["annotations"][0]["iscrowd"] = "0"
        else:
            payload["annotations"][0]["bbox"][0] = "0"
        annotation_file.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match=match):
            VLMDetectDataset(
                images,
                {0: "cat", 1: "dog"},
                QWEN_FMT,
                annotation_file=annotation_file,
            )

    def test_native_coco_rejects_duplicate_json_keys(self, tmp_path):
        images, annotation_file = _write_coco_dataset(tmp_path)
        text = annotation_file.read_text(encoding="utf-8")
        annotation_file.write_text(
            text.replace('"images":', '"images": [], "images":', 1),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="duplicate key|invalid"):
            VLMDetectDataset(
                images,
                {0: "cat", 1: "dog"},
                QWEN_FMT,
                annotation_file=annotation_file,
            )

    def test_native_coco_rejects_overflowing_json_float(self, tmp_path):
        images, annotation_file = _write_coco_dataset(tmp_path)
        text = annotation_file.read_text(encoding="utf-8")
        annotation_file.write_text(
            text.replace('"area": 600', '"area": 1e400', 1),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="invalid"):
            VLMDetectDataset(
                images,
                {0: "cat", 1: "dog"},
                QWEN_FMT,
                annotation_file=annotation_file,
            )

    def test_native_coco_preflight_decodes_image_bytes(self, tmp_path):
        images, annotation_file = _write_coco_dataset(tmp_path)
        image_path = images / "sample.png"
        Image.new("RGB", (100, 50), (120, 130, 140)).save(image_path, format="JPEG")
        payload = image_path.read_bytes()
        image_path.write_bytes(payload[:-20])

        with pytest.raises(ValueError, match="cannot be read"):
            VLMDetectDataset.validate_native_coco_source(
                images,
                {0: "cat", 1: "dog"},
                annotation_file,
            )

    def test_native_coco_rejects_image_above_pixel_limit(self, tmp_path, monkeypatch):
        images, annotation_file = _write_coco_dataset(tmp_path)
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

        with pytest.raises(ValueError, match="safe pixel limit"):
            VLMDetectDataset.validate_native_coco_source(
                images,
                {0: "cat", 1: "dog"},
                annotation_file,
            )

    def test_native_coco_checks_declared_image_dimensions(self, tmp_path):
        images, annotation_file = _write_coco_dataset(tmp_path)
        payload = json.loads(annotation_file.read_text(encoding="utf-8"))
        payload["images"][0]["width"] = 101
        annotation_file.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="annotations declare"):
            VLMDetectDataset(
                images,
                {0: "cat", 1: "dog"},
                QWEN_FMT,
                annotation_file=annotation_file,
            )

    def test_split_helpers_keep_source_and_resolved_annotation_separate(self):
        cfg = {"train": "imgs", "train_annotation_file": "x.json"}
        assert resolve_split_source(cfg, "train") == "imgs"
        assert resolve_split_annotation(cfg, "train") == "x.json"
        assert resolve_split_annotation(cfg, "val") is None

    def test_trainer_propagates_native_coco_annotations_to_both_splits(
        self, monkeypatch
    ):
        from types import SimpleNamespace

        from libreyolo.models.vlm.training import trainer as trainer_module
        from libreyolo.models.vlm.training.trainer import (
            VLMDetectionTrainer,
            VLMTrainConfig,
        )

        captured = []

        def fake_dataset(source, names, fmt, **kwargs):
            captured.append((source, names, fmt, kwargs))
            return [source]

        monkeypatch.setattr(trainer_module, "VLMDetectDataset", fake_dataset)
        monkeypatch.setattr(
            trainer_module, "VLMChatCollator", lambda *_args, **_kwargs: object()
        )
        monkeypatch.setattr(
            trainer_module,
            "DataLoader",
            lambda dataset, **_kwargs: dataset,
        )
        trainer = object.__new__(VLMDetectionTrainer)
        trainer.config = VLMTrainConfig(data="unused.yaml", workers=0, device="cpu")
        trainer.wrapper = SimpleNamespace(processor=object())
        trainer.recipe = SimpleNamespace(max_length_warn=4096)
        monkeypatch.setattr(trainer, "_resolve_device", lambda: torch.device("cpu"))

        train_loader, val_loader = trainer._build_dataloaders(
            {
                "train": "train-images",
                "val": "val-images",
                "train_annotation_file": "train.json",
                "val_annotation_file": "val.json",
            },
            {0: "cat"},
            QWEN_FMT,
        )

        assert train_loader == ["train-images"]
        assert val_loader == ["val-images"]
        assert captured[0][3]["annotation_file"] == "train.json"
        assert captured[1][3]["annotation_file"] == "val.json"

        with pytest.raises(ValueError, match="no val image source"):
            trainer._build_dataloaders(
                {
                    "train": "train-images",
                    "val_annotation_file": "val.json",
                },
                {0: "cat"},
                QWEN_FMT,
            )


# ---------------------------------------------------------------------------
# Collator (stub processor: ids are deterministic functions of the text)
# ---------------------------------------------------------------------------


class _StubTokenizer:
    padding_side = "left"  # collator must flip to right and restore


class _StubProcessor:
    """Tokenizes text as its character codes; images add three fixed tokens."""

    PAD = 0
    IMAGE_TOKENS = [900, 901, 902]

    def __init__(self, break_prefix=False):
        self.tokenizer = _StubTokenizer()
        self.break_prefix = break_prefix

    def _encode_conv(self, conversation, add_generation_prompt):
        ids = []
        for turn in conversation:
            for part in turn["content"]:
                if part["type"] == "image":
                    ids.extend(self.IMAGE_TOKENS)
                else:
                    ids.extend((ord(c) % 500) + 1000 for c in part["text"])
        if add_generation_prompt:
            ids.append(3)  # assistant header stub
        elif len(conversation) > 1:
            # Full conversation: header sits between prompt and answer.
            answer = ids[-sum(len(p["text"]) for p in conversation[-1]["content"]) :]
            prompt = ids[: len(ids) - len(answer)]
            first = [7] if self.break_prefix else []
            ids = first + prompt + [3] + answer + [4]  # 4 = eos stub
        return ids

    def apply_chat_template(
        self,
        conversations,
        tokenize,
        return_dict,
        return_tensors,
        padding,
        add_generation_prompt=False,
    ):
        assert tokenize and return_dict and padding
        assert self.tokenizer.padding_side == "right"
        encoded = [
            self._encode_conv(conv, add_generation_prompt) for conv in conversations
        ]
        width = max(len(e) for e in encoded)
        input_ids = torch.full((len(encoded), width), self.PAD, dtype=torch.long)
        attention = torch.zeros((len(encoded), width), dtype=torch.long)
        for i, ids in enumerate(encoded):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attention[i, : len(ids)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "token_type_ids": torch.zeros_like(input_ids),
        }


def _samples():
    image = Image.new("RGB", (8, 8))
    return [
        {"image": image, "prompt": "find cats", "target": "[]"},
        {
            "image": image,
            "prompt": "find cats",
            "target": '[{"bbox_2d": [1, 2, 3, 4]}]',
        },
    ]


class TestVLMChatCollator:
    def test_masks_prompt_and_padding_supervises_answer(self):
        collator = VLMChatCollator(_StubProcessor())
        batch = collator(_samples())
        labels, ids = batch["labels"], batch["input_ids"]
        for i, sample in enumerate(_samples()):
            prompt_len = 3 + len(sample["prompt"]) + 1  # image + text + header
            assert (labels[i, :prompt_len] == -100).all()
            answer_len = len(sample["target"]) + 1  # + eos stub
            answer = labels[i, prompt_len : prompt_len + answer_len]
            assert (answer != -100).all()
            assert torch.equal(answer, ids[i, prompt_len : prompt_len + answer_len])
            assert (labels[i, prompt_len + answer_len :] == -100).all()

    def test_supervised_fraction_is_answer_only(self):
        batch = VLMChatCollator(_StubProcessor())(_samples())
        supervised = int((batch["labels"] != -100).sum())
        expected = sum(len(s["target"]) + 1 for s in _samples())
        assert supervised == expected

    def test_prefix_violation_raises(self):
        collator = VLMChatCollator(_StubProcessor(break_prefix=True))
        with pytest.raises(RuntimeError, match="prompt-prefix"):
            collator(_samples())

    def test_padding_side_restored_and_drop_keys(self):
        processor = _StubProcessor()
        batch = VLMChatCollator(processor)(_samples())
        assert processor.tokenizer.padding_side == "left"
        assert "token_type_ids" not in batch


# ---------------------------------------------------------------------------
# Checkpoint contract
# ---------------------------------------------------------------------------


class _StubSaveable:
    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}', encoding="utf-8"
        )
        (Path(directory) / "adapter_model.safetensors").write_bytes(b"adapter")


class _FilesSaveable:
    def __init__(self, files):
        self.files = files

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, contents in self.files.items():
            (directory / name).write_text(contents, encoding="utf-8")


class _FailingSaveable:
    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "partial-processor.json").write_text("new", encoding="utf-8")
        raise RuntimeError("injected save failure")


class _StubWrapper:
    FAMILY = "qwen3vl"
    size = "2b"
    HF_REPOS = {"2b": "org/base-repo"}
    HF_REVISIONS = {}
    BBOX_KEY = "bbox_2d"
    COORD_DIVISOR = 1000.0
    BOX_FORMAT = "xyxy"
    names = {0: "cat", 1: "dog"}
    processor = _StubSaveable()

    def _format_detection_prompt(self, names):
        return (
            f"Detect all instances of: {names}. "
            "Output the result as a JSON array, one object per instance: "
            '[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
            "Only include objects that are actually visible; if there are none, "
            "respond with an empty array []."
        )

    def _detection_prompt(self):
        return self._format_detection_prompt(", ".join(self.names.values()))


class TestCheckpointContract:
    def test_save_then_read_round_trips(self, tmp_path):
        target = tmp_path / "best"
        save_vlm_checkpoint(
            target,
            peft_model=_StubSaveable(),
            processor=_StubSaveable(),
            wrapper=_StubWrapper(),
            metrics={"train/loss": 0.5},
        )
        assert is_vlm_checkpoint(target)
        contract = read_contract(target)
        assert contract["family"] == "qwen3vl"
        assert contract["size"] == "2b"
        assert contract["base_repo"] == "org/base-repo"
        assert contract["names"] == ["cat", "dog"]
        assert contract["bbox_key"] == "bbox_2d"
        assert contract["coord_divisor"] == 1000.0
        assert contract["box_format"] == "xyxy"
        assert contract["prompt"] == _StubWrapper()._detection_prompt()
        assert contract["task"] == "detect"
        assert contract["metrics"]["train/loss"] == 0.5

    @pytest.mark.parametrize("bad_metric", [float("nan"), float("inf"), True])
    def test_checkpoint_rejects_nonfinite_or_boolean_metrics(
        self, tmp_path, bad_metric
    ):
        target = tmp_path / "bad-metrics"

        with pytest.raises(ValueError, match="metrics must be finite"):
            save_vlm_checkpoint(
                target,
                peft_model=_StubSaveable(),
                processor=_StubSaveable(),
                wrapper=_StubWrapper(),
                metrics={"train/loss": bad_metric},
            )

        assert not target.exists()

    def test_save_refuses_incomplete_adapter_before_publication(self, tmp_path):
        target = tmp_path / "incomplete"

        with pytest.raises(ValueError, match="no adapter tensor payload"):
            save_vlm_checkpoint(
                target,
                peft_model=_FilesSaveable(
                    {"adapter_config.json": '{"peft_type":"LORA"}'}
                ),
                processor=_FilesSaveable({"preprocessor_config.json": "processor"}),
                wrapper=_StubWrapper(),
            )

        assert not target.exists()

    @pytest.mark.parametrize(
        "adapter_files",
        [
            {
                "adapter_config.json": '{"peft_type":"LORA"}',
                "adapter_model-00001-of-00002.safetensors": "partial",
            },
            {
                "adapter_config.json": '{"peft_type":"LORA"}',
                "adapter_model.safetensors.index.json": json.dumps(
                    {
                        "weight_map": {
                            "a": "adapter_model-00001-of-00002.safetensors",
                            "b": "adapter_model-00002-of-00002.safetensors",
                        }
                    }
                ),
                "adapter_model-00001-of-00002.safetensors": "first",
            },
            {
                "adapter_config.json": '{"peft_type":"LORA"}',
                "adapter_model.safetensors.index.json": json.dumps(
                    {"weight_map": {"a": []}}
                ),
            },
            {
                "adapter_config.json": '{"peft_type":"LORA"}',
                "adapter_model.safetensors.index.json": json.dumps(
                    {
                        "weight_map": {
                            "a": "adapter_model-00001-of-00002.safetensors",
                            "b": "adapter_model-00002-of-00002.safetensors",
                        }
                    }
                ),
                "adapter_model-00001-of-00002.safetensors": "first",
                "adapter_model-00002-of-00002.safetensors": "second",
            },
        ],
        ids=["loose-shard", "incomplete-index", "malformed-index", "complete-index"],
    )
    def test_sharded_adapters_are_rejected_before_publication(
        self, tmp_path, adapter_files
    ):
        target = tmp_path / "sharded-adapter"

        with pytest.raises(
            ValueError, match=r"sharded tensor payload.*PEFT loader cannot load"
        ):
            save_vlm_checkpoint(
                target,
                peft_model=_FilesSaveable(adapter_files),
                processor=_FilesSaveable({"preprocessor_config.json": "processor"}),
                wrapper=_StubWrapper(),
            )

        assert not target.exists()

    def test_full_model_shard_index_must_be_complete(self, tmp_path):
        target = tmp_path / "incomplete-full"
        index = json.dumps(
            {"weight_map": {"layer.weight": "model-00001-of-00002.safetensors"}}
        )

        with pytest.raises(ValueError, match="missing or empty shards"):
            save_vlm_checkpoint(
                target,
                peft_model=_FilesSaveable(
                    {
                        "config.json": "{}",
                        "model.safetensors.index.json": index,
                    }
                ),
                processor=_FilesSaveable({"preprocessor_config.json": "processor"}),
                wrapper=_StubWrapper(),
            )

        assert not target.exists()

    def test_complete_indexed_full_model_is_publishable(self, tmp_path):
        target = tmp_path / "indexed-full"
        index = json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )

        save_vlm_checkpoint(
            target,
            peft_model=_FilesSaveable(
                {
                    "config.json": "{}",
                    "model.safetensors.index.json": index,
                    "model-00001-of-00002.safetensors": "first",
                    "model-00002-of-00002.safetensors": "second",
                }
            ),
            processor=_FilesSaveable({"preprocessor_config.json": "processor"}),
            wrapper=_StubWrapper(),
        )

        assert read_contract(target)["family"] == "qwen3vl"
        assert (target / "model.safetensors.index.json").is_file()

    def test_is_vlm_checkpoint_negative_cases(self, tmp_path):
        assert not is_vlm_checkpoint(tmp_path)  # dir without contract
        assert not is_vlm_checkpoint(tmp_path / "missing")
        assert not is_vlm_checkpoint("qwen3-vl-4b")  # alias string

    @pytest.mark.parametrize(
        ("first_files", "replacement_files"),
        [
            (
                {"config.json": "{}", "model.safetensors": "full-weights"},
                {
                    "adapter_config.json": '{"peft_type":"LORA"}',
                    "adapter_model.safetensors": "adapter-weights",
                },
            ),
            (
                {
                    "adapter_config.json": '{"peft_type":"LORA"}',
                    "adapter_model.safetensors": "adapter-weights",
                },
                {"config.json": "{}", "model.safetensors": "full-weights"},
            ),
        ],
    )
    def test_save_replaces_existing_checkpoint_without_stale_files(
        self, tmp_path, first_files, replacement_files
    ):
        target = tmp_path / "best"
        processor = _FilesSaveable({"preprocessor_config.json": "processor"})
        save_vlm_checkpoint(
            target,
            peft_model=_FilesSaveable(first_files),
            processor=processor,
            wrapper=_StubWrapper(),
            metrics={"train/loss": 1.0},
        )

        save_vlm_checkpoint(
            target,
            peft_model=_FilesSaveable(replacement_files),
            processor=processor,
            wrapper=_StubWrapper(),
            metrics={"train/loss": 0.25},
        )

        assert all(not (target / name).exists() for name in first_files)
        assert all(
            (target / name).read_text(encoding="utf-8") == contents
            for name, contents in replacement_files.items()
        )
        assert is_vlm_checkpoint(target)
        assert read_contract(target)["metrics"] == {"train/loss": 0.25}

    def test_failed_replacement_leaves_existing_checkpoint_intact(self, tmp_path):
        target = tmp_path / "best"
        save_vlm_checkpoint(
            target,
            peft_model=_FilesSaveable(
                {"config.json": "{}", "model.safetensors": "full-weights"}
            ),
            processor=_FilesSaveable(
                {"preprocessor_config.json": "original-processor"}
            ),
            wrapper=_StubWrapper(),
            metrics={"train/loss": 1.0},
        )
        original = {
            path.relative_to(target): path.read_bytes()
            for path in target.rglob("*")
            if path.is_file()
        }

        with pytest.raises(RuntimeError, match="injected save failure"):
            save_vlm_checkpoint(
                target,
                peft_model=_FilesSaveable(
                    {
                        "adapter_config.json": '{"peft_type":"LORA"}',
                        "adapter_model.safetensors": "adapter-weights",
                    }
                ),
                processor=_FailingSaveable(),
                wrapper=_StubWrapper(),
                metrics={"train/loss": 0.25},
            )

        current = {
            path.relative_to(target): path.read_bytes()
            for path in target.rglob("*")
            if path.is_file()
        }
        assert current == original
        assert is_vlm_checkpoint(target)
        assert read_contract(target)["metrics"] == {"train/loss": 1.0}

    def test_publication_failure_rolls_back_existing_checkpoint(
        self, tmp_path, monkeypatch
    ):
        from libreyolo.models.vlm.training import checkpoint as checkpoint_module

        target = tmp_path / "best"
        save_vlm_checkpoint(
            target,
            peft_model=_FilesSaveable(
                {"config.json": "{}", "model.safetensors": "full-weights"}
            ),
            processor=_FilesSaveable(
                {"preprocessor_config.json": "original-processor"}
            ),
            wrapper=_StubWrapper(),
        )
        original = {
            path.relative_to(target): path.read_bytes()
            for path in target.rglob("*")
            if path.is_file()
        }
        real_replace = checkpoint_module.os.replace
        calls = 0

        def fail_publication(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected publication failure")
            return real_replace(source, destination)

        monkeypatch.setattr(checkpoint_module.os, "replace", fail_publication)

        with pytest.raises(OSError, match="injected publication failure"):
            save_vlm_checkpoint(
                target,
                peft_model=_StubSaveable(),
                processor=_FilesSaveable(
                    {"preprocessor_config.json": "replacement-processor"}
                ),
                wrapper=_StubWrapper(),
            )

        restored = {
            path.relative_to(target): path.read_bytes()
            for path in target.rglob("*")
            if path.is_file()
        }
        assert restored == original
        assert not list(tmp_path.glob(".best.backup-*"))

    def test_wrong_schema_rejected(self, tmp_path):
        (tmp_path / CONTRACT_FILENAME).write_text(
            json.dumps({"schema": 99, "family": "x", "size": "s", "names": []}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="schema"):
            read_contract(tmp_path)

    def test_missing_field_rejected(self, tmp_path):
        (tmp_path / CONTRACT_FILENAME).write_text(
            json.dumps({"schema": 1, "family": "x"}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="size"):
            read_contract(tmp_path)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("bbox_key", "", "bbox_key"),
            ("base_repo", "", "base_repo"),
            ("base_revision", "main", "base_revision"),
            ("base_revision", 123, "base_revision"),
            ("coord_divisor", 0, "coord_divisor"),
            ("coord_divisor", "1000", "coord_divisor"),
            ("box_format", "corners", "box_format"),
            ("box_format", ["xyxy"], "box_format"),
            ("prompt", None, "prompt"),
            ("names", [], "names"),
            ("names", ["cat", " CAT "], "unique case-insensitively"),
            ("task", "segment", "task"),
            ("task", None, "task"),
        ],
    )
    def test_invalid_contract_fields_are_rejected(self, tmp_path, field, value, match):
        contract = {
            "schema": 1,
            "family": "qwen3vl",
            "size": "2b",
            "base_repo": "Qwen/Qwen3-VL-2B-Instruct",
            "base_revision": None,
            "names": ["cat"],
            "bbox_key": "bbox_2d",
            "coord_divisor": 1000.0,
            "box_format": "xyxy",
            "prompt": "Detect cats.",
            "task": "detect",
        }
        contract[field] = value
        (tmp_path / CONTRACT_FILENAME).write_text(
            json.dumps(contract), encoding="utf-8"
        )
        with pytest.raises(ValueError, match=match):
            read_contract(tmp_path)

    def test_non_object_contract_is_rejected(self, tmp_path):
        (tmp_path / CONTRACT_FILENAME).write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            read_contract(tmp_path)

    def test_schema_one_prompt_and_coordinate_convention_are_restored(
        self, tmp_path, monkeypatch
    ):
        import sys
        from types import SimpleNamespace

        from libreyolo.models.base.model import BaseModel
        from libreyolo.models.vlm.qwen3vl import LibreQwen3VL

        target = tmp_path / "legacy"
        save_vlm_checkpoint(
            target,
            peft_model=_StubSaveable(),
            processor=_StubSaveable(),
            wrapper=_StubWrapper(),
        )
        contract_path = target / CONTRACT_FILENAME
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract.update(
            bbox_key="legacy_box",
            coord_divisor=512,
            box_format="xywh",
            prompt="Use the saved schema-one detection prompt.",
        )
        contract_path.write_text(json.dumps(contract), encoding="utf-8")

        class _Model:
            def eval(self):
                return self

        def _offline_base_init(instance, *, size, nb_classes, **kwargs):
            del kwargs
            instance.size = size
            instance.nb_classes = nb_classes
            instance.names = {i: str(i) for i in range(nb_classes)}
            instance.model = _Model()

        monkeypatch.setattr(BaseModel, "__init__", _offline_base_init)
        monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=object))
        with pytest.raises(ValueError, match="prompt must be a non-empty string"):
            LibreQwen3VL(size="2b", prompt="", device="cpu")
        model = LibreQwen3VL(
            size="2b",
            checkpoint_dir=str(target),
            names=["cat", "dog"],
            device="cpu",
        )
        assert model.BBOX_KEY == "legacy_box"
        assert model.COORD_DIVISOR == 512.0
        assert model.BOX_FORMAT == "xywh"
        assert model.HF_REPOS["2b"] == "org/base-repo"
        assert "2b" not in model.HF_REVISIONS
        assert model._detection_prompt() == "Use the saved schema-one detection prompt."
        with pytest.raises(ValueError, match="saved prompt and box convention"):
            model.set_classes(["fox"])
        assert model.names == {0: "cat", 1: "dog"}
        assert model._detection_prompt() == "Use the saved schema-one detection prompt."

        overridden = LibreQwen3VL(
            size="2b",
            checkpoint_dir=str(target),
            names=["cat", "dog"],
            prompt="Caller override.",
            device="cpu",
        )
        assert overridden._detection_prompt() == "Caller override."
        assert overridden.BBOX_KEY == "legacy_box"
        assert overridden.COORD_DIVISOR == 512.0
        assert overridden.BOX_FORMAT == "xywh"
        overridden.set_classes(["fox"])
        assert overridden.names == {0: "fox"}
        assert overridden._detection_prompt() == "Caller override."

        current_target = tmp_path / "current"
        save_vlm_checkpoint(
            current_target,
            peft_model=_StubSaveable(),
            processor=_StubSaveable(),
            wrapper=_StubWrapper(),
        )
        current = LibreQwen3VL(
            size="2b",
            checkpoint_dir=str(current_target),
            names=["cat", "dog"],
            device="cpu",
        )
        current.set_classes(["fox"])
        assert "fox" in current._detection_prompt()

        custom_target = tmp_path / "custom-prompt"
        save_vlm_checkpoint(
            custom_target,
            peft_model=_StubSaveable(),
            processor=_StubSaveable(),
            wrapper=_StubWrapper(),
        )
        custom_contract_path = custom_target / CONTRACT_FILENAME
        custom_contract = json.loads(custom_contract_path.read_text(encoding="utf-8"))
        custom_contract["prompt"] = "A learned custom detection prompt."
        custom_contract_path.write_text(json.dumps(custom_contract), encoding="utf-8")
        with pytest.raises(ValueError, match="requested class vocabulary"):
            LibreQwen3VL(
                size="2b",
                checkpoint_dir=str(custom_target),
                names=["fox"],
                device="cpu",
            )
        custom = LibreQwen3VL(
            size="2b",
            checkpoint_dir=str(custom_target),
            names=["cat", "dog"],
            device="cpu",
        )
        assert custom._detection_prompt() == "A learned custom detection prompt."
        with pytest.raises(ValueError, match="saved prompt and box convention"):
            custom.set_classes(["fox"])

        custom_override = LibreQwen3VL(
            size="2b",
            checkpoint_dir=str(custom_target),
            names=["cat", "dog"],
            prompt="Caller rebuilt prompt for fox.",
            device="cpu",
        )
        custom_override.set_classes(["fox"])
        assert custom_override._detection_prompt() == "Caller rebuilt prompt for fox."

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [("family", "lfm2vl", "belongs to family"), ("size", "4b", "trained at size")],
    )
    def test_checkpoint_identity_must_match_adapter(
        self, tmp_path, monkeypatch, field, value, match
    ):
        from libreyolo.models.base.model import BaseModel
        from libreyolo.models.vlm.qwen3vl import LibreQwen3VL

        target = tmp_path / field
        save_vlm_checkpoint(
            target,
            peft_model=_StubSaveable(),
            processor=_StubSaveable(),
            wrapper=_StubWrapper(),
        )
        contract_path = target / CONTRACT_FILENAME
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract[field] = value
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        monkeypatch.setattr(BaseModel, "__init__", lambda *args, **kwargs: None)

        with pytest.raises(ValueError, match=match):
            LibreQwen3VL(size="2b", checkpoint_dir=str(target), device="cpu")

    def test_direct_constructor_rejects_incomplete_checkpoint_before_base_init(
        self, tmp_path, monkeypatch
    ):
        from libreyolo.models.base.model import BaseModel
        from libreyolo.models.vlm.qwen3vl import LibreQwen3VL

        target = tmp_path / "incomplete"
        save_vlm_checkpoint(
            target,
            peft_model=_StubSaveable(),
            processor=_StubSaveable(),
            wrapper=_StubWrapper(),
        )
        (target / "adapter_model.safetensors").unlink()
        monkeypatch.setattr(
            BaseModel,
            "__init__",
            lambda *_args, **_kwargs: pytest.fail("incomplete adapter loaded base"),
        )

        with pytest.raises(ValueError, match="no adapter tensor payload"):
            LibreQwen3VL(size="2b", checkpoint_dir=str(target), device="cpu")

    def test_direct_constructor_rejects_indexed_adapter_before_base_init(
        self, tmp_path, monkeypatch
    ):
        from libreyolo.models.base.model import BaseModel
        from libreyolo.models.vlm.qwen3vl import LibreQwen3VL

        target = tmp_path / "indexed-adapter"
        save_vlm_checkpoint(
            target,
            peft_model=_StubSaveable(),
            processor=_StubSaveable(),
            wrapper=_StubWrapper(),
        )
        (target / "adapter_model.safetensors").unlink()
        (target / "adapter_model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "a": "adapter_model-00001-of-00002.safetensors",
                        "b": "adapter_model-00002-of-00002.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )
        (target / "adapter_model-00001-of-00002.safetensors").write_bytes(b"first")
        (target / "adapter_model-00002-of-00002.safetensors").write_bytes(b"second")
        monkeypatch.setattr(
            BaseModel,
            "__init__",
            lambda *_args, **_kwargs: pytest.fail("indexed adapter loaded base"),
        )

        with pytest.raises(
            ValueError, match=r"sharded tensor payload.*PEFT loader cannot load"
        ):
            LibreQwen3VL(size="2b", checkpoint_dir=str(target), device="cpu")

    def test_direct_adapter_constructor_requires_peft_before_base_init(
        self, tmp_path, monkeypatch
    ):
        import sys

        from libreyolo.models.base.model import BaseModel
        from libreyolo.models.vlm.qwen3vl import LibreQwen3VL

        target = tmp_path / "adapter"
        save_vlm_checkpoint(
            target,
            peft_model=_StubSaveable(),
            processor=_StubSaveable(),
            wrapper=_StubWrapper(),
        )
        monkeypatch.setitem(sys.modules, "peft", None)
        monkeypatch.setattr(
            BaseModel,
            "__init__",
            lambda *_args, **_kwargs: pytest.fail("missing PEFT loaded the base"),
        )

        with pytest.raises(ImportError, match=r"libreyolo\[vlm\]"):
            LibreQwen3VL(size="2b", checkpoint_dir=str(target), device="cpu")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("family", "lfm2vl"),
            ("size", "4b"),
            ("base_repo", "other/base"),
            ("base_revision", "a" * 40),
            ("bbox_key", "legacy_box"),
            ("coord_divisor", 512.0),
            ("box_format", "xywh"),
        ],
    )
    def test_resume_contract_must_match_pristine_base(self, tmp_path, field, value):
        from libreyolo.models.vlm.training.trainer import VLMDetectionTrainer

        target = tmp_path / field
        save_vlm_checkpoint(
            target,
            peft_model=_StubSaveable(),
            processor=_StubSaveable(),
            wrapper=_StubWrapper(),
        )
        contract_path = target / CONTRACT_FILENAME
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract[field] = value
        contract_path.write_text(json.dumps(contract), encoding="utf-8")

        trainer = object.__new__(VLMDetectionTrainer)
        trainer.wrapper = _StubWrapper()
        with pytest.raises(ValueError, match=field):
            trainer._validate_resume_contract(target)

    def test_resume_contract_requires_reconstructible_prompt(self, tmp_path):
        from libreyolo.models.vlm.training.trainer import VLMDetectionTrainer

        target = tmp_path / "prompt"
        save_vlm_checkpoint(
            target,
            peft_model=_StubSaveable(),
            processor=_StubSaveable(),
            wrapper=_StubWrapper(),
        )
        contract_path = target / CONTRACT_FILENAME
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["prompt"] = "A custom learned prompt."
        contract_path.write_text(json.dumps(contract), encoding="utf-8")

        trainer = object.__new__(VLMDetectionTrainer)
        trainer.wrapper = _StubWrapper()
        with pytest.raises(ValueError, match="exact prompt"):
            trainer._validate_resume_contract(target)


# ---------------------------------------------------------------------------
# Gating and recipes
# ---------------------------------------------------------------------------


class TestTrainGating:
    def _bare(self, cls, size=None):
        model = object.__new__(cls)
        if size is not None:
            model.size = size
        return model

    def test_qwen3vl_is_trainable_and_requires_data(self):
        from libreyolo.models.vlm import LibreQwen3VL

        assert LibreQwen3VL.TRAINABLE is True
        assert LibreQwen3VL.TRAINABLE_SIZES == ("2b", "4b")
        with pytest.raises(ValueError, match="data="):
            LibreQwen3VL.train(self._bare(LibreQwen3VL, "2b"))

    def test_qwen3vl_unverified_size_is_rejected_before_trainer_setup(self):
        from libreyolo.models.vlm import LibreQwen3VL

        with pytest.raises(NotImplementedError, match="size '8b'.*2b, 4b"):
            LibreQwen3VL.train(self._bare(LibreQwen3VL, "8b"), data="unused.yaml")

    def test_loaded_checkpoint_is_rejected_before_trainer_setup(self, tmp_path):
        from libreyolo.models.vlm import LibreQwen3VL

        model = self._bare(LibreQwen3VL, "2b")
        model._checkpoint_dir = tmp_path / "merged-adapter"

        with pytest.raises(NotImplementedError, match="already been merged"):
            LibreQwen3VL.train(model, data="unused.yaml")

    @pytest.mark.parametrize(
        "module_name, class_name, match",
        [
            ("smolvlm", "LibreSmolVLM2", "grounding"),
            ("kosmos2", "LibreKosmos2", "recipe"),
            ("locateanything", "LibreLocateAnything", "non-commercial"),
        ],
    )
    def test_untrainable_families_explain_why(self, module_name, class_name, match):
        import importlib

        module = importlib.import_module(f"libreyolo.models.vlm.{module_name}")
        cls = getattr(module, class_name)
        assert cls.TRAINABLE is False
        with pytest.raises(NotImplementedError, match=match):
            cls.train(self._bare(cls), data="x.yaml")

    def test_recipe_exists_for_every_trainable_family(self):
        from libreyolo.models.vlm import _ALIASES

        for cls, _size in set(_ALIASES.values()):
            if cls.TRAINABLE:
                assert cls.TRAINABLE_SIZES
                assert set(cls.TRAINABLE_SIZES) <= set(cls.HF_REPOS)
                recipe = get_recipe(cls.FAMILY)
                assert recipe.target_modules

    def test_lfm_candidate_recipe_is_pinned_and_scope_limited(self):
        from libreyolo.models.vlm.lfm2 import LibreLFM2VL

        assert LibreLFM2VL.HF_REVISIONS == {
            "450m": "fc6221ca597f3315e4f82fc2df606783267b34ba",
            "1.6b": "919fde3d022e3f90a4716006f993938ee8c2eb97",
            "3b": "5a414ead75d45db003906d06fb62bd5b6846cec0",
        }
        assert LibreLFM2VL.TRAINABLE is False
        recipe = get_recipe("lfm2vl")
        for name in (
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.in_proj",
            "model.language_model.layers.0.feed_forward.w1",
            "model.language_model.layers.0.feed_forward.w3",
        ):
            assert re.fullmatch(recipe.target_modules, name)
        for name in (
            "model.vision_tower.blocks.0.attn.q_proj",
            "model.multi_modal_projector.linear",
            "model.language_model.embed_tokens",
        ):
            assert re.fullmatch(recipe.target_modules, name) is None
        assert recipe.frozen_prefixes == (
            "model.vision_tower",
            "model.multi_modal_projector",
        )

    def test_unknown_recipe_raises(self):
        with pytest.raises(NotImplementedError):
            get_recipe("not-a-family")

    def test_download_scripts_are_an_explicit_vlm_train_config_field(self):
        from libreyolo.models.vlm.training.trainer import VLMTrainConfig

        assert VLMTrainConfig().allow_download_scripts is False
        assert (
            VLMTrainConfig(allow_download_scripts=True).allow_download_scripts is True
        )

    def test_dataset_name_mapping_normalizes_supported_numeric_string_ids(self):
        from libreyolo.models.vlm.training.trainer import _normalize_names

        assert _normalize_names({"0": "cat", 1: "dog"}, nc=2) == {
            0: "cat",
            1: "dog",
        }

    @pytest.mark.parametrize(
        "names",
        [
            {False: "cat"},
            {-1: "cat"},
            {0.0: "cat"},
            {"zero": "cat"},
            {0: "cat", "0": "dog"},
            {0: 7},
            ["cat", 7],
        ],
    )
    def test_dataset_name_mapping_rejects_ambiguous_ids_and_nonstring_names(
        self, names
    ):
        from libreyolo.models.vlm.training.trainer import _normalize_names

        with pytest.raises(ValueError):
            _normalize_names(names)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"allow_download_scripts": "false"}, "allow_download_scripts"),
            ({"lora": 1}, "lora must be a bool"),
            ({"workers": -1}, "workers"),
            ({"hflip": 1.1}, "hflip"),
            ({"lr0": 0.0}, "lr0"),
            ({"epochs": True}, "epochs"),
            ({"imgsz": 640}, "Unsupported VLM train.*imgsz"),
            ({"epoch": 2}, "Unsupported VLM train.*epoch"),
        ],
    )
    def test_vlm_trainer_rejects_invalid_public_config_types(self, kwargs, match):
        from libreyolo.models.vlm.training.trainer import VLMDetectionTrainer

        with pytest.raises(ValueError, match=match):
            VLMDetectionTrainer(_StubWrapper(), data="unused.yaml", **kwargs)

    def test_vlm_lora_dependency_floor_rejects_prerelease(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        from libreyolo.models.vlm.training import trainer as trainer_module

        monkeypatch.setitem(sys.modules, "peft", SimpleNamespace())
        monkeypatch.setattr(trainer_module, "version", lambda _name: "0.17.0rc1")

        with pytest.raises(ImportError, match="peft>=0.17.0"):
            trainer_module.require_vlm_lora_dependencies()

    @pytest.mark.parametrize("configured", [False, True])
    def test_vlm_hub_logger_is_rejected_before_resolution(
        self, configured, monkeypatch
    ):
        from libreyolo.models.vlm.training import trainer as trainer_module
        from libreyolo.models.vlm.training.trainer import VLMDetectionTrainer
        from libreyolo.training.loggers import HuggingFaceHubLogger

        logger_request = (
            object.__new__(HuggingFaceHubLogger)
            if configured
            else "hf:owner/vlm-adapter"
        )
        monkeypatch.setattr(
            trainer_module,
            "resolve_loggers",
            lambda *_args, **_kwargs: pytest.fail("Hub logger was resolved"),
        )

        with pytest.raises(NotImplementedError, match="checkpoint directories"):
            VLMDetectionTrainer(
                _StubWrapper(),
                data="unused.yaml",
                loggers=logger_request,
            )

    def test_vlm_hub_logger_is_rejected_when_passed_as_callback(self):
        from libreyolo.models.vlm.training.trainer import VLMDetectionTrainer
        from libreyolo.training.loggers import HuggingFaceHubLogger

        callback = object.__new__(HuggingFaceHubLogger)
        with pytest.raises(NotImplementedError, match="checkpoint directories"):
            VLMDetectionTrainer(
                _StubWrapper(),
                data="unused.yaml",
                callbacks=(item for item in [callback]),
            )

    @pytest.mark.parametrize(
        ("step", "total", "accumulate", "expected"),
        [
            (1, 1, 8, 1),
            (1, 10, 8, 8),
            (8, 10, 8, 8),
            (9, 10, 8, 2),
            (10, 10, 8, 2),
        ],
    )
    def test_accumulation_uses_the_true_final_group_size(
        self, step, total, accumulate, expected
    ):
        from libreyolo.models.vlm.training.trainer import (
            _accumulation_group_size,
        )

        assert _accumulation_group_size(step, total, accumulate) == expected

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [("cpu", "cpu"), ("0", "cuda:0"), (1, "cuda:1")],
    )
    def test_train_device_uses_standard_libreyolo_forms(self, requested, expected):
        from libreyolo.models.vlm.training.trainer import (
            VLMDetectionTrainer,
            VLMTrainConfig,
        )

        trainer = object.__new__(VLMDetectionTrainer)
        trainer.config = VLMTrainConfig(device=requested)
        trainer.wrapper = type("Wrapper", (), {"device": torch.device("cpu")})()

        assert str(trainer._resolve_device()) == expected

    def test_restore_synchronizes_wrapper_device_and_dtype(self):
        from libreyolo.models.vlm.training.trainer import VLMDetectionTrainer

        model = torch.nn.Linear(2, 2).to(dtype=torch.float64)
        wrapper = type(
            "Wrapper",
            (),
            {
                "model": model,
                "device": torch.device("cpu"),
                "_model_dtype": torch.float32,
            },
        )()
        trainer = object.__new__(VLMDetectionTrainer)
        trainer.wrapper = wrapper

        trainer._restore_inference_model(model)

        assert wrapper.device == next(model.parameters()).device
        assert wrapper._model_dtype == torch.float64

    def test_full_ft_reenables_parameters_frozen_by_a_prior_lora_merge(self):
        from libreyolo.models.vlm.training.trainer import (
            VLMDetectionTrainer,
            VLMTrainConfig,
        )

        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.visual = torch.nn.Linear(2, 2)
        model.model.language_model = torch.nn.Linear(2, 2)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        trainer = object.__new__(VLMDetectionTrainer)
        trainer.config = VLMTrainConfig(lora=False)
        trainer.wrapper = type("Wrapper", (), {"model": model})()
        trainer.recipe = get_recipe("qwen3vl")

        train_model = trainer._build_train_model(resume_dir=None)

        assert train_model is model
        assert all(
            not parameter.requires_grad for parameter in model.model.visual.parameters()
        )
        assert all(
            parameter.requires_grad
            for parameter in model.model.language_model.parameters()
        )

    def test_training_dtype_checks_the_requested_cuda_device(self, monkeypatch):
        from libreyolo.models.vlm.training.trainer import VLMDetectionTrainer

        entered = []

        class DeviceContext:
            def __init__(self, device):
                entered.append(str(device))

            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        monkeypatch.setattr(torch.cuda, "device", DeviceContext)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

        dtype = VLMDetectionTrainer._training_dtype(torch.device("cuda:2"))

        assert dtype == torch.bfloat16
        assert entered == ["cuda:2"]

    def test_training_rejects_unscaled_fp16_cuda(self, monkeypatch):
        from libreyolo.models.vlm.training.trainer import VLMDetectionTrainer

        class DeviceContext:
            def __init__(self, _device):
                pass

            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        monkeypatch.setattr(torch.cuda, "device", DeviceContext)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

        with pytest.raises(RuntimeError, match="BF16-capable"):
            VLMDetectionTrainer._training_dtype(torch.device("cuda:0"))

    @pytest.mark.parametrize("failure_stage", ["optimization", "callback"])
    def test_setup_failure_restores_the_inference_wrapper(
        self, failure_stage, monkeypatch, tmp_path
    ):
        from contextlib import nullcontext
        from types import SimpleNamespace

        from libreyolo.models.vlm.training import trainer as trainer_module
        from libreyolo.models.vlm.training.trainer import (
            VLMDetectionTrainer,
            VLMTrainConfig,
        )

        wrapper = _StubWrapper()
        wrapper.device = torch.device("cpu")
        wrapper._model_dtype = torch.float32
        wrapper.model = torch.nn.Linear(2, 2)

        def set_classes(names):
            wrapper.names = dict(enumerate(names))

        wrapper.set_classes = set_classes
        trainer = object.__new__(VLMDetectionTrainer)
        trainer.config = VLMTrainConfig(
            data="unused.yaml", epochs=1, device="cpu", gradient_checkpointing=False
        )
        trainer.wrapper = wrapper
        trainer.recipe = get_recipe("qwen3vl")
        trainer.save_dir = tmp_path / "run"
        trainer.save_dir.mkdir()
        monkeypatch.setattr(
            trainer_module,
            "load_data_config",
            lambda *_args, **_kwargs: {"names": ["cat"], "train": "unused"},
        )
        monkeypatch.setattr(
            trainer,
            "_build_dataloaders",
            lambda *_args, **_kwargs: ([{}], None),
        )
        monkeypatch.setattr(
            trainer, "_build_train_model", lambda _resume: wrapper.model
        )
        restored = []
        monkeypatch.setattr(
            trainer, "_restore_inference_model", lambda model: restored.append(model)
        )

        if failure_stage == "optimization":
            monkeypatch.setattr(
                trainer,
                "_prepare_optimization",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("setup")),
            )
            trainer.callbacks = SimpleNamespace()
        else:
            monkeypatch.setattr(
                trainer,
                "_prepare_optimization",
                lambda *_args, **_kwargs: (
                    1e-4,
                    list(wrapper.model.parameters()),
                    SimpleNamespace(),
                    SimpleNamespace(),
                    nullcontext(),
                ),
            )
            trainer.callbacks = SimpleNamespace(
                on_train_start=lambda _event: (_ for _ in ()).throw(
                    RuntimeError("callback")
                )
            )

        with pytest.raises(
            RuntimeError, match=failure_stage.replace("optimization", "setup")
        ):
            trainer.run()

        assert restored == [wrapper.model]

    def test_validation_loss_must_be_finite(self):
        from contextlib import nullcontext
        from types import SimpleNamespace

        from libreyolo.models.vlm.training.trainer import VLMDetectionTrainer

        class NonFiniteModel(torch.nn.Module):
            def forward(self, **_batch):
                return SimpleNamespace(loss=torch.tensor(float("nan")))

        trainer = object.__new__(VLMDetectionTrainer)
        with pytest.raises(FloatingPointError, match="validation loss"):
            trainer._eval_loss(
                NonFiniteModel(), [{}], torch.device("cpu"), nullcontext()
            )

    def test_nonfinite_gradients_fail_before_optimizer_step(self):
        from libreyolo.models.vlm.training.trainer import VLMDetectionTrainer

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        parameter.grad = torch.tensor(float("inf"))

        with pytest.raises(RuntimeError, match="non-finite"):
            VLMDetectionTrainer._clip_gradients([parameter], 1.0)
